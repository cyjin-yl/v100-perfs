"""Check explicit certified workspace reuse on a real V100, never CPU fallback."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.mxfp4_training.check_torch_cuda import Blocked, Runner, raw_tensor, sha256
from tools.mxfp4_training.check_torch_certified import (
    exact_output, library_metadata, patched_native, source_hashes as certified_sources,
)
from tools.mxfp4_training.reference import gemm_mx

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from tools.mxfp4_training.certified_workspace import CertifiedWorkspace, CertifiedWorkspaceBusy
    from tools.mxfp4_training.torch_ops import TorchMXTensor, _load_library


def source_hashes():
    root = Path(__file__).resolve().parent
    return {**certified_sources(), **{name: sha256(root / name) for name in
            ("certified_workspace.py", "check_certified_workspace.py")}}


def rejected(function, kind=RuntimeError, text=None):
    try:
        function()
    except kind as error:
        if text is not None and text not in str(error):
            raise AssertionError(f"Wrong rejection: {error}") from error
        return {"error_type": type(error).__name__, "message": str(error)}
    raise AssertionError("Expected rejection did not occur")


def run_checks(runner, report):
    device, stream = runner.device, runner.stream
    rng = np.random.default_rng(20261123)
    ah = raw_tensor(rng.integers(0, 16, (8, 64), dtype=np.uint8), 127)
    bh = raw_tensor(rng.integers(0, 16, (12, 64), dtype=np.uint8), 127)
    expected = gemm_mx(ah, bh)
    da, db = runner.execute(lambda: (runner.to_packed(ah), runner.to_packed(bh)))

    def workspace():
        return CertifiedWorkspace(8, 12, 64, device=device)

    def lifecycle():
        w = workspace()
        before = rejected(lambda: w.gemm(da, db), text="not active")
        fixed_properties = {name: rejected(lambda name=name: setattr(w, name, None), AttributeError)
                            for name in ("shape", "device", "padded_k", "library_path")}
        with w:
            initial = w.info()
            info = {}
            result = runner.execute(lambda: w.gemm(da, db, certified_info=info))
            if initial["submitted_calls"] != 0 or info["submitted_calls"] != 1 or info["accepted"] != 1:
                raise AssertionError("Initial/submission metadata differs")
            result_check = exact_output(result.cpu().numpy(), expected)
        w.close()
        if not w.closed:
            raise AssertionError("Workspace not closed")
        return {"pre_entry": before, "read_only_bindings": fixed_properties,
                "initial_info": initial, "submission_info": info,
                "output": result_check, "closed": True, "idempotent_close": True,
                "use_after_close": rejected(lambda: w.gemm(da, db), text="not active"),
                "reentry": rejected(lambda: w.__enter__(), text="exactly once")}
    runner.case("explicit_lifecycle_and_first_output", lifecycle)

    def padded():
        aa = raw_tensor(rng.integers(0, 16, (3, 33), dtype=np.uint8), 127)
        bb = raw_tensor(rng.integers(0, 16, (5, 33), dtype=np.uint8), 127)
        a, b = runner.execute(lambda: (runner.to_packed(aa), runner.to_packed(bb)))
        with CertifiedWorkspace(3, 5, 33, device=device) as w:
            out = runner.execute(lambda: w.gemm(a, b))
            checked = exact_output(out.cpu().numpy(), gemm_mx(aa, bb))
            with torch.cuda.stream(stream):
                invalid_codes = a.codes.clone()
                invalid_codes[:, 16] |= 0x80
                invalid = TorchMXTensor(invalid_codes, a.scales, a.logical_shape)
                failure = rejected(lambda: w.gemm(invalid, b), ValueError, "canonical positive zero")
            info = w.info()
            if info["padded_k"] != 64 or info["submitted_calls"] != 1:
                raise AssertionError("Padding failure changed submitted count or native K")
        return {"logical_k": 33, "native_padded_k": 64, "output": checked,
                "invalid_padding_rejected": failure, "final_info": info}
    runner.case("odd_logical_k_padding_and_native_shape_binding", padded)

    def initial_query_failure():
        lib = _load_library(report["library"]["path"], "cuda_certified")
        destroyed = []
        def get_info(original):
            def call(*args):
                code = original(*args)
                return code if code else 1
            return call
        def destroy(original):
            def call(*args):
                destroyed.append(1)
                return original(*args)
            return call
        w = workspace()
        with patched_native(lib, {"mxfp4_certified_get_info": get_info,
                                  "mxfp4_certified_destroy": destroy}):
            failure = rejected(w.__enter__, text="workspace diagnostic query")
        if not w.closed or destroyed != [1]:
            raise AssertionError("Initial ABI query failure did not clean up actual created context")
        return {"failure_after_real_create_and_real_info_call": failure,
                "native_destroy_calls": len(destroyed), "closed": w.closed}
    runner.case("initial_diagnostic_failure_rolls_back_native_allocation", initial_query_failure)

    def repeated():
        outputs, diagnostics = [], []
        with workspace() as w:
            for step in range(5):
                aa = raw_tensor(rng.integers(0, 16, (8, 64), dtype=np.uint8), 127)
                bb = raw_tensor(rng.integers(0, 16, (12, 64), dtype=np.uint8), 127)
                if step == 1:
                    aa.scales[:, 1] = 128
                elif step == 2:
                    aa.codes[:] = 0x88  # Signed zero input; output must canonicalize +0.
                elif step == 3:
                    aa.scales[:, 1] = 151  # Real certificate rejection, finite fallback.
                elif step == 4:
                    aa.scales[0, 0] = 255  # NaN scale must not reuse a stale certificate.
                reference = gemm_mx(aa, bb)
                info = {}
                def work():
                    da.codes.copy_(torch.from_numpy(aa.codes).to(device))
                    da.scales.copy_(torch.from_numpy(aa.scales).to(device))
                    db.codes.copy_(torch.from_numpy(bb.codes).to(device))
                    db.scales.copy_(torch.from_numpy(bb.scales).to(device))
                    return w.gemm(da, db, certified_info=info).clone()
                actual = runner.execute(work)
                outputs.append(exact_output(actual.cpu().numpy(), reference))
                diagnostics.append(info)
                if info["submitted_calls"] != step + 1 or bool(info["accepted"]) != (step < 3):
                    raise AssertionError("Reused context retained stale count/certificate")
            final = w.info()
        # Restore the stable shared fixture on its stream.
        runner.execute(lambda: (da.codes.copy_(torch.from_numpy(ah.codes).to(device)),
                               da.scales.copy_(torch.from_numpy(ah.scales).to(device)),
                               db.codes.copy_(torch.from_numpy(bh.codes).to(device)),
                               db.scales.copy_(torch.from_numpy(bh.scales).to(device))))
        return {"outputs": outputs, "diagnostics": diagnostics, "final_info": final,
                "same_device_input_storage_mutated_between_calls": True,
                "nan_payload_equivalence_only_for_nan_fixture": True}
    runner.case("mutated_inputs_certificate_and_nan_fallback", repeated)

    def validation():
        with workspace() as w:
            short = TorchMXTensor(da.codes[:7], da.scales[:7], (7, 64))
            shape = rejected(lambda: w.gemm(short, db), ValueError, "fixed logical shape")
            cpu_a = TorchMXTensor(torch.from_numpy(ah.codes), torch.from_numpy(ah.scales), ah.logical_shape)
            cpu_b = TorchMXTensor(torch.from_numpy(bh.codes), torch.from_numpy(bh.scales), bh.logical_shape)
            bound = rejected(lambda: w.gemm(cpu_a, cpu_b), ValueError, "bound CUDA device")
            invalid_info = rejected(lambda: w.gemm(da, db, certified_info={"stale": 1}),
                                    ValueError, "empty dict")
            if w.info()["submitted_calls"] != 0:
                raise AssertionError("Preflight failure submitted work")
            result = runner.execute(lambda: w.gemm(da, db))
            checked = exact_output(result.cpu().numpy(), expected)
        return {"wrong_shape": shape, "wrong_operand_device_cpu": bound,
                "nonempty_diagnostic": invalid_info, "recovery_output": checked}
    runner.case("preflight_failures_recover_without_submission", validation)

    def wrong_thread():
        with workspace() as w:
            def worker():
                return {name: rejected(function, text="entering host thread") for name, function in
                        (("gemm", lambda: w.gemm(da, db)), ("info", w.info), ("close", w.close))}
            with ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(worker).result(timeout=10)
            if w.closed or w.info()["submitted_calls"] != 0:
                raise AssertionError("Wrong thread touched or destroyed owner context")
            out = runner.execute(lambda: w.gemm(da, db))
            result["owner_recovery_output"] = exact_output(out.cpu().numpy(), expected)
        return result
    runner.case("wrong_host_thread_never_enters_native_context", wrong_thread)

    def streams():
        other = torch.cuda.Stream(device=device)
        if not hasattr(torch.cuda, "_sleep"):
            raise Blocked("Stream-busy test needs torch.cuda._sleep on this runtime")
        with workspace() as w:
            with torch.cuda.stream(stream):
                first = w.gemm(da, db)
                torch.cuda._sleep(400_000_000)  # Bounded ~0.3 s at V100 clocks.
            if stream.query():
                raise AssertionError("Busy-stream fixture completed before rejection check")
            with torch.cuda.stream(other):
                busy = rejected(lambda: w.gemm(da, db), CertifiedWorkspaceBusy, "still busy")
            if w.closed or w.submitted_calls != 1:
                raise AssertionError("Busy switch submitted work or closed workspace")
            stream.synchronize()
            first_check = exact_output(first.cpu().numpy(), expected)
            with torch.cuda.stream(other):
                last = w.gemm(da, db)
            other.synchronize()
            last_check = exact_output(last.cpu().numpy(), expected)
            info = w.info()
        return {"busy_switch": busy, "initial_output": first_check, "switched_output": last_check,
                "final_info": info, "different_stream_handles": stream.cuda_stream != other.cuda_stream}
    runner.case("busy_stream_reject_then_completed_stream_switch", streams)

    def queued_lifetime():
        with workspace() as w:
            retained_outputs = []
            with torch.cuda.stream(stream):
                for _ in range(4):
                    a = TorchMXTensor(da.codes.clone(), da.scales.clone(), da.logical_shape)
                    b = TorchMXTensor(db.codes.clone(), db.scales.clone(), db.logical_shape)
                    retained_outputs.append(w.gemm(a, b).clone())
                    del a, b
                # Context exit itself must wait; no explicit synchronize here.
            submitted = w.submitted_calls
        if not stream.query():
            raise AssertionError("Workspace close returned before its native stream completed")
        return {"submitted_calls": submitted, "close_waited": True,
                "outputs": [exact_output(x.cpu().numpy(), expected) for x in retained_outputs],
                "temporary_operand_references_released_after_enqueue": True}
    runner.case("queued_calls_tensor_lifetime_and_close_wait", queued_lifetime)

    def partial_failure():
        w = workspace()
        destroys = []
        with w:
            def wrap_gemm(original):
                def call(*args):
                    status = original(*args)
                    if status:
                        return status
                    return 1  # Inject only after REAL native work was submitted.
                return call
            def wrap_destroy(original):
                def call(*args):
                    destroys.append(1)
                    return original(*args)
                return call
            with patched_native(w._lib, {"mxfp4_certified_gemm": wrap_gemm,
                                         "mxfp4_certified_destroy": wrap_destroy}):
                with torch.cuda.stream(stream):
                    failure = rejected(lambda: w.gemm(da, db), text="workspace gemm")
            if not w.closed or destroys != [1] or not stream.query():
                raise AssertionError("Partial submission did not destroy exactly once and wait")
        return {"injected_after_real_native_submission": True, "failure": failure,
                "native_destroy_calls": len(destroys), "stream_complete_after_failure": True,
                "closed": w.closed}
    runner.case("partial_native_submission_failure_closes_and_waits", partial_failure)

    def body_failure():
        w = workspace()
        class BodyError(Exception):
            pass
        def body():
            with w:
                with torch.cuda.stream(stream):
                    w.gemm(da, db)
                raise BodyError("intentional context-body exception")
        failure = rejected(body, BodyError, "intentional")
        if not w.closed or not stream.query():
            raise AssertionError("Body exception failed to close and wait")
        return {"body_failure": failure, "closed": True, "stream_complete": True}
    runner.case("context_body_exception_cleans_up", body_failure)

    def capture():
        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream(device=device)
        marker = torch.zeros(1, device=device)
        torch.cuda.synchronize(device)
        with workspace() as w:
            # Catch the expected preflight exceptions INSIDE capture, leaving
            # the graph valid; close the active context only after capture ends.
            with torch.cuda.graph(graph, stream=capture_stream):
                marker.add_(1)
                call = rejected(lambda: w.gemm(da, db), text="CUDA Graph capture")
                close = rejected(w.close, text="CUDA Graph capture")
            capture_stream.synchronize()
            if w.closed or w.info()["submitted_calls"] != 0:
                raise AssertionError("Capture rejection changed native workspace")
            out = runner.execute(lambda: w.gemm(da, db))
            checked = exact_output(out.cpu().numpy(), expected)
        return {"gemm_capture_rejected_before_validation": call,
                "destroy_capture_rejected_without_consuming_context": close,
                "post_capture_recovery_output": checked}
    runner.case("capture_rejection_and_post_capture_recovery", capture)

    report["optional_coverage_unavailable"] = []
    if torch.cuda.device_count() < 2:
        report["optional_coverage_unavailable"].append("second CUDA device: only one visible device; CPU operand rejection executed")
    else:
        def second_device():
            index = (device.index + 1) % torch.cuda.device_count()
            other = torch.device(f"cuda:{index}")
            aa = TorchMXTensor(da.codes.to(other), da.scales.to(other), da.logical_shape)
            bb = TorchMXTensor(db.codes.to(other), db.scales.to(other), db.logical_shape)
            with workspace() as w:
                failure = rejected(lambda: w.gemm(aa, bb), ValueError, "bound CUDA device")
                info = w.info()
            return {"failure": failure, "no_submissions": info["submitted_calls"] == 0}
        runner.case("other_cuda_device_operand_rejection", second_device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-evidence", type=Path, required=True)
    parser.add_argument("--expected-certified-variant", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; use a new evidence path")
    if not args.device.startswith("cuda:") or not args.device[5:].isdigit():
        parser.error("require an explicit CUDA device index")
    report = {"schema_version": 1, "purpose": "explicit owner-thread certified workspace correctness",
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "pending", "passed": False,
              "device": args.device, "python": platform.python_version(), "numpy": np.__version__,
              "torch": None if torch is None else torch.__version__, "source_sha256": source_hashes(),
              "cases": [], "gpu_executed": False, "full_training_executed": False,
              "autograd_workspace_integration": False, "cpu_fallback": False}
    runner = None
    try:
        if torch is None or not torch.cuda.is_available():
            raise Blocked("A real CUDA runtime with Torch is required")
        device = torch.device(args.device)
        properties = torch.cuda.get_device_properties(device)
        report["gpu"] = {"name": properties.name, "compute_capability": [properties.major, properties.minor],
                         "total_memory_bytes": properties.total_memory,
                         "torch_cuda_build_version": torch.version.cuda,
                         "actual_SM70": (properties.major, properties.minor) == (7, 0)}
        if not report["gpu"]["actual_SM70"] or "V100" not in properties.name:
            raise Blocked("Require an actual V100 SM70")
        report["library"] = library_metadata(args.compile_evidence, args.expected_certified_variant,
                                             report["source_sha256"])
        report["library"]["context_lifecycle"] = "explicit_single_shape_owner_thread_workspace"
        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device)
            runner = Runner(device, stream, report)
            run_checks(runner, report)
            torch.cuda.synchronize(device)
        if source_hashes() != report["source_sha256"]:
            raise AssertionError("Source changed during checker")
        if sha256(Path(report["library"]["path"])) != report["library"]["sha256"]:
            raise AssertionError("Library changed during checker")
        report.update(status="passed", passed=True, source_unchanged_during_check=True,
                      library_unchanged_during_check=True)
    except Exception as error:
        report.update(status="blocked" if isinstance(error, Blocked) else "failed",
                      error_type=type(error).__name__, error=str(error))
    finally:
        if runner is not None:
            try:
                torch.cuda.synchronize(runner.device)
                runner.pending = False
            except Exception as error:
                report.update(status="failed", passed=False, cleanup_error=str(error))
            report.update(runner.state())
        report["completed_cases"] = sum(case.get("passed") is True for case in report["cases"])
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps({key: report[key] for key in ("status", "passed", "completed_cases", "gpu_executed")}))
    return 0 if report["passed"] else (2 if report["status"] == "blocked" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
