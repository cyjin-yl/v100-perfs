"""Real-SM70 correctness checks for the synchronous cuda_certified bridge.

This is a separate suite: it does not change or replace the 119 baseline/tiled
checks. It requires a successful build manifest binding the actual library to
the current native sources and the explicitly requested compiled variant.
Run from a fixed source snapshot with the intended virtualenv's Python:

  MXFP4_CUDA_LIBRARY=/path/libmxfp4_sm70.so python -m \
    tools.mxfp4_training.check_torch_certified \
    --compile-evidence /path/compile-evidence.json \
    --expected-certified-variant 8 --output /new/check-certified.json

No timing or LLM convergence claim is made. All packed outputs are checked in
full against the independent NumPy MX32 oracle. Certificate diagnostics are
read from the SAME submission and are deliberately outside any timing.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import threading

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.mxfp4_training.check_torch_cuda import Blocked, Runner, compare, raw_tensor, sha256
from tools.mxfp4_training.reference import gemm_mx, quantize_mx

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from tools.mxfp4_training import torch_ops as ops


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = ["check_torch_certified.py", "check_torch_cuda.py", "reference.py", "torch_ops.py"]
    names += [str(p.relative_to(root)) for p in sorted((root / "csrc").iterdir()) if p.is_file()]
    result = {name: sha256(root / name) for name in names}
    mapping = root.parents[1] / "evidence/benchmarks/sm70-packing/study/simulation/round3/cuda/fragment_mapping.hpp"
    result["fragment_mapping.hpp"] = sha256(mapping)
    return result


def library_metadata(manifest_path: Path, expected_variant: int, sources: dict) -> dict:
    value = os.environ.get("MXFP4_CUDA_LIBRARY")
    if not value:
        raise Blocked("Set MXFP4_CUDA_LIBRARY to the actual certified shared library")
    library = Path(value).expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    if not library.is_file() or not manifest_path.is_file():
        raise Blocked("The supplied shared library or compile manifest is absent")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema_version") != 2 or manifest.get("passed") is not True
            or manifest.get("certified_enabled") is not True
            or manifest.get("certified_variant") != expected_variant):
        raise RuntimeError("Compile manifest does not prove the requested successful certified build")
    artifact = manifest.get("artifacts", {}).get(library.name)
    digest = sha256(library)
    if artifact != {"sha256": digest, "bytes": library.stat().st_size}:
        raise RuntimeError("Library bytes/hash differ from the compile manifest")
    expected_native = {k: v for k, v in sources.items() if k.startswith("csrc/")}
    if manifest.get("source_sha256") != expected_native:
        raise RuntimeError("Compile manifest's native sources differ from this fixed snapshot")
    if manifest.get("fragment_mapping_sha256") != sources["fragment_mapping.hpp"]:
        raise RuntimeError("Compile manifest's fragment mapping differs from this snapshot")
    shared = [job for job in manifest.get("jobs", []) if job.get("name") == "shared_library"]
    if len(shared) != 1 or shared[0].get("returncode") != 0:
        raise RuntimeError("Compile manifest lacks a successful shared-library job")
    if f"-DMXFP4_CERTIFIED_VARIANT={expected_variant}" not in shared[0].get("argv", []):
        raise RuntimeError("Shared-library command does not identify the expected compiled variant")
    log = manifest_path.parent / shared[0]["log"]
    if log.parent != manifest_path.parent or sha256(log) != shared[0]["log_sha256"]:
        raise RuntimeError("Shared-library compile log does not match its manifest")
    ops._load_library(str(library), "cuda_certified")
    if ctypes.sizeof(ops._CertifiedInfo) != 128 or ops._CertifiedInfo.submitted_calls.offset != 120:
        raise RuntimeError("Python certified-info ABI v1 layout differs from the header")
    return {"path": str(library), "sha256": digest, "bytes": library.stat().st_size,
            "compiled_variant": expected_variant, "compile_evidence": str(manifest_path),
            "compile_evidence_sha256": sha256(manifest_path), "native_sources_match": True,
            "fragment_mapping_matches": True, "shared_compile_log_sha256": sha256(log),
            "context_lifecycle": "per_call_create_gemm_destroy_synchronous"}


def exact_output(actual, expected) -> dict:
    result = compare(actual, expected, exact=True)
    zeros = actual == 0
    if np.any(actual.view(np.uint32)[zeros] != 0):
        raise AssertionError("A zero GEMM output is not canonical positive zero")
    result["canonical_positive_zero_values"] = int(np.count_nonzero(zeros))
    return result


@contextmanager
def patched_native(lib, replacements):
    originals = {name: getattr(lib, name) for name in replacements}
    try:
        for name, replacement in replacements.items():
            setattr(lib, name, replacement(originals[name]))
        yield
    finally:
        for name, original in originals.items():
            setattr(lib, name, original)


class CertifiedRunner(Runner):
    def __init__(self, device, stream, report):
        super().__init__(device, stream, report)
        self.lib = ops._load_library(report["library"]["path"], "cuda_certified")

    def upload(self, host, offset=0):
        storage = torch.empty(host.codes.size + offset, dtype=torch.uint8, device=self.device)
        if storage.data_ptr() % 8:
            raise AssertionError("Fixture base is not eight-byte aligned")
        codes = storage[offset:].view(host.codes.shape)
        codes.copy_(torch.from_numpy(host.codes).to(self.device))
        scales = torch.from_numpy(host.scales).to(self.device)
        return ops.TorchMXTensor(codes, scales, host.logical_shape)

    def observed(self, a, b, info, *, expected_accepted=None):
        if (info.get("abi_version") != 1 or info.get("device") != self.device.index
                or info.get("metadata_valid") != 1 or info.get("submitted_calls") != 1
                or (info.get("m"), info.get("n"), info.get("padded_k")) !=
                (a.logical_shape[0], b.logical_shape[0], a.padded_k)):
            raise AssertionError("Submission diagnostic identity/shape/call count is wrong")
        if expected_accepted is not None and bool(info["accepted"]) != expected_accepted:
            raise AssertionError("Unexpected certificate acceptance")
        eligible = (a.logical_shape[0] % 128 == 0 and b.logical_shape[0] % 128 == 0
                    and a.codes.data_ptr() % 8 == b.codes.data_ptr() % 8 == 0)
        return {"certificate": info, "a_codes_address_mod8": a.codes.data_ptr() % 8,
                "b_codes_address_mod8": b.codes.data_ptr() % 8,
                "a_storage_offset_bytes": a.codes.storage_offset(),
                "b_storage_offset_bytes": b.codes.storage_offset(),
                "full_tile_aligned_input_eligible": eligible,
                "dispatch_evidence": "argument conditions; not a runtime kernel-name probe"}

    def packed(self, a, b, *, offset=0, accepted=True, reject_mask=0):
        expected = gemm_mx(a, b)
        evidence = {}

        def work():
            da, db = self.upload(a, offset), self.upload(b, offset)
            info = {}
            output = ops.gemm_packed(da, db, backend="cuda_certified", certified_info=info)
            evidence.update(self.observed(da, db, info, expected_accepted=accepted))
            if reject_mask and info["reject_flags"] & reject_mask != reject_mask:
                raise AssertionError("Expected certificate rejection flag was absent")
            return output.clone()  # Immediate GPU consumer on the nondefault stream.

        output = self.execute(work)
        return {"backend": "cuda_certified", "source": "independent_cpu_packed_fixture",
                "logical_shapes": [list(a.logical_shape), list(b.logical_shape)],
                "nondefault_stream": True, **evidence,
                "output": exact_output(output.cpu().numpy(), expected)}

    def linear(self, source, weight, grad, bias, *, noncontiguous=False):
        flat = source.reshape(-1, source.shape[-1])
        upstream = grad.reshape(-1, weight.shape[0])
        pairs = [(flat, weight), (upstream, weight.T), (upstream.T, flat.T)]
        quantized = [(quantize_mx(np.ascontiguousarray(a)), quantize_mx(np.ascontiguousarray(b)))
                     for a, b in pairs]
        observed = []
        original = ops.gemm_packed

        def tracked(a, b, *, backend="reference", certified_info=None):
            if backend != "cuda_certified" or certified_info is not None:
                raise AssertionError("Unexpected linear backend/diagnostic call")
            index = len(observed)
            if index >= 3:
                raise AssertionError("Unexpected extra linear GEMM")
            info = {}
            output = original(a, b, backend=backend, certified_info=info)
            expected_a, expected_b = quantized[index]
            for actual, expected in ((a, expected_a), (b, expected_b)):
                if (actual.logical_shape != expected.logical_shape
                        or not np.array_equal(actual.codes.cpu().numpy(), expected.codes)
                        or not np.array_equal(actual.scales.cpu().numpy(), expected.scales)):
                    raise AssertionError("F/D/W did not independently quantize the original FP32 direction")
            observed.append({"direction": ("forward", "dgrad", "wgrad")[index],
                             "original_fp32_directional_quantization_exact": True,
                             **self.observed(a, b, info, expected_accepted=True)})
            return output

        ops.gemm_packed = tracked
        try:
            result = super().linear_case(source, weight, grad, bias, (True, True, True),
                                         "cuda_certified", noncontiguous=noncontiguous)
        finally:
            ops.gemm_packed = original
        if len(observed) != 3:
            raise AssertionError("Forward/Dgrad/Wgrad did not each execute")
        result["certified_calls"] = observed
        return result

    def zero_k(self):
        evidence = {}

        def work():
            a = ops.TorchMXTensor(torch.empty((128, 0), dtype=torch.uint8, device=self.device),
                                  torch.empty((128, 0), dtype=torch.uint8, device=self.device), (128, 0))
            try:
                ops.gemm_packed(a, a, backend="cuda_certified")
            except ValueError as error:
                if "positive integers" not in str(error):
                    raise
            else:
                raise AssertionError("Public wrapper silently changed its positive-K contract")
            output = torch.empty((128, 128), dtype=torch.float32, device=self.device)
            info = {}
            # Explicit native-only boundary test: bypass the PUBLIC shape validator.
            ops._gemm_certified(self.lib, a, a, output, self.stream, info)
            evidence.update(self.observed(a, a, info, expected_accepted=True))
            return output.clone()

        output = self.execute(work)
        return {"public_wrapper_K0_rejected": True, "native_only_K0_executed": True,
                **evidence, "output": exact_output(output.cpu().numpy(), np.zeros((128, 128), np.float32))}

    def producer_consumer(self, a, b):
        expected = gemm_mx(a, b)
        producer = torch.cuda.Stream(device=self.device)
        evidence = {}

        def work():
            with torch.cuda.stream(producer):
                da, db = self.upload(a), self.upload(b)
                ready = torch.cuda.Event()
                ready.record(producer)
            self.stream.wait_event(ready)  # Explicit cross-stream data dependency.
            info = {}
            output = ops.gemm_packed(da, db, backend="cuda_certified", certified_info=info)
            evidence.update(self.observed(da, db, info, expected_accepted=True))
            del da, db
            pressure = [torch.empty((128, 128), dtype=torch.uint8, device=self.device) for _ in range(16)]
            for value in pressure:
                value.fill_(255)
            return output.clone()

        output = self.execute(work)
        return {"producer_to_consumer_event": True, "input_references_dropped_after_synchronous_return": True,
                "allocator_pressure_bytes": 16 * 128 * 128, **evidence,
                "output": exact_output(output.cpu().numpy(), expected)}

    def threads(self, a, b):
        expected = gemm_mx(a, b)
        lock, rendezvous = threading.Lock(), threading.Barrier(2, timeout=15)
        active, owners, destroyed = set(), {}, []
        peak = [0]

        def wrap_create(original):
            def create(m, n, k, destination):
                code = original(m, n, k, destination)
                if code == 0:
                    pointer = ctypes.cast(destination, ctypes.POINTER(ctypes.c_void_p))[0]
                    with lock:
                        active.add(pointer); owners[pointer] = threading.get_ident()
                        peak[0] = max(peak[0], len(active))
                    rendezvous.wait()  # Both contexts exist before either is used.
                return code
            return create

        def wrap_destroy(original):
            def destroy(context):
                pointer = context.value
                owner_match = owners.get(pointer) == threading.get_ident()
                code = original(context)
                with lock:
                    active.discard(pointer)
                    destroyed.append({"owner_thread_matches": owner_match, "native_status": int(code)})
                return code
            return destroy

        def worker(_index):
            with torch.cuda.device(self.device):
                stream = torch.cuda.Stream(device=self.device)
                with torch.cuda.stream(stream):
                    da, db = self.upload(a), self.upload(b)
                    info = {}
                    result = ops.gemm_packed(da, db, backend="cuda_certified", certified_info=info).clone()
                    observation = self.observed(da, db, info, expected_accepted=True)
                stream.synchronize()
                return {"host_thread": threading.get_ident(), **observation,
                        "output": exact_output(result.cpu().numpy(), expected)}

        def work():
            with patched_native(self.lib, {"mxfp4_certified_create": wrap_create,
                                           "mxfp4_certified_destroy": wrap_destroy}):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    return list(pool.map(worker, (0, 1)))

        results = self.execute(work)
        if (len({value["host_thread"] for value in results}) != 2 or peak[0] != 2 or active
                or len(destroyed) != 2 or any(not d["owner_thread_matches"] or d["native_status"] for d in destroyed)):
            raise AssertionError("Per-thread context creation/use/destruction isolation failed")
        return {"independent_host_threads": 2, "simultaneously_live_contexts": peak[0],
                "contexts_destroyed_on_owner_threads": destroyed, "workers": results}

    def post_submit_error(self, a, b):
        state = {"native_gemm_status": None, "destroy_statuses": []}

        def wrap_gemm(original):
            def gemm(*args):
                code = int(original(*args))
                state["native_gemm_status"] = code
                return 1 if code == 0 else code
            return gemm

        def wrap_destroy(original):
            def destroy(context):
                code = int(original(context)); state["destroy_statuses"].append(code)
                return code
            return destroy

        def work():
            da, db = self.upload(a), self.upload(b)
            with patched_native(self.lib, {"mxfp4_certified_gemm": wrap_gemm,
                                           "mxfp4_certified_destroy": wrap_destroy}):
                try:
                    ops.gemm_packed(da, db, backend="cuda_certified")
                except RuntimeError as error:
                    if state["native_gemm_status"] != 0 or "certified gemm failed: status=1" not in str(error):
                        raise
                else:
                    raise AssertionError("Controlled post-submit error was not propagated")
            if state["destroy_statuses"] != [0] or not self.stream.query():
                raise AssertionError("The failing call did not destroy/wait exactly once")

        self.execute(work)
        return {"injection": "test shim returns status 1 AFTER a successful native asynchronous submission",
                "invalid_device_addresses_used": False, "error_propagated": True,
                "stream_complete_on_error_return": True, **state}

    def capture_rejection(self, a, b):
        creates = []

        def wrap_create(original):
            def create(*args):
                creates.append(True)
                return original(*args)
            return create

        def work():
            da, db = self.upload(a), self.upload(b)
            marker = torch.ones(8, dtype=torch.float32, device=self.device)
            target = torch.empty_like(marker)
            self.stream.synchronize()
            graph, rejected = torch.cuda.CUDAGraph(), False
            with patched_native(self.lib, {"mxfp4_certified_create": wrap_create}):
                with torch.cuda.graph(graph, stream=self.stream):
                    try:
                        ops.gemm_packed(da, db, backend="cuda_certified")
                    except RuntimeError as error:
                        if "does not support CUDA Graph capture" not in str(error):
                            raise
                        rejected = True
                    torch.add(marker, 1, out=target)  # Keep the valid graph nonempty.
            graph.replay()
            if not rejected or creates:
                raise AssertionError("Capture was not rejected before native context creation")
            return target

        output = self.execute(work)
        np.testing.assert_array_equal(output.cpu().numpy(), np.full(8, 2., dtype=np.float32))
        return {"capture_rejected_before_native_create": True, "native_create_attempts": len(creates),
                "surrounding_graph_remains_valid": True}


def run_suite(runner, seed):
    rng = np.random.default_rng(seed)

    def fixture(rows, k):
        return raw_tensor(rng.integers(0, 16, (rows, k), dtype=np.uint8), 127)

    a, b = fixture(128, 128), fixture(128, 128)
    runner.case("packed_full_tile_128", lambda: runner.packed(a, b))
    ma, mb = fixture(256, 64), fixture(128, 64)
    runner.case("packed_multiple_tiles", lambda: runner.packed(ma, mb))
    ta, tb = fixture(129, 65), fixture(67, 65)
    runner.case("packed_tail_C6_compatibility", lambda: runner.packed(ta, tb))
    for offset in (1, 2, 4):
        runner.case(f"packed_offset_{offset}_C6_compatibility",
                    lambda offset=offset: runner.packed(a, b, offset=offset))
    runner.case("public_K0_reject_and_native_K0_positive_zero", runner.zero_k)
    z = raw_tensor(np.full((128, 32), 8, dtype=np.uint8), 0)
    runner.case("signed_zero_low_scale_positive_zero_output", lambda: runner.packed(z, z))
    nz = raw_tensor(np.zeros((128, 32), dtype=np.uint8), 255)
    runner.case("nan_scale_reject_even_for_zero_dot", lambda: runner.packed(nz, z, accepted=False, reject_mask=1))
    mixed_codes = np.full((128, 64), 8, dtype=np.uint8); mixed_codes[:, 32:] = 4
    mixed = raw_tensor(mixed_codes, np.tile(np.array([0, 127], dtype=np.uint8), (128, 1)))
    runner.case("zero_half_below_row_base", lambda: runner.packed(mixed, mixed))
    x, w, g = [rng.normal(size=(128, 128)).astype(np.float32) for _ in range(3)]
    runner.case("linear_F_D_W_full_tiles", lambda: runner.linear(x, w, g, np.zeros(128, np.float32)))
    tx = rng.normal(size=(2, 5, 33)).astype(np.float32)
    tw = rng.normal(size=(3, 33)).astype(np.float32)
    tg = rng.normal(size=(2, 5, 3)).astype(np.float32)
    runner.case("linear_F_D_W_noncontiguous_tails",
                lambda: runner.linear(tx, tw, tg, np.zeros(3, np.float32), noncontiguous=True))
    runner.case("producer_consumer_and_allocator_pressure", lambda: runner.producer_consumer(a, b))
    runner.case("two_host_threads_separate_contexts", lambda: runner.threads(a, b))
    runner.case("cleanup_after_controlled_post_submit_error", lambda: runner.post_submit_error(a, b))
    runner.case("reject_graph_capture", lambda: runner.capture_rejection(a, b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-evidence", type=Path, required=True)
    parser.add_argument("--expected-certified-variant", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a fresh evidence path")
    if not args.device.startswith("cuda:") or not args.device[5:].isdigit():
        parser.error("device must name an explicit CUDA index")
    if not 0 <= args.seed <= 2**32 - 1 or not 0 <= args.expected_certified_variant <= 255:
        parser.error("seed or certified variant is outside the supported integer range")
    sources = source_hashes()
    report = {"schema_version": 1, "purpose": "real SM70 synchronous certified PyTorch bridge correctness",
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "pending", "passed": False,
              "gpu_executed": False, "gpu_work_attempted": False, "gpu_work_call_returned": False,
              "gpu_work_synchronized": False, "all_attempted_work_synchronized": True,
              "tested_backends": [], "planned_backend": "cuda_certified",
              "context_lifecycle": "per_call_create_gemm_destroy_synchronous",
              "expected_certified_variant": args.expected_certified_variant,
              "device": args.device, "seed": args.seed, "source_sha256": sources,
              "python": platform.python_version(), "python_executable": sys.executable,
              "numpy": np.__version__, "torch": None if torch is None else torch.__version__,
              "cases": [], "performance_claim": None, "full_training_executed": False,
              "llm_convergence_validated": False, "cpu_fallback": False,
              "existing_baseline_tiled_119_suite_replaced": False,
              "minimum_free_bytes_after_context": 512 * 1024 * 1024}
    runner, code = None, 1
    try:
        if torch is None or not torch.cuda.is_available():
            raise Blocked("CUDA-enabled PyTorch is unavailable; no CPU fallback was attempted")
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        report["gpu"] = {"name": properties.name, "compute_capability": [properties.major, properties.minor],
                         "total_memory_bytes": properties.total_memory,
                         "multiprocessor_count": properties.multi_processor_count,
                         "actual_V100": "V100" in properties.name and (properties.major, properties.minor) == (7, 0),
                         "torch_cuda_build_version": torch.version.cuda}
        if (properties.major, properties.minor) != (7, 0):
            raise Blocked("The certified suite requires real SM70 hardware")
        free, total = torch.cuda.mem_get_info(device)
        report["gpu"].update(free_bytes_after_context=free, runtime_total_bytes=total)
        if free < report["minimum_free_bytes_after_context"]:
            raise Blocked("Less than 512 MiB is free; no services are stopped")
        report["library"] = library_metadata(args.compile_evidence, args.expected_certified_variant, sources)
        # Keep the exercised wrapper and the lifecycle instrumentation on the
        # same hash-verified CDLL cache key, including a symlinked input path.
        os.environ["MXFP4_CUDA_LIBRARY"] = report["library"]["path"]
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        stream = torch.cuda.Stream(device=device)
        if stream.cuda_stream == torch.cuda.default_stream(device).cuda_stream:
            raise RuntimeError("A nondefault stream was not obtained")
        runner = CertifiedRunner(device, stream, report)
        run_suite(runner, args.seed)
        torch.cuda.synchronize(device)
        if source_hashes() != sources:
            raise RuntimeError("Source changed during the check")
        if sha256(Path(report["library"]["path"])) != report["library"]["sha256"]:
            raise RuntimeError("Shared library changed during the check")
        if sha256(args.compile_evidence.resolve()) != report["library"]["compile_evidence_sha256"]:
            raise RuntimeError("Compile manifest changed during the check")
        names = [case["name"] for case in report["cases"]]
        if len(names) != len(set(names)) or len(names) != 16:
            raise RuntimeError("Suite did not execute every uniquely named planned case")
        report.update(status="passed", passed=True, tested_backends=["cuda_certified"],
                      source_unchanged_during_check=True)
        code = 0
    except Blocked as error:
        report.update(status="blocked", error_type=type(error).__name__, error=str(error)); code = 2
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
    finally:
        if runner is not None:
            report.update(runner.state())
        report["completed_cases"] = sum(case["passed"] for case in report["cases"])
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False); handle.write("\n")
        print(json.dumps({"status": report["status"], "passed": report["passed"],
                          "gpu_executed": report["gpu_executed"], "completed_cases": report["completed_cases"],
                          "output": str(args.output)}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
