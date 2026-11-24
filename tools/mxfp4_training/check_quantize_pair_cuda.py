"""Check the opt-in paired CUDA RNE wrapper against every NumPy output byte.

This checker executes only on an actual V100. It covers different shapes and
layouts, both independent nonfinite flags, capture and preflight rejection,
buffer stream registration before submission, and cleanup after a real native
enqueue followed by an injected host error. It never benchmarks or changes a
training default.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from unittest.mock import patch

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.mxfp4_training import check_quantize_cuda as check
from tools.mxfp4_training.reference import quantize_mx

torch = check.torch
if torch is not None:
    ops = check.ops

ORIGINAL_SINGLE_CUDA_SHA = "d6e8b0bb1431e836add9b72925463d80295faee8b8c0f2788f5219eca120cb37"


def source_hashes():
    root = Path(__file__).resolve().parent
    return {**check.source_hashes(), "check_quantize_pair_cuda.py": check.sha256(__file__)}


def single_quantizer_hash():
    text = (Path(__file__).resolve().parent / "torch_ops.py").read_text()
    function = next(node for node in ast.parse(text).body
                    if isinstance(node, ast.FunctionDef) and node.name == "quantize_mx_cuda_rne")
    return hashlib.sha256(ast.get_source_segment(text, function).encode()).hexdigest()


@contextmanager
def patched_kernel(library, replacement):
    original = library.mxfp4_quantize_rne_cuda
    library.mxfp4_quantize_rne_cuda = replacement(original)
    try:
        yield
    finally:
        library.mxfp4_quantize_rne_cuda = original


def rejected(function, kind, contains):
    try:
        function()
    except kind as error:
        check.need(contains in str(error), "Wrong rejection: " + str(error))
        return {"error_type": type(error).__name__, "message": str(error)}
    raise AssertionError("Expected paired quantizer rejection did not occur")


def compare_pair(actual, expected):
    check.need(isinstance(actual, tuple) and len(actual) == len(expected) == 2,
               "Paired wrapper must return exactly two packed tensors")
    return {name: check.packed_metric(got, wanted)
            for name, got, wanted in zip(("A", "B"), actual, expected)}


def gpu_metadata(device):
    properties = torch.cuda.get_device_properties(device)
    return {"name": properties.name, "compute_capability": [properties.major, properties.minor],
            "actual_V100": properties.major == 7 and properties.minor == 0 and "V100" in properties.name,
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "torch_cuda_build_version": torch.version.cuda}


def run_suite(runner, report, seed):
    rng = np.random.default_rng(seed)
    device, stream = runner.device, runner.stream
    native = ops._load_quantizer_library(os.environ["MXFP4_CUDA_LIBRARY"])

    def finite(a, b, layout_a="contiguous", layout_b="contiguous"):
        expected = tuple(quantize_mx(x) for x in (a, b))
        def work():
            x, y = (check.to_layout(array, device, layout)
                    for array, layout in ((a, layout_a), (b, layout_b)))
            paired = ops.quantize_pair_mx_cuda_rne(x, y)
            singles = (ops.quantize_mx_cuda_rne(x), ops.quantize_mx_cuda_rne(y))
            return x, y, paired, singles
        x, y, paired, singles = runner.execute(work)
        for source, original in ((x, a), (y, b)):
            check.need(np.array_equal(check.host_array(source).view(np.uint32),
                                      np.ascontiguousarray(original).view(np.uint32)), "Input changed")
        return {"layout_A": layout_a, "layout_B": layout_b,
                "strides_A": list(x.stride()), "strides_B": list(y.stride()),
                "pair_vs_numpy": compare_pair(paired, expected),
                "two_singles_vs_numpy": compare_pair(singles, expected),
                "input_fp32_bits_unchanged": True, "nondefault_stream": True}

    runner.case("all_scale_midpoints_and_different_shapes", lambda: finite(
        check.midpoint_matrix(), np.zeros((3, 65), dtype=np.float32)))
    runner.case("exponent_boundaries_and_random_finite_bits", lambda: finite(
        check.exponent_boundary_matrix(), check.finite_bits(rng, (17, 129)), "transpose", "positive_strides_offset"))
    zero_a = np.zeros((4, 65), dtype=np.float32); zero_a[1::2] = -0.
    zero_b = np.zeros((3, 33), dtype=np.float32); zero_b[:, 1::2] = -0.
    runner.case("signed_zero_blocks_and_positive_tail", lambda: finite(
        zero_a, zero_b, "positive_strides_offset", "transpose"))
    for rows, k in ((1, 1), (3, 31), (2, 32), (7, 33), (5, 63), (3, 64), (17, 65), (33, 97), (128, 128)):
        a = rng.normal(size=(rows, k)).astype(np.float32)
        b = rng.normal(size=(rows + 1, k + 2)).astype(np.float32)
        for la, lb in (("contiguous", "transpose"), ("row_offset", "positive_strides_offset")):
            runner.case(f"layouts_{rows}x{k}_{la}_{lb}", lambda a=a, b=b, la=la, lb=lb: finite(a, b, la, lb))
    row_broadcast = np.broadcast_to(rng.normal(size=(1, 65)).astype(np.float32), (7, 65)).copy()
    column_broadcast = np.broadcast_to(np.asarray([0., -0., 1.1, -4.5], np.float32)[:, None], (4, 33)).copy()
    runner.case("independent_zero_stride_operands", lambda: finite(
        row_broadcast, column_broadcast, "broadcast_rows", "broadcast_columns"))
    runner.case("int64_single_element_strides", lambda: finite(
        np.asarray([[1.25]], np.float32), np.asarray([[-0.]], np.float32),
        "wide_stride_single", "wide_stride_single"))
    if hasattr(torch, "_neg_view"):
        neg_a = rng.normal(size=(7, 33)).astype(np.float32); neg_a[0, 0] = -0.
        neg_b = rng.normal(size=(5, 65)).astype(np.float32); neg_b[-1, -1] = -0.
        runner.case("both_lazy_negative_views", lambda: finite(neg_a, neg_b, "lazy_negative", "lazy_negative"))
    else:
        report["optional_coverage_unavailable"].append("torch._neg_view is unavailable")

    def alias():
        values = rng.normal(size=(5, 65)).astype(np.float32)
        expected = quantize_mx(values)
        def work():
            source = check.to_layout(values, device, "transpose")
            return ops.quantize_pair_mx_cuda_rne(source, source)
        actual = runner.execute(work)
        check.need(actual[0].codes.data_ptr() != actual[1].codes.data_ptr(), "Output buffers unexpectedly alias")
        return {"same_input_tensor_for_both_operands": True, "outputs": compare_pair(actual, (expected, expected))}
    runner.case("aliased_input_with_independent_outputs", alias)

    def invalid_input(value, side, last):
        a, b = [rng.normal(size=shape).astype(np.float32) for shape in ((3, 65), (5, 33))]
        for index in ((0, 1) if side == "both" else (side,)):
            values = (a, b)[index]; values.flat[-1 if last else 0] = value
        good = [np.ones_like(a), np.ones_like(b)]
        expected = tuple(quantize_mx(x) for x in good)
        def work():
            x = check.to_layout(a, device, "positive_strides_offset")
            y = check.to_layout(b, device, "transpose")
            failure = rejected(lambda: ops.quantize_pair_mx_cuda_rne(x, y), ValueError, "non-finite")
            recovery = ops.quantize_pair_mx_cuda_rne(*(torch.from_numpy(x).to(device) for x in good))
            return failure, recovery
        failure, recovery = runner.execute(work)
        return {"nonfinite": str(value), "operand": side, "position": "last" if last else "first",
                "whole_pair_rejected": failure, "finite_recovery": compare_pair(recovery, expected)}
    for label, value in (("nan", np.nan), ("posinf", np.inf), ("neginf", -np.inf)):
        for side in (0, 1, "both"):
            for last in (False, True):
                runner.case(f"reject_{label}_{side}_{'last' if last else 'first'}",
                            lambda v=value, s=side, last=last: invalid_input(v, s, last))

    stable_np = rng.normal(size=(7, 65)).astype(np.float32)
    expected_stable = quantize_mx(stable_np)
    stable = runner.execute(lambda: torch.from_numpy(stable_np).to(device))

    def preflight():
        calls = []
        def count(original):
            def wrapped(*args):
                calls.append(1); return original(*args)
            return wrapped
        def work():
            failures = {}
            candidates = {"non_tensor": None, "cpu": torch.ones((2, 3), dtype=torch.float32),
                          "dtype": stable.to(torch.float64), "empty": stable[:0],
                          "rank": stable[0]}
            with patched_kernel(native, count):
                for name, bad in candidates.items():
                    for side in (0, 1):
                        pair = (bad, stable) if side == 0 else (stable, bad)
                        failures[f"{name}_{side}"] = rejected(
                            lambda pair=pair: ops.quantize_pair_mx_cuda_rne(*pair), ValueError, "requires")
            check.need(not calls, "Invalid metadata was detected after native submission")
            return failures, ops.quantize_pair_mx_cuda_rne(stable, stable)
        failures, recovery = runner.execute(work)
        return {"metadata_rejections": failures, "native_calls_during_rejections": 0,
                "recovery": compare_pair(recovery, (expected_stable, expected_stable))}
    runner.case("both_operand_metadata_preflight_before_native_calls", preflight)

    def registration_and_copy():
        ranges = []; snapshots = []; host_copies = []
        record_original, cpu_original = torch.Tensor.record_stream, torch.Tensor.cpu
        def record(tensor, target_stream):
            ranges.append((tensor.data_ptr(), tensor.numel() * tensor.element_size(), int(target_stream.cuda_stream)))
            return record_original(tensor, target_stream)
        def cpu(tensor, *args, **kwargs):
            host_copies.append({"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                                "bytes": tensor.numel() * tensor.element_size()})
            return cpu_original(tensor, *args, **kwargs)
        def observe(original):
            def wrapped(*args):
                snapshots.append({"pointers": tuple(int(p) for p in args[:4]), "records": list(ranges)})
                return original(*args)
            return wrapped
        def work():
            x = check.to_layout(stable_np, device, "lazy_negative" if hasattr(torch, "_neg_view") else "transpose")
            y = check.to_layout(stable_np, device, "positive_strides_offset")
            with patch.object(torch.Tensor, "record_stream", record), patch.object(torch.Tensor, "cpu", cpu), patched_kernel(native, observe):
                actual = ops.quantize_pair_mx_cuda_rne(x, y)
            check.need(len(snapshots) == 2, "Pair must call the native quantizer twice")
            first = snapshots[0]["records"]
            for snapshot in snapshots:
                for pointer in snapshot["pointers"]:
                    check.need(any(start <= pointer < start + length and sid == int(stream.cuda_stream)
                                   for start, length, sid in first), "Native buffer was not registered before first enqueue")
            for tensor in (x, y):
                check.need(any(start == tensor.data_ptr() for start, _, _ in first), "Original input storage was not registered")
            check.need(host_copies == [{"shape": [2], "dtype": "torch.int32", "bytes": 8}],
                       "Expected exactly one two-flag host copy")
            return actual
        actual = runner.execute(work)
        return {"all_native_buffers_registered_before_first_enqueue": True,
                "original_input_storage_registered": True, "native_calls": len(snapshots),
                "host_flag_transfers": host_copies,
                "outputs": compare_pair(actual, (expected_stable, expected_stable))}
    runner.case("both_native_calls_share_one_checked_host_transfer", registration_and_copy)

    def partial_enqueue(fail_at):
        calls = []; synchronizations = []
        synchronize_original = torch.cuda.Stream.synchronize
        def synchronize(target_stream):
            synchronizations.append(int(target_stream.cuda_stream))
            return synchronize_original(target_stream)
        def inject(original):
            def wrapped(*args):
                code = original(*args); calls.append(code)
                return 1 if len(calls) == fail_at and code == 0 else code
            return wrapped
        def work():
            with patched_kernel(native, inject), patch.object(torch.cuda.Stream, "synchronize", synchronize):
                failure = rejected(lambda: ops.quantize_pair_mx_cuda_rne(stable, stable), RuntimeError, "launch")
            check.need(len(calls) == fail_at and all(code == 0 for code in calls), "Injection did not follow real successful enqueues")
            check.need(synchronizations == [int(stream.cuda_stream)] and stream.query(),
                       "Partial enqueue error did not synchronize the actual stream")
            return failure, ops.quantize_pair_mx_cuda_rne(stable, stable)
        failure, recovery = runner.execute(work)
        return {"failure_after_real_native_call": fail_at, "successful_native_enqueues": calls,
                "failure": failure, "cleanup_waited_for_actual_stream": True,
                "recovery": compare_pair(recovery, (expected_stable, expected_stable))}
    for fail_at in (1, 2):
        runner.case(f"partial_enqueue_failure_{fail_at}_cleans_up", lambda fail_at=fail_at: partial_enqueue(fail_at))

    def capture():
        graph = torch.cuda.CUDAGraph(); capture_stream = torch.cuda.Stream(device=device)
        marker = torch.zeros(1, device=device); torch.cuda.synchronize(device)
        calls = []
        def observe(original):
            def wrapped(*args):
                calls.append(1); return original(*args)
            return wrapped
        with patched_kernel(native, observe):
            with torch.cuda.graph(graph, stream=capture_stream):
                marker.add_(1)
                failure = rejected(lambda: ops.quantize_pair_mx_cuda_rne(stable, stable), RuntimeError, "CUDA Graph capture")
        capture_stream.synchronize()
        check.need(not calls, "Capture rejection called the native quantizer")
        actual = runner.execute(lambda: ops.quantize_pair_mx_cuda_rne(stable, stable))
        return {"failure": failure, "native_calls_during_capture": 0,
                "post_capture_recovery": compare_pair(actual, (expected_stable, expected_stable))}
    runner.case("capture_preflight_and_recovery", capture)

    if torch.cuda.device_count() > 1:
        def second_device():
            other = torch.device(f"cuda:{(device.index + 1) % torch.cuda.device_count()}")
            second = stable.to(other); torch.cuda.synchronize(other)
            failure = rejected(lambda: ops.quantize_pair_mx_cuda_rne(stable, second), ValueError, "share one CUDA device")
            return {"failure": failure, "other_device": str(other)}
        runner.case("different_cuda_devices_rejected", second_device)
    else:
        report["optional_coverage_unavailable"].append("Only one CUDA device is visible; CPU-input rejection was executed")
    expected_count = 47 + int(hasattr(torch, "_neg_view")) + int(torch.cuda.device_count() > 1)
    names = [row["name"] for row in report["cases"]]
    check.need(len(names) == len(set(names)) == expected_count, "Paired checker coverage count differs")
    report["expected_cases"] = expected_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Output exists; use a new evidence path")
    if not args.device.startswith("cuda:") or not args.device[5:].isdigit(): parser.error("Require explicit CUDA index")
    if not 0 <= args.seed < 2**32: parser.error("Require uint32 seed")
    report = {"schema_version": 1, "purpose": "independent paired CUDA RNE wrapper correctness",
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "pending", "passed": False,
              "device": args.device, "seed": args.seed, "python": platform.python_version(), "numpy": np.__version__,
              "torch": None if torch is None else torch.__version__, "source_sha256": source_hashes(),
              "cases": [], "optional_coverage_unavailable": [], "gpu_executed": False,
              "performance_measured": False, "full_training_executed": False, "cpu_fallback": False,
              "single_cuda_quantizer_expected_source_sha256": ORIGINAL_SINGLE_CUDA_SHA}
    runner = None; code = 1
    try:
        check.need(single_quantizer_hash() == ORIGINAL_SINGLE_CUDA_SHA, "Existing single CUDA quantizer changed")
        if torch is None or not torch.cuda.is_available(): raise check.Blocked("CUDA-enabled Torch is unavailable")
        device = torch.device(args.device); torch.cuda.set_device(device)
        report["gpu"] = gpu_metadata(device)
        if not report["gpu"]["actual_V100"]: raise check.Blocked("Require an actual V100 SM70")
        library, report["library"] = check.library_metadata(args.compile_evidence)
        free, _ = torch.cuda.mem_get_info(device)
        if free < check.MINIMUM_FREE or check.CHECK_ALLOCATION_BUDGET > free - check.RESERVE_BYTES:
            raise check.Blocked("Insufficient guarded device memory")
        torch.set_num_threads(2)
        stream = torch.cuda.Stream(device=device)
        check.need(stream.cuda_stream != torch.cuda.default_stream(device).cuda_stream, "Require nondefault stream")
        runner = check.Runner(device, stream, report, library)
        with torch.cuda.stream(stream), torch.autocast(device_type="cuda", enabled=False):
            run_suite(runner, report, args.seed)
        stream.synchronize()
        check.need(source_hashes() == report["source_sha256"], "Sources changed during check")
        check.need(check.sha256(report["library"]["path"]) == report["library"]["sha256"], "Library changed during check")
        check.need(check.sha256(args.compile_evidence) == report["library"]["compile_evidence_sha256"], "Compile evidence changed")
        report.update(status="passed", passed=True, source_unchanged_during_check=True,
                      library_unchanged_during_check=True, original_single_quantizer_unchanged=True)
        code = 0
    except check.Blocked as error:
        report.update(status="blocked", error_type=type(error).__name__, error=str(error)); code = 2
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
    finally:
        if runner is not None:
            if runner.pending:
                try:
                    runner.stream.synchronize(); runner.pending = False
                except Exception as error:
                    report["cleanup_error"] = str(error)
            report.update(gpu_executed=True if runner.completed else (None if runner.attempted else False),
                          gpu_work_attempted=runner.attempted,
                          all_attempted_work_synchronized=not runner.pending)
        report["completed_cases"] = sum(row.get("passed") is True for row in report["cases"])
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False); handle.write("\n")
        print(json.dumps({key: report[key] for key in ("status", "passed", "gpu_executed", "completed_cases")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
