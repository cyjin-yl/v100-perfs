"""Measure the current PyTorch MXFP4 quantization and packed GEMM path on SM70.

First run the external integration checker against the SAME source/library:
  python -m tools.mxfp4_training.check_torch_cuda --output /tmp/check.json
Then, with MXFP4_CUDA_LIBRARY still set:
  python -m tools.mxfp4_training.bench_torch_cuda \
    --check-report /tmp/check.json --output /tmp/torch-bench.json

The FP16 Torch baselines return FP16, as ordinary torch.mm does. They are
context, not the FP16-input/FP32-output cuBLAS baseline in bench_mxfp4.cu.
This measures synthetic forward products, not training. The optional
--with-certified job includes the synchronous certified context lifecycle.
No CUDA device means blocked, never a CPU fallback.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Callable

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.mxfp4_training.check_torch_cuda import (
    Blocked, library_metadata, sha256, source_hashes as check_source_hashes,
)

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from tools.mxfp4_training.torch_ops import gemm_packed, quantize_mx_torch, quantize_mx_cuda_rne


MINIMUM_FREE_BYTES = 1024 * 1024 * 1024
RESERVE_BYTES = 512 * 1024 * 1024


def source_hashes() -> dict[str, str]:
    return {**check_source_hashes(), Path(__file__).name: sha256(Path(__file__))}


def memory_snapshot(device) -> dict:
    free, total = torch.cuda.mem_get_info(device)
    return {"device_free_bytes": free, "device_total_bytes": total,
            "torch_allocated_bytes": torch.cuda.memory_allocated(device),
            "torch_reserved_bytes": torch.cuda.memory_reserved(device),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def prerequisite(path: Path, report: dict) -> dict:
    """Consume evidence; never run or replace the external integration check."""
    if not path.is_file():
        raise Blocked("Run check_torch_cuda.py first and pass its existing --check-report JSON")
    try:
        evidence = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise Blocked(f"Cannot read the prerequisite check report: {error}") from error
    if not isinstance(evidence, dict):
        raise Blocked("The prerequisite integration report must be a JSON object")
    if (evidence.get("status") != "passed" or evidence.get("passed") is not True
            or evidence.get("gpu_executed") is not True
            or evidence.get("all_attempted_work_synchronized") is not True
            or evidence.get("source_unchanged_during_check") is not True
            or evidence.get("completed_cases", 0) <= 0
            or not evidence.get("cases")
            or not all(case.get("passed") is True for case in evidence["cases"])):
        raise Blocked("The prerequisite integration report is not a completed passing GPU check")
    expected_sources = check_source_hashes()
    if evidence.get("source_sha256") != expected_sources:
        raise Blocked("Integration check source hashes differ; rerun check_torch_cuda.py first")
    if evidence.get("library", {}).get("sha256") != report["library"]["sha256"]:
        raise Blocked("Integration check used a different CUDA library; rerun it first")
    prior_gpu, gpu = evidence.get("gpu", {}), report["gpu"]
    if (evidence.get("device") != report["device"]
            or evidence.get("torch") != report["torch"]
            or prior_gpu.get("actual_SM70") is not True
            or any(prior_gpu.get(key) != gpu[key] for key in (
                "name", "compute_capability", "total_memory_bytes", "torch_cuda_build_version"))):
        raise Blocked("Integration check Torch/CUDA/device identity differs; rerun it on this device")
    return {"path": str(path.resolve()), "sha256": sha256(path),
            "status": "passed", "completed_cases": evidence["completed_cases"],
            "finished_at_utc": evidence.get("finished_at_utc"),
            "source_and_library_hashes_match": True,
            "device_identity_scope": "index,name,compute_capability,total_memory,Torch_and_CUDA_build",
            "executed_inside_benchmark": False}



def quantizer_prerequisite(path: Path | None, report: dict) -> dict:
    """Require the independent byte/original-FP32 route check for native RNE."""
    if path is None or not path.is_file():
        raise Blocked("cuda_rne requires the passing --quantizer-check-report from check_quantize_cuda.py")
    from tools.mxfp4_training.check_quantize_cuda import source_hashes as quantizer_sources
    evidence = json.loads(path.read_text())
    if (evidence.get("status") != "passed" or evidence.get("passed") is not True
            or evidence.get("gpu_executed") is not True
            or evidence.get("all_attempted_work_synchronized") is not True
            or evidence.get("source_unchanged_during_check") is not True
            or not evidence.get("cases")
            or evidence.get("completed_cases") != len(evidence["cases"])
            or not all(case.get("passed") is True for case in evidence["cases"])):
        raise Blocked("Native RNE prerequisite is not a complete passing GPU check")
    if (evidence.get("source_sha256") != quantizer_sources()
            or evidence.get("library", {}).get("sha256") != report["library"]["sha256"]):
        raise Blocked("Native RNE checker source/library differs from this benchmark")
    prior_gpu, gpu = evidence.get("gpu", {}), report["gpu"]
    if (evidence.get("device") != report["device"] or evidence.get("torch") != report["torch"]
            or any(prior_gpu.get(key) != gpu[key] for key in
                   ("name", "compute_capability", "total_memory_bytes", "torch_cuda_build_version"))):
        raise Blocked("Native RNE checker device/Torch identity differs")
    return {"path": str(path.resolve()), "sha256": sha256(path), "status": "passed",
            "completed_cases": evidence["completed_cases"], "source_and_library_hashes_match": True,
            "library": evidence["library"], "executed_inside_benchmark": False}


def error_context(actual: np.ndarray, reference: np.ndarray) -> dict:
    if actual.shape != reference.shape:
        raise AssertionError("Output shape differs from the FP32 reference")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise AssertionError("Non-finite GEMM output; no timing result will be published")
    a, r = actual.astype(np.float64), reference.astype(np.float64)
    delta = np.abs(a - r)
    squared_reference = float(np.square(r).sum())
    squared_error = float(np.square(delta).sum())
    nrmse = math.sqrt(squared_error / squared_reference) if squared_reference else (
        0. if squared_error == 0 else None)
    return {"values": int(a.size), "all_finite": True,
            "exact_mismatches": int(np.count_nonzero(a != r)),
            "max_abs_error": float(delta.max(initial=0)),
            "max_relative_error": float((delta / np.maximum(np.abs(r), 1e-12)).max(initial=0)),
            "relative_denominator_floor": 1e-12, "nrmse": nrmse,
            "context_only_not_accuracy_gate": True}


def exact_outputs(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape or a.dtype != np.float32 or b.dtype != np.float32:
        raise AssertionError("Packed CUDA outputs must have identical shapes and FP32 dtype")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError("Packed CUDA outputs must be finite for this Gaussian input")
    mismatches = int(np.count_nonzero(a != b))
    if mismatches:
        raise AssertionError(f"Packed CUDA backends differ at {mismatches} output elements")
    return {"values": int(a.size), "exact_mismatches": 0,
            "comparison": "bitwise_except_signed_zero", "all_finite": True}


@dataclass
class ExecutionState:
    attempted: bool = False
    returned: bool = False
    completed: bool = False
    pending: bool = False

    def begin(self):
        self.attempted = self.pending = True

    def synchronize(self, stream):
        stream.synchronize()
        self.completed, self.pending = True, False

    def fields(self) -> dict:
        return {"gpu_executed": True if self.completed else (None if self.attempted else False),
                "gpu_work_attempted": self.attempted, "gpu_work_call_returned": self.returned,
                "gpu_work_synchronized": self.completed,
                "all_attempted_work_synchronized": not self.pending}


@dataclass
class Job:
    name: str
    scope: str
    function: Callable
    output_dtype: str
    is_gemm: bool = True
    event_ms: list[float] = field(default_factory=list)
    wall_ms: list[float] = field(default_factory=list)

    def summary(self, useful_flops: int) -> dict:
        event, wall = statistics.median(self.event_ms), statistics.median(self.wall_ms)
        return {"name": self.name, "scope": self.scope, "output_dtype": self.output_dtype,
                "primary_clock": "synchronized_perf_counter", "median_ms": wall,
                "useful_tflops": useful_flops / (wall * 1e9) if self.is_gemm else None,
                "cuda_event_median_ms": event, "synchronized_wall_median_ms": wall,
                "cuda_event_useful_tflops": useful_flops / (event * 1e9) if self.is_gemm else None,
                "synchronized_wall_useful_tflops": useful_flops / (wall * 1e9) if self.is_gemm else None,
                "cuda_event_round_ms": self.event_ms, "synchronized_wall_round_ms": self.wall_ms,
                "function_allocations_and_validation_included": True,
                "pytorch_caching_allocator_reuse_allowed": True,
                "timed_output_float32_conversion": False}


def benchmark(args, report: dict, device, stream, state: ExecutionState) -> None:
    def execute(function):
        state.begin()
        result = function()
        state.returned = True
        state.synchronize(stream)
        return result

    generator = torch.Generator(device=device).manual_seed(args.seed)
    a, b = execute(lambda: (torch.randn((args.m, args.k), device=device, dtype=torch.float32,
                                       generator=generator),
                            torch.randn((args.n, args.k), device=device, dtype=torch.float32,
                                       generator=generator)))
    report["input"]["generated"] = True
    if not execute(lambda: bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())):
        raise AssertionError("Synthetic Gaussian input unexpectedly contains non-finite values")
    ah, bh = execute(lambda: (a.to(torch.float16), b.to(torch.float16)))
    report["memory"]["after_resident_inputs"] = memory_snapshot(device)

    quantizer = quantize_mx_cuda_rne if args.quantizer_backend == "cuda_rne" else quantize_mx_torch

    def quantize_both():
        return quantizer(a), quantizer(b)  # Both functions use RNE by contract/default.

    def quantize_and_gemm(backend, diagnostic=None):
        qa, qb = quantize_both()
        return gemm_packed(qa, qb, backend=backend, certified_info=diagnostic)

    jobs = [
        Job("torch_fp32_matmul", "resident_fp32_A_at_B_rows_transpose", lambda: a @ b.T, "torch.float32"),
        Job("torch_fp16_compute", "preconverted_fp16_inputs_matmul_fp16_output", lambda: ah @ bh.T, "torch.float16"),
        Job("torch_fp16_casts_and_matmul", "fp32_to_fp16_both_inputs_then_matmul_fp16_output",
            lambda: a.to(torch.float16) @ b.to(torch.float16).T, "torch.float16"),
        Job("mxfp4_quantize_both", "fp32_to_packed_codes_and_scales_both_operands",
            quantize_both, "uint8_codes_and_scales", False),
        Job("mxfp4_cuda_quantize_and_gemm", "fp32_to_mxfp4_then_baseline_packed_gemm_fp32_output",
            lambda: quantize_and_gemm("cuda"), "torch.float32"),
        Job("mxfp4_cuda_tiled_quantize_and_gemm", "fp32_to_mxfp4_then_tiled_packed_gemm_fp32_output",
            lambda: quantize_and_gemm("cuda_tiled"), "torch.float32"),
    ]
    certified_name = "mxfp4_cuda_certified_quantize_and_gemm"
    if args.with_certified:
        jobs.append(Job(certified_name,
            "fp32_to_mxfp4_then_certified_create_prepare_certificate_gemm_scale_destroy_sync_fp32_output",
            lambda: quantize_and_gemm("cuda_certified"), "torch.float32"))

    def capture(job, result):
        if not job.is_gemm:
            host = []
            for packed, shape in zip(result, ((args.m, args.k), (args.n, args.k))):
                packed.validate()
                if packed.logical_shape != shape or packed.codes.device != device:
                    raise AssertionError("Quantizer returned the wrong logical shape or device")
                codes, scales = packed.codes.cpu().numpy().copy(), packed.scales.cpu().numpy().copy()
                if np.any(scales == 255):
                    raise AssertionError("Finite quantized input must not contain NaN E8M0 scales")
                host.append((codes, scales))
            return host
        if result.device != device or tuple(result.shape) != (args.m, args.n) or str(result.dtype) != job.output_dtype:
            raise AssertionError(f"Unexpected output device, shape or dtype from {job.name}")
        # No .float() GPU conversion is hidden in an FP16 timing path.
        array = result.cpu().numpy().copy()
        if not np.isfinite(array).all():
            raise AssertionError(f"Non-finite output from {job.name}")
        return array

    host_results = {}
    for job in jobs:
        if job.name == certified_name:
            diagnostic = {}
            report["native_certified_workspace_allocated"] = None
            result = execute(lambda: quantize_and_gemm("cuda_certified", diagnostic))
            report["native_certified_workspace_allocated"] = True
            report["certified_diagnostic_untimed_initial"] = diagnostic
        else:
            result = execute(job.function)
        host_results[job.name] = execute(lambda: capture(job, result))
        del result
    full_exact = exact_outputs(host_results["mxfp4_cuda_quantize_and_gemm"],
                               host_results["mxfp4_cuda_tiled_quantize_and_gemm"])
    reference = host_results["torch_fp32_matmul"]
    report["validation"] = {
        "passed_before_timing": True, "all_input_and_output_values_finite": True,
        "packed_cuda_vs_tiled_all_outputs": full_exact,
        "rounding": "nearest_ties_to_even", "same_resident_fp32_inputs_for_every_call": True,
        "errors_vs_torch_fp32_context_source": "untimed_initial_outputs",
        "errors_vs_torch_fp32_context": {job.name: error_context(host_results[job.name], reference)
                                         for job in jobs if job.is_gemm},
        "fp16_output_note": "ordinary Torch FP16 matmul rounds its output to FP16; not the native FP32-output cuBLAS primitive",
    }
    if args.with_certified:
        report["validation"]["packed_certified_vs_baseline_all_outputs"] = exact_outputs(
            host_results[certified_name], host_results["mxfp4_cuda_quantize_and_gemm"])
    for _ in range(args.warmup):
        for job in jobs:
            result = execute(job.function)
            del result

    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    state.begin()
    start.record(stream)
    stop.record(stream)
    stop.synchronize()  # Initialize timing events before the measured rounds.
    state.completed, state.pending = True, False
    report["memory"]["before_timed_rounds"] = memory_snapshot(device)
    torch.cuda.reset_peak_memory_stats(device)
    final_results = {}
    for round_index in range(args.rounds):
        for offset in range(len(jobs)):
            job = jobs[(round_index + offset) % len(jobs)]
            state.begin()
            wall_start = time.perf_counter()
            start.record(stream)
            for iteration in range(args.iters):
                result = job.function()
                state.returned = True
                if iteration + 1 < args.iters:
                    del result
            stop.record(stream)
            stop.synchronize()
            wall_end = time.perf_counter()
            state.completed, state.pending = True, False
            event_ms = start.elapsed_time(stop) / args.iters
            wall_ms = (wall_end - wall_start) * 1000. / args.iters
            if not (math.isfinite(event_ms) and math.isfinite(wall_ms) and event_ms > 0 and wall_ms > 0):
                raise RuntimeError("Invalid CUDA event or synchronized wall interval")
            job.event_ms.append(event_ms)
            job.wall_ms.append(wall_ms)
            if round_index + 1 == args.rounds:
                final_results[job.name] = execute(lambda: capture(job, result))
            del result
        print(f"Completed timing round {round_index + 1}/{args.rounds}", flush=True)

    # These are the actual final timed outputs, copied after both endpoints.
    final_exact = exact_outputs(final_results["mxfp4_cuda_quantize_and_gemm"],
                                final_results["mxfp4_cuda_tiled_quantize_and_gemm"])
    if args.with_certified:
        report["validation"]["final_timed_certified_vs_baseline_all_outputs"] = exact_outputs(
            final_results[certified_name], final_results["mxfp4_cuda_quantize_and_gemm"])
    repeat_checks = {}
    for job in jobs:
        if not job.is_gemm:
            continue
        actual, expected = final_results[job.name], host_results[job.name]
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise AssertionError(f"Output shape/dtype changed during timing for {job.name}")
        mismatches = int(np.count_nonzero(actual != expected))
        if mismatches:
            raise AssertionError(f"Final timed {job.name} output changed at {mismatches} elements")
        repeat_checks[job.name] = {"values": int(actual.size), "output_dtype": str(actual.dtype),
                                  "comparison": "exact_same_dtype_except_signed_zero",
                                  "exact_mismatches": 0}
    for actual, expected in zip(final_results["mxfp4_quantize_both"], host_results["mxfp4_quantize_both"]):
        if any(not np.array_equal(x, y) for x, y in zip(actual, expected)):
            raise AssertionError("RNE packed codes/scales changed for unchanged FP32 input")
    report["validation"].update(passed_after_timing=True,
                                final_timed_packed_cuda_vs_tiled_all_outputs=final_exact,
                                final_timed_rne_codes_and_scales_exact=True,
                                final_timed_vs_initial_all_outputs=repeat_checks,
                                final_timed_errors_vs_torch_fp32_context={
                                    job.name: error_context(final_results[job.name],
                                                           final_results["torch_fp32_matmul"])
                                    for job in jobs if job.is_gemm})
    state.synchronize(stream)
    report["memory"]["after_timed_rounds"] = memory_snapshot(device)
    report["timings"] = [job.summary(2 * args.m * args.n * args.k) for job in jobs]
    report["native_certified_capi_benchmarked"] = args.with_certified


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-report", type=Path, required=True)
    parser.add_argument("--quantizer-backend", choices=("torch", "cuda_rne"), default="torch")
    parser.add_argument("--quantizer-check-report", type=Path,
                        help="required independent byte/gradient check when selecting cuda_rne")
    parser.add_argument("--with-certified", action="store_true",
                        help="also measure the explicit synchronous certified wrapper, including context lifetime")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a new evidence file")
    if not args.device.startswith("cuda:") or not args.device[5:].isdigit():
        parser.error("device must name an explicit CUDA index, for example cuda:0")
    if not all(1 <= d <= 4096 for d in (args.m, args.n, args.k)):
        parser.error("each matrix dimension must be in [1, 4096]")
    if not (3 <= args.rounds <= 9 and 1 <= args.iters <= 20 and 0 <= args.warmup <= 3):
        parser.error("require rounds=3..9, iters=1..20, warmup=0..3")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("seed must be in [0, 2**32 - 1]")
    return args


def main() -> int:
    args = parse_args()
    padded_k = (args.k + 31) // 32 * 32
    # Deliberately conservative admission budget for the unfused quantizer's
    # FP32/int64/bool temporaries, input/output lifetimes and allocator slack.
    budget = (128 * max(args.m, args.n) * padded_k + 8 * (args.m + args.n) * padded_k
              + 12 * args.m * args.n + 64 * 1024 * 1024)
    if args.with_certified:
        # Conservatively cover normalized FP16 operands, metadata and workspace
        # even when the selected certified library uses metadata-only fused GEMM.
        budget += 4 * (args.m + args.n) * padded_k + 8 * 1024 * 1024
    report = {
        "schema_version": 1, "purpose": "SM70 PyTorch quantization and forward GEMM cost",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pending", "passed": False, "device": args.device,
        "quantizer_backend": args.quantizer_backend, "certified_job_enabled": args.with_certified,
        "python": platform.python_version(), "numpy": np.__version__,
        "torch": None if torch is None else torch.__version__, "source_sha256": source_hashes(),
        "shape": {"M": args.m, "N": args.n, "logical_K": args.k, "padded_K": padded_k},
        "input": {"distribution": "synthetic_gaussian", "mean": 0., "stddev": 1.,
                  "seed": args.seed, "generation_device": "cuda", "generated": False,
                  "dtype": "torch.float32",
                  "real_training_tensor": False},
        "warmup_per_job": args.warmup, "rounds": args.rounds, "iterations_per_round": args.iters,
        "backend_order": "rotating_each_round", "same_resident_inputs_reused": True,
        "useful_flops": 2 * args.m * args.n * args.k,
        "tflops_formula": "2*M*N*logical_K/(median_ms*1e9); quantize-only has no TFLOPS",
        "included_in_quantized_total": ["both_FP32_input_validations", "both_quantizations",
            "quantizer_temporaries_and_packed_output_allocations", "packed_validation",
            "Python_and_ctypes_dispatch", "GEMM_output_allocation", "packed_GEMM"],
        "excluded_from_times": ["context_and_library_initialization", "input_generation",
            "initial_preconversion_for_fp16_compute_only", "external_integration_check",
            "correctness_copies_and_comparisons", "timing_event_initialization", "untimed_certified_diagnostic"],
        "minimum_free_bytes_after_context": MINIMUM_FREE_BYTES,
        "reserve_bytes": RESERVE_BYTES, "estimated_peak_increment_budget_bytes": budget,
        "maximum_budget_fraction_of_free_memory": .75,
        "memory": {}, "timings": [], "native_certified_capi_benchmarked": False,
        "native_certified_workspace_allocated": False,
        "memory_accounting_note": ("Torch allocator peaks omit CUDA context/cuBLAS internal allocations; device free-memory snapshots include them. Packed baseline/tiled C APIs have no separately allocated native workspace. "
            + ("Optional certified totals include create/metadata or normalized inputs/native workspace, GEMM, destruction and synchronization."
               if args.with_certified else "No certified workspace is created or measured.")),
        "full_training_executed": False, "llm_convergence_validated": False,
        "cpu_fallback": False, "gpu_services_stopped": False, "clock_or_power_changes": False,
    }
    state = ExecutionState()
    exit_code = 1
    try:
        if torch is None:
            raise Blocked("PyTorch is not installed")
        if not torch.cuda.is_available():
            raise Blocked("CUDA is unavailable in this PyTorch process; no CPU fallback was attempted")
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        report["gpu"] = {
            "name": properties.name, "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "actual_SM70": (properties.major, properties.minor) == (7, 0),
            "actual_V100": (properties.major, properties.minor) == (7, 0) and "V100" in properties.name,
            "torch_cuda_build_version": torch.version.cuda,
        }
        if (properties.major, properties.minor) != (7, 0):
            raise Blocked("This benchmark requires actual SM70 hardware")
        report["library"] = library_metadata()
        report["prerequisite"] = prerequisite(args.check_report, report)
        if args.quantizer_backend == "cuda_rne":
            report["quantizer_prerequisite"] = quantizer_prerequisite(args.quantizer_check_report, report)
        report["memory"]["after_context_and_library"] = memory_snapshot(device)
        free = report["memory"]["after_context_and_library"]["device_free_bytes"]
        if free < MINIMUM_FREE_BYTES:
            raise Blocked("Requires at least 1 GiB free after context/library initialization; no services are stopped")
        if budget > free - RESERVE_BYTES or budget > free * 3 // 4:
            raise Blocked("Conservative quantization allocation budget exceeds the guarded free-memory allowance")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        report["torch_matmul_settings"] = {"allow_tf32": False,
            "allow_fp16_reduced_precision_reduction": False,
            "fp16_output_dtype": "torch.float16", "output_float32_cast_timed": False,
            "fp16_accumulation_note": "Requested no reduced-precision reductions; standard Torch FP16 output still rounds to FP16."}
        stream = torch.cuda.Stream(device=device)
        report["nondefault_stream"] = stream.cuda_stream != torch.cuda.default_stream(device).cuda_stream
        if not report["nondefault_stream"]:
            raise AssertionError("A nondefault stream was not obtained")
        with torch.inference_mode(), torch.cuda.stream(stream), torch.autocast(device_type="cuda", enabled=False):
            benchmark(args, report, device, stream, state)
        report["source_unchanged_during_benchmark"] = source_hashes() == report["source_sha256"]
        if not report["source_unchanged_during_benchmark"]:
            raise RuntimeError("Sources changed during measurement; rerun against a fixed snapshot")
        if sha256(Path(report["library"]["path"])) != report["library"]["sha256"]:
            raise RuntimeError("CUDA library changed during measurement")
        report["status"], report["passed"], exit_code = "passed", True, 0
    except Blocked as error:
        report.update(status="blocked", error_type=type(error).__name__, error=str(error))
        exit_code = 2
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        report["timings"] = []  # Never publish speed results after a failed gate.
        exit_code = 1
    finally:
        report.update(state.fields())
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"status": report["status"], "passed": report["passed"],
                          "gpu_executed": report["gpu_executed"], "output": str(args.output)}), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
