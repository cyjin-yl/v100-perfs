"""Measure complete FP32 quantization + certified GEMM with an explicit lifecycle.

Run the independent workspace and quantizer checkers first. Select per_call or
workspace in separate processes, then compare an ABBA ordering with identical
shape/seed and actual input hashes. Workspace creation and closing are measured
separately; repeated-call timings include quantization, validations, allocations,
Python/ctypes, preparation, certificate, GEMM and output scale/fallback. This is
a synthetic operator benchmark, not an autograd training-step benchmark.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.mxfp4_training.check_torch_cuda import Blocked, sha256
from tools.mxfp4_training.check_torch_certified import library_metadata
from tools.mxfp4_training.check_certified_workspace import source_hashes as checker_sources
from tools.mxfp4_training.bench_torch_cuda import quantizer_prerequisite

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from tools.mxfp4_training.certified_workspace import CertifiedWorkspace
    from tools.mxfp4_training.torch_ops import gemm_packed, quantize_mx_cuda_rne, quantize_mx_torch


def sources():
    root = Path(__file__).resolve().parent
    return {**checker_sources(), **{name: sha256(root / name) for name in
            ("bench_certified_workspace.py", "bench_torch_cuda.py")}}


def exact_bits(actual, expected):
    actual, expected = actual.cpu().numpy(), expected.cpu().numpy()
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.dtype != np.float32:
        raise AssertionError("Output shape/dtype differs")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError("Nonfinite output in Gaussian benchmark")
    mismatches = int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)))
    if mismatches or np.any(actual.view(np.uint32)[actual == 0] != 0):
        raise AssertionError(f"Full output bit comparison failed at {mismatches} values")
    return {"values": int(actual.size), "bit_mismatches": 0,
            "canonical_positive_zero": True, "all_finite": True}


def input_hash(a, b):
    return {name: hashlib.sha256(tensor.cpu().numpy().tobytes(order="C")).hexdigest()
            for name, tensor in (("a_fp32_bytes", a), ("b_fp32_bytes", b))}


def check_prerequisite(path, report):
    if not path.is_file():
        raise Blocked("Run the independent workspace checker first")
    data = json.loads(path.read_text())
    if (data.get("status") != "passed" or data.get("passed") is not True
            or data.get("gpu_executed") is not True
            or data.get("all_attempted_work_synchronized") is not True
            or data.get("source_unchanged_during_check") is not True
            or data.get("library_unchanged_during_check") is not True
            or len(data.get("cases", [])) < 11
            or data.get("completed_cases") != len(data["cases"])
            or not all(case.get("passed") is True for case in data["cases"])):
        raise Blocked("Workspace checker is not a complete passing GPU result")
    if data.get("source_sha256") != checker_sources():
        raise Blocked("Workspace checker used different sources")
    if (data.get("library", {}).get("sha256") != report["library"]["sha256"]
            or data.get("device") != report["device"] or data.get("torch") != report["torch"]
            or any(data.get("gpu", {}).get(key) != report["gpu"][key] for key in
                   ("name", "compute_capability", "total_memory_bytes", "torch_cuda_build_version"))):
        raise Blocked("Workspace checker library/device/runtime differs")
    return {"path": str(path.resolve()), "sha256": sha256(path),
            "completed_cases": data["completed_cases"], "source_and_library_hashes_match": True,
            "executed_inside_benchmark": False}


def benchmark(args, report, device, stream):
    generator = torch.Generator(device=device).manual_seed(args.seed)
    a = torch.randn((args.m, args.k), device=device, dtype=torch.float32, generator=generator)
    b = torch.randn((args.n, args.k), device=device, dtype=torch.float32, generator=generator)
    stream.synchronize()
    report["gpu_executed"] = True
    report["input_sha256_before"] = input_hash(a, b)
    quantizer = quantize_mx_cuda_rne if args.quantizer_backend == "cuda_rne" else quantize_mx_torch
    qa, qb = quantizer(a), quantizer(b)
    expected = gemm_packed(qa, qb, backend="cuda_tiled")
    stream.synchronize()
    del qa, qb
    workspace = None
    report["lifecycle_timing"] = {"context_create_and_initial_abi_query_ms": None,
                                   "context_close_and_final_wait_ms": None}
    try:
        if args.lifecycle == "workspace":
            start = time.perf_counter()
            workspace = CertifiedWorkspace(args.m, args.n, args.k, device=device)
            workspace.__enter__()
            report["lifecycle_timing"]["context_create_and_initial_abi_query_ms"] = (
                time.perf_counter() - start) * 1000
            report["workspace_bytes"] = workspace.workspace_bytes

        def operation(diagnostic=None):
            qa, qb = quantizer(a), quantizer(b)
            if workspace is not None:
                return workspace.gemm(qa, qb, certified_info=diagnostic)
            return gemm_packed(qa, qb, backend="cuda_certified", certified_info=diagnostic)

        initial_info = {}
        initial = operation(initial_info)
        stream.synchronize()
        report["initial_all_outputs"] = exact_bits(initial, expected)
        report["initial_certificate"] = initial_info
        if initial_info.get("accepted") != 1 or initial_info.get("reject_flags") != 0:
            raise AssertionError("Initial certificate did not accept this performance input")
        for _ in range(args.warmup):
            warm = operation()
            del warm
        stream.synchronize()
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record(stream)
        end_event.record(stream)
        end_event.synchronize()
        wall, events = [], []
        for round_index in range(args.rounds):
            stream.synchronize()
            started = time.perf_counter()
            start_event.record(stream)
            for iteration in range(args.iters):
                final = operation()
                if iteration + 1 < args.iters:
                    del final
            end_event.record(stream)
            end_event.synchronize()
            wall.append((time.perf_counter() - started) * 1000 / args.iters)
            events.append(start_event.elapsed_time(end_event) / args.iters)
            if not all(math.isfinite(value) and value > 0 for value in (wall[-1], events[-1])):
                raise AssertionError("Invalid elapsed interval")
            if round_index + 1 < args.rounds:
                del final
            print(f"Completed {args.lifecycle} round {round_index + 1}/{args.rounds}", flush=True)
        # Compare the ACTUAL final timed result, without recomputing GEMM.
        report["actual_last_timed_all_outputs"] = exact_bits(final, expected)
        report["actual_last_timed_vs_initial"] = exact_bits(final, initial)
        if workspace is not None:
            info = workspace.info()
            expected_calls = 1 + args.warmup + args.rounds * args.iters
            if (info["submitted_calls"] != expected_calls or info["accepted"] != 1
                    or info["reject_flags"] != 0):
                raise AssertionError("Actual last timed submission metadata differs")
            report["actual_last_timed_certificate"] = info
        else:
            report["actual_last_timed_certificate"] = None
            report["actual_last_timed_certificate_unavailable_reason"] = (
                "per-call contexts are destroyed inside measured operation; no timed diagnostic sync was added")
        report["input_sha256_after"] = input_hash(a, b)
        if report["input_sha256_after"] != report["input_sha256_before"]:
            raise AssertionError("Input bytes changed during timing")
        median_ms = statistics.median(wall)
        report["timing"] = {"primary_clock": "synchronized_perf_counter", "median_ms": median_ms,
                             "useful_tflops": 2 * args.m * args.n * args.k / (median_ms * 1e9),
                             "synchronized_wall_round_ms": wall, "cuda_event_round_ms": events,
                             "cuda_event_median_ms": statistics.median(events)}
        report["calls_in_one_lifecycle"] = 1 + args.warmup + args.rounds * args.iters
    finally:
        if workspace is not None:
            start = time.perf_counter()
            workspace.close()
            report["lifecycle_timing"]["context_close_and_final_wait_ms"] = (
                time.perf_counter() - start) * 1000
            report["workspace_closed"] = workspace.closed
    if workspace is not None:
        total_ms = sum(report["lifecycle_timing"].values())
        report["lifecycle_timing"]["separate_create_close_total_ms"] = total_ms
        report["lifecycle_timing"]["create_close_ms_per_all_completed_call"] = total_ms / report["calls_in_one_lifecycle"]
        report["lifecycle_timing"]["note"] = "Amortization denominator includes one validation, warmup, and measured calls; not a training-step rate"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-report", type=Path, required=True)
    parser.add_argument("--compile-evidence", type=Path, required=True)
    parser.add_argument("--quantizer-check-report", type=Path)
    parser.add_argument("--quantizer-backend", choices=("cuda_rne", "torch"), default="cuda_rne")
    parser.add_argument("--expected-certified-variant", type=int, default=2)
    parser.add_argument("--lifecycle", choices=("per_call", "workspace"), required=True)
    parser.add_argument("--device", default="cuda:0")
    for name in ("m", "n", "k"):
        parser.add_argument("--" + name, type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; use a new evidence path")
    if (not args.device.startswith("cuda:") or not args.device[5:].isdigit()
            or not all(1 <= value <= 4096 for value in (args.m, args.n, args.k))
            or not (10 <= args.warmup <= 100 and 3 <= args.rounds <= 9 and 1 <= args.iters <= 20)
            or not 0 <= args.seed < 2**32):
        parser.error("require explicit CUDA device, dimensions 1..4096, warmup 10..100, rounds 3..9, iters 1..20, uint32 seed")
    report = {"schema_version": 1, "purpose": "complete quantization plus certified packed GEMM lifecycle comparison",
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "pending", "passed": False,
              "device": args.device, "python": platform.python_version(), "numpy": np.__version__,
              "torch": None if torch is None else torch.__version__, "source_sha256": sources(),
              "lifecycle": args.lifecycle, "quantizer_backend": args.quantizer_backend,
              "shape": {"M": args.m, "N": args.n, "logical_K": args.k}, "seed": args.seed,
              "warmup": args.warmup, "rounds": args.rounds, "iterations_per_round": args.iters,
              "input_distribution": "synthetic_standard_normal_fp32_GPU_generator",
              "gpu_executed": False, "full_training_executed": False, "autograd_workspace_integration": False,
              "included_in_timed_operation": ["both_FP32_input_validations", "both_RNE_quantizations",
                  "packed_operand_allocations", "packed_validation", "Python_and_ctypes_dispatch",
                  "FP32_output_allocation", "preparation", "certificate", "GEMM", "output_scale_or_fallback"],
              "context_create_destroy_in_timed_operation": args.lifecycle == "per_call",
              "excluded_from_repeated_call_timing": ["input_generation", "correctness_comparisons",
                  "diagnostic_queries", "library_loading", "timing_event_initialization"],
              "tflops_formula": "2*M*N*logical_K/(median_ms*1e9)"}
    if args.lifecycle == "workspace":
        report["excluded_from_repeated_call_timing"].append("separately_measured_explicit_workspace_create_and_close")
    device = None
    try:
        if torch is None or not torch.cuda.is_available():
            raise Blocked("A real CUDA runtime with Torch is required")
        device = torch.device(args.device)
        p = torch.cuda.get_device_properties(device)
        report["gpu"] = {"name": p.name, "compute_capability": [p.major, p.minor],
                         "total_memory_bytes": p.total_memory, "torch_cuda_build_version": torch.version.cuda,
                         "actual_SM70": (p.major, p.minor) == (7, 0)}
        if not report["gpu"]["actual_SM70"] or "V100" not in p.name:
            raise Blocked("Require an actual V100 SM70")
        report["library"] = library_metadata(args.compile_evidence, args.expected_certified_variant,
                                             report["source_sha256"])
        report["library"]["context_lifecycle"] = args.lifecycle
        report["workspace_prerequisite"] = check_prerequisite(args.check_report, report)
        if args.quantizer_backend == "cuda_rne":
            report["quantizer_prerequisite"] = quantizer_prerequisite(args.quantizer_check_report, report)
        with torch.cuda.device(device):
            free, total = torch.cuda.mem_get_info(device)
            budget = 128 * max(args.m, args.n) * ((args.k + 31) // 32 * 32)
            budget += 24 * args.m * args.n + 8 * (args.m + args.n) * args.k + 256 * 1024**2
            report["memory_admission"] = {"free_bytes": free, "total_bytes": total,
                                           "conservative_increment_budget_bytes": budget,
                                           "reserve_bytes": 512 * 1024**2}
            if free < budget + 512 * 1024**2:
                raise Blocked("Insufficient free memory for conservative complete-wrapper budget")
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                benchmark(args, report, device, stream)
            stream.synchronize()
        if sources() != report["source_sha256"] or sha256(Path(report["library"]["path"])) != report["library"]["sha256"]:
            raise AssertionError("Source/library changed during benchmark")
        report.update(status="passed", passed=True, source_unchanged_during_benchmark=True,
                      library_unchanged_during_benchmark=True, all_attempted_work_synchronized=True)
    except Exception as error:
        report.update(status="blocked" if isinstance(error, Blocked) else "failed",
                      error_type=type(error).__name__, error=str(error))
    finally:
        if device is not None:
            try:
                torch.cuda.synchronize(device)
                report["all_attempted_work_synchronized"] = True
            except Exception as error:
                report.update(status="failed", passed=False, cleanup_error=str(error),
                              all_attempted_work_synchronized=False)
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps({key: report[key] for key in ("status", "passed", "gpu_executed")}))
    return 0 if report["passed"] else (2 if report["status"] == "blocked" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
