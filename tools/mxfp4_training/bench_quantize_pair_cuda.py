"""Measure two singles versus the paired CUDA RNE wrapper in separate processes.

Run the checker first, then launch this program in singles,pair,pair,singles
order for each size, retaining all reports. Each process validates every byte
against the independent NumPy oracle before timing and checks the ACTUAL last
timed result afterwards. Input SHA256 values permit cross-process comparison.
All allocations, validation, both native kernels and finite-flag host waits
belong in each measured call. No GEMM FLOPS or training speedup is claimed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from tools.mxfp4_training import check_quantize_pair_cuda as pair_check
from tools.mxfp4_training import check_quantize_cuda as check
from tools.mxfp4_training.bench_quantize_cuda import ClockSamples, input_oracle, input_hash, memory_sample

torch = check.torch
if torch is not None:
    ops = check.ops


def source_hashes():
    root = Path(__file__).resolve().parent
    return {**pair_check.source_hashes(), **{name: check.sha256(root / name) for name in
            ("bench_quantize_pair_cuda.py", "bench_quantize_cuda.py")}}


def validate_prerequisite(path, library, gpu):
    data = json.loads(path.read_text())
    check.need(data.get("status") == "passed" and data.get("passed") is True,
               "Paired checker did not pass")
    check.need(data.get("gpu_executed") is True and data.get("all_attempted_work_synchronized") is True,
               "Paired checker lacks completed GPU execution")
    check.need(data.get("source_unchanged_during_check") is True and data.get("library_unchanged_during_check") is True,
               "Paired checker lacks source/library stability checks")
    cases = data.get("cases", [])
    check.need(len(cases) >= 47 and all(row.get("passed") is True for row in cases)
               and data.get("expected_cases") == data.get("completed_cases") == len(cases),
               "Paired checker is incomplete")
    check.need(data.get("source_sha256") == pair_check.source_hashes(), "Checker used different source bytes")
    check.need(data.get("library", {}).get("sha256") == library["sha256"], "Checker used another library")
    check.need(data["library"].get("compile_evidence_sha256") == library["compile_evidence_sha256"],
               "Checker used another compile manifest")
    check.need(data.get("torch") == torch.__version__, "Checker used another Torch runtime")
    check.need(data.get("device") == str(gpu["device"]), "Checker used another device index")
    for key in ("name", "compute_capability", "total_memory_bytes", "multiprocessor_count", "torch_cuda_build_version"):
        check.need(data.get("gpu", {}).get(key) == gpu[key], "Checker GPU metadata differs: " + key)
    return {"path": str(path.resolve()), "sha256": check.sha256(path), "passed": True,
            "completed_cases": len(cases), "library_and_sources_match": True,
            "same_gpu_model_and_runtime": True,
            "physical_gpu_identity": "checker records model properties; benchmark separately records NVML UUID"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-evidence", type=Path, required=True)
    parser.add_argument("--check-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("singles", "pair"), required=True)
    parser.add_argument("--size", type=int, choices=(1024, 4096), default=1024)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Output exists; use a new evidence path")
    if not args.device.startswith("cuda:") or not args.device[5:].isdigit(): parser.error("Require explicit CUDA index")
    if not (0 <= args.seed < 2**32 and 1 <= args.warmup <= 100 and 3 <= args.rounds <= 20 and 1 <= args.iters <= 100):
        parser.error("Require uint32 seed, warmup1..100, rounds3..20, iters1..100")
    report = {"schema_version": 1, "purpose": "complete paired-input CUDA RNE wrapper latency",
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "pending", "passed": False,
              "gpu_executed": False, "device": args.device, "mode": args.mode, "seed": args.seed,
              "shape_A": [args.size, args.size], "shape_B": [args.size, args.size],
              "warmup": args.warmup, "rounds": args.rounds, "iterations_per_round": args.iters,
              "python": platform.python_version(), "numpy": np.__version__,
              "torch": None if torch is None else torch.__version__, "source_sha256": source_hashes(),
              "timing_scope": "both input quantizations, allocations, validation, native calls and finite-flag host waits",
              "event_scope": "stream interval spans complete wrappers and host submission gaps; not kernel-only",
              "excluded_from_timing": ["input_generation", "NumPy_oracle", "byte_checks", "clock_sampling", "source_hashing"],
              "same_resident_inputs_every_call": True, "packed_tensor_cache": False,
              "independent_process_runs_in_this_report": 1, "repeated_rounds_are_independent_samples": False,
              "recommended_cross_process_order": ["singles", "pair", "pair", "singles"],
              "clock_or_power_changes": False, "full_training_executed": False,
              "flops_or_tflops": None, "cpu_fallback": False, "raw_rounds": [], "memory": []}
    code = 1; stream = clocks = None; attempted = pending = synchronized = False
    try:
        if torch is None or not torch.cuda.is_available(): raise check.Blocked("CUDA-enabled Torch is unavailable")
        device = torch.device(args.device); torch.cuda.set_device(device)
        gpu = pair_check.gpu_metadata(device); gpu["device"] = str(device); report["gpu"] = gpu
        if not gpu["actual_V100"]: raise check.Blocked("Require an actual V100 SM70")
        _library, library = check.library_metadata(args.compile_evidence); report["library"] = library
        report["checker_prerequisite"] = validate_prerequisite(args.check_report, library, gpu)
        before = memory_sample(device, "before_inputs"); report["memory"].append(before)
        # Two resident FP32 inputs plus live packed outputs and allocator slack.
        # This is a guarded estimate, not an allocator-enforced cap.
        budget = 32 * args.size**2 + 128 * 1024**2
        report["memory_guard"] = {"conservative_allocation_budget_bytes": budget,
                                  "minimum_free_bytes": check.MINIMUM_FREE,
                                  "minimum_reserve_bytes": check.RESERVE_BYTES,
                                  "budget_is_preflight_estimate_not_allocator_cap": True}
        if before["free_bytes"] < check.MINIMUM_FREE or budget > before["free_bytes"] - check.RESERVE_BYTES:
            raise check.Blocked("Insufficient guarded device memory")
        torch.set_num_threads(2)
        stream = torch.cuda.Stream(device=device)
        check.need(stream.cuda_stream != torch.cuda.default_stream(device).cuda_stream, "Require nondefault stream")
        report["nondefault_stream"] = True
        clocks = ClockSamples(int(args.device[5:])); report["clock_monitor"] = clocks.metadata
        clocks.snapshot("before_inputs")
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False), torch.cuda.stream(stream):
            generator = torch.Generator(device=device).manual_seed(args.seed)
            attempted = pending = True
            a = torch.randn((args.size, args.size), device=device, dtype=torch.float32, generator=generator)
            b = torch.randn((args.size, args.size), device=device, dtype=torch.float32, generator=generator)
            stream.synchronize(); synchronized = True; pending = False
            expected_a, hash_a = input_oracle(a); expected_b, hash_b = input_oracle(b)
            expected = (expected_a, expected_b)
            report["input_sha256_before"] = {"A_fp32_bytes": hash_a, "B_fp32_bytes": hash_b}
            report["oracle_scope"] = "every input value, independent NumPy quantize_mx in 64-row chunks"

            def operation():
                if args.mode == "pair":
                    return ops.quantize_pair_mx_cuda_rne(a, b)
                return ops.quantize_mx_cuda_rne(a), ops.quantize_mx_cuda_rne(b)

            pending = True; initial = operation(); stream.synchronize(); pending = False
            report["initial_all_packed_bytes"] = pair_check.compare_pair(initial, expected)
            del initial
            for _ in range(args.warmup):
                pending = True; warm = operation(); del warm
            stream.synchronize(); pending = False
            report["memory"].append(memory_sample(device, "after_warmup"))
            if report["memory"][-1]["free_bytes"] < check.RESERVE_BYTES:
                raise check.Blocked("Post-warmup free memory fell below reserve")
            clocks.snapshot("after_warmup")
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record(stream); end_event.record(stream); end_event.synchronize()
            for index in range(args.rounds):
                stream.synchronize()
                started = time.perf_counter()
                start_event.record(stream)
                pending = True
                for iteration in range(args.iters):
                    final = operation()
                    if iteration + 1 < args.iters:
                        del final
                end_event.record(stream); end_event.synchronize(); pending = False
                wall_ms = (time.perf_counter() - started) * 1000 / args.iters
                event_ms = start_event.elapsed_time(end_event) / args.iters
                check.need(all(math.isfinite(x) and x > 0 for x in (wall_ms, event_ms)), "Invalid timing observation")
                report["raw_rounds"].append({"index": index, "mode": args.mode,
                                             "wall_ms_per_pair": wall_ms, "event_ms_per_pair": event_ms})
                if index + 1 < args.rounds:
                    del final
                print(f"Completed {args.mode} round {index + 1}/{args.rounds}", flush=True)
            clocks.snapshot("after_all_timed_rounds")
            # Keep the actual last output alive; no surrogate quantization call
            # replaces the final result generated inside the measured interval.
            report["actual_last_timed_all_packed_bytes"] = pair_check.compare_pair(final, expected)
            report["input_sha256_after"] = {"A_fp32_bytes": input_hash(a), "B_fp32_bytes": input_hash(b)}
            check.need(report["input_sha256_after"] == report["input_sha256_before"], "Inputs changed")
            report["memory"].append(memory_sample(device, "after_validation"))
            report["timing"] = {}
            for field in ("wall_ms_per_pair", "event_ms_per_pair"):
                values = [row[field] for row in report["raw_rounds"]]
                report["timing"][field] = {"median": statistics.median(values), "minimum": min(values),
                                             "maximum": max(values), "all_rounds": values}
        check.need(source_hashes() == report["source_sha256"], "Sources changed during benchmark")
        check.need(check.sha256(library["path"]) == library["sha256"], "Library changed during benchmark")
        check.need(check.sha256(args.compile_evidence) == library["compile_evidence_sha256"], "Compile evidence changed")
        report.update(status="passed", passed=True, gpu_executed=True,
                      source_and_library_unchanged=True, actual_last_timed_output_checked=True)
        code = 0
    except check.Blocked as error:
        report.update(status="blocked", error_type=type(error).__name__, error=str(error)); code = 2
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
    finally:
        if stream is not None and pending:
            try:
                stream.synchronize(); pending = False; synchronized = True
            except Exception as error:
                report["cleanup_error"] = str(error)
        if clocks is not None:
            clocks.close()
        report.update(gpu_executed=True if synchronized else (None if attempted else False),
                      gpu_work_attempted=attempted, all_attempted_work_synchronized=not pending,
                      finished_at_utc=datetime.now(timezone.utc).isoformat())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False); handle.write("\n")
        print(json.dumps({key: report[key] for key in ("status", "passed", "gpu_executed")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
