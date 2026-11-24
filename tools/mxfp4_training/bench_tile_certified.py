"""Packed-input C2/C10/C11 benchmark with exact tile and output evidence.

Run the matching check_tile_certified.py first. Compare separate processes in
ABBA order with identical shape/mode/seed and actual device-input SHA hashes.
Every timed call includes preparation, certificate, fast/fallback dispatch and
output production; no diagnostic D2H or host gate is inserted in its timing.
Input quantization, H2D, allocations and context lifetime are outside timing.
This is synthetic operator evidence, not an LLM training throughput result.
The complete large output oracle runs on the GPU, using the unconditional
ordered-MX32 tiled path verified against full NumPy outputs by the prerequisite
checker. CPU work here is O((M+N)*K + M*N + 64*K), never full M*N*K GEMM.
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

from tools.mxfp4_training.check_tile_certified import (
    GuardedInputs, GuardedOutput, NativeContext, assert_diagnostics, exact_output,
    source_hashes, submit_tiled_oracle,
)
from tools.mxfp4_training.check_torch_cuda import Blocked, sha256
from tools.mxfp4_training.check_torch_certified import library_metadata
from tools.mxfp4_training.reference import E2M1_Q
from tools.mxfp4_training.tile_certified_fixtures import benchmark_fixture, expected_certificates

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from tools.mxfp4_training import torch_ops as ops


def check_prerequisite(path, report):
    data = json.loads(path.read_text())
    if (data.get("status") != "passed" or data.get("passed") is not True
            or data.get("gpu_executed") is not True
            or data.get("all_attempted_work_synchronized") is not True
            or data.get("source_unchanged_during_check") is not True
            or data.get("library_unchanged_during_check") is not True
            or data.get("completed_cases") != len(data.get("cases", []))
            or len(data.get("cases", [])) != 21
            or data.get("unconditional_tiled_oracle_verified_cases") != 21
            or not all(case.get("passed") is True for case in data["cases"])):
        raise Blocked("The complete matching 21-case GPU tile checker must pass first")
    for case in data["cases"]:
        oracle = case.get("unconditional_tiled_oracle_check", {})
        checked = oracle.get("all_outputs", {})
        if (oracle.get("passed") is not True
                or oracle.get("backend") != "mxfp4_gemm_tiled_cuda_unconditional"
                or oracle.get("independent_semantic_oracle") != "full_NumPy_ordered_MX32"
                or checked.get("bit_mismatches") != 0
                or checked.get("compared_values") != case["shape"][0] * case["shape"][1]
                or checked.get("output_guards_unchanged") is not True):
            raise Blocked("The unconditional tiled GPU oracle lacks full independent NumPy validation")
    for key in ("source_sha256", "gpu", "device", "torch", "expected_certified_variant"):
        if data.get(key) != report[key]:
            raise Blocked(f"Tile checker snapshot differs for {key}")
    if data.get("library", {}).get("sha256") != report["library"]["sha256"]:
        raise Blocked("Tile checker used another binary")
    return {"path": str(path.resolve()), "sha256": sha256(path),
            "completed_cases": 21, "unconditional_tiled_oracle_verified_cases": 21,
            "executed_inside_benchmark": False}


def sampled_cpu_oracle(fixture, baseline, seed):
    rng = np.random.default_rng(seed ^ 0x93E712)
    count = min(64, fixture.m * fixture.n)
    indices = {0, fixture.m * fixture.n - 1, (fixture.m // 2) * fixture.n + fixture.n // 2}
    while len(indices) < count:
        indices.add(int(rng.integers(0, fixture.m * fixture.n)))
    indices = sorted(indices)
    decoded = {}

    def row_q(side, row):
        key = (side, row)
        if key not in decoded:
            packed = (fixture.a if side == 0 else fixture.b)[row]
            q = np.empty(fixture.padded_k, dtype=np.int16)
            q[0::2], q[1::2] = E2M1_Q[packed & 15], E2M1_Q[packed >> 4]
            decoded[key] = q.astype(np.int32)
        return decoded[key]

    expected, actual = [], []
    for flat in indices:
        i, j = divmod(flat, fixture.n)
        a, b = row_q(0, i), row_q(1, j)
        value = np.float32(0)
        for block in range(fixture.padded_k // 32):
            start = block * 32
            # At most 32*12*12 = 4608, exactly represented in int32/FP64.
            dot = int(np.dot(a[start:start + 32], b[start:start + 32]))
            sa, sb = int(fixture.a_scales[i, block]), int(fixture.b_scales[j, block])
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                term = np.float32(np.nan if sa == 255 or sb == 255 else np.ldexp(float(dot), sa + sb - 256))
                value = np.add(value, term, dtype=np.float32)
        expected.append(value)
        actual.append(baseline[i, j])
    result = exact_output(np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32))
    result.update(sample_flat_indices=indices, all_output_cpu_oracle=False,
                  baseline_full_output_used_for_candidate_comparisons=True)
    return result


def run(args, report, device, stream):
    fixture = benchmark_fixture(args.mode, args.m, args.n, args.k, args.seed)
    decision = expected_certificates(fixture)
    report["independent_cpu_certificate"] = decision
    report["generated_input_sha256"] = fixture.hashes()
    lib = ops._load_library(report["library"]["path"], "cuda_certified")
    import ctypes
    workspace_bytes = ctypes.c_uint64()
    code = lib.mxfp4_certified_workspace_bytes(args.m, args.n, fixture.padded_k, ctypes.byref(workspace_bytes))
    if code:
        raise ops._certified_error(lib, "workspace size", code)
    inputs_bytes = sum(x.nbytes for x in fixture.arrays()) + 4 * 64
    outputs_bytes = 2 * (args.m * args.n + 32) * 4
    explicit_bytes = inputs_bytes + outputs_bytes + workspace_bytes.value
    free, total = torch.cuda.mem_get_info(device)
    report["allocation"] = {"explicit_gpu_bytes_bound": explicit_bytes,
                            "native_workspace_bytes": workspace_bytes.value,
                            "required_free_reserve_bytes": 512 * 1024**2,
                            "observed_free_before_allocation": free, "device_total_bytes": total,
                            "includes_two_guarded_fp32_outputs": True,
                            "excludes_driver_library_and_allocator_internal_overheads": True}
    if explicit_bytes > 2 * 1024**3 or free < explicit_bytes + 512 * 1024**2:
        raise Blocked("Insufficient free GPU memory for the bounded benchmark plus reserve")
    uploaded = GuardedInputs(fixture, device)
    output = GuardedOutput(fixture.m, fixture.n, device)
    baseline_output = GuardedOutput(fixture.m, fixture.n, device)
    report["gpu_executed"] = True
    report["full_output_oracle"] = {
        "backend": "mxfp4_gemm_tiled_cuda_unconditional", "executed_on": "GPU",
        "native_source_sha256": report["source_sha256"]["csrc/mxfp4_gemm_tiled.cu"],
        "all_output_values": args.m * args.n,
        "independent_full_NumPy_semantic_cases_in_prerequisite": 21,
        "large_case_independent_CPU_sample_count": min(64, args.m * args.n),
        "shares_exact_fallback_source_with_candidate": True,
        "CPU_full_M_N_K_GEMM_executed": False,
        "CPU_work_complexity": "O((M+N)*padded_K + M*N + 64*padded_K)",
        "baseline_output_data_ptr": baseline_output.view.data_ptr(),
        "candidate_output_data_ptr": output.view.data_ptr(),
        "distinct_output_storage": baseline_output.storage.data_ptr() != output.storage.data_ptr(),
    }
    if not report["full_output_oracle"]["distinct_output_storage"]:
        raise AssertionError("Candidate and oracle output storage unexpectedly alias")
    submit_tiled_oracle(lib, fixture, uploaded, baseline_output, stream)
    stream.synchronize()
    raw_baseline = baseline_output.storage.cpu().numpy()
    if (np.any(raw_baseline[:baseline_output.start] != baseline_output.sentinel)
            or np.any(raw_baseline[baseline_output.stop:] != baseline_output.sentinel)):
        raise AssertionError("Tiled baseline changed output guards")
    expected = raw_baseline[baseline_output.start:baseline_output.stop].view(np.float32).reshape(args.m, args.n).copy()
    report["baseline_sampled_cpu_oracle"] = sampled_cpu_oracle(fixture, expected, args.seed)
    del raw_baseline, baseline_output
    report["actual_input_sha256_before"] = uploaded.verify()
    started = time.perf_counter()
    context = NativeContext(lib, fixture, stream, args.expected_certified_variant)
    report["context_create_and_initial_info_ms"] = (time.perf_counter() - started) * 1000
    try:
        context.call(uploaded, output)
        global_info, tile_info = context.info()
        report["initial_diagnostics"] = assert_diagnostics(
            global_info, tile_info, decision, args.expected_certified_variant)
        report["initial_all_outputs"] = output.verify(expected)
        for _ in range(args.warmup):
            context.call(uploaded, output)
        stream.synchronize()
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record(stream)
        end_event.record(stream)
        end_event.synchronize()
        # Preserve the actual first timed output before repeated calls overwrite
        # it. This single-call probe has its own timing and is excluded from the
        # repeated-batch median; validation occurs after its end event/wall stop.
        started = time.perf_counter()
        start_event.record(stream)
        context.call(uploaded, output)
        end_event.record(stream)
        end_event.synchronize()
        first_wall = (time.perf_counter() - started) * 1000
        first_event = start_event.elapsed_time(end_event)
        if not all(math.isfinite(value) and value > 0 for value in (first_wall, first_event)):
            raise AssertionError("Invalid first timed probe elapsed time")
        report["first_timed_probe"] = {"synchronized_wall_ms": first_wall,
                                      "cuda_event_ms": first_event,
                                      "submitted_call": context.calls,
                                      "included_in_repeated_batch_median": False}
        if context.calls != 2 + args.warmup:
            raise AssertionError("First timed probe is not the expected native submission")
        report["actual_first_timed_all_outputs"] = output.verify(expected)
        global_info, tile_info = context.info()
        report["actual_first_timed_diagnostics"] = assert_diagnostics(
            global_info, tile_info, decision, args.expected_certified_variant)
        wall, events = [], []
        for round_index in range(args.rounds):
            stream.synchronize()
            started = time.perf_counter()
            start_event.record(stream)
            for _ in range(args.iters):
                context.call(uploaded, output)
            end_event.record(stream)
            end_event.synchronize()
            wall.append((time.perf_counter() - started) * 1000 / args.iters)
            events.append(start_event.elapsed_time(end_event) / args.iters)
            if not all(math.isfinite(value) and value > 0 for value in (wall[-1], events[-1])):
                raise AssertionError("Invalid benchmark elapsed time")
            print(f"completed variant {args.expected_certified_variant} round {round_index + 1}/{args.rounds}", flush=True)
        # Do not launch another GEMM to stand in for the ACTUAL final timed output.
        report["actual_last_timed_all_outputs"] = output.verify(expected)
        report["actual_last_timed_submitted_call"] = context.calls
        if (report["actual_first_timed_all_outputs"]["actual_tensor_data_ptr"]
                != report["actual_last_timed_all_outputs"]["actual_tensor_data_ptr"]
                or output.view.data_ptr() != report["full_output_oracle"]["candidate_output_data_ptr"]):
            raise AssertionError("Timed output checks refer to a different native output tensor")
        report["actual_first_and_last_timed_tensor_identity_verified"] = True
        report["surrogate_GEMM_launched_before_actual_timed_output_checks"] = False
        global_info, tile_info = context.info()
        report["actual_last_timed_diagnostics"] = assert_diagnostics(
            global_info, tile_info, decision, args.expected_certified_variant)
        calls = 2 + args.warmup + args.rounds * args.iters
        if context.calls != calls or global_info["submitted_calls"] != calls or tile_info["submitted_calls"] != calls:
            raise AssertionError("Actual final timed call count differs")
        report["actual_input_sha256_after"] = uploaded.verify()
        if report["actual_input_sha256_after"] != report["actual_input_sha256_before"]:
            raise AssertionError("Inputs changed during timing")
        median = statistics.median(wall)
        report["timing"] = {"primary_clock": "synchronized_perf_counter", "median_ms": median,
                            "synchronized_wall_round_ms": wall, "cuda_event_round_ms": events,
                            "useful_tflops": 2 * args.m * args.n * args.k / (median * 1e9)}
        report["successful_submitted_calls"] = calls
        report["timed_submitted_calls"] = 1 + args.rounds * args.iters
    finally:
        started = time.perf_counter()
        context.close()
        report["context_close_and_final_wait_ms"] = (time.perf_counter() - started) * 1000
        report["context_closed"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-evidence", type=Path, required=True)
    parser.add_argument("--check-report", type=Path, required=True)
    parser.add_argument("--expected-certified-variant", type=int, choices=(2, 10, 11), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("unit", "block", "sparse_span", "sparse_energy"), required=True)
    for name in ("m", "n", "k"):
        parser.add_argument("--" + name, type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; use a new evidence path")
    if (not args.device.startswith("cuda:") or not args.device[5:].isdigit()
            or not all(1 <= value <= 8192 for value in (args.m, args.n, args.k))
            or not (10 <= args.warmup <= 100 and 3 <= args.rounds <= 10 and 1 <= args.iters <= 20)
            or not 0 <= args.seed < 2**32):
        parser.error("Require explicit device, shapes1..8192, warmup10..100, rounds3..10, iters1..20, uint32seed")
    report = {"schema_version": 1, "status": "pending", "passed": False,
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "complete packed-input tile-local certified pipeline comparison",
              "expected_certified_variant": args.expected_certified_variant,
              "device": args.device, "python": platform.python_version(), "numpy": np.__version__,
              "torch": None if torch is None else torch.__version__, "source_sha256": source_hashes(),
              "shape": {"M": args.m, "N": args.n, "logical_K": args.k, "padded_K": (args.k + 31) // 32 * 32},
              "mode": args.mode, "seed": args.seed, "warmup": args.warmup, "rounds": args.rounds,
              "iterations_per_round": args.iters, "gpu_executed": False,
              "real_training_tensor": False, "full_training_executed": False,
              "input_distribution": "deterministic_NumPy_uniform_full_E2M1_nibbles_with_labeled_scales",
              "included_in_timing": ["Python_ctypes_dispatch_and_record_stream", "native_prepare",
                  "full_K_global_and_optional_tile_certificates", "fast_and_fallback_dispatch",
                  "GEMM_and_output_scale"],
              "excluded_from_timing": ["input_generation", "FP32_quantization", "H2D", "allocations",
                  "separately_measured_context_create_close", "correctness_D2H", "diagnostic_queries",
                  "CPU_certificate_oracle", "input_hash_checks", "timing_event_initialization"],
              "host_gate_in_timed_call": False, "tflops_formula": "2*M*N*logical_K/(median_ms*1e9)",
              "clock_or_power_changes": False, "gpu_services_stopped": False}
    exit_code, device = 1, None
    try:
        if torch is None or not torch.cuda.is_available():
            raise Blocked("An actual V100 CUDA runtime with PyTorch is required")
        device = torch.device(args.device)
        p = torch.cuda.get_device_properties(device)
        report["gpu"] = {"name": p.name, "compute_capability": [p.major, p.minor],
                         "total_memory_bytes": p.total_memory, "torch_cuda_build_version": torch.version.cuda}
        if (p.major, p.minor) != (7, 0) or "V100" not in p.name:
            raise Blocked("Require a real V100 SM70")
        report["library"] = library_metadata(args.compile_evidence, args.expected_certified_variant,
                                             report["source_sha256"])
        report["library"]["context_lifecycle"] = "reused_native_context_create_close_outside_timing"
        report["checker_prerequisite"] = check_prerequisite(args.check_report, report)
        with torch.cuda.device(device), torch.no_grad():
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                run(args, report, device, stream)
        report["source_unchanged_during_benchmark"] = source_hashes() == report["source_sha256"]
        report["library_unchanged_during_benchmark"] = sha256(Path(report["library"]["path"])) == report["library"]["sha256"]
        if not report["source_unchanged_during_benchmark"] or not report["library_unchanged_during_benchmark"]:
            raise AssertionError("Source or library changed during benchmark")
        report.update(status="passed", passed=True)
        exit_code = 0
    except Blocked as error:
        report.update(status="blocked", error=str(error))
        exit_code = 2
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        if device is not None and report["gpu_executed"]:
            try:
                torch.cuda.synchronize(device)
                report["all_attempted_work_synchronized"] = True
            except BaseException as error:
                report.update(status="failed", passed=False, synchronization_error=str(error))
                exit_code = 1
        report["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "gpu_executed": report["gpu_executed"],
                      "output": str(args.output)}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
