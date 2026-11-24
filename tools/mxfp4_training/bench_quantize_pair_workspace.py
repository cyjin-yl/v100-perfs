"""Compare two single RNE calls with paired RNE through one persistent C2 workspace.

Run the existing paired-quantizer and workspace GPU checkers against the SAME
C2/RNE build first. Run this program in separate singles,pair,pair,singles
processes at each size. The default is 10 warmup plus 5x10 measured calls.
Every measured call freshly quantizes both resident FP32 inputs and includes
allocations, validation, flag waits, record_stream, Python/ctypes and the entire
certified GEMM pipeline. One-time context construction/closing is separate.

The actual first and last batch-timed outputs AND packed operands stay alive
until full comparison after timing. No substitute quantization or GEMM is used.
A NumPy-packed-input tiled GPU baseline supplies all expected output values;
an independent CPU ordered-MX32 oracle checks 32x32 selected output values.
This is synthetic inference-mode operator evidence, not an LLM training step.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.mxfp4_training import check_quantize_cuda as quant_check
from tools.mxfp4_training import check_quantize_pair_cuda as pair_check
from tools.mxfp4_training.check_certified_workspace import source_hashes as workspace_sources
from tools.mxfp4_training.check_torch_certified import library_metadata as certified_library_metadata
from tools.mxfp4_training.bench_quantize_cuda import ClockSamples, input_hash, input_oracle, memory_sample
from tools.mxfp4_training.bench_quantize_pair_cuda import validate_prerequisite as pair_prerequisite
from tools.mxfp4_training.reference import MXTensor, gemm_mx

torch = quant_check.torch
if torch is not None:
    ops = quant_check.ops
    from tools.mxfp4_training.certified_workspace import CertifiedWorkspace


def utc():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    root = Path(__file__).resolve().parent
    return {**workspace_sources(), **pair_check.source_hashes(), **{
        name: quant_check.sha256(root / name) for name in (
            "bench_quantize_cuda.py", "bench_quantize_pair_cuda.py", "bench_quantize_pair_workspace.py")}}


def workspace_prerequisite(path, library, gpu):
    path = path.expanduser().resolve(strict=True)
    contents = path.read_bytes()
    data = json.loads(contents)
    cases = data.get("cases", [])
    required = {
        "explicit_lifecycle_and_first_output", "odd_logical_k_padding_and_native_shape_binding",
        "initial_diagnostic_failure_rolls_back_native_allocation", "mutated_inputs_certificate_and_nan_fallback",
        "preflight_failures_recover_without_submission", "wrong_host_thread_never_enters_native_context",
        "busy_stream_reject_then_completed_stream_switch", "queued_calls_tensor_lifetime_and_close_wait",
        "partial_native_submission_failure_closes_and_waits", "context_body_exception_cleans_up",
        "capture_rejection_and_post_capture_recovery",
    }
    names = [case.get("name") for case in cases]
    good = (data.get("status") == "passed" and data.get("passed") is True
            and data.get("gpu_executed") is True and data.get("all_attempted_work_synchronized") is True
            and data.get("source_unchanged_during_check") is True
            and data.get("library_unchanged_during_check") is True
            and data.get("completed_cases") == len(cases) and required.issubset(names)
            and len(names) == len(set(names)) and all(case.get("passed") is True for case in cases))
    if not good:
        raise quant_check.Blocked("The complete independent workspace GPU checker must pass first")
    if data.get("source_sha256") != workspace_sources():
        raise quant_check.Blocked("Workspace checker source bytes differ")
    if (data.get("library", {}).get("sha256") != library["sha256"]
            or data["library"].get("compile_evidence_sha256") != library["compile_evidence_sha256"]
            or data["library"].get("compiled_variant") != 2
            or data.get("device") != gpu["device"] or data.get("torch") != torch.__version__):
        raise quant_check.Blocked("Workspace checker library, manifest, C2 variant, device or Torch differs")
    for key in ("name", "compute_capability", "total_memory_bytes", "torch_cuda_build_version"):
        if data.get("gpu", {}).get(key) != gpu[key]:
            raise quant_check.Blocked("Workspace checker GPU metadata differs: " + key)
    return {"path": str(path), "sha256": hashlib.sha256(contents).hexdigest(),
            "completed_cases": len(cases), "source_and_library_hashes_match": True,
            "compile_manifest_matches": True, "executed_inside_benchmark": False}


class ActivitySamples:
    """Read-only boundary observations; no CUDA context is created for mapping.

    ClockSamples uses Driver cuInit/cuDeviceGetPCIBusId, then NVML PCI lookup.
    CUDA-visible ordinals are never treated as nvidia-smi device indices.
    Process PID sets are compared without assuming host/container PID equality.
    """
    class Utilization(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    def __init__(self, index):
        self.clocks = ClockSamples(index)
        self.metadata = self.clocks.metadata
        self.metadata.update(
            activity_scope="NVML and compute-process boundary snapshots outside timing",
            initial_idle_required=True, process_identity_scope="PID namespace exposed by nvidia-smi",
            continuous_exclusivity_proven=False)

    def snapshot(self, label):
        row = self.clocks.snapshot(label)
        if self.metadata["available"]:
            try:
                lib, handle = self.clocks.nvml, self.clocks.handle
                function = lib.nvmlDeviceGetUtilizationRates
                function.argtypes = [ctypes.c_void_p, ctypes.POINTER(self.Utilization)]
                function.restype = ctypes.c_int
                usage = self.Utilization()
                status = function(handle, ctypes.byref(usage))
                row["utilization_nvml_status"] = status
                if status == 0:
                    row.update(gpu_utilization_percent=int(usage.gpu), memory_utilization_percent=int(usage.memory))
                command = ["nvidia-smi", "--id=" + self.metadata["gpu_uuid"],
                           "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits"]
                result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
                if result.returncode:
                    raise RuntimeError("nvidia-smi compute query failed: " + result.stderr.strip()[:512])
                processes = []
                for fields in csv.reader(io.StringIO(result.stdout)):
                    if not fields or not any(value.strip() for value in fields):
                        continue
                    if len(fields) != 3 or fields[0].strip() != self.metadata["gpu_uuid"]:
                        raise RuntimeError("Compute query returned an unexpected GPU identity or row")
                    pid = int(fields[1].strip())
                    if pid <= 0:
                        raise RuntimeError("Compute query returned an invalid PID")
                    memory = fields[2].strip()
                    processes.append({"pid": pid, "used_gpu_memory_mib": int(memory) if memory.isdigit() else None})
                row.update(compute_query_succeeded=True, compute_processes=processes,
                           compute_pids=sorted({item["pid"] for item in processes}))
            except Exception as error:
                row.update(compute_query_succeeded=False, activity_query_error=str(error))
        return row

    def require_idle(self, row):
        if (not self.metadata["available"] or row.get("utilization_nvml_status") != 0
                or row.get("compute_query_succeeded") is not True):
            raise quant_check.Blocked("Cannot establish the required initial NVML/process idle boundary")
        if row["gpu_utilization_percent"] != 0 or row["compute_pids"]:
            raise quant_check.Blocked("Selected GPU is not idle before this process creates a CUDA context")
        self.metadata["initial_idle_observed"] = True

    @staticmethod
    def require_one_process(row, previous=None):
        if row.get("compute_query_succeeded") is not True or len(row.get("compute_pids", [])) != 1:
            raise quant_check.Blocked("Expected exactly one visible compute process at the timing boundary")
        if previous is not None and row["compute_pids"] != previous:
            raise quant_check.Blocked("GPU compute-process identity changed across timing boundaries")
        return row["compute_pids"]

    def close(self):
        self.clocks.close()
        self.metadata["limitation"] = (
            "Initially idle and stable single-process boundary observations do not prove continuous exclusivity; "
            "short-lived competing work and within-round clock changes can be missed.")


def exact_array(actual, expected):
    quant_check.need(actual.shape == expected.shape and actual.dtype == expected.dtype == np.float32,
                     "FP32 output shape/dtype differs")
    quant_check.need(bool(np.isfinite(actual).all() and np.isfinite(expected).all()),
                     "Gaussian benchmark output is nonfinite")
    count = int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)))
    quant_check.need(count == 0, f"Output bit comparison failed at {count} values")
    quant_check.need(not np.any(actual.view(np.uint32)[actual == 0] != 0), "Output has noncanonical negative zero")
    return {"values": int(actual.size), "bit_mismatches": 0, "all_finite": True,
            "canonical_positive_zero": True}


def all_output_bits(tensor, expected):
    quant_check.need(tuple(tensor.shape) == expected.shape and tensor.dtype == torch.float32,
                     "Actual timed output tensor shape/dtype differs")
    digest = hashlib.sha256()
    for row in range(0, len(expected), 64):
        actual = np.ascontiguousarray(quant_check.host_array(tensor[row:row + 64]))
        exact_array(actual, expected[row:row + 64])
        digest.update(actual.tobytes())
    return {"values": int(expected.size), "bit_mismatches": 0, "all_finite": True,
            "canonical_positive_zero": True, "actual_fp32_sha256": digest.hexdigest(),
            "all_output_values_compared": True}


def cpu_output_sample(expected_packed, full_output, seed):
    rng = np.random.default_rng(seed ^ 0x50C2A11)
    rows = []
    reduced = []
    for packed in expected_packed:
        selected = {0, packed.logical_shape[0] // 2, packed.logical_shape[0] - 1}
        while len(selected) < min(32, packed.logical_shape[0]):
            selected.add(int(rng.integers(packed.logical_shape[0])))
        indices = np.asarray(sorted(selected), dtype=np.int64)
        rows.append(indices)
        reduced.append(MXTensor(np.ascontiguousarray(packed.codes[indices]),
                                np.ascontiguousarray(packed.scales[indices]),
                                (len(indices), packed.logical_shape[1])))
    cpu = gemm_mx(*reduced)
    result = exact_array(full_output[np.ix_(*rows)], cpu)
    result.update(a_rows=rows[0].tolist(), b_rows=rows[1].tolist(), all_outputs_cpu_oracle=False,
                  scope="independent CPU ordered-MX32 on 32x32 selected output values")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-evidence", type=Path, required=True)
    parser.add_argument("--pair-check-report", type=Path, required=True)
    parser.add_argument("--workspace-check-report", type=Path, required=True)
    parser.add_argument("--expected-certified-variant", type=int, choices=(2,), default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("singles", "pair"), required=True)
    parser.add_argument("--size", type=int, choices=(1024, 4096), default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--warmup", type=int, choices=(10,), default=10)
    parser.add_argument("--rounds", type=int, choices=(5,), default=5)
    parser.add_argument("--iters", type=int, choices=(10,), default=10)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a new evidence path")
    if not args.device.startswith("cuda:") or not args.device[5:].isdigit() or not 0 <= args.seed < 2**32:
        parser.error("Require an explicit CUDA device ordinal and a uint32 seed")
    report = {
        "schema_version": 1, "purpose": "paired versus two-single RNE plus persistent C2 workspace end-to-end latency",
        "started_at_utc": utc(), "status": "pending", "passed": False, "gpu_executed": False,
        "device": args.device, "mode": args.mode, "size": args.size, "seed": args.seed,
        "shape": {"M": args.size, "N": args.size, "logical_K": args.size, "padded_K": args.size},
        "warmup": args.warmup, "rounds": args.rounds, "iterations_per_round": args.iters,
        "python": platform.python_version(), "numpy": np.__version__,
        "torch": None if torch is None else torch.__version__, "source_sha256": source_hashes(),
        "expected_certified_variant": 2, "context_lifecycle": "one_explicit_owner_thread_workspace_per_process",
        "context_create_destroy_in_timed_operation": False,
        "included_in_timed_operation": ["both_FP32_input_validations", "fresh_two_input_RNE_quantization",
            "packed_and_flag_allocations", "both_finite_flag_validation_and_host_waits", "record_stream",
            "packed_tensor_validation", "Python_and_ctypes_dispatch", "new_FP32_output_allocation",
            "native_preparation", "certificate", "GEMM", "output_scale_or_exact_fallback"],
        "excluded_from_timing": ["input_generation", "NumPy_quantization_oracle", "tiled_GPU_output_baseline",
            "CPU_output_oracle", "D2H_correctness_comparisons", "diagnostic_queries", "library_loading",
            "timing_event_initialization", "NVML_and_process_snapshots", "source_and_evidence_hashing",
            "separately_measured_one_time_context_create_and_close"],
        "event_scope": "stream interval includes host submission gaps and flag waits; not kernel-only",
        "input_distribution": "synthetic_standard_normal_FP32_GPU_generator", "same_resident_inputs_every_call": True,
        "packed_tensor_cache": False, "torch_caching_allocator_reuse_allowed": True,
        "finite_flag_D2H_transfers_per_operation": 1 if args.mode == "pair" else 2,
        "native_quantizer_launches_per_operation": 2,
        "recommended_cross_process_order": ["singles", "pair", "pair", "singles"],
        "independent_process_runs_in_this_report": 1, "repeated_rounds_are_independent_samples": False,
        "first_timed_output_policy": "retain first batch result and packed operands; validate after all batches",
        "retained_first_and_last_output_storage_policy": "must_be_distinct_while_both_are_alive",
        "intermediate_output_allocator_reuse_allowed": True, "extra_first_timed_probe": False,
        "real_training_tensor": False, "full_training_executed": False, "autograd_workspace_integration": False,
        "clock_or_power_changes": False, "gpu_services_stopped": False, "cpu_fallback": False,
        "tflops_formula": "2*M*N*logical_K/(end_to_end_median_ms*1e9)",
        "tflops_scope": "useful GEMM work divided by complete quantization-plus-workspace latency; not kernel peak",
        "raw_rounds": [], "memory": [], "gpu_activity_boundaries": [], "lifecycle_timing": {},
    }
    exit_code = 1
    device = stream = activity = workspace = None
    attempted = completed = pending = False
    timed_completed = 0
    try:
        if torch is None:
            raise quant_check.Blocked("CUDA-enabled PyTorch is unavailable")
        # MUST precede torch.cuda.is_available/set_device/get_device_properties:
        # a runtime lazy initialization may otherwise create our own context.
        activity = ActivitySamples(int(args.device[5:]))
        report["clock_monitor"] = activity.metadata
        activity.require_idle(activity.snapshot("before_any_torch_cuda_initialization"))
        if not torch.cuda.is_available():
            raise quant_check.Blocked("A real CUDA runtime is unavailable")
        device = torch.device(args.device)
        gpu = pair_check.gpu_metadata(device)
        gpu["device"] = str(device)
        report["gpu"] = gpu
        if not gpu["actual_V100"]:
            raise quant_check.Blocked("Require an actual V100 SM70")
        _, quant_library = quant_check.library_metadata(args.compile_evidence)
        library = certified_library_metadata(args.compile_evidence, 2, report["source_sha256"])
        quant_check.need(quant_library["sha256"] == library["sha256"]
                         and quant_library["compile_evidence_sha256"] == library["compile_evidence_sha256"],
                         "Quantizer and C2 workspace must use one identical build")
        library.update(context_lifecycle=report["context_lifecycle"],
                       quantizer_backend=quant_library["quantizer_backend"], tiled_variant=1)
        report["library"] = library
        report["pair_prerequisite"] = pair_prerequisite(args.pair_check_report, quant_library, gpu)
        report["workspace_prerequisite"] = workspace_prerequisite(args.workspace_check_report, library, gpu)
        torch.set_num_threads(2)
        with torch.cuda.device(device), torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            lib = ops._load_library(library["path"], "cuda_certified")
            native_bytes = ctypes.c_uint64()
            code = lib.mxfp4_certified_workspace_bytes(args.size, args.size, args.size, ctypes.byref(native_bytes))
            if code:
                raise ops._certified_error(lib, "workspace memory admission", code)
            before = memory_sample(device, "before_inputs")
            report["memory"].append(before)
            budget = 48 * args.size**2 + native_bytes.value + 256 * 1024**2
            report["memory_guard"] = {
                "conservative_allocation_budget_bytes": budget, "native_workspace_bytes": native_bytes.value,
                "minimum_free_bytes": quant_check.MINIMUM_FREE, "minimum_reserve_bytes": quant_check.RESERVE_BYTES,
                "budget_is_preflight_estimate_not_allocator_cap": True,
                "includes_retained_first_and_last_outputs_and_packed_operands": True}
            if before["free_bytes"] < quant_check.MINIMUM_FREE or budget > before["free_bytes"] - quant_check.RESERVE_BYTES:
                raise quant_check.Blocked("Insufficient guarded GPU memory for complete-wrapper comparison")
            stream = torch.cuda.Stream(device=device)
            quant_check.need(stream.cuda_stream != torch.cuda.default_stream(device).cuda_stream, "Require a nondefault stream")
            report["nondefault_stream"] = True
            torch.cuda.reset_peak_memory_stats(device)
            with torch.cuda.stream(stream):
                generator = torch.Generator(device=device).manual_seed(args.seed)
                attempted = pending = True
                report["gpu_activity_boundaries"].append({"stage": "input_generation_attempted", "utc": utc()})
                a = torch.randn((args.size, args.size), device=device, dtype=torch.float32, generator=generator)
                b = torch.randn((args.size, args.size), device=device, dtype=torch.float32, generator=generator)
                stream.synchronize()
                completed = True
                pending = False
                report["gpu_activity_boundaries"].append({"stage": "input_generation_synchronized", "utc": utc()})
                expected_a, hash_a = input_oracle(a)
                expected_b, hash_b = input_oracle(b)
                expected_packed = (expected_a, expected_b)
                report["input_sha256_before"] = {"A_fp32_bytes": hash_a, "B_fp32_bytes": hash_b}
                # Build the all-output baseline from the INDEPENDENT NumPy
                # quantization, not from either wrapper under comparison.
                pending = True
                oracle_device = tuple(ops.TorchMXTensor(torch.from_numpy(x.codes).to(device),
                    torch.from_numpy(x.scales).to(device), x.logical_shape) for x in expected_packed)
                baseline = ops.gemm_packed(*oracle_device, backend="cuda_tiled")
                stream.synchronize()
                pending = False
                expected_output = np.ascontiguousarray(quant_check.host_array(baseline))
                report["baseline_output_fp32_sha256"] = hashlib.sha256(expected_output.tobytes()).hexdigest()
                report["baseline_sampled_cpu_oracle"] = cpu_output_sample(expected_packed, expected_output, args.seed)
                report["baseline_scope"] = "full tiled1 GPU output from independently NumPy-quantized inputs"
                del baseline, oracle_device
                workspace = CertifiedWorkspace(args.size, args.size, args.size, device=device)
                started = time.perf_counter()
                workspace.__enter__()
                report["lifecycle_timing"]["context_create_and_initial_ABI_query_ms"] = (time.perf_counter() - started) * 1000
                report["workspace_bytes"] = workspace.workspace_bytes
                quant_check.need(workspace.workspace_bytes == native_bytes.value, "Workspace size changed after admission")

                def operation():
                    qa, qb = (ops.quantize_pair_mx_cuda_rne(a, b) if args.mode == "pair" else
                              (ops.quantize_mx_cuda_rne(a), ops.quantize_mx_cuda_rne(b)))
                    return qa, qb, workspace.gemm(qa, qb)

                pending = True
                initial = operation()
                stream.synchronize()
                pending = False
                report["initial_all_packed_bytes"] = pair_check.compare_pair(initial[:2], expected_packed)
                report["initial_all_outputs"] = all_output_bits(initial[2], expected_output)
                report["initial_certificate"] = workspace.info()
                quant_check.need(report["initial_certificate"]["accepted"] == 1
                                 and report["initial_certificate"]["reject_flags"] == 0,
                                 "The Gaussian performance input did not pass the C2 certificate")
                del initial
                for _ in range(args.warmup):
                    pending = True
                    warm = operation()
                    del warm
                stream.synchronize()
                pending = False
                report["memory"].append(memory_sample(device, "after_warmup"))
                if report["memory"][-1]["free_bytes"] < quant_check.RESERVE_BYTES:
                    raise quant_check.Blocked("Post-warmup free memory fell below reserve")
                pids = activity.require_one_process(activity.snapshot("after_warmup_before_timing"))
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(stream)
                end_event.record(stream)
                end_event.synchronize()
                first_timed = last_timed = None
                report["gpu_activity_boundaries"].append({"stage": "timed_batches_start", "utc": utc(),
                    "workspace_submitted_calls": workspace.submitted_calls})
                for index in range(args.rounds):
                    stream.synchronize()
                    pending = True
                    started = time.perf_counter()
                    start_event.record(stream)
                    for iteration in range(args.iters):
                        current = operation()
                        if index == 0 and iteration == 0:
                            first_timed = current
                        if index + 1 == args.rounds and iteration + 1 == args.iters:
                            last_timed = current
                        del current
                    end_event.record(stream)
                    end_event.synchronize()
                    elapsed = (time.perf_counter() - started) * 1000 / args.iters
                    event_ms = start_event.elapsed_time(end_event) / args.iters
                    pending = False
                    timed_completed += args.iters
                    quant_check.need(all(math.isfinite(x) and x > 0 for x in (elapsed, event_ms)), "Invalid timing result")
                    report["raw_rounds"].append({"index": index, "wall_ms_per_operation": elapsed,
                        "event_ms_per_operation": event_ms, "completed_timed_calls": timed_completed})
                    print(f"Completed {args.mode} C2 workspace round {index + 1}/{args.rounds}", flush=True)
                report["gpu_activity_boundaries"].append({"stage": "timed_batches_synchronized", "utc": utc(),
                    "workspace_submitted_calls": workspace.submitted_calls})
                activity.require_one_process(activity.snapshot("after_all_timed_batches"), pids)
                quant_check.need(first_timed is not None and last_timed is not None, "Missing actual timed results")
                quant_check.need(first_timed[2].data_ptr() != last_timed[2].data_ptr(), "Retained first output was aliased")
                report["retained_first_and_last_output_storage_distinct"] = True
                for label, actual in (("first", first_timed), ("last", last_timed)):
                    report[f"actual_{label}_timed_all_packed_bytes"] = pair_check.compare_pair(actual[:2], expected_packed)
                    report[f"actual_{label}_timed_all_outputs"] = all_output_bits(actual[2], expected_output)
                    report[f"actual_{label}_timed_output_checked"] = True
                info = workspace.info()
                calls = 1 + args.warmup + args.rounds * args.iters
                quant_check.need(workspace.submitted_calls == info["submitted_calls"] == calls
                                 and info["accepted"] == 1 and info["reject_flags"] == 0,
                                 "Actual final timed C2 submission metadata differs")
                report["actual_last_timed_certificate"] = info
                report["successful_workspace_submissions"] = calls
                report["actual_first_timed_workspace_submission"] = 1 + args.warmup + 1
                report["actual_last_timed_workspace_submission"] = calls
                report["input_sha256_after"] = {"A_fp32_bytes": input_hash(a), "B_fp32_bytes": input_hash(b)}
                quant_check.need(report["input_sha256_after"] == report["input_sha256_before"], "Resident input bytes changed")
                report["memory"].append(memory_sample(device, "after_full_output_verification"))
                wall = [row["wall_ms_per_operation"] for row in report["raw_rounds"]]
                events = [row["event_ms_per_operation"] for row in report["raw_rounds"]]
                median = statistics.median(wall)
                report["timing"] = {"primary_clock": "synchronized_perf_counter", "median_ms": median,
                    "synchronized_wall_round_ms": wall, "cuda_event_round_ms": events,
                    "cuda_event_median_ms": statistics.median(events),
                    "useful_end_to_end_tflops": 2 * args.size**3 / (median * 1e9)}
        exit_code = 0
    except BaseException as error:
        # Write failed/blocked evidence even after partial allocation/submission.
        # GPU cleanup below still runs while retained operand/output locals live.
        blocked = isinstance(error, quant_check.Blocked)
        report.update(status="blocked" if blocked else "failed", passed=False,
                      error_type=type(error).__name__, error=str(error))
        exit_code = 2 if blocked else 1
    finally:
        if workspace is not None:
            try:
                started = time.perf_counter()
                workspace.close()
                report["lifecycle_timing"]["context_close_and_final_wait_ms"] = (time.perf_counter() - started) * 1000
                report["workspace_closed"] = workspace.closed
            except BaseException as error:
                report.update(status="failed", passed=False, workspace_cleanup_error=str(error))
                exit_code = 1
        if stream is not None and attempted:
            try:
                stream.synchronize()
                pending = False
                report["gpu_activity_boundaries"].append({"stage": "final_cleanup_synchronized", "utc": utc()})
            except BaseException as error:
                report.update(status="failed", passed=False, cleanup_error=str(error))
                pending = True
                exit_code = 1
        if activity is not None:
            try:
                activity.close()
            except BaseException as error:
                report.update(status="failed", passed=False, monitor_cleanup_error=str(error))
                exit_code = 1
        if exit_code == 0:
            try:
                quant_check.need(source_hashes() == report["source_sha256"], "Source snapshot changed during benchmark")
                quant_check.need(quant_check.sha256(report["library"]["path"]) == report["library"]["sha256"], "Library changed")
                quant_check.need(quant_check.sha256(args.compile_evidence) == report["library"]["compile_evidence_sha256"],
                                 "Compile manifest changed")
                for key in ("pair_prerequisite", "workspace_prerequisite"):
                    quant_check.need(quant_check.sha256(report[key]["path"]) == report[key]["sha256"], "Checker evidence changed")
                manifest = json.loads(args.compile_evidence.read_text())
                job = next(item for item in manifest["jobs"] if item["name"] == "shared_library")
                quant_check.need(quant_check.sha256(args.compile_evidence.parent / job["log"]) == job["log_sha256"],
                                 "Shared-library compilation log changed")
                report.update(status="passed", passed=True, source_unchanged_during_benchmark=True,
                    library_unchanged_during_benchmark=True, compile_evidence_unchanged_during_benchmark=True,
                    checker_evidence_unchanged_during_benchmark=True, observed_process_boundaries_match=True)
            except BaseException as error:
                report.update(status="failed", passed=False, evidence_validation_error=str(error))
                exit_code = 1
        report.update(gpu_executed=True if completed else (None if attempted else False),
            gpu_work_attempted=attempted, all_attempted_work_synchronized=not pending,
            completed_timed_calls=timed_completed, finished_at_utc=utc())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps({key: report[key] for key in ("status", "passed", "gpu_executed", "completed_timed_calls")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
