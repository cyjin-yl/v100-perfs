#!/usr/bin/env python3
"""One-shot, source-bound C2/C8/C9/C10 experiment. No remote operations/retries.

Compile first; every CUDA execution waits for the original B framework runner's
completion archive AND its exact same-UID flock. All child process groups are
created by this process. Absolute work cutoff: 2026-11-24 00:23:00 UTC.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time

NVCC = "/usr/local/cuda-12.8/bin/nvcc"
B_ROOT = "/workspace/user/mxfp4-sg-jincy26-261124-4abd5-20261123T204443Z-9fb4ed47"
B_COMMIT = "80d3a7c2bb0bb551a34b28bd7fda59978b6304ef"
B_SHA = "62378914e2b6ad7e34e3dd70736cb90aea07c962963494d97062d5f5738b0f57"
ABSOLUTE_CUTOFF = datetime(2026, 11, 24, 0, 23, tzinfo=timezone.utc)
BUDGET_SECONDS = 5400
MAX_ARCHIVE_BYTES = 1024**3
VARIANTS = (2, 8, 9, 10)
SHAPES = (2048, 4096, 8192)
MODES = {2048: "random_block_scales", 4096: "random_block_scales", 8192: "random_unit_scale"}
ORDER = (2, 8, 9, 10, 10, 9, 8, 2)
TRAINING = "tools/mxfp4_training/"
MAPPING = "evidence/benchmarks/sm70-packing/study/simulation/round3/cuda/fragment_mapping.hpp"
ARTIFACTS = {"shared_library": "libmxfp4_sm70.so", "standalone_harness": "check_mxfp4",
             "tiled_harness": "check_mxfp4_tiled", "certified_harness": "check_mxfp4_certified",
             "performance_benchmark": "bench_mxfp4"}


class Blocked(RuntimeError):
    pass


def need(value, message):
    if not value:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def regular(path, owned=False):
    info = Path(path).lstat()
    need(stat.S_ISREG(info.st_mode), "Not a nonsymlink regular file: " + str(path))
    if owned:
        need(info.st_uid == os.geteuid(), "File is not owned by this UID: " + str(path))
    return info


def document(path):
    regular(path)
    return json.loads(Path(path).read_text())


def write_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def safe_name(name):
    need(isinstance(name, str) and name and "\\" not in name, "Invalid archive member name")
    p = PurePosixPath(name)
    need(not p.is_absolute() and ".." not in p.parts and "." not in name.split("/")
         and str(p) == name.rstrip("/") and p.parts, "Unsafe archive member: " + name)
    return str(p)


def json_lines(path):
    regular(path)
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.startswith("{")]


def one(rows, kind):
    found = [r for r in rows if r.get("type") == kind]
    need(len(found) == 1, "Expected exactly one JSON record: " + kind)
    return found[0]


def fields(data, expected, label):
    for key, value in expected.items():
        need(data.get(key) == value and (not isinstance(value, bool) or data.get(key) is value),
             label + " field mismatch: " + key)


def extract(archive, binding_path, source):
    binding = document(binding_path)
    need(isinstance(binding, dict) and re.fullmatch(r"[0-9a-f]{40}", binding.get("source_commit", "")),
         "Missing source commit binding")
    need(re.fullmatch(r"[0-9a-f]{64}", binding.get("source_sha256", "")), "Bad archive hash binding")
    need(regular(archive).st_size <= MAX_ARCHIVE_BYTES and sha(archive) == binding["source_sha256"],
         "Archive size/hash mismatch")
    members = binding.get("members")
    need(isinstance(members, list) and members, "Missing bound members")
    expected = {}
    for member in members:
        name = safe_name(member["name"])
        need(name not in expected and re.fullmatch(r"[0-9a-f]{64}", member.get("sha256", "")),
             "Duplicate/invalid bound member")
        expected[name] = member["sha256"]
    source.mkdir(mode=0o700)
    seen, total = set(), 0
    with tarfile.open(archive, "r:*") as handle:
        entries = handle.getmembers()
        need(len(entries) <= 10000, "Excess archive entries")
        for item in entries:
            name = safe_name(item.name)
            need(name not in seen, "Duplicate archive path: " + name)
            seen.add(name)
            need(item.isdir() or item.isreg(), "Links/special archive entries are forbidden")
            need(not getattr(item, "sparse", None), "Sparse archive members are forbidden")
            if item.isdir():
                need(any(key.startswith(name + "/") for key in expected), "Unbound archive directory")
                continue
            need(name in expected and 0 <= item.size <= MAX_ARCHIVE_BYTES, "Unbound/oversized member")
            total += item.size
            need(total <= MAX_ARCHIVE_BYTES, "Expanded source exceeds limit")
        files = {safe_name(item.name) for item in entries if item.isreg()}
        need(files == set(expected), "Archive file set differs from exact source binding")
        for item in entries:
            if not item.isreg():
                continue
            name = safe_name(item.name)
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = handle.extractfile(item)
            need(stream is not None, "Unreadable member")
            with path.open("xb") as target:
                shutil.copyfileobj(stream, target)
            need(path.stat().st_size == item.size and sha(path) == expected[name], "Member hash mismatch: " + name)
    return binding, expected


def tree_hashes(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        regular(path)
        result[str(path.relative_to(root))] = sha(path)
    return result


def native_gate(rows, variant=None):
    fields(one(rows, "summary"), {"passed": True, "gpu_executed": True,
           "kernel_execution_completed": True, "actual_V100": True,
           "cases": 46, "values": 175429, "mismatches": 0}, "native summary")
    cases = [r for r in rows if r.get("type") == "correctness"]
    need(all(r.get("gpu_executed") is True and r.get("mismatches") == 0 and
             r.get("values") == r.get("M", -1) * r.get("N", -1) for r in cases), "Native case failed")
    if variant is None:
        need(len(cases) == 46 and sum(r["values"] for r in cases) == 175429, "Native counts mismatch")
        return {"passed": True, "gpu_executed": True, "calls": 46, "values": 175429, "mismatches": 0}
    fields(one(rows, "certified_summary"), {"passed": True, "gpu_executed": True,
           "certified_variant": variant, "full_domain_suite_passed": True, "extra_fixture_cases": 20,
           "extra_fixture_values": 230456, "context_reuse_calls": 3, "accepted_calls": 23,
           "fallback_calls": 46, "canonical_positive_zero_values": 2950, "mismatches": 0}, "certified summary")
    reuse = [r for r in rows if r.get("type") == "context_reuse"]
    need(len(reuse) == 3 and [r.get("accepted") for r in reuse] == [True, False, True]
         and [r.get("submitted_calls") for r in reuse] == [1, 2, 3]
         and all(r.get("passed") is True and r.get("mismatches") == 0 and r.get("values") == 15 for r in reuse),
         "Context reuse outputs not verified")
    count, values = 66, 405885
    if variant >= 8:
        extra = one(rows, "c8_fixture_summary")
        fields(extra, {"passed": True, "gpu_executed": True, "certified_variant": variant,
               "fixture_family": "c8_halfrow_contract", "cases": 9, "values": 180224,
               "c6_compatibility_cases": 3, "empty_K_cases": 1, "rejected_cases": 1,
               "accepted_calls": 8, "fallback_calls": 1, "mismatches": 0}, "halfrow contract")
        key = "nonempty_c8_loader_cases" if variant == 8 else "nonempty_derived_halfrow_loader_cases"
        need(extra.get(key) == 4, "Missing nonempty halfrow coverage")
        dispatch = [r for r in rows if r.get("type") == "c8_fixture_dispatch"]
        need(len(dispatch) == 9 and all(r.get("passed") is True and r.get("certified_variant") == variant
             and r.get("pointer_alignment_observed") is True for r in dispatch), "Dispatch fixture coverage missing")
        count += 9
        values += 180224
    if variant >= 9:
        fields(one(rows, "halfrow_boundary_summary"), {"passed": True, "gpu_executed": True,
               "certified_variant": variant, "cases": 1, "values": 16384, "mismatches": 0,
               "accepted": True, "K": 64, "canonical_positive_zero_values": 16384,
               "active_half_contains_signed_zero": True}, "two-stage cancellation")
        count += 1
        values += 16384
    need(len(cases) == count and sum(r["values"] for r in cases) == values, "Certified actual case/value counts mismatch")
    return {"passed": True, "gpu_executed": True, "certified_variant": variant,
            "calls": count + 3, "values": values + 45, "mismatches": 0,
            "kernel_path_instrumented": False}


def micro_gate(rows):
    selected = [r for r in rows if r.get("kind") == "mxfp4_half2_decode_microcheck"]
    need(len(selected) == 1, "Missing half2 microcheck result")
    fields(selected[0], {"passed": True, "status": "passed", "gpu_executed": True, "sm": 70,
           "patterns": 65536, "deltas": 13, "expected_cases": 851968, "cases": 851968,
           "half_values": 3407872, "nonzero_half_bitwise_comparisons": 2981888,
           "zero_half_sign_normalized_comparisons": 425984, "mismatched_half_values": 0,
           "first_mismatch": None, "comparison": "integer_bits_no_float_tolerance"}, "half2 exhaustive")
    return selected[0]


def metric(data, values, exact=True):
    fields(data, {"values": values, "nonfinite_mismatches": 0, "tolerance_failures": 0}, "Output comparison")
    if exact:
        fields(data, {"exact_mismatches": 0}, "Exact output comparison")


def benchmark_gate(rows, variant, size, fingerprints, mode, correctness_only=False):
    environment = one(rows, "environment")
    fields(environment, {"certified_variant": variant, "certified_compiled": True,
                         "actual_V100": True}, "Benchmark environment")
    fields(one(rows, "summary"), {"passed": True, "gpu_executed": True, "actual_V100": True,
           "completed_cases": 1, "correctness_only": correctness_only,
           "all_submitted_work_completed": True}, "Benchmark summary")
    initial = one(rows, "correctness")
    timed = None if correctness_only else one(rows, "benchmark")
    if correctness_only:
        need(not any(r.get("type") == "benchmark" for r in rows), "Correctness-only probe unexpectedly timed")
    fingerprint = None
    for record in ((initial,) if correctness_only else (initial, timed)):
        fields(record, {"M": size, "N": size, "logical_K": size, "padded_K": size,
               "input_mode": mode, "seed": 20261123, "gpu_executed": True,
               "actual_V100": True, "kernel_execution_completed": True,
               "all_submitted_work_completed": True}, "Benchmark case")
        current = record.get("input_fnv1a64")
        need(isinstance(current, str) and current, "Input hash missing")
        if fingerprint is None:
            fingerprint = current
        need(current == fingerprint, "Initial/timed packed inputs differ")
    need(initial.get("passed") is True, "Initial comparison failed")
    samples = initial.get("cpu_reference_values")
    need(isinstance(samples, int) and samples >= 256, "Insufficient CPU oracle sample coverage")
    for key in ("baseline_vs_cpu", "tiled_vs_cpu", "certified_vs_cpu"):
        metric(initial[key], samples)
    metric(initial["cublas_vs_cpu"], samples, exact=False)
    for key in ("tiled_vs_baseline_all_outputs", "certified_vs_baseline_all_outputs"):
        metric(initial[key], size * size)
    metric(initial["cublas_vs_baseline_all_outputs"], size * size, exact=False)
    fingerprint_key = mode + ":" + str(size)
    previous = fingerprints.get(fingerprint_key)
    need(previous is None or previous == fingerprint, "Same-mode/shape packed inputs differ")
    if correctness_only:
        certificate = initial["certificate"]
        fields(certificate, {"metadata_valid": True, "submitted_calls": 1}, "Boundary certificate")
        flags = certificate.get("reject_flags")
        need(type(flags) is int and 0 <= flags <= 15 and type(certificate.get("accepted")) is bool
             and certificate["accepted"] == (flags == 0), "Boundary certificate flags are inconsistent")
        fingerprints[fingerprint_key] = fingerprint
        return {"passed": True, "gpu_executed": True, "variant": variant, "shape": [size] * 3,
                "input_mode": mode, "input_fnv1a64": fingerprint, "certificate": certificate,
                "initial_all_output_values_per_comparison": size * size, "mismatches": 0,
                "performance_measured": False, "fast_performance_evidence": False}
    fields(timed, {"correctness_passed": True, "warmup_per_job": 100,
           "iterations_per_round": 10, "rounds": 5}, "Benchmark timing configuration")
    metric(timed["certified_post_timing_vs_baseline_all_outputs"], size * size)
    for cert, calls in ((initial["certificate"], 1), (timed["certificate_after_timing"], 151)):
        fields(cert, {"accepted": True, "metadata_valid": True, "reject_flags": 0,
               "submitted_calls": calls, "certificate_recomputed_each_call": True,
               "nan_scale_gate_pass": True, "span_gate_pass": True,
               "energy_gate_pass": True, "lattice_gate_pass": True}, "Accepted certificate")
    c0, c1 = dict(initial["certificate"]), dict(timed["certificate_after_timing"])
    c0.pop("submitted_calls"); c1.pop("submitted_calls")
    need(c0 == c1, "Certificate changed across identical inputs")
    jobs = timed.get("timings", [])
    expected = {"mxfp4_baseline", "mxfp4_tiled", "mxfp4_certified_total",
                "cublas_compute", "cublas_dequant", "cublas_total"}
    need(len(jobs) == 6 and {j.get("name") for j in jobs} == expected, "Benchmark backend set mismatch")
    for job in jobs:
        for key in ("round_ms", "cuda_event_round_ms", "synchronized_wall_round_ms"):
            values = job.get(key)
            need(isinstance(values, list) and len(values) == 5 and all(isinstance(v, (int, float))
                 and math.isfinite(v) and v > 0 for v in values), "Invalid timing array: " + key)
    certified = next(j for j in jobs if j["name"] == "mxfp4_certified_total")
    need(certified.get("primary_clock") == "synchronized_steady_clock", "Certified primary clock changed")
    fingerprints[fingerprint_key] = fingerprint
    return {"passed": True, "gpu_executed": True, "variant": variant, "shape": [size] * 3,
            "input_mode": mode, "input_fnv1a64": fingerprint, "certificate_accepted": True, "reject_flags": 0,
            "initial_all_output_values_per_comparison": size * size,
            "final_timed_certified_all_output_values": size * size, "mismatches": 0,
            "certified_timing": certified}


def torch_gate(path, variant, library, manifest):
    data = document(path)
    count = 119 if variant is None else 16
    fields(data, {"passed": True, "status": "passed", "gpu_executed": True,
           "completed_cases": count, "source_unchanged_during_check": True}, "Torch suite")
    cases = data.get("cases", [])
    need(len(cases) == count and all(c.get("passed") is True for c in cases), "Torch case coverage failed")
    metadata = data.get("library", {})
    fields(metadata, {"path": str(library), "sha256": sha(library), "bytes": library.stat().st_size}, "Torch library")
    if variant is not None:
        fields(metadata, {"compiled_variant": variant, "native_sources_match": True,
               "fragment_mapping_matches": True, "compile_evidence_sha256": sha(manifest)}, "Torch certified provenance")
    return {"passed": True, "gpu_executed": True, "cases": count, "numerical_case_gates_passed": True,
            "library_sha256": sha(library), "compiled_variant": variant}


class Runner:
    def __init__(self, args, run):
        self.args, self.run = args, run
        self.results, self.source = run / "results", run / "source"
        self.results.mkdir(mode=0o700)
        self.start = time.monotonic()
        self.deadline = min(self.start + BUDGET_SECONDS,
                            self.start + (ABSOLUTE_CUTOFF - datetime.now(timezone.utc)).total_seconds())
        self.active = None
        self.lock_fd = None
        self.expected = None
        self.binaries = {}
        self.report = {"schema_version": 1, "purpose": "C9/C10 source-bound native experiment",
                       "status": "running", "passed": False, "started_at_utc": utc(),
                       "run_directory": str(run), "python_execution_path": args.python,
                       "nvcc": NVCC, "global_budget_seconds": BUDGET_SECONDS,
                       "absolute_work_cutoff_utc": ABSOLUTE_CUTOFF.isoformat(),
                       "node_deadline_utc": "2026-11-24T00:24:00+00:00",
                       "cleanup_reserve_seconds": 15, "stages": [], "input_fingerprints": {},
                       "variants": list(VARIANTS), "shapes": list(SHAPES), "per_shape_variant_order": list(ORDER),
                       "benchmark_input_modes": {str(k): v for k, v in MODES.items()},
                       "correctness_only_boundary_probe": {"shape": [8192] * 3, "input_mode": "random_block_scales",
                                                           "variant": 2, "accept_or_reject_allowed": True},
                       "seed": 20261123, "warmup": 100, "iters": 10, "rounds": 5,
                       "framework_run": args.framework_run, "original_B_root": B_ROOT,
                       "gpu_executed": False, "automatic_retries": False,
                       "training_smoke_executed": False, "full_training_executed": False,
                       "runner_sha256": sha(Path(__file__))}
        self.env = os.environ.copy()
        self.env.pop("PYTHONHOME", None)
        self.env.pop("PYTHONPYCACHEPREFIX", None)
        self.env["OMP_NUM_THREADS"] = "2"
        self.env["MKL_NUM_THREADS"] = "2"
        self.env["PYTHONDONTWRITEBYTECODE"] = "1"
        self.env["CUDA_HOME"] = "/usr/local/cuda-12.8"
        self.env["PATH"] = "/usr/local/cuda-12.8/bin:" + self.env.get("PATH", "")
        self.env["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + self.env.get("LD_LIBRARY_PATH", "")
        self.env["PYTHONPATH"] = str(self.source)

    def save(self):
        write_json(self.results / "completion.json", self.report)

    def block(self, name, reason, gpu=False):
        row = {"name": name, "status": "blocked", "passed": False, "executed": False,
               "gpu_stage": gpu, "gpu_executed": False, "reason": reason, "recorded_at_utc": utc()}
        self.report["stages"].append(row)
        self.save()
        return False

    def cleanup(self):
        process = self.active
        if process is None:
            return
        # Only our start_new_session process group. Never broad pkill/proc scans.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        self.active = None

    def stage(self, name, argv, cap, gate=None, gpu=False, library=None):
        left = self.deadline - time.monotonic() - 15
        if left <= 0:
            return self.block(name, "Global/absolute deadline leaves no stage budget", gpu)
        if gpu and self.lock_fd is None:
            return self.block(name, "Framework completion/serialization lock not acquired", True)
        folder = self.results / "stages" / name
        folder.mkdir(parents=True, mode=0o700)
        timeout = min(cap, left)
        env = self.env.copy()
        if library is not None:
            env["MXFP4_CUDA_LIBRARY"] = str(library)
        row = {"name": name, "argv": [str(a) for a in argv], "cwd": str(self.source),
               "status": "running", "passed": False, "executed": False, "gpu_stage": gpu,
               "gpu_executed": False, "started_at_utc": utc(), "maximum_stage_seconds": cap,
               "effective_timeout_seconds": timeout, "timed_out": False, "returncode": None,
               "stdout": str(folder / "stdout.log"), "stderr": str(folder / "stderr.log")}
        self.report["stages"].append(row)
        self.save()
        start = time.monotonic()
        try:
            self.integrity()
            if library is not None:
                row["library_sha256_before"] = sha(library)
            if gpu and str(argv[0]) in self.binaries:
                row["binary_sha256_before"] = sha(argv[0])
            with (folder / "stdout.log").open("xb") as stdout, (folder / "stderr.log").open("xb") as stderr:
                process = subprocess.Popen([str(a) for a in argv], cwd=self.source, env=env,
                                           stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                           start_new_session=True)
                self.active = process
                row.update(executed=True, pid=process.pid, process_group=process.pid,
                           gpu_executed=None if gpu else False)
                self.save()
                try:
                    row["returncode"] = process.wait(timeout=timeout)
                    self.cleanup()  # Includes remaining children after the owned leader exited.
                except subprocess.TimeoutExpired:
                    row["timed_out"] = True
                    self.cleanup()
                    row["returncode"] = process.returncode
                    raise RuntimeError("Stage deadline expired; owned process group terminated")
            rows = json_lines(folder / "stdout.log")
            if gpu:
                evidence = [r.get("gpu_executed") for r in rows if "gpu_executed" in r]
                row["gpu_executed"] = True if True in evidence else (False if evidence and all(v is False for v in evidence) else None)
            if row["returncode"] != 0:
                reason = "Process exited with code " + str(row["returncode"])
                if gpu and (row["gpu_executed"] is False or row["returncode"] in (2, 77)):
                    raise Blocked(reason + "; CUDA/prerequisite gate did not pass")
                raise RuntimeError(reason)
            if gate is not None:
                row["validation"] = gate(rows)
            self.integrity()
            if library is not None:
                row["library_sha256_after"] = sha(library)
            if gpu and str(argv[0]) in self.binaries:
                row["binary_sha256_after"] = sha(argv[0])
            row.update(status="passed", passed=True)
        except (Exception, KeyboardInterrupt) as error:
            self.cleanup()
            row.update(status="blocked" if isinstance(error, Blocked) else "failed", passed=False,
                       error_type=type(error).__name__, error=str(error))
            if isinstance(error, KeyboardInterrupt):
                self.deadline = time.monotonic()
        finally:
            for key, path in (("binary_sha256_after", argv[0] if gpu and str(argv[0]) in self.binaries else None),
                              ("library_sha256_after", library)):
                if path is not None and Path(path).is_file():
                    row[key] = sha(path)
            row.update(finished_at_utc=utc(), wall_seconds=time.monotonic() - start)
            row["evidence_sha256"] = tree_hashes(folder)
            write_json(folder / "stage.json", row)
            self.save()
            print(json.dumps({"stage": name, "status": row["status"], "returncode": row["returncode"]}), flush=True)
        return row["passed"]

    def integrity(self):
        if self.expected is not None:
            need(tree_hashes(self.source) == self.expected, "Frozen source tree changed")
        for filename, digest in self.binaries.items():
            regular(filename)
            need(sha(filename) == digest, "Compiled artifact changed: " + filename)

    def build_gate(self, variant):
        build = self.run / ("build-c" + str(variant))
        data = document(build / "compile-evidence.json")
        fields(data, {"schema_version": 2, "passed": True, "gpu_executed": False,
               "certified_enabled": True, "certified_variant": variant, "tiled_variant": 1}, "Compilation manifest")
        need("release 12.8" in data.get("nvcc_version", ""), "Compiler is not CUDA 12.8")
        expected_native = {k[len(TRAINING):]: v for k, v in self.expected.items() if k.startswith(TRAINING + "csrc/")}
        need(data.get("source_sha256") == expected_native, "Compiled source hashes differ from archive")
        need(data.get("fragment_mapping_sha256") == self.expected[MAPPING], "Fragment mapping differs")
        jobs = data.get("jobs", [])
        need(len(jobs) == len(ARTIFACTS) and {j.get("name") for j in jobs} == set(ARTIFACTS), "Build jobs differ")
        need(set(data.get("artifacts", {})) == set(ARTIFACTS.values()), "Build artifacts differ")
        for job in jobs:
            command = job.get("argv", [])
            need(command and command[0] == NVCC and f"-DMXFP4_CERTIFIED_VARIANT={variant}" in command
                 and job.get("returncode") == 0, "Unbound/failed compile command")
            log_name = safe_name(job["log"])
            need("/" not in log_name and sha(build / log_name) == job["log_sha256"], "Compiler log hash mismatch")
        for name, meta in data["artifacts"].items():
            path = build / name
            regular(path)
            digest = sha(path)
            need(meta == {"sha256": digest, "bytes": path.stat().st_size}, "Binary hash mismatch")
            self.binaries[str(path)] = digest
        return {"passed": True, "gpu_executed": False, "certified_variant": variant,
                "compile_manifest_sha256": sha(build / "compile-evidence.json"),
                "artifacts": data["artifacts"]}

    def await_framework(self):
        start = time.monotonic()
        framework = Path(self.args.framework_run)
        lock_name = "/tmp/mxfp4-framework-followup-" + str(os.geteuid()) + "-" + hashlib.sha256(B_ROOT.encode()).hexdigest()[:20] + ".lock"
        row = {"name": "wait-framework-lock", "status": "waiting", "passed": False,
               "executed": False, "gpu_executed": False, "lock": lock_name, "started_at_utc": utc()}
        self.report["stages"].append(row)
        self.save()
        try:
            need(framework.is_dir() and not framework.is_symlink() and framework.stat().st_uid == os.geteuid(),
                 "Framework run is not an existing directory owned by this UID")
            need(str(Path(B_ROOT).resolve(strict=True)) == B_ROOT, "Original B root resolves differently than framework lock key")
            fd = os.open(lock_name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            try:
                info = os.fstat(fd)
                need(info.st_uid == os.geteuid() and stat.S_ISREG(info.st_mode), "Unsafe framework flock")
                manifest = framework / "archive-manifest.json"
                while time.monotonic() < self.deadline - 15:
                    if manifest.exists():
                        regular(manifest, owned=True)
                        receipt = document(manifest)
                        fields(receipt, {"source_commit": B_COMMIT, "source_archive_sha256": B_SHA}, "Framework source receipt")
                        need(receipt.get("status") in ("passed", "failed", "blocked"), "Framework receipt is not terminal")
                        archived = framework / "results.tar.gz"
                        regular(archived, owned=True)
                        need(sha(archived) == receipt.get("sha256"), "Framework completion archive hash mismatch")
                        try:
                            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            pass
                        else:
                            self.lock_fd, fd = fd, None
                            row.update(status="passed", passed=True, framework_status=receipt["status"],
                                       framework_manifest_sha256=sha(manifest), framework_archive_sha256=sha(archived),
                                       same_uid_lock_acquired=True)
                            return True
                    time.sleep(min(2, max(0, self.deadline - 15 - time.monotonic())))
                raise Blocked("Deadline while waiting for original framework completion and lock")
            finally:
                if fd is not None:
                    os.close(fd)
        except (Exception, KeyboardInterrupt) as error:
            row.update(status="blocked", error_type=type(error).__name__, reason=str(error))
            return False
        finally:
            row.update(finished_at_utc=utc(), wall_seconds=time.monotonic() - start)
            self.save()

    def execute(self):
        if self.deadline <= self.start + 15:
            raise Blocked("Global/absolute deadline is already exhausted")
        need(Path(self.args.python).is_file() and os.access(self.args.python, os.X_OK), "Explicit lexical Python is not executable")
        if not Path(NVCC).is_file() or not os.access(NVCC, os.X_OK):
            raise Blocked("Required /usr/local/cuda-12.8/bin/nvcc is unavailable")
        binding, self.expected = extract(Path(self.args.source_archive), Path(self.args.source_binding), self.source)
        write_json(self.results / "source-binding.json", binding)
        self.report.update(source_commit=binding["source_commit"], source_archive_sha256=binding["source_sha256"],
                           source_hashes_before=self.expected, source_binding_sha256=sha(self.args.source_binding))
        self.integrity()
        build_ok = {}
        for variant in VARIANTS:
            output = self.run / ("build-c" + str(variant))
            command = [self.args.python, "-m", "tools.mxfp4_training.build_cuda", "--output-dir", str(output),
                       "--nvcc", NVCC, "--with-certified", "--with-benchmark", "--certified-variant", str(variant)]
            build_ok[variant] = self.stage("build-c" + str(variant), command, 900,
                                          lambda rows, v=variant: self.build_gate(v))
        micro = self.run / "build-micro" / "check_mxfp4_half2_decode"
        micro.parent.mkdir(mode=0o700)
        csrc = self.source / TRAINING / "csrc"
        def micro_build_gate(rows):
            regular(micro)
            digest = sha(micro)
            self.binaries[str(micro)] = digest
            return {"passed": True, "gpu_executed": False, "binary_sha256": digest,
                    "actual_included_fused_sha256": sha(csrc / "mxfp4_gemm_fused.cu"),
                    "microcheck_source_sha256": sha(csrc / "check_mxfp4_half2_decode.cu")}
        micro_built = self.stage("build-half2-microcheck", [NVCC, "-O3", "-std=c++17", "--ftz=false", "-lineinfo",
                "-Xptxas=-v", "-gencode=arch=compute_70,code=[sm_70,compute_70]", "-I" + str(csrc),
                str(csrc / "check_mxfp4_half2_decode.cu"), "-o", str(micro)], 900, micro_build_gate)
        self.report["binary_hashes_before_gpu"] = dict(self.binaries)
        if not build_ok[2] or not build_ok[8]:
            raise Blocked("C2/C8 control compilation did not pass")
        if not self.await_framework():
            raise Blocked("Framework serialization prerequisite did not pass")
        native_ok = {}
        controls = self.stage("check-c2-baseline", [self.run / "build-c2/check_mxfp4", "--device", "0", "--correctness-only"], 600,
                              lambda rows: native_gate(rows), gpu=True)
        controls = self.stage("check-c2-tiled", [self.run / "build-c2/check_mxfp4_tiled", "--device", "0", "--correctness-only"], 600,
                              lambda rows: native_gate(rows), gpu=True) and controls
        for variant in VARIANTS:
            name = "check-certified-c" + str(variant)
            if not build_ok[variant]:
                native_ok[variant] = self.block(name, "Candidate compilation failed; no GPU test", True)
            else:
                native_ok[variant] = self.stage(name, [self.run / ("build-c" + str(variant)) / "check_mxfp4_certified", "--device", "0", "--correctness-only"],
                                                600, lambda rows, v=variant: native_gate(rows, v), gpu=True)
        micro_ok = self.stage("check-half2-microcheck", [micro], 300, micro_gate, gpu=True) if micro_built else self.block(
            "check-half2-microcheck", "Microchecker compilation failed", True)
        eligible = {v: build_ok[v] and native_ok.get(v, False) and (v != 10 or micro_ok) for v in VARIANTS}
        self.report["native_eligible_candidates"] = {str(v): ok for v, ok in eligible.items()}
        if not controls or not eligible[2] or not eligible[8]:
            raise Blocked("Baseline/tiled or C2/C8 full-suite correctness failed; no comparisons permitted")
        for variant in (8, 10):
            name = "torch-certified-c" + str(variant)
            if not eligible[variant]:
                self.block(name, "Candidate not eligible after native stages", True)
                continue
            output = self.results / (name + ".json")
            build = self.run / ("build-c" + str(variant))
            library, manifest = build / "libmxfp4_sm70.so", build / "compile-evidence.json"
            self.stage(name, [self.args.python, "-m", "tools.mxfp4_training.check_torch_certified",
                       "--compile-evidence", str(manifest), "--expected-certified-variant", str(variant),
                       "--output", str(output), "--device", "cuda:0", "--seed", "20261123"], 600,
                       lambda rows, p=output, v=variant, lib=library, man=manifest: torch_gate(p, v, lib, man),
                       gpu=True, library=library)
        output = self.results / "torch-baseline-tiled.json"
        library = self.run / "build-c2/libmxfp4_sm70.so"
        self.stage("torch-baseline-tiled", [self.args.python, "-m", "tools.mxfp4_training.check_torch_cuda",
                   "--output", str(output), "--device", "cuda:0", "--seed", "20261123"], 600,
                   lambda rows: torch_gate(output, None, library, None), gpu=True, library=library)

        # Correctness-only boundary evidence uses the unmodified block-scale distribution.
        # Its legitimate fallback is allowed and is never reported as fast-path timing.
        probe_output = self.results / "probe-8192-random_block_scales-c2.jsonl"
        probe = self.run / "build-c2/bench_mxfp4"
        probe_ok = self.stage("probe-8192-random_block_scales-c2", [probe, "--device", "0",
              "--m", "8192", "--n", "8192", "--k", "8192", "--mode", "random_block_scales",
              "--seed", "20261123", "--samples", "256", "--correctness-only", "--output", str(probe_output)],
              900, lambda rows: benchmark_gate(json_lines(probe_output), 2, 8192,
                  self.report["input_fingerprints"], "random_block_scales", correctness_only=True), gpu=True)
        if not probe_ok:
            raise Blocked("C2 boundary correctness probe failed; performance comparisons stopped")
        for size in SHAPES:
            for index, variant in enumerate(ORDER):
                name = "bench-" + str(size) + "-" + str(index + 1).zfill(2) + "-c" + str(variant)
                if not eligible[variant]:
                    self.block(name, "Candidate correctness/build/prior benchmark failed; no retry", True)
                    continue
                result_file = self.results / (name + ".jsonl")
                benchmark = self.run / ("build-c" + str(variant)) / "bench_mxfp4"
                command = [benchmark, "--device", "0", "--m", str(size), "--n", str(size), "--k", str(size),
                           "--mode", MODES[size], "--seed", "20261123", "--warmup", "100",
                           "--iters", "10", "--rounds", "5", "--samples", "256", "--output", str(result_file)]
                passed = self.stage(name, command, 900, lambda rows, v=variant, s=size, p=result_file:
                     benchmark_gate(json_lines(p), v, s, self.report["input_fingerprints"], MODES[s]), gpu=True)
                if not passed:
                    eligible[variant] = False
                    if variant in (2, 8):
                        raise Blocked("C2/C8 comparison control failed; remaining comparisons stopped")

    def finish(self, error=None):
        self.cleanup()
        if error is not None:
            self.report.update(error_type=type(error).__name__, error=str(error))
        planned = ["build-c" + str(v) for v in VARIANTS] + ["build-half2-microcheck", "wait-framework-lock",
                   "check-c2-baseline", "check-c2-tiled"] + ["check-certified-c" + str(v) for v in VARIANTS] + ["check-half2-microcheck"]
        planned += ["bench-" + str(s) + "-" + str(i + 1).zfill(2) + "-c" + str(v)
                    for s in SHAPES for i, v in enumerate(ORDER)]
        planned += ["torch-certified-c8", "torch-certified-c10", "torch-baseline-tiled", "probe-8192-random_block_scales-c2"]
        seen = {s["name"] for s in self.report["stages"]}
        for name in planned:
            if name not in seen:
                self.block(name, "Prerequisite/control/deadline prevented execution: " + str(error), not name.startswith("build") and name != "wait-framework-lock")
        try:
            self.integrity()
            self.report["source_hashes_after"] = tree_hashes(self.source) if self.expected is not None else None
            self.report["source_and_binaries_unchanged"] = self.expected is not None
            self.report["binary_hashes_after_gpu"] = {p: sha(p) for p in self.binaries}
        except Exception as integrity_error:
            self.report.update(integrity_error=str(integrity_error), source_and_binaries_unchanged=False)
            error = integrity_error
        for build in sorted(self.run.glob("build-*")):
            target = self.results / "build-evidence" / build.name
            target.mkdir(parents=True, exist_ok=True)
            for path in sorted(build.iterdir()):
                if path.suffix in (".json", ".log") and path.is_file() and not path.is_symlink():
                    shutil.copyfile(path, target / path.name)
        statuses = {s["status"] for s in self.report["stages"]}
        status = "failed" if "failed" in statuses or (error is not None and not isinstance(error, Blocked)) else (
                 "blocked" if "blocked" in statuses or error is not None else "passed")
        gpu = [s.get("gpu_executed") for s in self.report["stages"] if s.get("executed")]
        self.report.update(status=status, passed=status == "passed", finished_at_utc=utc(),
                           wall_seconds=time.monotonic() - self.start,
                           gpu_executed=True if True in gpu else (None if None in gpu else False))
        self.report["result_files_sha256"] = {k: v for k, v in tree_hashes(self.results).items() if k != "completion.json"}
        self.save()
        need(sum(p.stat().st_size for p in self.results.rglob("*") if p.is_file()) <= MAX_ARCHIVE_BYTES,
             "Results archive exceeds bounded size")
        archive = self.run / "results.tar.gz"
        with tarfile.open(archive, "x:gz") as handle:
            handle.add(self.results, arcname="results", recursive=True)
        receipt = {"status": status, "passed": status == "passed", "results_archive": str(archive),
                   "sha256": sha(archive), "bytes": archive.stat().st_size,
                   "completion_sha256": sha(self.results / "completion.json"),
                   "source_commit": self.report.get("source_commit"),
                   "source_archive_sha256": self.report.get("source_archive_sha256"),
                   "gpu_executed": self.report["gpu_executed"]}
        write_json(self.run / "archive-manifest.json", receipt)
        if self.lock_fd is not None:
            fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            os.close(self.lock_fd)
            self.lock_fd = None
        print(json.dumps(receipt), flush=True)
        return {"passed": 0, "blocked": 2, "failed": 1}[status]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-archive", "source-binding", "run-directory", "framework-run", "python"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    for name in ("source_archive", "source_binding", "run_directory", "framework_run", "python"):
        if not os.path.isabs(getattr(args, name)):
            parser.error("Every execution/input path must be explicitly absolute")
    # Intentionally preserve the venv symlink pathname, e.g. /opt/venv/bin/python3.
    run = Path(args.run_directory)
    need(run.parent.is_dir() and not run.exists() and not run.is_symlink(), "New run directory must not exist")
    run = run.parent.resolve(strict=True) / run.name
    framework = Path(args.framework_run).resolve(strict=True)
    need(run != framework and run not in framework.parents and framework not in run.parents,
         "New run must be separate from framework run")
    need(str(run) != B_ROOT and Path(B_ROOT) not in run.parents, "Do not modify original B run")
    run.mkdir(mode=0o700, exist_ok=False)
    runner = Runner(args, run)
    error = None
    def interrupted(signum, frame):
        for handled in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(handled, signal.SIG_IGN)
        raise KeyboardInterrupt("Runner received signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        runner.execute()
    except (Exception, KeyboardInterrupt) as caught:
        error = caught
    return runner.finish(error)


if __name__ == "__main__":
    raise SystemExit(main())
