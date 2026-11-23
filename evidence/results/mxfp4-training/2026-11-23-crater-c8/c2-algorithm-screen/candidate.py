#!/usr/bin/env python3
"""Build/run the standalone C2 cuBLAS algorithm screen; no installation or source edits."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time

FRAMEWORK_NATIVE_ROOT = "/workspace/user/mxfp4-sg-jincy26-261124-4abd5-20261123T204443Z-9fb4ed47"


def now():
    return datetime.now(timezone.utc).isoformat()


def need(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def load(path):
    value = json.loads(path.read_text())
    need(isinstance(value, dict), "Expected JSON object: " + str(path))
    return value


def source_hashes(source):
    paths = sorted((source / "tools/mxfp4_training/csrc").glob("*"))
    paths = [p for p in paths if p.is_file()]
    paths.append(source / "evidence/benchmarks/sm70-packing/study/simulation/round3/cuda/fragment_mapping.hpp")
    need(len(paths) >= 13, "Incomplete production source snapshot")
    return {p.relative_to(source).as_posix(): digest(p) for p in paths}


def new_directory(value):
    path = Path(value).absolute()
    need(path.parent.is_dir() and not path.exists() and not path.is_symlink(), "Use a new directory under an existing parent")
    path = path.parent.resolve() / path.name
    path.mkdir(mode=0o700, exist_ok=False)
    return path


def build(args):
    source = Path(args.source_root).resolve(strict=True)
    csrc = source / "tools/mxfp4_training/csrc"
    own = Path(__file__).absolute()
    candidate = own.with_name("bench_c2_algorithms.cu")
    nvcc = Path(args.nvcc).absolute()
    need(nvcc.is_file() and os.access(nvcc, os.X_OK), "Explicit nvcc must be executable")
    output = Path(args.output_directory).absolute()
    need(source != output and source not in output.parents, "Build output must be outside frozen production source")
    output = new_directory(output)
    original = source_hashes(source)
    shutil.copyfile(candidate, output / candidate.name)
    executable = output / "bench_c2_algorithms"
    command = [str(nvcc), "-O3", "-std=c++17", "--ftz=false", "-lineinfo", "-Xptxas=-v",
               "-gencode=arch=compute_70,code=[sm_70,compute_70]", "-DMXFP4_TILED_VARIANT=1",
               "-DMXFP4_CERTIFIED_VARIANT=2", "-I", str(csrc), str(csrc / "mxfp4_gemm.cu"),
               str(csrc / "mxfp4_gemm_tiled.cu"), str(output / candidate.name), "-lcublas", "-ldl", "-o", str(executable)]
    record = {"schema_version": 1, "kind": "C2 cuBLAS algorithm screen compilation", "started_at_utc": now(),
              "source_root": str(source), "production_source_sha256": original,
              "candidate_sha256": digest(candidate), "builder_sha256": digest(own), "argv": command,
              "certified_variant": 2, "tiled_variant": 1, "gpu_executed": False,
              "native_production_sources_modified": False, "passed": False}
    if args.native_manifest:
        native = Path(args.native_manifest).resolve(strict=True)
        value = load(native)
        need(Path(value["remote_dir"]).resolve() / "source" == source, "Native manifest does not bind this source directory")
        record["native_run_manifest"] = {"path": str(native), "sha256": digest(native), "record": value}
    if args.source_binding:
        bound_path, archive = Path(args.source_binding).resolve(strict=True), Path(args.source_archive).resolve(strict=True)
        bound = load(bound_path)
        need(digest(archive) == bound.get("source_sha256"), "Frozen source archive SHA differs")
        members = bound.get("members", [])
        expected = {}
        for item in members:
            relative = item.get("name", "")
            p = PurePosixPath(relative)
            need(relative and not p.is_absolute() and ".." not in p.parts and str(p) == relative and
                 relative not in expected, "Invalid/duplicate frozen source binding member")
            expected[relative] = item["sha256"]
            need(digest(source / relative) == item["sha256"], "Frozen source member changed: " + relative)
        need(expected and all(expected.get(k) == v for k, v in original.items()), "Csrc/fragment sources outside frozen binding")
        record["frozen_source_binding"] = {"path": str(bound_path), "sha256": digest(bound_path), "record": bound,
                                            "archive_path": str(archive), "archive_sha256": digest(archive)}
    def interrupted_compile(signum, _frame):
        # The compiler owns a separate session. Convert wrapper termination
        # into an exception so its existing finally block cleans that group.
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(sig, signal.SIG_IGN)
        raise InterruptedError("Compilation wrapper interrupted by signal " + str(signum))
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    for sig in previous_handlers:
        signal.signal(sig, interrupted_compile)
    try:
        version = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, timeout=30)
        (output / "nvcc-version.log").write_text(version.stdout + version.stderr)
        need(version.returncode == 0, "nvcc --version failed")
        record["nvcc_version"] = version.stdout.strip()
        with (output / "compile.stdout.log").open("xb") as out, (output / "compile.stderr.log").open("xb") as err:
            completed = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
            try:
                completed.wait(timeout=600)
            finally:
                # nvcc can leave compiler children even after its leader exits.
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(completed.pid, sig)
                    except ProcessLookupError:
                        pass
                    if sig == signal.SIGTERM:
                        try:
                            completed.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            pass
                completed.wait(timeout=5)
        record["returncode"] = completed.returncode
        need(source_hashes(source) == original, "Production source changed during compilation")
        need(digest(candidate) == record["candidate_sha256"], "Candidate source changed during compilation")
        if completed.returncode == 0:
            record["binary"] = {"path": str(executable), "sha256": digest(executable), "bytes": executable.stat().st_size}
            record["passed"] = True
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error))
    finally:
        record["finished_at_utc"] = now()
        record["log_sha256"] = {p.name: digest(p) for p in output.glob("*.log")}
        save(output / "compile-evidence.json", record)
        print(json.dumps({"passed": record["passed"], "build_directory": str(output),
                          "compile_evidence_sha256": digest(output / "compile-evidence.json")}))
        for sig, previous in previous_handlers.items():
            signal.signal(sig, previous)
    return 0 if record["passed"] else 1


def verify_build(directory):
    path = directory / "compile-evidence.json"
    record = load(path)
    need(record.get("passed") is True and record.get("returncode") == 0 and record.get("certified_variant") == 2,
         "Build was not a passing certified2 compilation")
    need(digest(Path(__file__).absolute()) == record["builder_sha256"], "Build/run wrapper content differs")
    source = Path(record["source_root"])
    need(source_hashes(source) == record["production_source_sha256"], "Frozen production source changed")
    if "frozen_source_binding" in record:
        bound = record["frozen_source_binding"]
        need(digest(bound["path"]) == bound["sha256"] and digest(bound["archive_path"]) == bound["archive_sha256"],
             "Frozen source publication/archive changed")
        for item in bound["record"]["members"]:
            need(digest(source / item["name"]) == item["sha256"], "Frozen source member changed: " + item["name"])
    need(digest(directory / "bench_c2_algorithms.cu") == record["candidate_sha256"], "Candidate source differs")
    binary = Path(record["binary"]["path"])
    need(binary == directory / "bench_c2_algorithms" and digest(binary) == record["binary"]["sha256"] and
         binary.stat().st_size == record["binary"]["bytes"], "Built executable binding failed")
    for name, expected in record["log_sha256"].items():
        need(Path(name).name == name and digest(directory / name) == expected, "Build log hash mismatch")
    return record, digest(path)


def acquire_gpu_gate(prior_directory, built):
    """Require the native archive to be finalized and hold the same B/framework lock."""
    root = Path(prior_directory).resolve(strict=True)
    lock_name = "/tmp/mxfp4-framework-followup-" + str(os.geteuid()) + "-" + hashlib.sha256(
        FRAMEWORK_NATIVE_ROOT.encode()).hexdigest()[:20] + ".lock"
    fd = os.open(lock_name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        need(info.st_uid == os.geteuid() and stat.S_ISREG(info.st_mode), "Unsafe serialization lock")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Prior framework/native GPU work still owns the B lock; no GPU work started") from error
        receipt_path, completion_path = root / "archive-manifest.json", root / "results/completion.json"
        archive = root / "results.tar.gz"
        receipt, completion = load(receipt_path), load(completion_path)
        need(receipt.get("status") in ("passed", "failed", "blocked") and completion.get("status") == receipt["status"] and
             completion.get("finished_at_utc") and
             all(s.get("status") in ("passed", "failed", "blocked") for s in completion.get("stages", [])),
             "Prior native completion is not finalized")
        need(digest(archive) == receipt.get("sha256") and archive.stat().st_size == receipt.get("bytes") and
             digest(completion_path) == receipt.get("completion_sha256"), "Prior native archive receipt does not match")
        with tarfile.open(archive, "r:gz") as t:
            matches = [m for m in t.getmembers() if m.name == "results/completion.json"]
            need(len(matches) == 1 and matches[0].isreg() and matches[0].size <= 32 * 1024 ** 2,
                 "Prior archived completion missing/duplicated/unsafe")
            need(hashlib.sha256(t.extractfile(matches[0]).read()).hexdigest() == receipt["completion_sha256"],
                 "Prior archived completion differs from finalized disk completion")
        stages = {s["name"]: s for s in completion.get("stages", [])}
        for name in ("check-c2-baseline", "check-c2-tiled", "check-certified-c2"):
            s = stages.get(name, {})
            need(s.get("status") == "passed" and s.get("passed") is True and s.get("returncode") == 0 and
                 s.get("validation", {}).get("passed") is True, "Prior repeated C2 gate did not pass: " + name)
        if "frozen_source_binding" in built:
            expected = built["frozen_source_binding"]["record"]
            need(receipt.get("source_commit") == expected["source_commit"] and
                 receipt.get("source_archive_sha256") == expected["source_sha256"], "Prior native source differs from microbench source")
        evidence = {"prior_native_directory": str(root), "archive_manifest_sha256": digest(receipt_path),
                    "archive_sha256": receipt["sha256"], "completion_sha256": receipt["completion_sha256"],
                    "prior_status": receipt["status"], "same_B_framework_lock": lock_name,
                    "same_B_framework_lock_acquired": True, "repeated_C2_native_gates_passed": True}
        return fd, evidence
    except BaseException:
        os.close(fd)
        raise


def validate_result(output, returncode, size, mode, binding, ordinary):
    rows = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    envs = [r for r in rows if r.get("type") == "screen_environment"]
    summaries = [r for r in rows if r.get("type") == "screen_summary"]
    need(len(envs) == 1 and len(summaries) == 1, "Environment/final summary missing or duplicated")
    env, summary = envs[0], summaries[0]
    for r in (env, summary):
        need(r.get("M") == r.get("N") == r.get("K") == size and r.get("mode") == mode and
             r.get("seed") == 20261123 and len(r.get("input_fnv1a64", "")) == 16, "Input identity differs")
        need(r.get("fullpipeline_performance_measured") is False, "Unexpected full-pipeline claim")
    need(env.get("build_binding_sha256") == binding and env.get("timescope") == "normalize_gemm_compute" and
         env.get("A_B_type") == "CUDA_R_16F" and env.get("C_type") == "CUDA_R_32F" and
         env.get("compute_type") == "CUBLAS_COMPUTE_32F" and env.get("blas_workspace_bytes") == 4 * 1024 ** 2 and
         env.get("explicit_allocation_bytes", 2 ** 40) <= 1024 ** 3 and env.get("reserve_bytes") == 512 * 1024 ** 2 and
         env.get("warmup_per_supported_algorithm") == 100 and env.get("rounds") == 5 and env.get("iterations_per_round") == 10,
         "Screen configuration binding differs")
    need(env["input_fnv1a64"] == summary["input_fnv1a64"], "Summary input fingerprint differs")
    cert = env.get("certificate", {})
    results = [r for r in rows if r.get("type") == "algorithm_result"]
    if returncode == 77:
        need(summary.get("status") == "blocked_certificate" and summary.get("passed") is False and
             summary.get("algorithm_timing_executed") is False and cert.get("accepted") is False and
             cert.get("reject_flags", 0) != 0 and not results, "Exit77 is not a real certificate block")
        return {"status": "blocked_certificate", "passed": False, "algorithm_timing_executed": False,
                "input_fnv1a64": env["input_fnv1a64"], "certificate": cert}
    need(returncode == 0 and summary.get("status") == "passed" and summary.get("passed") is True and
         summary.get("gpu_executed") is True and summary.get("all_submitted_work_completed") is True and
         summary.get("certificate_unchanged") is True and cert.get("accepted") is True and cert.get("reject_flags") == 0,
         "Screen process/content did not pass")
    gates = [r for r in rows if r.get("type") == "screen_prerequisite"]
    need(len(gates) == 1 and gates[0].get("passed") is True and gates[0].get("original_c2_finite_outputs") == size * size and
         gates[0].get("original_c2_negative_zero_outputs") == 0,
         "Original C2 prerequisite missing")
    def full_compare(value, canonical=True):
        need(value.get("values") == size * size and value.get("exact_mismatches") == 0 and
             value.get("nonfinite_values") == 0, "Full comparison failed")
        if canonical:
            need(value.get("negative_zero_values") == 0, "Postscaled C2/candidate violated canonical positive zero")
    full_compare(gates[0].get("original_c2_vs_tiled_all_outputs", {}), canonical=False)
    cpu = gates[0].get("original_c2_vs_cpu_selected", {})
    need(cpu.get("values") == 256 and cpu.get("exact_mismatches") == 0 and cpu.get("nonfinite_values") == 0,
         "Selected independent CPU oracle did not pass")
    expected_count = 42 if ordinary else 18
    need(len(results) == expected_count and len({r.get("algorithm") for r in results}) == expected_count,
         "Missing or duplicate algorithm result")
    checked = []
    for row in results:
        need(row.get("M") == row.get("N") == row.get("K") == size and row.get("mode") == mode and
             row.get("seed") == 20261123 and row.get("input_fnv1a64") == env["input_fnv1a64"],
             "Algorithm result belongs to different inputs")
        if row.get("status") == "not_supported":
            need(row.get("passed") is False and row.get("timing") is None and row.get("correctness_verified") is False,
                 "Unsupported algorithm falsely claimed passing/timed")
            continue
        need(row.get("status") == "passed" and row.get("passed") is True and row.get("successful_gemm_calls") == 151 and
             row.get("final_actual_timed_output_checked") is True and row.get("epilogue_applied_after_timer") is True and
             row.get("final_scaled_output_vs_tiled_verified_by_full_reference_transitivity") is True and
             row.get("fullpipeline_performance_measured") is False, "Algorithm final timed correctness missing")
        full_compare(row.get("final_scaled_output_vs_original_c2_all_outputs", {}))
        timing = row.get("timing", {})
        need(timing.get("scope") == "normalize_gemm_compute" and timing.get("primary_clock") == "synchronized_steady_clock",
             "Unexpected timing scope/primary clock")
        for key in ("cuda_event_round_ms", "synchronized_wall_round_ms"):
            values = timing.get(key, [])
            need(len(values) == 5 and all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in values),
                 "Timing samples invalid")
        checked.append(row["algorithm"])
    need("CUBLAS_GEMM_DEFAULT_TENSOR_OP" in checked and len(checked) == summary.get("supported_verified_algorithms"),
         "Current default algorithm missing or supported count differs")
    return {"status": "passed", "passed": True, "input_fnv1a64": env["input_fnv1a64"],
            "supported_algorithms": len(checked), "not_supported_algorithms": len(results) - len(checked),
            "best_screened_wall_algorithm": summary["best_screened_wall_algorithm"],
            "timescope": "normalize_gemm_compute", "fullpipeline_performance_measured": False}


def run(args):
    directory = Path(args.build_directory).resolve(strict=True)
    built, binding = verify_build(directory)
    lock_fd, gpu_gate = acquire_gpu_gate(args.prior_native_run, built)
    destination = Path(args.run_directory).absolute()
    source = Path(built["source_root"])
    need(source not in destination.parents and source != destination and directory != destination and directory not in destination.parents,
         "Run output must be separate from source/build")
    destination = new_directory(destination)
    results = destination / "results"
    results.mkdir()
    provenance = results / "provenance"
    provenance.mkdir()
    for p in directory.iterdir():
        if p.is_file():
            shutil.copyfile(p, provenance / p.name)
    shutil.copyfile(Path(__file__).absolute(), provenance / "candidate.py")
    for relative in built["production_source_sha256"]:
        target = provenance / "production-source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    started, active = time.monotonic(), None
    deadline = started + 1800
    report = {"schema_version": 1, "started_at_utc": now(), "status": "running", "passed": False,
              "build_manifest_sha256": binding, "binary_sha256": built["binary"]["sha256"],
              "production_source_sha256": built["production_source_sha256"], "cases": [],
              "timescope": "normalize_gemm_compute", "fullpipeline_performance_measured": False,
              "global_budget_seconds": 1800, "maximum_case_seconds": 600, "packages_installed": False,
              "services_stopped": False, "clock_or_power_changes": False, "source_modified": False}
    report["gpu_serialization_gate"] = gpu_gate
    env = os.environ.copy()
    cuda_lib = str(Path(built["argv"][0]).parent.parent / "lib64")
    env["LD_LIBRARY_PATH"] = cuda_lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    terminated = set()
    def cleanup(process):
        if process is None or process.pid in terminated:
            return
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
            if sig == signal.SIGTERM:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        process.wait(timeout=5)
        terminated.add(process.pid)
    def interrupted(signum, _frame):
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        cleanup(active)
        raise InterruptedError("Screen interrupted: " + str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    cases = [(4096, "random_block_scales"), (8192, "random_block_scales"), (8192, "random_unit_scale")]
    try:
        for size, mode in cases:
            verify_build(directory)
            label = str(size) + "-" + mode
            output = results / (label + ".jsonl")
            command = [built["binary"]["path"], "--size", str(size), "--mode", mode, "--device", "0",
                       "--build-binding-sha256", binding, "--output", str(output)]
            if args.include_ordinary:
                command.append("--include-ordinary")
            rec = {"size": size, "mode": mode, "argv": command, "status": "running", "passed": False,
                   "started_at_utc": now(), "stdout": label + ".stdout.log", "stderr": label + ".stderr.log"}
            report["cases"].append(rec)
            save(results / "completion.json", report)
            remaining = min(600, deadline - time.monotonic() - 15)
            if remaining <= 0:
                rec.update(status="blocked_budget", reason="Global budget exhausted", executed=False)
                continue
            process = None
            try:
                with (results / rec["stdout"]).open("xb") as out, (results / rec["stderr"]).open("xb") as err:
                    process = subprocess.Popen(command, cwd=str(directory), env=env, stdout=out, stderr=err, start_new_session=True)
                    active = process
                    rec.update(executed=True, owned_process_group=process.pid)
                    rec["returncode"] = process.wait(timeout=remaining)
                cleanup(process); active = None
                verify_build(directory)
                rec.update(validate_result(output, process.returncode, size, mode, binding, args.include_ordinary))
                rec["output_sha256"] = digest(output)
            except Exception as error:
                rec.update(status="failed", passed=False, error_type=type(error).__name__, error=str(error))
                if isinstance(error, (InterruptedError, subprocess.TimeoutExpired)):
                    raise
            finally:
                cleanup(process); active = None
                rec["finished_at_utc"] = now()
                save(results / "completion.json", report)
                print(json.dumps({"case": label, "status": rec["status"]}), flush=True)
        statuses = {c["status"] for c in report["cases"]}
        report["status"] = "failed" if "failed" in statuses else (
            "completed_with_blocked_cases" if statuses != {"passed"} else "passed")
        report["passed"] = report["status"] == "passed"
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", passed=False, error_type=type(error).__name__, error=str(error))
    finally:
        cleanup(active)
        recorded = {(c["size"], c["mode"]) for c in report["cases"]}
        for size, mode in cases:
            if (size, mode) not in recorded:
                report["cases"].append({"size": size, "mode": mode, "status": "blocked_runner", "passed": False,
                                         "executed": False, "reason": "Runner prerequisite/interruption failure"})
        try:
            verify_build(directory)
            report["production_source_and_binary_unchanged"] = True
        except Exception as error:
            report.update(status="failed", passed=False, production_source_and_binary_unchanged=False, integrity_error=str(error))
        report.update(finished_at_utc=now(), wall_seconds=time.monotonic() - started)
        report["result_files_sha256"] = {p.relative_to(results).as_posix(): digest(p)
                                         for p in sorted(results.rglob("*")) if p.is_file() and p.name != "completion.json"}
        save(results / "completion.json", report)
        archive = destination / "results.tar.gz"
        with tarfile.open(archive, "x:gz") as tar:
            tar.add(results, arcname="results")
        receipt = {"status": report["status"], "passed": report["passed"], "archive": str(archive),
                   "archive_sha256": digest(archive), "bytes": archive.stat().st_size,
                   "completion_sha256": digest(results / "completion.json")}
        save(destination / "archive-manifest.json", receipt)
        print(json.dumps(receipt), flush=True)
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    return 0 if report["status"] == "passed" else (2 if report["status"] == "completed_with_blocked_cases" else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--source-root", required=True)
    p.add_argument("--output-directory", required=True)
    p.add_argument("--nvcc", required=True)
    p.add_argument("--native-manifest")
    p.add_argument("--source-binding", help="Verified frozen source member/commit publication")
    p.add_argument("--source-archive", help="Required alongside --source-binding")
    p = sub.add_parser("run")
    p.add_argument("--build-directory", required=True)
    p.add_argument("--run-directory", required=True)
    p.add_argument("--prior-native-run", required=True,
                   help="C9/C10 native run with finalized completion, archive and archive-manifest")
    p.add_argument("--include-ordinary", action="store_true", help="Secondary ALGO0..23 sweep; can be much slower")
    args = parser.parse_args()
    if args.command == "build" and bool(args.source_binding) != bool(args.source_archive):
        parser.error("--source-binding and --source-archive must be provided together")
    return build(args) if args.command == "build" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
