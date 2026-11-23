#!/usr/bin/env python3
"""Bounded validation payload for an already allocated, owned V100 container.

The queue runner invokes this file with SOURCE_ARCHIVE RUN_DIRECTORY. It uses
only installed toolchains, creates a new source/results tree, never changes GPU
services or clocks, and preserves blocked/failed stages in results.tar.gz.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time


MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_SOURCE_BYTES = 256 * 1024 * 1024
FUSED_VARIANTS = (3, 4, 5, 6, 7, 8)
COMPARISON_SIZES = (2048, 4096)
COMPARISON_SEED = 20261123


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_source(archive, destination):
    """Accept regular files/directories only, even for an authenticated upload."""
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        if sum(item.size for item in members) > MAX_SOURCE_BYTES:
            raise ValueError("source archive exceeds the expanded-size limit")
        for item in members:
            parts = Path(item.name).parts
            if (not parts or item.name.startswith("/") or ".." in parts
                    or not (item.isfile() or item.isdir())):
                raise ValueError("unsafe source archive member: " + item.name)
        destination.mkdir(exist_ok=False)
        # Every member above is a relative regular file/directory, no links.
        source.extractall(destination, members=members)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_archive", type=Path)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    root = args.run_directory.resolve()
    archive = args.source_archive.resolve()
    if not root.is_dir() or not archive.is_file():
        parser.error("the uploaded archive and unique run directory must exist")
    results = root / "results"
    results.mkdir(exist_ok=False)
    source = root / "source"
    report = {"schema_version": 1, "started_at_utc": utc_now(),
              "status": "running", "source_archive_sha256": digest(archive),
              "deadline_seconds": 2700, "stages": [],
              "services_stopped": False, "clock_or_power_changes": False,
              "packages_installed": False,
              "training_scope": "synthetic tiny causal-decoder smoke, not LLM pretraining"}
    comparison_stages = ["control_c2_benchmark_" + str(size) for size in COMPARISON_SIZES]
    for variant in FUSED_VARIANTS:
        prefix = "fused_c" + str(variant)
        comparison_stages.extend([prefix + "_build", prefix + "_correctness"] +
                                 [prefix + "_benchmark_" + str(size) for size in COMPARISON_SIZES])
    report["fused_comparison"] = {
        "control_certified_variant": 2, "candidate_certified_variants": list(FUSED_VARIANTS),
        "shapes": [[size, size, size] for size in COMPARISON_SIZES],
        "mode": "random_block_scales", "seed": COMPARISON_SEED,
        "iters": 10, "rounds": 5, "warmup": 3,
        "candidate_correctness_gate": "full check_mxfp4_certified --correctness-only",
        "framework_library_certified_variant": 2,
        "stage_order": comparison_stages,
        "deadline_scope": "shared with all preceding native, framework and smoke stages",
    }
    deadline = time.monotonic() + report["deadline_seconds"]
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    active_process = None
    terminated_stages = set()

    def terminate_stage(process):
        if process is None or process in terminated_stages:
            return
        # A reaped leader does not imply its separately created process group
        # is empty. Always clean up that known owned group, including children
        # which ignore TERM after the leader has already exited.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        terminated_stages.add(process)

    def interrupted(signum, _frame):
        # Stages own their process groups; the SSH/GNU-timeout group alone
        # cannot terminate them. Stop only this payload's active child group.
        for name in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(name, signal.SIG_IGN)
        terminate_stage(active_process)
        raise InterruptedError("payload received signal " + str(signum))

    for name in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(name, interrupted)

    def save():
        temporary = results / "completion.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(results / "completion.json")

    def blocked(name, reason):
        report["stages"].append({"name": name, "status": "blocked", "reason": reason,
                                 "executed": False, "gpu_executed": False})
        save()

    def run(name, command, timeout=300, cwd=None, log_subdirectory=None):
        nonlocal active_process
        log_directory = Path(log_subdirectory) if log_subdirectory else Path()
        record = {"name": name, "argv": [str(x) for x in command],
                  "started_at_utc": utc_now(), "status": "running", "executed": True,
                  "stdout": str(log_directory / (name + ".stdout")),
                  "stderr": str(log_directory / (name + ".stderr"))}
        report["stages"].append(record)
        save()
        start = time.monotonic()
        remaining = min(float(timeout), deadline - start)
        if remaining <= 0:
            record.update(status="blocked", reason="batch deadline exhausted", executed=False,
                          gpu_executed=False)
            save()
            return False
        (results / log_directory).mkdir(parents=True, exist_ok=True)
        with (results / record["stdout"]).open("xb") as out, (results / record["stderr"]).open("xb") as err:
            process = None
            try:
                process = subprocess.Popen(record["argv"], cwd=str(cwd or source), env=environment,
                                           stdout=out, stderr=err, start_new_session=True)
                active_process = process
                record["returncode"] = process.wait(timeout=remaining)
                record["status"] = "passed" if process.returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                terminate_stage(process)
                record.update(status="failed", returncode=process.returncode, error="stage timeout")
            except BaseException as error:
                record.update(status="failed", error_type=type(error).__name__, error=str(error))
                if process is None:
                    record.update(executed=False, gpu_executed=False)
                raise
            finally:
                terminate_stage(process)
                active_process = None
                record.update(returncode=process.returncode if process is not None else None,
                              wall_seconds=time.monotonic() - start,
                              finished_at_utc=utc_now())
                save()
        print(json.dumps({"stage": name, "status": record["status"]}), flush=True)
        return record["status"] == "passed"

    def collect_build(build, subdirectory="build"):
        destination = results / subdirectory
        destination.mkdir(parents=True, exist_ok=True)
        if build.is_dir():
            for item in build.iterdir():
                if item.is_file() and item.suffix in (".json", ".log"):
                    shutil.copyfile(item, destination / item.name)

    def block_missing_comparison_stages(reason):
        recorded = {stage["name"] for stage in report["stages"]}
        for name in comparison_stages:
            if name not in recorded:
                blocked(name, reason)

    def comparison_benchmark(name, build, subdirectory, size):
        return run(name, [str(build / "bench_mxfp4"), "--device", "0",
                   "--m", str(size), "--n", str(size), "--k", str(size),
                   "--mode", "random_block_scales", "--seed", str(COMPARISON_SEED),
                   "--iters", "10", "--rounds", "5", "--warmup", "3",
                   "--output", str(results / subdirectory / ("benchmark-" + str(size) + ".jsonl"))],
                   600, log_subdirectory=subdirectory)

    try:
        extract_source(archive, source)
        report["source_files_sha256"] = {
            str(p.relative_to(source)): digest(p)
            for p in sorted(source.rglob("*")) if p.is_file()}
        run("gpu_environment", ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,"
             "utilization.gpu,clocks.current.sm,clocks.current.memory,power.draw,power.limit",
             "--format=csv"], timeout=30)
        nvcc_candidates = []
        if shutil.which("nvcc"):
            nvcc_candidates.append(Path(shutil.which("nvcc")))
        nvcc_candidates.extend(sorted(Path("/usr/local").glob("cuda*/bin/nvcc"), reverse=True))
        nvcc, versions = None, []
        for candidate in dict.fromkeys(p.resolve() for p in nvcc_candidates):
            version = subprocess.run([str(candidate), "--version"], capture_output=True,
                                     text=True, timeout=15)
            versions.append({"path": str(candidate), "returncode": version.returncode,
                             "version": version.stdout, "stderr": version.stderr})
            if version.returncode == 0 and re.search(r"release 12\.\d+", version.stdout):
                nvcc = candidate
                break
        report["installed_nvcc_probes"] = versions
        build = root / "build"
        native_passed = False
        if nvcc is None:
            blocked("cuda_build", "No installed CUDA 12.x nvcc capable of the required SM70 build was selected")
        else:
            cuda_lib = str(nvcc.parent.parent / "lib64")
            environment["LD_LIBRARY_PATH"] = cuda_lib + (":" + environment["LD_LIBRARY_PATH"]
                                                          if environment.get("LD_LIBRARY_PATH") else "")
            built = run("cuda_build", [sys.executable, "tools/mxfp4_training/build_cuda.py",
                         "--nvcc", str(nvcc), "--output-dir", str(build), "--with-benchmark",
                         "--with-certified", "--tiled-variant", "1", "--certified-variant", "2"], 900)
            collect_build(build)
            if built:
                native_passed = True
                for label, binary in (("native_baseline", "check_mxfp4"),
                                      ("native_tiled", "check_mxfp4_tiled"),
                                      ("native_certified", "check_mxfp4_certified")):
                    ok = run(label, [str(build / binary), "--device", "0", "--correctness-only"], 300)
                    native_passed = native_passed and ok
                if native_passed:
                    run("native_benchmark", [str(build / "bench_mxfp4"), "--device", "0",
                        "--iters", "10", "--rounds", "5", "--warmup", "3",
                        "--output", str(results / "native-benchmark.jsonl")], 600)
                else:
                    blocked("native_benchmark", "One or more native correctness gates failed")
        if not native_passed:
            blocked("torch_integration", "The current native library has not passed all correctness gates")
            blocked("torch_benchmark", "Requires the current native and framework correctness gates")
            blocked("gpu_training_smoke", "Requires the current native and framework correctness gates")
        else:
            environment["MXFP4_CUDA_LIBRARY"] = str(build / "libmxfp4_sm70.so")
            candidates = [sys.executable, shutil.which("python3"), shutil.which("python"),
                          "/opt/conda/bin/python", "/usr/local/bin/python3"]
            python, probes = None, []
            probe_code = ("import sys,json,numpy,torch; "
                          "print(json.dumps(dict(python=sys.version,torch=torch.__version__,"
                          "numpy=numpy.__version__,cuda=torch.version.cuda,"
                          "available=torch.cuda.is_available()))); "
                          "sys.exit(0 if sys.version_info >= (3,8) and torch.cuda.is_available() else 2)")
            for candidate in dict.fromkeys(str(Path(p).resolve()) for p in candidates if p and Path(p).is_file()):
                try:
                    probe = subprocess.run([candidate, "-c", probe_code], capture_output=True,
                                           text=True, env=environment, timeout=45)
                    probes.append({"path": candidate, "returncode": probe.returncode,
                                   "stdout": probe.stdout[-8192:], "stderr": probe.stderr[-8192:]})
                    if probe.returncode == 0:
                        python = candidate
                        break
                except subprocess.TimeoutExpired:
                    probes.append({"path": candidate, "error": "probe timeout"})
            report["installed_torch_probes"] = probes
            if python is None:
                blocked("torch_integration", "No installed Python >=3.8 with NumPy and CUDA-enabled PyTorch was found")
                blocked("torch_benchmark", "Requires the framework integration gate")
                blocked("gpu_training_smoke", "Requires the framework integration gate")
            else:
                check_report = results / "torch-cuda-check.json"
                checked = run("torch_integration", [python, "-m", "tools.mxfp4_training.check_torch_cuda",
                              "--output", str(check_report)], 600)
                if not checked and check_report.is_file():
                    if json.loads(check_report.read_text()).get("status") == "blocked":
                        report["stages"][-1]["status"] = "blocked"
                if checked:
                    run("torch_benchmark", [python, "-m", "tools.mxfp4_training.bench_torch_cuda",
                        "--check-report", str(check_report), "--output", str(results / "torch-benchmark.json")], 600)
                    for backend in ("cuda", "cuda_tiled"):
                        run("smoke_" + backend, [python, "-m", "tools.mxfp4_training.pretrain_smoke",
                            "--backend", backend, "--device", "cuda:0", "--steps", "40",
                            "--seed", "20261123", "--lr", "0.005",
                            "--output", str(results / ("smoke-" + backend + ".json"))], 600)
                else:
                    blocked("torch_benchmark", "Framework integration gate did not pass")
                    blocked("gpu_training_smoke", "Framework integration gate did not pass")

        # Keep the original framework/smoke stages and their certified2 library
        # above intact. These independently built native candidates never change
        # MXFP4_CUDA_LIBRARY and consume only the existing batch's remaining time.
        if nvcc is None:
            block_missing_comparison_stages("No installed CUDA 12.x SM70 toolchain was selected")
        elif not native_passed:
            block_missing_comparison_stages("The main certified2 library did not pass all native correctness gates")
        else:
            # The default benchmark includes 4096^3, but uses seed + 4*7919 for
            # that shape. Run explicit controls so both sizes use exactly the
            # same inputs as each fused candidate's one-shape invocation.
            for size in COMPARISON_SIZES:
                comparison_benchmark("control_c2_benchmark_" + str(size), build, "control-c2", size)
            for variant in FUSED_VARIANTS:
                prefix = "fused_c" + str(variant)
                subdirectory = "fused-c" + str(variant)
                candidate_build = root / ("build-" + subdirectory)
                built = run(prefix + "_build", [sys.executable, "tools/mxfp4_training/build_cuda.py",
                            "--nvcc", str(nvcc), "--output-dir", str(candidate_build),
                            "--with-benchmark", "--with-certified", "--tiled-variant", "1",
                            "--certified-variant", str(variant)], 900, log_subdirectory=subdirectory)
                collect_build(candidate_build, subdirectory + "/build")
                if not built:
                    reason = "Requires a successful " + prefix + "_build stage"
                    blocked(prefix + "_correctness", reason)
                    for size in COMPARISON_SIZES:
                        blocked(prefix + "_benchmark_" + str(size), reason)
                    continue
                checked = run(prefix + "_correctness", [str(candidate_build / "check_mxfp4_certified"),
                              "--device", "0", "--correctness-only"], 300, log_subdirectory=subdirectory)
                for size in COMPARISON_SIZES:
                    name = prefix + "_benchmark_" + str(size)
                    if checked:
                        comparison_benchmark(name, candidate_build, subdirectory, size)
                    else:
                        blocked(name, "Requires a successful full " + prefix + "_correctness stage")
        statuses = {stage["status"] for stage in report["stages"]}
        report["status"] = "failed" if "failed" in statuses else ("blocked" if "blocked" in statuses else "passed")
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        block_missing_comparison_stages("Payload prerequisite failed: " + type(error).__name__ + ": " + str(error))
    finally:
        report["finished_at_utc"] = utc_now()
        report["result_files_sha256"] = {str(p.relative_to(results)): digest(p)
            for p in sorted(results.rglob("*")) if p.is_file() and p.name != "completion.json"}
        save()
        if sum(p.stat().st_size for p in results.rglob("*")) > MAX_ARCHIVE_BYTES:
            raise RuntimeError("result files exceed bounded retrieval limit")
        output = root / "results.tar.gz"
        with tarfile.open(output, "x:gz") as packed:
            packed.add(results, arcname="results")
        print(json.dumps({"status": report["status"], "results_archive": str(output),
                          "sha256": digest(output), "bytes": output.stat().st_size}), flush=True)
    return {"passed": 0, "blocked": 2}.get(report["status"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
