#!/usr/bin/env python3
"""One bounded, source-bound C2/C10/C11 + paired-RNE V100 experiment.

Compatible with the audited queue runner's two positional arguments:
    python3 crater_remote_run.py SOURCE_ARCHIVE NEW_OWNED_RUN_DIRECTORY
The queue runner uploads run-manifest.json and enforces its 3000-second outer
timeout. This payload stops work at 2940 seconds, reserving time for cleanup
and results.tar.gz. It never imports Torch in this long-lived parent process.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
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


PYTHON = "/opt/venv/bin/python3"  # Keep lexical path: resolving loses the venv.
NVCC = "/usr/local/cuda-12.8/bin/nvcc"
TRAINING = "tools/mxfp4_training/"
MAPPING = "evidence/benchmarks/sm70-packing/study/simulation/round3/cuda/fragment_mapping.hpp"
VARIANTS = (2, 10, 11)
SEED = 20261124
TOTAL_SECONDS = 3000
WORK_SECONDS = 2940
MAX_SOURCE_BYTES = 64 << 20
MAX_ARCHIVE_BYTES = 32 << 20
MAX_RESULT_BYTES = 96 << 20
MAX_LOG_BYTES = 16 << 20


class Blocked(RuntimeError):
    pass


def need(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def regular(path, maximum=None):
    value = Path(path).lstat()
    need(stat.S_ISREG(value.st_mode) and value.st_uid == os.geteuid(),
         "Expected an owned regular file: " + str(path))
    need(value.st_mode & 0o022 == 0, "File is writable by other identities: " + str(path))
    if maximum is not None:
        need(value.st_size <= maximum, "File exceeds size bound: " + str(path))
    return value


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def source_hashes(source):
    result = {}
    for path in sorted(source.rglob("*")):
        need(not path.is_symlink(), "Source tree acquired a symlink")
        if path.is_file():
            regular(path)
            result[path.relative_to(source).as_posix()] = sha(path)
    return result


def extract_source(archive, destination):
    """Manually extract bounded regular files; reject links and duplicate paths."""
    destination.mkdir(mode=0o700, exist_ok=False)
    allowed_parents = {str(p) for p in PurePosixPath(MAPPING).parents}
    allowed_parents.update({"tools", "tools/mxfp4_training"})
    names, total = set(), 0
    with tarfile.open(archive, "r:gz") as bundle:
        for count, item in enumerate(bundle, 1):
            name = item.name.rstrip("/") if item.isdir() else item.name
            path = PurePosixPath(name)
            need(count <= 2048 and name and name == str(path) and not path.is_absolute()
                 and ".." not in path.parts and "\\" not in name and name not in names,
                 "Noncanonical, duplicate or excessive archive member: " + name)
            names.add(name)
            need(item.isfile() or item.isdir(), "Source archive contains a link or special file")
            allowed = name.startswith(TRAINING) or name == MAPPING
            need(allowed or (item.isdir() and name in allowed_parents),
                 "Source archive member outside the declared source scope: " + name)
            target = destination.joinpath(*path.parts)
            if item.isdir():
                target.mkdir(mode=0o700, parents=True, exist_ok=True)
                continue
            need(not item.sparse and item.size >= 0, "Sparse/invalid source member")
            total += item.size
            need(total <= MAX_SOURCE_BYTES, "Expanded source archive exceeds its bound")
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with bundle.extractfile(item) as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, 1 << 20)
            need(target.stat().st_size == item.size, "Truncated source archive member")
    hashes = source_hashes(destination)
    required = ("build_cuda.py", "torch_ops.py", "check_tile_certified.py",
                "bench_tile_certified.py", "check_quantize_pair_cuda.py",
                "bench_quantize_pair_cuda.py", "check_certified_workspace.py",
                "bench_quantize_pair_workspace.py")
    need(all(TRAINING + name in hashes for name in required) and MAPPING in hashes,
         "Archive is missing an experiment source file")
    return hashes


def make_plan(root, results):
    plan = []
    for variant in VARIANTS:
        build = root / ("build-c" + str(variant))
        plan.append(dict(name=f"build-c{variant}", kind="build", variant=variant, timeout=720,
                         argv=[PYTHON, TRAINING + "build_cuda.py", "--nvcc", NVCC,
                               "--output-dir", str(build), "--with-certified", "--with-quantizer-rne",
                               "--tiled-variant", "1", "--certified-variant", str(variant)]))
    for variant in VARIANTS:
        plan.append(dict(name=f"native-certified-c{variant}", kind="native", variant=variant,
                         summary="certified_summary", timeout=180,
                         argv=[str(root / f"build-c{variant}/check_mxfp4_certified"),
                               "--device", "0", "--correctness-only"]))
    for name, binary in (("native-tiled-c2", "check_mxfp4_tiled"), ("native-baseline-c2", "check_mxfp4")):
        plan.append(dict(name=name, kind="native", variant=2, summary="summary", timeout=180,
                         argv=[str(root / "build-c2" / binary), "--device", "0", "--correctness-only"]))

    def add_python(name, module, variant, arguments, kind="check", timeout=240, **extra):
        output = results / (name + ".json")
        plan.append(dict(name=name, kind=kind, variant=variant, timeout=timeout, output=str(output),
                         argv=[PYTHON, "-m", "tools.mxfp4_training." + module,
                               "--compile-evidence", str(root / f"build-c{variant}/compile-evidence.json"),
                               "--device", "cuda:0", *arguments, "--output", str(output)], **extra))

    for variant in VARIANTS:
        add_python(f"tile-check-c{variant}", "check_tile_certified", variant,
                   ["--expected-certified-variant", str(variant)])
    add_python("pair-check-c2", "check_quantize_pair_cuda", 2, ["--seed", str(SEED)])
    add_python("workspace-check-c2", "check_certified_workspace", 2,
               ["--expected-certified-variant", "2"])
    for size, modes, order, rounds, iters in (
            (4096, ("block", "sparse_span", "sparse_energy"), (2, 10, 11, 11, 10, 2), 5, 10),
            (8192, ("unit", "block"), (2, 11, 11, 2), 3, 3)):
        for mode in modes:
            for slot, variant in enumerate(order, 1):
                add_python(f"tile-{size}-{mode}-{slot:02d}-c{variant}", "bench_tile_certified", variant,
                           ["--expected-certified-variant", str(variant), "--check-report",
                            str(results / f"tile-check-c{variant}.json"), "--mode", mode,
                            "--m", str(size), "--n", str(size), "--k", str(size),
                            "--seed", str(SEED), "--warmup", "10", "--rounds", str(rounds),
                            "--iters", str(iters)], kind="tile_benchmark", size=size, mode=mode,
                           rounds=rounds, iters=iters, warmup=10)
    for family, module in (("quantize", "bench_quantize_pair_cuda"),
                           ("workspace", "bench_quantize_pair_workspace")):
        for size in (1024, 4096):
            for slot, mode in enumerate(("singles", "pair", "pair", "singles"), 1):
                arguments = ["--mode", mode, "--size", str(size), "--seed", str(SEED),
                             "--warmup", "10", "--rounds", "5", "--iters", "10"]
                if family == "quantize":
                    arguments += ["--check-report", str(results / "pair-check-c2.json")]
                else:
                    arguments += ["--pair-check-report", str(results / "pair-check-c2.json"),
                                  "--workspace-check-report", str(results / "workspace-check-c2.json"),
                                  "--expected-certified-variant", "2"]
                add_python(f"{family}-{size}-{slot:02d}-{mode}", module, 2, arguments,
                           kind=family + "_benchmark", size=size, mode=mode, rounds=5, iters=10, warmup=10)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_archive", type=Path)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    started = time.monotonic()
    root = Path(os.path.abspath(args.run_directory))
    archive = Path(os.path.abspath(args.source_archive))
    identity = root.lstat()
    need(stat.S_ISDIR(identity.st_mode) and identity.st_uid == os.geteuid()
         and identity.st_mode & 0o022 == 0 and root.resolve() == root,
         "Run directory must be an owned, private, nonsymlink directory")
    need(root.parent == Path("/workspace/user") and root.name.startswith("mxfp4-"),
         "Run directory is outside the queue runner's owned workspace scope")
    need(archive == root / "source.tar.gz", "Archive must be the queue runner's uploaded source.tar.gz")
    lock_fd = os.open(root / ".payload.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    lock_info = os.fstat(lock_fd)
    need(stat.S_ISREG(lock_info.st_mode) and lock_info.st_uid == os.geteuid()
         and lock_info.st_mode & 0o022 == 0, "Unsafe own-run lock")
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    results, source = root / "results", root / "source"
    results.mkdir(mode=0o700, exist_ok=False)
    need(not source.exists() and not source.is_symlink() and not (root / "results.tar.gz").exists(),
         "This run directory already contains an attempted payload")
    report = dict(schema_version=1, status="running", passed=False, started_at_utc=utc(),
                  uid=os.geteuid(), own_run_flock_acquired=True, run_directory=str(root),
                  total_runtime_bound_seconds=TOTAL_SECONDS, work_deadline_seconds=WORK_SECONDS,
                  outer_timeout_required_seconds=TOTAL_SECONDS, python_lexical_path=PYTHON,
                  packages_installed=False, clock_or_power_changes=False, gpu_services_stopped=False,
                  full_training_executed=False, llm_convergence_validated=False,
                  cleanup_scope="only_process_groups_created_by_this_payload",
                  gpu_exclusivity_scope="idle/no-compute-process benchmark boundaries; not continuous exclusivity",
                  source_commit_scope="authenticated publication label; archive bytes are independently hashed",
                  performance_scope="synthetic packed-input GEMM and finite FP32 RNE wrapper experiments",
                  first_tile_timed_probe_excluded_from_batch_medians=True,
                  stages=[], builds={}, input_fingerprints={}, gpu_snapshots=[])
    plan = make_plan(root, results)
    report["plan"] = plan
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(source), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                       PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                       CUDA_HOME="/usr/local/cuda-12.8", CUDA_CACHE_PATH=str(root / "cuda-cache"),
                       XDG_CACHE_HOME=str(root / "cache"), TORCH_EXTENSIONS_DIR=str(root / "torch-extensions"))
    environment["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64" + (
        ":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
    active_process, cleaned = None, set()
    frozen_sources, binaries = {}, {}

    def save():
        temporary = results / "completion.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(results / "completion.json")

    def cleanup(process):
        if process is None or process in cleaned:
            return
        # Even a reaped leader can leave children in its dedicated session.
        # Never select targets by process name, GPU PID, another run, or UID.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
        cleaned.add(process)

    def interrupted(signum, _frame):
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT, signal.SIGALRM):
            signal.signal(sig, signal.SIG_IGN)
        raise InterruptedError("Payload received signal " + str(signum))

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, interrupted)
    signal.setitimer(signal.ITIMER_REAL, max(0.01, WORK_SECONDS - (time.monotonic() - started)))

    def run(spec):
        nonlocal active_process
        name = spec["name"]
        row = {**spec, "status": "running", "executed": False, "started_at_utc": utc(),
               "stdout": name + ".stdout.log", "stderr": name + ".stderr.log"}
        report["stages"].append(row)
        save()
        begin, process = time.monotonic(), None
        env = environment.copy()
        if "variant" in spec:
            env["MXFP4_CUDA_LIBRARY"] = str(root / f"build-c{spec['variant']}/libmxfp4_sm70.so")
        try:
            if frozen_sources:
                need(source_hashes(source) == frozen_sources, "Source changed before stage")
            if spec.get("kind") not in ("probe", "build"):
                need(binaries and all(sha(path) == value for path, value in binaries.items()),
                     "Compiled artifact changed before GPU stage")
            remaining = min(spec.get("timeout", 30), started + WORK_SECONDS - begin)
            if remaining <= 0:
                raise Blocked("Shared work deadline exhausted")
            with (results / row["stdout"]).open("xb") as out, (results / row["stderr"]).open("xb") as err:
                process = subprocess.Popen(spec["argv"], cwd=source if source.is_dir() else root,
                                           env=env, stdout=out, stderr=err, start_new_session=True)
                active_process = process
                row.update(executed=True, owned_process_group=process.pid)
                while True:
                    left = begin + remaining - time.monotonic()
                    if left <= 0:
                        raise subprocess.TimeoutExpired(spec["argv"], remaining)
                    if os.fstat(out.fileno()).st_size + os.fstat(err.fileno()).st_size > MAX_LOG_BYTES:
                        raise RuntimeError("Stage output exceeded the bounded log size")
                    try:
                        code = process.wait(timeout=min(0.25, left))
                        break
                    except subprocess.TimeoutExpired:
                        continue
                row["returncode"] = code
                need(os.fstat(out.fileno()).st_size + os.fstat(err.fileno()).st_size <= MAX_LOG_BYTES,
                     "Stage output exceeded the bounded log size")
                need(code == 0, "Stage returned nonzero status " + str(code))
            row["status"] = "passed"
        except BaseException as error:
            row.update(status="blocked" if isinstance(error, Blocked) else "failed",
                       error_type=type(error).__name__, error=str(error))
            raise
        finally:
            try:
                cleanup(process)
            except BaseException as error:
                row.update(status="failed", cleanup_error=str(error))
                raise
            finally:
                active_process = None
                row.update(returncode=None if process is None else process.returncode,
                           finished_at_utc=utc(), wall_seconds=time.monotonic() - begin,
                           owned_child_group_cleanup_completed=process is None or process in cleaned)
                save()
                print(json.dumps({"stage": name, "status": row["status"]}), flush=True)
        return row

    def gpu_idle(label, require_idle=True):
        # Utilization is a trailing sample. Retry it only while the process list
        # is empty; a visible competing context is an immediate block, never a
        # signal target. Every attempt is recorded and shares the work deadline.
        for attempt in range(8):
            records = {}
            for suffix, query in (("gpu", ["--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu"]),
                                  ("processes", ["--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"])):
                spec = dict(name=f"gpu-{label}-{attempt:02d}-{suffix}", kind="probe", timeout=20,
                            argv=["nvidia-smi", *query, "--format=csv,noheader,nounits"])
                row = run(spec)
                records[suffix] = (results / row["stdout"]).read_text().strip()
            gpus = list(csv.reader(records["gpu"].splitlines(), skipinitialspace=True))
            need(len(gpus) == 1 and len(gpus[0]) == 6 and "V100" in gpus[0][2],
                 "Expected exactly one visible V100; no physical-index assumption is made")
            gpu_uuid = gpus[0][1].strip()
            need(gpu_uuid and report.setdefault("selected_gpu_uuid", gpu_uuid) == gpu_uuid,
                 "Visible GPU UUID changed during the experiment")
            memory_free, utilization = int(gpus[0][4]), int(gpus[0][5])
            no_process = not records["processes"] or records["processes"] == "No running processes found"
            snapshot = dict(label=label, attempt=attempt, observed_at_utc=utc(), physical_index=gpus[0][0],
                            uuid=gpu_uuid, name=gpus[0][2], memory_free_mib=memory_free,
                            utilization_percent=utilization, compute_process_list_empty=no_process, raw=records)
            report["gpu_snapshots"].append(snapshot)
            save()
            if not require_idle:
                return
            if not no_process or memory_free < 4096:
                raise Blocked("V100 idle boundary failed; no foreign process will be altered")
            if utilization == 0:
                return
            if attempt != 7:
                time.sleep(0.5)
        raise Blocked("V100 utilization did not settle at the bounded idle boundary")

    def check_build(variant):
        build = root / f"build-c{variant}"
        evidence_path = build / "compile-evidence.json"
        regular(evidence_path, 2 << 20)
        evidence = json.loads(evidence_path.read_text())
        need(evidence.get("passed") is True and evidence.get("certified_variant") == variant
             and evidence.get("certified_enabled") is True and evidence.get("quantizer_rne_enabled") is True
             and evidence.get("tiled_variant") == 1 and evidence.get("certified_prepare_raw_cache_enabled") is False,
             "Build did not match the frozen candidate configuration")
        expected = {name[len(TRAINING):]: value for name, value in frozen_sources.items()
                    if name.startswith(TRAINING + "csrc/")}
        need(evidence.get("source_sha256") == expected
             and evidence.get("fragment_mapping_sha256") == frozen_sources[MAPPING],
             "Compile evidence is not bound to the complete archive native sources")
        need(re.search(r"release 12\.8\b", evidence.get("nvcc_version", "")), "Require CUDA 12.8")
        need(len(evidence.get("jobs", [])) == 4, "Expected library and all three native harness builds")
        for job in evidence["jobs"]:
            need(job.get("returncode") == 0 and job["argv"][0] == NVCC, "A compile job did not pass")
            log = build / job["log"]
            need(log.parent == build and sha(log) == job["log_sha256"], "Compile log changed")
        expected_artifacts = {"libmxfp4_sm70.so", "check_mxfp4", "check_mxfp4_tiled", "check_mxfp4_certified"}
        need(set(evidence.get("artifacts", {})) == expected_artifacts, "Unexpected build artifact set")
        for name, metadata in evidence["artifacts"].items():
            path = build / name
            information = regular(path, 64 << 20)
            need(metadata == {"sha256": sha(path), "bytes": information.st_size}, "Binary differs from compile evidence")
            binaries[str(path)] = metadata["sha256"]
        report["builds"][str(variant)] = dict(compile_evidence_sha256=sha(evidence_path),
                                              artifacts=evidence["artifacts"], binary_bytes_archived=False)

    def validate_stage(spec, row):
        if spec["kind"] == "build":
            check_build(spec["variant"])
            return
        if spec["kind"] == "native":
            records = [json.loads(line) for line in (results / row["stdout"]).read_text().splitlines()
                       if line.startswith("{")]
            need(not any(x.get("passed") is False for x in records), "Native checker emitted a failed record")
            summary = [x for x in records if x.get("type") == spec["summary"]]
            need(len(summary) == 1 and summary[0].get("passed") is True
                 and summary[0].get("gpu_executed") is True and summary[0].get("mismatches") == 0,
                 "Native checker lacks completed exact correctness evidence")
            if spec["summary"] == "certified_summary":
                need(summary[0].get("certified_variant") == spec["variant"]
                     and summary[0].get("full_domain_suite_passed") is True,
                     "Native certified checker used another variant or skipped the full suite")
            row["verified_summary"] = summary[0]
            row["gpu_executed"] = True
            return
        output = Path(spec["output"])
        regular(output, MAX_LOG_BYTES)
        data = json.loads(output.read_text())
        need(data.get("passed") is True and data.get("status") == "passed"
             and data.get("gpu_executed") is True and data.get("all_attempted_work_synchronized") is True,
             "Python stage lacks successful synchronized GPU evidence")
        variant = spec["variant"]
        evidence_sha = report["builds"][str(variant)]["compile_evidence_sha256"]
        need(data.get("library", {}).get("sha256") == binaries[str(root / f"build-c{variant}/libmxfp4_sm70.so")]
             and data["library"].get("compile_evidence_sha256") == evidence_sha,
             "Python stage used another library or compile manifest")
        hashes = data.get("source_sha256")
        need(isinstance(hashes, dict) and hashes and all(frozen_sources.get(
             MAPPING if name == "fragment_mapping.hpp" else TRAINING + name) == value
             for name, value in hashes.items()), "Python stage source hashes differ from archive")
        row.update(output_sha256=sha(output), gpu_executed=True, source_and_library_bound=True)
        if spec["kind"] == "check":
            return
        need(data.get("mode") == spec["mode"] and data.get("seed") == SEED
             and data.get("warmup") == spec["warmup"] and data.get("rounds") == spec["rounds"]
             and data.get("iterations_per_round") == spec["iters"], "Benchmark configuration changed")
        if spec["kind"] == "tile_benchmark":
            need(data.get("actual_first_and_last_timed_tensor_identity_verified") is True
                 and data.get("context_closed") is True, "Actual timed tile outputs were not verified")
            for key in ("initial_all_outputs", "actual_first_timed_all_outputs", "actual_last_timed_all_outputs"):
                metric = data.get(key, {})
                need(metric.get("bit_mismatches") == 0 and metric.get("compared_values") == spec["size"] ** 2,
                     "Tile benchmark lacks full exact output checks")
            fingerprint = data.get("actual_input_sha256_before")
            need(fingerprint == data.get("generated_input_sha256") == data.get("actual_input_sha256_after"),
                 "Packed input bytes changed")
            group = f"tile:{spec['size']}:{spec['mode']}"
            # Keep global acceptance and tile path diagnostics separate, including
            # complete rejection. A passed timing is not a fast-path assertion.
            row["dispatch_diagnostics"] = data.get("actual_last_timed_diagnostics")
            row["fast_path_claimed"] = False
        else:
            need(data.get("actual_last_timed_output_checked") is True,
                 "Actual final timed quantizer result was not checked")
            if spec["kind"] == "workspace_benchmark":
                need(data.get("actual_first_timed_output_checked") is True and data.get("workspace_closed") is True,
                     "Workspace timing lacks actual first output or successful close")
            fingerprint = data.get("input_sha256_before")
            need(fingerprint == data.get("input_sha256_after"), "FP32 input bytes changed")
            group = f"fp32:{spec['size']}"
        need(isinstance(fingerprint, dict) and fingerprint, "Benchmark input fingerprint is missing")
        old = report["input_fingerprints"].setdefault(group, fingerprint)
        need(old == fingerprint, "The fixed comparison order used different input bytes")
        row["input_comparison_group"] = group

    def collect_builds():
        for variant in VARIANTS:
            build = root / f"build-c{variant}"
            if not build.is_dir() or build.is_symlink():
                continue
            target = results / f"build-c{variant}"
            target.mkdir(mode=0o700, exist_ok=True)
            for path in sorted(build.iterdir()):
                if path.suffix in (".json", ".log"):
                    regular(path, MAX_LOG_BYTES)
                    shutil.copyfile(path, target / path.name)

    failure = None
    try:
        save()
        regular(archive, MAX_ARCHIVE_BYTES)
        manifest_path = root / "run-manifest.json"
        regular(manifest_path, 1 << 20)
        manifest = json.loads(manifest_path.read_text())
        script = Path(os.path.abspath(__file__))
        regular(script, 1 << 20)
        report.update(source_archive_sha256=sha(archive), remote_script_sha256=sha(script),
                      run_manifest_sha256=sha(manifest_path), publication_manifest=manifest)
        need(script.parent == root and manifest.get("remote_dir") == str(root), "Run directory binding differs")
        need(manifest.get("schema_version") == 1 and manifest.get("owner") == "jincy26"
             and re.fullmatch(r"[0-9a-f]{40}", manifest.get("source_commit", "")),
             "Publication owner/commit label is missing")
        need(manifest.get("source_archive_sha256") == report["source_archive_sha256"]
             and manifest.get("remote_script_sha256") == report["remote_script_sha256"], "Uploaded source/payload hash differs")
        report.update(source_commit=manifest["source_commit"],
                      run_id=manifest.get("run_id"), job=manifest.get("job"))
        shutil.copyfile(manifest_path, results / "run-manifest.json")
        shutil.copyfile(script, results / "remote_run.py")
        frozen_sources = extract_source(archive, source)
        need(sha(archive) == report["source_archive_sha256"], "Source archive changed during extraction")
        report["source_files_sha256"] = frozen_sources
        write_json(results / "source-files.json", frozen_sources)
        need(Path(PYTHON).is_file() and os.access(PYTHON, os.X_OK), "Required lexical venv Python is unavailable")
        need(Path(NVCC).is_file() and os.access(NVCC, os.X_OK), "Required CUDA 12.8 nvcc is unavailable")
        gpu_idle("preflight")
        run(dict(name="nvcc-version", kind="probe", timeout=20, argv=[NVCC, "--version"]))
        probe = ("import json,sys,numpy,torch; "
                 "assert torch.cuda.is_available() and torch.cuda.device_count()==1; "
                 "p=torch.cuda.get_device_properties(0); "
                 "assert (p.major,p.minor)==(7,0) and 'V100' in p.name; "
                 "print(json.dumps(dict(python=sys.executable,python_version=sys.version,numpy=numpy.__version__,"
                 "torch=torch.__version__,cuda=torch.version.cuda,name=p.name,compute_capability=[p.major,p.minor])))")
        run(dict(name="torch-runtime", kind="probe", timeout=60, argv=[PYTHON, "-c", probe]))
        checks_complete = False
        for spec in plan:
            benchmark = spec["kind"].endswith("_benchmark")
            if benchmark and not checks_complete:
                prior = [x for x in report["stages"] if x.get("kind") in ("build", "native", "check")]
                need(len(prior) == 13 and all(x.get("status") == "passed" and x.get("evidence_validated") is True
                                             for x in prior), "All three builds and ten correctness gates must pass")
                checks_complete = True
                report["all_correctness_gates_passed_before_performance"] = True
            if benchmark:
                gpu_idle("before-" + spec["name"])
            row = run(spec)
            try:
                validate_stage(spec, row)
                if benchmark:
                    gpu_idle("after-" + spec["name"])
                    row["idle_gpu_boundaries_verified"] = True
                    row["continuous_gpu_exclusivity_proven"] = False
                row["evidence_validated"] = True
            except BaseException as error:
                row.update(status="failed", evidence_validated=False, validation_error=str(error))
                raise
            finally:
                save()
        gpu_idle("final", require_idle=False)
        need(source_hashes(source) == frozen_sources, "Final source tree changed")
        need(all(sha(path) == value for path, value in binaries.items()), "Final binary bytes changed")
        need(sha(archive) == report["source_archive_sha256"]
             and sha(script) == report["remote_script_sha256"]
             and sha(manifest_path) == report["run_manifest_sha256"], "Uploaded provenance changed during run")
        report.update(status="passed", passed=True, source_unchanged=True, binaries_unchanged=True)
    except BaseException as error:
        failure = str(error)
        report.update(status="blocked" if isinstance(error, Blocked) else "failed", passed=False,
                      error_type=type(error).__name__, error=str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(sig, signal.SIG_IGN)
        try:
            cleanup(active_process)
            collect_builds()
        except BaseException as error:
            report.update(status="failed", passed=False, collection_error=str(error))
        recorded = {row["name"] for row in report["stages"]}
        for spec in plan:
            if spec["name"] not in recorded:
                report["stages"].append({**spec, "status": "blocked", "executed": False,
                                          "gpu_executed": False, "reason": failure or "Prerequisite did not pass"})
        report.update(finished_at_utc=utc(), elapsed_seconds=time.monotonic() - started,
                      compiled_binary_sha256=binaries, binary_bytes_archived=False)
        files = [p for p in sorted(results.rglob("*")) if p.is_file() and p.name != "completion.json"]
        report["result_files_sha256"] = {p.relative_to(results).as_posix(): sha(p) for p in files}
        # Oversized evidence remains in the owned run directory; a failed run
        # still returns a bounded archive with its omissions explicitly listed.
        included, omitted, size = [], [], 0
        for path in files:
            info = regular(path)
            if size + info.st_size > MAX_RESULT_BYTES - (2 << 20):
                omitted.append(dict(path=path.relative_to(results).as_posix(), bytes=info.st_size, sha256=sha(path)))
            else:
                included.append(path)
                size += info.st_size
        if omitted:
            report.update(status="failed", passed=False, archive_omitted_files=omitted,
                          archive_error="Result byte bound exceeded; omitted files remain remotely")
        save()
        included.append(results / "completion.json")
        archive_manifest = {p.relative_to(results).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
                            for p in included}
        write_json(results / "archive-manifest.json", archive_manifest)
        included.append(results / "archive-manifest.json")
        temporary = root / "results.tar.gz.tmp"
        with tarfile.open(temporary, "w:gz", compresslevel=1) as bundle:
            for path in included:
                regular(path)
                bundle.add(path, arcname="results/" + path.relative_to(results).as_posix(), recursive=False)
        temporary.rename(root / "results.tar.gz")
        print(json.dumps(dict(status=report["status"], passed=report["passed"],
                              results_archive_sha256=sha(root / "results.tar.gz"),
                              results_archive_bytes=(root / "results.tar.gz").stat().st_size)), flush=True)
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    return 0 if report["passed"] else (2 if report["status"] == "blocked" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
