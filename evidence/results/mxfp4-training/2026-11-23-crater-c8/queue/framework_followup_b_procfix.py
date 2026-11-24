#!/usr/bin/env python3
"""Run the blocked framework stages against the completed, unchanged B snapshot.

This standalone runner uses only the standard library. It neither installs
packages nor builds or edits native code. Supply the venv interpreter itself;
its symlink is deliberately NOT resolved for execution. Run only after the
original queue watcher has finished collecting B and is inactive.
"""
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


SOURCE_COMMIT = "80d3a7c2bb0bb551a34b28bd7fda59978b6304ef"
SOURCE_SHA = "62378914e2b6ad7e34e3dd70736cb90aea07c962963494d97062d5f5738b0f57"
GLOBAL_SECONDS, STAGE_SECONDS = 1800, 600
MAX_BYTES = 1024 ** 3
TRAINING = "tools/mxfp4_training/"
MAPPING = "evidence/benchmarks/sm70-packing/study/simulation/round3/cuda/fragment_mapping.hpp"
CHECK_SOURCES = ("check_torch_cuda.py", "reference.py", "torch_ops.py",
                 "csrc/mxfp4_gemm.h", "csrc/mxfp4_gemm.cu",
                 "csrc/mxfp4_gemm_tiled.h", "csrc/mxfp4_gemm_tiled.cu")
BUILD_JOBS = {"shared_library": "libmxfp4_sm70.so",
              "standalone_harness": "check_mxfp4", "tiled_harness": "check_mxfp4_tiled",
              "certified_harness": "check_mxfp4_certified", "performance_benchmark": "bench_mxfp4"}
STAGES = ("python_environment", "torch_integration", "torch_benchmark", "smoke_cuda", "smoke_cuda_tiled")
ARMS = ("fp32", "mx_forward", "mx_forward_dgrad", "mx_forward_dgrad_wgrad", "mx_all_stochastic_grad")
TIMING_JOBS = {"torch_fp32_matmul", "torch_fp16_compute", "torch_fp16_casts_and_matmul",
               "mxfp4_quantize_both", "mxfp4_cuda_quantize_and_gemm", "mxfp4_cuda_tiled_quantize_and_gemm"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def document(path):
    need(Path(path).stat().st_size <= 32 * 1024 * 1024, "JSON exceeds read bound: " + str(path))
    value = json.loads(Path(path).read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    need(isinstance(value, dict), "JSON object required: " + str(path))
    return value


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def safe_name(value):
    need(isinstance(value, str) and value and "\\" not in value, "Invalid relative file name")
    p = PurePosixPath(value)
    need(not p.is_absolute() and ".." not in p.parts, "Unsafe relative file name: " + value)
    return p.as_posix().removeprefix("./")


def regular(path):
    need(stat.S_ISREG(Path(path).lstat().st_mode), "Regular non-symlink file required: " + str(path))


def tree_hashes(root):
    result = {}
    for p in sorted(root.rglob("*")):
        need(not p.is_symlink(), "Symlink found in evidence/source tree: " + str(p))
        if p.is_file():
            result[p.relative_to(root).as_posix()] = sha(p)
        else:
            need(p.is_dir(), "Special file found in tree: " + str(p))
    return result


def archive_hashes(path, prefix=""):
    """Hash safe regular members without extracting or executing archived content."""
    regular(path)
    need(path.stat().st_size <= MAX_BYTES, "Archive exceeds compressed size bound")
    result, total = {}, 0
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            name = safe_name(member.name)
            need(member.isdir() or (member.isfile() and not member.issparse()), "Unsafe archive member")
            if member.isdir():
                continue
            if prefix:
                need(name.startswith(prefix), "Archive member outside required prefix")
                name = name[len(prefix):]
            need(name and name not in result, "Duplicate/empty archive member")
            total += member.size
            need(member.size <= MAX_BYTES and total <= MAX_BYTES, "Archive exceeds expanded size bound")
            h = hashlib.sha256()
            with archive.extractfile(member) as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(block)
            result[name] = h.hexdigest()
    return result


def json_lines(path):
    rows = []
    for line in path.read_text().splitlines():
        if line.lstrip().startswith("{"):
            value = json.loads(line)
            need(isinstance(value, dict), "Expected JSONL object")
            rows.append(value)
    return rows


def ancestors():
    found, pid = set(), os.getpid()
    while pid > 0 and pid not in found:
        found.add(pid)
        try:
            fields = Path("/proc", str(pid), "stat").read_text().rsplit(")", 1)[1].split()
            pid = int(fields[1])
        except (OSError, ValueError, IndexError):
            break
    return found


def no_native_processes(root):
    """Exclude unrelated argv before cwd access; uncertain B candidates fail closed.

    Process inspection is read-only. A same-UID sshd session may prohibit cwd
    readlink even when its stat/cmdline are readable. That does not bind it to B.
    This remains a snapshot of B-related processes, not an exclusivity claim.
    """
    ignore, matches, excluded, disappeared = ancestors(), [], [], []
    uid, root_text = os.geteuid(), str(root)
    source_text = str(root / "source")
    runtime_names = {"crater_remote_run.py", "build_cuda.py", "nvcc", "ptxas", "cicc",
                     "fatbinary", "nvlink", "check_mxfp4", "check_mxfp4_tiled",
                     "check_mxfp4_certified", "bench_mxfp4", "bench_mxfp4_fused"}

    def identity(p):
        value = (p / "stat").read_text()
        left, right = value.find("("), value.rfind(")")
        fields = value[right + 1:].split()
        need(left > 0 and right > left and len(fields) >= 20,
             "Malformed process identity at overlap gate: " + p.name)
        need(int(value[:left].strip()) == int(p.name), "Process PID identity differs")
        return {"pid": int(p.name), "comm": value[left + 1:right], "state": fields[0],
                "start_time_ticks": int(fields[19])}

    def same_process(before, after):
        need(before["pid"] == after["pid"] and before["start_time_ticks"] == after["start_time_ticks"],
             "Process identity changed during overlap gate: " + str(before["pid"]))

    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) in ignore:
            continue
        phase, initial, candidate, argv = "uid", None, False, []
        try:
            if p.stat().st_uid != uid:
                continue
            phase = "stat"
            initial = identity(p)
            if initial["state"] in ("Z", "X", "x"):
                excluded.append({**initial, "reason": "nonexecuting_process_state", "cwd_read": False})
                continue
            phase = "cmdline"
            argv = [s.decode(errors="replace") for s in (p / "cmdline").read_bytes().split(b"\0") if s]
            after_argv = identity(p)
            same_process(initial, after_argv)
            if after_argv["state"] in ("Z", "X", "x"):
                excluded.append({**after_argv, "reason": "became_nonexecuting", "cwd_read": False})
                continue
            need(argv, "Live owned process has no inspectable argv at overlap gate: " + p.name)
            root_marker = any(root_text in a for a in argv)
            module_marker = any("tools.mxfp4_training" in a for a in argv)
            named_runtime = any(Path(a).name in runtime_names for a in argv)
            candidate = root_marker or module_marker or named_runtime
            if not candidate:
                # In particular sshd: mxfp4owner@notty has no B path/module/name.
                # Do not inspect unrelated cwd and do not record other argv data.
                excluded.append({**after_argv, "argv0_basename": Path(argv[0]).name,
                                 "reason": "argv_has_no_B_path_module_or_runtime_marker",
                                 "cwd_read": False})
                continue
            phase = "candidate_cwd"
            cwd = os.readlink(p / "cwd")
            after_cwd = identity(p)
            same_process(initial, after_cwd)
            if after_cwd["state"] in ("Z", "X", "x"):
                excluded.append({**after_cwd, "reason": "became_nonexecuting", "cwd_read": True})
                continue
            absolute_runtime = any(a.startswith(root_text + "/build/") or
                                   a.startswith(root_text + "/build-fused-") or
                                   a == str(root / "crater_remote_run.py") for a in argv)
            payload = any(Path(a).name == "crater_remote_run.py" for a in argv) and (
                root_text in argv or cwd == root_text)
            in_source = cwd == source_text or cwd.startswith(source_text + "/")
            in_build = cwd == root_text + "/build" or cwd.startswith(root_text + "/build/") or \
                       cwd.startswith(root_text + "/build-fused-")
            source_runtime = in_source and (module_marker or named_runtime)
            relative_native = in_build and named_runtime
            if absolute_runtime or payload or source_runtime or relative_native:
                matches.append({**after_cwd, "argv": argv, "cwd": cwd})
            else:
                excluded.append({**after_cwd, "argv0_basename": Path(argv[0]).name,
                                 "reason": "candidate_cwd_and_argv_do_not_bind_B_runtime", "cwd_read": True})
        except (FileNotFoundError, ProcessLookupError):
            disappeared.append({"pid": int(p.name), "phase": phase})
            continue
        except PermissionError as error:
            # A process can exit between readable stat/cmdline and cwd. Only a
            # confirmed exit/zombie permits exclusion; live/unknown candidates
            # and unreadable argv remain fail-closed, with useful diagnostics.
            if initial is not None:
                try:
                    latest = identity(p)
                    same_process(initial, latest)
                except (FileNotFoundError, ProcessLookupError):
                    disappeared.append({"pid": int(p.name), "phase": phase})
                    continue
                except PermissionError:
                    latest = None
                if latest is not None and latest["state"] in ("Z", "X", "x"):
                    excluded.append({**latest, "reason": "nonexecuting_after_access_error", "cwd_read": False})
                    continue
            diagnostic = {"pid": int(p.name), "phase": phase, "errno": error.errno,
                          "B_candidate_from_argv": candidate, "initial_identity": initial}
            raise RuntimeError("Cannot classify an owned process before overlap gate: " +
                               json.dumps(diagnostic, sort_keys=True)) from error
    need(not matches, "B-owned payload/native process still present: " + json.dumps(matches))
    return {"checked_at_utc": utc(), "same_uid_B_path_processes": [],
            "excluded_same_uid_processes": excluded, "disappeared_during_scan": disappeared,
            "scope": "B argv/cwd only; unrelated argv excluded before cwd; no claim of exclusive access to other GPU users"}


def native_suite(rows, certified=False):
    summary = [r for r in rows if r.get("type") == "summary"]
    need(len(summary) == 1, "Native summary missing/duplicated")
    s = summary[0]
    for k, v in {"passed": True, "gpu_executed": True, "kernel_execution_completed": True,
                 "actual_V100": True, "cases": 46, "values": 175429, "mismatches": 0}.items():
        need(s.get(k) == v, "Native full-domain summary mismatch: " + k)
    cases = [r for r in rows if r.get("type") == "correctness"]
    need(all(r.get("gpu_executed") is True and r.get("mismatches") == 0 and
             r.get("values") == r.get("M", -1) * r.get("N", -1) for r in cases), "Native case failed")
    if not certified:
        need(len(cases) == 46 and sum(r["values"] for r in cases) == 175429, "Native case count mismatch")
        return
    cert = [r for r in rows if r.get("type") == "certified_summary"]
    need(len(cert) == 1, "Certified summary missing/duplicated")
    for k, v in {"passed": True, "gpu_executed": True, "certified_variant": 2,
                 "full_domain_suite_passed": True, "extra_fixture_cases": 20,
                 "extra_fixture_values": 230456, "context_reuse_calls": 3, "accepted_calls": 23,
                 "fallback_calls": 46, "canonical_positive_zero_values": 2950, "mismatches": 0}.items():
        need(cert[0].get(k) == v, "C2 certificate summary mismatch: " + k)
    reuse = [r for r in rows if r.get("type") == "context_reuse"]
    need(len(cases) == 66 and sum(r["values"] for r in cases) == 405885 and len(reuse) == 3,
         "C2 fixture/reuse count mismatch")
    need([r.get("accepted") for r in reuse] == [True, False, True] and
         [r.get("submitted_calls") for r in reuse] == [1, 2, 3] and
         all(r.get("passed") is True and r.get("mismatches") == 0 and r.get("values") == 15 for r in reuse),
         "C2 reuse outputs not verified")


def verify_native(root):
    source, results, build = root / "source", root / "results", root / "build"
    for path in (root, source, results, build):
        need(path.is_dir() and not path.is_symlink(), "Expected fixed native directory: " + str(path))
    marker = root / "runner-payload-exit-code"
    regular(marker)
    need(marker.read_text().strip() == "2", "B must have exited with code 2 before the follow-up")
    manifest_path, completion_path = root / "run-manifest.json", results / "completion.json"
    manifest, completed = document(manifest_path), document(completion_path)
    need(manifest.get("source_commit") == SOURCE_COMMIT and manifest.get("source_archive_sha256") == SOURCE_SHA,
         "B source commit/archive binding mismatch")
    need(manifest.get("remote_dir") == str(root) and manifest.get("run_id"), "B run-directory identity mismatch")
    need(manifest.get("owner") == "jincy26" and manifest.get("endpoint", {}).get("owner") == "jincy26" and
         manifest.get("endpoint", {}).get("job") == manifest.get("job"), "B owner/job binding mismatch")
    need(completed.get("status") == "blocked" and completed.get("finished_at_utc") and
         completed.get("source_archive_sha256") == SOURCE_SHA, "B completion not finalized as blocked")
    stages = {s.get("name"): s for s in completed.get("stages", [])}
    need(len(stages) == len(completed.get("stages", [])) and stages and
         all(s.get("status") in ("passed", "blocked") for s in stages.values()), "B has duplicate/unfinished/failed stages")
    original_blocked = {n: stages.get(n) for n in ("torch_integration", "torch_benchmark", "gpu_training_smoke")}
    need(all(s and s.get("status") == "blocked" and s.get("executed") is False for s in original_blocked.values()),
         "Original B framework stages were not all unexecuted blocked stages")
    disk_results = tree_hashes(results)
    need(completed.get("result_files_sha256") == {k: v for k, v in disk_results.items() if k != "completion.json"},
         "B result files differ from its completion hash map")
    archived = archive_hashes(root / "results.tar.gz", "results/")
    need(archived == disk_results, "B results archive, including completion bytes, differs from finalized disk results")
    need(sha(root / "source.tar.gz") == SOURCE_SHA, "B source archive SHA mismatch")
    source_map = archive_hashes(root / "source.tar.gz")
    need(source_map == completed.get("source_files_sha256"), "B source member hashes differ from completion")
    actual_source = tree_hashes(source)
    need(all(actual_source.get(k) == v for k, v in source_map.items()), "B extracted source content changed")
    extras = {k: v for k, v in actual_source.items() if k not in source_map}
    need(all("__pycache__" in PurePosixPath(k).parts and k.endswith(".pyc") for k in extras),
         "Unexpected extra file in B source; only pre-existing Python bytecode is allowed")
    payload = root / "crater_remote_run.py"
    need(sha(payload) == manifest.get("remote_script_sha256") == source_map.get(TRAINING + "crater_remote_run.py"),
         "B payload/manifest/archived source hashes differ")
    build_path = build / "compile-evidence.json"
    need(sha(build_path) == disk_results.get("build/compile-evidence.json"), "Live/archived build manifest differs")
    compile_record = document(build_path)
    need(compile_record.get("passed") is True and compile_record.get("certified_enabled") is True and
         compile_record.get("certified_variant") == 2 and compile_record.get("tiled_variant") == 1,
         "Original library was not a successful C2/tiled1 build")
    need(compile_record.get("source_sha256") == {k[len(TRAINING):]: v for k, v in source_map.items()
                                                if k.startswith(TRAINING + "csrc/")}, "Build source hashes differ")
    need(compile_record.get("fragment_mapping_sha256") == source_map.get(MAPPING), "Fragment mapping hash differs")
    jobs, artifacts = compile_record.get("jobs", []), compile_record.get("artifacts", {})
    need(len(jobs) == 5 and {j.get("name") for j in jobs} == set(BUILD_JOBS) and
         set(artifacts) == set(BUILD_JOBS.values()), "Incomplete native build manifest")
    protected = {str(manifest_path): sha(manifest_path), str(completion_path): sha(completion_path),
                 str(marker): sha(marker), str(root / "source.tar.gz"): SOURCE_SHA,
                 str(root / "results.tar.gz"): sha(root / "results.tar.gz"), str(payload): sha(payload),
                 str(build_path): sha(build_path)}
    for j in jobs:
        name = BUILD_JOBS[j["name"]]
        argv = j.get("argv", [])
        need(j.get("returncode") == 0 and argv and argv[-1] == str(build / name) and
             "-gencode=arch=compute_70,code=[sm_70,compute_70]" in argv and
             "-DMXFP4_CERTIFIED_VARIANT=2" in argv and "-DMXFP4_TILED_VARIANT=1" in argv,
             "Native build command/return code mismatch")
        log_name = safe_name(j.get("log"))
        need("/" not in log_name and sha(build / log_name) == j.get("log_sha256") == disk_results.get("build/" + log_name),
             "Compile log SHA mismatch")
        binary = build / name
        regular(binary)
        need(sha(binary) == artifacts[name].get("sha256") and binary.stat().st_size == artifacts[name].get("bytes"),
             "Compiled binary SHA/size mismatch: " + name)
        protected[str(binary)] = sha(binary)
        protected[str(build / log_name)] = sha(build / log_name)
    gate = stages.get("cuda_build", {})
    need(gate.get("status") == "passed" and gate.get("executed") is True and gate.get("returncode") == 0,
         "Native build stage did not pass")
    for name, binary in (("native_baseline", "check_mxfp4"), ("native_tiled", "check_mxfp4_tiled"),
                         ("native_certified", "check_mxfp4_certified")):
        stage = stages.get(name, {})
        need(stage.get("status") == "passed" and stage.get("executed") is True and stage.get("returncode") == 0,
             "Native prerequisite did not pass: " + name)
        need(stage.get("argv") == [str(build / binary), "--device", "0", "--correctness-only"],
             "Native prerequisite command differs: " + name)
        output = safe_name(stage.get("stdout"))
        need(output in disk_results, "Missing native correctness log")
        native_suite(json_lines(results / output), name == "native_certified")
    return {"manifest": manifest, "completion": completed, "compile": compile_record,
            "source_sha256": source_map, "all_source_files_sha256": actual_source,
            "preexisting_bytecode_sha256": extras, "protected_files_sha256": protected,
            "original_framework_blocked": original_blocked, "original_torch_probes": completed.get("installed_torch_probes"),
            "native_overlap_check": no_native_processes(root), "native_results_sha256": disk_results}


def assert_unchanged(root, binding):
    need(tree_hashes(root / "source") == binding["all_source_files_sha256"], "B source tree changed during follow-up")
    for p, expected in binding["protected_files_sha256"].items():
        regular(p)
        need(sha(p) == expected, "B protected file changed: " + p)
    need(tree_hashes(root / "results") == binding["native_results_sha256"], "Original B evidence changed")


def validate_framework(path, name, binding, check_path):
    r = document(path)
    if r.get("status") == "blocked":
        return "blocked", {"report_error": r.get("error"), "gpu_executed": r.get("gpu_executed")}
    need(r.get("passed") is True and r.get("gpu_executed") is True, "Framework report is not a passing GPU result")
    need(r.get("device") == "cuda:0" and r.get("llm_convergence_validated") is False,
         "Unexpected device/training scope")
    if name.startswith("smoke_"):
        backend = name.removeprefix("smoke_")
        need(r.get("backend") == backend and r.get("seed") == 20261123 and r.get("learning_rate") == .005 and
             r.get("optimizer") == "AdamW FP32" and r.get("performance_claim") is None, "Smoke contract changed")
        expected = {k[len(TRAINING):]: v for k, v in binding["source_sha256"].items()
                    if k.startswith(TRAINING) and "/" not in k[len(TRAINING):] and k.endswith(".py")}
        need(r.get("source_sha256") == expected, "Smoke source hash mismatch")
        arms = r.get("arms", [])
        need(len(arms) == 5 and [a.get("name") for a in arms] == list(ARMS), "Smoke did not run all five arms")
        for arm in arms:
            need(all(arm.get(k) is True for k in ("passed", "parameters_finite", "master_parameters_fp32", "optimizer_states_fp32")),
                 "Smoke arm correctness failed")
            losses = arm.get("training_losses", [])
            need(arm.get("steps") == 40 and len(losses) == 40 and all(math.isfinite(x) for x in losses) and
                 arm.get("changed_parameter_tensors", 0) > 0 and
                 math.isfinite(arm["validation_loss_before"]) and math.isfinite(arm["validation_loss_after"]) and
                 arm["validation_loss_after"] < arm["validation_loss_before"], "Smoke losses/updates invalid")
            need(arm.get("recipe", {}).get("backend") == backend, "Smoke recipe backend differs")
        return "passed", {"gpu_executed": True, "arms": 5, "steps_per_arm": 40, "tested_backends": [backend]}
    need(r.get("status") == "passed" and r.get("all_attempted_work_synchronized") is True and
         r.get("nondefault_stream") is True and r.get("gpu", {}).get("actual_SM70") is True and
         r.get("gpu", {}).get("actual_V100") is True and r.get("cpu_fallback") is False,
         "Framework synchronization/device status invalid")
    lib = binding["compile"]["artifacts"]["libmxfp4_sm70.so"]
    library = r.get("library", {})
    need(library.get("sha256") == lib["sha256"] and library.get("bytes") == lib["bytes"] and
         library.get("path") == binding["library_path"], "Framework library binding mismatch")
    names = CHECK_SOURCES + (("bench_torch_cuda.py",) if name == "torch_benchmark" else ())
    need(r.get("source_sha256") == {n: binding["source_sha256"][TRAINING + n] for n in names},
         "Framework source hash mismatch")
    if name == "torch_integration":
        cases = r.get("cases", [])
        need(r.get("source_unchanged_during_check") is True and r.get("seed") == 20261123 and
             r.get("completed_cases") == 119 and len(cases) == 119 and all(c.get("passed") is True for c in cases),
             "The complete 119-case GPU integration suite did not pass")
        return "passed", {"gpu_executed": True, "completed_cases": 119, "tested_backends": ["cuda", "cuda_tiled"]}
    need(r.get("source_unchanged_during_benchmark") is True and r.get("native_certified_capi_benchmarked") is False and
         r.get("native_certified_workspace_allocated") is False and r.get("shape") == {
             "M": 1024, "N": 1024, "logical_K": 1024, "padded_K": 1024} and
         r.get("input", {}).get("seed") == 20261123 and r.get("rounds") == 3 and
         r.get("iterations_per_round") == 3 and r.get("warmup_per_job") == 1, "Torch benchmark contract differs")
    need(r.get("prerequisite", {}).get("sha256") == sha(check_path) and
         r.get("prerequisite", {}).get("completed_cases") == 119, "Benchmark not bound to this passing check")
    validation = r.get("validation", {})
    need(all(validation.get(k) is True for k in ("passed_before_timing", "passed_after_timing",
              "all_input_and_output_values_finite", "final_timed_rne_codes_and_scales_exact")), "Benchmark validation failed")
    for k in ("packed_cuda_vs_tiled_all_outputs", "final_timed_packed_cuda_vs_tiled_all_outputs"):
        need(validation.get(k, {}).get("exact_mismatches") == 0 and validation[k].get("values") == 1024 ** 2 and
             validation[k].get("all_finite") is True, "Benchmark packed outputs differ")
    repeats = validation.get("final_timed_vs_initial_all_outputs", {})
    need(set(repeats) == TIMING_JOBS - {"mxfp4_quantize_both"} and
         all(v.get("exact_mismatches") == 0 and v.get("values") == 1024 ** 2 for v in repeats.values()),
         "Final timed outputs did not match initial outputs")
    timings = r.get("timings", [])
    need(len(timings) == 6 and {j.get("name") for j in timings} == TIMING_JOBS, "Incomplete benchmark jobs")
    for job in timings:
        for k in ("cuda_event_round_ms", "synchronized_wall_round_ms"):
            need(len(job.get(k, [])) == 3 and all(math.isfinite(x) and x > 0 for x in job[k]), "Invalid timing rounds")
    return "passed", {"gpu_executed": True, "timed_jobs": 6, "tested_backends": ["cuda", "cuda_tiled"],
                       "native_certified_capi_benchmarked": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", required=True)
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--python", required=True, help="Absolute venv executable path; symlink is not resolved")
    args = parser.parse_args()
    for value in (args.native_root, args.run_directory, args.python):
        if not os.path.isabs(value):
            parser.error("all three paths must be explicitly absolute")
    root = Path(args.native_root).resolve(strict=True)
    run = Path(args.run_directory).absolute()
    need(run.parent.is_dir() and not run.exists() and not run.is_symlink(), "New run directory must not exist; parent must exist")
    run = run.parent.resolve(strict=True) / run.name
    need(root != run and root not in run.parents and run not in root.parents, "Follow-up directory must be separate from B")
    python = os.path.abspath(args.python)  # Never Path.resolve/realpath this execution argument.
    need(Path(python).is_file() and os.access(python, os.X_OK), "Explicit Python path is not executable")
    started = time.monotonic()
    deadline = started + GLOBAL_SECONDS
    lock_name = "/tmp/mxfp4-framework-followup-" + str(os.geteuid()) + "-" + hashlib.sha256(str(root).encode()).hexdigest()[:20] + ".lock"
    lock_fd = os.open(lock_name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    need(os.fstat(lock_fd).st_uid == os.geteuid() and stat.S_ISREG(os.fstat(lock_fd).st_mode), "Unsafe follow-up lock")
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run.mkdir(mode=0o700, exist_ok=False)
    results = run / "results"
    results.mkdir(mode=0o700)
    report = {"schema_version": 1, "purpose": "separate recovery of originally blocked B framework stages",
              "status": "running", "passed": False, "started_at_utc": utc(), "stages": [],
              "native_root": str(root), "run_directory": str(run), "python_execution_path": python,
              "global_budget_seconds": GLOBAL_SECONDS, "maximum_stage_seconds": STAGE_SECONDS,
              "cleanup_reserve_seconds": 15, "gpu_executed": False,
              "source_commit": SOURCE_COMMIT, "source_archive_sha256": SOURCE_SHA,
              "library_certified_variant": 2, "tested_torch_backends": ["cuda", "cuda_tiled"],
              "certified_or_C8_torch_backend_tested": False, "full_training_executed": False,
              "llm_convergence_validated": False, "packages_installed": False, "native_rebuilt": False,
              "services_stopped": False, "clock_or_power_changes": False,
              "original_blocked_conclusion_rewritten": False,
              "training_scope": "synthetic tiny causal-decoder integration; no LLM pretraining/convergence claim"}
    active = None
    binding = None
    terminated = set()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    # Fixed snapshot imports take precedence; do not import another checkout via PYTHONPATH.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPYCACHEPREFIX", None)

    def save():
        write_json(results / "completion.json", report)

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
                    process.wait(timeout=min(5, max(.1, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
        process.wait(timeout=5)
        terminated.add(process.pid)

    def interrupted(signum, _frame):
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(s, signal.SIG_IGN)
        cleanup(active)
        raise InterruptedError("Follow-up interrupted by signal " + str(signum))

    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, interrupted)

    def block(name, reason):
        report["stages"].append({"name": name, "status": "blocked", "executed": False,
                                 "gpu_executed": False, "reason": reason})
        save()

    def stage(name, argv, output=None):
        nonlocal active
        remaining = min(STAGE_SECONDS, deadline - time.monotonic() - 15)
        if remaining <= 0:
            block(name, "Global 1800-second budget exhausted")
            return False
        assert_unchanged(root, binding)
        overlap = no_native_processes(root)
        rec = {"name": name, "argv": argv, "cwd": str(root / "source"), "status": "running", "executed": False,
               "started_at_utc": utc(), "timeout_seconds": remaining, "gpu_executed": None,
               "stdout": name + ".stdout", "stderr": name + ".stderr", "native_overlap_check": overlap,
               "library_sha256": binding["compile"]["artifacts"]["libmxfp4_sm70.so"]["sha256"]}
        report["stages"].append(rec)
        save()
        begin, process = time.monotonic(), None
        try:
            with (results / rec["stdout"]).open("xb") as out, (results / rec["stderr"]).open("xb") as err:
                process = subprocess.Popen(argv, cwd=str(root / "source"), env=env, stdout=out, stderr=err,
                                           start_new_session=True)
                active = process
                rec.update(executed=True, owned_process_group=process.pid)
                rec["returncode"] = process.wait(timeout=max(.1, min(remaining, deadline - time.monotonic() - 10)))
            cleanup(process)
            active = None
            assert_unchanged(root, binding)
            if output is not None and output.is_file():
                rec["output"] = output.name
                rec["output_sha256"] = sha(output)
                if process.returncode == 2 and document(output).get("status") == "blocked":
                    rec.update(status="blocked", reason=document(output).get("error"),
                               gpu_executed=document(output).get("gpu_executed"))
                    return False
            need(process.returncode == 0, "Process exited with code " + str(process.returncode))
            if name == "python_environment":
                rows = json_lines(results / rec["stdout"])
                need(len(rows) == 1 and rows[0].get("available") is True and rows[0].get("executable") == python and
                     rows[0].get("numpy") and rows[0].get("torch") and rows[0].get("cuda") and
                     rows[0].get("compute_capability") == [7, 0] and "V100" in rows[0].get("device_name", ""),
                     "Explicit venv lacks NumPy/CUDA Torch or uses a different interpreter/device")
                rec.update(status="passed", gpu_executed=False, environment=rows[0],
                           note="Probe initializes CUDA context/device metadata but runs no GEMM/training workload")
            else:
                status, details = validate_framework(output, name, binding, results / "torch-cuda-check.json")
                rec.update(status=status, content_validation=details, gpu_executed=details.get("gpu_executed"))
            return rec["status"] == "passed"
        except subprocess.TimeoutExpired:
            cleanup(process)
            rec.update(status="failed", error="Stage/global timeout", gpu_executed=None)
            return False
        except (Exception, KeyboardInterrupt) as error:
            rec.update(status="failed", error_type=type(error).__name__, error=str(error))
            if isinstance(error, (InterruptedError, KeyboardInterrupt)):
                raise
            return False
        finally:
            cleanup(process)
            active = None
            rec.update(returncode=process.returncode if process is not None else None,
                       wall_seconds=time.monotonic() - begin, finished_at_utc=utc())
            if not rec["executed"]:
                rec["gpu_executed"] = False
            save()
            print(json.dumps({"stage": name, "status": rec["status"]}), flush=True)

    try:
        save()
        binding = verify_native(root)
        library = root / "build/libmxfp4_sm70.so"
        binding["library_path"] = str(library)
        report["native_prerequisites_verified"] = True
        report["original_framework_blocked"] = binding["original_framework_blocked"]
        report["original_torch_probes"] = binding["original_torch_probes"]
        report["original_failure_context"] = "B used Path.resolve for candidate Python paths, losing /opt/venv identity; originals remain unchanged"
        report["library_sha256"] = sha(library)
        provenance = results / "provenance"
        provenance.mkdir()
        for original, name in ((root / "run-manifest.json", "B-run-manifest.json"),
                               (root / "results/completion.json", "B-completion.json"),
                               (root / "build/compile-evidence.json", "B-compile-evidence.json"),
                               (root / "runner-payload-exit-code", "B-runner-payload-exit-code")):
            shutil.copyfile(original, provenance / name)
        script = Path(__file__).absolute()
        shutil.copyfile(script, provenance / "mxfp4_framework_followup.py")
        write_json(provenance / "verified-bindings.json", {k: v for k, v in binding.items()
                   if k not in ("completion", "compile")})
        env["MXFP4_CUDA_LIBRARY"] = str(library)
        compiler = Path(binding["compile"]["jobs"][0]["argv"][0])
        cuda_lib = str(compiler.parent.parent / "lib64")
        env["LD_LIBRARY_PATH"] = cuda_lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        write_json(results / "run-manifest.json", {"schema_version": 1, "native_run_id": binding["manifest"]["run_id"],
                   "native_root": str(root), "run_directory": str(run), "source_commit": SOURCE_COMMIT,
                   "source_archive_sha256": SOURCE_SHA, "helper_sha256": sha(script), "python_execution_path": python,
                   "MXFP4_CUDA_LIBRARY": str(library), "library_sha256": sha(library),
                   "cuda_lib_directory": cuda_lib, "PYTHONDONTWRITEBYTECODE": "1", "global_budget_seconds": GLOBAL_SECONDS,
                   "maximum_stage_seconds": STAGE_SECONDS, "native_prerequisites_verified": True,
                   "original_payload_results_unchanged": True, "certified_or_C8_torch_backend_tested": False})
        probe = ("import sys,json,numpy,torch; "
                 "a=torch.cuda.is_available(); "
                 "print(json.dumps(dict(executable=sys.executable,prefix=sys.prefix,base_prefix=sys.base_prefix,"
                 "python=sys.version,numpy=numpy.__version__,torch=torch.__version__,cuda=torch.version.cuda,"
                 "available=a,device_name=torch.cuda.get_device_name(0) if a else None,"
                 "compute_capability=list(torch.cuda.get_device_capability(0)) if a else None))); "
                 "sys.exit(0 if sys.version_info >= (3,8) and a else 2)")
        if stage("python_environment", [python, "-B", "-c", probe]):
            check = results / "torch-cuda-check.json"
            if stage("torch_integration", [python, "-B", "-m", "tools.mxfp4_training.check_torch_cuda", "--output", str(check)], check):
                bench = results / "torch-benchmark.json"
                stage("torch_benchmark", [python, "-B", "-m", "tools.mxfp4_training.bench_torch_cuda",
                                          "--check-report", str(check), "--output", str(bench)], bench)
                for backend in ("cuda", "cuda_tiled"):
                    output = results / ("smoke-" + backend + ".json")
                    stage("smoke_" + backend, [python, "-B", "-m", "tools.mxfp4_training.pretrain_smoke", "--backend", backend,
                                              "--device", "cuda:0", "--steps", "40", "--seed", "20261123", "--lr", "0.005",
                                              "--output", str(output)], output)
        seen = {s["name"] for s in report["stages"]}
        for name in STAGES:
            if name not in seen:
                block(name, "Requires the explicit Python and complete 119-case integration gates")
        assert_unchanged(root, binding)
        report["original_source_library_and_evidence_unchanged"] = True
        states = {s["status"] for s in report["stages"]}
        report["status"] = "failed" if "failed" in states else ("blocked" if "blocked" in states else "passed")
        report["passed"] = report["status"] == "passed"
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", passed=False, error_type=type(error).__name__, error=str(error))
        seen = {s["name"] for s in report["stages"]}
        for name in STAGES:
            if name not in seen:
                block(name, "Follow-up prerequisite/runner failure: " + type(error).__name__ + ": " + str(error))
    finally:
        cleanup(active)
        if binding is not None:
            try:
                assert_unchanged(root, binding)
                report["original_source_library_and_evidence_unchanged"] = True
            except Exception as error:
                report.update(status="failed", passed=False, original_source_library_and_evidence_unchanged=False,
                              integrity_error=str(error))
        values = [s.get("gpu_executed") for s in report["stages"] if s.get("executed")]
        report["gpu_executed"] = True if True in values else (None if None in values else False)
        report.update(finished_at_utc=utc(), wall_seconds=time.monotonic() - started)
        report["result_files_sha256"] = {k: v for k, v in tree_hashes(results).items() if k != "completion.json"}
        save()
        need(sum(p.stat().st_size for p in results.rglob("*") if p.is_file()) <= MAX_BYTES, "Follow-up archive exceeds size bound")
        archive = run / "results.tar.gz"
        with tarfile.open(archive, "x:gz") as output:
            output.add(results, arcname="results", recursive=True)
        receipt = {"status": report["status"], "passed": report["passed"], "results_archive": str(archive),
                   "sha256": sha(archive), "bytes": archive.stat().st_size,
                   "completion_sha256": sha(results / "completion.json"), "source_commit": SOURCE_COMMIT,
                   "source_archive_sha256": SOURCE_SHA, "original_B_results_rewritten": False}
        write_json(run / "archive-manifest.json", receipt)
        print(json.dumps(receipt), flush=True)
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    return {"passed": 0, "blocked": 2}.get(report["status"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
