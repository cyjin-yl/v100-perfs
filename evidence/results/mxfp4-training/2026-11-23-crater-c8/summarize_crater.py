#!/usr/bin/env python3
"""Read-only audit of a real frozen MXFP4 Crater result archive (stdlib only).

No extraction, imported archived code, subprocesses, network, CUDA, or fabricated
samples. The only write is a new --output JSON; an existing output is refused.
Trusted expected hashes/job/commit must come from the independent collection and
publication records, not be invented from this script's observed hashes.

Exit 0: audit verified and payload passed; 2: audit verified, payload blocked or
failed; 1: provenance/schema/evidence inconsistency. Native comparison rows are
eligible only after their own complete build/correctness/benchmark gates pass.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import sys
import tarfile

MIB = 1024 * 1024
PREFIX = "tools/mxfp4_training/"
PAYLOAD = PREFIX + "crater_remote_run.py"
MAPPING = "evidence/benchmarks/sm70-packing/study/simulation/round3/cuda/fragment_mapping.hpp"
SIZES = (2048, 4096)
VARIANTS = tuple(range(2, 9))
COMPARISON = [f"control_c2_benchmark_{n}" for n in SIZES]
for _v in range(3, 9):
    COMPARISON += [f"fused_c{_v}_build", f"fused_c{_v}_correctness"]
    COMPARISON += [f"fused_c{_v}_benchmark_{n}" for n in SIZES]
FIRST = ("gpu_environment", "cuda_build", "native_baseline", "native_tiled",
         "native_certified", "native_benchmark", "torch_integration", "torch_benchmark",
         "smoke_cuda", "smoke_cuda_tiled")
JOBS = {"shared_library": "libmxfp4_sm70.so", "standalone_harness": "check_mxfp4",
        "tiled_harness": "check_mxfp4_tiled", "certified_harness": "check_mxfp4_certified",
        "performance_benchmark": "bench_mxfp4"}
CHECK_SOURCES = {"check_torch_cuda.py", "reference.py", "torch_ops.py",
                 "csrc/mxfp4_gemm.h", "csrc/mxfp4_gemm.cu",
                 "csrc/mxfp4_gemm_tiled.h", "csrc/mxfp4_gemm_tiled.cu"}
NATIVE_BACKENDS = {"mxfp4_baseline", "mxfp4_tiled", "mxfp4_certified_total",
                   "cublas_compute", "cublas_dequant", "cublas_total"}


class AuditError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise AuditError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def object_pairs(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def json_load(data):
    def reject_constant(value):
        raise AuditError(f"non-JSON numeric constant {value}")
    result = json.loads(data, object_pairs_hook=object_pairs, parse_constant=reject_constant)
    need(isinstance(result, dict), "expected a JSON object")
    return result


def path_name(name):
    need(isinstance(name, str) and name and "\\" not in name and "\0" not in name,
         "invalid archive/member path")
    p = PurePosixPath(name)
    need(not p.is_absolute() and ".." not in p.parts and str(p) != ".", f"unsafe path {name!r}")
    return str(p)


class Bundle:
    def __init__(self, path, expected_sha, expanded_limit, prefix=None):
        self.path = str(path.resolve())
        need(valid_hash(expected_sha), "expected archive SHA must be lowercase SHA-256")
        need(path.stat().st_size <= expanded_limit, f"compressed archive too large: {path}")
        raw = path.read_bytes()
        need(len(raw) <= expanded_limit and sha(raw) == expected_sha, f"archive SHA mismatch: {path}")
        self.archive_sha = expected_sha
        self.archive_bytes = len(raw)
        self.files = {}
        self.hashes = {}
        seen = set()
        expanded = 0
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as packed:
            for member in packed:
                need(len(seen) < 20000, "too many archive members")
                name = path_name(member.name)
                need(name not in seen, f"duplicate archive member: {name}")
                seen.add(name)
                need((member.isdir() or member.isfile()) and not member.issparse(),
                     f"non-regular/sparse archive member: {name}")
                need(member.size >= 0, f"negative member size: {name}")
                expanded += member.size
                need(expanded <= expanded_limit, "archive exceeds expanded-size bound")
                if prefix:
                    need(name == prefix or name.startswith(prefix + "/"),
                         f"unexpected archive prefix: {name}")
                if member.isdir():
                    continue
                key = name[len(prefix) + 1:] if prefix else name
                need(bool(key) and name != prefix, "archive root cannot be a regular file")
                handle = packed.extractfile(member)
                need(handle is not None, f"unreadable archive member: {name}")
                data = handle.read(member.size + 1)
                need(len(data) == member.size, f"member size mismatch: {name}")
                self.files[key] = data
                self.hashes[key] = sha(data)
        self.expanded_bytes = expanded
        need(self.files, f"empty archive: {path}")

    def document(self, name):
        need(name in self.files, f"missing artifact: {name}")
        return json_load(self.files[name])

    def metadata(self):
        return {"path": self.path, "sha256": self.archive_sha, "bytes": self.archive_bytes,
                "expanded_bytes": self.expanded_bytes, "regular_members": len(self.files),
                "member_sha256": self.hashes}


def local_document(path):
    data = path.read_bytes()
    return json_load(data), {"path": str(path.resolve()), "sha256": sha(data), "bytes": len(data)}


def close(actual, expected, label):
    need(isinstance(actual, (float, int)) and not isinstance(actual, bool)
         and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=2e-8, abs_tol=1e-10),
         f"{label}: reported {actual!r}, recomputed {expected!r}")


def metric(record, values=None, exact=True):
    need(isinstance(record, dict), "missing numeric comparison metric")
    if values is not None:
        need(record.get("values") == values, "comparison element count mismatch")
    if exact:
        need(record.get("exact_mismatches") == 0, "exact comparison mismatch")
    if "tolerance_failures" in record:
        need(record["tolerance_failures"] == 0, "numeric tolerance failure")
    if "nonfinite_mismatches" in record:
        need(record["nonfinite_mismatches"] == 0, "nonfinite mismatch")


def execution(record, torch=False):
    need(record.get("gpu_executed") is True, "GPU execution not confirmed")
    key = "all_attempted_work_synchronized" if torch else "all_submitted_work_completed"
    need(record.get(key) is True, f"{key} not confirmed")
    if not torch:
        need(record.get("kernel_execution_completed") is True, "kernel completion not confirmed")


def timing(record, flops, rounds, counted=True):
    result = dict(record)
    for clock in ("synchronized_wall", "cuda_event"):
        raw = record.get(clock + "_round_ms")
        need(isinstance(raw, list) and len(raw) == rounds, f"{clock}: wrong round count")
        need(all(isinstance(x, (int, float)) and not isinstance(x, bool)
                 and math.isfinite(x) and x > 0 for x in raw), f"{clock}: invalid timing")
        median = statistics.median(raw)
        close(record.get(clock + "_median_ms"), median, clock + " median")
        if counted:
            close(record.get(clock + "_useful_tflops"), flops / (median * 1e9), clock + " TFLOPS")
        else:
            need(record.get(clock + "_useful_tflops") is None, "quantize/dequant has no GEMM TFLOPS")
        result[clock + "_recomputed_median_ms"] = median
        result[clock + "_recomputed_useful_tflops"] = flops / (median * 1e9) if counted else None
    primary = "synchronized_wall" if record.get("primary_clock") in (
        "synchronized_steady_clock", "synchronized_perf_counter") else "cuda_event"
    close(record.get("median_ms"), result[primary + "_recomputed_median_ms"], "primary median")
    if "round_ms" in record:
        need(record["round_ms"] == record[primary + "_round_ms"], "primary round array mismatch")
    if counted:
        close(record.get("useful_tflops"), result[primary + "_recomputed_useful_tflops"], "primary TFLOPS")
    else:
        need(record.get("useful_tflops") is None, "non-GEMM useful TFLOPS must be null")
    return result


class Audit:
    def __init__(self, args):
        self.args = args
        self.issues = []
        self.warnings = []
        self.results = Bundle(args.results_archive, args.archive_sha256, 128 * MIB, "results")
        self.source = Bundle(args.source_archive, args.source_sha256, 256 * MIB)
        self.report = self.results.document("completion.json")
        self.publication, self.publication_meta = local_document(args.source_publication)
        self.manifest, self.manifest_meta = local_document(args.run_manifest)
        self.stages = {}
        self.builds = {}
        self.suites = {}

    def bind_stage_command(self, stage):
        """Bind completion's absolute paths to this run-manifest, without execution."""
        command = stage.get("argv")
        if command is None:
            need(stage.get("executed") is not True, "executed stage has no command")
            return
        need(isinstance(command, list) and command and all(isinstance(x, str) for x in command),
             "invalid recorded stage command")
        name, root = stage["name"], self.manifest["remote_dir"]
        def option(flag, expected):
            need(command.count(flag) == 1, f"{name}: missing/duplicate {flag}")
            index = command.index(flag)
            need(index + 1 < len(command) and command[index + 1] == str(expected),
                 f"{name}: wrong {flag}")
        match = re.fullmatch(r"fused_c([3-8])_(build|correctness|benchmark_(2048|4096))", name)
        variant = int(match.group(1)) if match else 2
        build = root + ("/build" if variant == 2 else f"/build-fused-c{variant}")
        if name == "cuda_build" or (match and match.group(2) == "build"):
            need(len(command) > 1 and command[1] == PREFIX + "build_cuda.py", "build entry point mismatch")
            option("--output-dir", build)
            option("--certified-variant", variant)
            option("--tiled-variant", 1)
            need("--with-benchmark" in command and "--with-certified" in command
                 and "--with-fused-benchmark" not in command and "--library-only" not in command,
                 "build stage scope differs")
        elif name in ("native_baseline", "native_tiled", "native_certified") or (match and match.group(2) == "correctness"):
            binary = {"native_baseline": "check_mxfp4", "native_tiled": "check_mxfp4_tiled"}.get(name, "check_mxfp4_certified")
            need(command == [build + "/" + binary, "--device", "0", "--correctness-only"],
                 "correctness invocation differs from manifest run")
        elif name == "native_benchmark" or "_benchmark_" in name:
            need(command[0] == build + "/bench_mxfp4", "benchmark binary outside manifest run")
            for flag, value in (("--device", 0), ("--iters", 10), ("--rounds", 5), ("--warmup", 3)):
                option(flag, value)
            if name == "native_benchmark":
                option("--output", root + "/results/native-benchmark.jsonl")
            else:
                size = int(name.rsplit("_", 1)[1])
                directory = "control-c2" if variant == 2 else f"fused-c{variant}"
                option("--output", f"{root}/results/{directory}/benchmark-{size}.jsonl")
                for flag in ("--m", "--n", "--k"):
                    option(flag, size)
                option("--mode", "random_block_scales")
                option("--seed", 20261123)
        elif name in ("torch_integration", "torch_benchmark", "smoke_cuda", "smoke_cuda_tiled"):
            module = {"torch_integration": "check_torch_cuda", "torch_benchmark": "bench_torch_cuda"}.get(name, "pretrain_smoke")
            need(command[1:3] == ["-m", "tools.mxfp4_training." + module], "framework/smoke entry point differs")
            filename = {"torch_integration": "torch-cuda-check.json", "torch_benchmark": "torch-benchmark.json"}.get(name)
            if filename:
                option("--output", root + "/results/" + filename)
                if name == "torch_benchmark":
                    option("--check-report", root + "/results/torch-cuda-check.json")
            else:
                backend = name[len("smoke_"):]
                option("--backend", backend)
                option("--device", "cuda:0")
                option("--steps", 40)
                option("--seed", 20261123)
                option("--lr", "0.005")
                option("--output", root + "/results/smoke-" + backend + ".json")

    def provenance(self):
        a, r, p, m = self.args, self.report, self.publication, self.manifest
        need(r.get("schema_version") == 1 and r.get("status") in ("passed", "failed", "blocked"),
             "completion is not a finalized schema-1 report")
        need(r.get("source_archive_sha256") == a.source_sha256, "completion/source archive mismatch")
        result_hashes = dict(self.results.hashes)
        del result_hashes["completion.json"]
        need(r.get("result_files_sha256") == result_hashes, "completion member SHA map is incomplete or mismatched")
        need(p.get("schema_version") == 1 and p.get("ready") is True, "source publication is not ready")
        need(m.get("schema_version") == 1 and m.get("run_id"), "invalid run manifest")
        need(p.get("job_name") == m.get("job") == a.expected_job, "publication/run/job mismatch")
        need(m.get("owner") == a.owner, "run owner mismatch")
        need(p.get("source_sha256") == m.get("source_archive_sha256") == a.source_sha256,
             "publication/run/source archive mismatch")
        need(p.get("remote_script_sha256") == m.get("remote_script_sha256") == a.payload_sha256,
             "publication/run/payload mismatch")
        need(self.source.hashes.get(PAYLOAD) == a.payload_sha256, "uploaded payload does not match archived payload")
        need(p.get("source_commit") == m.get("source_commit") == a.source_commit,
             "source commit label mismatch")
        need(re.fullmatch(r"[0-9a-f]{40}", a.source_commit) is not None, "invalid source commit")
        endpoint = m.get("endpoint", {})
        need(endpoint.get("job") == a.expected_job and endpoint.get("owner") == a.owner,
             "endpoint ownership/job mismatch")
        need(all(isinstance(endpoint.get(k), int) and 1 <= endpoint[k] <= 65535
                 for k in ("container_port", "port"))
             and str(endpoint.get("host_key_sha256", "")).startswith("SHA256:")
             and endpoint.get("job_created_at") and endpoint.get("node")
             and endpoint.get("user") and endpoint["user"] != "root",
             "incomplete owned non-root SSH endpoint binding")
        need(isinstance(m.get("remote_dir"), str) and m["remote_dir"].startswith("/workspace/user/"),
             "unexpected remote work directory")
        for stage in r.get("stages", []):
            name = stage.get("name")
            need(name in set(FIRST) | set(COMPARISON) | {"gpu_training_smoke"}, f"unknown stage {name}")
            need(name not in self.stages, f"duplicate stage {name}")
            need(stage.get("status") in ("passed", "failed", "blocked"), f"unfinished stage {name}")
            self.stages[name] = stage
            self.bind_stage_command(stage)
            if stage.get("executed") is True:
                for field in ("stdout", "stderr"):
                    need(path_name(stage.get(field)) in self.results.files, f"missing {name} {field}")
            if stage["status"] == "passed":
                need(stage.get("executed") is True and stage.get("returncode") == 0,
                     f"passed process stage without successful execution: {name}")
        need(set(COMPARISON).issubset(self.stages), "comparison stage coverage missing from final completion")
        source_map = r.get("source_files_sha256")
        if source_map is not None:
            need(source_map == self.source.hashes, "payload source-files map differs from source archive")
        else:
            need(r["status"] == "failed" and not any(s.get("executed") for s in self.stages.values()),
                 "source-files SHA map missing after stage execution")
        statuses = {s["status"] for s in self.stages.values()}
        if r["status"] == "passed":
            need(statuses == {"passed"} and set(self.stages) == set(FIRST) | set(COMPARISON),
                 "passed payload must contain all 36 successful stages")
        elif r["status"] == "blocked":
            need("blocked" in statuses and "failed" not in statuses, "blocked top-level status inconsistent")
        else:
            need("failed" in statuses or r.get("error"), "failed payload without failed stage or exception")
        comparison = r.get("fused_comparison", {})
        expected = {"control_certified_variant": 2, "candidate_certified_variants": list(range(3, 9)),
                    "shapes": [[s, s, s] for s in SIZES], "mode": "random_block_scales",
                    "seed": 20261123, "iters": 10, "rounds": 5, "warmup": 3,
                    "framework_library_certified_variant": 2, "stage_order": COMPARISON}
        for key, value in expected.items():
            need(comparison.get(key) == value, f"payload comparison contract mismatch: {key}")
        for key in ("services_stopped", "clock_or_power_changes", "packages_installed"):
            need(r.get(key) is False, f"unexpected payload operational flag {key}")

    def stage(self, name):
        return self.stages.get(name, {"name": name, "status": "not_recorded", "executed": False})

    def component(self, name, function):
        try:
            result = function()
            result["audit_verified"] = True
            return result
        except (AuditError, KeyError, TypeError, ValueError, UnicodeError, OverflowError) as error:
            self.issues.append({"component": name, "error": str(error)})
            return {"audit_verified": False, "complete_pass": False, "error": str(error)}

    def doc_for_stage(self, path, stage):
        if path not in self.results.files:
            need(stage["status"] != "passed", f"passed {stage['name']} is missing {path}")
            return None
        try:
            return self.results.document(path)
        except (AuditError, ValueError, UnicodeError) as error:
            need(stage["status"] != "passed", f"passed {stage['name']} has invalid {path}: {error}")
            self.warnings.append({"artifact": path, "parse_error": str(error), "stage_status": stage["status"]})
            return None

    def jsonlines(self, path, strict):
        if path not in self.results.files:
            need(not strict, f"missing JSONL {path}")
            return []
        rows = []
        for number, line in enumerate(self.results.files[path].decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json_load(line))
            except (AuditError, ValueError) as error:
                need(not strict, f"{path}:{number}: {error}")
                self.warnings.append({"artifact": path, "line": number, "parse_error": str(error)})
        return rows

    def bind_sources(self, sources, expected=None):
        need(isinstance(sources, dict) and sources, "source SHA map missing")
        if expected is not None:
            need(set(sources) == set(expected), "source manifest file set differs")
        for name, digest in sources.items():
            need(self.source.hashes.get(PREFIX + path_name(name)) == digest,
                 f"source SHA differs from archive: {name}")

    def build(self, variant):
        stage = self.stage("cuda_build" if variant == 2 else f"fused_c{variant}_build")
        directory = "build" if variant == 2 else f"fused-c{variant}/build"
        path = directory + "/compile-evidence.json"
        record = self.doc_for_stage(path, stage)
        result = {"stage": stage, "artifact": path, "manifest": record, "complete_pass": False,
                  "binary_hash_scope": "compiler manifest only; binaries are not in results archive"}
        if record is None:
            return result
        need(record.get("schema_version") == 2 and record.get("evidence_type") == "cuda_compilation",
             "invalid compile manifest schema")
        need(record.get("certified_enabled") is True and record.get("certified_variant") == variant
             and record.get("tiled_variant") == 1 and record.get("fused_packed_benchmark_enabled") is False,
             "wrong compiled variant/benchmark profile")
        expected = {name[len(PREFIX):] for name in self.source.files
                    if name.startswith(PREFIX + "csrc/") and "/" not in name[len(PREFIX + "csrc/"):]}
        need(len(expected) == 12, "frozen source should contain 12 csrc inputs")
        self.bind_sources(record.get("source_sha256"), expected)
        need(record.get("fragment_mapping_sha256") == self.source.hashes.get(MAPPING), "fragment mapping mismatch")
        jobs = record.get("jobs", [])
        need(len({j.get("name") for j in jobs}) == len(jobs), "duplicate compiler job")
        for job in jobs:
            name = job.get("name")
            need(name in JOBS, "unexpected compiler job")
            need(job.get("log") == name + ".log", "compile log name mismatch")
            need(self.results.hashes.get(directory + "/" + job["log"]) == job.get("log_sha256"),
                 "compile log hash mismatch")
            command = job.get("argv", [])
            for flag in ("--ftz=false", "-gencode=arch=compute_70,code=[sm_70,compute_70]",
                         "-DMXFP4_TILED_VARIANT=1", f"-DMXFP4_CERTIFIED_VARIANT={variant}"):
                need(flag in command, f"compiler contract flag missing: {flag}")
            if name in ("shared_library", "certified_harness", "performance_benchmark"):
                has_fused = any(str(arg).endswith("/mxfp4_gemm_fused.cu") for arg in command)
                need(has_fused == (variant >= 3), "fused compilation route mismatch")
            need(PurePosixPath(command[-1]).name == JOBS[name], "compiler output name mismatch")
        for name, artifact in record.get("artifacts", {}).items():
            need(name in JOBS.values() and valid_hash(artifact.get("sha256"))
                 and isinstance(artifact.get("bytes"), int) and artifact["bytes"] > 0,
                 "invalid binary artifact metadata")
        if stage["status"] == "passed":
            need(record.get("passed") is True and len(jobs) == 5
                 and {j["name"] for j in jobs} == set(JOBS)
                 and all(j.get("returncode") == 0 for j in jobs)
                 and set(record.get("artifacts", {})) == set(JOBS.values()), "incomplete passing build")
            result["complete_pass"] = True
        return result

    def suite(self, stage_name, variant=None):
        stage = self.stage(stage_name)
        path = stage.get("stdout")
        rows = self.jsonlines(path, stage["status"] == "passed") if path else []
        result = {"stage": stage, "artifact": path, "records": rows, "complete_pass": False}
        if stage["status"] != "passed":
            return result
        base = [r for r in rows if r.get("type") == "summary"]
        need(len(base) == 1 and base[0].get("passed") is True and base[0].get("gpu_executed") is True
             and base[0].get("actual_V100") is True and base[0].get("cases") == 46
             and base[0].get("values") == 175429 and base[0].get("mismatches") == 0,
             "base 46-case correctness summary invalid")
        cases = [r for r in rows if r.get("type") == "correctness"]
        for case in cases:
            need(case.get("gpu_executed") is True and case.get("mismatches") == 0
                 and case.get("values") == case.get("M", -1) * case.get("N", -1),
                 "correctness case invalid")
        if variant is None:
            need(len(cases) == 46 and sum(c["values"] for c in cases) == 175429, "base case records mismatch")
            result.update(confirmed_calls=46, confirmed_output_values=175429, complete_pass=True)
            return result
        summaries = [r for r in rows if r.get("type") == "certified_summary"]
        need(len(summaries) == 1, "certified summary missing/duplicated")
        cert = summaries[0]
        expected = {"passed": True, "gpu_executed": True, "certified_variant": variant,
                    "full_domain_suite_passed": True, "extra_fixture_cases": 20,
                    "extra_fixture_values": 230456, "context_reuse_calls": 3,
                    "accepted_calls": 23, "fallback_calls": 46,
                    "canonical_positive_zero_values": 2950, "mismatches": 0}
        for key, value in expected.items():
            need(cert.get(key) == value, f"certified original suite mismatch: {key}")
        before = rows[:rows.index(cert)]
        ordinary = [r for r in before if r.get("type") == "correctness"]
        reuse = [r for r in before if r.get("type") == "context_reuse"]
        need(len(ordinary) == 66 and sum(c["values"] for c in ordinary) == 405885, "original case records mismatch")
        need(len(reuse) == 3 and [c.get("accepted") for c in reuse] == [True, False, True]
             and [c.get("submitted_calls") for c in reuse] == [1, 2, 3]
             and all(c.get("passed") is True and c.get("mismatches") == 0 and c.get("values") == 15 for c in reuse),
             "reused workspace evidence mismatch")
        result.update(original_suite_calls=69, original_suite_output_values=405930,
                      confirmed_calls=69, confirmed_output_values=405930)
        if variant == 8:
            c8 = [r for r in rows if r.get("type") == "c8_fixture_summary"]
            dispatch = [r for r in rows if r.get("type") == "c8_fixture_dispatch"]
            need(len(c8) == 1, "C8-specific suite missing")
            extra_expected = {"passed": True, "gpu_executed": True, "cases": 9, "values": 180224,
                              "nonempty_c8_loader_cases": 4, "c6_compatibility_cases": 3,
                              "empty_K_cases": 1, "rejected_cases": 1, "accepted_calls": 8,
                              "fallback_calls": 1, "canonical_positive_zero_values": 37961,
                              "peak_explicit_allocation_bytes": 170312, "mismatches": 0}
            for key, value in extra_expected.items():
                need(c8[0].get(key) == value, f"C8-specific suite mismatch: {key}")
            need(Counter(x.get("expected_execution_path") for x in dispatch) ==
                 {"c8_halfrow_loader": 4, "c6_compatible_fused": 3, "empty_k": 1, "exact_tiled_rejection": 1},
                 "C8 dispatch coverage mismatch")
            need(all(x.get("passed") is True and x.get("pointer_alignment_observed") is True
                     and x.get("kernel_path_instrumented") is False for x in dispatch), "C8 dispatch metadata mismatch")
            need(len(cases) == 75 and sum(c["values"] for c in cases) == 586109, "C8 case records mismatch")
            result.update(confirmed_calls=78, confirmed_output_values=586154,
                          c8_path_scope="inferred from observed alignment and frozen dispatch; not kernel instrumentation")
        else:
            need(len(cases) == 66, "unexpected extra correctness cases")
        result["complete_pass"] = True
        return result

    def certificate(self, before, after, variant, expected_calls):
        need(isinstance(before, dict) and isinstance(after, dict), "certificate metadata missing")
        need(before.get("metadata_valid") is True and after.get("metadata_valid") is True,
             "certificate metadata not valid")
        need(before.get("submitted_calls") == 1 and after.get("submitted_calls") == expected_calls,
             "certificate call count mismatch")
        need({k: v for k, v in before.items() if k != "submitted_calls"} ==
             {k: v for k, v in after.items() if k != "submitted_calls"}, "certificate changed with unchanged inputs")
        flags = before.get("reject_flags")
        need(isinstance(flags, int) and 0 <= flags <= 15 and before.get("accepted") is (flags == 0),
             "invalid certificate acceptance/rejection flags")
        for key, bit in (("nan_scale_gate_pass", 1), ("span_gate_pass", 2),
                         ("energy_gate_pass", 4), ("lattice_gate_pass", 8)):
            need(before.get(key) is (not bool(flags & bit)), "certificate gate flag mismatch")
        ua, ub = before.get("max_energy_a"), before.get("max_energy_b")
        need(isinstance(ua, int) and isinstance(ub, int) and ua >= 0 and ub >= 0, "invalid energy bound")
        bound_pass = ua == 0 or ub == 0 or ua <= (1 << 48) // ub
        need(before.get("energy_product_bound") == 1 << 48
             and before.get("energy_gate_pass_from_reported_bounds") is bound_pass, "integer energy report mismatch")
        need(before.get("workspace_allocation_timed") is False
             and before.get("certificate_recomputed_each_call") is True
             and before.get("diagnostic_get_info_in_timed_region") is False, "certificate timing contract mismatch")
        fused = variant >= 3
        for key in ("cublas_executes_on_rejected_calls", "normalized_operands_materialized",
                    "normalization_executes_on_rejected_calls"):
            need(before.get(key) is (not fused), f"incorrect variant metadata: {key}")
        if flags:
            need(before.get("path") == "exact_tiled_fallback", "rejected certificate path mismatch")
        elif fused:
            need(before.get("path") == "fused_normalized_hmma", "accepted fused path mismatch")

    def native_benchmark(self, stage_name, path, variant, size=None):
        stage = self.stage(stage_name)
        strict = stage["status"] == "passed"
        rows = self.jsonlines(path, strict)
        result = {"stage": stage, "artifact": path, "records": rows, "complete_pass": False, "cases": []}
        stdout = stage.get("stdout")
        if path in self.results.files and stdout in self.results.files:
            # Reporter writes the same JSONL twice. Never count stdout as a run.
            if strict:
                need(self.results.files[path] == self.results.files[stdout], "benchmark output/stdout duplication differs")
            elif self.results.files[path] != self.results.files[stdout]:
                self.warnings.append({"artifact": path, "stage_status": stage["status"],
                                      "note": "partial output/stdout differ; neither is counted as a completed run"})
        if not strict:
            return result
        envs = [r for r in rows if r.get("type") == "environment"]
        summaries = [r for r in rows if r.get("type") == "summary"]
        need(len(envs) == len(summaries) == 1, "benchmark environment/summary count mismatch")
        env, summary = envs[0], summaries[0]
        need(env.get("schema_version") == 3 and env.get("certified_variant") == variant
             and env.get("actual_V100") is True and env.get("compute_capability") == "7.0"
             and env.get("certified_global_normalized_fp16_workspace") is (variant == 2),
             "benchmark environment/variant mismatch")
        need(summary.get("passed") is True and summary.get("correctness_only") is False
             and summary.get("end_to_end_training_measured") is False, "benchmark did not finish successfully")
        execution(summary)
        correctness = [r for r in rows if r.get("type") == "correctness"]
        measured = [r for r in rows if r.get("type") == "benchmark"]
        count = 1 if size else 15
        need(len(rows) == 2 + count * 2 and len(correctness) == len(measured) == count
             and summary.get("completed_cases") == count, "benchmark case/record count mismatch")
        for correct, bench in zip(correctness, measured):
            identity = ("M", "N", "logical_K", "padded_K", "input_mode", "seed", "input_fnv1a64")
            need(all(correct.get(k) == bench.get(k) for k in identity), "initial/timed input identity mismatch")
            if size:
                need(all(bench.get(k) == size for k in ("M", "N", "logical_K", "padded_K"))
                     and bench.get("input_mode") == "random_block_scales" and bench.get("seed") == 20261123,
                     "comparison shape/mode/seed mismatch")
            need(isinstance(bench.get("input_fnv1a64"), str) and bench["input_fnv1a64"], "input fingerprint absent")
            need(correct.get("passed") is True and bench.get("correctness_passed") is True,
                 "benchmark correctness not passed")
            execution(correct)
            execution(bench)
            values, samples = bench["M"] * bench["N"], correct.get("cpu_reference_values")
            exhaustive = values * bench["logical_K"] <= 2000000
            expected_samples = values if exhaustive else min(values, 256)
            need(samples == expected_samples, "CPU oracle sample count differs")
            need(correct.get("reference_scope") == ("cpu_ordered_mx32_all_outputs" if exhaustive
                 else "cpu_ordered_mx32_selected_outputs"), "CPU oracle scope differs")
            for name in ("baseline_vs_cpu", "tiled_vs_cpu", "certified_vs_cpu"):
                metric(correct.get(name), samples)
            for name in ("tiled_vs_baseline_all_outputs", "certified_vs_baseline_all_outputs"):
                metric(correct.get(name), values)
            metric(correct.get("cublas_vs_cpu"), samples, exact=False)
            metric(correct.get("cublas_vs_baseline_all_outputs"), values, exact=False)
            metric(bench.get("certified_post_timing_vs_baseline_all_outputs"), values)
            need(bench.get("rounds") == 5 and bench.get("iterations_per_round") == 10
                 and bench.get("warmup_per_job") == 3 and bench.get("backend_order") == "rotating_each_round"
                 and bench.get("same_resident_inputs_reused") is True, "native timing design differs")
            self.certificate(correct.get("certificate"), bench.get("certificate_after_timing"), variant, 54)
            flops = 2 * bench["M"] * bench["N"] * bench["logical_K"]
            close(bench.get("useful_flops"), flops, "useful FLOPs")
            timing_rows = bench.get("timings", [])
            need(len(timing_rows) == 6 and {t.get("name") for t in timing_rows} == NATIVE_BACKENDS,
                 "native backend timing set differs")
            audited = [timing(t, flops, 5, t["name"] != "cublas_dequant") for t in timing_rows]
            certified = next(t for t in audited if t["name"] == "mxfp4_certified_total")
            need(certified.get("primary_clock") == "synchronized_steady_clock", "certified primary clock differs")
            result["cases"].append({"identity": {k: bench[k] for k in identity}, "useful_flops": flops,
                                    "certificate": bench["certificate_after_timing"],
                                    "all_backend_timings": audited, "certified_timing": certified,
                                    "telemetry_before": bench.get("telemetry_before"),
                                    "telemetry_after": bench.get("telemetry_after")})
        result.update(complete_pass=True, environment=env)
        return result

    def framework(self):
        stage = self.stage("torch_integration")
        record = self.doc_for_stage("torch-cuda-check.json", stage)
        result = {"stage": stage, "artifact": "torch-cuda-check.json", "report": record,
                  "complete_pass": False, "native_certified_path_tested": False}
        if record is None:
            return result
        self.bind_sources(record.get("source_sha256"), CHECK_SOURCES)
        if "library" in record:
            need(record["library"].get("sha256") == self.main_library_sha(), "framework used a different main library")
        if stage["status"] != "passed":
            return result
        need(record.get("status") == "passed" and record.get("passed") is True
             and record.get("source_unchanged_during_check") is True, "framework report not a complete pass")
        execution(record, torch=True)
        need(record.get("gpu", {}).get("actual_SM70") is True
             and record.get("gpu", {}).get("actual_V100") is True, "framework device is not V100/SM70")
        cases = record.get("cases", [])
        need(len(cases) == record.get("completed_cases") == 119 and all(c.get("passed") is True for c in cases),
             "framework 119-case gate incomplete")
        need(len({c.get("name") for c in cases}) == 119, "duplicate framework case names")
        def visit(obj):
            if isinstance(obj, dict):
                if obj.get("comparison") in ("exact_except_nan_payload_and_signed_zero", "fp32_allclose"):
                    metric(obj, exact=obj["comparison"].startswith("exact"))
                for value in obj.values():
                    visit(value)
            elif isinstance(obj, list):
                for value in obj:
                    visit(value)
        visit(cases)
        result.update(complete_pass=True, confirmed_cases=119,
                      comparison_scope="packed/quantization exactness plus declared FP32 gradient tolerances; not all 119 are bitwise")
        return result

    def main_library_sha(self):
        record = self.builds.get(2, {}).get("manifest") or {}
        return record.get("artifacts", {}).get("libmxfp4_sm70.so", {}).get("sha256")

    def torch_benchmark(self):
        stage = self.stage("torch_benchmark")
        record = self.doc_for_stage("torch-benchmark.json", stage)
        result = {"stage": stage, "artifact": "torch-benchmark.json", "report": record, "complete_pass": False}
        if record is None:
            return result
        self.bind_sources(record.get("source_sha256"), CHECK_SOURCES | {"bench_torch_cuda.py"})
        if "library" in record:
            need(record["library"].get("sha256") == self.main_library_sha(), "Torch benchmark library mismatch")
        if stage["status"] != "passed":
            return result
        need(record.get("status") == "passed" and record.get("passed") is True
             and record.get("source_unchanged_during_benchmark") is True, "Torch benchmark did not pass")
        execution(record, torch=True)
        prerequisite = record.get("prerequisite", {})
        need(prerequisite.get("sha256") == self.results.hashes.get("torch-cuda-check.json")
             and prerequisite.get("completed_cases") == 119
             and prerequisite.get("source_and_library_hashes_match") is True,
             "Torch benchmark prerequisite does not bind the 119-case report")
        need(record.get("native_certified_capi_benchmarked") is False
             and record.get("native_certified_workspace_allocated") is False, "Torch certified scope changed")
        validation = record.get("validation", {})
        for key in ("passed_before_timing", "passed_after_timing", "all_input_and_output_values_finite",
                    "final_timed_rne_codes_and_scales_exact"):
            need(validation.get(key) is True, f"Torch benchmark validation missing {key}")
        shape = record.get("shape", {})
        need(shape == {"M": 1024, "N": 1024, "logical_K": 1024, "padded_K": 1024}, "Torch benchmark shape changed")
        need(record.get("rounds") == 3 and record.get("iterations_per_round") == 3
             and record.get("warmup_per_job") == 1, "Torch timing design changed")
        metric(validation.get("packed_cuda_vs_tiled_all_outputs"), 1024 ** 2)
        metric(validation.get("final_timed_packed_cuda_vs_tiled_all_outputs"), 1024 ** 2)
        repeats = validation.get("final_timed_vs_initial_all_outputs", {})
        need(len(repeats) == 5, "final timed Torch GEMM validation incomplete")
        for value in repeats.values():
            metric(value, 1024 ** 2)
        records = record.get("timings", [])
        expected = {"torch_fp32_matmul", "torch_fp16_compute", "torch_fp16_casts_and_matmul",
                    "mxfp4_quantize_both", "mxfp4_cuda_quantize_and_gemm", "mxfp4_cuda_tiled_quantize_and_gemm"}
        need(len(records) == 6 and {r.get("name") for r in records} == expected, "Torch timing backends differ")
        result["timings"] = [timing(r, 2 * 1024 ** 3, 3, r["name"] != "mxfp4_quantize_both") for r in records]
        result["complete_pass"] = True
        return result

    def smoke(self, backend):
        stage = self.stage("smoke_" + backend)
        path = "smoke-" + backend + ".json"
        record = self.doc_for_stage(path, stage)
        result = {"stage": stage, "artifact": path, "report": record, "complete_pass": False,
                  "scope": "synthetic causal-decoder integration smoke; no LLM convergence or training throughput claim",
                  "library_binding": "payload environment and main build; smoke has no direct library SHA field"}
        if record is None:
            return result
        expected = {name[len(PREFIX):] for name in self.source.files if name.startswith(PREFIX)
                    and "/" not in name[len(PREFIX):] and name.endswith(".py")}
        self.bind_sources(record.get("source_sha256"), expected)
        if stage["status"] != "passed":
            return result
        need(record.get("passed") is True and record.get("backend") == backend
             and record.get("device") == "cuda:0" and record.get("gpu_executed") is True
             and record.get("llm_convergence_validated") is False and record.get("performance_claim") is None,
             "smoke metadata/scope mismatch")
        arms = record.get("arms", [])
        need(len(arms) == 5 and {a.get("name") for a in arms} ==
             {"fp32", "mx_forward", "mx_forward_dgrad", "mx_forward_dgrad_wgrad", "mx_all_stochastic_grad"},
             "smoke arm coverage mismatch")
        for arm in arms:
            need(arm.get("passed") is True and arm.get("steps") == 40
                 and len(arm.get("training_losses", [])) == 40
                 and all(math.isfinite(x) for x in arm["training_losses"])
                 and math.isfinite(arm["validation_loss_before"])
                 and math.isfinite(arm["validation_loss_after"])
                 and arm["validation_loss_after"] < arm["validation_loss_before"]
                 and arm.get("changed_parameter_tensors", 0) > 0
                 and all(arm.get(k) is True for k in ("parameters_finite", "master_parameters_fp32", "optimizer_states_fp32")),
                 "smoke training/finite-state gate failed")
        result["complete_pass"] = True
        return result

    def run(self):
        self.provenance()
        for variant in VARIANTS:
            self.builds[variant] = self.component(f"build_c{variant}", lambda v=variant: self.build(v))
        for label in ("native_baseline", "native_tiled"):
            self.suites[label] = self.component(label, lambda x=label: self.suite(x))
        for variant in VARIANTS:
            name = "native_certified" if variant == 2 else f"fused_c{variant}_correctness"
            self.suites[name] = self.component(name, lambda n=name, v=variant: self.suite(n, v))
        default = self.component("native_benchmark", lambda: self.native_benchmark(
            "native_benchmark", "native-benchmark.jsonl", 2))
        comparisons, fingerprints = [], {}
        for variant in VARIANTS:
            suite_name = "native_certified" if variant == 2 else f"fused_c{variant}_correctness"
            for size in SIZES:
                directory = "control-c2" if variant == 2 else f"fused-c{variant}"
                stage_name = f"control_c2_benchmark_{size}" if variant == 2 else f"fused_c{variant}_benchmark_{size}"
                result = self.component(stage_name, lambda n=stage_name, d=directory, s=size, v=variant:
                                        self.native_benchmark(n, f"{d}/benchmark-{s}.jsonl", v, s))
                eligible = all(x.get("audit_verified") and x.get("complete_pass") for x in
                               (result, self.builds[variant], self.suites[suite_name], self.builds[2],
                                self.suites["native_baseline"], self.suites["native_tiled"],
                                self.suites["native_certified"]))
                row = {"variant": variant, "M": size, "N": size, "logical_K": size,
                       "stage_status": self.stage(stage_name)["status"], "measurement_eligible": eligible,
                       "wall_median_ms": None, "wall_useful_tflops": None,
                       "event_median_ms": None, "event_useful_tflops": None,
                       "wall_round_ms": [], "event_round_ms": [],
                       "speed_ratio_vs_c2_wall": None, "throughput_change_vs_c2_percent": None,
                       "evidence": result}
                if eligible:
                    case = result["cases"][0]
                    fp = case["identity"]["input_fnv1a64"]
                    if size in fingerprints and fingerprints[size] != fp:
                        self.issues.append({"component": stage_name, "error": "same-shape input fingerprints differ across variants"})
                        row["measurement_eligible"] = False
                    else:
                        fingerprints[size] = fp
                    if row["measurement_eligible"]:
                        t = case["certified_timing"]
                        row.update(input_fnv1a64=fp, certificate=case["certificate"],
                                   wall_median_ms=t["synchronized_wall_recomputed_median_ms"],
                                   wall_useful_tflops=t["synchronized_wall_recomputed_useful_tflops"],
                                   event_median_ms=t["cuda_event_recomputed_median_ms"],
                                   event_useful_tflops=t["cuda_event_recomputed_useful_tflops"],
                                   wall_round_ms=t["synchronized_wall_round_ms"], event_round_ms=t["cuda_event_round_ms"])
                comparisons.append(row)
        controls = {r["M"]: r for r in comparisons if r["variant"] == 2 and r["measurement_eligible"]}
        for row in comparisons:
            control = controls.get(row["M"])
            if row["measurement_eligible"] and control:
                ratio = control["wall_median_ms"] / row["wall_median_ms"]
                row.update(speed_ratio_vs_c2_wall=ratio, throughput_change_vs_c2_percent=(ratio - 1) * 100)
        framework = self.component("torch_integration", self.framework)
        torch_bench = self.component("torch_benchmark", self.torch_benchmark)
        smokes = {b: self.component("smoke_" + b, lambda b=b: self.smoke(b)) for b in ("cuda", "cuda_tiled")}
        stage_counts = dict(Counter(s["status"] for s in self.stages.values()))
        return {"schema_version": 1, "audit_status": "verified" if not self.issues else "inconsistent",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "auditor_sha256": sha(Path(__file__).read_bytes()), "payload_status": self.report["status"],
                "issues": self.issues, "warnings": self.warnings,
                "provenance": {"results_archive": self.results.metadata(), "source_archive": self.source.metadata(),
                    "source_publication": {"file": self.publication_meta, "record": self.publication},
                    "run_manifest": {"file": self.manifest_meta, "record": self.manifest},
                    "result_member_hashes_verified": True,
                    "source_member_hashes_reported_and_verified": "source_files_sha256" in self.report,
                    "source_commit_binding_scope": "expected publication/manifest label plus verified source tar bytes; no git object checkout performed",
                    "bootstrap_scope": "payload packages_installed=false does not describe SSH bootstrap setup"},
                "completion": self.report, "stage_counts": stage_counts,
                "failed_stage_log_tails": [
                    {"stage": s["name"], "artifact": s[field],
                     "sha256": self.results.hashes[s[field]], "tail_max_bytes": 8192,
                     "text": self.results.files[s[field]][-8192:].decode("utf-8", errors="replace")}
                    for s in self.stages.values() if s["status"] == "failed"
                    for field in ("stdout", "stderr") if s.get(field) in self.results.files],
                "unrecorded_normal_stages": [x for x in FIRST if x not in self.stages],
                "builds": {str(k): v for k, v in self.builds.items()}, "native_correctness": self.suites,
                "native_default_benchmark": default, "native_comparison_rows": comparisons,
                "eligible_comparison_rows": sum(r["measurement_eligible"] for r in comparisons),
                "framework_119": framework, "framework_benchmark": torch_bench, "smokes": smokes,
                "gpu_environment_csv": self.results.files.get("gpu_environment.stdout", b"").decode("utf-8"),
                "interpretation": [
                    "14 comparison slots are C2 through C8 at 2048^3 and 4096^3; unavailable values remain null.",
                    "Every comparison uses synchronized wall time for the entire certified pipeline; event times remain separate.",
                    "Five raw rounds are repeated measurements in one process, not five independent runs; each round averages 10 invocations.",
                    "Variants run in fixed C2-to-C8 order, not ABBA; clock/load changes can confound cross-variant ratios.",
                    "Large native comparison shapes use 256 independent CPU-oracle outputs; small default cases are exhaustive. All also check baseline equality and final timed certified output equality.",
                    "A rejected certificate measures the exact tiled fallback, not the fused accepted path.",
                    "The default 15-case benchmark has different seeds; it is never pooled with explicit comparison rows or prior 1536 results.",
                    "Framework119 and both synthetic smokes exercise cuda/cuda_tiled, not the certified C2-C8 C API.",
                    "Torch FP16 matmul returns FP16; native cuBLAS returns FP32. These are different scopes.",
                    "Native telemetry consists of snapshots, not a continuous NVML trace. Preserve actual V100 model and node identity.",
                    "Build binary hashes are manifest claims; compiled binaries are not members of the result archive."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results_archive", type=Path)
    p.add_argument("--archive-sha256", required=True)
    p.add_argument("--source-archive", type=Path, required=True)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--source-publication", type=Path, required=True)
    p.add_argument("--run-manifest", type=Path, required=True)
    p.add_argument("--payload-sha256", required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--expected-job", required=True)
    p.add_argument("--owner", default="jincy26")
    p.add_argument("--output", type=Path, required=True, help="new JSON path; existing files are never overwritten")
    a = p.parse_args()
    if a.output.exists():
        p.error("--output already exists; use a fresh summary path")
    try:
        report = Audit(a).run()
    except (AuditError, OSError, ValueError, TypeError, KeyError, tarfile.TarError) as error:
        print(json.dumps({"audit_status": "rejected", "error_type": type(error).__name__,
                          "error": str(error), "performance_claim": None}), file=sys.stderr)
        return 1
    # Exclusive creation prevents accidental edits to input or earlier summaries.
    with a.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"audit_status": report["audit_status"], "payload_status": report["payload_status"],
                      "eligible_comparison_rows": report["eligible_comparison_rows"],
                      "stage_counts": report["stage_counts"], "issues": report["issues"],
                      "summary": str(a.output.resolve()), "summary_sha256": sha(a.output.read_bytes())}))
    if report["issues"]:
        return 1
    return 0 if report["payload_status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
