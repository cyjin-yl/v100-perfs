#!/usr/bin/env python3
"""Independently verify and summarize the archived C6/C8 V100 ABBA experiment.

Reads local evidence only. Does not invoke CUDA, a remote host, a compiler, or
an experiment. All round timings remain in the output. Pooling is descriptive:
rounds within a process are not treated as independent statistical samples.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import csv
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


VARIANTS = (6, 8)
BACKENDS = ("mxfp4_baseline", "mxfp4_tiled", "mxfp4_certified_total")
MAPPING = "evidence/benchmarks/sm70-packing/study/simulation/round3/cuda/fragment_mapping.hpp"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    with path.open("rb") as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
        return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(bool(rows), f"empty JSONL: {path}")
    return rows


def one(rows: list[dict], kind: str) -> dict:
    found = [row for row in rows if row.get("type") == kind]
    require(len(found) == 1, f"expected one {kind}, found {len(found)}")
    return found[0]


def close(actual: float, expected: float, label: str) -> None:
    require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=2e-11),
            f"{label}: {actual} != {expected}")


def exact(metrics: dict, values: int, label: str) -> None:
    require(metrics["values"] == values, f"{label}: wrong value count")
    for key in ("exact_mismatches", "tolerance_failures", "nonfinite_mismatches", "max_abs", "max_relative", "nrmse"):
        require(metrics[key] == 0, f"{label}: nonzero {key}")


def certificate(info: dict, calls: int) -> None:
    require(info["accepted"] is True and info["metadata_valid"] is True, "certificate not accepted/valid")
    require(info["reject_flags"] == 0 and info["submitted_calls"] == calls, "certificate flags/call count")
    require(info["workspace_bytes"] == 52296, "unexpected benchmark workspace")
    require(info["path"] == "fused_normalized_hmma", "wrong certified path")
    require(info["max_span_a"] <= 12 and info["max_span_b"] <= 12, "span bound")
    require(info["max_energy_a"] * info["max_energy_b"] <= 1 << 48, "energy bound")
    require(-126 <= info["lattice_exponent_min"] <= info["lattice_exponent_max"] <= 103, "lattice bound")
    for key in ("diagnostic_get_info_in_timed_region", "host_gate_synchronization_per_call",
                "cublas_executes_on_rejected_calls", "normalized_operands_materialized", "workspace_allocation_timed"):
        require(info[key] is False, f"unexpected timed path property: {key}")
    require(info["certificate_recomputed_each_call"] is True, "certificate not recomputed")


def timing_summary(wall: list[float], event: list[float], flops: int) -> dict:
    require(all(math.isfinite(v) and v > 0 for v in wall + event), "invalid round time")
    require(len(wall) == len(event), "clock round counts differ")
    wm, em = statistics.median(wall), statistics.median(event)
    return {
        "round_count": len(wall), "synchronized_wall_round_ms": wall, "cuda_event_round_ms": event,
        "synchronized_wall_median_ms": wm, "cuda_event_median_ms": em,
        "synchronized_wall_min_ms": min(wall), "synchronized_wall_max_ms": max(wall),
        "cuda_event_min_ms": min(event), "cuda_event_max_ms": max(event),
        "synchronized_wall_useful_tflops": flops / (wm * 1e9),
        "cuda_event_useful_tflops": flops / (em * 1e9),
    }


def verify_collection(path: Path, raw_hashes: dict[str, str]) -> dict:
    transport = path.read_bytes()
    is_base64 = path.suffix == ".b64"
    data = base64.b64decode(b"".join(transport.split()), validate=True) if is_base64 else transport
    members = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as handle:
        for item in handle.getmembers():
            if item.isdir():
                continue
            require(item.isfile(), "collection archive contains a nonregular member")
            member = PurePosixPath(item.name)
            require(not member.is_absolute() and ".." not in member.parts, "unexpected collection member path")
            name = str(member)
            require(name not in members, "duplicate collection member")
            digest = hashlib.sha256(handle.extractfile(item).read()).hexdigest()
            require(raw_hashes.get(name) == digest, f"collection member differs from audited local evidence: {name}")
            members[name] = digest
    require(bool(members), "empty collection archive")
    return {"archive_sha256": hashlib.sha256(data).hexdigest(), "archive_bytes": len(data),
            "sha256_recomputed_locally": True, "transport_file": str(path.resolve()),
            "transport_encoding": "base64" if is_base64 else "gzip_tar", "transport_sha256": hashlib.sha256(transport).hexdigest(),
            "regular_member_count": len(members), "all_members_match_audited_local_files": True,
            "member_sha256": members, "additional_audited_files_not_in_collection": sorted(set(raw_hashes) - set(members))}


def verify_sources(directory: Path, repo: Path, driver: dict) -> dict:
    archive = directory / "source.tar.gz"
    archive_sha = sha(archive)
    state = read_json(directory / "source-state.json")
    require(state["sha256"] == archive_sha == driver["source_archive_sha256"], "source archive SHA mismatch")
    require(state["bytes"] == archive.stat().st_size, "source archive size mismatch")
    members = {}
    with tarfile.open(archive, "r:gz") as handle:
        for item in handle.getmembers():
            if not item.isfile():
                continue
            path = PurePosixPath(item.name)
            require(not path.is_absolute() and ".." not in path.parts, "unexpected archive path")
            name = str(path)
            require(name not in members, f"duplicate archive member {name}")
            members[name] = hashlib.sha256(handle.extractfile(item).read()).hexdigest()
    require(members == state["source_files"], "source-state file manifest differs from archive contents")
    core = {key: value for key, value in members.items()
            if key.startswith("tools/mxfp4_training/csrc/")
            or key in ("tools/mxfp4_training/build_cuda.py", MAPPING)}
    require(len([key for key in core if "/csrc/" in key]) == 12, "expected exactly 12 csrc inputs")
    for name, digest in core.items():
        require(sha(repo / name) == digest, f"current core source differs from compiled snapshot: {name}")
    changed_other = [name for name, digest in members.items() if name not in core
                     and (not (repo / name).is_file() or sha(repo / name) != digest)]
    require(sha(directory / "run_iteration.py") == driver["driver_sha256"], "driver source SHA mismatch")
    builds = {}
    for variant in VARIANTS:
        build_dir = directory / f"build-c{variant}"
        manifest = read_json(build_dir / "compile-evidence.json")
        require(manifest["passed"] is True and manifest["certified_variant"] == variant, "build not successful/matched")
        require(manifest["tiled_variant"] == 1 and manifest["certified_enabled"] is True, "build backend selection")
        require(len(manifest["source_sha256"]) == 12, "compile manifest must bind 12 csrc files")
        for name, digest in manifest["source_sha256"].items():
            require(members.get("tools/mxfp4_training/" + name) == digest, f"compiled source mismatch: {name}")
        require(manifest["fragment_mapping_sha256"] == members[MAPPING], "fragment mapping SHA mismatch")
        logs = []
        for job in manifest["jobs"]:
            require(job["returncode"] == 0, f"failed compilation: {job['name']}")
            require(f"-DMXFP4_CERTIFIED_VARIANT={variant}" in job["argv"], "compiler variant flag")
            require("--ftz=false" in job["argv"], "FTZ contract flag missing")
            require("-gencode=arch=compute_70,code=[sm_70,compute_70]" in job["argv"], "SM70 build flag missing")
            digest = sha(build_dir / job["log"])
            require(digest == job["log_sha256"], f"compiler log hash mismatch: {job['log']}")
            logs.append({"name": job["name"], "log": job["log"], "sha256": digest, "returncode": job["returncode"]})
        require(len(logs) == 5, "unexpected compile job count")
        artifacts = {}
        for name, item in manifest["artifacts"].items():
            path = build_dir / name
            available = path.is_file()
            if available:
                require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], f"artifact mismatch: {name}")
            artifacts[name] = {**item, "available_locally": available, "sha256_recomputed_locally": available}
        builds[f"c{variant}"] = {"manifest_sha256": sha(build_dir / "compile-evidence.json"),
            "nvcc_version": manifest["nvcc_version"], "source_sha256": manifest["source_sha256"],
            "logs_verified": logs, "recorded_artifacts": artifacts}
    require(builds["c6"]["source_sha256"] == builds["c8"]["source_sha256"], "C6/C8 source snapshots differ")
    return {"source_archive_sha256": archive_sha, "source_archive_bytes": archive.stat().st_size,
            "source_parent_commit": state["source_parent_commit"], "archive_file_count": len(members),
            "all_archive_members_match_source_state": True, "compiled_core_matches_current_worktree": True,
            "core_source_sha256": core, "other_archived_files_changed_since_freeze": changed_other,
            "other_file_note": "Documentation and queue payload may change after the build; their archived hashes remain frozen.",
            "driver_sha256": driver["driver_sha256"], "builds": builds}


def verify_correctness(directory: Path, variant: int) -> dict:
    path = directory / f"correctness-c{variant}.jsonl"
    rows = read_jsonl(path)
    env, base, original = (one(rows, kind) for kind in ("environment", "summary", "certified_summary"))
    require(env["actual_V100"] is True and env["reserve_mib"] == 48, "correctness device/reserve")
    require(env["memory_profile"] == "small_memory_correctness" and env["max_explicit_allocation_bytes"] == 8 << 20,
            "correctness profile/cap")
    require(base["passed"] is True and base["cases"] == 46 and base["values"] == 175429, "base suite incomplete")
    require(original["passed"] is True and original["certified_variant"] == variant, "certified suite failed")
    require(original["extra_fixture_cases"] == 20 and original["extra_fixture_values"] == 230456, "certificate fixtures incomplete")
    require(original["accepted_calls"] == 23 and original["fallback_calls"] == 46, "original dispatch counts")
    require(original["canonical_positive_zero_values"] == 2950, "original canonical zero count")
    original_cases = [row for row in rows if row.get("type") == "correctness" and not row["case"].startswith("c8_")]
    reused = [row for row in rows if row.get("type") == "context_reuse"]
    require(len(original_cases) == 66 and len(reused) == 3, "original suite raw row count")
    values = sum(row["values"] for row in original_cases + reused)
    require(values == 405930, "original suite raw value count")
    for row in rows:
        if "mismatches" in row:
            require(row["mismatches"] == 0, "correctness mismatch")
        if "passed" in row:
            require(row["passed"] is True, "failed correctness record")
    result = {"file": path.name, "sha256": sha(path), "environment": env,
              "original_suite": {"calls": 69, "values": values, "accepted_calls": 23,
                  "fallback_calls": 46, "canonical_positive_zero_values": 2950, "mismatches": 0},
              "all_explicit_case_allocation_bytes_max": max(row["allocation_bytes"] for row in original_cases)}
    if variant == 8:
        extra = one(rows, "c8_fixture_summary")
        fixtures = [row for row in rows if row.get("type") == "correctness" and row["case"].startswith("c8_")]
        routes = [row for row in rows if row.get("type") == "c8_fixture_dispatch"]
        require(extra["cases"] == len(fixtures) == len(routes) == 9, "C8 fixture count")
        require(extra["values"] == sum(row["values"] for row in fixtures) == 180224, "C8 values")
        require(extra["accepted_calls"] == 8 and extra["fallback_calls"] == 1, "C8 accepted/rejected counts")
        require(extra["canonical_positive_zero_values"] == 37961, "C8 canonical zero count")
        require(extra["peak_explicit_allocation_bytes"] == max(row["allocation_bytes"] for row in fixtures) == 170312,
                "C8 explicit allocation peak")
        expected_routes = {"c8_halfrow_loader": 4, "c6_compatible_fused": 3, "empty_k": 1, "exact_tiled_rejection": 1}
        require(dict(Counter(row["expected_execution_path"] for row in routes)) == expected_routes, "C8 route classification")
        require(all(row["pointer_alignment_observed"] and not row["kernel_path_instrumented"] for row in routes), "C8 path evidence scope")
        result.update({"c8_fixture_summary": extra, "c8_fixture_cases": fixtures, "c8_fixture_dispatch": routes,
                       "total_calls_in_process": 78, "total_values_in_process": 586154})
    else:
        require(not any(row.get("type") == "c8_fixture_summary" for row in rows), "C6 unexpectedly ran C8-only fixtures")
        result.update({"total_calls_in_process": 69, "total_values_in_process": values})
    return result


def verify_run(directory: Path, job: dict, index: int, warmup: int) -> dict:
    path = directory / job["stdout"]
    rows = read_jsonl(path)
    require([row.get("type") for row in rows] == ["environment", "correctness", "benchmark", "summary"], "unexpected benchmark record sequence")
    env, correctness, measured, summary = rows
    require(summary["passed"] is True and summary["gpu_executed"] is True and summary["completed_cases"] == 1,
            "benchmark summary did not pass")
    require(not summary["c2_or_cublas_comparison_performed"] and not summary["end_to_end_training_measured"], "benchmark scope")
    require(env["actual_V100"] is True and env["certified_variant"] == job["variant"], "benchmark GPU/variant")
    require(env["name"] == "Tesla V100-PCIE-32GB" and env["sm_count"] == 80, "benchmark device model")
    require(not env["ordinary_cublas_handle_created"] and not env["global_fp16_operands_allocated"], "unexpected allocations")
    identity_keys = ("M", "N", "logical_K", "padded_K", "input_mode", "seed", "input_fnv1a64", "input_distribution", "real_training_tensor")
    identity = {key: measured[key] for key in identity_keys}
    require(identity == {key: correctness[key] for key in identity_keys}, "initial/timed input identity mismatch")
    require([identity[key] for key in ("M", "N", "logical_K", "padded_K")] == [1536] * 4, "benchmark shape")
    require(identity["input_mode"] == "random_block_scales" and identity["real_training_tensor"] is False, "benchmark input scope")
    require(identity["seed"] == 20261123 and identity["input_fnv1a64"] == "8729bc8ef4895a63", "benchmark frozen input")
    expected_args = {"--m": "1536", "--n": "1536", "--k": "1536", "--mode": "random_block_scales",
                     "--seed": "20261123", "--iters": "10", "--rounds": "5", "--warmup": str(warmup),
                     "--max-mib": "12", "--reserve-mib": "48"}
    argv = job["argv"]
    require(argv[0].endswith(f"/build-c{job['variant']}/bench_mxfp4_fused"), "benchmark executable path")
    require(len(argv) == 1 + 2 * len(expected_args), "unexpected benchmark arguments")
    require(dict(zip(argv[1::2], argv[2::2])) == expected_args, "benchmark argv differs from frozen workload")
    values = measured["M"] * measured["N"]
    require(correctness["cpu_reference_values"] == 256, "CPU oracle sample count")
    for key in ("baseline_vs_cpu", "tiled_vs_cpu", "certified_vs_cpu"):
        exact(correctness[key], correctness["cpu_reference_values"], key)
    for key in ("tiled_vs_baseline_all_outputs", "certified_vs_baseline_all_outputs"):
        exact(correctness[key], values, key)
    require(correctness["passed"] is True and correctness["baseline_nonfinite_values_all_outputs"] == 0, "initial output check")
    final_keys = ("baseline_final_timed_vs_initial_all_outputs", "tiled_final_timed_vs_baseline_all_outputs",
                  "certified_final_timed_vs_baseline_all_outputs")
    for key in final_keys:
        exact(measured[key], values, key)
    require(measured["correctness_passed"] and measured["all_final_timed_outputs_checked"], "final timed checks absent")
    require(measured["iterations_per_round"] == 10 and measured["rounds"] == 5 and measured["warmup_per_job"] == warmup,
            "benchmark iteration settings")
    require(measured["backend_order"] == "rotating_each_round", "backend rotation")
    require(measured["primary_clock"] == "synchronized_steady_clock", "timing clock")
    flops = 2 * measured["M"] * measured["N"] * measured["logical_K"]
    require(measured["useful_flops"] == flops == 7247757312, "FLOP numerator")
    require(measured["allocation_bytes"] == 11996232 and measured["allocation_breakdown"]["certified_workspace"] == 52296,
            "benchmark allocation budget")
    require(measured["reserve_mib"] == 48 and measured["max_allocation_mib"] == 12, "benchmark memory guards")
    certificate(correctness["certificate"], 1)
    certificate(measured["certificate_after_timing"], 1 + warmup + 5 * 10)
    require(measured["memory_before_allocation"]["free_bytes"] - measured["allocation_bytes"] >= 48 << 20,
            "preallocation reserve")
    for key in ("memory_after_warmup", "memory_after_timing"):
        require(measured[key]["free_bytes"] >= 48 << 20, "resident reserve")
    timings = {}
    require([timing["name"] for timing in measured["timings"]] == list(BACKENDS), "backend list")
    for timing in measured["timings"]:
        wall, event = timing["synchronized_wall_round_ms"], timing["cuda_event_round_ms"]
        require(len(wall) == len(event) == 5 and wall == timing["round_ms"], "round vectors")
        derived = timing_summary(wall, event, flops)
        for key in ("synchronized_wall_median_ms", "cuda_event_median_ms", "synchronized_wall_useful_tflops", "cuda_event_useful_tflops"):
            close(timing[key], derived[key], key)
        close(timing["median_ms"], derived["synchronized_wall_median_ms"], "primary median")
        close(timing["useful_tflops"], derived["synchronized_wall_useful_tflops"], "primary throughput")
        close(timing["min_ms"], min(wall), "wall minimum")
        close(timing["max_ms"], max(wall), "wall maximum")
        timings[timing["name"]] = {"scope": timing["scope"], **derived}
    return {"run_index": index, "variant": job["variant"], "file": path.name, "sha256": sha(path),
            "process_returncode": job["returncode"], "identity": identity, "environment": env,
            "cpu_oracle_samples_per_backend": correctness["cpu_reference_values"],
            "initial_full_output_comparisons": {key: correctness[key] for key in ("tiled_vs_baseline_all_outputs", "certified_vs_baseline_all_outputs")},
            "final_actual_timed_output_comparisons": {key: measured[key] for key in final_keys},
            "certificate_before_timing": correctness["certificate"], "certificate_after_timing": measured["certificate_after_timing"],
            "allocation_breakdown": measured["allocation_breakdown"], "memory_before_allocation": measured["memory_before_allocation"],
            "memory_after_warmup": measured["memory_after_warmup"], "memory_after_timing": measured["memory_after_timing"],
            "telemetry_before": measured["telemetry_before"], "telemetry_after": measured["telemetry_after"],
            "excluded_from_all_times": measured["excluded_from_all_times"], "timings": timings, "all_checks_passed": True}


def telemetry(directory: Path) -> dict:
    path = directory / "background-gpu-samples.csv"
    grouped = defaultdict(list)
    with path.open(newline="") as handle:
        for original in csv.DictReader(handle, skipinitialspace=True):
            row = {key.strip(): value.strip() for key, value in original.items()}
            values = {"timestamp_as_recorded": row["timestamp"], "gpu_index": int(row["index"])}
            for field, key in (("gpu_utilization_percent", "utilization.gpu [%]"),
                               ("memory_utilization_percent", "utilization.memory [%]"),
                               ("free_mib", "memory.free [MiB]"), ("sm_clock_mhz", "clocks.current.sm [MHz]"),
                               ("power_w", "power.draw [W]")):
                values[field] = float(row[key].split()[0])
            grouped[values["gpu_index"]].append(values)
    require(bool(grouped), "missing NVML samples")
    result = {}
    for gpu, rows in grouped.items():
        metrics = {}
        for key in ("gpu_utilization_percent", "memory_utilization_percent", "free_mib", "sm_clock_mhz", "power_w"):
            values = [row[key] for row in rows]
            metrics[key] = {"min": min(values), "max": max(values), "median": statistics.median(values), "mean": statistics.mean(values)}
        result[str(gpu)] = {"samples": rows, "sample_count": len(rows), "metrics": metrics}
    return {"file": path.name, "sha256": sha(path), "gpus": result,
            "scope": "Device-wide samples during the benchmark window include the benchmark itself and any other work.",
            "timestamp_note": "CSV timestamps have no timezone suffix and are retained verbatim; no per-round UTC attribution is inferred.",
            "clock_locked": False, "exclusive_device": False}


def compiler_resources(directory: Path) -> dict:
    resources = {}
    for variant in VARIANTS:
        path = directory / f"build-c{variant}" / "shared_library.log"
        source = path.read_text()
        blocks = re.split(r"ptxas info\s*:\s*Compiling entry function '", source)[1:]
        selected = []
        for block in blocks:
            function = block.split("'", 1)[0]
            if not ("fused_certified_prefetch32_halfrow" in function or "fused_certified_prefetch32IiLb0E" in function):
                continue
            count = re.search(r"Used (\d+) registers", block)
            spills = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", block)
            shared = re.search(r"(\d+) bytes smem", block)
            require(count is not None and spills is not None and shared is not None, "missing ptxas resources")
            selected.append({"function": function, "registers_per_thread": int(count[1]),
                             "stack_bytes": int(spills[1]), "spill_store_bytes": int(spills[2]),
                             "spill_load_bytes": int(spills[3]), "shared_bytes_per_cta": int(shared[1])})
        require(len(selected) == 2, "expected C6 full-tile int32 and C8 halfrow resource records")
        resources[f"c{variant}_build"] = {"file": str(path.relative_to(directory)), "sha256": sha(path), "kernels": selected}
    provenance = read_json(directory / "sass-provenance.json")
    require(provenance["source_archive_sha256"] == sha(directory / "source.tar.gz"), "SASS archive binding")
    manifest = read_json(directory / "build-c8/compile-evidence.json")
    require(provenance["binary_sha256"] == manifest["artifacts"]["bench_mxfp4_fused"]["sha256"], "SASS recorded binary binding")
    for section in provenance["sections"]:
        path = directory / section["name"]
        require(path.stat().st_size == section["bytes"] and sha(path) == section["sha256"], "SASS section SHA/size mismatch")
        require(section["function"] in path.read_text(), "SASS function mismatch")
    return {"ptxas": resources, "sass_provenance": provenance,
            "sass_section_hashes_verified": True,
            "note": "SASS binary SHA is cross-checked with compile metadata; absent archived binaries are not claimed to have been rehashed locally."}


def summarize(directory: Path, repo: Path, run_subdir: str | None = None) -> dict:
    initial_driver = read_json(directory / "driver-state.json")
    require(initial_driver["completed"] is True and not initial_driver.get("error"), "initial driver did not complete")
    require(len(initial_driver["jobs"]) == 8 and all(job["returncode"] == 0 for job in initial_driver["jobs"]),
            "initial driver jobs failed/incomplete")
    initial_jobs = [job for job in initial_driver["jobs"] if job["stage"] == "benchmark_abba"]
    require([job["variant"] for job in initial_jobs] == [6, 8, 8, 6], "initial ABBA order mismatch")
    for variant in VARIANTS:
        for stage in ("build", "correctness"):
            require(sum(job["stage"] == stage and job["variant"] == variant for job in initial_driver["jobs"]) == 1,
                    "initial driver stage coverage")
    provenance = verify_sources(directory, repo, initial_driver)
    correctness = {f"c{variant}": verify_correctness(directory, variant) for variant in VARIANTS}
    if run_subdir is None:
        run_directory, driver, jobs, warmup = directory, initial_driver, initial_jobs, 3
        reused_build = None
    else:
        require(run_subdir == "warm100", "unsupported independent run directory")
        run_directory = directory / run_subdir
        driver = read_json(run_directory / "driver-state.json")
        require(driver["completed"] is True and not driver.get("error"), "warm100 driver did not complete")
        require(driver["warmup_per_job"] == 100 and driver["benchmark_order"] == [6, 8, 8, 6], "warm100 experiment settings")
        require(driver["clock_or_power_changes"] is False, "warm100 clock/power setting changed")
        require(driver["source_archive_sha256"] == provenance["source_archive_sha256"], "warm100 source archive mismatch")
        jobs, warmup = driver["jobs"], 100
        require(len(jobs) == 4 and all(job["returncode"] == 0 for job in jobs), "warm100 jobs failed/incomplete")
        require([job["variant"] for job in jobs] == [6, 8, 8, 6], "warm100 ABBA order mismatch")
        recorded_binaries = {}
        for variant in VARIANTS:
            digest = provenance["builds"][f"c{variant}"]["recorded_artifacts"]["bench_mxfp4_fused"]["sha256"]
            require(driver["binary_sha256"][str(variant)] == digest, "warm100 recorded binary differs from original build")
            recorded_binaries[str(variant)] = digest
        reused_build = {"source_directory": str(directory), "original_driver_sha256": sha(directory / "driver-state.json"),
                        "warmup_driver_file": "run_warm100.py", "warmup_driver_sha256": sha(run_directory / "run_warm100.py"),
                        "recorded_binaries_match_original_compile_manifests": True, "binary_sha256": recorded_binaries,
                        "original_correctness_suites_reused": True, "original_correctness_suites_rerun_in_this_window": False,
                        "benchmarks_still_check_initial_and_final_outputs": True}
    runs = [verify_run(run_directory, job, index, warmup) for index, job in enumerate(jobs)]
    require(all(run["identity"] == runs[0]["identity"] for run in runs), "input fingerprint differs across ABBA")
    pooled = {}
    flops = 7247757312
    for variant in VARIANTS:
        selected = [run for run in runs if run["variant"] == variant]
        pooled[f"c{variant}"] = {"separate_executable_runs": len(selected), "backends": {}}
        for backend in BACKENDS:
            wall = [value for run in selected for value in run["timings"][backend]["synchronized_wall_round_ms"]]
            event = [value for run in selected for value in run["timings"][backend]["cuda_event_round_ms"]]
            pooled[f"c{variant}"]["backends"][backend] = timing_summary(wall, event, flops)
    comparison = {}
    for clock in ("synchronized_wall", "cuda_event"):
        c6 = pooled["c6"]["backends"]["mxfp4_certified_total"][clock + "_median_ms"]
        c8 = pooled["c8"]["backends"]["mxfp4_certified_total"][clock + "_median_ms"]
        comparison[clock] = {"c6_pooled_median_ms": c6, "c8_pooled_median_ms": c8,
                            "c8_throughput_over_c6": c6 / c8,
                            "c8_throughput_change_percent": (c6 / c8 - 1) * 100,
                            "c8_latency_change_percent": (c8 / c6 - 1) * 100}
    pairs = []
    for a, b in ((0, 1), (3, 2)):
        c6 = runs[a]["timings"]["mxfp4_certified_total"]["synchronized_wall_median_ms"]
        c8 = runs[b]["timings"]["mxfp4_certified_total"]["synchronized_wall_median_ms"]
        pairs.append({"c6_run_index": a, "c8_run_index": b, "c8_wall_throughput_over_c6": c6 / c8})
    # A later independent run lives underneath v6. Do not silently add its files
    # to the initial experiment's provenance or pool its timings with that run.
    raw_names = {"driver-state.json", "driver-launch.json", "driver.stdout", "background-gpu-samples.csv"}
    for job in driver["jobs"]:
        raw_names.update((job["stdout"], job["stdout"] + ".stderr"))
    if run_subdir:
        raw_names.add("run_warm100.py")
    else:
        raw_names.update(("source.tar.gz", "source-state.json", "run_iteration.py", "sass-provenance.json"))
        raw_names.update(section["name"] for section in read_json(directory / "sass-provenance.json")["sections"])
    raw_paths = [run_directory / name for name in sorted(raw_names) if (run_directory / name).is_file()]
    if run_subdir is None:
        raw_paths += [path for variant in VARIANTS for path in sorted((directory / f"build-c{variant}").iterdir()) if path.is_file()]
    raw_hashes = {str(path.relative_to(run_directory)): sha(path) for path in raw_paths}
    device = telemetry(run_directory)
    by_variant = {f"c{variant}": [run["timings"]["mxfp4_certified_total"]["synchronized_wall_median_ms"]
                                  for run in runs if run["variant"] == variant] for variant in VARIANTS}
    return {
        "schema_version": 1, "evidence_type": "independent_v6_c6_c8_local_evidence_audit", "passed": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "gpu_executed_by_this_script": False,
        "summarizer_sha256": sha(Path(__file__)), "raw_input_directory": str(run_directory),
        "experiment": run_subdir or "initial_warmup3", "reused_build_evidence": reused_build,
        "benchmark_order": [6, 8, 8, 6], "shared_input_identity": runs[0]["identity"], "useful_flops": flops,
        "statistics": {"primary_clock": "synchronized_steady_clock", "rounds_per_run": 5,
            "iterations_per_round": 10, "warmup_per_backend": warmup, "separate_runs_per_variant": 2,
            "pooled_rounds_per_variant": 10, "all_rounds_retained": True,
            "rounds_are_independent_samples": False, "confidence_interval_or_significance_claimed": False,
            "note": "Pooled medians are descriptive. Runs share a device/time window; rounds are correlated and clock/load drift is not controlled."},
        "provenance": provenance, "driver": driver, "correctness": correctness, "runs": runs,
        "pooled": pooled, "certified_c8_vs_c6": comparison, "adjacent_run_pairs": pairs,
        "observed_time_dependence": {"certified_wall_run_medians_ms_in_run_order": by_variant,
            "measured_gpu_sm_clock_mhz_range": [device["gpus"]["0"]["metrics"]["sm_clock_mhz"][key] for key in ("min", "max")],
            "note": "Run medians and sampled clocks can vary over the window; no per-round clock attribution or causal correction is inferred."},
        "device_telemetry": device, "compiled_resources": compiler_resources(directory),
        "scope_limits": {"c2_or_cublas_comparison_performed": False, "high_precision_quantization_included": False,
            "end_to_end_training_measured": False, "real_training_tensor": False, "exclusive_device": False,
            "single_mechanism_causality_established": False}, "raw_file_sha256": raw_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).parent / "verified/v6")
    parser.add_argument("--run-subdir", choices=("warm100",), help="Audit this independent performance run using the original V6 build evidence")
    parser.add_argument("--collection", type=Path, help="Also hash and bind a local collected .tar.gz archive or its .b64 transport file")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.input.resolve()
    output = args.output or directory / (args.run_subdir or ".") / "comparison-summary.json"
    if output.exists():
        parser.error("output exists; choose a new path to preserve previous evidence")
    repo = next(p for p in Path(__file__).resolve().parents if (p / "tools/mxfp4_training/build_cuda.py").is_file())
    try:
        report = summarize(directory, repo, args.run_subdir)
        if args.collection:
            report["collection_archive"] = verify_collection(args.collection, report["raw_file_sha256"])
    except (ValueError, KeyError, OSError, AssertionError, tarfile.TarError) as error:
        print(json.dumps({"passed": False, "gpu_executed_by_this_script": False, "error": str(error)}), file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "output": str(output), "sha256": sha(output),
                      "c8_vs_c6": report["certified_c8_vs_c6"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
