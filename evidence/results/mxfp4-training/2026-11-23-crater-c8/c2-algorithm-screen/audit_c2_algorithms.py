#!/usr/bin/env python3
"""Local, content-based audit of the actual C2 cuBLAS algorithm screen.

Uses the SHA-pinned independent native auditor for safe tar/JSON primitives and
certificate checks. Never imports archived production/candidate code, executes
ELF/CUDA/compiler/network commands, or manufactures benchmark observations.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

HELPER_SHA = "054bd0a94b17d48426f65bcd8cfea0566d1c63c167fe342ceed48347288be8b9"
CU_SHA = "ab55fc59c1f10d392d1d867a1a366b66ec159eabc3f58689f0e2069084c221ab"
WRAPPER_SHA = "e2229c8d04b6a9e6ffd32d72bca3dfe5104015e69aafb591d600ad236db43239"
RESULT_SHA = "b9bb395a2abaeaab11e08e44452361f931c7d710a0e16861d4305e899bfa88c6"
NATIVE_SHA = "d6775a7149d64b175ff20af085e1f9788937531561f10a80edccd5efb09462b8"
COMMIT = "ed39b0d715271e7b359672a081d758b4e4d5070f"
SOURCE_SHA = "3fb8cc55336a187f3c863eb98ae4d5089f57aabd6b8b585db85c0d09e03c1bd0"
PREFIX = "tools/mxfp4_training/"
MAPPING = "evidence/benchmarks/sm70-packing/study/simulation/round3/cuda/fragment_mapping.hpp"
ALGORITHMS = {"CUBLAS_GEMM_DEFAULT": -1, "CUBLAS_GEMM_DEFAULT_TENSOR_OP": 99,
              **{f"CUBLAS_GEMM_ALGO{i}_TENSOR_OP": 100 + i for i in range(16)}}
CASES = ((4096, "random_block_scales"), (8192, "random_block_scales"), (8192, "random_unit_scale"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    helper_path = args.native_directory / "audit_c9c10.py"
    if sha(helper_path.read_bytes()) != HELPER_SHA:
        raise RuntimeError("Independent audit helper SHA differs")
    spec = importlib.util.spec_from_file_location("independent_mxfp4_native_auditor", helper_path)
    h = importlib.util.module_from_spec(spec)
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(h)
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
    require, fields = h.require, h.eq_fields
    targets = [Path(str(args.output_prefix) + ext) for ext in (".json", ".md")]
    require(not any(p.exists() for p in targets), "Refusing to overwrite audit")
    report = {"schema_version": 1, "purpose": "independent actual C2 algorithm screen audit",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "pending", "issues": [],
              "auditor_sha256": sha(Path(__file__).read_bytes()), "helper_sha256": HELPER_SHA,
              "audit_gpu_executed": False, "audit_remote_executed": False,
              "archived_code_executed": False, "actual_elf_locally_rehashed": False,
              "timescope": "normalize_gemm_compute", "fullpipeline_performance_measured": False,
              "production_algorithm_changed": False, "cases": []}
    try:
        source = h.Bundle(args.native_directory / "source.tar.gz", SOURCE_SHA, 64 * 1024**2)
        original_binding = h.load_json((args.native_directory / "source-binding.json").read_bytes())
        fields(original_binding, {"source_commit": COMMIT, "source_sha256": SOURCE_SHA}, "frozen source")
        require(source.hashes == {r["name"]: r["sha256"] for r in original_binding["members"]}, "Original source hashes differ")
        native = h.Bundle(args.native_directory / "results.tar.gz", NATIVE_SHA, 1024**3, "results")
        native_complete = native.doc("completion.json")
        fields(native_complete, {"source_commit": COMMIT, "source_archive_sha256": SOURCE_SHA,
                                 "source_hashes_before": source.hashes, "source_hashes_after": source.hashes,
                                 "source_and_binaries_unchanged": True, "status": "passed"}, "native prerequisite")
        native_checks = {name: h.native(native.lines(f"stages/{name}/stdout.log"), 2 if name == "check-certified-c2" else None)
                         for name in ("check-c2-baseline", "check-c2-tiled", "check-certified-c2")}
        receipt_raw = (args.directory / "archive-manifest.json").read_bytes()
        receipt = h.load_json(receipt_raw)
        fields(receipt, {"archive_sha256": RESULT_SHA, "bytes": 1028506,
                         "status": "completed_with_blocked_cases", "passed": False}, "screen receipt")
        bundle = h.Bundle(args.directory / "results.tar.gz", RESULT_SHA, 128 * 1024**2, "results")
        require(bundle.meta["bytes"] == receipt["bytes"], "Archive byte size differs")
        complete = bundle.doc("completion.json")
        require(bundle.hashes["completion.json"] == receipt["completion_sha256"], "Completion hash differs")
        require(complete["result_files_sha256"] == {k: v for k, v in bundle.hashes.items() if k != "completion.json"}, "Result member hash set differs")
        fields(complete, {"status": "completed_with_blocked_cases", "passed": False,
                          "production_source_and_binary_unchanged": True, "timescope": "normalize_gemm_compute",
                          "fullpipeline_performance_measured": False, "source_modified": False,
                          "packages_installed": False, "services_stopped": False, "clock_or_power_changes": False}, "screen completion")
        manifest = bundle.doc("provenance/compile-evidence.json")
        manifest_sha = bundle.hashes["provenance/compile-evidence.json"]
        fields(manifest, {"schema_version": 1, "passed": True, "returncode": 0, "certified_variant": 2,
                          "tiled_variant": 1, "gpu_executed": False, "native_production_sources_modified": False,
                          "candidate_sha256": CU_SHA, "builder_sha256": WRAPPER_SHA}, "actual compile manifest")
        require("release 12.8" in manifest["nvcc_version"], "Unexpected CUDA compiler")
        argv = manifest["argv"]
        for flag in ("-std=c++17", "--ftz=false", "-DMXFP4_TILED_VARIANT=1", "-DMXFP4_CERTIFIED_VARIANT=2", "-lcublas"):
            require(flag in argv, "Required compile flag missing: " + flag)
        require(argv[0] == "/usr/local/cuda-12.8/bin/nvcc", "Compiler path differs")
        frozen = manifest["frozen_source_binding"]
        require(frozen["record"] == original_binding and frozen["archive_sha256"] == SOURCE_SHA
                and frozen["sha256"] == sha((args.native_directory / "source-binding.json").read_bytes()), "Compile source publication differs")
        require(bundle.hashes["provenance/bench_c2_algorithms.cu"] == CU_SHA
                and bundle.hashes["provenance/candidate.py"] == WRAPPER_SHA, "Archived candidate source/wrapper differs")
        for file, expected in (("bench_c2_algorithms.cu", CU_SHA), ("candidate.py", WRAPPER_SHA)):
            require(sha((args.directory / file).read_bytes()) == expected, "Review source differs from actual compiled candidate")
        production = {k: v for k, v in source.hashes.items() if k.startswith(PREFIX + "csrc/") or k == MAPPING}
        require(manifest["production_source_sha256"] == complete["production_source_sha256"] == production, "Production source binding differs")
        archived_production = {k[len("provenance/production-source/"):]: v for k, v in bundle.hashes.items() if k.startswith("provenance/production-source/")}
        require(archived_production == production, "Archived actual production source differs")
        for name, expected in manifest["log_sha256"].items():
            require(h.name_safe(name) == name and '/' not in name and bundle.hashes["provenance/" + name] == expected, "Compile log binding differs")
        binary = bundle.files["provenance/bench_c2_algorithms"]
        require(binary.startswith(b"\x7fELF") and sha(binary) == manifest["binary"]["sha256"] == complete["binary_sha256"]
                and len(binary) == manifest["binary"]["bytes"], "Actual ELF size/hash mismatch")
        require(manifest_sha == complete["build_manifest_sha256"], "Compile manifest binding differs")
        require(argv[argv.index("-o") + 1] == manifest["binary"]["path"], "Compile output executable path differs")
        require(manifest["source_root"] == native_complete["run_directory"] + "/source", "Production root differs from native run")
        gate = complete["gpu_serialization_gate"]
        fields(gate, {"prior_native_directory": native_complete["run_directory"], "archive_sha256": NATIVE_SHA,
                      "completion_sha256": native.hashes["completion.json"], "prior_status": "passed",
                      "same_B_framework_lock_acquired": True, "repeated_C2_native_gates_passed": True}, "GPU serialization prerequisite")
        require(gate["archive_manifest_sha256"] == sha((args.native_directory / "archive-manifest.json").read_bytes()), "Prior receipt SHA differs")
        require(datetime.fromisoformat(complete["started_at_utc"]) >= datetime.fromisoformat(native_complete["finished_at_utc"]), "Screen began before native completion")
        launch_raw = (args.directory / "launch.json").read_bytes()
        controller_raw = (args.directory / "controller-state.json").read_bytes()
        launch, controller = h.load_json(launch_raw), h.load_json(controller_raw)
        require(launch["candidate_sha256"] == controller["candidate_sha256"], "Launch/controller candidate bindings differ")
        for name, expected in launch["candidate_sha256"].items():
            require(h.name_safe(name) == name and '/' not in name and sha((args.directory / name).read_bytes()) == expected,
                    "Launch candidate file changed: " + name)
        fields(controller, {"status": "completed_with_blocked_cases"}, "controller terminal state")
        require([s["name"] for s in controller["stages"]] == ["build", "run"], "Controller stage set differs")
        fields(controller["stages"][0], {"status": "passed", "returncode": 0}, "controller compilation")
        fields(controller["stages"][1], {"status": "completed_with_blocked_cases", "returncode": 2}, "controller screen")
        report["provenance"] = {"results_archive": bundle.meta, "source_archive": source.meta,
            "native_results_archive": native.meta, "archive_manifest_sha256": sha(receipt_raw),
            "completion_sha256": bundle.hashes["completion.json"], "compile_manifest_sha256": manifest_sha,
            "source_commit": COMMIT, "candidate_sha256": CU_SHA, "wrapper_sha256": WRAPPER_SHA,
            "binary_sha256": sha(binary), "binary_bytes": len(binary), "member_sha256": bundle.hashes,
            "native_C2_prerequisites_independently_rechecked": native_checks, "gpu_serialization_gate": gate,
            "launch_sha256": sha(launch_raw), "controller_state_sha256": sha(controller_raw),
            "controller_source_sha256": launch["candidate_sha256"]["controller.py"]}
        report["actual_elf_locally_rehashed"] = True
        entries = complete["cases"]
        require([(r["size"], r["mode"]) for r in entries] == list(CASES), "Screen case order/set differs")
        def compare(data, count, canonical=True):
            fields(data, {"values": count, "exact_mismatches": 0, "nonfinite_values": 0, "first_bad_index": None,
                          "comparison": "bitwise_except_signed_zero; nonfinite always fails"}, "full bit comparison")
            if canonical:
                require(data["negative_zero_values"] == 0, "Canonical +0 was violated")
        for rec in entries:
            size, mode = rec["size"], rec["mode"]
            label = str(size) + "-" + mode
            path = label + ".jsonl"
            require(rec["executed"] is True and rec["output_sha256"] == bundle.hashes[path], "Case execution/raw output binding differs")
            rows = bundle.lines(path)
            require(rows == bundle.lines(rec["stdout"]), "JSONL/stdout records differ")
            env, summary = h.one(rows, "screen_environment"), h.one(rows, "screen_summary")
            env_identity = {"M": size, "N": size, "K": size, "mode": mode, "seed": 20261123,
                            "input_fnv1a64": env["input_fnv1a64"]}
            for r in rows:
                if r["type"] != "round_environment":
                    fields(r, env_identity, "case identity")
            native_file = "probe-8192-random_block_scales-c2.jsonl" if (size, mode) == CASES[1] else f"bench-{size}-01-c2.jsonl"
            native_initial = h.one(native.lines(native_file), "correctness")
            require(env["input_fnv1a64"] == native_initial["input_fnv1a64"], "Screen inputs differ from independently checked native run")
            fields(env, {"actual_SM70": True, "actual_V100": True, "build_binding_sha256": manifest_sha,
                         "cublas_math_flags": 16, "A_B_type": "CUDA_R_16F", "C_type": "CUDA_R_32F",
                         "compute_type": "CUBLAS_COMPUTE_32F", "alpha": 1, "beta": 0, "pointer_mode": "HOST",
                         "transa": "T", "transb": "N", "blas_workspace_bytes": 4 * 1024**2,
                         "max_allocation_bytes": 1024**3, "reserve_bytes": 512 * 1024**2,
                         "warmup_per_supported_algorithm": 100, "rounds": 5, "iterations_per_round": 10,
                         "timescope": "normalize_gemm_compute", "normalization_timed": False, "certificate_timed": False,
                         "epilogue_timed": False, "fullpipeline_performance_measured": False,
                         "services_stopped": False, "clock_or_power_changes": False, "exclusive_gpu_claim": False}, "actual handle and timing configuration")
            allocation = 2 * size * (size // 2 + size // 32) + 4 * size**2 + env["workspace_bytes"]
            require(env["explicit_allocation_bytes"] == allocation <= 1024**3 and env["free_bytes_after_allocations"] >= 512 * 1024**2
                    and allocation <= env["free_bytes_before"] - 512 * 1024**2 and allocation <= env["free_bytes_before"] * 3 // 4,
                    "Memory budget/reserve evidence failed")
            cert = env["certificate"]
            h.certificate(cert, 1, (size, mode) != CASES[1])
            require(cert == native_initial["certificate"], "Actual normalization certificate differs from same-input native C2")
            command = rec["argv"]
            require(command[0] == manifest["binary"]["path"] and "--include-ordinary" not in command, "Algorithm executable/set differs")
            for flag, value in (("--size", str(size)), ("--mode", mode), ("--device", "0"), ("--build-binding-sha256", manifest_sha)):
                require(command.count(flag) == 1 and command[command.index(flag) + 1] == value, "Run argument differs: " + flag)
            result = {"size": size, "input_mode": mode, "input_fnv1a64": env["input_fnv1a64"],
                      "timescope": "normalize_gemm_compute", "fullpipeline_performance_measured": False,
                      "certificate": cert, "environment": env, "raw_sha256": bundle.hashes[path], "algorithms": []}
            if (size, mode) == CASES[1]:
                fields(rec, {"returncode": 77, "status": "blocked_certificate", "passed": False,
                             "algorithm_timing_executed": False}, "legitimate rejected case")
                fields(summary, {"status": "blocked_certificate", "passed": False, "algorithm_timing_executed": False,
                                 "throughput": None, "fullpipeline_performance_measured": False,
                                 "gpu_executed": True, "all_submitted_work_completed": True, "certificate": cert}, "rejected summary")
                require(cert["accepted"] is False and cert["reject_flags"] == 4 and len(rows) == 2, "Rejected case exposed timing/algorithm results")
                result.update(status="blocked_certificate", evidence_valid=True, algorithm_timing_executed=False,
                              original_C2_output_numerically_compared_in_this_screen=False,
                              same_input_fallback_outputs_compared_in_prior_native_probe=size**2)
                report["cases"].append(result)
                continue
            fields(rec, {"status": "passed", "passed": True, "returncode": 0}, "screen execution")
            fields(summary, {"status": "passed", "passed": True, "requested_algorithms": 18,
                             "supported_verified_algorithms": 18, "not_supported_algorithms": 0,
                             "gpu_executed": True, "all_submitted_work_completed": True,
                             "certificate_unchanged": True, "selection_requires_independent_fullpipeline_ABBA": True,
                             "round_samples_are_correlated": True, "fullpipeline_performance_measured": False}, "screen summary")
            prereq = h.one(rows, "screen_prerequisite")
            fields(prereq, {"passed": True, "original_c2_finite_outputs": size**2,
                            "original_c2_negative_zero_outputs": 0, "cpu_reference_exhaustive": False,
                            "candidate_vs_tiled_method": "exact transitivity through complete original C2 array"}, "original C2 reference")
            compare(prereq["original_c2_vs_cpu_selected"], 256)
            compare(prereq["original_c2_vs_tiled_all_outputs"], size**2, False)
            initial = [r for r in rows if r["type"] == "algorithm_initial_check"]
            final = [r for r in rows if r["type"] == "algorithm_result"]
            require(len(initial) == len(final) == 18 and {r["algorithm"] for r in initial} == {r["algorithm"] for r in final} == set(ALGORITHMS), "Actual algorithm set incomplete")
            start_by_name = {r["algorithm"]: r for r in initial}
            for row in final:
                name = row["algorithm"]; first = start_by_name[name]
                fields(first, {"algorithm_id": ALGORITHMS[name], "passed": True, "warmup_completed": 100}, "initial algorithm")
                compare(first["initial_scaled_output_vs_original_c2_all_outputs"], size**2)
                fields(row, {"algorithm_id": ALGORITHMS[name], "status": "passed", "passed": True,
                             "cublas_status": 0, "successful_gemm_calls": 151, "timescope": "normalize_gemm_compute",
                             "final_actual_timed_output_checked": True, "epilogue_applied_after_timer": True,
                             "final_scaled_output_vs_tiled_verified_by_full_reference_transitivity": True,
                             "fullpipeline_performance_measured": False}, "last actual timed algorithm output")
                compare(row["final_scaled_output_vs_original_c2_all_outputs"], size**2)
                timing = row["timing"]
                fields(timing, {"name": name, "scope": "normalize_gemm_compute", "primary_clock": "synchronized_steady_clock"}, "algorithm timer")
                wall, event = timing["synchronized_wall_round_ms"], timing["cuda_event_round_ms"]
                require(timing["round_ms"] == wall and all(isinstance(x, list) and len(x) == 5 and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in x) for x in (wall, event)), "Algorithm raw samples invalid")
                wm, em = statistics.median(wall), statistics.median(event)
                for key, wanted in (("median_ms", wm), ("min_ms", min(wall)), ("max_ms", max(wall)),
                                    ("synchronized_wall_median_ms", wm), ("cuda_event_median_ms", em),
                                    ("useful_tflops", 2 * size**3 / (wm * 1e9)), ("synchronized_wall_useful_tflops", 2 * size**3 / (wm * 1e9)),
                                    ("cuda_event_useful_tflops", 2 * size**3 / (em * 1e9))):
                    h.close(timing[key], wanted, key)
                result["algorithms"].append({"algorithm": name, "algorithm_id": ALGORITHMS[name], "status": "passed",
                    "successful_gemm_calls": 151, "initial_full_output_values": size**2, "last_actual_timed_full_output_values": size**2,
                    "exact_mismatches": 0, "nonfinite_values": 0, "negative_zero_values": 0,
                    "wall_round_ms": wall, "event_round_ms": event, "wall_median_ms": wm, "event_median_ms": em,
                    "wall_compute_tflops": 2 * size**3 / (wm * 1e9), "event_compute_tflops": 2 * size**3 / (em * 1e9)})
            result["algorithms"].sort(key=lambda a: a["wall_median_ms"])
            require(summary["best_screened_wall_algorithm"] == result["algorithms"][0]["algorithm"], "Claimed winner differs from independently recomputed median")
            rounds = [r for r in rows if r["type"] == "round_environment"]
            require([r["round"] for r in rounds] == list(range(5)), "Round telemetry incomplete")
            telemetry = [{"point": "before_screen", **env["telemetry"]}]
            telemetry += [{"point": f"before_round_{r['round']}", **r["telemetry"]} for r in rounds]
            telemetry += [{"point": "after_screen", **summary["telemetry"]}]
            result.update(status="passed", evidence_valid=True, supported_verified_algorithms=18, not_supported_algorithms=0,
                          best_observed_wall_algorithm=result["algorithms"][0]["algorithm"],
                          original_reference=prereq, telemetry_snapshots=telemetry,
                          independent_process_runs=1, rounds_per_algorithm=5, within_run_rounds_are_correlated=True)
            result["telemetry_ranges"] = {}
            for key in ("sm_clock_mhz", "memory_clock_mhz", "power_w", "power_limit_w"):
                values = [r[key] for r in telemetry if type(r.get(key)) in (int, float)]
                result["telemetry_ranges"][key] = {"min": min(values), "max": max(values)} if values else None
            report["cases"].append(result)
        require(len(report["cases"]) == 3 and sum(r["status"] == "passed" for r in report["cases"]) == 2, "Screen coverage incomplete")
        report.update(status="verified", payload_status=complete["status"],
            accepted_algorithm_shape_pairs=36, not_supported_algorithm_shape_pairs=0, blocked_certificate_cases=1,
            initial_plus_last_full_output_comparisons=2 * 18 * (4096**2 + 8192**2),
            actual_raw_gemm_calls=36 * 151, actual_timed_raw_gemm_calls=36 * 50,
            started_at_utc=complete["started_at_utc"], finished_at_utc=complete["finished_at_utc"],
            limitations=[
                "Only normalized FP16 GEMM compute is timed; certificate, packed decode/normalization, original epilogue, fallback, allocations, quantization and copies are excluded.",
                "Every actual final timed output receives the production C2 epilogue only after both endpoints, then full exact comparison against the preserved C2 reference and transitively exact tiled output; no replacement GEMM is used.",
                "The certificate is computed by production C2 once, then its actual normalized buffers/handle/4MiB workspace are reused. This differs from production full-pipeline timing, which recomputes the certificate/normalization each call.",
                "One process per accepted shape/mode and five correlated rounds per algorithm; this is a screen, not independent ABBA validation or a stable production selection.",
                "Boundary NVML samples can vary and are not per-algorithm continuous frequency traces. Clock/power changes were not requested; exclusive GPU use is not established.",
                "The 8192 block-scale case legitimately rejects on energy with no algorithm timing. Its numerical fallback output was verified in the separate same-input native probe, not copied/compared by this blocked screen.",
                "4096 block-scale and 8192 unit-scale distributions are distinct. These compute figures must not replace the independent full-pipeline numbers or be presented as complete MXFP4 training throughput.",
                "No production algorithm was changed. Any proposed winner still requires independent full-pipeline validation under paired comparable conditions."])
    except (h.AuditError, ValueError, KeyError, TypeError, OSError) as error:
        report.update(status="invalid_evidence")
        report["issues"].append({"error_type": type(error).__name__, "error": str(error)})
        report["cases"] = []
    targets[0].parent.mkdir(parents=True, exist_ok=True)
    with targets[0].open("x") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")
    lines = ["# C2 cuBLAS 算法筛选：独立原始证据审计", "", f"状态：`{report['status']}`。本审计仅处理真实本地归档，未执行 GPU、编译器、远端操作或归档代码。", "",
             "**这里全部性能数字仅为 normalized GEMM compute，未测完整 MXFP4 算子流水线，也未改变 production 算法。**", ""]
    if report["status"] == "verified":
        lines += [f"结果 tar SHA-256 `{RESULT_SHA}`；冻结生产源提交 `{COMMIT}`，源 tar SHA-256 `{SOURCE_SHA}`。",
                  "", f"实际 ELF 已在本地重新计算 SHA-256：`{report['provenance']['binary_sha256']}`；编译 manifest/log、候选源码、生产源码、全部归档成员及前序 native 正确性来源均一致。", "",
                  "## 观测到的筛选结果", "", "| 输入 | 最快观测算法 | Wall compute TFLOPS | 默认 DEFAULT_TENSOR_OP | 默认 DEFAULT |", "|---|---|---:|---:|---:|"]
        for case in report["cases"]:
            if case["status"] != "passed":
                continue
            by_name = {a["algorithm"]: a for a in case["algorithms"]}
            best = case["algorithms"][0]
            lines.append(f"| {case['size']}³ {case['input_mode']} | {best['algorithm']} | {best['wall_compute_tflops']:.6f} | {by_name['CUBLAS_GEMM_DEFAULT_TENSOR_OP']['wall_compute_tflops']:.6f} | {by_name['CUBLAS_GEMM_DEFAULT']['wall_compute_tflops']:.6f} |")
        lines += ["", "两个 accepted 输入各 18 种算法全部完成初次与最后实际 timed 输出的全量位比较。每种算法 151 次成功 GEMM，100 次 warmup，5×10 次计时；无 nonfinite、负零或位不一致。中位数直接由全部五个原始 wall/event round 重算。", "",
                  "8192³ random_block_scales 的证书因 energy bound 拒绝（flags=4），没有任何算法计时或虚构吞吐。上层 `completed_with_blocked_cases` 是此预期边界结果，不能改写成三个 case 均 passed。", ""]
        for case in report["cases"]:
            if case["status"] != "passed":
                continue
            lines += [f"## {case['size']}³ {case['input_mode']} 完整排序", "", "| 算法 | Wall ms | Wall compute TFLOPS | Event ms | Event compute TFLOPS |", "|---|---:|---:|---:|---:|"]
            for row in case["algorithms"]:
                lines.append(f"| {row['algorithm']} | {row['wall_median_ms']:.9f} | {row['wall_compute_tflops']:.6f} | {row['event_median_ms']:.9f} | {row['event_compute_tflops']:.6f} |")
            lines += ["", "边界 NVML 读数范围：`" + json.dumps(case["telemetry_ranges"], ensure_ascii=False) + "`。这些快照不是各算法执行期间的连续频率测量。", ""]
        lines += ["## 结论边界", ""] + ["- " + item for item in report["limitations"]]
    if report["issues"]:
        lines += ["## 阻断项", "", "```json", json.dumps(report["issues"], ensure_ascii=False, indent=2), "```"]
    with targets[1].open("x") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"status": report["status"], "issues": report["issues"], "outputs": [str(p) for p in targets]}, ensure_ascii=False))
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
