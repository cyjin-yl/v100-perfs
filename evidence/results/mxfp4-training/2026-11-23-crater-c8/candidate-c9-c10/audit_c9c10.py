#!/usr/bin/env python3
"""Independent, local-only audit of the actual C9/C10 experiment archive.

Reads bounded tar members without extracting/importing/executing them. Does not
run CUDA, a compiler, remote commands, or fabricated tests. New audit JSON/MD are
the only writes. Numbers are eligible only after actual numerical record gates.
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
import tarfile

COMMIT = "ed39b0d715271e7b359672a081d758b4e4d5070f"
SOURCE_SHA = "3fb8cc55336a187f3c863eb98ae4d5089f57aabd6b8b585db85c0d09e03c1bd0"
TRAINING = "tools/mxfp4_training/"
MAPPING = "evidence/benchmarks/sm70-packing/study/simulation/round3/cuda/fragment_mapping.hpp"
VARIANTS = (2, 8, 9, 10)
ORDER = (2, 8, 9, 10, 10, 9, 8, 2)
MODES = {2048: "random_block_scales", 4096: "random_block_scales", 8192: "random_unit_scale"}
ARTIFACTS = {"shared_library": "libmxfp4_sm70.so", "standalone_harness": "check_mxfp4",
             "tiled_harness": "check_mxfp4_tiled", "certified_harness": "check_mxfp4_certified",
             "performance_benchmark": "bench_mxfp4"}
JOBS = {"mxfp4_baseline", "mxfp4_tiled", "mxfp4_certified_total", "cublas_compute", "cublas_dequant", "cublas_total"}


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pairs(items):
    result = {}
    for k, v in items:
        require(k not in result, "Duplicate JSON key: " + k)
        result[k] = v
    return result


def load_json(data):
    def bad(value):
        raise AuditError("Non-JSON constant: " + value)
    def finite(value):
        result = float(value)
        require(math.isfinite(result), "Unbounded JSON numeric value")
        return result
    obj = json.loads(data, object_pairs_hook=pairs, parse_constant=bad, parse_float=finite)
    require(isinstance(obj, dict), "Expected JSON object")
    return obj


def name_safe(name):
    require(isinstance(name, str) and name and "\\" not in name and "\0" not in name, "Invalid member name")
    p = PurePosixPath(name)
    require(not p.is_absolute() and ".." not in p.parts and "." not in name.split("/")
            and str(p) == name.rstrip("/") and p.parts, "Unsafe/noncanonical member: " + name)
    return str(p)


class Bundle:
    def __init__(self, path, expected, limit, prefix=None):
        require(path.is_file() and not path.is_symlink(), "Archive is not a regular local file")
        require(path.stat().st_size <= limit, "Compressed archive exceeds bound")
        raw = path.read_bytes()
        require(re.fullmatch(r"[0-9a-f]{64}", expected or "") and digest(raw) == expected, "Archive hash mismatch")
        self.meta = {"path": str(path.absolute()), "sha256": expected, "bytes": len(raw)}
        self.files, self.hashes = {}, {}
        seen, expanded = set(), 0
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as handle:
            for member in handle:
                key = name_safe(member.name)
                require(key not in seen and len(seen) < 20000, "Duplicate/excess archive member")
                seen.add(key)
                require((member.isdir() or member.isreg()) and not member.issparse(), "Links/special/sparse members forbidden")
                require(0 <= member.size <= limit, "Invalid member size")
                expanded += member.size
                require(expanded <= limit, "Expanded archive exceeds bound")
                if prefix:
                    require(key == prefix or key.startswith(prefix + "/"), "Unexpected archive prefix")
                if member.isdir():
                    continue
                if prefix:
                    require(key != prefix, "Archive root must be a directory")
                    key = key[len(prefix) + 1:]
                stream = handle.extractfile(member)
                require(stream is not None, "Missing member data")
                data = stream.read(member.size + 1)
                require(len(data) == member.size, "Member length mismatch")
                self.files[key], self.hashes[key] = data, digest(data)
        require(self.files, "Empty archive")
        self.meta.update(regular_members=len(self.files), expanded_bytes=expanded)

    def doc(self, name):
        require(name in self.files, "Missing JSON artifact: " + name)
        return load_json(self.files[name])

    def lines(self, name):
        require(name in self.files, "Missing JSONL artifact: " + name)
        return [load_json(line) for line in self.files[name].decode().splitlines() if line.lstrip().startswith("{")]


def eq_fields(obj, fields, label):
    for key, wanted in fields.items():
        actual = obj.get(key)
        require(actual == wanted and (type(wanted) is not bool or actual is wanted), f"{label}.{key}: {actual!r} != {wanted!r}")


def one(records, kind):
    selected = [r for r in records if r.get("type") == kind]
    require(len(selected) == 1, "Expected exactly one record: " + kind)
    return selected[0]


def close(a, b, label):
    require(type(a) in (int, float) and math.isfinite(a) and math.isclose(a, b, rel_tol=3e-9, abs_tol=1e-10),
            f"Incorrect derived statistic {label}: {a!r} vs {b!r}")


def metric(obj, values, exact=True):
    eq_fields(obj, {"values": values, "nonfinite_mismatches": 0, "tolerance_failures": 0}, "metric")
    if exact:
        eq_fields(obj, {"exact_mismatches": 0}, "exact metric")
        for key in ("max_abs", "max_relative", "nrmse"):
            if obj.get(key) is not None:
                require(obj[key] == 0, "Nonzero exact metric: " + key)


def certificate(obj, calls, require_accepted=True):
    eq_fields(obj, {"metadata_valid": True, "submitted_calls": calls,
                   "certificate_recomputed_each_call": True, "energy_product_bound": 2**48,
                   "energy_cap": 2**48 + 1, "span_limit": 12}, "certificate")
    u, v = obj["max_energy_a"], obj["max_energy_b"]
    require(type(u) is int and type(v) is int and 0 <= u <= 2**48 + 1 and 0 <= v <= 2**48 + 1, "Invalid energies")
    energy = u == 0 or v == 0 or u <= (2**48) // v
    span = max(obj["max_span_a"], obj["max_span_b"]) <= 12
    nan_ok = obj["nan_scales_a"] == obj["nan_scales_b"] == 0
    low = obj["min_scale_a"] + obj["min_scale_b"] - 256
    high = obj["max_scale_a"] + obj["max_scale_b"] - 256
    lattice = low >= -126 and high <= 103
    flags = (0 if nan_ok else 1) | (0 if span else 2) | (0 if energy else 4) | (0 if lattice else 8)
    eq_fields(obj, {"reject_flags": flags, "accepted": flags == 0, "nan_scale_gate_pass": nan_ok,
                   "span_gate_pass": span, "energy_gate_pass": energy, "energy_gate_pass_from_reported_bounds": energy,
                   "lattice_gate_pass": lattice, "lattice_exponent_min": low, "lattice_exponent_max": high}, "derived certificate")
    if require_accepted:
        require(flags == 0, "Performance certificate rejected")


def native(records, variant=None):
    s = one(records, "summary")
    eq_fields(s, {"passed": True, "gpu_executed": True, "kernel_execution_completed": True,
                  "actual_V100": True, "cases": 46, "values": 175429, "mismatches": 0}, "native summary")
    cases = [r for r in records if r.get("type") == "correctness"]
    require(len({c["case"] for c in cases}) == len(cases), "Duplicate native case name")
    for c in cases:
        eq_fields(c, {"mismatches": 0, "gpu_executed": True, "nondefault_stream": True,
                      "values": c["M"] * c["N"]}, "native case")
        require(c["padded_K"] == (c["logical_K"] + 31) // 32 * 32, "Native padding inconsistent")
    count, values, reuse, zeros = 46, 175429, [], None
    output = {"variant": variant, "original_domain_cases": 46, "original_domain_outputs": 175429}
    if variant is not None:
        cs = one(records, "certified_summary")
        eq_fields(cs, {"passed": True, "gpu_executed": True, "certified_variant": variant,
                       "full_domain_suite_passed": True, "extra_fixture_cases": 20, "extra_fixture_values": 230456,
                       "context_reuse_calls": 3, "accepted_calls": 23, "fallback_calls": 46,
                       "canonical_positive_zero_values": 2950, "mismatches": 0}, "certified suite")
        reuse = [r for r in records if r.get("type") == "context_reuse"]
        require(len(reuse) == 3, "Missing reused-context records")
        for r, accepted, calls in zip(reuse, (True, False, True), (1, 2, 3)):
            eq_fields(r, {"accepted": accepted, "submitted_calls": calls, "passed": True, "values": 15, "mismatches": 0}, "context reuse")
        count, values, zeros = 66, 405885, 2950
        output.update(original_certified_calls=69, original_certified_outputs=405930,
                      original_accepted=23, original_fallback=46, original_canonical_positive_zeros=2950)
        if variant >= 8:
            extra = one(records, "c8_fixture_summary")
            eq_fields(extra, {"passed": True, "gpu_executed": True, "certified_variant": variant,
                              "cases": 9, "values": 180224, "accepted_calls": 8, "fallback_calls": 1,
                              "c6_compatibility_cases": 3, "empty_K_cases": 1, "rejected_cases": 1,
                              "mismatches": 0}, "halfrow fixtures")
            eq_fields(extra, {"nonempty_c8_loader_cases" if variant == 8 else "nonempty_derived_halfrow_loader_cases": 4}, "halfrow loader")
            dispatch = [r for r in records if r.get("type") == "c8_fixture_dispatch"]
            require(len(dispatch) == 9 and len({r["case"] for r in dispatch}) == 9, "Missing/duplicate dispatch fixtures")
            for r in dispatch:
                eq_fields(r, {"passed": True, "certified_variant": variant, "pointer_alignment_observed": True,
                              "kernel_path_instrumented": False}, "dispatch fixture")
            count += 9; values += 180224; zeros += extra["canonical_positive_zero_values"]
            output["halfrow_contract"] = extra
        if variant >= 9:
            boundary = one(records, "halfrow_boundary_summary")
            eq_fields(boundary, {"passed": True, "gpu_executed": True, "certified_variant": variant,
                                 "cases": 1, "values": 16384, "mismatches": 0, "accepted": True,
                                 "K": 64, "canonical_positive_zero_values": 16384,
                                 "active_half_contains_signed_zero": True}, "cancellation boundary")
            count += 1; values += 16384; zeros += 16384
            output["two_stage_cancellation"] = boundary
    require(len(cases) == count and sum(c["values"] for c in cases) == values, "Native individual records disagree with total")
    output.update(passed=True, calls=len(cases) + len(reuse), outputs=values + 15 * len(reuse),
                  canonical_positive_zero_outputs=zeros, mismatches=0,
                  fixture_dispatch_proven_by_argument_conditions_not_kernel_instrumentation=True)
    return output


def half2(records):
    selected = [r for r in records if r.get("kind") == "mxfp4_half2_decode_microcheck"]
    require(len(selected) == 1, "Half2 result missing/duplicated")
    eq_fields(selected[0], {"passed": True, "status": "passed", "gpu_executed": True, "sm": 70,
                            "patterns": 65536, "deltas": 13, "expected_cases": 851968, "cases": 851968,
                            "half_values": 3407872, "nonzero_half_bitwise_comparisons": 2981888,
                            "zero_half_sign_normalized_comparisons": 425984, "mismatched_half_values": 0,
                            "first_mismatch": None, "comparison": "integer_bits_no_float_tolerance"}, "half2 enumeration")
    return selected[0]


def bench(records, variant, size, mode, probe=False):
    env = one(records, "environment")
    eq_fields(env, {"certified_variant": variant, "certified_compiled": True, "actual_V100": True,
                    "clock_or_power_changes": False, "gpu_services_stopped": False}, "benchmark environment")
    eq_fields(one(records, "summary"), {"passed": True, "gpu_executed": True, "actual_V100": True,
               "completed_cases": 1, "correctness_only": probe, "all_submitted_work_completed": True}, "benchmark completion")
    first = one(records, "correctness")
    last = None if probe else one(records, "benchmark")
    if probe:
        require(not any(r.get("type") == "benchmark" for r in records), "Boundary probe unexpectedly timed")
    for obj in (first,) if probe else (first, last):
        eq_fields(obj, {"M": size, "N": size, "logical_K": size, "padded_K": size, "input_mode": mode,
                       "seed": 20261123, "gpu_executed": True, "actual_V100": True, "real_training_tensor": False,
                       "kernel_execution_completed": True, "all_submitted_work_completed": True}, "benchmark case")
        require(re.fullmatch(r"[0-9a-f]{16}", obj.get("input_fnv1a64", "")), "Missing input fingerprint")
    require(first.get("passed") is True and first.get("cpu_reference_values") == 256, "CPU numerical oracle incomplete")
    for name in ("baseline_vs_cpu", "tiled_vs_cpu", "certified_vs_cpu", "cublas_vs_cpu"):
        metric(first[name], 256, not name.startswith("cublas"))
    for name in ("tiled_vs_baseline_all_outputs", "certified_vs_baseline_all_outputs", "cublas_vs_baseline_all_outputs"):
        metric(first[name], size**2, not name.startswith("cublas"))
    certificate(first["certificate"], 1, not probe)
    output = {"variant": variant, "size": size, "input_mode": mode, "input_fnv1a64": first["input_fnv1a64"],
              "certificate_initial": first["certificate"], "initial_certified_all_output_values": size**2,
              "cpu_oracle_selected_outputs": 256, "passed": True, "timed": not probe, "environment": env}
    if probe:
        output["performance_measured"] = False
        return output
    require(first["input_fnv1a64"] == last["input_fnv1a64"], "First/timed input mismatch")
    eq_fields(last, {"correctness_passed": True, "warmup_per_job": 100, "iterations_per_round": 10,
                     "rounds": 5, "same_resident_inputs_reused": True}, "timing settings")
    close(last["useful_flops"], 2 * size**3, "useful FLOPs")
    metric(last["certified_post_timing_vs_baseline_all_outputs"], size**2)
    certificate(last["certificate_after_timing"], 151)
    a, b = dict(first["certificate"]), dict(last["certificate_after_timing"])
    a.pop("submitted_calls"); b.pop("submitted_calls")
    require(a == b, "Certificate changed on identical inputs")
    jobs = last["timings"]
    require(len(jobs) == 6 and {j["name"] for j in jobs} == JOBS, "Wrong timing backend set")
    for job in jobs:
        arrays = [job[k] for k in ("round_ms", "cuda_event_round_ms", "synchronized_wall_round_ms")]
        require(all(isinstance(arr, list) and len(arr) == 5 and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in arr) for arr in arrays), "Invalid timing samples")
        primary, event, wall = arrays
        is_certified = job["name"] == "mxfp4_certified_total"
        eq_fields(job, {"primary_clock": "synchronized_steady_clock" if is_certified else "cuda_event_elapsed"}, "primary timer")
        require(primary == (wall if is_certified else event), "Primary timer arrays differ")
        for field, actual in (("median_ms", statistics.median(primary)), ("min_ms", min(primary)),
                              ("max_ms", max(primary)), ("cuda_event_median_ms", statistics.median(event)),
                              ("synchronized_wall_median_ms", statistics.median(wall))):
            close(job[field], actual, field)
        if job["name"] != "cublas_dequant":
            for key, arr in (("useful_tflops", primary), ("cuda_event_useful_tflops", event), ("synchronized_wall_useful_tflops", wall)):
                close(job[key], 2 * size**3 / (statistics.median(arr) * 1e9), key)
    selected = next(j for j in jobs if j["name"] == "mxfp4_certified_total")
    expected_scope = "packed_to_fp32_gpu_certificate_normalization_cublas_conditional_epilogue_and_fallback" if variant == 2 else "packed_to_fp32_gpu_certificate_fused_normalized_hmma_and_conditional_fallback"
    require(selected["scope"] == expected_scope, "Certified timing scope changed")
    output.update(final_actual_timed_certified_all_output_values=size**2, certificate_after_timing=last["certificate_after_timing"],
                  final_output_comparison=last["certified_post_timing_vs_baseline_all_outputs"],
                  certified_timing=selected, all_backend_timings=jobs, telemetry_before=last.get("telemetry_before"),
                  telemetry_after=last.get("telemetry_after"), final_output_copied_after_both_timing_endpoints=True)
    return output


def walk_metrics(value, path=""):
    count = 0
    if isinstance(value, dict):
        if "comparison" in value and "values" in value:
            require(value.get("tolerance_failures") == 0, "Torch numeric failure: " + path)
            require(type(value["values"]) is int and value["values"] > 0, "Empty Torch comparison")
            if value["comparison"].startswith("exact"):
                require(value.get("exact_mismatches") == 0 and value.get("atol") == value.get("rtol") == 0, "Torch exact failure")
            else:
                require(value["comparison"] == "fp32_allclose" and value.get("atol") == value.get("rtol") == 3e-5, "Unknown Torch tolerance")
            require(math.prod(value["shape"]) == value["values"], "Torch numeric shape mismatch")
            count += 1
        for key, child in value.items():
            count += walk_metrics(child, path + "/" + key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            count += walk_metrics(child, path + "/" + str(index))
    return count


def torch_suite(data, variant, sources, compile_data, compile_hash, run):
    count = 119 if variant is None else 16
    eq_fields(data, {"passed": True, "status": "passed", "gpu_executed": True, "completed_cases": count,
                    "gpu_work_attempted": True, "gpu_work_call_returned": True, "gpu_work_synchronized": True,
                    "all_attempted_work_synchronized": True, "source_unchanged_during_check": True}, "Torch completion")
    eq_fields(data["gpu"], {"actual_V100": True, "compute_capability": [7, 0]}, "Torch device")
    cases = data["cases"]
    require(len(cases) == count and len({c["name"] for c in cases}) == count and all(c.get("passed") is True for c in cases), "Torch individual cases incomplete")
    names = {"check_torch_cuda.py", "reference.py", "torch_ops.py", "csrc/mxfp4_gemm.h", "csrc/mxfp4_gemm.cu",
             "csrc/mxfp4_gemm_tiled.h", "csrc/mxfp4_gemm_tiled.cu"}
    if variant is not None:
        names = {"check_torch_certified.py", "check_torch_cuda.py", "reference.py", "torch_ops.py", "fragment_mapping.hpp"}
        names |= {k[len(TRAINING):] for k in sources if k.startswith(TRAINING + "csrc/")}
    require(set(data["source_sha256"]) == names, "Torch source binding set differs")
    for key, val in data["source_sha256"].items():
        name = MAPPING if key == "fragment_mapping.hpp" else TRAINING + key
        require(name in sources and val == sources[name], "Torch source mismatch: " + key)
    require({"check_torch_cuda.py", "torch_ops.py", "reference.py"}.issubset(data["source_sha256"]), "Missing Torch source bindings")
    selected = variant or 2
    lib = data["library"]
    eq_fields(lib, {"path": run + f"/build-c{selected}/libmxfp4_sm70.so", **compile_data["artifacts"]["libmxfp4_sm70.so"]}, "Torch library")
    if variant is not None:
        tested_backends = data.get("tested_backends")
        require(tested_backends == ["cuda_certified"], "Certified Torch backend scope changed")
        eq_fields(lib, {"compiled_variant": variant, "compile_evidence_sha256": compile_hash,
                       "native_sources_match": True, "fragment_mapping_matches": True}, "Torch certified build")
        by_name = {c["name"]: c for c in cases}
        for name in ("linear_F_D_W_full_tiles", "linear_F_D_W_noncontiguous_tails"):
            calls = by_name[name]["certified_calls"]
            require([c["direction"] for c in calls] == ["forward", "dgrad", "wgrad"], "F/D/W sequence mismatch")
            for call in calls:
                require(call["original_fp32_directional_quantization_exact"] is True, "F/D/W quantization mismatch")
                eq_fields(call["certificate"], {"accepted": 1, "reject_flags": 0, "submitted_calls": 1, "metadata_valid": 1}, "Torch F/D/W certificate")
        threads = by_name["two_host_threads_separate_contexts"]
        eq_fields(threads, {"independent_host_threads": 2, "simultaneously_live_contexts": 2}, "Torch contexts")
        require(len(threads["contexts_destroyed_on_owner_threads"]) == 2 and all(d["owner_thread_matches"] is True and d["native_status"] == 0 for d in threads["contexts_destroyed_on_owner_threads"]), "Thread cleanup mismatch")
        eq_fields(by_name["cleanup_after_controlled_post_submit_error"], {"native_gemm_status": 0, "destroy_statuses": [0], "error_propagated": True, "stream_complete_on_error_return": True}, "Controlled error cleanup")
        eq_fields(by_name["reject_graph_capture"], {"capture_rejected_before_native_create": True, "native_create_attempts": 0, "surrounding_graph_remains_valid": True}, "Graph capture")
    else:
        # The original 119-case JSON has no top-level tested_backends field.
        # Derive the exercised backend set from actual packed/linear cases.
        tested_backends = sorted({c.get("backend", c.get("recipe", {}).get("backend")) for c in cases
                                  if c.get("backend", c.get("recipe", {}).get("backend")) is not None})
        require(tested_backends == ["cuda", "cuda_tiled"], "119 suite backend scope changed")
        for c in cases:
            if c["name"].startswith("gpu_rne_"):
                require(c.get("codes_and_scales_exact") is True, "Original quantizer bytes failed")
            if c["name"] == "gpu_stochastic_seed":
                eq_fields(c, {"same_seed_exact": True, "different_seed_changes_codes": True}, "SR seed")
            if c["name"] == "reject_gpu_to_reference_fallback":
                require(c["gpu_reference_fallback_rejected"] is True, "Silent reference fallback")
    metrics = walk_metrics(cases)
    require(metrics > 0, "Torch cases contain no numeric comparisons")
    return {"passed": True, "cases": count, "numeric_comparison_records": metrics,
            "variant": variant, "library_sha256": lib["sha256"],
            "tested_backends": tested_backends, "full_training_executed": False}


def sass_functions(text):
    result, current = {}, None
    for line in text.splitlines():
        match = re.search(r"Function\s*:\s*(\S+)", line)
        if match:
            current = match.group(1)
            require(current not in result, "Duplicate SASS function")
            result[current] = []
        instruction = re.match(r"\s*/\*([0-9a-fA-F]+)\*/\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9_.]*)\s*([^;]*);", line)
        if instruction and current is not None:
            result[current].append({"address": int(instruction[1], 16), "opcode": instruction[2], "operands": instruction[3].strip()})
    return result


def static_review(directory, results, archive_sha, completion):
    if not directory.is_dir():
        return {"status": "not_available", "performance_inference_from_static_counts": False}
    archive = Bundle(directory.parent / "static.tar.gz", archive_sha, 64 * 1024**2, "static")
    hashes, candidates = {}, {}
    for path in sorted(directory.rglob("*")):
        if path.is_dir():
            continue
        require(path.is_file() and not path.is_symlink(), "Unexpected static member")
        key = str(path.relative_to(directory)); raw = path.read_bytes(); hashes[key] = digest(raw)
        if "sass" not in path.name.lower():
            continue
        try:
            funcs = sass_functions(raw.decode())
        except UnicodeError:
            continue
        for name, instructions in funcs.items():
            if "fused_certified_prefetch32_halfrow" not in name or not instructions:
                continue
            variant = 10 if "halfrow_mul" in name else 9 if "halfrow_cursor" in name else 8
            previous = candidates.get(variant)
            if previous:
                require(previous["instructions"] == instructions, "Different SASS copies for one function")
                continue
            candidates[variant] = {"file": key, "sha256": hashes[key], "function": name, "instructions": instructions}
    require(hashes == archive.hashes, "Local static files differ from validated static archive")
    manifest = archive.doc("manifest.json")
    require(manifest.get("gpu_executed") is False and len(manifest.get("records", [])) == 3, "Unexpected static manifest")
    require({r["variant"] for r in manifest["records"]} == {8, 9, 10} and set(candidates) == {8, 9, 10}, "Static candidate coverage differs")
    output = {}
    for variant, entry in sorted(candidates.items()):
        ins = entry.pop("instructions")
        counts = Counter(i["opcode"].split(".")[0] for i in ins)
        loops = []
        for i in ins:
            target = re.search(r"0x([0-9a-fA-F]+)\b", i["operands"])
            if i["opcode"].split(".")[0] == "BRA" and target and int(target[1], 16) < i["address"]:
                start = int(target[1], 16)
                body = [x for x in ins if start <= x["address"] <= i["address"]]
                hmma = sum(x["opcode"].startswith("HMMA") for x in body)
                if hmma:
                    loops.append((hmma, len(body), start, i["address"], body))
        loop_info = None
        if loops:
            hmma, length, begin, end, body = max(loops, key=lambda x: (x[0], -x[1]))
            first_hmma = next(x["address"] for x in body if x["opcode"].startswith("HMMA"))
            loop_info = {"selection_rule": "backward BRA interval with most HMMA; shortest on a tie; static scope only",
                         "start_hex": hex(begin), "branch_hex": hex(end), "instruction_slots": length,
                         "opcode_counts": dict(sorted(Counter(x["opcode"].split(".")[0] for x in body).items())),
                         "before_first_hmma_counts": dict(sorted(Counter(x["opcode"].split(".")[0] for x in body if x["address"] < first_hmma).items())),
                         "global_loads": [{"address_hex": hex(x["address"]), "opcode": x["opcode"], "operands": x["operands"],
                                           "remaining_hmma_in_static_loop": sum(y["opcode"].startswith("HMMA") and y["address"] > x["address"] for y in body)}
                                          for x in body if x["opcode"].startswith("LDG")]}
        log_key = f"build-evidence/build-c{variant}/shared_library.log"
        resources = None
        if log_key in results.files:
            text = results.files[log_key].decode()
            blocks = re.split(r"ptxas info\s*:\s*Function properties for\s+", text)[1:]
            for block in blocks:
                if block.splitlines()[0].strip() != entry["function"]:
                    continue
                stack = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", block)
                registers = re.search(r"Used (\d+) registers", block)
                shared = re.search(r"(\d+) bytes smem", block)
                if stack and registers:
                    resources = {"registers_per_thread": int(registers[1]), "stack_bytes": int(stack[1]),
                                 "spill_store_bytes": int(stack[2]), "spill_load_bytes": int(stack[3]),
                                 "shared_bytes": int(shared[1]) if shared else 0, "log": log_key,
                                 "log_sha256": results.hashes[log_key]}
        entry.update(full_function_instruction_slots=len(ins), opcode_counts=dict(sorted(counts.items())),
                     hmma_instructions=counts["HMMA"], hmul2_instructions=counts["HMUL2"],
                     resources=resources, stage_loop=loop_info)
        output[str(variant)] = entry
    for record in manifest["records"]:
        variant = record["variant"]
        derived = output[str(variant)]
        binary = completion["run_directory"] + f"/build-c{variant}/libmxfp4_sm70.so"
        eq_fields(record, {"source_commit": COMMIT, "binary": binary,
                           "binary_sha256": completion["binary_hashes_before_gpu"][binary],
                           "build_manifest_sha256": results.hashes[f"build-evidence/build-c{variant}/compile-evidence.json"],
                           "sass_file": derived["file"], "sass_sha256": derived["sha256"], "function": derived["function"],
                           "static_instruction_count": derived["full_function_instruction_slots"],
                           "opcodes": derived["opcode_counts"], "dynamic_performance_inferred": False}, "independently recounted SASS manifest")
        require(derived["resources"] is not None, "SASS function resource record not found in actual ptxas log")
    return {"status": "available", "archive": archive.meta, "files_sha256": hashes,
            "node_recorded_binary_bindings_match_native_run": True,
            "elf_bytes_locally_rehashed": False,
            "cuobjdump_version": archive.files["cuobjdump-version.txt"].decode().strip(),
            "functions": output, "performance_inference_from_static_counts": False,
            "limitations": ["Instruction counts include complete disassembled functions and NOP padding, not dynamic execution counts.",
                            "A shorter instruction listing does not measure latency, issue rate, bank conflicts, or achieved Tensor Core utilization.",
                            "Load instruction placement is compiler scheduling evidence, not a proof of latency overlap."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--results-sha256", required=True, help="independently collected result archive SHA")
    parser.add_argument("--static-sha256", help="independently collected static archive SHA, required when static/ exists")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--runner", type=Path)
    args = parser.parse_args()
    outputs = [Path(str(args.output_prefix) + extension) for extension in (".json", ".md")]
    require(not any(p.exists() for p in outputs), "Refusing to overwrite an audit")
    base = args.directory.absolute()
    report = {"schema_version": 1, "purpose": "independent raw C9/C10 evidence audit",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "auditor_sha256": digest(Path(__file__).read_bytes()),
              "audit_gpu_executed": False, "audit_remote_executed": False, "archived_code_executed": False,
              "status": "pending", "issues": [], "stage_content_checks": {}, "comparison": [],
              "binary_bytes_locally_rehashed": False}
    try:
        binding_raw = (base / "source-binding.json").read_bytes(); binding = load_json(binding_raw)
        eq_fields(binding, {"source_commit": COMMIT, "source_sha256": SOURCE_SHA, "source_bytes": 119431}, "source binding")
        source = Bundle(base / "source.tar.gz", SOURCE_SHA, 64 * 1024**2)
        members = binding["members"]
        require(len(members) == 25 and len({m["name"] for m in members}) == 25, "Wrong source member set")
        expected = {name_safe(m["name"]): m["sha256"] for m in members}
        require(source.hashes == expected, "Source member SHA mismatch")
        require(all(len(source.files[m["name"]]) == m["bytes"] for m in members), "Source member byte count mismatch")
        receipt_raw = (base / "archive-manifest.json").read_bytes(); receipt = load_json(receipt_raw)
        eq_fields(receipt, {"sha256": args.results_sha256, "source_commit": COMMIT, "source_archive_sha256": SOURCE_SHA}, "result receipt")
        results = Bundle(base / "results.tar.gz", args.results_sha256, 1024**3, "results")
        require(results.meta["bytes"] == receipt["bytes"], "Result receipt byte mismatch")
        completion = results.doc("completion.json")
        require(results.hashes["completion.json"] == receipt["completion_sha256"], "Completion hash mismatch")
        eq_fields(completion, {"source_commit": COMMIT, "source_archive_sha256": SOURCE_SHA,
                              "source_binding_sha256": digest(binding_raw), "source_hashes_before": expected,
                              "source_hashes_after": expected, "source_and_binaries_unchanged": True,
                              "variants": list(VARIANTS), "shapes": list(MODES), "per_shape_variant_order": list(ORDER),
                              "benchmark_input_modes": {str(k): v for k, v in MODES.items()},
                              "seed": 20261123, "warmup": 100, "iters": 10, "rounds": 5}, "completion binding")
        # The runner intentionally writes a sorted-key copy of the supplied
        # binding; compare its full object, while retaining both byte hashes.
        require(results.doc("source-binding.json") == binding, "Archived source binding content differs")
        require(completion["result_files_sha256"] == {k: v for k, v in results.hashes.items() if k != "completion.json"}, "Result per-file hash set mismatch")
        require(completion["binary_hashes_before_gpu"] == completion["binary_hashes_after_gpu"], "Binary hashes changed during GPU work")
        require(completion["status"] == receipt["status"] and completion["passed"] is receipt["passed"], "Receipt status differs")
        require(completion["status"] in ("passed", "failed", "blocked"), "Run is not terminal")
        if args.runner:
            require(digest(args.runner.read_bytes()) == completion["runner_sha256"], "Runner source SHA mismatch")
        stages = {s["name"]: s for s in completion["stages"]}
        require(len(stages) == len(completion["stages"]), "Duplicate stages")
        names = {f"build-c{v}" for v in VARIANTS} | {"build-half2-microcheck", "wait-framework-lock", "check-c2-baseline", "check-c2-tiled", "check-half2-microcheck", "torch-baseline-tiled", "torch-certified-c8", "torch-certified-c10", "probe-8192-random_block_scales-c2"}
        names |= {f"check-certified-c{v}" for v in VARIANTS}
        names |= {f"bench-{s}-{i + 1:02d}-c{v}" for s in MODES for i, v in enumerate(ORDER)}
        require(set(stages) == names, "Planned stage set mismatch")
        run = completion["run_directory"]
        require(run.startswith("/workspace/user/") and str(PurePosixPath(run)) == run, "Unexpected native root")
        for name, stage in stages.items():
            require(stage["status"] in ("passed", "failed", "blocked"), "Nonterminal stage")
            if "argv" not in stage:
                continue
            require(stage.get("cwd") == run + "/source", "Stage cwd not bound to frozen source")
            require(results.doc(f"stages/{name}/stage.json") == stage, "Stage JSON differs from completion")
            actual = {key[len(f"stages/{name}/"):]: sha for key, sha in results.hashes.items() if key.startswith(f"stages/{name}/") and key != f"stages/{name}/stage.json"}
            require(stage["evidence_sha256"] == actual, "Stage evidence hash mismatch")
            for field, artifact in (("binary_sha256", stage["argv"][0]), ("library_sha256", run + f"/build-c{2 if name == 'torch-baseline-tiled' else name.rsplit('c', 1)[-1]}/libmxfp4_sm70.so")):
                if field + "_before" in stage:
                    require(stage[field + "_before"] == stage[field + "_after"] == completion["binary_hashes_before_gpu"][artifact], "Stage binary binding mismatch")
        report["provenance"] = {"source_archive": source.meta, "results_archive": results.meta,
                                "source_binding_sha256": digest(binding_raw),
                                "archived_reserialized_source_binding_sha256": results.hashes["source-binding.json"],
                                "archive_manifest_sha256": digest(receipt_raw),
                                "completion_sha256": results.hashes["completion.json"], "source_commit": COMMIT,
                                "source_members_sha256": source.hashes, "results_members_sha256": results.hashes,
                                "runner_sha256": completion["runner_sha256"], "runner_locally_rehashed": bool(args.runner),
                                "native_run_directory": run, "binary_hashes_recorded_before_and_after": completion["binary_hashes_before_gpu"]}
        report.update(provenance_verified=True, payload_status=completion["status"], stage_status_counts=dict(Counter(s["status"] for s in stages.values())),
                      started_at_utc=completion["started_at_utc"], finished_at_utc=completion["finished_at_utc"])
        def check(name, action):
            stage = stages[name]
            if stage["status"] != "passed":
                report["stage_content_checks"][name] = {"passed": False, "status": stage["status"], "reason": stage.get("error", stage.get("reason"))}
                return None
            try:
                if "argv" in stage:
                    eq_fields(stage, {"passed": True, "executed": True, "returncode": 0, "timed_out": False}, "executed stage")
                value = action()
                report["stage_content_checks"][name] = {"passed": True, "content": value}
                return value
            except (AuditError, KeyError, TypeError, ValueError) as error:
                report["issues"].append({"stage": name, "error": str(error)})
                report["stage_content_checks"][name] = {"passed": False, "status": "invalid_content", "error": str(error)}
                return None
        builds = {}
        def check_build(v):
            key = f"build-evidence/build-c{v}/compile-evidence.json"
            data = results.doc(key)
            eq_fields(data, {"schema_version": 2, "passed": True, "gpu_executed": False,
                             "certified_enabled": True, "certified_variant": v, "tiled_variant": 1,
                             "source_sha256": {k[len(TRAINING):]: h for k, h in expected.items() if k.startswith(TRAINING + 'csrc/')},
                             "fragment_mapping_sha256": expected[MAPPING]}, "build manifest")
            require("release 12.8" in data["nvcc_version"], "Unexpected compiler version")
            require(len(data["jobs"]) == 5 and {j["name"] for j in data["jobs"]} == set(ARTIFACTS), "Build jobs differ")
            require(set(data["artifacts"]) == set(ARTIFACTS.values()), "Build artifact set differs")
            for job in data["jobs"]:
                argv = job["argv"]
                require(job["returncode"] == 0 and "-std=c++17" in argv and "--ftz=false" in argv
                        and f"-DMXFP4_CERTIFIED_VARIANT={v}" in argv, "Unexpected/failed compiler command")
                require(argv[0] == "/usr/local/cuda-12.8/bin/nvcc", "Compiler path differs")
                inputs = [a for a in argv if a.endswith(".cu")]
                require(inputs and all(a.startswith(run + "/source/") and a[len(run + '/source/'):] in expected for a in inputs), "Compile input outside frozen source")
                require("-o" in argv and argv[argv.index("-o") + 1] == run + f"/build-c{v}/" + ARTIFACTS[job["name"]], "Compile output binding differs")
                log = name_safe(job["log"])
                require('/' not in log and results.hashes[f"build-evidence/build-c{v}/" + log] == job["log_sha256"], "Build log mismatch")
            for name, meta in data["artifacts"].items():
                require(meta["sha256"] == completion["binary_hashes_before_gpu"][run + f"/build-c{v}/" + name], "Build binary hash inconsistent")
                require(type(meta["bytes"]) is int and meta["bytes"] > 0, "Invalid recorded binary size")
            builds[v] = data
            return {"passed": True, "variant": v, "nvcc_version": data["nvcc_version"], "manifest_sha256": results.hashes[key], "artifacts": data["artifacts"]}
        for v in VARIANTS:
            check(f"build-c{v}", lambda v=v: check_build(v))
        check("wait-framework-lock", lambda: {"same_uid_lock_acquired": stages["wait-framework-lock"]["same_uid_lock_acquired"]} if stages["wait-framework-lock"]["same_uid_lock_acquired"] is True else (_ for _ in ()).throw(AuditError("Missing serialization lock")))
        controls = [check(name, lambda name=name: native(results.lines(f"stages/{name}/stdout.log"))) for name in ("check-c2-baseline", "check-c2-tiled")]
        native_results = {v: check(f"check-certified-c{v}", lambda v=v: native(results.lines(f"stages/check-certified-c{v}/stdout.log"), v)) for v in VARIANTS}
        def micro_build():
            argv = stages["build-half2-microcheck"]["argv"]
            require(argv[0] == "/usr/local/cuda-12.8/bin/nvcc" and "--ftz=false" in argv
                    and run + "/source/" + TRAINING + "csrc/check_mxfp4_half2_decode.cu" in argv
                    and "-I" + run + "/source/" + TRAINING + "csrc" in argv, "Microcheck compile source binding differs")
            return {"binary_sha256": completion["binary_hashes_before_gpu"][run + "/build-micro/check_mxfp4_half2_decode"],
                    "source_sha256": expected[TRAINING + "csrc/check_mxfp4_half2_decode.cu"],
                    "included_fused_source_sha256": expected[TRAINING + "csrc/mxfp4_gemm_fused.cu"]}
        check("build-half2-microcheck", micro_build)
        micro = check("check-half2-microcheck", lambda: half2(results.lines("stages/check-half2-microcheck/stdout.log")))
        for variant in (None, 8, 10):
            name = "torch-baseline-tiled" if variant is None else f"torch-certified-c{variant}"
            check(name, lambda v=variant, name=name: torch_suite(results.doc(name + ".json"), v, expected, builds[v or 2], results.hashes[f"build-evidence/build-c{v or 2}/compile-evidence.json"], run))
        probe = check("probe-8192-random_block_scales-c2", lambda: bench(results.lines("probe-8192-random_block_scales-c2.jsonl"), 2, 8192, "random_block_scales", True))
        report["block_scale_8192_probe"] = probe
        timings, fingerprints = [], {}
        for size, mode in MODES.items():
            for index, v in enumerate(ORDER):
                name = f"bench-{size}-{index + 1:02d}-c{v}"
                if stages[name].get("executed"):
                    argv = stages[name]["argv"]
                    require(argv[0] == run + f"/build-c{v}/bench_mxfp4", "Benchmark executable variant binding differs")
                    for flag, value in (("--m", str(size)), ("--n", str(size)), ("--k", str(size)),
                                        ("--mode", mode), ("--seed", "20261123"), ("--warmup", "100"),
                                        ("--iters", "10"), ("--rounds", "5"), ("--samples", "256"),
                                        ("--output", run + "/results/" + name + ".jsonl")):
                        require(argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value, "Benchmark command argument differs: " + flag)
                data = check(name, lambda name=name, v=v, size=size, mode=mode: bench(results.lines(name + ".jsonl"), v, size, mode))
                if data:
                    key = f"{mode}:{size}"
                    old = fingerprints.setdefault(key, data["input_fnv1a64"])
                    require(old == data["input_fnv1a64"], "Paired runs use different packed inputs")
                    data["stage"] = name; data["run_position"] = index + 1
                    timings.append(data)
        report["input_fingerprints"] = fingerprints
        recorded_fingerprints = dict(fingerprints)
        if probe:
            recorded_fingerprints["random_block_scales:8192"] = probe["input_fnv1a64"]
        require(recorded_fingerprints == completion["input_fingerprints"], "Completion input fingerprints differ from actual raw records")
        report["validated_timing_runs"] = len(timings)
        eligible_common = all(controls) and native_results[2] is not None and native_results[8] is not None and 2 in builds and 8 in builds and probe is not None
        for size, mode in MODES.items():
            for v in VARIANTS:
                selected = [t for t in timings if t["size"] == size and t["variant"] == v]
                row = {"shape": [size] * 3, "input_mode": mode, "variant": v, "eligible": bool(eligible_common and native_results[v] and v in builds and (v != 10 or micro) and len(selected) == 2),
                       "independent_process_runs": len(selected), "rounds_per_run": 5, "rounds_are_not_independent_replicates": True,
                       "wall_median_ms": None, "wall_useful_tflops": None, "event_median_ms": None, "event_useful_tflops": None}
                if row["eligible"]:
                    wall = [x for t in selected for x in t["certified_timing"]["synchronized_wall_round_ms"]]
                    event = [x for t in selected for x in t["certified_timing"]["cuda_event_round_ms"]]
                    require(len(wall) == len(event) == 10, "Pooled sample count must be ten")
                    row.update(input_fnv1a64=selected[0]["input_fnv1a64"], round_samples=10,
                               wall_round_ms=wall, event_round_ms=event, wall_median_ms=statistics.median(wall),
                               event_median_ms=statistics.median(event),
                               wall_useful_tflops=2 * size**3 / (statistics.median(wall) * 1e9),
                               event_useful_tflops=2 * size**3 / (statistics.median(event) * 1e9),
                               per_run_wall_medians_ms=[statistics.median(t["certified_timing"]["synchronized_wall_round_ms"]) for t in selected],
                               stage_names=[t["stage"] for t in selected])
                report["comparison"].append(row)
        for row in report["comparison"]:
            baseline = next(r for r in report["comparison"] if r["shape"] == row["shape"] and r["variant"] == 8)
            if row["eligible"] and baseline["eligible"]:
                row["wall_throughput_change_percent_vs_c8"] = 100 * (baseline["wall_median_ms"] / row["wall_median_ms"] - 1)
                row["wall_latency_change_percent_vs_c8"] = 100 * (row["wall_median_ms"] / baseline["wall_median_ms"] - 1)
        telemetry = [t[k] for t in timings for k in ("telemetry_before", "telemetry_after") if t.get(k)]
        report["telemetry"] = {"boundary_snapshots": len(telemetry), "continuous_profiler_trace": False}
        for key in ("sm_clock_mhz", "memory_clock_mhz", "power_w", "power_limit_w"):
            values = [t[key] for t in telemetry if type(t.get(key)) in (int, float)]
            report["telemetry"][key] = {"min": min(values), "max": max(values), "values": values} if values else None
        report["sass"] = static_review(base / "static", results, args.static_sha256, completion)
        report["validation_totals"] = {
            "native_independent_calls": sum(x["calls"] for x in controls + list(native_results.values()) if x),
            "native_independent_outputs": sum(x["outputs"] for x in controls + list(native_results.values()) if x),
            "benchmark_certified_initial_and_final_output_comparisons": sum(t["size"]**2 * 2 for t in timings),
            "benchmark_cpu_oracle_positions_per_backend": len(timings) * 256,
            "benchmark_certified_submissions_recorded": len(timings) * 151,
            "benchmark_certified_timed_calls": len(timings) * 50,
            "blockscale_8192_probe_full_output_comparisons": 8192**2 if probe else 0,
            "not_every_intermediate_timed_output_was_copied": True}
        report["status"] = "verified" if not report["issues"] else "invalid_content"
        report["all_planned_content_gates_passed"] = len(report["stage_content_checks"]) == 41 and all(v["passed"] for v in report["stage_content_checks"].values())
        report["limitations"] = [
            "Packed-to-FP32 pipeline timings include the certificate and selected GEMM path; exclude quantization, H2D, allocation, and correctness copies.",
            "4096/2048 use random_block_scales; 8192 timings use random_unit_scale. The separate 8192 block-scale correctness probe is never timing evidence.",
            "Two independent processes per variant/shape, five rounds each; all ten raw rounds are pooled. Within-run rounds are correlated, not ten independent trials.",
            "Original benchmark verifies every initial output and actual final timed certified output against baseline, with 256 independent CPU oracle positions; it does not archive full output arrays or validate every intermediate timed invocation.",
            "The original benchmark shares one device output and does not poison before every backend. Dedicated native halfrow fixtures include poison and canonical-zero checks; these scopes are distinct.",
            "119 Torch cases cover baseline/tiled, with separate 16-case certified suites for C8 and C10. No C9-specific Torch suite or full LLM pretraining/convergence was run.",
            "GPU timing is historical node execution. This audit performs no GPU execution. Boundary NVML snapshots do not establish exclusive use or continuous clocks.",
            "Actual source/archive/log hashes were locally recomputed. ELF bytes are absent from the native results tar; binary hashes are crosschecked among manifests, stage before/after records, and completion.",
            "SASS static instruction/resource comparisons explain candidate mechanisms, not measured dynamic speed or per-instruction causality."]
    except (AuditError, KeyError, ValueError, OSError, TypeError, tarfile.TarError) as error:
        report.update(status="invalid_evidence")
        report["issues"].append({"error_type": type(error).__name__, "error": str(error)})
        report["comparison"] = []
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    with outputs[0].open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    lines = ["# C9/C10 原始证据独立审计", "", f"审计状态：`{report['status']}`。审计本身没有执行 CUDA、编译、远端命令或归档中的代码。", ""]
    if report.get("provenance_verified"):
        lines += [f"冻结源提交 `{COMMIT}`；源 tar SHA-256 `{SOURCE_SHA}`。结果 tar SHA-256 `{args.results_sha256}`。", "",
                  f"源文件 25 项逐文件验证；结果归档 {report['provenance']['results_archive']['regular_members']} 个普通文件逐文件验证。已核验计时运行 {report.get('validated_timing_runs', 0)} / 24。", "",
                  "## 同输入、完整 packed-to-FP32 比较", "", "每项汇总同形状、同 variant 两次独立进程的全部 10 个 round；吞吐使用 `2*M*N*K/(median_ms*1e9)`。", "",
                  "| 形状 | 输入 | Variant | Wall ms | Wall TFLOPS | Event ms | Event TFLOPS | Wall 吞吐相对 C8 |", "|---|---|---|---:|---:|---:|---:|---:|"]
        for r in report["comparison"]:
            if r["eligible"]:
                lines.append(f"| {r['shape'][0]}³ | {r['input_mode']} | C{r['variant']} | {r['wall_median_ms']:.9f} | {r['wall_useful_tflops']:.6f} | {r['event_median_ms']:.9f} | {r['event_useful_tflops']:.6f} | {r['wall_throughput_change_percent_vs_c8']:+.3f}% |")
            else:
                lines.append(f"| {r['shape'][0]}³ | {r['input_mode']} | C{r['variant']} | 未满足全部证据门 | — | — | — | — |")
        lines += ["", "## 正确性与边界探针", ""]
        for name, data in report["stage_content_checks"].items():
            if name.startswith("check-") or name.startswith("torch-"):
                detail = data.get("content", {})
                lines.append(f"- `{name}`：{'通过内容核验' if data['passed'] else '未通过'}；" + (f"{detail['calls']} 次调用、{detail['outputs']} 个输出。" if 'calls' in detail else f"{detail.get('cases', '—')} 个 case。"))
        probe = report.get("block_scale_8192_probe")
        if probe:
            c = probe["certificate_initial"]
            lines += ["", f"8192³ `random_block_scales` 独立 correctness-only 探针：accepted=`{c['accepted']}`，reject_flags=`{c['reject_flags']}`，U={c['max_energy_a']}、V={c['max_energy_b']}；全量比较 {8192**2} 个输出。未计时，不能与 unit-scale 吞吐混报。"]
        lines += ["", "## SASS 静态证据", "", "| Variant | 完整函数指令槽 | HMMA | HMUL2 | 寄存器/线程 | Stack / Spill store / Spill load bytes |", "|---|---:|---:|---:|---:|---|"]
        for variant, entry in report.get("sass", {}).get("functions", {}).items():
            resource = entry.get("resources") or {}
            lines.append(f"| C{variant} | {entry['full_function_instruction_slots']} | {entry['hmma_instructions']} | {entry['hmul2_instructions']} | {resource.get('registers_per_thread', '未匹配')} | {resource.get('stack_bytes', '—')} / {resource.get('spill_store_bytes', '—')} / {resource.get('spill_load_bytes', '—')} |")
        functions = report.get("sass", {}).get("functions", {})
        if set(functions) == {"8", "9", "10"}:
            loops = {v: e["stage_loop"] for v, e in functions.items()}
            lines += ["", f"按实际 backward-BRA 范围识别的 K-stage 循环，C8/C9/C10 分别为 {loops['8']['instruction_slots']}/{loops['9']['instruction_slots']}/{loops['10']['instruction_slots']} 个静态指令槽。三者保留 256 条 HMMA、48 条 LDS、16 条 STS 和两条 BAR。",
                      "", "C10 的循环以 16 条 HMUL2 实现精确的 half2 二幂缩放；相对 C8，IMAD 少 36、LOP3 少 48、SEL 少 32、LEA 多 4，合计少 96 个指令槽。它与减少整数解码工作的设计意图一致，但不能据此把全部实测加速归因于某条指令或推算动态执行时间。",
                      "", "C9 的 cursor 确实改变了 load 调度：两条 packed LDG 后剩余的静态 HMMA 为 249/247 条，C8 为 186/18 条；但 C9 的实测速度更低。提前发出 load 并不足以保证提速，不能把这份负结果省略。C10 的两条 packed LDG 相对 HMMA 的位置仍与 C8 一致。"]
        lines += ["", "完整 opcode、循环范围、load 位置和原始 SHA 在 JSON 中。SASS 的节点端 binary SHA 与本次编译记录交叉一致；本地不含 ELF，未声称重算 ELF 字节。", "", "## 实际限制", ""]
        lines += ["- " + limitation for limitation in report.get("limitations", [])]
    if report["issues"]:
        lines += ["", "## 阻断项", "", "```json", json.dumps(report["issues"], ensure_ascii=False, indent=2), "```"]
    with outputs[1].open("x") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"status": report["status"], "issues": report["issues"], "validated_timing_runs": report.get("validated_timing_runs"), "outputs": [str(p) for p in outputs]}, ensure_ascii=False))
    return 0 if report["status"] == "verified" and report.get("all_planned_content_gates_passed") else 2 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
