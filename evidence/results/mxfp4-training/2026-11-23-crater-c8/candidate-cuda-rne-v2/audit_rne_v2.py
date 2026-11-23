#!/usr/bin/env python3
"""Local-only independent audit of the actual initial/v2 CUDA RNE archives.

No GPU, compiler, network, archived-code import, extraction, or synthetic test.
Only SHA-pinned independent audit helpers are imported. Never overwrite reports.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import sys

HELPER_SHA = "054bd0a94b17d48426f65bcd8cfea0566d1c63c167fe342ceed48347288be8b9"
RESULT_SHA = "95f8691ff4f437d0c2c3f8e93cad326f28f98796274d1257a89cef961b93d45e"
SOURCE_SHA = "58e95bb116ba08d49e1fc0f51e3fa9fae17d442f45bbc0ea617ca163ae099ec9"
COMMIT = "f338faaa5cbdcd900aa82ae95af00ed5fb770235"
OLD_RESULT_SHA = "36682c293f811545d1ed8f67e68602c8d94a30bd1d87160db762136f6566d03f"
OLD_SOURCE_SHA = "5b730b12d6eeab8fca67799e90f8e1eac6d7d6201c6feb794b658d1865d1ef26"
OLD_COMMIT = "0ff69f8e9a9522aced2f3be887a71ba318fdb867"
ORIGINAL_QUANTIZER_SHA = "1a74501035e2b99f5ccc8335037f8d9d169416e5e6210b9ac9d193c5ca1e5910"
TRAINING = "tools/mxfp4_training/"
MAPPING = "evidence/benchmarks/sm70-packing/study/simulation/round3/cuda/fragment_mapping.hpp"
BACKENDS = ("cuda", "cuda_tiled", "cuda_certified")
ORDER = ("torch", "cuda_rne", "cuda_rne", "torch")
JOBS = ("torch_fp32_matmul", "torch_fp16_compute", "torch_fp16_casts_and_matmul",
        "mxfp4_quantize_both", "mxfp4_cuda_quantize_and_gemm",
        "mxfp4_cuda_tiled_quantize_and_gemm", "mxfp4_cuda_certified_quantize_and_gemm")
GEMM_JOBS = set(JOBS) - {"mxfp4_quantize_both"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    directory, parent = args.directory, args.directory.parent
    helper = parent / "candidate-c9-c10/audit_c9c10.py"
    if sha(helper.read_bytes()) != HELPER_SHA:
        raise RuntimeError("Independent audit helper changed")
    spec = importlib.util.spec_from_file_location("independent_native_auditor", helper)
    h = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(h)
    need, fields = h.require, h.eq_fields
    targets = [Path(str(args.output_prefix) + ext) for ext in (".json", ".md")]
    need(not any(p.exists() for p in targets), "Refusing to overwrite audit")
    report = {"schema_version": 1, "purpose": "independent actual CUDA RNE v2 evidence audit",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "pending", "issues": [],
              "auditor_sha256": sha(Path(__file__).read_bytes()), "helper_sha256": HELPER_SHA,
              "audit_gpu_executed": False, "audit_remote_executed": False, "archived_code_executed": False,
              "ELF_bytes_present_locally": False, "quantization": [], "end_to_end": []}

    def archive(root, expected, size, status, commit, source_sha):
        receipt = h.load_json((root / "archive-manifest.json").read_bytes())
        fields(receipt, {"sha256": expected, "bytes": size, "status": status,
                        "source_commit": commit, "source_archive_sha256": source_sha}, "archive receipt")
        b = h.Bundle(root / "results.tar.gz", expected, 64 << 20, "results")
        need(b.meta["bytes"] == size, "Archive actual length differs")
        c = b.doc("completion.json")
        need(b.hashes["completion.json"] == receipt["completion_sha256"], "Completion hash differs")
        need(c["result_files_sha256"] == {k: v for k, v in b.hashes.items() if k != "completion.json"},
             "Result member hash set differs")
        fields(c, {"status": status, "source_commit": commit, "source_archive_sha256": source_sha}, "completion identity")
        return b, c, receipt

    def source(root, expected, commit):
        b = h.Bundle(root / "source.tar.gz", expected, 16 << 20)
        binding = h.load_json((root / "source-binding.json").read_bytes())
        fields(binding, {"source_commit": commit, "source_sha256": expected,
                         "source_bytes": b.meta["bytes"]}, "source binding")
        need(len(binding["members"]) == len(b.files) == 29, "Frozen source member count differs")
        need({r["name"]: r["sha256"] for r in binding["members"]} == b.hashes, "Source member hashes differ")
        for row in binding["members"]:
            need(len(b.files[row["name"]]) == row["bytes"], "Source member byte count differs")
        return b, binding

    def packed(obj, shape=None):
        rows, logical = obj["logical_shape"]
        padded = (logical + 31) // 32 * 32
        need(type(rows) is int and type(logical) is int and rows > 0 and logical > 0, "Invalid packed shape")
        if shape:
            need(obj["logical_shape"] == list(shape), "Packed comparison shape differs")
        fields(obj, {"padded_K": padded, "code_bytes": rows * padded // 2,
                     "scale_bytes": rows * padded // 32, "code_byte_mismatches": 0,
                     "scale_byte_mismatches": 0}, "full packed comparison")
        for key in ("expected_codes_sha256", "expected_scales_sha256"):
            need(re.fullmatch(r"[0-9a-f]{64}", obj[key]) is not None, "Missing oracle byte hash")
        return obj["code_bytes"] + obj["scale_bytes"]

    def scan(value):
        totals = Counter()
        if isinstance(value, dict):
            if "code_byte_mismatches" in value:
                totals["byte_comparison_records"] += 1
                totals["compared_packed_bytes"] += packed(value)
            if "comparison" in value and "tolerance_failures" in value:
                h.walk_metrics(value)
                totals["numeric_comparison_records"] += 1
                totals["numeric_compared_values"] += value["values"]
                totals["exact_numeric_records" if value["comparison"].startswith("exact") else "allclose_numeric_records"] += 1
                if not value["comparison"].startswith("exact"):
                    totals["permitted_allclose_bit_mismatches"] += value["exact_mismatches"]
            for child in value.values():
                totals.update(scan(child))
        elif isinstance(value, list):
            for child in value:
                totals.update(scan(child))
        return totals

    try:
        src, binding = source(directory, SOURCE_SHA, COMMIT)
        bundle, complete, receipt = archive(directory, RESULT_SHA, 146654, "passed", COMMIT, SOURCE_SHA)
        oldroot = parent / "candidate-cuda-rne"
        oldsrc, oldbinding = source(oldroot, OLD_SOURCE_SHA, OLD_COMMIT)
        old, oldcomplete, oldreceipt = archive(oldroot, OLD_RESULT_SHA, 80304, "failed", OLD_COMMIT, OLD_SOURCE_SHA)
        fields(oldcomplete, {"passed": False, "source_and_binaries_unchanged": True,
                             "source_hashes_before": oldsrc.hashes, "source_hashes_after": oldsrc.hashes}, "initial source integrity")
        # The runner serializes a parsed copy with sorted keys. Bind the original
        # input's bytes to completion, and independently compare the copy's data.
        need(bundle.doc("source-binding.json") == binding, "Archived source publication data differs")
        fields(complete, {"passed": True, "gpu_executed": True, "source_and_binaries_unchanged": True,
                          "source_hashes_before": src.hashes, "source_hashes_after": src.hashes,
                          "source_binding_sha256": sha((directory / "source-binding.json").read_bytes()),
                          "source_archive_bytes": src.meta["bytes"], "automatic_retries": False,
                          "clock_or_power_changes": False, "services_stopped": False,
                          "packages_installed": False, "full_training_executed": False,
                          "llm_convergence_validated": False, "training_smoke_executed": True,
                          "rounds": 5, "iters": 10, "seed": 20261123, "quantizer_warmup": 10,
                          "end_to_end_warmup": 3, "end_to_end_order_per_shape": list(ORDER),
                          "rounds_are_independent_samples": False}, "v2 completion scope")
        changed = [k for k in sorted(src.files) if src.files[k] != oldsrc.files[k]]
        need(changed == [TRAINING + "check_quantize_cuda.py"], "Recovery changed more than checker")
        oldtext, newtext = (s.files[changed[0]].decode() for s in (oldsrc, src))
        old_call = "            old = ops.quantize_mx_torch(x, rounding='nearest')"
        new_call = """            # The unchanged Torch control uses x.new_tensor(), which this
            # runtime cannot dispatch through a lazy Negative key. Materialize
            # only its control input; native wrapper/direct still receive x.
            old_input = x.resolve_conj().resolve_neg() if (x.is_conj() or x.is_neg()) else x
            old = ops.quantize_mx_torch(old_input, rounding='nearest')"""
        old_fields = "            'lazy_negative': x.is_neg(), 'nondefault_stream': True, 'coverage': coverage,"
        new_fields = old_fields + """
            'torch_control_lazy_view_resolved': bool(x.is_conj() or x.is_neg()),
            'torch_control_lazy_view_resolution_reason': (
                'unchanged Torch control x.new_tensor does not support the Negative dispatch key; compare resolved logical FP32 values'
                if x.is_neg() else ('resolve logical conjugate values for the unchanged Torch control' if x.is_conj() else None)),"""
        need(oldtext.count(old_call) == oldtext.count(old_fields) == 1 and
             oldtext.replace(old_call, new_call).replace(old_fields, new_fields) == newtext,
             "Checker delta differs from reviewed control-only repair")
        torch_text = src.files[TRAINING + "torch_ops.py"].decode()
        function = next(n for n in ast.parse(torch_text).body if isinstance(n, ast.FunctionDef) and n.name == "quantize_mx_torch")
        need(sha(ast.get_source_segment(torch_text, function).encode()) == ORIGINAL_QUANTIZER_SHA,
             "Original Torch RNE/SR implementation changed")
        oldcheck = old.doc("check-quantize-cuda.json")
        fields(oldcheck, {"status": "failed", "passed": False, "completed_cases": 44,
                          "error": "DispatchKey Negative doesn't correspond to a device",
                          "all_attempted_work_synchronized": False}, "initial failure preserved")
        need(len(oldcheck["cases"]) == 45 and sum(c["passed"] is True for c in oldcheck["cases"]) == 44
             and oldcheck["cases"][-1]["name"] == "lazy_negative_input", "Initial failure location differs")
        oldcounts = Counter(s["status"] for s in oldcomplete["stages"])
        need(oldcounts == {"passed": 7, "failed": 1, "blocked": 12}, "Initial failure stage accounting differs")
        diagnostic = h.load_json((oldroot / "lazy-negative-diagnostic.json").read_bytes())
        checks = {v["name"]: v for v in diagnostic["results"]}
        need(checks["old_quantizer"]["passed"] is False and "x.new_tensor" in checks["old_quantizer"]["traceback"]
             and checks["native_wrapper"]["passed"] is True and diagnostic["all_work_synchronized"] is True,
             "Diagnostic does not support failure attribution")
        report["initial_failure"] = {"results_archive": old.meta, "source_archive": oldsrc.meta,
            "status_counts": dict(oldcounts), "successful_quantizer_cases_before_failure": 44,
            "failed_case": "lazy_negative_input", "error": oldcheck["error"],
            "original_all_attempted_work_synchronized": False,
            "diagnostic_sha256": sha((oldroot / "lazy-negative-diagnostic.json").read_bytes()),
            "recovery_changed_files": changed, "native_and_torch_ops_bytes_unchanged": True,
            "comparison_contract_unchanged": True, "old_failure_not_reclassified": True}

        run = complete["run_directory"]
        manifest_path = "build-evidence/build-c2-rne/compile-evidence.json"
        manifest, manifest_hash = bundle.doc(manifest_path), bundle.hashes[manifest_path]
        native_hashes = {k[len(TRAINING):]: v for k, v in src.hashes.items() if k.startswith(TRAINING + "csrc/")}
        fields(manifest, {"schema_version": 2, "passed": True, "gpu_executed": False, "tiled_variant": 1,
                          "certified_variant": 2, "certified_enabled": True, "quantizer_rne_enabled": True,
                          "quantizer_rne_backend": "sm70_mx32_rne_warp_fp32_v1",
                          "source_sha256": native_hashes, "fragment_mapping_sha256": src.hashes[MAPPING]}, "actual build")
        need("V12.8.93" in manifest["nvcc_version"], "Compiler version differs")
        artifacts = manifest["artifacts"]
        need(set(artifacts) == {"libmxfp4_sm70.so", "check_mxfp4", "check_mxfp4_tiled", "check_mxfp4_certified"}, "Build artifacts differ")
        binaries = {run + "/build-c2-rne/" + name: value["sha256"] for name, value in artifacts.items()}
        need(complete["binary_hashes_before_gpu"] == complete["binary_hashes_after_gpu"] == binaries, "Binary before/after bindings differ")
        need(len(manifest["jobs"]) == 4, "Incomplete compilation jobs")
        for job in manifest["jobs"]:
            need(job["returncode"] == 0 and bundle.hashes["build-evidence/build-c2-rne/" + job["log"]] == job["log_sha256"], "Build log differs")
            need(job["argv"][0] == complete["nvcc"] and job["argv"][-1] in binaries, "Compile output binding differs")
            for flag in ("--ftz=false", "-DMXFP4_TILED_VARIANT=1", "-DMXFP4_CERTIFIED_VARIANT=2", "-gencode=arch=compute_70,code=[sm_70,compute_70]"):
                need(flag in job["argv"], "Required compiler flag missing")
            for arg in job["argv"]:
                if arg.endswith(".cu"):
                    need(arg.startswith(run + "/source/") and arg[len(run + "/source/"):] in src.files, "Unbound compiled source")
        library = {"path": run + "/build-c2-rne/libmxfp4_sm70.so", **artifacts["libmxfp4_sm70.so"]}
        launch = h.load_json((directory / "launch.json").read_bytes())
        fields(launch, {"source_commit": COMMIT, "source_sha256": SOURCE_SHA,
                        "runner_sha256": complete["runner_sha256"], "previous_archive_sha256": OLD_RESULT_SHA}, "launch")
        need(bundle.hashes["quantizer-experiment-runner.py"] == complete["runner_sha256"] == receipt["runner_sha256"]
             and bundle.hashes["reviewed-native-stage-helper.py"] == complete["reviewed_helper_sha256"] == receipt["reviewed_helper_sha256"],
             "Runner/helper source binding differs")
        plan = ["build-c2-rne", "wait-prior-native-algorithm-lock", "check-c2-baseline", "check-c2-tiled",
                "check-certified-c2", "torch-baseline-tiled", "torch-certified-c2", "check-quantize-cuda",
                "smoke-cuda_tiled", "smoke-cuda_certified", "quant-latency-1024", "quant-latency-4096"]
        plan += [f"end-to-end-{size}-{i:02d}-{q}" for size in (1024, 4096) for i, q in enumerate(ORDER, 1)]
        need(complete["planned_stages"] == [s["name"] for s in complete["stages"]] == plan, "20-stage plan/order differs")
        stages = {s["name"]: s for s in complete["stages"]}
        for stage in complete["stages"]:
            fields(stage, {"status": "passed", "passed": True}, "terminal stage")
            if stage["name"].startswith("wait-"):
                fields(stage, {"executed": False, "same_uid_lock_acquired": True}, "serialization gate")
                continue
            fields(stage, {"executed": True, "returncode": 0, "timed_out": False, "cwd": run + "/source"}, "executed stage")
            argv = stage["argv"]
            if stage["name"] in ("check-c2-baseline", "check-c2-tiled", "check-certified-c2"):
                binary_name = {"check-c2-baseline": "check_mxfp4", "check-c2-tiled": "check_mxfp4_tiled",
                               "check-certified-c2": "check_mxfp4_certified"}[stage["name"]]
                need(argv == [run + "/build-c2-rne/" + binary_name, "--device", "0", "--correctness-only"],
                     "Native check executable/arguments differ")
            else:
                need(argv[:2] == [complete["python_execution_path"], "-m"], "Python stage interpreter/module invocation differs")
                label = stage["name"]
                module = ("build_cuda" if label == "build-c2-rne" else
                          "check_torch_cuda" if label == "torch-baseline-tiled" else
                          "check_torch_certified" if label == "torch-certified-c2" else
                          "check_quantize_cuda" if label == "check-quantize-cuda" else
                          "pretrain_smoke" if label.startswith("smoke-") else
                          "bench_quantize_cuda" if label.startswith("quant-latency-") else "bench_torch_cuda")
                need(argv[2] == "tools.mxfp4_training." + module, "Unexpected Python source entrypoint")
                if label == "build-c2-rne":
                    need("--with-quantizer-rne" in argv and "--with-certified" in argv, "Required opt-in build missing")
                else:
                    for flag, value in (("--device", "cuda:0"), ("--seed", "20261123"),
                                        ("--output", run + "/results/" + label + ".json")):
                        need(argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value, "Python stage flag differs: " + flag)
            need(bundle.doc("stages/" + stage["name"] + "/stage.json") == stage, "Stage record differs from completion")
            for name, value in stage["evidence_sha256"].items():
                need(bundle.hashes["stages/" + stage["name"] + "/" + name] == value, "Stage log hash mismatch")
            if stage["name"] != "build-c2-rne":
                fields(stage, {"gpu_executed": True, "gpu_stage": True}, "GPU execution record")
            for key in ("library_sha256_before", "library_sha256_after"):
                if key in stage:
                    need(stage[key] == library["sha256"], "Per-stage library binding differs")
            if "output_sha256" in stage.get("validation", {}):
                need(stage["validation"]["output_sha256"] == bundle.hashes[stage["name"] + ".json"], "Stage output content binding differs")
        need(datetime.fromisoformat(stages["quant-latency-1024"]["started_at_utc"]) >
             datetime.fromisoformat(stages["smoke-cuda_certified"]["finished_at_utc"]), "Performance preceded numerical gates")
        prior_bundles = {}
        for tag, dirname, expected in (("native", "candidate-c9-c10", "d6775a7149d64b175ff20af085e1f9788937531561f10a80edccd5efb09462b8"),
                                       ("algorithm", "c2-algorithm-screen", "b9bb395a2abaeaab11e08e44452361f931c7d710a0e16861d4305e899bfa88c6")):
            pb = h.Bundle(parent / dirname / "results.tar.gz", expected, 128 << 20, "results")
            pc = pb.doc("completion.json")
            need(pc["result_files_sha256"] == {k: v for k, v in pb.hashes.items() if k != "completion.json"}, "Prior result map differs")
            need(bundle.files[f"prior-evidence/{tag}-completion.json"] == pb.files["completion.json"]
                 and bundle.files[f"prior-evidence/{tag}-archive-manifest.json"] == (parent / dirname / "archive-manifest.json").read_bytes(),
                 "Copied prerequisite differs from actual prior archive")
            evidence = stages["wait-prior-native-algorithm-lock"][tag]
            fields(evidence, {"archive_sha256": expected, "archive_bytes": pb.meta["bytes"],
                              "completion_sha256": pb.hashes["completion.json"],
                              "receipt_sha256": bundle.hashes[f"prior-evidence/{tag}-archive-manifest.json"]}, "prior gate")
            need(datetime.fromisoformat(stages["check-c2-baseline"]["started_at_utc"]) >= datetime.fromisoformat(pc["finished_at_utc"]), "GPU stages overlap prior completed run")
            prior_bundles[tag] = pb.meta
        report["provenance"] = {"source_archive": src.meta, "results_archive": bundle.meta, "source_commit": COMMIT,
            "compile_manifest_sha256": manifest_hash, "source_member_sha256": src.hashes,
            "input_source_binding_sha256": sha((directory / "source-binding.json").read_bytes()),
            "archived_sorted_source_binding_sha256": bundle.hashes["source-binding.json"],
            "source_binding_copy_semantically_identical": True,
            "result_member_sha256": bundle.hashes, "library": library, "binary_hashes": binaries,
            "binary_binding_scope": "node-generated actual ELF hash records cross-checked; ELF bytes are absent locally",
            "launch_sha256": sha((directory / "launch.json").read_bytes()), "prior_archives": prior_bundles,
            "same_B_lock_path": stages["wait-prior-native-algorithm-lock"]["same_B_lock_path"],
            "stages": 20, "GPU_stages": 18}
        report["native_correctness"] = {name: h.native(bundle.lines("stages/" + name + "/stdout.log"),
            2 if name == "check-certified-c2" else None) for name in ("check-c2-baseline", "check-c2-tiled", "check-certified-c2")}

        def sources_for(names, all_native=False, mapping=False):
            result = {name: src.hashes[TRAINING + name] for name in names}
            if all_native:
                result.update(native_hashes)
            if mapping:
                result["fragment_mapping.hpp"] = src.hashes[MAPPING]
            return result

        def bound(data, names, all_native=False, mapping=False):
            fields(data["library"], library, "report library")
            need(data["source_sha256"] == sources_for(names, all_native, mapping), "Report source set/hash differs")
            fields(data["gpu"], {"name": "Tesla V100-SXM2-32GB", "compute_capability": [7, 0],
                                 "actual_V100": True, "multiprocessor_count": 80,
                                 "torch_cuda_build_version": "12.8"}, "report device")
            fields(data, {"python": "3.12.15", "numpy": "2.3.5", "torch": "2.10.0+cu128"}, "runtime")

        torch_results = {}
        for filename, variant, count in (("torch-baseline-tiled.json", None, 119), ("torch-certified-c2.json", 2, 16)):
            data = bundle.doc(filename)
            names = ["check_torch_cuda.py", "reference.py", "torch_ops.py"]
            names += (["check_torch_certified.py"] if variant else ["csrc/mxfp4_gemm.h", "csrc/mxfp4_gemm.cu", "csrc/mxfp4_gemm_tiled.h", "csrc/mxfp4_gemm_tiled.cu"])
            bound(data, names, variant is not None, variant is not None)
            # The earlier independent helper expects its historical build-c2
            # directory. The actual path is checked above; adapt only a copy's
            # spelling to reuse its full numerical/lifecycle checks, never raw evidence.
            adapted = {**data, "library": {**data["library"], "path": run + "/build-c2/libmxfp4_sm70.so"}}
            torch_results[filename] = h.torch_suite(adapted, variant, src.hashes, manifest, manifest_hash, run)
            need(torch_results[filename]["cases"] == count, "Torch case count differs")
        report["torch_correctness"] = torch_results
        quant = bundle.doc("check-quantize-cuda.json")
        bound(quant, ["check_quantize_cuda.py", "reference.py", "torch_ops.py", "build_cuda.py"], True)
        fields(quant, {"status": "passed", "passed": True, "gpu_executed": True, "completed_cases": 135,
                       "expected_cases": 135, "all_attempted_work_synchronized": True, "source_unchanged_during_check": True,
                       "tested_gemm_backends": list(BACKENDS), "nondefault_stream": True}, "quantizer completion")
        fields(quant["library"], {"compile_evidence_sha256": manifest_hash, "quantizer_abi_version": 1,
                                 "quantizer_rne_enabled": True, "original_torch_quantizer_source_sha256": ORIGINAL_QUANTIZER_SHA,
                                 "quantizer_backend": "sm70_mx32_rne_warp_fp32_v1", "certified_variant": 2}, "quantizer ABI")
        cases = {c["name"]: c for c in quant["cases"]}
        need(len(cases) == len(quant["cases"]) == 135 and all(c["passed"] is True for c in cases.values()), "Quantizer cases incomplete")
        finite = [c for c in cases.values() if "cuda_rne_vs_numpy" in c]
        need(len(finite) == 45, "Finite quantization fixtures incomplete")
        for c in finite:
            fields(c, {"nondefault_stream": True, "direct_initial_error_flag": 123, "finite_error_flag": 0,
                       "input_bits_unchanged": True, "direct_output_poison": "bitwise_inverse_of_every_expected_code_and_scale_byte"}, "native direct fixture")
            metrics = [c[k] for k in ("torch_vs_numpy", "cuda_rne_vs_numpy", "direct_cuda_vs_numpy")]
            for m in metrics:
                packed(m)
            need(metrics[0] == metrics[1] == metrics[2], "Byte comparisons use different expected outputs")
        fields(cases["all_253_scale_midpoint_neighbors_and_signs"]["coverage"],
               {"scale_count": 253, "scale_exponents": [-127, 125], "midpoints_per_scale": 7,
                "neighbors_per_midpoint": 3, "signs": 2, "target_boundary_values": 10626}, "tie coverage")
        fields(cases["amax_power_boundaries"]["coverage"], {"binary_exponents": [-149, 127], "powers": 277, "neighbors_per_power": 3}, "amax coverage")
        fields(cases["lazy_negative_input"], {"lazy_negative": True, "torch_control_lazy_view_resolved": True}, "lazy control repair")
        rejects = [c for n, c in cases.items() if n.startswith("reject_")]
        need(len(rejects) == 12, "Nonfinite coverage incomplete")
        for c in rejects:
            fields(c["rejected"], {"torch": True, "cuda_rne": True}, "nonfinite rejection")
            packed(c["finite_recovery"])
        linear_names = {f"linear_{shape}_{be}_{bits}" for shape in ("tail", "strided", "original_fp32_transpose")
                        for be in BACKENDS for bits in ("".join(map(str, flags)) for flags in itertools.product((0, 1), repeat=3))}
        need({n for n in cases if n.startswith("linear_")} == linear_names, "F/D/W coverage differs")
        for name in linear_names:
            c = cases[name]; recipe = c["recipe"]
            active = sum(recipe[x] for x in ("forward", "dgrad", "wgrad"))
            fields(c, {"original_fp32_axis_inputs_verified": True, "native_quantizer_calls": 2 * active,
                       "torch_quantizer_calls": 0, "nondefault_stream": True}, "F/D/W route")
            fields(recipe, {"quantizer_backend": "cuda_rne", "rounding": "nearest", "gradient_rounding": "nearest"}, "RNE recipe")
            for key, flag in (("output", "forward"), ("dgrad", "dgrad"), ("wgrad", "wgrad")):
                need(c["comparisons"][key]["comparison"].startswith("exact") == recipe[flag], "Declared surrogate comparison differs")
        sr = [c for n, c in cases.items() if n.startswith("unchanged_sr_")]
        need(len(sr) == 6 and {c["backend"] for c in sr} == set(BACKENDS), "SR coverage incomplete")
        for c in sr:
            route = ["cuda_rne" if v == "nearest" else "torch:stochastic" for v in
                     (c["rounding"], c["rounding"], c["gradient_rounding"], c["rounding"], c["gradient_rounding"], c["rounding"])]
            fields(c, {"actual_quantizer_routes": route, "same_seed": 20261123, "original_torch_quantizer_source_unchanged": True}, "SR unchanged route")
        negative = quant["transpose_negative_control"]
        need(negative["effective"] is True and negative["dgrad_mismatches_if_row_quantized_values_reused"] > 0
             and negative["wgrad_mismatches_if_row_quantized_values_reused"] > 0, "Transpose negative control ineffective")
        counts = scan(quant["cases"])
        need(counts["byte_comparison_records"] == 147 and counts["numeric_comparison_records"] == 306
             and counts["exact_numeric_records"] == 126 and counts["allclose_numeric_records"] == 180,
             "Numerical comparison accounting differs")
        report["quantizer_correctness"] = {"passed": True, "cases": 135, "finite_quantization_fixtures": 45,
            "nonfinite_rejection_fixtures": 12, "linear_recipe_fixtures": 72, "unchanged_SR_fixtures": 6,
            "metric_counts": dict(counts), "negative_transpose_control": negative,
            "unsupported_coverage_not_executed": quant["unsupported_coverage_not_executed"],
            "original_torch_quantizer_sha256": ORIGINAL_QUANTIZER_SHA}

        smoke_results = {}
        smoke_names = ("fp32", "mx_forward", "mx_forward_dgrad", "mx_forward_dgrad_wgrad", "mx_all_stochastic_grad")
        recipes = ((False, False, False), (True, False, False), (True, True, False), (True, True, True), (True, True, True))
        for be in ("cuda_tiled", "cuda_certified"):
            data = bundle.doc("smoke-" + be + ".json")
            fields(data, {"passed": True, "gpu_executed": True, "backend": be, "quantizer_backend": "cuda_rne",
                          "seed": 20261123, "learning_rate": 0.005, "optimizer": "AdamW FP32",
                          "llm_convergence_validated": False, "performance_claim": None}, "GPU smoke")
            expected = {k[len(TRAINING):]: v for k, v in src.hashes.items() if k.startswith(TRAINING)
                        and "/" not in k[len(TRAINING):] and k.endswith(".py")}
            need(data["source_sha256"] == expected, "Smoke source hashes differ")
            need([a["name"] for a in data["arms"]] == list(smoke_names), "Smoke arms differ")
            for index, (arm, flags) in enumerate(zip(data["arms"], recipes)):
                fields(arm, {"steps": 40, "passed": True, "parameters_finite": True, "master_parameters_fp32": True,
                             "optimizer_states_fp32": True, "changed_parameter_tensors": 17}, "smoke updates")
                fields(arm["recipe"], {**dict(zip(("forward", "dgrad", "wgrad"), flags)),
                       "rounding": "nearest", "gradient_rounding": "stochastic" if index == 4 else "nearest",
                       "backend": be, "quantizer_backend": "cuda_rne"}, "smoke recipe")
                need(len(arm["training_losses"]) == 40 and all(math.isfinite(x) for x in arm["training_losses"])
                     and 0 <= arm["validation_loss_after"] < arm["validation_loss_before"]
                     and math.isfinite(arm["max_abs_gradient"]) and arm["max_abs_gradient"] > 0, "Smoke finite/update content failed")
            smoke_results[be] = data["arms"]
        report["training_smoke"] = {"backends": smoke_results, "arms_per_backend": 5, "steps_per_arm": 40,
            "full_LLM_training_executed": False, "throughput_claim": None}

        clock_samples = []
        for size in (1024, 4096):
            name = f"quant-latency-{size}.json"; data = bundle.doc(name)
            bound(data, ["check_quantize_cuda.py", "bench_quantize_cuda.py", "reference.py", "torch_ops.py", "build_cuda.py"], True)
            fields(data, {"status": "passed", "passed": True, "gpu_executed": True, "all_attempted_work_synchronized": True,
                          "source_unchanged_during_benchmark": True, "shape_A": [size, size], "shape_B": [size, size],
                          "rounds_per_backend": 5, "iterations_per_round": 10, "warmup_per_backend": 10,
                          "seed": 20261123, "input_bits_unchanged": True, "packed_tensor_cache": False,
                          "same_resident_input_tensors_for_every_call": True, "flops_or_tflops": None}, "quantizer timing")
            fields(data["calls_per_backend"], {"initial": 1, "warmup": 10, "timed": 50, "quantizer_calls_per_pair": 2}, "quantizer calls")
            need(data["checker_prerequisite"]["sha256"] == bundle.hashes["check-quantize-cuda.json"], "Quantizer timing prerequisite differs")
            need(data["initial_correctness"]["torch"] == data["initial_correctness"]["cuda_rne"], "Initial expected quantized inputs differ")
            for p in data["initial_correctness"]["torch"].values():
                packed(p, (size, size))
            rows = data["raw_rounds"]
            need(len(rows) == 10, "Quantizer raw rounds missing")
            for index in range(5):
                order = ["torch", "cuda_rne"] if index % 2 == 0 else ["cuda_rne", "torch"]
                selected = rows[2 * index:2 * index + 2]
                need([x["quantizer_backend"] for x in selected] == order, "Quantizer alternating order differs")
                for position, row in enumerate(selected):
                    fields(row, {"round_index": index, "position_in_round": position, "order": order, "iterations": 10,
                                 "passed": True, "byte_checks_after_both_timing_endpoints": True}, "actual timed quantized output")
                    need(row["last_timed_outputs_vs_numpy"] == data["initial_correctness"][row["quantizer_backend"]], "Final actual timed quantized bytes differ")
                    for clock in ("wall", "event"):
                        need(row[clock + "_ms_per_pair"] > 0, "Nonpositive quantization time")
                        h.close(row[clock + "_ms_total"] / 10, row[clock + "_ms_per_pair"], "per-pair division")
            medians = {}
            for be in ("torch", "cuda_rne"):
                medians[be] = {}
                for clock in ("wall", "event"):
                    key = clock + "_ms_per_pair"; values = [r[key] for r in rows if r["quantizer_backend"] == be]
                    observed = data["summary"]["backends"][be][key]
                    need(observed["raw_round_values"] == values, "Quantizer summary omitted/changed rounds")
                    for k, wanted in (("median", statistics.median(values)), ("minimum", min(values)), ("maximum", max(values))):
                        h.close(observed[k], wanted, "quantizer " + k)
                    medians[be][key] = statistics.median(values)
            speed = medians["torch"]["wall_ms_per_pair"] / medians["cuda_rne"]["wall_ms_per_pair"]
            h.close(data["summary"]["median_wall_speedup_torch_over_cuda_rne"], speed, "quantizer wall speedup")
            h.close(data["summary"]["median_wall_latency_reduction_percent"], 100 * (1 - 1 / speed), "quantizer latency reduction")
            fields(data["summary"], {"independent_process_runs": 1, "repeated_rounds_are_independent_samples": False,
                                     "flops_or_tflops": None}, "quantizer inference scope")
            for fingerprint in ("A_fp32_bytes_sha256", "B_fp32_bytes_sha256"):
                need(re.fullmatch("[a-f0-9]{64}", data["input_fingerprint"][fingerprint]) is not None, "Missing input fingerprint")
            before = data["memory"][0]["free_bytes"]; budget = 192 * size**2 + (256 << 20)
            need(data["memory_guard"]["conservative_allocation_budget_bytes"] == budget and before >= 1 << 30
                 and budget <= before - (512 << 20) and all(m["free_bytes"] >= 512 << 20 for m in data["memory"]), "Quantizer memory guard failed")
            monitor = data["clock_monitor"]
            fields(monitor, {"available": True, "clock_or_power_changes": False, "continuous_load_measurement": False}, "clock monitor scope")
            need(len(monitor["samples"]) == 12, "Missing quantizer boundary clocks")
            clock_samples.extend(monitor["samples"])
            report["quantization"].append({"size": size, "scope": data["timing_scope"], "event_scope": data["event_scope"],
                "wall_speedup_torch_over_cuda_rne": speed, "medians": medians, "raw_rounds": rows,
                "input_fingerprint": data["input_fingerprint"], "independent_process_runs": 1,
                "clock_monitor": monitor, "raw_sha256": bundle.hashes[name]})

        for size in (1024, 4096):
            by_quantizer = {q: {job: {"wall": [], "event": []} for job in JOBS} for q in set(ORDER)}
            run_rows = []
            for index, q in enumerate(ORDER, 1):
                name = f"end-to-end-{size}-{index:02d}-{q}.json"; data = bundle.doc(name)
                bound(data, ["check_torch_cuda.py", "reference.py", "torch_ops.py", "bench_torch_cuda.py",
                    "csrc/mxfp4_gemm.h", "csrc/mxfp4_gemm.cu", "csrc/mxfp4_gemm_tiled.h", "csrc/mxfp4_gemm_tiled.cu"])
                fields(data, {"status": "passed", "passed": True, "gpu_executed": True, "all_attempted_work_synchronized": True,
                              "source_unchanged_during_benchmark": True, "quantizer_backend": q, "certified_job_enabled": True,
                              "warmup_per_job": 3, "rounds": 5, "iterations_per_round": 10,
                              "shape": {"M": size, "N": size, "logical_K": size, "padded_K": size},
                              "same_resident_inputs_reused": True, "useful_flops": 2 * size**3, "full_training_executed": False,
                              "llm_convergence_validated": False, "cpu_fallback": False, "gpu_services_stopped": False,
                              "clock_or_power_changes": False}, "complete Torch wrapper")
                fields(data["input"], {"seed": 20261123, "distribution": "synthetic_gaussian", "dtype": "torch.float32",
                                       "mean": 0.0, "stddev": 1.0, "generated": True, "real_training_tensor": False}, "wrapper input")
                need(data["prerequisite"]["sha256"] == bundle.hashes["torch-baseline-tiled.json"], "Wrapper Torch prerequisite differs")
                if q == "cuda_rne":
                    need(data["quantizer_prerequisite"]["sha256"] == bundle.hashes["check-quantize-cuda.json"], "Wrapper quantizer prerequisite differs")
                fields(data["torch_matmul_settings"], {"allow_tf32": False, "allow_fp16_reduced_precision_reduction": False,
                                                       "fp16_output_dtype": "torch.float16", "output_float32_cast_timed": False}, "Torch numerical flags")
                diagnostic = data["certified_diagnostic_untimed_initial"]
                fields(diagnostic, {"accepted": 1, "reject_flags": 0, "metadata_valid": 1, "submitted_calls": 1,
                                    "m": size, "n": size, "padded_k": size,
                                    "context_lifecycle": "per_call_create_gemm_destroy_synchronous"}, "initial certified diagnostic")
                need(diagnostic["max_energy_a"] * diagnostic["max_energy_b"] <= 2**48
                     and max(diagnostic["max_span_a"], diagnostic["max_span_b"]) <= 12, "Initial certificate bounds failed")
                validation = data["validation"]
                fields(validation, {"passed_before_timing": True, "passed_after_timing": True, "all_input_and_output_values_finite": True,
                                    "final_timed_rne_codes_and_scales_exact": True, "same_resident_fp32_inputs_for_every_call": True}, "wrapper correctness")
                for key in ("packed_cuda_vs_tiled_all_outputs", "final_timed_packed_cuda_vs_tiled_all_outputs",
                            "packed_certified_vs_baseline_all_outputs", "final_timed_certified_vs_baseline_all_outputs"):
                    fields(validation[key], {"values": size**2, "exact_mismatches": 0, "all_finite": True,
                                             "comparison": "bitwise_except_signed_zero"}, "whole-output exact comparison")
                repeats = validation["final_timed_vs_initial_all_outputs"]
                need(set(repeats) == GEMM_JOBS, "Final actual timed outputs missing")
                for job, value in repeats.items():
                    fields(value, {"values": size**2, "exact_mismatches": 0,
                                   "output_dtype": "float16" if job.startswith("torch_fp16") else "float32",
                                   "comparison": "exact_same_dtype_except_signed_zero"}, "final same-dtype output")
                need(len(data["timings"]) == 7 and {t["name"] for t in data["timings"]} == set(JOBS), "Timing jobs missing")
                for timing in data["timings"]:
                    job = timing["name"]
                    fields(timing, {"primary_clock": "synchronized_perf_counter", "function_allocations_and_validation_included": True,
                                    "timed_output_float32_conversion": False}, "wrapper timing scope")
                    if job == JOBS[-1]:
                        need(timing["scope"] == "fp32_to_mxfp4_then_certified_create_prepare_certificate_gemm_scale_destroy_sync_fp32_output", "Lifecycle costs excluded")
                    for clock, key, median_key, rate_key in (("wall", "synchronized_wall_round_ms", "synchronized_wall_median_ms", "synchronized_wall_useful_tflops"),
                                                           ("event", "cuda_event_round_ms", "cuda_event_median_ms", "cuda_event_useful_tflops")):
                        values = timing[key]
                        need(len(values) == 5 and all(math.isfinite(x) and x > 0 for x in values), "Invalid raw wrapper rounds")
                        med = statistics.median(values)
                        h.close(timing[median_key], med, "wrapper median")
                        if job == "mxfp4_quantize_both":
                            need(timing[rate_key] is None, "Quantize-only reports TFLOPS")
                        else:
                            h.close(timing[rate_key], 2 * size**3 / (med * 1e9), "wrapper TFLOPS")
                        by_quantizer[q][job][clock].extend(values)
                    h.close(timing["median_ms"], timing["synchronized_wall_median_ms"], "primary clock")
                budget = data["estimated_peak_increment_budget_bytes"]
                free = data["memory"]["after_context_and_library"]["device_free_bytes"]
                need(free >= 1 << 30 and budget <= free - (512 << 20) and budget <= free * 3 // 4, "Wrapper memory preflight failed")
                run_rows.append({"name": name, "raw_sha256": bundle.hashes[name], "quantizer_backend": q,
                                 "initial_certificate": diagnostic, "timings": data["timings"]})
            pooled = []
            for q in ("torch", "cuda_rne"):
                for job in JOBS:
                    times = by_quantizer[q][job]
                    need(len(times["wall"]) == len(times["event"]) == 10, "Pooled rounds omitted")
                    wall, event = statistics.median(times["wall"]), statistics.median(times["event"])
                    pooled.append({"quantizer_backend": q, "job": job, "wall_median_ms": wall, "event_median_ms": event,
                        "wall_useful_tflops": None if job == "mxfp4_quantize_both" else 2 * size**3 / (wall * 1e9),
                        "event_useful_tflops": None if job == "mxfp4_quantize_both" else 2 * size**3 / (event * 1e9),
                        "wall_round_ms": times["wall"], "event_round_ms": times["event"]})
            report["end_to_end"].append({"size": size, "scope": "complete Torch wrappers, not a full training step",
                "process_order": list(ORDER), "independent_processes_per_quantizer": 2, "pooled_rounds_per_job": 10,
                "repeated_rounds_are_independent_samples": False, "cross_process_input_byte_fingerprints_recorded": False,
                "cross_process_input_identity": "same seed/shape/generator/runtime/model; byte identity was not recorded",
                "per_call_synchronous_context_included": True, "raw_runs": run_rows, "pooled": pooled})
        report["quantization_boundary_NVML"] = {"samples": len(clock_samples),
            "sm_clock_mhz_min": min(s["sm_clock_mhz"] for s in clock_samples),
            "sm_clock_mhz_max": max(s["sm_clock_mhz"] for s in clock_samples),
            "memory_clock_mhz": sorted({s["memory_clock_mhz"] for s in clock_samples}),
            "power_limit_mw": sorted({s["power_limit_mw"] for s in clock_samples}),
            "continuous_trace": False, "complete_wrapper_ABBA_clock_trace_present": False}
        report["limitations"] = [
            "No local ELF bytes in these RNE archives; actual binary hashes are node records cross-bound to build and each stage.",
            "Quantizer GPU capture rejection and missing-library rejection were not executed by the 135-case checker; certified GEMM capture rejection is a separate executed test.",
            "The initial failed check remains failed and reports all_attempted_work_synchronized=false; later diagnostic synchronized its own work.",
            "Quantization latency measures two complete wrappers, with allocations, validation and finite-flag synchronization; CUDA events include host submission gaps.",
            "Quantization uses one process per shape with five correlated rounds, no confidence interval or causal clock claim.",
            "Torch complete-wrapper comparison uses two processes per quantizer with ten correlated pooled rounds; input bytes are not fingerprinted across processes.",
            "Only final actual timed outputs per complete-wrapper job are checked, not every intermediate timed call; standalone quantization checks the final output of every round.",
            "Certified initial diagnostic records one fresh context; timed calls have the same per-call synchronous lifecycle, not native reused workspace timing.",
            "Boundary NVML samples exist for standalone quantization, not a continuous full-wrapper ABBA clock trace or exclusivity proof.",
            "Synthetic smoke proves finite updates for its five recipes only; it establishes no real-corpus LLM convergence, training throughput, or peak-memory benefit.",
        ]
        report["status"] = "verified"
    except Exception as error:
        report["status"] = "failed"
        report["issues"].append({"type": type(error).__name__, "message": str(error)})
    targets[0].parent.mkdir(parents=True, exist_ok=True)
    with targets[0].open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    lines = ["# CUDA RNE v2 原始归档独立审计", "", f"审计状态：**{report['status']}**。审计本身没有运行 GPU、编译器、远端命令或归档代码。", ""]
    if report["status"] == "verified":
        lines += ["## 来源与正确性", "", f"实际 results.tar.gz 为 146,654 B，SHA `{RESULT_SHA}`；源 commit `{COMMIT}`。",
            "29 个冻结源文件与全部结果成员按实际字节重算，完整 20 个阶段的输出、日志、源码、编译与库绑定一致。RNE 归档没有 ELF；二进制 SHA 来自节点实测记录，已交叉绑定到各阶段前后记录。", "",
            "C2 native 69 次调用、405,930 输出、2,950 canonical +0 通过；baseline/tiled 各46例。原 Torch 119 项与 C2 独立16项通过。",
            "CUDA RNE 135项包括45个有限量化fixture、12个非有限拒绝/恢复、72个F/D/W recipe及6个SR路由对照。147个逐字节比较和306个数值比较逐项核验，exact记录零mismatch；FP32对照使用已声明的3e-5容差。",
            "253个尺度的midpoint邻域、FP32二幂边界、次正规数、负零、padding、非连续/零stride、int64 stride和lazy view均有实际记录。native direct输出预先填oracle逐字节反码。", "",
            "## 初始失败与唯一修复", "", "原运行7个stage通过、1个失败、12个blocked，量化先通过44例，随后旧Torch control的`x.new_tensor`不支持lazy Negative dispatch key。原失败与同步状态没有改写。",
            "v2源归档逐文件比较只有check_quantize_cuda.py不同：仅解析旧Torch control输入的lazy view并记录原因。native wrapper/direct仍接收原view；量化kernel、torch_ops、SR及比较标准未改。", "",
            "## 两输入完整量化 wrapper", "", "同步wall；每个backend预热10次、5轮×10次，轮内交替顺序。每轮最后实际timed codes/scales在两个timer结束后全量对独立NumPy oracle。", "",
            "| 输入尺寸（两矩阵） | Torch ms | CUDA RNE ms | wall速度比 |", "|---|---:|---:|---:|"]
        for row in report["quantization"]:
            lines.append(f"| {row['size']}² × 2 | {row['medians']['torch']['wall_ms_per_pair']:.9f} | {row['medians']['cuda_rne']['wall_ms_per_pair']:.9f} | {row['wall_speedup_torch_over_cuda_rne']:.6f}× |")
        lines += ["", "这是两个完整量化调用，包含分配、验证与有限值标志同步，不是kernel-only或GEMM吞吐。输入真实SHA和全部round保存在audit.json。", "",
            "## 完整 Torch 量化＋certified GEMM", "", "每个形状按Torch→CUDA RNE→CUDA RNE→Torch进行4个独立进程；每job预热3次、5轮×10次。每种量化器合并两个进程的10个round取中位数。", "",
            "| 形状 | 量化器 | 完整wall ms | useful TFLOPS |", "|---|---|---:|---:|"]
        for row in report["end_to_end"]:
            for timing in row["pooled"]:
                if timing["job"] == JOBS[-1]:
                    lines.append(f"| {row['size']}³ | {timing['quantizer_backend']} | {timing['wall_median_ms']:.9f} | {timing['wall_useful_tflops']:.9f} |")
        lines += ["", "计时包含每次context创建、workspace分配、prepare/证书/GEMM/epilogue和同步销毁。此处不能与native复用context的TFLOPS当作同一scope，也不是完整训练step。七个job的全部wall/event原始round和重算统计保留在audit.json。", "",
            "## GPU 训练 smoke", "", "cuda_tiled与cuda_certified均配CUDA RNE，各五组配方×40步；十组全部17个参数张量实际更新，参数与AdamW状态保持有限FP32，训练loss有限且验证loss下降。SR组继续使用原Torch stochastic路由。合成width-32 decoder不建立LLM训练吞吐或收敛结论。", "", "## 限制", ""]
        lines.extend("- " + value for value in report["limitations"])
    else:
        lines += ["审计失败，不发布性能结果：", "", "```json", json.dumps(report["issues"], ensure_ascii=False, indent=2), "```"]
    with targets[1].open("x") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"status": report["status"], "issues": report["issues"], "output": [str(p) for p in targets],
                      "quantization_speedups": [x["wall_speedup_torch_over_cuda_rne"] for x in report["quantization"]]}, ensure_ascii=False))
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
