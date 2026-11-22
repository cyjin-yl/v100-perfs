#!/usr/bin/env python3
"""Validate the V100 integer model, then run the proposed packing schemes."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

import numpy as np

from v100_model import (UPSTREAM_COMMIT, bits_to_float, float_to_bits,
                        ieee_once_rounded_dot, ieee_serial_dot, v100_bfma4, v100_dot)

BASE = Path(__file__).resolve().parent
UPSTREAM = BASE / "MATLAB-tensor-core"
SNAPSHOT_MANIFEST_SHA256 = "ca1ecccb9eb760606b789ab740cb4fc314955a34eefb1933469f24da4373a165"


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_upstream_provenance(directory):
    """Validate the frozen export even when the deliverable omits Git metadata.

    The manifest digest is pinned in this driver, and the manifest binds every
    unmodified model, helper and fixture to the audited upstream commit.
    No skip/ignore/force switch is provided.
    """
    embedded = directory / "SNAPSHOT.json"
    manifest_path = embedded if embedded.exists() else BASE / "upstream-snapshot" / "SNAPSHOT.json"
    if not manifest_path.is_file() or hash_file(manifest_path) != SNAPSHOT_MANIFEST_SHA256:
        raise RuntimeError("Missing or altered frozen upstream manifest; provenance verification failed")
    manifest = json.loads(manifest_path.read_text())
    if manifest["commit"] != UPSTREAM_COMMIT:
        raise RuntimeError("Snapshot commit does not match the audited model commit")
    if manifest["repository"] != "https://github.com/north-numerical-computing/MATLAB-tensor-core":
        raise RuntimeError("Unexpected upstream repository")
    for relative, expected in manifest["files"].items():
        path = directory / relative
        if not path.is_file() or path.stat().st_size != expected["bytes"] or hash_file(path) != expected["sha256"]:
            raise RuntimeError(f"Upstream snapshot hash/size mismatch: {relative}")
    has_git = (directory / ".git").exists()
    if has_git:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip()
        if commit != UPSTREAM_COMMIT:
            raise RuntimeError("Git checkout does not match the audited commit")
    return {"commit": manifest["commit"],
            "verification_mode": "git_commit_and_frozen_file_hashes" if has_git else "snapshot_frozen_file_hashes",
            "manifest_sha256": SNAPSHOT_MANIFEST_SHA256,
            "verified_files": len(manifest["files"]),
            "git_metadata_present": has_git}


def data_coverage(values, minimum_normal):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    nonzero = x[x != 0]
    return {
        "values": len(x), "finite": int(np.count_nonzero(np.isfinite(x))),
        "zeros": int(np.count_nonzero(x == 0)),
        "subnormals_in_declared_format": int(np.count_nonzero((np.abs(x) < minimum_normal) & (x != 0))),
        "minimum_nonzero_magnitude": float(np.min(np.abs(nonzero))) if len(nonzero) else None,
        "maximum_magnitude": float(np.max(np.abs(x))),
        "minimum_unbiased_exponent": min(math.frexp(abs(float(v)))[1]-1 for v in nonzero) if len(nonzero) else None,
        "maximum_unbiased_exponent": max(math.frexp(abs(float(v)))[1]-1 for v in nonzero) if len(nonzero) else None,
    }


def fixture_validation():
    directory = UPSTREAM / "model_validation/V100/fp16"
    files = {key: directory / name for key, name in {
        "a": "a_V100_fp16.txt", "b": "b_V100_fp16.txt",
        "c": "c_V100_fp32.txt", "d": "d_V100_fp32.txt"}.items()}
    a = [[bits_to_float(int(t, 16)) for t in row.split()] for row in files["a"].read_text().splitlines()]
    b = [[bits_to_float(int(t, 16)) for t in row.split()] for row in files["b"].read_text().splitlines()]
    c = [bits_to_float(int(row, 2)) for row in files["c"].read_text().splitlines()]
    expected_bits = [int(row, 2) for row in files["d"].read_text().splitlines()]
    assert len(a) == len(b) == len(c) == len(expected_bits) == 5000
    assert all(len(x) == 4 for x in a + b)
    errors = {"ported_v100": 0, "ieee_serial_fp32": 0, "once_rounded_reference": 0}
    first_difference = None
    with (BASE / "fixture_comparison.csv").open("w", newline="") as out:
        w = csv.writer(out)
        w.writerow(["index", "upstream_v100_hex", "ported_v100_hex", "ieee_serial_hex", "once_rounded_hex"])
        for i, (av, bv, cv, expected) in enumerate(zip(a, b, c, expected_bits)):
            model = v100_bfma4(av, bv, cv)
            ordinary = ieee_serial_dot(av, bv, cv)
            rounded = ieee_once_rounded_dot(av, bv, cv)
            mb, ob, rb = (float_to_bits(x) for x in (model, ordinary, rounded))
            errors["ported_v100"] += mb != expected
            errors["ieee_serial_fp32"] += ob != expected
            errors["once_rounded_reference"] += rb != expected
            w.writerow([i, *(f"{x:08x}" for x in (expected, mb, ob, rb))])
            if first_difference is None and ob != expected:
                first_difference = {"index": i, "a": av, "b": bv, "c": cv,
                                    "upstream_expected": bits_to_float(expected),
                                    "v100_model": model, "ieee_serial": ordinary,
                                    "v100_trace": v100_bfma4(av, bv, cv, trace=True)[1]}
    result = {
        "fixture_count": len(a), "comparison": "Bitwise equality of binary32 results",
        "mismatches": errors, "coverage": {
            "a_fp16": data_coverage(a, 2**-14), "b_fp16": data_coverage(b, 2**-14),
            "c_fp32": data_coverage(c, 2**-126),
            "d_fp32": data_coverage([bits_to_float(v) for v in expected_bits], 2**-126),
        },
        "sha256": {str(path.relative_to(UPSTREAM)): hash_file(path) for path in files.values()},
        "all_a_b_exactly_fp16": bool(np.all(np.asarray(a, dtype=np.float16).astype(np.float64) == a)
                                    and np.all(np.asarray(b, dtype=np.float16).astype(np.float64) == b)),
        "first_ieee_difference": first_difference,
        "scope": "5000 published hardware-output fixtures. This is not an exhaustive validation of the V100 input space and not new hardware measurements.",
    }
    assert errors["ported_v100"] == 0
    return result


def case_record(label, a, b, c=0.0):
    a, b = [float(v) for v in a], [float(v) for v in b]
    model = v100_dot(a, b, c)
    return {"label": label, "a": a, "b": b, "c": float(c), "model": model,
            "expected_hex": f"{float_to_bits(model):08x}",
            "ieee_serial": ieee_serial_dot(a, b, c),
            "once_rounded": ieee_once_rounded_dot(a, b, c)}


def simulate_scheme(name, domain, k, radix, two_sided, rng, n_random, original_cases):
    n_inputs = 4 if two_sided else 3
    random = rng.choice(domain, size=(n_random, n_inputs, k))
    constants = np.array([[np.repeat(v, k) for v in item]
                          for item in itertools.product(domain, repeat=n_inputs)], dtype=np.int64)
    # Structured stress: sign-alternating, reversed, and cancellation patterns.
    structured = []
    for constants_item in itertools.product((min(domain), 0, max(domain)), repeat=n_inputs):
        structured.append([np.array([v if j % 2 == lane % 2 else -v for j in range(k)])
                           for lane, v in enumerate(constants_item)])
    cases = np.concatenate((random, constants, np.asarray(structured)), axis=0)
    errors = packed_mismatch = baseline_mismatch = lost_alignment = 0
    max_packed = max_accumulator = 0
    for index, vectors in enumerate(cases):
        if two_sided:
            a0, a1, b0, b1 = vectors
            p, q = a0 + radix * a1, b0 + radix * b1
            target = [int(a0 @ b0), int(a1 @ b1)]
            exact_z = int(p @ q)
        else:
            a0, a1, b0 = vectors
            p, q = a0 + radix * a1, b0
            target = [int(a0 @ b0), int(a1 @ b0)]
            exact_z = int(p @ q)
        p = p.astype(np.float16).astype(np.float64).tolist()
        q = q.astype(np.float16).astype(np.float64).tolist()
        z = 0.0
        for start in range(0, k, 4):
            z, tr = v100_bfma4(p[start:start+4], q[start:start+4], z, trace=True)
            lost_alignment += int(any(tr.get("discarded_magnitude_bits", [])))
            max_accumulator = max(max_accumulator, abs(z))
        max_packed = max(max_packed, max(abs(x) for x in p), max(abs(x) for x in q))
        if two_sided:
            high = round(z / (radix * radix))
            residual = z - (radix * radix) * high
            cross = round(residual / radix)
            low = round(residual - radix * cross)
            decoded = [low, high]
        else:
            high = round(z / radix)
            decoded = [round(z - radix * high), high]
        errors += decoded != target
        packed_mismatch += z != exact_z
        baseline_mismatch += z != ieee_serial_dot(p, q)
        if index < 64 or (n_random <= index < n_random + min(len(constants), 81)):
            original_cases.append(case_record(f"{name}_{index}", p, q))
    result = {"name": name, "domain_q": list(domain), "k": k, "radix": radix,
              "independent_two_sided": two_sided,
              "random_cases": n_random, "constant_extreme_cases": len(constants),
              "alternating_cases": len(structured), "total_cases": len(cases),
              "v100_decode_failures": errors, "packed_value_vs_exact_integer_mismatches": packed_mismatch,
              "packed_value_vs_ieee_serial_mismatches": baseline_mismatch,
              "bfma_groups_that_discarded_alignment_bits": lost_alignment,
              "observed_max_abs_packed_operand": max_packed,
              "observed_max_abs_accumulator": max_accumulator}
    assert errors == packed_mismatch == lost_alignment == 0
    return result


def scalar_full_e2m1(original_cases):
    values = [-12, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 12]
    table = {}
    ambiguous = set()
    cases = exact_mismatch = 0
    for a0, a1, b in itertools.product(values, repeat=3):
        p = float(np.float16(a0 + 128*a1))
        z = v100_bfma4([p,0,0,0], [b,0,0,0])
        target = (a0*b, a1*b)
        if z in table and table[z] != target:
            ambiguous.add(z)
        table[z] = target
        exact_mismatch += z != (a0+128*a1)*b
        cases += 1
    collision_a = case_record("full_e2m1_collision_a", [8,0,0,0], [8,1,0,0])
    collision_b = case_record("full_e2m1_collision_b", [-8,128,0,0], [8,1,0,0])
    collision_a["wanted_decoded_q_dots"] = [64,0]
    collision_b["wanted_decoded_q_dots"] = [-64,1]
    original_cases.extend((collision_a, collision_b))
    assert not ambiguous and exact_mismatch == 0
    assert collision_a["model"] == collision_b["model"] == 64
    return {"scalar_cases": cases, "distinct_encoded_products": len(table),
            "ambiguous_scalar_keys": len(ambiguous), "scalar_exact_product_mismatches": exact_mismatch,
            "k2_collision_confirmed_in_v100_model": True, "collision_cases": [collision_a, collision_b]}


def semantic_controls(original_cases):
    cases = [
        case_record("two_small_products_lost_at_global_alignment", [2**-12,2**-12,1,0], [2**-12,2**-12,1,0]),
        case_record("cancellation_cannot_recover_lost_alignment_bits", [1,1,2**-12,0], [1,-1,2**-12,0]),
        case_record("paper_v100_denormalized_product_1p5_times_1p5", [1.5,2**-11,2**-11,0], [1.5,2**-12,2**-12,0]),
        case_record("paper_v100_normalized_product_1_times_2p25", [1,2**-11,2**-11,0], [2.25,2**-12,2**-12,0]),
        case_record("paper_nonmonotonic_lower_c", [2**-12]*4, [2**-12]*4, 1-2**-24),
        case_record("paper_nonmonotonic_higher_c", [2**-12]*4, [2**-12]*4, 1),
        case_record("fp16_min_subnormal_input_product", [2**-24,0,0,0], [2**-24,0,0,0]),
        case_record("fp32_subnormal_accumulator_preserved", [0,0,0,0], [0,0,0,0], 2**-149),
        case_record("fp16_max_finite_products", [65504]*4, [65504]*4),
        case_record("fp16_max_finite_products_cancel", [65504]*4, [65504,-65504,65504,-65504]),
        case_record("fp32_max_finite_accumulator", [65504]*4, [65504]*4, bits_to_float(0x7f7fffff)),
        case_record("unsafe_ternary_radix1024_k16", [1025]*3+[1024]*13, [1025]*3+[1024]*13),
    ]
    for case in cases:
        if len(case["a"]) <= 4:
            case["alignment_trace"] = v100_bfma4(case["a"],case["b"],case["c"],trace=True)[1]
    assert cases[0]["model"] == 1 and cases[0]["ieee_serial"] == 1+2**-23
    assert cases[1]["model"] == 0 and cases[1]["ieee_serial"] == 2**-24
    assert cases[2]["model"] == 2.25+2**-22 and cases[3]["model"] == 2.25
    assert cases[4]["model"] > cases[5]["model"]
    cases[-1]["exact_integer_packed_sum"] = sum(int(x)*int(y) for x,y in zip(cases[-1]["a"],cases[-1]["b"]))
    cases[-1]["wanted_q_dots"] = [3,16]
    def unpack(z):
        h=round(z/(1024*1024)); rem=z-h*1024*1024; cross=round(rem/1024)
        return [round(rem-cross*1024),h]
    cases[-1]["v100_decoded"] = unpack(cases[-1]["model"])
    cases[-1]["ieee_decoded"] = unpack(cases[-1]["ieee_serial"])
    narrow_unsafe = case_record("unsafe_independent_q3_radix512_k8", [1537]+[1536]*7, [1537]+[1536]*7)
    narrow_unsafe["exact_integer_packed_sum"] = 18877441
    narrow_unsafe["wanted_q_dots"] = [1,72]
    h = round(narrow_unsafe["model"] / (512*512))
    rem = narrow_unsafe["model"] - h*512*512
    cross = round(rem/512)
    narrow_unsafe["v100_decoded"] = [round(rem-cross*512), h]
    c = 0.0
    traces = []
    for start in (0,4):
        c, tr = v100_bfma4(narrow_unsafe["a"][start:start+4],narrow_unsafe["b"][start:start+4],c,trace=True)
        traces.append({"block_start":start,"result":c,"trace":tr})
    narrow_unsafe["group4_trace"] = traces
    assert narrow_unsafe["v100_decoded"] == [0,72]
    cases.append(narrow_unsafe)
    original_cases.extend(cases)
    return cases


def main():
    global UPSTREAM
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-cases", type=int, default=4096)
    default_upstream = BASE / "MATLAB-tensor-core"
    if not default_upstream.is_dir():
        default_upstream = BASE / "upstream-snapshot"
    parser.add_argument("--upstream", type=Path, default=default_upstream,
                        help="Git checkout or the bundled hash-verified minimal upstream snapshot")
    args = parser.parse_args()
    UPSTREAM = args.upstream.resolve()
    start = time.perf_counter()
    provenance = verify_upstream_provenance(UPSTREAM)
    commit = provenance["commit"]
    original_cases = []
    rng = np.random.default_rng(20261122)
    result = {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "execution": "Python integer translation of the authors' V100 numerical model, validated against published hardware fixtures",
            "actual_new_gpu_measurement": False, "cycle_or_throughput_simulation": False,
            "original_matlab_octave_code_executed_by_this_script": False,
            "upstream_repo": "https://github.com/north-numerical-computing/MATLAB-tensor-core",
            "upstream_commit": commit, "numpy": np.__version__, "python": platform.python_version(),
            "upstream_provenance": provenance,
            "model_source_sha256": hash_file(BASE / "v100_model.py"),
            "driver_source_sha256": hash_file(__file__),
            "upstream_source_sha256": {path:hash_file(UPSTREAM/path) for path in (
                "models/V100TC.m", "models/tools/Generic_BFMA_TC.m", "models/tools/GEMM.m")},
            "command": shlex.join(["python3", *sys.argv]),
            "argv": sys.argv,
            "seed": 20261122,
        },
        "model_features": {"products_per_bfma":4,"extra_alignment_bits":0,"sticky_bits":0,
                           "normalize_individual_products":False,"global_alignment":True,
                           "include_c_in_global_alignment":True,"final_rounding":"toward_zero",
                           "input_subnormals_preserved":True,"output_subnormals_preserved":True},
        "upstream_hardware_fixture_validation": fixture_validation(),
        "packing_schemes": [
            simulate_scheme("shared_rhs_q3_r512_k16", list(range(-3,4)),16,512,False,rng,args.random_cases,original_cases),
            simulate_scheme("independent_ternary_r128_k16",[-1,0,1],16,128,True,rng,args.random_cases,original_cases),
            simulate_scheme("independent_ternary_r256_k32",[-1,0,1],32,256,True,rng,args.random_cases,original_cases),
            simulate_scheme("independent_q3_r512_k4",list(range(-3,4)),4,512,True,rng,args.random_cases,original_cases),
            simulate_scheme("independent_q2_r256_k8",list(range(-2,3)),8,256,True,rng,args.random_cases,original_cases),
            simulate_scheme("independent_q3_r256_k4",list(range(-3,4)),4,256,True,rng,args.random_cases,original_cases),
        ],
        "full_e2m1_scalar_and_collision":scalar_full_e2m1(original_cases),
        "numerical_semantics_and_boundary_controls":semantic_controls(original_cases),
        "limitations": [
            "Port implements finite FP16 A/B and FP32 C/D; FP16-output, other architectures and arbitrary NaN payload rules are not implemented.",
            "Published 5000-case fixtures do not exhaust the input space. Coverage statistics show which exponent/subnormal regimes they cover.",
            "Additional subnormal/extreme cases compare model behavior and mathematical expectations, not new V100 hardware outputs.",
            "Recursive groups of four use the authors' GEMM driver ordering; a real WMMA/cuBLAS kernel can group larger reductions differently.",
            "No instruction execution, scheduling, registers, memory bandwidth, timing, training gradients or convergence is modeled.",
            "Safe packed integer cases have every partial sum below 2**24 and common alignment grid finer than one, so these controls show no lost alignment bits.",
        ],
    }
    result["metadata"]["elapsed_seconds"] = time.perf_counter()-start
    (BASE/"numerical_results.json").write_text(json.dumps(result,indent=2)+"\n")
    # A modest deterministic subset to run in the unchanged original MATLAB/Octave model.
    (BASE/"original_model_cases.json").write_text(json.dumps(original_cases,indent=2)+"\n")
    print(json.dumps({"fixture":result["upstream_hardware_fixture_validation"]["mismatches"],
                      "schemes":result["packing_schemes"],"original_cases":len(original_cases),
                      "elapsed_seconds":result["metadata"]["elapsed_seconds"]},indent=2))


if __name__ == "__main__":
    main()
