#!/usr/bin/env python3
"""CPU numerical search for FP16 operand packing and FP32 output decoding.

This program DOES NOT run CUDA, benchmark a GPU, or emulate the undocumented
internal arithmetic/order of Volta Tensor Cores. NumPy float16 conversions and
explicit float32 operations are a reproducible scalar numerical model.

q = 2*x converts each finite E2M1 numerical value into an integer. The two signed
zero encodings are one numerical value here. The actual dot product is q_dot/4,
before applying the two MX block scales.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform

import numpy as np

FULL = np.array([-12, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 12], dtype=np.int64)
NARROW = np.arange(-3, 4, dtype=np.int64)
TERNARY = np.array([-1, 0, 1], dtype=np.int64)


def scalar_shared_rhs(domain: np.ndarray, shift: int) -> dict:
    """All (a0, a1, b), with p=half(a0+R*a1), z=float32(p*b)."""
    radix = 1 << shift
    pairs = np.array(list(itertools.product(domain.tolist(), repeat=2)), dtype=np.int64)
    exact_packed = pairs[:, 0] + radix * pairs[:, 1]
    packed = exact_packed.astype(np.float16).astype(np.float32)
    triples = np.array(list(itertools.product(domain.tolist(), repeat=3)), dtype=np.int64)
    p = (triples[:, 0] + radix * triples[:, 1]).astype(np.float16).astype(np.float32)
    z = np.multiply(p, triples[:, 2].astype(np.float16).astype(np.float32), dtype=np.float32)
    target = triples[:, :2] * triples[:, 2:3]
    high = np.rint(z / np.float32(radix)).astype(np.int64)
    low = (z - np.float32(radix) * high.astype(np.float32)).astype(np.int64)
    decoded = np.stack((low, high), axis=1)
    bad_nearest = np.any(decoded != target, axis=1)
    mapping: dict[float, tuple[int, int]] = {}
    first_input: dict[float, tuple[int, int, int]] = {}
    ambiguous: set[float] = set()
    example = None
    for inp, encoded, wanted in zip(triples, z, target):
        key = float(encoded)
        value = (int(wanted[0]), int(wanted[1]))
        if key in mapping and mapping[key] != value:
            ambiguous.add(key)
            if example is None:
                example = {
                    "encoded_product": key,
                    "first_input_a0_a1_b": list(first_input[key]),
                    "first_outputs": list(mapping[key]),
                    "second_input_a0_a1_b": inp.tolist(),
                    "second_outputs": list(value),
                }
        else:
            mapping.setdefault(key, value)
            first_input.setdefault(key, tuple(int(x) for x in inp))
    return {
        "shift": shift,
        "radix": radix,
        "operand_pairs": len(pairs),
        "operand_pairs_rounded_by_fp16": int(np.count_nonzero(packed != exact_packed)),
        "distinct_packed_operands": len(set(float(x) for x in packed)),
        "input_triples": len(triples),
        "distinct_encoded_products": len(mapping),
        "encoded_products_with_conflicting_targets": len(ambiguous),
        "output_only_lookup_is_unambiguous": not ambiguous,
        "nearest_radix_decoder_wrong_triples": int(np.count_nonzero(bad_nearest)),
        "nearest_radix_decoder_max_abs_error": int(np.max(np.abs(decoded - target))),
        "first_lookup_collision": example,
    }


def simulate_dot(a0: list[int], a1: list[int], b: list[int], radix: int) -> dict:
    a0n, a1n, bn = (np.asarray(x, dtype=np.int64) for x in (a0, a1, b))
    intended = a0n + radix * a1n
    packed = intended.astype(np.float16).astype(np.float32)
    products = np.multiply(packed, bn.astype(np.float16).astype(np.float32), dtype=np.float32)
    accumulator = np.float32(0)
    for p in products:
        accumulator = np.float32(accumulator + p)
    high = int(np.rint(accumulator / np.float32(radix)))
    low = int(np.float32(accumulator - np.float32(radix) * np.float32(high)))
    return {
        "a0_q": a0,
        "a1_q": a1,
        "b_q": b,
        "radix": radix,
        "fp16_packed_operands": packed.tolist(),
        "fp32_products": products.tolist(),
        "fp32_accumulator": float(accumulator),
        "target_q_dots": [int(a0n @ bn), int(a1n @ bn)],
        "nearest_radix_decoded_q_dots": [low, high],
        "packed_operand_rounding_count": int(np.count_nonzero(packed != intended)),
    }


def exhaustive_narrow_dot_bound() -> dict:
    """Exhaust a superset of all K=16 output pairs, plus an exactness argument.

    Every intermediate integer is smaller than 2**24, independent of summation
    order, so this interval proof is stronger than a finite random test.
    """
    radius, radix, k = 3, 512, 16
    bound = k * radius * radius
    low, high = np.meshgrid(np.arange(-bound, bound + 1), np.arange(-bound, bound + 1))
    low, high = low.ravel(), high.ravel()
    exact_z = low + radix * high
    z = exact_z.astype(np.float32)
    decoded_high = np.rint(z / np.float32(radix)).astype(np.int64)
    decoded_low = (z - np.float32(radix) * decoded_high.astype(np.float32)).astype(np.int64)
    bad = (decoded_low != low) | (decoded_high != high)
    return {
        "domain_q": list(range(-radius, radius + 1)),
        "domain_e2m1_values": [x / 2 for x in range(-radius, radius + 1)],
        "k": k,
        "radix": radix,
        "max_abs_packed_operand": radius * (radix + 1),
        "max_abs_individual_product": radius * radius * (radix + 1),
        "max_abs_any_partial_accumulator": bound * (radix + 1),
        "max_abs_unpacked_dot": bound,
        "half_radix_guard_bound": radix // 2,
        "superset_integer_output_pairs_tested": int(len(low)),
        "wrong_output_pairs": int(np.count_nonzero(bad)),
        "fp32_rounded_accumulators": int(np.count_nonzero(z != exact_z)),
        "proof": [
            "All packed operands are integers with magnitude <=1539, exactly representable in binary16.",
            "Every product and every possible partial sum has magnitude <=73872, exactly representable in binary32.",
            "The low dot is an integer in [-144,144], strictly inside (-256,256).",
            "Therefore nearest(z/512) recovers the high dot, and subtraction recovers the low dot.",
            "The output-pair enumeration is a superset of all actual K=16 possibilities, not an enumeration of all input sequences.",
        ],
    }


def exhaustive_two_sided_ternary_bound(k: int = 16, radix: int = 128) -> dict:
    """Two independent ternary dot products in one packed multiply-accumulate.

    (a0+R*a1)*(b0+R*b1) = L + R*C + R^2*H, where C is the cross
    coefficient. The proof checks an integer-box superset of all legal results.
    """
    low, cross, high = np.meshgrid(np.arange(-k, k + 1),
                                   np.arange(-2 * k, 2 * k + 1),
                                   np.arange(-k, k + 1), indexing="ij")
    low, cross, high = low.ravel(), cross.ravel(), high.ravel()
    exact = low + radix * cross + radix * radix * high
    z = exact.astype(np.float32)
    decoded_high = np.rint(z / np.float32(radix * radix)).astype(np.int64)
    residual = z - decoded_high.astype(np.float32) * np.float32(radix * radix)
    decoded_cross = np.rint(residual / np.float32(radix)).astype(np.int64)
    decoded_low = (residual - decoded_cross.astype(np.float32) * np.float32(radix)).astype(np.int64)
    bad = (low != decoded_low) | (cross != decoded_cross) | (high != decoded_high)
    scalar = np.array(list(itertools.product(TERNARY.tolist(), repeat=4)), dtype=np.int64)
    a0, a1, b0, b1 = scalar.T
    packed_a = (a0 + radix * a1).astype(np.float16).astype(np.float32)
    packed_b = (b0 + radix * b1).astype(np.float16).astype(np.float32)
    product = np.multiply(packed_a, packed_b, dtype=np.float32)
    intended_product = a0 * b0 + radix * (a0 * b1 + a1 * b0) + radix * radix * a1 * b1
    return {
        "domain_q": [-1, 0, 1],
        "radix": radix,
        "k": k,
        "target": "Two independent dot products sum(a0*b0) and sum(a1*b1), not just a shared RHS.",
        "packed_operands": f"A=a0+{radix}*a1, B=b0+{radix}*b1",
        "polynomial": f"z=L+{radix}*C+{radix*radix}*H, with C=sum(a0*b1+a1*b0)",
        "max_abs_packed_operand": radix + 1,
        "max_abs_any_partial_accumulator": k * (radix + 1) ** 2,
        "low_high_bound": k,
        "cross_bound": 2 * k,
        "scalar_quadruples_tested": len(scalar),
        "scalar_product_mismatches": int(np.count_nonzero(product != intended_product)),
        "superset_integer_coefficient_triples_tested": len(low),
        "wrong_coefficient_triples": int(np.count_nonzero(bad)),
        "fp32_rounded_accumulators": int(np.count_nonzero(z != exact)),
        "proof": [
            f"All operands are exactly representable binary16 integers with magnitude <={radix+1}.",
            f"Every product and every possible partial sum is an integer of magnitude <={k*(radix+1)**2}, exactly representable in binary32.",
            f"abs({radix}*C+L)<={2*k*radix+k}<{radix*radix//2} so rounding z/{radix*radix} recovers H.",
            f"After removing H, abs(L)<={k}<{radix//2} so rounding the residual/{radix} recovers C.",
            "Subtracting the cross term recovers L; C is discarded, yielding both independent dot products.",
            "This is a mathematical and CPU numerical result; actual Volta execution and performance remain to be measured.",
        ],
    }


def sampled_dot_trials(seed: int = 20261122, trials: int = 4096) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for name, domain, radix in (("ternary", TERNARY, 256), ("e2m1_abs_le_1p5", NARROW, 512), ("full_e2m1", FULL, 128)):
        for k in (1, 2, 16, 32, 64, 128, 512):
            a0, a1, b = (rng.choice(domain, size=(trials, k)) for _ in range(3))
            intended = a0 + radix * a1
            packed = intended.astype(np.float16).astype(np.float32)
            products = np.multiply(packed, b.astype(np.float16).astype(np.float32), dtype=np.float32)
            z = np.zeros(trials, dtype=np.float32)
            for i in range(k):
                z = np.add(z, products[:, i], dtype=np.float32)
            high = np.rint(z / np.float32(radix)).astype(np.int64)
            low = (z - np.float32(radix) * high.astype(np.float32)).astype(np.int64)
            want0, want1 = np.sum(a0 * b, axis=1), np.sum(a1 * b, axis=1)
            bad = (low != want0) | (high != want1)
            out.append({
                "domain": name,
                "radix": radix,
                "k": k,
                "trials": trials,
                "seed": seed,
                "wrong_dot_pairs": int(np.count_nonzero(bad)),
                "max_abs_error_either_q_dot": int(max(np.max(np.abs(low - want0)), np.max(np.abs(high - want1)))),
                "packed_operand_rounding_count": int(np.count_nonzero(packed != intended)),
                "decoder": "nearest_radix",
            })
    return out


def prime_encoding_scalar_exhaustive() -> dict:
    """Nonlinear existence proof for TWO independently multiplied scalar lanes.

    Same prime table is used on both operands. The prime product determines the
    unordered pair of input pairs. Swapping whole input pairs leaves both target
    products unchanged. This construction is NOT a dot-product homomorphism or
    a useful fast GEMM algorithm.
    """
    pairs = list(itertools.product(FULL.tolist(), repeat=2))
    primes: list[int] = []
    candidate = 2
    while len(primes) < len(pairs):
        if all(candidate % p for p in primes if p * p <= candidate):
            primes.append(candidate)
        candidate += 1
    table = {}
    collisions = 0
    rounding = 0
    for ia, a in enumerate(pairs):
        for ib, b in enumerate(pairs):
            fp32_product = np.float32(np.float32(np.float16(primes[ia])) * np.float32(np.float16(primes[ib])))
            exact_product = primes[ia] * primes[ib]
            rounding += int(fp32_product != exact_product)
            key = int(fp32_product)
            wanted = (a[0] * b[0], a[1] * b[1])
            if key in table and table[key] != wanted:
                collisions += 1
            table[key] = wanted
    return {
        "purpose": "Existence counterexample to a blanket information-theoretic impossibility claim about scalar FP16-input/FP32-output multiplication.",
        "encoding": "Map the sorted 225 numerical E2M1 pairs to the first 225 primes.",
        "input_pair_count": len(pairs),
        "max_prime": max(primes),
        "max_exact_product": max(primes) ** 2,
        "independent_four_value_combinations_tested": len(pairs) ** 2,
        "distinct_product_keys": len(table),
        "conflicting_target_products": collisions,
        "fp32_product_rounding_count": rounding,
        "caveat": "Decoding requires a large table/factorization and products cannot simply be accumulated. No performance or training benefit is claimed.",
    }


def bilateral_polynomial_scalar(domain: np.ndarray, shift: int) -> dict:
    """Compress a length-two scalar dot into the middle polynomial coefficient.

    (a0+R*a1)*(b1+R*b0) = a0*b1 + R*(a0*b0+a1*b1) + R^2*a1*b0.
    Decode the center coefficient using signed balanced digits.
    """
    r = 1 << shift
    values = np.array(list(itertools.product(domain.tolist(), repeat=4)), dtype=np.int64)
    a0, a1, b0, b1 = values.T
    p = (a0 + r * a1).astype(np.float16).astype(np.float32)
    q = (b1 + r * b0).astype(np.float16).astype(np.float32)
    z = np.multiply(p, q, dtype=np.float32)
    outer = np.rint(z / np.float32(r * r))
    middle = np.rint((z - outer * np.float32(r * r)) / np.float32(r)).astype(np.int64)
    wanted = a0 * b0 + a1 * b1
    return {
        "radix": r,
        "input_quadruples": len(values),
        "wrong_middle_coefficients": int(np.count_nonzero(middle != wanted)),
        "max_abs_middle_error": int(np.max(np.abs(middle - wanted))),
        "rounded_packed_operands": int(np.count_nonzero(p != a0 + r * a1) + np.count_nonzero(q != b1 + r * b0)),
        "scope": "Two-term scalar dot only. This test alone says nothing about accumulating several packed products.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(Path(__file__).with_name("packing_results.json")))
    args = parser.parse_args()
    a = simulate_dot([8, 0], [0, 0], [8, 1], 128)
    b = simulate_dot([-8, 0], [0, 1], [8, 1], 128)
    assert a["fp32_accumulator"] == b["fp32_accumulator"]
    assert a["target_q_dots"] != b["target_q_dots"]
    result = {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "execution_kind": "CPU NumPy scalar numerical model and exact integer reasoning",
            "gpu_executed": False,
            "volta_internal_arithmetic_emulated": False,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "format": {
            "q_equals": "2 times the E2M1 numerical value",
            "full_q_values": FULL.tolist(),
            "e2m1_values": (FULL / 2).tolist(),
            "signed_zero_handling": "positive and negative zero are one numerical value; bitwise sign-of-zero preservation is outside this experiment",
            "mxfp4_scales": "Not modeled in the scalar searches. Apply per-block scales after decoding normalized q dot products, divided by 4.",
        },
        "full_e2m1_scalar_shared_rhs_radix_sweep": [scalar_shared_rhs(FULL, s) for s in range(13)],
        "narrow_e2m1_exact_k16": exhaustive_narrow_dot_bound(),
        "two_sided_ternary_exact_k16": exhaustive_two_sided_ternary_bound(),
        "two_sided_ternary_exact_k32": exhaustive_two_sided_ternary_bound(k=32, radix=256),
        "narrow_e2m1_scalar_exhaustive": scalar_shared_rhs(NARROW, 9),
        "full_e2m1_k2_lookup_collision": {"case_a": a, "case_b": b, "zero_padding_to_k16_preserves_collision": True},
        "narrow_k32_constructed_carry_failure": simulate_dot([3] * 32, [0] * 32, [3] * 32, 512),
        "full_e2m1_radix512_operand_collision": {
            "case_a": simulate_dot([0], [12], [1], 512),
            "case_b": simulate_dot([1], [12], [1], 512),
        },
        "random_dot_model": sampled_dot_trials(),
        "bilateral_polynomial_scalar": {
            "narrow": [bilateral_polynomial_scalar(NARROW, s) for s in (4, 5, 6, 7, 8, 9)],
            "full": [bilateral_polynomial_scalar(FULL, s) for s in (4, 5, 6, 7, 8, 9)],
        },
        "prime_nonlinear_scalar_existence_proof": prime_encoding_scalar_exhaustive(),
        "not_established": [
            "Actual V100 correctness, throughput, latency or speedup",
            "Any advantage over a tuned full-size cuBLAS FP16 GEMM after packing and unpacking",
            "A general exact full-domain MXFP4 dot-product packing scheme",
            "Training convergence, quantization error or gradient behavior",
            "Impossibility of other encodings, sparse/block-specialized algorithms or approximate methods",
        ],
    }
    assert result["narrow_e2m1_exact_k16"]["wrong_output_pairs"] == 0
    assert result["two_sided_ternary_exact_k16"]["wrong_coefficient_triples"] == 0
    assert result["two_sided_ternary_exact_k16"]["scalar_product_mismatches"] == 0
    assert result["two_sided_ternary_exact_k32"]["wrong_coefficient_triples"] == 0
    assert result["narrow_e2m1_scalar_exhaustive"]["operand_pairs_rounded_by_fp16"] == 0
    assert result["narrow_e2m1_scalar_exhaustive"]["nearest_radix_decoder_wrong_triples"] == 0
    r128 = result["full_e2m1_scalar_shared_rhs_radix_sweep"][7]
    assert r128["output_only_lookup_is_unambiguous"]
    assert r128["operand_pairs_rounded_by_fp16"] == 0
    assert result["prime_nonlinear_scalar_existence_proof"]["conflicting_target_products"] == 0
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(Path(args.output).resolve()), "full_scalar_r128": r128, "narrow_k16": result["narrow_e2m1_exact_k16"], "two_sided_ternary": result["two_sided_ternary_exact_k16"], "prime": result["prime_nonlinear_scalar_existence_proof"]}, indent=2))


if __name__ == "__main__":
    main()
