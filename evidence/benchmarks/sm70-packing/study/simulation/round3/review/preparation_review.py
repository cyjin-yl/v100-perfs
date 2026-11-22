#!/usr/bin/env python3
"""Independent E2M1 preparation/scale/layout-cost checks, with no GPU execution.

The LUT is an exact encoding candidate, not a measured GPU optimization.
Physical storage is lane-major: each byte holds adjacent K positions of one
lane, so a pair-LUT key has to be assembled across two lane byte streams.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

Q = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.int64)
SPEC = "https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf/"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def pair_lut(radix: int, radius: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(256, dtype=np.int64)
    low, high = Q[indices & 15], Q[indices >> 4]
    valid = (np.abs(low) <= radius) & (np.abs(high) <= radius)
    integers = low + radix * high
    table = np.where(valid, integers, 0).astype(np.float16)
    require(np.array_equal(table[valid].astype(np.int64), integers[valid]), "A valid LUT entry rounded")
    return table, valid


def verify_preparation(radix: int, radius: int) -> dict:
    table, valid = pair_lut(radix, radius)
    # Exhaust all pairs of physical bytes: low/high nibble means even/odd K,
    # while b0/b1 means independently scaled logical lane 0/1.
    b0, b1 = np.meshgrid(np.arange(256, dtype=np.int64), np.arange(256, dtype=np.int64), indexing="ij")
    even_key = (b0 & 15) | ((b1 & 15) << 4)
    odd_key = (b0 >> 4) | ((b1 >> 4) << 4)
    accepted = valid[even_key] & valid[odd_key]
    expected_even = Q[b0 & 15] + radix * Q[b1 & 15]
    expected_odd = Q[b0 >> 4] + radix * Q[b1 >> 4]
    even_errors = np.count_nonzero(table[even_key][accepted].astype(np.int64) != expected_even[accepted])
    odd_errors = np.count_nonzero(table[odd_key][accepted].astype(np.int64) != expected_odd[accepted])
    require(even_errors == odd_errors == 0, "Cross-lane nibble packing failed")
    raw = table.astype("<f2").tobytes()
    return {
        "radix": radix, "q_abs_limit": radius, "pair_table_entries": 256,
        "valid_pair_codes_including_signed_zeros": int(valid.sum()),
        "physical_byte_pairs_checked": int(b0.size), "accepted_physical_byte_pairs": int(accepted.sum()),
        "even_output_errors": int(even_errors), "odd_output_errors": int(odd_errors),
        "table_bytes": len(raw), "validity_mask_bytes_if_bitpacked": 32,
        "little_endian_half_table_sha256": hashlib.sha256(raw).hexdigest(),
        "half_bits_hex": [f"{int(x):04x}" for x in table.view(np.uint16)],
        "valid_pair_mask_hex": np.packbits(valid, bitorder="little").tobytes().hex(),
        "scope": "Exact CPU encoding lookup; no constant-memory throughput or GPU performance claim",
    }


def verify_half_scale_domain() -> dict:
    powers = np.arange(-127, 128)
    exact = np.ldexp(Q.astype(np.float64)[None, :] / 2, powers[:, None])
    with np.errstate(over="ignore", under="ignore"):
        half = exact.astype(np.float16).astype(np.float64)
    safe = np.all(half == exact, axis=1)
    found = powers[safe].tolist()
    require(found == list(range(-23, 14)), "Unexpected all-E2M1 half-dequantization domain")
    samples = []
    for exponent in [-24, -23, -2, 2, 13, 14]:
        index = exponent + 127
        samples.append({"scale_exponent": exponent,
                        "inexact_element_codes": np.flatnonzero(half[index] != exact[index]).tolist(),
                        "all_codes_exact_finite": bool(safe[index])})
    return {"finite_E8M0_exponents_checked": len(powers), "element_codes_per_exponent": 16,
            "all_E2M1_values_convert_exactly_to_half_exponents": found,
            "safe_interval_inclusive": [-23, 13], "boundary_checks": samples,
            "note": "This concerns scaled individual FP16 inputs only; it does not prove dot-product/output range or training accuracy. E8M0 NaN is excluded."}


def middle_scale_collision() -> dict:
    radix = 512
    w0, w1 = 2, 4
    records = []
    for a0, a1, b0, b1 in [(1, 0, 1, 0), (0, 1, 0, 1)]:
        p, q = a0 + radix * a1, b1 + radix * b0
        low_dot, high_dot = a0 * b0, a1 * b1
        records.append({"inputs_a0_a1_b0_b1": [a0, a1, b0, b1], "packed_product": p * q,
                        "unweighted_middle": low_dot + high_dot,
                        "scaled_dot": (w0 * low_dot + w1 * high_dot) / 4})
    require(records[0]["packed_product"] == records[1]["packed_product"] == radix,
            "Expected equal packed products")
    require(records[0]["scaled_dot"] != records[1]["scaled_dot"], "Expected distinct scaled outputs")
    return {"radix": radix, "block_scale_products": [w0, w1], "cases": records,
            "collision_confirmed": True,
            "consequence": "Reverse-B middle extraction can combine two K slices directly only when the required scale products agree, or when additional information is retained. Independent-output extraction can apply distinct scales after decoding."}


def byte_models() -> list[dict]:
    records = []
    for m, n, k in [(16, 16, 32), (64, 64, 32), (128, 128, 256), (512, 512, 32), (512, 512, 4096)]:
        f = (m + n) * k  # A+B scalar count for ONE of the two independent GEMMs.
        scales = 2 * (m + n) * ((k + 31) // 32)
        output = 8 * m * n
        baseline_staged = 9 * f + scales + output
        candidate_staged = 5 * f + scales + output
        fused_codes = f + scales + output
        records.append({
            "M": m, "N": n, "K": k, "two_lane_input_element_count": 2 * f,
            "compressed_E2M1_codes_bytes": f, "E8M0_scale_bytes": scales, "two_FP32_outputs_bytes": output,
            "baseline_dequant_half_temporary_bytes": 4 * f, "candidate_packed_half_temporary_bytes": 2 * f,
            "baseline_staged_min_traffic_bytes": baseline_staged,
            "candidate_staged_min_traffic_bytes": candidate_staged,
            "hypothetical_fused_compressed_baseline_min_traffic_bytes": fused_codes,
            "staged_baseline_over_staged_candidate_byte_ratio": baseline_staged / candidate_staged,
            "fused_baseline_over_staged_candidate_byte_ratio": fused_codes / candidate_staged,
            "candidate_source_compression_bits_per_original_element": 8,
            "caveat": "Minimum whole-matrix traffic, one read of each input, both temporaries written/read once, scales once, outputs once, no alpha/beta read. Tile reloads/cache effects/scaling compute/quantizer/training excluded. Ratios are bytes, not measured speedups.",
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("preparation-review-results.json"))
    args = parser.parse_args()
    result = {
        "gpu_executed": False, "timing_measured": False, "source_format_spec": SPEC,
        "q_from_E2M1_code": Q.tolist(), "q_definition": "q=2*x; the sign of zero is intentionally merged for numeric dots",
        "pair_luts": {"A_ternary_R512": verify_preparation(512, 1), "B_full_E2M1_R128": verify_preparation(128, 12)},
        "scaled_half_baseline_domain": verify_half_scale_domain(),
        "cross_scale_middle_counterexample": middle_scale_collision(),
        "traffic_models": byte_models(),
        "all_checks_passed": True,
        "interpretation": [
            "The new asymmetric packing can use a 512-byte exact pair LUT per operand plus an optional validity bitmap.",
            "A lane-major byte is not itself the pair-LUT key; even/odd K positions must be joined across the two input lanes.",
            "A constant-memory LUT may serialize distinct warp addresses; no throughput benefit was measured.",
            "The staged candidate reduces the size of FP16 temporaries relative to staged dequantization, but expands the original compressed 4-bit input to 8 bits per original scalar.",
            "A fused compressed-input FP16 baseline is a stronger memory baseline; the staged byte ratio is not a universal acceleration claim.",
            "Two independent dot outputs and two K slices of one dot are different targets; middle-only folding must respect per-block scale products.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"all_checks_passed": True, "physical_byte_pairs_checked": 2 * 65536,
                      "safe_all_E2M1_half_scale_exponents": [-23, 13],
                      "middle_scale_collision_confirmed": True, "gpu_executed": False}, indent=2))


if __name__ == "__main__":
    main()
