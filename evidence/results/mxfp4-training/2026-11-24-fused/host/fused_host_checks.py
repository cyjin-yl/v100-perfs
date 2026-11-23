#!/usr/bin/env python3
"""Persist the previously in-session fused helper/layout checks; no GPU use.

The integer helper model mirrors the frozen CUDA source. Its independent
reference converts mathematical signed q * 2**delta to NumPy binary16.
Fragment formulas mirror the recorded fragment_mapping.hpp. Coverage/bank
properties here concern those formulas, not execution of a CUDA binary.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[5]
CUDA = REPO / "tools/mxfp4_training/csrc/mxfp4_gemm_fused.cu"
HEADER = CUDA.with_suffix(".h")
MAPPING = REPO / "evidence/benchmarks/sm70-packing/study/simulation/round3/cuda/fragment_mapping.hpp"
EXPECTED = {
    str(CUDA.relative_to(REPO)): "e99b3b4444b8ebdbb6df1f25358b358a88c038fc2d211d1d86d3b029ea0f54b6",
    str(HEADER.relative_to(REPO)): "65230a2b2329e096b2112ccf0d44de118c81f767edd4d9836721149003714b69",
}


def q_half_pair(packed):
    selector = ((packed & 7) << 4) | ((packed & 0x70) << 8)
    source_bytes = np.array([0x00, 0x3c, 0x40, 0x42, 0x44, 0x46, 0x48, 0x4a], dtype=np.uint32)
    magnitude = np.zeros_like(packed)
    for output_byte in range(4):
        magnitude |= source_bytes[(selector >> (4 * output_byte)) & 7] << (8 * output_byte)
    return magnitude | ((packed & 8) << 12) | ((packed & 0x80) << 24)


def check_normalization():
    packed = np.arange(65536, dtype=np.uint32)
    codes = np.stack([(packed >> (4 * index)) & 15 for index in range(4)], axis=1)
    q = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float64)
    lo_mask = np.where(packed & 7, 0xffff, 0).astype(np.uint32)
    lo_mask |= np.where(packed & 0x70, 0xffff0000, 0).astype(np.uint32)
    hi_mask = np.where(packed & 0x700, 0xffff, 0).astype(np.uint32)
    hi_mask |= np.where(packed & 0x7000, 0xffff0000, 0).astype(np.uint32)
    cases = 0
    by_delta = []
    for delta in range(13):
        shift = np.uint32(delta << 10)
        both = shift | (shift << np.uint32(16))
        lo = (q_half_pair(packed & 255) + both) & lo_mask
        hi = (q_half_pair(packed >> 8) + both) & hi_mask
        actual = np.stack([lo & 65535, lo >> 16, hi & 65535, hi >> 16], axis=1).astype(np.uint16)
        expected = (q[codes] * (2.0 ** delta)).astype(np.float16).view(np.uint16)
        mismatch = int(np.count_nonzero(actual != expected))
        assert mismatch == 0, (delta, mismatch)
        assert np.all(actual[(codes & 7) == 0] == 0), "both zero encodings must become positive zero"
        cases += actual.size
        by_delta.append({"delta": delta, "half_bit_comparisons": int(actual.size), "mismatches": mismatch})
    assert cases == 3407872
    return {"packed_four_nibble_inputs": 65536, "deltas": [0, 12],
            "half_bit_comparisons": cases, "mismatches": 0,
            "canonical_positive_zero_checked": True, "by_delta": by_delta}


def quadrant_row(lane):
    return (((lane >> 2) & 3) >> 1) * 8


def quadrant_col(lane):
    return (((lane >> 2) & 3) & 1) * 8


def c_row(lane, index):
    return quadrant_row(lane) + (lane & 1) + (index & 2) + ((lane >> 4) * 4)


def c_col(lane, index):
    return quadrant_col(lane) + (index & 4) + (lane & 2) + (index & 1)


def check_layouts():
    variants = [(0, 64, 64, 64, 128, 32, 32),
                (1, 128, 128, 32, 256, 32, 64),
                (2, 128, 128, 64, 256, 32, 64)]
    results = []
    for variant, tm, tn, bk, threads, wm, wn in variants:
        writes = Counter()
        pairs = 0
        for tid in range(threads):
            lane, warp = tid & 31, tid >> 5
            warp_row, warp_col = (warp // (tn // wn)) * wm, (warp % (tn // wn)) * wn
            for fr in range(wm // 16):
                for fc in range(wn // 16):
                    for pair in range(4):
                        index = pair * 2
                        assert c_row(lane, index) == c_row(lane, index + 1)
                        assert c_col(lane, index) % 2 == 0
                        assert c_col(lane, index + 1) == c_col(lane, index) + 1
                        for component in (index, index + 1):
                            row = warp_row + fr * 16 + c_row(lane, component)
                            col = warp_col + fc * 16 + c_col(lane, component)
                            assert 0 <= row < tm and 0 <= col < tn
                            writes[(row, col)] += 1
                        pairs += 1
        assert len(writes) == tm * tn and set(writes.values()) == {1}
        stride = bk // 4 + 1
        bank_cases = 0
        for operand in ("a", "b"):
            rows = {(quadrant_row(lane) if operand == "a" else quadrant_col(lane))
                    + (lane & 3) + ((lane >> 4) * 4) for lane in range(32)}
            assert rows == set(range(16))
            for vector in range(bk // 4):
                words = {(row * stride + vector) * 2 + component
                         for row in rows for component in (0, 1)}
                banks = Counter(word % 32 for word in words)
                assert set(banks) == set(range(32)) and set(banks.values()) == {1}
                bank_cases += 1
        shared = (tm + tn) * stride * 8 + tm + tn
        assert shared == (17536, 18688, 35072)[variant]
        results.append({"internal_variant": variant, "certified_variant": variant + 3,
                        "tile": [tm, tn, bk], "threads": threads,
                        "fp32_accumulator_elements_per_thread": wm * wn // 32,
                        "output_elements_checked": len(writes), "stores_per_output_element": 1,
                        "adjacent_same_row_even_column_pairs_checked": pairs,
                        "shared_bank_cases_checked": bank_cases,
                        "distinct_banks_per_16_unique_row_half4_loads": 32,
                        "static_shared_bytes_from_source": shared})
    return results


def main():
    source_hashes = {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (CUDA, HEADER, MAPPING)}
    assert all(source_hashes[path] == expected for path, expected in EXPECTED.items()), "frozen source changed"
    result = {"schema_version": 1, "evidence_type": "host_bit_pattern_and_layout_checks",
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "gpu_executed": False, "cuda_compilation_executed": False,
              "provenance_note": "Durable rerun of previously in-session checks; earlier raw stdout was not saved.",
              "source_sha256": source_hashes,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "numpy_version": np.__version__,
              "limitations": "Host helper models and mapping enumeration only; no GPU correctness or performance claim.",
              "normalization": check_normalization(), "layouts": check_layouts(), "passed": True}
    output = Path(__file__).with_name("fused-host-checks.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "gpu_executed": False,
                      "half_bit_comparisons": result["normalization"]["half_bit_comparisons"],
                      "mismatches": 0, "layouts_checked": len(result["layouts"]), "output": str(output)}))


if __name__ == "__main__":
    main()
