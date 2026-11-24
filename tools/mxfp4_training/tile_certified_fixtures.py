"""Independent CPU fixtures and integer certificates for tile-local dispatch.

These fixtures preserve the existing ordered-MX32 arithmetic.  They are
synthetic, not sampled training tensors.  No GPU or timing is executed here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .reference import E2M1_Q, MXTensor, gemm_mx


ENERGY_BOUND = 1 << 48
ENERGY_CAP = ENERGY_BOUND + 1
TILE = 128


@dataclass
class TileFixture:
    name: str
    a: np.ndarray
    a_scales: np.ndarray
    b: np.ndarray
    b_scales: np.ndarray
    logical_k: int

    @property
    def m(self):
        return self.a.shape[0]

    @property
    def n(self):
        return self.b.shape[0]

    @property
    def padded_k(self):
        return self.a.shape[1] * 2

    def arrays(self):
        return (self.a, self.a_scales, self.b, self.b_scales)

    def hashes(self):
        return {name: hashlib.sha256(value.tobytes(order="C")).hexdigest()
                for name, value in zip(("a_codes", "a_scales", "b_codes", "b_scales"), self.arrays())}

    def oracle(self):
        if not self.logical_k:
            return np.zeros((self.m, self.n), dtype=np.float32)
        if self.m * self.n * self.padded_k > 64 * 1024**2:
            raise ValueError("Full CPU GEMM oracle is restricted to small correctness fixtures; "
                             "large benchmarks must use the validated GPU tiled oracle")
        return gemm_mx(MXTensor(self.a, self.a_scales, (self.m, self.logical_k)),
                       MXTensor(self.b, self.b_scales, (self.n, self.logical_k)))


def clear_padding(codes, logical_k):
    if logical_k % 2:
        codes[:, logical_k // 2] &= np.uint8(15)
    codes[:, (logical_k + 1) // 2:] = 0


def uniform_fixture(name="accepted", m=257, n=259, k=64, code=2):
    padded_k = (k + 31) // 32 * 32
    a = np.full((m, padded_k // 2), code | (code << 4), dtype=np.uint8)
    b = np.full((n, padded_k // 2), code | (code << 4), dtype=np.uint8)
    clear_padding(a, k)
    clear_padding(b, k)
    return TileFixture(name, a, np.full((m, padded_k // 32), 127, dtype=np.uint8),
                       b, np.full((n, padded_k // 32), 127, dtype=np.uint8), k)


def correctness_fixtures():
    result = [uniform_fixture(), uniform_fixture("accepted_full_tiles", 256, 384, 256),
              uniform_fixture("accepted_k65_padding", k=65), uniform_fixture("native_k0", k=0)]
    mixed = uniform_fixture("mixed_span_cross", m=384, n=384)
    mixed.a_scales[128, 0] = 140
    mixed.b_scales[256, 0] = 140
    result.append(mixed)
    tail = uniform_fixture("mixed_span_tail_k65", k=65)
    tail.a_scales[128, 0] = 140
    tail.b_scales[258, 0] = 140
    result.append(tail)
    energy = uniform_fixture("mixed_energy_only", code=7)
    energy.a_scales[128, 0] = 139
    result.append(energy)
    nan_zero = uniform_fixture("nan_scale_in_zero_row")
    nan_zero.a[129, :] = 0x88
    nan_zero.a_scales[129, 0] = 255
    result.append(nan_zero)
    lower = uniform_fixture("mixed_low_lattice")
    lower.a_scales[128, :] = 0
    result.append(lower)
    upper = uniform_fixture("mixed_high_lattice")
    upper.b_scales[128, :] = 254
    result.append(upper)
    all_rejected = uniform_fixture("all_rejected_span")
    all_rejected.a_scales[:, 0] = 140
    result.append(all_rejected)
    zeros = uniform_fixture("signed_zero_and_inactive_low_scale", k=96)
    zeros.a[::31, :] = 0x88
    zeros.b[::29, :] = 0x88
    zeros.a[:, :16] = 0x88
    zeros.b[:, :16] = 0x88
    zeros.a_scales[:, 0] = 0
    zeros.b_scales[:, 0] = 0
    result.append(zeros)
    return result


def benchmark_fixture(mode, m, n, k, seed):
    if mode not in ("unit", "block", "sparse_span", "sparse_energy"):
        raise ValueError("unknown tile benchmark input mode")
    if min(m, n, k) <= 0 or max(m, n, k) > 8192:
        raise ValueError("bounded positive dimensions up to 8192 are required")
    padded = (k + 31) // 32 * 32
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, size=(m, padded // 2), dtype=np.uint8)
    b = rng.integers(0, 256, size=(n, padded // 2), dtype=np.uint8)
    clear_padding(a, k)
    clear_padding(b, k)
    if mode == "unit":
        sa, sb = (np.full((rows, padded // 32), 127, dtype=np.uint8) for rows in (m, n))
    else:
        sa = rng.integers(125, 130, size=(m, padded // 32), dtype=np.uint8)
        sb = rng.integers(125, 130, size=(n, padded // 32), dtype=np.uint8)
    fixture = TileFixture(mode, a, sa, b, sb, k)
    if mode in ("sparse_span", "sparse_energy"):
        if padded < 64:
            raise ValueError("sparse rejection fixtures require at least two MX32 blocks")
        for codes, scales, row in ((a, sa, 0), (b, sb, n - 1)):
            # Force finite nonzero blocks at min scale and max delta. The
            # remaining packed values still cover the entire E2M1 alphabet.
            codes[row, :32] = 0x77
            scales[row, 1] = 125
            scales[row, 0] = 138 if mode == "sparse_span" else 137
        clear_padding(a, k)
        clear_padding(b, k)
    return fixture


def row_certificates(codes, scales):
    """CPU integer arithmetic with bounded chunk memory, including NaN blocks.

    Match the mathematical row definition, independently of CUDA thread
    grouping: nonzero finite blocks choose the normalization minimum; NaN
    scales are counted even for zero codes; energy saturates at 2**48+1.
    """
    rows = []
    for start in range(0, len(codes), 32):
        raw = codes[start:start + 32]
        q = np.empty((len(raw), raw.shape[1] * 2), dtype=np.int16)
        q[:, 0::2], q[:, 1::2] = E2M1_Q[raw & 15], E2M1_Q[raw >> 4]
        block_energy = (q.astype(np.int32) ** 2).reshape(len(raw), -1, 32).sum(axis=-1)
        for energy_blocks, scale_row in zip(block_energy, scales[start:start + 32]):
            active = energy_blocks != 0
            finite_active = active & (scale_row != 255)
            base = int(scale_row[finite_active].min()) if finite_active.any() else 127
            high = int(scale_row[finite_active].max()) if finite_active.any() else base
            energy = 0
            for raw_energy, scale in zip(energy_blocks[finite_active], scale_row[finite_active]):
                delta = int(scale) - base
                term = ENERGY_CAP if delta > 24 else int(raw_energy) << (2 * delta)
                energy = min(ENERGY_CAP, energy + term)
            rows.append({"energy": energy, "min_scale": base, "span": high - base,
                         "nan_scales": int(np.count_nonzero(scale_row == 255)),
                         "zero": int(not active.any()), "saturated": int(energy == ENERGY_CAP)})
    return rows


def group_certificate(a_rows, b_rows):
    u, v = max(x["energy"] for x in a_rows), max(x["energy"] for x in b_rows)
    amin, amax = min(x["min_scale"] for x in a_rows), max(x["min_scale"] for x in a_rows)
    bmin, bmax = min(x["min_scale"] for x in b_rows), max(x["min_scale"] for x in b_rows)
    aspan, bspan = max(x["span"] for x in a_rows), max(x["span"] for x in b_rows)
    anan, bnan = sum(x["nan_scales"] for x in a_rows), sum(x["nan_scales"] for x in b_rows)
    flags = (int(bool(anan or bnan)) | (2 if max(aspan, bspan) > 12 else 0)
             | (4 if u and v and u * v > ENERGY_BOUND else 0)
             | (8 if amin + bmin - 256 < -126 or amax + bmax - 256 > 103 else 0))
    return {"accepted": int(flags == 0), "reject_flags": flags,
            "max_energy_a": u, "max_energy_b": v, "min_scale_a": amin,
            "max_scale_a": amax, "min_scale_b": bmin, "max_scale_b": bmax,
            "max_span_a": aspan, "max_span_b": bspan,
            "nan_scales_a": anan, "nan_scales_b": bnan,
            "zero_rows_a": sum(x["zero"] for x in a_rows),
            "zero_rows_b": sum(x["zero"] for x in b_rows),
            "energy_saturated_a": max(x["saturated"] for x in a_rows),
            "energy_saturated_b": max(x["saturated"] for x in b_rows)}


def expected_certificates(fixture):
    a = row_certificates(fixture.a, fixture.a_scales)
    b = row_certificates(fixture.b, fixture.b_scales)
    global_info = group_certificate(a, b)
    masks = [[group_certificate(a[i:i + TILE], b[j:j + TILE])["accepted"]
              for j in range(0, fixture.n, TILE)] for i in range(0, fixture.m, TILE)]
    fast = sum(sum(row) for row in masks)
    total = len(masks) * len(masks[0])
    return {"global": global_info, "tile_mask": masks, "fast_tiles": fast,
            "fallback_tiles": total - fast, "total_tiles": total,
            "tiles_m": len(masks), "tiles_n": len(masks[0])}
