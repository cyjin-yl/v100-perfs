#!/usr/bin/env python3
"""Finite-input V100 numerical model: an integer port of the authors' model.

Derived from north-numerical-computing/MATLAB-tensor-core at commit
6fd0520850164bdd1c37f901954e27ec63eada56, models/V100TC.m and the global
alignment path in models/tools/Generic_BFMA_TC.m.

Copyright (c) 2025, Faizan A. Khattak and Mantas Mikaitis.
The original BSD-2-Clause license is retained in UPSTREAM-LICENCE.

This Python translation uses exact integers for alignment/summation and explicit
bit truncation. It is NOT a call to a normal FP32 matrix multiplication, a CUDA
execution, an ISA interpreter, a cycle model, or a claim of universal V100
bit-equivalence. It is checked against all 5000 upstream V100 validation cases.
Only the FP16-input / FP32-output V100 branch is implemented.
"""

from __future__ import annotations

import math
import struct
from typing import Sequence

import numpy as np

UPSTREAM_COMMIT = "6fd0520850164bdd1c37f901954e27ec63eada56"


def float_to_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def exact_integer_scaled_to_fp32_rz(mantissa: int, scale: int) -> float:
    """Round mantissa * 2**scale to binary32, toward zero, via integer bits."""
    if not mantissa:
        return 0.0
    sign_bit = 0x80000000 if mantissa < 0 else 0
    mag = abs(mantissa)
    leading = mag.bit_length() - 1
    exponent = leading + scale
    if exponent > 127:
        # Matches the upstream final overflow path, which returns Infinity.
        return bits_to_float(sign_bit | 0x7F800000)
    if exponent >= -126:
        shift = leading - 23
        significand = mag >> shift if shift >= 0 else mag << -shift
        return bits_to_float(sign_bit | ((exponent + 127) << 23) | (significand & 0x7FFFFF))
    # Subnormal grid is 2**-149. Truncation preserves the upstream RZ behavior.
    shift = scale + 149
    significand = mag << shift if shift >= 0 else mag >> -shift
    return bits_to_float(sign_bit | significand)


def fp16_significand_and_exponent(x: float) -> tuple[int, int]:
    """Signed 11-bit significand and unbiased exponent, clamped to -14."""
    if not math.isfinite(x):
        raise ValueError("The port's primitive requires finite FP16 operands")
    if x == 0:
        return 0, -14
    exponent = max(math.frexp(abs(x))[1] - 1, -14)
    scaled = math.ldexp(x, 10 - exponent)
    if scaled != int(scaled) or abs(scaled) >= 2048:
        raise ValueError(f"Operand is not an exact finite FP16 value: {x!r}")
    return int(scaled), exponent


def v100_bfma4(a: Sequence[float], b: Sequence[float], c: float = 0.0, *, trace: bool = False):
    """Four products and C: unnormalized products, global alignment, no guards.

    Parameters match V100TC: fma=4, neab=0, denorm_prd=1,
    global_alignment=1, late_partial_sum=0, sticky=0, final rounding=RZ.
    Positive and negative product magnitudes are truncated before signed sum.
    """
    if len(a) != len(b) or len(a) > 4:
        raise ValueError("V100 primitive accepts at most four pairs")
    if not math.isfinite(c):
        return (float(c), {"special_c": True}) if trace else float(c)
    terms: list[tuple[int, int]] = []
    for ai, bi in zip(a, b):
        if ai == 0 or bi == 0:
            continue
        ma, ea = fp16_significand_and_exponent(float(ai))
        mb, eb = fp16_significand_and_exponent(float(bi))
        # Upstream product significand is NOT normalized after multiplication.
        # ma*mb has a 2**20 scale; AlignSignficand stores it with 23 frac bits.
        terms.append((ma * mb * 8, ea + eb))
    if c != 0:
        bits = float_to_bits(c)
        raw_exp, frac = (bits >> 23) & 255, bits & 0x7FFFFF
        exp_c = raw_exp - 127 if raw_exp else -126
        sig_c = frac | (0x800000 if raw_exp else 0)
        if bits >> 31:
            sig_c = -sig_c
        terms.append((sig_c, exp_c))
    if not terms:
        return (0.0, {"zero": True}) if trace else 0.0
    maximum_exponent = max(exp for _, exp in terms)
    aligned = []
    lost_bits = []
    for sig, exp in terms:
        shift = maximum_exponent - exp
        magnitude = abs(sig)
        trunc = magnitude >> shift
        aligned.append(-trunc if sig < 0 else trunc)
        lost_bits.append(magnitude - (trunc << shift))
    total = sum(aligned)
    value = exact_integer_scaled_to_fp32_rz(total, maximum_exponent - 23)
    if trace:
        return value, {
            "terms_signed_significand_exponent": terms,
            "max_exponent": maximum_exponent,
            "aligned_signed_significands": aligned,
            "discarded_magnitude_bits": lost_bits,
            "integer_sum": total,
            "output_bits_hex": f"{float_to_bits(value):08x}",
        }
    return value


def v100_dot(a: Sequence[float], b: Sequence[float], c: float = 0.0) -> float:
    """Recursive groups of four, exactly the upstream GEMM driver order."""
    if len(a) != len(b):
        raise ValueError("Length mismatch")
    accumulator = float(c)
    for start in range(0, len(a), 4):
        accumulator = v100_bfma4(a[start:start + 4], b[start:start + 4], accumulator)
    return accumulator


def ieee_serial_dot(a: Sequence[float], b: Sequence[float], c: float = 0.0) -> float:
    """IEEE binary32 round-to-nearest at each product/addition (comparison)."""
    accumulator = np.float32(c)
    for ai, bi in zip(a, b):
        accumulator = np.float32(accumulator + np.float32(np.float32(ai) * np.float32(bi)))
    return float(accumulator)


def ieee_once_rounded_dot(a: Sequence[float], b: Sequence[float], c: float = 0.0) -> float:
    """FP64 reference sum, rounded to binary32 once; finite fixture comparison."""
    return float(np.float32(math.fsum([float(ai) * float(bi) for ai, bi in zip(a, b)] + [c])))
