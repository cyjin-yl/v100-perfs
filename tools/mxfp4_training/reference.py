"""Independent NumPy reference for the MXFP4 training experiments.

This module defines arithmetic, not a convergence claim. A tensor stores rows
quantized along the last (reduction) axis in blocks of 32. Transposing already
quantized codes does not produce the other reduction-axis representation:
quantize the transpose of the original high-precision tensor separately.

The quantizer uses ``floor(log2(amax)) - 2`` and saturates to E2M1's maximum
magnitude, 6. Stochastic rounding is unbiased *between representable values*,
but saturation remains biased. This recipe is explicit and is not implied by
the MX storage format itself.

The GEMM oracle sums integer ``q = 2 * E2M1`` products exactly within each
block, applies both E8M0 scales with one ldexp, rounds to FP32, then adds the
block result to the FP32 output. It intentionally does not materialize a scale
product or an FP16 dequantized operand. E8M0 byte 0 means 2**-127; byte 255
propagates NaN even when the corresponding integer dot product is zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


BLOCK_SIZE = 32
E2M1_Q = np.array([0, 1, 2, 3, 4, 6, 8, 12,
                  0, -1, -2, -3, -4, -6, -8, -12], dtype=np.int16)
E2M1_Q.flags.writeable = False
_MAGNITUDES = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)
_MAGNITUDES.flags.writeable = False


@dataclass(frozen=True)
class MXTensor:
    """A contiguous, row-wise MXFP4 tensor with zero-filled K padding.

    ``codes`` is uint8 [rows, padded_k/2], low nibble first. ``scales`` is
    uint8 [rows, padded_k/32]. ``logical_shape`` omits the padded K elements.
    Empty dimensions are rejected. Arrays are not copied; public operations
    revalidate them, because callers can still mutate NumPy array storage.
    """

    codes: np.ndarray
    scales: np.ndarray
    logical_shape: tuple[int, int]

    def __post_init__(self) -> None:
        self.validate()

    @property
    def padded_k(self) -> int:
        return ((int(self.logical_shape[1]) + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE

    def validate(self) -> None:
        if (not isinstance(self.logical_shape, tuple)
                or len(self.logical_shape) != 2
                or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral)
                       for v in self.logical_shape)):
            raise TypeError("logical_shape must be a pair of integer dimensions")
        rows, k = self.logical_shape
        if rows <= 0 or k <= 0:
            raise ValueError("MXFP4 reference requires positive rows and K")
        for name, value, shape in (
            ("codes", self.codes, (rows, self.padded_k // 2)),
            ("scales", self.scales, (rows, self.padded_k // BLOCK_SIZE)),
        ):
            if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
                raise TypeError(f"{name} must be a uint8 NumPy array")
            if value.shape != shape:
                raise ValueError(f"{name} shape must be {shape}, got {value.shape}")
            if not value.flags.c_contiguous:
                raise ValueError(f"{name} must be C-contiguous")
        # A partial byte's high nibble, followed by whole padding bytes, must
        # be bitwise zero (the canonical positive-zero code).
        if k & 1 and np.any(self.codes[:, k // 2] & np.uint8(0xF0)):
            raise ValueError("MXFP4 K padding must contain zero codes")
        if np.any(self.codes[:, (k + 1) // 2:]):
            raise ValueError("MXFP4 K padding must contain zero codes")


def unpack_codes(tensor: MXTensor) -> np.ndarray:
    """Return uint8 nibbles including K padding, validating the public input."""
    if not isinstance(tensor, MXTensor):
        raise TypeError("expected an MXTensor")
    tensor.validate()
    unpacked = np.empty((tensor.logical_shape[0], tensor.padded_k), dtype=np.uint8)
    unpacked[:, 0::2] = tensor.codes & np.uint8(15)
    unpacked[:, 1::2] = tensor.codes >> np.uint8(4)
    return unpacked


def unpack_q(tensor: MXTensor) -> np.ndarray:
    """Return exact signed int16 q=2*E2M1, including K padding."""
    return E2M1_Q[unpack_codes(tensor)]


def _generator(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is not None and (isinstance(rng, (bool, np.bool_)) or not isinstance(rng, Integral)):
        raise TypeError("rng must be a NumPy Generator, an integer seed, or None")
    return np.random.default_rng(rng)


def quantize_mx(x: np.ndarray, *, rounding: str = "nearest",
                rng: np.random.Generator | int | None = None) -> MXTensor:
    """Quantize a finite float32/float64 matrix along K, with explicit padding.

    RNE ties choose the even E2M1 code (the trailing significand bit). Negative
    values use sign-magnitude coding, preserving negative zero. SR samples the
    two neighboring magnitudes after saturation; an integer ``rng`` seed makes
    it reproducible without changing NumPy's process-global random state.
    """
    x = np.asarray(x)
    if x.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("quantize_mx requires a float32 or float64 matrix")
    if x.ndim != 2 or min(x.shape, default=0) <= 0:
        raise ValueError("quantize_mx requires a matrix with positive rows and K")
    if rounding not in ("nearest", "stochastic"):
        raise ValueError("rounding must be 'nearest' or 'stochastic'")
    if not np.isfinite(x).all():
        raise ValueError("quantize_mx rejects non-finite inputs")

    rows, k = x.shape
    padded_k = ((k + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    padded = np.zeros((rows, padded_k), dtype=np.float64)
    padded[:, :k] = x
    blocks = padded.reshape(rows, -1, BLOCK_SIZE)
    amax = np.max(np.abs(blocks), axis=-1)
    # frexp avoids log2 rounding just below a power of two and covers even
    # binary64 subnormals. Zero blocks deliberately use exponent zero.
    _, frexp_exponent = np.frexp(amax)
    exponents = np.where(amax == 0, 0, np.clip(frexp_exponent - 3, -127, 127))
    scales = np.ascontiguousarray(exponents + 127, dtype=np.uint8)
    normalized = np.ldexp(np.abs(blocks), -exponents[..., None])
    normalized = np.minimum(normalized, 6.0)

    upper = np.searchsorted(_MAGNITUDES, normalized, side="left")
    lower = np.maximum(upper - 1, 0)
    lower_value, upper_value = _MAGNITUDES[lower], _MAGNITUDES[upper]
    if rounding == "nearest":
        lower_distance = normalized - lower_value
        upper_distance = upper_value - normalized
        choose_upper = ((upper_distance < lower_distance)
                        | ((upper_distance == lower_distance) & ((upper & 1) == 0)))
    else:
        probabilities = np.divide(normalized - lower_value,
                                  upper_value - lower_value,
                                  out=np.zeros_like(normalized),
                                  where=upper_value != lower_value)
        choose_upper = _generator(rng).random(normalized.shape) < probabilities
    magnitude_codes = np.where(choose_upper, upper, lower).astype(np.uint8)
    sign_codes = np.signbit(blocks).astype(np.uint8) << np.uint8(3)
    codes = (magnitude_codes | sign_codes).reshape(rows, padded_k)
    packed = codes[:, 0::2] | (codes[:, 1::2] << np.uint8(4))
    return MXTensor(np.ascontiguousarray(packed), scales, (rows, k))


def dequantize_mx(tensor: MXTensor, *, dtype=np.float32) -> np.ndarray:
    """Decode to float32 (or float64), omitting K padding and preserving -0.

    E8M0 byte 255 produces a NaN for every element of its block. Finite scale
    bytes can still cause overflow/underflow in the requested output dtype.
    """
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("dequantize_mx output dtype must be float32 or float64")
    codes = unpack_codes(tensor)
    magnitude = _MAGNITUDES[codes & np.uint8(7)]
    values = np.copysign(magnitude, np.where(codes & np.uint8(8), -1.0, 1.0))
    exponents = tensor.scales.astype(np.int32) - 127
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        values = np.ldexp(values.reshape(*tensor.scales.shape, BLOCK_SIZE),
                          exponents[..., None])
        values = np.where(tensor.scales[..., None] == 255, np.nan, values)
        return np.ascontiguousarray(values.reshape(codes.shape)[:, :tensor.logical_shape[1]], dtype=dtype)


def gemm_mx(a: MXTensor, b: MXTensor) -> np.ndarray:
    """Compute A @ B_rows.T using exact block products and FP32 block sums.

    The independent integer reference is designed for correctness fixtures,
    not CPU performance. Operands must have the same logical reduction length.
    In particular, use separately quantized X.T / W.T / dY.T representations
    when changing reduction axes for dgrad and wgrad.
    """
    aq, bq = unpack_q(a), unpack_q(b)
    if a.logical_shape[1] != b.logical_shape[1]:
        raise ValueError("MXFP4 GEMM requires equal logical K dimensions")
    out = np.zeros((a.logical_shape[0], b.logical_shape[0]), dtype=np.float32)
    for block in range(a.padded_k // BLOCK_SIZE):
        lo, hi = block * BLOCK_SIZE, (block + 1) * BLOCK_SIZE
        qsum = aq[:, lo:hi].astype(np.int32) @ bq[:, lo:hi].astype(np.int32).T
        sa, sb = a.scales[:, block], b.scales[:, block]
        exponents = sa.astype(np.int32)[:, None] + sb.astype(np.int32)[None, :] - 256
        # binary64 ldexp exactly represents every finite intermediate needed
        # for the FP32 contract (max integer magnitude 4608, exponent ±256).
        # Round *after* scaling, preventing scale-only overflow/underflow.
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            contribution = np.ldexp(qsum.astype(np.float64), exponents).astype(np.float32)
            contribution[(sa[:, None] == 255) | (sb[None, :] == 255)] = np.nan
            out = np.add(out, contribution, dtype=np.float32)
    return out
