"""Independent storage, rounding and training-axis checks for MXFP4."""

from __future__ import annotations

import unittest

import numpy as np

from tools.mxfp4_training.reference import (
    MXTensor, dequantize_mx, gemm_mx, quantize_mx, unpack_codes,
)


# Literal value table intentionally independent of the implementation's q LUT.
VALUES = np.array([0., .5, 1., 1.5, 2., 3., 4., 6.,
                   -0., -.5, -1., -1.5, -2., -3., -4., -6.])


def encoded(codes, scales=127) -> MXTensor:
    """Build a test fixture from logical nibbles, without using the quantizer."""
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.ndim == 1:
        codes = codes[None, :]
    rows, k = codes.shape
    kp = ((k + 31) // 32) * 32
    packed = np.zeros((rows, kp // 2), dtype=np.uint8)
    for row in range(rows):
        for column in range(k):
            packed[row, column // 2] |= int(codes[row, column]) << (4 * (column % 2))
    shape = (rows, kp // 32)
    scales = np.broadcast_to(np.asarray(scales, dtype=np.uint8), shape).copy()
    return MXTensor(packed, scales, (rows, k))


class MXFP4StorageTests(unittest.TestCase):
    def test_all_sixteen_codes_and_signed_zero(self):
        fixture = encoded(np.arange(16))
        decoded = dequantize_mx(fixture, dtype=np.float64)
        np.testing.assert_array_equal(decoded, VALUES[None, :])
        np.testing.assert_array_equal(np.signbit(decoded), np.signbit(VALUES[None, :]))
        expected_bytes = [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]
        self.assertEqual(fixture.codes[0, :8].tolist(), expected_bytes)
        requantized = quantize_mx(VALUES[None, :])
        np.testing.assert_array_equal(requantized.codes, fixture.codes)
        np.testing.assert_array_equal(requantized.scales, fixture.scales)

    def test_zero_scale_byte_is_nonzero_and_255_is_nan(self):
        fixture = encoded([[2, 0], [2, 0], [2, 0]], [[0], [254], [255]])
        decoded = dequantize_mx(fixture)
        self.assertEqual(decoded[0, 0], np.float32(2.**-127))
        self.assertNotEqual(decoded[0, 0], 0)
        self.assertEqual(decoded[1, 0], np.float32(2.**127))
        self.assertTrue(np.isnan(decoded[2]).all())

    def test_tail_padding_for_odd_and_multiple_block_lengths(self):
        for k in (1, 2, 15, 31, 32, 33, 63, 65):
            with self.subTest(k=k):
                source = np.ones((3, k), dtype=np.float32)
                result = quantize_mx(source)
                self.assertEqual(result.padded_k, ((k + 31) // 32) * 32)
                self.assertTrue(np.all(unpack_codes(result)[:, k:] == 0))
                np.testing.assert_array_equal(dequantize_mx(result), source)
                self.assertTrue(result.codes.flags.c_contiguous)
                self.assertTrue(result.scales.flags.c_contiguous)

    def test_constructor_rejects_shape_dtype_contiguity_and_padding(self):
        valid = encoded([[1]])
        invalid_arguments = [
            (valid.codes.astype(np.int8), valid.scales, valid.logical_shape),
            (valid.codes, valid.scales.astype(np.int32), valid.logical_shape),
            (valid.codes[:, :1], valid.scales, valid.logical_shape),
            (valid.codes, np.ones((1, 2), dtype=np.uint8), valid.logical_shape),
            (valid.codes, valid.scales, [1, 1]),
            (valid.codes, valid.scales, (True, 1)),
            (valid.codes, valid.scales, (0, 1)),
            (valid.codes, valid.scales, (1, 0)),
            (np.zeros((1, 32), dtype=np.uint8)[:, ::2], valid.scales, valid.logical_shape),
        ]
        for arguments in invalid_arguments:
            with self.subTest(shapes=[getattr(v, "shape", v) for v in arguments]):
                with self.assertRaises((TypeError, ValueError)):
                    MXTensor(*arguments)
        for byte, value in [(0, 0x10), (1, 0x01), (15, 0x80)]:
            bad = valid.codes.copy()
            bad[0, byte] |= value
            with self.assertRaisesRegex(ValueError, "padding"):
                MXTensor(bad, valid.scales, valid.logical_shape)

    def test_mutated_padding_is_revalidated(self):
        fixture = encoded([[1]])
        fixture.codes[0, 1] = 1
        with self.assertRaisesRegex(ValueError, "padding"):
            dequantize_mx(fixture)


class MXFP4QuantizationTests(unittest.TestCase):
    def test_rne_ties_even_for_both_signs(self):
        ties = np.array([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.])
        expected = np.array([0., 1., 1., 2., 2., 4., 4.])
        source = np.concatenate([ties, -ties, [6.]])[None, :]
        result = quantize_mx(source)
        np.testing.assert_array_equal(result.scales, [[127]])
        decoded = dequantize_mx(result)
        np.testing.assert_array_equal(decoded[0, :7], expected)
        np.testing.assert_array_equal(decoded[0, 7:14], -expected)
        self.assertTrue(np.signbit(decoded[0, 7]))

    def test_rne_distinguishes_adjacent_values_around_ties(self):
        source = np.array([[np.nextafter(.25, 0.), .25,
                            np.nextafter(.25, 1.), 6.]])
        np.testing.assert_array_equal(dequantize_mx(quantize_mx(source)),
                                      [[0., 0., .5, 6.]])

    def test_scale_uses_exact_binary_exponent(self):
        source = np.array([[np.nextafter(4., 0.)], [4.],
                           [np.nextafter(8., 0.)], [8.]])
        result = quantize_mx(source)
        np.testing.assert_array_equal(result.scales[:, 0], [126, 127, 127, 128])
        # The declared floor-exponent recipe clips [6*s, 8*s); this expected
        # discontinuity must not be mistaken for an unbiased quantizer.
        np.testing.assert_array_equal(dequantize_mx(result)[:, 0], [3., 4., 6., 8.])

    def test_zero_blocks_have_canonical_scale_and_preserve_sign(self):
        source = np.array([[0., -0., 0.], [-0., -0., -0.]], dtype=np.float32)
        result = quantize_mx(source)
        np.testing.assert_array_equal(result.scales, [[127], [127]])
        np.testing.assert_array_equal(np.signbit(dequantize_mx(result)), np.signbit(source))

    def test_finite_binary64_extremes_clamp_scale_without_crashing(self):
        source = np.array([[np.nextafter(0., 1.)], [np.finfo(np.float64).max]])
        result = quantize_mx(source)
        np.testing.assert_array_equal(result.scales[:, 0], [0, 254])
        decoded = dequantize_mx(result, dtype=np.float64)
        self.assertEqual(decoded[0, 0], 0.)
        self.assertEqual(decoded[1, 0], 6. * 2.**127)

    def test_stochastic_seed_reproducibility_and_local_rng(self):
        source = np.tile(np.array([.75] * 31 + [6.]), (64, 1))
        np.random.seed(10)
        global_before = np.random.get_state()
        first = quantize_mx(source, rounding="stochastic", rng=123)
        second = quantize_mx(source, rounding="stochastic", rng=123)
        third = quantize_mx(source, rounding="stochastic", rng=124)
        np.testing.assert_array_equal(first.codes, second.codes)
        self.assertFalse(np.array_equal(first.codes, third.codes))
        global_after = np.random.get_state()
        self.assertEqual(global_before[0], global_after[0])
        np.testing.assert_array_equal(global_before[1], global_after[1])
        self.assertEqual(global_before[2:], global_after[2:])

    def test_stochastic_interior_frequency_and_negative_sign(self):
        # 63,488 independent 50/50 samples per sign; a fixed deterministic
        # seed makes this a repeatable statistical check, not a flaky test.
        source = np.tile(np.array([.75] * 31 + [6.]), (2048, 1))
        positive = dequantize_mx(quantize_mx(source, rounding="stochastic", rng=11))[:, :31]
        negative = dequantize_mx(quantize_mx(-source, rounding="stochastic", rng=11))[:, :31]
        self.assertTrue(np.isin(positive, [.5, 1.]).all())
        self.assertLess(abs(float(positive.mean()) - .75), .004)
        np.testing.assert_array_equal(negative, -positive)

    def test_stochastic_saturation_is_explicitly_biased(self):
        source = np.tile(np.array([7.5] * 32), (64, 1))
        decoded = dequantize_mx(quantize_mx(source, rounding="stochastic", rng=3))
        np.testing.assert_array_equal(decoded, np.full_like(source, 6.))

    def test_invalid_quantization_arguments(self):
        cases = [np.ones(4, dtype=np.float32), np.ones((1, 2), dtype=np.int32),
                 np.ones((1, 2), dtype=np.float16), np.ones((0, 2), dtype=np.float32),
                 np.array([[np.nan]]), np.array([[np.inf]]), np.array([[-np.inf]])]
        for source in cases:
            with self.subTest(shape=source.shape, dtype=source.dtype):
                with self.assertRaises((TypeError, ValueError)):
                    quantize_mx(source)
        with self.assertRaisesRegex(ValueError, "rounding"):
            quantize_mx(np.ones((1, 1)), rounding="truncate")
        with self.assertRaisesRegex(TypeError, "rng"):
            quantize_mx(np.ones((1, 1)), rounding="stochastic", rng="seed")


class MXFP4GemmTests(unittest.TestCase):
    def test_every_pair_of_e2m1_codes(self):
        operand = encoded(np.arange(16).reshape(16, 1))
        np.testing.assert_array_equal(gemm_mx(operand, operand),
                                      np.outer(VALUES, VALUES).astype(np.float32))

    def test_every_pair_of_finite_e8m0_scales(self):
        operand = encoded(np.full((255, 1), 3, dtype=np.uint8),
                          np.arange(255, dtype=np.uint8)[:, None])
        # Independent high-precision dequantize-then-multiply oracle: binary64
        # represents every single-element product in this finite scale range.
        high_precision = np.array([1.5 * 2.**(byte - 127) for byte in range(255)])
        with np.errstate(over="ignore", under="ignore"):
            expected = np.outer(high_precision, high_precision).astype(np.float32)
        np.testing.assert_array_equal(gemm_mx(operand, operand), expected)

    def test_combined_scaling_avoids_intermediate_overflow(self):
        operand = encoded([[1]], 191)  # 0.5 * 2**64; final dot = 2**126.
        result = gemm_mx(operand, operand)
        self.assertEqual(result[0, 0], np.float32(2.**126))
        self.assertTrue(np.isfinite(result[0, 0]))
        zero = encoded([[0]], 254)
        self.assertEqual(gemm_mx(zero, zero)[0, 0], 0.)

    def test_combined_scaling_preserves_output_subnormal(self):
        operand = encoded([[4]], 52)  # 2 * 2**-75; final dot = 2**-148.
        result = gemm_mx(operand, operand)
        self.assertEqual(result[0, 0], np.float32(2.**-148))
        self.assertNotEqual(result[0, 0], 0.)

    def test_nan_scale_propagates_even_for_zero_dot(self):
        a = encoded([[0], [0]], [[255], [127]])
        b = encoded([[0], [0]], [[127], [255]])
        actual = gemm_mx(a, b)
        np.testing.assert_array_equal(np.isnan(actual), [[True, True], [False, True]])
        self.assertEqual(actual[1, 0], 0.)

    def test_fp32_block_accumulation_order_is_part_of_contract(self):
        # First two block contributions overflow with opposite signs. A global
        # exact dot would cancel them; the specified FP32 block sum is NaN.
        a = encoded([[7] + [0] * 31 + [7] + [0] * 31], [[254, 254]])
        b = encoded([[7] + [0] * 31 + [15] + [0] * 31], [[254, 254]])
        self.assertTrue(np.isnan(gemm_mx(a, b)[0, 0]))

    def test_random_full_codes_scales_and_rectangular_tails(self):
        rng = np.random.default_rng(2026)
        for m, n, k in [(1, 1, 1), (3, 5, 31), (17, 19, 33), (5, 2, 67)]:
            with self.subTest(m=m, n=n, k=k):
                blocks = (k + 31) // 32
                a = encoded(rng.integers(0, 16, (m, k)),
                            rng.integers(124, 131, (m, blocks)))
                b = encoded(rng.integers(0, 16, (n, k)),
                            rng.integers(124, 131, (n, blocks)))
                expected = dequantize_mx(a, dtype=np.float64) @ dequantize_mx(b, dtype=np.float64).T
                # Bounds here guarantee exact binary32 arithmetic, so an
                # approximate tolerance would conceal a storage/axis error.
                np.testing.assert_array_equal(gemm_mx(a, b), expected.astype(np.float32))

    def test_forward_dgrad_wgrad_layouts(self):
        rng = np.random.default_rng(42)
        # Bounded dyadic inputs avoid quantization loss, isolating axis errors.
        x = rng.integers(-2, 3, (7, 5)).astype(np.float32)
        w = rng.integers(-2, 3, (3, 5)).astype(np.float32)
        dy = rng.integers(-2, 3, (7, 3)).astype(np.float32)
        np.testing.assert_array_equal(gemm_mx(quantize_mx(x), quantize_mx(w)), x @ w.T)
        np.testing.assert_array_equal(gemm_mx(quantize_mx(dy), quantize_mx(w.T)), dy @ w)
        np.testing.assert_array_equal(gemm_mx(quantize_mx(dy.T), quantize_mx(x.T)), dy.T @ x)

    def test_transpose_requires_requantization_of_original_values(self):
        x = np.ones((33, 33), dtype=np.float32)
        x[0, :] = 64.
        # Row-wise quantization preserves the rows exactly. Column-wise MX32
        # shares a scale with the 64 outlier, and legitimately rounds 1 to 0.
        naive_transpose = dequantize_mx(quantize_mx(x)).T
        correct_column_representation = dequantize_mx(quantize_mx(x.T))
        self.assertEqual(naive_transpose[0, 1], 1.)
        self.assertEqual(correct_column_representation[0, 1], 0.)
        self.assertFalse(np.array_equal(naive_transpose, correct_column_representation))

    def test_gemm_rejects_logical_k_mismatch_even_with_same_padding(self):
        with self.assertRaisesRegex(ValueError, "logical K"):
            gemm_mx(encoded([[1]]), encoded([[1, 1]]))


if __name__ == "__main__":
    unittest.main()
