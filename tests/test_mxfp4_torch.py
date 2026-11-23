"""First-order MXFP4 training contracts, checked against independent NumPy.

Quantized backward deliberately implements a surrogate. Finite-difference
gradcheck of the rounding function would test a different mathematical object.
"""

from __future__ import annotations

import itertools
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from tools.mxfp4_training.reference import gemm_mx, quantize_mx

if torch is not None:
    from tools.mxfp4_training.torch_ops import (
        MXLinear, MXRecipe, TorchMXTensor, gemm_packed, mxfp4_linear,
        quantize_mx_torch,
    )


def oracle_product(a, b):
    return gemm_mx(quantize_mx(np.ascontiguousarray(a)),
                   quantize_mx(np.ascontiguousarray(b)))


@unittest.skipIf(torch is None, "optional PyTorch is not installed")
class TorchMXQuantizationTests(unittest.TestCase):
    def assert_encoding_matches_numpy(self, source):
        expected = quantize_mx(source.detach().numpy())
        actual = quantize_mx_torch(source)
        actual.validate()
        self.assertEqual(actual.logical_shape, expected.logical_shape)
        self.assertEqual(actual.padded_k, expected.padded_k)
        np.testing.assert_array_equal(actual.codes.numpy(), expected.codes)
        np.testing.assert_array_equal(actual.scales.numpy(), expected.scales)
        self.assertFalse(actual.codes.requires_grad)

    def test_all_nibbles_signed_zero_and_ties_match_numpy(self):
        codes = np.array([0., .5, 1., 1.5, 2., 3., 4., 6.,
                          -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=np.float32)
        ties = np.array([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], dtype=np.float32)
        source = np.concatenate([codes, ties, -ties, [6., -6.]]).astype(np.float32)
        self.assert_encoding_matches_numpy(torch.from_numpy(source[None, :]))
        zeros = torch.tensor([[0., -0., 0.], [-0., 0., -0.]], dtype=torch.float32)
        self.assert_encoding_matches_numpy(zeros)

    def test_adjacent_fp32_values_around_each_tie_match_numpy(self):
        ties = np.array([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], dtype=np.float32)
        source = np.concatenate([
            np.nextafter(ties, np.float32(-np.inf)),
            ties,
            np.nextafter(ties, np.float32(np.inf)),
            np.array([6.], dtype=np.float32),
        ])[None, :]
        self.assert_encoding_matches_numpy(torch.from_numpy(source))
        self.assert_encoding_matches_numpy(torch.from_numpy(-source))

    def test_subnormal_and_large_finite_scales_match_numpy(self):
        values = np.array([0., .5, 1., 1.5, 2., 3., 4., 6.,
                           -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=np.float32)
        for exponent in (-127, -126, -125, -64, -2, 0, 32, 64, 120, 125):
            with self.subTest(exponent=exponent):
                source = np.ldexp(values, exponent)[None, :]
                self.assert_encoding_matches_numpy(torch.from_numpy(source))
        extrema = np.array([
            [np.nextafter(np.float32(0), np.float32(1)), 0., -0.],
            [np.finfo(np.float32).tiny, 0., -0.],
            [np.finfo(np.float32).max, 0., -np.finfo(np.float32).max],
        ], dtype=np.float32)
        self.assert_encoding_matches_numpy(torch.from_numpy(extrema))

    def test_tails_and_noncontiguous_inputs_match_numpy(self):
        rng = np.random.default_rng(33)
        for k in (1, 31, 32, 33, 65):
            with self.subTest(k=k):
                x = torch.from_numpy(rng.normal(size=(k, 5)).astype(np.float32)).T
                self.assert_encoding_matches_numpy(x)

    def test_stochastic_seed_is_reproducible_without_global_rng_changes(self):
        source = torch.tensor([[.75] * 31 + [6.]] * 128, dtype=torch.float32)
        before = torch.get_rng_state().clone()
        first = quantize_mx_torch(source, rounding="stochastic",
                                  generator=torch.Generator().manual_seed(7))
        second = quantize_mx_torch(source, rounding="stochastic",
                                   generator=torch.Generator().manual_seed(7))
        third = quantize_mx_torch(source, rounding="stochastic",
                                  generator=torch.Generator().manual_seed(8))
        torch.testing.assert_close(first.codes, second.codes, rtol=0, atol=0)
        self.assertFalse(torch.equal(first.codes, third.codes))
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)

    def test_stochastic_interior_probability_and_saturation(self):
        source = torch.tensor([[.75] * 31 + [6.]] * 2048, dtype=torch.float32)
        result = quantize_mx_torch(source, rounding="stochastic",
                                   generator=torch.Generator().manual_seed(9))
        codes = torch.empty_like(source, dtype=torch.uint8)
        codes[:, ::2] = result.codes & 15
        codes[:, 1::2] = result.codes >> 4
        interior = codes[:, :31]
        self.assertTrue(bool(((interior == 1) | (interior == 2)).all()))
        self.assertLess(abs(float((interior == 2).float().mean()) - .5), .008)
        saturated = quantize_mx_torch(torch.full((2, 32), 7.5), rounding="stochastic",
                                      generator=torch.Generator().manual_seed(9))
        torch.testing.assert_close(saturated.codes, torch.full((2, 16), 0x77, dtype=torch.uint8),
                                   rtol=0, atol=0)

    def test_quantizer_rejects_invalid_dtype_shape_nonfinite_and_rounding(self):
        invalid = [torch.ones(2), torch.ones(1, 2, dtype=torch.float64),
                   torch.ones(1, 2, dtype=torch.float16), torch.ones(0, 2),
                   torch.tensor([[float("nan")]]), torch.tensor([[float("inf")]]),
                   torch.tensor([[-float("inf")]])]
        for source in invalid:
            with self.subTest(shape=source.shape, dtype=source.dtype):
                with self.assertRaises(ValueError):
                    quantize_mx_torch(source)
        with self.assertRaises(ValueError):
            quantize_mx_torch(torch.ones(1, 1), rounding="truncate")


@unittest.skipIf(torch is None, "optional PyTorch is not installed")
class TorchMXPackedTests(unittest.TestCase):
    def test_reference_gemm_matches_independent_numpy(self):
        rng = np.random.default_rng(901)
        for m, n, k in ((1, 1, 1), (7, 5, 31), (17, 19, 33), (4, 7, 65)):
            with self.subTest(shape=(m, n, k)):
                x = rng.normal(size=(m, k)).astype(np.float32)
                y = rng.normal(size=(n, k)).astype(np.float32)
                actual = gemm_packed(quantize_mx_torch(torch.from_numpy(x)),
                                     quantize_mx_torch(torch.from_numpy(y)))
                np.testing.assert_array_equal(actual.numpy(), oracle_product(x, y))

    def test_packed_validation_and_no_cuda_fallback_on_cpu(self):
        a = quantize_mx_torch(torch.ones(3, 1))
        b = quantize_mx_torch(torch.ones(2, 1))
        for backend in ("cuda", "cuda_tiled", "unknown"):
            with self.assertRaises(ValueError):
                gemm_packed(a, b, backend=backend)
        with self.assertRaisesRegex(ValueError, "same logical K"):
            gemm_packed(a, quantize_mx_torch(torch.ones(2, 2)))
        bad = TorchMXTensor(a.codes.to(torch.int8), a.scales, a.logical_shape)
        with self.assertRaises(ValueError):
            gemm_packed(bad, b)
        bad = TorchMXTensor(a.codes, a.scales.to("meta"), a.logical_shape)
        with self.assertRaisesRegex(ValueError, "same device"):
            gemm_packed(bad, b)
        a.codes[0, 0] |= 0x10
        with self.assertRaisesRegex(ValueError, "padding"):
            gemm_packed(a, b)


@unittest.skipIf(torch is None, "optional PyTorch is not installed")
class TorchMXLinearTests(unittest.TestCase):
    def test_all_eight_recipe_arms_against_independent_forward_and_grad_oracles(self):
        rng = np.random.default_rng(51)
        original_x = rng.normal(size=(5, 33)).astype(np.float32)
        original_w = rng.normal(size=(7, 33)).astype(np.float32)
        original_g = rng.normal(size=(5, 7)).astype(np.float32)
        original_b = rng.normal(size=(7,)).astype(np.float32)
        for f, d, w in itertools.product((False, True), repeat=3):
            with self.subTest(forward=f, dgrad=d, wgrad=w):
                x = torch.tensor(original_x, requires_grad=True)
                weight = torch.tensor(original_w, requires_grad=True)
                bias = torch.tensor(original_b, requires_grad=True)
                recipe = MXRecipe(forward=f, dgrad=d, wgrad=w)
                actual = mxfp4_linear(x, weight, bias, recipe=recipe)
                actual.backward(torch.tensor(original_g))
                expected_y = (oracle_product(original_x, original_w) if f
                              else original_x @ original_w.T) + original_b
                expected_dx = (oracle_product(original_g, original_w.T) if d
                               else original_g @ original_w)
                expected_dw = (oracle_product(original_g.T, original_x.T) if w
                               else original_g.T @ original_x)
                np.testing.assert_allclose(actual.detach().numpy(), expected_y, rtol=2e-6, atol=2e-6)
                np.testing.assert_allclose(x.grad.numpy(), expected_dx, rtol=2e-6, atol=2e-6)
                np.testing.assert_allclose(weight.grad.numpy(), expected_dw, rtol=2e-6, atol=2e-6)
                np.testing.assert_allclose(bias.grad.numpy(), original_g.sum(axis=0), rtol=2e-6, atol=2e-6)

    def test_all_fp32_recipe_matches_nn_linear_with_autocast_enabled(self):
        with torch.random.fork_rng():
            torch.manual_seed(110)
            standard = torch.nn.Linear(33, 7, dtype=torch.float32)
            experimental = MXLinear(33, 7, recipe=MXRecipe(False, False, False))
            experimental.load_state_dict(standard.state_dict())
            x = torch.randn(2, 3, 33, requires_grad=True)
            other_x = x.detach().clone().requires_grad_(True)
            g = torch.randn(2, 3, 7)
            expected = standard(x)
            # The experimental API promises FP32 masters and explicitly
            # disables the caller's autocast inside forward and backward.
            with torch.autocast("cpu", dtype=torch.bfloat16):
                actual = experimental(other_x)
            expected.backward(g)
            actual.backward(g)
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
            torch.testing.assert_close(other_x.grad, x.grad, rtol=2e-6, atol=2e-6)
            torch.testing.assert_close(experimental.weight.grad, standard.weight.grad, rtol=2e-6, atol=2e-6)
            torch.testing.assert_close(experimental.bias.grad, standard.bias.grad, rtol=2e-6, atol=2e-6)

    def test_noncontiguous_batch_weight_and_upstream_gradient(self):
        rng = np.random.default_rng(2)
        x = torch.from_numpy(rng.normal(size=(5, 2, 33)).astype(np.float32)).transpose(0, 1).requires_grad_(True)
        weight = torch.from_numpy(rng.normal(size=(33, 3)).astype(np.float32)).T.requires_grad_(True)
        g = torch.from_numpy(rng.normal(size=(5, 2, 3)).astype(np.float32)).transpose(0, 1)
        self.assertFalse(x.is_contiguous())
        self.assertFalse(weight.is_contiguous())
        self.assertFalse(g.is_contiguous())
        y = mxfp4_linear(x, weight)
        y.backward(g)
        flat_x = x.detach().numpy().reshape(-1, 33)
        flat_g = g.numpy().reshape(-1, 3)
        original_w = weight.detach().numpy()
        np.testing.assert_array_equal(y.detach().numpy(), oracle_product(flat_x, original_w).reshape(2, 5, 3))
        np.testing.assert_array_equal(x.grad.numpy(), oracle_product(flat_g, original_w.T).reshape(x.shape))
        np.testing.assert_array_equal(weight.grad.numpy(), oracle_product(flat_g.T, flat_x.T))

    def test_dgrad_and_wgrad_requantize_original_transposes(self):
        # Outliers produce materially different column scales from row scales.
        # Independent NumPy matrices use the original FP32 data in each arm.
        x = torch.ones(33, 33)
        x[0, :] = 64.
        x.requires_grad_(True)
        weight = torch.ones(33, 33)
        weight[0, :] = 64.
        weight.requires_grad_(True)
        g = torch.ones(33, 33)
        g[0, :] = 64.
        y = mxfp4_linear(x, weight)
        y.backward(g)
        expected_dx = oracle_product(g.numpy(), weight.detach().numpy().T)
        expected_dw = oracle_product(g.numpy().T, x.detach().numpy().T)
        np.testing.assert_array_equal(x.grad.numpy(), expected_dx)
        np.testing.assert_array_equal(weight.grad.numpy(), expected_dw)
        self.assertNotEqual(float(x.grad[0, 1]), float((g @ weight.detach())[0, 1]))
        self.assertNotEqual(float(weight.grad[1, 1]), float((g.T @ x.detach())[1, 1]))

    def test_bias_is_unquantized_fp32_and_gradient_is_fp32_sum(self):
        x = torch.zeros(5, 33, requires_grad=True)
        weight = torch.zeros(3, 33, requires_grad=True)
        bias = torch.tensor([.123456789, -1.e-9, 1.e8], dtype=torch.float32, requires_grad=True)
        g = torch.tensor([[.123456789, .25, -.875]] * 5, dtype=torch.float32)
        result = mxfp4_linear(x, weight, bias)
        result.backward(g)
        torch.testing.assert_close(result, bias.detach().expand(5, -1), rtol=0, atol=0)
        torch.testing.assert_close(bias.grad, g.sum(0), rtol=0, atol=0)
        self.assertEqual(bias.grad.dtype, torch.float32)

    def test_frozen_inputs_and_no_bias(self):
        layer = MXLinear(5, 3, bias=False)
        x = torch.ones(2, 5)
        layer(x).sum().backward()
        self.assertIsNone(x.grad)
        self.assertIsNone(layer.bias)
        self.assertIsNotNone(layer.weight.grad)
        layer.weight.requires_grad_(False)
        x = x.requires_grad_(True)
        layer(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_optimizer_updates_fp32_masters_and_state_dict_roundtrip(self):
        with torch.random.fork_rng():
            torch.manual_seed(4)
            layer = MXLinear(33, 7)
            x = torch.randn(5, 33)
            optimizer = torch.optim.AdamW(layer.parameters(), lr=.01)
            before = layer.weight.detach().clone()
            loss = layer(x).square().mean()
            loss.backward()
            self.assertTrue(bool(torch.isfinite(layer.weight.grad).all()))
            optimizer.step()
            self.assertFalse(torch.equal(layer.weight, before))
            self.assertEqual(layer.weight.dtype, torch.float32)
            self.assertEqual(optimizer.state[layer.weight]["exp_avg"].dtype, torch.float32)
            self.assertEqual(optimizer.state[layer.weight]["exp_avg_sq"].dtype, torch.float32)
            restored = MXLinear(33, 7)
            restored.load_state_dict(layer.state_dict())
            torch.testing.assert_close(restored(x), layer(x), rtol=0, atol=0)
            self.assertEqual(set(layer.state_dict()), {"weight", "bias"})

    def test_recipe_rejects_invalid_values(self):
        for kwargs in ({"forward": 1}, {"dgrad": "yes"}, {"wgrad": None},
                       {"backend": "auto"}, {"rounding": "truncate"},
                       {"gradient_rounding": "truncate"}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    MXRecipe(**kwargs)

    def test_linear_rejects_bad_shapes_dtypes_devices_and_no_fallback(self):
        x, w = torch.ones(2, 5), torch.ones(3, 5)
        invalid = [(torch.ones(5), w, None), (torch.ones(0, 5), w, None),
                   (torch.ones(2, 6), w, None), (x.double(), w, None),
                   (x, w.half(), None), (x, w, torch.ones(3, dtype=torch.float64)),
                   (x, w, torch.ones(1, 3)), (x, w.to("meta"), None)]
        for inputs in invalid:
            with self.subTest(shapes=[getattr(v, "shape", None) for v in inputs]):
                with self.assertRaises(ValueError):
                    mxfp4_linear(*inputs)
        with self.assertRaisesRegex(ValueError, "requires cuda"):
            mxfp4_linear(x, w, recipe=MXRecipe(backend="cuda"))
        with self.assertRaisesRegex(ValueError, "requires cpu"):
            mxfp4_linear(x.to("meta"), w.to("meta"))

    def test_linear_rejects_nonfinite_inputs_weight_and_bias_in_all_fp32_arm(self):
        for enabled in (False, True):
            recipe = MXRecipe(enabled, enabled, enabled)
            for position, value in itertools.product(range(3), (float("nan"), float("inf"), -float("inf"))):
                with self.subTest(quantized=enabled, position=position, value=value):
                    tensors = [torch.ones(2, 5), torch.ones(3, 5), torch.ones(3)]
                    tensors[position].flatten()[0] = value
                    with self.assertRaisesRegex(ValueError, "non-finite"):
                        mxfp4_linear(*tensors, recipe=recipe)


if __name__ == "__main__":
    unittest.main()
