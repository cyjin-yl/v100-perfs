"""First-order MXFP4 linear training with an explicit surrogate backward.

The CPU backend is an independent NumPy arithmetic oracle, not acceleration.
The CUDA backend launches the repository's SM70 kernel on PyTorch's current
stream. Quantization is currently unfused PyTorch, including validation syncs.
Weights and optimizer state remain FP32. This is a research building block;
neither successful backpropagation nor a tiny loss curve proves LLM convergence.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from .reference import MXTensor, gemm_mx


@dataclass(frozen=True)
class TorchMXTensor:
    codes: torch.Tensor
    scales: torch.Tensor
    logical_shape: tuple[int, int]

    @property
    def padded_k(self) -> int:
        return ((self.logical_shape[1] + 31) // 32) * 32

    def validate(self) -> None:
        if (not isinstance(self.logical_shape, tuple) or len(self.logical_shape) != 2
                or any(type(d) is not int or d <= 0 for d in self.logical_shape)):
            raise ValueError("logical_shape must contain two positive integers")
        rows, k = self.logical_shape
        for name, tensor, shape in (
            ("codes", self.codes, (rows, self.padded_k // 2)),
            ("scales", self.scales, (rows, self.padded_k // 32)),
        ):
            if tensor.dtype != torch.uint8 or tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must have uint8 dtype and shape {shape}")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if self.codes.device != self.scales.device:
            raise ValueError("codes and scales must be on the same device")
        if k % 2 and bool(torch.any(self.codes[:, k // 2] & 0xf0)):
            raise ValueError("K padding must be canonical positive zero")
        if bool(torch.any(self.codes[:, (k + 1) // 2:])):
            raise ValueError("K padding must be canonical positive zero")


@torch.no_grad()
def quantize_mx_torch(x: torch.Tensor, *, rounding: str = "nearest",
                      generator: torch.Generator | None = None) -> TorchMXTensor:
    """Row-wise MX32 encoding of finite FP32 values, without CPU transfers.

    Scale: floor(log2(amax))-2, clamped to E8M0's finite range. Magnitudes above
    6 saturate. SR is unbiased only inside the representable interval; it does
    not undo clipping. RNE picks the even trailing E2M1 significand bit.
    """
    if x.dtype != torch.float32 or x.ndim != 2 or min(x.shape) <= 0:
        raise ValueError("quantization requires a nonempty FP32 matrix")
    if rounding not in ("nearest", "stochastic"):
        raise ValueError("rounding must be nearest or stochastic")
    if x.device.type not in ("cpu", "cuda"):
        raise ValueError("quantization requires CPU or CUDA tensors")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("quantization rejects non-finite inputs")
    rows, k = x.shape
    padded_k = ((k + 31) // 32) * 32
    padded = torch.nn.functional.pad(x.detach(), (0, padded_k - k))
    blocks = padded.reshape(rows, -1, 32)
    amax = blocks.abs().amax(dim=-1)
    _, exponent = torch.frexp(amax)
    exponent = torch.where(amax == 0, 0, (exponent - 3).clamp(-127, 127))
    normalized = torch.ldexp(blocks.abs(), -exponent.unsqueeze(-1)).clamp(max=6)
    levels = x.new_tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
    upper = torch.bucketize(normalized.contiguous(), levels, right=False)
    lower = (upper - 1).clamp(min=0)
    lo, hi = levels[lower], levels[upper]
    if rounding == "nearest":
        d_lo, d_hi = normalized - lo, hi - normalized
        choose_upper = (d_hi < d_lo) | ((d_hi == d_lo) & ((upper & 1) == 0))
    else:
        probability = torch.where(hi != lo, (normalized - lo) / (hi - lo), 0)
        samples = torch.rand(normalized.shape, device=x.device, dtype=x.dtype,
                             generator=generator)
        choose_upper = samples < probability
    codes = torch.where(choose_upper, upper, lower).to(torch.uint8)
    codes = (codes | (torch.signbit(blocks).to(torch.uint8) << 3)).reshape(rows, padded_k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return TorchMXTensor(packed, (exponent + 127).to(torch.uint8).contiguous(), (rows, k))


@lru_cache(maxsize=4)
def _load_library(path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(Path(path).expanduser().resolve(strict=True)))
    lib.mxfp4_cuda_abi_version.restype = ctypes.c_int
    if lib.mxfp4_cuda_abi_version() != 1:
        raise RuntimeError("unsupported MXFP4 CUDA ABI")
    lib.mxfp4_gemm_cuda.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int64] * 3 + [ctypes.c_void_p]
    lib.mxfp4_gemm_cuda.restype = ctypes.c_int
    return lib


@torch.no_grad()
def gemm_packed(a: TorchMXTensor, b: TorchMXTensor, *, backend: str = "reference") -> torch.Tensor:
    """A @ B_rows.T, with exact integer block products and FP32 accumulation."""
    a.validate()
    b.validate()
    if a.logical_shape[1] != b.logical_shape[1] or a.codes.device != b.codes.device:
        raise ValueError("GEMM operands must have the same logical K and device")
    if backend == "reference":
        if a.codes.device.type != "cpu":
            raise ValueError("reference backend requires CPU; no silent device fallback")
        an = MXTensor(a.codes.numpy(), a.scales.numpy(), a.logical_shape)
        bn = MXTensor(b.codes.numpy(), b.scales.numpy(), b.logical_shape)
        return torch.from_numpy(gemm_mx(an, bn))
    if backend != "cuda":
        raise ValueError("backend must be reference or cuda")
    if a.codes.device.type != "cuda":
        raise ValueError("cuda backend requires CUDA inputs")
    if torch.cuda.get_device_capability(a.codes.device) != (7, 0):
        raise ValueError("this experimental backend is gated to SM70")
    library_path = os.environ.get("MXFP4_CUDA_LIBRARY")
    if not library_path:
        raise RuntimeError("set MXFP4_CUDA_LIBRARY to the compiled libmxfp4_sm70.so")
    lib = _load_library(library_path)
    with torch.cuda.device(a.codes.device):
        stream = torch.cuda.current_stream(a.codes.device)
        out = torch.empty((a.logical_shape[0], b.logical_shape[0]),
                          device=a.codes.device, dtype=torch.float32)
        tensors = [a.codes, a.scales, b.codes, b.scales, out]
        code = lib.mxfp4_gemm_cuda(*(t.data_ptr() for t in tensors),
                                 a.logical_shape[0], b.logical_shape[0], a.padded_k,
                                 stream.cuda_stream)
        if code:
            raise RuntimeError(f"MXFP4 CUDA launch failed with cudaError_t={code}")
        for tensor in tensors:
            tensor.record_stream(stream)
    return out


@dataclass(frozen=True)
class MXRecipe:
    """Explicit, independently switchable Fprop/Dgrad/Wgrad experiment arms."""
    forward: bool = True
    dgrad: bool = True
    wgrad: bool = True
    rounding: str = "nearest"
    gradient_rounding: str = "nearest"
    backend: str = "reference"

    def __post_init__(self) -> None:
        if any(type(v) is not bool for v in (self.forward, self.dgrad, self.wgrad)):
            raise ValueError("forward, dgrad and wgrad must be bool")
        if self.backend not in ("reference", "cuda"):
            raise ValueError("unsupported backend")
        if any(r not in ("nearest", "stochastic") for r in (self.rounding, self.gradient_rounding)):
            raise ValueError("unsupported rounding")


def _product(a: torch.Tensor, b: torch.Tensor, recipe: MXRecipe,
             *, gradient: bool = False) -> torch.Tensor:
    qa = quantize_mx_torch(a, rounding=recipe.gradient_rounding if gradient else recipe.rounding)
    qb = quantize_mx_torch(b, rounding=recipe.rounding)
    return gemm_packed(qa, qb, backend=recipe.backend)


class _MXLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor,
                bias: torch.Tensor | None, recipe: MXRecipe):
        ctx.save_for_backward(x, weight)
        ctx.recipe = recipe
        ctx.has_bias = bias is not None
        flat = x.reshape(-1, x.shape[-1])
        # Explicitly disable autocast for the FP32 reference and master policy.
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = _product(flat, weight, recipe) if recipe.forward else flat @ weight.T
            if bias is not None:
                out = out + bias
        return out.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        x, weight = ctx.saved_tensors
        recipe = ctx.recipe
        g = grad_output.reshape(-1, weight.shape[0]).contiguous()
        flat = x.reshape(-1, x.shape[-1])
        dx = dw = db = None
        with torch.autocast(device_type=x.device.type, enabled=False):
            if ctx.needs_input_grad[0]:
                # Each transpose is quantized anew FROM ORIGINAL FP32 data.
                dx = (_product(g, weight.T, recipe, gradient=True)
                      if recipe.dgrad else g @ weight)
                dx = dx.reshape_as(x)
            if ctx.needs_input_grad[1]:
                dw = (_product(g.T, flat.T, recipe, gradient=True)
                      if recipe.wgrad else g.T @ flat)
            if ctx.has_bias and ctx.needs_input_grad[2]:
                db = g.sum(dim=0)
        return dx, dw, db, None


def mxfp4_linear(x: torch.Tensor, weight: torch.Tensor,
                 bias: torch.Tensor | None = None, *, recipe: MXRecipe | None = None) -> torch.Tensor:
    """Linear with a declared quantized surrogate gradient, FP32 masters.

    Only first-order differentiation is supported. This is not the derivative
    of rounding; numerical finite-difference gradcheck is therefore inapposite.
    """
    recipe = recipe or MXRecipe()
    if x.ndim < 2 or weight.ndim != 2 or min(x.shape) <= 0 or min(weight.shape) <= 0:
        raise ValueError("nonempty input [..., K] and weight [N,K] are required")
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError("input and weight K dimensions differ")
    values = [x, weight] + ([] if bias is None else [bias])
    if any(t.dtype != torch.float32 or t.device != x.device for t in values):
        raise ValueError("input, FP32 master weight and bias must share FP32 dtype and device")
    expected = "cpu" if recipe.backend == "reference" else "cuda"
    if x.device.type != expected:
        raise ValueError(f"{recipe.backend} backend requires {expected} tensors")
    if any(not bool(torch.isfinite(t).all()) for t in values):
        raise ValueError("linear rejects non-finite input, weight and bias")
    if bias is not None and tuple(bias.shape) != (weight.shape[0],):
        raise ValueError("bias must have shape [N]")
    return _MXLinear.apply(x, weight, bias, recipe)


class MXLinear(nn.Linear):
    """Drop-in FP32 parameter layout, with explicit experimental MX arithmetic."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 *, recipe: MXRecipe | None = None, device=None):
        super().__init__(in_features, out_features, bias, device=device, dtype=torch.float32)
        self.recipe = recipe or MXRecipe()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mxfp4_linear(x, self.weight, self.bias, recipe=self.recipe)
