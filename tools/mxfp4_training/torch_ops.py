"""First-order MXFP4 linear training with an explicit surrogate backward.

The CPU backend is an independent NumPy arithmetic oracle, not acceleration.
The CUDA backend launches the repository's SM70 kernel on PyTorch's current
stream. Quantization defaults to unfused PyTorch, including validation syncs.
An explicit cuda_rne quantizer uses one native warp per MX32 and synchronizes
its nonfinite flag before returning; stochastic rounding stays in PyTorch.
The opt-in cuda_certified bridge owns a fresh native context per call and
synchronizes on destruction; it is not a cached asynchronous training backend.
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


class _CertifiedInfo(ctypes.Structure):
    """Exact ABI v1 layout from csrc/mxfp4_gemm_certified.h (128 bytes)."""
    _fields_ = [
        ("abi_version", ctypes.c_uint32), ("device", ctypes.c_int32),
        ("m", ctypes.c_int64), ("n", ctypes.c_int64), ("padded_k", ctypes.c_int64),
        ("workspace_bytes", ctypes.c_uint64),
        ("max_energy_a", ctypes.c_uint64), ("max_energy_b", ctypes.c_uint64),
        ("accepted", ctypes.c_uint32), ("reject_flags", ctypes.c_uint32),
        ("min_scale_a", ctypes.c_int32), ("max_scale_a", ctypes.c_int32),
        ("min_scale_b", ctypes.c_int32), ("max_scale_b", ctypes.c_int32),
        ("max_span_a", ctypes.c_uint32), ("max_span_b", ctypes.c_uint32),
        ("nan_scales_a", ctypes.c_uint32), ("nan_scales_b", ctypes.c_uint32),
        ("zero_rows_a", ctypes.c_uint32), ("zero_rows_b", ctypes.c_uint32),
        ("energy_saturated_a", ctypes.c_uint32), ("energy_saturated_b", ctypes.c_uint32),
        ("metadata_valid", ctypes.c_uint32), ("reserved", ctypes.c_uint32),
        ("submitted_calls", ctypes.c_uint64),
    ]


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
        if k < self.padded_k and bool(torch.any(self.codes[:, (k + 1) // 2:])):
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


@lru_cache(maxsize=8)
def _load_quantizer_library(path: str) -> ctypes.CDLL:
    """Load only the new quantizer ABI; never substitute a different backend."""
    lib = ctypes.CDLL(str(Path(path).expanduser().resolve(strict=True)))
    try:
        version = lib.mxfp4_quantize_rne_cuda_abi_version
        identity = lib.mxfp4_quantize_rne_cuda_backend
        kernel = lib.mxfp4_quantize_rne_cuda
    except AttributeError as error:
        raise RuntimeError("CUDA library has no RNE quantizer; rebuild with --with-quantizer-rne") from error
    version.restype = ctypes.c_int
    identity.restype = ctypes.c_char_p
    if version() != 1 or identity() != b"sm70_mx32_rne_warp_fp32_v1":
        raise RuntimeError("unsupported MXFP4 RNE quantizer ABI/backend")
    kernel.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 4 + [ctypes.c_void_p]
    kernel.restype = ctypes.c_int
    return lib


@torch.no_grad()
def quantize_mx_cuda_rne(x: torch.Tensor) -> TorchMXTensor:
    """Explicit SM70 RNE encoder for finite FP32 strided CUDA matrices.

    The native kernel computes scale/encoding and detects nonfinite values in
    one pass. Reading its device flag synchronizes the current stream before
    returning or rejecting the input; this wait is part of wrapper timings.
    Positive/zero strides, transposes, offsets and original negative zero are
    supported. Output K padding is canonical positive zero. CUDA Graph capture
    is rejected before allocation/submission. Missing native support is an
    error; there is no Torch/CPU replacement inside this opt-in entry point.
    """
    if (not isinstance(x, torch.Tensor) or x.layout != torch.strided
            or x.dtype != torch.float32 or x.ndim != 2 or min(x.shape) <= 0):
        raise ValueError("cuda_rne quantization requires a nonempty strided FP32 matrix")
    if x.device.type != "cuda":
        raise ValueError("cuda_rne quantization requires CUDA inputs; no device fallback")
    rows, k = x.shape
    strides = x.stride()
    limit = 2**63 - 1
    if (any(s < 0 or s > limit for s in strides) or rows > limit or k > limit - 31):
        raise ValueError("cuda_rne requires nonnegative strides and signed-64-bit-safe dimensions")
    with torch.cuda.device(x.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cuda_rne does not support CUDA Graph capture")
        if torch.cuda.get_device_capability(x.device) != (7, 0):
            raise ValueError("this experimental RNE quantizer is gated to SM70")
        path = os.environ.get("MXFP4_CUDA_LIBRARY")
        if not path:
            raise RuntimeError("set MXFP4_CUDA_LIBRARY to a library built with --with-quantizer-rne")
        lib = _load_quantizer_library(path)
        stream = torch.cuda.current_stream(x.device)
        # Record original storage before any optional lazy-view resolution.
        x.record_stream(stream)
        source = x.detach().resolve_conj().resolve_neg()
        padded_k = ((k + 31) // 32) * 32
        codes = torch.empty((rows, padded_k // 2), dtype=torch.uint8, device=x.device)
        scales = torch.empty((rows, padded_k // 32), dtype=torch.uint8, device=x.device)
        error_flag = torch.empty((), dtype=torch.int32, device=x.device)
        tensors = (source, codes, scales, error_flag)
        for tensor in tensors:
            tensor.record_stream(stream)  # Every native buffer is retained BEFORE submission.
        code = lib.mxfp4_quantize_rne_cuda(*(tensor.data_ptr() for tensor in tensors),
                                          rows, k, *source.stride(), stream.cuda_stream)
        if code:
            raise RuntimeError(f"MXFP4 CUDA RNE launch failed with cudaError_t={code}")
        # The single flag read waits for the entire quantization kernel on this
        # stream. Invalid outputs are never exposed to the caller or GEMM.
        if error_flag.item() != 0:
            raise ValueError("quantization rejects non-finite inputs")
        return TorchMXTensor(codes, scales, (rows, k))


@torch.no_grad()
def quantize_pair_mx_cuda_rne(a: torch.Tensor, b: torch.Tensor) -> tuple[TorchMXTensor, TorchMXTensor]:
    """Quantize two independent SM70 inputs with one finite-flag host read.

    This opt-in wrapper uses the unchanged native RNE kernel twice on the
    current stream. Both inputs are validated before either native call, and
    every native buffer is registered with that stream before submission.
    A single two-int device-to-host copy waits for both results; either
    nonfinite input rejects the entire pair without exposing packed outputs.

    Shapes may differ. Inputs must share one CUDA device; row/column strides,
    lazy negative views and K padding follow quantize_mx_cuda_rne. Errors after
    a native call was attempted synchronize this stream while buffers remain
    alive. This function does not change the single-input API, MXRecipe,
    stochastic rounding, or the GEMM arithmetic contract.
    """
    limit = 2**63 - 1
    pointer_limit = 2**(ctypes.sizeof(ctypes.c_void_p) * 8 - 1) - 1

    def validate(x: torch.Tensor) -> tuple[int, int]:
        if (not isinstance(x, torch.Tensor) or x.layout != torch.strided
                or x.dtype != torch.float32 or x.ndim != 2 or min(x.shape) <= 0):
            raise ValueError("paired cuda_rne requires nonempty strided FP32 matrices")
        if x.device.type != "cuda":
            raise ValueError("paired cuda_rne requires CUDA inputs; no device fallback")
        rows, k = x.shape
        strides = x.stride()
        if (any(s < 0 or s > limit for s in strides) or rows > limit or k > limit - 31):
            raise ValueError("paired cuda_rne requires nonnegative strides and signed-64-bit-safe dimensions")
        # Mirror the native ABI extent checks for BOTH inputs before the
        # first native enqueue. Python integers make the products overflow-safe.
        if (rows * ((k + 31) // 32) * 16 > pointer_limit
                or (rows - 1) * strides[0] + (k - 1) * strides[1] > pointer_limit // 4):
            raise ValueError("paired cuda_rne input extent exceeds the native pointer range")
        return rows, k

    shapes = (validate(a), validate(b))
    if a.device != b.device:
        raise ValueError("paired cuda_rne inputs must share one CUDA device")
    with torch.cuda.device(a.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("paired cuda_rne does not support CUDA Graph capture")
        if torch.cuda.get_device_capability(a.device) != (7, 0):
            raise ValueError("this experimental paired RNE quantizer is gated to SM70")
        path = os.environ.get("MXFP4_CUDA_LIBRARY")
        if not path:
            raise RuntimeError("set MXFP4_CUDA_LIBRARY to a library built with --with-quantizer-rne")
        lib = _load_quantizer_library(path)
        stream = torch.cuda.current_stream(a.device)
        for x in (a, b):
            x.record_stream(stream)  # Includes original storage of lazy views.
        sources = tuple(x.detach().resolve_conj().resolve_neg() for x in (a, b))
        # A lazy-view resolution can change strides; check its actual metadata
        # too, still before the first native enqueue.
        for source in sources:
            validate(source)
        outputs = tuple(TorchMXTensor(
            torch.empty((rows, ((k + 31) // 32) * 16), dtype=torch.uint8, device=a.device),
            torch.empty((rows, (k + 31) // 32), dtype=torch.uint8, device=a.device),
            (rows, k)) for rows, k in shapes)
        flags = torch.empty((2,), dtype=torch.int32, device=a.device)
        for tensor in (*sources, outputs[0].codes, outputs[0].scales,
                       outputs[1].codes, outputs[1].scales, flags):
            tensor.record_stream(stream)
        attempted = False
        try:
            for index, (source, output, (rows, k)) in enumerate(zip(sources, outputs, shapes)):
                attempted = True  # A failing ABI call may already have enqueued work.
                code = lib.mxfp4_quantize_rne_cuda(
                    source.data_ptr(), output.codes.data_ptr(), output.scales.data_ptr(),
                    flags.data_ptr() + index * flags.element_size(),
                    rows, k, *source.stride(), stream.cuda_stream)
                if code:
                    raise RuntimeError(f"MXFP4 paired CUDA RNE launch {index} failed with cudaError_t={code}")
            # One blocking D2H transfer, not two independent .item() waits.
            # Both flags share storage but the unchanged ABI clears only its
            # own aligned int, so the second call cannot erase the first error.
            flag_values = flags.cpu().tolist()
        except BaseException as error:
            if attempted:
                try:
                    stream.synchronize()
                except BaseException as cleanup:
                    raise RuntimeError(f"{error}; paired quantizer cleanup failed: {cleanup}") from error
            raise
        if flag_values != [0, 0]:
            raise ValueError("paired quantization rejects non-finite inputs")
        return outputs


@lru_cache(maxsize=8)
def _load_library(path: str, backend: str = "cuda") -> ctypes.CDLL:
    lib = ctypes.CDLL(str(Path(path).expanduser().resolve(strict=True)))
    lib.mxfp4_cuda_abi_version.restype = ctypes.c_int
    if lib.mxfp4_cuda_abi_version() != 1:
        raise RuntimeError("unsupported MXFP4 CUDA ABI")
    lib.mxfp4_gemm_cuda.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int64] * 3 + [ctypes.c_void_p]
    lib.mxfp4_gemm_cuda.restype = ctypes.c_int
    if backend == "cuda_tiled":
        try:
            version = lib.mxfp4_tiled_cuda_abi_version
            kernel = lib.mxfp4_gemm_tiled_cuda
        except AttributeError as error:
            raise RuntimeError("CUDA library has no tiled backend; rebuild with build_cuda.py") from error
        version.restype = ctypes.c_int
        if version() != 1:
            raise RuntimeError("unsupported tiled MXFP4 CUDA ABI")
        kernel.argtypes = lib.mxfp4_gemm_cuda.argtypes
        kernel.restype = ctypes.c_int
    if backend == "cuda_certified":
        signatures = {
            "mxfp4_certified_create": ([ctypes.c_int64] * 3 +
                                        [ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int),
            "mxfp4_certified_gemm": ([ctypes.c_void_p] * 7 +
                                      [ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
            "mxfp4_certified_get_info": ([ctypes.c_void_p, ctypes.POINTER(_CertifiedInfo)],
                                          ctypes.c_int),
            "mxfp4_certified_destroy": ([ctypes.c_void_p], ctypes.c_int),
            "mxfp4_certified_workspace_bytes": ([ctypes.c_int64] * 3 +
                                                 [ctypes.POINTER(ctypes.c_uint64)], ctypes.c_int),
            "mxfp4_certified_error_string": ([ctypes.c_int], ctypes.c_char_p),
        }
        for name, (arguments, result) in signatures.items():
            try:
                function = getattr(lib, name)
            except AttributeError as error:
                raise RuntimeError("CUDA library has no certified API; rebuild with --with-certified") from error
            function.argtypes, function.restype = arguments, result
    return lib


def _certified_error(lib, operation: str, code: int) -> RuntimeError:
    description = lib.mxfp4_certified_error_string(code)
    text = description.decode("utf-8", errors="replace") if description else "unknown native status"
    return RuntimeError(f"MXFP4 certified {operation} failed: status={code}, {text}")


def _gemm_certified(lib, a, b, out, stream, diagnostic: dict | None) -> None:
    """Own create/use/destroy on this host thread and current CUDA device.

    Retain all tensors and the stream until destroy has waited for native work,
    including when gemm returns an error after partially enqueuing its pipeline.
    There is deliberately no shared context, finalizer, or cross-thread cache.
    """
    tensors = (a.codes, a.scales, b.codes, b.scales, out)
    for tensor in tensors:
        tensor.record_stream(stream)  # Register lifetime BEFORE any submission.
    context = ctypes.c_void_p()
    failure = None
    observed = None
    try:
        code = lib.mxfp4_certified_create(a.logical_shape[0], b.logical_shape[0],
                                         a.padded_k, ctypes.byref(context))
        if code:
            raise _certified_error(lib, "create", code)
        if not context.value:
            raise RuntimeError("MXFP4 certified create returned a null context")
        initial = _CertifiedInfo()
        code = lib.mxfp4_certified_get_info(context, ctypes.byref(initial))
        if code:
            raise _certified_error(lib, "initial ABI query", code)
        if (initial.abi_version != 1 or initial.metadata_valid != 0 or initial.submitted_calls != 0
                or initial.device != a.codes.device.index
                or (initial.m, initial.n, initial.padded_k) !=
                (a.logical_shape[0], b.logical_shape[0], a.padded_k)):
            raise RuntimeError("Unsupported or inconsistent MXFP4 certified ABI v1 context")
        code = lib.mxfp4_certified_gemm(context, *(t.data_ptr() for t in tensors),
                                       stream.cuda_stream, None)
        if code:
            raise _certified_error(lib, "gemm", code)
        if diagnostic is not None:
            info = _CertifiedInfo()
            code = lib.mxfp4_certified_get_info(context, ctypes.byref(info))
            if code:
                raise _certified_error(lib, "diagnostic query", code)
            if info.abi_version != 1 or info.metadata_valid != 1 or info.submitted_calls != 1:
                raise RuntimeError("MXFP4 certified diagnostic metadata is invalid")
            observed = {name: int(getattr(info, name)) for name, _ in info._fields_}
    except BaseException as error:
        failure = error
        raise
    finally:
        if context.value:
            # This call waits on the same stream, on the creating host thread.
            code = lib.mxfp4_certified_destroy(context)
            if code:
                cleanup = _certified_error(lib, "destroy", code)
                if failure is not None:
                    raise RuntimeError(f"{failure}; cleanup also failed: {cleanup}") from failure
                raise cleanup
    if diagnostic is not None:
        diagnostic.update(observed)
        diagnostic["context_lifecycle"] = "per_call_create_gemm_destroy_synchronous"


@torch.no_grad()
def gemm_packed(a: TorchMXTensor, b: TorchMXTensor, *, backend: str = "reference",
                certified_info: dict | None = None) -> torch.Tensor:
    """A @ B_rows.T, with exact integer block products and FP32 accumulation.

    cuda_certified creates and destroys its context per call, synchronously.
    Its create/free and waiting costs belong in any wrapper/step measurement.
    An optional empty certified_info dict requests diagnostics for this exact
    submission. Diagnostic queries synchronize; keep them outside performance
    timing. Other backends do not accept this argument. CUDA Graph capture is
    unsupported, as is the native context API. Public packed shapes still
    require positive K; the native C API alone also supports K=0.
    """
    if certified_info is not None:
        if backend != "cuda_certified" or type(certified_info) is not dict or certified_info:
            raise ValueError("certified_info requires cuda_certified and an empty dict")
    # Reject capture before validation can synchronize on padding predicates.
    if backend == "cuda_certified" and a.codes.device.type == "cuda":
        with torch.cuda.device(a.codes.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("cuda_certified does not support CUDA Graph capture")
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
    if backend not in ("cuda", "cuda_tiled", "cuda_certified"):
        raise ValueError("backend must be reference, cuda, cuda_tiled, or cuda_certified")
    if a.codes.device.type != "cuda":
        raise ValueError("cuda backend requires CUDA inputs")
    if torch.cuda.get_device_capability(a.codes.device) != (7, 0):
        raise ValueError("this experimental backend is gated to SM70")
    library_path = os.environ.get("MXFP4_CUDA_LIBRARY")
    if not library_path:
        raise RuntimeError("set MXFP4_CUDA_LIBRARY to the compiled libmxfp4_sm70.so")
    lib = _load_library(library_path, backend)
    kernel = lib.mxfp4_gemm_tiled_cuda if backend == "cuda_tiled" else lib.mxfp4_gemm_cuda
    with torch.cuda.device(a.codes.device):
        stream = torch.cuda.current_stream(a.codes.device)
        out = torch.empty((a.logical_shape[0], b.logical_shape[0]),
                          device=a.codes.device, dtype=torch.float32)
        if backend == "cuda_certified":
            _gemm_certified(lib, a, b, out, stream, certified_info)
            return out
        tensors = [a.codes, a.scales, b.codes, b.scales, out]
        code = kernel(*(t.data_ptr() for t in tensors),
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
    quantizer_backend: str = "torch"

    def __post_init__(self) -> None:
        if any(type(v) is not bool for v in (self.forward, self.dgrad, self.wgrad)):
            raise ValueError("forward, dgrad and wgrad must be bool")
        if self.backend not in ("reference", "cuda", "cuda_tiled", "cuda_certified"):
            raise ValueError("unsupported backend")
        if any(r not in ("nearest", "stochastic") for r in (self.rounding, self.gradient_rounding)):
            raise ValueError("unsupported rounding")
        if self.quantizer_backend not in ("torch", "cuda_rne"):
            raise ValueError("unsupported quantizer_backend")
        if self.quantizer_backend == "cuda_rne" and self.backend == "reference":
            raise ValueError("cuda_rne quantizer requires a CUDA GEMM backend")


def _product(a: torch.Tensor, b: torch.Tensor, recipe: MXRecipe,
             *, gradient: bool = False) -> torch.Tensor:
    # Each direction still encodes its ORIGINAL FP32 operands. SR deliberately
    # keeps the existing Torch path and RNG consumption, independently per input.
    a_rounding = recipe.gradient_rounding if gradient else recipe.rounding
    qa = (quantize_mx_cuda_rne(a) if recipe.quantizer_backend == "cuda_rne" and a_rounding == "nearest"
          else quantize_mx_torch(a, rounding=a_rounding))
    qb = (quantize_mx_cuda_rne(b) if recipe.quantizer_backend == "cuda_rne" and recipe.rounding == "nearest"
          else quantize_mx_torch(b, rounding=recipe.rounding))
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
