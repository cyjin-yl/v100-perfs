"""Explicit, bounded reuse of one SM70 certified native GEMM context.

Use ``with CertifiedWorkspace(m, n, logical_k, device='cuda:0') as workspace``
and call ``workspace.gemm(a, b)`` on that same host thread. Successful calls
enqueue work asynchronously on PyTorch's current stream. Closing waits for the
native work and releases the fixed workspace. There is no global context cache,
implicit finalizer, or integration with autograd's potentially different worker
threads. The existing ``gemm_packed(..., backend='cuda_certified')`` API retains
its per-call synchronous context lifecycle.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import threading

import torch

from .torch_ops import TorchMXTensor, _CertifiedInfo, _certified_error, _load_library


class CertifiedWorkspaceBusy(RuntimeError):
    """A stream switch would race unfinished work; no new work was submitted."""


class CertifiedWorkspace:
    """A single fixed shape/device/library context, owned by its entering thread.

    Construction itself allocates no CUDA resources. Enter exactly once, and
    always leave the context manager or call ``close`` on the entering thread.
    A preflight validation error leaves an active context usable. An error
    after a native submission attempt closes it synchronously before propagating
    the error. No context can be reopened. ``info`` synchronizes and therefore
    belongs outside performance timing. CUDA Graph capture is unsupported.

    A different current stream is accepted only after the previous stream has
    completed. This API never inserts a hidden event dependency to make a busy
    stream switch succeed. Callers still own the input tensor data dependencies.
    """

    def __init__(self, m: int, n: int, logical_k: int, *, device,
                 library_path: str | None = None):
        if any(type(value) is not int or value <= 0 for value in (m, n, logical_k)):
            raise ValueError("workspace dimensions must be positive integers")
        self._shape = (m, n, logical_k)
        self._padded_k = (logical_k + 31) // 32 * 32
        self._device = torch.device(device)
        if self._device.type != "cuda" or self._device.index is None:
            raise ValueError("workspace device must name an explicit CUDA index")
        path = library_path if library_path is not None else os.environ.get("MXFP4_CUDA_LIBRARY")
        if not path:
            raise RuntimeError("set MXFP4_CUDA_LIBRARY or pass library_path")
        self._library_path = str(Path(path).expanduser().resolve(strict=True))
        self._lib = None
        self._context = ctypes.c_void_p()
        self._owner = None
        self._last_stream = None
        self._state = "new"
        self._submitted_calls = 0
        self._workspace_bytes = None

    @property
    def shape(self):
        return self._shape

    @property
    def padded_k(self):
        return self._padded_k

    @property
    def device(self):
        return self._device

    @property
    def library_path(self):
        return self._library_path

    @property
    def workspace_bytes(self):
        return self._workspace_bytes

    @property
    def closed(self) -> bool:
        return self._state == "closed"

    @property
    def submitted_calls(self) -> int:
        return self._submitted_calls

    def _assert_owner(self) -> None:
        # Compare Thread objects as well as native C++'s thread id, so a reused
        # Python/native numeric id cannot revive an abandoned owner context.
        if self._owner is not None and threading.current_thread() is not self._owner:
            raise RuntimeError("certified workspace requires its entering host thread")

    def _assert_active(self) -> None:
        self._assert_owner()
        if self._state != "active" or not self._context.value:
            raise RuntimeError("certified workspace is not active; enter it exactly once")

    @staticmethod
    def _reject_capture() -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("certified workspace does not support CUDA Graph capture")

    def __enter__(self):
        self._assert_owner()
        if self._state != "new":
            raise RuntimeError("certified workspace can be entered exactly once")
        self._owner = threading.current_thread()
        try:
            with torch.cuda.device(self._device):
                self._reject_capture()
                if torch.cuda.get_device_capability(self._device) != (7, 0):
                    raise ValueError("certified workspace is gated to SM70")
                self._lib = _load_library(self._library_path, "cuda_certified")
                code = self._lib.mxfp4_certified_create(
                    self._shape[0], self._shape[1], self._padded_k, ctypes.byref(self._context))
                if code:
                    raise _certified_error(self._lib, "workspace create", code)
                if not self._context.value:
                    raise RuntimeError("certified create returned a null context")
                self._state = "active"
                initial = self._read_info()
                self._workspace_bytes = initial["workspace_bytes"]
        except BaseException as error:
            self._close_after_error(error)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is None:
            self.close()
        else:
            self._close_after_error(exc_value)
        return False

    def _close_after_error(self, failure: BaseException) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            raise RuntimeError(f"{failure}; workspace cleanup also failed: {cleanup}") from failure

    def close(self) -> None:
        """Release once, on the owner thread and bound device; wait for native work."""
        self._assert_owner()
        if self._state == "closed":
            return
        if not self._context.value:
            self._state = "closed"
            return
        with torch.cuda.device(self._device):
            self._reject_capture()
            context = self._context
            # Native destroy can report a CUDA cleanup error after deleting the
            # C++ object. Never retry that consumed pointer, even on failure.
            try:
                code = self._lib.mxfp4_certified_destroy(context)
            finally:
                self._context = ctypes.c_void_p()
                self._state = "closed"
                self._last_stream = None
            if code:
                raise _certified_error(self._lib, "workspace destroy", code)

    def _read_info(self) -> dict:
        result = _CertifiedInfo()
        code = self._lib.mxfp4_certified_get_info(self._context, ctypes.byref(result))
        if code:
            raise _certified_error(self._lib, "workspace diagnostic query", code)
        expected_valid = int(self._submitted_calls > 0)
        if (result.abi_version != 1 or result.device != self._device.index
                or (result.m, result.n, result.padded_k) !=
                (self._shape[0], self._shape[1], self._padded_k)
                or result.submitted_calls != self._submitted_calls
                or result.metadata_valid != expected_valid):
            raise RuntimeError("certified workspace ABI/shape/call metadata is inconsistent")
        info = {name: int(getattr(result, name)) for name, _ in result._fields_}
        info["context_lifecycle"] = "explicit_single_shape_owner_thread_workspace"
        return info

    def info(self) -> dict:
        """Read diagnostics for the latest submission, synchronizing its stream."""
        self._assert_active()
        with torch.cuda.device(self._device):
            self._reject_capture()
            try:
                return self._read_info()
            except BaseException as error:
                self._close_after_error(error)
                raise

    @torch.no_grad()
    def gemm(self, a: TorchMXTensor, b: TorchMXTensor, *,
             certified_info: dict | None = None) -> torch.Tensor:
        """Validate and enqueue A @ B_rows.T, returning a new FP32 output tensor.

        An optional empty dictionary receives this exact submission's metadata;
        that diagnostic query synchronizes. Without diagnostics, successful
        calls do not force a host wait (padding validation can still synchronize).
        """
        self._assert_active()
        if certified_info is not None and (type(certified_info) is not dict or certified_info):
            raise ValueError("certified_info must be an empty dict")
        with torch.cuda.device(self._device):
            self._reject_capture()  # Before padding validation can synchronize.
            if not isinstance(a, TorchMXTensor) or not isinstance(b, TorchMXTensor):
                raise TypeError("workspace operands must be TorchMXTensor instances")
            a.validate()
            b.validate()
            if (a.logical_shape != (self._shape[0], self._shape[2])
                    or b.logical_shape != (self._shape[1], self._shape[2])):
                raise ValueError("workspace operands do not match its fixed logical shape")
            if a.codes.device != self._device or b.codes.device != self._device:
                raise ValueError("workspace operands do not match its bound CUDA device")
            stream = torch.cuda.current_stream(self._device)
            if (self._last_stream is not None
                    and stream.cuda_stream != self._last_stream.cuda_stream
                    and not self._last_stream.query()):
                raise CertifiedWorkspaceBusy("previous workspace stream is still busy; synchronize it before switching")
            out = torch.empty((self._shape[0], self._shape[1]), device=self._device, dtype=torch.float32)
            tensors = (a.codes, a.scales, b.codes, b.scales, out)
            for tensor in tensors:
                tensor.record_stream(stream)  # BEFORE any native submission.
            try:
                code = self._lib.mxfp4_certified_gemm(
                    self._context, *(tensor.data_ptr() for tensor in tensors), stream.cuda_stream, None)
                if code:
                    raise _certified_error(self._lib, "workspace gemm", code)
                self._last_stream = stream
                self._submitted_calls += 1
                if certified_info is not None:
                    certified_info.update(self._read_info())
            except BaseException as error:
                # A failure may follow partially enqueued preparation/GEMM.
                # Retain local tensors through same-thread synchronous close.
                self._close_after_error(error)
                raise
            return out
