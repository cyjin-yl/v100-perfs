"""Bounded real-SM70 validation of the PyTorch MXFP4 CUDA integration.

Run from the repository root after building the shared library:
  MXFP4_CUDA_LIBRARY=/path/to/libmxfp4_sm70.so python -m \
    tools.mxfp4_training.check_torch_cuda --output /tmp/torch-cuda-check.json

This checks both CUDA implementations, GPU quantization and the intentional
first-order surrogate backward against independent NumPy arithmetic. It does
not time kernels, measure training throughput or establish LLM convergence.
No CUDA device means a blocked result, never a CPU fallback or passing skip.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.mxfp4_training.reference import MXTensor, gemm_mx, quantize_mx

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from tools.mxfp4_training.torch_ops import (
        MXRecipe, TorchMXTensor, gemm_packed, mxfp4_linear, quantize_mx_torch,
    )


class Blocked(RuntimeError):
    """An absent execution prerequisite, not an executed numerical failure."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = ["check_torch_cuda.py", "reference.py", "torch_ops.py",
             "csrc/mxfp4_gemm.h", "csrc/mxfp4_gemm.cu",
             "csrc/mxfp4_gemm_tiled.h", "csrc/mxfp4_gemm_tiled.cu"]
    return {name: sha256(root / name) for name in names}


def library_metadata() -> dict:
    value = os.environ.get("MXFP4_CUDA_LIBRARY")
    if not value:
        raise Blocked("Set MXFP4_CUDA_LIBRARY to the compiled shared library")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise Blocked(f"CUDA shared library does not exist: {path}")
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as error:
        raise Blocked(f"CUDA shared library could not be loaded: {error}") from error
    identities = {}
    for backend, prefix in (("cuda", "mxfp4"), ("cuda_tiled", "mxfp4_tiled")):
        try:
            version = getattr(lib, prefix + "_cuda_abi_version")
            name = getattr(lib, prefix + "_cuda_backend")
            getattr(lib, "mxfp4_gemm_tiled_cuda" if backend == "cuda_tiled" else "mxfp4_gemm_cuda")
        except AttributeError as error:
            raise Blocked(f"CUDA shared library lacks the required {backend} API") from error
        version.restype = ctypes.c_int
        name.restype = ctypes.c_char_p
        abi = int(version())
        if abi != 1:
            raise RuntimeError(f"Unsupported {backend} ABI {abi}")
        identities[backend] = {"abi_version": abi, "name": name().decode("utf-8")}
    runtime = {}
    for symbol in ("cudaRuntimeGetVersion", "cudaDriverGetVersion"):
        try:
            function = getattr(lib, symbol)
            function.argtypes = [ctypes.POINTER(ctypes.c_int)]
            function.restype = ctypes.c_int
            value = ctypes.c_int()
            status = int(function(ctypes.byref(value)))
            runtime[symbol] = {"version": value.value if status == 0 else None,
                               "cuda_error": status}
        except AttributeError:
            runtime[symbol] = {"version": None, "cuda_error": None,
                               "reason": "symbol not visible through shared library"}
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
            "backends": identities, "runtime": runtime}


def compare(actual: np.ndarray, expected: np.ndarray, *, exact: bool) -> dict:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"Shape/dtype mismatch: {actual.shape}/{actual.dtype} vs {expected.shape}/{expected.dtype}")
    same = (actual == expected) | (np.isnan(actual) & np.isnan(expected))
    finite = np.isfinite(actual) & np.isfinite(expected)
    errors = np.abs(actual[finite].astype(np.float64) - expected[finite].astype(np.float64))
    # Quantized GEMMs require exact equality. Only ordinary FP32 GEMM / bias
    # reduction allow independent cuBLAS/NumPy accumulation-order differences.
    atol, rtol = (0., 0.) if exact else (3e-5, 3e-5)
    close = same.copy()
    close[finite] |= errors <= atol + rtol * np.abs(expected[finite].astype(np.float64))
    result = {"shape": list(actual.shape), "values": int(actual.size),
              "comparison": "exact_except_nan_payload_and_signed_zero" if exact else "fp32_allclose",
              "atol": atol, "rtol": rtol, "exact_mismatches": int(np.count_nonzero(~same)),
              "tolerance_failures": int(np.count_nonzero(~close)),
              "max_finite_abs_error": float(errors.max(initial=0))}
    if not bool(close.all()):
        first = np.argwhere(~close)[0].tolist()
        raise AssertionError(f"Numerical check failed at {first}: {result}")
    return result


def raw_tensor(codes: np.ndarray, scales: np.ndarray | int = 127) -> MXTensor:
    """Independent nibble fixture assembly; never invokes a GPU quantizer."""
    rows, k = codes.shape
    kp = ((k + 31) // 32) * 32
    padded = np.zeros((rows, kp), dtype=np.uint8)
    padded[:, :k] = codes
    packed = padded[:, ::2] | (padded[:, 1::2] << np.uint8(4))
    scale_bytes = np.broadcast_to(np.asarray(scales, dtype=np.uint8), (rows, kp // 32)).copy()
    return MXTensor(np.ascontiguousarray(packed), scale_bytes, (rows, k))


def oracle_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return gemm_mx(quantize_mx(np.ascontiguousarray(a)), quantize_mx(np.ascontiguousarray(b)))


class Runner:
    def __init__(self, device, stream, report: dict):
        self.device, self.stream, self.report = device, stream, report
        self.attempted = self.returned = self.completed = False
        self.pending = False

    def execute(self, function):
        self.attempted = True
        self.pending = True
        with torch.cuda.stream(self.stream):
            if torch.cuda.current_stream(self.device).cuda_stream != self.stream.cuda_stream:
                raise AssertionError("The requested nondefault CUDA stream is not current")
            result = function()
        self.returned = True
        self.stream.synchronize()
        self.completed = True
        self.pending = False
        return result

    def state(self) -> dict:
        return {"gpu_executed": True if self.completed else (None if self.attempted else False),
                "gpu_work_attempted": self.attempted,
                "gpu_work_call_returned": self.returned,
                "gpu_work_synchronized": self.completed,
                "all_attempted_work_synchronized": not self.pending}

    def case(self, name: str, function):
        record = {"name": name, "passed": False}
        self.report["cases"].append(record)
        try:
            record.update(function())
            record["passed"] = True
        except Exception as error:
            record["error_type"] = type(error).__name__
            record["error"] = str(error)
            raise
        print(f"PASS {name}", flush=True)

    def to_packed(self, tensor: MXTensor):
        return TorchMXTensor(torch.from_numpy(tensor.codes).to(self.device),
                             torch.from_numpy(tensor.scales).to(self.device), tensor.logical_shape)

    def packed_case(self, a: MXTensor, b: MXTensor, backend: str) -> dict:
        expected = gemm_mx(a, b)

        def work():
            out = gemm_packed(self.to_packed(a), self.to_packed(b), backend=backend)
            if out.device != self.device or out.dtype != torch.float32:
                raise AssertionError("CUDA backend returned the wrong device/dtype")
            # Queue a GPU consumer on the same nondefault stream, rather than
            # relying solely on a later synchronizing CPU read of the result.
            return out.clone()

        output = self.execute(work)
        return {"backend": backend, "logical_shapes": [list(a.logical_shape), list(b.logical_shape)],
                "source": "independent_cpu_packed_fixture", "nondefault_stream": True,
                "output": compare(output.cpu().numpy(), expected, exact=True)}

    def quantization_case(self, source: np.ndarray) -> dict:
        expected = quantize_mx(source)
        result = self.execute(lambda: quantize_mx_torch(torch.from_numpy(source).to(self.device)))
        codes = result.codes.cpu().numpy()
        scales = result.scales.cpu().numpy()
        if not np.array_equal(codes, expected.codes) or not np.array_equal(scales, expected.scales):
            raise AssertionError("GPU RNE codes/scales differ from the independent NumPy quantizer")
        return {"logical_shape": list(source.shape), "codes_and_scales_exact": True,
                "nibble_bytes": int(codes.size), "scale_bytes": int(scales.size),
                "nondefault_stream": True}

    def stochastic_case(self, seed: int) -> dict:
        host = np.tile(np.array([.75] * 31 + [6.], dtype=np.float32), (64, 1))

        def work():
            source = torch.from_numpy(host).to(self.device)
            return [quantize_mx_torch(source, rounding="stochastic",
                                     generator=torch.Generator(device=self.device).manual_seed(s))
                    for s in (seed, seed, seed + 1)]

        results = self.execute(work)
        codes = [r.codes.cpu().numpy() for r in results]
        if not np.array_equal(codes[0], codes[1]) or np.array_equal(codes[0], codes[2]):
            raise AssertionError("GPU stochastic rounding seed reproducibility failed")
        return {"same_seed_exact": True, "different_seed_changes_codes": True,
                "nondefault_stream": True, "distribution_claim": None}

    def linear_case(self, source: np.ndarray, weight: np.ndarray, grad: np.ndarray,
                    bias: np.ndarray, flags: tuple[bool, bool, bool], backend: str,
                    *, noncontiguous: bool = False) -> dict:
        f, d, w = flags
        flat = source.reshape(-1, source.shape[-1])
        upstream = grad.reshape(-1, weight.shape[0])
        expected = {
            "output": ((oracle_product(flat, weight) if f else flat @ weight.T) + bias).reshape(grad.shape),
            "dgrad": (oracle_product(upstream, weight.T) if d else upstream @ weight).reshape(source.shape),
            "wgrad": oracle_product(upstream.T, flat.T) if w else upstream.T @ flat,
            "bias_grad": upstream.sum(axis=0, dtype=np.float32),
        }
        recipe = MXRecipe(f, d, w, backend=backend)

        def work():
            if noncontiguous:
                # Construct transposed CUDA leaves explicitly, ensuring .to()
                # cannot accidentally make the test input contiguous.
                x = torch.from_numpy(np.ascontiguousarray(source.swapaxes(0, 1))).to(self.device).transpose(0, 1)
                wt = torch.from_numpy(np.ascontiguousarray(weight.T)).to(self.device).T
                g = torch.from_numpy(np.ascontiguousarray(grad.swapaxes(0, 1))).to(self.device).transpose(0, 1)
                if x.is_contiguous() or wt.is_contiguous() or g.is_contiguous():
                    raise AssertionError("Noncontiguous CUDA fixture unexpectedly became contiguous")
            else:
                x = torch.from_numpy(np.ascontiguousarray(source)).to(self.device)
                wt = torch.from_numpy(np.ascontiguousarray(weight)).to(self.device)
                g = torch.from_numpy(np.ascontiguousarray(grad)).to(self.device)
            x = x.detach().requires_grad_(True)
            wt = wt.detach().requires_grad_(True)
            b = torch.from_numpy(bias).to(self.device).requires_grad_(True)
            y = mxfp4_linear(x, wt, b, recipe=recipe)
            y.backward(g)
            results = {"output": y.detach().clone(), "dgrad": x.grad,
                       "wgrad": wt.grad, "bias_grad": b.grad}
            if any(value is None or value.device != self.device or value.dtype != torch.float32
                   for value in results.values()):
                raise AssertionError("Missing gradient, CPU fallback or loss of FP32 master/gradient dtype")
            return results

        outputs = self.execute(work)
        exact = {"output": f, "dgrad": d, "wgrad": w, "bias_grad": False}
        metrics = {key: compare(value.detach().cpu().numpy(), expected[key], exact=exact[key])
                   for key, value in outputs.items()}
        return {"recipe": asdict(recipe), "input_shape": list(source.shape),
                "weight_shape": list(weight.shape), "noncontiguous": noncontiguous,
                "nondefault_stream": True, "gradient_contract": "declared_first_order_surrogate",
                "comparisons": metrics}

    def rejection_case(self) -> dict:
        def work():
            a = quantize_mx_torch(torch.ones(2, 5, device=self.device))
            b = quantize_mx_torch(torch.ones(3, 5, device=self.device))
            try:
                gemm_packed(a, b, backend="reference")
            except ValueError as error:
                if "requires CPU" not in str(error):
                    raise
            else:
                raise AssertionError("GPU packed tensors silently fell back to the CPU reference backend")

        self.execute(work)
        return {"gpu_reference_fallback_rejected": True}


def run_suite(runner: Runner, seed: int) -> None:
    rng = np.random.default_rng(seed)
    shapes = [(3, 5, 1), (7, 5, 31), (17, 19, 33), (65, 33, 65)]
    for m, n, k in shapes:
        blocks = (k + 31) // 32
        a = raw_tensor(rng.integers(0, 16, (m, k), dtype=np.uint8),
                       rng.integers(124, 131, (m, blocks), dtype=np.uint8))
        b = raw_tensor(rng.integers(0, 16, (n, k), dtype=np.uint8),
                       rng.integers(124, 131, (n, blocks), dtype=np.uint8))
        for backend in ("cuda", "cuda_tiled"):
            runner.case(f"packed_{backend}_{m}_{n}_{k}",
                        lambda a=a, b=b, backend=backend: runner.packed_case(a, b, backend))

    # Full code and E8M0 sentinels reach the native wrapper independently of
    # the finite-input quantizer. NaN scales must propagate even for zero dots.
    for title, a, b in (
        ("all_codes", raw_tensor(np.arange(16, dtype=np.uint8)[:, None]),
         raw_tensor(np.arange(16, dtype=np.uint8)[:, None])),
        ("scale_edges", raw_tensor(np.array([[0], [1], [4], [2], [0]], dtype=np.uint8),
                                   np.array([[255], [191], [52], [0], [254]], dtype=np.uint8)),
         raw_tensor(np.array([[0], [1], [4], [2], [0]], dtype=np.uint8),
                    np.array([[127], [191], [52], [254], [255]], dtype=np.uint8))),
    ):
        for backend in ("cuda", "cuda_tiled"):
            runner.case(f"packed_{backend}_{title}",
                        lambda a=a, b=b, backend=backend: runner.packed_case(a, b, backend))

    levels = np.array([0., .5, 1., 1.5, 2., 3., 4., 6.,
                       -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=np.float32)
    ties = np.array([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], dtype=np.float32)
    neighbors = np.concatenate([ties, np.nextafter(ties, np.float32(0)),
                                np.nextafter(ties, np.float32(np.inf)), [6.]]).astype(np.float32)[None, :]
    runner.case("gpu_rne_all_codes", lambda: runner.quantization_case(levels[None, :]))
    runner.case("gpu_rne_midpoints_and_neighbors", lambda: runner.quantization_case(neighbors))
    for exponent in (-127, -126, -64, 0, 64, 125):
        source = np.ldexp(levels, exponent)[None, :]
        runner.case(f"gpu_rne_exponent_{exponent}", lambda source=source: runner.quantization_case(source))
    extrema = np.array([[np.nextafter(np.float32(0), np.float32(1)), 0., -0.],
                        [np.finfo(np.float32).max, 0., -np.finfo(np.float32).max]], dtype=np.float32)
    runner.case("gpu_rne_fp32_extremes", lambda: runner.quantization_case(extrema))
    runner.case("gpu_stochastic_seed", lambda: runner.stochastic_case(seed))

    linear_inputs = []
    for m, n, k in shapes:
        linear_inputs.append((f"tail_{m}_{n}_{k}", rng.normal(size=(m, k)).astype(np.float32),
                              rng.normal(size=(n, k)).astype(np.float32),
                              rng.normal(size=(m, n)).astype(np.float32),
                              rng.normal(size=(n,)).astype(np.float32), False))
    linear_inputs.append(("noncontiguous_batch", rng.normal(size=(2, 5, 33)).astype(np.float32),
                          rng.normal(size=(3, 33)).astype(np.float32),
                          rng.normal(size=(2, 5, 3)).astype(np.float32),
                          rng.normal(size=(3,)).astype(np.float32), True))
    x = np.ones((33, 33), dtype=np.float32)
    x[0, :] = 64.
    g = x.copy()
    # Verify the fixture really distinguishes original-FP32 requantization
    # from simply reusing a row-quantized representation after transposition.
    if np.array_equal(oracle_product(g, x.T), g @ x):
        raise AssertionError("Transpose-axis negative control is ineffective")
    linear_inputs.append(("original_fp32_transpose", x, x.copy(), g,
                          np.zeros(33, dtype=np.float32), False))
    for title, x, w, g, bias, noncontiguous in linear_inputs:
        for backend in ("cuda", "cuda_tiled"):
            for flags in itertools.product((False, True), repeat=3):
                arm = "".join("1" if flag else "0" for flag in flags)
                runner.case(f"linear_{backend}_{title}_fdw{arm}",
                            lambda x=x, w=w, g=g, bias=bias, flags=flags, backend=backend,
                            noncontiguous=noncontiguous: runner.linear_case(
                                x, w, g, bias, flags, backend, noncontiguous=noncontiguous))
    runner.case("reject_gpu_to_reference_fallback", runner.rejection_case)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0", help="Explicit CUDA device; CPU fallback is unavailable")
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a new evidence path")
    if not args.device.startswith("cuda:") or not args.device[5:].isdigit():
        parser.error("device must name an explicit CUDA index, for example cuda:0")
    if args.seed < 0 or args.seed > 2**32 - 1:
        parser.error("seed must be an integer in [0, 2**32 - 1]")
    report = {
        "schema_version": 1, "purpose": "real SM70 PyTorch MXFP4 integration correctness",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pending", "passed": False, "gpu_executed": False,
        "gpu_work_attempted": False, "gpu_work_call_returned": False,
        "gpu_work_synchronized": False, "all_attempted_work_synchronized": True,
        "device": args.device, "seed": args.seed,
        "python": platform.python_version(), "numpy": np.__version__,
        "torch": None if torch is None else torch.__version__,
        "source_sha256": source_hashes(), "cases": [],
        "minimum_free_bytes_after_context": 512 * 1024 * 1024,
        "largest_linear_shape": {"M": 65, "N": 33, "K": 65},
        "performance_claim": None, "full_training_executed": False,
        "llm_convergence_validated": False, "cpu_fallback": False,
    }
    runner = None
    exit_code = 1
    try:
        if torch is None:
            raise Blocked("PyTorch is not installed")
        if not torch.cuda.is_available():
            raise Blocked("CUDA is unavailable in this PyTorch process; no CPU fallback was attempted")
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        report["gpu"] = {
            "name": properties.name, "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "actual_SM70": (properties.major, properties.minor) == (7, 0),
            "actual_V100": (properties.major, properties.minor) == (7, 0) and "V100" in properties.name,
            "torch_cuda_build_version": torch.version.cuda,
        }
        if (properties.major, properties.minor) != (7, 0):
            raise Blocked("This integration check requires actual SM70 hardware")
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        report["gpu"].update(free_bytes_after_context=free_bytes, runtime_total_bytes=total_bytes)
        if free_bytes < report["minimum_free_bytes_after_context"]:
            raise Blocked("Requires at least 512 MiB free after CUDA context initialization; no services are stopped")
        report["library"] = library_metadata()
        torch.set_num_threads(2)
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        report["matmul_allow_tf32"] = False
        stream = torch.cuda.Stream(device=device)
        if stream.cuda_stream == torch.cuda.default_stream(device).cuda_stream:
            raise AssertionError("The check did not obtain a nondefault CUDA stream")
        report["nondefault_stream"] = True
        torch.cuda.reset_peak_memory_stats(device)
        runner = Runner(device, stream, report)
        run_suite(runner, args.seed)
        torch.cuda.synchronize(device)
        report["gpu"].update(max_memory_allocated_bytes=torch.cuda.max_memory_allocated(device),
                              max_memory_reserved_bytes=torch.cuda.max_memory_reserved(device))
        report["source_unchanged_during_check"] = source_hashes() == report["source_sha256"]
        if not report["source_unchanged_during_check"]:
            raise RuntimeError("Source changed during the integration check; rerun against a fixed snapshot")
        if sha256(Path(report["library"]["path"])) != report["library"]["sha256"]:
            raise RuntimeError("CUDA library changed during the integration check")
        report["status"], report["passed"] = "passed", True
        exit_code = 0
    except Blocked as error:
        report.update(status="blocked", error_type=type(error).__name__, error=str(error))
        exit_code = 2
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        exit_code = 1
    finally:
        if runner is not None:
            report.update(runner.state())
        report["completed_cases"] = sum(case["passed"] for case in report["cases"])
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"status": report["status"], "passed": report["passed"],
                          "gpu_executed": report["gpu_executed"],
                          "completed_cases": report["completed_cases"],
                          "output": str(args.output)}), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
