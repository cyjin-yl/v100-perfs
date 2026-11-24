"""Independent GPU check of the opt-in C11 tile-local certificate dispatcher.

Uses the public native ABI, an independent Python-integer certificate, and the
NumPy ordered-MX32 output oracle.  No timing is performed.  C2/C10 controls use
the same fixtures and must report tile dispatch disabled without changing the
global certificate. Each case also checks the unconditional GPU tiled oracle
in full against NumPy, before that oracle may be used for large benchmarks.
Run only from a frozen source/build snapshot.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.mxfp4_training.check_torch_cuda import Blocked, sha256
from tools.mxfp4_training.check_torch_certified import library_metadata, source_hashes as base_sources
from tools.mxfp4_training.tile_certified_fixtures import (
    benchmark_fixture, correctness_fixtures, expected_certificates, uniform_fixture,
)

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from tools.mxfp4_training import torch_ops as ops


class TileInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in
                ("abi_version", "enabled", "metadata_valid", "tile_m", "tile_n", "reserved")]
    _fields_ += [(name, ctypes.c_uint64) for name in
                 ("tiles_m", "tiles_n", "total_tiles", "fast_tiles", "fallback_tiles", "submitted_calls")]


def source_hashes():
    root = Path(__file__).resolve().parent
    names = ("tile_certified_fixtures.py", "check_tile_certified.py", "bench_tile_certified.py")
    return {**base_sources(), **{name: sha256(root / name) for name in names}}


def exact_output(actual, expected):
    actual = np.ascontiguousarray(actual)
    expected = np.ascontiguousarray(expected)
    if actual.dtype != np.float32 or expected.dtype != np.float32 or actual.shape != expected.shape:
        raise AssertionError("Output dtype/shape differs from the independent oracle")
    # Finite values and signed infinities must match every bit; NaN payloads
    # are intentionally not part of the existing ordered-MX32 contract. Bound
    # comparison scratch for an 8192^2 output instead of making full masked
    # copies of both 256-MiB matrices. Every value still participates.
    actual_flat, expected_flat = actual.reshape(-1), expected.reshape(-1)
    mismatch = nan_count = infinite_count = zero_count = 0
    comparison_chunk = 1 << 20
    for start in range(0, actual.size, comparison_chunk):
        got = actual_flat[start:start + comparison_chunk]
        wanted = expected_flat[start:start + comparison_chunk]
        nan = np.isnan(wanted)
        if not np.array_equal(np.isnan(got), nan):
            raise AssertionError("NaN locations differ")
        mismatch += int(np.count_nonzero(got.view(np.uint32)[~nan] != wanted.view(np.uint32)[~nan]))
        zero = got == 0
        if np.any(got.view(np.uint32)[zero] != 0):
            raise AssertionError("Output contains a noncanonical negative zero")
        nan_count += int(np.count_nonzero(nan))
        infinite_count += int(np.count_nonzero(np.isinf(wanted)))
        zero_count += int(np.count_nonzero(zero))
    if mismatch:
        raise AssertionError(f"Ordered-MX32 output differs at {mismatch} values")
    return {"compared_values": int(actual.size), "bit_mismatches": 0,
            "nan_values": nan_count, "infinite_values": infinite_count,
            "canonical_positive_zero_values": zero_count,
            "nan_payload_bits_compared": False,
            "maximum_comparison_chunk_values": comparison_chunk,
            "output_fp32_sha256": hashlib.sha256(memoryview(actual).cast("B")).hexdigest()}


class GuardedInputs:
    def __init__(self, fixture, device, offset=0):
        self.fixture = fixture
        self.storages, self.views, self.expected_storage = [], [], []
        self.guard = 32
        for index, host in enumerate(fixture.arrays()):
            # Misalign codes only; scales have no vector alignment requirement.
            extra = offset if index in (0, 2) else 0
            start = self.guard + extra
            expected = np.full(start + host.size + self.guard, 0xA7, dtype=np.uint8)
            expected[start:start + host.size] = host.reshape(-1)
            storage = torch.from_numpy(expected).to(device)
            self.storages.append(storage)
            self.views.append(storage[start:start + host.size].view(host.shape))
            self.expected_storage.append(expected)

    def update(self, fixture):
        if (fixture.m, fixture.n, fixture.logical_k) != (
                self.fixture.m, self.fixture.n, self.fixture.logical_k):
            raise ValueError("Storage-reuse fixture changed shape")
        for host, view, expected in zip(fixture.arrays(), self.views, self.expected_storage):
            view.copy_(torch.from_numpy(host).to(view.device))
            start = view.storage_offset()
            expected[start:start + host.size] = host.reshape(-1)
        self.fixture = fixture

    def verify(self):
        actual = {}
        for name, storage, view, expected in zip(
                ("a_codes", "a_scales", "b_codes", "b_scales"),
                self.storages, self.views, self.expected_storage):
            got = storage.cpu().numpy()
            if not np.array_equal(got, expected):
                raise AssertionError("Native call changed packed input bytes or their guards")
            actual[name] = hashlib.sha256(view.cpu().numpy().tobytes()).hexdigest()
        if actual != self.fixture.hashes():
            raise AssertionError("Actual device input SHA differs from generated input")
        return actual


class GuardedOutput:
    def __init__(self, m, n, device, offset=0):
        self.start = 16 + offset
        self.stop = self.start + m * n
        self.sentinel = 0x5A17C0DE
        self.storage = torch.full((self.stop + 16,), self.sentinel, dtype=torch.int32, device=device)
        self.view = self.storage[self.start:self.stop].view(torch.float32).view(m, n)

    def reset(self):
        self.storage.fill_(self.sentinel)

    def verify(self, expected):
        raw = self.storage.cpu().numpy()
        if np.any(raw[:self.start] != self.sentinel) or np.any(raw[self.stop:] != self.sentinel):
            raise AssertionError("Native call wrote outside the output tensor")
        result = exact_output(raw[self.start:self.stop].view(np.float32).reshape(expected.shape), expected)
        result["output_guards_unchanged"] = True
        result["actual_tensor_data_ptr"] = self.view.data_ptr()
        result["actual_storage_data_ptr"] = self.storage.data_ptr()
        result["actual_storage_offset_bytes"] = self.start * 4
        return result


def submit_tiled_oracle(lib, fixture, inputs, output, stream):
    """Enqueue the unconditional, ordered-MX32 GPU oracle on these exact buffers.

    It is separate from the candidate certified dispatch, but shares the
    candidate's exact fallback source. Independent semantic evidence therefore
    comes from the complete NumPy checks below and the large-case CPU samples.
    """
    lib.mxfp4_gemm_tiled_cuda.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int64] * 3 + [ctypes.c_void_p]
    lib.mxfp4_gemm_tiled_cuda.restype = ctypes.c_int
    for tensor in [*inputs.storages, *inputs.views, output.storage, output.view]:
        tensor.record_stream(stream)
    code = lib.mxfp4_gemm_tiled_cuda(
        *(tensor.data_ptr() for tensor in inputs.views), output.view.data_ptr(),
        fixture.m, fixture.n, fixture.padded_k, stream.cuda_stream)
    if code:
        raise RuntimeError(f"Unconditional tiled oracle launch failed: cudaError_t={code}")


def check_tiled_oracle(fixture, lib, inputs, device, stream, expected, *, offset=0):
    output = GuardedOutput(fixture.m, fixture.n, device, offset)
    submit_tiled_oracle(lib, fixture, inputs, output, stream)
    stream.synchronize()
    return {"passed": True, "backend": "mxfp4_gemm_tiled_cuda_unconditional",
            "independent_semantic_oracle": "full_NumPy_ordered_MX32",
            "shares_exact_fallback_source_with_candidate": True,
            "all_outputs": output.verify(expected)}


class NativeContext:
    def __init__(self, lib, fixture, stream, variant):
        self.lib, self.stream, self.fixture, self.variant = lib, stream, fixture, variant
        self.context, self.calls = ctypes.c_void_p(), 0
        self.lib.mxfp4_certified_get_tile_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(TileInfo)]
        self.lib.mxfp4_certified_get_tile_info.restype = ctypes.c_int
        if ctypes.sizeof(TileInfo) != 72 or TileInfo.submitted_calls.offset != 64:
            raise RuntimeError("Tile-info ABI layout does not match the declared 72 bytes")
        try:
            self.checked(self.lib.mxfp4_certified_create(
                fixture.m, fixture.n, fixture.padded_k, ctypes.byref(self.context)), "create")
            if not self.context.value:
                raise AssertionError("Native create returned an empty context")
            global_info, tiles = self.info()
            if global_info["metadata_valid"] or tiles["metadata_valid"]:
                raise AssertionError("Fresh context exposes stale certificate data")
        except BaseException:
            self.close()
            raise

    def checked(self, code, operation):
        if code:
            raise ops._certified_error(self.lib, operation, code)

    def call(self, inputs, output):
        for tensor in [*inputs.storages, *inputs.views, output.storage, output.view]:
            tensor.record_stream(self.stream)
        self.checked(self.lib.mxfp4_certified_gemm(
            self.context, *(tensor.data_ptr() for tensor in inputs.views), output.view.data_ptr(),
            self.stream.cuda_stream, None), "gemm")
        self.calls += 1

    def info(self):
        base, tile = ops._CertifiedInfo(), TileInfo()
        self.checked(self.lib.mxfp4_certified_get_info(self.context, ctypes.byref(base)), "get_info")
        self.checked(self.lib.mxfp4_certified_get_tile_info(self.context, ctypes.byref(tile)), "get_tile_info")
        global_info = {name: int(getattr(base, name)) for name, _ in base._fields_}
        tiles = {name: int(getattr(tile, name)) for name, _ in tile._fields_}
        if (global_info["abi_version"] != 1 or tiles["abi_version"] != 1
                or global_info["device"] != self.stream.device.index
                or global_info["submitted_calls"] != self.calls or tiles["submitted_calls"] != self.calls
                or global_info["metadata_valid"] != int(bool(self.calls))
                or tiles["metadata_valid"] != int(bool(self.calls))
                or (global_info["m"], global_info["n"], global_info["padded_k"]) !=
                (self.fixture.m, self.fixture.n, self.fixture.padded_k)):
            raise AssertionError("Native diagnostic identity or submitted-call count differs")
        if tiles["enabled"] != int(self.variant == 11) or tiles["reserved"]:
            raise AssertionError("Tile-dispatch enablement or reserved field differs")
        if self.variant != 11 and any(tiles[k] for k in
                ("tile_m", "tile_n", "tiles_m", "tiles_n", "total_tiles", "fast_tiles", "fallback_tiles")):
            raise AssertionError("A control variant claims tile-local execution")
        return global_info, tiles

    def close(self):
        if self.context.value:
            pointer, self.context = self.context, ctypes.c_void_p()
            self.checked(self.lib.mxfp4_certified_destroy(pointer), "destroy")


def assert_diagnostics(global_info, tile_info, expected, variant):
    for key, value in expected["global"].items():
        if global_info[key] != value:
            raise AssertionError(f"CPU/global certificate differs for {key}: {global_info[key]} vs {value}")
    if variant == 11:
        if (tile_info["tile_m"], tile_info["tile_n"]) != (128, 128):
            raise AssertionError("Unexpected output tile dimensions")
        for key in ("tiles_m", "tiles_n", "total_tiles", "fast_tiles", "fallback_tiles"):
            if tile_info[key] != expected[key]:
                raise AssertionError(f"CPU/tile certificate differs for {key}")
    return {"global_certificate": global_info, "tile_diagnostics": tile_info,
            "cpu_expected_tile_mask": expected["tile_mask"], "all_integer_diagnostics_match": True}


def check_fixture(fixture, lib, device, stream, variant, *, offset=0):
    expected, decision = fixture.oracle(), expected_certificates(fixture)
    uploaded = GuardedInputs(fixture, device, offset)
    output = GuardedOutput(fixture.m, fixture.n, device, int(bool(offset)))
    context = NativeContext(lib, fixture, stream, variant)
    try:
        before = uploaded.verify()
        oracle_check = check_tiled_oracle(fixture, lib, uploaded, device, stream,
                                         expected, offset=int(bool(offset)))
        context.call(uploaded, output)
        global_info, tiles = context.info()
        diagnostic = assert_diagnostics(global_info, tiles, decision, variant)
        compared = output.verify(expected)
        after = uploaded.verify()
        if after != before:
            raise AssertionError("Input fingerprint changed")
        return {"name": fixture.name, "passed": True, "shape": [fixture.m, fixture.n, fixture.logical_k],
                "padded_K": fixture.padded_k, "input_byte_offset": offset,
                "actual_input_sha256_before": before, "actual_input_sha256_after": after,
                "output": compared, "unconditional_tiled_oracle_check": oracle_check, **diagnostic}
    finally:
        context.close()


def check_reuse(lib, device, stream, variant):
    fixture = uniform_fixture("same_storage_reuse", m=384, n=384)
    uploaded, output = GuardedInputs(fixture, device), GuardedOutput(fixture.m, fixture.n, device)
    context = NativeContext(lib, fixture, stream, variant)
    records = []
    pointers = [tensor.data_ptr() for tensor in uploaded.views]
    try:
        for iteration, mode in enumerate(("accepted", "mixed", "all_rejected", "nan", "accepted")):
            current = uniform_fixture(mode, m=384, n=384)
            if mode == "mixed":
                current.a_scales[128, 0] = 140
                current.b_scales[256, 0] = 140
            elif mode == "all_rejected":
                current.a_scales[:, 0] = 140
            elif mode == "nan":
                current.b[256, :] = 0
                current.b_scales[256, 0] = 255
            uploaded.update(current)
            output.reset()
            expected, decision = current.oracle(), expected_certificates(current)
            oracle_check = check_tiled_oracle(current, lib, uploaded, device, stream, expected)
            context.call(uploaded, output)
            global_info, tiles = context.info()
            diagnostic = assert_diagnostics(global_info, tiles, decision, variant)
            if pointers != [tensor.data_ptr() for tensor in uploaded.views]:
                raise AssertionError("Fixture reuse changed input storage")
            records.append({"name": "reuse_" + mode, "iteration": iteration, "passed": True,
                            "submitted_calls": iteration + 1, "same_input_storage": True,
                            "actual_input_sha256": uploaded.verify(),
                            "shape": [current.m, current.n, current.logical_k],
                            "output": output.verify(expected),
                            "unconditional_tiled_oracle_check": oracle_check, **diagnostic})
        return records
    finally:
        context.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-evidence", type=Path, required=True)
    parser.add_argument("--expected-certified-variant", type=int, choices=(2, 10, 11), default=11)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; use a new evidence path")
    report = {"schema_version": 1, "status": "pending", "passed": False,
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "gpu_executed": False,
              "performance_measured": False, "real_training_tensor": False,
              "python": platform.python_version(), "numpy": np.__version__,
              "torch": None if torch is None else torch.__version__, "source_sha256": source_hashes(),
              "device": args.device, "expected_certified_variant": args.expected_certified_variant,
              "cases": []}
    exit_code, device = 1, None
    try:
        if torch is None or not torch.cuda.is_available():
            raise Blocked("An actual V100 CUDA runtime with PyTorch is required")
        if not args.device.startswith("cuda:") or not args.device[5:].isdigit():
            raise ValueError("Use an explicit CUDA device ordinal")
        device = torch.device(args.device)
        properties = torch.cuda.get_device_properties(device)
        report["gpu"] = {"name": properties.name, "compute_capability": [properties.major, properties.minor],
                         "total_memory_bytes": properties.total_memory, "torch_cuda_build_version": torch.version.cuda}
        if (properties.major, properties.minor) != (7, 0) or "V100" not in properties.name:
            raise Blocked("Require a real V100 SM70 for this acceptance suite")
        report["library"] = library_metadata(args.compile_evidence, args.expected_certified_variant,
                                             report["source_sha256"])
        report["library"]["context_lifecycle"] = "native_explicit_context_with_reuse_and_separate_diagnostics"
        lib = ops._load_library(report["library"]["path"], "cuda_certified")
        if not hasattr(lib, "mxfp4_certified_get_tile_info"):
            raise Blocked("Rebuild the snapshot with the optional tile diagnostics ABI")
        with torch.cuda.device(device), torch.no_grad():
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                report["gpu_executed"] = True
                fixtures = correctness_fixtures()
                fixtures.append(benchmark_fixture("block", 257, 259, 96, 20261124))
                for fixture in fixtures:
                    report["cases"].append(check_fixture(fixture, lib, device, stream,
                                                        args.expected_certified_variant))
                    print("passed " + fixture.name, flush=True)
                for index in (1, 4, 5):
                    fixture = fixtures[index]
                    report["cases"].append(check_fixture(fixture, lib, device, stream,
                                                        args.expected_certified_variant, offset=1))
                report["cases"].extend(check_reuse(lib, device, stream, args.expected_certified_variant))
                stream.synchronize()
        report["completed_cases"] = len(report["cases"])
        report["unconditional_tiled_oracle_verified_cases"] = sum(
            case["unconditional_tiled_oracle_check"]["passed"] for case in report["cases"])
        report["source_unchanged_during_check"] = source_hashes() == report["source_sha256"]
        report["library_unchanged_during_check"] = sha256(Path(report["library"]["path"])) == report["library"]["sha256"]
        if not report["source_unchanged_during_check"] or not report["library_unchanged_during_check"]:
            raise AssertionError("Source or library changed during the GPU check")
        report.update(status="passed", passed=True)
        exit_code = 0
    except Blocked as error:
        report.update(status="blocked", error=str(error))
        exit_code = 2
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        report["completed_cases"] = len(report["cases"])
        if device is not None and report["gpu_executed"]:
            try:
                torch.cuda.synchronize(device)
                report["all_attempted_work_synchronized"] = True
            except BaseException as error:
                report.update(status="failed", passed=False, synchronization_error=str(error))
                exit_code = 1
        report["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "completed_cases": report["completed_cases"],
                      "gpu_executed": report["gpu_executed"], "output": str(args.output)}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
