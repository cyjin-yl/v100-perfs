"""Independent NumPy byte checks for the opt-in CUDA RNE quantizer.

The ordinary Torch quantizer (including stochastic rounding) must remain
unchanged. All CUDA fixtures use a nondefault stream. Native direct checks
start codes/scales at the bitwise inverse of the oracle, exposing unwritten
bytes; wrapper checks exercise the public finite-validation/flag-sync path.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import ctypes
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mxfp4_training.reference import MXTensor, quantize_mx, gemm_mx, dequantize_mx
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from tools.mxfp4_training import torch_ops as ops

ORIGINAL_TORCH_QUANTIZER_SHA = "1a74501035e2b99f5ccc8335037f8d9d169416e5e6210b9ac9d193c5ca1e5910"
MINIMUM_FREE = 1024 ** 3
RESERVE_BYTES = 512 * 1024 ** 2
CHECK_ALLOCATION_BUDGET = 128 * 1024 ** 2


class Blocked(RuntimeError):
    pass


def need(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parent
    names = ['check_quantize_cuda.py', 'reference.py', 'torch_ops.py', 'build_cuda.py']
    names += [str(p.relative_to(root)) for p in sorted((root / 'csrc').iterdir()) if p.is_file()]
    return {name: sha256(root / name) for name in names}


def original_quantizer_hash():
    path = Path(__file__).resolve().parent / 'torch_ops.py'
    text = path.read_text()
    function = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'quantize_mx_torch')
    return hashlib.sha256(ast.get_source_segment(text, function).encode()).hexdigest()


def library_metadata(manifest_path):
    value = os.environ.get('MXFP4_CUDA_LIBRARY')
    if not value:
        raise Blocked('Set MXFP4_CUDA_LIBRARY to the shared library built with --with-quantizer-rne')
    path = Path(value).expanduser().resolve(strict=True)
    manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    need(manifest.get('passed') is True and manifest.get('quantizer_rne_enabled') is True,
         'Compile manifest must identify a passing opt-in CUDA RNE build')
    need(manifest.get('quantizer_rne_backend') == 'sm70_mx32_rne_warp_fp32_v1',
         'Compile manifest has an unexpected quantizer backend')
    need(manifest.get('tiled_variant') == 1, 'Use the unchanged tiled1 GEMM control')
    artifact = {'sha256': sha256(path), 'bytes': path.stat().st_size}
    need(manifest.get('artifacts', {}).get(path.name) == artifact, 'Library bytes differ from build manifest')
    root = Path(__file__).resolve().parent
    expected = {str(p.relative_to(root)): sha256(p) for p in sorted((root / 'csrc').iterdir()) if p.is_file()}
    need(manifest.get('source_sha256') == expected, 'Native sources differ from compile manifest')
    mapping = root.parents[1] / 'evidence/benchmarks/sm70-packing/study/simulation/round3/cuda/fragment_mapping.hpp'
    need(manifest.get('fragment_mapping_sha256') == sha256(mapping), 'Fragment mapping differs from build')
    jobs = [j for j in manifest.get('jobs', []) if j.get('name') == 'shared_library']
    need(len(jobs) == 1 and jobs[0].get('returncode') == 0, 'Shared-library compile did not pass')
    need(sha256(manifest_path.parent / jobs[0]['log']) == jobs[0]['log_sha256'], 'Shared compile log changed')
    lib = ctypes.CDLL(str(path))
    lib.mxfp4_quantize_rne_cuda_abi_version.restype = ctypes.c_int
    lib.mxfp4_quantize_rne_cuda_backend.restype = ctypes.c_char_p
    need(lib.mxfp4_quantize_rne_cuda_abi_version() == 1, 'Unsupported native RNE ABI')
    need(lib.mxfp4_quantize_rne_cuda_backend() == b'sm70_mx32_rne_warp_fp32_v1',
         'Unexpected native RNE backend identity')
    lib.mxfp4_quantize_rne_cuda.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 4 + [ctypes.c_void_p]
    lib.mxfp4_quantize_rne_cuda.restype = ctypes.c_int
    need(original_quantizer_hash() == ORIGINAL_TORCH_QUANTIZER_SHA,
         'Ordinary Torch quantizer changed; this candidate must preserve its RNE/SR implementation')
    return lib, {'path': str(path), **artifact, 'compile_evidence': str(manifest_path),
        'compile_evidence_sha256': sha256(manifest_path), 'native_sources_match': True,
        'fragment_mapping_matches': True, 'quantizer_rne_enabled': True, 'quantizer_abi_version': 1,
        'quantizer_backend': lib.mxfp4_quantize_rne_cuda_backend().decode(),
        'tiled_variant': 1, 'certified_enabled': manifest.get('certified_enabled') is True,
        'certified_variant': manifest.get('certified_variant'),
        'nvcc': jobs[0]['argv'][0], 'original_torch_quantizer_source_sha256': ORIGINAL_TORCH_QUANTIZER_SHA}


def host_array(tensor):
    return tensor.detach().resolve_conj().resolve_neg().cpu().numpy()


def packed_metric(actual, expected):
    need(tuple(actual.logical_shape) == expected.logical_shape, 'Logical packed shape differs')
    actual.validate()
    codes, scales = host_array(actual.codes), host_array(actual.scales)
    need(codes.dtype == scales.dtype == np.uint8, 'Packed dtype must be uint8')
    need(codes.shape == expected.codes.shape and scales.shape == expected.scales.shape, 'Packed storage shape differs')
    cm = int(np.count_nonzero(codes != expected.codes)); sm = int(np.count_nonzero(scales != expected.scales))
    result = {'logical_shape': list(expected.logical_shape), 'padded_K': expected.padded_k,
        'code_bytes': int(codes.size), 'scale_bytes': int(scales.size),
        'code_byte_mismatches': cm, 'scale_byte_mismatches': sm,
        'expected_codes_sha256': hashlib.sha256(expected.codes.tobytes()).hexdigest(),
        'expected_scales_sha256': hashlib.sha256(expected.scales.tobytes()).hexdigest()}
    need(cm == sm == 0, 'Codes/scales differ from independent NumPy oracle: ' + json.dumps(result))
    return result


def numeric_metric(actual, expected, exact):
    actual = np.asarray(actual); expected = np.asarray(expected, dtype=np.float32)
    need(actual.dtype == np.float32 and actual.shape == expected.shape, 'Gradient/output dtype or shape differs')
    same = (actual == expected) | (np.isnan(actual) & np.isnan(expected))
    finite = np.isfinite(actual) & np.isfinite(expected)
    delta = np.abs(actual[finite].astype(np.float64) - expected[finite].astype(np.float64))
    atol = rtol = 0.0 if exact else 3e-5
    close = same.copy(); close[finite] |= delta <= atol + rtol * np.abs(expected[finite].astype(np.float64))
    result = {'shape': list(actual.shape), 'values': int(actual.size),
        'comparison': 'exact_except_nan_payload_and_signed_zero' if exact else 'fp32_allclose',
        'atol': atol, 'rtol': rtol, 'exact_mismatches': int(np.count_nonzero(~same)),
        'tolerance_failures': int(np.count_nonzero(~close)), 'max_finite_abs_error': float(delta.max(initial=0))}
    need(bool(close.all()), 'Output/gradient mismatch: ' + json.dumps(result))
    return result


def midpoint_matrix():
    midpoints = np.asarray([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], dtype=np.float64)
    rows, exponents = [], []
    for exponent in range(-127, 126):
        for midpoint in midpoints:
            tie = np.float32(np.ldexp(midpoint, exponent))
            lo = np.nextafter(tie, np.float32(-np.inf)); hi = np.nextafter(tie, np.float32(np.inf))
            anchor = np.float32(np.ldexp(6., exponent))
            row = np.zeros(32, dtype=np.float32)
            row[:10] = [lo, tie, hi, -lo, -tie, -hi, anchor, -anchor, np.float32(0), np.float32(-0.)]
            rows.append(row); exponents.append(exponent)
    result = np.asarray(rows, dtype=np.float32)
    oracle = quantize_mx(result)
    need(np.array_equal(oracle.scales[:, 0], np.asarray(exponents) + 127), 'Midpoint anchors did not select all 253 scales')
    unpacked = np.empty((len(rows), 32), dtype=np.uint8)
    unpacked[:, ::2] = oracle.codes & 15; unpacked[:, 1::2] = oracle.codes >> 4
    for row in range(len(rows)):
        midpoint_index = row % 7
        even_code = midpoint_index if midpoint_index % 2 == 0 else midpoint_index + 1
        need(unpacked[row, 1] == even_code and unpacked[row, 4] == (even_code | 8), 'Independent RNE tie fixture malformed')
    return result


def exponent_boundary_matrix():
    rows = []
    for exponent in range(-149, 128):
        power = np.float32(np.ldexp(1., exponent))
        for amax in (np.nextafter(power, np.float32(0)), power, np.nextafter(power, np.float32(np.inf))):
            row = np.zeros(32, dtype=np.float32)
            row[:6] = [amax, -amax, np.float32(amax * .5), np.float32(-amax * .5), 0., -0.]
            rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def finite_bits(rng, shape):
    wanted = math.prod(shape); parts = []; count = 0
    while count < wanted:
        raw = rng.integers(0, 2**32, size=wanted, dtype=np.uint32)
        good = raw[(raw & np.uint32(0x7f800000)) != np.uint32(0x7f800000)]
        parts.append(good); count += len(good)
    return np.concatenate(parts)[:wanted].view(np.float32).reshape(shape).copy()


def to_layout(array, device, layout):
    a = np.ascontiguousarray(array, dtype=np.float32); rows, k = a.shape
    if layout == 'contiguous':
        return torch.from_numpy(a).to(device)
    if layout == 'transpose':
        return torch.from_numpy(np.ascontiguousarray(a.T)).to(device).T
    if layout == 'positive_strides_offset':
        storage = torch.empty((rows * 2 + 1, k * 2 + 3), device=device, dtype=torch.float32)
        view = storage[1:1 + rows * 2:2, 1:1 + k * 2:2]
        view.copy_(torch.from_numpy(a).to(device)); return view
    if layout == 'row_offset':
        storage = torch.empty((rows + 2, k + 3), device=device, dtype=torch.float32)
        view = storage[1:1 + rows, 1:1 + k]
        view.copy_(torch.from_numpy(a).to(device)); return view
    if layout == 'broadcast_rows':
        need(np.array_equal(a, np.broadcast_to(a[:1], a.shape)), 'Bad broadcast fixture')
        return torch.from_numpy(a[:1].copy()).to(device).expand(rows, k)
    if layout == 'broadcast_columns':
        need(np.array_equal(a, np.broadcast_to(a[:, :1], a.shape)), 'Bad broadcast fixture')
        return torch.from_numpy(a[:, :1].copy()).to(device).expand(rows, k)
    if layout == 'lazy_negative':
        need(hasattr(torch, '_neg_view'), 'lazy negative API unavailable')
        return torch._neg_view(torch.from_numpy(np.ascontiguousarray(-a)).to(device))
    if layout == 'wide_stride_single':
        need(a.shape == (1, 1), 'Wide-stride fixture must contain one element')
        return torch.as_strided(torch.from_numpy(a).to(device), (1, 1), (2**31, 2**31))
    raise ValueError('Unknown layout ' + layout)


class Runner:
    def __init__(self, device, stream, report, library):
        self.device, self.stream, self.report, self.library = device, stream, report, library
        self.attempted = self.returned = self.completed = self.pending = False

    def execute(self, function):
        self.attempted = self.pending = True
        with torch.cuda.stream(self.stream):
            need(torch.cuda.current_stream(self.device).cuda_stream == self.stream.cuda_stream, 'Nondefault stream not current')
            result = function()
        self.returned = True
        self.stream.synchronize(); self.completed = True; self.pending = False
        return result

    def case(self, name, function):
        row = {'name': name, 'passed': False}; self.report['cases'].append(row)
        try:
            row.update(function()); row['passed'] = True
        except Exception as error:
            row.update(error_type=type(error).__name__, error=str(error)); raise
        print('PASS ' + name, flush=True)

    def direct(self, x, expected):
        resolved = x.resolve_conj().resolve_neg()
        codes = torch.from_numpy(np.bitwise_xor(expected.codes, np.uint8(255))).to(self.device)
        scales = torch.from_numpy(np.bitwise_xor(expected.scales, np.uint8(255))).to(self.device)
        flag = torch.full((1,), 123, dtype=torch.int32, device=self.device)
        for t in (x, resolved, codes, scales, flag):
            t.record_stream(self.stream)
        status = self.library.mxfp4_quantize_rne_cuda(resolved.data_ptr(), codes.data_ptr(), scales.data_ptr(),
            flag.data_ptr(), *resolved.shape, *resolved.stride(), self.stream.cuda_stream)
        need(status == 0, 'Direct native quantizer launch status ' + str(status))
        need(flag.item() == 0, 'Direct native rejected finite input')
        return ops.TorchMXTensor(codes, scales, tuple(x.shape))

    def quantization(self, source, layout='contiguous', coverage=None):
        expected = quantize_mx(source)
        def work():
            x = to_layout(source, self.device, layout)
            old = ops.quantize_mx_torch(x, rounding='nearest')
            new = ops.quantize_mx_cuda_rne(x)
            direct = self.direct(x, expected)
            return x, old, new, direct
        x, old, new, direct = self.execute(work)
        a, b, c = packed_metric(old, expected), packed_metric(new, expected), packed_metric(direct, expected)
        need(np.array_equal(host_array(x).view(np.uint32), np.ascontiguousarray(source).view(np.uint32)), 'Input was modified')
        return {'layout': layout, 'input_strides': list(x.stride()), 'storage_offset': x.storage_offset(),
            'lazy_negative': x.is_neg(), 'nondefault_stream': True, 'coverage': coverage,
            'torch_vs_numpy': a, 'cuda_rne_vs_numpy': b, 'direct_cuda_vs_numpy': c,
            'direct_output_poison': 'bitwise_inverse_of_every_expected_code_and_scale_byte',
            'direct_initial_error_flag': 123, 'finite_error_flag': 0, 'input_bits_unchanged': True}

    def nonfinite(self, value, position, layout):
        source = np.ones((3, 65), dtype=np.float32); source.flat[position] = value
        good = np.arange(195, dtype=np.float32).reshape(3, 65) / np.float32(31)
        expected = quantize_mx(good)
        def work():
            x = to_layout(source, self.device, layout)
            rejected = {}
            for name, fn in (('torch', ops.quantize_mx_torch), ('cuda_rne', ops.quantize_mx_cuda_rne)):
                try:
                    fn(x)
                except ValueError:
                    rejected[name] = True
                else:
                    raise AssertionError(name + ' returned packed output for nonfinite input')
            recovery = ops.quantize_mx_cuda_rne(to_layout(good, self.device, layout))
            return rejected, recovery
        rejected, recovered = self.execute(work)
        return {'nonfinite': str(value), 'flat_position': position, 'layout': layout,
                'rejected': rejected, 'finite_recovery': packed_metric(recovered, expected),
                'nondefault_stream': True}

    def linear(self, source, weight, upstream, bias, flags, backend, layout='contiguous'):
        flat = source.reshape(-1, source.shape[-1]); g = upstream.reshape(-1, weight.shape[0])
        product = lambda a, b: gemm_mx(quantize_mx(np.ascontiguousarray(a)), quantize_mx(np.ascontiguousarray(b)))
        f, d, w = flags
        expected = {'output': ((product(flat, weight) if f else flat @ weight.T) + bias).reshape(upstream.shape),
                    'dgrad': (product(g, weight.T) if d else g @ weight).reshape(source.shape),
                    'wgrad': product(g.T, flat.T) if w else g.T @ flat,
                    'bias_grad': g.sum(axis=0, dtype=np.float32)}
        expected_inputs = ([] if not f else [flat, weight]) + ([] if not d else [g, weight.T]) + ([] if not w else [g.T, flat.T])
        trace = []
        original_new, original_old = ops.quantize_mx_cuda_rne, ops.quantize_mx_torch
        def tracked_new(x):
            trace.append({'kind': 'cuda_rne', 'input': host_array(x).copy()})
            return original_new(x)
        def tracked_old(x, **kwargs):
            trace.append({'kind': 'torch', 'rounding': kwargs.get('rounding', 'nearest')})
            return original_old(x, **kwargs)
        recipe = ops.MXRecipe(*flags, backend=backend, quantizer_backend='cuda_rne')
        def work():
            x = to_layout(flat, self.device, layout).reshape(source.shape).detach().requires_grad_(True)
            wt = to_layout(weight, self.device, layout).detach().requires_grad_(True)
            gg = torch.from_numpy(np.ascontiguousarray(upstream)).to(self.device)
            bb = torch.from_numpy(bias.copy()).to(self.device).requires_grad_(True)
            ops.quantize_mx_cuda_rne, ops.quantize_mx_torch = tracked_new, tracked_old
            try:
                y = ops.mxfp4_linear(x, wt, bb, recipe=recipe)
                y.backward(gg)
            finally:
                ops.quantize_mx_cuda_rne, ops.quantize_mx_torch = original_new, original_old
            return y.detach(), x.grad, wt.grad, bb.grad
        actual = self.execute(work)
        need(len(trace) == len(expected_inputs) and all(t['kind'] == 'cuda_rne' for t in trace),
             'MXRecipe did not dispatch every selected RNE operand to the native wrapper')
        for observed, reference in zip(trace, expected_inputs):
            need(np.array_equal(observed['input'].view(np.uint32), np.ascontiguousarray(reference).view(np.uint32)),
                 'F/D/W quantizer did not receive original FP32 values in the correct reduction-axis orientation')
        exact = (f, d, w, False)
        metrics = {name: numeric_metric(host_array(value), expected[name], is_exact)
                   for name, value, is_exact in zip(('output', 'dgrad', 'wgrad', 'bias_grad'), actual, exact)}
        return {'recipe': asdict(recipe), 'layout': layout, 'input_shape': list(source.shape),
                'weight_shape': list(weight.shape), 'nondefault_stream': True,
                'gradient_contract': 'declared_first_order_surrogate', 'comparisons': metrics,
                'native_quantizer_calls': len(trace), 'torch_quantizer_calls': 0,
                'original_fp32_axis_inputs_verified': True,
                'dispatch_trace': 'checker wraps and delegates actual functions; input capture outside performance measurements'}

    def stochastic_routing(self, backend, seed, rounding, gradient_rounding):
        rng = np.random.default_rng(seed)
        x_np, w_np, g_np = (rng.normal(size=s).astype(np.float32) for s in ((7, 33), (5, 33), (7, 5)))
        original_new, original_old = ops.quantize_mx_cuda_rne, ops.quantize_mx_torch
        trace = []
        def new(x):
            trace.append('cuda_rne'); return original_new(x)
        def old(x, **kwargs):
            trace.append('torch:' + kwargs.get('rounding', 'nearest')); return original_old(x, **kwargs)
        def run(quantizer):
            torch.manual_seed(seed)
            x = torch.from_numpy(x_np).to(self.device).requires_grad_(True)
            w = torch.from_numpy(w_np).to(self.device).requires_grad_(True)
            g = torch.from_numpy(g_np).to(self.device)
            recipe = ops.MXRecipe(True, True, True, rounding=rounding, gradient_rounding=gradient_rounding,
                                  backend=backend, quantizer_backend=quantizer)
            y = ops.mxfp4_linear(x, w, recipe=recipe); y.backward(g)
            return y.detach(), x.grad, w.grad
        def work():
            baseline = run('torch')
            ops.quantize_mx_cuda_rne, ops.quantize_mx_torch = new, old
            try:
                candidate = run('cuda_rne')
            finally:
                ops.quantize_mx_cuda_rne, ops.quantize_mx_torch = original_new, original_old
            return baseline, candidate
        baseline, candidate = self.execute(work)
        expected = [('cuda_rne' if r == 'nearest' else 'torch:stochastic') for r in
                    (rounding, rounding, gradient_rounding, rounding, gradient_rounding, rounding)]
        need(trace == expected, 'Stochastic route changed or native RNE incorrectly consumed SR operand')
        checks = [numeric_metric(host_array(a), host_array(b), True) for a, b in zip(candidate, baseline)]
        return {'backend': backend, 'rounding': rounding, 'gradient_rounding': gradient_rounding,
                'same_seed': seed, 'actual_quantizer_routes': trace, 'same_seed_output_and_gradients': checks,
                'original_torch_quantizer_source_unchanged': True, 'nondefault_stream': True}


def run_suite(runner, seed):
    rng = np.random.default_rng(seed)
    backends = ('cuda', 'cuda_tiled')
    if runner.report['library']['certified_enabled']:
        backends += ('cuda_certified',)
    runner.report['tested_gemm_backends'] = list(backends)
    runner.case('all_253_scale_midpoint_neighbors_and_signs', lambda: runner.quantization(midpoint_matrix(), coverage={
        'scale_exponents': [-127, 125], 'scale_count': 253, 'midpoints_per_scale': 7,
        'neighbors_per_midpoint': 3, 'signs': 2, 'target_boundary_values': 253 * 7 * 3 * 2}))
    runner.case('amax_power_boundaries', lambda: runner.quantization(exponent_boundary_matrix(), coverage={
        'binary_exponents': [-149, 127], 'powers': 277, 'neighbors_per_power': 3}))
    extreme = np.asarray([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
        0x00800000, 0x80800000, 0x7f7fffff, 0xff7fffff, 0x3f800000, 0xbf800000], dtype=np.uint32).view(np.float32)
    runner.case('fp32_extremes_subnormals_signed_zero', lambda: runner.quantization(extreme[:, None]))
    zeros = np.zeros((4, 65), dtype=np.float32); zeros[1::2] = np.float32(-0.)
    runner.case('zero_blocks_and_canonical_tail', lambda: runner.quantization(zeros, 'positive_strides_offset'))
    random_bits = finite_bits(rng, (257, 129))
    runner.case('random_finite_fp32_bits', lambda: runner.quantization(random_bits, 'transpose',
        {'finite_bit_patterns': int(random_bits.size), 'seed': seed}))
    for shape in ((1, 1), (3, 31), (2, 32), (7, 33), (5, 63), (3, 64), (17, 65), (33, 97), (128, 128)):
        source = rng.normal(size=shape).astype(np.float32)
        for layout in ('contiguous', 'transpose', 'positive_strides_offset', 'row_offset'):
            runner.case(f'layout_{shape[0]}x{shape[1]}_{layout}', lambda x=source, l=layout: runner.quantization(x, l))
    row_broadcast = np.broadcast_to(rng.normal(size=(1, 65)).astype(np.float32), (7, 65)).copy()
    col_broadcast = np.broadcast_to(np.asarray([0., -0., 1.1, -4.5], dtype=np.float32)[:, None], (4, 33)).copy()
    runner.case('zero_stride_rows', lambda: runner.quantization(row_broadcast, 'broadcast_rows'))
    runner.case('zero_stride_columns', lambda: runner.quantization(col_broadcast, 'broadcast_columns'))
    runner.case('int64_stride_single_element', lambda: runner.quantization(
        np.asarray([[1.25]], dtype=np.float32), 'wide_stride_single',
        {'native_dispatch_expected': 'int64', 'stride_elements': 2**31, 'actual_input_elements': 1}))
    if hasattr(torch, '_neg_view'):
        negative_view = rng.normal(size=(7, 33)).astype(np.float32); negative_view[0, 0] = -0.
        runner.case('lazy_negative_input', lambda: runner.quantization(negative_view, 'lazy_negative'))
    else:
        runner.report['optional_coverage_unavailable'].append('torch._neg_view is unavailable')
    for label, value in (('nan', np.nan), ('posinf', np.inf), ('neginf', -np.inf)):
        for position in (0, 194):
            for layout in ('contiguous', 'positive_strides_offset'):
                runner.case(f'reject_{label}_{position}_{layout}', lambda v=value, p=position, l=layout: runner.nonfinite(v, p, l))
    fixtures = []
    for title, m, n, k, layout in (('tail', 7, 5, 33, 'contiguous'), ('strided', 9, 7, 65, 'positive_strides_offset')):
        x, w, g, b = (rng.normal(size=s).astype(np.float32) for s in ((m, k), (n, k), (m, n), (n,)))
        if title == 'strided':
            x, g = x.reshape(3, 3, k), g.reshape(3, 3, n)
        fixtures.append((title, x, w, g, b, layout))
    # A negative control distinguishes requantizing original FP32 transposes
    # from first losing values to a different row-wise representation.
    x, w, g, b = (rng.normal(size=s).astype(np.float32) for s in ((11, 33), (13, 33), (11, 13), (13,)))
    correct = gemm_mx(quantize_mx(g), quantize_mx(w.T.copy()))
    wrong = gemm_mx(quantize_mx(g), quantize_mx(dequantize_mx(quantize_mx(w)).T.copy()))
    need(not np.array_equal(correct, wrong), 'Original-FP32 transpose negative control ineffective')
    correct_w = gemm_mx(quantize_mx(g.T.copy()), quantize_mx(x.T.copy()))
    wrong_w = gemm_mx(quantize_mx(g.T.copy()), quantize_mx(dequantize_mx(quantize_mx(x)).T.copy()))
    need(not np.array_equal(correct_w, wrong_w), 'Original-FP32 wgrad negative control ineffective')
    runner.report['transpose_negative_control'] = {'dgrad_values': int(correct.size),
        'dgrad_mismatches_if_row_quantized_values_reused': int(np.count_nonzero(correct != wrong)),
        'wgrad_values': int(correct_w.size),
        'wgrad_mismatches_if_row_quantized_values_reused': int(np.count_nonzero(correct_w != wrong_w)), 'effective': True}
    fixtures.append(('original_fp32_transpose', x, w, g, b, 'transpose'))
    for title, x, w, g, b, layout in fixtures:
        for backend in backends:
            for flags in itertools.product((False, True), repeat=3):
                bits = ''.join('1' if v else '0' for v in flags)
                runner.case(f'linear_{title}_{backend}_{bits}',
                    lambda x=x, w=w, g=g, b=b, f=flags, be=backend, l=layout: runner.linear(x, w, g, b, f, be, l))
    for backend in backends:
        for rounding, gradient_rounding in (('nearest', 'stochastic'), ('stochastic', 'stochastic')):
            runner.case(f'unchanged_sr_{backend}_{rounding}_{gradient_rounding}',
                lambda be=backend, r=rounding, gr=gradient_rounding: runner.stochastic_routing(be, seed, r, gr))
    expected_cases = 56 + int(hasattr(torch, '_neg_view')) + 26 * len(backends)
    names = [case['name'] for case in runner.report['cases']]
    need(len(names) == len(set(names)) == expected_cases, 'Planned checker coverage is incomplete')
    runner.report['expected_cases'] = expected_cases
    runner.report['unsupported_coverage_not_executed'] = [
        'CUDA Graph capture rejection', 'unavailable-library rejection', 'negative strides (unsupported by Torch construction)']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile-evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=20261123)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Output exists; preserve evidence using a new path')
    if not args.device.startswith('cuda:') or not args.device[5:].isdigit(): parser.error('Require explicit CUDA index')
    if not 0 <= args.seed < 2**32: parser.error('Require a uint32 seed')
    report = {'schema_version': 1, 'purpose': 'independent opt-in CUDA RNE quantizer and training-route correctness',
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'pending', 'passed': False,
        'gpu_executed': False, 'device': args.device, 'seed': args.seed, 'python': platform.python_version(),
        'numpy': np.__version__, 'torch': None if torch is None else torch.__version__,
        'source_sha256': source_hashes(), 'cases': [], 'optional_coverage_unavailable': [],
        'quantizer': 'cuda_rne', 'oracle': 'independent_NumPy_quantize_mx', 'performance_claim': None,
        'full_training_executed': False, 'llm_convergence_validated': False, 'cpu_fallback': False,
        'minimum_free_bytes_after_context': MINIMUM_FREE, 'reserve_bytes': RESERVE_BYTES,
        'conservative_allocation_budget_bytes': CHECK_ALLOCATION_BUDGET}
    runner = None; code = 1
    try:
        if torch is None or not torch.cuda.is_available(): raise Blocked('CUDA-enabled Torch is unavailable')
        device = torch.device(args.device); torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        report['gpu'] = {'name': properties.name, 'compute_capability': [properties.major, properties.minor],
            'actual_V100': properties.major == 7 and properties.minor == 0 and 'V100' in properties.name,
            'total_memory_bytes': properties.total_memory, 'multiprocessor_count': properties.multi_processor_count,
            'torch_cuda_build_version': torch.version.cuda}
        if not report['gpu']['actual_V100']: raise Blocked('Require actual V100 SM70')
        library, report['library'] = library_metadata(args.compile_evidence)
        free, total = torch.cuda.mem_get_info(device)
        report['gpu'].update(free_bytes_after_context_and_library=free, runtime_total_bytes=total)
        if free < MINIMUM_FREE or CHECK_ALLOCATION_BUDGET > free - RESERVE_BYTES:
            raise Blocked('Insufficient guarded device memory; no services are stopped')
        torch.set_num_threads(2); torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        stream = torch.cuda.Stream(device=device)
        need(stream.cuda_stream != torch.cuda.default_stream(device).cuda_stream, 'A nondefault stream is required')
        report['nondefault_stream'] = True
        runner = Runner(device, stream, report, library)
        with torch.cuda.stream(stream), torch.autocast(device_type='cuda', enabled=False):
            run_suite(runner, args.seed)
        torch.cuda.synchronize(device)
        need(source_hashes() == report['source_sha256'], 'Sources changed during checker')
        need(sha256(report['library']['path']) == report['library']['sha256'], 'Library changed during checker')
        need(sha256(args.compile_evidence) == report['library']['compile_evidence_sha256'], 'Compile manifest changed')
        report.update(status='passed', passed=True, source_unchanged_during_check=True)
        code = 0
    except Blocked as error:
        report.update(status='blocked', error_type=type(error).__name__, error=str(error)); code = 2
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
    finally:
        if runner is not None:
            report.update(gpu_executed=True if runner.completed else (None if runner.attempted else False),
                gpu_work_attempted=runner.attempted, gpu_work_call_returned=runner.returned,
                gpu_work_synchronized=runner.completed, all_attempted_work_synchronized=not runner.pending)
        report['completed_cases'] = sum(c.get('passed') is True for c in report['cases'])
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as handle:
            json.dump(report, handle, indent=2, allow_nan=False); handle.write('\n')
        print(json.dumps({k: report[k] for k in ('status', 'passed', 'gpu_executed', 'completed_cases')}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
