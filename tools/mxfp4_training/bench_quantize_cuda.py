"""Two-input complete-wrapper RNE latency on resident, identical FP32 inputs.

Run the independent checker first against the same source and library. Every
reported round includes both quantizer calls, allocations, validation, and all
finite-flag synchronizations. It excludes initial input generation, NumPy
reference construction, clock sampling, and post-timing byte comparisons.
CUDA events span the entire wrapper sequence, including host submission gaps;
they are not kernel-only timings. Quantization is reported as latency, never
GEMM-equivalent FLOPS. Repeated rounds are one process, not independent runs.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.mxfp4_training import check_quantize_cuda as check
from tools.mxfp4_training.reference import MXTensor, quantize_mx

torch = check.torch
if torch is not None:
    ops = check.ops


class ClockSamples:
    """Read-only NVML snapshots, mapped by the active CUDA device PCI address."""
    def __init__(self, device_index):
        self.metadata = {'available': False, 'clock_or_power_changes': False,
                         'sampling_scope': 'outside all timing intervals', 'samples': []}
        self.nvml = None
        try:
            driver = ctypes.CDLL('libcuda.so.1')
            driver.cuInit.argtypes = [ctypes.c_uint]
            driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
            driver.cuDeviceGetPCIBusId.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            check.need(driver.cuInit(0) == 0, 'cuInit failed for PCI mapping')
            dev = ctypes.c_int(); bus = ctypes.create_string_buffer(64)
            check.need(driver.cuDeviceGet(ctypes.byref(dev), device_index) == 0, 'cuDeviceGet failed')
            check.need(driver.cuDeviceGetPCIBusId(bus, len(bus), dev.value) == 0, 'PCI lookup failed')
            lib = ctypes.CDLL('libnvidia-ml.so.1')
            lib.nvmlInit_v2.argtypes = []; lib.nvmlInit_v2.restype = ctypes.c_int
            check.need(lib.nvmlInit_v2() == 0, 'NVML initialization failed')
            self.nvml = lib; self.handle = ctypes.c_void_p()
            lib.nvmlDeviceGetHandleByPciBusId_v2.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
            check.need(lib.nvmlDeviceGetHandleByPciBusId_v2(bus.value, ctypes.byref(self.handle)) == 0,
                       'NVML PCI mapping failed')
            lib.nvmlDeviceGetClockInfo.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
            for name in ('nvmlDeviceGetPowerUsage', 'nvmlDeviceGetPowerManagementLimit'):
                getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
            lib.nvmlDeviceGetUUID.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
            uuid = ctypes.create_string_buffer(96)
            check.need(lib.nvmlDeviceGetUUID(self.handle, uuid, len(uuid)) == 0, 'NVML UUID failed')
            self.metadata.update(available=True, pci_bus_id=bus.value.decode(), gpu_uuid=uuid.value.decode())
        except Exception as error:
            self.metadata['initialization_error'] = str(error)

    def snapshot(self, label):
        row = {'label': label, 'utc': datetime.now(timezone.utc).isoformat()}
        if self.metadata['available']:
            for key, name, extra in (
                ('sm_clock_mhz', 'nvmlDeviceGetClockInfo', (1,)),
                ('memory_clock_mhz', 'nvmlDeviceGetClockInfo', (2,)),
                ('power_usage_mw', 'nvmlDeviceGetPowerUsage', ()),
                ('power_limit_mw', 'nvmlDeviceGetPowerManagementLimit', ())):
                value = ctypes.c_uint()
                status = getattr(self.nvml, name)(self.handle, *extra, ctypes.byref(value))
                row[key] = value.value if status == 0 else None
                if status != 0: row[key + '_nvml_status'] = status
        self.metadata['samples'].append(row)
        return row

    def close(self):
        if self.nvml is not None:
            self.nvml.nvmlShutdown()
            self.nvml = None
        for key in ('sm_clock_mhz', 'memory_clock_mhz', 'power_usage_mw', 'power_limit_mw'):
            values = [r[key] for r in self.metadata['samples'] if r.get(key) is not None]
            if values:
                self.metadata[key + '_observed'] = {'minimum': min(values), 'maximum': max(values),
                                                   'samples': len(values)}
        self.metadata['continuous_load_measurement'] = False
        self.metadata['limitation'] = 'Boundary snapshots can miss within-round clock and competing-load changes.'


def source_hashes():
    result = check.source_hashes()
    result['bench_quantize_cuda.py'] = check.sha256(Path(__file__))
    return result


def validate_prerequisite(path, library, gpu):
    path = Path(path).resolve(strict=True)
    data = json.loads(path.read_text())
    check.need(data.get('passed') is True and data.get('status') == 'passed', 'Independent checker did not pass')
    check.need(data.get('gpu_executed') is True and data.get('all_attempted_work_synchronized') is True,
               'Checker lacks completed, synchronized GPU execution')
    cases = data.get('cases', [])
    check.need(bool(cases) and all(c.get('passed') is True for c in cases), 'Checker contains an incomplete case')
    check.need(data.get('completed_cases') == data.get('expected_cases') == len(cases), 'Checker coverage is incomplete')
    check.need(data.get('source_sha256') == check.source_hashes(), 'Checker belongs to different source bytes')
    check.need(data.get('library', {}).get('sha256') == library['sha256'], 'Checker used a different native library')
    check.need(data['library'].get('compile_evidence_sha256') == library['compile_evidence_sha256'],
               'Checker used a different compile manifest')
    check.need(data.get('torch') == torch.__version__, 'Checker used a different Torch version')
    for key in ('name', 'compute_capability', 'total_memory_bytes', 'multiprocessor_count', 'torch_cuda_build_version'):
        check.need(data.get('gpu', {}).get(key) == gpu[key], 'Checker GPU metadata differs: ' + key)
    return {'path': str(path), 'sha256': check.sha256(path), 'passed': True,
            'completed_cases': len(cases), 'library_and_sources_match': True,
            'same_gpu_model_and_runtime': True,
            'physical_gpu_identity': 'checker records model properties; benchmark separately records NVML UUID'}


def input_oracle(tensor, rows_per_chunk=64):
    rows, k = tuple(tensor.shape); padded = ((k + 31) // 32) * 32
    codes = np.empty((rows, padded // 2), dtype=np.uint8)
    scales = np.empty((rows, padded // 32), dtype=np.uint8)
    fingerprint = hashlib.sha256()
    for first in range(0, rows, rows_per_chunk):
        values = check.host_array(tensor[first:first + rows_per_chunk])
        check.need(values.dtype == np.float32 and bool(np.isfinite(values).all()), 'Benchmark input is not finite FP32')
        fingerprint.update(np.ascontiguousarray(values).tobytes())
        reference = quantize_mx(values)
        codes[first:first + len(values)] = reference.codes
        scales[first:first + len(values)] = reference.scales
    return MXTensor(codes, scales, (rows, k)), fingerprint.hexdigest()


def input_hash(tensor, rows_per_chunk=64):
    digest = hashlib.sha256()
    for first in range(0, len(tensor), rows_per_chunk):
        digest.update(np.ascontiguousarray(check.host_array(tensor[first:first + rows_per_chunk])).tobytes())
    return digest.hexdigest()


def compare_pair(actual, expected):
    check.need(len(actual) == len(expected) == 2, 'Both quantized inputs are required')
    return {name: check.packed_metric(a, e) for name, a, e in zip(('A', 'B'), actual, expected)}


def memory_sample(device, label):
    free, total = torch.cuda.mem_get_info(device)
    return {'label': label, 'free_bytes': free, 'total_bytes': total,
            'allocated_bytes': torch.cuda.memory_allocated(device),
            'reserved_bytes': torch.cuda.memory_reserved(device),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}


def summarize_rounds(rows):
    result = {}
    for name in ('torch', 'cuda_rne'):
        selected = [r for r in rows if r['quantizer_backend'] == name]
        result[name] = {'rounds': len(selected)}
        for field in ('wall_ms_per_pair', 'event_ms_per_pair'):
            values = [r[field] for r in selected]
            check.need(values and all(math.isfinite(v) and v > 0 for v in values), 'Invalid timing observations')
            result[name][field] = {'median': statistics.median(values), 'minimum': min(values),
                                  'maximum': max(values), 'raw_round_values': values}
    wall = result['torch']['wall_ms_per_pair']['median'] / result['cuda_rne']['wall_ms_per_pair']['median']
    event = result['torch']['event_ms_per_pair']['median'] / result['cuda_rne']['event_ms_per_pair']['median']
    return {'backends': result, 'median_wall_speedup_torch_over_cuda_rne': wall,
            'median_event_speedup_torch_over_cuda_rne': event,
            'median_wall_latency_reduction_percent': 100 * (1 - 1 / wall),
            'independent_process_runs': 1, 'repeated_rounds_are_independent_samples': False,
            'flops_or_tflops': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile-evidence', type=Path, required=True)
    parser.add_argument('--check-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--size', type=int, choices=(1024, 4096), default=1024)
    parser.add_argument('--seed', type=int, default=20261123)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--iters', type=int, default=10)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Output exists; preserve evidence using a new path')
    if not args.device.startswith('cuda:') or not args.device[5:].isdigit(): parser.error('Require explicit CUDA index')
    if not (0 <= args.seed < 2**32 and 1 <= args.warmup <= 20 and 3 <= args.rounds <= 9 and 1 <= args.iters <= 20):
        parser.error('Require uint32 seed, warmup1..20, rounds3..9, iters1..20')
    report = {'schema_version': 1, 'purpose': 'two-input complete-wrapper quantization latency',
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'pending', 'passed': False,
        'gpu_executed': False, 'seed': args.seed, 'device': args.device, 'shape_A': [args.size, args.size],
        'shape_B': [args.size, args.size], 'rounds_per_backend': args.rounds, 'iterations_per_round': args.iters,
        'warmup_per_backend': args.warmup, 'python': platform.python_version(), 'numpy': np.__version__,
        'torch': None if torch is None else torch.__version__, 'source_sha256': source_hashes(),
        'timing_scope': 'two complete quantizer wrapper calls, including allocations, validation and finite-flag synchronizations',
        'event_scope': 'stream interval across complete wrappers, including host submission gaps; not kernel-only',
        'excluded_from_timing': ['input_generation', 'NumPy_oracle', 'byte_checks', 'clock_sampling', 'source_hashing'],
        'same_resident_input_tensors_for_every_call': True, 'packed_tensor_cache': False,
        'torch_caching_allocator_reuse_allowed': True, 'clock_or_power_changes': False,
        'order_rule': 'rotate [torch, cuda_rne] once per round', 'full_training_executed': False,
        'cpu_fallback': False, 'flops_or_tflops': None, 'raw_rounds': [], 'memory': [], 'initial_correctness': {}}
    code = 1; clocks = None; attempted = synchronized = False; in_flight = False
    try:
        if torch is None or not torch.cuda.is_available(): raise check.Blocked('CUDA-enabled Torch is unavailable')
        device = torch.device(args.device); torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        gpu = {'name': properties.name, 'compute_capability': [properties.major, properties.minor],
               'actual_V100': properties.major == 7 and properties.minor == 0 and 'V100' in properties.name,
               'total_memory_bytes': properties.total_memory, 'multiprocessor_count': properties.multi_processor_count,
               'torch_cuda_build_version': torch.version.cuda}
        report['gpu'] = gpu
        if not gpu['actual_V100']: raise check.Blocked('Require actual V100 SM70')
        _library, library = check.library_metadata(args.compile_evidence); report['library'] = library
        report['checker_prerequisite'] = validate_prerequisite(args.check_report, library, gpu)
        before = memory_sample(device, 'before_inputs'); report['memory'].append(before)
        budget = 192 * args.size**2 + 256 * 1024**2
        report['memory_guard'] = {'conservative_allocation_budget_bytes': budget,
            'minimum_free_bytes': check.MINIMUM_FREE, 'minimum_reserve_bytes': check.RESERVE_BYTES,
            'budget_is_conservative_preflight_estimate_not_cuda_allocator_cap': True}
        if before['free_bytes'] < check.MINIMUM_FREE or budget > before['free_bytes'] - check.RESERVE_BYTES:
            raise check.Blocked('Insufficient guarded device memory; no services are stopped')
        torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
        stream = torch.cuda.Stream(device=device)
        check.need(stream.cuda_stream != torch.cuda.default_stream(device).cuda_stream, 'Use a nondefault stream')
        report['nondefault_stream'] = True
        clocks = ClockSamples(int(args.device[5:])); report['clock_monitor'] = clocks.metadata
        clocks.snapshot('before_inputs')
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False), torch.cuda.stream(stream):
            generator = torch.Generator(device=device); generator.manual_seed(args.seed)
            attempted = in_flight = True
            a = torch.randn((args.size, args.size), dtype=torch.float32, device=device, generator=generator)
            b = torch.randn((args.size, args.size), dtype=torch.float32, device=device, generator=generator)
            stream.synchronize(); synchronized = True; in_flight = False
            expected_a, hash_a = input_oracle(a); expected_b, hash_b = input_oracle(b)
            expected = (expected_a, expected_b)
            report['input_fingerprint'] = {'A_fp32_bytes_sha256': hash_a, 'B_fp32_bytes_sha256': hash_b,
                'dtype': 'float32', 'order': 'C', 'generator': 'torch.Generator normal; one fixed resident pair',
                'oracle': 'independent NumPy quantize_mx over every input byte; 64-row chunks'}
            def old():
                return ops.quantize_mx_torch(a, rounding='nearest'), ops.quantize_mx_torch(b, rounding='nearest')
            def new():
                return ops.quantize_mx_cuda_rne(a), ops.quantize_mx_cuda_rne(b)
            jobs = {'torch': old, 'cuda_rne': new}
            for name, fn in jobs.items():
                in_flight = True; actual = fn(); stream.synchronize(); in_flight = False
                report['initial_correctness'][name] = compare_pair(actual, expected)
                del actual
            for name, fn in jobs.items():
                for _ in range(args.warmup):
                    in_flight = True; actual = fn(); del actual
                stream.synchronize(); in_flight = False
            warm = memory_sample(device, 'after_warmup'); report['memory'].append(warm)
            if warm['free_bytes'] < check.RESERVE_BYTES:
                raise check.Blocked('Post-warmup free memory fell below the fixed reserve')
            clocks.snapshot('after_warmup')
            start_event = torch.cuda.Event(enable_timing=True); end_event = torch.cuda.Event(enable_timing=True)
            # Materialize event handles before measuring either backend.
            start_event.record(stream); end_event.record(stream); end_event.synchronize()
            names = list(jobs)
            for round_index in range(args.rounds):
                order = names[round_index % 2:] + names[:round_index % 2]
                clocks.snapshot(f'round_{round_index}_before')
                for order_index, name in enumerate(order):
                    stream.synchronize(); actual = None; fn = jobs[name]
                    wall_start = time.perf_counter(); start_event.record(stream); in_flight = True
                    for _ in range(args.iters):
                        if actual is not None: del actual
                        actual = fn()
                    end_event.record(stream); end_event.synchronize()
                    wall_end = time.perf_counter(); in_flight = False
                    event_ms = start_event.elapsed_time(end_event)
                    row = {'round_index': round_index, 'position_in_round': order_index, 'order': order,
                        'quantizer_backend': name, 'iterations': args.iters,
                        'wall_ms_total': (wall_end - wall_start) * 1000, 'event_ms_total': event_ms,
                        'wall_ms_per_pair': (wall_end - wall_start) * 1000 / args.iters,
                        'event_ms_per_pair': event_ms / args.iters, 'passed': False}
                    report['raw_rounds'].append(row)
                    # These are the actual last TIMED outputs. There is no new quantizer call here.
                    row['last_timed_outputs_vs_numpy'] = compare_pair(actual, expected)
                    row['byte_checks_after_both_timing_endpoints'] = True; row['passed'] = True
                    del actual
                clocks.snapshot(f'round_{round_index}_after')
            stream.synchronize()
            check.need(input_hash(a) == hash_a and input_hash(b) == hash_b, 'Input bytes changed during benchmark')
            report['input_bits_unchanged'] = True
        report['memory'].append(memory_sample(device, 'after_final_checks'))
        check.need(len(report['raw_rounds']) == 2 * args.rounds, 'Incomplete timing rounds')
        report['summary'] = summarize_rounds(report['raw_rounds'])
        report['calls_per_backend'] = {'initial': 1, 'warmup': args.warmup, 'timed': args.rounds * args.iters,
                                      'quantizer_calls_per_pair': 2}
        check.need(source_hashes() == report['source_sha256'], 'Sources changed during benchmark')
        check.need(check.sha256(library['path']) == library['sha256'], 'Library changed during benchmark')
        check.need(check.sha256(args.compile_evidence) == library['compile_evidence_sha256'], 'Compile manifest changed')
        check.need(check.sha256(args.check_report) == report['checker_prerequisite']['sha256'], 'Prerequisite changed')
        report.update(status='passed', passed=True, source_unchanged_during_benchmark=True); code = 0
    except check.Blocked as error:
        report.update(status='blocked', error_type=type(error).__name__, error=str(error)); code = 2
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
    finally:
        if clocks is not None: clocks.close()
        report.update(gpu_work_attempted=attempted, gpu_executed=True if synchronized else (None if attempted else False),
                      gpu_work_synchronized=synchronized, all_attempted_work_synchronized=not in_flight,
                      finished_at_utc=datetime.now(timezone.utc).isoformat())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as handle:
            json.dump(report, handle, indent=2, allow_nan=False); handle.write('\n')
        print(json.dumps({k: report[k] for k in ('status', 'passed', 'gpu_executed')}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
