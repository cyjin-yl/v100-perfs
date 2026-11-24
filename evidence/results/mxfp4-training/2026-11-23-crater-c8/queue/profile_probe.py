#!/usr/bin/env python3
"""One exact C2 pipeline inside cudaProfilerStart/Stop; diagnostic, not timing."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap

SIZE, SEED, WARMUP = 4096, 20261123, 10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(value, message):
    if not value: raise RuntimeError(message)


def api_binding(function, name):
    """Verify installed Torch exposes the documented checked CUDA runtime call."""
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    need(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef), 'Unexpected profiler API source')
    body = tree.body[0].body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    expected = ast.parse('check_error(cudart().' + name + '())').body
    need(ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False) ==
         ast.dump(ast.Module(body=expected, type_ignores=[]), include_attributes=False),
         'Installed profiler API is not the documented checked CUDA runtime entry point')
    path = Path(inspect.getsourcefile(function)).resolve(strict=True)
    return {'module': function.__module__, 'function': function.__name__, 'runtime_symbol': name,
            'installed_source_path': str(path), 'installed_module_sha256': sha(path),
            'function_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'actual_function_source': source, 'checked_runtime_call_AST_verified': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--compile-evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(not args.output.exists(), 'Output already exists')
    source = args.source_root.resolve(strict=True)
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import torch.cuda.profiler
    from tools.mxfp4_training import certified_workspace as workspace_module
    from tools.mxfp4_training.check_certified_workspace import source_hashes
    from tools.mxfp4_training.check_torch_certified import library_metadata
    from tools.mxfp4_training.torch_ops import gemm_packed, quantize_mx_cuda_rne

    need(Path(workspace_module.__file__).resolve() == source / 'tools/mxfp4_training/certified_workspace.py',
         'Imported workspace module is not the frozen source')
    report = {'schema_version': 1, 'status': 'pending', 'passed': False,
        'purpose': 'one complete C2 pipeline within explicit CUDA profiler range',
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'gpu_executed': False,
        'probe_sha256': sha(__file__), 'source_root': str(source), 'source_sha256': source_hashes(),
        'torch': torch.__version__, 'device': 'cuda:0', 'size': SIZE, 'seed': SEED, 'warmup': WARMUP,
        'profiled_workspace_submissions': 0, 'profile_range_entered': False, 'profile_range_stopped': False,
        'quantization_in_profile_range': False, 'context_creation_in_profile_range': False,
        'correctness_comparison_or_D2H_in_profile_range': False, 'last_stream_sync_in_profile_range': True,
        'diagnostic_only': True, 'unprofiled_performance_claim': False, 'full_training_executed': False}
    device = torch.device('cuda:0')
    started = False
    try:
        need(torch.cuda.is_available(), 'CUDA is unavailable')
        props = torch.cuda.get_device_properties(device)
        need((props.major, props.minor) == (7, 0) and 'V100' in props.name, 'Require actual V100 SM70')
        report['gpu'] = {'name': props.name, 'compute_capability': [props.major, props.minor],
                         'total_memory_bytes': props.total_memory, 'sm_count': props.multi_processor_count}
        report['library'] = library_metadata(args.compile_evidence, 2, report['source_sha256'])
        report['profiler_API'] = {
            'start': api_binding(torch.cuda.profiler.start, 'cudaProfilerStart'),
            'stop': api_binding(torch.cuda.profiler.stop, 'cudaProfilerStop')}
        with torch.cuda.device(device):
            free, total = torch.cuda.mem_get_info(device)
            need(free >= 2 * 1024**3, 'Require two GiB free for bounded diagnostic inputs/output/workspace')
            report['memory_admission'] = {'free_bytes': free, 'total_bytes': total, 'minimum_free_bytes': 2 * 1024**3}
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                generator = torch.Generator(device=device).manual_seed(SEED)
                a = torch.randn((SIZE, SIZE), device=device, dtype=torch.float32, generator=generator)
                b = torch.randn((SIZE, SIZE), device=device, dtype=torch.float32, generator=generator)
                qa, qb = quantize_mx_cuda_rne(a), quantize_mx_cuda_rne(b)
                expected = gemm_packed(qa, qb, backend='cuda_tiled')
                stream.synchronize(); report['gpu_executed'] = True

                def fingerprints():
                    return {name: hashlib.sha256(tensor.cpu().numpy().tobytes(order='C')).hexdigest()
                            for name, tensor in (('a_fp32', a), ('b_fp32', b),
                                ('a_codes', qa.codes), ('a_scales', qa.scales),
                                ('b_codes', qb.codes), ('b_scales', qb.scales))}

                def compare(actual):
                    host = actual.cpu().numpy(); oracle = expected.cpu().numpy()
                    need(host.shape == oracle.shape == (SIZE, SIZE) and host.dtype == oracle.dtype == np.float32,
                         'Output dtype/shape differs')
                    need(np.isfinite(host).all() and np.isfinite(oracle).all(), 'Nonfinite diagnostic output')
                    count = int(np.count_nonzero(host.view(np.uint32) != oracle.view(np.uint32)))
                    need(count == 0 and np.all(host.view(np.uint32)[host == 0] == 0), 'Full output bit comparison failed')
                    return {'values': SIZE * SIZE, 'bit_mismatches': count,
                            'all_finite': True, 'canonical_positive_zero': True}

                report['input_sha256_before'] = fingerprints()
                with workspace_module.CertifiedWorkspace(SIZE, SIZE, SIZE, device=device) as workspace:
                    initial_info = {}
                    initial = workspace.gemm(qa, qb, certified_info=initial_info)
                    report['initial_all_outputs'] = compare(initial)
                    need(initial_info['accepted'] == 1 and initial_info['reject_flags'] == 0,
                         'Initial certificate rejected this diagnostic workload')
                    report['initial_certificate'] = initial_info
                    for _ in range(WARMUP):
                        warm = workspace.gemm(qa, qb)
                        del warm
                    stream.synchronize()
                    report['before_profile_certificate'] = workspace.info()
                    need(report['before_profile_certificate']['submitted_calls'] == 11,
                         'Initial + warmup call count differs')
                    torch.cuda.profiler.start(); started = True
                    report['profile_range_entered'] = True
                    try:
                        final = workspace.gemm(qa, qb)
                        report['profiled_workspace_submissions'] = 1
                        stream.synchronize()
                    finally:
                        torch.cuda.profiler.stop(); started = False
                        report['profile_range_stopped'] = True
                    report['actual_profiled_all_outputs'] = compare(final)
                    report['actual_profiled_certificate'] = workspace.info()
                    need(report['actual_profiled_certificate']['submitted_calls'] == 12
                         and report['actual_profiled_certificate']['accepted'] == 1
                         and report['actual_profiled_certificate']['reject_flags'] == 0,
                         'Actual profiled submission certificate/count differs')
                report['workspace_closed'] = workspace.closed
                report['input_sha256_after'] = fingerprints()
                need(report['input_sha256_after'] == report['input_sha256_before'], 'Input bytes changed')
            stream.synchronize()
        need(source_hashes() == report['source_sha256'] and sha(report['library']['path']) == report['library']['sha256'],
             'Frozen source/library changed')
        report.update(status='passed', passed=True, all_attempted_work_synchronized=True,
                      source_and_library_unchanged=True)
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
    finally:
        if started:
            try:
                torch.cuda.profiler.stop(); report['profile_range_stopped'] = True
            except Exception as error:
                report['profiler_stop_cleanup_error'] = str(error)
        try:
            torch.cuda.synchronize(device); report['all_attempted_work_synchronized'] = True
        except Exception as error:
            report.update(status='failed', passed=False, cleanup_error=str(error), all_attempted_work_synchronized=False)
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        with args.output.open('x') as handle:
            json.dump(report, handle, indent=2, allow_nan=False); handle.write('\n')
        print(json.dumps({key: report[key] for key in ('status', 'passed', 'gpu_executed')}), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
