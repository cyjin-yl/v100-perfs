#!/usr/bin/env python3
"""One-shot frozen CUDA RNE quantizer experiment; no remote operations/retries.

CPU compilation may start before the earlier algorithm screen finishes. Every
CUDA stage waits for both prior finalized archives and the exact same owned B
flock. The reviewed native helper supplies stage execution and owned-process
cleanup, while this runner owns its independent 20-stage plan and final archive.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import stat
import statistics
import sys
import tarfile
import time

HELPER = Path('/workspace/user/mxfp4-c9-c10-20261123T2128.py')
HELPER_SHA = 'f7f45628bcf7616a7e7915823c77f2f75602cff66be29bf640c5ec3be8ecce1e'
SOURCE_COMMIT = 'f338faaa5cbdcd900aa82ae95af00ed5fb770235'
SOURCE_SHA = '58e95bb116ba08d49e1fc0f51e3fa9fae17d442f45bbc0ea617ca163ae099ec9'
SOURCE_BYTES = 138658
SOURCE_FILES = 29
FROZEN_HARNESS = {
    'check_quantize_cuda.py': 'baa4323496efc3c8360ef1d697e2843153dd6c985cb17e18aabe5d9ec3147c38',
    'bench_quantize_cuda.py': '3748307ea1a25a62e719e6275af6190d8302da0389e4eaf849d36f642fb0a53a',
    'bench_torch_cuda.py': 'a7bf192784bdcb3f95542002f706252c98f234ee04b2b8bbedd4ac7a51915ae8',
}
DEFAULT_ALGO_RUN = '/workspace/user/mxfp4-c2-algorithm-20261123T2145/run'
ORDER = ('torch', 'cuda_rne', 'cuda_rne', 'torch')
SHAPES = (1024, 4096)
SEED = 20261123
QUANT_WARMUP, END_TO_END_WARMUP, ROUNDS, ITERS = 10, 3, 5, 10
ARTIFACTS = {'shared_library': 'libmxfp4_sm70.so', 'standalone_harness': 'check_mxfp4',
             'tiled_harness': 'check_mxfp4_tiled', 'certified_harness': 'check_mxfp4_certified'}
GEMM_JOBS = ('torch_fp32_matmul', 'torch_fp16_compute', 'torch_fp16_casts_and_matmul',
             'mxfp4_cuda_quantize_and_gemm', 'mxfp4_cuda_tiled_quantize_and_gemm',
             'mxfp4_cuda_certified_quantize_and_gemm')
ALL_JOBS = set(GEMM_JOBS) | {'mxfp4_quantize_both'}
H = None


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            result.update(block)
    return result.hexdigest()


def load_helper():
    info = HELPER.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or digest(HELPER) != HELPER_SHA:
        raise RuntimeError('The exact owned reviewed helper is missing or changed')
    spec = importlib.util.spec_from_file_location('mxfp4_frozen_native_helper', HELPER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def planned_names():
    return ['build-c2-rne', 'wait-prior-native-algorithm-lock', 'check-c2-baseline', 'check-c2-tiled',
            'check-certified-c2', 'torch-baseline-tiled', 'torch-certified-c2', 'check-quantize-cuda',
            'smoke-cuda_tiled', 'smoke-cuda_certified'] + [
            'quant-latency-' + str(s) for s in SHAPES] + [
            f'end-to-end-{s}-{i + 1:02d}-{q}' for s in SHAPES for i, q in enumerate(ORDER)]


def archive_evidence(root, kind):
    """Verify the complete prior archive, disk completion, and every result hash."""
    H.need(root.is_dir() and not root.is_symlink() and root.stat().st_uid == os.geteuid(),
           'Prior run is not an owned nonsymlink directory')
    receipt_path, completion_path = root / 'archive-manifest.json', root / 'results/completion.json'
    H.regular(receipt_path, owned=True); H.regular(completion_path, owned=True)
    receipt, completion = H.document(receipt_path), H.document(completion_path)
    allowed = ('passed', 'completed_with_blocked_cases') if kind == 'algorithm' else ('passed', 'failed', 'blocked')
    H.need(receipt.get('status') in allowed and completion.get('status') == receipt['status']
           and completion.get('finished_at_utc'), 'Prior run is not a supported terminal result')
    archive = root / 'results.tar.gz'; size = H.regular(archive, owned=True).st_size
    archive_hash = receipt.get('archive_sha256' if kind == 'algorithm' else 'sha256')
    H.need(size == receipt.get('bytes') and size <= H.MAX_ARCHIVE_BYTES and H.sha(archive) == archive_hash,
           'Prior archive size/hash mismatch')
    H.need(H.sha(completion_path) == receipt.get('completion_sha256'), 'Prior completion hash mismatch')
    expected = completion.get('result_files_sha256')
    H.need(isinstance(expected, dict) and expected, 'Prior result-file map is absent')
    found = {}; seen = set(); expanded = 0
    with tarfile.open(archive, 'r:gz') as source:
        members = source.getmembers()
        H.need(len(members) < 10000, 'Prior archive has too many entries')
        for item in members:
            name = H.safe_name(item.name)
            H.need(name not in seen and (name == 'results' or name.startswith('results/')),
                   'Duplicate/out-of-root prior member')
            seen.add(name)
            H.need(item.isdir() or item.isreg(), 'Prior archive has links/special members')
            H.need(not getattr(item, 'sparse', None), 'Prior archive has sparse members')
            if item.isdir(): continue
            expanded += item.size
            H.need(0 <= item.size <= H.MAX_ARCHIVE_BYTES and expanded <= H.MAX_ARCHIVE_BYTES,
                   'Prior archive is too large')
            member_hash = hashlib.sha256(); stream = source.extractfile(item)
            H.need(stream is not None, 'Unreadable prior archive member')
            for block in iter(lambda: stream.read(1024**2), b''):
                member_hash.update(block)
            if name == 'results/completion.json':
                H.need(member_hash.hexdigest() == receipt['completion_sha256'], 'Archived completion differs')
            else:
                found[name[len('results/'):]] = member_hash.hexdigest()
    H.need('results/completion.json' in seen and found == expected, 'Prior complete member hash map differs')
    for name, value in found.items():
        path = root / 'results' / H.safe_name(name)
        H.regular(path, owned=True)
        H.need(H.sha(path) == value, 'Prior disk result differs from archived member: ' + name)
    return completion, {'run': str(root), 'status': receipt['status'], 'archive_sha256': archive_hash,
        'archive_bytes': size, 'receipt_sha256': H.sha(receipt_path),
        'completion_sha256': receipt['completion_sha256'], 'verified_result_files': len(found)}


def native_prior_gate(root):
    completion, evidence = archive_evidence(root, 'native')
    H.need(completion.get('source_and_binaries_unchanged') is True, 'Prior native source/binary integrity failed')
    stages = completion.get('stages', [])
    H.need(stages and all(s.get('status') in ('passed', 'failed', 'blocked') for s in stages),
           'Prior native stages are not all terminal')
    by_name = {s['name']: s for s in stages}
    H.need(len(by_name) == len(stages), 'Prior native stage names repeat')
    wanted = ['check-c2-baseline', 'check-c2-tiled'] + [f'check-certified-c{v}' for v in (2, 8, 9, 10)]
    wanted += ['check-half2-microcheck', 'torch-certified-c8', 'torch-certified-c10', 'torch-baseline-tiled']
    checks = {}
    for name in wanted:
        stage = by_name.get(name, {})
        H.fields(stage, {'status': 'passed', 'passed': True, 'returncode': 0}, 'Prior native numerical stage ' + name)
        H.need(stage.get('validation', {}).get('passed') is True, 'Prior native stage lacks its numerical gate')
        rows = H.json_lines(root / 'results/stages' / name / 'stdout.log')
        if name in ('check-c2-baseline', 'check-c2-tiled'):
            checks[name] = H.native_gate(rows)
        elif name.startswith('check-certified-c'):
            checks[name] = H.native_gate(rows, int(name.rsplit('c', 1)[1]))
        elif name == 'check-half2-microcheck':
            checks[name] = H.micro_gate(rows)
        else:
            data = H.document(root / 'results' / (name + '.json'))
            count = 119 if name == 'torch-baseline-tiled' else 16
            H.fields(data, {'passed': True, 'status': 'passed', 'completed_cases': count,
                           'gpu_executed': True, 'all_attempted_work_synchronized': True}, 'Prior Torch numerical suite')
            H.need(len(data.get('cases', [])) == count and all(c.get('passed') is True for c in data['cases']),
                   'Prior Torch cases incomplete')
            checks[name] = {'passed': True, 'cases': count}
    evidence['numerical_gates'] = checks
    return evidence


def algorithm_prior_gate(root, native_evidence):
    completion, evidence = archive_evidence(root, 'algorithm')
    H.need(completion.get('production_source_and_binary_unchanged') is True, 'Prior algorithm integrity failed')
    prior = completion.get('gpu_serialization_gate', {})
    H.need(prior.get('prior_native_directory') == native_evidence['run']
           and prior.get('archive_sha256') == native_evidence['archive_sha256']
           and prior.get('completion_sha256') == native_evidence['completion_sha256'],
           'Algorithm screen was serialized after a different native run')
    cases = completion.get('cases', [])
    H.need(len(cases) == 3 and len({(c.get('size'), c.get('mode')) for c in cases}) == 3,
           'Prior algorithm case set incomplete')
    passed = []
    for size, mode in ((4096, 'random_block_scales'), (8192, 'random_unit_scale')):
        selected = [c for c in cases if c.get('size') == size and c.get('mode') == mode]
        H.need(len(selected) == 1, 'Required algorithm case missing')
        case = selected[0]
        H.fields(case, {'passed': True, 'status': 'passed', 'returncode': 0}, 'Required prior algorithm case')
        output = root / 'results' / (str(size) + '-' + mode + '.jsonl')
        H.need(H.sha(output) == case.get('output_sha256'), 'Prior algorithm output differs')
        rows = H.json_lines(output); env, summary = H.one(rows, 'screen_environment'), H.one(rows, 'screen_summary')
        for data in (env, summary):
            H.fields(data, {'M': size, 'N': size, 'K': size, 'mode': mode, 'seed': SEED,
                            'fullpipeline_performance_measured': False}, 'Algorithm input/scope')
        H.fields(summary, {'status': 'passed', 'passed': True, 'gpu_executed': True,
                           'all_submitted_work_completed': True, 'certificate_unchanged': True}, 'Algorithm summary')
        H.fields(env.get('certificate', {}), {'accepted': True, 'reject_flags': 0}, 'Algorithm certificate')
        gate = H.one(rows, 'screen_prerequisite')
        H.fields(gate, {'passed': True, 'original_c2_finite_outputs': size**2,
                        'original_c2_negative_zero_outputs': 0}, 'Algorithm C2 reference')
        for key, values in (('original_c2_vs_tiled_all_outputs', size**2), ('original_c2_vs_cpu_selected', 256)):
            H.fields(gate.get(key, {}), {'values': values, 'exact_mismatches': 0, 'nonfinite_values': 0}, 'Algorithm reference')
        algorithms = [r for r in rows if r.get('type') == 'algorithm_result']
        H.need(len(algorithms) in (18, 42) and len({r.get('algorithm') for r in algorithms}) == len(algorithms),
               'Prior algorithm rows missing/duplicated')
        good = []
        for row in algorithms:
            if row.get('status') == 'not_supported':
                H.need(row.get('passed') is False and row.get('timing') is None, 'Unsupported algorithm claimed timing')
                continue
            H.fields(row, {'passed': True, 'status': 'passed', 'successful_gemm_calls': 151,
                           'final_actual_timed_output_checked': True}, 'Prior algorithm final output')
            H.fields(row.get('final_scaled_output_vs_original_c2_all_outputs', {}),
                     {'values': size**2, 'exact_mismatches': 0, 'nonfinite_values': 0,
                      'negative_zero_values': 0}, 'Prior algorithm final full comparison')
            good.append(row['algorithm'])
        H.need('CUBLAS_GEMM_DEFAULT_TENSOR_OP' in good and len(good) == summary.get('supported_verified_algorithms'),
               'Prior algorithm default/supported count failed')
        passed.append({'size': size, 'mode': mode, 'supported_algorithms': len(good), 'output_sha256': H.sha(output)})
    evidence['required_passing_cases'] = passed
    evidence['other_case'] = [{k: c.get(k) for k in ('size', 'mode', 'status', 'passed', 'returncode')}
                              for c in cases if (c.get('size'), c.get('mode')) == (8192, 'random_block_scales')]
    return evidence


def walk_metrics(value):
    counts = {'byte_comparisons': 0, 'numerical_comparisons': 0}
    def visit(node):
        if isinstance(node, dict):
            if 'code_byte_mismatches' in node:
                H.fields(node, {'code_byte_mismatches': 0, 'scale_byte_mismatches': 0}, 'Packed byte comparison')
                H.need(node.get('code_bytes', 0) > 0 and node.get('scale_bytes', 0) > 0, 'Empty packed comparison')
                counts['byte_comparisons'] += 1
            if 'tolerance_failures' in node:
                H.need(node['tolerance_failures'] == 0, 'Numerical tolerance failed')
                if str(node.get('comparison', '')).startswith(('exact', 'bitwise')):
                    H.need(node.get('exact_mismatches') == 0, 'Exact numerical comparison failed')
                counts['numerical_comparisons'] += 1
            for child in node.values(): visit(child)
        elif isinstance(node, list):
            for child in node: visit(child)
    visit(value)
    return counts


def create_runner_class():
    class QuantizerRunner(H.Runner):
        def __init__(self, args, run):
            super().__init__(args, run)
            self.own_hash = H.sha(Path(__file__))
            self.report = {'schema_version': 1, 'purpose': 'frozen C2 plus CUDA RNE quantizer experiment',
                'status': 'running', 'passed': False, 'started_at_utc': H.utc(), 'run_directory': str(run),
                'python_execution_path': args.python, 'nvcc': H.NVCC, 'runner_sha256': self.own_hash,
                'reviewed_helper_path': str(HELPER), 'reviewed_helper_sha256': HELPER_SHA,
                'source_expected_commit': SOURCE_COMMIT, 'source_expected_archive_sha256': SOURCE_SHA,
                'global_budget_seconds': H.BUDGET_SECONDS, 'absolute_work_cutoff_utc': H.ABSOLUTE_CUTOFF.isoformat(),
                'cleanup_reserve_seconds': 15, 'planned_stages': planned_names(), 'stages': [],
                'original_B_root': H.B_ROOT, 'prior_native_run': args.prior_native_run,
                'prior_algorithm_run': args.prior_algorithm_run, 'certified_variant': 2, 'tiled_variant': 1,
                'quantizer_rne_enabled': True, 'seed': SEED, 'quantizer_shapes': list(SHAPES),
                'end_to_end_order_per_shape': list(ORDER), 'quantizer_warmup': QUANT_WARMUP,
                'end_to_end_warmup': END_TO_END_WARMUP, 'rounds': ROUNDS, 'iters': ITERS,
                'smokes_run_before_performance': True, 'independent_process_runs_per_end_to_end_quantizer_shape': 2,
                'rounds_are_independent_samples': False, 'gpu_executed': False, 'automatic_retries': False,
                'training_smoke_executed': False, 'full_training_executed': False,
                'llm_convergence_validated': False, 'clock_or_power_changes': False,
                'packages_installed': False, 'services_stopped': False}
            self.library = self.run / 'build-c2-rne/libmxfp4_sm70.so'
            self.manifest = self.run / 'build-c2-rne/compile-evidence.json'
            self.binding_hash = None

        def integrity(self):
            super().integrity()
            H.need(H.sha(HELPER) == HELPER_SHA and H.sha(Path(__file__)) == self.own_hash, 'Reviewed runner/helper changed')
            if self.binding_hash is not None:
                H.need(H.sha(self.args.source_archive) == SOURCE_SHA and H.sha(self.args.source_binding) == self.binding_hash,
                       'Frozen input archive or binding changed')

        def stage(self, *args, **kwargs):
            passed = super().stage(*args, **kwargs)
            row = self.report['stages'][-1]
            if passed and row.get('validation', {}).get('gpu_executed') is True:
                row['gpu_executed'] = True
                row['gpu_execution_evidence'] = 'validated stage JSON report or native summary'
                H.write_json(self.results / 'stages' / row['name'] / 'stage.json', row)
                self.save()
            return passed

        def build_gate(self):
            build = self.manifest.parent; data = H.document(self.manifest)
            H.fields(data, {'schema_version': 2, 'passed': True, 'gpu_executed': False,
                'certified_enabled': True, 'certified_variant': 2, 'tiled_variant': 1,
                'quantizer_rne_enabled': True, 'quantizer_rne_backend': 'sm70_mx32_rne_warp_fp32_v1'}, 'C2 RNE build')
            H.need('release 12.8' in data.get('nvcc_version', ''), 'Compiler is not CUDA12.8')
            native = {k[len(H.TRAINING):]: v for k, v in self.expected.items() if k.startswith(H.TRAINING + 'csrc/')}
            H.need(data.get('source_sha256') == native and data.get('fragment_mapping_sha256') == self.expected[H.MAPPING],
                   'Compile source/mapping differs from frozen source')
            jobs = data.get('jobs', [])
            H.need(len(jobs) == 4 and {j.get('name') for j in jobs} == set(ARTIFACTS)
                   and set(data.get('artifacts', {})) == set(ARTIFACTS.values()), 'Expected exactly four C2 RNE artifacts')
            for job in jobs:
                argv = job.get('argv', [])
                H.need(argv and argv[0] == H.NVCC and job.get('returncode') == 0
                    and '-DMXFP4_CERTIFIED_VARIANT=2' in argv and '-DMXFP4_TILED_VARIANT=1' in argv
                    and '--ftz=false' in argv and '-gencode=arch=compute_70,code=[sm_70,compute_70]' in argv,
                    'Compile command/architecture/variant differs')
                if job['name'] == 'shared_library':
                    H.need(str(self.source / H.TRAINING / 'csrc/mxfp4_quantize_rne.cu') in argv,
                           'Native RNE source is not in the shared-library compile')
                log = H.safe_name(job['log'])
                H.need('/' not in log and H.sha(build / log) == job.get('log_sha256'), 'Compile log hash differs')
            for name, meta in data['artifacts'].items():
                path = build / name; H.regular(path)
                H.need(meta == {'sha256': H.sha(path), 'bytes': path.stat().st_size}, 'Compiled artifact hash differs')
                self.binaries[str(path)] = meta['sha256']
            return {'passed': True, 'gpu_executed': False, 'artifacts': data['artifacts'],
                    'compile_manifest_sha256': H.sha(self.manifest), 'quantizer_rne_enabled': True}

        def await_prior(self):
            row = {'name': 'wait-prior-native-algorithm-lock', 'status': 'waiting', 'passed': False,
                   'executed': False, 'gpu_executed': False, 'started_at_utc': H.utc()}
            self.report['stages'].append(row); self.save(); fd = None; start = time.monotonic()
            try:
                native = Path(self.args.prior_native_run); algorithm = Path(self.args.prior_algorithm_run)
                H.need(str(Path(H.B_ROOT).resolve(strict=True)) == H.B_ROOT, 'Original B lock identity differs')
                lock = '/tmp/mxfp4-framework-followup-' + str(os.geteuid()) + '-' + hashlib.sha256(H.B_ROOT.encode()).hexdigest()[:20] + '.lock'
                row['same_B_lock_path'] = lock
                fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                info = os.fstat(fd)
                H.need(info.st_uid == os.geteuid() and stat.S_ISREG(info.st_mode), 'Unsafe same-UID GPU serialization lock')
                while time.monotonic() < self.deadline - 15:
                    if (native / 'archive-manifest.json').is_file() and (algorithm / 'archive-manifest.json').is_file():
                        try:
                            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            pass
                        else:
                            native_info = native_prior_gate(native)
                            algorithm_info = algorithm_prior_gate(algorithm, native_info)
                            self.lock_fd, fd = fd, None
                            row.update(status='passed', passed=True, same_uid_lock_acquired=True,
                                       native=native_info, algorithm=algorithm_info)
                            self.report['prior_numerical_evidence'] = {'native': native_info, 'algorithm': algorithm_info}
                            provenance = self.results / 'prior-evidence'; provenance.mkdir(mode=0o700)
                            for tag, root in (('native', native), ('algorithm', algorithm)):
                                shutil.copyfile(root / 'archive-manifest.json', provenance / (tag + '-archive-manifest.json'))
                                shutil.copyfile(root / 'results/completion.json', provenance / (tag + '-completion.json'))
                            return True
                    time.sleep(min(2, max(0, self.deadline - 15 - time.monotonic())))
                raise H.Blocked('Deadline before finalized prior runs and same B GPU lock were available')
            except (Exception, KeyboardInterrupt) as error:
                row.update(status='blocked' if isinstance(error, H.Blocked) else 'failed',
                           error_type=type(error).__name__, error=str(error))
                if isinstance(error, KeyboardInterrupt): self.deadline = time.monotonic()
                return False
            finally:
                if fd is not None: os.close(fd)
                row.update(finished_at_utc=H.utc(), wall_seconds=time.monotonic() - start); self.save()

        def wanted_sources(self, names, all_native=False, mapping=False):
            result = {name: self.expected[H.TRAINING + name] for name in names}
            if all_native:
                result.update({k[len(H.TRAINING):]: v for k, v in self.expected.items()
                               if k.startswith(H.TRAINING + 'csrc/')})
            if mapping: result['fragment_mapping.hpp'] = self.expected[H.MAPPING]
            return result

        def bound_library(self, data):
            H.fields(data.get('library', {}), {'path': str(self.library), 'sha256': H.sha(self.library),
                     'bytes': self.library.stat().st_size}, 'Stage native library identity')

        def torch_gate(self, path, certified):
            value = H.torch_gate(path, 2 if certified else None, self.library, self.manifest if certified else None)
            data = H.document(path)
            H.need(data.get('all_attempted_work_synchronized') is True, 'Torch work was not fully synchronized')
            names = ['check_torch_cuda.py', 'reference.py', 'torch_ops.py']
            if certified:
                names += ['check_torch_certified.py']
            else:
                names += ['csrc/mxfp4_gemm.h', 'csrc/mxfp4_gemm.cu', 'csrc/mxfp4_gemm_tiled.h', 'csrc/mxfp4_gemm_tiled.cu']
            H.need(data.get('source_sha256') == self.wanted_sources(names, certified, certified), 'Torch checker sources differ')
            value['checked_metric_counts'] = walk_metrics(data['cases'])
            value['output_sha256'] = H.sha(path)
            return value

        def quantizer_gate(self, path):
            data = H.document(path); cases = data.get('cases', []); names = [c.get('name') for c in cases]
            count = 135 if 'lazy_negative_input' in names else 134
            H.fields(data, {'status': 'passed', 'passed': True, 'gpu_executed': True,
                'all_attempted_work_synchronized': True, 'source_unchanged_during_check': True,
                'completed_cases': count, 'expected_cases': count,
                'tested_gemm_backends': ['cuda', 'cuda_tiled', 'cuda_certified']}, 'Independent CUDA RNE checker')
            H.need(len(cases) == len(set(names)) == count and all(c.get('passed') is True for c in cases),
                   'Independent checker case set failed')
            self.bound_library(data)
            H.fields(data['library'], {'compile_evidence_sha256': H.sha(self.manifest), 'quantizer_rne_enabled': True,
                'quantizer_backend': 'sm70_mx32_rne_warp_fp32_v1', 'quantizer_abi_version': 1,
                'certified_enabled': True, 'certified_variant': 2, 'tiled_variant': 1}, 'RNE library/compile binding')
            expected = self.wanted_sources(['check_quantize_cuda.py', 'reference.py', 'torch_ops.py', 'build_cuda.py'], True)
            H.need(data.get('source_sha256') == expected, 'RNE checker source hashes differ')
            linear = [c for c in cases if c.get('name', '').startswith('linear_')]
            H.need(len(linear) == 72 and all(c.get('original_fp32_axis_inputs_verified') is True for c in linear),
                   'Original-FP32 F/D/W recipe coverage incomplete')
            H.need(len([c for c in cases if c.get('name', '').startswith('unchanged_sr_')]) == 6,
                   'Stochastic route controls incomplete')
            H.need(data.get('transpose_negative_control', {}).get('effective') is True,
                   'Transpose negative control ineffective')
            counts = walk_metrics(cases)
            H.need(counts['byte_comparisons'] >= 144 and counts['numerical_comparisons'] == 306,
                   'Independent byte/numerical comparisons incomplete')
            return {'passed': True, 'gpu_executed': True, 'cases': count, 'metric_counts': counts,
                    'full_numpy_byte_checks': True, 'output_sha256': H.sha(path),
                    'unsupported_coverage_not_executed': data.get('unsupported_coverage_not_executed')}

        def quant_latency_gate(self, path, size):
            data = H.document(path)
            H.fields(data, {'status': 'passed', 'passed': True, 'gpu_executed': True,
                'all_attempted_work_synchronized': True, 'source_unchanged_during_benchmark': True,
                'shape_A': [size, size], 'shape_B': [size, size], 'seed': SEED,
                'rounds_per_backend': ROUNDS, 'iterations_per_round': ITERS, 'warmup_per_backend': QUANT_WARMUP,
                'input_bits_unchanged': True, 'packed_tensor_cache': False, 'flops_or_tflops': None}, 'Quantizer latency')
            self.bound_library(data)
            expected = self.wanted_sources(['check_quantize_cuda.py', 'bench_quantize_cuda.py', 'reference.py', 'torch_ops.py', 'build_cuda.py'], True)
            H.need(data.get('source_sha256') == expected, 'Quantizer latency sources differ')
            H.need(data.get('checker_prerequisite', {}).get('sha256') == H.sha(self.results / 'check-quantize-cuda.json'),
                   'Quantizer latency checker prerequisite differs')
            rows = data.get('raw_rounds', [])
            H.need(len(rows) == 2 * ROUNDS and all(r.get('passed') is True for r in rows), 'Quantizer timing rows incomplete')
            H.need(walk_metrics(data['initial_correctness'])['byte_comparisons'] == 4, 'Initial full byte comparisons missing')
            for q in ('torch', 'cuda_rne'):
                selected = [r for r in rows if r.get('quantizer_backend') == q]
                H.need(sorted(r.get('round_index') for r in selected) == list(range(ROUNDS)), 'Quantizer rounds differ')
                for row in selected:
                    H.fields(row, {'iterations': ITERS, 'byte_checks_after_both_timing_endpoints': True}, 'Quantizer timed output')
                    H.need(walk_metrics(row['last_timed_outputs_vs_numpy'])['byte_comparisons'] == 2,
                           'Both actual last timed quantized outputs were not checked')
                for clock in ('wall', 'event'):
                    key = clock + '_ms_per_pair'; values = [r[key] for r in selected]
                    H.need(all(math.isfinite(v) and v > 0 for v in values), 'Quantizer timing invalid')
                    observed = data['summary']['backends'][q][key]
                    H.need(observed['raw_round_values'] == values and observed['median'] == statistics.median(values),
                           'Quantizer median/raw rounds differ')
            return {'passed': True, 'gpu_executed': True, 'output_sha256': H.sha(path),
                    'input_fingerprint': data['input_fingerprint'], 'summary': data['summary'],
                    'clock_monitor': data.get('clock_monitor'), 'raw_rounds_retained_in_output': True}

        def end_to_end_gate(self, path, size, quantizer):
            data = H.document(path)
            H.fields(data, {'status': 'passed', 'passed': True, 'gpu_executed': True,
                'all_attempted_work_synchronized': True, 'source_unchanged_during_benchmark': True,
                'quantizer_backend': quantizer, 'certified_job_enabled': True,
                'native_certified_capi_benchmarked': True, 'native_certified_workspace_allocated': True,
                'shape': {'M': size, 'N': size, 'logical_K': size, 'padded_K': size},
                'warmup_per_job': END_TO_END_WARMUP, 'rounds': ROUNDS, 'iterations_per_round': ITERS}, 'End-to-end benchmark')
            self.bound_library(data)
            names = ['check_torch_cuda.py', 'reference.py', 'torch_ops.py', 'bench_torch_cuda.py',
                     'csrc/mxfp4_gemm.h', 'csrc/mxfp4_gemm.cu', 'csrc/mxfp4_gemm_tiled.h', 'csrc/mxfp4_gemm_tiled.cu']
            H.need(data.get('source_sha256') == self.wanted_sources(names), 'End-to-end benchmark sources differ')
            H.fields(data.get('input', {}), {'seed': SEED, 'generated': True, 'distribution': 'synthetic_gaussian',
                                          'dtype': 'torch.float32'}, 'End-to-end input')
            H.need(data.get('prerequisite', {}).get('sha256') == H.sha(self.results / 'torch-baseline-tiled.json'),
                   'End-to-end original checker prerequisite differs')
            if quantizer == 'cuda_rne':
                H.need(data.get('quantizer_prerequisite', {}).get('sha256') == H.sha(self.results / 'check-quantize-cuda.json'),
                       'End-to-end RNE checker prerequisite differs')
            validation = data.get('validation', {})
            H.fields(validation, {'passed_before_timing': True, 'passed_after_timing': True,
                'all_input_and_output_values_finite': True, 'final_timed_rne_codes_and_scales_exact': True}, 'End-to-end correctness')
            for key in ('packed_cuda_vs_tiled_all_outputs', 'final_timed_packed_cuda_vs_tiled_all_outputs',
                        'packed_certified_vs_baseline_all_outputs', 'final_timed_certified_vs_baseline_all_outputs'):
                H.fields(validation.get(key, {}), {'values': size**2, 'exact_mismatches': 0, 'all_finite': True}, 'Full packed outputs')
            repeats = validation.get('final_timed_vs_initial_all_outputs', {})
            H.need(set(repeats) == set(GEMM_JOBS), 'Missing final actual timed output repeat check')
            for value in repeats.values(): H.fields(value, {'values': size**2, 'exact_mismatches': 0}, 'Timed output repeat')
            timings = data.get('timings', [])
            H.need(len(timings) == 7 and {t.get('name') for t in timings} == ALL_JOBS, 'Missing/duplicate timing job')
            for timing in timings:
                H.fields(timing, {'primary_clock': 'synchronized_perf_counter',
                    'function_allocations_and_validation_included': True}, 'Complete-wrapper clock')
                for clock, median_key in (('cuda_event_round_ms', 'cuda_event_median_ms'),
                                           ('synchronized_wall_round_ms', 'synchronized_wall_median_ms')):
                    values = timing.get(clock, [])
                    H.need(len(values) == ROUNDS and all(math.isfinite(v) and v > 0 for v in values)
                           and timing[median_key] == statistics.median(values), 'End-to-end timing median/rounds invalid')
                H.need(timing['median_ms'] == timing['synchronized_wall_median_ms'], 'Primary median is not wall time')
                if timing['name'] == 'mxfp4_quantize_both':
                    H.need(timing['useful_tflops'] is None, 'Quantizer wrongly received GEMM FLOPS')
            diagnostic = data.get('certified_diagnostic_untimed_initial', {})
            H.fields(diagnostic, {'abi_version': 1, 'metadata_valid': 1, 'submitted_calls': 1,
                'm': size, 'n': size, 'padded_k': size, 'context_lifecycle': 'per_call_create_gemm_destroy_synchronous'},
                'Untimed certified diagnostic')
            H.need(diagnostic.get('accepted') in (0, 1) and
                ((diagnostic['accepted'] == 1) == (diagnostic.get('reject_flags') == 0)), 'Certified accept/reject inconsistent')
            return {'passed': True, 'gpu_executed': True, 'size': size, 'quantizer_backend': quantizer,
                'output_sha256': H.sha(path), 'timings': timings, 'untimed_initial_certificate': diagnostic,
                'raw_rounds_retained_in_output': True, 'full_initial_and_actual_final_timed_outputs_checked': True,
                'interprocess_input_identity': 'same seed/shape/Torch/GPU; this existing benchmark has no input-byte fingerprint'}

        def smoke_gate(self, path, backend):
            data = H.document(path)
            H.fields(data, {'passed': True, 'gpu_executed': True, 'backend': backend, 'quantizer_backend': 'cuda_rne',
                'device': 'cuda:0', 'seed': SEED, 'optimizer': 'AdamW FP32',
                'llm_convergence_validated': False, 'performance_claim': None}, 'Training smoke')
            expected = {Path(k).name: v for k, v in self.expected.items()
                        if k.startswith(H.TRAINING) and '/' not in k[len(H.TRAINING):] and k.endswith('.py')}
            H.need(data.get('source_sha256') == expected, 'Smoke source set differs')
            wanted = ('fp32', 'mx_forward', 'mx_forward_dgrad', 'mx_forward_dgrad_wgrad', 'mx_all_stochastic_grad')
            arms = data.get('arms', [])
            H.need([a.get('name') for a in arms] == list(wanted), 'Smoke arm set differs')
            for arm in arms:
                H.fields(arm, {'steps': 40, 'passed': True, 'parameters_finite': True,
                    'master_parameters_fp32': True, 'optimizer_states_fp32': True}, 'Smoke arm')
                H.fields(arm.get('recipe', {}), {'backend': backend, 'quantizer_backend': 'cuda_rne'}, 'Smoke quantizer route')
                losses = arm.get('training_losses', [])
                H.need(len(losses) == 40 and all(math.isfinite(v) for v in losses), 'Smoke training losses invalid')
                before, after = arm.get('validation_loss_before'), arm.get('validation_loss_after')
                H.need(math.isfinite(before) and math.isfinite(after) and after < before and
                    arm.get('changed_parameter_tensors', 0) > 0 and math.isfinite(arm.get('max_abs_gradient'))
                    and arm.get('wall_seconds', 0) > 0, 'Smoke loss/updates did not pass')
            self.report['training_smoke_executed'] = True
            return {'passed': True, 'gpu_executed': True, 'output_sha256': H.sha(path), 'arms': arms,
                    'timing_scope': 'instrumented synthetic smoke; not a training throughput benchmark'}

        def execute(self):
            if self.deadline <= self.start + 15: raise H.Blocked('Absolute deadline already exhausted')
            H.need(Path(self.args.python).is_file() and os.access(self.args.python, os.X_OK), 'Lexical venv Python is not executable')
            if not Path(H.NVCC).is_file() or not os.access(H.NVCC, os.X_OK): raise H.Blocked('CUDA12.8 nvcc unavailable')
            H.need(H.regular(self.args.source_archive, owned=True).st_size == SOURCE_BYTES,
                   'Frozen source archive byte count differs')
            binding, self.expected = H.extract(Path(self.args.source_archive), Path(self.args.source_binding), self.source)
            H.fields(binding, {'source_commit': SOURCE_COMMIT, 'source_sha256': SOURCE_SHA}, 'Frozen source binding')
            H.need(len(self.expected) == SOURCE_FILES, 'Frozen source must contain exactly29 files')
            for name, value in FROZEN_HARNESS.items():
                H.need(self.expected.get(H.TRAINING + name) == value, 'Frozen harness differs: ' + name)
            self.binding_hash = H.sha(self.args.source_binding)
            H.write_json(self.results / 'source-binding.json', binding)
            shutil.copyfile(Path(__file__), self.results / 'quantizer-experiment-runner.py')
            shutil.copyfile(HELPER, self.results / 'reviewed-native-stage-helper.py')
            self.report.update(source_commit=SOURCE_COMMIT, source_archive_sha256=SOURCE_SHA,
                source_archive_bytes=SOURCE_BYTES, source_hashes_before=self.expected, source_binding_sha256=self.binding_hash)
            self.integrity()
            command = [self.args.python, '-m', 'tools.mxfp4_training.build_cuda', '--output-dir', str(self.manifest.parent),
                '--nvcc', H.NVCC, '--with-certified', '--with-quantizer-rne', '--certified-variant', '2', '--tiled-variant', '1']
            if not self.stage('build-c2-rne', command, 900, lambda rows: self.build_gate()):
                raise H.Blocked('C2 RNE compilation failed; no GPU stage')
            self.report['binary_hashes_before_gpu'] = dict(self.binaries)
            if not self.await_prior(): raise H.Blocked('Prior terminal numerical evidence and same B lock were not obtained')
            build = self.manifest.parent
            for name, exe, variant in (('check-c2-baseline', 'check_mxfp4', None),
                                       ('check-c2-tiled', 'check_mxfp4_tiled', None),
                                       ('check-certified-c2', 'check_mxfp4_certified', 2)):
                if not self.stage(name, [build / exe, '--device', '0', '--correctness-only'], 600,
                                  lambda rows, v=variant: H.native_gate(rows, v), gpu=True):
                    raise H.Blocked('Native numerical control did not pass: ' + name)
            def python_stage(name, module, options, gate, cap=600):
                output = self.results / (name + '.json')
                argv = [self.args.python, '-m', 'tools.mxfp4_training.' + module,
                        '--output', str(output), '--device', 'cuda:0', '--seed', str(SEED)] + options
                if not self.stage(name, argv, cap, lambda rows: gate(output), gpu=True, library=self.library):
                    raise H.Blocked('Numerical/performance stage did not pass: ' + name)
            python_stage('torch-baseline-tiled', 'check_torch_cuda', [], lambda p: self.torch_gate(p, False))
            python_stage('torch-certified-c2', 'check_torch_certified', ['--compile-evidence', str(self.manifest),
                         '--expected-certified-variant', '2'], lambda p: self.torch_gate(p, True))
            python_stage('check-quantize-cuda', 'check_quantize_cuda', ['--compile-evidence', str(self.manifest)], self.quantizer_gate)
            for backend in ('cuda_tiled', 'cuda_certified'):
                python_stage('smoke-' + backend, 'pretrain_smoke', ['--steps', '40', '--backend', backend,
                             '--quantizer-backend', 'cuda_rne'], lambda p, be=backend: self.smoke_gate(p, be), 900)
            self.report['all_numerical_gates_passed_before_performance'] = True; self.save()
            for size in SHAPES:
                python_stage('quant-latency-' + str(size), 'bench_quantize_cuda',
                    ['--compile-evidence', str(self.manifest), '--check-report', str(self.results / 'check-quantize-cuda.json'),
                     '--size', str(size), '--warmup', str(QUANT_WARMUP), '--rounds', str(ROUNDS), '--iters', str(ITERS)],
                    lambda p, s=size: self.quant_latency_gate(p, s), 900)
            for size in SHAPES:
                for index, quantizer in enumerate(ORDER):
                    python_stage(f'end-to-end-{size}-{index + 1:02d}-{quantizer}', 'bench_torch_cuda',
                        ['--check-report', str(self.results / 'torch-baseline-tiled.json'),
                         '--quantizer-check-report', str(self.results / 'check-quantize-cuda.json'),
                         '--quantizer-backend', quantizer, '--with-certified', '--m', str(size), '--n', str(size), '--k', str(size),
                         '--warmup', str(END_TO_END_WARMUP), '--rounds', str(ROUNDS), '--iters', str(ITERS)],
                        lambda p, s=size, q=quantizer: self.end_to_end_gate(p, s, q), 900)

        def finish(self, error=None):
            self.cleanup()
            if error is not None: self.report.update(error_type=type(error).__name__, error=str(error))
            seen = {s['name'] for s in self.report['stages']}
            for name in planned_names():
                if name not in seen:
                    self.block(name, 'Prerequisite/control/deadline prevented execution: ' + str(error),
                               name not in ('build-c2-rne', 'wait-prior-native-algorithm-lock'))
            try:
                self.integrity()
                self.report['source_hashes_after'] = H.tree_hashes(self.source) if self.expected is not None else None
                self.report['source_and_binaries_unchanged'] = self.expected is not None
                self.report['binary_hashes_after_gpu'] = {p: H.sha(p) for p in self.binaries}
            except Exception as changed:
                self.report.update(integrity_error=str(changed), source_and_binaries_unchanged=False); error = changed
            try:
                for build in sorted(self.run.glob('build-*')):
                    target = self.results / 'build-evidence' / build.name; target.mkdir(parents=True, exist_ok=True)
                    for path in sorted(build.iterdir()):
                        if path.suffix in ('.json', '.log') and path.is_file() and not path.is_symlink():
                            shutil.copyfile(path, target / path.name)
                statuses = {s['status'] for s in self.report['stages']}
                status = 'failed' if 'failed' in statuses or (error is not None and not isinstance(error, H.Blocked)) else (
                         'blocked' if 'blocked' in statuses or error is not None else 'passed')
                gpu = [s.get('gpu_executed') for s in self.report['stages'] if s.get('executed')]
                self.report.update(status=status, passed=status == 'passed', finished_at_utc=H.utc(),
                    wall_seconds=time.monotonic() - self.start,
                    gpu_executed=True if True in gpu else (None if None in gpu else False))
                H.need(len(self.report['stages']) == len(planned_names()) and
                       {s['name'] for s in self.report['stages']} == set(planned_names()), 'Final stage plan differs')
                self.report['result_files_sha256'] = {k: v for k, v in H.tree_hashes(self.results).items() if k != 'completion.json'}
                self.save()
                H.need(sum(p.stat().st_size for p in self.results.rglob('*') if p.is_file()) <= H.MAX_ARCHIVE_BYTES,
                       'Results archive exceeds fixed size bound')
                archive = self.run / 'results.tar.gz'
                with tarfile.open(archive, 'x:gz') as target: target.add(self.results, arcname='results', recursive=True)
                receipt = {'status': status, 'passed': status == 'passed', 'results_archive': str(archive),
                    'sha256': H.sha(archive), 'bytes': archive.stat().st_size,
                    'completion_sha256': H.sha(self.results / 'completion.json'), 'source_commit': self.report.get('source_commit'),
                    'source_archive_sha256': self.report.get('source_archive_sha256'), 'gpu_executed': self.report['gpu_executed'],
                    'runner_sha256': self.own_hash, 'reviewed_helper_sha256': HELPER_SHA}
                H.write_json(self.run / 'archive-manifest.json', receipt)
                print(json.dumps(receipt), flush=True)
                return {'passed': 0, 'blocked': 2, 'failed': 1}[status]
            finally:
                if self.lock_fd is not None:
                    fcntl.flock(self.lock_fd, fcntl.LOCK_UN); os.close(self.lock_fd); self.lock_fd = None

    return QuantizerRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-archive', 'source-binding', 'run-directory', 'prior-native-run', 'python'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--prior-algorithm-run', default=DEFAULT_ALGO_RUN)
    args = parser.parse_args()
    for name in ('source_archive', 'source_binding', 'run_directory', 'prior_native_run', 'prior_algorithm_run', 'python'):
        if not os.path.isabs(getattr(args, name)): parser.error('Every input/execution path must be explicitly absolute')
    global H
    H = load_helper()
    run = Path(args.run_directory)
    H.need(run.parent.is_dir() and not run.exists() and not run.is_symlink(), 'New run directory must not exist')
    run = run.parent.resolve(strict=True) / run.name
    for name in ('prior_native_run', 'prior_algorithm_run'):
        prior = Path(getattr(args, name))
        H.need(prior.is_dir() and not prior.is_symlink() and prior.stat().st_uid == os.geteuid(), 'Prior run is not owned/existing')
        prior = prior.resolve(strict=True); setattr(args, name, str(prior))
        H.need(run != prior and run not in prior.parents and prior not in run.parents, 'New and prior runs overlap')
    H.need(str(run) != H.B_ROOT and Path(H.B_ROOT) not in run.parents and run not in Path(H.B_ROOT).parents,
           'Do not modify original B run')
    args.framework_run = args.prior_native_run  # Required only by the inherited constructor, never its original execute().
    run.mkdir(mode=0o700, exist_ok=False)
    runner = create_runner_class()(args, run)
    def interrupted(signum, frame):
        for handled in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(handled, signal.SIG_IGN)
        raise KeyboardInterrupt('Quantizer runner received signal ' + str(signum))
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(signum, interrupted)
    error = None
    try:
        runner.execute()
    except (Exception, KeyboardInterrupt) as caught:
        error = caught
    return runner.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
