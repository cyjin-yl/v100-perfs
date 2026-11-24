#!/usr/bin/env python3
"""Frozen explicit-workspace experiment, reusing a completed cache build.

No compilation, remote access, hidden retry or unrelated cleanup occurs here.
The completed cache archive and exact original B flock precede every CUDA
stage. The reviewed H.Runner owns stage process groups and cleanup; the RNE-v2
archive verifier checks all prior result members. Existing libraries/manifests
are used read-only, bound to identical native sources in the new snapshot.
"""
import argparse
import fcntl
import hashlib
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
import types

HELPER = Path('/workspace/user/mxfp4-c9-c10-20261123T2128.py')
HELPER_SHA = 'f7f45628bcf7616a7e7915823c77f2f75602cff66be29bf640c5ec3be8ecce1e'
QUANTIZER_HELPER = Path('/workspace/user/mxfp4-cuda-rne-v2-20261123T2204/results/quantizer-experiment-runner.py')
QUANTIZER_HELPER_SHA = '96d3db40d25426370cfc8542dff908edf00a385c0055019c727c9c005fb8a548'
CACHE_RUNNER_SHA = '65e23628156a4430421cd6ad21359a62e4f2ab922f9daf58ee7d1621b5ecdde7'
DEFAULT_CACHE_RUN = '/workspace/user/mxfp4-c2-cache-20261123T2240'
CACHE_COMMIT = 'c581389551be2233ba23404fa72c471cc4c6f961'
CACHE_SOURCE_SHA = 'a7062abcfa1b9263d7a7bf0d282f2de1b51cfa279939f3b969eacb543f5417c0'
SOURCE_COMMIT = 'f74de5be95687d75758c1f636ffbd00d6b63d161'
SOURCE_SHA = 'aa7ac287d389d1e8705fdeef60713bae1927116c85bb7355a52a817e4a806a89'
SOURCE_BYTES = 157598
SOURCE_FILES = 33
NEW_MODULES = {
    'certified_workspace.py': '03a2222d6efb4ddf5369e01220041c16923da05ceb40235a92e48d3269cfe6cf',
    'check_certified_workspace.py': '00b59608ad22b2d2d236b20c1fa64c0f7110481fac1d6525e4ef74d77bf797fc',
    'bench_certified_workspace.py': 'b0d1a07dc08b8856a31c6226ec25c367264fe438c9559256739fff90247ac4bc',
}
SHAPES = (1024, 4096)
ORDER = ('per_call', 'workspace', 'workspace', 'per_call')
SEED, WARMUP, ROUNDS, ITERS = 20261123, 10, 5, 10
CHECK_CASES = {
    'explicit_lifecycle_and_first_output', 'odd_logical_k_padding_and_native_shape_binding',
    'initial_diagnostic_failure_rolls_back_native_allocation',
    'mutated_inputs_certificate_and_nan_fallback', 'preflight_failures_recover_without_submission',
    'wrong_host_thread_never_enters_native_context',
    'busy_stream_reject_then_completed_stream_switch', 'queued_calls_tensor_lifetime_and_close_wait',
    'partial_native_submission_failure_closes_and_waits', 'context_body_exception_cleans_up',
    'capture_rejection_and_post_capture_recovery',
}
INCLUDED = ['both_FP32_input_validations', 'both_RNE_quantizations', 'packed_operand_allocations',
            'packed_validation', 'Python_and_ctypes_dispatch', 'FP32_output_allocation',
            'preparation', 'certificate', 'GEMM', 'output_scale_or_fallback']
EXCLUDED = ['input_generation', 'correctness_comparisons', 'diagnostic_queries',
            'library_loading', 'timing_event_initialization']
H = Q = None


def reviewed_module(path, expected, name):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
        raise RuntimeError('Reviewed helper is not an owned nonsymlink regular file: ' + str(path))
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError('Reviewed helper hash differs: ' + str(path))
    module = types.ModuleType(name); module.__file__ = str(path); sys.modules[name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def planned_names():
    return ['wait-cache-lock', 'check-certified-workspace'] + [
        f'bench-{size}-{index + 1:02d}-{lifecycle}'
        for size in SHAPES for index, lifecycle in enumerate(ORDER)]


def cache_stage_names():
    return ['build-off', 'build-on', 'build-prepare-micro', 'wait-rnev2-lock'] + [
        f'check-{tag}-{kind}' for tag in ('off', 'on') for kind in ('baseline', 'tiled', 'certified')] + [
        'check-prepare-micro', 'torch-certified-on', 'quantize-cuda-off', 'quantize-cuda-on'] + [
        f'bench-{size}-{index + 1:02d}-{tag}'
        for size in (2048, 4096, 8192) for index, tag in enumerate(('off', 'on', 'on', 'off'))]


def create_runner_class():
    ReviewedQuantizerRunner = Q.create_runner_class()

    class WorkspaceRunner(ReviewedQuantizerRunner):
        def __init__(self, args, run):
            H.Runner.__init__(self, args, run)
            self.own_hash = H.sha(Path(__file__))
            self.binding_hash = None
            self.cache = Path(args.prior_cache_run)
            self.library = self.cache / ('build-' + args.cache_build) / 'libmxfp4_sm70.so'
            self.manifest = self.library.parent / 'compile-evidence.json'
            self.quantizer_report = self.cache / 'results' / ('quantize-cuda-' + args.cache_build + '.json')
            self.prior_files = {}
            self.prior_source_hashes = None
            self.report = {
                'schema_version': 1, 'purpose': 'explicit owner-thread workspace complete-wrapper ABBA experiment',
                'status': 'running', 'passed': False, 'started_at_utc': H.utc(), 'run_directory': str(run),
                'python_execution_path': args.python, 'runner_sha256': self.own_hash,
                'reviewed_native_helper_path': str(HELPER), 'reviewed_native_helper_sha256': HELPER_SHA,
                'reviewed_quantizer_helper_path': str(QUANTIZER_HELPER),
                'reviewed_quantizer_helper_sha256': QUANTIZER_HELPER_SHA,
                'reviewed_prior_cache_runner_sha256': CACHE_RUNNER_SHA,
                'global_budget_seconds': H.BUDGET_SECONDS,
                'absolute_work_cutoff_utc': H.ABSOLUTE_CUTOFF.isoformat(), 'cleanup_reserve_seconds': 15,
                'planned_stages': planned_names(), 'stages': [], 'original_B_root': H.B_ROOT,
                'prior_cache_run': str(self.cache), 'cache_build': args.cache_build,
                'prepare_raw_cache_enabled': args.cache_build == 'on',
                'compiled_in_this_run': False, 'prior_libraries_used_read_only': True,
                'source_expected_commit': SOURCE_COMMIT, 'source_expected_archive_sha256': SOURCE_SHA,
                'source_expected_archive_bytes': SOURCE_BYTES, 'certified_variant': 2, 'tiled_variant': 1,
                'quantizer_backend': 'cuda_rne', 'seed': SEED, 'shapes': list(SHAPES),
                'per_shape_lifecycle_order': list(ORDER), 'warmup': WARMUP, 'rounds': ROUNDS, 'iters': ITERS,
                'input_fingerprints': {}, 'independent_process_runs_per_lifecycle_shape': 2,
                'rounds_are_independent_samples': False, 'gpu_executed': False,
                'automatic_retries': False, 'full_training_executed': False,
                'autograd_workspace_integration': False, 'llm_convergence_validated': False,
                'clock_or_power_changes': False, 'packages_installed': False, 'services_stopped': False,
            }

        def integrity(self):
            H.Runner.integrity(self)
            H.need(H.sha(HELPER) == HELPER_SHA and H.sha(QUANTIZER_HELPER) == QUANTIZER_HELPER_SHA
                   and H.sha(Path(__file__)) == self.own_hash, 'Frozen runner/helper changed')
            if self.binding_hash is not None:
                H.need(H.sha(self.args.source_archive) == SOURCE_SHA
                       and H.sha(self.args.source_binding) == self.binding_hash,
                       'Frozen new source archive/binding changed')
            for name, digest in self.prior_files.items():
                H.regular(name, owned=True)
                H.need(H.sha(name) == digest, 'Read-only prior evidence changed: ' + name)
            if self.prior_source_hashes is not None:
                H.need(H.tree_hashes(self.cache / 'source') == self.prior_source_hashes,
                       'Frozen prior cache source changed')

        def remember(self, path):
            H.regular(path, owned=True)
            self.prior_files[str(path)] = H.sha(path)

        def reuse_build_gate(self, tag):
            folder = self.cache / ('build-' + tag)
            manifest = folder / 'compile-evidence.json'
            data = H.document(manifest); enabled = tag == 'on'
            H.fields(data, {'schema_version': 2, 'passed': True, 'gpu_executed': False,
                'certified_enabled': True, 'certified_variant': 2, 'tiled_variant': 1,
                'quantizer_rne_enabled': True, 'quantizer_rne_backend': 'sm70_mx32_rne_warp_fp32_v1',
                'certified_prepare_raw_cache_enabled': enabled,
                'certified_prepare_raw_cache_backend': 'sm70_c2_row_raw_cache_k8192_v1' if enabled else None,
                'certified_prepare_raw_cache_max_k': 8192 if enabled else None,
                'certified_prepare_raw_cache_bytes_per_mx32': 20 if enabled else None}, 'Reused cache build')
            expected_native = self.wanted_sources([], all_native=True)
            H.need(data.get('source_sha256') == expected_native
                   and data.get('fragment_mapping_sha256') == self.expected[H.MAPPING],
                   'Reused native/mapping sources differ from new frozen snapshot')
            H.need('release 12.8' in data.get('nvcc_version', ''), 'Reused build compiler is not CUDA12.8')
            jobs = data.get('jobs', [])
            H.need(len(jobs) == 5 and {job.get('name') for job in jobs} == set(H.ARTIFACTS)
                   and set(data.get('artifacts', {})) == set(H.ARTIFACTS.values()),
                   'Reused build artifact/job set differs')
            self.remember(manifest)
            for job in jobs:
                argv = job.get('argv', [])
                H.need(argv and argv[0] == H.NVCC and job.get('returncode') == 0
                       and '-DMXFP4_CERTIFIED_VARIANT=2' in argv and '-DMXFP4_TILED_VARIANT=1' in argv
                       and '--ftz=false' in argv and '-gencode=arch=compute_70,code=[sm_70,compute_70]' in argv,
                       'Reused compile command differs')
                defines = [part for part in argv if part.startswith('-DMXFP4_CERTIFIED_PREPARE_RAW_CACHE')]
                H.need(defines == (['-DMXFP4_CERTIFIED_PREPARE_RAW_CACHE=1'] if enabled else []),
                       'Reused cache macro differs')
                if job['name'] == 'shared_library':
                    H.need(str(self.cache / 'source' / H.TRAINING / 'csrc/mxfp4_quantize_rne.cu') in argv,
                           'Reused library did not compile the bound native RNE source')
                log = H.safe_name(job['log'])
                H.need('/' not in log and H.sha(folder / log) == job.get('log_sha256'),
                       'Reused compile log hash differs')
                self.remember(folder / log)
            for name, meta in data['artifacts'].items():
                binary = folder / name; H.regular(binary, owned=True)
                H.need(meta == {'sha256': H.sha(binary), 'bytes': binary.stat().st_size},
                       'Reused artifact bytes/hash differ')
                self.binaries[str(binary)] = meta['sha256']
            return {'passed': True, 'build_tag': tag, 'compiled_in_this_run': False,
                    'compile_manifest_sha256': H.sha(manifest), 'artifacts': data['artifacts'],
                    'native_and_mapping_hashes_match_new_source': True}

        def prior_gate(self):
            completion, evidence = Q.archive_evidence(self.cache, 'native')
            H.fields(completion, {'status': 'passed', 'passed': True, 'gpu_executed': True,
                'source_and_binaries_unchanged': True, 'all_numerical_gates_passed_before_performance': True,
                'source_commit': CACHE_COMMIT, 'source_archive_sha256': CACHE_SOURCE_SHA,
                'runner_sha256': CACHE_RUNNER_SHA, 'certified_variant': 2, 'tiled_variant': 1},
                'Completed cache experiment')
            stages = completion.get('stages', [])
            H.need(len(stages) == 26 and {stage.get('name') for stage in stages} == set(cache_stage_names())
                   and all(stage.get('status') == 'passed' and stage.get('passed') is True for stage in stages),
                   'Completed cache independent26-stage plan did not pass')
            old_hashes = completion.get('source_hashes_before')
            H.need(isinstance(old_hashes, dict) and len(old_hashes) == 30
                   and completion.get('source_hashes_after') == old_hashes,
                   'Prior cache source map is incomplete/changed')
            additions = {H.TRAINING + name: digest for name, digest in NEW_MODULES.items()}
            H.need(not (set(old_hashes) & set(additions)) and self.expected == {**old_hashes, **additions},
                   'New source is not exactly unchanged prior source plus three frozen workspace modules')
            H.need(H.tree_hashes(self.cache / 'source') == old_hashes, 'Prior disk source differs')
            self.prior_source_hashes = old_hashes
            numerical = {}
            for tag in ('off', 'on'):
                for kind, variant in (('baseline', None), ('tiled', None), ('certified', 2)):
                    name = f'check-{tag}-{kind}'
                    numerical[name] = H.native_gate(H.json_lines(self.cache / 'results/stages' / name / 'stdout.log'), variant)
            micro_rows = H.json_lines(self.cache / 'results/stages/check-prepare-micro/stdout.log')
            micro = H.one(micro_rows, 'prepare_cache_summary')
            H.fields(micro, {'status': 'passed', 'passed': True, 'gpu_executed': True,
                'all_submitted_work_completed': True, 'cases': 327, 'direct_cached_vs_original_cases': 323,
                'public_dispatch_cases': 4, 'normalized_half_values': 1079840,
                'rowdata_rows': 1635, 'normalized_half_bit_mismatches': 0, 'rowdata_byte_mismatches': 0,
                'independent_cpu_oracle_mismatches': 0, 'public_output_bit_mismatches': 0,
                'guard_bytes_unchanged': True, 'input_bytes_unchanged': True}, 'Prior actual cache microcheck')
            H.need(len([row for row in micro_rows if row.get('type') == 'prepare_cache_correctness']) == 327,
                   'Prior actual microcheck case count differs')
            numerical['check-prepare-micro'] = micro
            build_evidence = {tag: self.reuse_build_gate(tag) for tag in ('off', 'on')}
            selected_library, selected_manifest = self.library, self.manifest
            try:
                for tag in ('off', 'on'):
                    self.library = self.cache / ('build-' + tag) / 'libmxfp4_sm70.so'
                    self.manifest = self.library.parent / 'compile-evidence.json'
                    path = self.cache / 'results' / ('quantize-cuda-' + tag + '.json')
                    numerical['quantize-cuda-' + tag] = self.quantizer_gate(path)
                    H.need(numerical['quantize-cuda-' + tag]['cases'] == 135
                           and numerical['quantize-cuda-' + tag]['metric_counts'] ==
                           {'byte_comparisons': 147, 'numerical_comparisons': 306},
                           'Prior selected/other RNE135 coverage differs')
                    self.remember(path)
            finally:
                self.library, self.manifest = selected_library, selected_manifest
            for path in (self.cache / 'archive-manifest.json', self.cache / 'results.tar.gz',
                         self.cache / 'results/completion.json',
                         self.cache / 'results/prepare-cache-experiment-runner.py'):
                self.remember(path)
            H.need(H.sha(self.cache / 'results/prepare-cache-experiment-runner.py') == CACHE_RUNNER_SHA,
                   'Prior archived cache runner differs')
            destination = self.results / 'prior-cache'; destination.mkdir(mode=0o700)
            for name in ('archive-manifest.json', 'results/completion.json',
                         'results/quantize-cuda-off.json', 'results/quantize-cuda-on.json',
                         'results/prepare-cache-experiment-runner.py'):
                shutil.copyfile(self.cache / name, destination / Path(name).name)
            for tag in ('off', 'on'):
                target = destination / ('build-' + tag); target.mkdir(mode=0o700)
                for file in sorted((self.cache / ('build-' + tag)).iterdir()):
                    if file.suffix in ('.json', '.log'):
                        H.regular(file, owned=True); shutil.copyfile(file, target / file.name)
            evidence.update(numerical_gates=numerical, reused_builds=build_evidence,
                            selected_build=self.args.cache_build,
                            source_delta='three_new_workspace_python_modules_only')
            self.report['binary_hashes_before_gpu'] = dict(self.binaries)
            self.report['read_only_prior_file_hashes_before'] = dict(self.prior_files)
            return evidence

        def await_cache(self):
            row = {'name': 'wait-cache-lock', 'status': 'waiting', 'passed': False,
                   'executed': False, 'gpu_executed': False, 'started_at_utc': H.utc()}
            self.report['stages'].append(row); self.save()
            fd = None; started = time.monotonic()
            try:
                H.need(str(Path(H.B_ROOT).resolve(strict=True)) == H.B_ROOT, 'Original B lock identity differs')
                lock = '/tmp/mxfp4-framework-followup-' + str(os.geteuid()) + '-' + hashlib.sha256(H.B_ROOT.encode()).hexdigest()[:20] + '.lock'
                row['same_B_lock_path'] = lock
                fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                info = os.fstat(fd)
                H.need(info.st_uid == os.geteuid() and stat.S_ISREG(info.st_mode), 'Unsafe same-UID GPU lock')
                while time.monotonic() < self.deadline - 15:
                    if (self.cache / 'archive-manifest.json').is_file():
                        try:
                            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            pass
                        else:
                            prior = self.prior_gate()
                            self.lock_fd, fd = fd, None
                            row.update(status='passed', passed=True, same_uid_lock_acquired=True, prior=prior)
                            self.report['prior_cache_evidence'] = prior
                            return True
                    time.sleep(min(2, max(0, self.deadline - 15 - time.monotonic())))
                raise H.Blocked('Deadline before completed cache experiment and same B lock became available')
            except (Exception, KeyboardInterrupt) as error:
                row.update(status='blocked' if isinstance(error, H.Blocked) else 'failed',
                           error_type=type(error).__name__, error=str(error))
                if isinstance(error, KeyboardInterrupt): self.deadline = time.monotonic()
                return False
            finally:
                if fd is not None: os.close(fd)
                row.update(finished_at_utc=H.utc(), wall_seconds=time.monotonic() - started); self.save()

        def checker_sources(self):
            return self.wanted_sources(['check_torch_certified.py', 'check_torch_cuda.py', 'reference.py',
                                       'torch_ops.py', 'certified_workspace.py', 'check_certified_workspace.py'],
                                      all_native=True, mapping=True)

        def workspace_gate(self, path):
            data = H.document(path); cases = data.get('cases', [])
            H.fields(data, {'schema_version': 1, 'status': 'passed', 'passed': True, 'device': 'cuda:0',
                'gpu_executed': True, 'all_attempted_work_synchronized': True,
                'source_unchanged_during_check': True, 'library_unchanged_during_check': True,
                'completed_cases': 11, 'full_training_executed': False,
                'autograd_workspace_integration': False, 'cpu_fallback': False}, 'Workspace independent checker')
            H.need(len(cases) == 11 and {case.get('name') for case in cases} == CHECK_CASES
                   and all(case.get('passed') is True for case in cases), 'Workspace checker case coverage differs')
            H.need(data.get('source_sha256') == self.checker_sources(), 'Workspace checker source map differs')
            self.bound_library(data)
            H.fields(data['library'], {'compiled_variant': 2, 'compile_evidence': str(self.manifest),
                'compile_evidence_sha256': H.sha(self.manifest), 'native_sources_match': True,
                'fragment_mapping_matches': True,
                'context_lifecycle': 'explicit_single_shape_owner_thread_workspace'}, 'Workspace library binding')
            gpu = data.get('gpu', {})
            H.need(gpu.get('actual_SM70') is True and gpu.get('compute_capability') == [7, 0]
                   and 'V100' in gpu.get('name', ''), 'Workspace checker did not execute on V100')
            metrics = Q.walk_metrics(cases)
            H.need(metrics == {'byte_comparisons': 0, 'numerical_comparisons': 16},
                   'Workspace checker independent full-output comparisons differ')
            by_name = {case['name']: case for case in cases}
            repeated = by_name['mutated_inputs_certificate_and_nan_fallback']
            H.need(len(repeated.get('diagnostics', [])) == 5
                   and [d.get('submitted_calls') for d in repeated['diagnostics']] == [1, 2, 3, 4, 5]
                   and [d.get('accepted') for d in repeated['diagnostics']] == [1, 1, 1, 0, 0],
                   'Reused mutated inputs retained stale certificate/count')
            for name in ('initial_diagnostic_failure_rolls_back_native_allocation',
                         'partial_native_submission_failure_closes_and_waits'):
                H.fields(by_name[name], {'native_destroy_calls': 1, 'closed': True}, 'Failure cleanup coverage')
            H.fields(by_name['queued_calls_tensor_lifetime_and_close_wait'],
                     {'submitted_calls': 4, 'close_waited': True}, 'Queued lifetime coverage')
            self.report['workspace_gpu_identity'] = gpu
            self.report['workspace_torch_version'] = data.get('torch')
            return {'passed': True, 'gpu_executed': True, 'cases': 11,
                    'independent_output_comparisons': metrics['numerical_comparisons'],
                    'output_sha256': H.sha(path),
                    'optional_coverage_unavailable': data.get('optional_coverage_unavailable')}

        def benchmark_gate(self, path, size, lifecycle):
            data = H.document(path)
            H.fields(data, {'schema_version': 1, 'status': 'passed', 'passed': True, 'device': 'cuda:0',
                'gpu_executed': True, 'all_attempted_work_synchronized': True,
                'source_unchanged_during_benchmark': True, 'library_unchanged_during_benchmark': True,
                'lifecycle': lifecycle, 'quantizer_backend': 'cuda_rne', 'seed': SEED,
                'shape': {'M': size, 'N': size, 'logical_K': size},
                'warmup': WARMUP, 'rounds': ROUNDS, 'iterations_per_round': ITERS,
                'full_training_executed': False, 'autograd_workspace_integration': False,
                'context_create_destroy_in_timed_operation': lifecycle == 'per_call',
                'calls_in_one_lifecycle': 1 + WARMUP + ROUNDS * ITERS,
                'included_in_timed_operation': INCLUDED,
                'tflops_formula': '2*M*N*logical_K/(median_ms*1e9)'}, 'Workspace complete-wrapper benchmark')
            expected_excluded = EXCLUDED + (['separately_measured_explicit_workspace_create_and_close']
                                            if lifecycle == 'workspace' else [])
            H.need(data.get('excluded_from_repeated_call_timing') == expected_excluded,
                   'Complete-wrapper timing scope changed')
            expected_sources = {**self.checker_sources(), **self.wanted_sources(
                ['bench_certified_workspace.py', 'bench_torch_cuda.py'])}
            H.need(data.get('source_sha256') == expected_sources, 'Benchmark source map differs')
            self.bound_library(data)
            H.fields(data['library'], {'compiled_variant': 2, 'compile_evidence': str(self.manifest),
                'compile_evidence_sha256': H.sha(self.manifest), 'native_sources_match': True,
                'context_lifecycle': lifecycle}, 'Benchmark reused library')
            H.need(data.get('gpu') == self.report['workspace_gpu_identity']
                   and data.get('torch') == self.report['workspace_torch_version'],
                   'Benchmark GPU/Torch differs from checker')
            H.fields(data.get('workspace_prerequisite', {}),
                     {'sha256': H.sha(self.results / 'workspace-check.json'), 'completed_cases': 11,
                      'source_and_library_hashes_match': True, 'executed_inside_benchmark': False},
                     'Workspace benchmark prerequisite')
            H.fields(data.get('quantizer_prerequisite', {}),
                     {'sha256': H.sha(self.quantizer_report), 'completed_cases': 135,
                      'source_and_library_hashes_match': True, 'executed_inside_benchmark': False},
                     'Quantizer benchmark prerequisite')
            for field in ('initial_all_outputs', 'actual_last_timed_all_outputs', 'actual_last_timed_vs_initial'):
                H.fields(data.get(field, {}), {'values': size * size, 'bit_mismatches': 0,
                    'canonical_positive_zero': True, 'all_finite': True}, 'Actual full output bits ' + field)
            first = data.get('initial_certificate', {})
            H.fields(first, {'abi_version': 1, 'm': size, 'n': size, 'padded_k': size,
                'accepted': 1, 'reject_flags': 0, 'metadata_valid': 1, 'submitted_calls': 1},
                'Initial actual submission certificate')
            create_close = data.get('lifecycle_timing', {})
            if lifecycle == 'workspace':
                H.fields(data.get('actual_last_timed_certificate', {}), {'abi_version': 1, 'm': size, 'n': size,
                    'padded_k': size, 'accepted': 1, 'reject_flags': 0, 'metadata_valid': 1,
                    'submitted_calls': 1 + WARMUP + ROUNDS * ITERS}, 'Actual last timed workspace certificate')
                H.need(data.get('workspace_closed') is True and data.get('workspace_bytes', 0) > 0,
                       'Workspace did not close or lacks actual bound allocation size')
                c = create_close.get('context_create_and_initial_abi_query_ms')
                d = create_close.get('context_close_and_final_wait_ms')
                H.need(all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in (c, d)),
                       'Separate create/close timing is invalid')
                H.need(create_close.get('separate_create_close_total_ms') == c + d
                       and create_close.get('create_close_ms_per_all_completed_call') ==
                       (c + d) / (1 + WARMUP + ROUNDS * ITERS), 'Create/close amortization differs')
            else:
                H.need(data.get('actual_last_timed_certificate') is None
                       and data.get('actual_last_timed_certificate_unavailable_reason') ==
                       'per-call contexts are destroyed inside measured operation; no timed diagnostic sync was added',
                       'Per-call destroyed final context diagnostic limitation was concealed')
                H.need(create_close == {'context_create_and_initial_abi_query_ms': None,
                                       'context_close_and_final_wait_ms': None},
                       'Per-call create/destroy cost was incorrectly removed from timing')
            before, after = data.get('input_sha256_before'), data.get('input_sha256_after')
            H.need(isinstance(before, dict) and set(before) == {'a_fp32_bytes', 'b_fp32_bytes'}
                   and before == after and all(isinstance(v, str) and len(v) == 64
                   and all(c in '0123456789abcdef' for c in v) for v in before.values()),
                   'Actual input-byte hash changed or is absent')
            old = self.report['input_fingerprints'].setdefault(str(size), before)
            H.need(old == before, 'ABBA processes used different actual FP32 input bytes')
            timing = data.get('timing', {})
            H.fields(timing, {'primary_clock': 'synchronized_perf_counter'}, 'Primary performance clock')
            wall, event = timing.get('synchronized_wall_round_ms'), timing.get('cuda_event_round_ms')
            H.need(isinstance(wall, list) and isinstance(event, list) and len(wall) == len(event) == ROUNDS
                   and all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in wall + event),
                   'Missing/invalid complete raw timing rounds')
            median = statistics.median(wall)
            H.need(timing.get('median_ms') == median and timing.get('cuda_event_median_ms') == statistics.median(event)
                   and math.isclose(timing.get('useful_tflops', -1), 2 * size**3 / (median * 1e9), rel_tol=1e-12),
                   'Reported performance differs from actual complete timing rounds')
            return {'passed': True, 'gpu_executed': True, 'output_sha256': H.sha(path),
                    'size': size, 'lifecycle': lifecycle, 'input_sha256': before,
                    'timing': timing, 'separate_lifecycle_timing': create_close,
                    'scope': 'both_FP32_quantizations_plus_validated_complete_certified_pipeline'}

        def execute(self):
            if self.deadline <= self.start + 15: raise H.Blocked('Absolute/global deadline exhausted')
            H.need(Path(self.args.python).is_file() and os.access(self.args.python, os.X_OK),
                   'Explicit lexical venv Python is not executable')
            H.need(SOURCE_BYTES > 0 and len(SOURCE_SHA) == 64, 'Runner source binding is not finalized')
            H.need(H.regular(self.args.source_archive, owned=True).st_size == SOURCE_BYTES,
                   'Frozen source archive byte count differs')
            binding, self.expected = H.extract(Path(self.args.source_archive), Path(self.args.source_binding), self.source)
            H.fields(binding, {'source_commit': SOURCE_COMMIT, 'source_sha256': SOURCE_SHA}, 'Frozen workspace source')
            H.need(len(self.expected) == SOURCE_FILES and
                   all(self.expected.get(H.TRAINING + name) == digest for name, digest in NEW_MODULES.items()),
                   'Frozen workspace module/file set differs')
            self.binding_hash = H.sha(self.args.source_binding)
            H.write_json(self.results / 'source-binding.json', binding)
            shutil.copyfile(Path(__file__), self.results / 'workspace-experiment-runner.py')
            shutil.copyfile(HELPER, self.results / 'reviewed-native-helper.py')
            shutil.copyfile(QUANTIZER_HELPER, self.results / 'reviewed-rnev2-helper.py')
            self.report.update(source_commit=SOURCE_COMMIT, source_archive_sha256=SOURCE_SHA,
                               source_archive_bytes=SOURCE_BYTES, source_hashes_before=self.expected,
                               source_binding_sha256=self.binding_hash)
            self.integrity()
            if not self.await_cache(): raise H.Blocked('Completed cache experiment/same B lock prerequisite failed')
            self.integrity()
            output = self.results / 'workspace-check.json'
            if not self.stage('check-certified-workspace', [self.args.python, '-m',
                'tools.mxfp4_training.check_certified_workspace', '--compile-evidence', str(self.manifest),
                '--expected-certified-variant', '2', '--device', 'cuda:0', '--output', str(output)],
                600, lambda rows: self.workspace_gate(output), gpu=True, library=self.library):
                raise H.Blocked('Workspace ownership/lifetime/numerical checker failed')
            self.report['all_numerical_gates_passed_before_performance'] = True; self.save()
            for size in SHAPES:
                for index, lifecycle in enumerate(ORDER):
                    name = f'bench-{size}-{index + 1:02d}-{lifecycle}'
                    output = self.results / (name + '.json')
                    command = [self.args.python, '-m', 'tools.mxfp4_training.bench_certified_workspace',
                        '--compile-evidence', str(self.manifest), '--check-report', str(self.results / 'workspace-check.json'),
                        '--quantizer-check-report', str(self.quantizer_report), '--quantizer-backend', 'cuda_rne',
                        '--expected-certified-variant', '2', '--lifecycle', lifecycle, '--device', 'cuda:0',
                        '--m', str(size), '--n', str(size), '--k', str(size), '--seed', str(SEED),
                        '--warmup', str(WARMUP), '--rounds', str(ROUNDS), '--iters', str(ITERS), '--output', str(output)]
                    if not self.stage(name, command, 900,
                        lambda rows, p=output, s=size, l=lifecycle: self.benchmark_gate(p, s, l),
                        gpu=True, library=self.library):
                        raise H.Blocked('Complete-wrapper correctness/scope control failed; comparison stopped')

        def finish(self, error=None):
            self.cleanup()
            if error is not None: self.report.update(error_type=type(error).__name__, error=str(error))
            seen = {stage['name'] for stage in self.report['stages']}
            for name in planned_names():
                if name not in seen:
                    self.block(name, 'Prerequisite/control/deadline prevented execution: ' + str(error),
                               gpu=name != 'wait-cache-lock')
            try:
                self.integrity()
                self.report['source_hashes_after'] = H.tree_hashes(self.source) if self.expected is not None else None
                self.report['source_and_binaries_unchanged'] = self.expected is not None
                self.report['binary_hashes_after_gpu'] = {path: H.sha(path) for path in self.binaries}
                self.report['read_only_prior_file_hashes_after'] = {path: H.sha(path) for path in self.prior_files}
            except Exception as changed:
                self.report.update(integrity_error=str(changed), source_and_binaries_unchanged=False); error = changed
            try:
                statuses = {stage['status'] for stage in self.report['stages']}
                status = 'failed' if 'failed' in statuses or (error is not None and not isinstance(error, H.Blocked)) else (
                    'blocked' if 'blocked' in statuses or error is not None else 'passed')
                gpu = [stage.get('gpu_executed') for stage in self.report['stages'] if stage.get('executed')]
                self.report.update(status=status, passed=status == 'passed', finished_at_utc=H.utc(),
                    wall_seconds=time.monotonic() - self.start,
                    gpu_executed=True if True in gpu else (None if None in gpu else False))
                H.need(len(self.report['stages']) == 10
                       and {stage['name'] for stage in self.report['stages']} == set(planned_names()),
                       'Final independent10-stage plan differs')
                self.report['result_files_sha256'] = {
                    name: digest for name, digest in H.tree_hashes(self.results).items() if name != 'completion.json'}
                self.save()
                H.need(sum(path.stat().st_size for path in self.results.rglob('*') if path.is_file()) <= H.MAX_ARCHIVE_BYTES,
                       'Workspace results archive exceeds bound')
                archive = self.run / 'results.tar.gz'
                with tarfile.open(archive, 'x:gz') as target:
                    target.add(self.results, arcname='results', recursive=True)
                receipt = {'status': status, 'passed': status == 'passed', 'results_archive': str(archive),
                    'sha256': H.sha(archive), 'bytes': archive.stat().st_size,
                    'completion_sha256': H.sha(self.results / 'completion.json'),
                    'source_commit': self.report.get('source_commit'),
                    'source_archive_sha256': self.report.get('source_archive_sha256'),
                    'gpu_executed': self.report['gpu_executed'], 'runner_sha256': self.own_hash,
                    'reviewed_native_helper_sha256': HELPER_SHA,
                    'reviewed_quantizer_helper_sha256': QUANTIZER_HELPER_SHA,
                    'prior_cache_runner_sha256': CACHE_RUNNER_SHA, 'cache_build': self.args.cache_build,
                    'compiled_in_this_run': False}
                H.write_json(self.run / 'archive-manifest.json', receipt)
                print(json.dumps(receipt), flush=True)
                return {'passed': 0, 'blocked': 2, 'failed': 1}[status]
            finally:
                if self.lock_fd is not None:
                    fcntl.flock(self.lock_fd, fcntl.LOCK_UN); os.close(self.lock_fd); self.lock_fd = None

    return WorkspaceRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-archive', 'source-binding', 'run-directory', 'python'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--prior-cache-run', default=DEFAULT_CACHE_RUN)
    parser.add_argument('--cache-build', choices=('off', 'on'), required=True)
    args = parser.parse_args()
    for name in ('source_archive', 'source_binding', 'run_directory', 'python', 'prior_cache_run'):
        if not os.path.isabs(getattr(args, name)): parser.error('Every path must be explicitly absolute')
    global H, Q
    H = reviewed_module(HELPER, HELPER_SHA, 'mxfp4_frozen_native_helper')
    Q = reviewed_module(QUANTIZER_HELPER, QUANTIZER_HELPER_SHA, 'mxfp4_frozen_rnev2_helper'); Q.H = H
    cache = Path(args.prior_cache_run)
    H.need(cache.is_dir() and not cache.is_symlink() and cache.stat().st_uid == os.geteuid(),
           'Prior cache directory must already exist and be owned')
    cache = cache.resolve(strict=True); args.prior_cache_run = str(cache)
    run = Path(args.run_directory)
    H.need(run.parent.is_dir() and not run.exists() and not run.is_symlink(), 'New run directory must not exist')
    run = run.parent.resolve(strict=True) / run.name
    for old in (cache, Path(H.B_ROOT), QUANTIZER_HELPER.parent.parent):
        H.need(run != old and run not in old.parents and old not in run.parents,
               'New run overlaps a read-only prior run')
    args.framework_run = str(cache)
    run.mkdir(mode=0o700, exist_ok=False)
    runner = create_runner_class()(args, run)
    def interrupted(signum, frame):
        for handled in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(handled, signal.SIG_IGN)
        raise KeyboardInterrupt('Workspace runner received signal ' + str(signum))
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(signum, interrupted)
    error = None
    try:
        runner.execute()
    except (Exception, KeyboardInterrupt) as caught:
        error = caught
    return runner.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
