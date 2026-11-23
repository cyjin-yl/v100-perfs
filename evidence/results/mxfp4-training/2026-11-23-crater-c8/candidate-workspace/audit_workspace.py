#!/usr/bin/env python3
"""Independently audit an actual explicit-workspace archive, without CUDA.

Prepared before the run; use only with actual downloaded archives and observed
SHA256 values. No archive extraction, archived-code execution, GPU/compiler,
network, synthetic fixtures, mutation of raw evidence, or report overwrite.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys

HELPER_SHA = '054bd0a94b17d48426f65bcd8cfea0566d1c63c167fe342ceed48347288be8b9'
RUNNER_SHA = '4018f7b5b08d3493ee746458cd98dea58925e1bc233b4e456b4adfea2fc9ee03'
NATIVE_HELPER_SHA = 'f7f45628bcf7616a7e7915823c77f2f75602cff66be29bf640c5ec3be8ecce1e'
RNE_HELPER_SHA = '96d3db40d25426370cfc8542dff908edf00a385c0055019c727c9c005fb8a548'
CACHE_RUNNER_SHA = '65e23628156a4430421cd6ad21359a62e4f2ab922f9daf58ee7d1621b5ecdde7'
COMMIT = 'f74de5be95687d75758c1f636ffbd00d6b63d161'
SOURCE_SHA = 'aa7ac287d389d1e8705fdeef60713bae1927116c85bb7355a52a817e4a806a89'
CACHE_COMMIT = 'c581389551be2233ba23404fa72c471cc4c6f961'
CACHE_SOURCE_SHA = 'a7062abcfa1b9263d7a7bf0d282f2de1b51cfa279939f3b969eacb543f5417c0'
TRAINING = 'tools/mxfp4_training/'
MAPPING = 'evidence/benchmarks/sm70-packing/study/simulation/round3/cuda/fragment_mapping.hpp'
MODULES = {
    'certified_workspace.py': '03a2222d6efb4ddf5369e01220041c16923da05ceb40235a92e48d3269cfe6cf',
    'check_certified_workspace.py': '00b59608ad22b2d2d236b20c1fa64c0f7110481fac1d6525e4ef74d77bf797fc',
    'bench_certified_workspace.py': 'b0d1a07dc08b8856a31c6226ec25c367264fe438c9559256739fff90247ac4bc',
}
ORDER = ('per_call', 'workspace', 'workspace', 'per_call')
CHECK_NAMES = {
    'explicit_lifecycle_and_first_output', 'odd_logical_k_padding_and_native_shape_binding',
    'initial_diagnostic_failure_rolls_back_native_allocation', 'mutated_inputs_certificate_and_nan_fallback',
    'preflight_failures_recover_without_submission', 'wrong_host_thread_never_enters_native_context',
    'busy_stream_reject_then_completed_stream_switch', 'queued_calls_tensor_lifetime_and_close_wait',
    'partial_native_submission_failure_closes_and_waits', 'context_body_exception_cleans_up',
    'capture_rejection_and_post_capture_recovery',
}
INCLUDED = ['both_FP32_input_validations', 'both_RNE_quantizations', 'packed_operand_allocations',
            'packed_validation', 'Python_and_ctypes_dispatch', 'FP32_output_allocation',
            'preparation', 'certificate', 'GEMM', 'output_scale_or_fallback']
EXCLUDED = ['input_generation', 'correctness_comparisons', 'diagnostic_queries',
            'library_loading', 'timing_event_initialization']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--cache-directory', type=Path, required=True)
    parser.add_argument('--expected-results-sha256', required=True)
    parser.add_argument('--expected-cache-results-sha256', required=True)
    parser.add_argument('--cache-audit', type=Path, required=True)
    parser.add_argument('--cache-audit-sha256', required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    helper_path = args.directory.parent / 'candidate-c9-c10/audit_c9c10.py'
    if sha(helper_path.read_bytes()) != HELPER_SHA:
        raise RuntimeError('Independent local audit helper changed')
    spec = importlib.util.spec_from_file_location('independent_workspace_helpers', helper_path)
    h = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(h)
    need, fields = h.require, h.eq_fields
    outputs = [Path(str(args.output_prefix) + ext) for ext in ('.json', '.md')]
    need(not any(p.exists() for p in outputs), 'Never overwrite audit evidence')
    result = {'schema_version': 1, 'purpose': 'independent actual explicit-workspace archive audit',
              'audit_started_at_utc': datetime.now(timezone.utc).isoformat(),
              'audit_status': 'pending', 'audit_passed': False, 'issues': [],
              'auditor_sha256': sha(Path(__file__).read_bytes()), 'helper_sha256': HELPER_SHA,
              'audit_gpu_executed': False, 'audit_remote_executed': False,
              'archived_code_executed': False, 'performance_rows': [], 'comparisons': []}

    def source(root, expected, commit, count, size):
        bundle = h.Bundle(root / 'source.tar.gz', expected, 16 << 20)
        binding = h.load_json((root / 'source-binding.json').read_bytes())
        fields(binding, {'source_commit': commit, 'source_sha256': expected, 'source_bytes': size}, 'source binding')
        need(bundle.meta['bytes'] == size and len(bundle.files) == len(binding['members']) == count, 'Source size/count differs')
        need(bundle.hashes == {r['name']: r['sha256'] for r in binding['members']}, 'Frozen source members differ')
        for row in binding['members']:
            need(len(bundle.files[row['name']]) == row['bytes'], 'Frozen member byte length differs')
        return bundle, binding

    def archive(root, expected):
        bundle = h.Bundle(root / 'results.tar.gz', expected, 128 << 20, 'results')
        receipt = h.load_json((root / 'archive-manifest.json').read_bytes())
        complete = bundle.doc('completion.json')
        fields(receipt, {'sha256': expected, 'bytes': bundle.meta['bytes'],
                         'completion_sha256': bundle.hashes['completion.json']}, 'result receipt')
        need(complete['result_files_sha256'] == {k: v for k, v in bundle.hashes.items() if k != 'completion.json'},
             'Result member map differs from actual complete archive')
        status = complete['status']
        need(status in ('passed', 'failed', 'blocked'), 'Archive is not terminal')
        fields(receipt, {'status': status, 'passed': status == 'passed'}, 'terminal receipt')
        fields(complete, {'passed': status == 'passed'}, 'terminal result')
        return bundle, complete, receipt

    def info(data, calls, m, n, logical, accepted=None):
        fields(data, {'abi_version': 1, 'device': 0, 'm': m, 'n': n, 'padded_k': (logical + 31) // 32 * 32,
                      'submitted_calls': calls, 'metadata_valid': int(calls > 0)}, 'actual native context metadata')
        need(data['workspace_bytes'] > 0, 'Missing actual workspace allocation')
        if calls:
            u, v = data['max_energy_a'], data['max_energy_b']
            need(type(u) is int and type(v) is int and 0 <= u <= 2**48 + 1 and 0 <= v <= 2**48 + 1, 'Invalid energy cap')
            span = max(data['max_span_a'], data['max_span_b']) <= 12
            energy = u == 0 or v == 0 or u <= 2**48 // v
            nan = data['nan_scales_a'] == data['nan_scales_b'] == 0
            lattice = data['min_scale_a'] + data['min_scale_b'] - 256 >= -126 and data['max_scale_a'] + data['max_scale_b'] - 256 <= 103
            flags = (0 if nan else 1) | (0 if span else 2) | (0 if energy else 4) | (0 if lattice else 8)
            fields(data, {'reject_flags': flags, 'accepted': int(flags == 0)}, 'derived certificate')
            if accepted is not None:
                need(data['accepted'] == int(accepted), 'Certificate admission differs')

    def metrics(value):
        count, values, zeros = 0, 0, 0
        if isinstance(value, dict):
            if 'comparison' in value and 'tolerance_failures' in value:
                h.walk_metrics(value)
                need(value['comparison'].startswith('exact'), 'Workspace checker silently allowed tolerance')
                z = value['canonical_positive_zero_values']
                need(type(z) is int and 0 <= z <= value['values'], 'Invalid canonical-zero count')
                count, values, zeros = 1, value['values'], z
            for child in value.values():
                a, b, c = metrics(child); count += a; values += b; zeros += c
        elif isinstance(value, list):
            for child in value:
                a, b, c = metrics(child); count += a; values += b; zeros += c
        return count, values, zeros

    try:
        src, binding = source(args.directory, SOURCE_SHA, COMMIT, 33, 157598)
        cache_src, cache_binding = source(args.cache_directory, CACHE_SOURCE_SHA, CACHE_COMMIT, 30, 146431)
        additions = {TRAINING + key: value for key, value in MODULES.items()}
        need(src.hashes == {**cache_src.hashes, **additions} and not set(cache_src.hashes) & set(additions),
             'Workspace snapshot changed existing native/default/quantizer source')
        bundle, complete, receipt = archive(args.directory, args.expected_results_sha256)
        cache, cache_complete, cache_receipt = archive(args.cache_directory, args.expected_cache_results_sha256)
        need(sha(args.cache_audit.read_bytes()) == args.cache_audit_sha256, 'Independent cache audit changed')
        cache_audit = h.load_json(args.cache_audit.read_bytes())
        fields(cache_audit, {'audit_passed': True, 'audit_status': 'verified', 'experiment_status': 'passed',
                            'experiment_passed': True, 'complete_numerical_gate_passed': True}, 'independent cache prerequisite')
        fields(cache_audit['result_archive'], {'sha256': cache.meta['sha256'], 'bytes': cache.meta['bytes'],
                                             'completion_sha256': cache.hashes['completion.json']}, 'cache audit archive binding')
        fields(cache_complete, {'status': 'passed', 'passed': True, 'source_commit': CACHE_COMMIT,
                               'source_archive_sha256': CACHE_SOURCE_SHA, 'source_hashes_before': cache_src.hashes,
                               'source_hashes_after': cache_src.hashes, 'source_and_binaries_unchanged': True,
                               'runner_sha256': CACHE_RUNNER_SHA}, 'prior cache completion')
        need(len(cache_complete['stages']) == 26 and all(s['status'] == 'passed' and s['passed'] is True for s in cache_complete['stages']),
             'Prior cache 26-stage suite incomplete')
        fields(complete, {'source_commit': COMMIT, 'source_archive_sha256': SOURCE_SHA,
                          'source_archive_bytes': 157598, 'source_hashes_before': src.hashes,
                          'source_hashes_after': src.hashes, 'source_and_binaries_unchanged': True,
                          'runner_sha256': RUNNER_SHA, 'certified_variant': 2, 'tiled_variant': 1,
                          'compiled_in_this_run': False, 'prior_libraries_used_read_only': True,
                          'quantizer_backend': 'cuda_rne', 'seed': 20261123, 'warmup': 10, 'rounds': 5, 'iters': 10,
                          'per_shape_lifecycle_order': list(ORDER), 'automatic_retries': False,
                          'autograd_workspace_integration': False, 'full_training_executed': False,
                          'llm_convergence_validated': False, 'clock_or_power_changes': False,
                          'services_stopped': False, 'packages_installed': False}, 'workspace run binding/scope')
        fields(receipt, {'source_commit': COMMIT, 'source_archive_sha256': SOURCE_SHA,
                         'runner_sha256': RUNNER_SHA, 'compiled_in_this_run': False}, 'workspace publication')
        need(bundle.doc('source-binding.json') == binding and complete['source_binding_sha256'] == sha((args.directory / 'source-binding.json').read_bytes()),
             'Source publication content/input hash differs')
        for name, expected in (('workspace-experiment-runner.py', RUNNER_SHA),
                               ('reviewed-native-helper.py', NATIVE_HELPER_SHA), ('reviewed-rnev2-helper.py', RNE_HELPER_SHA),
                               ('prior-cache/prepare-cache-experiment-runner.py', CACHE_RUNNER_SHA)):
            need(bundle.hashes[name] == expected, 'Reviewed archived runner/helper differs: ' + name)
        need(bundle.files['prior-cache/completion.json'] == cache.files['completion.json'], 'Copied cache completion differs')
        need(bundle.doc('prior-cache/archive-manifest.json') == cache_receipt, 'Copied cache receipt data differs')
        run, prior_run = complete['run_directory'], cache_complete['run_directory']
        need(complete['prior_cache_run'] == prior_run, 'Prior run path differs')
        tag = complete['cache_build']
        need(tag in ('off', 'on') and receipt['cache_build'] == tag and complete['prepare_raw_cache_enabled'] is (tag == 'on'), 'Selected cache build differs')
        plans = ['wait-cache-lock', 'check-certified-workspace'] + [f'bench-{size}-{i:02d}-{life}'
                 for size in (1024, 4096) for i, life in enumerate(ORDER, 1)]
        need(complete['planned_stages'] == [s['name'] for s in complete['stages']] == plans, 'Actual 10-stage order differs')
        stages = {s['name']: s for s in complete['stages']}
        fields(stages['wait-cache-lock'], {'status': 'passed', 'passed': True, 'executed': False,
                                         'gpu_executed': False, 'same_uid_lock_acquired': True}, 'same B flock')
        need(stages['wait-cache-lock']['prior']['archive_sha256'] == cache.meta['sha256']
             and stages['wait-cache-lock']['prior']['completion_sha256'] == cache.hashes['completion.json'], 'Actual prerequisite archive differs')
        launch = h.load_json((args.directory / 'launch.json').read_bytes())
        launch_argv = launch['argv']
        need(launch.get('runner_sha256') == RUNNER_SHA, 'Launch runner binding differs')
        for flag, wanted in (('--run-directory', run), ('--cache-build', tag),
                             ('--python', complete['python_execution_path'])):
            need(launch_argv.count(flag) == 1 and launch_argv[launch_argv.index(flag) + 1] == wanted, 'Launch option differs: ' + flag)
        # The reviewed controller has a fixed default prior path. The real
        # launch may omit this optional flag; that must still bind identically.
        need(launch_argv.count('--prior-cache-run') <= 1, 'Duplicate prior-cache-run option')
        launch_prior = (launch_argv[launch_argv.index('--prior-cache-run') + 1]
                        if '--prior-cache-run' in launch_argv else '/workspace/user/mxfp4-c2-cache-20261123T2240')
        need(launch_prior == prior_run, 'Launched default/explicit prior cache path differs')

        native = {k[len(TRAINING):]: v for k, v in src.hashes.items() if k.startswith(TRAINING + 'csrc/')}
        builds, binary_hashes, bound_prior_files = {}, {}, {}
        for current in ('off', 'on'):
            prefix = f'prior-cache/build-{current}/'
            oldprefix = f'build-evidence/build-{current}/'
            manifest = bundle.doc(prefix + 'compile-evidence.json')
            need(bundle.files[prefix + 'compile-evidence.json'] == cache.files[oldprefix + 'compile-evidence.json'], 'Reused manifest differs from actual cache archive')
            fields(manifest, {'passed': True, 'gpu_executed': False, 'certified_variant': 2, 'tiled_variant': 1,
                              'certified_enabled': True, 'quantizer_rne_enabled': True,
                              'certified_prepare_raw_cache_enabled': current == 'on', 'source_sha256': native,
                              'fragment_mapping_sha256': src.hashes[MAPPING]}, 'reused build')
            need(len(manifest['jobs']) == len(manifest['artifacts']) == 5, 'Reused build artifact count differs')
            bound_prior_files[prior_run + f'/build-{current}/compile-evidence.json'] = bundle.hashes[prefix + 'compile-evidence.json']
            for job in manifest['jobs']:
                need(job['returncode'] == 0 and bundle.hashes[prefix + job['log']] == job['log_sha256']
                     and bundle.files[prefix + job['log']] == cache.files[oldprefix + job['log']], 'Reused compilation log differs')
                flags = [x for x in job['argv'] if x.startswith('-DMXFP4_CERTIFIED_PREPARE_RAW_CACHE')]
                need(flags == (['-DMXFP4_CERTIFIED_PREPARE_RAW_CACHE=1'] if current == 'on' else []), 'Cache compile macro differs')
                for flag in ('--ftz=false', '-DMXFP4_CERTIFIED_VARIANT=2', '-DMXFP4_TILED_VARIANT=1', '-gencode=arch=compute_70,code=[sm_70,compute_70]'):
                    need(flag in job['argv'], 'Reused numerical compiler flag differs')
                bound_prior_files[prior_run + f'/build-{current}/' + job['log']] = bundle.hashes[prefix + job['log']]
            for name, meta in manifest['artifacts'].items():
                binary_hashes[prior_run + f'/build-{current}/' + name] = meta['sha256']
            quant_name = f'quantize-cuda-{current}.json'
            need(bundle.files['prior-cache/' + quant_name] == cache.files[quant_name], 'Reused quantizer report differs')
            bound_prior_files[prior_run + '/results/' + quant_name] = cache.hashes[quant_name]
            builds[current] = {'data': manifest, 'sha256': bundle.hashes[prefix + 'compile-evidence.json']}
        need(complete['binary_hashes_before_gpu'] == complete['binary_hashes_after_gpu'] == binary_hashes, 'Reused binary changed')
        bound_prior_files.update({prior_run + '/results.tar.gz': cache.meta['sha256'],
            prior_run + '/archive-manifest.json': bundle.hashes['prior-cache/archive-manifest.json'],
            prior_run + '/results/completion.json': cache.hashes['completion.json'],
            prior_run + '/results/prepare-cache-experiment-runner.py': CACHE_RUNNER_SHA})
        need(complete['read_only_prior_file_hashes_before'] == complete['read_only_prior_file_hashes_after'] == bound_prior_files,
             'Read-only prior evidence binding changed or omitted')
        selected = builds[tag]
        lib = {'path': prior_run + f'/build-{tag}/libmxfp4_sm70.so', **selected['data']['artifacts']['libmxfp4_sm70.so']}
        manifest_remote = prior_run + f'/build-{tag}/compile-evidence.json'
        checker_sources = {name: src.hashes[TRAINING + name] for name in
                           ('check_torch_certified.py', 'check_torch_cuda.py', 'reference.py', 'torch_ops.py',
                            'certified_workspace.py', 'check_certified_workspace.py')}
        checker_sources.update(native); checker_sources['fragment_mapping.hpp'] = src.hashes[MAPPING]
        bench_sources = {**checker_sources, **{n: src.hashes[TRAINING + n] for n in ('bench_certified_workspace.py', 'bench_torch_cuda.py')}}
        result['provenance'] = {'source_archive': src.meta, 'results_archive': bundle.meta,
            'source_commit': COMMIT, 'new_modules': MODULES, 'source_member_sha256': src.hashes,
            'result_member_sha256': bundle.hashes, 'prior_cache_archive': cache.meta,
            'cache_independent_audit_sha256': args.cache_audit_sha256, 'source_delta': 'three Python modules only',
            'selected_cache_build': tag, 'selected_library': lib, 'selected_compile_manifest_sha256': selected['sha256'],
            'ELF_bytes_present_locally': False, 'ELF_scope': 'node-recorded hashes cross-bound to old manifests and stage before/after records',
            'launch_sha256': sha((args.directory / 'launch.json').read_bytes()),
            'same_B_lock_path': stages['wait-cache-lock']['same_B_lock_path']}
        for stage in complete['stages'][1:]:
            state = stage['status']; need(state in ('passed', 'failed', 'blocked'), 'Nonterminal stage')
            if not stage.get('executed'):
                need(state == 'blocked', 'Missing execution for nonblocked stage')
                continue
            name = stage['name']; argv = stage['argv']
            need(stage['cwd'] == run + '/source' and argv[:2] == [complete['python_execution_path'], '-m'], 'Actual Python entrypoint binding differs')
            need(argv[2] == 'tools.mxfp4_training.' + ('check_certified_workspace' if name == 'check-certified-workspace' else 'bench_certified_workspace'),
                 'Unexpected checker/benchmark module')
            expected_output = 'workspace-check.json' if name == 'check-certified-workspace' else name + '.json'
            for flag, value in (('--compile-evidence', manifest_remote), ('--expected-certified-variant', '2'),
                                ('--device', 'cuda:0'), ('--output', run + '/results/' + expected_output)):
                need(argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value, 'Executed argument differs: ' + flag)
            prefix = 'stages/' + name + '/'
            need(bundle.doc(prefix + 'stage.json') == stage, 'Stage copy differs')
            need(stage['evidence_sha256'] == {n: bundle.hashes[prefix + n] for n in ('stdout.log', 'stderr.log')}, 'Stage log hashes differ')
            need(stage.get('library_sha256_before') == stage.get('library_sha256_after') == lib['sha256'], 'Actual executed library changed')
            if state == 'passed':
                fields(stage, {'passed': True, 'returncode': 0, 'gpu_executed': True, 'timed_out': False}, 'executed successful stage')
                need(stage['validation']['output_sha256'] == bundle.hashes[expected_output], 'Numerical output binding differs')
        result['stage_counts'] = dict(Counter(s['status'] for s in complete['stages']))
        result['stage_statuses'] = [{k: s.get(k) for k in ('name', 'status', 'passed', 'executed', 'gpu_executed', 'returncode', 'error', 'reason')}
                                    for s in complete['stages']]
        result['experiment_status'], result['experiment_passed'] = complete['status'], complete['passed']
        need(datetime.fromisoformat(stages['check-certified-workspace']['started_at_utc']) >= datetime.fromisoformat(cache_complete['finished_at_utc']),
             'Workspace GPU stages overlap unfinished cache prerequisite')
        if complete['status'] != 'passed':
            # Preserve the actual failure; no partial ABBA throughput is eligible.
            need(any(s['status'] != 'passed' for s in complete['stages']), 'Failure state lacks failed/blocked stage')
            result.update(audit_status='verified_terminal_nonpassing', audit_passed=True,
                          performance_eligible=False, partial_windows_not_published=True)
        else:
            need(all(s['status'] == 'passed' for s in complete['stages']), 'Passing run has incomplete stages')
            check = bundle.doc('workspace-check.json')
            fields(check, {'passed': True, 'status': 'passed', 'gpu_executed': True, 'completed_cases': 11,
                           'all_attempted_work_synchronized': True, 'source_unchanged_during_check': True,
                           'library_unchanged_during_check': True, 'full_training_executed': False,
                           'autograd_workspace_integration': False, 'cpu_fallback': False}, 'actual workspace checker')
            need(check['source_sha256'] == checker_sources, 'Checker source snapshot differs')
            fields(check['library'], {**lib, 'compiled_variant': 2, 'compile_evidence': manifest_remote,
                'compile_evidence_sha256': selected['sha256'], 'native_sources_match': True,
                'fragment_mapping_matches': True, 'context_lifecycle': 'explicit_single_shape_owner_thread_workspace'}, 'checker library')
            fields(check['gpu'], {'name': 'Tesla V100-SXM2-32GB', 'compute_capability': [7, 0],
                                  'actual_SM70': True, 'torch_cuda_build_version': '12.8'}, 'actual GPU')
            cases = {c['name']: c for c in check['cases']}
            need(set(cases) == CHECK_NAMES and len(check['cases']) == 11 and all(c['passed'] is True for c in cases.values()), 'Checker case set differs')
            counts, values, zeros = metrics(check['cases'])
            need(counts == 16 and values == 1455 and zeros >= 96, 'Full checker output accounting differs')
            c = cases['explicit_lifecycle_and_first_output']
            info(c['initial_info'], 0, 8, 12, 64); info(c['submission_info'], 1, 8, 12, 64, True)
            fields(c, {'closed': True, 'idempotent_close': True}, 'explicit lifecycle close')
            for key in ('pre_entry', 'use_after_close', 'reentry'):
                need(c[key]['error_type'] == 'RuntimeError', 'Lifecycle rejected call was not rejected')
            need(set(c['read_only_bindings']) == {'shape', 'device', 'padded_k', 'library_path'}
                 and all(v['error_type'] == 'AttributeError' for v in c['read_only_bindings'].values()), 'Mutable public bindings')
            c = cases['odd_logical_k_padding_and_native_shape_binding']; info(c['final_info'], 1, 3, 5, 33, True)
            need(c['invalid_padding_rejected']['error_type'] == 'ValueError', 'Noncanonical padding not rejected')
            repeated = cases['mutated_inputs_certificate_and_nan_fallback']
            need(len(repeated['diagnostics']) == len(repeated['outputs']) == 5
                 and repeated['same_device_input_storage_mutated_between_calls'] is True, 'Mutation/stale certificate fixture incomplete')
            for i, data in enumerate(repeated['diagnostics'], 1):
                info(data, i, 8, 12, 64, i <= 3)
            need(repeated['outputs'][2]['canonical_positive_zero_values'] == 96, 'Signed-zero input did not canonicalize all outputs')
            need(repeated['diagnostics'][4]['nan_scales_a'] > 0, 'NaN rejection not actually exercised')
            for key in ('initial_diagnostic_failure_rolls_back_native_allocation', 'partial_native_submission_failure_closes_and_waits'):
                fields(cases[key], {'native_destroy_calls': 1, 'closed': True}, 'actual native cleanup')
            fields(cases['partial_native_submission_failure_closes_and_waits'],
                   {'injected_after_real_native_submission': True, 'stream_complete_after_failure': True}, 'post-submit failure cleanup')
            fields(cases['queued_calls_tensor_lifetime_and_close_wait'],
                   {'submitted_calls': 4, 'close_waited': True, 'temporary_operand_references_released_after_enqueue': True}, 'asynchronous lifetime')
            fields(cases['context_body_exception_cleans_up'], {'closed': True, 'stream_complete': True}, 'body exception cleanup')
            thread = cases['wrong_host_thread_never_enters_native_context']
            for key in ('gemm', 'info', 'close'):
                need(thread[key]['error_type'] == 'RuntimeError' and 'entering host thread' in thread[key]['message'], 'Wrong-thread path reached context')
            stream_case = cases['busy_stream_reject_then_completed_stream_switch']
            need(stream_case['different_stream_handles'] is True and stream_case['busy_switch']['error_type'] == 'CertifiedWorkspaceBusy', 'Busy switch gate missing')
            info(stream_case['final_info'], 2, 8, 12, 64, True)
            capture = cases['capture_rejection_and_post_capture_recovery']
            for key in ('gemm_capture_rejected_before_validation', 'destroy_capture_rejected_without_consuming_context'):
                need(capture[key]['error_type'] == 'RuntimeError' and 'CUDA Graph capture' in capture[key]['message'], 'Capture rejection missing')
            result['checker'] = {'cases': 11, 'independent_full_output_comparisons': 16, 'compared_output_values': values,
                'canonical_positive_zero_values': zeros, 'actual_call_counts': [1, 2, 3, 4, 5],
                'mutation_admission': [1, 1, 1, 0, 0], 'optional_coverage_unavailable': check['optional_coverage_unavailable'],
                'raw_sha256': bundle.hashes['workspace-check.json'], 'all_cases': check['cases']}
            fingerprints, raw_runs = {}, []
            pending_rows, pending_comparisons = [], []
            for size in (1024, 4096):
                grouped = {life: {'wall': [], 'event': [], 'lifecycle': []} for life in set(ORDER)}
                for index, life in enumerate(ORDER, 1):
                    name = f'bench-{size}-{index:02d}-{life}'
                    data = bundle.doc(name + '.json')
                    fields(data, {'status': 'passed', 'passed': True, 'gpu_executed': True,
                                  'all_attempted_work_synchronized': True, 'source_unchanged_during_benchmark': True,
                                  'library_unchanged_during_benchmark': True, 'lifecycle': life,
                                  'quantizer_backend': 'cuda_rne', 'seed': 20261123, 'warmup': 10,
                                  'rounds': 5, 'iterations_per_round': 10, 'calls_in_one_lifecycle': 61,
                                  'shape': {'M': size, 'N': size, 'logical_K': size},
                                  'context_create_destroy_in_timed_operation': life == 'per_call',
                                  'full_training_executed': False, 'autograd_workspace_integration': False,
                                  'included_in_timed_operation': INCLUDED}, 'actual complete wrapper benchmark')
                    excluded = EXCLUDED + (['separately_measured_explicit_workspace_create_and_close'] if life == 'workspace' else [])
                    need(data['excluded_from_repeated_call_timing'] == excluded, 'Timing removed production work')
                    need(data['source_sha256'] == bench_sources and data['gpu'] == check['gpu'] and data['torch'] == check['torch'], 'Benchmark source/GPU/runtime differs')
                    fields(data['library'], {**lib, 'compiled_variant': 2, 'compile_evidence': manifest_remote,
                        'compile_evidence_sha256': selected['sha256'], 'native_sources_match': True,
                        'context_lifecycle': life}, 'benchmark library')
                    fields(data['workspace_prerequisite'], {'sha256': bundle.hashes['workspace-check.json'],
                        'completed_cases': 11, 'source_and_library_hashes_match': True, 'executed_inside_benchmark': False}, 'workspace prerequisite')
                    fields(data['quantizer_prerequisite'], {'sha256': cache.hashes[f'quantize-cuda-{tag}.json'],
                        'completed_cases': 135, 'source_and_library_hashes_match': True, 'executed_inside_benchmark': False}, 'quantizer prerequisite')
                    argv = stages[name]['argv']
                    for flag, value in (('--lifecycle', life), ('--quantizer-backend', 'cuda_rne'), ('--m', str(size)),
                        ('--n', str(size)), ('--k', str(size)), ('--seed', '20261123'), ('--warmup', '10'), ('--rounds', '5'), ('--iters', '10')):
                        need(argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value, 'Actual timing command differs')
                    for key in ('initial_all_outputs', 'actual_last_timed_all_outputs', 'actual_last_timed_vs_initial'):
                        fields(data[key], {'values': size**2, 'bit_mismatches': 0, 'canonical_positive_zero': True,
                                          'all_finite': True}, 'initial/actual-last full bit check')
                    info(data['initial_certificate'], 1, size, size, size, True)
                    lifetime = data['lifecycle_timing']
                    if life == 'workspace':
                        info(data['actual_last_timed_certificate'], 61, size, size, size, True)
                        need(data['workspace_closed'] is True and data['workspace_bytes'] == data['initial_certificate']['workspace_bytes'], 'Workspace release/allocation metadata differs')
                        create, close = lifetime['context_create_and_initial_abi_query_ms'], lifetime['context_close_and_final_wait_ms']
                        need(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (create, close)), 'Missing separate resource timing')
                        h.close(lifetime['separate_create_close_total_ms'], create + close, 'resource timing sum')
                        h.close(lifetime['create_close_ms_per_all_completed_call'], (create + close) / 61, '61-operation amortization')
                        grouped[life]['lifecycle'].append(lifetime)
                    else:
                        need(data['actual_last_timed_certificate'] is None and data['actual_last_timed_certificate_unavailable_reason'] ==
                             'per-call contexts are destroyed inside measured operation; no timed diagnostic sync was added', 'Per-call metadata absence concealed')
                        need(lifetime == {'context_create_and_initial_abi_query_ms': None, 'context_close_and_final_wait_ms': None}, 'Per-call resource timing excluded')
                    before, after = data['input_sha256_before'], data['input_sha256_after']
                    need(before == after and set(before) == {'a_fp32_bytes', 'b_fp32_bytes'}
                         and all(re.fullmatch('[0-9a-f]{64}', v) is not None for v in before.values()), 'Input byte fingerprint changed/missing')
                    need(fingerprints.setdefault(str(size), before) == before, 'ABBA processes used different actual FP32 inputs')
                    timing = data['timing']; need(timing['primary_clock'] == 'synchronized_perf_counter', 'Wrong primary clock')
                    wall, event = timing['synchronized_wall_round_ms'], timing['cuda_event_round_ms']
                    need(len(wall) == len(event) == 5 and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in wall + event), 'Incomplete raw timing rounds')
                    h.close(timing['median_ms'], statistics.median(wall), 'wall median')
                    h.close(timing['cuda_event_median_ms'], statistics.median(event), 'event median')
                    h.close(timing['useful_tflops'], 2 * size**3 / (statistics.median(wall) * 1e9), 'full-wrapper TFLOPS')
                    memory = data['memory_admission']
                    budget = 128 * size**2 + 24 * size**2 + 16 * size**2 + (256 << 20)
                    fields(memory, {'conservative_increment_budget_bytes': budget, 'reserve_bytes': 512 << 20}, 'memory admission')
                    need(memory['free_bytes'] >= budget + (512 << 20), 'Memory admission violated')
                    grouped[life]['wall'].extend(wall); grouped[life]['event'].extend(event)
                    raw_runs.append({'name': name, 'raw_sha256': bundle.hashes[name + '.json'], 'lifecycle': life,
                        'size': size, 'timing': timing, 'separate_lifecycle_timing': lifetime,
                        'input_sha256': before, 'initial_certificate': data['initial_certificate'],
                        'actual_last_timed_certificate': data['actual_last_timed_certificate']})
                for life in ('per_call', 'workspace'):
                    times = grouped[life]
                    need(len(times['wall']) == len(times['event']) == 10, 'All ten pooled rounds were not retained')
                    w, e = statistics.median(times['wall']), statistics.median(times['event'])
                    pending_rows.append({'size': size, 'lifecycle': life, 'eligible': True,
                        'pooled_wall_median_ms': w, 'pooled_event_median_ms': e,
                        'wall_useful_tflops': 2 * size**3 / (w * 1e9), 'event_useful_tflops': 2 * size**3 / (e * 1e9),
                        'raw_wall_round_ms': times['wall'], 'raw_event_round_ms': times['event'],
                        'processes': 2, 'rounds_per_process': 5, 'separate_lifecycle_observations': times['lifecycle']})
                a, b = pending_rows[-2:]
                ratio = a['pooled_wall_median_ms'] / b['pooled_wall_median_ms']
                pending_comparisons.append({'size': size, 'workspace_over_per_call_throughput_ratio': ratio,
                    'wall_latency_reduction_percent': 100 * (1 - 1 / ratio),
                    'same_actual_FP32_input_bytes_all_four_processes': True,
                    'create_close_excluded_only_for_explicit_workspace': True})
            need(fingerprints == complete['input_fingerprints'], 'Completion fingerprint map differs')
            resource_rows = [{'run': r['name'], 'size': r['size'], **r['separate_lifecycle_timing']}
                             for r in raw_runs if r['lifecycle'] == 'workspace']
            totals = [r['separate_create_close_total_ms'] for r in resource_rows]
            result.update(audit_status='verified', audit_passed=True, performance_eligible=True,
                          performance_rows=pending_rows, comparisons=pending_comparisons,
                          raw_runs=raw_runs, input_fingerprints=fingerprints,
                          operation_calls_per_process=61, timed_calls_per_process=50,
                          independent_checker_cases=11, repeated_rounds_are_independent_samples=False,
                          separate_resource_timing={'scope': 'first explicit context creation plus final close in each benchmark process',
                              'total_ms_min': min(totals), 'total_ms_max': max(totals), 'observations': resource_rows,
                              'included_in_reported_repeated_call_TFLOPS': False,
                              'not_a_universal_per_context_cold_cost_estimate': True})
        result['limitations'] = [
            'This API is explicit, single-shape, owner-thread reuse; autograd and MXLinear do not use it, and existing default interfaces are unchanged.',
            'Repeated-call timing includes both full quantizations, validation, allocations, Python/ctypes, preparation, certificate, GEMM and epilogue/fallback.',
            'Workspace creation and close are measured separately; per-call creation and synchronous destruction remain inside each timed operation.',
            'The separate amortization divisor 61 includes one initial validation, ten warmups and fifty timed operations, not sixty-one independent cases or training steps.',
            'Each lifecycle has two independent processes per size and ten correlated round observations; no confidence interval or statistical significance is established.',
            'The individual per-call process medians vary; ABBA reduces ordering bias but does not establish a fixed universal speedup or isolate every source of variance.',
            'Only the final actual timed GEMM output is fully validated per process; intermediate timed outputs are not individually copied back.',
            'Per-call final native contexts are destroyed before the timer returns, so no final certificate is available; the initial per-call diagnostic and final output checks are retained.',
            'The large Gaussian output oracle is the unchanged exact tiled path; independent NumPy checks are provided by the workspace and prerequisite suites, not full large-matrix CPU GEMM.',
            'CUDA event intervals include host submission gaps and wrapper synchronizations; they are not isolated kernel timings.',
            'This runner has no continuous NVML clock/load trace and does not assert exclusive device activity; selected cache build is held identical across both lifecycles.',
            'Only node-recorded ELF hashes are available locally, cross-bound to build manifests and before/after execution records.',
            'No real-corpus LLM pretraining, backward/autograd integration, optimizer step, training throughput or peak-memory benefit is established here.',
        ]
    except Exception as error:
        result.update(audit_status='inconsistent_or_incomplete', audit_passed=False, performance_eligible=False,
                      performance_rows=[], comparisons=[])
        result['issues'].append({'type': type(error).__name__, 'message': str(error)})
    result['audit_finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    with outputs[0].open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write('\n')
    lines = ['# 显式 certified workspace 原始归档独立审计', '',
             f"审计：**{result['audit_status']}**；实际实验：**{result.get('experiment_status', 'unverified')}**。", '',
             '本地审计没有执行 GPU、编译器、远端命令或归档代码，也没有修改原始证据。', '']
    if result.get('performance_eligible'):
        lines += ['## 来源与正确性', '',
            f"源 commit `{COMMIT}`，33 个源文件；相对已核验 cache 源仅增加三个 workspace Python 模块。",
            f"本轮复用 cache `{result['provenance']['selected_cache_build']}` 库，未重新编译；四个 ABBA 进程使用同一个实际库。",
            '11 个独立检查案例、16 次全输出精确比较、1,455 个比较值通过；覆盖线程、stream、输入变更后的证书刷新、NaN回退、canonical +0、capture拒绝与异常清理。', '',
            '## 完整两输入量化＋certified GEMM', '',
            '每形状按per_call→workspace→workspace→per_call，逐进程1次初始验证、10次预热、5轮×10次计时。每种生命周期合并全部10个round。四个进程实际FP32输入SHA完全一致。', '',
            '| 形状 | 生命周期 | pooled wall ms | useful TFLOPS |', '|---|---|---:|---:|']
        for row in result['performance_rows']:
            lines.append(f"| {row['size']}³ | {row['lifecycle']} | {row['pooled_wall_median_ms']:.9f} | {row['wall_useful_tflops']:.9f} |")
        lines += ['', 'workspace重复调用计时排除一次性创建与关闭；per_call每次创建、分配和同步销毁仍在计时内。两种路径都包含两输入量化、验证、分配、dispatch、prepare、证书、GEMM和最终缩放／回退。', '',
                  'workspace末次实际timed输出的证书计数为61，结果全量逐位匹配且零值均为+0；per_call最后context已销毁，其缺失诊断被明确保留。没有用重新执行GEMM替换最后计时结果。', '',
                  '## 一次性资源成本必须另计', '',
                  '每个workspace进程的首次创建（含初始ABI查询）与最终关闭实际观察如下。关闭前的最终结果检查已同步GPU；这些观察不代表任意环境的固定冷启动成本。', '',
                  '| 形状／ABBA位置 | 创建 ms | 关闭 ms | 总计 ms |', '|---|---:|---:|---:|']
        for row in result['separate_resource_timing']['observations']:
            position = row['run'].split('-')[2]
            lines.append(f"| {row['size']}³／{position} | {row['context_create_and_initial_abi_query_ms']:.6f} | {row['context_close_and_final_wait_ms']:.6f} | {row['separate_create_close_total_ms']:.6f} |")
        resource = result['separate_resource_timing']
        lines += ['', f"本次总计范围为 **{resource['total_ms_min']:.6f}–{resource['total_ms_max']:.6f} ms**，未包含于上表重复调用TFLOPS。原报告另保留除以61次完整操作的描述性摊销；61含初始验证和warmup，不能偷换为50次计时调用或训练step的平均成本。", '',
                  'per_call两个进程各自的wall median：', '']
        for size in (1024, 4096):
            medians = [r['timing']['median_ms'] for r in result['raw_runs'] if r['size'] == size and r['lifecycle'] == 'per_call']
            lines.append(f"- {size}³：{medians[0]:.6f}／{medians[1]:.6f} ms。")
        lines += ['', '报告保留全部round并按既定ABBA顺序汇总；未锁定频率、没有连续NVML轨迹，不据此声称固定普遍倍率或统计显著性。', '', '## 限制', '']
        lines += ['- ' + item for item in result['limitations']]
    else:
        lines += ['没有发布可比较的性能数字；失败／blocked或不完整窗口不得视为通过。', '',
                  '```json', json.dumps(result['issues'], ensure_ascii=False, indent=2), '```']
    with outputs[1].open('x') as handle:
        handle.write('\n'.join(lines) + '\n')
    print(json.dumps({'audit_status': result['audit_status'], 'audit_passed': result['audit_passed'],
                      'issues': result['issues'], 'outputs': [str(p) for p in outputs]}, ensure_ascii=False))
    return 0 if result['audit_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
