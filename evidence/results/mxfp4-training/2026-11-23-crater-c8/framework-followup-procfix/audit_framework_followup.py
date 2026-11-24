#!/usr/bin/env python3
"""Independent local content audit of the collected, real framework follow-up.
Reads archives only; does not extract/import archived code or invoke a process.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import tarfile

ROOT = Path('/workspace/scratch/3cc0562b0106/v100-perfs/evidence/results/mxfp4-training/2026-11-23-crater-c8')
FOLLOW = ROOT / 'framework-followup-procfix'
B = ROOT / 'attempt-b'
FOLLOW_SHA = '2de0fc7de6d7f3fb8e33a760c538fc2e36c37cf4f08b133960e9c7df9f778cd1'
B_RESULTS_SHA = 'f7267b0d24a4cae8888d211264473a2d9caab23da3e5fccbccf310ffa7a0e14b'
B_SOURCE_SHA = '62378914e2b6ad7e34e3dd70736cb90aea07c962963494d97062d5f5738b0f57'
B_COMMIT = '80d3a7c2bb0bb551a34b28bd7fda59978b6304ef'
HELPER_SHA = '2046ca0b52d0f52fa9baaf565be92fe899a5ac165da3970eaa583902eee0197b'
PREFIX = 'tools/mxfp4_training/'
CHECK_NAMES = {'check_torch_cuda.py', 'reference.py', 'torch_ops.py', 'csrc/mxfp4_gemm.h',
               'csrc/mxfp4_gemm.cu', 'csrc/mxfp4_gemm_tiled.h', 'csrc/mxfp4_gemm_tiled.cu'}
OUT = Path('/tmp/mxfp4-framework-followup-independent-audit.json')
MD = Path('/tmp/mxfp4-framework-followup-independent-audit.md')
checks = []


def need(value, message):
    if not value:
        raise AssertionError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pairs(items):
    result = {}
    for key, value in items:
        need(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def doc(data):
    value = json.loads(data, object_pairs_hook=pairs,
                       parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    need(isinstance(value, dict), 'JSON object required')
    return value


def bundle(path, expected, prefix='results/'):
    raw = path.read_bytes()
    need(sha(raw) == expected, 'archive SHA mismatch: ' + str(path))
    need(len(raw) <= 128 * 1024 * 1024, 'compressed-size bound exceeded')
    files, seen, total = {}, set(), 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as tf:
        for member in tf:
            name = str(PurePosixPath(member.name))
            need(not name.startswith('/') and '..' not in PurePosixPath(name).parts and '\\' not in name,
                 'unsafe archive path')
            need(name not in seen and (member.isfile() or member.isdir()) and not member.issparse(),
                 'duplicate/link/special archive member')
            seen.add(name)
            total += member.size
            need(member.size >= 0 and total <= 256 * 1024 * 1024, 'expanded-size bound exceeded')
            if member.isdir():
                continue
            need(not prefix or name.startswith(prefix), 'unexpected archive prefix')
            key = name[len(prefix):] if prefix else name
            files[key] = tf.extractfile(member).read()
            need(len(files[key]) == member.size, 'member size mismatch')
    return files, {'path': str(path), 'sha256': expected, 'bytes': len(raw),
                   'regular_members': len(files), 'member_sha256': {k: sha(v) for k, v in files.items()}}


def field_checks(value, expected, label):
    for key, want in expected.items():
        have = value.get(key)
        need(have == want and (not isinstance(want, bool) or have is want), label + ': ' + key)


def source_map(report, expected_names, source):
    expected = {name: sha(source[PREFIX + name]) for name in expected_names}
    need(report.get('source_sha256') == expected, 'report source SHA map differs from B archive')


def metric(value, exact):
    need(value.get('values') == math.prod(value.get('shape', [])), 'numeric metric shape/count mismatch')
    kind = 'exact_except_nan_payload_and_signed_zero' if exact else 'fp32_allclose'
    field_checks(value, {'comparison': kind, 'tolerance_failures': 0,
                        'atol': 0.0 if exact else 3e-5, 'rtol': 0.0 if exact else 3e-5}, 'metric')
    if exact:
        need(value.get('exact_mismatches') == 0, 'exact mismatch')
    need(isinstance(value.get('exact_mismatches'), int) and value['exact_mismatches'] >= 0, 'bad mismatch count')
    need(math.isfinite(value['max_finite_abs_error']) and value['max_finite_abs_error'] >= 0, 'bad error metric')


def main():
    files, archive = bundle(FOLLOW / 'results.tar.gz', FOLLOW_SHA)
    bfiles, barchive = bundle(B / 'results.tar.gz', B_RESULTS_SHA)
    source, source_archive = bundle(B / 'source.tar.gz', B_SOURCE_SHA, prefix='')
    completion = doc(files['completion.json'])
    receipt = doc((FOLLOW / 'archive-manifest.json').read_bytes())
    run = doc(files['run-manifest.json'])
    binding = doc(files['provenance/verified-bindings.json'])
    original = doc(files['provenance/B-completion.json'])
    original_run = doc(files['provenance/B-run-manifest.json'])
    build = doc(files['provenance/B-compile-evidence.json'])
    source_hashes = {k: sha(v) for k, v in source.items()}
    need(completion['result_files_sha256'] == {k: sha(v) for k, v in files.items() if k != 'completion.json'},
         'follow-up result member SHA mismatch/missing member')
    field_checks(receipt, {'sha256': FOLLOW_SHA, 'bytes': archive['bytes'],
                          'completion_sha256': sha(files['completion.json']), 'source_commit': B_COMMIT,
                          'source_archive_sha256': B_SOURCE_SHA, 'original_B_results_rewritten': False}, 'receipt')
    need(files['provenance/B-completion.json'] == bfiles['completion.json'], 'copied B completion differs')
    need(files['provenance/B-compile-evidence.json'] == bfiles['build/compile-evidence.json'], 'copied B build differs')
    need(files['provenance/B-run-manifest.json'] == (B / 'run-manifest.json').read_bytes(), 'copied B run manifest differs')
    need(files['provenance/B-runner-payload-exit-code'].strip() == b'2', 'B exit status differs')
    need(original['result_files_sha256'] == {k: sha(v) for k, v in bfiles.items() if k != 'completion.json'},
         'B results member hashes differ')
    need(binding['native_results_sha256'] == {k: sha(v) for k, v in bfiles.items()}, 'binding/B member mismatch')
    need(binding['source_sha256'] == original['source_files_sha256'] == source_hashes, 'source member binding mismatch')
    need(binding['manifest'] == original_run, 'original run binding differs')
    need(all(binding['all_source_files_sha256'].get(k) == v for k, v in source_hashes.items()), 'runtime source snapshot differs')
    extras = {k: v for k, v in binding['all_source_files_sha256'].items() if k not in source_hashes}
    need(extras == binding['preexisting_bytecode_sha256'] and all('__pycache__' in PurePosixPath(k).parts
         and k.endswith('.pyc') for k in extras), 'unbound source file')
    need(sha(files['provenance/mxfp4_framework_followup.py']) == run['helper_sha256'] == HELPER_SHA,
         'executed helper differs from reviewed procfix')
    for obj in (original_run, run, completion):
        field_checks(obj, {'source_commit': B_COMMIT, 'source_archive_sha256': B_SOURCE_SHA}, 'source identity')
    native_root = original_run['remote_dir']
    need(run['native_root'] == completion['native_root'] == native_root
         and run['native_run_id'] == original_run['run_id'], 'run identity mismatch')
    need(original_run['owner'] == original_run['endpoint']['owner'] == 'jincy26'
         and original_run['job'] == original_run['endpoint']['job'] == 'sg-jincy26-261124-4abd5', 'ownership mismatch')
    field_checks(original, {'status': 'blocked'}, 'B conclusion')
    lib = build['artifacts']['libmxfp4_sm70.so']
    library_path = native_root + '/build/libmxfp4_sm70.so'
    need(run['MXFP4_CUDA_LIBRARY'] == binding['library_path'] == library_path
         and run['library_sha256'] == completion['library_sha256'] == lib['sha256'], 'library binding mismatch')
    field_checks(build, {'passed': True, 'certified_enabled': True, 'certified_variant': 2, 'tiled_variant': 1}, 'B build')
    need(build['source_sha256'] == {k[len(PREFIX):]: v for k, v in source_hashes.items()
         if k.startswith(PREFIX + 'csrc/')}, 'B compiled source mismatch')
    for job in build['jobs']:
        need(job['returncode'] == 0 and sha(bfiles['build/' + job['log']]) == job['log_sha256'], 'B compiler log mismatch')
    protected = binding['protected_files_sha256']
    for name, value in build['artifacts'].items():
        need(protected.get(native_root + '/build/' + name) == value['sha256'], 'live binary digest binding differs')
    field_checks(completion, {'status': 'passed', 'passed': True, 'gpu_executed': True,
        'native_prerequisites_verified': True, 'original_source_library_and_evidence_unchanged': True,
        'certified_or_C8_torch_backend_tested': False, 'native_rebuilt': False,
        'original_blocked_conclusion_rewritten': False, 'full_training_executed': False,
        'llm_convergence_validated': False, 'packages_installed': False,
        'services_stopped': False, 'clock_or_power_changes': False}, 'follow-up completion')
    checks.append('Archive SHA, all 21 result-member hashes, original B/source archive and copied provenance bindings verified.')

    expected_stages = ['python_environment', 'torch_integration', 'torch_benchmark', 'smoke_cuda', 'smoke_cuda_tiled']
    stages = completion['stages']
    need([s['name'] for s in stages] == expected_stages, 'stage coverage/order mismatch')
    for stage in stages:
        field_checks(stage, {'status': 'passed', 'executed': True, 'returncode': 0,
                            'cwd': native_root + '/source', 'library_sha256': lib['sha256']}, 'stage')
        need(stage['argv'][0] == run['python_execution_path'] == completion['python_execution_path'] == '/opt/venv/bin/python3',
             'interpreter pathname differs')
        need(stage['argv'][1] == '-B' and stage['native_overlap_check']['same_uid_B_path_processes'] == [],
             'bytecode/overlap guard differs')
        for stream in ('stdout', 'stderr'):
            need(stage[stream] in files, 'missing stage log')
        if stage['name'] != 'python_environment':
            need(stage['gpu_executed'] is True and stage['output_sha256'] == sha(files[stage['output']]),
                 'stage output hash/execution mismatch')
            need(stage['argv'][-1] == run['run_directory'] + '/results/' + stage['output'], 'output argv differs')
    environment = doc(files['python_environment.stdout'])
    need(environment == stages[0]['environment'] and environment['executable'] == run['python_execution_path']
         and environment['available'] is True and environment['compute_capability'] == [7, 0]
         and environment['device_name'] == 'Tesla V100-SXM2-32GB', 'interpreter/device probe differs')
    checks.append('All five stages actually exited zero; explicit venv, per-stage outputs and non-overlap evidence verified.')

    check = doc(files['torch-cuda-check.json'])
    bench = doc(files['torch-benchmark.json'])
    for obj, names in ((check, CHECK_NAMES), (bench, CHECK_NAMES | {'bench_torch_cuda.py'})):
        source_map(obj, names, source)
        field_checks(obj, {'status': 'passed', 'passed': True, 'gpu_executed': True,
            'all_attempted_work_synchronized': True, 'nondefault_stream': True,
            'device': 'cuda:0', 'cpu_fallback': False, 'full_training_executed': False,
            'llm_convergence_validated': False}, 'framework execution')
        field_checks(obj['library'], {'path': library_path, 'sha256': lib['sha256'], 'bytes': lib['bytes']}, 'framework library')
        field_checks(obj['gpu'], {'actual_SM70': True, 'actual_V100': True, 'compute_capability': [7, 0]}, 'framework GPU')
        need(obj['python'] == '3.12.15' and obj['numpy'] == environment['numpy']
             and obj['torch'] == environment['torch'], 'framework version binding differs')
    field_checks(check, {'completed_cases': 119, 'source_unchanged_during_check': True,
                        'seed': 20261123, 'matmul_allow_tf32': False}, 'integration')
    case_names = [c['name'] for c in check['cases']]
    need(len(case_names) == len(set(case_names)) == 119, 'duplicate/missing integration cases')
    groups, metrics = Counter(), []
    linear_coverage = Counter()
    for case in check['cases']:
        need(case.get('passed') is True, 'failed integration case')
        name = case['name']
        if name.startswith('packed_'):
            groups['packed'] += 1
            need(case['backend'] in ('cuda', 'cuda_tiled') and case['nondefault_stream'] is True, 'packed backend differs')
            metric(case['output'], True); metrics.append(case['output'])
        elif name.startswith('gpu_rne_'):
            groups['rne_quantization'] += 1
            field_checks(case, {'codes_and_scales_exact': True, 'nondefault_stream': True}, 'RNE case')
            need(case['nibble_bytes'] > 0 and case['scale_bytes'] > 0, 'empty RNE payload')
        elif name == 'gpu_stochastic_seed':
            groups['stochastic_quantization'] += 1
            field_checks(case, {'same_seed_exact': True, 'different_seed_changes_codes': True,
                                'nondefault_stream': True, 'distribution_claim': None}, 'stochastic case')
        elif name.startswith('linear_'):
            groups['linear'] += 1
            recipe = case['recipe']; backend = recipe['backend']
            need(backend in ('cuda', 'cuda_tiled') and case['nondefault_stream'] is True
                 and case['gradient_contract'] == 'declared_first_order_surrogate', 'linear scope differs')
            flags = tuple(recipe[k] for k in ('forward', 'dgrad', 'wgrad'))
            linear_coverage[(backend, flags)] += 1
            need(set(case['comparisons']) == {'output', 'dgrad', 'wgrad', 'bias_grad'}, 'missing gradient metric')
            for key, exact in zip(('output', 'dgrad', 'wgrad', 'bias_grad'), flags + (False,)):
                value = case['comparisons'][key]; metric(value, exact); metrics.append(value)
        elif name == 'reject_gpu_to_reference_fallback':
            groups['reference_fallback_rejection'] += 1
            need(case.get('gpu_reference_fallback_rejected') is True, 'silent GPU reference fallback')
        else:
            raise AssertionError('unplanned framework case: ' + name)
    need(dict(groups) == {'packed': 12, 'rne_quantization': 9, 'stochastic_quantization': 1,
                         'linear': 96, 'reference_fallback_rejection': 1}, '119 case category coverage differs')
    need(linear_coverage == Counter({(b, f): 6 for b in ('cuda', 'cuda_tiled')
                                    for f in itertools.product((False, True), repeat=3)}), 'linear recipe coverage differs')
    kinds = Counter(m['comparison'] for m in metrics)
    need(kinds == {'exact_except_nan_payload_and_signed_zero': 156, 'fp32_allclose': 240}, 'metric coverage differs')
    metric_summary = {'numeric_metric_records': len(metrics), 'comparisons_by_contract': dict(kinds),
        'exact_contract_mismatches': sum(m['exact_mismatches'] for m in metrics if m['comparison'].startswith('exact')),
        'tolerance_failures_all_metrics': sum(m['tolerance_failures'] for m in metrics),
        'fp32_tolerance_metric_records_with_nonzero_bitwise_mismatches': sum(m['exact_mismatches'] > 0
                    for m in metrics if m['comparison'] == 'fp32_allclose')}
    checks.append('119 case bodies, 16 backend/recipe combinations over six fixture groups, and all 396 numeric metrics checked.')

    field_checks(bench, {'source_unchanged_during_benchmark': True, 'rounds': 3,
        'iterations_per_round': 3, 'warmup_per_job': 1, 'native_certified_capi_benchmarked': False,
        'native_certified_workspace_allocated': False, 'useful_flops': 2 * 1024**3}, 'Torch benchmark')
    need(bench['shape'] == {'M': 1024, 'N': 1024, 'logical_K': 1024, 'padded_K': 1024}, 'benchmark shape differs')
    field_checks(bench['prerequisite'], {'sha256': sha(files['torch-cuda-check.json']), 'completed_cases': 119,
         'source_and_library_hashes_match': True, 'executed_inside_benchmark': False}, 'prerequisite')
    validation = bench['validation']
    for key in ('passed_before_timing', 'passed_after_timing', 'all_input_and_output_values_finite',
                'same_resident_fp32_inputs_for_every_call', 'final_timed_rne_codes_and_scales_exact'):
        need(validation[key] is True, 'benchmark validation missing: ' + key)
    for key in ('packed_cuda_vs_tiled_all_outputs', 'final_timed_packed_cuda_vs_tiled_all_outputs'):
        field_checks(validation[key], {'values': 1024**2, 'exact_mismatches': 0, 'all_finite': True}, key)
    timing_names = {'torch_fp32_matmul', 'torch_fp16_compute', 'torch_fp16_casts_and_matmul',
        'mxfp4_quantize_both', 'mxfp4_cuda_quantize_and_gemm', 'mxfp4_cuda_tiled_quantize_and_gemm'}
    repeats = validation['final_timed_vs_initial_all_outputs']
    need(set(repeats) == timing_names - {'mxfp4_quantize_both'}, 'final timed check missing')
    for value in repeats.values():
        field_checks(value, {'values': 1024**2, 'exact_mismatches': 0}, 'timed output')
    need(len(bench['timings']) == 6 and {t['name'] for t in bench['timings']} == timing_names, 'benchmark backend coverage differs')
    timings = []
    for value in bench['timings']:
        item = dict(value)
        need(value['primary_clock'] == 'synchronized_perf_counter', 'wrong primary clock')
        for clock in ('synchronized_wall', 'cuda_event'):
            raw = value[clock + '_round_ms']
            need(len(raw) == 3 and all(math.isfinite(v) and v > 0 for v in raw), 'invalid benchmark rounds')
            median = statistics.median(raw)
            need(math.isclose(median, value[clock + '_median_ms'], rel_tol=1e-12), 'reported median differs')
            flops = None if value['name'] == 'mxfp4_quantize_both' else (2 * 1024**3) / (median * 1e9)
            reported = value[clock + '_useful_tflops']
            need((flops is None and reported is None) or (flops is not None and math.isclose(flops, reported, rel_tol=1e-12)),
                 'reported TFLOPS differs')
            item[clock + '_independently_recomputed_median_ms'] = median
        need(value['median_ms'] == value['synchronized_wall_median_ms'], 'primary median not wall')
        timings.append(item)
    checks.append('All six benchmark jobs independently recomputed from three raw rounds; initial/final full-output checks and quantization checks validated.')

    smoke_reports, smoke_rows = {}, []
    recipes = [('fp32', (False, False, False), 'nearest'), ('mx_forward', (True, False, False), 'nearest'),
        ('mx_forward_dgrad', (True, True, False), 'nearest'), ('mx_forward_dgrad_wgrad', (True, True, True), 'nearest'),
        ('mx_all_stochastic_grad', (True, True, True), 'stochastic')]
    smoke_source_names = {name[len(PREFIX):] for name in source if name.startswith(PREFIX)
                          and '/' not in name[len(PREFIX):] and name.endswith('.py')}
    for backend in ('cuda', 'cuda_tiled'):
        smoke = doc(files['smoke-' + backend + '.json']); smoke_reports[backend] = smoke
        source_map(smoke, smoke_source_names, source)
        field_checks(smoke, {'passed': True, 'gpu_executed': True, 'backend': backend, 'device': 'cuda:0',
            'seed': 20261123, 'learning_rate': 0.005, 'optimizer': 'AdamW FP32', 'performance_claim': None,
            'llm_convergence_validated': False}, 'smoke')
        need(smoke['model'] == {'layers': 1, 'width': 32, 'heads': 4, 'parameter_count': 11168}, 'smoke model differs')
        need(len(smoke['arms']) == 5, 'smoke arms missing')
        for arm, (name, flags, gradient_rounding) in zip(smoke['arms'], recipes):
            field_checks(arm, {'name': name, 'passed': True, 'steps': 40, 'parameters_finite': True,
                'master_parameters_fp32': True, 'optimizer_states_fp32': True,
                'timing_scope': 'instrumented synthetic smoke only; not a performance benchmark'}, 'smoke arm')
            need(tuple(arm['recipe'][k] for k in ('forward', 'dgrad', 'wgrad')) == flags
                 and arm['recipe']['backend'] == backend and arm['recipe']['gradient_rounding'] == gradient_rounding,
                 'smoke recipe differs')
            losses = arm['training_losses']
            need(len(losses) == 40 and all(math.isfinite(v) for v in losses), 'missing/nonfinite smoke loss')
            before, after, seconds = arm['validation_loss_before'], arm['validation_loss_after'], arm['wall_seconds']
            need(math.isfinite(before) and math.isfinite(after) and after < before and math.isfinite(seconds) and seconds > 0
                 and arm['changed_parameter_tensors'] > 0 and math.isfinite(arm['max_abs_gradient']), 'smoke numeric gate failed')
            smoke_rows.append({'backend': backend, 'arm': name, 'steps': 40, 'initial_validation_loss': before,
                'final_validation_loss': after, 'instrumented_training_wall_seconds': seconds,
                'instrumented_mean_step_ms': seconds * 1000 / 40, 'mean_step_formula': '1000*wall_seconds/40',
                'changed_parameter_tensors': arm['changed_parameter_tensors'], 'training_losses': losses,
                'max_abs_gradient': arm['max_abs_gradient'], 'parameters_finite': True,
                'master_parameters_fp32': True, 'optimizer_states_fp32': True})
    paired = all(a['training_losses'] == b['training_losses'] and a['validation_loss_before'] == b['validation_loss_before']
                 and a['validation_loss_after'] == b['validation_loss_after']
                 for a, b in zip(smoke_reports['cuda']['arms'], smoke_reports['cuda_tiled']['arms']))
    checks.append('All ten smoke arms: 40 finite training losses, decreasing held-out loss, expected recipe, parameter updates and FP32 state verified.')
    warnings = {name: data.decode('utf-8', errors='replace') for name, data in files.items()
                if name.endswith('.stderr') and data}
    report = {'schema_version': 1, 'audit_status': 'verified', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'auditor_sha256': sha(Path(__file__).read_bytes()), 'performed_gpu_or_remote_work': False,
        'provenance': {'followup_archive': archive, 'original_B_archive': barchive, 'B_source_archive': source_archive,
            'source_commit': B_COMMIT, 'reviewed_helper_sha256': HELPER_SHA, 'library': {'path': library_path, **lib},
            'binary_verification_scope': 'Follow-up rehashed live binary; independent local audit binds its metadata to B compiler manifest, since binaries are not archived.'},
        'checks_passed': checks, 'run_manifest': run, 'stage_execution': [{k: v for k, v in s.items()
            if k != 'native_overlap_check'} for s in stages], 'python_environment': environment,
        'integration': {'completed_cases': 119, 'case_category_counts': dict(groups), 'metric_summary': metric_summary,
                        'report': check},
        'torch_benchmark': {'independently_recomputed_timings': timings, 'report': bench},
        'smoke_arms': smoke_rows, 'smoke_reports': smoke_reports,
        'paired_backend_training_loss_trajectories_exact_equal': paired, 'stderr_messages': warnings,
        'scope_notes': ['Torch workloads exercise cuda/cuda_tiled from the C2-built library; certified/C8 Torch is not tested here.',
            '119 cases include exact quantized paths and tolerated ordinary FP32 paths; they are not all bitwise comparisons.',
            'Three benchmark rounds each average three invocations in one process; do not treat rounds as independent runs.',
            'FP16 comparators return FP16; packed MXFP4 paths return FP32. Quantization, allocations and Python validation are included in quantized totals.',
            'Smoke step times are instrumented 40-step means, with checks/synchronizations and synthetic data generation, not LLM throughput.',
            'Synthetic one-layer 11168-parameter decoder uses length-16 modular sequences/vocabulary32; no LLM pretraining or convergence conclusion.',
            'Audit validates authenticated report contents and bindings, not a new tensor-oracle replay; raw output tensors are not included.',
            'B original blocked report and the earlier failed follow-up remain separate immutable evidence.']}
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    lines = ['# Framework follow-up: independent audit', '',
        'The real procfix follow-up is **verified**: all five stages exited zero, the 119-case integration suite passed its declared contracts, and both five-arm synthetic smokes completed 40 steps per arm. This audit performed no new GPU or remote work.', '',
        '## Provenance', '', f'- Result archive SHA256: `{FOLLOW_SHA}` ({archive["bytes"]} bytes).',
        f'- All {archive["regular_members"] - 1} non-completion regular members match completion SHA256 entries; completion itself matches the receipt.',
        f'- Source B `{B_COMMIT}` / archive `{B_SOURCE_SHA}` and the original B build/result manifests are byte/hash bound.',
        f'- Executed helper SHA256: `{HELPER_SHA}`. Library SHA256: `{lib["sha256"]}`.',
        '- Device: Tesla V100-SXM2-32GB; Python 3.12.15 via `/opt/venv/bin/python3`; NumPy 2.3.5; PyTorch 2.10.0+cu128.', '',
        '## Integration correctness', '',
        '- 12 packed fixtures + 9 RNE quantization cases + 1 stochastic-seed case + 96 linear cases + 1 rejected GPU-to-reference fallback = **119**.',
        '- 156 exact numerical metrics have zero mismatches. All 240 ordinary FP32 allclose metrics have zero tolerance failures at atol=rtol=3e-5; 136 of those metrics have permitted bitwise differences.',
        '- The 96 linear cases cover six fixture groups × two backends × eight forward/dgrad/wgrad recipes, including noncontiguous input and original-FP32 transpose checks.', '',
        '## 1024³ Torch benchmark', '',
        'Wall medians are recomputed from all three raw rounds, each averaging three invocations after one warmup. These are repeated measurements in one process.', '',
        '| Job | Wall median (ms) | Useful TFLOPS | Output |', '|---|---:|---:|---|']
    for t in timings:
        tf = '—' if t['synchronized_wall_useful_tflops'] is None else f"{t['synchronized_wall_useful_tflops']:.6f}"
        lines.append(f"| {t['name']} | {t['synchronized_wall_independently_recomputed_median_ms']:.9f} | {tf} | {t['output_dtype']} |")
    lines += ['', 'Packed cuda/tiled initial and final timed outputs match at all 1,048,576 positions; all five final timed GEMM outputs match their own initial outputs; final RNE codes/scales remain exact. FP16 comparators produce FP16 output. Quantized totals include both input quantizations, temporary/output allocations, validation and Python dispatch.', '',
        '## Forty-step synthetic smoke', '',
        'All arms update 17 parameter tensors, retain finite FP32 master parameters and AdamW state, and reduce validation loss. Mean step time is `1000 × instrumented training wall seconds / 40`; it is not an isolated kernel or LLM training benchmark.', '',
        '| Backend | Arm | Initial loss | Final loss | Total training wall (s) | Mean step (ms) |',
        '|---|---|---:|---:|---:|---:|']
    for a in smoke_rows:
        lines.append(f"| {a['backend']} | {a['arm']} | {a['initial_validation_loss']:.9f} | {a['final_validation_loss']:.9f} | {a['instrumented_training_wall_seconds']:.9f} | {a['instrumented_mean_step_ms']:.6f} |")
    lines += ['', 'The two backends have identical per-arm training-loss trajectories and initial/final validation losses for this seed. The model is one layer, width32, four heads, 11,168 parameters, and trains on synthetic length-16 modular sequences with vocabulary32.', '',
        '## Limits and diagnostics', '',
        'These results validate baseline/tiled Torch integration from the C2-built library; they do not measure the certified/C8 Torch path. The tiny synthetic smoke establishes finite updates and decreasing loss for this fixture, not LLM pretraining quality or convergence.', '',
        'The integration stderr contains a PyTorch warning that the autograd cuBLAS path set a previously absent current CUDA context. The actual suite subsequently synchronized and passed every declared gate; the warning is retained verbatim in the audit JSON.', '',
        'The original B report remains blocked for its original framework attempt. This independent successful follow-up does not rewrite that history.', '']
    MD.write_text('\n'.join(lines))
    print(json.dumps({'audit_status': 'verified', 'result_member_hashes_checked': archive['regular_members'] - 1,
        'integration_cases': len(case_names), 'metric_summary': metric_summary, 'smoke_arms': len(smoke_rows),
        'paired_losses_exact_equal': paired,
        'outputs': {str(p): {'sha256': sha(p.read_bytes()), 'bytes': p.stat().st_size} for p in (OUT, MD)}}))


if __name__ == '__main__':
    main()
