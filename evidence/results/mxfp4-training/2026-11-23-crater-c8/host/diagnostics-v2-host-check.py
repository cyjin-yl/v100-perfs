#!/usr/bin/env python3
"""Bounded local validation of the actual v1 archive and v2 command repair."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import types

ROOT = Path('/workspace/scratch/3cc0562b0106/v100-perfs/evidence/results/mxfp4-training/2026-11-23-crater-c8/post-workspace-diagnostics/attempt-a')
PATHS = {name: Path('/tmp/' + filename) for name, filename in {
    'H': 'mxfp4_native_c9c10_experiment.py', 'Q': 'mxfp4_quantizer_experiment_v2.py',
    'W': 'mxfp4_certified_workspace_experiment.py', 'C': 'mxfp4_c2_prepare_cache_experiment.py',
    'V': 'mxfp4_post_workspace_diagnostics.py', 'D': 'mxfp4_post_workspace_diagnostics_v2.py',
}.items()}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(name, path):
    module = types.ModuleType('host_v2_' + name); module.__file__ = str(path)
    sys.modules[module.__name__] = module
    text = path.read_text(); ast.parse(text)
    exec(compile(text, str(path), 'exec'), module.__dict__)
    return module

m = {name: load(name, path) for name, path in PATHS.items()}
H, Q, W, C, V, D = (m[name] for name in ('H', 'Q', 'W', 'C', 'V', 'D'))
Q.H = H; W.H, W.Q = H, Q; C.H, C.Q = H, Q
for module in (V, D): module.H, module.Q, module.W, module.C = H, Q, W, C
for name, digest in (('H', D.HELPER_SHA), ('Q', D.QSHA), ('W', D.WSHA), ('C', D.CSHA), ('V', D.PRIOR_RUNNER_SHA)):
    assert sha(PATHS[name]) == digest, name
assert sha('/tmp/mxfp4_profile_probe.py') == D.PROBE_SHA
assert (ROOT / 'results.tar.gz').stat().st_size == 826674
assert sha(ROOT / 'results.tar.gz') == D.PRIOR_ARCHIVE_SHA

with tempfile.TemporaryDirectory(prefix='mxfp4-diag-v2-actual-host-') as temporary:
    base = Path(temporary); prior = base / 'prior'; prior.mkdir()
    shutil.copyfile(ROOT / 'archive-manifest.json', prior / 'archive-manifest.json')
    shutil.copyfile(ROOT / 'results.tar.gz', prior / 'results.tar.gz')
    with tarfile.open(prior / 'results.tar.gz', 'r:gz') as archive:
        seen = set(); total = 0
        for member in archive.getmembers():
            name = H.safe_name(member.name)
            assert name not in seen and (name == 'results' or name.startswith('results/'))
            seen.add(name)
            assert member.isdir() or member.isreg()
            assert not getattr(member, 'sparse', None)
            destination = prior / name
            if member.isdir(): destination.mkdir(parents=True, exist_ok=True); continue
            total += member.size
            assert 0 <= member.size <= H.MAX_ARCHIVE_BYTES and total <= H.MAX_ARCHIVE_BYTES
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, destination.open('xb') as target:
                shutil.copyfileobj(source, target)
    completion, evidence = Q.archive_evidence(prior, 'native')
    H.fields(evidence, {'status': 'failed', 'archive_sha256': D.PRIOR_ARCHIVE_SHA,
        'archive_bytes': 826674, 'completion_sha256': D.PRIOR_COMPLETION_SHA}, 'Actual prior archive')
    H.fields(completion, {'status': 'failed', 'runner_sha256': D.PRIOR_RUNNER_SHA,
        'probe_sha256': D.PROBE_SHA, 'source_commit': D.SOURCE_COMMIT,
        'source_archive_sha256': D.SOURCE_SHA, 'source_and_binaries_unchanged': True, 'cache_build': 'off'}, 'Actual prior binding')
    workspace = H.document(prior / 'results/prior-workspace/completion.json')
    cache = H.document(prior / 'results/prior-cache/completion.json')
    assert completion['source_hashes_before'] == completion['source_hashes_after'] == workspace['source_hashes_after']
    assert completion['binary_hashes_before_gpu'] == completion['binary_hashes_after_gpu']
    micro = completion['cache_run'] + '/build-micro/check_mxfp4_prepare_cache'
    assert completion['binary_hashes_after_gpu'][micro] == cache['binary_hashes_after_gpu'][micro]
    expected_binaries = dict(workspace['binary_hashes_after_gpu'])
    expected_binaries[micro] = cache['binary_hashes_after_gpu'][micro]
    assert completion['binary_hashes_after_gpu'] == expected_binaries
    stages = {stage['name']: stage for stage in completion['stages']}
    assert len(stages) == len(completion['stages']) == 7
    assert {name for name, stage in stages.items() if stage['status'] == 'failed'} == {'ncu-gv100-catalog'}
    assert stages['ncu-gv100-catalog']['returncode'] == 0
    assert stages['ncu-gv100-catalog']['error'] == 'Metric catalog exceeds diagnostic bound'
    assert stages['ncu-profile']['status'] == 'blocked' and stages['ncu-profile']['executed'] is False
    for name in ('wait-workspace-lock', 'ncu-version', 'ncu-help', 'compute-sanitizer-version', 'memcheck-prepare-cache'):
        H.fields(stages[name], {'status': 'passed', 'passed': True}, 'Prior stage')
    memcheck = C.micro_gate(H.json_lines(prior / 'results/stages/memcheck-prepare-cache/stdout.log'))
    H.fields(stages['memcheck-prepare-cache'], {'executed': True, 'gpu_executed': True, 'returncode': 0}, 'Actual memcheck')
    H.fields(stages['memcheck-prepare-cache']['validation'], {'passed': True, 'memory_errors': 0,
        'binary_sha256': expected_binaries[micro], 'memcheck_log_sha256': sha(prior / 'results/memcheck.log')}, 'Memcheck binding')
    assert 'ERROR SUMMARY: 0 errors' in (prior / 'results/memcheck.log').read_text()
    actual_help = prior / 'results/stages/ncu-help/stdout.log'
    assert sha(actual_help) == D.ACTUAL_HELP_SHA
    catalog = (prior / 'results/stages/ncu-gv100-catalog/stdout.log').read_text()
    advertised = [name for name in D.METRICS if name in catalog]
    assert len(advertised) == len(D.METRICS) == len(D.BASE_METRICS) == 9
    del catalog
    commands = {}
    for label, module in (('v1', V), ('v2', D)):
        run = base / label; run.mkdir()
        args = types.SimpleNamespace(run_directory=str(run), python=sys.executable,
            probe='/tmp/mxfp4_profile_probe.py', workspace_run=completion['workspace_run'],
            prior_cache_run=completion['cache_run'], prior_diagnostics_run=str(prior),
            cache_build='off', framework_run=completion['workspace_run'])
        runner = module.create_runner()(args, run)
        helper = runner.results / 'stages/ncu-help'; helper.mkdir(parents=True)
        shutil.copyfile(actual_help, helper / 'stdout.log'); (helper / 'stderr.log').write_text('')
        runner.ncu_help([])
        captured = {}
        def capture(name, argv, cap, gate=None, gpu=False, library=None):
            captured[name] = [str(value) for value in argv]
            if name == 'ncu-gv100-catalog': runner.report['selected_profile_metrics'] = advertised
            return True
        runner.stage = capture; runner.integrity = lambda: None
        runner.await_workspace = lambda: True; runner.reuse_prior_memcheck = lambda: True
        runner.tool = lambda path: True
        runner.execute()
        commands[label] = {name: [value.replace(str(runner.results), '<results>') for value in argv]
                           for name, argv in captured.items()}
    assert commands['v1']['ncu-profile'] == commands['v2']['ncu-profile']
    assert set(commands['v2']) == {'ncu-version', 'ncu-help', 'ncu-gv100-catalog', 'ncu-profile'}
    query = commands['v2']['ncu-gv100-catalog']
    assert query[query.index('--query-metrics-mode') + 1] == 'suffix'
    assert query[query.index('--metrics') + 1] == ','.join(D.BASE_METRICS)
    assert query[query.index('--config-file') + 1] == 'off'
    report = {'status': 'bounded_host_checks_passed', 'gpu_executed': False,
        'runner_bytes': PATHS['D'].stat().st_size, 'runner_sha256': sha(PATHS['D']),
        'probe_sha256': D.PROBE_SHA, 'preserved_v1_sha256': sha(PATHS['V']),
        'actual_prior_archive_evidence': evidence, 'actual_help_sha256': D.ACTUAL_HELP_SHA,
        'actual_all_nine_candidate_metrics_advertised': advertised,
        'actual_memcheck_summary': memcheck, 'actual_memcheck_memory_errors': 0,
        'source_and_binary_bindings_match_actual_prior_workspace_cache': True,
        'profile_command_identical_to_v1': True, 'catalog_command': query,
        'new_GPU_memcheck_command_absent': True, 'new_stage_plan': list(D.PLAN),
        'node_physical_binary_tool_checks_remain_runtime_requirements': True}
Path('/tmp/mxfp4_diagnostics_v2_host_checks.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
