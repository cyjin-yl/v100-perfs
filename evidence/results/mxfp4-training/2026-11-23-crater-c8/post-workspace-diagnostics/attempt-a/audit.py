#!/usr/bin/env python3
"""Audit an actual downloaded post-workspace diagnostics archive, without CUDA.

Reads immutable local archives only. No extraction, archived-code execution,
GPU, remote access, synthetic samples, or modification of raw results.
Outputs are created exclusively and must not already exist.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import tarfile

RUNNER_SHA = 'ab8802a3c077106f01ae63eb94b44f85ec006aa49b847bdcf94faa94b43915f1'
PROBE_SHA = '8b7f0a9f2b5dbeb6b1f0ceaa0297ab4b0ef10e474a550f5d97f6925a8b33502c'
H_SHA = 'f7f45628bcf7616a7e7915823c77f2f75602cff66be29bf640c5ec3be8ecce1e'
Q_SHA = '96d3db40d25426370cfc8542dff908edf00a385c0055019c727c9c005fb8a548'
W_SHA = '4018f7b5b08d3493ee746458cd98dea58925e1bc233b4e456b4adfea2fc9ee03'
C_SHA = '65e23628156a4430421cd6ad21359a62e4f2ab922f9daf58ee7d1621b5ecdde7'
SOURCE_COMMIT = 'f74de5be95687d75758c1f636ffbd00d6b63d161'
SOURCE_SHA = 'aa7ac287d389d1e8705fdeef60713bae1927116c85bb7355a52a817e4a806a89'
CACHE_COMMIT = 'c581389551be2233ba23404fa72c471cc4c6f961'
CACHE_SOURCE_SHA = 'a7062abcfa1b9263d7a7bf0d282f2de1b51cfa279939f3b969eacb543f5417c0'
WORKSPACE_RESULTS_SHA = 'ac5998366a536800bac705eb03c185e01b089b0e7b24b29d84aa8e5aa2b65c30'
CACHE_RESULTS_SHA = '9bea8409d639e984fa7834a16681312f23b14158b4c6f568796165f7bda0c797'
WORKSPACE_AUDIT_SHA = 'c3a72c02889e5482412e09e07ef1a9af949253e9533efcde4aa4eec94d2637d3'
CACHE_AUDIT_SHA = 'b034f7662c0a17781213b7c130aec5ccd3fd8c336561f956a653744d6867fe6c'
TRAINING = 'tools/mxfp4_training/'
MAPPING = 'evidence/benchmarks/sm70-packing/study/simulation/round3/cuda/fragment_mapping.hpp'
NCU = '/usr/local/cuda/bin/ncu'
SANITIZER = '/usr/local/cuda/bin/compute-sanitizer'
PLAN = ('wait-workspace-lock', 'ncu-version', 'ncu-help', 'ncu-gv100-catalog',
        'ncu-profile', 'compute-sanitizer-version', 'memcheck-prepare-cache')
METRICS = (
    'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active',
    'sm__warps_active.avg.pct_of_peak_sustained_active',
    'smsp__issue_active.avg.pct_of_peak_sustained_active',
    'smsp__warps_eligible.avg.per_cycle_active',
    'dram__throughput.avg.pct_of_peak_sustained_elapsed',
    'dram__bytes_read.sum.per_second', 'dram__bytes_write.sum.per_second',
    'sm__cycles_elapsed.avg.per_second', 'gpu__time_duration.sum',
)
REQUIRED = {METRICS[0], METRICS[1], METRICS[4]}
DENIED = ('ERR_NVGPUCTRPERM', 'CUPTI_ERROR_INSUFFICIENT_PRIVILEGES',
          'does not have permission to access NVIDIA GPU Performance Counters')


class AuditError(ValueError):
    pass


def need(value, message):
    if not value:
        raise AuditError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pairs(values):
    result = {}
    for key, value in values:
        need(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def load(data):
    def invalid(value):
        raise AuditError('Nonfinite JSON number: ' + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def fields(data, expected, label):
    need(isinstance(data, dict), label + ' is not an object')
    for key, value in expected.items():
        need(data.get(key) == value and (not isinstance(value, bool) or data[key] is value), label + ': ' + key)


def safe(name):
    path = PurePosixPath(name)
    need(name and '\\' not in name and not path.is_absolute() and '..' not in path.parts
         and '.' not in name.split('/') and str(path) == name.rstrip('/'), 'Unsafe tar name: ' + name)
    return str(path)


class Bundle:
    def __init__(self, path, expected, strip_results=False):
        path = Path(path)
        need(path.is_file() and not path.is_symlink(), 'Archive must be a regular nonsymlink file')
        need(path.stat().st_size <= 1024**3, 'Compressed archive exceeds bound')
        raw = path.read_bytes()
        self.meta = {'path': str(path), 'sha256': sha(raw), 'bytes': len(raw)}
        need(self.meta['sha256'] == expected, 'Archive SHA mismatch: ' + str(path))
        self.files, self.hashes = {}, {}
        seen, expanded = set(), 0
        with tarfile.open(fileobj=io.BytesIO(raw), mode='r:*') as tar:
            members = tar.getmembers()
            need(len(members) < 10000, 'Too many tar members')
            for item in members:
                name = safe(item.name)
                need(name not in seen, 'Duplicate tar member'); seen.add(name)
                need(item.isdir() or item.isreg(), 'Tar link/special member forbidden')
                need(not getattr(item, 'sparse', None), 'Sparse member forbidden')
                if strip_results:
                    need(name == 'results' or name.startswith('results/'), 'Unexpected results root')
                    name = '' if name == 'results' else name[len('results/'):]
                if item.isdir():
                    continue
                expanded += item.size
                need(0 <= item.size <= 1024**3 and expanded <= 1024**3, 'Expanded archive exceeds bound')
                value = tar.extractfile(item).read()
                need(len(value) == item.size, 'Short tar member')
                self.files[name], self.hashes[name] = value, sha(value)

    def doc(self, name):
        need(name in self.files, 'Missing JSON artifact: ' + name)
        return load(self.files[name])

    def text(self, name):
        need(name in self.files, 'Missing text artifact: ' + name)
        return self.files[name].decode('utf-8', errors='replace')

    def rows(self, name):
        return [load(line) for line in self.text(name).splitlines() if line.startswith('{')]


def one(rows, kind):
    matches = [row for row in rows if row.get('type') == kind]
    need(len(matches) == 1, 'Expected one record: ' + kind)
    return matches[0]


def result_bundle(directory, expected, allowed):
    bundle = Bundle(directory / 'results.tar.gz', expected, True)
    receipt_bytes = (directory / 'archive-manifest.json').read_bytes()
    receipt, complete = load(receipt_bytes), bundle.doc('completion.json')
    fields(receipt, {'sha256': expected, 'bytes': bundle.meta['bytes'],
                     'completion_sha256': bundle.hashes['completion.json']}, 'archive receipt')
    need(complete['result_files_sha256'] == {k: v for k, v in bundle.hashes.items() if k != 'completion.json'},
         'Archived member SHA map differs')
    need(complete['status'] in allowed, 'Archive is not an allowed terminal state')
    fields(receipt, {'status': complete['status'], 'passed': complete['status'] == 'passed'}, 'receipt terminal status')
    fields(complete, {'passed': complete['status'] == 'passed'}, 'completion terminal status')
    bundle.meta.update(completion_sha256=bundle.hashes['completion.json'], receipt_sha256=sha(receipt_bytes),
                       verified_hashed_result_files=len(complete['result_files_sha256']))
    return bundle, complete, receipt


def source_bundle(directory, expected, commit, count, size):
    source = Bundle(directory / 'source.tar.gz', expected)
    binding = load((directory / 'source-binding.json').read_bytes())
    fields(binding, {'source_commit': commit, 'source_sha256': expected, 'source_bytes': size}, 'source binding')
    need(source.meta['bytes'] == size and len(source.files) == len(binding['members']) == count, 'Source count/size differs')
    need(source.hashes == {m['name']: m['sha256'] for m in binding['members']}, 'Source member SHA map differs')
    for member in binding['members']:
        need(len(source.files[member['name']]) == member['bytes'], 'Source member size differs')
    return source, binding


def metric_value(raw):
    text = raw.strip()
    # Preserve unavailable/non-numeric strings verbatim; never coerce them to 0.
    unavailable = {'', 'n/a', 'na', 'nan', '-nan', '+nan', 'not available', 'not supported', '--', 'inf', '-inf', '+inf'}
    if text.lower() in unavailable:
        return {'raw_value': raw, 'numeric_value': None, 'availability': 'not_available'}
    number = r'[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
    if not re.fullmatch(number, text):
        return {'raw_value': raw, 'numeric_value': None, 'availability': 'non_numeric'}
    value = float(text.replace(',', ''))
    if not math.isfinite(value):
        return {'raw_value': raw, 'numeric_value': None, 'availability': 'not_available'}
    return {'raw_value': raw, 'numeric_value': value, 'availability': 'numeric'}


def parse_profile_csv(raw, selected):
    lines = raw.splitlines()
    start = next((i for i, line in enumerate(lines) if '"ID"' in line and '"Kernel Name"' in line
                  and '"Metric Name"' in line), None)
    if start is None:
        return {'metric_header_present': False, 'kernels': [], 'metric_rows': [], 'selected_metric_rows': [],
                'numeric_counter_rows': 0, 'unavailable_counter_rows': 0, 'all_selected_counters_numeric': False}
    parsed = list(csv.DictReader(io.StringIO('\n'.join(lines[start:]))))
    raw_rows = [row for row in parsed if row.get('ID', '').isdigit() and row.get('Metric Name')]
    kernels, rows, identities = {}, [], {}
    for row in raw_rows:
        ident, name, metric = row['ID'], row['Kernel Name'], row['Metric Name']
        need(ident not in kernels or kernels[ident] == name, 'One profiler launch ID has conflicting kernel names')
        kernels[ident] = name
        key = (ident, metric)
        if key in identities:
            need(identities[key] == (row.get('Metric Unit'), row.get('Metric Value')), 'Duplicate metric rows disagree')
        identities[key] = (row.get('Metric Unit'), row.get('Metric Value'))
        value = metric_value(row.get('Metric Value', ''))
        rows.append({'launch_id': ident, 'kernel_name': name, 'metric_name': metric,
                     'metric_unit': row.get('Metric Unit'), 'requested_counter': metric in selected,
                     **value, 'raw_fields': row})
    counters = [row for row in rows if row['requested_counter']]
    return {'metric_header_present': True, 'kernels': [{'launch_id': key, 'name': value} for key, value in kernels.items()],
            'metric_rows': rows, 'selected_metric_rows': counters,
            'numeric_counter_rows': sum(r['availability'] == 'numeric' for r in counters),
            'unavailable_counter_rows': sum(r['availability'] != 'numeric' for r in counters),
            'all_selected_counters_numeric': bool(counters) and all(r['availability'] == 'numeric' for r in counters)}


def certificate(data, calls):
    fields(data, {'abi_version': 1, 'device': 0, 'm': 4096, 'n': 4096, 'padded_k': 4096,
                  'metadata_valid': 1, 'submitted_calls': calls, 'accepted': 1, 'reject_flags': 0}, 'actual profile certificate')
    need(type(data['workspace_bytes']) is int and data['workspace_bytes'] > 0, 'Invalid workspace bytes')
    u, v = data['max_energy_a'], data['max_energy_b']
    need(type(u) is int and type(v) is int and 0 <= u <= 2**48 + 1 and 0 <= v <= 2**48 + 1, 'Invalid capped energies')
    need(max(data['max_span_a'], data['max_span_b']) <= 12 and (u == 0 or v == 0 or u <= 2**48 // v),
         'Accepted profile certificate violates span/energy gate')
    need(data['nan_scales_a'] == data['nan_scales_b'] == 0 and
         data['min_scale_a'] + data['min_scale_b'] - 256 >= -126 and
         data['max_scale_a'] + data['max_scale_b'] - 256 <= 103, 'Accepted profile certificate violates scale gate')


def api_proof(data, name):
    fields(data, {'runtime_symbol': name, 'checked_runtime_call_AST_verified': True}, 'profiler API binding')
    source = data['actual_function_source']
    need(sha(source.encode()) == data['function_source_sha256'], 'Profiler API source SHA differs')
    tree = ast.parse(source)
    need(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef), 'Unexpected profiler API source')
    body = tree.body[0].body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    expected = ast.parse('check_error(cudart().' + name + '())').body
    need(ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False) ==
         ast.dump(ast.Module(body=expected, type_ignores=[]), include_attributes=False), 'Profiler API AST is not checked CUDA runtime call')


def check_micro(rows, prior_rows):
    env = one(rows, 'prepare_cache_environment'); summary = one(rows, 'prepare_cache_summary')
    fields(env, {'actual_V100': True, 'certified_variant': 2, 'certified_prepare_raw_cache': True,
                 'explicit_allocation_cap_bytes': 8*1024**2, 'reserve_bytes': 512*1024**2,
                 'minimum_free_bytes': 1024**3, 'shared_dispatch_helper_checks': 7,
                 'performance_timing_executed': False}, 'memchecked environment')
    cases = [r for r in rows if r.get('type') == 'prepare_cache_correctness']
    old_cases = [r for r in prior_rows if r.get('type') == 'prepare_cache_correctness']
    need(len(cases) == len({r.get('name') for r in cases}) == 327, 'Memchecked case count/name differs')
    need(cases == old_cases, 'Memchecked actual case records differ from already verified exact source fixture results')
    need(summary == one(prior_rows, 'prepare_cache_summary'), 'Memchecked summary differs from verified native 327-case summary')
    fields(summary, {'passed': True, 'status': 'passed', 'gpu_executed': True, 'all_submitted_work_completed': True,
                     'cases': 327, 'direct_cached_vs_original_cases': 323, 'public_dispatch_cases': 4,
                     'normalized_half_values': 1079840, 'rowdata_rows': 1635, 'public_FP32_output_values': 24,
                     'normalized_half_bit_mismatches': 0, 'rowdata_byte_mismatches': 0,
                     'independent_cpu_oracle_mismatches': 0, 'public_output_bit_mismatches': 0,
                     'guard_bytes_unchanged': True, 'input_bytes_unchanged': True}, 'actual memchecked summary')
    need(sum(r['normalized_half_values'] for r in cases) == 1079840 and sum(r['rowdata_rows'] for r in cases) == 1635,
         'Memchecked per-case totals differ')
    for row in cases:
        fields(row, {'passed': True, 'gpu_executed': True, 'M': 3, 'N': 2, 'rowdata_rows': 5,
                     'rowdata_bytes_per_row': 16, 'normalized_half_bit_mismatches': 0, 'rowdata_byte_mismatches': 0,
                     'independent_cpu_oracle_mismatches': 0, 'input_bytes_unchanged': True,
                     'guard_bytes_unchanged': True, 'actual_original_kernel_compared': True,
                     'poison': 'bitwise_inverse_of_every_CPU_oracle_output_byte'}, 'memchecked case')
    return {'summary': summary, 'environment': env, 'all_327_cases': cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--expected-results-sha256', required=True)
    parser.add_argument('--workspace-directory', type=Path)
    parser.add_argument('--cache-directory', type=Path)
    parser.add_argument('--output-prefix', required=True, type=Path)
    args = parser.parse_args()
    args.directory = args.directory.resolve(strict=True)
    wsdir = (args.workspace_directory or args.directory.parent / 'candidate-workspace').resolve(strict=True)
    cadir = (args.cache_directory or args.directory.parent / 'candidate-c2-prepare-cache').resolve(strict=True)
    outputs = [Path(str(args.output_prefix) + ext) for ext in ('.json', '.md')]
    need(not any(p.exists() for p in outputs), 'Never overwrite an existing audit')
    report = {'schema_version': 1, 'purpose': 'independent actual post-workspace diagnostics audit',
              'audit_started_at_utc': datetime.now(timezone.utc).isoformat(), 'auditor_sha256': sha(Path(__file__).read_bytes()),
              'audit_passed': False, 'audit_status': 'pending', 'gpu_executed_by_auditor': False,
              'remote_executed_by_auditor': False, 'archived_code_executed': False,
              'all_diagnostics_passed': False, 'profiling': {'status': 'unavailable', 'complete': False},
              'memcheck': {'status': 'unavailable', 'complete': False}}
    code = 1
    try:
        ws, wc, wr = result_bundle(wsdir, WORKSPACE_RESULTS_SHA, {'passed'})
        ca, cc, cr = result_bundle(cadir, CACHE_RESULTS_SHA, {'passed'})
        src, binding = source_bundle(wsdir, SOURCE_SHA, SOURCE_COMMIT, 33, 157598)
        oldsrc, _ = source_bundle(cadir, CACHE_SOURCE_SHA, CACHE_COMMIT, 30, 146431)
        for directory, filename, expected, archived in ((wsdir, 'audit.json', WORKSPACE_AUDIT_SHA, ws),
                                                        (cadir, 'independent-audit.json', CACHE_AUDIT_SHA, ca)):
            raw = (directory / filename).read_bytes(); audit = load(raw)
            need(sha(raw) == expected, 'Independent prerequisite audit SHA differs')
            fields(audit, {'audit_passed': True, 'audit_status': 'verified'}, 'independent prerequisite audit')
            # Current prior reports use distinct top-level provenance field names.
            need(archived.meta['sha256'] in raw.decode(), 'Prerequisite audit does not bind actual archive')
        fields(wc, {'status': 'passed', 'passed': True, 'source_commit': SOURCE_COMMIT,
                    'source_archive_sha256': SOURCE_SHA, 'runner_sha256': W_SHA, 'cache_build': 'off',
                    'source_hashes_before': src.hashes, 'source_hashes_after': src.hashes,
                    'source_and_binaries_unchanged': True}, 'verified workspace prerequisite')
        fields(cc, {'status': 'passed', 'passed': True, 'source_commit': CACHE_COMMIT,
                    'source_archive_sha256': CACHE_SOURCE_SHA, 'runner_sha256': C_SHA,
                    'source_hashes_before': oldsrc.hashes, 'source_hashes_after': oldsrc.hashes,
                    'source_and_binaries_unchanged': True}, 'verified cache prerequisite')
        need(len(wc['stages']) == 10 and all(s['status'] == 'passed' and s['passed'] is True for s in wc['stages']), 'Workspace plan incomplete')
        need(len(cc['stages']) == 26 and all(s['status'] == 'passed' and s['passed'] is True for s in cc['stages']), 'Cache plan incomplete')
        for name, digest in oldsrc.hashes.items():
            need(src.hashes.get(name) == digest, 'Workspace source changed an existing cache source')
        d, complete, receipt = result_bundle(args.directory, args.expected_results_sha256,
                                             {'passed', 'failed', 'completed_with_blocked_diagnostics'})
        fields(complete, {'runner_sha256': RUNNER_SHA, 'probe_sha256': PROBE_SHA,
                          'native_helper_sha256': H_SHA, 'quantizer_helper_sha256': Q_SHA,
                          'workspace_helper_sha256': W_SHA, 'cache_helper_sha256': C_SHA,
                          'source_commit': SOURCE_COMMIT, 'source_archive_sha256': SOURCE_SHA,
                          'cache_build': 'off', 'quantization_in_profile_scope': False,
                          'profile_is_performance_benchmark': False, 'compiled_in_this_run': False,
                          'automatic_retries': False, 'new_tuning_candidates': False,
                          'clock_or_power_changes': False, 'privilege_escalation': False,
                          'packages_installed': False, 'services_stopped': False,
                          'full_training_executed': False, 'prior_performance_evidence_modified': False,
                          'planned_stages': list(PLAN)}, 'diagnostic frozen binding/scope')
        fields(receipt, {'runner_sha256': RUNNER_SHA, 'probe_sha256': PROBE_SHA,
                         'source_commit': SOURCE_COMMIT, 'source_archive_sha256': SOURCE_SHA,
                         'prior_performance_evidence_modified': False}, 'diagnostic receipt binding')
        need(d.hashes.get('diagnostics-runner.py') == RUNNER_SHA and d.hashes.get('profile-probe.py') == PROBE_SHA,
             'Archived frozen diagnostic/probe source differs')
        need([s['name'] for s in complete['stages']] == list(PLAN), 'Diagnostic seven-stage order differs')
        stages = {s['name']: s for s in complete['stages']}
        for name, stage in stages.items():
            need(stage['status'] in ('passed', 'blocked', 'failed'), 'Nonterminal stage')
            fields(stage, {'passed': stage['status'] == 'passed'}, 'stage status')
            if stage.get('executed'):
                need(d.doc('stages/' + name + '/stage.json') == stage, 'Stage snapshot differs: ' + name)
                for sub, digest in stage.get('evidence_sha256', {}).items():
                    need(d.hashes.get('stages/' + name + '/' + safe(sub)) == digest, 'Stage evidence SHA differs')
                if stage['passed']:
                    fields(stage, {'returncode': 0, 'timed_out': False}, 'passed process')
        report.update(result_archive=d.meta, prior_workspace_archive=ws.meta, prior_cache_archive=ca.meta,
                      prior_workspace_audit_sha256=WORKSPACE_AUDIT_SHA, prior_cache_audit_sha256=CACHE_AUDIT_SHA,
                      source_archive=src.meta, source_commit=SOURCE_COMMIT,
                      raw_status=complete['status'], stage_counts=dict(Counter(s['status'] for s in stages.values())),
                      raw_stages=complete['stages'])

        run, wsrun, carun = complete['run_directory'], wc['run_directory'], cc['run_directory']
        source_root, manifest_path = wsrun + '/source', carun + '/build-off/compile-evidence.json'
        library_path, micro_path = carun + '/build-off/libmxfp4_sm70.so', carun + '/build-micro/check_mxfp4_prepare_cache'
        fields(complete, {'workspace_run': wsrun, 'cache_run': carun, 'source_root_used_read_only': source_root}, 'read-only prior paths')
        wait = stages['wait-workspace-lock']
        if wait['passed']:
            fields(wait, {'executed': False, 'gpu_executed': False, 'same_uid_lock_acquired': True}, 'same B flock')
            need(wait['same_B_lock_path'] == next(s for s in wc['stages'] if s['name'] == 'wait-cache-lock')['same_B_lock_path'], 'GPU serialization lock differs')
            fields(complete, {'source_hashes_before': src.hashes, 'source_hashes_after': src.hashes,
                              'source_and_binaries_unchanged': True}, 'read-only source before/after')
            need(complete['binary_hashes_before_gpu'] == complete['binary_hashes_after_gpu'], 'Native binaries changed')
            expected_binaries = dict(wc['binary_hashes_after_gpu'])
            expected_binaries[micro_path] = cc['binary_hashes_after_gpu'][micro_path]
            need(complete['binary_hashes_after_gpu'] == expected_binaries, 'Reused binaries differ from prior verified artifacts')
            fields(wait, {'selected_library_sha256': expected_binaries[library_path],
                          'actual_microchecker_sha256': expected_binaries[micro_path]}, 'selected native identities')
            for prefix, prior, prior_receipt in (('prior-workspace/', ws, wr), ('prior-cache/', ca, cr)):
                need(d.files[prefix + 'completion.json'] == prior.files['completion.json'], 'Copied prior completion differs')
                need(d.doc(prefix + 'archive-manifest.json') == prior_receipt, 'Copied prior receipt differs')
                evidence = complete['prior_workspace_evidence' if prefix.startswith('prior-workspace') else 'prior_cache_evidence']
                fields(evidence, {'archive_sha256': prior.meta['sha256'], 'archive_bytes': prior.meta['bytes'],
                                  'completion_sha256': prior.meta['completion_sha256'],
                                  'receipt_sha256': prior.meta['receipt_sha256'], 'status': 'passed'}, 'prior archive evidence')
            need(d.doc('prior-workspace/source-binding.json') == binding and
                 d.files['prior-workspace/workspace-check.json'] == ws.files['workspace-check.json'], 'Copied workspace source/check differs')
            for name in d.files:
                if name.startswith('prior-cache/build-'):
                    prior_name = 'build-evidence/' + name[len('prior-cache/'):]
                    need(d.files[name] == ca.files[prior_name], 'Copied compile evidence/log differs')
            for name in ('quantize-cuda-off.json', 'quantize-cuda-on.json', 'prepare-cache-experiment-runner.py'):
                need(d.files['prior-cache/' + name] == ca.files[name], 'Copied native numerical/runner evidence differs')
        else:
            need(all(not s.get('executed') and s['status'] != 'passed' for n, s in stages.items() if n != 'wait-workspace-lock'),
                 'Diagnostics executed without completed locked prerequisite')

        help_text = '\n'.join(d.text('stages/ncu-help/' + part) for part in ('stdout.log', 'stderr.log')) if stages['ncu-help'].get('executed') else ''
        rename = '--rename-kernels' in help_text
        selected = []
        if stages['ncu-version']['passed']:
            text = '\n'.join(d.text('stages/ncu-version/' + part) for part in ('stdout.log', 'stderr.log'))
            need(re.search(r'\b2025\.1\.1(?:\.0)?\b', text), 'Actual profiler version is not pinned Volta-compatible NCU')
        if stages['ncu-help']['passed']:
            for option in ('profile-from-start', 'clock-control', 'cache-control', 'replay-mode', 'launch-count',
                           'query-metrics-mode', 'chips', 'target-processes', 'log-file', 'export', 'csv', 'page', 'config-file'):
                need('--' + option in help_text, 'Actual installed NCU help missing option: ' + option)
            need(complete['ncu_rename_kernels_off_supported'] is rename, 'Rename-kernels support differs from actual help')
        catalog_text = ('\n'.join(d.text('stages/ncu-gv100-catalog/' + part) for part in ('stdout.log', 'stderr.log'))
                        if stages['ncu-gv100-catalog'].get('executed') else '')
        catalog_over_bound = len(catalog_text) > 32*1024**2
        if stages['ncu-gv100-catalog'].get('error') == 'Metric catalog exceeds diagnostic bound':
            need(catalog_over_bound and stages['ncu-gv100-catalog']['returncode'] == 0
                 and stages['ncu-gv100-catalog']['status'] == 'failed', 'Catalog bound failure is not supported by raw output')
        if stages['ncu-gv100-catalog']['passed']:
            catalog = catalog_text
            available = set(re.findall(r'\b[a-z][a-z0-9]*__[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+', catalog))
            selected = [name for name in METRICS if name in available]
            need(REQUIRED.issubset(selected), 'Actual GV100 catalog lacks required counters')
            need(complete['selected_profile_metrics'] == selected and
                 complete['requested_metrics_not_in_actual_catalog'] == [name for name in METRICS if name not in available],
                 'Metric selection differs from actual installed GV100 catalog')
            need(stages['ncu-gv100-catalog']['validation']['catalog_sha256'] == d.hashes['stages/ncu-gv100-catalog/stdout.log'],
                 'Actual catalog SHA differs')
        commands = {'ncu-version': [NCU, '--version'], 'ncu-help': [NCU, '--help'],
                    'ncu-gv100-catalog': [NCU, '--config-file', 'off', '--query-metrics', '--chips', 'gv100', '--query-metrics-mode', 'all'],
                    'compute-sanitizer-version': [SANITIZER, '--version'],
                    'memcheck-prepare-cache': [SANITIZER, '--tool', 'memcheck', '--error-exitcode', '99', '--leak-check', 'full',
                                              '--target-processes', 'application-only', '--log-file', run + '/results/memcheck.log',
                                              micro_path, '--device', '0', '--correctness-only']}
        profile_stage = stages['ncu-profile']
        if profile_stage.get('executed'):
            actual = profile_stage['argv']
            need(actual.count('-B') == 1, 'Profile probe Python argument missing/duplicated')
            python = actual[actual.index('-B') - 1]
            need(python == wc['python_execution_path'], 'Profile Python differs from validated workspace Python')
            probe_path = actual[actual.index('-B') + 1]
            need(Path(probe_path).is_absolute(), 'Probe execution path is not absolute')
            commands['ncu-profile'] = [NCU, '--config-file', 'off'] + (['--rename-kernels', 'off'] if rename else []) + [
                '--devices', '0', '--target-processes', 'application-only', '--profile-from-start', 'off',
                '--clock-control', 'none', '--cache-control', 'none', '--replay-mode', 'kernel', '--launch-count', '10',
                '--kernel-name-base', 'demangled', '--section', 'LaunchStats', '--metrics', ','.join(selected),
                '--csv', '--page', 'raw', '--log-file', run + '/results/profile.csv', '--export', run + '/results/profile',
                python, '-B', probe_path, '--source-root', source_root, '--compile-evidence', manifest_path,
                '--output', run + '/results/profile-probe.json']
            fields(profile_stage, {'library_sha256_before': cc['binary_hashes_after_gpu'][library_path],
                                   'library_sha256_after': cc['binary_hashes_after_gpu'][library_path]}, 'profile library before/after')
        for name, command in commands.items():
            stage = stages[name]
            if stage.get('executed'):
                need(stage['argv'] == command and stage['cwd'] == source_root, 'Actual command/CWD differs: ' + name)
                cap = 300 if name in ('ncu-profile', 'memcheck-prepare-cache') else 60 if name == 'ncu-gv100-catalog' else 30
                need(stage['maximum_stage_seconds'] == cap and 0 < stage['effective_timeout_seconds'] <= cap,
                     'Stage budget differs')
        observed_requested = [name for name in METRICS if re.search(
            r'(?<![A-Za-z0-9_.])' + re.escape(name) + r'(?![A-Za-z0-9_.])', catalog_text)]
        report['ncu_catalog'] = {'passed': stages['ncu-gv100-catalog']['passed'], 'selected_metrics': selected,
                                 'controller_metric_selection_completed': stages['ncu-gv100-catalog']['passed'],
                                 'independently_observed_requested_metrics': observed_requested,
                                 'missing_requested_metrics': ([name for name in METRICS if name not in observed_requested]
                                                               if catalog_text else None),
                                 'query_returncode': stages['ncu-gv100-catalog'].get('returncode'),
                                 'raw_combined_text_characters': len(catalog_text),
                                 'controller_character_bound': 32*1024**2,
                                 'raw_catalog_exceeds_controller_bound': catalog_over_bound,
                                 'config_file_off_in_actual_commands': True, 'rename_kernels_off_supported': rename}
        raw_csv = d.text('profile.csv') if 'profile.csv' in d.files else ''
        parsed = parse_profile_csv(raw_csv, selected)
        denied_text = raw_csv + '\n' + '\n'.join(d.text('stages/ncu-profile/' + part) for part in ('stdout.log', 'stderr.log')
                                                   if 'stages/ncu-profile/' + part in d.files)
        denial_tokens = [token for token in DENIED if token in denied_text]
        report['profiling'] = {'status': profile_stage['status'], 'complete': False,
                               'counter_permission_denied': bool(denial_tokens), 'observed_denial_tokens': denial_tokens,
                               'raw_csv_sha256': d.hashes.get('profile.csv'), 'raw_profile_report_sha256': d.hashes.get('profile.ncu-rep'),
                               'diagnostic_only': True, 'unprofiled_performance_claim': False,
                               **parsed, 'probe': d.doc('profile-probe.json') if 'profile-probe.json' in d.files else None}
        if denial_tokens:
            fields(profile_stage, {'status': 'blocked', 'passed': False, 'permission_blocked': True,
                                   'no_escalation_or_counter_retry': True}, 'honest counter permission block')
            need(complete['status'] != 'passed', 'Denied profiler was incorrectly declared complete')
        if profile_stage['passed']:
            need(wait['passed'] and stages['ncu-gv100-catalog']['passed'] and not denial_tokens, 'Profile lacks prerequisite')
            p = d.doc('profile-probe.json')
            fields(p, {'status': 'passed', 'passed': True, 'gpu_executed': True, 'all_attempted_work_synchronized': True,
                       'source_and_library_unchanged': True, 'probe_sha256': PROBE_SHA, 'source_root': source_root,
                       'size': 4096, 'seed': 20261123, 'warmup': 10, 'profiled_workspace_submissions': 1,
                       'profile_range_entered': True, 'profile_range_stopped': True, 'quantization_in_profile_range': False,
                       'context_creation_in_profile_range': False, 'correctness_comparison_or_D2H_in_profile_range': False,
                       'last_stream_sync_in_profile_range': True, 'diagnostic_only': True,
                       'unprofiled_performance_claim': False, 'full_training_executed': False, 'workspace_closed': True}, 'actual profiling probe')
            names = ['check_torch_certified.py', 'check_torch_cuda.py', 'reference.py', 'torch_ops.py',
                     'certified_workspace.py', 'check_certified_workspace.py']
            expected_sources = {name: src.hashes[TRAINING + name] for name in names}
            expected_sources.update({key[len(TRAINING):]: value for key, value in src.hashes.items() if key.startswith(TRAINING + 'csrc/')})
            expected_sources['fragment_mapping.hpp'] = src.hashes[MAPPING]
            need(p['source_sha256'] == expected_sources, 'Profile probe source map differs')
            manifest = ca.doc('build-evidence/build-off/compile-evidence.json')
            fields(p['library'], {'path': library_path, **manifest['artifacts']['libmxfp4_sm70.so'],
                                   'compiled_variant': 2, 'compile_evidence': manifest_path,
                                   'compile_evidence_sha256': ca.hashes['build-evidence/build-off/compile-evidence.json'],
                                   'native_sources_match': True, 'fragment_mapping_matches': True}, 'actual profile library')
            need(p['gpu']['compute_capability'] == [7, 0] and 'V100' in p['gpu']['name'], 'Profile target not V100')
            for key in ('initial_all_outputs', 'actual_profiled_all_outputs'):
                fields(p[key], {'values': 4096**2, 'bit_mismatches': 0, 'all_finite': True,
                                 'canonical_positive_zero': True}, 'actual full-output bit comparison')
            for key, calls in (('initial_certificate', 1), ('before_profile_certificate', 11), ('actual_profiled_certificate', 12)):
                certificate(p[key], calls)
            certs = [{key: value for key, value in p[name].items() if key != 'submitted_calls'}
                     for name in ('initial_certificate', 'before_profile_certificate', 'actual_profiled_certificate')]
            need(certs[0] == certs[1] == certs[2], 'Profile certificate changed for identical packed inputs')
            fingerprints = p['input_sha256_before']
            need(fingerprints == p['input_sha256_after'] and set(fingerprints) == {'a_fp32','b_fp32','a_codes','a_scales','b_codes','b_scales'},
                 'Input fingerprint set changed/incomplete')
            need(all(re.fullmatch('[0-9a-f]{64}', value) for value in fingerprints.values()), 'Malformed input hash')
            target = wc['input_fingerprints']['4096']
            need(fingerprints['a_fp32'] == target['a_fp32_bytes'] and fingerprints['b_fp32'] == target['b_fp32_bytes'],
                 'Profile inputs differ from actual prior 4096 benchmark')
            for key, name in (('start', 'cudaProfilerStart'), ('stop', 'cudaProfilerStop')):
                api_proof(p['profiler_API'][key], name)
            kernels = parsed['kernels']; kernel_names = [k['name'] for k in kernels]
            need(parsed['metric_header_present'] and 1 <= len(kernels) < 10, 'Profile capture absent/capped')
            for needle in ('prepare_rows', 'finalize_certificate', 'postscale_rows_if_certified', 'gemm_tiled_mx32'):
                need(any(needle in name for name in kernel_names), 'Actual C2 pipeline kernel absent: ' + needle)
            need(any('gemm' in name.lower() and 'gemm_tiled_mx32' not in name for name in kernel_names), 'cuBLAS GEMM absent')
            need(not any('quantize' in name.lower() for name in kernel_names), 'Quantization leaked inside profile range')
            for kernel in kernels:
                observed = {r['metric_name'] for r in parsed['metric_rows'] if r['launch_id'] == kernel['launch_id']}
                need(set(selected).issubset(observed), 'Requested metric row missing for kernel')
            need(len(d.files.get('profile.ncu-rep', b'')) > 0, 'Binary NCU report absent/empty')
            summary = d.doc('profile-csv-summary.json')
            fields(summary, {'kernels': {r['launch_id']: r['name'] for r in kernels}, 'metric_rows': len(parsed['metric_rows']),
                             'selected_metrics': selected, 'diagnostic_only': True, 'profile_csv_sha256': d.hashes['profile.csv']}, 'CSV summary')
            fields(profile_stage['validation'], {'probe_json_sha256': d.hashes['profile-probe.json'],
                                                 'profile_sha256': d.hashes['profile.ncu-rep']}, 'profile validator hashes')
            report['profiling'].update(capture_complete=True, complete=parsed['all_selected_counters_numeric'],
                status='passed' if parsed['all_selected_counters_numeric'] else 'captured_with_unavailable_counters',
                actual_context_lifecycle='one_explicit_resident_CertifiedWorkspace; initial1+warmup10+profile1=12',
                generic_library_metadata_lifecycle=p['library'].get('context_lifecycle'),
                metadata_note='The generic library_metadata helper reports its historical per-call label; actual probe source and counters establish the explicit resident workspace execution.')

        mstage = stages['memcheck-prepare-cache']
        report['memcheck'] = {'status': mstage['status'], 'complete': False,
                              'raw_log_sha256': d.hashes.get('memcheck.log'), 'tool': 'memcheck', 'leak_check': 'full',
                              'memory_error_count': None, 'actual_cases': None,
                              'scope': 'actual cache-ON preparation microchecker, not a full training or whole-GEMM sanitizer campaign'}
        mtext = d.text('memcheck.log') if 'memcheck.log' in d.files else ''
        errors = [int(x.replace(',', '')) for x in re.findall(r'ERROR SUMMARY:\s*([\d,]+)\s+errors?\b', mtext)]
        report['memcheck']['observed_error_summaries'] = errors
        if errors:
            report['memcheck']['memory_error_count'] = max(errors)
        if mstage['passed']:
            need(wait['passed'] and stages['compute-sanitizer-version']['passed'], 'Memcheck lacks prerequisite')
            need(errors and all(value == 0 for value in errors), 'No complete zero-error sanitizer summary')
            need('========= ERROR:' not in mtext and 'FATAL' not in mtext.upper(), 'Sanitizer diagnostic failure contradicts zero errors')
            rows = d.rows('stages/memcheck-prepare-cache/stdout.log')
            checked = check_micro(rows, ca.rows('stages/check-prepare-micro/stdout.log'))
            fields(mstage['validation'], {'passed': True, 'gpu_executed': True, 'memory_errors': 0,
                                           'actual_microcheck_summary': checked['summary'],
                                           'memcheck_log_sha256': d.hashes['memcheck.log'],
                                           'binary_sha256': cc['binary_hashes_after_gpu'][micro_path],
                                           'tool': 'memcheck', 'leak_check': 'full'}, 'actual memcheck validator')
            report['memcheck'].update(complete=True, actual_cases=327, **checked)
        states = {s['status'] for s in stages.values()}
        if 'failed' in states:
            need(complete['status'] == 'failed', 'Failed stage hidden by terminal status')
        elif 'blocked' in states:
            need(complete['status'] in ('completed_with_blocked_diagnostics', 'failed'), 'Blocked diagnostic hidden by complete pass')
        else:
            need(complete['status'] in ('passed', 'failed'), 'Complete stage set disagrees with terminal state')
        fields(complete, {'diagnostic_results': {name: stages[name]['status'] for name in ('ncu-profile','memcheck-prepare-cache')}},
               'diagnostic terminal result map')
        fields(receipt, {'diagnostic_results': complete['diagnostic_results']}, 'diagnostic receipt result map')
        report.update(audit_passed=True, audit_status='verified',
                      all_diagnostics_passed=complete['status'] == 'passed' and report['profiling']['complete'] and report['memcheck']['complete'],
                      installed_tool_files_sha256=complete.get('installed_tool_files_sha256', {}),
                      limitations=['Profiler counters are diagnostic kernel-replay measurements, not unprofiled wall-time or a new throughput result.',
                                   'Unavailable, N/A and nonnumeric counter values are retained as null plus their raw text; zero is reported only when the raw metric is numeric zero.',
                                   'Counter permission denial is blocked, never a successful measurement; no retry/escalation is inferred.',
                                   'Library and installed-tool ELF bytes are not necessarily in the archive; recorded remote hashes are cross-bound to immutable prior build evidence and stage records.',
                                   'Memcheck covers the actual 327-case cache-ON preparation checker only. It does not imply racecheck, whole-training sanitization, or LLM convergence.'])
        code = 0
    except Exception as error:
        report.update(audit_status='inconsistent_or_incomplete', error_type=type(error).__name__, error=str(error),
                      all_diagnostics_passed=False)
        report['profiling']['complete'] = report['memcheck']['complete'] = False
    report['audit_finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    with outputs[0].open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    lines = ['# Independent post-workspace diagnostics audit', '', 'Audit: **' + report['audit_status'] + '**.', '']
    if report['audit_passed']:
        lines += ['Raw experiment status: `' + report['raw_status'] + '`; stage counts: `' + str(report['stage_counts']) + '`.', '',
                  '## Nsight Compute', '', 'Status: **' + report['profiling']['status'] + '**.', '',
                  'Numeric requested counter rows: ' + str(report['profiling']['numeric_counter_rows']) +
                  '; unavailable or nonnumeric requested rows: ' + str(report['profiling']['unavailable_counter_rows']) + '.', '']
        if report['profiling'].get('counter_permission_denied'):
            lines += ['GPU performance-counter permission was explicitly denied. No profiler metric is treated as a completed measurement.', '']
        if report['profiling'].get('capture_complete'):
            lines += ['| Launch ID | Captured kernel |', '|---|---|']
            lines += ['| ' + k['launch_id'] + ' | ' + k['name'].replace('|', '\\|') + ' |' for k in report['profiling']['kernels']]
            lines += ['', 'One profiled workspace submission; actual native counters are 1 initial, 11 after warmup, 12 after profiling. Both full 4096² output comparisons and input SHA checks passed. Quantization and context creation are outside the profiler range.', '',
                      report['profiling']['metadata_note'], '']
        lines += ['## Compute Sanitizer', '', 'Status: **' + report['memcheck']['status'] + '**.', '',
                  'Actual checker cases: ' + str(report['memcheck']['actual_cases']) +
                  '; reported memory errors: ' + str(report['memcheck']['memory_error_count']) + '.', '',
                  '## Limits', ''] + ['- ' + item for item in report['limitations']]
    else:
        lines += [report.get('error', 'Audit did not complete.'), '', 'No diagnostic is eligible for a completed-result claim.']
    with outputs[1].open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps({'audit_status': report['audit_status'], 'audit_passed': report['audit_passed'],
                      'all_diagnostics_passed': report['all_diagnostics_passed'], 'output': str(outputs[0])}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
