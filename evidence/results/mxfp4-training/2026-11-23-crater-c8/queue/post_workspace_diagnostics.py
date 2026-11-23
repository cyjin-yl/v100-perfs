#!/usr/bin/env python3
"""Bounded optional NCU and memcheck diagnostics after the frozen workspace run.

No compilation/install/clock/admin/service changes. Existing source and native
artifacts are read-only. All target launches use reviewed owned-group cleanup.
Blocked counter access is recorded separately; no permissions are escalated.
"""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import sys
import tarfile
import time
import types

HELPER = Path('/workspace/user/mxfp4-c9-c10-20261123T2128.py')
HELPER_SHA = 'f7f45628bcf7616a7e7915823c77f2f75602cff66be29bf640c5ec3be8ecce1e'
QHELPER = Path('/workspace/user/mxfp4-cuda-rne-v2-20261123T2204/results/quantizer-experiment-runner.py')
QSHA = '96d3db40d25426370cfc8542dff908edf00a385c0055019c727c9c005fb8a548'
WSHA = '4018f7b5b08d3493ee746458cd98dea58925e1bc233b4e456b4adfea2fc9ee03'
CSHA = '65e23628156a4430421cd6ad21359a62e4f2ab922f9daf58ee7d1621b5ecdde7'
PROBE_SHA = '8b7f0a9f2b5dbeb6b1f0ceaa0297ab4b0ef10e474a550f5d97f6925a8b33502c'
SOURCE_COMMIT = 'f74de5be95687d75758c1f636ffbd00d6b63d161'
SOURCE_SHA = 'aa7ac287d389d1e8705fdeef60713bae1927116c85bb7355a52a817e4a806a89'
NCU = Path('/usr/local/cuda/bin/ncu')
SANITIZER = Path('/usr/local/cuda/bin/compute-sanitizer')
METRICS = (
    'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active',
    'sm__warps_active.avg.pct_of_peak_sustained_active',
    'smsp__issue_active.avg.pct_of_peak_sustained_active',
    'smsp__warps_eligible.avg.per_cycle_active',
    'dram__throughput.avg.pct_of_peak_sustained_elapsed',
    'dram__bytes_read.sum.per_second', 'dram__bytes_write.sum.per_second',
    'sm__cycles_elapsed.avg.per_second', 'gpu__time_duration.sum',
)
REQUIRED_METRICS = {METRICS[0], METRICS[1], METRICS[4]}
PLAN = ('wait-workspace-lock', 'ncu-version', 'ncu-help', 'ncu-gv100-catalog',
        'ncu-profile', 'compute-sanitizer-version', 'memcheck-prepare-cache')
H = Q = W = C = None


def load(path, digest, name):
    item = path.lstat()
    if not stat.S_ISREG(item.st_mode) or item.st_uid != os.geteuid():
        raise RuntimeError('Helper is not an owned nonsymlink regular file: ' + str(path))
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest: raise RuntimeError('Helper hash differs: ' + str(path))
    module = types.ModuleType(name); module.__file__ = str(path); sys.modules[name] = module
    exec(compile(data, str(path), 'exec'), module.__dict__)
    return module


def create_runner():
    Parent = W.create_runner_class()
    class Diagnostics(Parent):
        def __init__(self, args, run):
            super().__init__(args, run)
            self.workspace = Path(args.workspace_run)
            self.whelper = self.workspace / 'results/workspace-experiment-runner.py'
            self.chelper = self.cache / 'results/prepare-cache-experiment-runner.py'
            self.source = self.workspace / 'source'
            self.env['PYTHONPATH'] = str(self.source)
            self.own_hash = H.sha(Path(__file__))
            self.probe = Path(args.probe)
            self.micro = self.cache / 'build-micro/check_mxfp4_prepare_cache'
            self.system_tools = {}
            self.report = {'schema_version': 1, 'purpose': 'optional bounded post-workspace diagnostics',
                'status': 'running', 'passed': False, 'started_at_utc': H.utc(), 'run_directory': str(run),
                'runner_sha256': self.own_hash, 'probe_sha256': PROBE_SHA,
                'native_helper_sha256': HELPER_SHA, 'quantizer_helper_sha256': QSHA,
                'workspace_helper_sha256': WSHA, 'cache_helper_sha256': CSHA,
                'source_commit': SOURCE_COMMIT, 'source_archive_sha256': SOURCE_SHA,
                'workspace_run': str(self.workspace), 'cache_run': str(self.cache),
                'source_root_used_read_only': str(self.source), 'cache_build': 'off',
                'profile_scope': 'one_resident_packed_to_FP32_C2_pipeline_inside_CUDA_profiler_start_stop',
                'quantization_in_profile_scope': False, 'profile_is_performance_benchmark': False,
                'memcheck_scope': 'actual_cache_ON_preparation327_microchecker_binary',
                'planned_stages': list(PLAN), 'stages': [], 'gpu_executed': False,
                'compiled_in_this_run': False, 'automatic_retries': False, 'new_tuning_candidates': False,
                'clock_or_power_changes': False, 'privilege_escalation': False,
                'packages_installed': False, 'services_stopped': False,
                'full_training_executed': False, 'global_budget_seconds': H.BUDGET_SECONDS,
                'absolute_work_cutoff_utc': H.ABSOLUTE_CUTOFF.isoformat(), 'cleanup_reserve_seconds': 15}

        def integrity(self):
            H.Runner.integrity(self)
            for path, digest in ((HELPER, HELPER_SHA), (QHELPER, QSHA), (self.whelper, WSHA),
                                 (self.chelper, CSHA), (self.probe, PROBE_SHA), (Path(__file__), self.own_hash)):
                H.regular(path, owned=True); H.need(H.sha(path) == digest, 'Frozen diagnostic/helper changed: ' + str(path))
            for path, digest in self.prior_files.items():
                H.regular(path, owned=True); H.need(H.sha(path) == digest, 'Read-only prior evidence changed: ' + path)
            for path, digest in self.system_tools.items():
                H.regular(path); H.need(H.sha(path) == digest, 'Installed diagnostic tool changed: ' + path)
            if self.prior_source_hashes is not None:
                H.need(H.tree_hashes(self.cache / 'source') == self.prior_source_hashes, 'Prior native cache source changed')

        def log_text(self, name):
            folder = self.results / 'stages' / name
            return '\n'.join((folder / part).read_text(errors='replace') for part in ('stdout.log', 'stderr.log')
                             if (folder / part).is_file())

        def stage(self, name, *args, **kwargs):
            ok = H.Runner.stage(self, name, *args, **kwargs)
            row = self.report['stages'][-1]
            if not ok and name == 'ncu-profile':
                message = self.log_text(name)
                csv_path = self.results / 'profile.csv'
                if csv_path.is_file(): message += '\n' + csv_path.read_text(errors='replace')
                denied = any(token in message for token in (
                    'ERR_NVGPUCTRPERM', 'CUPTI_ERROR_INSUFFICIENT_PRIVILEGES',
                    'does not have permission to access NVIDIA GPU Performance Counters'))
                if denied:
                    row.update(status='blocked', passed=False, permission_blocked=True,
                               no_escalation_or_counter_retry=True,
                               error='Actual profiler denied access to GPU performance counters; raw error retained')
            if ok and row.get('validation', {}).get('gpu_executed') is True:
                row['gpu_executed'] = True
            if row.get('executed'):
                H.write_json(self.results / 'stages' / name / 'stage.json', row)
            self.save()
            return ok

        def await_workspace(self):
            row = {'name': 'wait-workspace-lock', 'status': 'waiting', 'passed': False,
                   'executed': False, 'gpu_executed': False, 'started_at_utc': H.utc()}
            self.report['stages'].append(row); self.save(); fd = None; started = time.monotonic()
            try:
                lock = '/tmp/mxfp4-framework-followup-' + str(os.geteuid()) + '-' + hashlib.sha256(H.B_ROOT.encode()).hexdigest()[:20] + '.lock'
                H.need(str(Path(H.B_ROOT).resolve(strict=True)) == H.B_ROOT, 'Original B lock key differs')
                row['same_B_lock_path'] = lock
                fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                item = os.fstat(fd)
                H.need(item.st_uid == os.geteuid() and stat.S_ISREG(item.st_mode), 'Unsafe same-UID GPU flock')
                while time.monotonic() < self.deadline - 15:
                    if (self.workspace / 'archive-manifest.json').is_file():
                        try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError: pass
                        else:
                            completion, evidence = Q.archive_evidence(self.workspace, 'native')
                            H.fields(completion, {'status': 'passed', 'passed': True, 'gpu_executed': True,
                                'source_and_binaries_unchanged': True,
                                'all_numerical_gates_passed_before_performance': True,
                                'source_commit': SOURCE_COMMIT, 'source_archive_sha256': SOURCE_SHA,
                                'runner_sha256': WSHA, 'cache_build': 'off'}, 'Completed workspace prerequisite')
                            stages = completion.get('stages', [])
                            H.need(len(stages) == 10 and {s.get('name') for s in stages} == set(W.planned_names())
                                   and all(s.get('passed') is True and s.get('status') == 'passed' for s in stages),
                                   'Workspace ten-stage plan is not complete/passed')
                            self.expected = completion.get('source_hashes_before')
                            H.need(isinstance(self.expected, dict) and len(self.expected) == 33
                                   and completion.get('source_hashes_after') == self.expected
                                   and H.tree_hashes(self.source) == self.expected, 'Frozen workspace source differs')
                            cache_evidence = self.prior_gate()  # Exact reviewed cache archive/artifact/quant gates.
                            self.workspace_gate(self.workspace / 'results/workspace-check.json')
                            self.report['target_fp32_input_sha256'] = completion['input_fingerprints']['4096']
                            cache_completion = H.document(self.cache / 'results/completion.json')
                            micro_sha = cache_completion.get('binary_hashes_after_gpu', {}).get(str(self.micro))
                            H.need(isinstance(micro_sha, str) and H.sha(self.micro) == micro_sha,
                                   'Actual preparation microchecker binary hash differs')
                            self.binaries[str(self.micro)] = micro_sha
                            destination = self.results / 'prior-workspace'; destination.mkdir(mode=0o700)
                            for name in ('archive-manifest.json', 'results/completion.json',
                                         'results/workspace-check.json', 'results/source-binding.json'):
                                self.remember(self.workspace / name)
                                shutil.copyfile(self.workspace / name, destination / Path(name).name)
                            self.remember(self.workspace / 'results.tar.gz')
                            self.report.update(prior_workspace_evidence=evidence, prior_cache_evidence=cache_evidence,
                                               source_hashes_before=self.expected,
                                               binary_hashes_before_gpu=dict(self.binaries))
                            self.lock_fd, fd = fd, None
                            row.update(status='passed', passed=True, same_uid_lock_acquired=True,
                                       prior_workspace=evidence, selected_library_sha256=H.sha(self.library),
                                       actual_microchecker_sha256=micro_sha)
                            return True
                    time.sleep(min(2, max(0, self.deadline - 15 - time.monotonic())))
                raise H.Blocked('Deadline before completed workspace archive/same B lock')
            except (Exception, KeyboardInterrupt) as error:
                row.update(status='blocked' if isinstance(error, H.Blocked) else 'failed',
                           error_type=type(error).__name__, error=str(error))
                if isinstance(error, KeyboardInterrupt): self.deadline = time.monotonic()
                return False
            finally:
                if fd is not None: os.close(fd)
                row.update(finished_at_utc=H.utc(), wall_seconds=time.monotonic() - started); self.save()

        def tool(self, path):
            if not path.is_file() or not os.access(path, os.X_OK): return False
            target = path.resolve(strict=True); H.regular(target)
            self.system_tools[str(target)] = H.sha(target)
            return True

        def ncu_version(self, rows):
            text = self.log_text('ncu-version')
            if not re.search(r'\b2025\.1\.1(?:\.0)?\b', text):
                raise H.Blocked('Installed NCU is not the observed Volta-compatible 2025.1.1')
            return {'passed': True, 'version': '2025.1.1', 'actual_version_output': text.strip(),
                    'gv100_supported_by_this_version': True}

        def ncu_help(self, rows):
            text = self.log_text('ncu-help')
            for option in ('profile-from-start', 'clock-control', 'cache-control', 'replay-mode',
                           'launch-count', 'query-metrics-mode', 'chips', 'target-processes',
                           'log-file', 'export', 'csv', 'page', 'config-file'):
                if '--' + option not in text: raise H.Blocked('Installed NCU lacks required option: ' + option)
            self.report['ncu_rename_kernels_off_supported'] = '--rename-kernels' in text
            return {'passed': True, 'required_installed_options_observed': True,
                    'config_file_off_option_observed': True,
                    'rename_kernels_off_option_observed': self.report['ncu_rename_kernels_off_supported']}

        def catalog_gate(self, rows):
            text = self.log_text('ncu-gv100-catalog')
            H.need(len(text) < 32 * 1024**2, 'Metric catalog exceeds diagnostic bound')
            available = set(re.findall(r'\b[a-z][a-z0-9]*__[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+', text))
            selected = [name for name in METRICS if name in available]
            if not REQUIRED_METRICS.issubset(selected):
                raise H.Blocked('Installed GV100 catalog lacks required tensor/occupancy/DRAM metrics')
            self.report['selected_profile_metrics'] = selected
            self.report['requested_metrics_not_in_actual_catalog'] = [name for name in METRICS if name not in available]
            return {'passed': True, 'selected_metrics': selected,
                    'missing_optional_metrics': self.report['requested_metrics_not_in_actual_catalog'],
                    'catalog_sha256': H.sha(self.results / 'stages/ncu-gv100-catalog/stdout.log')}

        def profile_gate(self, rows):
            data = H.document(self.results / 'profile-probe.json')
            H.fields(data, {'status': 'passed', 'passed': True, 'gpu_executed': True,
                'all_attempted_work_synchronized': True, 'source_and_library_unchanged': True,
                'probe_sha256': PROBE_SHA, 'size': 4096, 'seed': 20261123, 'warmup': 10,
                'profiled_workspace_submissions': 1, 'profile_range_entered': True,
                'profile_range_stopped': True, 'quantization_in_profile_range': False,
                'context_creation_in_profile_range': False,
                'correctness_comparison_or_D2H_in_profile_range': False,
                'diagnostic_only': True, 'unprofiled_performance_claim': False,
                'workspace_closed': True}, 'Actual ranged profiling probe')
            H.need(data['source_sha256'] == self.checker_sources(), 'Probe frozen source map differs')
            self.bound_library(data)
            for key in ('initial_all_outputs', 'actual_profiled_all_outputs'):
                H.fields(data[key], {'values': 4096**2, 'bit_mismatches': 0, 'all_finite': True,
                                    'canonical_positive_zero': True}, 'Full output bit check')
            H.fields(data['before_profile_certificate'], {'submitted_calls': 11, 'accepted': 1}, 'Warmup certificate')
            H.fields(data['actual_profiled_certificate'], {'submitted_calls': 12, 'accepted': 1, 'reject_flags': 0},
                     'Actual profiled submission certificate')
            H.need(data['input_sha256_before'] == data['input_sha256_after'], 'Profile probe input bytes changed')
            target = self.report['target_fp32_input_sha256']
            H.need(data['input_sha256_before']['a_fp32'] == target['a_fp32_bytes']
                   and data['input_sha256_before']['b_fp32'] == target['b_fp32_bytes'],
                   'Profile probe did not use the actual prior benchmark FP32 inputs')
            H.need(all(data['profiler_API'][name]['checked_runtime_call_AST_verified'] is True
                       for name in ('start', 'stop')), 'Installed Torch profiler API was not checked')
            raw = (self.results / 'profile.csv').read_text(errors='replace')
            lines = raw.splitlines(); start = next((i for i, line in enumerate(lines)
                if '"ID"' in line and '"Kernel Name"' in line and '"Metric Name"' in line), None)
            H.need(start is not None, 'NCU raw CSV has no metric header')
            records = list(csv.DictReader(io.StringIO('\n'.join(lines[start:]))))
            records = [r for r in records if r.get('ID', '').isdigit() and r.get('Metric Name')]
            kernels = {r['ID']: r['Kernel Name'] for r in records}
            H.need(1 <= len(kernels) < 10, 'No kernels or profile cap reached; cannot assert full pipeline coverage')
            names = list(kernels.values())
            for needle in ('prepare_rows', 'finalize_certificate', 'postscale_rows_if_certified', 'gemm_tiled_mx32'):
                H.need(any(needle in name for name in names), 'Expected C2 pipeline kernel absent: ' + needle)
            H.need(any('gemm' in name.lower() and 'gemm_tiled_mx32' not in name for name in names),
                   'No actual cuBLAS GEMM kernel captured')
            H.need(not any('quantize' in name.lower() for name in names), 'Quantization leaked inside profile range')
            for ident, name in kernels.items():
                observed = {r['Metric Name'] for r in records if r['ID'] == ident}
                H.need(set(self.report['selected_profile_metrics']).issubset(observed),
                       'Selected metrics missing from captured kernel: ' + name)
            profile = self.results / 'profile.ncu-rep'; H.regular(profile)
            H.need(profile.stat().st_size > 0, 'NCU binary report is empty')
            H.write_json(self.results / 'profile-csv-summary.json', {
                'kernels': kernels, 'metric_rows': len(records), 'selected_metrics': self.report['selected_profile_metrics'],
                'diagnostic_only': True, 'profile_csv_sha256': H.sha(self.results / 'profile.csv')})
            return {'passed': True, 'gpu_executed': True, 'profiled_kernels': kernels,
                    'metric_rows': len(records), 'probe_json_sha256': H.sha(self.results / 'profile-probe.json'),
                    'profile_sha256': H.sha(profile), 'diagnostic_only': True}

        def memcheck_gate(self, rows):
            checked = C.micro_gate(rows)
            text = (self.results / 'memcheck.log').read_text(errors='replace')
            H.need('ERROR SUMMARY: 0 errors' in text, 'Compute Sanitizer did not report zero memory errors')
            return {'passed': True, 'gpu_executed': True, 'memory_errors': 0,
                    'actual_microcheck_summary': checked, 'memcheck_log_sha256': H.sha(self.results / 'memcheck.log'),
                    'binary_sha256': H.sha(self.micro), 'tool': 'memcheck', 'leak_check': 'full'}

        def execute(self):
            self.integrity()
            shutil.copyfile(Path(__file__), self.results / 'diagnostics-runner.py')
            shutil.copyfile(self.probe, self.results / 'profile-probe.py')
            if not self.await_workspace(): raise H.Blocked('Passed workspace and same B flock prerequisite failed')
            self.integrity()
            available = self.tool(NCU)
            version = available and self.stage('ncu-version', [NCU, '--version'], 30, self.ncu_version)
            if not available: self.block('ncu-version', 'Installed NCU is unavailable')
            help_ok = version and self.stage('ncu-help', [NCU, '--help'], 30, self.ncu_help)
            if not version: self.block('ncu-help', 'NCU version prerequisite unavailable')
            catalog = help_ok and self.stage('ncu-gv100-catalog', [NCU, '--config-file', 'off', '--query-metrics', '--chips', 'gv100',
                '--query-metrics-mode', 'all'], 60, self.catalog_gate)
            if not help_ok: self.block('ncu-gv100-catalog', 'NCU version/options prerequisite unavailable')
            if catalog:
                command = [NCU, '--config-file', 'off']
                if self.report['ncu_rename_kernels_off_supported']:
                    command += ['--rename-kernels', 'off']
                command += ['--devices', '0', '--target-processes', 'application-only',
                    '--profile-from-start', 'off', '--clock-control', 'none', '--cache-control', 'none',
                    '--replay-mode', 'kernel', '--launch-count', '10', '--kernel-name-base', 'demangled',
                    '--section', 'LaunchStats', '--metrics', ','.join(self.report['selected_profile_metrics']),
                    '--csv', '--page', 'raw', '--log-file', str(self.results / 'profile.csv'),
                    '--export', str(self.results / 'profile'), self.args.python, '-B', str(self.probe),
                    '--source-root', str(self.source), '--compile-evidence', str(self.manifest),
                    '--output', str(self.results / 'profile-probe.json')]
                self.stage('ncu-profile', command, 300, self.profile_gate, gpu=True, library=self.library)
            else:
                self.block('ncu-profile', 'NCU compatible version/options/actual metric catalog unavailable', gpu=True)
            if self.tool(SANITIZER):
                tool_ok = self.stage('compute-sanitizer-version', [SANITIZER, '--version'], 30,
                    lambda rows: {'passed': True, 'actual_version_output': self.log_text('compute-sanitizer-version').strip()})
                if tool_ok:
                    self.stage('memcheck-prepare-cache', [SANITIZER, '--tool', 'memcheck', '--error-exitcode', '99',
                        '--leak-check', 'full', '--target-processes', 'application-only',
                        '--log-file', str(self.results / 'memcheck.log'), self.micro,
                        '--device', '0', '--correctness-only'], 300, self.memcheck_gate, gpu=True)
                else:
                    self.block('memcheck-prepare-cache', 'Compute Sanitizer version query failed', gpu=True)
            else:
                self.block('compute-sanitizer-version', 'Compute Sanitizer is not installed at the fixed CUDA path')
                self.block('memcheck-prepare-cache', 'Compute Sanitizer unavailable', gpu=True)

        def finish(self, error=None):
            self.cleanup()
            if error is not None: self.report.update(error_type=type(error).__name__, error=str(error))
            seen = {s['name'] for s in self.report['stages']}
            for name in PLAN:
                if name not in seen: self.block(name, 'Diagnostic prerequisite/deadline prevented execution',
                                               gpu=name in ('ncu-profile', 'memcheck-prepare-cache'))
            try:
                self.integrity(); self.report['source_and_binaries_unchanged'] = self.expected is not None
                self.report['source_hashes_after'] = H.tree_hashes(self.source) if self.expected else None
                self.report['binary_hashes_after_gpu'] = {p: H.sha(p) for p in self.binaries}
                self.report['installed_tool_files_sha256'] = self.system_tools
            except Exception as changed:
                self.report.update(integrity_error=str(changed), source_and_binaries_unchanged=False); error = changed
            try:
                statuses = {s['status'] for s in self.report['stages']}
                status = ('failed' if 'failed' in statuses or (error is not None and not isinstance(error, H.Blocked))
                          else 'completed_with_blocked_diagnostics' if 'blocked' in statuses or error is not None else 'passed')
                gpu = [s.get('gpu_executed') for s in self.report['stages'] if s.get('executed')]
                self.report.update(status=status, passed=status == 'passed', finished_at_utc=H.utc(),
                    wall_seconds=time.monotonic() - self.start,
                    gpu_executed=True if True in gpu else (None if None in gpu else False),
                    diagnostic_results={s['name']: s['status'] for s in self.report['stages']
                                        if s['name'] in ('ncu-profile', 'memcheck-prepare-cache')},
                    prior_performance_evidence_modified=False)
                H.need(len(self.report['stages']) == len(PLAN)
                       and {s['name'] for s in self.report['stages']} == set(PLAN), 'Diagnostic stage plan differs')
                self.report['result_files_sha256'] = {k: v for k, v in H.tree_hashes(self.results).items() if k != 'completion.json'}
                self.save()
                H.need(sum(p.stat().st_size for p in self.results.rglob('*') if p.is_file()) <= H.MAX_ARCHIVE_BYTES,
                       'Diagnostic archive exceeds bound')
                archive = self.run / 'results.tar.gz'
                with tarfile.open(archive, 'x:gz') as out: out.add(self.results, arcname='results')
                receipt = {'status': status, 'passed': status == 'passed', 'sha256': H.sha(archive),
                    'bytes': archive.stat().st_size, 'completion_sha256': H.sha(self.results / 'completion.json'),
                    'source_commit': SOURCE_COMMIT, 'source_archive_sha256': SOURCE_SHA,
                    'runner_sha256': self.own_hash, 'probe_sha256': PROBE_SHA,
                    'gpu_executed': self.report['gpu_executed'],
                    'diagnostic_results': self.report['diagnostic_results'], 'prior_performance_evidence_modified': False}
                H.write_json(self.run / 'archive-manifest.json', receipt); print(json.dumps(receipt), flush=True)
                return {'passed': 0, 'completed_with_blocked_diagnostics': 2, 'failed': 1}[status]
            finally:
                if self.lock_fd is not None:
                    fcntl.flock(self.lock_fd, fcntl.LOCK_UN); os.close(self.lock_fd); self.lock_fd = None
    return Diagnostics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run-directory', 'python', 'probe'): p.add_argument('--' + name, required=True)
    p.add_argument('--workspace-run', default='/workspace/user/mxfp4-workspace-20261123T2300')
    p.add_argument('--prior-cache-run', default='/workspace/user/mxfp4-c2-cache-20261123T2240')
    args = p.parse_args()
    for name in ('run_directory', 'python', 'probe', 'workspace_run', 'prior_cache_run'):
        if not os.path.isabs(getattr(args, name)): p.error('Every path must be explicitly absolute')
    args.cache_build = 'off'; args.framework_run = args.workspace_run
    global H, Q, W, C
    H = load(HELPER, HELPER_SHA, 'diagnostic_native_helper')
    Q = load(QHELPER, QSHA, 'diagnostic_quantizer_helper'); Q.H = H
    for name in ('workspace_run', 'prior_cache_run'):
        folder = Path(getattr(args, name))
        H.need(folder.is_dir() and not folder.is_symlink() and folder.stat().st_uid == os.geteuid(),
               'Prior run must be an owned nonsymlink existing directory')
        setattr(args, name, str(folder.resolve(strict=True)))
    W = load(Path(args.workspace_run) / 'results/workspace-experiment-runner.py', WSHA, 'diagnostic_workspace_helper'); W.H = H; W.Q = Q
    C = load(Path(args.prior_cache_run) / 'results/prepare-cache-experiment-runner.py', CSHA, 'diagnostic_cache_helper'); C.H = H; C.Q = Q
    H.regular(args.probe, owned=True); H.need(H.sha(args.probe) == PROBE_SHA, 'Profiling probe is not the frozen candidate')
    run = Path(args.run_directory)
    H.need(run.parent.is_dir() and not run.exists() and not run.is_symlink(), 'Use a new diagnostic run directory')
    run = run.parent.resolve(strict=True) / run.name
    for old in (Path(args.workspace_run), Path(args.prior_cache_run), Path(H.B_ROOT)):
        H.need(run != old and run not in old.parents and old not in run.parents, 'Diagnostic output overlaps prior run')
    run.mkdir(mode=0o700); runner = create_runner()(args, run)
    def interrupted(signum, frame):
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(sig, signal.SIG_IGN)
        raise KeyboardInterrupt('Diagnostics received signal ' + str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(sig, interrupted)
    error = None
    try: runner.execute()
    except (Exception, KeyboardInterrupt) as caught: error = caught
    return runner.finish(error)


if __name__ == '__main__': raise SystemExit(main())
