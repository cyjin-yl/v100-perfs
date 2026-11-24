#!/usr/bin/env python3
"""Local fake-CLI/fake-launcher checks; never contacts Crater or starts a job."""
import ast
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

SOURCE = Path(__file__).with_name('wait_auth_and_recover.py')
ast.parse(SOURCE.read_text())
spec = importlib.util.spec_from_file_location('auth_recovery', SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
results = []

with tempfile.TemporaryDirectory(prefix='mxfp4-auth-wait-mock-') as temporary:
    base = Path(temporary)
    launcher = base / 'reviewed-submit.py'
    launcher.write_text('raise RuntimeError("This mock file must never execute")\n')
    digest = hashlib.sha256(launcher.read_bytes()).hexdigest()
    for scenario in ('expired_then_recovers', 'wrong_account', 'other_api_error', 'launcher_nonzero', 'launcher_timeout', 'deadline_expired'):
        clock = {'now': datetime(2026, 11, 23, 20, 0, tzinfo=timezone.utc), 'mono': 0.0}
        if scenario == 'deadline_expired': clock['now'] = mod.DEADLINE + timedelta(seconds=1)
        calls = {'read': 0, 'job': 0, 'launch': 0, 'sleep': []}
        outdir = base / scenario
        args = SimpleNamespace(outdir=outdir, launcher=launcher, expected_launcher_sha256=digest)

        def now(): return clock['now']
        def mono(): return clock['mono']
        def sleep(seconds):
            assert 0 < seconds <= 60
            calls['sleep'].append(seconds)
            clock['now'] += timedelta(seconds=seconds)
            clock['mono'] += seconds

        def read(argv, **kwargs):
            calls['read'] += 1
            assert argv[0] == mod.CLI
            assert argv[1:3] in (['auth', 'ls'], ['job', 'get'])
            assert kwargs['timeout'] <= 25
            if argv[1:3] == ['auth', 'ls']:
                data = {'active_context': {'username': 'other' if scenario == 'wrong_account' else mod.OWNER,
                                           'platform_url': mod.PLATFORM}}
            else:
                calls['job'] += 1
                if scenario == 'expired_then_recovers' and calls['job'] == 1:
                    return subprocess.CompletedProcess(argv, 1, '', json.dumps({'code': 'ERR_API_HTTP_ERROR',
                        'context': {'http_status': 401, 'message': 'token has invalid claims: token is expired'}}))
                if scenario == 'other_api_error':
                    return subprocess.CompletedProcess(argv, 1, '', json.dumps({'code': 'ERR_FORBIDDEN_403'}))
                data = {'job': {'jobName': mod.ORIGINAL_JOB, 'username': mod.OWNER, 'namespace': 'crater-workspace',
                                'createdAt': mod.ORIGINAL_CREATED, 'status': 'Failed'}}
            return subprocess.CompletedProcess(argv, 0, json.dumps({'data': data}), '')

        def launch(argv, **kwargs):
            calls['launch'] += 1
            state = json.loads((outdir / 'auth-wait-state.json').read_text())
            assert state['launcher_attempted'] and state['launcher_attempt_count'] == 1
            assert state['phase'] == 'launching'
            assert Path(argv[1]) == outdir / 'reviewed-launcher.py'
            assert kwargs['cwd'] == launcher.parent
            assert state['executed_launcher'] == argv[1] and state['executed_launcher_sha256'] == digest
            assert hashlib.sha256(Path(argv[1]).read_bytes()).hexdigest() == digest
            assert kwargs['timeout'] <= 300
            if scenario == 'launcher_timeout': raise subprocess.TimeoutExpired('fake launcher', 300)
            return 9 if scenario == 'launcher_nonzero' else 0

        recovery = mod.Recovery(args, read=read, launch=launch, now=now, monotonic=mono, sleep=sleep)
        rc = recovery.run()
        state = json.loads((outdir / 'auth-wait-state.json').read_text())
        expected = {'expired_then_recovers': (0, 'launcher_returned_zero', 1),
                    'wrong_account': (1, 'failed', 0), 'other_api_error': (1, 'failed', 0),
                    'launcher_nonzero': (1, 'launcher_nonzero', 1),
                    'launcher_timeout': (1, 'launcher_outcome_unknown', 1),
                    'deadline_expired': (1, 'deadline_reached', 0)}[scenario]
        assert (rc, state['phase'], calls['launch']) == expected
        assert not state['login_attempted'] and not state['account_changed'] and not state['token_read']
        assert state['deadline_utc'] == mod.DEADLINE.isoformat()
        if scenario == 'expired_then_recovers':
            assert calls['sleep'] == [60] and state['waiting_for_auth_count'] == 1
            prior = (outdir / 'auth-wait-state.json').read_bytes()
            try:
                mod.Recovery(args, read=read, launch=launch, now=now, monotonic=mono, sleep=sleep).run()
            except mod.Stop:
                pass
            else:
                raise AssertionError('Repeated invocation did not refuse existing state')
            assert calls['launch'] == 1 and (outdir / 'auth-wait-state.json').read_bytes() == prior
        if scenario == 'deadline_expired': assert calls['read'] == 0
        results.append({'case': scenario, 'passed': True, 'phase': state['phase'], 'launcher_calls': calls['launch']})

report = {'passed': True, 'tests': results, 'case_count': len(results), 'ast_passed': True,
          'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'real_cli_calls': 0, 'real_launcher_calls': 0, 'gpu_executed': False}
Path(__file__).with_name('auth-wait-host-checks.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
