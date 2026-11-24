#!/usr/bin/env python3
"""Compile the frozen C2 screen, then wait for the owned native GPU batch."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
NATIVE = Path('/workspace/user/mxfp4-c9-c10-20261123T2128')
INPUTS = Path('/workspace/user/mxfp4-c9-c10-inputs-20261123T2118')
PYTHON = '/opt/venv/bin/python3'
active = None
started = time.monotonic()
report = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
          'gpu_serialization': 'native finalized archive and candidate same B flock',
          'stages': []}


def save():
    p = ROOT / 'controller-state.json'
    q = p.with_suffix('.tmp')
    q.write_text(json.dumps(report, indent=2) + '\n')
    q.replace(p)


def cleanup():
    global active
    if active is None:
        return
    # Give the candidate wrapper time to clean its separately owned children.
    for sig, seconds in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(active.pid, sig)
        except ProcessLookupError:
            pass
        try:
            active.wait(timeout=seconds)
            break
        except subprocess.TimeoutExpired:
            pass
    active = None


def interrupted(signum, _frame):
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, signal.SIG_IGN)
    cleanup()
    raise InterruptedError('Controller interrupted: ' + str(signum))


def stage(name, argv, timeout):
    global active
    row = {'name': name, 'argv': argv, 'status': 'running',
           'started_at_utc': datetime.now(timezone.utc).isoformat()}
    report['stages'].append(row)
    save()
    with (ROOT / (name + '.stdout.log')).open('xb') as out, (ROOT / (name + '.stderr.log')).open('xb') as err:
        active = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
        row['pid'] = active.pid
        save()
        try:
            row['returncode'] = active.wait(timeout=timeout)
        finally:
            cleanup()
    row['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    row['status'] = 'passed' if row['returncode'] == 0 else ('completed_with_blocked_cases' if name == 'run' and row['returncode'] == 2 else 'failed')
    save()
    return row['returncode']


for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(sig, interrupted)
try:
    binding = json.loads((ROOT / 'candidate-binding.json').read_text())
    for name, expected in binding['sha256'].items():
        assert Path(name).name == name and hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    report['candidate_sha256'] = binding['sha256']
    save()
    rc = stage('build', [PYTHON, '-B', str(ROOT / 'candidate.py'), 'build',
               '--source-root', str(NATIVE / 'source'), '--source-binding', str(INPUTS / 'source-binding.json'),
               '--source-archive', str(INPUTS / 'source.tar.gz'), '--nvcc', '/usr/local/cuda-12.8/bin/nvcc',
               '--output-directory', str(ROOT / 'build')], 660)
    if rc != 0:
        raise RuntimeError('Candidate compilation failed')
    gate_started = time.monotonic()
    while not (NATIVE / 'archive-manifest.json').is_file():
        if time.monotonic() - started > 3500:
            raise TimeoutError('Native batch did not finalize within controller wait budget')
        time.sleep(5)
    report['native_archive_observed_at_utc'] = datetime.now(timezone.utc).isoformat()
    report['native_wait_seconds'] = time.monotonic() - gate_started
    save()
    # The candidate verifies the archive, source, repeated C2 gates and takes the
    # same exclusive flock before submitting any GPU work.
    rc = stage('run', [PYTHON, '-B', str(ROOT / 'candidate.py'), 'run',
               '--build-directory', str(ROOT / 'build'), '--run-directory', str(ROOT / 'run'),
               '--prior-native-run', str(NATIVE)], 1845)
    report['status'] = 'passed' if rc == 0 else ('completed_with_blocked_cases' if rc == 2 else 'failed')
except BaseException as error:
    report.update(status='failed', error_type=type(error).__name__, error=str(error))
finally:
    cleanup()
    report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    report['wall_seconds'] = time.monotonic() - started
    save()
    print(json.dumps(report), flush=True)
sys.exit(0 if report['status'] == 'passed' else (2 if report['status'] == 'completed_with_blocked_cases' else 1))
