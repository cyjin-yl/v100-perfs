#!/usr/bin/env python3
"""Replace the observed terminal bootstrap failure once, within the old budget."""
from pathlib import Path
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import time

BASE = Path('/tank/shared/projects/research/v100-mxfp4-20261123')
CLI = '/home/ezra/verdant-dev4-transfer/tools/crater-1.4.1'
OLD_JOB = 'sg-jincy26-261123-e5bb3'
OWNER = 'jincy26'
DISPLAY = 'mxfp4-c8-1124-sshfix-v4'
SOURCE_COMMIT = '91e776034aba6e215b085a373affb5c71f2a79e6'
SOURCE_SHA = '50a4d7a0444bf6544b2ea2f269d26013e36a6f18356fb0c53e5ab5116abedf8a'
PAYLOAD_SHA = '023b140da884955f3e36c308215b930a9c0a9269761feb06d85c8f66dc5bcf73'
RUNNER_SHA = 'd53282e098400575b499f3e3a82ce0790220b72a3fb62c8a5ae0cd75ade575fb'
DEADLINE = datetime(2026, 11, 24, 4, 52, 24, tzinfo=timezone.utc)
BOOTSTRAP = BASE / 'crater-bootstrap-c8-recovery-v4.sh'
ROOT = BASE / 'c8-replacement-v4-20261124'
BOOTSTRAP_SHA = '2a1e116cebd01fd2a13f5310717fc26b4f52040e6d0f316d593477787606ef97'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        os.fchmod(f.fileno(), 0o600)
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def query(*args):
    r = subprocess.run([CLI, *args, '--json', '--no-interactive'],
                       capture_output=True, text=True, timeout=30)
    if r.returncode:
        try:
            code = json.loads(r.stderr).get('code', 'unknown')
        except (ValueError, TypeError):
            code = 'unknown'
        raise RuntimeError(f'Crater read failed: exit={r.returncode} code={code}')
    return json.loads(r.stdout)['data']


def put(path, data):
    if path.exists():
        assert path.read_bytes() == data, str(path)
    else:
        with path.open('xb') as f:
            os.fchmod(f.fileno(), 0o600)
            f.write(data)


def main():
    ROOT.mkdir(exist_ok=True, mode=0o700)
    with (ROOT / 'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        final = ROOT / 'launch-audit.json'
        if final.exists():
            prior = json.loads(final.read_text())
            assert prior.get('phase') == 'launched', 'Inspect incomplete launch before any retry'
            print(json.dumps(prior))
            return
        assert (DEADLINE - datetime.now(timezone.utc)).total_seconds() > 120
        ctx = query('auth', 'ls').get('active_context', {})
        assert ctx.get('username') == OWNER and ctx.get('platform_url') == 'https://gpu.act.buaa.edu.cn'
        old = query('job', 'get', OLD_JOB)['job']
        assert old['username'] == OWNER and old['namespace'] == 'crater-workspace'
        assert old['createdAt'] == '2026-11-23T15:57:51Z' and old['status'] == 'Failed'
        failed_replacement = query('job', 'get', 'sg-jincy26-261124-de561')['job']
        assert failed_replacement['username'] == OWNER
        assert failed_replacement['createdAt'] == '2026-11-23T19:41:53Z'
        assert failed_replacement['status'] == 'Failed'
        v3_job = query('job', 'get', 'sg-jincy26-261124-5a8e7')['job']
        assert v3_job['username'] == OWNER and v3_job['namespace'] == 'crater-workspace'
        assert v3_job['createdAt'] == '2026-11-23T20:17:37Z' and v3_job['status'] == 'Failed'
        v3_watch = BASE / 'c8-replacement-v3-20261124/watch'
        v3_state = json.loads((v3_watch / 'state.json').read_text())
        assert v3_state['phase'] == 'failed' and v3_state['remote_payload_attempted'] is False
        v3_lock = json.loads((v3_watch / 'runner-lock.json').read_text())
        v3_proc = Path('/proc') / str(v3_lock['pid'])
        if v3_proc.exists():
            assert str(BASE / 'crater_queue_runner.py').encode() not in (v3_proc / 'cmdline').read_bytes().split(b'\0'), 'Prior owned watcher still running'
        # The v2 API returned an error; reconcile its display name before any new create.
        reconciled = query('job', 'ls', '--search', 'mxfp4-c8-1124-sshfix-v2',
                           '--owner', OWNER, '--all-pages')
        assert isinstance(reconciled.get('jobs'), list), 'Unknown job list schema; stop'
        assert not reconciled['jobs'], 'v2 submission has a matching job; inspect before recovery'
        save(ROOT / 'v2-reconciliation.json', dict(observed_at_utc=now(),
             display_name='mxfp4-c8-1124-sshfix-v2', owner=OWNER, matching_jobs=[]))
        old_state = json.loads((BASE / 'queue-watch-20261124-recovered/state.json').read_text())
        assert old_state['phase'] == 'failed' and old_state['remote_payload_attempted'] is False
        old_lock = json.loads((BASE / 'queue-watch-20261124-recovered/runner-lock.json').read_text())
        old_proc = Path('/proc') / str(old_lock['pid'])
        if old_proc.exists():
            old_argv = (old_proc / 'cmdline').read_bytes().split(b'\0')
            assert str(BASE / 'crater_queue_runner.py').encode() not in old_argv, 'Old owned watcher is still live'
        source = (BASE / 'queue-inputs-91e7760/source.tar.gz').read_bytes()
        payload = (BASE / 'queue-inputs-91e7760/crater_remote_run.py').read_bytes()
        assert hashlib.sha256(source).hexdigest() == SOURCE_SHA
        assert hashlib.sha256(payload).hexdigest() == PAYLOAD_SHA
        assert hashlib.sha256((BASE / 'crater_queue_runner.py').read_bytes()).hexdigest() == RUNNER_SHA
        bootstrap = BOOTSTRAP.read_text()
        assert hashlib.sha256(bootstrap.encode()).hexdigest() == BOOTSTRAP_SHA
        subprocess.run(['bash', '-n', str(BOOTSTRAP)], check=True, timeout=10)
        create_args = [CLI, 'job', 'create', 'custom', '--name', DISPLAY,
                       '--image', 'gpu-harbor.act.buaa.edu.cn/user-jincy26/onecat-main-sm70-cu128-ngc22:1.5.3-upgrade-20261114',
                       '--arch', 'amd64', '--cpu', '4', '--memory', '16Gi', '--gpu', '1',
                       '--gpu-resource', 'nvidia.com/v100', '--working-dir', '/workspace/user',
                       '--volume', 'user:/workspace/user', '--forward', 'ssh:2222', '--env', 'MXFP4_DEADLINE_UTC=2026-11-24T04:52:24Z',
                       '--shell', 'bash', '--command', bootstrap, '--json', '--no-interactive']
        result_path = ROOT / 'create-response.json'
        attempt_path = ROOT / 'submit-attempt.json'
        if not result_path.exists():
            assert not attempt_path.exists(), 'Submission outcome is uncertain; reconcile by display name before retry'
            assert not (ROOT / 'watch').exists(), 'Fresh watcher output directory already exists'
            occupied = subprocess.run(['tmux', 'has-session', '-t', 'mxfp4-c8-replacement-v4'],
                                      capture_output=True, timeout=10)
            assert occupied.returncode == 1, 'Watcher tmux session already exists or tmux check failed'
            prior_same_name = query('job', 'ls', '--search', DISPLAY, '--owner', OWNER, '--all-pages')
            assert isinstance(prior_same_name.get('jobs'), list), 'Unknown job list schema; stop'
            assert not prior_same_name['jobs'], 'v4 display already exists; inspect before any create'
            assert (DEADLINE - datetime.now(timezone.utc)).total_seconds() > 180
            save(attempt_path, dict(schema_version=1, started_at_utc=now(), old_job=OLD_JOB,
                 old_job_status='Failed', display_name=DISPLAY, argv=create_args,
                 bootstrap_sha256=hashlib.sha256(bootstrap.encode()).hexdigest(),
                 original_deadline_utc=DEADLINE.isoformat(), source_commit=SOURCE_COMMIT,
                 job_cancelled=False, existing_queued_job_recreated=False))
            result = subprocess.run(create_args, capture_output=True, text=True, timeout=60)
            put(ROOT / 'create.stdout', result.stdout.encode())
            put(ROOT / 'create.stderr', result.stderr.encode())
            assert result.returncode == 0, f'Create exit {result.returncode}; inspect recorded response before retry'
            response = json.loads(result.stdout)
            assert response.get('status') == 'OK'
            save(result_path, response)
        response = json.loads(result_path.read_text())
        created = response['data']['job']
        job_name = created['metadata']['name']
        assert job_name != OLD_JOB and created['metadata']['namespace'] == 'crater-workspace'
        assert created['metadata']['labels']['crater.raids.io/task-user'] == OWNER
        spec = created['spec']['tasks'][0]['template']['spec']
        assert created['spec']['tasks'][0]['replicas'] == 1
        assert spec['containers'][0]['resources']['requests'] == {'cpu': '4', 'memory': '16Gi', 'nvidia.com/v100': '1'}
        container = spec['containers'][0]
        mounts = [m for m in container['volumeMounts'] if m['mountPath'] == '/workspace/user']
        assert len(mounts) == 1 and mounts[0]['subPath'] == 'sugon-gpu-home-lab/jincy26'
        assert container['securityContext']['runAsUser'] == 0
        environment = {v['name']: v.get('value') for v in container.get('env', [])}
        assert environment.get('MXFP4_DEADLINE_UTC') == '2026-11-24T04:52:24Z'
        observed = query('job', 'get', job_name)['job']
        assert observed['username'] == OWNER and observed['namespace'] == 'crater-workspace'
        assert observed['status'] not in ['Failed', 'Completed', 'Terminated', 'Aborted']
        inputs = ROOT / 'inputs'
        inputs.mkdir(exist_ok=True, mode=0o700)
        put(inputs / 'source.tar.gz', source)
        put(inputs / 'crater_remote_run.py', payload)
        ready = dict(schema_version=1, ready=True, job_name=job_name,
                     source_commit=SOURCE_COMMIT, source_sha256=SOURCE_SHA,
                     remote_script_sha256=PAYLOAD_SHA, created_at_utc=now())
        assert not (inputs / 'ready.json').exists(), 'Inspect partial ready publication before retry'
        save(inputs / 'ready.json', ready)
        out = ROOT / 'watch'
        out.mkdir(mode=0o700)
        remaining = (DEADLINE - datetime.now(timezone.utc)).total_seconds() - 5
        assert 60 < remaining <= 12 * 3600
        argv = ['python3', '-u', str(BASE / 'crater_queue_runner.py'), '--cli', CLI,
                '--job', job_name, '--owner', OWNER, '--namespace', 'crater-workspace',
                '--pod', job_name + '-default0-0', '--source-archive', str(inputs / 'source.tar.gz'),
                '--remote-script', str(inputs / 'crater_remote_run.py'), '--ready-json', str(inputs / 'ready.json'),
                '--outdir', str(out), '--proxy-command',
                'ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ForwardAgent=no doescontainer docker exec -i act-easyconnect /usr/bin/socat STDIO TCP:%h:%p,connect-timeout=10',
                '--poll-seconds', '60', '--max-wait-hours', str(remaining / 3600)]
        record = dict(schema_version=1, phase='prepared', old_job=OLD_JOB,
                      old_job_terminal_failure='Image has no sshd', previous_replacement='sg-jincy26-261124-de561',
                      previous_replacement_terminal_failure='required_tool_missing:python3; own workspace mount missing in create response',
                      rejected_v2_display='mxfp4-c8-1124-sshfix-v2',
                      v2_reconciled_empty=True, volume_cli='user:/workspace/user',
                      previous_v3_job='sg-jincy26-261124-5a8e7',
                      previous_v3_failure='login_cannot_write_persistent_workspace',
                      new_job=job_name,
                      job_created_at=observed['createdAt'], job_status=observed['status'],
                      resources=observed['resources'], source_commit=SOURCE_COMMIT,
                      source_sha256=SOURCE_SHA, remote_script_sha256=PAYLOAD_SHA,
                      runner_sha256=RUNNER_SHA,
                      bootstrap_sha256=hashlib.sha256(bootstrap.encode()).hexdigest(),
                      original_deadline_utc=DEADLINE.isoformat(), max_wait_seconds=remaining,
                      argv=argv, outdir=str(out), tmux_session='mxfp4-c8-replacement-v4',
                      job_cancelled=False, existing_queued_job_recreated=False,
                      source_operator_changed=False, gpu_execution_observed=False)
        # Clip the entire local watcher to the original absolute deadline too.
        # INT runs the frozen runner's KeyboardInterrupt/finally cleanup for
        # its own SSH process group; KILL bounds the remaining five seconds.
        wrapper = ('import os,sys,time; '
                   f'left=int({DEADLINE.timestamp()}-time.time()); '
                   'sys.exit(70) if left <= 5 else None; '
                   "os.execvp('timeout', ['timeout','--signal=INT','--kill-after=5s',"
                   "str(left-5)+'s',*sys.argv[1:]])")
        wrapper_argv = ['python3', '-c', wrapper, *argv]
        record['absolute_deadline_wrapper_argv'] = wrapper_argv
        save(final, record)
        command = 'exec ' + shlex.join(wrapper_argv) + ' > ' + shlex.quote(str(out / 'watcher.stdout')) + ' 2>&1'
        subprocess.run(['tmux', 'new-session', '-d', '-s', record['tmux_session'], '-c', str(BASE), command], check=True, timeout=10)
        until = time.monotonic() + 6
        while not (out / 'runner-lock.json').exists() and time.monotonic() < until:
            time.sleep(0.1)
        running = json.loads((out / 'runner-lock.json').read_text())
        proc = Path('/proc') / str(running['pid'])
        live_argv = [os.fsdecode(x) for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
        assert live_argv == argv
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        record.update(phase='launched', launched_at_utc=now(), pid=running['pid'],
                      starttime=int(fields[19]), run_id=running['run_id'], process_state=fields[0])
        save(final, record)
        print(json.dumps(record))


if __name__ == '__main__':
    main()
