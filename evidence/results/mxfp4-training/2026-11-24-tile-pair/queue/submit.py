"""Submit one authorized 1-V100 experiment, with a four-hour container bound."""
from pathlib import Path
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import subprocess

BASE = Path('/tank/shared/projects/research/v100-mxfp4-20261124')
ROOT = BASE / 'tile-pair-job'
CLI = '/home/ezra/verdant-dev4-transfer/tools/crater-1.4.1'
OWNER = 'jincy26'
DISPLAY = 'mxfp4-tile-pair-1124'

def save(name, value):
    with (ROOT / name).open('x') as f:
        os.fchmod(f.fileno(), 0o600)
        json.dump(value, f, indent=2)
        f.write('\n')

def read(*args):
    result = subprocess.run([CLI, *args, '--json', '--no-interactive'],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, 'Crater read failed; no submission attempted'
    value = json.loads(result.stdout)
    assert value['status'] == 'OK'
    return value['data']

def main():
    ROOT.mkdir(mode=0o700, exist_ok=True)
    with (ROOT / 'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (ROOT/'submit-attempt.json').exists(), 'Reconcile the prior attempt before retry'
        context=read('auth','ls')['active_context']
        assert context['username']==OWNER and context['platform_url']=='https://gpu.act.buaa.edu.cn'
        jobs=read('job','ls','--owner',OWNER,'--search','mxfp4','--all-pages')['jobs']
        assert all(j['status'].lower() in {'failed','aborted','terminated','deleted','completed','succeeded'} for j in jobs), 'An existing experiment is active; inspect first'
        assert not any(j['name']==DISPLAY for j in jobs), 'Display name already exists; reconcile first'
        save('prior-jobs.json', jobs)
        bootstrap=(BASE/'bootstrap.sh').read_text()
        subprocess.run(['bash','-n',str(BASE/'bootstrap.sh')],check=True,timeout=10)
        argv=[CLI,'job','create','custom','--name',DISPLAY,
              '--image','gpu-harbor.act.buaa.edu.cn/user-jincy26/onecat-main-sm70-cu128-ngc22:1.5.3-upgrade-20261114',
              '--arch','amd64','--cpu','4','--memory','16Gi','--gpu','1',
              '--gpu-resource','nvidia.com/v100','--working-dir','/workspace/user',
              '--volume','user:/workspace/user','--forward','ssh:22',
              '--shell','bash','--command',bootstrap,'--json','--no-interactive']
        save('submit-attempt.json',{'started_at_utc':datetime.now(timezone.utc).isoformat(),
             'display_name':DISPLAY,'owner':OWNER,'bootstrap_sha256':hashlib.sha256(bootstrap.encode()).hexdigest(),
             'container_lifetime_bound_seconds':14400,'existing_jobs_modified':False})
        r=subprocess.run(argv,capture_output=True,text=True,timeout=60)
        (ROOT/'create.stdout').write_text(r.stdout)
        (ROOT/'create.stderr').write_text(r.stderr)
        assert r.returncode==0, 'Create result recorded; reconcile before any retry'
        response=json.loads(r.stdout)
        assert response['status']=='OK'
        save('create-response.json',response)
        job=response['data']['job']
        assert job['metadata']['namespace']=='crater-workspace'
        assert job['metadata']['labels']['crater.raids.io/task-user']==OWNER
        spec=job['spec']['tasks'][0]['template']['spec']
        assert job['spec']['tasks'][0]['replicas']==1
        assert spec['containers'][0]['resources']['requests']=={'cpu':'4','memory':'16Gi','nvidia.com/v100':'1'}
        name=job['metadata']['name']
        current=read('job','get',name)
        save('created-observation.json',current)
        print(json.dumps({'job_name':name,'display':DISPLAY,'status':current['job']['status'],
                          'created_at':current['job']['createdAt'],'resources':current['job']['resources']}))

if __name__=='__main__':
    main()
