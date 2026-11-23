import datetime,hashlib,json,os,pathlib,shlex,subprocess
base=pathlib.Path('/tank/shared/projects/research/v100-mxfp4-20261123')
old=base/'queue-watch-20261124'
new=base/'queue-watch-20261124-recovered'
cli='/home/ezra/verdant-dev4-transfer/tools/crater-1.4.1'
job='sg-jincy26-261123-e5bb3'
owner='jincy26'
deadline=datetime.datetime(2026,11,24,4,52,24,tzinfo=datetime.timezone.utc)
audit=base/'queue-recovery-20261124.json'
if audit.exists():
 print(audit.read_text());raise SystemExit(0)
state=json.loads((old/'state.json').read_text())
assert state['phase']=='failed' and 'ERR_UNAUTHORIZED_401' in state['error']
assert state['remote_payload_attempted'] is False and state['remote_payload_completed'] is False
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  cmd=(proc/'cmdline').read_bytes().split(b'\0')
  if str(base/'crater_queue_runner.py').encode() in cmd and job.encode() in cmd:
   raise RuntimeError('An owned watcher is still live: '+proc.name)
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert not new.exists(), 'Recovery directory already exists; inspect before retrying'
auth=json.loads(subprocess.run([cli,'auth','ls','--json'],check=True,capture_output=True,text=True,timeout=25).stdout)
ctx=auth.get('data',auth).get('active_context',{})
assert ctx.get('username')==owner and ctx.get('platform_url')=='https://gpu.act.buaa.edu.cn',ctx
query=json.loads(subprocess.run([cli,'job','get',job,'--json','--no-interactive'],check=True,capture_output=True,text=True,timeout=25).stdout)
j=query['data']['job']
assert j['jobName']==job and j['username']==owner and j['namespace']=='crater-workspace'
assert j['createdAt']=='2026-11-23T15:57:51Z' and j['resources']['nvidia.com/v100']=='1'
ready=json.loads((base/'queue-source-ready.json').read_text())
for path,key in [(base/'queued-source.tar.gz','source_sha256'),(base/'crater_remote_run.py','remote_script_sha256')]:
 assert hashlib.sha256(path.read_bytes()).hexdigest()==ready[key]
now=datetime.datetime.now(datetime.timezone.utc)
remaining=(deadline-now).total_seconds()-5
assert 60<remaining<=12*3600
args=['python3','-u',str(base/'crater_queue_runner.py'),'--cli',cli,'--job',job,'--owner',owner,'--namespace','crater-workspace','--pod',job+'-default0-0','--source-archive',str(base/'queued-source.tar.gz'),'--remote-script',str(base/'crater_remote_run.py'),'--ready-json',str(base/'queue-source-ready.json'),'--outdir',str(new),'--proxy-command','ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ForwardAgent=no doescontainer docker exec -i act-easyconnect /usr/bin/socat STDIO TCP:%h:%p,connect-timeout=10','--poll-seconds','60','--max-wait-hours',str(remaining/3600)]
new.mkdir(mode=0o700)
(new/'previous-failed-state.json').write_text(json.dumps(state,indent=2)+'\n')
(new/'recovery-job-query.json').write_text(json.dumps(query,indent=2)+'\n')
record={'schema_version':1,'recovered_at_utc':now.isoformat(),'original_deadline_utc':deadline.isoformat(),'max_wait_seconds':remaining,'job':job,'job_created_at':j['createdAt'],'job_status':j['status'],'job_phase':j['podGroupPhase'],'source_commit':ready['source_commit'],'source_sha256':ready['source_sha256'],'remote_script_sha256':ready['remote_script_sha256'],'same_account_verified':True,'login_or_account_switch_attempted':False,'job_recreated':False,'old_outdir':str(old),'new_outdir':str(new),'argv':args,'tmux_session':'mxfp4-queue-e5bb3-recovered'}
(new/'launch-record.json').write_text(json.dumps(record,indent=2)+'\n')
command='exec '+shlex.join(args)+' > '+shlex.quote(str(new/'watcher.stdout'))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s',record['tmux_session'],'-c',str(base),command],check=True,timeout=10)
audit.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
