import pathlib,json,hashlib,subprocess,time,os,traceback
root=pathlib.Path(__file__).resolve().parent;base=root.parent
state={'schema_version':1,'purpose':'resolve observed V6 clock ramp with longer ordinary workload warmup','source_archive_sha256':hashlib.sha256((base/'source.tar.gz').read_bytes()).hexdigest(),'started_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'warmup_per_job':100,'benchmark_order':[6,8,8,6],'jobs':[],'binary_sha256':{},'completed':False,'clock_or_power_changes':False}
def save():
 p=root/'driver-state.json.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(root/'driver-state.json')
for v in (6,8):
 exe=base/f'build-c{v}/bench_mxfp4_fused';h=hashlib.sha256(exe.read_bytes()).hexdigest();m=json.loads((exe.parent/'compile-evidence.json').read_text());assert h==m['artifacts'][exe.name]['sha256'];state['binary_sha256'][str(v)]=h
logger=None
try:
 save()
 with (root/'background-gpu-samples.csv').open('x') as f:
  logger=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.free,clocks.current.sm,power.draw','--format=csv','--loop-ms=200'],stdout=f,stderr=subprocess.STDOUT)
  for seq,v in enumerate((6,8,8,6)):
   cmd=[str(base/f'build-c{v}/bench_mxfp4_fused'),'--m','1536','--n','1536','--k','1536','--mode','random_block_scales','--seed','20261123','--iters','10','--rounds','5','--warmup','100','--max-mib','12','--reserve-mib','48']
   name=f'abba-{seq:02d}-c{v}-1536.jsonl';start=time.monotonic()
   with (root/name).open('x') as so,(root/(name+'.stderr')).open('x') as se:r=subprocess.run(cmd,stdout=so,stderr=se,timeout=90,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'))
   state['jobs'].append({'variant':v,'returncode':r.returncode,'argv':cmd,'stdout':name,'seconds':time.monotonic()-start});save();assert r.returncode==0
 state['completed']=True
except Exception as e:state.update(error=str(e),traceback=traceback.format_exc())
finally:
 if logger:
  logger.terminate()
  try:logger.wait(timeout=5)
  except subprocess.TimeoutExpired:logger.kill();logger.wait()
 state['finished_at_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save()
