import concurrent.futures,subprocess,pathlib,json,time,hashlib
root=pathlib.Path('/tmp/v100-mxfp4-perf-3cc0562b0106')
nvcc='/usr/local/cuda-12.9/bin/nvcc'
source=root/'tools/mxfp4_training/csrc'
def build(v):
 out=root/('build-perf-c'+str(v))
 out.mkdir(exist_ok=True)
 cmd=['python3',str(root/'tools/mxfp4_training/build_cuda.py'),'--nvcc',nvcc,'--library-only','--with-benchmark','--with-certified','--tiled-variant','1','--certified-variant',str(v),'--output-dir',str(out)]
 print(json.dumps({'build_start':v}),flush=True)
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=150)
 (out/'driver-build.log').write_text(r.stdout+r.stderr)
 if r.returncode:return v,r.returncode
 cmd=[nvcc,'-O3','-std=c++17','--ftz=false','-gencode=arch=compute_70,code=[sm_70,compute_70]',str(source/'check_mxfp4_certified.cu'),'-L'+str(out),'-lmxfp4_sm70','-Xlinker=-rpath','-Xlinker='+str(out),'-o',str(out/'check_mxfp4_certified')]
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
 (out/'certified-harness-build.log').write_text(r.stdout+r.stderr)
 rec={'argv':cmd,'returncode':r.returncode,'source_sha256':hashlib.sha256((source/'check_mxfp4_certified.cu').read_bytes()).hexdigest(),'included_oracle_sha256':hashlib.sha256((source/'check_mxfp4.cu').read_bytes()).hexdigest(),'library_sha256':hashlib.sha256((out/'libmxfp4_sm70.so').read_bytes()).hexdigest()}
 if not r.returncode:rec['artifact_sha256']=hashlib.sha256((out/'check_mxfp4_certified').read_bytes()).hexdigest()
 (out/'certified-harness-build.json').write_text(json.dumps(rec,indent=2)+'\n')
 print(json.dumps({'build_complete':v,'exit':r.returncode}),flush=True)
 return v,r.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 results=list(pool.map(build,[0,1,2]))
summary={'builds':results,'runs':[]}
for v,status in results:
 if status:continue
 out=root/('build-perf-c'+str(v))
 for binary,args,label in [
  ('check_mxfp4_certified',['--correctness-only','--reserve-mib','64'],'correctness'),
  ('bench_mxfp4',['--m','2048','--n','2048','--k','2048','--mode','random_block_scales','--iters','10','--rounds','5','--warmup','3','--reserve-mib','64','--max-mib','64'],'varied-2048')]:
  r=subprocess.run([str(out/binary)]+args,capture_output=True,text=True,timeout=45)
  (out/(label+'.jsonl')).write_text(r.stdout);(out/(label+'.stderr')).write_text(r.stderr)
  summary['runs'].append({'variant':v,'label':label,'exit':r.returncode})
  for line in r.stdout.splitlines():
   d=json.loads(line)
   if d.get('type') in ['certified_summary','summary']:print(json.dumps({'variant':v,'record':d}),flush=True)
   if d.get('type')=='benchmark':
    print(json.dumps({'variant':v,'timings':d['timings']}),flush=True)
  if r.returncode:print(r.stderr,flush=True);break
(root/'vector-comparison-complete.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
