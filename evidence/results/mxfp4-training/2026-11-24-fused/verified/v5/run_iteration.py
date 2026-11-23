import concurrent.futures,json,subprocess,time,traceback,hashlib,re
from pathlib import Path
root=Path("/tmp/v100-mxfp4-fused-3cc0562b0106/v5")
record={"schema_version":1,"source_archive_sha256":"5a3010b8c2ed97e203d545453d7992af6e4e82e73fe6b671c4b2cdcecb1fb88e","jobs":[],"started_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"completed":False}
def save():
 t=root/"driver-state.json.tmp";t.write_text(json.dumps(record,indent=2)+"\n");t.replace(root/"driver-state.json")
def execute(label,variant,command,outfile,timeout):
 start=time.monotonic()
 with (root/outfile).open("w") as so,(root/(outfile+".stderr")).open("w") as se:
  try:r=subprocess.run(command,stdout=so,stderr=se,timeout=timeout);rc=r.returncode
  except subprocess.TimeoutExpired:rc=124
 return dict(stage=label,variant=variant,returncode=rc,seconds=time.monotonic()-start,argv=command,stdout=outfile)
def build(v):
 cmd=["python3",str(root/"tools/mxfp4_training/build_cuda.py"),"--nvcc","/usr/local/cuda-12.9/bin/nvcc","--output-dir",str(root/f"build-c{v}"),"--with-certified","--with-fused-benchmark","--tiled-variant","1","--certified-variant",str(v)]
 return execute("build",v,cmd,f"build-c{v}.stdout",900)
logger=None
try:
 save()
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for item in pool.map(build,[6,7]):record["jobs"].append(item);save()
 assert all(x["returncode"]==0 for x in record["jobs"]),"Build failed"
 for v in [6,7]:
  cmd=[str(root/f"build-c{v}/check_mxfp4_certified"),"--correctness-only","--small-memory-correctness","--reserve-mib","48"]
  item=execute("correctness",v,cmd,f"correctness-c{v}.jsonl",180);record["jobs"].append(item);save()
  assert item["returncode"]==0,"Correctness failed"
 with (root/"background-gpu-samples.csv").open("w") as f:
  logger=subprocess.Popen(["nvidia-smi","--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.free,clocks.current.sm,power.draw","--format=csv","--loop-ms=200"],stdout=f,stderr=subprocess.STDOUT)
  for seq,v in enumerate([6,7,7,6]):
   cmd=[str(root/f"build-c{v}/bench_mxfp4_fused"),"--m","1536","--n","1536","--k","1536","--mode","random_block_scales","--seed","20261123","--iters","10","--rounds","5","--warmup","3","--max-mib","12","--reserve-mib","48"]
   item=execute("benchmark_abba",v,cmd,f"abba-{seq:02d}-c{v}-1536.jsonl",300);record["jobs"].append(item);save()
   if item["returncode"]:
    record["benchmark_stopped_on_failure"]=True;break
 record["completed"]=True
except Exception as exc:
 record["error"]=str(exc);record["traceback"]=traceback.format_exc()
finally:
 if logger is not None:
  logger.terminate()
  try:logger.wait(timeout=5)
  except subprocess.TimeoutExpired:logger.kill();logger.wait()
 record["finished_at_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime());save()
