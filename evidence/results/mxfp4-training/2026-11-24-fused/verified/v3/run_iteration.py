import concurrent.futures,json,subprocess,time,traceback,hashlib
from pathlib import Path
root=Path("/tmp/v100-mxfp4-fused-3cc0562b0106/v3")
prior=root.parent/"v2"
record={"schema_version":1,"source_archive_sha256":"402df7d9013fb8ebec2e576f8bcb0503bf624979ac673c0343711e171063220f","jobs":[],"started_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"completed":False}
def save():
 t=root/"driver-state.json.tmp";t.write_text(json.dumps(record,indent=2)+"\n");t.replace(root/"driver-state.json")
def build(v):
 out=root/("build-c"+str(v));out.mkdir(exist_ok=True)
 cmd=["python3",str(root/"tools/mxfp4_training/build_cuda.py"),"--nvcc","/usr/local/cuda-12.9/bin/nvcc","--output-dir",str(out),"--with-certified","--with-fused-benchmark","--library-only","--tiled-variant","1","--certified-variant",str(v)]
 start=time.monotonic()
 with (out/"driver-build.stdout").open("w") as so,(out/"driver-build.stderr").open("w") as se:
  try:r=subprocess.run(cmd,stdout=so,stderr=se,timeout=900);rc=r.returncode
  except subprocess.TimeoutExpired:rc=124
 return dict(stage="build",variant=v,returncode=rc,seconds=time.monotonic()-start,argv=cmd)
try:
 save()
 previous=json.loads((prior/"driver-state.json").read_text())
 assert previous.get("completed"),"Previous correctness driver has not finished"
 for v in [3,4,5]:
  summary=next(x for x in map(json.loads,(prior/f"build-c{v}/correctness.jsonl").read_text().splitlines()) if x["type"]=="certified_summary")
  assert summary["passed"] and summary["gpu_executed"] and summary["certified_variant"]==v
 record["correctness_gate"]="Full 69-call V2 suite; exact source and variant match checked before each run";save()
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for item in pool.map(build,[3,4,5]):record["jobs"].append(item);save()
 for v in [3,4,5]:
  if not any(x["variant"]==v and x["returncode"]==0 for x in record["jobs"]):continue
  out=root/("build-c"+str(v))
  old=json.loads((prior/f"build-c{v}/compile-evidence.json").read_text());new=json.loads((out/"compile-evidence.json").read_text())
  assert new["certified_variant"]==old["certified_variant"]==v
  assert old["fragment_mapping_sha256"]==new["fragment_mapping_sha256"]
  assert all(new["source_sha256"].get(k)==val for k,val in old["source_sha256"].items()),"Previously checked CUDA/helper source changed"
  cmd=[str(out/"bench_mxfp4_fused"),"--device","0","--m","1536","--n","1536","--k","1536","--mode","random_block_scales","--seed","20261123","--iters","10","--rounds","5","--warmup","3","--max-mib","12","--reserve-mib","48"]
  with (out/"packed-1536.jsonl").open("w") as so,(out/"packed-1536.stderr").open("w") as se:
   try:r=subprocess.run(cmd,stdout=so,stderr=se,timeout=300);rc=r.returncode
   except subprocess.TimeoutExpired:rc=124
  record["jobs"].append(dict(stage="benchmark",variant=v,returncode=rc,argv=cmd,matched_correctness_source=True,correctness_report_sha256=hashlib.sha256((prior/f"build-c{v}/correctness.jsonl").read_bytes()).hexdigest()));save()
 record["completed"]=True
except Exception as exc:
 record["error"]=str(exc);record["traceback"]=traceback.format_exc()
finally:
 record["finished_at_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime());save()
