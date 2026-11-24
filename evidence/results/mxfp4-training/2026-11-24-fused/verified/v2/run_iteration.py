import concurrent.futures,json,subprocess,time,traceback,hashlib
from pathlib import Path
root=Path("/tmp/v100-mxfp4-fused-3cc0562b0106/v2")
record={"schema_version":1,"source_archive_sha256":"ee91ede25497f0ce2b363d1356378c1629f3b43c6aba2f5877a018c281cace48","jobs":[],"started_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"completed":False}
def save():
 (root/"driver-state.json").write_text(json.dumps(record,indent=2)+"\n")
def build(v):
 out=root/("build-c"+str(v));out.mkdir(exist_ok=True)
 cmd=["python3",str(root/"tools/mxfp4_training/build_cuda.py"),"--nvcc","/usr/local/cuda-12.9/bin/nvcc","--output-dir",str(out),"--with-certified","--with-benchmark","--tiled-variant","1","--certified-variant",str(v)]
 start=time.monotonic()
 with (out/"driver-build.stdout").open("w") as so,(out/"driver-build.stderr").open("w") as se:
  try:r=subprocess.run(cmd,stdout=so,stderr=se,timeout=900);rc=r.returncode
  except subprocess.TimeoutExpired:rc=124
 return dict(stage="build",variant=v,returncode=rc,seconds=time.monotonic()-start,argv=cmd)
try:
 save()
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for item in pool.map(build,[3,4,5]):
   record["jobs"].append(item);save()
 for v in [3,4,5]:
  if not any(x["variant"]==v and x["returncode"]==0 for x in record["jobs"]):continue
  out=root/("build-c"+str(v))
  cmd=[str(out/"check_mxfp4_certified"),"--device","0","--correctness-only","--small-memory-correctness","--reserve-mib","48"]
  with (out/"correctness.jsonl").open("w") as so,(out/"correctness.stderr").open("w") as se:
   try:r=subprocess.run(cmd,stdout=so,stderr=se,timeout=180);rc=r.returncode
   except subprocess.TimeoutExpired:rc=124
  record["jobs"].append(dict(stage="correctness",variant=v,returncode=rc,argv=cmd));save()
  if rc:continue
 record["completed"]=True
except Exception as exc:
 record["error"]=str(exc);record["traceback"]=traceback.format_exc()
finally:
 record["finished_at_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime());save()
