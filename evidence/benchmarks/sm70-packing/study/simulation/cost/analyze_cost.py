#!/usr/bin/env python3
"""Reproducible static compilation cost evidence, not timing simulation."""
from collections import Counter
from pathlib import Path
import hashlib,json,re

ROOT=Path(__file__).resolve().parent
regs={m[0]:int(m[1]) for m in re.findall(r'Function (\w+):\s+REG:(\d+)',(ROOT/'resources.txt').read_text())}
parts=re.split(r'Function : (\w+)',(ROOT/'sm70-sass.txt').read_text())
records=[]
for i in range(1,len(parts),2):
    name,body=parts[i],parts[i+1]
    ops=[]
    for line in body.splitlines():
        m=re.match(r'\s*/\*[0-9a-f]+\*/\s+((?:@!?\w+\s+)?)([A-Z][A-Z0-9.]*)(?:\s|;)',line)
        if not m:continue
        if '@!PT' in m[1]:continue # statically false startup instruction
        ops.append(m[2])
        if m[2]=='EXIT':break # padding/trap branch after return is not executed
    c=Counter(x.split('.')[0] for x in ops)
    warps_per_cta=4 if name.endswith('_w4') else 1
    reg_per_warp=((regs[name]*32+255)//256)*256
    ctas=min(32,64//warps_per_cta,65536//(reg_per_warp*warps_per_cta))
    fpcore=sum(c[x] for x in ('FMUL','FFMA','FADD'))
    intcore=sum(c[x] for x in ('IADD3','SHF','LOP3'))
    floors=dict(scheduler_issue=len(ops)/4,tensor_arithmetic=c['HMMA']*.5,
                fp32_basic=fpcore*.5,int32_basic=intcore*.5)
    records.append(dict(kernel=name,registers_per_thread=regs[name],warps_per_CTA=warps_per_cta,
      shared_memory_bytes=0,spill_bytes=0,resource_resident_warps=ctas*warps_per_cta,
      resource_occupancy_fraction=ctas*warps_per_cta/64,
      straight_line_warp_instructions=len(ops),opcode_counts=dict(sorted(c.items())),
      per_sm_amortized_cycle_floors=floors,
      optimistic_floor_max=max(floors.values()),
      notes='No measured cycles. Occupancy is resource-only with 256-register-per-warp allocation; issue floors ignore dependencies, cache misses, load/store, conversion throughput, register banks and scheduler partition details.'))

roof=[]
for m,n,k in [(16,16,32),(64,64,32),(512,512,32),(4096,4096,32),(512,512,512),(512,512,4096)]:
    raw=4*m*k+4*k*n+8*m*n
    prepacked=2*m*k+2*k*n+8*m*n
    with_pack=8*m*k+8*k*n+8*m*n
    roof.append(dict(M=m,N=n,K=k,useful_flops=4*m*n*k,
      baseline_min_HBM_bytes=raw,prepacked_fused_min_HBM_bytes=prepacked,
      materialized_pack_then_fused_min_HBM_bytes=with_pack,
      ideal_prepacked_bandwidth_ratio=raw/prepacked,
      ideal_with_pack_bandwidth_ratio=raw/with_pack,
      caveat='Read each matrix exactly once, two FP32 outputs, no alpha/beta C read. Long K requires repeated decode and compatible scales; no predicted execution time.'))

files=['packing_cost_kernels.cu','packing_cost_kernels.o','sm70-sass.txt','resources.txt',
       'm8n8k4_probe.cu','m8n8k4_probe.o','m8n8k4-sass.txt','magic-decode-validation.json']
out=dict(compiled_for='sm_70',gpu_executed=False,cycle_accurate_simulation=False,
  assumptions=dict(SM_registers=65536,register_allocation_quantum_per_warp=256,maximum_warps_per_SM=64,
    maximum_CTAs_per_SM=32,schedulers_per_SM=4,FP32_basic_scalar_instructions_per_SM_cycle=64,
    INT32_basic_scalar_instructions_per_SM_cycle=64,tensor_FLOPs_per_SM_cycle=1024,
    HMMA_884_step_FLOPs_per_warp=512),
  kernels=records,roofline_byte_models=roof,
  evidence_hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
(ROOT/'static-cost-results.json').write_text(json.dumps(out,indent=2)+'\n')
for x in records:
    print(x['kernel'],x['registers_per_thread'],x['straight_line_warp_instructions'],x['opcode_counts'].get('HMMA',0),x['resource_occupancy_fraction'],x['optimistic_floor_max'])
print(json.dumps(roof,indent=2))
