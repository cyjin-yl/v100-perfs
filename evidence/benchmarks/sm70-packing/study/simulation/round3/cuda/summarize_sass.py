#!/usr/bin/env python3
"""Summarize actual sm70 object disassembly; no execution-time inference."""
from pathlib import Path
from collections import Counter
import re,json,hashlib
ROOT=Path(__file__).resolve().parent
regs={m[0]:int(m[1]) for m in re.findall(r'Function (\S+):\s+REG:(\d+)',(ROOT/'resources.txt').read_text())}
parts=re.split(r'Function : (\S+)',(ROOT/'sm70-sass.txt').read_text())
records=[]
for i in range(1,len(parts),2):
    name,body=parts[i],parts[i+1];ops=[]
    for line in body.splitlines():
        m=re.match(r'\s*/\*[0-9a-f]+\*/\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9.]*)(?:\s|;)',line)
        if m:ops.append(m[1])
    counts=Counter(ops)
    row=dict(symbol=name,registers=regs[name],static_instruction_lines=len(ops),opcodes=dict(sorted(counts.items())))
    spec=re.search(r'mapped_kernelILi(\d+)ELi(\d+)ELi(\d+)ELb([01])ELb([01])',name)
    if spec:
        chunk,bits,mode,scaled,vector=map(int,spec.groups())
        row.update(decode_chunk=chunk,low_bits=bits,decode_mode=mode,scaled=bool(scaled),vector_store=bool(vector),
                   HMMA=sum(v for op,v in counts.items() if op.startswith('HMMA')),
                   global_store_instructions=sum(v for op,v in counts.items() if op.startswith('STG')),
                   scalar_store_instructions=counts['STG.E.SYS'],vector64_store_instructions=counts['STG.E.64.SYS'],
                   FRND=counts['FRND'],F2I=sum(v for op,v in counts.items() if op.startswith('F2I')),
                   I2F=sum(v for op,v in counts.items() if op.startswith('I2F')))
        assert row['HMMA']==chunk # Static inner body: 4 STEP per K4, unrolled when chunk=32.
        assert row['global_store_instructions']==(8 if vector else 16)
        if mode==2:assert row['FRND']==row['F2I']==row['I2F']==0
    records.append(row)
sources=['mapped_packing.cu','fragment_mapping.hpp','validate_fragments.py','mapped_packing.o','sm70-sass.txt','resources.txt']
out=dict(gpu_executed=False,hardware_status='pending',object_compile='passed',target='sm_70',
         evidence_kind='Real CUDA compilation / static SASS counts only',kernels=records,
         kernel_count=len(records),mapped_kernel_count=sum('decode_chunk' in x for x in records),
         notes=['Looped kernels: static instruction count is not a dynamic trace or execution time.',
                'The 16 E2M1 nibble codes are decoded arithmetically; no constant-LUT throughput assumption.',
                'Compiler can retain address/control instructions; none of the counts imply an end-to-end speedup.'],
         sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources})
(ROOT/'compile-evidence.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('kernels','sha256')},indent=2))
for r in records:
    if 'decode_chunk' in r and r['decode_mode']==2:
        print(r['decode_chunk'],r['low_bits'],r['scaled'],r['vector_store'],r['registers'],r['HMMA'],r['global_store_instructions'])
