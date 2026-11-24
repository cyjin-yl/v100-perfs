#!/usr/bin/env python3
"""Run synthetic timing probes, not hardware benchmarks or numeric emulation."""
from pathlib import Path
import csv
import hashlib
import json
import re
import subprocess
import time

ROOT=Path(__file__).resolve().parent
exe=ROOT/'build/accel-sim.out'
cases=json.loads((ROOT/'synthetic/manifest.json').read_text())
results=[]
for variant in ['upstream_qv100','whatif_hmma_latency8_init2']:
    for case in cases:
        if variant!='upstream_qv100' and case['warps']!=1:
            continue
        name=case['name']
        log=ROOT/'logs'/f'{variant}-{name}.log'
        argv=[str(exe),'-trace',str(ROOT/'synthetic'/name/'kernelslist.g'),
              '-config',str(ROOT/'configs/qv100-upstream.config'),
              '-config',str(ROOT/'configs/qv100-upstream-trace.config')]
        if variant!='upstream_qv100':
            argv+=['-config',str(ROOT/'configs/latency8-whatif.config')]
        start=time.monotonic()
        with log.open('w') as f:
            proc=subprocess.run(argv,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=120)
        elapsed=time.monotonic()-start
        raw=log.read_text()
        parsed={}
        for key in ['gpu_sim_cycle','gpu_sim_insn','gpu_tot_sim_cycle','gpu_tot_sim_insn','gpu_ipc']:
            m=re.findall(r'^'+re.escape(key)+r'\s*=\s*(\S+)',raw,re.M)
            if m:parsed[key]=float(m[-1]) if key=='gpu_ipc' else int(m[-1])
        success=proc.returncode==0 and '*** exit detected ***' in raw
        result={'variant':variant,**case,'returncode':proc.returncode,'completed':success,
                'scope':'synthetic dependency-graph cycle simulation, not hardware timing',
                'simulator_wall_seconds':elapsed,'log':str(log.relative_to(ROOT)),
                'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),'argv':argv,**parsed}
        results.append(result)
        (ROOT/'synthetic-results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps({k:result.get(k) for k in ['variant','name','completed','gpu_sim_cycle','gpu_sim_insn','simulator_wall_seconds']}),flush=True)
        if not success:raise RuntimeError(f'{name} did not complete; inspect {log}')
fields=['variant','name','hmma_per_warp','independent_accumulator_chains','warps',
        'hmma_warp_instructions','other_warp_instructions','gpu_sim_cycle','gpu_sim_insn','gpu_ipc','completed','log']
with (ROOT/'synthetic-results.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(results)
