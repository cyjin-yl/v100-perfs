#!/usr/bin/env python3
"""Generate synthetic register-dependency probes for Accel-Sim.

These ARE NOT hardware traces and ARE NOT traces of v100_packing_probe.cu.
There are no data values, memory operations, matrix outputs, or training steps.
Only the scheduler/operand-collector/execution-unit timing model is exercised.
Each HMMA names the base register of each SASS operand, matching the upstream
NVBit tracer's register-operand abstraction. This is not a lane-accurate model.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent

def generate(name, hmma_per_warp=256, chains=1, warps=1):
    out = ROOT / 'synthetic' / name
    out.mkdir(parents=True, exist_ok=True)
    lines = [
        f'-kernel name = SYNTHETIC_NOT_HARDWARE_{name}',
        '-kernel id = 1', '-grid dim = (1,1,1)',
        f'-block dim = ({warps*32},1,1)', '-shmem = 0', '-nregs = 80',
        '-binary version = 70', '-cuda stream id = 0',
        '-enable lineinfo = 0', '-nvbit version = NONE_SYNTHETIC',
        '-accelsim tracer version = 3', '-shmem base_addr = 0x0',
        '-local mem base_addr = 0x0',
        '# SYNTHETIC register-dependency probe, no hardware execution',
        '#BEGIN_TB', 'thread block = 0,0,0',
    ]
    for warp in range(warps):
        instrs = []
        for reg in [0,2,4,6,8,10,12,14,64,66]:
            instrs.append(f'ffffffff 1 R{reg} MOV 0 0 0')
        for i in range(hmma_per_warp):
            reg = 2*(i % chains)
            instrs.append(f'ffffffff 1 R{reg} HMMA.884.F32.F32.STEP0 3 R64 R66 R{reg} 0 0')
        instrs.append('ffffffff 0 EXIT 0 0 0')
        lines.extend([f'warp = {warp}', f'insts = {len(instrs)}'])
        lines.extend(f'{pc*16:04x} {ins}' for pc,ins in enumerate(instrs))
    lines.extend(['#END_TB', ''])
    data='\n'.join(lines).encode()
    (out/'kernel-1.traceg').write_bytes(data)
    (out/'kernelslist.g').write_text('kernel-1.traceg\n')
    return {'name':name,'provenance':'synthetic, not GPU captured',
            'hmma_per_warp':hmma_per_warp,'independent_accumulator_chains':chains,
            'warps':warps,'hmma_warp_instructions':hmma_per_warp*warps,
            'other_warp_instructions':11*warps,
            'sha256':hashlib.sha256(data).hexdigest()}

cases=[]
for chains in [1,4,8]:
    cases.append(generate(f'hmma256_chains{chains}_warps1',chains=chains))
for warps in [4,16]:
    cases.append(generate(f'hmma256_chains1_warps{warps}',warps=warps))
cases.append(generate('no_hmma_warps1',hmma_per_warp=0))
(ROOT/'synthetic'/'manifest.json').write_text(json.dumps(cases,indent=2)+'\n')
print(json.dumps(cases,indent=2))
