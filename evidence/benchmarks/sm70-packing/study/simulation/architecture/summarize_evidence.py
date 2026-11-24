#!/usr/bin/env python3
"""Parse recorded simulation output; never label model cycles as GPU timings."""
from pathlib import Path
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent
raw = (ROOT / 'logs/official-lud64-qv100.log').read_text()
chunks = re.split(r'^kernel_name =\s*', raw, flags=re.M)[1:]
rows = []
for index, chunk in enumerate(chunks, 1):
    row = {'kernel_index': index, 'kernel_name': chunk.splitlines()[0].strip()}
    for key in ['gpu_sim_cycle', 'gpu_sim_insn', 'gpu_tot_sim_cycle', 'gpu_tot_sim_insn']:
        row[key] = int(re.search(r'^' + key + r'\s*=\s*(\d+)', chunk, re.M).group(1))
    rows.append(row)
assert len(rows) == 10 and '*** exit detected ***' in raw
assert sum(x['gpu_sim_cycle'] for x in rows) == rows[-1]['gpu_tot_sim_cycle']
assert sum(x['gpu_sim_insn'] for x in rows) == rows[-1]['gpu_tot_sim_insn']
summary = {
    'scope': 'Official published Tesla-V100-directory trace replay in upstream QV100 cycle model; not packing-kernel execution, not numeric validation, not hardware timing',
    'completed': True,
    'kernel_count': len(rows),
    'gpu_tot_sim_cycle': rows[-1]['gpu_tot_sim_cycle'],
    'gpu_tot_sim_insn': rows[-1]['gpu_tot_sim_insn'],
    'argv': ['./build/accel-sim.out', '-trace', './official-lud64/kernelslist.g',
             '-config', './configs/qv100-upstream.config', '-config', './configs/qv100-upstream-trace.config'],
    'log_sha256': hashlib.sha256(raw.encode()).hexdigest(),
    'kernels': rows,
}
(ROOT / 'official-lud64-results.json').write_text(json.dumps(summary, indent=2) + '\n')
with (ROOT / 'official-lud64-results.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({k: v for k, v in summary.items() if k != 'kernels'}, indent=2))
