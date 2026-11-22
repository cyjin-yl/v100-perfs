#!/usr/bin/env python3
"""Static compiler evidence; instruction counts are not runtime timings."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent
DECODER_OPS = {'F2I','IADD3','SHF','SGXT','I2F','LEA','FMUL','FRND','FFMA'}


def main():
    sass = (BASE/'sm70-sass.txt').read_text()
    resources = (BASE/'resources.txt').read_text()
    compiler_log = (BASE/'compile.stderr').read_text()
    sections = re.split(r'Function : (\w+)\n', sass)
    result = {}
    for name, body in zip(sections[1::2], sections[2::2]):
        instructions = re.findall(r'/\*[0-9a-f]+\*/\s+(.*?)\s*;', body)
        decoder = [s for s in instructions if s.split()[0].split('.')[0] in DECODER_OPS]
        reg = re.search(r'Function '+name+r':\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)', resources)
        assert reg is not None
        counts = Counter(s.split()[0].split('.')[0] for s in decoder)
        result[name] = dict(decoder_sass_instructions=decoder,
                            decoder_instruction_count=len(decoder),
                            decoder_opcode_counts=dict(counts),
                            registers=int(reg[1]),stack_bytes=int(reg[2]),
                            shared_bytes=int(reg[3]),local_bytes=int(reg[4]))
    expected={'middle_float_output':5,'middle_bfe_float_output':5,
              'middle_shift_float_output':4,'middle_bfe_integer_output':4,
              'middle_shift_integer_output':3}
    assert set(result)==set(expected)
    for name,count in expected.items():
        assert result[name]['decoder_instruction_count']==count
    spills=re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',compiler_log)
    assert len(spills)==5 and all(a=='0' and b=='0' for a,b in spills)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.iterdir()
            if p.is_file() and p.name in ['middle_decode.cu','middle_decode.o','sm70-sass.txt','resources.txt']}
    report=dict(scope='Actual CUDA 12.8.93 SM70 compilation and static SASS only',
                gpu_executed=False,cycle_simulation=False,kernel_timing_ms=None,
                compile_success=True,all_spills_zero=True,kernels=result,sha256=hashes,
                matched_float_output_decoder_counts=dict(original_round=5,ptx_bfe=5,shift_lea=4),
                integer_output_endpoint_counts=dict(ptx_bfe=4,shift_lea=3),
                warning='Counts exclude identical launch, addressing, global load/store and control; no speedup is measured.')
    (BASE/'compile-evidence.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:{'instructions':v['decoder_instruction_count'],'registers':v['registers']} for name,v in result.items()}))


if __name__=='__main__':
    main()
