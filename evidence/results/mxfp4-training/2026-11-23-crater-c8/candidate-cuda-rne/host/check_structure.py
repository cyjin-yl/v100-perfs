#!/usr/bin/env python3
"""Host-only source preservation check; does not import Torch or execute CUDA."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

BASE = Path('/workspace/scratch/3cc0562b0106/v100-perfs')
ROOT = Path(__file__).resolve().parents[1]
REL = Path('tools/mxfp4_training')

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def nodes(p):
    return {n.name: n for n in ast.parse(p.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

def same_file_nodes(filename, names):
    old, new = nodes(BASE/REL/filename), nodes(ROOT/REL/filename)
    for name in names:
        assert ast.dump(old[name], include_attributes=False) == ast.dump(new[name], include_attributes=False), name
    return list(names)

result = {'gpu_executed': False, 'cuda_compiled': False, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
          'base_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=BASE, text=True).strip()}
result['unchanged_torch_nodes'] = same_file_nodes('torch_ops.py', (
    'TorchMXTensor', 'quantize_mx_torch', '_load_library', '_gemm_certified', 'gemm_packed',
    '_MXLinear', 'mxfp4_linear', 'MXLinear'))
result['unchanged_smoke_nodes'] = same_file_nodes('pretrain_smoke.py', (
    'TinyDecoder', 'batch_for', 'loss_for', 'run_arm'))
recipe = nodes(ROOT/REL/'torch_ops.py')['MXRecipe']
fields = {n.target.id: ast.literal_eval(n.value) for n in recipe.body if isinstance(n, ast.AnnAssign)}
assert fields['quantizer_backend'] == 'torch' and fields['backend'] == 'reference'
assert fields['rounding'] == fields['gradient_rounding'] == 'nearest'
smoke_calls = [n for n in ast.walk(ast.parse((ROOT/REL/'pretrain_smoke.py').read_text()))
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'MXRecipe']
assert len(smoke_calls) == 5 and all(any(k.arg == 'quantizer_backend' for k in n.keywords) for n in smoke_calls)
result['all_five_smoke_recipes_explicit_quantizer'] = True
result['defaults'] = fields
files = ['torch_ops.py', 'build_cuda.py', 'pretrain_smoke.py', 'csrc/mxfp4_quantize_rne.cu', 'csrc/mxfp4_quantize_rne.h']
result['candidate_sha256'] = {name: digest(ROOT/REL/name) for name in files}
result['original_sha256'] = {name: digest(BASE/REL/name) for name in files if (BASE/REL/name).is_file()}
result['script_sha256'] = digest(Path(__file__))
result['passed'] = True
result['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
out = ROOT/'host/structure-result.json'
out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
