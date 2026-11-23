from pathlib import Path
import hashlib,json,tarfile
root=Path('.')
path=root/'tools/mxfp4_training/csrc/mxfp4_gemm_fused.cu'
new=path.read_text()
with tarfile.open(root/'evidence/results/mxfp4-training/2026-11-24-fused/verified/v2/source.tar.gz','r:gz') as t:
    member=next(x for x in t.getmembers() if x.name.endswith('tools/mxfp4_training/csrc/mxfp4_gemm_fused.cu'))
    old=t.extractfile(member).read().decode()
assert hashlib.sha256(old.encode()).hexdigest()=='e99b3b4444b8ebdbb6df1f25358b358a88c038fc2d211d1d86d3b029ea0f54b6'
assert new.startswith(old[:old.index('bool product_fits')])
start='bool product_fits'; end='template<class Index>\nvoid dispatch_variant'
assert old[old.index(start):old.index(end)]==new[new.index(start):new.index(end)]
start='  const bool vector_output =';end='\nbool product_fits'
old_epilogue=old[old.index(start):old.index(end)].replace('fr < FragM','fr < 2').replace('fc < FragN','fc < 4')
new_epilogue=new[new.rindex(start):new.index(end)]
assert old_epilogue==new_epilogue
checks=0; loads=0; cases=[]
for m,n in [(128,128),(129,131),(131,129),(17,19)]:
  for k in [0,32,64,96,192,288]:
    blocks=k//32;case_checks=0;case_loads=0
    for tm in range((m+127)//128):
      for tn in range((n+127)//128):
        def raw_stage(block):
          global loads,checks
          state={}
          for operand,rows,tile in [('a',m,tm*128),('b',n,tn*128)]:
            for tid in range(256):
              for copy in range(4):
                index=tid+copy*256;row=tile+index//8;vector=index%8
                if row<rows and block<blocks:
                  address=row*(k//2)+block*16+vector*2
                  scale_address=row*blocks+block
                  assert 0<=address and address+1<rows*(k//2)
                  assert 0<=scale_address<rows*blocks
                  state[operand,tid,copy]=(row,block,vector,address,scale_address)
                else:state[operand,tid,copy]=None
          return state
        raw=raw_stage(0) if blocks else None
        case_loads+=sum(v is not None for v in raw.values()) if raw else 0
        for block in range(blocks):
          expected=raw_stage(block)
          assert raw==expected
          case_checks+=len(raw)
          published=[]
          for operand in ['a','b']:
            slots=[((tid+copy*256)//8,(tid+copy*256)%8) for tid in range(256) for copy in range(4)]
            assert len(set(slots))==128*8
          if block+1<blocks:
            raw=raw_stage(block+1)
            case_loads+=sum(v is not None for v in raw.values())
    checks+=case_checks;loads+=case_loads
    cases.append({'shape':[m,n,k],'stage_group_comparisons':case_checks,'valid_raw_groups_loaded':case_loads})
assert 'cudaFuncSetAttribute' not in new
print(json.dumps({'passed':True,'gpu_executed':False,'cuda_compilation_executed':False,'original_kernel_and_helpers_byte_unchanged':True,'new_epilogue_matches_old_BK32':True,'stage_group_comparisons':checks,'valid_raw_groups_loaded':loads,'shape_cases':len(cases),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'header_sha256':hashlib.sha256(path.with_suffix('.h').read_bytes()).hexdigest(),'cases':cases},indent=2))
