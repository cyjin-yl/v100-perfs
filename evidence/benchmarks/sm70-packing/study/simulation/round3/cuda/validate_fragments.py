#!/usr/bin/env python3
"""Independent CPU reconstruction of documented m8n8k4 fragment layouts.

This is an arithmetic/layout emulator, not a cycle simulator and not hardware
evidence. Every tested packed integer dot is bounded within binary32 exact
integer range, so the selected cases do not depend on a claimed general
Volta rounding model.
"""
from pathlib import Path
import json
import shutil,subprocess,tempfile
import numpy as np

ROOT=Path(__file__).resolve().parent
Q=np.array([0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12],dtype=np.int32)
CASES={
 'ternary_k32':dict(chunk=32,ra=256,rb=256,low_bits=8,a=[-1,0,1],b=[-1,0,1]),
 'seven_k4':dict(chunk=4,ra=256,rb=256,low_bits=8,a=list(range(-3,4)),b=list(range(-3,4))),
 'ternary_full_k4':dict(chunk=4,ra=512,rb=128,low_bits=7,a=[-1,0,1],b=sorted(set(Q.tolist())))
}

def group(lane):return (lane>>2)&3
def off(lane):return 8*(group(lane)>>1),8*(group(lane)&1)
def ar(lane):return off(lane)[0]+(lane&3)+(4 if lane>=16 else 0)
def bc(lane):return off(lane)[1]+(lane&3)+(4 if lane>=16 else 0)
def cr(lane,i):return off(lane)[0]+(lane&1)+(i&2)+(4 if lane>=16 else 0)
def cc(lane,i):return off(lane)[1]+(i&4)+(lane&2)+(i&1)

def load_fragment(a,b,kk):
    # Each lane has two f16x2 registers for A and two for B.
    fa=np.array([a[ar(l),kk:kk+4] for l in range(32)],dtype=np.float16)
    fb=np.array([b[kk:kk+4,bc(l)] for l in range(32)],dtype=np.float16)
    assert np.array_equal(fa.astype(np.float32),np.array([a[ar(l),kk:kk+4] for l in range(32)]))
    assert np.array_equal(fb.astype(np.float32),np.array([b[kk:kk+4,bc(l)] for l in range(32)]))
    # The uint32 view also checks lower/upper half ordering of f16x2.
    return fa.view(np.uint32).copy(),fb.view(np.uint32).copy()

def mma_fragment(fa_bits,fb_bits,fc):
    fa=fa_bits.view(np.float16).astype(np.float32)
    fb=fb_bits.view(np.float16).astype(np.float32)
    out=np.empty((32,8),dtype=np.float32)
    for g in range(4):
        ga=np.zeros((8,4),np.float32);gb=np.zeros((4,8),np.float32);gc=np.zeros((8,8),np.float32)
        lanes=[l for l in range(32) if group(l)==g]
        for l in lanes:
            local=(l&3)+(4 if l>=16 else 0)
            ga[local]=fa[l];gb[:,local]=fb[l]
            for i in range(8):gc[cr(l,i)%8,cc(l,i)%8]=fc[l,i]
        for k in range(4):gc=np.float32(gc+np.float32(ga[:,k,None]*gb[k,None,:]))
        for l in lanes:
            for i in range(8):out[l,i]=gc[cr(l,i)%8,cc(l,i)%8]
    return out

def scatter(fc):
    out=np.empty((16,16),dtype=np.float32)
    for l in range(32):
        for i in range(8):out[cr(l,i),cc(l,i)]=fc[l,i]
    return out

def decode(z,bits,mode):
    r=2**bits
    if mode=='float':
        h=np.rint(z*np.float32(1/65536)).astype(np.float32)
        rem=np.float32(z-h*np.float32(65536))
        mid=np.rint(rem*np.float32(1/r)).astype(np.float32)
        low=np.float32(rem-mid*np.float32(r))
        return low*np.float32(.25),h*np.float32(.25)
    if mode=='integer':
        iz=z.astype(np.int32)
        return ((((iz+r//2)&(r-1))-r//2)*.25).astype(np.float32),(((iz+32768)>>16)*.25).astype(np.float32)
    u=(z+np.float32(12582912)).view(np.int32)
    hb=((u+32768)>>16)+(0x4b400000-0x4b40)
    lb=((u+r//2)&(r-1))+(0x4b400000-r//2)
    h=hb.view(np.float32)*np.float32(.25)-np.float32(3145728)
    low=lb.view(np.float32)*np.float32(.25)-np.float32(3145728)
    return low,h

def pack_nibbles(q):
    lut={int(v):i for i,v in enumerate(Q) if i!=8}
    codes=np.array([lut[int(x)] for x in q.ravel()],np.uint8)
    return codes[0::2]|(codes[1::2]<<4)

def unpack_nibbles(data,shape):
    out=np.empty(data.size*2,np.int32);out[0::2]=Q[data&15];out[1::2]=Q[data>>4]
    return out.reshape(shape)

def domain_gate(q,cfg):
    if not np.isin(q[:,:16],cfg['a']).all() or not np.isin(q[:,16:],cfg['b']).all():
        raise ValueError('Unsupported packing domain; values are never clipped.')

def check_mapping():
    out_coords=[(cr(l,i),cc(l,i)) for l in range(32) for i in range(8)]
    assert len(set(out_coords))==256
    for g in range(4):
        lanes=[l for l in range(32) if group(l)==g]
        assert sorted((l&3)+(4 if l>=16 else 0) for l in lanes)==list(range(8))
    # Impulse products prove every output and every local reduction position
    # from independently reconstructed group matrices, not just a round-trip.
    trials=0
    for row in range(16):
        for col in range(16):
            for kk in range(4):
                a=np.zeros((16,4),np.float32);b=np.zeros((4,16),np.float32)
                a[row,kk]=1;b[kk,col]=1
                fa,fb=load_fragment(a,b,0)
                out=scatter(mma_fragment(fa,fb,np.zeros((32,8),np.float32)))
                exp=np.zeros((16,16),np.float32);exp[row,col]=1
                assert np.array_equal(out,exp)
                trials+=1
    return trials

def check_cpp_header():
    compiler=shutil.which('c++') or shutil.which('g++')
    if not compiler:raise RuntimeError('A host C++ compiler is required to cross-check the actual CUDA mapping header.')
    with tempfile.TemporaryDirectory(prefix='m8-layout-') as tmp:
        binary=Path(tmp)/'emit-map'
        subprocess.run([compiler,'-std=c++17','-O2',str(ROOT/'emit_header_map.cpp'),'-o',str(binary)],check=True,capture_output=True,text=True)
        lines=subprocess.run([str(binary)],check=True,capture_output=True,text=True).stdout.splitlines()
    assert len(lines)==32
    for lane,line in enumerate(lines):
        expected=[lane,group(lane),ar(lane),bc(lane)]
        for i in range(8):expected.extend([cr(lane,i),cc(lane,i)])
        assert list(map(int,line.split()))==expected
    return True

def run_case(name,k,variable_scales,seed,extreme=False):
    cfg=CASES[name];rng=np.random.default_rng(seed)
    # Storage matches hardware: [stream][A rows then B columns][K].
    q=np.empty((2,32,k),np.int32)
    q[:,:16]=rng.choice(cfg['a'],size=(2,16,k))
    q[:,16:]=rng.choice(cfg['b'],size=(2,16,k))
    if extreme:
        q[:,:16]=max(cfg['a']);q[:,16:]=max(cfg['b'])
        q[1,::2]*=-1
    domain_gate(q,cfg)
    codes=pack_nibbles(q);assert np.array_equal(unpack_nibbles(codes,q.shape),q)
    sb=(k+31)//32
    exp=rng.integers(-2,3,size=(2,32,sb)) if variable_scales else np.zeros((2,32,sb),np.int32)
    scale=np.exp2(exp).astype(np.float32)
    packed=(q[0]+np.array([cfg['ra']]*16+[cfg['rb']]*16)[:,None]*q[1]).astype(np.float16)
    assert np.array_equal(packed.astype(np.int32),q[0]+np.array([cfg['ra']]*16+[cfg['rb']]*16)[:,None]*q[1])
    ap=packed[:16].astype(np.float32);bp=packed[16:].T.astype(np.float32)
    refs=np.zeros((2,16,16),np.float64)
    for s in range(2):
        for blk in range(sb):
            lo=blk*32;hi=min(lo+32,k)
            aa=q[s,:16,lo:hi]*.5*scale[s,:16,blk,None]
            bb=(q[s,16:,lo:hi]*.5*scale[s,16:,blk,None]).T
            h_a=aa.astype(np.float16);h_b=bb.astype(np.float16)
            assert np.isfinite(h_a).all() and np.isfinite(h_b).all()
            assert np.array_equal(h_a.astype(np.float64),aa) and np.array_equal(h_b.astype(np.float64),bb)
            refs[s]+=aa@bb
    errors={}
    for mode in ['float','integer','magic']:
        out=np.zeros((2,16,16),np.float32)
        for blk in range(sb):
            physical=np.zeros((2,16,16),np.float32)
            lo=blk*32;hi=min(lo+32,k)
            for start in range(lo,hi,cfg['chunk']):
                fc=np.zeros((32,8),np.float32)
                for kk in range(start,min(start+cfg['chunk'],hi),4):
                    fa,fb=load_fragment(ap,bp,kk);fc=mma_fragment(fa,fb,fc)
                low,high=decode(fc,cfg['low_bits'],mode)
                physical[0]+=scatter(low);physical[1]+=scatter(high)
            for s in range(2):out[s]+=physical[s]*scale[s,:16,blk,None]*scale[s,16:,blk][None,:]
        errors[mode]=int(np.count_nonzero(out!=refs))
        assert errors[mode]==0,(name,k,variable_scales,mode,errors)
    return dict(candidate=name,K=k,scales='E8M0 exponents [-2,2]' if variable_scales else 'all one',
                extreme=extreme,seed=seed,output_values=512,errors=errors,
                baseline_half_conversion_exact=True)

def main():
    header_verified=check_cpp_header();impulses=check_mapping();cases=[]
    rejected=0
    for cfg in CASES.values():
        try:domain_gate(np.full((2,32,32),12,np.int32),cfg)
        except ValueError:rejected+=1
    assert rejected==len(CASES)
    for name,cfg in CASES.items():
        for k in sorted(set([cfg['chunk'],32,64,128])):
            for scales in [False,True]:
                for extreme in [False,True]:cases.append(run_case(name,k,scales,20261122+len(cases),extreme))
    mapping=[dict(lane=l,group=group(l),a_row=ar(l),b_col=bc(l),
                  outputs=[dict(i=i,row=cr(l,i),col=cc(l,i)) for i in range(8)]) for l in range(32)]
    (ROOT/'fragment-map.json').write_text(json.dumps(mapping,indent=2)+'\n')
    out=dict(gpu_executed=False,cycle_simulation=False,official_layout_source='https://docs.nvidia.com/cuda/archive/11.4.0/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-884-f16',
             host_compiled_header_map_verified=header_verified,
             impulse_cases=impulses,cases=cases,case_count=len(cases),
             full_E2M1_by_full_E2M1_rejected_candidates=rejected,
             errors_total=sum(sum(x['errors'].values()) for x in cases),
             scope='Documented fragment map, exact bounded-integer arithmetic, E2M1 nibbles, finite small E8M0 scales; not general Volta rounding or training convergence.')
    (ROOT/'cpu-fragment-validation.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='cases'},indent=2))

if __name__=='__main__':main()
