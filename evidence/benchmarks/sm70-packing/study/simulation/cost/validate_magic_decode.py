#!/usr/bin/env python3
"""CPU arithmetic validation; explicitly not a GPU simulator or benchmark."""
import json
from pathlib import Path
import numpy as np

def verify(k,qmax=1):
    bound=k*qmax*qmax
    low=np.arange(-bound,bound+1,dtype=np.int64)[None,:,None]
    high=np.arange(-bound,bound+1,dtype=np.int64)[:,None,None]
    cross=np.arange(-2*bound,2*bound+1,dtype=np.int64)[None,None,:]
    z=low+256*cross+65536*high
    fz=z.astype(np.float32)
    assert np.array_equal(fz.astype(np.int64),z)
    u=(fz+np.float32(12582912)).view(np.int32)
    hb=((u+32768)>>16)+(0x4b400000-0x4b40)
    lb=((u+128)&255)+(0x4b400000-128)
    h=hb.view(np.float32)*np.float32(.25)-np.float32(3145728)
    l=lb.view(np.float32)*np.float32(.25)-np.float32(3145728)
    hbad=np.count_nonzero(h!=high*.25)
    lbad=np.count_nonzero(l!=low*.25)
    return dict(K=k,qmax=qmax,radix=256,cases=z.size,low_errors=int(lbad),high_errors=int(hbad),
                max_abs_Z=int(np.abs(z).max()),arithmetic="CPU NumPy float32 / int32")

def domain_search():
    rows=[]
    for shift in range(2,11):
        r=2**shift
        exact=all(np.float16(a+r*b)==a+r*b for a in [-1,0,1] for b in [-1,0,1])
        candidates=[]
        for k in range(1,2049):
            if exact and k<r/2 and k*(2*r+1)<r*r/2 and k*(r+1)**2<2**24:
                candidates.append(k)
        rows.append(dict(radix=r,operand_exact=bool(exact),
            sufficient_max_K=max(candidates,default=0),
            max_multiple_16=max([k for k in candidates if k%16==0],default=0)))
    return rows

out=dict(gpu_executed=False,cycle_simulation=False,
         validation=[verify(32),verify(48),verify(4,3)],
         sufficient_bound_search=domain_search(),
         note="All independently bounded L,H,cross tuples tested, a superset of reachable ternary dot products. Power-of-two quarter scaling and subtraction are exact here; no general FMA emulation claim.")
assert all(v['low_errors']==0 and v['high_errors']==0 for v in out['validation'])
path=Path(__file__).with_name('magic-decode-validation.json')
path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
