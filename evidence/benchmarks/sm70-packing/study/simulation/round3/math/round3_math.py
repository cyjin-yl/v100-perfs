#!/usr/bin/env python3
"""Round 3: asymmetric packing and a robust middle-coefficient V100 path.

CPU execution of the validated integer translation of the published V100 model.
No hardware timing, instruction-throughput, or training claim is made.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

BASE = Path(__file__).resolve().parent
NUMERICS = BASE.parents[1] / "numerics"
sys.path.insert(0, str(NUMERICS))
from v100_model import v100_bfma4, v100_dot, float_to_bits, UPSTREAM_COMMIT
from run_numerical_study import verify_upstream_provenance

FULL = (-12,-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8,12)
RADII = (1,2,3,4,6,8,12)


def alphabet(radius):
    return tuple(x for x in FULL if abs(x) <= radius)


def operand_exact(radius, radix):
    q=alphabet(radius)
    p=np.array([x+radix*y for x,y in itertools.product(q,repeat=2)],dtype=np.int64)
    with np.errstate(over="ignore"):
        h=p.astype(np.float16).astype(np.float64)
    return bool(np.all(h==p))


OPERAND_EXACT={(q,1<<s):operand_exact(q,1<<s) for q in RADII for s in range(1,11)}


def bound(kind, amax, bmax, ra, rb, k):
    d=k*amax*bmax
    source_ok=OPERAND_EXACT[(amax,ra)] and (kind=="shared" or OPERAND_EXACT[(bmax,rb)])
    if kind=="shared":
        abs_sum=d*(ra+1)
        error=0
        low_safe=2*d<ra
        high_safe=True
        precision_safe=abs_sum<2**24
    else:
        abs_sum=d*(ra+1)*(rb+1)
        if kind=="middle":
            assert ra==rb
            # Global alignment/final RZ grid at most 2**(Emax-23).
            # Magnitudes never exceed the sum of absolute raw products.
            grid=1<<max(0,abs_sum.bit_length()-1-23)
            error=6*((k+3)//4)*(grid-1)
            low_safe=2*(d+error)<ra
            high_safe=2*(2*d*ra+d+error)<ra*ra
            precision_safe=True  # Error bound replaces exact packed-sum demand.
        else:
            error=0
            low_safe=2*d<min(ra,rb)
            high_safe=2*d*(1+ra+rb)<ra*rb
            precision_safe=abs_sum<2**24
    return {"kind":kind,"amax":amax,"bmax":bmax,"radix_a":ra,"radix_b":rb,"compressed_k":k,
            "operand_encodings_exact":source_ok,"sum_absolute_products_bound":abs_sum,
            "coefficient_outer_bound":d,"model_packed_sum_error_bound":error,
            "low_or_middle_extraction_bound_pass":low_safe,"high_extraction_bound_pass":high_safe,
            "precision_bound_pass":precision_safe,
            "sufficient_conditions_pass":bool(source_ok and low_safe and high_safe and precision_safe)}


def parameter_search():
    count=0; summaries=[]
    for kind in ("shared","independent","middle"):
        for amax,bmax in itertools.product(RADII,repeat=2):
            safe=[]
            radices=((1<<s,1<<t) for s in range(1,11) for t in range(1,11)) if kind=="independent" else ((1<<s,1<<s) for s in range(1,11))
            for ra,rb in radices:
                for k in (4,8,12,16,24,32,48,64):
                    row=bound(kind,amax,bmax,ra,rb,k);count+=1
                    if row["sufficient_conditions_pass"]:safe.append(row)
            best=max(safe,key=lambda r:(r['compressed_k'],-r['sum_absolute_products_bound'])) if safe else None
            summaries.append({"kind":kind,"amax":amax,"bmax":bmax,"best_by_k_then_abs_bound":best})
    return {"candidates_checked":count,"summaries":summaries,
            "scope":"Linear radix families and sufficient bounds only; failed bounds are not impossibility proofs.",
            "middle_scope":"One same-scale dot's two K subsequences, not two separately recoverable outputs."}


def model_with_trace(p,q):
    c=0.0;dropped=0;traces=[]
    for start in range(0,len(p),4):
        c,t=v100_bfma4(p[start:start+4],q[start:start+4],c,trace=True)
        dropped+=bool(any(t.get('discarded_magnitude_bits',[])))
        traces.append({"start":start,"result":c,"trace":t})
    return c,dropped,traces


def encode_decode(kind,vectors,ra,rb):
    if kind=="shared":
        a0,a1,b0=vectors
        p=a0+ra*a1;q=b0
        wanted=[int(a0@b0),int(a1@b0)]
    else:
        a0,a1,b0,b1=vectors
        p=a0+ra*a1
        q=b1+rb*b0 if kind=="middle" else b0+rb*b1
        wanted=[int(a0@b0+a1@b1)] if kind=="middle" else [int(a0@b0),int(a1@b1)]
    ph=p.astype(np.float16).astype(np.float64).tolist()
    qh=q.astype(np.float16).astype(np.float64).tolist()
    z,dropped,traces=model_with_trace(ph,qh)
    if kind=="shared":
        high=round(z/ra);decoded=[round(z-ra*high),high]
    elif kind=="middle":
        high=round(z/(ra*rb));decoded=[round((z-ra*rb*high)/ra)]
    else:
        high=round(z/(ra*rb));r=min(ra,rb)
        decoded=[round(z-r*round(z/r)),high]
    bitfield_decoded=None
    if kind=="middle" and ra==rb==512:
        assert z==int(z) and abs(z)<2**31-256
        bits=((int(z)+256)>>9)&511
        bitfield_decoded=[bits-512 if bits>=256 else bits]
    elif kind=="independent" and (ra,rb)==(512,128):
        zi=int(z)
        bitfield_decoded=[((zi+64)&127)-64,(zi+32768)>>16]
    return {"wanted":wanted,"decoded":decoded,"bitfield_decoded":bitfield_decoded,
            "packed_result":z,"exact_integer_packed_result":int(p@q),
            "packed_error":z-int(p@q),"dropped_alignment_groups":dropped,
            "inputs_q":[x.tolist() for x in vectors],"packed_a":ph,"packed_b":qh,"traces":traces}


def run_candidate(label,kind,amax,bmax,ra,rb,k,n,rng):
    qa,qb=alphabet(amax),alphabet(bmax)
    arrays=[rng.choice(qa,(n,k)),rng.choice(qa,(n,k)),rng.choice(qb,(n,k))]
    if kind!="shared":arrays.append(rng.choice(qb,(n,k)))
    cases=[list(x) for x in zip(*arrays)]
    product_domains=(qa,qa,qb) if kind=="shared" else (qa,qa,qb,qb)
    constants=list(itertools.product(*product_domains))
    cases.extend([[np.repeat(x,k) for x in vals] for vals in constants])
    alternating=list(itertools.product(*[(-max(abs(x) for x in dom),0,max(abs(x) for x in dom)) for dom in product_domains]))
    cases.extend([[np.array([v if (j+lane)%2==0 else -v for j in range(k)]) for lane,v in enumerate(vals)] for vals in alternating])
    if kind=="middle" and amax==bmax==3:
        low=np.zeros(k,dtype=np.int64);low[0]=1
        cases.append([low.copy(),np.repeat(3,k),np.repeat(3,k),low.copy()])
    errors=rounded=drops=scale_errors=bitfield_errors=0;max_error=0;first_loss=None;first_failure=None
    for index,vectors in enumerate(cases):
        r=encode_decode(kind,vectors,ra,rb)
        wrong=r['wanted']!=r['decoded'];errors+=wrong
        if r['bitfield_decoded'] is not None:bitfield_errors+=r['bitfield_decoded']!=r['wanted']
        rounded+=r['packed_error']!=0;max_error=max(max_error,abs(r['packed_error']))
        drops+=r['dropped_alignment_groups']
        if wrong and first_failure is None:first_failure=r
        if r['packed_error']!=0 and first_loss is None:first_loss=r
        if kind=="independent":
            # Different exact power-of-two scales for all four lanes.
            exps=(index%9-4,(index*3)%9-4,(index*5)%9-4,(index*7)%9-4)
            expected=[r['wanted'][0]*2.0**(exps[0]+exps[2]-2),r['wanted'][1]*2.0**(exps[1]+exps[3]-2)]
            got=[r['decoded'][0]*2.0**(exps[0]+exps[2]-2),r['decoded'][1]*2.0**(exps[1]+exps[3]-2)]
            scale_errors+=got!=expected
    proven=bound(kind,amax,bmax,ra,rb,k)
    assert proven['sufficient_conditions_pass'] and errors==0 and scale_errors==0 and bitfield_errors==0
    assert max_error<=proven['model_packed_sum_error_bound']
    return {"label":label,"proof":proven,"random_cases":n,"constant_cases":len(constants),
            "alternating_cases":len(alternating),"total_cases":len(cases),"decoded_failures":errors,
            "packed_sum_not_exact_cases":rounded,"maximum_observed_packed_sum_error":max_error,
            "groups_discarding_alignment_bits":drops,"independent_scale_failures":scale_errors,
            "bitfield_decoder_failures":bitfield_errors,
            "first_loss_with_correct_output":first_loss,"first_failure":first_failure}


def block32_middle(n,rng):
    a=rng.integers(-3,4,(n,32));b=rng.integers(-3,4,(n,32))
    a=np.concatenate((a,np.full((1,32),3)));b=np.concatenate((b,np.full((1,32),3)))
    wrong=0;maximum_error=0
    for av,bv in zip(a,b):
        result=0
        for start in (0,16):
            aa,bb=av[start:start+16],bv[start:start+16]
            rec=encode_decode('middle',[aa[0::2],aa[1::2],bb[0::2],bb[1::2]],512,512)
            result+=rec['decoded'][0]
            maximum_error=max(maximum_error,abs(rec['packed_error']))
        wrong+=result!=int(av@bv)
    assert wrong==0
    return {"original_k":32,"packed_k_total":16,"packed_k_per_decode":8,
            "decodes_per_original_block":2,"bfma4_groups_per_decode":2,"packed_bfma4_groups_total":4,
            "ordinary_bfma4_groups_total":8,"cases":len(a),"failures":wrong,
            "maximum_intermediate_packed_error":maximum_error,
            "scale_rule":"Both halves belong to the same original MX32 scale product; multiply decoded sum by scaleA*scaleB/4.",
            "qualification":"Arithmetic operation counts only. Packing, epilogue, memory and register costs are not timed."}


def failure_examples():
    def rec(kind,arrays,ra,rb):return encode_decode(kind,[np.array(x,dtype=np.int64) for x in arrays],ra,rb)
    asym_a=rec('independent',[[1]*8,[0]*8,[12]*8,[0]*8],512,128)
    asym_b=rec('independent',[[1]*8,[0]*8,[-4]*8,[1]+[0]*7],512,128)
    assert asym_a['packed_result']==asym_b['packed_result']==96 and asym_a['wanted']!=asym_b['wanted']
    int2_a=rec('independent',[[-2,-2,-2,-1],[0]*4,[-12,-12,-8,0],[0]*4],512,128)
    int2_b=rec('independent',[[-2,-2,-2,-1],[0]*4,[12,12,8,0],[0,0,0,-1]],512,128)
    assert int2_a['packed_result']==int2_b['packed_result']==64 and int2_a['wanted']!=int2_b['wanted']
    middle16=rec('middle',[[3]*16]*4,512,512)
    assert middle16['decoded']!=middle16['wanted']
    # Even the full encoded scalar result cannot resolve different scale products.
    scale_a=rec('middle',[[1],[0],[1],[0]],512,512)
    scale_b=rec('middle',[[0],[1],[0],[1]],512,512)
    assert scale_a['packed_result']==scale_b['packed_result']==512
    return {"asymmetric_k8_exact_collision":[asym_a,asym_b],
            "signed_int2_by_full_e2m1_k4_exact_collision":[int2_a,int2_b],
            "middle_q3_k16_carry_failure":middle16,
            "different_scale_products_collision":{"case_a":scale_a,"case_b":scale_b,"scale_products":[2,4],
                "actual_e2m1_outputs":[0.5,1.0],"reason":"A single middle coefficient cannot retain differently weighted diagonal terms."}}


def quantize_mx32(x):
    maxima=np.max(np.abs(x),axis=1)
    scale=np.where(maxima>0,np.exp2(np.floor(np.log2(np.maximum(maxima,2**-100)))-2),1.0)
    values=np.array([0,.5,1,1.5,2,3,4,6])
    distance=np.abs(np.abs(x/scale[:,None])[:,:,None]-values)
    minimum=distance.min(axis=2,keepdims=True)
    # Exact ties prefer an even E2M1 numerical code (the RNE requirement).
    rank=np.where(distance==minimum,np.arange(8)[None,None,:]%2*8+np.arange(8),100)
    code=np.argmin(rank,axis=2)
    q=(2*values[code]*np.sign(x)).astype(np.int64)
    return q,scale


def normalized_chunks(chunks,mode):
    out=chunks.copy();factors=np.ones(len(chunks),dtype=np.int64)
    for i,row in enumerate(chunks):
        nonzero=np.abs(row[row!=0])
        if not len(nonzero):continue
        if mode=='power2':
            factor=1
            while np.all(nonzero%(factor*2)==0):factor*=2
        elif mode=='gcd':factor=int(np.gcd.reduce(nonzero))
        else:factor=1
        out[i]//=factor;factors[i]=factor
    return out,factors


def admission_and_repair(blocks,rng):
    distributions={
        'gaussian':rng.normal(size=(blocks,32)),
        'laplace':rng.laplace(size=(blocks,32)),
        'student_t3':rng.standard_t(3,size=(blocks,32)),
        'sparse90_gaussian':rng.normal(size=(blocks,32))*(rng.random((blocks,32))<.1),
        'ternary_positive_control':rng.integers(-1,2,(blocks,32)).astype(float),
    }
    result=[]
    for label,source in distributions.items():
        q,scales=quantize_mx32(source)
        chunks=q.reshape(-1,4)
        pair_indices=rng.permutation(len(chunks)).reshape(-1,2)
        chunks16=q.reshape(-1,16)
        pairs16=rng.permutation(len(chunks16)).reshape(-1,2)
        rows={}
        for mode in ('none','power2','gcd'):
            normalized,factors=normalized_chunks(chunks,mode)
            narrow=np.max(np.abs(normalized),axis=1)<=3
            ternary=np.max(np.abs(normalized),axis=1)<=1
            normalized16,_=normalized_chunks(chunks16,mode)
            middle_narrow=np.max(np.abs(normalized16),axis=1)<=3
            rows[mode]={"q3_chunk_fraction":float(narrow.mean()),"ternary_chunk_fraction":float(ternary.mean()),
                        "shared_two_rows_q3_pair_fraction":float(np.all(narrow[pair_indices],axis=1).mean()),
                        "asymmetric_two_ternary_A_lanes_pair_fraction":float(np.all(ternary[pair_indices],axis=1).mean()),
                        "middle_original_k16_operand_fraction":float(middle_narrow.mean()),
                        "middle_original_k16_two_operands_pair_fraction":float(np.all(middle_narrow[pairs16],axis=1).mean()),
                        "factored_chunks":int(np.count_nonzero(factors!=1))}
        # An exact core+residual fallback: only A is narrowed; B stays full E2M1.
        core=np.clip(chunks,-3,3);residual=chunks-core
        residual_per_pair=np.count_nonzero(residual[pair_indices],axis=(1,2))
        repairs_failed=0;repairs_tested=min(512,len(pair_indices))
        for (i,j),b in zip(pair_indices[:repairs_tested],chunks[::-1][:repairs_tested]):
            r=encode_decode('shared',[core[i],core[j],b],512,512)
            corrections=[int(residual[i]@b),int(residual[j]@b)]
            got=[r['decoded'][lane]+corrections[lane] for lane in range(2)]
            repairs_failed+=got!=[int(chunks[i]@b),int(chunks[j]@b)]
        # Exponent bins each hold normalized values in q[-3,3].
        exponent=np.where(np.abs(chunks)>=8,2,np.where(np.abs(chunks)>=4,1,0))
        buckets_per_pair=[]
        for i,j in pair_indices:
            active={int(e) for row in (i,j) for e,v in zip(exponent[row],chunks[row]) if v!=0}
            buckets_per_pair.append(len(active))
        maxima=np.max(np.abs(q),axis=1)
        result.append({"distribution":label,"blocks32":blocks,"chunks4":len(chunks),
                       "full_block_max_q_histogram":{str(v):int(np.count_nonzero(maxima==v)) for v in np.unique(maxima)},
                       "normalization_admission":rows,
                       "residual_value_fraction":float(np.count_nonzero(residual)/residual.size),
                       "residual_scalar_products_per_two_outputs_mean":float(residual_per_pair.mean()),
                       "residual_count_histogram":{str(v):int(np.count_nonzero(residual_per_pair==v)) for v in range(9)},
                       "pair_fraction_requiring_at_most_one_scalar_product":float(np.mean(residual_per_pair<=1)),
                       "repair_model_cases":repairs_tested,"repair_failures":repairs_failed,
                       "nonempty_exponent_bins_per_pair_mean":float(np.mean(buckets_per_pair)),
                       "nonempty_exponent_bins_histogram":{str(v):int(np.count_nonzero(np.array(buckets_per_pair)==v)) for v in range(4)}})
        assert repairs_failed==0
    return {"quantizer":"OCP-style max scaling: scale=2**(floor(log2(maxabs))-2), nearest-even E2M1 with saturation.",
            "synthetic_only":True,"datasets":result,
            "cost_accounting":{
                "dense_one_sided_residual":"For two shared-B outputs, baseline is 2 dot products. Packed core costs 1; two dense residual dots add 2, total 3 versus 2 (1.5x arithmetic), before packing overhead.",
                "dense_two_sided_residual":"AB=AcBc+ArB+AcBr; packing the two core outputs gives 1 packed core plus 4 residual products versus 2 baselines, or 2.5x dense arithmetic if residuals use dense GEMM.",
                "sparse_residual":"Additional scalar products are counted, not equated to Tensor Core cycles. Indices, gathers, zeros, predication and scheduling are not timed.",
                "exponent_bins":"q=2**e*m with e=0/1/2 and |m|<=3. Up to three packed shared-B products versus two baseline products; only one active bin gives a raw arithmetic reduction.",
                "normalization":"Common powers of two or gcd factors are exact internal refactorizations. Gcd factors including 3 require a postscale multiplication; this is not evidence for free MX format conversion."}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--random-cases',type=int,default=2048);parser.add_argument('--blocks',type=int,default=2048)
    args=parser.parse_args();start=time.perf_counter();rng=np.random.default_rng(20261123)
    provenance=verify_upstream_provenance(NUMERICS/'upstream-snapshot')
    candidates=[
        ('asymmetric_ternary_by_full_e2m1','independent',1,12,512,128,4),
        ('shared_q3_by_full_e2m1','shared',3,12,512,512,4),
        ('shared_ternary_by_full_e2m1_longer_k','shared',1,12,1024,1024,32),
        ('middle_q3_compressed_k8','middle',3,3,512,512,8),
        ('middle_q3_compressed_k12','middle',3,3,512,512,12),
    ]
    output={"metadata":{"created_utc":datetime.now(timezone.utc).isoformat(),"upstream_commit":UPSTREAM_COMMIT,
                        "model_provenance":provenance,"model_sha256":hashlib.sha256((NUMERICS/'v100_model.py').read_bytes()).hexdigest(),
                        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"seed":20261123,
                        "gpu_execution":False,"performance_measured":False,"argv":sys.argv},
            "search":parameter_search(),
            "candidates":[run_candidate(*spec,args.random_cases,rng) for spec in candidates],
            "same_scale_mx32_middle_pipeline":block32_middle(args.random_cases,rng),
            "failure_examples":failure_examples(),
            "admission_and_repair":admission_and_repair(args.blocks,rng),
            "proof_scope":"Published V100 BFMA4 numerical model, sequential blocks of four; no theorem for arbitrary future hardware or every nonlinear encoding."}
    output['metadata']['elapsed_seconds']=time.perf_counter()-start
    (BASE/'round3_math_results.json').write_text(json.dumps(output,indent=2)+'\n')
    compact={"candidates":[{k:r[k] for k in ('label','total_cases','decoded_failures','packed_sum_not_exact_cases','maximum_observed_packed_sum_error')} for r in output['candidates']],
             "mx32_pipeline":output['same_scale_mx32_middle_pipeline'],"search_candidates":output['search']['candidates_checked'],
             "elapsed_seconds":output['metadata']['elapsed_seconds']}
    print(json.dumps(compact,indent=2))


if __name__=='__main__':main()
