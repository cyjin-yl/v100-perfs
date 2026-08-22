#!/usr/bin/env python3
"""用 imatrix 加权, 逐张量测量"量化后相对于 Q8_0 源的信号损失"。

指标 rel = Σ_ij m_j (W_ij - W'_ij)^2 / Σ_ij m_j W_ij^2
   m_j = imatrix 的 in_sum2[j]/counts, 即第 j 个输入通道的平均激活能量。
在"输入通道互不相关"的假设下, rel 正比于该张量输出被量化扰动的相对能量,
这正是 llama.cpp 的 imatrix 量化所最小化的目标。
"""
import sys, re, json
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

def load_imatrix(path):
    r = GGUFReader(path, "r")
    m = {}
    counts = {}
    for t in r.tensors:
        if t.name.endswith(".in_sum2"):
            m[t.name[:-len(".in_sum2")]] = np.array(t.data, dtype=np.float64)
        elif t.name.endswith(".counts"):
            counts[t.name[:-len(".counts")]] = float(np.array(t.data).reshape(-1)[0])
    out = {}
    for k, v in m.items():
        c = counts.get(k, 1.0) or 1.0
        out[k] = v / c
    return out

def tensors_by_name(path):
    r = GGUFReader(path, "r")
    return r, {t.name: t for t in r.tensors}

def deq(t):
    return dequantize(t.data, t.tensor_type).astype(np.float32)

def main():
    src_path, cmp_path, imat_path, layer_csv, out_json = sys.argv[1:6]
    layers = set(int(x) for x in layer_csv.split(",")) if layer_csv != "all" else None
    imat = load_imatrix(imat_path)
    rs, src = tensors_by_name(src_path)
    rc, cmp_ = tensors_by_name(cmp_path)

    rows = []
    for name, ts in src.items():
        if not name.endswith(".weight"):
            continue
        mm = re.match(r"blk\.(\d+)\.", name)
        if layers is not None:
            if mm is None or int(mm.group(1)) not in layers:
                continue
        if name not in imat:
            continue
        tc = cmp_.get(name)
        if tc is None:
            continue
        if tc.tensor_type == ts.tensor_type:
            rows.append(dict(name=name, src=ts.tensor_type.name, dst=tc.tensor_type.name,
                             rel=0.0, rel_unw=0.0, note="未重新量化(逐字节相同)"))
            continue
        ne0 = int(ts.shape[0]); ne1 = int(ts.shape[1]) if len(ts.shape) > 1 else 1
        m = imat[name]
        if m.shape[0] != ne0:
            rows.append(dict(name=name, src=ts.tensor_type.name, dst=tc.tensor_type.name,
                             rel=None, note=f"imatrix 长度 {m.shape[0]} != ne0 {ne0}"))
            continue
        a = deq(ts).reshape(ne1, ne0)
        b = deq(tc).reshape(ne1, ne0)
        # 分块累加, 避免为 89M 参数的 ffn 张量一次性开 float64 中间数组(单个就 700MB)
        num = den = num_u = den_u = 0.0
        STEP = 1024
        for r0 in range(0, ne1, STEP):
            ar = a[r0:r0+STEP].astype(np.float64)
            dr = ar - b[r0:r0+STEP].astype(np.float64)
            dr *= dr
            ar *= ar
            num += float((dr @ m).sum()); den += float((ar @ m).sum())
            num_u += float(dr.sum()); den_u += float(ar.sum())
        rows.append(dict(name=name, src=ts.tensor_type.name, dst=tc.tensor_type.name,
                         rel=num/den, rel_unw=num_u/den_u, nelem=ne0*ne1))
        del a, b
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"完成 {len(rows)} 个张量 -> {out_json}")

main()
