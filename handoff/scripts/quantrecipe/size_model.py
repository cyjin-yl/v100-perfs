#!/usr/bin/env python3
"""按张量类别统计参数量, 并计算"若把某类升档"的体积代价。

读取 dump_tensor_types.py 产出的 TSV。
"""
import sys, re
from collections import defaultdict

# ggml 每种类型的 (block 元素数, block 字节数)
BLK = {
    "F32":    (1, 4),
    "F16":    (1, 2),
    "Q4_K":   (256, 144),
    "Q5_K":   (256, 176),
    "Q6_K":   (256, 210),
    "Q8_0":   (32, 34),
    "IQ4_XS": (256, 136),
    "IQ4_NL": (32, 18),
}

def bytes_for(nelem, typ):
    n, b = BLK[typ]
    return nelem // n * b

def cat(name):
    m = re.sub(r"^blk\.\d+\.", "", name)
    return m

def load(path):
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            name, typ, shape, ne, nb = line.rstrip("\n").split("\t")
            rows.append((name, typ, shape, int(ne), int(nb)))
    return rows

def report(path, label):
    rows = load(path)
    agg = defaultdict(lambda: [0, 0, 0, set()])  # cat -> [count, nelem, nbytes, types]
    for name, typ, shape, ne, nb in rows:
        a = agg[cat(name)]
        a[0] += 1; a[1] += ne; a[2] += nb; a[3].add(typ)
    tot_e = sum(a[1] for a in agg.values())
    tot_b = sum(a[2] for a in agg.values())
    print(f"### {label}  总张量={len(rows)}  总参数={tot_e/1e9:.3f}B  总字节={tot_b/2**30:.3f} GiB")
    print(f"{'类别':<34}{'个数':>5}{'参数量':>12}{'占比':>8}{'字节GiB':>10}{'体积占比':>9}  类型")
    for k in sorted(agg, key=lambda x: -agg[x][2]):
        c, ne, nb, ts = agg[k]
        print(f"{k:<34}{c:>5}{ne/1e6:>11.1f}M{100*ne/tot_e:>7.2f}%{nb/2**30:>10.3f}{100*nb/tot_b:>8.2f}%  {'/'.join(sorted(ts))}")
    print()
    return agg, tot_b

if __name__ == "__main__":
    for p, l in zip(sys.argv[1::2], sys.argv[2::2]):
        report(p, l)
