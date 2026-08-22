#!/usr/bin/env python3
"""把逐张量误差换算成"整模型损伤"和"预期 PPL 变化", 给出配方选型表。

损伤 = Σ_类别 (该类张量数 x 该类加权相对误差中位数)
  加性是**近似**: 各子层的量化扰动在残差流上大致独立叠加。真正的检验是 PPL,
  这里只用来在配方之间排序、以及把"能省多少字节"换算成"值不值得测"。

预期 PPL% = 实测的 0.417%(IQ4_XS vs 我们的 Q5_K_M) x (损伤差 / 两者的损伤差)
  即假设 PPL 增量与损伤线性相关, 用唯一一个已实测的点做标定。
"""
import json, re, sys, statistics as st
from collections import defaultdict

CNT = {"ffn_down.weight":64,"ffn_gate.weight":64,"ffn_up.weight":64,"attn_qkv.weight":48,
       "attn_gate.weight":48,"ssm_out.weight":48,"attn_q.weight":16,"attn_output.weight":16,
       "attn_v.weight":16,"attn_k.weight":16,"ssm_alpha.weight":48,"ssm_beta.weight":48}
PARAM = {"ffn_down.weight":5704.3e6,"ffn_gate.weight":5704.3e6,"ffn_up.weight":5704.3e6,
         "attn_qkv.weight":2516.6e6,"attn_gate.weight":1509.9e6,"ssm_out.weight":1509.9e6,
         "attn_q.weight":1006.6e6,"attn_output.weight":503.3e6,"attn_v.weight":83.9e6,
         "attn_k.weight":83.9e6,"ssm_alpha.weight":11.8e6,"ssm_beta.weight":11.8e6}

def load(path):
    out = defaultdict(list)
    for r in json.load(open(path)):
        if r.get("rel") is None: continue
        out[re.sub(r"^blk\.\d+\.","",r["name"])].append(r["rel"])
    return {k: st.median(v) for k, v in out.items()}

def parse_probe(path):
    """requant_probe 的输出 -> {(类别, 类型): 中位误差}"""
    acc = defaultdict(list)
    for line in open(path):
        p = line.split()
        if len(p) < 3 or not p[0].startswith("blk."): continue
        cat = re.sub(r"^blk\.\d+\.","",p[0]); acc[(cat,p[1])].append(float(p[2]))
    return {k: st.median(v) for k, v in acc.items()}

def damage(cfg, err):
    return sum(CNT[c]*err[(c,t)] for c, t in cfg.items())

if __name__ == "__main__":
    iq = load(sys.argv[1])            # 我们的 IQ4_XS 实测(逐类别)
    q5 = load(sys.argv[2])            # 我们的 Q5_K_M 实测(逐类别)
    pr = parse_probe(sys.argv[3])     # 探针补的 (类别,类型) 点
    E = dict(pr)
    for c, v in iq.items(): E.setdefault((c,"IQ4_XS"), v)
    for c in CNT:
        E.setdefault((c,"Q8_0"), 0.0)  # Q8_0 就是量化源, 误差为 0(逐字节相同)
    D_iq = sum(CNT[c]*iq[c] for c in CNT if c in iq)
    D_q5 = sum(CNT[c]*q5[c] for c in CNT if c in q5)
    print(f"损伤: 我们的 IQ4_XS = {D_iq:.4f}   我们的 Q5_K_M = {D_q5:.4f}   差 = {D_iq-D_q5:.4f}")
    print(f"标定: 实测 ΔPPL 0.417%  ->  每单位损伤 {0.417/(D_iq-D_q5):.4f}%\n")
    k = 0.417/(D_iq-D_q5)

    base = {c: "IQ4_XS" for c in CNT}
    base["attn_qkv.weight"] = "Q5_K"; base["attn_v.weight"] = "Q5_K"   # llama.cpp 自动升档
    # 修正版: attn_k/attn_v 用 Q8_0 而不是 Q6_K —— 实测 Q6_K 在部分层反而比 Q5_K 差
    frugal = dict(base, **{"ssm_alpha.weight":"Q8_0","ssm_beta.weight":"Q8_0",
                           "attn_k.weight":"Q8_0","attn_v.weight":"Q8_0","attn_output.weight":"Q6_K"})
    fplus  = dict(frugal, **{"ssm_out.weight":"Q6_K"})
    cfgs = [("A 现状(已上生产)", base, 15311.11),
            ("C frugal", frugal, 15534.31),
            ("C2 frugal-plus", fplus, 15950.56),
            ("D 只把 attn_qkv 压回", dict(base, **{"attn_qkv.weight":"IQ4_XS"}), 14936.11),
            ("E frugal + qkv 压回", dict(frugal, **{"attn_qkv.weight":"IQ4_XS"}), 15159.31),
            ("F frugal-plus + qkv 压回", dict(fplus, **{"attn_qkv.weight":"IQ4_XS"}), 15575.56)]
    print(f"{'方案':<26}{'体积MiB':>10}{'ΔMiB':>8}{'损伤':>9}{'Δ损伤%':>9}{'预期ΔPPL%':>11}")
    D0 = damage(base, E)
    for nm, cfg, size in cfgs:
        try: d = damage(cfg, E)
        except KeyError as e: print(f"{nm:<26} 缺数据 {e}"); continue
        print(f"{nm:<26}{size:>10.1f}{size-15311.11:>+8.1f}{d:>9.4f}{100*(d-D0)/D0:>+8.1f}%{k*(d-D0):>+10.3f}%")
    print("\n预期 ΔPPL% 相对现状 A; 负号 = 更好。")
