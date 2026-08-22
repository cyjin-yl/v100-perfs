#!/usr/bin/env python3
"""配对 PPL 检验的功效分析: 要多少块才能把某个大小的差异从噪声里分出来。

输入用已经实测到的一组配对结果(块数 n0 与配对 SE), 反推逐块标准差
    sd = SE * sqrt(n0)
再按标准的配对 t 检验样本量公式(正态近似)
    n = (z_{1-a/2} + z_{1-b})^2 * sd^2 / delta^2
算出目标效应量 delta 所需的块数。

块数换算成语料: 每块 n_ctx 个 token, 按实测语料的字节/块比例折算字节。
"""
import math, sys

Z = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449}
Z_ALPHA = 1.9600  # 双侧 0.05

def main():
    n0, se0 = 60, 0.187          # 实测: IQ4_XS vs 我们的 Q5_K_M, 60 块, 配对 SE 0.187%
    obs = 0.417                  # 实测差异 %
    corpus_bytes, n_ctx = 439981, 512
    sd = se0 * math.sqrt(n0)
    bpb = corpus_bytes / n0      # 每块字节
    print(f"实测基线: n={n0} 块, 配对SE={se0}% -> 逐块标准差 sd={sd:.3f}%")
    print(f"          实测差异 {obs}% (t={obs/se0:.2f})")
    print(f"          留出集 {corpus_bytes/1e6:.2f} MB / {n0} 块 = {bpb:.0f} 字节每块, {n_ctx} token 每块\n")
    print(f"{'目标效应 δ%':>12}{'80%功效块数':>13}{'90%功效块数':>13}{'80%所需语料':>14}{'GPU分钟x2组':>13}")
    for d in (0.417, 0.30, 0.20, 0.15, 0.104, 0.05):
        rows = []
        for p in (0.80, 0.90):
            n = (Z_ALPHA + Z[p])**2 * sd**2 / d**2
            rows.append(math.ceil(n))
        mb = rows[0]*bpb/1e6
        # imatrix 实测速度: 584 块 / 35 分钟 = 16.7 块每分钟(同 n_ctx=512, 同卡)
        mins = rows[0]/16.7*2
        print(f"{d:>12.3f}{rows[0]:>13}{rows[1]:>13}{mb:>12.2f}MB{mins:>12.0f}")
    print("\n注: 正态近似, 未做 t 分布自由度修正(n>30 时差别可忽略)。")
    print("    假设扩样后的逐块方差与现有留出集同量级 —— 换语料分布会改变 sd。")

main()
