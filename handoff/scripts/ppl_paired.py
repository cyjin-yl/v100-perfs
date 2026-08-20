#!/usr/bin/env python3
"""对 llama-perplexity 的多组结果做**配对**比较, 而不是比两个独立置信区间。

为什么要这么做:
  llama-perplexity 只报 `PPL = 2.6989 +/- 0.0471` 这种边际置信区间。拿两组的
  区间去比是**错的**, 会严重低估判别力 —— 因为两组评的是**同一份文本的同一批
  分块**, 误差高度相关。文本本身难不难(某块是密集代码、某块是自然语言)对两个
  模型是共同的, 这部分方差在配对后会被完全抵消。

  实例: A=2.6989±0.0471, B=2.6877±0.0470, 差值 0.0112 远小于任一区间宽度,
  按独立区间判读会得出"完全分不出来"。但那 ±0.047 里绝大部分是"分块难度"的
  方差, 不是"模型差异"的方差。配对之后往往能把不确定度压小一个量级, 从而
  真正回答"降位宽到底代价多大"。

原理:
  llama.cpp 打印的 `[n]v` 是**前 n 块的累计 PPL**, 即 v = exp(S_n / n),
  其中 S_n 是前 n 块平均负对数似然之和。于是可以逐块还原:
      nll_n = n*ln(PPL_n) - (n-1)*ln(PPL_{n-1})
  拿到逐块 nll 后, 两组的配对差 d_i = nll_i(A) - nll_i(B) 就是每块上的
  对数似然差。exp(mean(d)) 即 PPL 比值, 其标准误按配对样本算。

  注意这是"块级"配对而不是"token 级"配对 —— 块内 token 数固定(n_ctx=512)
  所以块级平均是无偏的; 只是自由度只有块数, 这一点在报告里如实标注。

用法:
  python ppl_paired.py /path/to/gpu-verify.log --labels A:IQ4_XS+imat B:Q5_K_M+imat C:官方Q5_K_M
"""

import argparse
import math
import re
import sys

TOK = re.compile(r"\[(\d+)\]\s*([0-9]+\.[0-9]+)")


def split_runs(text):
    """把日志里连续追加的多次 perplexity 运行切开。

    判据是分块序号回到 1 —— 每次运行都从 [1] 开始重新计数。
    """
    runs, cur, last = [], [], 0
    for m in TOK.finditer(text):
        n, v = int(m.group(1)), float(m.group(2))
        if n <= last and cur:
            runs.append(cur)
            cur = []
        cur.append((n, v))
        last = n
    if cur:
        runs.append(cur)
    return runs


def per_chunk_nll(run):
    """累计 PPL 序列 -> 逐块 nll。"""
    out, prev_sum = [], 0.0
    for n, v in run:
        s = n * math.log(v)
        out.append(s - prev_sum)
        prev_sum = s
    return out


def paired(a, b):
    n = min(len(a), len(b))
    d = [a[i] - b[i] for i in range(n)]
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n > 1 else 0.0
    t = mean / se if se > 0 else float("inf")
    worse = sum(1 for x in d if x > 0)
    return n, mean, se, t, worse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--labels", nargs="*", default=[],
                    help="按运行顺序给每组起名, 如 A:IQ4_XS B:Q5_K_M")
    args = ap.parse_args()

    text = open(args.log, encoding="utf-8", errors="replace").read()
    runs = split_runs(text)
    if not runs:
        print("日志里没找到分块 PPL 序列", file=sys.stderr)
        return 2

    names = list(args.labels) + [f"run{i+1}" for i in range(len(runs))]
    names = names[:len(runs)]
    nlls = [per_chunk_nll(r) for r in runs]

    print(f"== 配对困惑度比较 ({len(runs)} 组) ==\n")
    print(f"  {'组':<22}{'块数':>5}{'PPL':>10}")
    for nm, r, x in zip(names, runs, nlls):
        ppl = math.exp(sum(x) / len(x))
        print(f"  {nm:<22}{len(r):>5}{ppl:>10.4f}")

    print(f"\n  {'比较':<30}{'ΔPPL%':>9}{'配对SE%':>10}{'t':>8}{'更差块数':>10}")
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            n, mean, se, t, worse = paired(nlls[i], nlls[j])
            # exp(mean)-1 就是 PPL 的相对变化; se 同阶做一阶传播
            print(f"  {names[i]+' vs '+names[j]:<30}"
                  f"{(math.exp(mean)-1)*100:>+9.3f}"
                  f"{se*100:>10.3f}{t:>8.2f}{worse:>6}/{n}")

    print("\n  读法:")
    print("    ΔPPL% > 0 表示前者更差(困惑度更高)。")
    print("    |t| < 2 大致可视为在块级上分不出差异(双侧 p>0.05, 自由度=块数-1)。")
    print("    '更差块数' 是符号检验的直观版: 接近半数说明差异没有一致方向。")
    print("    局限: 配对单位是 512-token 的块, 不是 token; 自由度只有块数。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
