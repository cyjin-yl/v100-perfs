#!/usr/bin/env python3
"""从 omp 会话日志里测"逐字抄写破坏率"(真实负载, 零 GPU 成本)。

背景:
  agent 反复要把上下文里的字面量原样抄进命令——仓库路径、包名、commit hash。
  抄歪一个字符命令就失败, 而且更糟的是: agent 之后**读回自己写坏的内容**,
  发现前后矛盾, 会得出"我正被 prompt injection 攻击"的结论并拒绝干活。
  生产上真实出现过 /home/ezra/Documents/Proto-UI -> /home/eze/Documents/PotouI。

为什么用会话日志而不是造压测:
  后端 batch=1 且长期排着几十个请求, 想独占 GPU 做 A/B 很贵。而 agent 每天
  产出几万次这些字面量, 本身就是一个持续运行的、真实分布的探针。

用法:
  # 逐日
  python corruption_rate.py --canon /home/ezra/Documents/Proto-UI --canon Proto-UI
  # 修复前后对比(按小时, 只看某段时间)
  python corruption_rate.py --by hour --since 2026-08-19T00
"""

import argparse
import glob
import os
import re
import sys
from collections import Counter

DEFAULT_SESSION_GLOB = "/home/ezra/.omp/agent/sessions/*/*.jsonl"
DEFAULT_CANON = ["/home/ezra/Documents/Proto-UI", "Proto-UI"]

TS_RE = re.compile(r'"timestamp":"(\d{4}-\d\d-\d\d)T(\d\d)')


def build_candidate_re(canon):
    """只在"长得像目标字面量"的片段上做距离计算, 否则 400MB 的日志扫不动。"""
    pats = []
    for c in canon:
        if c.startswith("/"):
            head = re.escape(c.split("/")[1][:2]) if len(c.split("/")) > 1 else ""
            pats.append(r'/[A-Za-z0-9_./-]{2,80}')
        else:
            # 首字母 + 长度相近的词
            pats.append(r'(?<![A-Za-z])' + re.escape(c[0]) +
                        r'[A-Za-z0-9_-]{%d,%d}(?![A-Za-z])' %
                        (max(1, len(c) - 4), len(c) + 4))
    return re.compile("|".join(pats))


# 误报来源(实测): 编辑距离阈值 4 会把这些合法串算成"破坏" ——
#   /Proto-UI(只是带了前导斜杠)、Prototype、Proposed、Protocol、Protobuf…
# 它们把真实破坏率 ~0.8% 抬成了 6.31%。这里做两层过滤:
#   1) 显式排除已知的合法英文词(它们恰好落在距离阈值内)
#   2) 距离阈值收紧, 并要求长度接近 —— 真实破坏是"抄歪一两个字符",
#      不是换成另一个词
LEGIT_WORDS = {
    "prototype", "prototypes", "prototyping", "proposed", "proposal",
    "protocol", "protocols", "protobuf", "protect", "protected",
    "promote", "promoted", "proto", "photo", "photos",
}


def is_legit_variant(tok, canon_lower):
    """判断这个 token 是不是合法写法而非破坏。"""
    bare = tok.strip("/.,\"'`)]}").lower()
    if not bare:
        return True
    if bare in LEGIT_WORDS:
        return True
    # 只差前后标点/斜杠 -> 合法
    for c in canon_lower:
        if bare == c.strip("/").lower():
            return True
        # 路径: 去掉前导斜杠后相同
        if c.startswith("/") and bare == c.lower().lstrip("/"):
            return True
    return False


def lev(a, b, cap):
    """带上界的编辑距离; 超过 cap 直接返回 cap+1, 避免在无关串上做满矩阵。"""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        best = i
        for j, cb in enumerate(b, 1):
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            cur.append(v)
            best = min(best, v)
        if best > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=DEFAULT_SESSION_GLOB)
    ap.add_argument("--canon", action="append", default=None,
                    help="正确写法, 可给多次; 不给则用 Proto-UI 那组")
    ap.add_argument("--by", choices=["day", "hour"], default="day")
    ap.add_argument("--since", default=None,
                    help="只统计不早于该前缀的时间, 如 2026-08-19 或 2026-08-19T00")
    ap.add_argument("--max-dist", type=int, default=2,
                    help="编辑距离上限; 默认 2(实测 4 会把 Prototype/Proposed 这类合法词算成破坏)")
    ap.add_argument("--newest", type=int, default=2,
                    help="只扫最近改动的 N 个会话文件")
    ap.add_argument("--samples", type=int, default=25)
    args = ap.parse_args()

    canon = args.canon or DEFAULT_CANON
    cand_re = build_candidate_re(canon)
    canon_lower = [c.lower() for c in canon]

    files = sorted(glob.glob(args.glob), key=os.path.getmtime,
                   reverse=True)[:args.newest]
    if not files:
        print(f"没找到会话文件: {args.glob}", file=sys.stderr)
        return 2

    good, bad, samples = Counter(), Counter(), Counter()
    for path in files:
        print(f"扫描 {os.path.basename(path)} "
              f"({os.path.getsize(path)/1e6:.0f}MB)", flush=True)
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = TS_RE.search(line)
                if not m:
                    continue
                key = m.group(1) if args.by == "day" else f"{m.group(1)}T{m.group(2)}"
                if args.since and key < args.since:
                    continue
                for tok in cand_re.findall(line):
                    if not tok:
                        continue
                    tok = tok.rstrip('/.,":\\\'`)]}')
                    low = tok.lower()
                    # 纯大小写变体算正确写法(PROTO-UI 之类是合法排版, 不是破坏)
                    if low in canon_lower or any(
                            low.startswith(c) for c in canon_lower if c.startswith("/")):
                        good[key] += 1
                        continue
                    if is_legit_variant(tok, canon_lower):
                        good[key] += 1
                        continue
                    for c in canon:
                        d = lev(tok, c, args.max_dist)
                        # 真实破坏是"抄歪一两个字符", 所以同时要求长度接近;
                        # 距离大而长度也差很多的, 基本都是另一个词
                        if 1 <= d <= args.max_dist and \
                                abs(len(tok) - len(c)) <= args.max_dist:
                            bad[key] += 1
                            samples[tok] += 1
                            break

    keys = sorted(set(good) | set(bad))
    if not keys:
        print("该时间范围内没有可统计的字面量")
        return 0
    print(f"\n{'时间':<14}{'正确':>9}{'破坏':>7}{'破坏率':>10}")
    tot_g = tot_b = 0
    for k in keys:
        g, b = good.get(k, 0), bad.get(k, 0)
        t = g + b
        if not t:
            continue
        tot_g += g
        tot_b += b
        mark = "  <<<" if b / t > 0.005 else ""
        print(f"{k:<14}{g:>9}{b:>7}{b/t:>9.2%}{mark}")
    tot = tot_g + tot_b
    print(f"{'合计':<14}{tot_g:>9}{tot_b:>7}"
          f"{(tot_b/tot if tot else 0):>9.2%}")
    if samples:
        print("\n破坏变体:")
        for s, n in samples.most_common(args.samples):
            print(f"  {n:>4}  {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
