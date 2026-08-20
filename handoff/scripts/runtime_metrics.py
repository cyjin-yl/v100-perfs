#!/usr/bin/env python3
"""从后端日志采样运行时指标, 用于两个量化档位的同口径对比。

为什么需要它:
  切换量化档(Q5_K_M -> IQ4_XS)之后要判断"值不值", 光看 PPL 不够 —— PPL 衡量的是
  "预测下一个 token 的平均难度", 不回答"实际服务变好还是变差"。真正要看的是:
    decode tok/s        速度有没有掉(IQ 量化在 SM70 上是查表密集的, 可能更慢)
    MTP 接受率          draft 质量; 掉了 MTP 就白开
    kv_pool 占比        省下的显存有没有真的变成可用页
    L1trie / hits       前缀缓存有没有真的活过来
    vram other=         有没有对不上账的显存
  这些散在日志各处, 肉眼比两个档位不可靠, 所以做成同口径采样。

用法:
  python runtime_metrics.py --log <backend.log> --since-restart   # 只统计最近一次启动之后
  python runtime_metrics.py --log a.log --compare b.log           # 两档对比
"""

import os

# ---------------------------------------------------------------------------
# 绕过代理访问本机。
# 这台机器的 ~/.zshrc 里 export 了 http_proxy/https_proxy=127.0.0.1:10808
# (给 HuggingFace 上传用)。Python 的 urllib **不会**自动把 localhost 排除在
# 代理之外 —— 于是打 127.0.0.1:8000 的请求会被塞进那个代理转一圈:
# 请求送得到(后端确实会处理), 但响应卡在代理回程上, 表现为"探针挂住不返回",
# 而后端日志里明明写着 prefill 6.29s 就完成了。
# 更阴的是它会污染**时延测量** —— 任何本地 HTTP 基准都可能悄悄多算一段代理开销。
for _v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_v, None)
os.environ["no_proxy"] = "127.0.0.1,localhost"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
# ---------------------------------------------------------------------------

import argparse
import re
import statistics
import sys

RE_METRICS = re.compile(
    r"running=(?P<run>\d+) pending=(?P<pend>\d+).*?"
    r"prefill (?P<pf>\d+) tok \((?P<pfs>[\d.]+) tok/s\).*?"
    r"decode (?P<dec>\d+) tok \((?P<decs>[\d.]+) tok/s\).*?"
    r"done (?P<done>\d+) req.*?"
    r"kv_pool=(?P<used>\d+)/(?P<total>\d+) pg \((?P<pct>\d+)%\).*?"
    r"L1trie=(?P<trie>\d+) pg")
RE_HITS = re.compile(r"hits=(\d+)")
RE_VRAM = re.compile(r"vram=(\d+)/(\d+)MB\(pool=(\d+) .*?other=(-?\d+)\)")
RE_ACCEPT = re.compile(r"pos_accept_rate=\[([\d.]+)%, ([\d.]+)%\]")
RE_READY = re.compile(r"\[Server\] ready, listening")
RE_ABORT = re.compile(r"batch forward failed|丢弃已断开的排队请求|disk byte limit exceeded")


def collect(path, since_restart):
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
    except OSError as e:
        print(f"读不了 {path}: {e}", file=sys.stderr)
        return None
    if since_restart:
        last = max((i for i, l in enumerate(lines) if RE_READY.search(l)),
                   default=None)
        if last is None:
            print(f"  警告: {path} 里找不到 '[Server] ready', 统计全文件")
        else:
            lines = lines[last:]

    d = {"decode_tps": [], "prefill_tps": [], "pool_pct": [], "trie": [],
         "hits": [], "accept0": [], "accept1": [], "other": [], "anomalies": 0,
         "samples": 0}
    for l in lines:
        m = RE_METRICS.search(l)
        if m:
            d["samples"] += 1
            # 0 tok/s 的样本是空闲期, 计入会把均值拉垮 —— 只统计真的在产出的窗口
            if float(m["decs"]) > 0:
                d["decode_tps"].append(float(m["decs"]))
            if float(m["pfs"]) > 0:
                d["prefill_tps"].append(float(m["pfs"]))
            d["pool_pct"].append(int(m["pct"]))
            d["trie"].append(int(m["trie"]))
            h = RE_HITS.search(l)
            if h:
                d["hits"].append(int(h.group(1)))
            v = RE_VRAM.search(l)
            if v:
                d["other"].append(int(v.group(4)))
        a = RE_ACCEPT.search(l)
        if a:
            d["accept0"].append(float(a.group(1)))
            d["accept1"].append(float(a.group(2)))
        if RE_ABORT.search(l):
            d["anomalies"] += 1
    return d


def fmt(d, label):
    def stat(v, unit="", pct=False):
        if not v:
            return "  无样本"
        med = statistics.median(v)
        return (f"{med:.1f}{unit} (中位, n={len(v)}, "
                f"max={max(v):.1f}{unit})")
    print(f"── {label} ──")
    print(f"  metrics 样本数    : {d['samples']}")
    print(f"  decode tok/s      : {stat(d['decode_tps'])}")
    print(f"  prefill tok/s     : {stat(d['prefill_tps'])}")
    print(f"  MTP 接受率 pos0   : {stat(d['accept0'], '%')}")
    print(f"  MTP 接受率 pos1   : {stat(d['accept1'], '%')}")
    print(f"  kv_pool 占用      : {stat(d['pool_pct'], '%')}")
    print(f"  L1trie 页数       : {stat(d['trie'])}")
    print(f"  前缀缓存 hits     : {max(d['hits']) if d['hits'] else '无'}"
          f"  (取最大值; 恒为 0 说明缓存没生效)")
    print(f"  vram other=(无主) : {stat(d['other'], 'MB')}")
    print(f"  异常事件          : {d['anomalies']}"
          f"  (batch forward failed / 断连丢弃 / 磁盘配额超限)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--compare", default=None, help="第二份日志, 做同口径对比")
    ap.add_argument("--since-restart", action="store_true", default=True)
    ap.add_argument("--whole-file", dest="since_restart", action="store_false")
    args = ap.parse_args()

    a = collect(args.log, args.since_restart)
    if a is None:
        return 2
    fmt(a, args.log.split("/")[-1])
    if args.compare:
        b = collect(args.compare, args.since_restart)
        if b:
            print()
            fmt(b, args.compare.split("/")[-1])
            print()
            print("── 判读提示 ──")
            print("  decode tok/s 若 IQ4_XS 反而更慢, 那是**没走到快路径**, 要查不要回退 ——")
            print("    27B decode 是带宽瓶颈, IQ4_XS 每 token 少读约 26% 字节, 理论上更快;")
            print("    SM70 走 DP4A MMQ 直接在量化权重上做整数点积, 不做完整反量化。")
            print("    查: FASTLLM_CUDA_SM70_IQ4XS_MMQ 是否生效 / tile 配置 / CUDA graph 兼容性 / MTP 是否也走 MMQ;")
            print("  MTP 接受率掉了说明 draft 与主模型不匹配, MTP 收益会大打折扣;")
            print("  L1trie 恒为 0 + hits 恒为 0 = 前缀缓存整层没生效, 优先查这个。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
