#!/usr/bin/env python3
"""把扫描矩阵的逐档 JSON 汇总成两张 markdown 对比表(解限版 Cyber / 普通版)。

用法: python make_tables.py [--out MATRIX.md]
读取 projects/EzraVastLLM/reports/<tag>.json(sweep_one.sh 的汇总产物)。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

REPORTS = "/run/media/ezra/13D010B6FDBC1A06/projects/EzraVastLLM/reports"

# tag 形如 q38-n5-n-q4kxl-turbo4-mtp2-cachetuned-sm70
TAG_RE = re.compile(
    r"q38-(?P<id>[nc]\d+)-(?P<model>[nc]-[a-z0-9_]+)-(?P<kv>turbo4|turbo3|fp8_e4m3)"
    r"-mtp(?P<mtp>\d)(?P<cache>-cachetuned)?(?P<sm70>-sm70)?$")

MODEL_LABEL = {
    "n-q4kxl": "UD-Q4_K_XL", "n-q5km": "Q5_K_M", "n-q6k": "Q6_K",
    "n-awq": "W4A16-AWQ",
    "c-q4km": "Cyber Q4_K_M+mtp", "c-q5km": "Cyber Q5_K_M+mtp",
    "c-q6k": "Cyber Q6_K+mtp", "c-awq": "Cyber W4A16-AWQ",
}


def bench_pick(rows, target, field):
    """取某个 prompt 规模下 pass1/pass2 的引擎侧指标。"""
    for r in rows or []:
        if r.get("target_tokens") == target and r.get("pass") == 1:
            return r.get(field)
    return None


def fmt(v, suffix="", nd=1):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}{suffix}"
    return f"{v}{suffix}"


def row_of(rep):
    tag = rep.get("tag", "")
    m = TAG_RE.match(tag)
    meta = m.groupdict() if m else {}
    rows = rep.get("bench") or []
    acc = rep.get("mtp_accept_rate_last")
    acc_s = f"{acc[0]}/{acc[1]}%" if isinstance(acc, (list, tuple)) and len(acc) == 2 else "-"
    vram = rep.get("vram_peak_mib")
    return {
        "id": meta.get("id", tag[:8]),
        "model": MODEL_LABEL.get(meta.get("model", ""), meta.get("model", "?")),
        "kv": meta.get("kv", "?"),
        "mtp": meta.get("mtp", "?"),
        "cache": "调优" if meta.get("cache") else "默认",
        "sm70": "开" if meta.get("sm70") else "关",
        "cold": fmt(rep.get("cold_start_s"), "s", 0),
        "vram": fmt(vram / 1024 if vram else None, "G", 1),
        "mtp_on": "是" if rep.get("mtp_enabled") else "否",
        "acc": acc_s,
        "hit": fmt(rep.get("prefix_hit_ratio"), "", 3),
        "pf32k": fmt(bench_pick(rows, 32000, "engine_prefill_tok_s"), "", 0),
        "pf128k": fmt(bench_pick(rows, 131072, "engine_prefill_tok_s"), "", 0),
        "dec": fmt(bench_pick(rows, 8000, "engine_decode_tok_s"), "", 1),
        "e2e32k": fmt(bench_pick(rows, 32000, "engine_e2e_tok_s"), "", 0),
        "tool": f"{rep.get('matrix_fail')}/{rep.get('matrix_total')}",
        "garble": f"{rep.get('garble_fail')}/{rep.get('garble_total')}",
        "loop": f"{rep.get('toolloop_fail')}/{rep.get('toolloop_total')}",
        "conc": f"{rep.get('conc_fail')}/{rep.get('conc_total')}",
    }


HEAD = ("| 档 | 模型 | KV | MTP | 缓存 | SM70 | 冷启 | 显存峰值 | MTP生效 | 接受率 | "
        "前缀命中 | prefill@32K | prefill@128K | decode | e2e@32K | 工具失败 | "
        "乱码失败 | 多轮失败 | 并发失败 |")
SEP = "|" + "---|" * 19


def table(rows):
    out = [HEAD, SEP]
    for r in rows:
        out.append("| " + " | ".join([
            r["id"], r["model"], r["kv"], r["mtp"], r["cache"], r["sm70"],
            r["cold"], r["vram"], r["mtp_on"], r["acc"], r["hit"],
            r["pf32k"], r["pf128k"], r["dec"], r["e2e32k"],
            r["tool"], r["garble"], r["loop"], r["conc"]]) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPORTS, "MATRIX.md"))
    args = ap.parse_args()
    reps = []
    for p in sorted(glob.glob(os.path.join(REPORTS, "*.json"))):
        if re.search(r"\.(matrix|garble|toolloop|bench|conc|conc10|longctx)\.json$", p):
            continue
        try:
            rep = json.load(open(p))
        except Exception:
            continue
        if rep.get("status") != "ok":
            continue
        reps.append(rep)
    cyber = [row_of(r) for r in reps if r.get("tag", "").startswith("q38-c")]
    normal = [row_of(r) for r in reps if r.get("tag", "").startswith("q38-n")]
    cyber.sort(key=lambda r: r["id"])
    normal.sort(key=lambda r: r["id"])
    body = [
        "# Qwen3.8-27B on V100 32G — 扫描矩阵结果",
        "",
        "口径: 全部走生产链路 nginx:80 → thinking_proxy:8000 → fastllm:8002; ",
        "prefill/decode/e2e 取后端日志里引擎自己打的 `[req N] done:` 拆分; ",
        "工具/乱码/多轮/并发列是 `失败数/总数`。",
        "",
        "## 解限版 (Cyber)",
        "",
        table(cyber) if cyber else "_(尚无数据)_",
        "",
        "## 普通版 (官方 Qwen3.8-27B)",
        "",
        table(normal) if normal else "_(尚无数据)_",
        "",
    ]
    text = "\n".join(body)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
