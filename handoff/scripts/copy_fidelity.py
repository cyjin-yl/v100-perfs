#!/usr/bin/env python3
"""逐字抄写保真度测试 (verbatim copy fidelity).

为什么是这个指标:
  agent 干活时最依赖的不是"文采", 而是把上下文里出现过的字面量——文件路径、
  函数名、commit hash、PR 号——**一个字符不差地**抄进下一条命令。抄歪一个字符,
  整条命令就失败。生产上观察到的正是这种破坏:
      /home/ezra/Documents/Proto-UI  ->  /home/eze/Documents/PotouI
      ezra -> eZrA / EZa / EzA          Documents -> Dotments
  形态是"随机大小写翻转 + 邻近 token 替换", 不是语义错误。

  这类破坏有两个嫌疑, 本脚本用来把它们分开:
    (a) MTP 投机解码验证/回滚有 bug —— 正确实现下投机解码与非投机**逐字节等价**,
        所以只要 MTP 开关会改变保真度, 就证明实现有问题, 与 draft 头准不准无关。
    (b) KV cache 量化(turbo3/turbo4/fp8) 在深上下文里丢失细粒度信息 ——
        表现为保真度随**上下文深度**单调下降, 而浅上下文正常。

  两者的判据不同:
    MTP 问题   -> 保真度差异与深度**弱相关**, 但与 MTP 开关强相关。
    KV 量化问题 -> 保真度随深度显著下降, 且与 MTP 开关无关。

用法:
  python copy_fidelity.py --depths 2000,16000,64000,120000 --repeats 3 \
      --tag q5-mtp2-turbo4 --json out.json
"""

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------- 针 (needles)
# 刻意覆盖生产上真实坏掉的那几类字面量。混合大小写是重点:
# 大小写是最细粒度的信息, 也是观察到最先丢的东西。
NEEDLE_TEMPLATES = [
    ("path",   "/home/{user}/Documents/{proj}/packages/{pkg}/src/index.ts"),
    ("path",   "/run/media/{user}/13D010B6FDBC1A06/{proj}/build-rw/apiserver"),
    ("ident",  "{Proj}CacheManagerV{n}::GrowPagedBuffer"),
    ("ident",  "handle{Proj}StreamAbort_{n}"),
    ("hash",   "commit {hex40}"),
    ("mixed",  "{USER}-{proj}-{hex8}.{ext}"),
    ("url",    "https://github.com/{Proj}/{Proj}/pull/{num}#issuecomment-{num2}"),
]

USERS = ["ezra", "haru", "cyjin", "doeswork"]
PROJS = ["Proto-UI", "VastLLM", "Zerm-Core", "Turbo-KV"]
PKGS = ["prototypes", "brutalist", "runtime-core", "adapter-web"]
EXTS = ["yaml", "jsonl", "gguf", "env"]

FILLER_WORDS = """
the prototype adapter emits a boundary event whenever the renderer commits a
layout pass while the scheduler drains its pending queue and the reconciler
walks the tree comparing prior nodes against the freshly produced fiber list
which the compositor then rasterizes into tiles that the presenter uploads to
the swap chain before signalling the frame fence so downstream consumers can
observe a coherent snapshot of the document without tearing or partial paint
""".split()


def build_needles(rng, count):
    out = []
    for i in range(count):
        kind, tmpl = NEEDLE_TEMPLATES[i % len(NEEDLE_TEMPLATES)]
        user = rng.choice(USERS)
        proj = rng.choice(PROJS)
        s = tmpl.format(
            user=user,
            USER=user.upper(),
            proj=proj,
            Proj=proj.replace("-", ""),
            pkg=rng.choice(PKGS),
            ext=rng.choice(EXTS),
            n=rng.randint(2, 9),
            num=rng.randint(100, 999),
            num2=rng.randint(1000000000, 9999999999),
            hex8="".join(rng.choice("0123456789abcdef") for _ in range(8)),
            hex40="".join(rng.choice("0123456789abcdef") for _ in range(40)),
        )
        out.append({"id": f"N{i:02d}", "kind": kind, "text": s})
    return out


def build_context(rng, needles, target_chars):
    """把针均匀埋进填充文本, 返回 (文本, 每根针的相对深度)."""
    parts, depths = [], {}
    per = max(1, target_chars // (len(needles) + 1))
    for idx, nd in enumerate(needles):
        chunk = []
        n = 0
        while n < per:
            w = rng.choice(FILLER_WORDS)
            chunk.append(w)
            n += len(w) + 1
            if len(chunk) % 22 == 21:
                chunk.append("\n")
        parts.append(" ".join(chunk))
        parts.append(
            f"\n\n[RECORD {nd['id']}] 归档条目 {nd['id']} 的值是: {nd['text']}\n\n")
        depths[nd["id"]] = None  # 稍后按字符位置回填
    text = "".join(parts)
    for nd in needles:
        pos = text.find(f"[RECORD {nd['id']}]")
        depths[nd["id"]] = round(pos / max(1, len(text)), 3)
    return text, depths


def ask(base_url, model, messages, max_tokens, timeout, api_key=None):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
        # 抄写任务不需要思考链; 关掉可以隔离解码本身, 也让测试跑得完
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key or 'x'}"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    msg = data.get("choices", [{}])[0].get("message", {}) or {}
    return (msg.get("content") or ""), elapsed, data.get("usage", {}) or {}


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(expected, reply, needle_id):
    """判定这根针是否被逐字抄对。

    返回 (是否逐字命中, 模型实际写的值, 编辑距离)。
    优先按约定的 `编号<TAB>值` 逐行解析——这样报告里的"实得"就是模型真正写的
    那一串, 而不是一个随意截取的窗口; 解析不到才退回全文窗口搜索。
    """
    if expected in reply:
        return True, expected, 0
    for line in reply.splitlines():
        stripped = line.strip().lstrip("-*` ").strip()
        if not stripped.startswith(needle_id):
            continue
        value = stripped[len(needle_id):].strip().strip(":：\t ").strip("`")
        if value:
            return value == expected, value, levenshtein(expected, value)
    # 退路: 等长窗口里找最接近的
    best, bestd = "", len(expected) + 1
    win = len(expected)
    for i in range(max(0, len(reply) - win + 1)):
        d = levenshtein(expected, reply[i:i + win])
        if d < bestd:
            bestd, best = d, reply[i:i + win]
            if d <= 1:
                break
    return False, best, bestd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get(
        "COPY_FIDELITY_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--model", default=os.environ.get(
        "COPY_FIDELITY_MODEL", "qwen3.8-fastllm"))
    ap.add_argument("--api-key", default=os.environ.get("COPY_FIDELITY_KEY"))
    ap.add_argument("--depths", default="2000,16000,64000",
                    help="上下文 token 数, 逗号分隔 (按 ~3.6 字符/token 估算)")
    ap.add_argument("--needles", type=int, default=7)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--timeout", type=float, default=3000.0)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    results = []
    print(f"[copy-fidelity] tag={args.tag} model={args.model} "
          f"depths={depths} needles={args.needles} repeats={args.repeats}",
          flush=True)

    for depth_tokens in depths:
        for rep in range(args.repeats):
            rng = random.Random(args.seed + depth_tokens * 131 + rep)
            needles = build_needles(rng, args.needles)
            # 粗估 3.6 字符/token; 精确值不重要, 深度只需可比
            ctx, depth_map = build_context(rng, needles,
                                           int(depth_tokens * 3.6))
            asked = needles  # 全部都问, 一次问完省一遍 prefill
            ask_list = "\n".join(f"- {n['id']}" for n in asked)
            prompt = (
                "下面是一份归档。请**逐字**复述指定条目的值, 一个字符都不要改动、"
                "不要补全、不要修正你认为的拼写错误。\n\n"
                f"=== 归档开始 ===\n{ctx}\n=== 归档结束 ===\n\n"
                f"请复述以下条目, 每行一条, 格式严格为 `编号<TAB>值`:\n{ask_list}\n"
            )
            try:
                reply, elapsed, usage = ask(
                    args.base_url, args.model,
                    [{"role": "user", "content": prompt}],
                    max_tokens=1200, timeout=args.timeout,
                    api_key=args.api_key)
            except (urllib.error.URLError, urllib.error.HTTPError,
                    TimeoutError, OSError) as e:
                print(f"  depth={depth_tokens} rep={rep} 请求失败: {e}",
                      flush=True)
                results.append({"depth": depth_tokens, "rep": rep,
                                "error": str(e)})
                continue

            exact = 0
            details = []
            for nd in asked:
                ok, got, dist = score(nd["text"], reply, nd["id"])
                exact += 1 if ok else 0
                details.append({
                    "id": nd["id"], "kind": nd["kind"],
                    "rel_depth": depth_map[nd["id"]],
                    "exact": ok, "edit_distance": dist,
                    "expected": nd["text"],
                    "got": None if ok else got,
                })
            rate = exact / len(asked)
            bad = [d for d in details if not d["exact"]]
            print(f"  depth={depth_tokens:>7} rep={rep} "
                  f"逐字命中 {exact}/{len(asked)} ({rate:.0%}) "
                  f"wall={elapsed:.1f}s "
                  f"prompt_tok={usage.get('prompt_tokens','?')}", flush=True)
            for d in bad:
                print(f"      ✗ {d['id']}({d['kind']}) 距离={d['edit_distance']}",
                      flush=True)
                print(f"        期望: {d['expected']}", flush=True)
                print(f"        实得: {d['got']}", flush=True)
            results.append({
                "depth": depth_tokens, "rep": rep, "exact": exact,
                "total": len(asked), "rate": rate, "wall_s": round(elapsed, 2),
                "usage": usage, "details": details,
            })

    ok_runs = [r for r in results if "rate" in r]
    overall = (sum(r["exact"] for r in ok_runs) /
               max(1, sum(r["total"] for r in ok_runs))) if ok_runs else 0.0
    print(f"\n[copy-fidelity] tag={args.tag} 总体逐字保真度: {overall:.1%} "
          f"({len(ok_runs)} 轮有效)", flush=True)
    by_depth = {}
    for r in ok_runs:
        by_depth.setdefault(r["depth"], []).append(r["rate"])
    for d in sorted(by_depth):
        v = by_depth[d]
        print(f"    depth={d:>7}: {sum(v)/len(v):.1%}", flush=True)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"tag": args.tag, "model": args.model,
                       "overall": overall, "runs": results}, f,
                      ensure_ascii=False, indent=2)
        print(f"[copy-fidelity] 已写入 {args.json}", flush=True)
    return 0 if overall >= 0.999 else 1


if __name__ == "__main__":
    sys.exit(main())
