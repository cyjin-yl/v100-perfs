#!/usr/bin/env python3
"""Qwen3.8-27B on V100 的验收判定表。

对应约定的目标: 工具调用正确、无乱码、262K 上下文、OpenAI/Anthropic 双协议的
thinking effort 与 on/off、流式与非流式都正常、MTP 开启、5~10 并发无报错。

设计原则:
  - 每条判据都必须能**自动判定通过/失败**, 不能靠人眼看输出。
  - 失败时打印足以定位的证据(实际返回了什么), 而不是只说 FAIL。
  - 后端是 batch=1, 请求会排队; 因此默认串行跑, 并发那条单独开关。

用法:
  python goal_acceptance.py --base-url http://127.0.0.1:8000 --token "$AUTH_TOKEN"
  python goal_acceptance.py --only stream,thinking     # 只跑某几组
  python goal_acceptance.py --concurrency 6            # 加跑并发组
"""

import argparse
import json
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

import sys
import threading
import time
import urllib.error
import urllib.request

RESULTS = []


def post(base, token, path, body, timeout, stream=False):
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}",
                 "anthropic-version": "2023-06-01"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if stream:
            chunks = []
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("data:"):
                    chunks.append(line[5:].strip())
            return chunks, time.time() - t0
        return json.loads(resp.read().decode("utf-8")), time.time() - t0


def record(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"  {'✓' if ok else '✗'} {name}" + (f"   {detail}" if detail else ""),
          flush=True)


def msg(text):
    return [{"role": "user", "content": text}]


# ---------------------------------------------------------------- 各组判据

def group_nonstream(base, token, model, timeout):
    print("[非流式] OpenAI /v1/chat/completions")
    try:
        d, w = post(base, token, "/v1/chat/completions",
                    {"model": model, "max_tokens": 64, "stream": False,
                     "temperature": 0.0, "messages": msg("只回答一个词: 你好")},
                    timeout)
        content = (d.get("choices") or [{}])[0].get("message", {}).get("content")
        record("非流式返回非空正文", bool(content and content.strip()),
               f"wall={w:.1f}s len={len(content or '')}")
    except Exception as e:
        record("非流式返回非空正文", False, f"{type(e).__name__}: {e}")


def group_stream(base, token, model, timeout):
    print("[流式] SSE 分块与 [DONE]")
    try:
        chunks, w = post(base, token, "/v1/chat/completions",
                         {"model": model, "max_tokens": 64, "stream": True,
                          "temperature": 0.0, "messages": msg("数到五")},
                         timeout, stream=True)
        record("流式收到多个分块", len(chunks) > 2, f"chunks={len(chunks)}")
        record("流式以 [DONE] 收尾", any(c == "[DONE]" for c in chunks),
               f"末尾={chunks[-1][:40] if chunks else '无'}")
    except Exception as e:
        record("流式可用", False, f"{type(e).__name__}: {e}")


def group_thinking(base, token, model, timeout):
    print("[thinking] on/off 与 effort")
    for label, extra in [
        ("enable_thinking=False 时无思考内容",
         {"chat_template_kwargs": {"enable_thinking": False},
          "reasoning_effort": "none"}),
        ("enable_thinking=True 时有思考内容",
         {"chat_template_kwargs": {"enable_thinking": True},
          "reasoning_effort": "medium"}),
    ]:
        try:
            body = {"model": model, "max_tokens": 400, "stream": False,
                    "temperature": 0.0,
                    "messages": msg("13 乘以 17 等于多少? 简短回答。")}
            body.update(extra)
            d, w = post(base, token, "/v1/chat/completions", body, timeout)
            m = (d.get("choices") or [{}])[0].get("message", {}) or {}
            reasoning = m.get("reasoning_content") or ""
            want_thinking = "True" in label
            ok = bool(reasoning.strip()) == want_thinking
            record(label, ok, f"reasoning_len={len(reasoning)} wall={w:.1f}s")
        except Exception as e:
            record(label, False, f"{type(e).__name__}: {e}")

    for effort in ("low", "high"):
        try:
            d, w = post(base, token, "/v1/chat/completions",
                        {"model": model, "max_tokens": 500, "stream": False,
                         "temperature": 0.0, "reasoning_effort": effort,
                         "chat_template_kwargs": {"enable_thinking": True},
                         "messages": msg("简述快速排序的思路。")}, timeout)
            m = (d.get("choices") or [{}])[0].get("message", {}) or {}
            record(f"reasoning_effort={effort} 请求成功",
                   bool((m.get("content") or m.get("reasoning_content"))),
                   f"reasoning_len={len(m.get('reasoning_content') or '')} wall={w:.1f}s")
        except Exception as e:
            record(f"reasoning_effort={effort} 请求成功", False,
                   f"{type(e).__name__}: {e}")


def group_anthropic(base, token, model, timeout):
    print("[Anthropic] /v1/messages")
    try:
        d, w = post(base, token, "/v1/messages",
                    {"model": model, "max_tokens": 64,
                     "messages": msg("只回答一个词: 你好")}, timeout)
        blocks = d.get("content") or []
        text = "".join(b.get("text", "") for b in blocks
                       if isinstance(b, dict))
        record("Anthropic 协议返回正文", bool(text.strip()),
               f"wall={w:.1f}s len={len(text)}")
    except Exception as e:
        record("Anthropic 协议返回正文", False, f"{type(e).__name__}: {e}")


def group_toolcall(base, token, model, timeout):
    print("[工具调用] 名字保真 + 参数名正确")
    tools = [{"type": "function", "function": {
        "name": "bash",
        "description": "Run a shell command",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string",
                                                  "description": "要执行的命令"}},
                       "required": ["command"]}}}]
    try:
        d, w = post(base, token, "/v1/chat/completions",
                    {"model": model, "max_tokens": 256, "stream": False,
                     "temperature": 0.0, "tools": tools, "tool_choice": "auto",
                     "reasoning_effort": "none",
                     "chat_template_kwargs": {"enable_thinking": False},
                     "messages": msg("用工具执行 `echo hello`。只调用工具。")},
                    timeout)
        m = (d.get("choices") or [{}])[0].get("message", {}) or {}
        calls = m.get("tool_calls") or []
        names = [c.get("function", {}).get("name") for c in calls]
        record("工具名与声明完全一致", names == ["bash"], f"实得={names}")
        args = {}
        if calls:
            try:
                args = json.loads(calls[0]["function"].get("arguments") or "{}")
            except Exception:
                args = {}
        record("参数名为 command(非 action 之类)", "command" in args,
               f"参数键={list(args)} wall={w:.1f}s")
    except Exception as e:
        record("工具调用可用", False, f"{type(e).__name__}: {e}")


def group_copy(base, token, model, timeout):
    print("[逐字抄写] 字面量不被改写")
    literals = [
        "/home/ezra/Documents/Proto-UI/packages/prototypes/brutalist/src/card/index.ts",
        "Proto-UI",
        "13D010B6FDBC1A06",
        "3f9a2c7e15b8d046",
    ]
    for lit in literals:
        try:
            d, w = post(base, token, "/v1/chat/completions",
                        {"model": model, "max_tokens": 200, "stream": False,
                         "temperature": 0.0, "reasoning_effort": "none",
                         "chat_template_kwargs": {"enable_thinking": False},
                         "messages": msg(
                             f"把下面这行原样重复一遍, 不要加任何其它内容:\n{lit}")},
                        timeout)
            out = ((d.get("choices") or [{}])[0]
                   .get("message", {}) or {}).get("content") or ""
            record(f"逐字抄写 {lit[:44]}", lit in out,
                   "" if lit in out else f"实得={out.strip()[:70]!r}")
        except Exception as e:
            record(f"逐字抄写 {lit[:44]}", False, f"{type(e).__name__}: {e}")


def group_concurrency(base, token, model, timeout, n):
    print(f"[并发] {n} 路同时请求")
    errs, lock = [], threading.Lock()

    def one(i):
        try:
            d, _ = post(base, token, "/v1/chat/completions",
                        {"model": model, "max_tokens": 96, "stream": False,
                         "temperature": 0.0,
                         "messages": msg(f"用一句话介绍数字 {i}")}, timeout)
            c = (d.get("choices") or [{}])[0].get("message", {}).get("content")
            if not (c and c.strip()):
                with lock:
                    errs.append(f"#{i} 空正文")
        except Exception as e:
            with lock:
                errs.append(f"#{i} {type(e).__name__}: {e}")

    ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    record(f"{n} 路并发全部成功", not errs,
           f"wall={time.time()-t0:.1f}s" + (f" 失败={errs[:3]}" if errs else ""))


GROUPS = {
    "nonstream": group_nonstream, "stream": group_stream,
    "thinking": group_thinking, "anthropic": group_anthropic,
    "toolcall": group_toolcall, "copy": group_copy,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--token", default=os.environ.get("AUTH_TOKEN", "x"))
    ap.add_argument("--model", default="qwen3.8-fastllm")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--only", default=None,
                    help="逗号分隔的组名: " + ",".join(GROUPS))
    ap.add_argument("--concurrency", type=int, default=0,
                    help=">0 时额外跑并发组")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    names = ([g.strip() for g in args.only.split(",")] if args.only
             else list(GROUPS))
    for n in names:
        if n not in GROUPS:
            print(f"未知组名: {n}", file=sys.stderr)
            return 2
        GROUPS[n](args.base_url, args.token, args.model, args.timeout)
        print()

    if args.concurrency > 0:
        group_concurrency(args.base_url, args.token, args.model,
                          args.timeout, args.concurrency)
        print()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"验收: {passed}/{len(RESULTS)} 通过")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  未过: {name}  {detail}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([{"name": n, "ok": o, "detail": d}
                       for n, o, d in RESULTS], f,
                      ensure_ascii=False, indent=2)
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
