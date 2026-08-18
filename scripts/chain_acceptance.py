#!/usr/bin/env python
"""生产链路验收探针: nginx:80 /v1 → thinking_proxy:8000 → fastllm apiserver。

存在的理由: 之前的"修好了"结论都是小样本 + 单协议 + 非生产链路测出来的, 一上 omp
就寄。这个脚本按 haru 的验收线把矩阵铺满, 并且**只走生产链路**(默认打 nginx:80)。

四个 suite:
  matrix       {openai, anthropic} x {stream, non-stream} x {off, low, medium,
               high, xhigh} 的工具调用保真度 + thinking 开关是否真的生效
  toolloop     多轮工具对话(工具结果回灌), 检查工具历史往返与参数漂移
  concurrency  K 路并发流式 toolloop(默认 6), 检查 5~10 并发无报错
  longctx      大上下文 prefill + 重放, 量 TTFT / prefill 速度 / 前缀命中

判定口径(hard failure):
  - HTTP 非 200 / 流中断 / SSE 解析失败
  - 该调工具时没有 tool_call, 或 tool_call 参数为空 / 缺 required 键
  - thinking=off 时仍产出 reasoning, thinking=on 时完全没有 reasoning
  - 输出解码循环(同一行连续重复超过阈值)

用法:
  python chain_acceptance.py --suite matrix
  python chain_acceptance.py --suite toolloop --rounds 6
  python chain_acceptance.py --suite concurrency --concurrency 6
  python chain_acceptance.py --suite longctx --ctx-tokens 200000
  python chain_acceptance.py --suite all --json runtime/handoff/acceptance.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from typing import Any

import httpx

MODEL = os.environ.get("ACCEPT_MODEL", "qwen3.8-27b")
BASE = os.environ.get("ACCEPT_BASE", "http://127.0.0.1:80")
TOKEN = os.environ.get("AUTH_TOKEN", "")

LIST_DIR_TOOL = {
    "name": "list_dir",
    "description": "List entries under an absolute directory path on this machine.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string",
                     "description": "Absolute directory path, e.g. /etc"},
            "max_depth": {"type": "integer",
                          "description": "Recursion depth, default 1"},
        },
        "required": ["path"],
    },
}
READ_FILE_TOOL = {
    "name": "read_file",
    "description": "Read the full text content of a file.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute file path"},
        },
        "required": ["path"],
    },
}
TOOLS = [LIST_DIR_TOOL, READ_FILE_TOOL]

TOOL_PROMPT = (
    "列出 /etc 目录下的条目。你必须通过工具调用完成, 不要凭记忆直接回答, "
    "也不要只描述你要做什么。"
)


# ─── 协议封装 ────────────────────────────────────────────────────


def _openai_tools() -> list[dict]:
    return [{"type": "function", "function": t} for t in TOOLS]


def _anthropic_tools() -> list[dict]:
    return [{"name": t["name"], "description": t["description"],
             "input_schema": t["parameters"]} for t in TOOLS]


def _headers() -> dict[str, str]:
    h = {"content-type": "application/json"}
    if TOKEN:
        h["authorization"] = f"Bearer {TOKEN}"
        h["x-api-key"] = TOKEN
    return h


def openai_body(messages: list[dict], effort: str | None, stream: bool,
                max_tokens: int = 2048) -> dict:
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "tools": _openai_tools(),
        "stream": stream,
        "max_tokens": max_tokens,
        "temperature": 1.0,
    }
    if effort in (None, "off"):
        body["enable_thinking"] = False
    else:
        body["reasoning_effort"] = effort
    return body


_ANTHROPIC_BUDGET = {"low": 2048, "medium": 4096, "high": 8192, "xhigh": 16384}


def anthropic_body(messages: list[dict], effort: str | None, stream: bool,
                   max_tokens: int = 2048, system: str | None = None) -> dict:
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "tools": _anthropic_tools(),
        "stream": stream,
        "max_tokens": max_tokens,
    }
    if system:
        body["system"] = system
    if effort in (None, "off"):
        body["thinking"] = {"type": "disabled"}
    else:
        body["thinking"] = {"type": "enabled",
                            "budget_tokens": _ANTHROPIC_BUDGET[effort]}
    return body


# ─── 结果结构 ────────────────────────────────────────────────────


DUMP_DIR = ""
_DUMP_SEQ = [0]


def dump_failure(label: str, body: dict, turn: "Turn") -> str:
    """把失败样本的请求体 + 原始 SSE 落盘 —— 空参数这类间歇故障, 没有原始
    流就只能猜。"""
    if not DUMP_DIR:
        return ""
    os.makedirs(DUMP_DIR, exist_ok=True)
    _DUMP_SEQ[0] += 1
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", label)
    path = os.path.join(DUMP_DIR, f"fail-{_DUMP_SEQ[0]:03d}-{safe}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"label": label, "errors": turn.errors,
                   "request": body,
                   "text": turn.text, "reasoning": turn.reasoning,
                   "tool_calls": turn.tool_calls,
                   "finish": turn.raw_finish, "usage": turn.usage,
                   "raw_events": turn.raw_lines}, fh,
                  ensure_ascii=False, indent=2)
    print(f"    [dump] {path}", flush=True)
    return path


class Turn:
    """一次请求的归一化结果, 两个协议共用。"""

    def __init__(self) -> None:
        self.ok = True
        self.raw_lines: list[str] = []
        self.errors: list[str] = []
        self.text = ""
        self.reasoning = ""
        self.tool_calls: list[dict] = []   # {id, name, arguments(dict|str)}
        self.ttft: float | None = None
        self.latency = 0.0
        self.raw_finish = ""
        self.usage: dict = {}

    def fail(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def summary(self) -> str:
        names = ",".join(str(c.get("name")) for c in self.tool_calls) or "-"
        return (f"ok={self.ok} tools={names} think={len(self.reasoning)}c "
                f"text={len(self.text)}c ttft={self.ttft} "
                f"lat={self.latency:.1f}s")


_TRANSIENT_PREFIXES = ("backend_error", "http_503", "http_502", "transport",
                       "stream_broken")


async def resilient(make_turn, tries: int = 3, delay: float = 20.0) -> "Turn":
    """瞬时后端不可用时重试。

    后端在显存水位压力/懒加载权重期间会短暂返回 503 backend_reloading;
    这台机器上还有外部流量(haru 自己的 omp 会打 15 万 token 的请求)会把 KV pool
    打满。不重试的话, 这类环境噪声会被记成"模型把字丢了/不调工具"。
    """
    turn = await make_turn()
    for i in range(tries - 1):
        transient = [e for e in turn.errors
                     if e.startswith(_TRANSIENT_PREFIXES)]
        if not transient:
            return turn
        print(f"    [retry] 瞬时后端不可用: {transient[0][:70]}; "
              f"{delay:.0f}s 后第 {i + 2} 次", flush=True)
        await asyncio.sleep(delay)
        turn = await make_turn()
    return turn


LOOP_RE = re.compile(r"(.{24,}?)\1{4,}", re.S)


def check_loop(turn: Turn) -> None:
    """解码循环检测: 同一段 >=24 字符的串连续重复 5 次以上。"""
    for field, blob in (("text", turn.text), ("reasoning", turn.reasoning)):
        m = LOOP_RE.search(blob)
        if m:
            turn.fail(f"decode_loop_in_{field}:{m.group(1)[:40]!r}")


def check_tool_call(turn: Turn, expect: str, required: list[str]) -> None:
    if not turn.tool_calls:
        turn.fail("no_tool_call")
        return
    for call in turn.tool_calls:
        if call.get("name") != expect:
            turn.fail(f"wrong_tool_name:{call.get('name')!r}")
            continue
        args = call.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError:
                turn.fail(f"arguments_not_json:{str(call.get('arguments'))[:60]!r}")
                continue
        if not isinstance(args, dict) or not args:
            turn.fail("empty_arguments")
            continue
        for key in required:
            value = args.get(key)
            if value is None or (isinstance(value, str) and not value.strip()):
                turn.fail(f"missing_required_arg:{key}")


def check_thinking(turn: Turn, effort: str | None) -> None:
    if effort in (None, "off"):
        if turn.reasoning.strip():
            turn.fail(f"thinking_leaked_when_off:{len(turn.reasoning)}c")
    else:
        if not turn.reasoning.strip():
            turn.fail("thinking_missing_when_on")


# ─── 请求执行 ────────────────────────────────────────────────────


async def run_openai(client: httpx.AsyncClient, body: dict) -> Turn:
    turn = Turn()
    t0 = time.monotonic()
    if not body.get("stream"):
        try:
            r = await client.post(f"{BASE}/v1/chat/completions", json=body,
                                  headers=_headers())
        except Exception as exc:                      # noqa: BLE001
            turn.fail(f"transport:{type(exc).__name__}:{exc}")
            turn.latency = time.monotonic() - t0
            return turn
        turn.latency = time.monotonic() - t0
        if r.status_code != 200:
            turn.fail(f"http_{r.status_code}:{r.text[:200]}")
            return turn
        data = r.json()
        turn.usage = data.get("usage") or {}
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        turn.raw_finish = choice.get("finish_reason") or ""
        turn.text = msg.get("content") or ""
        turn.reasoning = msg.get("reasoning_content") or ""
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            turn.tool_calls.append({"id": tc.get("id"), "name": fn.get("name"),
                                    "arguments": fn.get("arguments")})
        return turn

    acc: dict[int, dict] = {}
    try:
        async with client.stream("POST", f"{BASE}/v1/chat/completions",
                                 json=body, headers=_headers()) as r:
            if r.status_code != 200:
                payload = await r.aread()
                turn.fail(f"http_{r.status_code}:{payload[:200]!r}")
                turn.latency = time.monotonic() - t0
                return turn
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if DUMP_DIR and len(turn.raw_lines) < 4000:
                    turn.raw_lines.append(payload)
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    turn.fail(f"bad_sse_json:{payload[:80]!r}")
                    continue
                # 后端重载/繁忙时 proxy 会在流中间插一个 error 事件然后 [DONE]。
                # 不识别它的话, 空响应会被上层误判成"模型没输出/字丢了"——
                # 2026-08-18 的 garble suite 就是这么误报的。
                if event.get("error"):
                    err = event["error"]
                    msg = err.get("message") if isinstance(err, dict) else str(err)
                    turn.fail(f"backend_error:{event.get('status', '')}:{str(msg)[:120]}")
                    continue
                if event.get("usage"):
                    turn.usage = event["usage"]
                choice = (event.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                if choice.get("finish_reason"):
                    turn.raw_finish = choice["finish_reason"]
                chunk_text = delta.get("content") or ""
                chunk_reason = delta.get("reasoning_content") or ""
                if (chunk_text or chunk_reason) and turn.ttft is None:
                    turn.ttft = round(time.monotonic() - t0, 2)
                turn.text += chunk_text
                turn.reasoning += chunk_reason
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = acc.setdefault(idx, {"id": None, "name": None,
                                                "arguments": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
                    if turn.ttft is None:
                        turn.ttft = round(time.monotonic() - t0, 2)
    except Exception as exc:                          # noqa: BLE001
        turn.fail(f"stream_broken:{type(exc).__name__}:{exc}")
    turn.latency = time.monotonic() - t0
    turn.tool_calls = [acc[i] for i in sorted(acc)]
    return turn


async def run_anthropic(client: httpx.AsyncClient, body: dict) -> Turn:
    turn = Turn()
    t0 = time.monotonic()
    if not body.get("stream"):
        try:
            r = await client.post(f"{BASE}/v1/messages", json=body,
                                  headers=_headers())
        except Exception as exc:                      # noqa: BLE001
            turn.fail(f"transport:{type(exc).__name__}:{exc}")
            turn.latency = time.monotonic() - t0
            return turn
        turn.latency = time.monotonic() - t0
        if r.status_code != 200:
            turn.fail(f"http_{r.status_code}:{r.text[:200]}")
            return turn
        data = r.json()
        turn.usage = data.get("usage") or {}
        turn.raw_finish = data.get("stop_reason") or ""
        for block in data.get("content") or []:
            btype = block.get("type")
            if btype == "text":
                turn.text += block.get("text") or ""
            elif btype == "thinking":
                turn.reasoning += block.get("thinking") or ""
            elif btype == "tool_use":
                turn.tool_calls.append({"id": block.get("id"),
                                        "name": block.get("name"),
                                        "arguments": block.get("input")})
        return turn

    blocks: dict[int, dict] = {}
    try:
        async with client.stream("POST", f"{BASE}/v1/messages", json=body,
                                 headers=_headers()) as r:
            if r.status_code != 200:
                payload = await r.aread()
                turn.fail(f"http_{r.status_code}:{payload[:200]!r}")
                turn.latency = time.monotonic() - t0
                return turn
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload:
                    continue
                if DUMP_DIR and len(turn.raw_lines) < 4000:
                    turn.raw_lines.append(payload)
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    turn.fail(f"bad_sse_json:{payload[:80]!r}")
                    continue
                etype = event.get("type")
                if etype == "content_block_start":
                    blk = event.get("content_block") or {}
                    blocks[event.get("index", 0)] = {
                        "type": blk.get("type"), "name": blk.get("name"),
                        "id": blk.get("id"), "json": "", "text": ""}
                elif etype == "content_block_delta":
                    slot = blocks.setdefault(event.get("index", 0),
                                             {"type": None, "json": "",
                                              "text": ""})
                    delta = event.get("delta") or {}
                    dtype = delta.get("type")
                    if turn.ttft is None:
                        turn.ttft = round(time.monotonic() - t0, 2)
                    if dtype == "text_delta":
                        turn.text += delta.get("text") or ""
                    elif dtype == "thinking_delta":
                        turn.reasoning += delta.get("thinking") or ""
                    elif dtype == "input_json_delta":
                        slot["json"] += delta.get("partial_json") or ""
                elif etype == "message_delta":
                    turn.raw_finish = ((event.get("delta") or {})
                                       .get("stop_reason") or turn.raw_finish)
                    if event.get("usage"):
                        turn.usage = event["usage"]
                elif etype == "error":
                    turn.fail(f"sse_error:{json.dumps(event)[:160]}")
    except Exception as exc:                          # noqa: BLE001
        turn.fail(f"stream_broken:{type(exc).__name__}:{exc}")
    turn.latency = time.monotonic() - t0
    for idx in sorted(blocks):
        slot = blocks[idx]
        if slot.get("type") == "tool_use":
            turn.tool_calls.append({"id": slot.get("id"),
                                    "name": slot.get("name"),
                                    "arguments": slot.get("json") or "{}"})
    return turn


# ─── suite: matrix ───────────────────────────────────────────────

EFFORTS = ["off", "low", "medium", "high", "xhigh"]


async def suite_matrix(client: httpx.AsyncClient, args) -> list[dict]:
    rows = []
    messages = [{"role": "user", "content": TOOL_PROMPT}]
    apis = [a for a in ("openai", "anthropic") if a in args.apis.split(",")]
    efforts = [e for e in EFFORTS if e in args.efforts.split(",")]
    streams = [s == "1" for s in args.streams.split(",")]
    for api in apis:
        for stream in streams:
            for effort in efforts:
                label = f"{api}/{'stream' if stream else 'block'}/{effort}"
                for attempt in range(args.repeat):
                    if api == "openai":
                        body = openai_body(messages, effort, stream)
                        turn = await resilient(lambda: run_openai(client, body))
                    else:
                        body = anthropic_body(messages, effort, stream)
                        turn = await resilient(lambda: run_anthropic(client, body))
                    check_tool_call(turn, "list_dir", ["path"])
                    check_thinking(turn, effort)
                    check_loop(turn)
                    if not turn.ok:
                        dump_failure(f"{label}-{attempt}", body, turn)
                    rows.append({"case": label, "attempt": attempt,
                                 "ok": turn.ok, "errors": turn.errors,
                                 "ttft": turn.ttft, "latency": turn.latency,
                                 "reasoning_chars": len(turn.reasoning),
                                 "tools": [c.get("name")
                                           for c in turn.tool_calls]})
                    print(f"  {label} #{attempt}: {turn.summary()}"
                          + (f" ERR={turn.errors}" if turn.errors else ""),
                          flush=True)
    return rows


# ─── suite: toolloop ─────────────────────────────────────────────


FAKE_LS = ("hostname\nhosts\nnginx/\npasswd\nresolv.conf\nssh/\n"
           "systemd/\nfstab\nos-release\nprofile")


async def one_toolloop(client: httpx.AsyncClient, api: str, rounds: int,
                       effort: str, tag: str) -> dict:
    """一路多轮工具对话:每轮都必须产出一个合法 tool_call。"""
    failures: list[str] = []
    ttfts: list[float] = []
    if api == "openai":
        messages: list[dict] = [{"role": "user", "content": TOOL_PROMPT}]
    else:
        messages = [{"role": "user", "content": TOOL_PROMPT}]
    for rnd in range(rounds):
        if api == "openai":
            turn = await run_openai(
                client, openai_body(messages, effort, True, max_tokens=1536))
        else:
            turn = await run_anthropic(
                client, anthropic_body(messages, effort, True,
                                       max_tokens=1536))
        expect = "list_dir" if rnd == 0 else None
        if expect:
            check_tool_call(turn, expect, ["path"])
        elif not turn.tool_calls:
            turn.fail("no_tool_call")
        else:
            for call in turn.tool_calls:
                if call.get("name") not in ("list_dir", "read_file"):
                    turn.fail(f"wrong_tool_name:{call.get('name')!r}")
                args = call.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args or "{}")
                    except json.JSONDecodeError:
                        turn.fail("arguments_not_json")
                        args = {}
                if not args or not str(args.get("path", "")).strip():
                    turn.fail("empty_arguments")
        check_loop(turn)
        if turn.ttft is not None:
            ttfts.append(turn.ttft)
        if not turn.ok:
            failures.append(f"round{rnd}:{','.join(turn.errors)}")
            print(f"  [{tag}] round {rnd} FAIL {turn.errors}", flush=True)
            break
        print(f"  [{tag}] round {rnd} ok "
              f"tool={turn.tool_calls[0].get('name')} ttft={turn.ttft} "
              f"lat={turn.latency:.1f}s", flush=True)

        # 回灌工具结果, 并要求下一步继续用工具
        call = turn.tool_calls[0]
        raw_args = call.get("arguments")
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args or "{}")
            except json.JSONDecodeError:
                raw_args = {}
        if api == "openai":
            messages.append({
                "role": "assistant",
                "content": turn.text or "",
                "tool_calls": [{"id": call.get("id") or f"call_{rnd}",
                                "type": "function",
                                "function": {
                                    "name": call.get("name"),
                                    "arguments": json.dumps(raw_args,
                                                            ensure_ascii=False)}}],
            })
            messages.append({"role": "tool",
                             "tool_call_id": call.get("id") or f"call_{rnd}",
                             "content": FAKE_LS})
            messages.append({"role": "user", "content":
                             "很好。现在用 read_file 读取 /etc/hostname, "
                             "同样必须通过工具调用。"})
        else:
            messages.append({"role": "assistant", "content": [
                {"type": "tool_use", "id": call.get("id") or f"toolu_{rnd}",
                 "name": call.get("name"), "input": raw_args or {}}]})
            messages.append({"role": "user", "content": [
                {"type": "tool_result",
                 "tool_use_id": call.get("id") or f"toolu_{rnd}",
                 "content": FAKE_LS}]})
            messages.append({"role": "user", "content":
                             "很好。现在用 read_file 读取 /etc/hostname, "
                             "同样必须通过工具调用。"})
    return {"tag": tag, "api": api, "rounds_done": rounds - len(failures),
            "failures": failures,
            "ttft_median": round(statistics.median(ttfts), 2) if ttfts else None}


async def suite_toolloop(client: httpx.AsyncClient, args) -> list[dict]:
    out = []
    for api in ("openai", "anthropic"):
        out.append(await one_toolloop(client, api, args.rounds, "high",
                                     f"toolloop-{api}"))
    return out


# ─── suite: concurrency ──────────────────────────────────────────


async def suite_concurrency(client: httpx.AsyncClient, args) -> list[dict]:
    k = args.concurrency
    print(f"  启动 {k} 路并发流式 toolloop(每路 {args.rounds} 轮)", flush=True)
    tasks = [one_toolloop(client, "openai" if i % 2 == 0 else "anthropic",
                          args.rounds, "medium", f"conc{i}")
             for i in range(k)]
    t0 = time.monotonic()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    wall = time.monotonic() - t0
    rows = []
    for i, res in enumerate(results):
        if isinstance(res, BaseException):
            rows.append({"tag": f"conc{i}", "failures": [f"exception:{res}"],
                         "rounds_done": 0})
        else:
            rows.append(res)
    print(f"  并发 wall={wall:.1f}s", flush=True)
    return rows


# ─── suite: longctx ──────────────────────────────────────────────


def build_long_context(target_tokens: int) -> tuple[str, str]:
    """造一段带唯一标记的长上下文, 用于同时测 prefill 与深处召回。"""
    needle = "ZK-7731-ORION"
    chunk = (
        "记录 {i}: 服务 svc-{i} 在区域 r{r} 的 p99 延迟为 {ms}ms, "
        "副本数 {n}, 归属团队 team-{t}。\n"
    )
    parts = ["以下是一份运维台账, 请在被问到时严格依据台账内容回答。\n"]
    approx = 0
    i = 0
    # 粗估 1 token ≈ 1.6 字符(中英混排), 只需量级正确
    while approx < target_tokens * 1.6:
        line = chunk.format(i=i, r=i % 7, ms=100 + (i * 37) % 900,
                            n=1 + i % 5, t=i % 13)
        if i == 3:
            line += f"重要: 审计密钥是 {needle}。\n"
        parts.append(line)
        approx += len(line)
        i += 1
    return "".join(parts), needle


async def suite_longctx(client: httpx.AsyncClient, args) -> list[dict]:
    ledger, needle = build_long_context(args.ctx_tokens)
    print(f"  上下文约 {len(ledger)} 字符(目标 ~{args.ctx_tokens} token)",
          flush=True)
    rows = []
    for pass_no in (1, 2):        # 第二遍应该吃到前缀缓存
        messages = [
            {"role": "user", "content": ledger},
            {"role": "assistant", "content": "台账已读取。"},
            {"role": "user", "content":
             "台账里记录的审计密钥是什么? 然后用 read_file 工具读取 "
             "/etc/hostname。两件事都要做。"},
        ]
        turn = await run_openai(
            client, openai_body(messages, "low", True, max_tokens=1024))
        recalled = needle in (turn.text + turn.reasoning)
        if not recalled:
            turn.fail("needle_not_recalled")
        if not turn.tool_calls:
            turn.fail("no_tool_call")
        check_loop(turn)
        prompt_tokens = (turn.usage or {}).get("prompt_tokens")
        prefill_rate = (round(prompt_tokens / turn.ttft, 1)
                        if prompt_tokens and turn.ttft else None)
        rows.append({"pass": pass_no, "ok": turn.ok, "errors": turn.errors,
                     "ttft": turn.ttft, "latency": round(turn.latency, 1),
                     "prompt_tokens": prompt_tokens,
                     "prefill_tok_s_est": prefill_rate,
                     "needle_recalled": recalled,
                     "tools": [c.get("name") for c in turn.tool_calls]})
        print(f"  pass{pass_no}: ok={turn.ok} ttft={turn.ttft} "
              f"prompt_tokens={prompt_tokens} needle={recalled} "
              f"errors={turn.errors}", flush=True)
    return rows


# ─── main ────────────────────────────────────────────────────────


# ─── suite: garble(乱码) ─────────────────────────────────────────

RARE_CHARS = "饕餮 魑魅魍魉 齉齾 靐龘 兲丆 瓛璩 黼黻 龏龗"
GARBLE_PROMPT = (
    "逐字复写下面这一行, 不要解释, 不要加拼音:\n"
    f"{RARE_CHARS}\n"
    "然后另起一行复写: αβγ ∑∫∂ ✓✗ 🌸🔧🧪 ①②③ ｱｲｳ"
)
GARBLE_EXPECT = ["饕餮", "魑魅魍魉", "齉齾", "靐龘", "黼黻",
                 "αβγ", "∑∫∂", "🌸", "①②③"]


async def suite_garble(client: httpx.AsyncClient, args) -> list[dict]:
    """低频字/多字节 token 保真度: 量化档位掉字节会在这里现形。"""
    rows = []
    for attempt in range(max(args.repeat, 3)):
        messages = [{"role": "user", "content": GARBLE_PROMPT}]
        # 非流式: 乱码判定只关心最终文本, 而流式路径在后端热身/显存水位窗口里
        # 会插入 503 事件, 把空响应误判成"字丢了"(整晚的 garble 假失败都是这么来的)。
        gbody = openai_body(messages, "off", False, max_tokens=512)
        gbody.pop("tools", None)     # 复写题不该带工具, 否则模型可能改去调工具
        turn = await resilient(lambda: run_openai(client, gbody))
        blob = turn.text + turn.reasoning
        bad_bytes = blob.count("�")
        missing = [tok for tok in GARBLE_EXPECT if tok not in blob]
        if bad_bytes:
            turn.fail(f"replacement_chars:{bad_bytes}")
        if missing:
            turn.fail(f"missing_glyphs:{missing}")
        check_loop(turn)
        if not turn.ok:
            dump_failure(f"garble-{attempt}",
                         openai_body(messages, "off", True), turn)
        rows.append({"attempt": attempt, "ok": turn.ok, "errors": turn.errors,
                     "bad_bytes": bad_bytes, "missing": missing,
                     "chars": len(blob), "latency": round(turn.latency, 1)})
        print(f"  garble #{attempt}: bad_bytes={bad_bytes} missing={missing} "
              f"ok={turn.ok}", flush=True)
    return rows


# ─── suite: bench(prefill / decode / e2e) ────────────────────────


_REQ_DONE_RE = re.compile(
    r"\[req (\d+)\] done: prefill (\d+) tok ([\d.]+)s \| "
    r"decode (\d+) tok in ([\d.]+)s")


def tail_backend_metrics(log_path: str, since_offset: int) -> tuple[list[dict], int]:
    """从后端日志里取权威的 prefill/decode 拆分(引擎自己打的)。"""
    if not log_path or not os.path.exists(log_path):
        return [], since_offset
    size = os.path.getsize(log_path)
    if size < since_offset:
        since_offset = 0
    with open(log_path, "r", errors="replace") as fh:
        fh.seek(since_offset)
        blob = fh.read()
    out = []
    for m in _REQ_DONE_RE.finditer(blob):
        prefill_tok, prefill_s = int(m.group(2)), float(m.group(3))
        decode_tok, decode_s = int(m.group(4)), float(m.group(5))
        out.append({
            "prefill_tok": prefill_tok, "prefill_s": prefill_s,
            "prefill_tok_s": round(prefill_tok / prefill_s, 1) if prefill_s else None,
            "decode_tok": decode_tok, "decode_s": decode_s,
            "decode_tok_s": round(decode_tok / decode_s, 1) if decode_s else None,
            "e2e_tok_s": round((prefill_tok + decode_tok) /
                               (prefill_s + decode_s), 1)
            if (prefill_s + decode_s) else None,
        })
    mtp_off = blob.count("[Qwen3.5 MTP] not enabled")
    if out:
        out[-1]["mtp_not_enabled_lines"] = mtp_off
    return out, size


def _filler(target_tokens: int, salt: str) -> str:
    ledger, _ = build_long_context(target_tokens)
    return f"[batch {salt}]\n" + ledger


async def suite_bench(client: httpx.AsyncClient, args) -> list[dict]:
    sizes = [int(s) for s in args.bench_sizes.split(",") if s.strip()]
    rows = []
    offset = (os.path.getsize(args.log_path)
              if args.log_path and os.path.exists(args.log_path) else 0)
    for size in sizes:
        for pass_no in (1, 2):     # pass2 = 同前缀重放, 量前缀缓存命中
            ctx = _filler(size, f"s{size}") if size > 400 else "你好。"
            messages = [{"role": "user", "content": ctx},
                        {"role": "assistant", "content": "已读取。"},
                        {"role": "user", "content":
                         "用一句话总结上面的内容, 不超过 30 字。"}]
            turn = await run_openai(client, openai_body(
                messages, "off", True, max_tokens=args.bench_out_tokens))
            check_loop(turn)
            metrics, offset = tail_backend_metrics(args.log_path, offset)
            eng = metrics[-1] if metrics else {}
            prompt_tokens = (turn.usage or {}).get("prompt_tokens")
            row = {"target_tokens": size, "pass": pass_no, "ok": turn.ok,
                   "errors": turn.errors, "ttft": turn.ttft,
                   "wall": round(turn.latency, 2),
                   "prompt_tokens": prompt_tokens,
                   "client_e2e_tok_s": round(
                       ((prompt_tokens or 0) + len(turn.text) // 3)
                       / turn.latency, 1) if turn.latency else None,
                   **{f"engine_{k}": v for k, v in eng.items()}}
            rows.append(row)
            print(f"  bench {size}tok pass{pass_no}: ttft={turn.ttft} "
                  f"wall={row['wall']}s engine_prefill="
                  f"{eng.get('prefill_tok_s')} tok/s engine_decode="
                  f"{eng.get('decode_tok_s')} tok/s e2e="
                  f"{eng.get('e2e_tok_s')} mtp_off_lines="
                  f"{eng.get('mtp_not_enabled_lines')}", flush=True)
    return rows


# ─── suite: images(多图不崩) ─────────────────────────────────────


def _png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """不依赖 PIL 生成一张纯色 PNG(测多图时只关心解码通路与显存, 不关心内容)。"""
    import struct
    import zlib

    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))


async def suite_images(client: httpx.AsyncClient, args) -> list[dict]:
    """omp 的 snapcompact 会一次贴很多张图: 2/5/7 张都不能崩, 且要真的走多模态通路。"""
    import base64

    rows = []
    for count in [int(x) for x in args.image_counts.split(",") if x.strip()]:
        blocks = []
        for i in range(count):
            png = _png(args.image_size, args.image_size,
                       (30 + i * 30 % 200, 60, 200 - i * 20 % 180))
            url = "data:image/png;base64," + base64.b64encode(png).decode()
            blocks.append({"type": "image_url", "image_url": {"url": url}})
        blocks.append({"type": "text", "text":
                       f"上面一共有几张图? 每张图的主色调是什么? "
                       f"请直接回答, 不要调用工具。(应为 {count} 张)"})
        body = openai_body([{"role": "user", "content": blocks}], "off", True,
                           max_tokens=512)
        body.pop("tools", None)
        turn = await resilient(lambda: run_openai(client, body), tries=2, delay=15)
        check_loop(turn)
        answered = bool(turn.text.strip())
        if not answered:
            turn.fail("empty_answer")
        rows.append({"images": count, "size": args.image_size, "ok": turn.ok,
                     "errors": turn.errors, "chars": len(turn.text),
                     "latency": round(turn.latency, 1),
                     "head": turn.text[:120]})
        print(f"  images x{count} ({args.image_size}px): ok={turn.ok} "
              f"chars={len(turn.text)} lat={turn.latency:.1f}s "
              f"{turn.errors if turn.errors else ''}", flush=True)
        if not turn.ok:
            dump_failure(f"images-{count}", {"images": count}, turn)
    return rows


# ─── 预检: 绝不允许静默走云端 ────────────────────────────────────


async def preflight(client: httpx.AsyncClient) -> dict:
    """确认目标 model 真的在本地别名表里。

    2026-08-18 的教训: proxy 换 profile 后 /v1/models 只剩 qwen3.6-*,
    我们照旧打 qwen3.8-27b, 请求被静默路由到 OpenRouter/NIM ——
    20 格矩阵全 no_tool_call, 却与本地后端毫无关系。
    """
    info: dict[str, Any] = {}
    r = await client.get(f"{BASE}/v1/models", headers=_headers())
    r.raise_for_status()
    ids = [m.get("id") for m in (r.json().get("data") or [])]
    info["advertised_models"] = ids
    if MODEL not in ids:
        raise SystemExit(
            f"[preflight] 目标模型 {MODEL!r} 不在 /v1/models {ids} 里 —— "
            "请求会被路由到云端 fallback, 测出来的一切都无效。先修 profile 别名。")
    try:
        h = await client.get(f"{BASE.rsplit(':', 1)[0]}:8000/health", timeout=10)
        info["health"] = h.json()
    except Exception:                                  # noqa: BLE001
        try:
            h = await client.get(f"{BASE}/health", timeout=10)
            info["health"] = h.json()
        except Exception:                              # noqa: BLE001
            info["health"] = "unavailable"
    ready = (info.get("health") or {})
    if isinstance(ready, dict) and ready.get("ready") is False:
        raise SystemExit("[preflight] 后端 not ready, 先等加载完成。")

    # /health 的 ready 只说明子进程起来了。SKIP_WARMUP 下权重要等第一个真实请求才上传,
    # 这期间请求会拿到 503 backend_reloading —— 直接开跑会把"后端还没热"记成模型缺陷。
    for attempt in range(12):
        # 用有一定长度的 prompt + 输出, 逼后端真正把权重铺开; 8 token 的探针会
        # "通过"但后端下一刻仍会对正常请求返回 503。
        probe = await run_openai(client, {
            "model": MODEL, "stream": False, "max_tokens": 64,
            "messages": [{"role": "user", "content":
                          "用一句话说明什么是 KV cache。" * 8}],
            "enable_thinking": False})
        if probe.ok or not any(e.startswith(("http_", "backend_error", "transport"))
                               for e in probe.errors):
            info["warmup_attempts"] = attempt + 1
            print(f"[preflight] warmup 通过(第 {attempt + 1} 次, "
                  f"{probe.latency:.1f}s)", flush=True)
            break
        print(f"[preflight] warmup 第 {attempt + 1} 次未通过: {probe.errors}, 15s 后重试",
              flush=True)
        await asyncio.sleep(15)
    else:
        raise SystemExit("[preflight] warmup 连续 12 次失败, 后端没有真正可用。")
    print(f"[preflight] models={ids} "
          f"backend={(ready or {}).get('backend') if isinstance(ready, dict) else '?'} "
          f"state={((ready or {}).get('lifecycle') or {}).get('state') if isinstance(ready, dict) else '?'}",
          flush=True)
    return info


SUITES = {"matrix": suite_matrix, "toolloop": suite_toolloop,
          "concurrency": suite_concurrency, "longctx": suite_longctx,
          "garble": suite_garble, "bench": suite_bench,
          "images": suite_images}


async def main() -> int:
    global BASE, MODEL, DUMP_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="matrix",
                    choices=[*SUITES, "all"])
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--repeat", type=int, default=1,
                    help="matrix 每格重复次数(抗幸存者偏差, 建议 >=3)")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--ctx-tokens", type=int, default=120000)
    ap.add_argument("--timeout", type=float, default=2400)
    ap.add_argument("--json", default="")
    ap.add_argument("--bench-sizes", default="200,8000,32000,128000",
                    help="bench suite 的 prompt token 目标档位")
    ap.add_argument("--bench-out-tokens", type=int, default=192)
    ap.add_argument("--log-path", default="",
                    help="后端日志路径, 用于取引擎侧权威 prefill/decode 拆分")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--dump-dir", default="",
                    help="失败样本的请求体+原始 SSE 转储目录")
    ap.add_argument("--apis", default="openai,anthropic")
    ap.add_argument("--efforts", default="off,low,medium,high,xhigh")
    ap.add_argument("--streams", default="0,1")
    ap.add_argument("--image-counts", default="2,5,7")
    ap.add_argument("--image-size", type=int, default=512)
    args = ap.parse_args()

    DUMP_DIR = args.dump_dir

    BASE = args.base.rstrip("/")
    MODEL = args.model

    suites = list(SUITES) if args.suite == "all" else [args.suite]
    report: dict[str, Any] = {"base": BASE, "model": MODEL,
                              "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                              "suites": {}}
    limits = httpx.Limits(max_connections=32, max_keepalive_connections=16)
    timeout = httpx.Timeout(args.timeout, connect=15.0)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        if not args.skip_preflight:
            report["preflight"] = await preflight(client)
        for name in suites:
            print(f"\n=== suite: {name} ===", flush=True)
            t0 = time.monotonic()
            rows = await SUITES[name](client, args)
            report["suites"][name] = {
                "wall": round(time.monotonic() - t0, 1), "rows": rows}

    # 汇总
    hard_fail = 0
    print("\n=== 汇总 ===")
    for name, block in report["suites"].items():
        rows = block["rows"]
        if name in ("toolloop", "concurrency"):
            bad = [r for r in rows if r.get("failures")]
            print(f"{name}: {len(rows) - len(bad)}/{len(rows)} 路全绿, "
                  f"wall={block['wall']}s")
            for r in bad:
                print(f"  FAIL {r['tag']}: {r['failures']}")
            hard_fail += len(bad)
        else:
            bad = [r for r in rows if not r.get("ok")]
            print(f"{name}: {len(rows) - len(bad)}/{len(rows)} 格通过, "
                  f"wall={block['wall']}s")
            for r in bad:
                key = r.get("case", r.get("pass"))
                print(f"  FAIL {key}: {r.get('errors')}")
            hard_fail += len(bad)
    report["hard_failures"] = hard_fail
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"\n报告已写入 {args.json}")
    print(f"\nhard_failures={hard_fail}")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
