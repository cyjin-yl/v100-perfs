#!/usr/bin/env python3
"""Hermes WeChat tool-call fidelity benchmark.

Replays real tool-call turns from the Hermes WeChat session
(session 20260704_201000_4181eaeb, stored on the hermes host) against
each provider and scores tool-call validity:

  name_valid      tool name in Hermes's registered tool set
  name_repaired   name resolvable via Hermes's repair (prefix/fuzzy)
  args_valid      arguments parse as JSON
  required_ok     required fields present for known tools

Providers: primary FastLLM proxy (qwen3.6-27b-awq) and the two NVIDIA
fallbacks that produced the malformed calls (z-ai/glm-5.2,
minimaxai/minimax-m3).

Usage:
  python3 hermes_toolcall_bench.py --ssh-host hermes --turns 6
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time
import urllib.request
from typing import Any

HERMES_TOOLS = {
    "write_file": {"required": ["path", "content"]},
    "read_file": {"required": ["path"]},
    "terminal": {"required": ["command"]},
    "execute_code": {"required": ["code"]},
    "session_search": {"required": ["query"]},
    "web_search": {"required": ["query"]},
    "kanban_create": {"required": []},
    "kanban_list": {"required": []},
    "todo_write": {"required": []},
}
SESSION_ID = "20260704_201000_4181eaeb"
DB = "/home/ezra/.hermes/state.db"

PROVIDERS = {
    "fastllm-proxy": {
        "base": "http://127.0.0.1:8000/v1",
        "model": "qwen3.6-27b-awq",
        "key_env": "AUTH_TOKEN",
    },
    "nvidia-glm52": {
        "base": "https://integrate.api.nvidia.com/v1",
        "model": "z-ai/glm-5.2",
        "key_env": "NVIDIA_API_KEY",
    },
    "nvidia-minimax3": {
        "base": "https://integrate.api.nvidia.com/v1",
        "model": "minimaxai/minimax-m3",
        "key_env": "NVIDIA_API_KEY",
    },
}


def _ssh(host: str, cmd: str) -> str:
    r = subprocess.run(["ssh", host, cmd], capture_output=True, text=True,
                       timeout=120)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-200:])
    return r.stdout


def fetch_turns(host: str, n: int) -> list[dict]:
    """Fetch the last n real user messages (skipping tool-result payloads)."""
    q = (
        "SELECT content FROM messages WHERE "
        "session_id='" + SESSION_ID + "' AND role='user' AND content IS NOT NULL "
        "AND content != '' AND content NOT LIKE '{' || '%' "
        "ORDER BY timestamp DESC LIMIT " + str(n * 4) + ";"
    )
    rows = _ssh(host, "sqlite3 " + DB + " \"" + q + "\"")
    turns = []
    for line in rows.strip().split("\n"):
        content = line.strip()
        if content and len(content) > 4:
            turns.append({"user": content[:800]})
            if len(turns) >= n:
                break
    turns.reverse()
    return turns


def repair_name(name: str, valid: set[str]) -> str | None:
    if not name:
        return None
    lowered = name.lower().strip()
    for sep in ('"', "'", "<", ">"):
        idx = lowered.find(sep)
        if idx > 0:
            lowered = lowered[:idx]
    if lowered in valid:
        return lowered
    norm = lowered.replace("-", "_").replace(" ", "_")
    if norm in valid:
        return norm
    if len(lowered) >= 4:
        prefixed = [n for n in valid if n.startswith(lowered)]
        if len(prefixed) == 1:
            return prefixed[0]
    return None


def call_provider(provider: dict, messages: list, api_keys: dict,
                  timeout: int = 300) -> dict:
    key = api_keys.get(provider["key_env"], "")
    tools = [{"type": "function", "function": {
        "name": name, "description": f"Tool {name}",
        "parameters": {"type": "object",
                       "properties": {r: {"type": "string"}
                                      for r in reqs},
                       "required": reqs}}}
        for name, meta in HERMES_TOOLS.items()
        for reqs in [meta["required"]]]
    body = json.dumps({"model": provider["model"], "max_tokens": 300,
                       "temperature": 0, "tools": tools,
                       "messages": messages}).encode()
    req = urllib.request.Request(
        f"{provider['base']}/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return {"response": out, "wall": time.time() - t0}


def score(provider_name: str, response: dict) -> dict:
    valid_names = set(HERMES_TOOLS)
    try:
        choice = response["response"]["choices"][0]
        msg = choice.get("message") or {}
        tcs = msg.get("tool_calls") or []
    except Exception:
        return {"provider": provider_name, "error": "bad response"}
    scores = []
    for tc in tcs:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "")
        try:
            args = json.loads(raw_args) if raw_args else {}
            args_ok = isinstance(args, dict)
        except json.JSONDecodeError:
            args = {}
            args_ok = False
        repaired = repair_name(name, valid_names)
        required_ok = False
        if repaired and repaired in HERMES_TOOLS:
            required_ok = all(k in args for k in HERMES_TOOLS[repaired]["required"])
        scores.append({
            "name": name,
            "name_valid": name in valid_names,
            "name_repaired": repaired,
            "args_valid": args_ok,
            "required_ok": required_ok,
        })
    return {"provider": provider_name,
            "finish": choice.get("finish_reason"),
            "wall_s": round(response["wall"], 1),
            "tool_calls": scores}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh-host", default="hermes")
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    turns = fetch_turns(args.ssh_host, args.turns)
    print(f"replaying {len(turns)} real WeChat turns", flush=True)

    # api keys: primary from local .env, nvidia from hermes .env
    import os
    import pathlib
    import re
    env = pathlib.Path("/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/.env").read_text()
    keys = {"AUTH_TOKEN": re.search(
        r'^\s*AUTH_TOKEN\s*=\s*["\']?([^"\'\n]+)', env, re.M).group(1)}
    keys["NVIDIA_API_KEY"] = _ssh(args.ssh_host,
        "grep -oE '^NVIDIA_API_KEY=.+' /home/ezra/.hermes/.env | cut -d= -f2").strip()

    results = {p: [] for p in PROVIDERS}
    for i, turn in enumerate(turns):
        for pname, prov in PROVIDERS.items():
            system = ("You are Hermes, a helpful agent with tools. "
                      "Prefer calling a tool over answering from memory.")
            if pname == "fastllm-proxy":
                system = "<|think_off|>" + system
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": turn["user"]}]
            try:
                resp = call_provider(prov, messages, keys)
                results[pname].append(score(pname, resp))
            except Exception as exc:
                results[pname].append({"provider": pname,
                                       "error": f"{type(exc).__name__}: "
                                                f"{str(exc)[:120]}"})
            print(f"turn {i+1}/{len(turns)} {pname}: "
                  f"{json.dumps(results[pname][-1], ensure_ascii=False)[:160]}",
                  flush=True)

    summary = {
        "schema_version": 1,
        "benchmark": "hermes_wechat_toolcall_fidelity",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session_id": SESSION_ID,
        "turns": len(turns),
        "providers": {},
    }
    for pname, entries in results.items():
        calls = [s for e in entries for s in e.get("tool_calls", [])]
        errors = sum(1 for e in entries if "error" in e)
        summary["providers"][pname] = {
            "errors": errors,
            "tool_calls": len(calls),
            "name_valid": sum(1 for c in calls if c["name_valid"]),
            "name_repaired": sum(1 for c in calls if c["name_repaired"]),
            "args_valid": sum(1 for c in calls if c["args_valid"]),
            "required_ok": sum(1 for c in calls if c["required_ok"]),
        }
    out_path = pathlib.Path(args.output) if args.output else pathlib.Path(
        "/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/v100-perfs/benchmarks/"
        f"fastllm/results/hermes_wechat_toolcall_fidelity_{time.strftime('%Y%m%d_%H%M%S')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
