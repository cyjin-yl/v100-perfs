#!/usr/bin/env python3
"""Agentic workload benchmark: N concurrent agents doing multi-round tool
loops through the proxy, with a SHARED system prompt (prefix cache).

Simulates the production shape: 1-2 main agents + 5-10 subagents, each
round = chat(request) -> structured tool_call -> fake tool result -> next.

Metrics per request: wall, prompt_tokens, evaluated_tokens (prefix hits),
finish_reason, tool_calls count. Plus proxy concurrency samples.

Usage:
  python3 agentic_load_bench.py --main-agents 2 --subagents 8 --rounds 4 \
      --output results/agentic_load_2m8s_4r.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import re
import threading
import time
import urllib.request
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROXY_HEALTH = "http://127.0.0.1:8000/health"

SYSTEM_PROMPT = (
    "You are a helpful coding agent. You have access to tools. "
    "Prefer calling a tool over answering from memory when the user asks "
    "you to execute something. Follow the tool call format exactly. "
    + ("The quick brown fox jumps over the lazy dog. " * 60)
    + "Never invent tool outputs; wait for the real tool result."
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command and return its output",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string",
                            "description": "The shell command to run"}
            },
            "required": ["command"],
        },
    },
}]


def _post(endpoint: str, api_key: str, body: dict,
          timeout: int = 300) -> tuple[dict, float]:
    req = urllib.request.Request(
        f"{endpoint}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, time.time() - t0


def agent_loop(worker_id: int, rounds: int, endpoint: str, api_key: str,
               model: str, results: list, errors: list) -> None:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for rnd in range(rounds):
        tag = f"agent{worker_id}-round{rnd}"
        messages.append({"role": "user", "content": (
            f"用 bash 工具运行命令：echo {tag}，然后把输出报告给我。")})
        body = {
            "model": model, "stream": False, "max_tokens": 256,
            "temperature": 0, "tools": TOOLS, "messages": messages,
        }
        try:
            out, wall = _post(endpoint, api_key, body)
            choice = out["choices"][0]
            msg = choice["message"]
            tool_calls = msg.get("tool_calls") or []
            usage = out.get("usage") or {}
            results.append({
                "worker": worker_id, "round": rnd,
                "wall_s": round(wall, 2),
                "prompt_tokens": usage.get("prompt_tokens"),
                "evaluated_tokens": usage.get("prompt_tokens_evaluated"),
                "completion_tokens": usage.get("completion_tokens"),
                "finish": choice.get("finish_reason"),
                "tool_calls": len(tool_calls),
                "tool_args_ok": bool(tool_calls and tool_calls[0].get(
                    "function", {}).get("arguments")),
            })
            # append assistant message + fake tool result
            messages.append(msg)
            for tc in tool_calls:
                messages.append({"role": "tool",
                                 "tool_call_id": tc.get("id", f"id{rnd}"),
                                 "content": f"{tag}"})
        except Exception as exc:
            errors.append({"worker": worker_id, "round": rnd,
                           "error": f"{type(exc).__name__}: {str(exc)[:120]}"})


def sample_concurrency(stop_event: threading.Event,
                       samples: list) -> None:
    while not stop_event.is_set():
        try:
            with urllib.request.urlopen(PROXY_HEALTH, timeout=3) as r:
                lc = json.loads(r.read()).get("lifecycle", {})
            samples.append((lc.get("active", 0), lc.get("pending", 0)))
        except Exception:
            pass
        time.sleep(1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-agents", type=int, default=2)
    parser.add_argument("--subagents", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key-env", default="AUTH_TOKEN")
    parser.add_argument("--model", default="qwen3.6-27b-heretic")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "")

    def _hit_pages() -> float:
        try:
            with urllib.request.urlopen(
                    "http://127.0.0.1:8002/props", timeout=30) as r:
                return float(json.loads(r.read()).get(
                    "prefix_cache_gpu_hit_pages", 0))
        except Exception:
            return 0.0

    hit_pages_before = _hit_pages()
    total_workers = args.main_agents + args.subagents
    results: list[dict] = []
    errors: list[dict] = []
    samples: list[tuple[int, int]] = []
    stop = threading.Event()
    sampler = threading.Thread(target=sample_concurrency, args=(stop, samples))
    sampler.start()

    t0 = time.time()
    threads = []
    for w in range(total_workers):
        t = threading.Thread(target=agent_loop,
                             args=(w, args.rounds, args.endpoint, api_key,
                                   args.model, results, errors))
        t.start()
        threads.append(t)
    deadline = t0 + args.timeout
    for t in threads:
        t.join(max(0, deadline - time.time()))
    stop.set()
    sampler.join(timeout=2)
    wall_total = time.time() - t0

    # The backend does not emit prompt_tokens_evaluated; estimate prefix
    # hits from wall time: a request finishing well under the first
    # (cold-prefill) request's time must have hit the shared prefix.
    first_wall = min((r["wall_s"] for r in results), default=0.0)
    hit_cutoff = max(3.0, first_wall * 0.35)
    hits = [r for r in results
            if r.get("prompt_tokens", 0) > 500 and r["wall_s"] < hit_cutoff]
    hit_pages_delta = _hit_pages() - hit_pages_before
    wall_s = sorted(r["wall_s"] for r in results)
    act = [s[0] for s in samples]
    pend = [s[1] for s in samples]
    summary = {
        "schema_version": 1,
        "benchmark": "fastllm_agentic_load",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workload": {"main_agents": args.main_agents,
                     "subagents": args.subagents,
                     "rounds": args.rounds,
                     "total_requests_expected":
                         total_workers * args.rounds},
        "endpoint": args.endpoint,
        "model": args.model,
        "results": {
            "requests_completed": len(results),
            "errors": len(errors),
            "error_samples": errors[:5],
            "wall_total_s": round(wall_total, 1),
            "request_p50_s": wall_s[len(wall_s) // 2] if wall_s else None,
            "request_p95_s": wall_s[int(len(wall_s) * 0.95) - 1] if wall_s else None,
            "request_max_s": wall_s[-1] if wall_s else None,
            "prefix_hit_rate": round(len(hits) / len(results), 3)
                if results else 0,
            "prefix_gpu_hit_pages_delta": round(hit_pages_delta, 1),
            "first_request_wall_s": round(first_wall, 2),
            "tool_call_rate": round(
                sum(1 for r in results if r.get("tool_calls")) / len(results), 3)
                if results else 0,
            "tool_args_ok_rate": round(
                sum(1 for r in results if r.get("tool_args_ok")) / len(results), 3)
                if results else 0,
            "concurrency_peak_active": max(act) if act else 0,
            "concurrency_mean_active": round(sum(act) / len(act), 2) if act else 0,
            "concurrency_peak_pending": max(pend) if pend else 0,
        },
        "requests": results,
    }
    out_path = pathlib.Path(args.output) if args.output else (
        REPO_ROOT / "benchmarks" / "fastllm" / "results" /
        f"agentic_load_{args.main_agents}m{args.subagents}s_"
        f"{args.rounds}r_{time.strftime('%Y%m%d_%H%M%S')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"done: {len(results)}/{total_workers*args.rounds} ok, "
          f"{len(errors)} errors, hit={summary['results']['prefix_hit_rate']}, "
          f"peak_active={summary['results']['concurrency_peak_active']}, "
          f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
