#!/usr/bin/env python3
"""Long-context KV recall A/B test: turbo3 vs turbo4.

Embeds N unique key->value facts at spread positions inside a long filler
prompt, then asks the model to recall specific facts. Scores exact-match
recall per fact and writes a structured JSON result.

Usage:
  python3 kv_recall_ab.py --endpoint http://127.0.0.1:8000 \
      --api-key-env AUTH_TOKEN --model qwen3.6-fastllm \
      --facts 24 --span 200000 --label turbo4
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.request
from pathlib import Path

FILLER = (
    "The quick brown fox jumps over the lazy dog while the analysis continues. "
    "Numbers and trends evolve across sections without repetition of key facts. "
)


def make_facts(n: int, seed: int = 42) -> dict[str, str]:
    rng = random.Random(seed)
    facts = {}
    for i in range(n):
        key = f"k{seed}_{i:03d}"
        value = f"VALUE{rng.randint(0, 999999):06d}"
        facts[key] = value
    return facts


def build_prompt(facts: dict[str, str], span: int) -> tuple[str, dict[int, str]]:
    """Interleave facts into a long filler prompt; returns (prompt, pos->key)."""
    filler_tokens = len(FILLER.split())
    per_gap = max(4, span // (len(facts) + 1) // filler_tokens)
    parts: list[str] = []
    positions: dict[int, str] = {}
    tokens_so_far = 0
    for idx, (key, value) in enumerate(facts.items()):
        gap = FILLER * per_gap
        parts.append(gap)
        tokens_so_far += len(gap.split())
        fact_line = f"Remember: the code {key} maps to {value}."
        parts.append(fact_line)
        positions[idx] = key
        tokens_so_far += len(fact_line.split())
    parts.append(FILLER * per_gap)
    return "".join(parts), positions


def chat(endpoint: str, api_key: str, model: str, messages: list,
         max_tokens: int = 64, timeout: int = 900) -> dict:
    body = json.dumps({
        "model": model, "stream": False, "max_tokens": max_tokens,
        "temperature": 0, "messages": messages,
    }).encode()
    req = urllib.request.Request(
        f"{endpoint}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    out["_wall"] = time.time() - t0
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key-env", default="AUTH_TOKEN")
    parser.add_argument("--model", default="qwen3.6-fastllm")
    parser.add_argument("--facts", type=int, default=24)
    parser.add_argument("--span", type=int, default=200000,
                        help="approximate prompt length in tokens")
    parser.add_argument("--queries", type=int, default=12)
    parser.add_argument("--label", default="unknown")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    facts = make_facts(args.facts)
    prompt, _ = build_prompt(facts, args.span)
    print(f"prompt ~{len(prompt.split())} tokens, facts={len(facts)}", flush=True)

    rng = random.Random(7)
    queried = rng.sample(sorted(facts), min(args.queries, len(facts)))
    results = []
    t0 = time.time()
    for key in queried:
        expected = facts[key]
        try:
            resp = chat(
                args.endpoint, api_key, args.model,
                [{"role": "system", "content": "<|think_off|>Answer with ONLY the requested value, no explanation."},
                 {"role": "user", "content": (
                     prompt + f"\nWhat value does the code {key} map to? "
                     "Answer with ONLY the value.")}],
                max_tokens=64)
            content = resp["choices"][0]["message"].get("content", "")
            match = expected in content
            results.append({
                "key": key, "expected": expected,
                "got": content.strip()[:80], "match": match,
                "wall_s": round(resp.get("_wall", 0), 1),
                "prompt_tokens": resp.get("usage", {}).get("prompt_tokens"),
                "evaluated": resp.get("usage", {}).get(
                    "prompt_tokens_evaluated"),
            })
        except Exception as exc:
            results.append({"key": key, "expected": expected,
                            "error": f"{type(exc).__name__}: {str(exc)[:100]}"})
        print(f"{key}: {'HIT' if results[-1].get('match') else 'miss'} "
              f"({len(results)}/{len(queried)})", flush=True)

    hits = sum(1 for r in results if r.get("match"))
    summary = {
        "schema_version": 1,
        "benchmark": "fastllm_kv_recall_ab",
        "label": args.label,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoint": args.endpoint,
        "model": args.model,
        "facts_total": len(facts),
        "queries": len(queried),
        "span_tokens_approx": len(prompt.split()),
        "hits": hits,
        "recall_rate": round(hits / len(queried), 3) if queried else 0,
        "wall_total_s": round(time.time() - t0, 1),
        "results": results,
    }
    out_path = Path(args.output) if args.output else Path(
        f"/tmp/kv_recall_{args.label}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"recall {hits}/{len(queried)} -> wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
