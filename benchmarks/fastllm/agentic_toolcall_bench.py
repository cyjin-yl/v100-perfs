#!/usr/bin/env python3
"""Agentic tool-call benchmark against the local FastLLM proxy.

Runs verifiable multi-step tasks through omp headless (or opencode) in a
sandbox directory and records three optimization-facing metric groups:

1. Parallelism      - proxy active/pending concurrency sampled during the run
2. Prefix cache     - backend /props deltas: GPU hit pages, disk hits,
                      restore hits, recompute tokens/s
3. Agent behavior   - tool invocations counted from the tmux transcript:
                      write/read/bash/edit calls, tool-call blocks,
                      error marks, recap lines

Usage:
  python3 agentic_toolcall_bench.py --sandbox /tmp/agentbench \
      --runner omp --tasks file_roundtrip,fib_run
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.request
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # v100-perfs
BENCH_SESSION = "agentbench"
PROXY_HEALTH = "http://127.0.0.1:8000/health"
BACKEND_PROPS = "http://127.0.0.1:8002/props"

CACHE_KEYS = (
    "prefix_cache_gpu_hit_pages",
    "prefix_cache_disk_hits",
    "prefix_cache_disk_read_bytes",
    "prefix_cache_persistence_restore_hits",
    "prefix_cache_recompute_tokens_per_second",
)


def _read_file(path: pathlib.Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


def verify_file_roundtrip(sandbox: pathlib.Path) -> tuple[bool, str]:
    content = _read_file(sandbox / "roundtrip.txt")
    ok = content is not None and content.strip() == "42"
    return ok, f"roundtrip.txt={content!r}" if not ok else "roundtrip.txt=42"


def verify_fib_run(sandbox: pathlib.Path) -> tuple[bool, str]:
    script = sandbox / "fib.py"
    if not script.exists():
        return False, "fib.py missing"
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=60
    )
    ok = "6765" in result.stdout
    return ok, f"stdout={result.stdout.strip()!r}" if not ok else "fib(20)=6765"


def verify_multi_file(sandbox: pathlib.Path) -> tuple[bool, str]:
    a = _read_file(sandbox / "a.txt")
    b = _read_file(sandbox / "b.txt")
    ok = (a is not None and b is not None
          and "hello" in (a or "") and "world" in (b or ""))
    return ok, f"a={a!r} b={b!r}" if not ok else "a/b ok"


def verify_count_py(sandbox: pathlib.Path) -> tuple[bool, str]:
    probe = sandbox / "count_probe"
    if not probe.exists():
        return False, "count_probe dir missing"
    expected = len([p for p in probe.rglob("*.py") if p.is_file()])
    answer = _read_file(sandbox / "count.txt")
    try:
        got = int((answer or "").strip())
    except ValueError:
        return False, f"count.txt={answer!r}"
    ok = got == expected
    return ok, f"expected={expected} got={got}" if not ok else f"count={expected}"


TASKS: dict[str, dict[str, Any]] = {
    "file_roundtrip": {
        "prompt": (
            "在当前目录创建文件 roundtrip.txt，内容只写数字 42，"
            "然后读回该文件并把读到的内容原样告诉我。"
        ),
        "verify": verify_file_roundtrip,
    },
    "fib_run": {
        "prompt": (
            "在当前目录写一个 fib.py 脚本，计算斐波那契第 20 项并 print 出来，"
            "然后实际运行它，把运行输出告诉我。"
        ),
        "verify": verify_fib_run,
    },
    "multi_file": {
        "prompt": (
            "在当前目录创建 a.txt（内容 hello）和 b.txt（内容 world）两个文件，"
            "然后用一条命令同时显示两个文件的内容，把结果告诉我。"
        ),
        "verify": verify_multi_file,
    },
    "count_py": {
        "prompt": (
            "先创建目录 count_probe，在里面创建 3 个 .py 文件和 2 个 .txt 文件，"
            "然后统计 count_probe 目录下 .py 文件的数量，"
            "把统计结果写进 count.txt（只写数字），并告诉我结果。"
        ),
        "verify": verify_count_py,
    },
}


def fetch_backend_props() -> dict[str, Any]:
    try:
        with urllib.request.urlopen(BACKEND_PROPS, timeout=30) as r:
            props = json.loads(r.read())
        return {k: props.get(k, 0) for k in CACHE_KEYS}
    except Exception:
        return {k: 0 for k in CACHE_KEYS}


def fetch_health() -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(PROXY_HEALTH, timeout=5) as r:
            h = json.loads(r.read())
        return h.get("lifecycle", {})
    except Exception:
        return None


def _tmux(*args: str) -> str:
    result = subprocess.run(["tmux", *args], capture_output=True, text=True)
    return result.stdout


def _shq(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


def analyze_transcript(transcript: str) -> dict[str, Any]:
    """Count tool invocations visible in the omp pane transcript."""
    return {
        "write_calls": len(re.findall(r"Write[:\u2026]? [^\n]*", transcript)),
        "read_calls": len(re.findall(r"Read [^\n]*", transcript)),
        "bash_calls": len(re.findall(
            r"(?:Run|\$) [^\n]*(?:command|mkdir|ls|cat|echo|python)", transcript)),
        "edit_calls": len(re.findall(r"Edit [^\n]*", transcript)),
        "task_subagent_spawns": len(re.findall(
            r"task[^\n]*(?:subagent|agent|Launch)", transcript)),
        "error_marks": len(re.findall(
            r"(?:SchemaError|invalid arguments|failed|Error)", transcript)),
        "recap_lines": len(re.findall(r"recap:", transcript)),
        "tool_call_blocks": len(re.findall(r"<tool_call>", transcript)),
    }


def run_task(task_name: str, prompt: str, sandbox: pathlib.Path,
             model: str, runner: str, timeout: int) -> dict[str, Any]:
    task_dir = sandbox / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    window = f"task-{task_name}"
    _tmux("kill-window", "-t", f"{BENCH_SESSION}:{window}")
    _tmux("new-session", "-d", "-s", BENCH_SESSION, "-n", window,
          "-c", str(task_dir))

    if runner == "omp":
        # omp v17 已无 --auto-approve; 用 --plan-yolo 让模型在计划批准后
        # 自动开始实现(等效于旧版的自动批准工作流)。
        shell = (
            f"omp --model {model} --cwd {task_dir} "
            f"--tools bash,read,write,edit,glob,grep,task "
            f"--plan-yolo --plan-yolo-into {model} "
            f"{_shq(prompt)}"
        )
    elif runner == "claude":
        # Claude Code 走 Anthropic /v1/messages。用 .env 里的 AUTH_TOKEN 指向
        # 本网关; ANTHROPIC_API_KEY 置空避免它去要云端 key。
        # --dangerously-skip-permissions 让 headless 跑工具不再逐个要批准。
        # 这条路径同时真实检验"剥离 Claude Code 归因头"——Claude Code 每请求
        # 都会发那行会变的 x-anthropic-billing-header, 网关剥掉后前缀才稳定。
        shell = (
            f"set -a; . {REPO_ROOT.parent}/.env; set +a; "
            f"ANTHROPIC_BASE_URL=http://127.0.0.1:8000 "
            f"ANTHROPIC_AUTH_TOKEN=\"$AUTH_TOKEN\" "
            f"ANTHROPIC_API_KEY=\"\" "
            f"ANTHROPIC_MODEL={model} "
            f"claude -p {_shq(prompt)} --model {model} "
            f"--dangerously-skip-permissions"
        )
    else:
        shell = f"opencode run --model {model} -- {_shq(prompt)}"
    _tmux("send-keys", "-t", f"{BENCH_SESSION}:{window}", shell, "Enter")

    props_before = fetch_backend_props()
    started = time.time()
    deadline = started + timeout

    active_samples: list[int] = []
    pending_samples: list[int] = []
    transcript = ""
    while time.time() < deadline:
        health = fetch_health()
        if health:
            active_samples.append(health.get("active", 0))
            pending_samples.append(health.get("pending", 0))
        transcript = _tmux("capture-pane", "-t", f"{BENCH_SESSION}:{window}", "-p")
        try:
            last_verify, _ = TASKS[task_name]["verify"](task_dir)
        except Exception:
            last_verify = False
        # verify 通过即视为任务完成。不再死等 "recap" 关键字——任务 prompt
        # 里从没要求过它, 导致每次都空等到 deadline(实测 890s 外部超时还
        # 比 900s deadline 短, 直接杀掉进程、一行结果都没输出)。
        # 给 5s 缓冲抓一次尾部转录就收。
        if last_verify:
            time.sleep(5)
            transcript = _tmux(
                "capture-pane", "-t", f"{BENCH_SESSION}:{window}", "-p")
            break
        dead = _tmux("list-panes", "-t", f"{BENCH_SESSION}:{window}",
                     "-F", "#{pane_dead}").strip()
        if dead == "1":
            break
        time.sleep(5)

    wall = time.time() - started
    transcript = _tmux("capture-pane", "-t", f"{BENCH_SESSION}:{window}", "-p")
    _tmux("kill-window", "-t", f"{BENCH_SESSION}:{window}")

    ok, detail = TASKS[task_name]["verify"](task_dir)
    props_after = fetch_backend_props()
    cache_delta = {k: round(props_after.get(k, 0) - props_before.get(k, 0), 3)
                   for k in CACHE_KEYS}

    peak_active = max(active_samples) if active_samples else 0
    mean_active = (sum(active_samples) / len(active_samples)
                   if active_samples else 0)
    peak_pending = max(pending_samples) if pending_samples else 0

    return {
        "task": task_name,
        "runner": runner,
        "status": "ok" if ok else ("timeout" if wall >= timeout else "failed"),
        "verified": ok,
        "detail": detail,
        "wall_s": round(wall, 1),
        "parallelism": {
            "peak_active": peak_active,
            "mean_active": round(mean_active, 2),
            "peak_pending": peak_pending,
            "samples": len(active_samples),
        },
        "prefix_cache_delta": cache_delta,
        "agent": analyze_transcript(transcript),
        "output_tail": transcript[-400:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox", default="/tmp/agentbench")
    parser.add_argument("--runner",
                        choices=["omp", "opencode", "claude"], default="omp")
    parser.add_argument("--model", default=None)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    model = args.model or (
        "vllm/qwen3.6-27b-awq" if args.runner == "omp"
        else "catvllm/qwen3.6-27b-awq")

    sandbox = pathlib.Path(args.sandbox)
    sandbox.mkdir(parents=True, exist_ok=True)
    names = [n for n in args.tasks.split(",") if n in TASKS]
    results = []
    for name in names:
        print(f"[{name}] running...", flush=True)
        results.append(run_task(name, TASKS[name]["prompt"], sandbox,
                                model, args.runner, args.timeout))
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)

    summary = {
        "schema_version": 1,
        "benchmark": "fastllm_agentic_toolcall",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "runner": args.runner,
        "sandbox": str(sandbox),
        "results": results,
        "summary": {
            "total": len(results),
            "verified_ok": sum(1 for r in results if r.get("verified")),
            "gpu_hit_pages_added": sum(
                r.get("prefix_cache_delta", {}).get(
                    "prefix_cache_gpu_hit_pages", 0) for r in results),
            "error_marks": sum(
                r.get("agent", {}).get("error_marks", 0) for r in results),
        },
    }
    out_path = pathlib.Path(args.output) if args.output else (
        REPO_ROOT / "benchmarks" / "fastllm" / "results" /
        f"fastllm_agentic_toolcall_{args.runner}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
