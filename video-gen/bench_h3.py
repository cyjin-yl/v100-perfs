#!/usr/bin/env python3
"""Reproducible MiniMax H3 ComfyUI benchmark for a single V100.

The harness uses the official ComfyUI API graph, submits one job at a time,
polls queue/history, samples ComfyUI's torch VRAM telemetry, and writes JSON
artifacts. It intentionally reports wall-clock generation time (including
model loading, text encoding, decode, and video muxing) as the primary user
metric; optional denoise/progress timings can be added by a future ComfyUI
websocket adapter.

Examples:
  python bench_h3.py --workflow graph.json --matrix resolution --steps 1 --output results/res.json
  python bench_h3.py --workflow graph.json --matrix duration --steps 1 --output results/duration.json
  python bench_h3.py --workflow graph.json --matrix aspect --steps 1 --output results/aspect.json
  python bench_h3.py --workflow graph.json --matrix anchor --steps 20 --output results/anchor_20.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

FPS = 24
PROMPT_NODE = "104"
SCHEDULER_NODE = "9"
NOISE_NODE = "15"
SAVE_NODE = "92"

# Official H3 16:9 examples, rounded to the model's 32-pixel grid.  The 1.0
# point is capped at the native 768 x 1344 area budget.
RESOLUTION_POINTS = [
    (0.1, 416, 224),
    (0.2, 608, 352),
    (0.3, 736, 416),
    (0.4, 864, 480),
    (0.5, 960, 544),
    (0.6, 1056, 608),
    (0.7, 1152, 640),
    (0.8, 1216, 672),
    (0.9, 1280, 736),
    (1.0, 1344, 768),
]

# Common landscape, square, and portrait formats at approximately 0.4 MP.
# Dimensions stay on ComfyUI's 32-pixel grid.
ASPECT_POINTS = [
    ("16:9", 864, 480),
    ("4:3", 736, 544),
    ("1:1", 640, 640),
    ("3:4", 544, 736),
    ("9:16", 480, 864),
]


def align_frames(requested_seconds: float) -> int:
    """Mirror ComfyUI's align_frame_count(max(5, length)) rule."""
    requested = max(5, math.ceil(requested_seconds * FPS))
    return requested + ((5 - requested) % 17)


def actual_seconds(frames: int) -> float:
    return frames / FPS


def api_json(
    base: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    request_timeout: float = 120.0,
) -> dict[str, Any]:
    url = base.rstrip("/") + path
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=max(0.1, request_timeout)) as response:
        raw = response.read()
    return json.loads(raw or b"{}")


def stats(
    base: str,
    *,
    request_timeout: float = 120.0,
) -> dict[str, Any]:
    try:
        return api_json(
            base,
            "/system_stats",
            request_timeout=request_timeout,
        )
    except Exception as exc:  # telemetry must never abort a generation
        return {"error": str(exc)}


def require_free_vram(base: str, minimum_bytes: int) -> int:
    free_bytes = _free_vram(api_json(base, "/system_stats"))
    if free_bytes is None:
        raise RuntimeError("ComfyUI did not report free device VRAM")
    if free_bytes < minimum_bytes:
        free_gib = free_bytes / 1024**3
        required_gib = minimum_bytes / 1024**3
        raise RuntimeError(
            f"GPU is in use: {free_gib:.2f} GiB free; "
            f"{required_gib:.2f} GiB required"
        )
    return free_bytes


def submit(
    base: str,
    graph: dict[str, Any],
    client_id: str,
    *,
    request_timeout: float = 120.0,
) -> dict[str, Any]:
    return api_json(
        base,
        "/prompt",
        {"prompt": graph, "client_id": client_id},
        request_timeout=request_timeout,
    )

def queue_state(base: str, *, request_timeout: float = 120.0) -> dict[str, Any]:
    state = api_json(base, "/queue", request_timeout=request_timeout)
    if not all(
        isinstance(state.get(key), list)
        for key in ("queue_running", "queue_pending")
    ):
        raise RuntimeError("ComfyUI /queue response is missing queue lists")
    return state


def prompt_is_running(payload: dict[str, Any], prompt_id: str) -> bool:
    return any(
        isinstance(item, (list, tuple))
        and len(item) > 1
        and item[1] == prompt_id
        for item in payload.get("queue_running", [])
    )


def prompt_is_queued(payload: dict[str, Any], prompt_id: str) -> bool:
    return any(
        isinstance(item, (list, tuple))
        and len(item) > 1
        and item[1] == prompt_id
        for key in ("queue_running", "queue_pending")
        for item in payload.get(key, [])
    )


def cancel_prompt(base: str, prompt_id: str, *, poll: float,
                  timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    request_timeout = min(30.0, max(0.1, timeout))
    api_json(
        base,
        "/interrupt",
        {"prompt_id": prompt_id},
        request_timeout=request_timeout,
    )
    api_json(
        base,
        "/queue",
        {"delete": [prompt_id]},
        request_timeout=request_timeout,
    )
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"prompt {prompt_id} did not stop within {timeout:g} seconds"
            )
        if not prompt_is_queued(
            queue_state(base, request_timeout=min(30.0, remaining)),
            prompt_id,
        ):
            return True
        time.sleep(min(max(0.1, poll), remaining))


def wait_for_idle(base: str, poll: float, timeout: float) -> None:
    announced = False
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"ComfyUI queue did not become idle within {timeout:g} seconds"
            )
        state = queue_state(base, request_timeout=min(30.0, remaining))
        if not state.get("queue_running") and not state.get("queue_pending"):
            if announced:
                print("ComfyUI queue is idle; starting benchmark case", flush=True)
            return
        if not announced:
            print("Waiting for existing ComfyUI jobs before benchmark case", flush=True)
            announced = True
        time.sleep(min(max(1.0, poll), remaining))


def make_graph(template: dict[str, Any], *, width: int, height: int, frames: int,
               steps: int, seed: int, prefix: str) -> dict[str, Any]:
    graph = copy.deepcopy(template.get("prompt", template))
    graph[PROMPT_NODE]["inputs"].update({"width": width, "height": height, "length": frames})
    graph[SCHEDULER_NODE]["inputs"]["steps"] = steps
    graph[NOISE_NODE]["inputs"].update({"noise_seed": seed, "control_after_generate": "fixed"})
    graph[SAVE_NODE]["inputs"]["filename_prefix"] = prefix
    return graph


def extract_video(history: dict[str, Any]) -> dict[str, Any] | None:
    for node in history.get("outputs", {}).values():
        for key in ("gifs", "videos", "images"):
            for item in node.get(key, []) if isinstance(node, dict) else []:
                if isinstance(item, dict) and str(item.get("filename", "")).lower().endswith((".mp4", ".webm", ".mov")):
                    return item
    return None


def extract_node_errors(history: dict[str, Any]) -> dict[str, Any]:
    prompt = history.get("prompt")
    if not isinstance(prompt, dict):
        return {}
    errors = prompt.get("node_errors", {})
    return errors if isinstance(errors, dict) else {}


def execution_times(
    history: dict[str, Any],
) -> tuple[float | None, float | None]:
    status = history.get("status", {})
    messages = status.get("messages", []) if isinstance(status, dict) else []
    started: float | None = None
    finished: float | None = None
    terminal_events = {
        "execution_success",
        "execution_error",
        "execution_interrupted",
    }
    for message in messages:
        if not isinstance(message, (list, tuple)) or len(message) != 2:
            continue
        event, data = message
        if not isinstance(data, dict):
            continue
        timestamp = data.get("timestamp")
        if not isinstance(timestamp, (int, float)):
            continue
        seconds = timestamp / 1000
        if event == "execution_start" and started is None:
            started = seconds
        elif event in terminal_events:
            finished = seconds
    return started, finished


def one_run(base: str, template: dict[str, Any], case: dict[str, Any], *,
            poll: float, run_no: int, timeout: float,
            client_id: str | None = None,
            on_submit: Callable[[dict[str, Any]], None] | None = None,
            ) -> dict[str, Any]:
    seed = int(case.get("seed", 424242))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        f"{stamp}_{case['kind']}_{case['label']}_s{case['steps']}_{run_no:02d}"
    )
    graph = make_graph(
        template,
        width=int(case["width"]),
        height=int(case["height"]),
        frames=int(case["frames"]),
        steps=int(case["steps"]),
        seed=seed,
        prefix=f"video/MiniMaxH3_bench_{run_id}",
    )
    client_id = client_id or str(uuid.uuid4())
    before = stats(base)
    submitted = time.time()
    response = submit(
        base,
        graph,
        client_id,
        request_timeout=min(120.0, max(0.1, timeout)),
    )
    if "prompt_id" not in response:
        return {
            **case,
            "client_id": client_id,
            "run_id": run_id,
            "submitted": _iso(submitted),
            "submit_response": response,
            "status": "submit_error",
        }
    prompt_id = response["prompt_id"]
    if on_submit is not None:
        on_submit({
            "client_id": client_id,
            "run_id": run_id,
            "prompt_id": prompt_id,
            "submitted": _iso(submitted),
        })
    minimum_free = _free_vram(before)
    history: dict[str, Any] = {}
    last_stats = before
    poll_started: float | None = None
    last_poll_error: str | None = None
    deadline = time.monotonic() + timeout
    timed_out = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        time.sleep(min(max(0.1, poll), remaining))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        last_stats = stats(
            base,
            request_timeout=min(30.0, remaining),
        )
        minimum_free = _minimum_vram(minimum_free, _free_vram(last_stats))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        try:
            state = queue_state(
                base,
                request_timeout=min(30.0, remaining),
            )
            if poll_started is None and prompt_is_running(state, prompt_id):
                poll_started = time.time()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            history_all = api_json(
                base,
                f"/history/{prompt_id}",
                request_timeout=min(120.0, remaining),
            )
            last_poll_error = None
        except Exception as exc:
            last_poll_error = f"{type(exc).__name__}: {exc}"
            continue
        history = (
            history_all.get(prompt_id, {})
            if isinstance(history_all, dict)
            else {}
        )
        if history:
            break
    observed_finished = time.time()
    cancellation_dispatched: bool | None = None
    cancellation_error: str | None = None
    if timed_out:
        try:
            cancellation_dispatched = cancel_prompt(
                base,
                prompt_id,
                poll=poll,
                timeout=min(300.0, max(30.0, poll * 4)),
            )
        except Exception as exc:
            cancellation_error = f"{type(exc).__name__}: {exc}"
    after_stats = stats(base)
    history_started, history_finished = execution_times(history)
    if history_started is not None and history_finished is not None:
        execution_started = history_started
        finished = history_finished
        timing_source = "history_messages"
    else:
        execution_started = (
            None if timed_out else (poll_started or submitted)
        )
        finished = observed_finished
        timing_source = (
            "poll_fallback" if execution_started is not None else None
        )
    queue_wait_s = (
        round(max(0.0, execution_started - submitted), 3)
        if execution_started is not None
        else None
    )
    execution_wall_s = (
        round(max(0.0, finished - execution_started), 3)
        if execution_started is not None
        else None
    )
    status = history.get("status", {}) if isinstance(history, dict) else {}
    return {
        **case,
        "client_id": client_id,
        "run_id": run_id,
        "prompt_id": prompt_id,
        "submitted": _iso(submitted),
        "execution_started": (
            _iso(execution_started) if execution_started is not None else None
        ),
        "finished": _iso(finished),
        "timing_source": timing_source,
        "queue_wait_s": queue_wait_s,
        "execution_wall_s": execution_wall_s,
        "end_to_end_s": round(finished - submitted, 3),
        "wall_s": execution_wall_s,
        "status": (
            "timeout"
            if timed_out
            else status.get("status_str", "unknown")
        ),
        "completed": False if timed_out else status.get("completed"),
        "messages": status.get("messages", []),
        "node_errors": extract_node_errors(history),
        "video": extract_video(history),
        "vram_free_before": _free_vram(before),
        "vram_free_min_sampled": minimum_free,
        "vram_free_after": _free_vram(after_stats),
        "cancellation_dispatched": cancellation_dispatched,
        "cancellation_error": cancellation_error,
        "last_poll_error": last_poll_error,
        "last_poll_stats": last_stats,
        "system_stats": after_stats,
    }


def _free_vram(payload: dict[str, Any]) -> int | None:
    try:
        device = payload["devices"][0]
        raw = device.get("vram_free")
        if raw is None:
            raw = device.get("torch_vram_free")
        return int(raw) if raw is not None else None
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _minimum_vram(current: int | None, sample: int | None) -> int | None:
    if current is None:
        return sample
    if sample is None:
        return current
    return min(current, sample)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as target:
        json.dump(payload, target, ensure_ascii=False, indent=2)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def cases(matrix: str, steps: int, repeat: int) -> list[dict[str, Any]]:
    if steps < 1 or repeat < 1:
        raise ValueError("steps and repeat must both be positive")
    out: list[dict[str, Any]] = []
    if matrix in ("resolution", "all"):
        for mp, width, height in RESOLUTION_POINTS:
            for r in range(repeat):
                out.append({"kind": "resolution", "label": f"mp_{mp:.1f}", "target_mp": mp,
                            "width": width, "height": height, "requested_seconds": 2.0,
                            "frames": align_frames(2.0), "actual_seconds": actual_seconds(align_frames(2.0)),
                            "steps": steps, "repeat": r + 1})
    if matrix in ("duration", "all"):
        for sec in range(1, 16):
            frames = align_frames(float(sec))
            for r in range(repeat):
                out.append({"kind": "duration", "label": f"sec_{sec:02d}", "target_mp": 0.4,
                            "width": 864, "height": 480, "requested_seconds": sec,
                            "frames": frames, "actual_seconds": actual_seconds(frames),
                            "steps": steps, "repeat": r + 1})
    if matrix in ("aspect", "all"):
        frames = align_frames(2.0)
        for aspect_ratio, width, height in ASPECT_POINTS:
            for r in range(repeat):
                out.append({
                    "kind": "aspect",
                    "label": f"aspect_{aspect_ratio.replace(':', 'x')}",
                    "aspect_ratio": aspect_ratio,
                    "target_mp": 0.4,
                    "width": width,
                    "height": height,
                    "requested_seconds": 2.0,
                    "frames": frames,
                    "actual_seconds": actual_seconds(frames),
                    "steps": steps,
                    "repeat": r + 1,
                })
    if matrix in ("anchor", "all"):
        frames = align_frames(2.0)
        for r in range(repeat):
            out.append({
                "kind": "anchor",
                "label": "mp_0.4_sec_02",
                "target_mp": 0.4,
                "width": 864,
                "height": 480,
                "requested_seconds": 2.0,
                "frames": frames,
                "actual_seconds": actual_seconds(frames),
                "steps": steps,
                "repeat": r + 1,
            })
    return out


def run_succeeded(row: dict[str, Any]) -> bool:
    return row.get("status") == "success" and row.get("completed") is True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8188")
    ap.add_argument("--workflow", type=Path, required=True, help="successful API-format H3 T2V graph JSON")
    ap.add_argument(
        "--matrix",
        choices=("resolution", "duration", "aspect", "anchor", "all"),
        default="anchor",
    )
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument(
        "--timeout",
        type=float,
        default=21600.0,
        help="maximum seconds for one submitted generation",
    )
    ap.add_argument(
        "--idle-timeout",
        type=float,
        default=3600.0,
        help="maximum seconds to wait for an already-busy ComfyUI queue",
    )
    ap.add_argument(
        "--min-free-vram-gib",
        type=float,
        default=30.0,
        help="fail closed before submission when less device VRAM is free",
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    template = json.loads(args.workflow.read_text())
    todo = cases(args.matrix, args.steps, args.repeat)
    started = time.time()
    result = {
        "schema": "v100-perfs/minimax-h3/1",
        "started": _iso(started),
        "host": os.uname().nodename,
        "endpoint": args.endpoint,
        "workflow": str(args.workflow),
        "workflow_sha256": _sha256_file(args.workflow),
        "matrix": args.matrix,
        "steps": args.steps,
        "repeat": args.repeat,
        "timeout_seconds": args.timeout,
        "idle_timeout_seconds": args.idle_timeout,
        "min_free_vram_gib": args.min_free_vram_gib,
        "frame_rule": "ceil(max(5, requested_seconds*24)) rounded up to n % 17 == 5",
        "baseline_stats": stats(args.endpoint),
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(args.output, result)
    for index, case in enumerate(todo, 1):
        client_id = str(uuid.uuid4())
        result["active_case"] = {
            **case,
            "client_id": client_id,
            "state": "admission",
            "recorded": _iso(time.time()),
        }
        _write_json_atomic(args.output, result)
        failure_stage = "queue_admission"
        try:
            wait_for_idle(
                args.endpoint,
                args.poll,
                args.idle_timeout,
            )
            require_free_vram(
                args.endpoint,
                int(args.min_free_vram_gib * 1024**3),
            )
            failure_stage = "submit_or_execute"
            print(
                f"[{index}/{len(todo)}] {case['label']} "
                f"{case['width']}x{case['height']} {case['frames']}f "
                f"{args.steps} steps",
                flush=True,
            )
            def record_submission(identity: dict[str, Any]) -> None:
                result["active_case"].update(identity)
                result["active_case"]["state"] = "submitted"
                _write_json_atomic(args.output, result)

            row = one_run(
                args.endpoint,
                template,
                case,
                poll=args.poll,
                run_no=case["repeat"],
                timeout=args.timeout,
                client_id=client_id,
                on_submit=record_submission,
            )
        except Exception as exc:
            row = {
                **case,
                "client_id": client_id,
                "status": "harness_error",
                "completed": False,
                "failure_stage": failure_stage,
                "error": f"{type(exc).__name__}: {exc}",
            }
            active_case = result.get("active_case", {})
            if isinstance(active_case, dict) and active_case.get("prompt_id"):
                row["prompt_id"] = active_case["prompt_id"]
                try:
                    row["cancellation_dispatched"] = cancel_prompt(
                        args.endpoint,
                        active_case["prompt_id"],
                        poll=args.poll,
                        timeout=min(300.0, max(30.0, args.poll * 4)),
                    )
                except Exception as cancel_exc:
                    row["cancellation_dispatched"] = False
                    row["cancellation_error"] = (
                        f"{type(cancel_exc).__name__}: {cancel_exc}"
                    )
        result["runs"].append(row)
        result.pop("active_case", None)
        _write_json_atomic(args.output, result)
        print(
            f"    {row.get('status')} {row.get('wall_s', 'n/a')}s",
            flush=True,
        )
        if not run_succeeded(row):
            break
    result["finished"] = _iso(time.time())
    result["total_wall_s"] = round(time.time() - started, 3)
    _write_json_atomic(args.output, result)
    return 0 if all(run_succeeded(row) for row in result["runs"]) else 2


if __name__ == "__main__":
    sys.exit(main())
