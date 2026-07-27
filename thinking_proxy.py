#!/usr/bin/env python3
"""
Thinking Proxy for llama-server.

- OpenAI /v1/chat/completions: transparent passthrough, maps reasoning_effort
  to chat_template_kwargs.enable_thinking.
- Anthropic /v1/messages: full request/response/streaming format conversion,
  maps thinking.type to enable_thinking.
- Auth: Bearer token (OpenAI) or x-api-key (Anthropic).
  Tailscale (100.64.0.0/10) and localhost bypass auth.
- Process manager: starts, health-checks, and auto-restarts llama-server.

Run:  python thinking_proxy.py
  or:  uvicorn thinking_proxy:app --host 0.0.0.0 --port 8000
"""

import asyncio
import base64
import datetime
import hashlib
import io
import ipaddress
import json
import os
import signal
import sqlite3
import sys
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse as _StarletteJSONResponse, Response

# ─── Config ─────────────────────────────────────────────────────

PROJECT_DIR = os.environ.get("PROJECT_DIR", os.getcwd())
BACKEND_PORT = int(os.environ.get("BACKEND_PORT", "8001"))
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8000"))
PROXY_HOST = os.environ.get("PROXY_HOST", "0.0.0.0")
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "")
HEALTH_INTERVAL = int(os.environ.get("HEALTH_INTERVAL", "15"))
HEALTH_TIMEOUT = int(os.environ.get("HEALTH_TIMEOUT", "5"))
MAX_RESTART_BACKOFF = int(os.environ.get("MAX_RESTART_BACKOFF", "60"))
QUEUE_TIMEOUT = int(os.environ.get("QUEUE_TIMEOUT", "180"))
FALLBACK_ENABLED = os.environ.get("FALLBACK_ENABLED", "1") == "1"
BENCHMARK_MODE = os.environ.get("BENCHMARK_MODE", "0") == "1"

NIM_API_KEY = os.environ.get("NIM_API_KEY", "")
NIM_BASE_URL = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_MODEL_MULTIMODAL = os.environ.get("NIM_MODEL_MULTIMODAL", "nvidia/minimax-m3")
NIM_MODEL_TEXT = os.environ.get("NIM_MODEL_TEXT", "z-ai/glm-5.2")

OR_API_KEY = os.environ.get("OR_API_KEY", "")
OR_BASE_URL = os.environ.get("OR_BASE_URL", "https://openrouter.ai/api/v1")
OR_FREE_MODELS = os.environ.get("OR_FREE_MODELS", "[]")

ZEN_API_KEY = os.environ.get("ZEN_API_KEY", "")
ZEN_BASE_URL = os.environ.get("ZEN_BASE_URL", "https://opencode.ai/zen/v1")
ZEN_MODELS = os.environ.get("ZEN_MODELS", "[]")

CC_SWITCH_BASE_URL = os.environ.get("CC_SWITCH_BASE_URL", "http://127.0.0.1:15721")

BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"
TAILSCALE_NET = ipaddress.ip_network("100.64.0.0/10")

LLAMA_BINARY = os.path.join(
    PROJECT_DIR, "llama.cpp-turboquant", "build", "bin", "llama-server"
)
LLAMA_ARGS = [
    "-m", os.path.join(PROJECT_DIR, os.environ.get(
        "TURBO_MODEL", "models/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-LOW-MTP-IQ4_XS.gguf")),
    "--mmproj", os.path.join(
        PROJECT_DIR, "models/Qwen3.6-27B-DFlash-GGUF/mmproj-BF16.gguf"),
    "-c", "262144", "-ngl", "99", "-np", "1", "-fa", "on", "-fit", "off",
    "--cache-type-k", "turbo4", "--cache-type-v", "turbo4",
    "--spec-type", "draft-mtp", "--spec-draft-n-max", "9",
    "--jinja",
    "--chat-template-file", os.path.join(
        PROJECT_DIR, "chat_templates/qwen3.6_merged.jinja"),
    "--reasoning", "on",
    "--alias", "qwen3.6-27b-awq,qwen3.6-27b-heretic",
    "--host", "127.0.0.1", "--port", str(BACKEND_PORT),
]

STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


# ─── Context helpers ────────────────────────────────────────────

def _get_context_limit() -> int:
    total = 32768
    slots = 1
    args = LLAMA_ARGS
    for i, a in enumerate(args):
        if a == "-c" and i + 1 < len(args):
            total = int(args[i + 1])
        if a == "-np" and i + 1 < len(args):
            slots = int(args[i + 1])
    return total // slots


_CONTEXT_LIMIT = _get_context_limit()


def _estimate_tokens(messages: list) -> int:
    total = 0
    for msg in messages:
        c = msg.get("content", "")
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):
            for block in c:
                if isinstance(block, dict):
                    txt = block.get("text", "") or block.get("content", "") or ""
                    total += len(txt)
                    if block.get("type") in ("image_url", "image"):
                        total += 1000
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            total += len(str(tc))
            total += len(str(fn.get("arguments", "")))
        total += len(str(msg.get("reasoning_content", "")))
    return total // 2


def _over_context(body: dict) -> bool:
    prompt_est = _estimate_tokens(body.get("messages", []))
    max_tokens = body.get("max_tokens", 16384)
    return prompt_est + max_tokens > _CONTEXT_LIMIT

def _has_images(messages: list) -> bool:
    for msg in messages:
        c = msg.get("content", "")
        if isinstance(c, str):
            continue
        if isinstance(c, list):
            for block in c:
                if isinstance(block, dict) and block.get("type") in ("image_url", "image"):
                    return True
    return False


# ─── Prefix cache simulator ──────────────────────────────────────

class PrefixTracker:
    AGENT_WINDOW = 120
    STALE_WINDOW = 3600

    def __init__(self, local_slots: int = 2, remote_ttl: int = 3600):
        self.local_slots = local_slots
        self.remote_ttl = remote_ttl
        self._local: OrderedDict[str, float] = OrderedDict()
        self._nim: OrderedDict[str, float] = OrderedDict()
        self._or: OrderedDict[str, float] = OrderedDict()
        self._zen: OrderedDict[str, float] = OrderedDict()
        self._cc: OrderedDict[str, float] = OrderedDict()
        self._local_active: set[str] = set()
        self._local_hits: dict[str, dict] = {}

    def _fp(self, messages: list) -> str:
        recent = json.dumps(messages[-3:] if messages else [], sort_keys=True)
        return hashlib.md5(recent.encode()).hexdigest()[:16]

    def hit(self, messages: list, backend: str) -> bool:
        fp = self._fp(messages)
        cache = getattr(self, f"_{backend}")
        if fp not in cache:
            return False
        age = time.time() - cache[fp]
        if backend == "local":
            return True
        return age < self.remote_ttl

    def slot_free(self, messages: list) -> bool:
        fp = self._fp(messages)
        if fp in self._local:
            return True
        return len(self._local) < self.local_slots

    def slot_value(self, fp: str) -> float:
        info = self._local_hits.get(fp)
        if not info:
            return 0.0
        age = time.time() - info["last_time"]
        if age < self.AGENT_WINDOW:
            return 100.0 + min(info["count"], 100) * 2.0
        if age < self.STALE_WINDOW:
            return 50.0
        return 0.0

    def eviction_cost(self, messages: list) -> float:
        fp = self._fp(messages)
        if fp in self._local:
            return 0.0
        if len(self._local) < self.local_slots:
            return 0.0
        lru_fp = next(iter(self._local))
        return self.slot_value(lru_fp)

    def reserve(self, messages: list):
        fp = self._fp(messages)
        self._local_active.add(fp)

    def record(self, messages: list, backend: str, hit_confirmed: bool = False):
        fp = self._fp(messages)
        cache = getattr(self, f"_{backend}")
        if backend == "local":
            self._local_active.discard(fp)
            info = self._local_hits.setdefault(fp, {"last_time": 0, "count": 0, "first_time": time.time()})
            info["last_time"] = time.time()
            info["count"] += 1
            if hit_confirmed:
                cache[fp] = time.time()
                cache.move_to_end(fp)
                return
            if fp in cache:
                cache.move_to_end(fp)
                cache[fp] = time.time()
                return
        cache[fp] = time.time()
        cache.move_to_end(fp)
        if backend == "local":
            while len(cache) > self.local_slots:
                cache.popitem(last=False)

    def all_hits(self, messages: list) -> dict:
        return {
            b: self.hit(messages, b)
            for b in ("local", "nim", "or")
        }


prefix_tracker = PrefixTracker(local_slots=2)


# ─── Model routing config ────────────────────────────────────────

MODEL_HERETIC = "qwen3.6-27b-heretic"

def route_for_model(model: str) -> dict:
    if MODEL_HERETIC in model:
        return {"local_only": True, "priority": 0}
    return {"local_only": False, "priority": 1}

def parse_or_models(raw: str) -> list[dict]:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []

OR_FREE = parse_or_models(OR_FREE_MODELS)
if not OR_FREE:
    OR_FREE = [
        {"id": "qwen/qwen-2.5-72b-instruct", "multimodal": False},
        {"id": "qwen/qwen-2.5-32b-instruct", "multimodal": False},
        {"id": "meta-llama/llama-3.3-70b-instruct", "multimodal": False},
    ]

ZEN_MODEL_LIST = parse_or_models(ZEN_MODELS)
if not ZEN_MODEL_LIST:
    ZEN_MODEL_LIST = [
        {"id": "deepseek-v4-flash", "multimodal": False},
    ]


# ─── Remote backend helpers ──────────────────────────────────────

def _build_payload(body: dict, stream: bool, model: str) -> dict:
    return {
        "model": model,
        "messages": body.get("messages", []),
        "temperature": body.get("temperature", 0.7),
        "max_tokens": body.get("max_tokens", 16384),
        "stream": stream,
    }


async def _nim_chat(body: dict, tracker: bool = True) -> dict:
    if tracker:
        prefix_tracker.record(body.get("messages", []), "nim")
    pkey = _nim_penalty_key(body)
    has_img = _has_images(body.get("messages", []))
    model = NIM_MODEL_MULTIMODAL if has_img else NIM_MODEL_TEXT
    async with httpx.AsyncClient(timeout=120) as c:
        try:
            r = await c.post(
                f"{NIM_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {NIM_API_KEY}"},
                json=_build_payload(body, False, model),
            )
            if r.status_code != 200:
                backend_penalty.record_failure(pkey)
                service_history.record(pkey, False)
                raise RuntimeError(f"NIM returned {r.status_code}: {r.text[:200]}")
            backend_penalty.record_success(pkey)
            service_history.record(pkey, True)
            return r.json()
        except (httpx.TimeoutException, httpx.RequestError):
            backend_penalty.record_failure(pkey)
            service_history.record(pkey, False)
            raise


async def _nim_stream(body: dict):
    prefix_tracker.record(body.get("messages", []), "nim")
    pkey = _nim_penalty_key(body)
    has_img = _has_images(body.get("messages", []))
    model = NIM_MODEL_MULTIMODAL if has_img else NIM_MODEL_TEXT
    async with httpx.AsyncClient(timeout=600) as c:
        async with c.stream(
            "POST", f"{NIM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {NIM_API_KEY}"},
            json=_build_payload(body, True, model),
        ) as resp:
            if resp.status_code != 200:
                backend_penalty.record_failure(pkey)
                service_history.record(pkey, False)
                err = await resp.aread()
                raise RuntimeError(f"NIM stream returned {resp.status_code}: {err.decode()[:200]}")
            backend_penalty.record_success(pkey)
            service_history.record(pkey, True)
            async for chunk in resp.aiter_bytes():
                yield chunk


async def _or_chat(body: dict, model_id: str | None = None) -> dict:
    prefix_tracker.record(body.get("messages", []), "or")
    pkey = _or_penalty_key(body)
    model = model_id or (OR_FREE[0]["id"] if OR_FREE else "qwen/qwen-2.5-72b-instruct")
    async with httpx.AsyncClient(timeout=120) as c:
        try:
            r = await c.post(
                f"{OR_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {OR_API_KEY}"},
                json=_build_payload(body, False, model),
            )
            if r.status_code != 200:
                backend_penalty.record_failure(pkey)
                service_history.record(pkey, False)
                raise RuntimeError(f"OpenRouter returned {r.status_code}: {r.text[:200]}")
            backend_penalty.record_success(pkey)
            service_history.record(pkey, True)
            return r.json()
        except (httpx.TimeoutException, httpx.RequestError):
            backend_penalty.record_failure(pkey)
            service_history.record(pkey, False)
            raise


async def _or_stream(body: dict):
    pkey = _or_penalty_key(body)
    model = OR_FREE[0]["id"] if OR_FREE else "qwen/qwen-2.5-72b-instruct"
    prefix_tracker.record(body.get("messages", []), "or")
    async with httpx.AsyncClient(timeout=600) as c:
        async with c.stream(
            "POST", f"{OR_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {OR_API_KEY}"},
            json=_build_payload(body, True, model),
        ) as resp:
            if resp.status_code != 200:
                backend_penalty.record_failure(pkey)
                service_history.record(pkey, False)
                err = await resp.aread()
                raise RuntimeError(f"OR stream returned {resp.status_code}: {err.decode()[:200]}")
            backend_penalty.record_success(pkey)
            service_history.record(pkey, True)
            async for chunk in resp.aiter_bytes():
                yield chunk


def _or_supports_images(model_id: str) -> bool:
    for m in OR_FREE:
        if m["id"] == model_id:
            return m.get("multimodal", False)
    return False


async def _zen_chat(body: dict) -> dict:
    prefix_tracker.record(body.get("messages", []), "zen")
    pkey = _zen_penalty_key(body)
    has_img = _has_images(body.get("messages", []))
    model = None
    for m in ZEN_MODEL_LIST:
        if has_img and not _zen_supports_images(m["id"]):
            continue
        model = m["id"]
        break
    if not model:
        model = ZEN_MODEL_LIST[0]["id"] if ZEN_MODEL_LIST else "deepseek-v4-flash"
    async with httpx.AsyncClient(timeout=120) as c:
        try:
            r = await c.post(
                f"{ZEN_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {ZEN_API_KEY}",
                    "User-Agent": "CatVLLM/1.0",
                },
                json=_build_payload(body, False, model),
            )
            if r.status_code != 200:
                backend_penalty.record_failure(pkey)
                service_history.record(pkey, False)
                raise RuntimeError(f"Zen returned {r.status_code}: {r.text[:200]}")
            backend_penalty.record_success(pkey)
            service_history.record(pkey, True)
            return r.json()
        except (httpx.TimeoutException, httpx.RequestError):
            backend_penalty.record_failure(pkey)
            service_history.record(pkey, False)
            raise


async def _zen_stream(body: dict):
    pkey = _zen_penalty_key(body)
    has_img = _has_images(body.get("messages", []))
    model = None
    for m in ZEN_MODEL_LIST:
        if has_img and not _zen_supports_images(m["id"]):
            continue
        model = m["id"]
        break
    if not model:
        model = ZEN_MODEL_LIST[0]["id"] if ZEN_MODEL_LIST else "deepseek-v4-flash"
    prefix_tracker.record(body.get("messages", []), "zen")
    async with httpx.AsyncClient(timeout=600) as c:
        async with c.stream(
            "POST", f"{ZEN_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {ZEN_API_KEY}",
                "User-Agent": "CatVLLM/1.0",
            },
            json=_build_payload(body, True, model),
        ) as resp:
            if resp.status_code != 200:
                backend_penalty.record_failure(pkey)
                service_history.record(pkey, False)
                err = await resp.aread()
                raise RuntimeError(f"Zen stream returned {resp.status_code}: {err.decode()[:200]}")
            backend_penalty.record_success(pkey)
            service_history.record(pkey, True)
            async for chunk in resp.aiter_bytes():
                yield chunk


# ─── Priority queue for local backend ────────────────────────────

class _QueueItem:
    __slots__ = ("body", "priority", "event", "response", "error")
    def __init__(self, body: dict, priority: int):
        self.body = body
        self.priority = priority
        self.event = asyncio.Event()
        self.response = None
        self.error = None

    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority < other.priority
        return False


class BackendScheduler:
    def __init__(self, max_concurrent: int = 2):
        self.max_concurrent = max_concurrent
        self.active = 0
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._workers: list[asyncio.Task] = []
        self._stream_sem = asyncio.Semaphore(1)

    async def start_workers(self):
        for _ in range(self.max_concurrent):
            w = asyncio.create_task(self._worker())
            self._workers.append(w)

    def pending(self) -> int:
        return self._queue.qsize()

    async def _worker(self):
        while True:
            _, item = await self._queue.get()
            self.active += 1
            try:
                async with httpx.AsyncClient(timeout=600) as c:
                    resp = await c.post(
                        f"{BACKEND_URL}/v1/chat/completions", json=item.body)
                    item.response = resp
                    item.event.set()
            except Exception as e:
                item.error = e
                item.event.set()
            finally:
                self.active -= 1
                hit = False
                ok = item.response and item.response.status_code == 200
                if ok:
                    try:
                        data = item.response.json()
                        u = data.get("usage", {})
                        pt = u.get("prompt_tokens", 0)
                        pe = u.get("prompt_tokens_evaluated", pt)
                        hit = pt > 0 and pe is not None and pe < pt
                    except Exception:
                        pass
                    backend_penalty.record_success("local")
                    service_history.record("local", True)
                else:
                    backend_penalty.record_failure("local")
                    service_history.record("local", False)
                prefix_tracker.record(item.body.get("messages", []), "local", hit_confirmed=hit)

    async def submit(self, body: dict, priority: int, timeout: float | None = None):
        item = _QueueItem(body, priority)
        await self._queue.put((priority, item))
        try:
            await asyncio.wait_for(item.event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise
        if item.error:
            raise item.error
        return item.response

    async def acquire_stream(self, timeout: float | None = None):
        await asyncio.wait_for(self._stream_sem.acquire(), timeout=timeout)

    def release_stream(self):
        self._stream_sem.release()


scheduler = BackendScheduler()

BACKEND_READY = False


# ─── Fair routing decision ───────────────────────────────────────

BACKEND_NAMES = ["local", "nim", "or", "zen"]


class ServiceHistory:
    DB_PATH = Path(PROJECT_DIR) / "service_history.db"
    RETENTION_DAYS = 7

    def __init__(self):
        self._conn = sqlite3.connect(str(self.DB_PATH))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS service_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backend TEXT NOT NULL,
                model TEXT NOT NULL,
                ts INTEGER NOT NULL,
                success INTEGER NOT NULL,
                hour INTEGER NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_svc_model_hour
            ON service_log(model, hour, ts)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_svc_ts
            ON service_log(ts)
        """)
        self._conn.commit()
        self._prune()

    def _prune(self) -> None:
        cutoff = int(time.time()) - self.RETENTION_DAYS * 86400
        self._conn.execute("DELETE FROM service_log WHERE ts < ?", (cutoff,))
        self._conn.commit()

    def record(self, model_key: str, success: bool) -> None:
        now = int(time.time())
        hour = now // 3600
        backend = model_key.split(":")[0]
        self._conn.execute(
            "INSERT INTO service_log (backend, model, ts, success, hour) VALUES (?, ?, ?, ?, ?)",
            (backend, model_key, now, 1 if success else 0, hour),
        )
        self._conn.commit()

    def bias(self, model_key: str) -> float:
        now = int(time.time())
        cur_hour = now // 3600
        lookback = self.RETENTION_DAYS * 24
        # current hour ± 1
        hours = [cur_hour - 1, cur_hour, cur_hour + 1]
        total = 0
        success = 0
        for h in hours:
            cur = self._conn.execute(
                "SELECT COUNT(*), SUM(success) FROM service_log "
                "WHERE model = ? AND hour = ?",
                (model_key, h),
            ).fetchone()
            if cur and cur[0] > 0:
                total += cur[0]
                success += cur[1] or 0
        if total < 5:
            return 0.0
        rate = success / total
        return (rate - 0.5) * 20


service_history = ServiceHistory()


class BackendPenalty:
    DECAY_HALFLIFE = 30.0
    BASE_INCREASE = 15.0

    def __init__(self):
        self._penalties: dict[str, float] = {}
        self._consecutive: dict[str, int] = {}
        self._last_fail: dict[str, float] = {}
        self._quota_half: dict[str, float] = {}

    def _decay(self, key: str) -> None:
        now = time.time()
        age = now - self._last_fail.get(key, now)
        hl = self._quota_half.get(key, self.DECAY_HALFLIFE)
        if age > 0 and self._penalties.get(key, 0) > 0:
            factor = 2 ** (-age / hl)
            self._penalties[key] *= factor
            if self._penalties[key] < 0.1:
                self._penalties[key] = 0.0

    def record_success(self, key: str) -> None:
        self._penalties[key] = 0.0
        self._consecutive[key] = 0
        self._quota_half.pop(key, None)

    def record_failure(self, key: str, error: str = "") -> None:
        if "key limit" in error.lower() or "quota" in error.lower() or "rate limit" in error.lower():
            self._penalties[key] = 999
            self._consecutive[key] = 99
            self._quota_half[key] = 7200.0
            self._last_fail[key] = time.time()
            return
        self._decay(key)
        cons = self._consecutive.get(key, 0) + 1
        self._consecutive[key] = cons
        self._penalties[key] = (
            self._penalties.get(key, 0)
            + self.BASE_INCREASE * (1 + cons * 0.5)
        )
        self._last_fail[key] = time.time()

    def score(self, key: str) -> float:
        self._decay(key)
        return -self._penalties.get(key, 0)


backend_penalty = BackendPenalty()


def _nim_penalty_key(body: dict) -> str:
    has_img = _has_images(body.get("messages", []))
    model = NIM_MODEL_MULTIMODAL if has_img else NIM_MODEL_TEXT
    return f"nim:{model}"


def _or_penalty_key(body: dict) -> str:
    has_img = _has_images(body.get("messages", []))
    model = None
    for m in OR_FREE:
        if has_img and not _or_supports_images(m["id"]):
            continue
        model = m["id"]
        break
    return f"or:{model or 'unknown'}"


def _zen_supports_images(model_id: str) -> bool:
    for m in ZEN_MODEL_LIST:
        if m["id"] == model_id:
            return m.get("multimodal", False)
    return False


def _zen_penalty_key(body: dict) -> str:
    has_img = _has_images(body.get("messages", []))
    model = None
    for m in ZEN_MODEL_LIST:
        if has_img and not _zen_supports_images(m["id"]):
            continue
        model = m["id"]
        break
    return f"zen:{model or 'unknown'}"


def _is_h2(model: str) -> bool:
    return "haiku" in model.lower()


async def _rescue_stream(gen):
    """Wrap a backend stream so that mid-stream errors yield OpenAI error SSE
    instead of crashing the whole StreamingResponse."""
    try:
        async for chunk in gen:
            yield chunk
    except Exception:
        yield b"data: " + json.dumps(
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "error"}]}
        ).encode() + b"\n\n"
    yield b"data: [DONE]\n\n"


async def _cc_chat(anthropic_body: dict) -> dict:
    pkey = f"cc:{anthropic_body.get('model', '')}"
    prefix_tracker.record(anthropic_body.get("messages", []), "cc")
    async with httpx.AsyncClient(timeout=60) as c:
        try:
            r = await c.post(
                f"{CC_SWITCH_BASE_URL}/v1/messages",
                json=anthropic_body,
            )
            if r.status_code != 200:
                backend_penalty.record_failure(pkey)
                service_history.record(pkey, False)
                raise RuntimeError(f"cc-switch returned {r.status_code}: {r.text[:200]}")
            backend_penalty.record_success(pkey)
            service_history.record(pkey, True)
            return r.json()
        except (httpx.TimeoutException, httpx.RequestError):
            backend_penalty.record_failure(pkey)
            service_history.record(pkey, False)
            raise


async def _cc_stream(anthropic_body: dict):
    pkey = f"cc:{anthropic_body.get('model', '')}"
    prefix_tracker.record(anthropic_body.get("messages", []), "cc")
    async with httpx.AsyncClient(timeout=300) as c:
        async with c.stream(
            "POST", f"{CC_SWITCH_BASE_URL}/v1/messages",
            json={**anthropic_body, "stream": True},
        ) as resp:
            if resp.status_code != 200:
                backend_penalty.record_failure(pkey)
                service_history.record(pkey, False)
                err = await resp.aread()
                raise RuntimeError(f"cc-switch stream returned {resp.status_code}: {err.decode()[:200]}")
            backend_penalty.record_success(pkey)
            service_history.record(pkey, True)
            buffer = ""
            async for chunk in resp.aiter_bytes():
                text = chunk.decode("utf-8", errors="replace")
                buffer += text
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line.startswith("data: "):
                        yield f"{line}\n\n"
            if buffer.startswith("data: "):
                yield f"{buffer}\n\n"


async def _pick_backend(body: dict) -> str:
    has_img = _has_images(body.get("messages", []))
    msgs = body.get("messages", [])
    hits = prefix_tracker.all_hits(msgs)
    slot_free = prefix_tracker.slot_free(msgs)
    pending = scheduler.pending()
    active = scheduler.active
    local_busy = pending + active >= scheduler.max_concurrent

    # query real llama-server slots for busy state
    local_slots_idle = 0
    try:
        async with httpx.AsyncClient(timeout=2) as c:
            r = await c.get(f"{BACKEND_URL}/slots")
            if r.status_code == 200:
                for s in r.json():
                    if not s.get("is_processing", True):
                        local_slots_idle += 1
    except Exception:
        pass

    over_ctx = _over_context(body)
    scores: dict[str, float] = {}
    for b in BACKEND_NAMES:
        s = 0.0

        if b == "local":
            if not BACKEND_READY or over_ctx:
                s = -999
                continue
            if hits.get("local"):
                s += 35
            else:
                s -= prefix_tracker.eviction_cost(msgs)
                if not slot_free:
                    s -= 100
            s -= pending * 4
            s -= active * 3
            if local_busy:
                s -= 10
            if local_slots_idle == 0:
                s -= 50
            pkey = "local"
            s += backend_penalty.score(pkey)
            s += service_history.bias(pkey)

        elif b == "nim":
            if not NIM_API_KEY:
                s = -999
                continue
            if not has_img:
                s -= 25
            if hits.get("nim"):
                s += 20
            if has_img:
                s += 8
            s -= 2
            pkey = _nim_penalty_key(body)
            s += backend_penalty.score(pkey)
            s += service_history.bias(pkey)

        elif b == "or":
            if not OR_API_KEY:
                s = -999
                continue
            if hits.get("or"):
                s += 15
            if has_img:
                mm = any(_or_supports_images(m["id"]) for m in OR_FREE)
                if not mm:
                    s -= 10
            s -= 5
            pkey = _or_penalty_key(body)
            s += backend_penalty.score(pkey)
            s += service_history.bias(pkey)

        elif b == "zen":
            if not ZEN_API_KEY or not ZEN_MODEL_LIST:
                s = -999
                continue
            if hits.get("zen"):
                s += 12
            if has_img:
                mm = any(_zen_supports_images(m["id"]) for m in ZEN_MODEL_LIST)
                if not mm:
                    s -= 10
            s -= 12
            pkey = _zen_penalty_key(body)
            s += backend_penalty.score(pkey)
            s += service_history.bias(pkey)

        scores[b] = s

    best = max(scores, key=scores.get)
    print(f"[route] {best}  scores={scores}", flush=True)
    return best


# ─── Process manager ────────────────────────────────────────────

class LlamaManager:
    def __init__(self):
        self.proc = None
        self.restarts = 0
        self._stopping = False
        self._adopted = False

    async def start(self):
        try:
            async with httpx.AsyncClient(timeout=3) as c:
                r = await c.get(f"{BACKEND_URL}/health")
                if r.status_code in (200, 503):
                    print(f"[manager] adopting existing backend on port {BACKEND_PORT}", flush=True)
                    self.proc = None
                    self._adopted = True
                    if r.status_code == 200:
                        global BACKEND_READY
                        BACKEND_READY = True
                    asyncio.create_task(self.monitor())
                    return
        except Exception:
            pass
        print(f"[manager] starting llama-server on port {BACKEND_PORT}", flush=True)
        log_path = Path(PROJECT_DIR) / f"llama_server_{datetime.date.today()}.log"
        log_file = open(log_path, "a", buffering=1)
        self.proc = await asyncio.create_subprocess_exec(
            *([LLAMA_BINARY] + LLAMA_ARGS),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=PROJECT_DIR,
            env=os.environ.copy(),
        )
        asyncio.create_task(self._forward_logs(log_file))
        asyncio.create_task(self._wait_ready_and_monitor())

    async def _wait_ready_and_monitor(self):
        try:
            await self._wait_ready(deadline=1200)
            print(f"[manager] llama-server ready (pid {self.proc.pid})", flush=True)
        except RuntimeError as e:
            print(f"[manager] {e}", flush=True)
        asyncio.create_task(self.monitor())

    async def _forward_logs(self, log_file):
        try:
            async for line in self.proc.stdout:
                text = line.decode("utf-8", errors="replace")
                log_file.write(text)
                log_file.flush()
                sys.stdout.write(text)
                sys.stdout.flush()
        except Exception:
            pass
        finally:
            try:
                log_file.close()
            except Exception:
                pass

    async def _wait_ready(self, deadline=300):
        global BACKEND_READY
        start = time.time()
        while time.time() - start < deadline:
            if self.proc is not None and self.proc.returncode is not None:
                raise RuntimeError(
                    f"llama-server exited early with code {self.proc.returncode}")
            try:
                async with httpx.AsyncClient() as c:
                    r = await c.get(f"{BACKEND_URL}/health", timeout=3)
                    if r.status_code == 200:
                        BACKEND_READY = True
                        return
            except Exception:
                pass
            await asyncio.sleep(5)

    async def monitor(self):
        global BACKEND_READY
        backoff = 5
        while not self._stopping:
            await asyncio.sleep(HEALTH_INTERVAL)
            if self._stopping:
                break
            crashed = self.proc is not None and self.proc.returncode is not None
            if crashed:
                print(f"[manager] llama-server crashed "
                      f"(exit {self.proc.returncode})", flush=True)
                await self.proc.wait()
                self.restarts += 1
                print(f"[manager] restart #{self.restarts} "
                      f"in {backoff}s", flush=True)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, MAX_RESTART_BACKOFF)
                self.proc = None
                try:
                    await self.start()
                    backoff = 5
                except Exception as e:
                    print(f"[manager] restart failed: {e}", flush=True)
                continue

            try:
                async with httpx.AsyncClient() as c:
                    r = await c.get(
                        f"{BACKEND_URL}/health", timeout=HEALTH_TIMEOUT)
                    BACKEND_READY = r.status_code == 200
                    if r.status_code not in (200, 503):
                        print(f"[manager] unexpected health status {r.status_code}", flush=True)
            except Exception:
                pass
            backoff = 5

    async def stop(self):
        self._stopping = True
        if self.proc and self.proc.returncode is None:
            print("[manager] shutting down llama-server", flush=True)
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                self.proc.kill()


manager = LlamaManager()


# ─── Auth middleware ─────────────────────────────────────────────

NO_AUTH_PATHS = {"/health", "/", "/favicon.ico"}


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in NO_AUTH_PATHS:
            return await call_next(request)

        client_ip = request.client.host if request.client else ""
        try:
            ip = ipaddress.ip_address(client_ip)
            if ip in TAILSCALE_NET:
                return await call_next(request)
        except ValueError:
            pass

        bearer = request.headers.get("authorization", "")
        if bearer.startswith("Bearer "):
            token = bearer[7:].strip()
        else:
            token = request.headers.get("x-api-key", "").strip()

        if token and token == AUTH_TOKEN:
            return await call_next(request)

        return JSONResponse(
            status_code=401,
            content={
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "invalid x-api-key or Authorization header",
                },
            },
        )


_NATIVE_IMAGE_FORMATS = {"png", "jpeg", "jpg", "gif", "bmp"}


def _convert_images(messages):
    try:
        from PIL import Image
    except ImportError:
        return

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            url = (part.get("image_url") or {}).get("url", "")
            if not url.startswith("data:"):
                continue
            header, _, b64 = url.partition(",")
            fmt = ""
            if "/" in header:
                fmt = header.split("/")[1].split(";")[0].lower()
            if fmt in _NATIVE_IMAGE_FORMATS:
                continue
            try:
                raw = base64.b64decode(b64)
                img = Image.open(io.BytesIO(raw))
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="PNG")
                new_b64 = base64.b64encode(buf.getvalue()).decode()
                part["image_url"]["url"] = f"data:image/png;base64,{new_b64}"
            except Exception as e:
                print(f"[proxy] image convert failed ({fmt}): {e}", flush=True)


# ─── FastAPI app ────────────────────────────────────────────────

app = FastAPI(title="Thinking Proxy")
app.add_middleware(AuthMiddleware)


# ─── Model listing ──────────────────────────────────────────────

MODEL_LIST = {
    "object": "list",
    "data": [
        {
            "id": "qwen3.6-27b-awq",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
            "context_length": 262144,
        },
        {
            "id": "qwen3.6-27b-heretic",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
            "context_length": 262144,
        },
    ],
}


@app.get("/v1/models")
async def list_models():
    return JSONResponse(MODEL_LIST)


# ─── OpenAI endpoint ────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def openai_chat(request: Request):
    try:
        body = await request.json()
    except ClientDisconnect:
        return Response(status_code=499)

    effort = body.pop("reasoning_effort", None)
    reasoning_obj = body.pop("reasoning", None)
    enable_thinking_top = body.pop("enable_thinking", None)
    thinking_budget = body.pop("thinking_budget", None)

    enable = True
    if effort == "none":
        enable = False
    if isinstance(reasoning_obj, dict):
        eff = reasoning_obj.get("effort", "")
        if eff == "none":
            enable = False
    elif reasoning_obj == "off":
        enable = False
    if enable_thinking_top is True:
        enable = True
    elif enable_thinking_top is False:
        enable = False

    kwargs = body.setdefault("chat_template_kwargs", {})
    kwargs["enable_thinking"] = enable

    if enable:
        _EFFORT_BUDGETS = {"low": 2048, "medium": 4096, "high": 8192, "xhigh": 16384}
        budget = thinking_budget
        key = (effort or "").lower()
        if isinstance(reasoning_obj, dict):
            key = (reasoning_obj.get("effort") or "").lower()
        if budget is None or budget <= 0:
            budget = _EFFORT_BUDGETS.get(key, 4096)
        if key == "max":
            body.pop("reasoning", None)
        else:
            body.setdefault("reasoning", {})["budget"] = budget

    _convert_images(body.get("messages", []))

    route = route_for_model(body.get("model", ""))
    is_heretic = route["local_only"]
    priority = route["priority"]

    if body.get("stream"):
        if not is_heretic and FALLBACK_ENABLED and not BENCHMARK_MODE and _over_context(body):
            return StreamingResponse(
                _rescue_stream(_nim_stream(body)), media_type="text/event-stream")
        timeout = QUEUE_TIMEOUT * 2 if is_heretic else QUEUE_TIMEOUT
        try:
            await scheduler.acquire_stream(timeout=timeout)
        except asyncio.TimeoutError:
            if is_heretic:
                return JSONResponse({"error": "queue timeout"}, status_code=503)
            if FALLBACK_ENABLED and not BENCHMARK_MODE:
                try:
                    return StreamingResponse(
                        _rescue_stream(_nim_stream(body)), media_type="text/event-stream")
                except Exception:
                    if OR_API_KEY:
                        try:
                            return StreamingResponse(
                                _rescue_stream(_or_stream(body)), media_type="text/event-stream")
                        except Exception:
                            pass
                    if ZEN_API_KEY:
                        try:
                            return StreamingResponse(
                                _rescue_stream(_zen_stream(body)), media_type="text/event-stream")
                        except Exception:
                            pass
            return JSONResponse({"error": "all backends busy"}, status_code=503)
        return StreamingResponse(
            _local_stream(f"{BACKEND_URL}/v1/chat/completions", body, request),
            media_type="text/event-stream",
        )

    if not is_heretic and FALLBACK_ENABLED and not BENCHMARK_MODE and _over_context(body):
        try:
            return JSONResponse(await _nim_chat(body))
        except (RuntimeError, httpx.TimeoutException, httpx.RequestError):
            if ZEN_API_KEY:
                try:
                    return JSONResponse(await _zen_chat(body))
                except Exception:
                    pass
            return JSONResponse({"error": "context exceeds limit, fallback failed"}, status_code=429)

    if is_heretic:
        prefix_tracker.reserve(body.get("messages", []))
        try:
            resp = await scheduler.submit(body, priority, timeout=QUEUE_TIMEOUT * 2)
            return Response(content=resp.content, status_code=resp.status_code,
                          media_type=resp.headers.get("content-type", "application/json"))
        except asyncio.TimeoutError:
            return JSONResponse({"error": "local backend busy, retry later"}, status_code=503)
        except (httpx.ConnectError, httpx.TimeoutException):
            return JSONResponse({"error": "backend unavailable"}, status_code=503)

    last_err = ""
    backend = await _pick_backend(body) if not is_heretic else "local"

    if backend == "local":
        prefix_tracker.reserve(body.get("messages", []))
        try:
            to = QUEUE_TIMEOUT * 2 if is_heretic else 1
            resp = await scheduler.submit(body, priority, timeout=to)
            if resp.status_code == 200:
                return Response(content=resp.content, status_code=200,
                              media_type=resp.headers.get("content-type", "application/json"))
            print(f"[route] local non-200 {resp.status_code}", flush=True)
        except asyncio.TimeoutError:
            last_err = "local_timeout"
            print(f"[route] local timeout ({to}s)", flush=True)
        except (httpx.ConnectError, httpx.TimeoutException):
            last_err = "local_down"
            print(f"[route] local down", flush=True)
        if is_heretic:
            return JSONResponse({"error": f"local backend {last_err}"}, status_code=503)

    if not is_heretic:
        try:
            print(f"[route] routing to NIM", flush=True)
            return JSONResponse(await _nim_chat(body))
        except (RuntimeError, httpx.TimeoutException, httpx.RequestError) as e:
            last_err = "nim_down"
            print(f"[route] NIM failed: {e}", flush=True)

    if not is_heretic and OR_API_KEY:
        try:
            print(f"[route] routing to OpenRouter", flush=True)
            return JSONResponse(await _or_chat(body))
        except (RuntimeError, httpx.TimeoutException, httpx.RequestError) as e:
            last_err = "or_down"
            print(f"[route] OR failed: {e}", flush=True)

    if not is_heretic and ZEN_API_KEY:
        try:
            print(f"[route] routing to Zen", flush=True)
            return JSONResponse(await _zen_chat(body))
        except (RuntimeError, httpx.TimeoutException, httpx.RequestError) as e:
            last_err = "zen_down"
            print(f"[route] Zen failed: {e}", flush=True)

    return JSONResponse({"error": f"all backends exhausted ({last_err})"}, status_code=429,
                        headers={"Retry-After": "30"})

    if is_heretic:
        return JSONResponse({"error": f"local backend {last_err}"}, status_code=503)
    return JSONResponse({"error": f"all backends exhausted ({last_err})"}, status_code=429,
                        headers={"Retry-After": "30"})


# ─── Anthropic endpoint ─────────────────────────────────────────

@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    try:
        body = await request.json()
    except ClientDisconnect:
        return Response(status_code=499)
    raw_model = body.get("model", "")
    if _is_h2(raw_model):
        pkey = f"cc:{raw_model}"
        if backend_penalty.score(pkey) > -50:
            try:
                if body.get("stream"):
                    return StreamingResponse(
                        _cc_stream(body), media_type="text/event-stream")
                return JSONResponse(await _cc_chat(body))
            except Exception:
                backend_penalty.record_failure(pkey)
                service_history.record(pkey, False)
                print(f"[route] cc-switch failed, falling through", flush=True)

    stream = body.get("stream", False)
    openai_body = _anthropic_to_openai(body)
    _convert_images(openai_body.get("messages", []))

    route = route_for_model(openai_body.get("model", ""))
    is_heretic = route["local_only"]
    priority = route["priority"]

    if stream:
        if not is_heretic and FALLBACK_ENABLED and not BENCHMARK_MODE and _over_context(openai_body):
            try:
                nim_data = await _nim_chat(openai_body)
                anthro = _openai_to_anthropic(nim_data)
            except Exception:
                nim_data = None
            if nim_data is None and ZEN_API_KEY:
                try:
                    nim_data = await _zen_chat(openai_body)
                    anthro = _openai_to_anthropic(nim_data)
                except Exception:
                    nim_data = None
            if nim_data is not None:
                async def _ctx_overflow_stream():
                    yield _sse("message_start", {"type": "message_start", "message": anthro})
                    for i, block in enumerate(anthro.get("content", [])):
                        yield _sse("content_block_start", {"type": "content_block_start", "index": i, "content_block": block})
                        yield _sse("content_block_stop", {"type": "content_block_stop", "index": i})
                    usage = anthro.get("usage", {})
                    yield _sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": anthro.get("stop_reason", "end_turn"), "stop_sequence": None}, "usage": {"output_tokens": usage.get("output_tokens", 0)}})
                    yield _sse("message_stop", {"type": "message_stop"})
                return StreamingResponse(_ctx_overflow_stream(), media_type="text/event-stream")
            return JSONResponse({"error": "context exceeds limit, fallback failed"}, status_code=429)
        return StreamingResponse(
            _anthropic_stream_limited(openai_body, request),
            media_type="text/event-stream",
        )

    if not is_heretic and FALLBACK_ENABLED and not BENCHMARK_MODE and _over_context(openai_body):
        try:
            nim_data = await _nim_chat(openai_body)
            return JSONResponse(_openai_to_anthropic(nim_data))
        except Exception:
            if ZEN_API_KEY:
                try:
                    zen_data = await _zen_chat(openai_body)
                    return JSONResponse(_openai_to_anthropic(zen_data))
                except Exception:
                    pass
            return JSONResponse({"error": "context exceeds limit, fallback failed"}, status_code=429)

    if is_heretic:
        prefix_tracker.reserve(openai_body.get("messages", []))
        try:
            resp = await scheduler.submit(openai_body, priority, timeout=QUEUE_TIMEOUT * 2)
            if resp.status_code == 200:
                return JSONResponse(_openai_to_anthropic(resp.json()))
            return Response(content=resp.content, status_code=resp.status_code,
                          media_type="application/json")
        except asyncio.TimeoutError:
            return JSONResponse({"error": "local backend busy, retry later"}, status_code=503)
        except (httpx.ConnectError, httpx.TimeoutException):
            return JSONResponse({"error": "backend unavailable"}, status_code=503)

    last_err = ""
    backend = await _pick_backend(openai_body) if not is_heretic else "local"

    if backend == "local":
        prefix_tracker.reserve(openai_body.get("messages", []))
        try:
            to = QUEUE_TIMEOUT * 2 if is_heretic else 1
            resp = await scheduler.submit(openai_body, priority, timeout=to)
            if resp.status_code == 200:
                return JSONResponse(_openai_to_anthropic(resp.json()))
            print(f"[route] local non-200 {resp.status_code}", flush=True)
        except asyncio.TimeoutError:
            last_err = "local_timeout"
            print(f"[route] local timeout ({to}s)", flush=True)
        except (httpx.ConnectError, httpx.TimeoutException):
            last_err = "local_down"
            print(f"[route] local down", flush=True)
        if is_heretic:
            return JSONResponse({"error": f"local backend {last_err}"}, status_code=503)

    if not is_heretic:
        try:
            print(f"[route] routing to NIM", flush=True)
            nim_data = await _nim_chat(openai_body)
            return JSONResponse(_openai_to_anthropic(nim_data))
        except (RuntimeError, httpx.TimeoutException, httpx.RequestError) as e:
            last_err = "nim_down"
            print(f"[route] NIM failed: {e}", flush=True)

    if not is_heretic and OR_API_KEY:
        try:
            print(f"[route] routing to OpenRouter", flush=True)
            or_data = await _or_chat(openai_body)
            return JSONResponse(_openai_to_anthropic(or_data))
        except (RuntimeError, httpx.TimeoutException, httpx.RequestError) as e:
            last_err = "or_down"
            print(f"[route] OR failed: {e}", flush=True)

    if not is_heretic and ZEN_API_KEY:
        try:
            print(f"[route] routing to Zen", flush=True)
            zen_data = await _zen_chat(openai_body)
            return JSONResponse(_openai_to_anthropic(zen_data))
        except (RuntimeError, httpx.TimeoutException, httpx.RequestError) as e:
            last_err = "zen_down"
            print(f"[route] Zen failed: {e}", flush=True)

    return JSONResponse({"error": f"all backends exhausted ({last_err})"}, status_code=429,
                        headers={"Retry-After": "30"})


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(request: Request):
    try:
        body = await request.json()
    except ClientDisconnect:
        return Response(status_code=499)
    openai_body = _anthropic_to_openai(body)
    openai_body["max_tokens"] = 1
    openai_body["stream"] = False
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{BACKEND_URL}/v1/chat/completions", json=openai_body)
        data = resp.json()
    usage = data.get("usage", {})
    return JSONResponse({
        "input_tokens": usage.get("prompt_tokens", 0),
        "context_management": {
            "original_input_tokens": usage.get("prompt_tokens", 0),
        },
    })


# ─── Anthropic conversion helpers ───────────────────────────────

def _image_source_to_url(source: dict) -> str:
    if source.get("type") == "url":
        return source.get("url", "")
    media = source.get("media_type", "image/jpeg")
    return f"data:{media};base64,{source.get('data', '')}"


def _anthropic_to_openai(body: dict) -> dict:
    messages = []

    system = body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            parts = [b.get("text", "") for b in system
                     if isinstance(b, dict) and b.get("type") == "text"
                     and not b.get("text", "").startswith("x-anthropic-billing")]
            if parts:
                messages.append({"role": "system", "content": " ".join(parts)})

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        if role == "system":
            continue

        content = msg.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        parts = []
        tool_calls = []
        reasoning = []

        for block in (content or []):
            if not isinstance(block, dict):
                if isinstance(block, str):
                    parts.append({"type": "text", "text": block})
                continue
            bt = block.get("type")
            if bt == "text":
                parts.append({"type": "text", "text": block.get("text", "")})
            elif bt == "image":
                parts.append({"type": "image_url", "image_url": {
                    "url": _image_source_to_url(block.get("source", {}))}})
            elif bt == "thinking":
                reasoning.append(block.get("thinking", ""))
            elif bt == "redacted_thinking":
                pass
            elif bt == "tool_use":
                tool_calls.append({
                    "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })
            elif bt == "tool_result":
                tc = block.get("content", "")
                if isinstance(tc, list):
                    tc = "\n".join(
                        b.get("text", "") for b in tc
                        if isinstance(b, dict) and b.get("type") == "text")
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": str(tc),
                })
                continue

        msg_dict = {"role": role}
        if reasoning:
            msg_dict["reasoning_content"] = "".join(reasoning)
        if tool_calls:
            msg_dict["tool_calls"] = tool_calls
        if parts:
            if len(parts) == 1 and parts[0]["type"] == "text":
                msg_dict["content"] = parts[0]["text"]
            else:
                msg_dict["content"] = parts
        elif not tool_calls and not reasoning:
            continue
        messages.append(msg_dict)

    thinking_param = body.get("thinking") or {}
    enable = thinking_param.get("type") != "disabled"

    output_config = body.get("output_config") or {}
    if output_config.get("effort") and output_config["effort"] != "none":
        enable = True

    result = {
        "model": body.get("model", "qwen3.6-27b-awq"),
        "messages": messages,
        "max_tokens": body.get("max_tokens", 16384),
        "temperature": body.get("temperature", 1.0),
        "top_p": body.get("top_p", 1.0),
        "stream": body.get("stream", False),
        "chat_template_kwargs": {"enable_thinking": enable},
    }

    if enable:
        if body.get("thinking", {}).get("type") == "max":
            result.pop("reasoning", None)
        else:
            budget = thinking_param.get("budget_tokens") or 4096
            result.setdefault("reasoning", {})["budget"] = budget

    if body.get("stop_sequences"):
        result["stop"] = body["stop_sequences"]
    if body.get("top_k"):
        result["top_k"] = body["top_k"]

    if body.get("tools"):
        result["tools"] = [{
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        } for t in body["tools"]]
        result.setdefault("tool_choice", "auto")

    tc = body.get("tool_choice")
    if isinstance(tc, dict):
        tct = tc.get("type", "auto")
        mapping = {"auto": "auto", "any": "required", "none": "none"}
        if tct in mapping:
            result["tool_choice"] = mapping[tct]
        elif tct == "tool":
            result["tool_choice"] = {
                "type": "function",
                "function": {"name": tc.get("name", "")},
            }

    return result


def _openai_to_anthropic(oai: dict) -> dict:
    choice = (oai.get("choices") or [{}])[0]
    msg = choice.get("message", {})

    content = []
    if msg.get("reasoning_content"):
        content.append({
            "type": "thinking",
            "thinking": msg["reasoning_content"],
            "signature": uuid.uuid4().hex,
        })
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls", []):
        fn = tc.get("function", {})
        try:
            inp = json.loads(fn.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            inp = {}
        content.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
            "name": fn.get("name", ""),
            "input": inp,
        })
    if not content:
        content.append({"type": "text", "text": ""})

    finish = choice.get("finish_reason", "stop")
    usage = oai.get("usage", {})

    return {
        "id": oai.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": oai.get("model", ""),
        "stop_reason": STOP_REASON_MAP.get(finish, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ─── Streaming ──────────────────────────────────────────────────

async def _local_stream(url: str, body: dict, request: Request):
    prefix_tracker.reserve(body.get("messages", []))
    local_failed = False
    try:
        async for chunk in _proxy_stream(url, body):
            yield chunk
            if await request.is_disconnected():
                return
    except Exception:
        local_failed = True
    finally:
        prefix_tracker.record(body.get("messages", []), "local")
        scheduler.release_stream()
        if local_failed:
            backend_penalty.record_failure("local")
            service_history.record("local", False)
        else:
            backend_penalty.record_success("local")
            service_history.record("local", True)

    if local_failed:
        if FALLBACK_ENABLED and not BENCHMARK_MODE:
            try:
                async for chunk in _nim_stream(body):
                    yield chunk
                    if await request.is_disconnected():
                        return
                return
            except Exception:
                if OR_API_KEY:
                    try:
                        async for chunk in _or_stream(body):
                            yield chunk
                            if await request.is_disconnected():
                                return
                        return
                    except Exception:
                        pass
                if ZEN_API_KEY:
                    try:
                        async for chunk in _zen_stream(body):
                            yield chunk
                            if await request.is_disconnected():
                                return
                        return
                    except Exception:
                        pass
        yield b"data: " + json.dumps(
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "error"}]}
        ).encode() + b"\n\n"
        yield b"data: [DONE]\n\n"


async def _anthropic_stream_limited(openai_body: dict, request: Request):
    route = route_for_model(openai_body.get("model", ""))
    timeout = QUEUE_TIMEOUT * 2 if route["local_only"] else QUEUE_TIMEOUT
    try:
        await scheduler.acquire_stream(timeout=timeout)
    except asyncio.TimeoutError:
        if route["local_only"]:
            yield _sse("error", {"error": {"message": "queue timeout"}})
            return
        if FALLBACK_ENABLED and not BENCHMARK_MODE:
            nim_data = None
            try:
                nim_data = await _nim_chat(openai_body)
            except Exception:
                if OR_API_KEY:
                    try:
                        nim_data = await _or_chat(openai_body)
                    except Exception:
                        pass
                if nim_data is None and ZEN_API_KEY:
                    try:
                        nim_data = await _zen_chat(openai_body)
                    except Exception:
                        pass
            if nim_data is None:
                yield _sse("error", {"error": {"message": "fallback failed"}})
                return
            anthro = _openai_to_anthropic(nim_data)
            yield _sse("message_start", {
                "type": "message_start", "message": anthro,
            })
            for block in anthro.get("content", []):
                yield _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": anthro["content"].index(block),
                    "content_block": block,
                })
                yield _sse("content_block_stop", {
                    "type": "content_block_stop",
                    "index": anthro["content"].index(block),
                })
                if await request.is_disconnected():
                    return
            usage = anthro.get("usage", {})
            yield _sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": anthro.get("stop_reason", "end_turn"),
                          "stop_sequence": None},
                "usage": {
                    "output_tokens": usage.get("output_tokens", 0),
                },
            })
            yield _sse("message_stop", {"type": "message_stop"})
            return
        yield _sse("error", {"error": {"message": "queue timeout"}})
        return
    prefix_tracker.reserve(openai_body.get("messages", []))
    local_failed = False
    try:
        async for chunk in _anthropic_stream(openai_body):
            yield chunk
            if await request.is_disconnected():
                return
    except Exception:
        local_failed = True
    finally:
        prefix_tracker.record(openai_body.get("messages", []), "local")
        scheduler.release_stream()

    if local_failed:
        if not route["local_only"] and FALLBACK_ENABLED and not BENCHMARK_MODE:
            nim_data = None
            try:
                nim_data = await _nim_chat(openai_body)
            except Exception:
                if OR_API_KEY:
                    try:
                        nim_data = await _or_chat(openai_body)
                    except Exception:
                        pass
                if nim_data is None and ZEN_API_KEY:
                    try:
                        nim_data = await _zen_chat(openai_body)
                    except Exception:
                        pass
            if nim_data is None:
                yield _sse("error", {"error": {"message": "all backends failed"}})
                return
            anthro = _openai_to_anthropic(nim_data)
            yield _sse("message_start", {
                "type": "message_start", "message": anthro,
            })
            for block in anthro.get("content", []):
                yield _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": anthro["content"].index(block),
                    "content_block": block,
                })
                yield _sse("content_block_stop", {
                    "type": "content_block_stop",
                    "index": anthro["content"].index(block),
                })
                if await request.is_disconnected():
                    return
            usage = anthro.get("usage", {})
            yield _sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": anthro.get("stop_reason", "end_turn"),
                          "stop_sequence": None},
                "usage": {
                    "output_tokens": usage.get("output_tokens", 0),
                },
            })
            yield _sse("message_stop", {"type": "message_stop"})
            return
        yield _sse("error", {"error": {"message": "stream error"}})


async def _proxy_stream(url: str, body: dict):
    async with httpx.AsyncClient(timeout=600) as client:
        async with client.stream("POST", url, json=body) as resp:
            if resp.status_code != 200:
                err_body = await resp.aread()
                raise RuntimeError(
                    f"backend returned {resp.status_code}: {err_body.decode()[:200]}")
            async for chunk in resp.aiter_bytes():
                yield chunk


async def _anthropic_stream(openai_body: dict):
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "content": [], "model": openai_body.get("model", ""),
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    block_idx = 0
    cur_type = None
    finish_reason = "stop"
    in_tokens = 0
    out_tokens = 0

    async with httpx.AsyncClient(timeout=600) as client:
        async with client.stream(
            "POST", f"{BACKEND_URL}/v1/chat/completions", json=openai_body
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = ev.get("choices", [])
                if not choices:
                    u = ev.get("usage")
                    if u:
                        in_tokens = u.get("prompt_tokens", in_tokens)
                        out_tokens = u.get("completion_tokens", out_tokens)
                    continue

                delta = choices[0].get("delta", {})
                if choices[0].get("finish_reason"):
                    finish_reason = choices[0]["finish_reason"]

                rc = delta.get("reasoning_content")
                if rc:
                    if cur_type != "thinking":
                        if cur_type is not None:
                            yield _sse("content_block_stop", {
                                "type": "content_block_stop", "index": block_idx})
                            block_idx += 1
                        yield _sse("content_block_start", {
                            "type": "content_block_start", "index": block_idx,
                            "content_block": {"type": "thinking", "thinking": ""},
                        })
                        cur_type = "thinking"
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta", "index": block_idx,
                        "delta": {"type": "thinking_delta", "thinking": rc},
                    })

                tc = delta.get("content")
                if tc:
                    if cur_type != "text":
                        if cur_type is not None:
                            yield _sse("content_block_stop", {
                                "type": "content_block_stop", "index": block_idx})
                            block_idx += 1
                        yield _sse("content_block_start", {
                            "type": "content_block_start", "index": block_idx,
                            "content_block": {"type": "text", "text": ""},
                        })
                        cur_type = "text"
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta", "index": block_idx,
                        "delta": {"type": "text_delta", "text": tc},
                    })

                u = ev.get("usage")
                if u:
                    in_tokens = u.get("prompt_tokens", in_tokens)
                    out_tokens = u.get("completion_tokens", out_tokens)

    if cur_type is not None:
        yield _sse("content_block_stop", {
            "type": "content_block_stop", "index": block_idx})

    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {
            "stop_reason": STOP_REASON_MAP.get(finish_reason, "end_turn"),
            "stop_sequence": None,
        },
        "usage": {"input_tokens": in_tokens, "output_tokens": out_tokens},
    })
    yield _sse("message_stop", {"type": "message_stop"})


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ─── Pass-through ───────────────────────────────────────────────

@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def passthrough(path: str, request: Request):
    url = f"{BACKEND_URL}/{path}"
    body = await request.body()
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "authorization", "x-api-key")
    }
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            resp = await client.request(
                request.method, url,
                content=body, headers=headers,
                params=request.query_params,
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type"),
            )
    except httpx.ConnectError:
        return JSONResponse(
            {"error": "backend loading", "status": "unavailable"},
            status_code=503)


# ─── Startup / shutdown ─────────────────────────────────────────

@app.on_event("startup")
async def _startup():
    asyncio.create_task(manager.start())
    asyncio.create_task(scheduler.start_workers())
    print("[proxy] starting in background, backend not ready yet", flush=True)


@app.on_event("shutdown")
async def _shutdown():
    if not manager._adopted and manager.proc is not None:
        await manager.stop()


# ─── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[proxy] starting on {PROXY_HOST}:{PROXY_PORT}, "
          f"backend on port {BACKEND_PORT}", flush=True)
    print(f"[proxy] auth token: {'set' if AUTH_TOKEN else 'EMPTY - INSECURE'}", flush=True)
    print(f"[proxy] tailscale + localhost bypass auth", flush=True)

    def _handle_sigterm(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_sigterm)

    uvicorn.run(
        app,
        host=PROXY_HOST,
        port=PROXY_PORT,
        log_level="info",
        timeout_keep_alive=300,
    )
