#!/usr/bin/env python3
"""
Thinking Proxy for llama-server (default) or a FastLLM backend.

- OpenAI /v1/chat/completions: transparent passthrough, maps reasoning_effort
  to chat_template_kwargs.enable_thinking.
- Anthropic /v1/messages: full request/response/streaming format conversion,
  maps thinking.type to enable_thinking.
- Auth: Bearer token (OpenAI) or x-api-key (Anthropic).
  Tailscale (100.64.0.0/10) and localhost bypass auth.
- Process manager: starts, health-checks, and auto-restarts llama-server.
  FastLLM remains external by default. With FASTLLM_OWNED=1 the proxy starts
  one child on the first request, drains it after idle/VRAM pressure, and
  never signals a process it did not spawn.

Backend selection (env-driven; default is llama-server):
  FASTLLM_BACKEND_URL   Set the FastLLM OpenAI endpoint.
  FASTLLM_OWNED         Opt in to proxy-owned cold-start/unload. Requires
                        FASTLLM_BACKEND_COMMAND. Default 0 (external).
  FASTLLM_MODEL_SLUG    Backend slug sent in outbound payloads; never exposed
                        to clients. Default "qwen3.6-fastllm".
  FASTLLM_PUBLIC_ALIASES  Comma-separated public model names that route to
                        FastLLM. Default "qwen3.6-27b-heretic,qwen3.6-27b-awq".
                        They are kept public in /v1/models and in responses;
                        only the outbound payload model field is rewritten to
                        the slug. local-only heretic/Fable requests never
                        silently fall through to cloud providers.

When FASTLLM_BACKEND_URL is unset, behavior is unchanged from the original
llama-server proxy. A new owned child starts a new tensor-cache epoch; only
remote-provider routing hints survive child shutdown.

Run:  python thinking_proxy.py
  or:  uvicorn thinking_proxy:app --host 0.0.0.0 --port 8000
"""

import asyncio
import base64
import datetime
import hashlib
import io
import inspect
import ipaddress
import json
import os
import re
import signal
import sqlite3
import sys
import shlex
import secrets
import time
import uuid
from collections import OrderedDict
from urllib.parse import urlparse
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse as _StarletteJSONResponse, Response

_FASTLLM_SUPPORT_DIR = Path(__file__).resolve().parent / "v100-perfs"
if str(_FASTLLM_SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_FASTLLM_SUPPORT_DIR))
from fastllm_adapter import adapt_fastllm_response, prepare_fastllm_body

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
# Explicit NIM_MODELS wins; else an explicit NIM_MODEL_TEXT is honored for
# backward compatibility; else the currently available NVIDIA NIM catalog.
NIM_MODEL_TEXT = os.environ.get("NIM_MODEL_TEXT", "")
NIM_MODELS = os.environ.get("NIM_MODELS", "")
if NIM_MODELS:
    _nim_ids = [s.strip() for s in NIM_MODELS.split(",") if s.strip()]
elif NIM_MODEL_TEXT:
    _nim_ids = [NIM_MODEL_TEXT]
else:
    _nim_ids = [
        "meta/muse-glimmer-30b",
        "poolside/laguna-xs-2.1",
        "stepfun-ai/step-3.7-flash",
        "nvidia/minimax-m3",
    ]
NIM_MODEL_LIST = [
    {"id": m, "multimodal": (m == NIM_MODEL_MULTIMODAL)}
    for m in _nim_ids
]
if not NIM_MODEL_LIST:
    NIM_MODEL_LIST = [{"id": NIM_MODEL_MULTIMODAL, "multimodal": True}]

OR_API_KEY = os.environ.get("OR_API_KEY", "")
OR_BASE_URL = os.environ.get("OR_BASE_URL", "https://openrouter.ai/api/v1")
OR_FREE_MODELS = os.environ.get("OR_FREE_MODELS", "[]")

ZEN_API_KEY = os.environ.get("ZEN_API_KEY", "")
ZEN_BASE_URL = os.environ.get("ZEN_BASE_URL", "https://opencode.ai/zen/v1")
ZEN_MODELS = os.environ.get("ZEN_MODELS", "[]")

CC_SWITCH_BASE_URL = os.environ.get("CC_SWITCH_BASE_URL", "http://127.0.0.1:15721")

# ── FastLLM backend (external by default; owned lifecycle is opt-in) ──
FASTLLM_BACKEND_URL = os.environ.get("FASTLLM_BACKEND_URL", "").rstrip("/")
FASTLLM_MODEL_SLUG = os.environ.get("FASTLLM_MODEL_SLUG", "qwen3.6-fastllm")
FASTLLM_PUBLIC_ALIASES = {
    s.strip() for s in os.environ.get(
        "FASTLLM_PUBLIC_ALIASES", "qwen3.6-fastllm"
    ).split(",") if s.strip()
}
FASTLLM_ENABLED = bool(FASTLLM_BACKEND_URL)
FASTLLM_MODE = FASTLLM_ENABLED
FASTLLM_OWNED = os.environ.get("FASTLLM_OWNED", "0") == "1"
FASTLLM_BACKEND_COMMAND = os.environ.get("FASTLLM_BACKEND_COMMAND", "").strip()
FASTLLM_BACKEND_CWD = os.environ.get("FASTLLM_BACKEND_CWD", PROJECT_DIR)
FASTLLM_BACKEND_LOG = os.environ.get(
    "FASTLLM_BACKEND_LOG",
    os.path.join(PROJECT_DIR, "fastllm_owned_backend.log"),
)
FASTLLM_START_TIMEOUT = float(os.environ.get("FASTLLM_START_TIMEOUT", "900"))
FASTLLM_STOP_TIMEOUT = float(os.environ.get("FASTLLM_STOP_TIMEOUT", "30"))
FASTLLM_IDLE_TIMEOUT = float(os.environ.get("FASTLLM_IDLE_TIMEOUT", "0"))
FASTLLM_PREFIX_CACHE_PERSIST = os.environ.get(
    "FASTLLM_PREFIX_CACHE_PERSIST", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }
FASTLLM_PREFIX_CACHE_CHECKPOINT_TIMEOUT = max(
    1.0,
    float(os.environ.get(
        "FASTLLM_PREFIX_CACHE_CHECKPOINT_TIMEOUT", "300")),
)
# 空闲/显存压力卸载方式：1（默认）→ POST /admin/suspend 让后端常驻释放 GPU；
# 0 → 保持旧行为 kill 进程。suspend 的 HTTP 超时上限。
FASTLLM_IDLE_SUSPEND = os.environ.get(
    "FASTLLM_IDLE_SUSPEND", "1").strip().lower() in {
        "1", "true", "yes", "on",
    }
FASTLLM_SUSPEND_TIMEOUT = max(
    1.0,
    float(os.environ.get("FASTLLM_SUSPEND_TIMEOUT", "300")),
)
FASTLLM_RESUME_TIMEOUT = max(
    1.0,
    float(os.environ.get("FASTLLM_RESUME_TIMEOUT", "900")),
)
FASTLLM_LIFECYCLE_INTERVAL = max(
    1.0, float(os.environ.get("FASTLLM_LIFECYCLE_INTERVAL", "5")))
FASTLLM_VRAM_MIN_FREE_BYTES = int(
    float(os.environ.get("FASTLLM_VRAM_MIN_FREE_GIB", "0")) * 1024 ** 3)
FASTLLM_VRAM_RESUME_FREE_BYTES = int(
    float(os.environ.get(
        "FASTLLM_VRAM_RESUME_FREE_GIB",
        str(FASTLLM_VRAM_MIN_FREE_BYTES / 1024 ** 3),
    )) * 1024 ** 3)
FASTLLM_VRAM_HIGH_WATERMARK = float(
    os.environ.get("FASTLLM_VRAM_HIGH_WATERMARK", "0"))
FASTLLM_VRAM_RESUME_WATERMARK = float(
    os.environ.get("FASTLLM_VRAM_RESUME_WATERMARK", "0"))
FASTLLM_GPU_INDEX = os.environ.get("FASTLLM_GPU_INDEX", "0")
if FASTLLM_OWNED and not FASTLLM_BACKEND_COMMAND:
    raise RuntimeError(
        "FASTLLM_OWNED=1 requires a non-empty FASTLLM_BACKEND_COMMAND")
if FASTLLM_OWNED and not FASTLLM_MODE:
    raise RuntimeError("FASTLLM_OWNED=1 requires FASTLLM_BACKEND_URL")
FASTLLM_CHAT_TEMPLATE = Path(os.environ.get(
    "FASTLLM_CHAT_TEMPLATE",
    os.path.join(PROJECT_DIR, "chat_templates", "qwen3.6_gguf_original.jinja"),
))
LLAMA_ENABLED = os.environ.get("LLAMA_ENABLED", "1") == "1"
if FASTLLM_MODE and QUEUE_TIMEOUT < 600:
    QUEUE_TIMEOUT = 600

if FASTLLM_MODE:
    BACKEND_URL = FASTLLM_BACKEND_URL
else:
    BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"
BACKEND_HEALTH_PATH = "/health"
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

    def reset_local(self):
        self._local.clear()
        self._local_active.clear()
        self._local_hits.clear()

    def all_hits(self, messages: list) -> dict:
        return {
            b: self.hit(messages, b)
            for b in ("local", "nim", "or")
        }


prefix_tracker = PrefixTracker(local_slots=2)


class BackendLifecycleError(RuntimeError):
    pass


class BackendMemoryPressure(BackendLifecycleError):
    pass


async def _await_callback(result):
    if inspect.isawaitable(result):
        return await result
    return result


class BackendLease:
    def __init__(self, manager):
        self._manager = manager
        self._released = False

    async def release(self):
        if self._released:
            return
        self._released = True
        await self._manager._release()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        await self.release()


class BackendLifecycleManager:
    """Serialize backend epochs and account for every in-flight request."""

    def __init__(
        self,
        *,
        owned,
        start_backend,
        stop_backend,
        probe_ready,
        reset_local,
        checkpoint_backend=None,
        suspend_backend=None,
        resume_backend=None,
        start_timeout=900.0,
        readiness_interval=0.25,
        idle_timeout=0.0,
        minimum_free_bytes=0,
        resume_free_bytes=0,
        high_used_ratio=0.0,
        resume_used_ratio=0.0,
    ):
        if minimum_free_bytes < 0 or resume_free_bytes < 0:
            raise ValueError("VRAM thresholds must be non-negative")
        if resume_free_bytes and resume_free_bytes < minimum_free_bytes:
            raise ValueError("resume_free_bytes must be >= minimum_free_bytes")
        high_used_ratio = float(high_used_ratio)
        resume_used_ratio = float(resume_used_ratio)
        if high_used_ratio and not 0.0 < high_used_ratio < 1.0:
            raise ValueError("high_used_ratio must be between 0 and 1")
        if resume_used_ratio and not 0.0 < resume_used_ratio < 1.0:
            raise ValueError("resume_used_ratio must be between 0 and 1")
        if high_used_ratio and resume_used_ratio >= high_used_ratio:
            raise ValueError("resume_used_ratio must be below high_used_ratio")
        self.owned = bool(owned)
        self.start_backend = start_backend
        self.stop_backend = stop_backend
        self.probe_ready = probe_ready
        self.checkpoint_backend = checkpoint_backend
        self.suspend_backend = suspend_backend
        self.resume_backend = resume_backend
        self.reset_local = reset_local
        self.start_timeout = float(start_timeout)
        self.readiness_interval = max(0.01, float(readiness_interval))
        self.idle_timeout = max(0.0, float(idle_timeout))
        self.minimum_free_bytes = int(minimum_free_bytes)
        self.resume_free_bytes = int(resume_free_bytes)
        self.high_used_ratio = high_used_ratio
        self.resume_used_ratio = resume_used_ratio
        self.state = "COLD"
        self.generation = 0
        self.active = 0
        self.last_idle_at = time.monotonic()
        self.last_error = None
        self.last_stop_reason = None
        self.child = None
        self.checkpoint_successes = 0
        self.checkpoint_failures = 0
        self.last_checkpoint_error = None
        self.last_checkpoint_generation = 0
        self.last_checkpoint_pages = 0
        self.last_checkpoint_bytes = 0
        self.last_checkpoint_duration_ms = 0.0
        self._pressure = False
        self._pressure_since = None
        self._high_samples = 0
        self._resume_samples = 0
        self._resume_pending = False
        self._accepting = True
        self._drained = asyncio.Event()
        self._drained.set()
        self._lock = asyncio.Lock()
        self._stop_lock = asyncio.Lock()
        self._activation_task = None

    @property
    def pressure(self):
        return self._pressure

    async def acquire(self, timeout=None):
        deadline = (None if timeout is None
                    else time.monotonic() + max(0.0, float(timeout)))
        while True:
            async with self._lock:
                if not self._accepting:
                    raise BackendLifecycleError(
                        "FastLLM backend is draining for shutdown")
                if self._pressure:
                    raise BackendMemoryPressure(
                        "FastLLM activation is blocked by the VRAM free-space watermark")
                if self.state in ("DRAINING", "STOPPING"):
                    continue
                if self.state == "READY":
                    self.active += 1
                    self._drained.clear()
                    return BackendLease(self)
                task = self._activation_task
                if task is None or task.done():
                    # 记录来自挂起态的激活意图：_activate 用快照决定走
                    # resume 还是冷启动（state 随即被覆盖为 STARTING）。
                    self._resume_pending = (
                        self.state == "SUSPENDED"
                        and self.child is not None
                        and self.resume_backend is not None
                    )
                    self.state = "STARTING"
                    self.last_error = None
                    task = asyncio.create_task(self._activate())
                    self._activation_task = task
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise BackendLifecycleError(
                    "FastLLM backend is unloading (timed out waiting)")
            await asyncio.sleep(0.25)

        waiter = asyncio.shield(task)
        if deadline is None:
            await waiter
        else:
            remaining = max(0.0, deadline - time.monotonic())
            await asyncio.wait_for(waiter, timeout=remaining)

        async with self._lock:
            if not self._accepting:
                raise BackendLifecycleError(
                    "FastLLM backend is draining for shutdown")
            if self._pressure:
                raise BackendMemoryPressure(
                    "FastLLM became unavailable under VRAM pressure")
            if self.state != "READY":
                raise BackendLifecycleError(
                    self.last_error or f"FastLLM backend is {self.state}")
            self.active += 1
            self._drained.clear()
            return BackendLease(self)

    async def wait_for_activation(self):
        async with self._lock:
            task = self._activation_task
        if task is not None:
            await asyncio.shield(task)

    async def observe_child_exit(self, child):
        async with self._lock:
            if not self.owned or self.child is not child:
                return False
            self.child = None
            self.state = "FAILED"
            self.last_error = (
                f"FastLLM backend exited with code {child.returncode}")
            return True

    async def _activate(self):
        child = None
        resumed = False
        next_generation = self.generation + 1
        try:
            async with asyncio.timeout(self.start_timeout):
                if self.owned:
                    if self._resume_pending:
                        # 常驻后端：从挂起状态恢复，避免重新 spawn。
                        child = self.child
                        try:
                            await _await_callback(
                                self.resume_backend(child))
                            resumed = True
                        except BaseException:
                            # resume 失败 → 回退冷启动（kill + 重新 spawn）。
                            resumed = False
                            try:
                                await _await_callback(
                                    self.stop_backend(child))
                            except BaseException:
                                pass
                            async with self._lock:
                                if self.child is child:
                                    self.child = None
                            child = await _await_callback(
                                self.start_backend(next_generation))
                    else:
                        child = await _await_callback(
                            self.start_backend(next_generation))
                    while not await _await_callback(
                        self.probe_ready(child)
                    ):
                        await asyncio.sleep(self.readiness_interval)
                    await _await_callback(self.reset_local())
            async with self._lock:
                self._resume_pending = False
                self.child = child
                if self.owned:
                    if not resumed:
                        self.generation = next_generation
                self.state = "READY"
                self.last_error = None
        except BaseException as exc:
            if child is not None and self.owned:
                try:
                    await _await_callback(self.stop_backend(child))
                except BaseException:
                    pass
            message = (
                f"FastLLM backend activation failed: {type(exc).__name__}: {exc}")
            async with self._lock:
                self._resume_pending = False
                self.child = None
                self.state = "FAILED"
                self.last_error = message
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise BackendLifecycleError(message) from exc

    async def _release(self):
        async with self._lock:
            if self.active <= 0:
                raise BackendLifecycleError("backend lease accounting underflow")
            self.active -= 1
            if self.active == 0:
                self.last_idle_at = time.monotonic()
                self._drained.set()
        # No immediate stop on pressure here: a transient VRAM peak (e.g. a
        # vision-encode burst) clears within seconds once the request ends.
        # Sustained pressure is handled by the watchdog's grace period.

    async def check_idle(self, now=None):
        if not self.owned or self.idle_timeout <= 0:
            return False
        timestamp = time.monotonic() if now is None else float(now)
        async with self._lock:
            should_stop = (
                self.state == "READY"
                and self.active == 0
                and timestamp - self.last_idle_at >= self.idle_timeout
            )
            idle_seconds = (
                timestamp - self.last_idle_at if should_stop else 0.0
            )
        if not should_stop:
            return False
        # 分级卸载：短空闲 → 权重驻留内存（memory，快速恢复）；
        # 超长空闲（>12h）→ disk（删除 RAM 权重快照，resume 时从源
        # GGUF 重新加载）。
        tier = "disk" if idle_seconds >= 12 * 3600 else "memory"
        return await self.stop("idle", tier=tier)

    async def observe_memory(self, *, free_bytes, total_bytes):
        if not self.owned or (
            self.minimum_free_bytes <= 0 and self.high_used_ratio <= 0
        ):
            return False
        if self.state not in {"READY", "DRAINING"}:
            # Loading peaks legitimately exhaust VRAM (model + KV pool
            # prefill); evaluating pressure mid-activation would reject the
            # very request that is bringing the backend up.
            return False
        free_bytes = int(free_bytes)
        total_bytes = int(total_bytes)
        if free_bytes < 0 or total_bytes <= 0 or free_bytes > total_bytes:
            raise ValueError("invalid VRAM sample")
        used_ratio = 1.0 - (free_bytes / total_bytes)
        crossed_high = (
            (self.minimum_free_bytes > 0
             and free_bytes < self.minimum_free_bytes)
            or (self.high_used_ratio > 0
                and used_ratio >= self.high_used_ratio)
        )
        crossed_resume = (
            (self.resume_free_bytes <= 0
             or free_bytes >= self.resume_free_bytes)
            and (self.resume_used_ratio <= 0
                 or used_ratio <= self.resume_used_ratio)
        )

        changed = False
        async with self._lock:
            if crossed_high:
                # Require consecutive samples before declaring pressure:
                # transient peaks (multimodal encode, prefill spikes) must
                # not kill a healthy backend.
                self._high_samples += 1
                if self._high_samples >= 2:
                    changed = not self._pressure
                    if not self._pressure:
                        self._pressure = True
                        self._pressure_since = time.monotonic()
                    if self.state == "READY":
                        self.state = "DRAINING"
            else:
                self._high_samples = 0
            if crossed_resume:
                self._resume_samples += 1
                if self._resume_samples >= 2 and self._pressure:
                    self._pressure = False
                    self._pressure_since = None
                    changed = True
                    if self.state == "DRAINING":
                        self.state = "READY"
            else:
                self._resume_samples = 0
        return changed

    def pressure_stale(self, grace_seconds: float) -> bool:
        """True when pressure has persisted past the grace period and no
        request is in flight (safe to unload)."""
        if not self._pressure or self._pressure_since is None:
            return False
        if self.active > 0:
            return False
        return time.monotonic() - self._pressure_since >= grace_seconds

    async def drain_and_stop(self, reason, timeout):
        if not self.owned:
            return False
        async with self._lock:
            self._accepting = False
            if self.active > 0:
                self.state = "DRAINING"
                self.last_stop_reason = str(reason)
        timed_out = False
        try:
            await asyncio.wait_for(
                self._drained.wait(), timeout=max(0.0, float(timeout)))
        except asyncio.TimeoutError:
            timed_out = True
        # 代理退出是真正的关闭：必须 kill，不能把常驻后端留在内存里。
        return await self.stop(reason, force=timed_out, suspend=False)

    async def stop(self, reason, force=False, suspend=True, tier="memory"):
        if not self.owned:
            return False
        async with self._stop_lock:
            activation = None
            async with self._lock:
                if self.active > 0 and not force:
                    self.state = "DRAINING"
                    self.last_stop_reason = str(reason)
                    return False
                if (
                    self.state == "STARTING"
                    and self._activation_task is not None
                    and not self._activation_task.done()
                ):
                    activation = self._activation_task
                    activation.cancel()
                child = self.child
                if activation is None and child is None:
                    if self.state != "FAILED":
                        self.state = "COLD"
                    return False
                self.state = "STOPPING"
                self.last_stop_reason = str(reason)
            if activation is not None:
                await asyncio.gather(activation, return_exceptions=True)
                async with self._lock:
                    self.child = None
                    self._activation_task = None
                    self._resume_pending = False
                    self.state = "COLD"
                    self.last_idle_at = time.monotonic()
                return True
            should_checkpoint = (
                child is not None
                and self.active == 0
                and self.state != "SUSPENDED"
                and self.checkpoint_backend is not None
            )
            if should_checkpoint:
                checkpoint_started = time.monotonic()
                try:
                    result = await _await_callback(
                        self.checkpoint_backend(child))
                    result = result if isinstance(result, dict) else {}
                    duration_ms = float(result.get(
                        "duration_ms",
                        (time.monotonic() - checkpoint_started) * 1000.0,
                    ))
                    async with self._lock:
                        self.checkpoint_successes += 1
                        self.last_checkpoint_error = None
                        self.last_checkpoint_generation = int(
                            result.get("generation", 0))
                        self.last_checkpoint_pages = int(
                            result.get("pages", 0))
                        self.last_checkpoint_bytes = int(
                            result.get("bytes", 0))
                        self.last_checkpoint_duration_ms = duration_ms
                except Exception as exc:
                    message = (
                        "FastLLM prefix-cache checkpoint failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    async with self._lock:
                        self.checkpoint_failures += 1
                        self.last_checkpoint_error = message
                        self.last_checkpoint_duration_ms = (
                            time.monotonic() - checkpoint_started
                        ) * 1000.0
                    print(f"[lifecycle] {message}", flush=True)
            if (
                suspend
                and self.suspend_backend is not None
                and child is not None
                and self.active == 0
            ):
                try:
                    await _await_callback(
                        self.suspend_backend(child, tier=tier))
                except Exception as exc:
                    message = (
                        "FastLLM suspend failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    print(f"[lifecycle] {message}", flush=True)
                    async with self._lock:
                        self.last_error = message
                else:
                    async with self._lock:
                        if self.child is child:
                            self.state = "SUSPENDED"
                            self.last_stop_reason = str(reason)
                            # GPU 已释放；陈旧的压力标记会死锁后续激活。
                            self._pressure = False
                            self._pressure_since = None
                    print(
                        f"[lifecycle] suspended owned FastLLM "
                        f"pid={child.pid} on {reason}",
                        flush=True,
                    )
                    return True
                # suspend 失败 → 回退到 kill 流程
            try:
                await _await_callback(self.stop_backend(child))
            finally:
                async with self._lock:
                    if self.child is child:
                        self.child = None
                    self.state = "COLD"
                    self.last_idle_at = time.monotonic()
                    # The GPU is free again; a stale pressure flag would
                    # deadlock every subsequent activation.
                    self._pressure = False
                    self._pressure_since = None
            return True

    def snapshot(self):
        return {
            "owned": self.owned,
            "state": self.state,
            "suspended": self.state == "SUSPENDED",
            "generation": self.generation,
            "active": self.active,
            "accepting": self._accepting,
            "pressure": self._pressure,
            "last_error": self.last_error,
            "last_stop_reason": self.last_stop_reason,
            "checkpoint_successes": self.checkpoint_successes,
            "checkpoint_failures": self.checkpoint_failures,
            "last_checkpoint_error": self.last_checkpoint_error,
            "last_checkpoint_generation":
                self.last_checkpoint_generation,
            "last_checkpoint_pages": self.last_checkpoint_pages,
            "last_checkpoint_bytes": self.last_checkpoint_bytes,
            "last_checkpoint_duration_ms":
                self.last_checkpoint_duration_ms,
        }


# ─── Model routing config ────────────────────────────────────────

HERETIC_MODEL_SUFFIX = "heretic"


def _is_heretic_model(model: str) -> bool:
    return bool(model) and model.endswith(HERETIC_MODEL_SUFFIX)


def _is_fastllm_alias(model: str) -> bool:
    """True when a public model name should route to the FastLLM backend."""
    if not model:
        return False
    if model == FASTLLM_MODEL_SLUG:
        return True
    return any(alias in model for alias in FASTLLM_PUBLIC_ALIASES)


def _is_local_alias(model: str) -> bool:
    """True when the caller explicitly selected the local FastLLM backend
    (heretic suffix, public alias, or backend slug). These requests bypass
    the fair router: rerouting an explicitly-local model to external
    providers serves a different model and breaks multi-round tool calls."""
    return bool(model) and (_is_heretic_model(model) or _is_fastllm_alias(model))

def _to_backend_model(public_model: str) -> str:
    """Rewrite a public alias to the configured FastLLM backend slug in
    outbound payloads. Leaves other model names (incl. cloud models) intact."""
    if FASTLLM_MODE and _is_fastllm_alias(public_model):
        return FASTLLM_MODEL_SLUG
    return public_model


def _to_public_model(backend_model: str, requested_model: str) -> str:
    """Restore the public model name the caller actually requested in responses,
    hiding the internal backend slug. Only rewrites when we are in FastLLM mode
    and the request targeted a FastLLM alias."""
    if FASTLLM_MODE and backend_model == FASTLLM_MODEL_SLUG and requested_model:
        return requested_model
    return backend_model


def _rewrite_local_response(resp, requested_model: str):
    """Restore the public model name in a non-streaming local-backend response
    when serving FastLLM. Returns a FastAPI Response preserving status, content
    type and body. No-op outside FastLLM mode or for non-JSON bodies."""
    if not (FASTLLM_MODE and requested_model and resp.status_code == 200):
        return resp
    ctype = resp.headers.get("content-type", "")
    if "json" not in ctype:
        return resp
    try:
        data = json.loads(resp.body)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
        return resp
    if isinstance(data, dict) and data.get("model") == FASTLLM_MODEL_SLUG:
        data["model"] = requested_model
        return JSONResponse(data)
    return resp


def _rewrite_stream_model(event: bytes, public_model: str) -> bytes:
    """Restore the public model in one complete OpenAI SSE event."""
    if not (FASTLLM_MODE and public_model):
        return event
    separator = b"\r\n\r\n" if event.endswith(b"\r\n\r\n") else b"\n\n"
    lines = event[:-len(separator)].splitlines() if event.endswith(separator) else event.splitlines()
    data_lines = [line[5:].lstrip() for line in lines if line.startswith(b"data:")]
    if not data_lines:
        return event
    payload = b"\n".join(data_lines)
    if payload == b"[DONE]":
        return event
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return event
    if not isinstance(data, dict) or data.get("model") != FASTLLM_MODEL_SLUG:
        return event
    data["model"] = public_model
    return (b"data: " + json.dumps(data, ensure_ascii=False,
                                    separators=(",", ":")).encode()
            + separator)


def _normalize_tool_schema(body: dict) -> dict:
    """Cherry Studio's builtin_web_search ships the search terms in the tool
    description ("Prepared queries: ...") behind an optional-only schema, so
    compliant models emit {} and the client's search gets no parameters.
    Promote the prepared query to a required string parameter so the model
    generates real arguments (the output stays fully model-authored)."""
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict):
            continue
        desc = fn.get("description", "") or ""
        if "Prepared queries" not in desc:
            continue
        m = re.search(r'Prepared queries?: "([^"]+)"', desc)
        if not m or not m.group(1).strip():
            continue
        params = fn.get("parameters")
        if not isinstance(params, dict):
            params = {"type": "object"}
            fn["parameters"] = params
        props = params.get("properties")
        if not isinstance(props, dict):
            props = {}
            params["properties"] = props
        if "query" not in props:
            props["query"] = {
                "type": "string",
                "description": "The search query. Use the prepared query "
                                "from the tool description when it fits.",
            }
        required = params.get("required")
        if not isinstance(required, list):
            required = []
            params["required"] = required
        if "query" in props and "query" not in required:
            required.append("query")
    return body




def _restore_public_model(oai: dict, requested_model: str) -> dict:
    """Restore the public model name in a parsed OpenAI completion dict (used
    before Anthropic conversion) when serving FastLLM."""
    if (FASTLLM_MODE and isinstance(oai, dict)
            and oai.get("model") == FASTLLM_MODEL_SLUG and requested_model):
        oai["model"] = requested_model
    return oai


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
    model = _pick_nim_model(has_img)
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
    model = _pick_nim_model(has_img)
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
    __slots__ = ("body", "priority", "event", "response", "error", "cancelled")
    def __init__(self, body: dict, priority: int):
        self.body = body
        self.priority = priority
        self.event = asyncio.Event()
        self.response = None
        self.error = None
        self.cancelled = False

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
        # FastLLM runs batch=1: exactly one in-flight request at a time
        # across BOTH the stream path and the worker queue (rotation, not
        # fan-out), so concurrent client requests never stack VRAM.
        self._slot = asyncio.Semaphore(1)

    async def start_workers(self):
        for _ in range(self.max_concurrent):
            w = asyncio.create_task(self._worker())
            self._workers.append(w)

    def pending(self) -> int:
        return self._queue.qsize()

    async def _worker(self):
        while True:
            _, item = await self._queue.get()
            lease = None
            sent = False
            slot_held = False
            self.active += 1
            try:
                await asyncio.wait_for(
                    self._slot.acquire(), timeout=QUEUE_TIMEOUT)
                slot_held = True
                if item.cancelled:
                    continue
                lease = await backend_lifecycle.acquire(timeout=QUEUE_TIMEOUT)
                if item.cancelled:
                    continue
                outbound = item.body
                if FASTLLM_MODE:
                    outbound = prepare_fastllm_body(
                        item.body, FASTLLM_CHAT_TEMPLATE)
                async with httpx.AsyncClient(timeout=QUEUE_TIMEOUT + 60) as c:
                    sent = True
                    resp = await c.post(
                        f"{BACKEND_URL}/v1/chat/completions", json=outbound)
                    if FASTLLM_MODE and resp.status_code == 200:
                        try:
                            adapted = adapt_fastllm_response(resp.json())
                            resp = httpx.Response(
                                status_code=200,
                                content=json.dumps(adapted).encode(),
                                headers={**resp.headers,
                                         "content-type": "application/json"},
                                request=resp.request,
                            )
                        except Exception:
                            pass
                    item.response = resp
                    if not item.cancelled:
                        item.event.set()
            except Exception as exc:
                if not item.cancelled:
                    item.error = exc
                    item.event.set()
            finally:
                if slot_held:
                    self._slot.release()
                if lease is not None:
                    await lease.release()
                self.active -= 1
                self._queue.task_done()
                if sent:
                    hit = False
                    ok = item.response and item.response.status_code == 200
                    if ok:
                        try:
                            data = item.response.json()
                            usage = data.get("usage", {})
                            prompt_tokens = usage.get("prompt_tokens", 0)
                            evaluated = usage.get(
                                "prompt_tokens_evaluated", prompt_tokens)
                            hit = (
                                prompt_tokens > 0
                                and evaluated is not None
                                and evaluated < prompt_tokens
                            )
                        except Exception:
                            pass
                        backend_penalty.record_success("local")
                        service_history.record("local", True)
                    else:
                        backend_penalty.record_failure("local")
                        service_history.record("local", False)
                    prefix_tracker.record(
                        item.body.get("messages", []),
                        "local",
                        hit_confirmed=hit,
                    )

    async def submit(self, body: dict, priority: int, timeout: float | None = None):
        item = _QueueItem(body, priority)
        await self._queue.put((priority, item))
        try:
            await asyncio.wait_for(item.event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            item.cancelled = True
            raise
        if item.error:
            raise item.error
        return item.response

    async def acquire_stream(self, timeout: float | None = None):
        await asyncio.wait_for(self._slot.acquire(), timeout=timeout)

    def release_stream(self):
        self._slot.release()


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


def _pick_nim_model(has_img: bool) -> str:
    """Pick the NIM model: multimodal always wins for images; text requests
    spread across the available text models by lowest penalty (fair routing)."""
    if has_img:
        return NIM_MODEL_MULTIMODAL
    candidates = [m["id"] for m in NIM_MODEL_LIST if not m.get("multimodal")]
    if not candidates:
        candidates = [m["id"] for m in NIM_MODEL_LIST]
    return min(candidates, key=lambda mid: backend_penalty.score(f"nim:{mid}"))


def _nim_penalty_key(body: dict) -> str:
    has_img = _has_images(body.get("messages", []))
    model = _pick_nim_model(has_img)
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


async def _local_fallback_stream(body: dict):
    """Last-resort stream from the local backend, tried after every external
    provider failed. Waits out an unloading backend instead of failing."""
    try:
        await scheduler.acquire_stream(timeout=QUEUE_TIMEOUT)
    except asyncio.TimeoutError:
        raise
    stream_url = f"{BACKEND_URL}/v1/chat/completions"
    try:
        stream_body = (prepare_fastllm_body(body, FASTLLM_CHAT_TEMPLATE)
                       if FASTLLM_MODE else body)
        opened_stream = await _open_backend_stream(stream_url, stream_body)
    except (httpx.TransportError, BackendLifecycleError):
        scheduler.release_stream()
        raise
    try:
        async for chunk in _proxy_stream(
                stream_url, stream_body, opened_stream=opened_stream):
            yield chunk
    finally:
        scheduler.release_stream()

def _external_backend_order(preferred_backend: str = "") -> list[str]:
    backends = ["nim"]
    if OR_API_KEY:
        backends.append("or")
    if ZEN_API_KEY:
        backends.append("zen")
    if preferred_backend in backends:
        backends.remove(preferred_backend)
        backends.insert(0, preferred_backend)
    return backends


async def _external_chat(backend: str, body: dict) -> dict:
    if backend == "nim":
        return await _nim_chat(body)
    if backend == "or":
        return await _or_chat(body)
    if backend == "zen":
        return await _zen_chat(body)
    raise ValueError(f"unsupported external backend: {backend}")


async def _fallback_openai_stream(
    body: dict,
    preferred_backend: str = "",
):
    providers = [
        {
            "nim": _nim_stream,
            "or": _or_stream,
            "zen": _zen_stream,
        }[backend]
        for backend in _external_backend_order(preferred_backend)
    ]
    if not BENCHMARK_MODE:
        providers.append(_local_fallback_stream)

    for provider in providers:
        yielded = False
        try:
            async for chunk in provider(body):
                yielded = True
                yield chunk
            yield b"data: [DONE]\n\n"
            return
        except Exception:
            if yielded:
                break
    yield b"data: " + json.dumps(
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "error"}]}
    ).encode() + b"\n\n"
    yield b"data: [DONE]\n\n"


async def _rescue_stream(gen):
    """Wrap a backend stream so that mid-stream errors yield OpenAI error SSE
    instead of crashing the whole StreamingResponse."""
    try:
        async for chunk in gen:
            yield chunk
    except Exception as exc:
        print(f"[backend] stream failed url={BACKEND_URL} "
              f"error={type(exc).__name__}: {exc}", flush=True)
        yield b"data: " + json.dumps(
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "error"}]}
        ).encode() + b"\n\n"
    yield b"data: [DONE]\n\n"

def _backend_reloading_error():
    return {
        "error": {
            "message": "FastLLM backend is reloading; retry shortly.",
            "type": "service_unavailable",
            "param": None,
            "code": "backend_reloading",
        }
    }

def _backend_reloading_response():
    return JSONResponse(_backend_reloading_error(), status_code=503)


def _log_backend_failure(stage: str, exc: BaseException):
    print(f"[backend] stage={stage} url={BACKEND_URL} "
          f"error={type(exc).__name__}: {exc}", flush=True)


def _log_backend_status(stage: str, status: int):
    if status >= 500:
        print(f"[backend] stage={stage} url={BACKEND_URL} status={status}", flush=True)


def _backend_reloading_sse():
    payload = _backend_reloading_error()
    payload["status"] = 503
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


async def _open_backend_stream(url: str, body: dict, timeout: float = 600):
    lease = await backend_lifecycle.acquire(
        timeout=min(float(timeout), float(QUEUE_TIMEOUT)))
    client = httpx.AsyncClient(timeout=timeout)
    try:
        request = client.build_request("POST", url, json=body)
        response = await client.send(request, stream=True)
    except BaseException:
        await client.aclose()
        await lease.release()
        raise
    return client, response, lease


async def _close_backend_stream(opened_stream):
    client, response = opened_stream[:2]
    lease = opened_stream[2] if len(opened_stream) > 2 else None
    try:
        await response.aclose()
    finally:
        try:
            await client.aclose()
        finally:
            if lease is not None:
                await lease.release()


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


async def _tcp_backend_ready(timeout: float = 3) -> bool:
    """TCP readiness for backends without an HTTP /health endpoint (FastLLM)."""
    from urllib.parse import urlparse
    parsed = urlparse(BACKEND_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout)
        writer.close()
        await writer.wait_closed()
        return True
    except Exception:
        return False

# ─── Process manager ────────────────────────────────────────────

class LlamaManager:
    def __init__(self):
        self.proc = None
        self.restarts = 0
        self._stopping = False
        self._adopted = False

    async def start(self):
        if FASTLLM_MODE:
            # FastLLM has a separate lifecycle manager; this legacy manager
            # only keeps the cached TCP readiness signal current.
            self._adopted = True
            self.proc = None
            asyncio.create_task(self.monitor())
            ownership = "owned, cold-start" if FASTLLM_OWNED else "external"
            print(
                f"[manager] FastLLM backend {BACKEND_URL} configured "
                f"({ownership}); llama-server spawn disabled",
                flush=True,
            )
            return
        try:
            async with httpx.AsyncClient(timeout=3) as c:
                r = await c.get(f"{BACKEND_URL}{BACKEND_HEALTH_PATH}")
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
                    r = await c.get(f"{BACKEND_URL}{BACKEND_HEALTH_PATH}", timeout=3)
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
                if FASTLLM_MODE:
                    BACKEND_READY = await _tcp_backend_ready(timeout=HEALTH_TIMEOUT)
                else:
                    async with httpx.AsyncClient() as c:
                        r = await c.get(
                            f"{BACKEND_URL}{BACKEND_HEALTH_PATH}", timeout=HEALTH_TIMEOUT)
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


class OwnedFastLLMChild:
    __slots__ = ("process", "generation", "_control_token")

    def __init__(self, process, generation, control_token):
        self.process = process
        self.generation = int(generation)
        self._control_token = str(control_token)

    @property
    def pid(self):
        return self.process.pid

    @property
    def returncode(self):
        return self.process.returncode

    def __repr__(self):
        return (
            f"OwnedFastLLMChild(generation={self.generation}, "
            f"pid={self.pid}, returncode={self.returncode})"
        )


async def _spawn_owned_fastllm(generation):
    argv = shlex.split(FASTLLM_BACKEND_COMMAND)
    if not argv:
        raise BackendLifecycleError("FASTLLM_BACKEND_COMMAND is empty")
    log_path = Path(FASTLLM_BACKEND_LOG)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "ab", buffering=0)
    log_file.write(
        f"\n[proxy] starting FastLLM generation {generation} at "
        f"{datetime.datetime.now().isoformat()}\n".encode())
    control_token = secrets.token_urlsafe(32)
    child_env = os.environ.copy()
    child_env["FASTLLM_PREFIX_CACHE_CONTROL_TOKEN"] = control_token
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=log_file,
            stderr=asyncio.subprocess.STDOUT,
            cwd=FASTLLM_BACKEND_CWD,
            env=child_env,
            start_new_session=True,
        )
    finally:
        log_file.close()
    print(
        f"[lifecycle] starting owned FastLLM generation {generation}, "
        f"pid={process.pid}",
        flush=True,
    )
    return OwnedFastLLMChild(process, generation, control_token)
async def _checkpoint_owned_fastllm(child):
    process = child.process
    if process.returncode is not None:
        raise BackendLifecycleError(
            f"FastLLM backend exited with code {process.returncode} "
            "before prefix-cache checkpoint")
    if not child._control_token:
        raise BackendLifecycleError(
            "owned FastLLM control token is unavailable")
    headers = {
        "Authorization": f"Bearer {child._control_token}"
    }
    async with httpx.AsyncClient(
            timeout=FASTLLM_PREFIX_CACHE_CHECKPOINT_TIMEOUT) as client:
        response = await client.post(
            f"{BACKEND_URL}/admin/prefix-cache/checkpoint",
            headers=headers,
        )
    if response.status_code != 200:
        detail = response.text[:512].replace("\n", " ")
        raise BackendLifecycleError(
            "FastLLM prefix-cache checkpoint returned "
            f"HTTP {response.status_code}: {detail}")
    result = response.json()
    if not isinstance(result, dict) or result.get("status") != "ok":
        raise BackendLifecycleError(
            "FastLLM prefix-cache checkpoint returned an invalid response")
    return result


async def _suspend_owned_fastllm(child, tier="memory"):
    """POST /admin/suspend：后端释放全部 GPU 内存但保持进程常驻。"""
    process = child.process
    if process.returncode is not None:
        raise BackendLifecycleError(
            f"FastLLM backend exited with code {process.returncode} "
            "before suspend")
    if not child._control_token:
        raise BackendLifecycleError(
            "owned FastLLM control token is unavailable")
    headers = {
        "Authorization": f"Bearer {child._control_token}"
    }
    async with httpx.AsyncClient(
            timeout=FASTLLM_SUSPEND_TIMEOUT) as client:
        response = await client.post(
            f"{BACKEND_URL}/admin/suspend",
            headers=headers,
            json={"tier": tier},
        )
    if response.status_code != 200:
        detail = response.text[:512].replace("\n", " ")
        raise BackendLifecycleError(
            "FastLLM suspend returned "
            f"HTTP {response.status_code}: {detail}")
    result = response.json()
    if not isinstance(result, dict) or result.get("status") != "ok":
        raise BackendLifecycleError(
            "FastLLM suspend returned an invalid response")
    return result


async def _resume_owned_fastllm(child):
    """POST /admin/resume：把权重+KV 状态恢复回 GPU，进程保持同一 pid。"""
    process = child.process
    if process.returncode is not None:
        raise BackendLifecycleError(
            f"FastLLM backend exited with code {process.returncode} "
            "before resume")
    if not child._control_token:
        raise BackendLifecycleError(
            "owned FastLLM control token is unavailable")
    headers = {
        "Authorization": f"Bearer {child._control_token}"
    }
    async with httpx.AsyncClient(
            timeout=FASTLLM_RESUME_TIMEOUT) as client:
        response = await client.post(
            f"{BACKEND_URL}/admin/resume",
            headers=headers,
        )
    if response.status_code != 200:
        detail = response.text[:512].replace("\n", " ")
        raise BackendLifecycleError(
            "FastLLM resume returned "
            f"HTTP {response.status_code}: {detail}")
    result = response.json()
    if not isinstance(result, dict) or result.get("status") != "ok":
        raise BackendLifecycleError(
            "FastLLM resume returned an invalid response")
    return result


async def _stop_owned_fastllm(child):
    global BACKEND_READY
    BACKEND_READY = False
    process = child.process
    if process.returncode is not None:
        await process.wait()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        await process.wait()
        return
    try:
        await asyncio.wait_for(
            process.wait(), timeout=FASTLLM_STOP_TIMEOUT)
    except asyncio.TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
    print(
        f"[lifecycle] stopped owned FastLLM pid={process.pid}, "
        f"code={process.returncode}",
        flush=True,
    )


async def _probe_lifecycle_backend(child):
    global BACKEND_READY
    process = child.process
    if process.returncode is not None:
        raise BackendLifecycleError(
            f"FastLLM backend exited with code {process.returncode}")
    ready = await _tcp_backend_ready(timeout=HEALTH_TIMEOUT)
    BACKEND_READY = ready
    return ready


def _reset_local_cache_expectations():
    prefix_tracker.reset_local()


backend_lifecycle = BackendLifecycleManager(
    owned=FASTLLM_OWNED,
    start_backend=_spawn_owned_fastllm,
    stop_backend=_stop_owned_fastllm,
    probe_ready=_probe_lifecycle_backend,
    reset_local=_reset_local_cache_expectations,
    checkpoint_backend=(
        _checkpoint_owned_fastllm
        if FASTLLM_PREFIX_CACHE_PERSIST
        else None
    ),
    suspend_backend=(
        _suspend_owned_fastllm
        if FASTLLM_OWNED and FASTLLM_IDLE_SUSPEND
        else None
    ),
    resume_backend=(
        _resume_owned_fastllm
        if FASTLLM_OWNED and FASTLLM_IDLE_SUSPEND
        else None
    ),
    start_timeout=(
        FASTLLM_START_TIMEOUT if FASTLLM_OWNED else HEALTH_TIMEOUT),
    idle_timeout=FASTLLM_IDLE_TIMEOUT,
    minimum_free_bytes=FASTLLM_VRAM_MIN_FREE_BYTES,
    resume_free_bytes=FASTLLM_VRAM_RESUME_FREE_BYTES,
    high_used_ratio=FASTLLM_VRAM_HIGH_WATERMARK,
    resume_used_ratio=FASTLLM_VRAM_RESUME_WATERMARK,
)
lifecycle_watchdog_task = None


async def _read_vram_sample():
    if (
        FASTLLM_VRAM_MIN_FREE_BYTES <= 0
        and FASTLLM_VRAM_HIGH_WATERMARK <= 0
    ):
        return None
    try:
        process = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            f"--id={FASTLLM_GPU_INDEX}",
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(
            process.communicate(), timeout=HEALTH_TIMEOUT)
        if process.returncode != 0:
            return None
        values = stdout.decode().strip().splitlines()[0].split(",")
        free_mib, total_mib = (int(value.strip()) for value in values[:2])
        return free_mib * 1024 ** 2, total_mib * 1024 ** 2
    except (IndexError, OSError, ValueError, asyncio.TimeoutError):
        return None


async def _lifecycle_watchdog():
    while True:
        try:
            await asyncio.sleep(FASTLLM_LIFECYCLE_INTERVAL)
            child = backend_lifecycle.child
            if child is not None and child.returncode is not None:
                if await backend_lifecycle.observe_child_exit(child):
                    print(
                        f"[lifecycle] owned FastLLM exited, "
                        f"code={child.returncode}; next request will reload",
                        flush=True,
                    )
            await backend_lifecycle.check_idle()
            sample = await _read_vram_sample()
            if sample is not None:
                changed = await backend_lifecycle.observe_memory(
                    free_bytes=sample[0], total_bytes=sample[1])
                if changed:
                    print(
                        f"[lifecycle] VRAM pressure="
                        f"{backend_lifecycle.pressure}, "
                        f"free={sample[0] / 1024 ** 3:.2f} GiB",
                        flush=True,
                    )
                if backend_lifecycle.pressure_stale(
                        grace_seconds=60.0):
                    print(
                        "[lifecycle] VRAM pressure sustained >60s, unloading",
                        flush=True,
                    )
                    await backend_lifecycle.stop("memory_pressure")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(
                f"[lifecycle] watchdog error: {type(exc).__name__}: {exc}",
                flush=True,
            )


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


# FastLLM's vision path is safest when every inline image is normalized to
# an RGB PNG.  Do this even for PNG inputs: palette/alpha/EXIF variants can
# otherwise take different decoder paths or arrive with an unexpected layout.
def _convert_images(messages):
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url") or {}
            url = image_url.get("url", "")
            if not url.startswith("data:"):
                continue
            header, _, b64 = url.partition(",")
            fmt = ""
            if "/" in header:
                fmt = header.split("/", 1)[1].split(";", 1)[0].lower()
            try:
                raw = base64.b64decode(b64)
                with Image.open(io.BytesIO(raw)) as source:
                    orientation = source.getexif().get(0x0112, 1)
                    if orientation not in (1, None) or source.mode != "RGB":
                        # EXIF rotation applied or non-RGB: re-encode as PNG.
                        img = ImageOps.exif_transpose(source).convert("RGB")
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        new_b64 = base64.b64encode(buf.getvalue()).decode()
                        image_url["url"] = f"data:image/png;base64,{new_b64}"
                    # Otherwise keep the original bytes (JPEG/PNG as-is);
                    # the backend decodes by magic, so re-encoding only
                    # inflates the request body (JPEG 4K -> PNG ~2.7x).
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
            "max_tokens": 16384,
            "reasoning": True,
            "capabilities": {
                "vision": True,
                "function_calling": True,
                "streaming": True,
            },
            "supported_features": {
                "vision": True,
                "functionCall": True,
                "streaming": True,
                "reasoning": True,
            },
            "modalities": ["text", "image"],
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
        },
        {
            "id": "qwen3.6-27b-heretic",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
            "context_length": 262144,
            "max_tokens": 16384,
            "reasoning": True,
            "capabilities": {
                "vision": True,
                "function_calling": True,
                "streaming": True,
            },
            "supported_features": {
                "vision": True,
                "functionCall": True,
                "streaming": True,
                "reasoning": True,
            },
            "modalities": ["text", "image"],
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
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

    _normalize_tool_schema(body)

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
    requested_model = body.get("model", "")
    # Classify the caller's public model before rewriting aliases. Any local
    # alias (heretic suffix, public alias, or slug) bypasses the fair router;
    # rerouting an explicitly-local model to external providers breaks tool
    # calls and serves a different model than the caller asked for.
    is_heretic = _is_local_alias(requested_model)
    if requested_model:
        body["model"] = _to_backend_model(requested_model)
    # FastLLM is the only backend with the loaded local vision projector. A
    # FastLLM /slots probe is not available on this apiserver, so image calls
    # must remain local even when they use the fair-routed AWQ alias.
    is_heretic = is_heretic or (
        FASTLLM_MODE and _has_images(body.get("messages", []))
    )
    priority = 0 if is_heretic else 1

    if body.get("stream"):
        if not is_heretic and FALLBACK_ENABLED and not BENCHMARK_MODE and _over_context(body):
            return StreamingResponse(
                _rescue_stream(_nim_stream(body)), media_type="text/event-stream")
        if not is_heretic and FALLBACK_ENABLED and not BENCHMARK_MODE:
            backend = await _pick_backend(body)
            if backend != "local":
                return StreamingResponse(
                    _fallback_openai_stream(body, backend),
                    media_type="text/event-stream",
                )
        timeout = QUEUE_TIMEOUT * 2 if is_heretic else QUEUE_TIMEOUT
        try:
            await scheduler.acquire_stream(timeout=timeout)
        except asyncio.TimeoutError:
            if is_heretic:
                return JSONResponse({"error": "queue timeout"}, status_code=503)
            if FALLBACK_ENABLED and not BENCHMARK_MODE:
                return StreamingResponse(
                    _fallback_openai_stream(body),
                    media_type="text/event-stream")
            return JSONResponse({"error": "all backends busy"}, status_code=503)
        stream_url = f"{BACKEND_URL}/v1/chat/completions"
        try:
            stream_body = (prepare_fastllm_body(body, FASTLLM_CHAT_TEMPLATE)
                           if FASTLLM_MODE else body)
            opened_stream = await _open_backend_stream(stream_url, stream_body)
        except (httpx.TransportError, BackendLifecycleError) as exc:
            scheduler.release_stream()
            backend_penalty.record_failure("local")
            service_history.record("local", False)
            _log_backend_failure("openai_stream", exc)
            return _backend_reloading_response()
        except BaseException:
            scheduler.release_stream()
            raise
        return StreamingResponse(
            _local_stream(
                stream_url, stream_body, request, requested_model,
                allow_fallback=not is_heretic,
                opened_stream=opened_stream, body_is_prepared=True),
            media_type="text/event-stream")

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
            _log_backend_status("openai_chat", resp.status_code)
            return _rewrite_local_response(
                Response(content=resp.content, status_code=resp.status_code,
                         media_type=resp.headers.get("content-type", "application/json")),
                requested_model)
        except asyncio.TimeoutError:
            return JSONResponse({"error": "local backend busy, retry later"}, status_code=503)
        except (httpx.TransportError, BackendLifecycleError) as exc:
            _log_backend_failure("openai_chat", exc)
            return _backend_reloading_response()

    last_err = ""
    backend = await _pick_backend(body) if not is_heretic else "local"

    if backend == "local":
        prefix_tracker.reserve(body.get("messages", []))
        try:
            to = QUEUE_TIMEOUT * 2 if is_heretic else 1
            resp = await scheduler.submit(body, priority, timeout=to)
            if resp.status_code == 200:
                return _rewrite_local_response(
                    Response(content=resp.content, status_code=200,
                             media_type=resp.headers.get("content-type", "application/json")),
                    requested_model)
            print(f"[route] local non-200 {resp.status_code}", flush=True)
        except asyncio.TimeoutError:
            last_err = "local_timeout"
            print(f"[route] local timeout ({to}s)", flush=True)
        except (httpx.TransportError, BackendLifecycleError) as exc:
            last_err = "local_down"
            _log_backend_failure("openai_chat_fallback", exc)
            print("[route] local down", flush=True)
        if is_heretic:
            return JSONResponse({"error": f"local backend {last_err}"}, status_code=503)

    if not is_heretic:
        for external_backend in _external_backend_order(backend):
            try:
                print(
                    f"[route] routing to {external_backend}",
                    flush=True,
                )
                return JSONResponse(
                    await _external_chat(external_backend, body)
                )
            except (
                RuntimeError,
                httpx.TimeoutException,
                httpx.RequestError,
            ) as exc:
                last_err = f"{external_backend}_down"
                print(
                    f"[route] {external_backend} failed: {exc}",
                    flush=True,
                )

    # External providers rejected or errored (sensitive-content refusal,
    # delisted model, outage): fall back to the local Heretic backend so the
    # caller gets a response instead of a hard failure.
    if not is_heretic and not BENCHMARK_MODE:
        prefix_tracker.reserve(body.get("messages", []))
        try:
            print("[route] external backends failed, falling back to local", flush=True)
            resp = await scheduler.submit(body, priority, timeout=QUEUE_TIMEOUT * 2)
            _log_backend_status("openai_chat_local_fallback", resp.status_code)
            if resp.status_code == 200:
                return _rewrite_local_response(
                    Response(content=resp.content, status_code=200,
                             media_type=resp.headers.get("content-type", "application/json")),
                    requested_model)
            last_err = f"local_fallback_{resp.status_code}"
        except asyncio.TimeoutError:
            last_err = "local_fallback_timeout"
        except (httpx.TransportError, BackendLifecycleError) as exc:
            last_err = "local_fallback_down"
            _log_backend_failure("openai_chat_local_fallback", exc)

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
    _normalize_tool_schema(openai_body)
    _convert_images(openai_body.get("messages", []))
    requested_model = openai_body.get("model", "")
    # Preserve public-ID routing semantics before the backend slug rewrite.
    # Local aliases bypass the fair router (see _is_local_alias).
    is_heretic = _is_local_alias(requested_model)
    if requested_model:
        openai_body["model"] = _to_backend_model(requested_model)
    # Keep Anthropic/mobile image requests on the local FastLLM projector too.
    is_heretic = is_heretic or (
        FASTLLM_MODE and _has_images(openai_body.get("messages", []))
    )
    priority = 0 if is_heretic else 1
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
                    async for chunk in _completed_anthropic_stream(anthro, request):
                        yield chunk
                return StreamingResponse(_ctx_overflow_stream(), media_type="text/event-stream")
            return JSONResponse({"error": "context exceeds limit, fallback failed"}, status_code=429)
        return StreamingResponse(
            _anthropic_stream_limited(openai_body, request, requested_model),
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
            _log_backend_status("anthropic_chat", resp.status_code)
            if resp.status_code == 200:
                return JSONResponse(_openai_to_anthropic(_restore_public_model(resp.json(), requested_model)))
            return Response(content=resp.content, status_code=resp.status_code,
                          media_type="application/json")
        except asyncio.TimeoutError:
            return JSONResponse({"error": "local backend busy, retry later"}, status_code=503)
        except (httpx.TransportError, BackendLifecycleError) as exc:
            _log_backend_failure("anthropic_chat", exc)
            return _backend_reloading_response()

    last_err = ""
    backend = await _pick_backend(openai_body) if not is_heretic else "local"

    if backend == "local":
        prefix_tracker.reserve(openai_body.get("messages", []))
        try:
            to = QUEUE_TIMEOUT * 2 if is_heretic else 1
            resp = await scheduler.submit(openai_body, priority, timeout=to)
            if resp.status_code == 200:
                return JSONResponse(_openai_to_anthropic(_restore_public_model(resp.json(), requested_model)))
            print(f"[route] local non-200 {resp.status_code}", flush=True)
        except asyncio.TimeoutError:
            last_err = "local_timeout"
            print(f"[route] local timeout ({to}s)", flush=True)
        except (httpx.TransportError, BackendLifecycleError) as exc:
            last_err = "local_down"
            _log_backend_failure("anthropic_chat_fallback", exc)
            print("[route] local down", flush=True)
        if is_heretic:
            return JSONResponse({"error": f"local backend {last_err}"}, status_code=503)

    if not is_heretic:
        for external_backend in _external_backend_order(backend):
            try:
                print(
                    f"[route] routing to {external_backend}",
                    flush=True,
                )
                external_data = await _external_chat(
                    external_backend, openai_body
                )
                return JSONResponse(_openai_to_anthropic(external_data))
            except (
                RuntimeError,
                httpx.TimeoutException,
                httpx.RequestError,
            ) as exc:
                last_err = f"{external_backend}_down"
                print(
                    f"[route] {external_backend} failed: {exc}",
                    flush=True,
                )

    # External providers rejected or errored: fall back to the local Heretic
    # backend instead of a hard failure (sensitive-content refusal etc.).
    if not is_heretic and not BENCHMARK_MODE:
        prefix_tracker.reserve(openai_body.get("messages", []))
        try:
            print("[route] external backends failed, falling back to local", flush=True)
            resp = await scheduler.submit(openai_body, priority, timeout=QUEUE_TIMEOUT * 2)
            _log_backend_status("anthropic_chat_local_fallback", resp.status_code)
            if resp.status_code == 200:
                local_data = json.loads(resp.content)
                return JSONResponse(_openai_to_anthropic(local_data))
            last_err = f"local_fallback_{resp.status_code}"
        except (asyncio.TimeoutError, httpx.TransportError, BackendLifecycleError) as exc:
            last_err = "local_fallback_down"
            _log_backend_failure("anthropic_chat_local_fallback", exc)

    return JSONResponse({"error": f"all backends exhausted ({last_err})"}, status_code=429,
                        headers={"Retry-After": "30"})


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(request: Request):
    try:
        body = await request.json()
    except ClientDisconnect:
        return Response(status_code=499)
    openai_body = _anthropic_to_openai(body)
    openai_body["model"] = _to_backend_model(openai_body.get("model", ""))
    openai_body["max_tokens"] = 1
    openai_body["stream"] = False
    count_body = openai_body
    if FASTLLM_MODE:
        count_body = prepare_fastllm_body(openai_body, FASTLLM_CHAT_TEMPLATE)
    try:
        lease = await backend_lifecycle.acquire(timeout=QUEUE_TIMEOUT)
    except (asyncio.TimeoutError, BackendLifecycleError):
        return _backend_reloading_response()
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{BACKEND_URL}/v1/chat/completions", json=count_body)
            data = resp.json()
    except httpx.TransportError as exc:
        _log_backend_failure("count_tokens", exc)
        return _backend_reloading_response()
    finally:
        await lease.release()
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
        result["tools"] = []
        for tool in body["tools"]:
            if tool.get("type", "").startswith("web_search") and not tool.get(
                "input_schema"
            ):
                name = "web_search"
                description = tool.get(
                    "description", "Search the web for current information."
                )
                parameters = {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        },
                    },
                    "required": ["query"],
                }
            else:
                name = tool.get("name", "")
                description = tool.get("description", "")
                parameters = tool.get("input_schema", {})
            result["tools"].append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            })
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

async def _local_stream(
    url: str,
    body: dict,
    request: Request,
    public_model: str = "",
    allow_fallback: bool = True,
    opened_stream=None,
    body_is_prepared: bool = False,
):
    if FASTLLM_MODE and not body_is_prepared:
        body = prepare_fastllm_body(body, FASTLLM_CHAT_TEMPLATE)
    prefix_tracker.reserve(body.get("messages", []))
    local_failed = False
    pending = b""
    try:
        stream = (_proxy_stream(url, body, opened_stream=opened_stream)
                  if opened_stream is not None else _proxy_stream(url, body))
        async for chunk in stream:
            if FASTLLM_MODE:
                pending += chunk
                while True:
                    crlf_end = pending.find(b"\r\n\r\n")
                    lf_end = pending.find(b"\n\n")
                    ends = [end for end in (crlf_end, lf_end) if end >= 0]
                    if not ends:
                        break
                    end = min(ends)
                    separator = (b"\r\n\r\n"
                                 if pending.startswith(b"\r\n\r\n", end)
                                 else b"\n\n")
                    event_end = end + len(separator)
                    event, pending = pending[:event_end], pending[event_end:]
                    yield _rewrite_stream_model(event, public_model)
                    data_lines = [
                        line[5:].lstrip()
                        for line in event.splitlines()
                        if line.startswith(b"data:")
                    ]
                    if b"\n".join(data_lines) == b"[DONE]":
                        return
            else:
                yield chunk
            if await request.is_disconnected():
                return
        if pending:
            yield _rewrite_stream_model(pending, public_model)
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
        if allow_fallback and FALLBACK_ENABLED and not BENCHMARK_MODE:
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
        yield _backend_reloading_sse()
        yield b"data: [DONE]\n\n"


async def _completed_anthropic_stream(
    anthro: dict,
    request: Request | None = None,
):
    message = {
        key: value
        for key, value in anthro.items()
        if key not in {"content", "stop_reason", "stop_sequence", "usage"}
    }
    usage = anthro.get("usage", {})
    message.update({
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": 0,
        },
    })
    yield _sse("message_start", {
        "type": "message_start",
        "message": message,
    })

    for index, block in enumerate(anthro.get("content", [])):
        block_type = block.get("type")
        if block_type == "thinking":
            start_block = {
                "type": "thinking",
                "thinking": "",
                "signature": "",
            }
        elif block_type == "text":
            start_block = {"type": "text", "text": ""}
        elif block_type == "tool_use":
            start_block = {
                "type": "tool_use",
                "id": block.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                "name": block.get("name", ""),
                "input": {},
            }
        else:
            raise RuntimeError(
                f"unsupported Anthropic content block: {block_type!r}"
            )

        yield _sse("content_block_start", {
            "type": "content_block_start",
            "index": index,
            "content_block": start_block,
        })
        if block_type == "thinking":
            if block.get("thinking"):
                yield _sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "thinking_delta",
                        "thinking": block["thinking"],
                    },
                })
            if block.get("signature"):
                yield _sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "signature_delta",
                        "signature": block["signature"],
                    },
                })
        elif block_type == "text" and block.get("text"):
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": block["text"]},
            })
        elif block_type == "tool_use":
            partial_json = json.dumps(
                block.get("input", {}),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": partial_json,
                },
            })
        yield _sse("content_block_stop", {
            "type": "content_block_stop",
            "index": index,
        })
        if request is not None and await request.is_disconnected():
            return

    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {
            "stop_reason": anthro.get("stop_reason", "end_turn"),
            "stop_sequence": anthro.get("stop_sequence"),
        },
        "usage": {"output_tokens": usage.get("output_tokens", 0)},
    })
    yield _sse("message_stop", {"type": "message_stop"})


async def _anthropic_external_stream(
    openai_body: dict,
    request: Request,
    preferred_backend: str = "",
):
    for external_backend in _external_backend_order(preferred_backend):
        try:
            external_data = await _external_chat(
                external_backend, openai_body
            )
        except (
            RuntimeError,
            httpx.TimeoutException,
            httpx.RequestError,
        ):
            continue

        anthro = _openai_to_anthropic(external_data)
        async for chunk in _completed_anthropic_stream(anthro, request):
            yield chunk
        return

    yield _sse("error", {"error": {"message": "all backends failed"}})


async def _anthropic_stream_limited(
    openai_body: dict,
    request: Request,
    public_model: str = "",
):
    requested_model = public_model or openai_body.get("model", "")
    is_heretic = _is_local_alias(requested_model) or (
        FASTLLM_MODE and _has_images(openai_body.get("messages", []))
    )
    if not is_heretic and FALLBACK_ENABLED and not BENCHMARK_MODE:
        backend = await _pick_backend(openai_body)
        if backend != "local":
            async for chunk in _anthropic_external_stream(
                openai_body, request, backend
            ):
                yield chunk
            return
    timeout = QUEUE_TIMEOUT * 2 if is_heretic else QUEUE_TIMEOUT
    try:
        await scheduler.acquire_stream(timeout=timeout)
    except asyncio.TimeoutError:
        if is_heretic:
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
            async for chunk in _completed_anthropic_stream(anthro, request):
                yield chunk
            return
        yield _sse("error", {"error": {"message": "queue timeout"}})
        return
    try:
        lease = await backend_lifecycle.acquire(timeout=timeout)
    except (asyncio.TimeoutError, BackendLifecycleError) as exc:
        scheduler.release_stream()
        yield _sse("error", {"error": {"message": str(exc)}})
        return
    prefix_tracker.reserve(openai_body.get("messages", []))
    local_failed = False
    yielded_local_event = False
    try:
        async for chunk in _anthropic_stream(openai_body, public_model):
            yielded_local_event = True
            yield chunk
            if await request.is_disconnected():
                return
    except Exception:
        local_failed = True
    finally:
        prefix_tracker.record(openai_body.get("messages", []), "local")
        scheduler.release_stream()
        await lease.release()

    if local_failed:
        if yielded_local_event:
            yield _sse("error", _backend_reloading_error())
            return
        if not is_heretic and FALLBACK_ENABLED and not BENCHMARK_MODE:
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
            async for chunk in _completed_anthropic_stream(anthro, request):
                yield chunk
            return
        yield _sse("error", _backend_reloading_error())


async def _iter_sse_payloads(lines):
    data_lines = []
    async for line in lines:
        if line:
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            continue
        if not data_lines:
            continue
        yield "\n".join(data_lines)
        data_lines.clear()
    if data_lines:
        yield "\n".join(data_lines)


async def _proxy_stream(url: str, body: dict, opened_stream=None):
    if opened_stream is None:
        opened_stream = await _open_backend_stream(url, body)
    resp = opened_stream[1]
    try:
        if resp.status_code != 200:
            err_body = await resp.aread()
            raise RuntimeError(
                f"backend returned {resp.status_code}: {err_body.decode()[:200]}")
        async for chunk in resp.aiter_bytes():
            yield chunk
    finally:
        await _close_backend_stream(opened_stream)


async def _anthropic_stream(openai_body: dict, public_model: str = ""):
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "content": [], "model": public_model or openai_body.get("model", ""),
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    block_idx = 0
    cur_type = None
    finish_reason = None
    in_tokens = 0
    out_tokens = 0
    saw_json_event = False

    stream_body = openai_body
    if FASTLLM_MODE:
        stream_body = prepare_fastllm_body(openai_body, FASTLLM_CHAT_TEMPLATE)
    async with httpx.AsyncClient(timeout=600) as client:
        async with client.stream(
            "POST", f"{BACKEND_URL}/v1/chat/completions", json=stream_body
        ) as resp:
            if resp.status_code != 200:
                err_body = await resp.aread()
                raise RuntimeError(
                    f"backend returned {resp.status_code}: {err_body.decode()[:200]}")
            async for data in _iter_sse_payloads(resp.aiter_lines()):
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("malformed backend SSE event") from exc
                saw_json_event = True

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


                # Convert OpenAI tool_calls deltas to Anthropic tool_use blocks.
                tcs = delta.get("tool_calls")
                if tcs:
                    for tc in tcs:
                        tc_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                        fn = tc.get("function", {})
                        fn_name = fn.get("name", "")
                        fn_args = fn.get("arguments", "")
                        if fn_name:
                            if cur_type is not None:
                                yield _sse("content_block_stop", {
                                    "type": "content_block_stop", "index": block_idx})
                                block_idx += 1
                            yield _sse("content_block_start", {
                                "type": "content_block_start", "index": block_idx,
                                "content_block": {
                                    "type": "tool_use", "id": tc_id,
                                    "name": fn_name, "input": {},
                                },
                            })
                            cur_type = "tool_use"
                        if fn_args and cur_type == "tool_use":
                            yield _sse("content_block_delta", {
                                "type": "content_block_delta", "index": block_idx,
                                "delta": {"type": "input_json_delta", "partial_json": fn_args},
                            })
                u = ev.get("usage")
                if u:
                    in_tokens = u.get("prompt_tokens", in_tokens)
                    out_tokens = u.get("completion_tokens", out_tokens)
    if not saw_json_event or finish_reason is None:
        raise RuntimeError("backend stream ended without a terminal event")

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


# ─── Health ────────────────────────────────────────────────────

@app.get("/health")
async def health(request: Request):
    """Report backend state without activating an owned cold backend."""
    lifecycle = backend_lifecycle.snapshot()
    ready = (
        lifecycle["state"] == "READY"
        if FASTLLM_OWNED
        else BACKEND_READY
    )
    if request.query_params.get("probe") == "1":
        try:
            if FASTLLM_MODE:
                ready = (
                    lifecycle["state"] == "READY"
                    and await _tcp_backend_ready(timeout=HEALTH_TIMEOUT)
                )
            else:
                async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:
                    r = await client.get(f"{BACKEND_URL}{BACKEND_HEALTH_PATH}")
                    ready = r.status_code == 200
        except Exception:
            ready = False
    return JSONResponse({
        "status": "ok" if ready else "loading",
        "backend": "fastllm" if FASTLLM_MODE else "llama",
        "backend_url": BACKEND_URL,
        "ready": ready,
        "lifecycle": lifecycle,
    })


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
        lease = await backend_lifecycle.acquire(timeout=QUEUE_TIMEOUT)
    except (asyncio.TimeoutError, BackendLifecycleError):
        return _backend_reloading_response()
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
    except httpx.TransportError as exc:
        _log_backend_failure("passthrough", exc)
        return _backend_reloading_response()
    finally:
        await lease.release()


# ─── Startup / shutdown ─────────────────────────────────────────

@app.on_event("startup")
async def _startup():
    global lifecycle_watchdog_task
    asyncio.create_task(manager.start())
    asyncio.create_task(scheduler.start_workers())
    if FASTLLM_OWNED:
        lifecycle_watchdog_task = asyncio.create_task(
            _lifecycle_watchdog())
    print("[proxy] starting in background, backend not ready yet", flush=True)


@app.on_event("shutdown")
async def _shutdown():
    global lifecycle_watchdog_task
    if lifecycle_watchdog_task is not None:
        lifecycle_watchdog_task.cancel()
        await asyncio.gather(
            lifecycle_watchdog_task, return_exceptions=True)
        lifecycle_watchdog_task = None
    await backend_lifecycle.drain_and_stop(
        "proxy_shutdown", timeout=FASTLLM_STOP_TIMEOUT)
    if not manager._adopted and manager.proc is not None:
        await manager.stop()


# ─── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[proxy] starting on {PROXY_HOST}:{PROXY_PORT}, "
          f"backend on port {BACKEND_PORT}", flush=True)
    if FASTLLM_MODE:
        print(f"[proxy] FastLLM backend: {BACKEND_URL} (slug {FASTLLM_MODEL_SLUG}, "
              f"aliases {FASTLLM_PUBLIC_ALIASES}); llama-server spawn disabled", flush=True)
    else:
        print(f"[proxy] local backend: llama-server", flush=True)
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
