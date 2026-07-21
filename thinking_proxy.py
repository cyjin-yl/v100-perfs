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
import io
import ipaddress
import json
import os
import signal
import sys
import time
import uuid

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

# ─── Config ─────────────────────────────────────────────────────

PROJECT_DIR = os.environ.get("PROJECT_DIR", os.getcwd())
BACKEND_PORT = int(os.environ.get("BACKEND_PORT", "8001"))
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8000"))
PROXY_HOST = os.environ.get("PROXY_HOST", "0.0.0.0")
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "")
HEALTH_INTERVAL = int(os.environ.get("HEALTH_INTERVAL", "15"))
HEALTH_TIMEOUT = int(os.environ.get("HEALTH_TIMEOUT", "5"))
MAX_RESTART_BACKOFF = int(os.environ.get("MAX_RESTART_BACKOFF", "60"))
MAX_STREAMING_CONCURRENCY = int(os.environ.get("MAX_STREAMING_CONCURRENCY", "1"))
QUEUE_TIMEOUT = int(os.environ.get("QUEUE_TIMEOUT", "180"))
FALLBACK_ENABLED = os.environ.get("FALLBACK_ENABLED", "1") == "1"
BENCHMARK_MODE = os.environ.get("BENCHMARK_MODE", "0") == "1"

NIM_API_KEY = os.environ.get("NIM_API_KEY", "")
NIM_BASE_URL = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_MODEL_MULTIMODAL = os.environ.get("NIM_MODEL_MULTIMODAL", "nvidia/minimax-m3")
NIM_MODEL_TEXT = os.environ.get("NIM_MODEL_TEXT", "deepseek-ai/deepseek-v4-flash")

STREAMING_SEMAPHORE = asyncio.Semaphore(MAX_STREAMING_CONCURRENCY)
REQUEST_SEMAPHORE = asyncio.Semaphore(MAX_STREAMING_CONCURRENCY)

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
    "-c", "327680", "-ngl", "99", "-np", "2", "-fa", "on", "-fit", "off",
    "--cache-type-k", "turbo3", "--cache-type-v", "turbo3",
    "--cache-ram", "0",
    "--spec-type", "draft-mtp", "--spec-draft-n-max", "5",
    "--jinja",
    "--chat-template-file", os.path.join(
        PROJECT_DIR, "chat_templates/qwen3.6_merged.jinja"),
    "--chat-template-kwargs", '{"enable_thinking":true}',
    "--alias", "qwen3.6-27b-awq",
    "--host", "127.0.0.1", "--port", str(BACKEND_PORT),
]

STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


# ─── Fallback helpers ────────────────────────────────────────────

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
    max_tokens = body.get("max_tokens", 4096)
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


async def _nim_chat(body: dict) -> dict:
    has_img = _has_images(body.get("messages", []))
    model = NIM_MODEL_MULTIMODAL if has_img else NIM_MODEL_TEXT
    payload = {
        "model": model,
        "messages": body["messages"],
        "temperature": body.get("temperature", 0.7),
        "max_tokens": body.get("max_tokens", 4096),
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(
            f"{NIM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {NIM_API_KEY}"},
            json=payload,
        )
        if r.status_code != 200:
            raise RuntimeError(f"NIM returned {r.status_code}: {r.text[:200]}")
        return r.json()


async def _nim_stream(body: dict):
    has_img = _has_images(body.get("messages", []))
    model = NIM_MODEL_MULTIMODAL if has_img else NIM_MODEL_TEXT
    payload = {
        "model": model,
        "messages": body["messages"],
        "temperature": body.get("temperature", 0.7),
        "max_tokens": body.get("max_tokens", 4096),
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=600) as c:
        async with c.stream(
            "POST", f"{NIM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {NIM_API_KEY}"},
            json=payload,
        ) as resp:
            async for chunk in resp.aiter_bytes():
                yield chunk


# ─── Process manager ────────────────────────────────────────────

class LlamaManager:
    def __init__(self):
        self.proc = None
        self.restarts = 0
        self._stopping = False

    async def start(self):
        try:
            async with httpx.AsyncClient(timeout=3) as c:
                r = await c.get(f"{BACKEND_URL}/health")
                if r.status_code == 200:
                    print(f"[manager] adopting existing backend on port {BACKEND_PORT}", flush=True)
                    self.proc = None
                    asyncio.create_task(self.monitor())
                    return
        except Exception:
            pass
        print(f"[manager] starting llama-server on port {BACKEND_PORT}", flush=True)
        self.proc = await asyncio.create_subprocess_exec(
            *([LLAMA_BINARY] + LLAMA_ARGS),
            stdout=sys.stdout,
            stderr=sys.stderr,
            cwd=PROJECT_DIR,
            env=os.environ.copy(),
        )
        asyncio.create_task(self._wait_ready_and_monitor())

    async def _wait_ready_and_monitor(self):
        try:
            await self._wait_ready(deadline=600)
            print(f"[manager] llama-server ready (pid {self.proc.pid})", flush=True)
        except RuntimeError as e:
            print(f"[manager] {e}", flush=True)
        asyncio.create_task(self.monitor())

    async def _wait_ready(self, deadline=300):
        start = time.time()
        while time.time() - start < deadline:
            if self.proc.returncode is not None:
                raise RuntimeError(
                    f"llama-server exited early with code {self.proc.returncode}")
            try:
                async with httpx.AsyncClient() as c:
                    r = await c.get(f"{BACKEND_URL}/health", timeout=3)
                    if r.status_code == 200:
                        return
            except Exception:
                pass
            await asyncio.sleep(2)
        raise RuntimeError("llama-server did not become healthy in time")

    async def monitor(self):
        backoff = 5
        while not self._stopping:
            await asyncio.sleep(HEALTH_INTERVAL)
            if self._stopping:
                break
            crashed = self.proc is not None and self.proc.returncode is not None
            unhealthy = False
            if not crashed and self.proc is not None:
                try:
                    async with httpx.AsyncClient() as c:
                        r = await c.get(
                            f"{BACKEND_URL}/health", timeout=HEALTH_TIMEOUT)
                        unhealthy = r.status_code not in (200, 503)
                except Exception:
                    unhealthy = True

            if crashed or unhealthy:
                if crashed:
                    print(f"[manager] llama-server crashed "
                          f"(exit {self.proc.returncode})", flush=True)
                    await self.proc.wait()
                else:
                    print("[manager] health check failed, killing", flush=True)
                    self.proc.kill()
                    await self.proc.wait()
                self.restarts += 1
                print(f"[manager] restart #{self.restarts} "
                      f"in {backoff}s", flush=True)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, MAX_RESTART_BACKOFF)
                try:
                    await self.start()
                    backoff = 5
                except Exception as e:
                    print(f"[manager] restart failed: {e}", flush=True)
            else:
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


# ─── OpenAI endpoint ────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def openai_chat(request: Request):
    body = await request.json()

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
        _EFFORT_BUDGETS = {"low": 2048, "medium": 4096, "high": 8192}
        budget = thinking_budget
        if budget is None or budget <= 0:
            key = (effort or "").lower()
            if isinstance(reasoning_obj, dict):
                key = (reasoning_obj.get("effort") or "").lower()
            budget = _EFFORT_BUDGETS.get(key, 4096)
        body.setdefault("reasoning", {})["budget"] = budget

    _convert_images(body.get("messages", []))

    if body.get("stream"):
        return StreamingResponse(
            _proxy_stream_limited(f"{BACKEND_URL}/v1/chat/completions", body),
            media_type="text/event-stream",
        )

    if FALLBACK_ENABLED and not BENCHMARK_MODE and _over_context(body):
        try:
            nim_data = await _nim_chat(body)
            return JSONResponse(nim_data)
        except Exception:
            return JSONResponse({"error": "context exceeds limit, fallback failed"}, status_code=429)

    try:
        await asyncio.wait_for(REQUEST_SEMAPHORE.acquire(), timeout=QUEUE_TIMEOUT)
    except asyncio.TimeoutError:
        if not FALLBACK_ENABLED or BENCHMARK_MODE:
            return JSONResponse({"error": "queue timeout"}, status_code=503)
        try:
            nim_data = await _nim_chat(body)
            return JSONResponse(nim_data)
        except Exception:
            return JSONResponse({"error": "service unavailable"}, status_code=429)

    try:
        async with httpx.AsyncClient(timeout=600) as client:
            resp = await client.post(
                f"{BACKEND_URL}/v1/chat/completions", json=body)
            if resp.status_code == 200 or not FALLBACK_ENABLED or BENCHMARK_MODE:
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    media_type=resp.headers.get("content-type", "application/json"),
                )
    except (httpx.ConnectError, httpx.TimeoutException):
        if not FALLBACK_ENABLED or BENCHMARK_MODE:
            return JSONResponse({"error": "backend unavailable"}, status_code=503)
    finally:
        REQUEST_SEMAPHORE.release()

    try:
        nim_data = await _nim_chat(body)
        return JSONResponse(nim_data)
    except Exception:
        return JSONResponse({"error": "service unavailable"}, status_code=429)


# ─── Anthropic endpoint ─────────────────────────────────────────

@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    body = await request.json()
    stream = body.get("stream", False)
    openai_body = _anthropic_to_openai(body)
    _convert_images(openai_body.get("messages", []))

    if stream:
        return StreamingResponse(
            _anthropic_stream_limited(openai_body),
            media_type="text/event-stream",
        )

    if FALLBACK_ENABLED and not BENCHMARK_MODE and _over_context(openai_body):
        try:
            nim_data = await _nim_chat(openai_body)
            return JSONResponse(_openai_to_anthropic(nim_data))
        except Exception:
            return JSONResponse({"error": "context exceeds limit, fallback failed"}, status_code=429)

    try:
        await asyncio.wait_for(REQUEST_SEMAPHORE.acquire(), timeout=QUEUE_TIMEOUT)
    except asyncio.TimeoutError:
        if not FALLBACK_ENABLED or BENCHMARK_MODE:
            return JSONResponse({"error": "queue timeout"}, status_code=503)
        try:
            nim_data = await _nim_chat(openai_body)
            return JSONResponse(_openai_to_anthropic(nim_data))
        except Exception:
            return JSONResponse({"error": "service unavailable"}, status_code=429)

    try:
        async with httpx.AsyncClient(timeout=600) as client:
            resp = await client.post(
                f"{BACKEND_URL}/v1/chat/completions", json=openai_body)
            if resp.status_code != 200:
                if not FALLBACK_ENABLED or BENCHMARK_MODE:
                    return Response(
                        content=resp.content, status_code=resp.status_code,
                        media_type="application/json")
            else:
                return JSONResponse(_openai_to_anthropic(resp.json()))
    except (httpx.ConnectError, httpx.TimeoutException):
        if not FALLBACK_ENABLED or BENCHMARK_MODE:
            return JSONResponse({"error": "backend unavailable"}, status_code=503)
    finally:
        REQUEST_SEMAPHORE.release()

    try:
        nim_data = await _nim_chat(openai_body)
        return JSONResponse(_openai_to_anthropic(nim_data))
    except Exception:
        return JSONResponse({"error": "service unavailable"}, status_code=429)


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(request: Request):
    body = await request.json()
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
        "max_tokens": body.get("max_tokens", 4096),
        "temperature": body.get("temperature", 1.0),
        "top_p": body.get("top_p", 1.0),
        "stream": body.get("stream", False),
        "chat_template_kwargs": {"enable_thinking": enable},
    }

    if enable:
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

async def _proxy_stream_limited(url: str, body: dict):
    try:
        await asyncio.wait_for(STREAMING_SEMAPHORE.acquire(), timeout=QUEUE_TIMEOUT)
    except asyncio.TimeoutError:
        if not FALLBACK_ENABLED or BENCHMARK_MODE:
            yield b"data: " + json.dumps({"error": "queue timeout"}).encode() + b"\n\n"
            yield b"data: [DONE]\n\n"
            return
        async for chunk in _nim_stream(body):
            yield chunk
        return
    try:
        async for chunk in _proxy_stream(url, body):
            yield chunk
    finally:
        STREAMING_SEMAPHORE.release()


async def _anthropic_stream_limited(openai_body: dict):
    try:
        await asyncio.wait_for(STREAMING_SEMAPHORE.acquire(), timeout=QUEUE_TIMEOUT)
    except asyncio.TimeoutError:
        if not FALLBACK_ENABLED or BENCHMARK_MODE:
            yield _sse("error", {"error": {"message": "queue timeout"}})
            return
        try:
            nim_data = await _nim_chat(openai_body)
        except Exception:
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
    try:
        async for chunk in _anthropic_stream(openai_body):
            yield chunk
    finally:
        STREAMING_SEMAPHORE.release()


async def _proxy_stream(url: str, body: dict):
    async with httpx.AsyncClient(timeout=600) as client:
        async with client.stream("POST", url, json=body) as resp:
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
    print("[proxy] starting in background, backend not ready yet", flush=True)


@app.on_event("shutdown")
async def _shutdown():
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
