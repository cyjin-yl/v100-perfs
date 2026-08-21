#!/usr/bin/env python3
"""Behavioral checks for local streaming error classification."""

import asyncio
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "thinking_proxy_under_test", ROOT / "thinking_proxy.py"
)
proxy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proxy)

REAL_PROXY_STREAM = proxy._proxy_stream


class FakePrefix:
    def reserve(self, *_args):
        pass

    def record(self, *_args):
        pass


class FakeScheduler:
    def release_stream(self):
        pass


class FakePenalty:
    def __init__(self):
        self.failures = 0
        self.successes = 0

    def record_failure(self, *_args):
        self.failures += 1

    def record_success(self, *_args):
        self.successes += 1


class FakeHistory:
    def __init__(self):
        self.rows = []

    def record(self, *args):
        self.rows.append(args)


class FakeLifecycle:
    def __init__(self, state):
        self.state = state

    def snapshot(self):
        return {"state": self.state}


class FakeRequest:
    def __init__(self, disconnected=False):
        self.disconnected = disconnected

    async def is_disconnected(self):
        return self.disconnected


class FakeResponse:
    status_code = 200

    def __init__(self, chunks):
        self.chunks = chunks

    async def aiter_bytes(self):
        for chunk in self.chunks:
            yield chunk

    async def aread(self):
        return b""

    async def aclose(self):
        pass


class FakeClient:
    async def aclose(self):
        pass


class FakeLease:
    async def release(self):
        pass


async def run_invalid_utf8_case():
    penalty = FakePenalty()
    history = FakeHistory()
    proxy.prefix_tracker = FakePrefix()
    proxy.scheduler = FakeScheduler()
    proxy.backend_penalty = penalty
    proxy.service_history = history
    proxy.backend_lifecycle = FakeLifecycle("READY")
    proxy._proxy_stream = REAL_PROXY_STREAM
    proxy.FASTLLM_MODE = True
    proxy.FALLBACK_ENABLED = False
    response = FakeResponse([
        b'data: {"choices":[{"delta":{"content":"ok \xff"},'
        b'"index":0}]}\n\n',
        b"data: [DONE]\n\n",
    ])
    text = await collect(
        proxy._local_stream(
            "http://backend",
            {"messages": []},
            FakeRequest(),
            opened_stream=(FakeClient(), response, FakeLease()),
            body_is_prepared=True,
        )
    )
    return text, penalty, history


async def partial_break(*_args, **_kwargs):
    yield b'data: {"choices":[{"delta":{"content":"x"},"index":0}]}\n\n'
    raise RuntimeError("synthetic read break")


async def immediate_break(*_args, **_kwargs):
    if False:
        yield b""
    raise RuntimeError("synthetic open break")


async def collect(stream):
    return b"".join([chunk async for chunk in stream]).decode()


async def run_case(state, broken, disconnected=False, fallback=False):
    penalty = FakePenalty()
    history = FakeHistory()
    fallback_calls = []
    proxy.prefix_tracker = FakePrefix()
    proxy.scheduler = FakeScheduler()
    proxy.backend_penalty = penalty
    proxy.service_history = history
    proxy.backend_lifecycle = FakeLifecycle(state)
    proxy._proxy_stream = broken
    proxy.FASTLLM_MODE = True
    proxy.FALLBACK_ENABLED = fallback
    proxy.BENCHMARK_MODE = False
    proxy.OR_API_KEY = ""
    proxy.ZEN_API_KEY = ""

    async def forbidden_fallback(_body):
        fallback_calls.append(1)
        yield b"data: forbidden\n\n"

    proxy._nim_stream = forbidden_fallback
    text = await collect(
        proxy._local_stream(
            "http://backend",

            {"messages": []},
            FakeRequest(disconnected),
            opened_stream=(object(), object(), None),
            body_is_prepared=True,
        )
    )
    return text, penalty, history, fallback_calls


async def main():
    proxy.FASTLLM_MODE = True
    assert proxy._is_auto_route_alias("auto")
    assert proxy._to_backend_model("auto") == proxy.FASTLLM_MODEL_SLUG
    assert not proxy._must_route_local("auto", [])
    text, penalty, history = await run_invalid_utf8_case()
    assert "\ufffd" in text
    assert "data: [DONE]" in text
    assert "backend_stream_interrupted" not in text
    assert penalty.failures == 0
    assert penalty.successes == 1
    assert history.rows[-1][-1] is True

    text, penalty, history, fallback_calls = await run_case(
        "READY", partial_break, fallback=True
    )
    assert "backend_stream_interrupted" in text
    assert "backend_reloading" not in text
    assert penalty.failures == 1
    assert history.rows[-1][-1] is False
    assert not fallback_calls, "partial local output must not mix with fallback output"

    text, _, _, _ = await run_case("LOADING", immediate_break)
    assert "backend_reloading" in text
    assert "backend_stream_interrupted" not in text

    text, penalty, history, _ = await run_case(
        "READY", immediate_break, disconnected=True
    )
    assert text == ""
    assert penalty.failures == 0
    assert penalty.successes == 0
    assert history.rows == []


if __name__ == "__main__":
    asyncio.run(main())
    print("proxy stream error invariants: PASS")
