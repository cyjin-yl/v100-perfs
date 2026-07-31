import asyncio
import json
import os
import tempfile
import unittest
from unittest import mock

from fastapi.responses import Response


_TEST_PROJECT = tempfile.TemporaryDirectory()
os.environ["PROJECT_DIR"] = _TEST_PROJECT.name
os.environ["FASTLLM_BACKEND_URL"] = "http://127.0.0.1:8002"
os.environ["FASTLLM_MODEL_SLUG"] = "qwen3.6-fastllm"
os.environ["FASTLLM_PUBLIC_ALIASES"] = (
    "qwen3.6-27b-awq,qwen3.6-27b-heretic"
)

import thinking_proxy


def tearDownModule():
    thinking_proxy.service_history._conn.close()
    _TEST_PROJECT.cleanup()


class ModelRewriteTests(unittest.TestCase):
    def test_nonstream_response_restores_requested_public_model(self):
        response = Response(
            content=json.dumps({
                "model": "qwen3.6-fastllm",
                "choices": [{"message": {"content": "OK"}}],
            }),
            status_code=200,
            media_type="application/json",
        )

        rewritten = thinking_proxy._rewrite_local_response(
            response, "qwen3.6-27b-awq"
        )

        self.assertEqual(
            json.loads(rewritten.body)["model"], "qwen3.6-27b-awq"
        )

    def test_stream_event_restores_public_model_with_json_whitespace(self):
        chunk = (
            b'data: {"model": "qwen3.6-fastllm",'
            b'"choices":[]}\r\n\r\n'
        )

        rewritten = thinking_proxy._rewrite_stream_model(
            chunk, "qwen3.6-27b-heretic"
        )

        payload = rewritten.split(b"data: ", 1)[1].strip()
        self.assertEqual(
            json.loads(payload)["model"], "qwen3.6-27b-heretic"
        )

    def test_stream_event_restores_public_model_in_multiline_json(self):
        event = (
            b"data: {\r\n"
            b'data: \t"model": "qwen3.6-fastllm",\r\n'
            b'data: \t"choices": []\r\n'
            b"data: }\r\n\r\n"
        )

        rewritten = thinking_proxy._rewrite_stream_model(
            event, "qwen3.6-27b-awq"
        )

        payload = b"\n".join(
            line[5:].lstrip()
            for line in rewritten.splitlines()
            if line.startswith(b"data:")
        )
        self.assertEqual(
            json.loads(payload)["model"], "qwen3.6-27b-awq"
        )



class LocalOnlyStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_fastllm_failure_never_calls_cloud_fallback(self):
        cloud_calls = 0

        async def failing_local_stream(url, body):
            if False:
                yield b""
            raise RuntimeError("local backend failed")

        async def cloud_stream(body):
            nonlocal cloud_calls
            cloud_calls += 1
            yield b'data: {"model":"cloud"}\n\n'

        class ConnectedRequest:
            async def is_disconnected(self):
                return False

        body = {
            "model": "qwen3.6-fastllm",
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
        }

        with (
            mock.patch.object(
                thinking_proxy, "prepare_fastllm_body", side_effect=lambda b, _: b
            ),
            mock.patch.object(thinking_proxy, "_proxy_stream", failing_local_stream),
            mock.patch.object(thinking_proxy, "_nim_stream", cloud_stream),
            mock.patch.object(thinking_proxy.prefix_tracker, "reserve"),
            mock.patch.object(thinking_proxy.prefix_tracker, "record"),
            mock.patch.object(thinking_proxy.scheduler, "release_stream"),
            mock.patch.object(thinking_proxy.backend_penalty, "record_failure"),
            mock.patch.object(thinking_proxy.service_history, "record"),
        ):
            chunks = [
                chunk
                async for chunk in thinking_proxy._local_stream(
                    "http://127.0.0.1:8002/v1/chat/completions",
                    body,
                    ConnectedRequest(),
                    "qwen3.6-27b-heretic",
                    allow_fallback=False,
                )
            ]

        self.assertEqual(cloud_calls, 0)
        self.assertIn(b'"finish_reason": "error"', b"".join(chunks))
        self.assertTrue(b"".join(chunks).endswith(b"data: [DONE]\n\n"))

    async def test_fastllm_done_stops_consuming_upstream_immediately(self):
        consumed_after_done = False

        async def stream_with_late_bytes(url, body):
            nonlocal consumed_after_done
            yield (
                b'data: {"model":"qwen3.6-fastllm",'
                b'"choices":[{"delta":{"content":"OK"}}]}\n\n'
                b'data: [DONE]\n\n'
            )
            consumed_after_done = True
            yield b"late bytes that must not be consumed"

        class ConnectedRequest:
            async def is_disconnected(self):
                return False

        body = {
            "model": "qwen3.6-fastllm",
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
        }
        with (
            mock.patch.object(
                thinking_proxy, "prepare_fastllm_body", side_effect=lambda b, _: b
            ),
            mock.patch.object(thinking_proxy, "_proxy_stream", stream_with_late_bytes),
            mock.patch.object(thinking_proxy.prefix_tracker, "reserve"),
            mock.patch.object(thinking_proxy.prefix_tracker, "record"),
            mock.patch.object(thinking_proxy.scheduler, "release_stream"),
            mock.patch.object(thinking_proxy.backend_penalty, "record_success"),
            mock.patch.object(thinking_proxy.service_history, "record"),
        ):
            chunks = [
                chunk
                async for chunk in thinking_proxy._local_stream(
                    "http://127.0.0.1:8002/v1/chat/completions",
                    body,
                    ConnectedRequest(),
                    "qwen3.6-27b-heretic",
                    allow_fallback=False,
                )
            ]

        self.assertFalse(consumed_after_done)
        self.assertTrue(b"".join(chunks).endswith(b"data: [DONE]\n\n"))


class AnthropicStreamTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _client_for(lines, status_code=200):
        class FakeResponse:
            def __init__(self):
                self.status_code = status_code

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aiter_lines(self):
                for line in lines:
                    yield line

            async def aread(self):
                return b"upstream error"

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def stream(self, method, url, json):
                return FakeResponse()

        return FakeClient()

    async def test_multiline_fastllm_events_produce_anthropic_stream(self):
        lines = [
            "data: {",
            'data: "choices": [{"delta": {"reasoning_content": "plan"}, "finish_reason": null}],',
            'data: "usage": {"prompt_tokens": 7, "completion_tokens": 1}',
            "data: }",
            "",
            "data: {",
            'data: "choices": [{"delta": {"content": "answer"}, "finish_reason": null}]',
            "data: }",
            "",
            "data: {",
            'data: "choices": [{"delta": {"content": ""}, "finish_reason": "length"}],',
            'data: "usage": {"prompt_tokens": 7, "completion_tokens": 3}',
            "data: }",
            "",
            "data: [DONE]",
            "",
        ]
        client = self._client_for(lines)
        body = {"model": "qwen3.6-fastllm", "messages": [], "stream": True}
        with (
            mock.patch.object(
                thinking_proxy, "prepare_fastllm_body", side_effect=lambda b, _: b
            ),
            mock.patch.object(
                thinking_proxy.httpx, "AsyncClient", return_value=client
            ),
        ):
            output = "".join([
                chunk async for chunk in thinking_proxy._anthropic_stream(body)
            ])

        self.assertIn('"type": "thinking_delta", "thinking": "plan"', output)
        self.assertIn('"type": "text_delta", "text": "answer"', output)
        self.assertIn('"stop_reason": "max_tokens"', output)
        self.assertIn('"input_tokens": 7, "output_tokens": 3', output)
        self.assertTrue(output.endswith("event: message_stop\n" +
                                        'data: {"type": "message_stop"}\n\n'))

    async def test_done_without_terminal_event_is_an_error(self):
        lines = [
            'data: {"choices": [{"delta": {"role": "assistant"}, "finish_reason": null}]}',
            "",
            "data: [DONE]",
            "",
        ]
        client = self._client_for(lines)
        body = {"model": "qwen3.6-fastllm", "messages": [], "stream": True}
        with (
            mock.patch.object(
                thinking_proxy, "prepare_fastllm_body", side_effect=lambda b, _: b
            ),
            mock.patch.object(
                thinking_proxy.httpx, "AsyncClient", return_value=client
            ),
            self.assertRaisesRegex(RuntimeError, "without a terminal event"),
        ):
            _ = [chunk async for chunk in thinking_proxy._anthropic_stream(body)]


if __name__ == "__main__":
    unittest.main()
