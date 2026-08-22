import base64
import io
import asyncio
import json
import os
import tempfile
import unittest
import sys
from pathlib import Path

from unittest import mock
from fastapi.responses import Response


_TEST_PROJECT = tempfile.TemporaryDirectory()
os.environ["PROJECT_DIR"] = _TEST_PROJECT.name
os.environ["FASTLLM_BACKEND_URL"] = "http://127.0.0.1:8002"
os.environ["FASTLLM_MODEL_SLUG"] = "qwen3.6-fastllm"
os.environ["FASTLLM_PUBLIC_ALIASES"] = (
    "qwen3.6-27b-awq,qwen3.6-27b-heretic"
)
os.environ["FASTLLM_AUTO_ROUTE_ALIASES"] = "auto"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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



class ModelRoutingTests(unittest.TestCase):
    def test_old_public_model_ids_remain_advertised(self):
        model_ids = {item["id"] for item in thinking_proxy.MODEL_LIST["data"]}
        self.assertTrue({
            "qwen3.6-27b-awq",
            "qwen3.6-27b-heretic",
        }.issubset(model_ids))

    def test_model_list_is_built_from_configured_public_aliases(self):
        model_list = thinking_proxy._build_local_model_list({
            "qwen3.8-27b",
            "qwen3.8-27b-heretic",
        })

        self.assertEqual(
            [item["id"] for item in model_list["data"]],
            ["qwen3.8-27b", "qwen3.8-27b-heretic"],
        )
        for item in model_list["data"]:
            self.assertEqual(item["context_length"], 262144)
            self.assertTrue(item["capabilities"]["vision"])
            self.assertTrue(item["supported_features"]["reasoning"])

    def test_auto_route_alias_is_advertised_and_not_local_only(self):
        advertised = {
            item["id"]: item for item in thinking_proxy.MODEL_LIST["data"]
        }

        self.assertIn("auto", advertised)
        self.assertEqual(advertised["auto"]["owned_by"], "router")
        self.assertTrue(thinking_proxy._is_auto_route_alias("auto"))
        self.assertFalse(thinking_proxy._is_local_alias("auto"))
        self.assertEqual(
            thinking_proxy._to_backend_model("auto"),
            "qwen3.6-fastllm",
        )

    def test_auto_route_images_can_use_fair_router(self):
        image_message = [{
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,AA=="},
            }],
        }]

        self.assertFalse(
            thinking_proxy._must_route_local("auto", image_message)
        )
        self.assertTrue(
            thinking_proxy._must_route_local(
                "qwen3.6-27b-heretic",
                image_message,
            )
        )

    def test_auto_public_id_uses_fair_router_then_backend_slug(self):
        self.assertFalse(thinking_proxy._is_local_alias("auto"))
        self.assertEqual(
            thinking_proxy._to_backend_model("auto"),
            "qwen3.6-fastllm",
        )

    def test_only_ids_ending_in_heretic_force_local_routing(self):
        self.assertTrue(
            thinking_proxy._is_heretic_model("qwen3.6-27b-heretic")
        )
        self.assertTrue(
            thinking_proxy._is_heretic_model("vendor/qwen3.6-27b-heretic")
        )
        self.assertFalse(
            thinking_proxy._is_heretic_model("qwen3.6-27b-heretic-preview")
        )
        self.assertEqual(
            thinking_proxy._to_backend_model("qwen3.6-27b-heretic"),
            "qwen3.6-fastllm",
        )

    def test_internal_fastllm_slug_is_not_a_heretic_public_id(self):
        self.assertFalse(
            thinking_proxy._is_heretic_model("qwen3.6-fastllm")
        )


class FairRouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_ready_owned_fastllm_is_preferred_while_slots_are_free(self):
        class EmptySlotsClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def get(self, _url):
                return type("Response", (), {"status_code": 404})()

        body = {
            "model": "qwen3.6-fastllm",
            "messages": [{"role": "user", "content": "test"}],
        }
        with (
            mock.patch.object(thinking_proxy, "FASTLLM_MODE", True),
            mock.patch.object(thinking_proxy, "FASTLLM_OWNED", True),
            mock.patch.object(thinking_proxy, "BACKEND_READY", False),
            mock.patch.object(thinking_proxy, "NIM_API_KEY", "enabled"),
            mock.patch.object(thinking_proxy, "OR_API_KEY", ""),
            mock.patch.object(thinking_proxy, "ZEN_API_KEY", ""),
            mock.patch.object(
                thinking_proxy.backend_lifecycle,
                "snapshot",
                return_value={"state": "READY"},
            ),
            mock.patch.object(
                thinking_proxy.prefix_tracker,
                "all_hits",
                return_value={},
            ),
            mock.patch.object(
                thinking_proxy.prefix_tracker,
                "slot_free",
                return_value=True,
            ),
            mock.patch.object(
                thinking_proxy.prefix_tracker,
                "eviction_cost",
                return_value=0,
            ),
            mock.patch.object(
                thinking_proxy.scheduler,
                "pending",
                return_value=0,
            ),
            mock.patch.object(thinking_proxy.scheduler, "active", 0),
            mock.patch.object(
                thinking_proxy.backend_penalty,
                "score",
                return_value=0,
            ),
            mock.patch.object(
                thinking_proxy.service_history,
                "bias",
                return_value=0,
            ),
            mock.patch.object(
                thinking_proxy,
                "_over_context",
                return_value=False,
            ),
            mock.patch.object(
                thinking_proxy.httpx,
                "AsyncClient",
                side_effect=lambda **_kwargs: EmptySlotsClient(),
            ),
        ):
            backend = await thinking_proxy._pick_backend(body)

        self.assertEqual(backend, "local")


class AnthropicToolConversionTests(unittest.TestCase):
    def test_builtin_web_search_gets_query_schema(self):
        body = {
            "model": "qwen3.6-27b-heretic",
            "messages": [{"role": "user", "content": "Search the web"}],
            "max_tokens": 128,
            "tools": [{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 3,
            }],
        }

        converted = thinking_proxy._anthropic_to_openai(body)

        self.assertEqual(converted["tools"], [{
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        },
                    },
                    "required": ["query"],
                },
            },
        }])

    def test_prepared_query_does_not_require_optional_context(self):
        body = {
            "tools": [{
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": 'Prepared queries: "FastLLM offload"',
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "additionalContext": {"type": "string"},
                        },
                    },
                },
            }],
        }

        normalized = thinking_proxy._normalize_tool_schema(body)

        self.assertEqual(
            normalized["tools"][0]["function"]["parameters"]["required"],
            ["query"],
        )
    def test_disabled_thinking_is_forwarded_to_fastllm(self):
        body = {
            "model": "qwen3.6-27b-heretic",
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "Describe this image."}],
            "max_tokens": 128,
        }

        converted = thinking_proxy._anthropic_to_openai(body)

        self.assertFalse(converted["chat_template_kwargs"]["enable_thinking"])
        self.assertNotIn("reasoning", converted)

class ReasoningControlTests(unittest.TestCase):
    def test_named_effort_is_forwarded_to_qwen_chat_template(self):
        body = {"reasoning_effort": "low"}

        thinking_proxy._normalize_openai_reasoning_controls(body)

        self.assertEqual(
            body["chat_template_kwargs"],
            {"enable_thinking": True, "reasoning_effort": "low"},
        )
        self.assertEqual(body["reasoning"]["budget"], 2048)

    def test_effort_aliases_match_qwen38_levels(self):
        for requested, expected in (
            ("minimal", "low"),
            ("high", "high"),
            ("max", "xhigh"),
            ("xhigh", "xhigh"),
        ):
            with self.subTest(requested=requested):
                body = {"reasoning_effort": requested}
                thinking_proxy._normalize_openai_reasoning_controls(body)
                self.assertEqual(
                    body["chat_template_kwargs"]["reasoning_effort"],
                    expected,
                )

    def test_disabled_thinking_does_not_forward_an_effort(self):
        body = {
            "reasoning_effort": "none",
            "chat_template_kwargs": {"reasoning_effort": "xhigh"},
        }

        thinking_proxy._normalize_openai_reasoning_controls(body)

        self.assertEqual(
            body["chat_template_kwargs"], {"enable_thinking": False}
        )
        self.assertNotIn("reasoning", body)



class ImageConversionTests(unittest.TestCase):
    def test_rgb_jpeg_data_url_preserves_original_bytes(self):
        from PIL import Image

        source = io.BytesIO()
        Image.new("RGB", (1, 1), (255, 0, 0)).save(source, format="JPEG")
        messages = [{
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": (
                        "data:image/jpeg;base64,"
                        + base64.b64encode(source.getvalue()).decode()
                    ),
                },
            }],
        }]

        original = messages[0]["content"][0]["image_url"]["url"]
        thinking_proxy._convert_images(messages)

        converted = messages[0]["content"][0]["image_url"]["url"]
        self.assertEqual(converted, original)
        payload = base64.b64decode(converted.partition(",")[2])
        self.assertTrue(payload.startswith(b"\xff\xd8\xff"))


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
        self.assertIn(b'"code": "backend_reloading"', b"".join(chunks))
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
    async def test_completed_response_uses_incremental_anthropic_sse_shape(self):
        cloud_response = {
            "id": "cloud-response",
            "model": "cloud-model",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "reasoning_content": "plan",
                    "content": "answer",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }
        request = mock.Mock()
        request.is_disconnected = mock.AsyncMock(return_value=False)

        anthropic_response = thinking_proxy._openai_to_anthropic(
            cloud_response
        )
        output = "".join([
            chunk async for chunk in thinking_proxy._completed_anthropic_stream(
                anthropic_response,
                request,
            )
        ])

        events = []
        for frame in output.strip().split("\n\n"):
            data_line = next(
                line for line in frame.splitlines() if line.startswith("data: ")
            )
            events.append(json.loads(data_line[6:]))

        message_start = events[0]
        self.assertEqual(message_start["type"], "message_start")
        self.assertEqual(message_start["message"]["content"], [])
        self.assertIsNone(message_start["message"]["stop_reason"])
        starts = [
            event["content_block"]
            for event in events
            if event["type"] == "content_block_start"
        ]
        self.assertEqual(
            starts,
            [
                {"type": "thinking", "thinking": "", "signature": ""},
                {"type": "text", "text": ""},
            ],
        )
        deltas = [
            event["delta"]
            for event in events
            if event["type"] == "content_block_delta"
        ]
        self.assertIn({"type": "thinking_delta", "thinking": "plan"}, deltas)
        self.assertIn({"type": "text_delta", "text": "answer"}, deltas)

    async def test_tool_call_deltas_produce_anthropic_tool_use_block(self):
        lines = [
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_search", "type": "function", "function": {"name": "web_search", "arguments": ""}}]}, "finish_reason": null}]}',
            "",
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"query\\":"}}]}, "finish_reason": null}]}',
            "",
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\\"weather\\"}"}}]}, "finish_reason": null}]}',
            "",
            'data: {"choices": [{"delta": {"content": ""}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 11, "completion_tokens": 4}}',
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

        self.assertIn(
            '"type": "tool_use", "id": "call_search", '
            '"name": "web_search", "input": {}',
            output,
        )
        self.assertIn(
            '"type": "input_json_delta", "partial_json": "{\\"query\\":"',
            output,
        )
        self.assertIn(
            '"type": "input_json_delta", "partial_json": "\\"weather\\"}"',
            output,
        )
        self.assertIn('"stop_reason": "tool_use"', output)
        self.assertIn('"input_tokens": 11, "output_tokens": 4', output)

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



class BackendReloadingTests(unittest.IsolatedAsyncioTestCase):
    class JsonRequest:
        def __init__(self, body, method="POST", path="/v1/chat/completions"):
            self._body = body
            self.method = method
            self.headers = {}
            self.query_params = {}
            self.url = type("URL", (), {"path": path})()

        async def json(self):
            return dict(self._body)

        async def body(self):
            return json.dumps(self._body).encode()

        async def is_disconnected(self):
            return False

    @staticmethod
    def assert_reloading_response(testcase, response):
        testcase.assertEqual(response.status_code, 503)
        payload = json.loads(response.body)
        testcase.assertEqual(payload["error"]["type"], "service_unavailable")
        testcase.assertEqual(payload["error"]["code"], "backend_reloading")
    async def test_anthropic_auto_uses_fair_route_while_local_reloads(self):
        request = self.JsonRequest({
            "model": "auto",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 8,
            "stream": False,
        }, path="/v1/messages")
        cloud_response = {
            "id": "cloud-response",
            "model": "cloud-model",
            "choices": [{
                "message": {"role": "assistant", "content": "cloud"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

        with (
            mock.patch.object(
                thinking_proxy, "_pick_backend",
                new=mock.AsyncMock(return_value="or"),
            ) as pick_backend,
            mock.patch.object(
                thinking_proxy, "_nim_chat",
                new=mock.AsyncMock(),
            ) as nim_chat,
            mock.patch.object(
                thinking_proxy, "_or_chat",
                new=mock.AsyncMock(return_value=cloud_response),
            ) as or_chat,
            mock.patch.object(thinking_proxy, "OR_API_KEY", "enabled"),
            mock.patch.object(
                thinking_proxy.scheduler, "submit",
                new=mock.AsyncMock(),
            ) as local_submit,
        ):
            response = await thinking_proxy.anthropic_messages(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            json.loads(response.body)["content"][0]["text"], "cloud"
        )
        self.assertEqual(json.loads(response.body)["model"], "auto")
        pick_backend.assert_awaited_once()
        nim_chat.assert_not_awaited()
        or_chat.assert_awaited_once()
        local_submit.assert_not_awaited()

    async def test_openai_auto_image_uses_fair_route(self):
        request = self.JsonRequest({
            "model": "auto",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.png"},
                }],
            }],
            "max_tokens": 8,
            "stream": False,
        })
        cloud_response = {
            "id": "cloud-response",
            "model": "cloud-model",
            "choices": [{
                "message": {"role": "assistant", "content": "cloud"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

        with (
            mock.patch.object(
                thinking_proxy,
                "_pick_backend",
                new=mock.AsyncMock(return_value="or"),
            ) as pick_backend,
            mock.patch.object(
                thinking_proxy,
                "_or_chat",
                new=mock.AsyncMock(return_value=cloud_response),
            ) as or_chat,
            mock.patch.object(thinking_proxy, "OR_API_KEY", "enabled"),
            mock.patch.object(
                thinking_proxy.scheduler,
                "submit",
                new=mock.AsyncMock(),
            ) as local_submit,
        ):
            response = await thinking_proxy.openai_chat(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            json.loads(response.body)["choices"][0]["message"]["content"],
            "cloud",
        )
        self.assertEqual(json.loads(response.body)["model"], "auto")
        pick_backend.assert_awaited_once()
        or_chat.assert_awaited_once()
        local_submit.assert_not_awaited()

    async def test_openai_initial_stream_protocol_error_is_http_503(self):
        request = self.JsonRequest({
            "model": "qwen3.6-27b-heretic",
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
        })
        with (
            mock.patch.object(thinking_proxy.scheduler, "acquire_stream"),
            mock.patch.object(
                thinking_proxy.scheduler, "release_stream") as release_stream,
            mock.patch.object(thinking_proxy.backend_penalty, "record_failure"),
            mock.patch.object(thinking_proxy.service_history, "record"),
            mock.patch.object(
                thinking_proxy, "prepare_fastllm_body", side_effect=lambda b, _: b),
            mock.patch.object(
                thinking_proxy, "_open_backend_stream",
                side_effect=thinking_proxy.httpx.RemoteProtocolError("empty reply")),
        ):
            response = await thinking_proxy.openai_chat(request)
        self.assert_reloading_response(self, response)
        release_stream.assert_called_once()

    async def test_openai_stream_preparation_failure_releases_slot(self):
        request = self.JsonRequest({
            "model": "qwen3.6-27b-heretic",
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
        })
        with (
            mock.patch.object(thinking_proxy.scheduler, "acquire_stream"),
            mock.patch.object(
                thinking_proxy.scheduler, "release_stream") as release_stream,
            mock.patch.object(
                thinking_proxy, "prepare_fastllm_body",
                side_effect=ValueError("bad rendered prompt")),
            self.assertRaisesRegex(ValueError, "bad rendered prompt"),
        ):
            await thinking_proxy.openai_chat(request)
        release_stream.assert_called_once()

    async def test_openai_nonstream_protocol_error_is_http_503(self):
        request = self.JsonRequest({
            "model": "qwen3.6-27b-heretic",
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
        })
        with mock.patch.object(
                thinking_proxy.scheduler, "submit",
                side_effect=thinking_proxy.httpx.RemoteProtocolError("empty reply")):
            response = await thinking_proxy.openai_chat(request)
        self.assert_reloading_response(self, response)

    async def test_count_tokens_protocol_error_is_http_503(self):
        request = self.JsonRequest({
            "model": "qwen3.6-27b-heretic",
            "messages": [{"role": "user", "content": "test"}],
        }, path="/v1/messages/count_tokens")

        class FailingClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, *args, **kwargs):
                raise thinking_proxy.httpx.RemoteProtocolError("empty reply")

        with (
            mock.patch.object(
                thinking_proxy, "prepare_fastllm_body", side_effect=lambda b, _: b),
            mock.patch.object(
                thinking_proxy.httpx, "AsyncClient", return_value=FailingClient()),
        ):
            response = await thinking_proxy.anthropic_count_tokens(request)
        self.assert_reloading_response(self, response)

    async def test_passthrough_protocol_error_is_http_503(self):
        request = self.JsonRequest({}, method="GET", path="/props")

        class FailingClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def request(self, *args, **kwargs):
                raise thinking_proxy.httpx.RemoteProtocolError("empty reply")

        with mock.patch.object(
                thinking_proxy.httpx, "AsyncClient", return_value=FailingClient()):
            response = await thinking_proxy.passthrough("props", request)
        self.assert_reloading_response(self, response)

    async def test_midstream_error_is_structured_sse_and_done(self):
        async def failing_local_stream(url, body):
            yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            raise thinking_proxy.httpx.RemoteProtocolError("truncated body")

        request = self.JsonRequest({})
        body = {"model": "qwen3.6-fastllm", "messages": [], "stream": True}
        with (
            mock.patch.object(
                thinking_proxy, "prepare_fastllm_body", side_effect=lambda b, _: b),
            mock.patch.object(thinking_proxy, "_proxy_stream", failing_local_stream),
            mock.patch.object(thinking_proxy.prefix_tracker, "reserve"),
            mock.patch.object(thinking_proxy.prefix_tracker, "record"),
            mock.patch.object(thinking_proxy.scheduler, "release_stream"),
            mock.patch.object(thinking_proxy.backend_penalty, "record_failure"),
            mock.patch.object(thinking_proxy.service_history, "record"),
        ):
            chunks = [chunk async for chunk in thinking_proxy._local_stream(
                "http://127.0.0.1:8002/v1/chat/completions", body, request,
                allow_fallback=False)]
        output = b"".join(chunks)
        self.assertIn(b'"type": "service_unavailable"', output)
        self.assertIn(b'"code": "backend_reloading"', output)
        self.assertTrue(output.endswith(b"data: [DONE]\n\n"))

class StreamFallbackBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_fallback_advances_only_before_first_byte(self):
        calls = []

        async def nim(body):
            calls.append("nim")
            if False:
                yield b""
            raise RuntimeError("nim unavailable")

        async def openrouter(body):
            calls.append("or")
            yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'

        async def zen(body):
            calls.append("zen")
            yield b"should not run"

        with (
            mock.patch.object(thinking_proxy, "_nim_stream", nim),
            mock.patch.object(thinking_proxy, "_or_stream", openrouter),
            mock.patch.object(thinking_proxy, "_zen_stream", zen),
            mock.patch.object(thinking_proxy, "OR_API_KEY", "enabled"),
            mock.patch.object(thinking_proxy, "ZEN_API_KEY", "enabled"),
        ):
            output = b"".join([
                chunk async for chunk in thinking_proxy._fallback_openai_stream({})])
        self.assertEqual(calls, ["nim", "or"])
        self.assertIn(b'"content":"ok"', output)
        self.assertTrue(output.endswith(b"data: [DONE]\n\n"))

    async def test_anthropic_partial_local_stream_never_starts_cloud_message(self):
        cloud_calls = 0

        async def partial_local(body, public_model=""):
            yield thinking_proxy._sse("message_start", {
                "type": "message_start", "message": {"id": "partial"}})
            raise thinking_proxy.httpx.RemoteProtocolError("truncated body")

        async def cloud(body):
            nonlocal cloud_calls
            cloud_calls += 1
            return {"choices": [{"message": {"content": "cloud"}}]}

        request = BackendReloadingTests.JsonRequest({})
        body = {"model": "qwen3.6-fastllm", "messages": [], "stream": True}
        with (
            mock.patch.object(thinking_proxy, "_anthropic_stream", partial_local),
            mock.patch.object(
                thinking_proxy, "_pick_backend",
                new=mock.AsyncMock(return_value="local"),
            ),
            mock.patch.object(thinking_proxy, "_nim_chat", cloud),
            mock.patch.object(thinking_proxy.scheduler, "acquire_stream"),
            mock.patch.object(thinking_proxy.scheduler, "release_stream"),
            mock.patch.object(thinking_proxy.prefix_tracker, "reserve"),
            mock.patch.object(thinking_proxy.prefix_tracker, "record"),
            mock.patch.object(thinking_proxy, "FALLBACK_ENABLED", True),
            mock.patch.object(thinking_proxy, "BENCHMARK_MODE", False),
        ):
            output = "".join([
                chunk async for chunk in thinking_proxy._anthropic_stream_limited(
                    body, request, "auto")])
        self.assertEqual(cloud_calls, 0)
        self.assertEqual(output.count("event: message_start"), 1)
        self.assertIn('"code": "backend_reloading"', output)
    async def test_anthropic_auto_stream_uses_external_token_stream(self):
        request = BackendReloadingTests.JsonRequest({})
        body = {
            "model": "qwen3.6-fastllm",
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
        }
        stream_calls = 0

        async def openrouter_stream(_body):
            nonlocal stream_calls
            stream_calls += 1
            yield (
                b'data: {"model":"cloud-model","choices":[{"delta":'
                b'{"role":"assistant","content":"cl"},"finish_reason":null}]}\n\n'
            )
            yield (
                b'data: {"model":"cloud-model","choices":[{"delta":'
                b'{"content":"oud"},"finish_reason":"stop"}],'
                b'"usage":{"prompt_tokens":1,"completion_tokens":2}}\n\n'
            )
            yield b"data: [DONE]\n\n"

        with (
            mock.patch.object(
                thinking_proxy,
                "_pick_backend",
                new=mock.AsyncMock(return_value="or"),
            ) as pick_backend,
            mock.patch.object(
                thinking_proxy,
                "_or_stream",
                new=openrouter_stream,
            ),
            mock.patch.object(
                thinking_proxy,
                "_or_chat",
                new=mock.AsyncMock(
                    side_effect=AssertionError("buffered chat path used")
                ),
            ) as or_chat,
            mock.patch.object(
                thinking_proxy.scheduler,
                "acquire_stream",
                new=mock.AsyncMock(),
            ) as acquire_local,
            mock.patch.object(thinking_proxy, "OR_API_KEY", "enabled"),
            mock.patch.object(thinking_proxy, "FALLBACK_ENABLED", True),
            mock.patch.object(thinking_proxy, "BENCHMARK_MODE", False),
        ):
            output = "".join([
                chunk async for chunk in thinking_proxy._anthropic_stream_limited(
                    body, request, "auto"
                )
            ])

        self.assertEqual(stream_calls, 1)
        self.assertIn('"text": "cl"', output)
        self.assertIn('"text": "oud"', output)
        self.assertIn("event: message_stop", output)
        pick_backend.assert_awaited_once()
        or_chat.assert_not_awaited()
        acquire_local.assert_not_awaited()

if __name__ == "__main__":
    unittest.main()
