import json
import os
import unittest
from pathlib import Path
from unittest import mock

import fastllm_adapter


TEMPLATE = Path(__file__).parent / "chat_templates" / "qwen3.6_merged.jinja"


class FastLLMAdapterTests(unittest.TestCase):
    def test_prepare_body_renders_raw_qwen_prompt_with_tools(self):
        body = {
            "model": "qwen3.6-fastllm",
            "messages": [
                {"role": "system", "content": "Use tools when needed."},
                {"role": "user", "content": "What is the weather in Paris?"},
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Read weather by city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }],
            "max_tokens": 64,
            "temperature": 0.0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        actual = fastllm_adapter.prepare_fastllm_body(body, TEMPLATE)

        self.assertIsNot(actual, body)
        self.assertTrue(actual["raw_prompt"])
        self.assertEqual(actual["prompt"].count("<|im_start|>assistant"), 1)
        self.assertIn("get_weather(city: string)", actual["prompt"])
        self.assertIn("What is the weather in Paris?", actual["prompt"])
        self.assertTrue(actual["prompt"].endswith("<think>\n\n</think>\n\n"))
        self.assertEqual(actual["messages"], body["messages"])
        self.assertEqual(actual["max_tokens"], 64)

    def test_prepare_body_adds_qwen_end_marker_stop(self):
        body = {
            "model": "qwen3.6-fastllm",
            "messages": [{"role": "user", "content": "Say ready."}],
        }

        actual = fastllm_adapter.prepare_fastllm_body(body, TEMPLATE)

        self.assertEqual(actual["stop"], ["<|im_end|>"])

    def test_prepare_body_renders_plain_assistant_history_without_tool_calls(self):
        body = {
            "model": "qwen3.6-fastllm",
            "messages": [
                {"role": "user", "content": "First question."},
                {"role": "assistant", "content": "First answer."},
                {"role": "user", "content": "Follow up."},
            ],
            "stream": True,
        }

        actual = fastllm_adapter.prepare_fastllm_body(body, TEMPLATE)

        self.assertIn("First answer.", actual["prompt"])
        self.assertIn("Follow up.", actual["prompt"])
        self.assertEqual(actual["messages"], body["messages"])

    def test_prepare_body_renders_historical_tool_call_and_result(self):
        body = {
            "model": "qwen3.6-fastllm",
            "messages": [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "Paris"}),
                        },
                    }],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "18 C"},
            ],
            "max_tokens": 32,
        }

        actual = fastllm_adapter.prepare_fastllm_body(body, TEMPLATE)

        self.assertIn("<function=get_weather>", actual["prompt"])
        self.assertIn("<parameter=city>\nParis\n</parameter>", actual["prompt"])
        self.assertIn("<tool_response>\n18 C\n</tool_response>", actual["prompt"])

    def test_adapt_response_promotes_multiple_xml_tool_calls(self):
        response = {
            "id": "fastllm-1",
            "model": "qwen3.6-fastllm",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": (
                        "checking\n</think>\n\n"
                        "<tool_call>\n<function=get_weather>\n"
                        "<parameter=city>\nParis\n</parameter>\n"
                        "</function>\n</tool_call>\n"
                        "<tool_call>\n<function=convert_temperature>\n"
                        "<parameter=value>\n18\n</parameter>\n"
                        "<parameter=unit>\nC\n</parameter>\n"
                        "</function>\n</tool_call>"
                    ),
                },
                "finish_reason": None,
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 12},
        }

        actual = fastllm_adapter.adapt_fastllm_response(response)
        message = actual["choices"][0]["message"]

        self.assertEqual(message["reasoning_content"], "checking")
        self.assertEqual(message["content"], "")
        self.assertEqual(actual["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual([c["function"]["name"] for c in message["tool_calls"]],
                         ["get_weather", "convert_temperature"])
        self.assertEqual(json.loads(message["tool_calls"][0]["function"]["arguments"]),
                         {"city": "Paris"})
        self.assertEqual(json.loads(message["tool_calls"][1]["function"]["arguments"]),
                         {"value": 18, "unit": "C"})

    def test_parse_tolerates_qwen_function_typo(self):
        content = (
            "<tool_call>\n<fuction=get_weather>\n"
            "<parameter=city>\n北京\n</parameter>\n"
            "</function>\n</tool_call>"
        )

        calls = fastllm_adapter.parse_fastllm_tool_calls(content)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"city": "北京"})
    def test_adapt_response_keeps_normal_answer_and_extracts_reasoning(self):
        response = {
            "choices": [{
                "message": {"role": "assistant", "content": "work\n</think>\n\n42"},
                "finish_reason": None,
            }],
        }

        actual = fastllm_adapter.adapt_fastllm_response(response)

        self.assertEqual(actual["choices"][0]["message"]["reasoning_content"], "work")
        self.assertEqual(actual["choices"][0]["message"]["content"], "42")
        self.assertEqual(actual["choices"][0]["finish_reason"], "stop")
    def test_adapt_response_strips_generated_qwen_tail(self):
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "</think>\nREADY<|im_end|>\n<|im_start|>user",
                },
                "finish_reason": None,
            }],
        }

        actual = fastllm_adapter.adapt_fastllm_response(response)

        self.assertEqual(actual["choices"][0]["message"]["content"], "READY")


if __name__ == "__main__":
    unittest.main()
