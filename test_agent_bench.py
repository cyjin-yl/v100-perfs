import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parent / "benchmarks" / "agent_bench.py"
SPEC = importlib.util.spec_from_file_location("agent_bench_under_test", MODULE_PATH)
agent_bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = agent_bench
SPEC.loader.exec_module(agent_bench)


class SseEventParsingTest(unittest.TestCase):
    def test_joins_multiline_data_fields_at_event_boundary(self):
        lines = [
            b"data: {\r\n",
            b'data:   "choices": [\r\n',
            b'data:     {"delta": {"content": "OK"}}\r\n',
            b"data:   ]\r\n",
            b"data: }\r\n",
            b"\r\n",
            b'data: {"choices": [], "usage": {"prompt_tokens": 7, '
            b'"completion_tokens": 2}}\n',
            b"\n",
            b"data: [DONE]\n",
            b"\n",
        ]

        events = list(agent_bench.iter_sse_data(lines))

        self.assertEqual(len(events), 3)
        self.assertEqual(events[0]["choices"][0]["delta"]["content"], "OK")
        self.assertEqual(events[1]["usage"]["completion_tokens"], 2)
        self.assertIsNone(events[2])

    def test_reasoning_content_marks_first_generated_token(self):
        clock = [0.0]

        class FakeResponse:
            def __iter__(self):
                events = [
                    (1.0, b'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\n'),
                    (1.0, b"\n"),
                    (3.0, b'data: {"choices":[{"delta":{"content":"answer"}}]}\n'),
                    (3.0, b"\n"),
                    (5.0, b'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2}}\n'),
                    (5.0, b"\n"),
                    (5.0, b"data: [DONE]\n"),
                    (5.0, b"\n"),
                ]
                for timestamp, line in events:
                    clock[0] = timestamp
                    yield line

        original_now = agent_bench.now
        original_urlopen = agent_bench.urllib.request.urlopen
        agent_bench.now = lambda: clock[0]
        agent_bench.urllib.request.urlopen = lambda request, timeout: FakeResponse()
        try:
            result = agent_bench.call_chat(
                "http://127.0.0.1:1",
                "test-model",
                [{"role": "user", "content": "test"}],
            )
        finally:
            agent_bench.now = original_now
            agent_bench.urllib.request.urlopen = original_urlopen

        self.assertEqual(result["content"], "answer")
        self.assertEqual(result["ttft"], 1.0)
        self.assertEqual(result["decode_time"], 4.0)


class RunAgentsFailureTest(unittest.TestCase):
    def test_worker_failure_is_raised_to_caller(self):
        original = agent_bench.run_one_turn

        def fail_worker(*args, **kwargs):
            raise RuntimeError("request failed")

        agent_bench.run_one_turn = fail_worker
        try:
            with self.assertRaisesRegex(RuntimeError, "request failed"):
                agent_bench.run_agents(
                    endpoint="http://127.0.0.1:1",
                    model="test-model",
                    num_agents=2,
                    max_model_len=1024,
                    shared_prefix_tokens=1,
                    turns_per_agent=1,
                    max_tokens=1,
                )
        finally:
            agent_bench.run_one_turn = original


if __name__ == "__main__":
    unittest.main()
