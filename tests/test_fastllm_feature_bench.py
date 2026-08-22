from __future__ import annotations

import importlib.util
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parent / "benchmarks" / "fastllm" / "feature_bench.py"
)
SPEC = importlib.util.spec_from_file_location("fastllm_feature_bench", MODULE_PATH)
feature_bench = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(feature_bench)


class SseAccumulatorTest(unittest.TestCase):
    def test_multiline_events_survive_transport_chunk_boundaries(self):
        accumulator = feature_bench.SSEAccumulator()
        chunks = [
            b": keepalive\r\ndata: {\"choices\":[{\"delta\":",
            b"{\"content\":\"hello\"}}]}\r\n\r\n",
            b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\n",
            b"data: \"usage\":{\"prompt_tokens\":4,\"completion_tokens\":1,\"total_tokens\":5}}\n\n",
            b"data: [DONE]\n\n",
        ]
        for index, chunk in enumerate(chunks):
            accumulator.feed(chunk, timestamp=10.0 + index)
        accumulator.finish(timestamp=20.0)

        result = accumulator.result()
        self.assertEqual(result["content"], "hello")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["usage"]["total_tokens"], 5)
        self.assertEqual(result["done_events"], 1)
        self.assertEqual(result["malformed_events"], 0)
        self.assertEqual(result["first_token_time"], 11.0)

    def test_invalid_utf8_event_is_counted_not_raised(self):
        accumulator = feature_bench.SSEAccumulator()
        accumulator.feed(b"data: {\"choices\":\xff}\n\ndata: [DONE]\n\n", 1.0)
        accumulator.finish(2.0)
        result = accumulator.result()
        self.assertEqual(result["malformed_events"], 1)
        self.assertEqual(result["done_events"], 1)


class ValidationTest(unittest.TestCase):
    def test_result_validation_and_reference_comparison_are_strict(self):
        good = {
            "request_id": "a",
            "iteration": 0,
            "status": 200,
            "done_events": 1,
            "malformed_events": 0,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 1,
                "total_tokens": 5,
            },
            "content_sha256": "abc",
        }
        self.assertEqual(feature_bench.validate_request_result(good), [])

        broken = dict(good)
        broken["done_events"] = 0
        broken["usage"] = dict(good["usage"], total_tokens=6)
        errors = feature_bench.validate_request_result(broken)
        self.assertTrue(any("DONE" in error for error in errors))
        self.assertTrue(any("total_tokens" in error for error in errors))

        reference = {"requests": [good]}
        changed = dict(good, content_sha256="def")
        comparison = feature_bench.compare_with_reference(
            {"requests": [changed]}, reference
        )
        self.assertFalse(comparison["match"])
        self.assertTrue(any("content_sha256" in error for error in comparison["errors"]))


class _FixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format, *_args):
        pass

    def do_GET(self):
        if self.path == "/props":
            body = json.dumps(
                {
                    "model": "fixture",
                    "cpu_request_swap_disk_restores": 3,
                    "prefix_cache_disk_hits": 5,
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        payload = (
            b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            b'"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}\n\n'
            b"data: [DONE]\n\n"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class RequestFixtureTest(unittest.TestCase):
    def test_repeat_fixture_expands_compact_raw_prompt(self):
        expanded = feature_bench.expand_request_spec(
            {
                "id": "long-a",
                "body": {"model": "fixture", "stream": True},
                "repeat": {
                    "field": "prompt",
                    "prefix": "pre",
                    "unit": " x",
                    "count": 3,
                    "suffix": "post",
                },
            },
            "fixture.json",
        )
        self.assertEqual(expanded["body"]["prompt"], "pre x x xpost")
        self.assertNotIn("repeat", expanded)

    def test_repeat_fixture_rejects_negative_count(self):
        with self.assertRaisesRegex(ValueError, "count"):
            feature_bench.expand_request_spec(
                {
                    "id": "bad",
                    "body": {},
                    "repeat": {
                        "field": "prompt",
                        "unit": "x",
                        "count": -1,
                    },
                },
                "bad.json",
            )


class PropsValidationTest(unittest.TestCase):
    def test_required_prop_deltas_are_numeric_and_enforced(self):
        self.assertEqual(
            feature_bench.validate_prop_deltas(
                {"restores": 2, "hits": 4},
                {"restores": 3, "hits": 6},
                {"restores": 1, "hits": 2},
            ),
            [],
        )
        errors = feature_bench.validate_prop_deltas(
            {"restores": 2},
            {"restores": 2},
            {"restores": 1, "missing": 1},
        )
        self.assertTrue(any("restores" in error for error in errors))
        self.assertTrue(any("missing" in error for error in errors))
        upper_errors = feature_bench.validate_prop_deltas(
            {"spills": 7},
            {"spills": 8},
            {},
            {"spills": 0},
        )
        self.assertEqual(len(upper_errors), 1)
        self.assertIn("above 0", upper_errors[0])


class HttpBenchmarkTest(unittest.TestCase):
    def test_http_run_captures_props_hash_usage_and_protocol(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _FixtureHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = feature_bench.run_http_benchmark(
                endpoint=f"http://127.0.0.1:{server.server_port}",
                requests=[
                    {
                        "id": "fixture",
                        "body": {
                            "model": "fixture",
                            "messages": [{"role": "user", "content": "x"}],
                            "stream": True,
                        },
                    }
                ],
                runs=1,
                concurrency=1,
                timeout=5.0,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["props_before"]["model"], "fixture")
        self.assertEqual(result["props_after"]["prefix_cache_disk_hits"], 5)
        self.assertEqual(result["requests"][0]["content"], "ok")
        self.assertEqual(result["requests"][0]["usage"]["total_tokens"], 5)
        self.assertEqual(result["validation"]["errors"], [])


class CommandBenchmarkTest(unittest.TestCase):
    def test_command_ab_interleaves_variants_and_parses_metrics(self):
        config = {
            "name": "fixture_operator",
            "command": [
                str(Path(feature_bench.sys.executable)),
                "-c",
                (
                    "import os; "
                    "print('latency_ms=' + os.environ['BENCH_LATENCY'] + "
                    "' output_hash=same')"
                ),
            ],
            "variants": [
                {
                    "name": "baseline",
                    "env": {"BENCH_LATENCY": "4.0"},
                },
                {
                    "name": "optimized",
                    "env": {"BENCH_LATENCY": "2.0"},
                },
            ],
            "warmups": 1,
            "runs": 2,
            "metrics": {
                "latency_ms": {
                    "regex": r"latency_ms=([0-9.]+)",
                    "unit": "ms",
                    "lower_is_better": True,
                }
            },
            "correctness": {
                "output_hash": r"output_hash=([a-z]+)"
            },
        }
        result = feature_bench.run_command_benchmark(config)
        self.assertEqual(
            [run["variant"] for run in result["runs"]],
            ["baseline", "optimized", "baseline", "optimized"],
        )
        self.assertEqual(
            result["summary"]["metrics"]["latency_ms"]["optimized"][
                "speedup_vs_baseline"
            ],
            2.0,
        )
        self.assertEqual(result["validation"]["errors"], [])

    def test_command_config_rejects_secret_bearing_environment_keys(self):
        config = {
            "name": "unsafe",
            "command": [str(Path(feature_bench.sys.executable)), "-V"],
            "variants": [
                {
                    "name": "baseline",
                    "env": {"API_TOKEN": "must-not-be-recorded"},
                }
            ],
            "runs": 1,
            "metrics": {},
        }
        with self.assertRaisesRegex(ValueError, "sensitive"):
            feature_bench.run_command_benchmark(config)

    def test_process_metadata_uses_stdout_when_stderr_is_empty(self):
        self.assertEqual(
            feature_bench.subprocess_error_text(
                "driver/library mismatch\n", ""
            ),
            "driver/library mismatch",
        )


if __name__ == "__main__":
    unittest.main()
