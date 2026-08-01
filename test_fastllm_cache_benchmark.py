from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parent
    / "benchmarks"
    / "fastllm"
    / "run_cache_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location(
    "fastllm_cache_benchmark", MODULE_PATH
)
cache_benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(cache_benchmark)


class ScenarioTest(unittest.TestCase):
    def test_resident_and_l3_scenarios_are_capacity_specific(self):
        resident = cache_benchmark.build_scenario("resident")
        l3 = cache_benchmark.build_scenario("l3")
        self.assertEqual(resident["concurrency"], 2)
        self.assertEqual(len(resident["request_files"]), 2)
        self.assertEqual(resident["maximum_prop_deltas"], {
            "cpu_request_swap_disk_spills": 0.0,
        })
        self.assertEqual(l3["concurrency"], 3)
        self.assertEqual(len(l3["request_files"]), 3)
        self.assertIn(
            "paged_cache_snapshot_zstd_decompress_calls",
            l3["required_prop_deltas"],
        )
        prefix = cache_benchmark.build_scenario("prefix")
        self.assertEqual(prefix["concurrency"], 1)
        self.assertEqual(len(prefix["request_files"]), 4)
        self.assertIn(
            "prefix_cache_disk_hits",
            prefix["required_prop_deltas"],
        )
        self.assertIn(
            "prefix_cache_gpu_hit_pages",
            prefix["required_prop_deltas"],
        )
        self.assertIn(
            "prefix_cache_zstd_compress_calls",
            prefix["required_prop_deltas"],
        )
        self.assertIn(
            "prefix_cache_zstd_decompress_calls",
            prefix["required_prop_deltas"],
        )

    def test_profile_validation_rejects_wrong_pool_or_dtype(self):
        props = {
            "backend": "fastllm",
            "kv_cache_dtype": "fp8_e4m3",
            "token_pool": 262144,
            "cpu_request_swap_enabled": True,
            "cpu_request_swap_zstd_enabled": True,
            "cpu_request_swap_disk_enabled": True,
        }
        errors = cache_benchmark.validate_profile(
            props, "l3", expected_token_pool=32768
        )
        self.assertTrue(any("kv_cache_dtype" in error for error in errors))
        self.assertTrue(any("token_pool" in error for error in errors))

    def test_prefix_profile_requires_runtime_cost_metrics(self):
        props = {
            "backend": "fastllm",
            "kv_cache_dtype": "turbo3",
            "token_pool": 16384,
            "prefix_cache_cpu_tier_enabled": True,
            "prefix_cache_disk_tier_enabled": True,
            "prefix_cache_recompute_tokens_per_second": 0.0,
            "prefix_cache_disk_read_mib_per_second": 0.0,
            "prefix_cache_zstd_compress_calls": 0.0,
            "prefix_cache_zstd_compress_input_bytes": 0.0,
            "prefix_cache_zstd_compress_output_bytes": 0.0,
            "prefix_cache_zstd_compress_seconds": 0.0,
            "prefix_cache_zstd_decompress_calls": 0.0,
            "prefix_cache_zstd_decompress_input_bytes": 0.0,
            "prefix_cache_zstd_decompress_output_bytes": 0.0,
            "prefix_cache_zstd_decompress_seconds": 0.0,
        }
        self.assertEqual(
            cache_benchmark.validate_profile(
                props, "prefix", expected_token_pool=16384
            ),
            [],
        )
        props["prefix_cache_disk_tier_enabled"] = False
        errors = cache_benchmark.validate_profile(
            props, "prefix", expected_token_pool=16384
        )
        self.assertTrue(
            any("prefix_cache_disk_tier_enabled" in error for error in errors)
        )

    def test_prefix_scenario_checks_partial_hit_restore_and_output(self):
        common = {
            "content_sha256": "content",
            "reasoning_sha256": "reason",
            "finish_reason": "length",
            "usage": {
                "prompt_tokens": 12000,
                "completion_tokens": 32,
                "total_tokens": 12032,
            },
        }
        result = {
            "requests": [
                {"request_id": "prefix_seed_a", "ttft_seconds": 20.0, **common},
                {
                    "request_id": "prefix_shared_variant",
                    "ttft_seconds": 1.0,
                    **{**common, "content_sha256": "variant"},
                },
                {
                    "request_id": "prefix_pressure_b",
                    "ttft_seconds": 20.0,
                    **{**common, "content_sha256": "pressure"},
                },
                {
                    "request_id": "prefix_restore_a",
                    "ttft_seconds": 4.0,
                    **common,
                },
            ]
        }
        self.assertEqual(
            cache_benchmark.validate_prefix_scenario(result),
            [],
        )
        result["requests"][-1]["content_sha256"] = "wrong"
        errors = cache_benchmark.validate_prefix_scenario(result)
        self.assertTrue(any("content_sha256" in error for error in errors))

    def test_repeat_consistency_compares_observable_output(self):
        result = {
            "requests": [
                {
                    "request_id": "a",
                    "iteration": 0,
                    "content_sha256": "same",
                    "reasoning_sha256": "reason",
                    "finish_reason": "stop",
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1,
                              "total_tokens": 3},
                },
                {
                    "request_id": "a",
                    "iteration": 1,
                    "content_sha256": "different",
                    "reasoning_sha256": "reason",
                    "finish_reason": "stop",
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1,
                              "total_tokens": 3},
                },
            ]
        }
        errors = cache_benchmark.validate_repeat_consistency(result)
        self.assertEqual(len(errors), 1)
        self.assertIn("content_sha256", errors[0])

    def test_zstd_summary_uses_counter_deltas(self):
        result = {
            "props_before": {
                "paged_cache_snapshot_zstd_compress_input_bytes": 100,
                "paged_cache_snapshot_zstd_compress_output_bytes": 40,
                "paged_cache_snapshot_zstd_compress_seconds": 1.0,
                "paged_cache_snapshot_zstd_decompress_output_bytes": 100,
                "paged_cache_snapshot_zstd_decompress_seconds": 2.0,
            },
            "props_after": {
                "paged_cache_snapshot_zstd_compress_input_bytes": 1100,
                "paged_cache_snapshot_zstd_compress_output_bytes": 540,
                "paged_cache_snapshot_zstd_compress_seconds": 2.0,
                "paged_cache_snapshot_zstd_decompress_output_bytes": 1100,
                "paged_cache_snapshot_zstd_decompress_seconds": 2.5,
            },
        }
        summary = cache_benchmark.summarize_zstd(result)
        self.assertAlmostEqual(summary["stored_ratio"], 0.5)
        self.assertAlmostEqual(summary["compress_mib_per_second"], 1000 / 1048576)
        self.assertAlmostEqual(summary["decompress_mib_per_second"], 2000 / 1048576)

    def test_zstd_summary_supports_prefix_cache_counters(self):
        result = {
            "props_before": {
                "prefix_cache_zstd_compress_input_bytes": 10,
                "prefix_cache_zstd_compress_output_bytes": 4,
                "prefix_cache_zstd_compress_seconds": 0.25,
                "prefix_cache_zstd_decompress_output_bytes": 10,
                "prefix_cache_zstd_decompress_seconds": 0.5,
            },
            "props_after": {
                "prefix_cache_zstd_compress_input_bytes": 1010,
                "prefix_cache_zstd_compress_output_bytes": 504,
                "prefix_cache_zstd_compress_seconds": 1.25,
                "prefix_cache_zstd_decompress_output_bytes": 1010,
                "prefix_cache_zstd_decompress_seconds": 1.0,
            },
        }
        summary = cache_benchmark.summarize_zstd(
            result, metric_prefix="prefix_cache_zstd"
        )
        self.assertAlmostEqual(summary["stored_ratio"], 0.5)
        self.assertAlmostEqual(
            summary["compress_mib_per_second"], 1000 / 1048576
        )
        self.assertAlmostEqual(
            summary["decompress_mib_per_second"], 2000 / 1048576
        )


if __name__ == "__main__":
    unittest.main()
