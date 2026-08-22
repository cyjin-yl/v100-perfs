#!/usr/bin/env python3
"""Run capacity-aware FastLLM resident-batch or CPU L3 acceptance benchmarks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
FEATURE_BENCH_PATH = SCRIPT_DIR / "feature_bench.py"
FEATURE_SPEC = importlib.util.spec_from_file_location(
    "fastllm_feature_bench_for_cache", FEATURE_BENCH_PATH
)
feature_bench = importlib.util.module_from_spec(FEATURE_SPEC)
assert FEATURE_SPEC.loader is not None
FEATURE_SPEC.loader.exec_module(feature_bench)

_OBSERVABLE_FIELDS = (
    "content_sha256",
    "reasoning_sha256",
    "finish_reason",
    "usage",
)


def build_scenario(mode: str) -> dict[str, Any]:
    requests_dir = SCRIPT_DIR / "requests"
    if mode == "resident":
        return {
            "mode": mode,
            "request_files": [
                requests_dir / "resident_32k_a.json",
                requests_dir / "resident_32k_b.json",
            ],
            "concurrency": 2,
            "default_runs": 2,
            "expected_token_pool": 262144,
            "required_prop_deltas": {},
            "maximum_prop_deltas": {
                "cpu_request_swap_disk_spills": 0.0,
            },
        }
    if mode == "l3":
        return {
            "mode": mode,
            "request_files": [
                requests_dir / "resident_32k_a.json",
                requests_dir / "resident_32k_b.json",
                requests_dir / "resident_32k_c.json",
            ],
            "concurrency": 3,
            "default_runs": 1,
            "expected_token_pool": 32768,
            "required_prop_deltas": {
                "cpu_request_swap_disk_spills": 1.0,
                "cpu_request_swap_disk_restores": 1.0,
                "cpu_request_swap_disk_write_bytes": 1.0,
                "cpu_request_swap_disk_read_bytes": 1.0,
                "paged_cache_snapshot_zstd_compress_calls": 1.0,
                "paged_cache_snapshot_zstd_decompress_calls": 1.0,
            },
            "maximum_prop_deltas": {
                "cpu_request_swap_suspended": 0.0,
            },
        }
    if mode == "prefix":
        return {
            "mode": mode,
            "request_files": [
                requests_dir / "prefix_seed_a.json",
                requests_dir / "prefix_shared_variant.json",
                requests_dir / "prefix_pressure_b.json",
                requests_dir / "prefix_restore_a.json",
            ],
            "warmup_file": requests_dir / "prefix_warmup.json",
            "concurrency": 1,
            "default_runs": 1,
            "expected_token_pool": 16384,
            "required_prop_deltas": {
                "prefix_cache_gpu_hit_pages": 1.0,
                "prefix_cache_disk_write_bytes": 1.0,
                "prefix_cache_disk_read_bytes": 1.0,
                "prefix_cache_disk_hits": 1.0,
                "prefix_cache_zstd_compress_calls": 1.0,
                "prefix_cache_zstd_decompress_calls": 1.0,
            },
            "maximum_prop_deltas": {},
        }
    raise ValueError(f"unknown scenario {mode!r}")


def validate_profile(
    props: dict[str, Any], mode: str, expected_token_pool: int
) -> list[str]:
    expected: dict[str, Any] = {
        "backend": "fastllm",
        "kv_cache_dtype": "turbo3",
        "token_pool": expected_token_pool,
    }
    if mode in ("resident", "l3"):
        expected["cpu_request_swap_enabled"] = True
    if mode == "l3":
        expected.update({
            "cpu_request_swap_zstd_enabled": True,
            "cpu_request_swap_disk_enabled": True,
        })
    elif mode == "prefix":
        expected.update({
            "prefix_cache_cpu_tier_enabled": True,
            "prefix_cache_disk_tier_enabled": True,
        })
    errors: list[str] = []
    if props.get("_error"):
        errors.append(f"props request failed: {props['_error']}")
    for key, value in expected.items():
        if props.get(key) != value:
            errors.append(
                f"profile {key} is {props.get(key)!r}; expected {value!r}"
            )
    if mode == "l3":
        for key in (
            "paged_cache_snapshot_zstd_compress_calls",
            "paged_cache_snapshot_zstd_decompress_calls",
        ):
            if not isinstance(props.get(key), (int, float)):
                errors.append(f"profile does not expose numeric {key}")
    if mode == "prefix":
        for key in (
            "prefix_cache_recompute_tokens_per_second",
            "prefix_cache_disk_read_mib_per_second",
            "prefix_cache_zstd_compress_calls",
            "prefix_cache_zstd_compress_input_bytes",
            "prefix_cache_zstd_compress_output_bytes",
            "prefix_cache_zstd_compress_seconds",
            "prefix_cache_zstd_decompress_calls",
            "prefix_cache_zstd_decompress_input_bytes",
            "prefix_cache_zstd_decompress_output_bytes",
            "prefix_cache_zstd_decompress_seconds",
        ):
            value = props.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                errors.append(f"profile does not expose numeric {key}")
    return errors


def validate_repeat_consistency(result: dict[str, Any]) -> list[str]:
    by_request: dict[str, list[dict[str, Any]]] = {}
    for item in result.get("requests", []):
        by_request.setdefault(str(item.get("request_id", "")), []).append(item)
    errors: list[str] = []
    for request_id, items in by_request.items():
        items.sort(key=lambda item: int(item.get("iteration", -1)))
        if len(items) < 2:
            continue
        reference = items[0]
        for item in items[1:]:
            for field in _OBSERVABLE_FIELDS:
                if item.get(field) != reference.get(field):
                    errors.append(
                        f"{request_id} iteration {item.get('iteration')} "
                        f"field {field} differs from iteration 0"
                    )
    return errors

def validate_prefix_scenario(result: dict[str, Any]) -> list[str]:
    by_id = {
        str(item.get("request_id", "")): item
        for item in result.get("requests", [])
    }
    required_ids = (
        "prefix_seed_a",
        "prefix_shared_variant",
        "prefix_pressure_b",
        "prefix_restore_a",
    )
    errors = [
        f"prefix scenario is missing {request_id}"
        for request_id in required_ids
        if request_id not in by_id
    ]
    if errors:
        return errors
    seed = by_id["prefix_seed_a"]
    variant = by_id["prefix_shared_variant"]
    restored = by_id["prefix_restore_a"]
    for field in _OBSERVABLE_FIELDS:
        if restored.get(field) != seed.get(field):
            errors.append(
                f"prefix restore field {field} differs from seed"
            )
    seed_ttft = seed.get("ttft_seconds")
    variant_ttft = variant.get("ttft_seconds")
    restore_ttft = restored.get("ttft_seconds")
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in (seed_ttft, variant_ttft, restore_ttft)
    ):
        errors.append("prefix scenario did not produce numeric TTFT")
        return errors
    if float(variant_ttft) >= float(seed_ttft):
        errors.append(
            "page-aligned shared-prefix hit was not faster than seed prefill"
        )
    if float(restore_ttft) >= float(seed_ttft):
        errors.append(
            "disk prefix restore was not faster than recompute"
        )
    return errors



def _numeric_delta(
    result: dict[str, Any], key: str
) -> float:
    before = result.get("props_before", {}).get(key)
    after = result.get("props_after", {}).get(key)
    if (
        isinstance(before, bool)
        or isinstance(after, bool)
        or not isinstance(before, (int, float))
        or not isinstance(after, (int, float))
    ):
        return 0.0
    return float(after) - float(before)


def summarize_zstd(
    result: dict[str, Any],
    metric_prefix: str = "paged_cache_snapshot_zstd",
) -> dict[str, float | None]:
    compress_input = _numeric_delta(
        result, f"{metric_prefix}_compress_input_bytes"
    )
    compress_output = _numeric_delta(
        result, f"{metric_prefix}_compress_output_bytes"
    )
    compress_seconds = _numeric_delta(
        result, f"{metric_prefix}_compress_seconds"
    )
    decompress_output = _numeric_delta(
        result, f"{metric_prefix}_decompress_output_bytes"
    )
    decompress_seconds = _numeric_delta(
        result, f"{metric_prefix}_decompress_seconds"
    )
    mib = 1024.0 * 1024.0
    return {
        "compress_input_bytes": compress_input,
        "compress_output_bytes": compress_output,
        "stored_ratio": (
            compress_output / compress_input if compress_input > 0 else None
        ),
        "compress_seconds": compress_seconds,
        "compress_mib_per_second": (
            compress_input / mib / compress_seconds
            if compress_seconds > 0 else None
        ),
        "decompress_output_bytes": decompress_output,
        "decompress_seconds": decompress_seconds,
        "decompress_mib_per_second": (
            decompress_output / mib / decompress_seconds
            if decompress_seconds > 0 else None
        ),
    }


def _metadata(values: Iterable[str]) -> dict[str, str]:
    return feature_bench._parse_metadata(values)


def run_scenario(
    *,
    mode: str,
    endpoint: str,
    output: Path,
    runs: int | None,
    timeout: float,
    expected_token_pool: int | None,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    scenario = build_scenario(mode)
    pool = (
        expected_token_pool
        if expected_token_pool is not None
        else int(scenario["expected_token_pool"])
    )
    props = feature_bench._fetch_json(endpoint, "/props", timeout)
    profile_errors = validate_profile(props, mode, pool)
    if profile_errors:
        raise RuntimeError("; ".join(profile_errors))

    warmup_result: dict[str, Any] | None = None
    warmup_file = scenario.get("warmup_file")
    if warmup_file is not None:
        warmup_result = feature_bench.run_http_benchmark(
            endpoint=endpoint,
            requests=[feature_bench._load_request(warmup_file)],
            runs=1,
            concurrency=1,
            timeout=timeout,
            metadata={**dict(metadata or {}), "phase": "warmup"},
        )
        warmup_path = output.with_name(output.stem + "_warmup.json")
        feature_bench._atomic_write_json(warmup_path, warmup_result)
    request_specs = [
        feature_bench._load_request(path)
        for path in scenario["request_files"]
    ]
    current_runs = runs if runs is not None else int(scenario["default_runs"])
    control_result: dict[str, Any] | None = None
    if mode == "l3":
        control_result = feature_bench.run_http_benchmark(
            endpoint=endpoint,
            requests=request_specs,
            runs=1,
            concurrency=1,
            timeout=timeout,
            metadata={**dict(metadata or {}), "phase": "single_control"},
        )
        controls_path = output.with_name(output.stem + "_controls.json")
        feature_bench._atomic_write_json(controls_path, control_result)

    result = feature_bench.run_http_benchmark(
        endpoint=endpoint,
        requests=request_specs,
        runs=current_runs,
        concurrency=int(scenario["concurrency"]),
        timeout=timeout,
        metadata={**dict(metadata or {}), "scenario": mode},
        required_prop_deltas=dict(scenario["required_prop_deltas"]),
        maximum_prop_deltas=dict(scenario["maximum_prop_deltas"]),
    )
    result["scenario"] = {
        "mode": mode,
        "expected_token_pool": pool,
        "request_files": [str(path) for path in scenario["request_files"]],
    }
    extra_errors = validate_repeat_consistency(result)
    if mode == "prefix":
        extra_errors.extend(validate_prefix_scenario(result))
        result["warmup_result"] = output.with_name(
            output.stem + "_warmup.json"
        ).name
        result["zstd_summary"] = summarize_zstd(
            result, metric_prefix="prefix_cache_zstd"
        )
    if control_result is not None:
        comparison = feature_bench.compare_with_reference(
            result, control_result
        )
        result["reference_comparison"] = comparison
        extra_errors.extend(comparison["errors"])
        result["control_result"] = output.with_name(
            output.stem + "_controls.json"
        ).name
        result["zstd_summary"] = summarize_zstd(result)
    if extra_errors:
        result["validation"]["errors"].extend(extra_errors)
        result["validation"]["passed"] = False
    feature_bench._atomic_write_json(output, result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("resident", "l3", "prefix"), required=True
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8002")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--expected-token-pool", type=int)
    parser.add_argument("--metadata", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_scenario(
        mode=args.mode,
        endpoint=args.endpoint,
        output=args.output,
        runs=args.runs,
        timeout=args.timeout,
        expected_token_pool=args.expected_token_pool,
        metadata=_metadata(args.metadata),
    )
    summary = {
        "output": str(args.output),
        "validation": result["validation"],
        "request_count": len(result.get("requests", [])),
        "reference_comparison": result.get("reference_comparison"),
        "zstd_summary": result.get("zstd_summary"),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if result["validation"]["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
