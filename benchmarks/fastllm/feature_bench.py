#!/usr/bin/env python3
"""Reproducible FastLLM feature benchmarks with strict SSE validation."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1


class SSEAccumulator:
    """Incrementally parses SSE events and accumulates OpenAI stream fields."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._data_lines: list[bytes] = []
        self._content: list[str] = []
        self._reasoning: list[str] = []
        self._usage: dict[str, Any] = {}
        self._finish_reason: str | None = None
        self._first_token_time: float | None = None
        self._done_events = 0
        self._malformed_events = 0
        self._event_count = 0

    def feed(self, chunk: bytes, timestamp: float | None = None) -> None:
        if not chunk:
            return
        self._buffer.extend(chunk)
        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                return
            line = bytes(self._buffer[:newline])
            del self._buffer[: newline + 1]
            if line.endswith(b"\r"):
                line = line[:-1]
            self._consume_line(line, timestamp)

    def finish(self, timestamp: float | None = None) -> None:
        if self._buffer:
            line = bytes(self._buffer)
            self._buffer.clear()
            if line.endswith(b"\r"):
                line = line[:-1]
            self._consume_line(line, timestamp)
        if self._data_lines:
            self._dispatch(timestamp)

    def _consume_line(self, line: bytes, timestamp: float | None) -> None:
        if not line:
            if self._data_lines:
                self._dispatch(timestamp)
            return
        if line.startswith(b":"):
            return
        if line == b"data" or line.startswith(b"data:"):
            value = line[5:] if line.startswith(b"data:") else b""
            if value.startswith(b" "):
                value = value[1:]
            self._data_lines.append(value)

    def _dispatch(self, timestamp: float | None) -> None:
        payload = b"\n".join(self._data_lines)
        self._data_lines.clear()
        self._event_count += 1
        if payload == b"[DONE]":
            self._done_events += 1
            return
        try:
            event = json.loads(payload.decode("utf-8", errors="replace"))
        except (UnicodeError, json.JSONDecodeError, TypeError):
            self._malformed_events += 1
            return
        if not isinstance(event, dict):
            self._malformed_events += 1
            return

        usage = event.get("usage")
        if isinstance(usage, dict):
            self._usage = dict(usage)
        choices = event.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                self._finish_reason = str(finish_reason)
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            reasoning = delta.get("reasoning_content")
            emitted = False
            if isinstance(content, str) and content:
                self._content.append(content)
                emitted = True
            if isinstance(reasoning, str) and reasoning:
                self._reasoning.append(reasoning)
                emitted = True
            if emitted and self._first_token_time is None:
                self._first_token_time = timestamp

    def result(self) -> dict[str, Any]:
        return {
            "content": "".join(self._content),
            "reasoning_content": "".join(self._reasoning),
            "usage": dict(self._usage),
            "finish_reason": self._finish_reason,
            "first_token_time": self._first_token_time,
            "done_events": self._done_events,
            "malformed_events": self._malformed_events,
            "event_count": self._event_count,
        }


def validate_request_result(result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if result.get("status") != 200:
        errors.append(f"HTTP status is {result.get('status')!r}, expected 200")
    if result.get("done_events") != 1:
        errors.append(
            f"[DONE] event count is {result.get('done_events')!r}, expected 1"
        )
    if result.get("malformed_events") != 0:
        errors.append(
            f"malformed event count is {result.get('malformed_events')!r}, expected 0"
        )
    if result.get("finish_reason") is None:
        errors.append("finish_reason is missing")

    usage = result.get("usage")
    if not isinstance(usage, dict):
        errors.append("usage is missing")
    else:
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        if not all(isinstance(value, int) and value >= 0 for value in (prompt, completion, total)):
            errors.append("usage token counts are not non-negative integers")
        elif prompt + completion != total:
            errors.append(
                "usage total_tokens does not equal prompt_tokens + completion_tokens"
            )
    if not result.get("content_sha256"):
        errors.append("content_sha256 is missing")
    return errors


def _result_key(result: dict[str, Any]) -> tuple[str, int]:
    return str(result.get("request_id", "")), int(result.get("iteration", -1))


def compare_with_reference(
    current: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    reference_by_key = {
        _result_key(item): item for item in reference.get("requests", [])
    }
    errors: list[str] = []
    checked = 0
    for item in current.get("requests", []):
        key = _result_key(item)
        expected = reference_by_key.get(key)
        if expected is None:
            errors.append(f"missing reference request {key[0]} iteration {key[1]}")
            continue
        checked += 1
        for field in ("content_sha256", "reasoning_sha256", "finish_reason", "usage"):
            if item.get(field) != expected.get(field):
                errors.append(
                    f"{key[0]} iteration {key[1]} field {field} differs"
                )
    current_keys = {_result_key(item) for item in current.get("requests", [])}
    for key in reference_by_key.keys() - current_keys:
        errors.append(f"current run omitted request {key[0]} iteration {key[1]}")
    return {"match": not errors, "checked": checked, "errors": errors}

def validate_prop_deltas(
    before: dict[str, Any],
    after: dict[str, Any],
    required: dict[str, float],
    maximums: dict[str, float] | None = None,
) -> list[str]:
    errors: list[str] = []
    maximums = dict(maximums or {})
    for key in sorted(set(required) | set(maximums)):
        before_value = before.get(key)
        after_value = after.get(key)
        if (
            isinstance(before_value, bool)
            or isinstance(after_value, bool)
            or not isinstance(before_value, (int, float))
            or not isinstance(after_value, (int, float))
        ):
            errors.append(f"required prop {key} is missing or non-numeric")
            continue
        delta = float(after_value) - float(before_value)
        if not math.isfinite(delta):
            errors.append(f"required prop {key} delta is non-finite")
        elif key in required and delta < required[key]:
            errors.append(
                f"required prop {key} delta {delta!r} is below {required[key]}"
            )
        elif key in maximums and delta > maximums[key]:
            errors.append(
                f"required prop {key} delta {delta!r} is above {maximums[key]}"
            )
    return errors


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()


def _fetch_json(endpoint: str, path: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint.rstrip("/") + path,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            parsed = json.loads(body.decode("utf-8", errors="replace"))
            if not isinstance(parsed, dict):
                raise ValueError("response is not a JSON object")
            return parsed
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}


def _call_request(
    endpoint: str,
    request_spec: dict[str, Any],
    iteration: int,
    ordinal: int,
    timeout: float,
    api_key: str | None,
) -> tuple[int, dict[str, Any]]:
    request_id = str(request_spec["id"])
    body = request_spec["body"]
    encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
    headers = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    path = str(request_spec.get("path", "/v1/chat/completions"))
    request = urllib.request.Request(
        endpoint.rstrip("/") + path,
        data=encoded,
        headers=headers,
        method="POST",
    )

    accumulator = SSEAccumulator()
    raw = bytearray()
    started = time.perf_counter()
    status: int | None = None
    transport_error: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            read_chunk = getattr(response, "read1", response.read)
            while True:
                chunk = read_chunk(64 * 1024)
                if not chunk:
                    break
                raw.extend(chunk)
                accumulator.feed(chunk, time.perf_counter())
    except urllib.error.HTTPError as exc:
        status = exc.code
        error_body = exc.read()
        raw.extend(error_body)
        transport_error = error_body.decode("utf-8", errors="replace")
    except Exception as exc:
        transport_error = f"{type(exc).__name__}: {exc}"
    ended = time.perf_counter()
    accumulator.finish(ended)
    stream = accumulator.result()
    first_token_time = stream.pop("first_token_time")
    content = str(stream["content"])
    reasoning = str(stream["reasoning_content"])
    result = {
        "request_id": request_id,
        "iteration": iteration,
        "status": status,
        "wall_seconds": ended - started,
        "ttft_seconds": (
            first_token_time - started if first_token_time is not None else None
        ),
        "request_sha256": hashlib.sha256(encoded).hexdigest(),
        "response_bytes": len(raw),
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": _sha256_text(content),
        "reasoning_sha256": _sha256_text(reasoning),
        "transport_error": transport_error,
        **stream,
    }
    return ordinal, result


def subprocess_error_text(stdout: str, stderr: str) -> str:
    return stderr.strip() or stdout.strip() or "command failed without diagnostics"


def _gpu_metadata() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if completed.returncode != 0:
        return {
            "available": False,
            "error": subprocess_error_text(
                completed.stdout, completed.stderr
            ),
        }
    return {
        "available": True,
        "query": command[1],
        "rows": [line for line in completed.stdout.splitlines() if line],
    }


def _summary(requests: list[dict[str, Any]]) -> dict[str, Any]:
    by_id: dict[str, dict[str, Any]] = {}
    for request_id in sorted({str(item["request_id"]) for item in requests}):
        selected = [item for item in requests if item["request_id"] == request_id]
        walls = [float(item["wall_seconds"]) for item in selected]
        ttfts = [
            float(item["ttft_seconds"])
            for item in selected
            if item.get("ttft_seconds") is not None
        ]
        by_id[request_id] = {
            "runs": len(selected),
            "wall_median_seconds": statistics.median(walls),
            "wall_min_seconds": min(walls),
            "wall_max_seconds": max(walls),
            "ttft_median_seconds": statistics.median(ttfts) if ttfts else None,
        }
    return {"by_request": by_id, "request_count": len(requests)}


def run_http_benchmark(
    *,
    endpoint: str,
    requests: list[dict[str, Any]],
    runs: int,
    concurrency: int,
    timeout: float,
    api_key: str | None = None,
    metadata: dict[str, str] | None = None,
    required_prop_deltas: dict[str, float] | None = None,
    maximum_prop_deltas: dict[str, float] | None = None,
) -> dict[str, Any]:
    if runs < 1:
        raise ValueError("runs must be positive")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if not requests:
        raise ValueError("at least one request is required")
    for request_spec in requests:
        if not isinstance(request_spec.get("id"), str) or not request_spec["id"]:
            raise ValueError("every request needs a non-empty string id")
        if not isinstance(request_spec.get("body"), dict):
            raise ValueError("every request needs a JSON object body")

    props_before = _fetch_json(endpoint, "/props", timeout)
    completed_requests: list[dict[str, Any]] = []
    ordinal = 0
    for iteration in range(runs):
        jobs: list[tuple[int, dict[str, Any], int]] = []
        for request_spec in requests:
            jobs.append((ordinal, request_spec, iteration))
            ordinal += 1
        if concurrency == 1:
            iteration_results = [
                _call_request(
                    endpoint,
                    request_spec,
                    current_iteration,
                    current_ordinal,
                    timeout,
                    api_key,
                )
                for current_ordinal, request_spec, current_iteration in jobs
            ]
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=concurrency
            ) as executor:
                futures = [
                    executor.submit(
                        _call_request,
                        endpoint,
                        request_spec,
                        current_iteration,
                        current_ordinal,
                        timeout,
                        api_key,
                    )
                    for current_ordinal, request_spec, current_iteration in jobs
                ]
                iteration_results = [future.result() for future in futures]
        completed_requests.extend(
            result for _, result in sorted(iteration_results, key=lambda item: item[0])
        )
    props_after = _fetch_json(endpoint, "/props", timeout)

    validation_errors: list[str] = []
    for result in completed_requests:
        prefix = f"{result['request_id']} iteration {result['iteration']}: "
        validation_errors.extend(
            prefix + error for error in validate_request_result(result)
        )
    validation_errors.extend(
        validate_prop_deltas(
            props_before,
            props_after,
            dict(required_prop_deltas or {}),
            dict(maximum_prop_deltas or {}),
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "fastllm_feature_http",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gpu": _gpu_metadata(),
            "user_metadata": dict(metadata or {}),
        },
        "endpoint": endpoint,
        "configuration": {
            "runs": runs,
            "concurrency": concurrency,
            "timeout_seconds": timeout,
            "request_ids": [item["id"] for item in requests],
            "required_prop_deltas": dict(required_prop_deltas or {}),
            "maximum_prop_deltas": dict(maximum_prop_deltas or {}),
        },
        "props_before": props_before,
        "props_after": props_after,
        "requests": completed_requests,
        "summary": _summary(completed_requests),
        "validation": {
            "passed": not validation_errors,
            "errors": validation_errors,
        },
    }


_SENSITIVE_ENV_PARTS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "API_KEY",
    "AUTHORIZATION",
    "COOKIE",
    "CREDENTIAL",
)


def _validate_command_config(config: dict[str, Any]) -> None:
    if not isinstance(config.get("name"), str) or not config["name"]:
        raise ValueError("command benchmark needs a non-empty name")
    command = config.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) and item for item in command)
    ):
        raise ValueError("command must be a non-empty string array")
    variants = config.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("at least one command variant is required")
    names: set[str] = set()
    for variant in variants:
        if not isinstance(variant, dict):
            raise ValueError("every variant must be a JSON object")
        name = variant.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("variant names must be non-empty and unique")
        names.add(name)
        environment = variant.get("env", {})
        if not isinstance(environment, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in environment.items()
        ):
            raise ValueError("variant env must map strings to strings")
        for key in environment:
            upper = key.upper()
            if any(part in upper for part in _SENSITIVE_ENV_PARTS):
                raise ValueError(
                    f"variant environment key {key!r} is sensitive and "
                    "must not be recorded"
                )
    for field in ("warmups", "runs"):
        value = config.get(field, 0 if field == "warmups" else 1)
        if not isinstance(value, int) or value < (0 if field == "warmups" else 1):
            raise ValueError(f"{field} has an invalid count")
    metrics = config.get("metrics", {})
    correctness = config.get("correctness", {})
    if not isinstance(metrics, dict) or not isinstance(correctness, dict):
        raise ValueError("metrics and correctness must be JSON objects")
    for name, metric in metrics.items():
        if (
            not isinstance(name, str)
            or not isinstance(metric, dict)
            or not isinstance(metric.get("regex"), str)
        ):
            raise ValueError("every metric needs a name and regex")
        re.compile(metric["regex"])
    for name, pattern in correctness.items():
        if not isinstance(name, str) or not isinstance(pattern, str):
            raise ValueError("correctness must map names to regex strings")
        re.compile(pattern)


def _fingerprint_executable(command: str) -> dict[str, Any]:
    resolved = shutil.which(command)
    path = Path(resolved if resolved is not None else command)
    if not path.is_file():
        return {"path": str(path), "available": False}
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "available": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _parse_command_output(
    output: str,
    metrics: dict[str, dict[str, Any]],
    correctness: dict[str, str],
) -> tuple[dict[str, float], dict[str, str], list[str]]:
    parsed_metrics: dict[str, float] = {}
    parsed_correctness: dict[str, str] = {}
    errors: list[str] = []
    for name, definition in metrics.items():
        match = re.search(definition["regex"], output, re.MULTILINE)
        if match is None:
            errors.append(f"metric {name} did not match")
            continue
        try:
            value = float(match.group(1))
        except (IndexError, ValueError):
            errors.append(f"metric {name} did not capture a number")
            continue
        if not math.isfinite(value):
            errors.append(f"metric {name} is not finite")
            continue
        parsed_metrics[name] = value
    for name, pattern in correctness.items():
        match = re.search(pattern, output, re.MULTILINE)
        if match is None:
            errors.append(f"correctness field {name} did not match")
            continue
        try:
            parsed_correctness[name] = match.group(1)
        except IndexError:
            errors.append(f"correctness field {name} has no capture group")
    return parsed_metrics, parsed_correctness, errors


def _run_command_variant(
    config: dict[str, Any],
    variant: dict[str, Any],
    iteration: int,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(variant.get("env", {}))
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            config["command"],
            cwd=config.get("cwd"),
            env=environment,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=float(config.get("timeout_seconds", 3600.0)),
        )
        return_code: int | None = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        transport_error = None
    except (OSError, subprocess.SubprocessError) as exc:
        return_code = None
        stdout = ""
        stderr = ""
        transport_error = f"{type(exc).__name__}: {exc}"
    ended = time.perf_counter()
    output = stdout + "\n" + stderr
    metrics, correctness, parse_errors = _parse_command_output(
        output,
        config.get("metrics", {}),
        config.get("correctness", {}),
    )
    return {
        "variant": variant["name"],
        "iteration": iteration,
        "return_code": return_code,
        "wall_seconds": ended - started,
        "stdout": stdout,
        "stderr": stderr,
        "output_sha256": hashlib.sha256(
            output.encode("utf-8", errors="replace")
        ).hexdigest(),
        "metrics": metrics,
        "correctness": correctness,
        "parse_errors": parse_errors,
        "transport_error": transport_error,
    }


def _command_summary(
    runs: list[dict[str, Any]],
    variants: list[dict[str, Any]],
    metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    baseline = variants[0]["name"]
    metric_summary: dict[str, Any] = {}
    for metric_name, definition in metrics.items():
        by_variant: dict[str, Any] = {}
        medians: dict[str, float] = {}
        for variant in variants:
            name = variant["name"]
            values = [
                float(run["metrics"][metric_name])
                for run in runs
                if run["variant"] == name and metric_name in run["metrics"]
            ]
            if not values:
                continue
            median = statistics.median(values)
            medians[name] = median
            by_variant[name] = {
                "values": values,
                "median": median,
                "min": min(values),
                "max": max(values),
            }
        baseline_value = medians.get(baseline)
        if baseline_value is not None:
            for name, median in medians.items():
                if median == 0.0 or baseline_value == 0.0:
                    speedup = None
                elif definition.get("lower_is_better", True):
                    speedup = baseline_value / median
                else:
                    speedup = median / baseline_value
                by_variant[name]["speedup_vs_baseline"] = speedup
        metric_summary[metric_name] = by_variant
        metric_summary[metric_name]["unit"] = definition.get("unit")
    return {
        "baseline": baseline,
        "metrics": metric_summary,
    }


def run_command_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    _validate_command_config(config)
    variants = config["variants"]
    warmups = int(config.get("warmups", 0))
    runs_count = int(config.get("runs", 1))
    validation_errors: list[str] = []
    for warmup in range(warmups):
        for variant in variants:
            result = _run_command_variant(config, variant, warmup)
            if result["return_code"] != 0:
                raise RuntimeError(
                    f"warmup {warmup} for {variant['name']} failed: "
                    f"{result['transport_error'] or result['stderr']}"
                )

    runs: list[dict[str, Any]] = []
    for iteration in range(runs_count):
        for variant in variants:
            result = _run_command_variant(config, variant, iteration)
            runs.append(result)
            prefix = f"{variant['name']} iteration {iteration}: "
            if result["return_code"] != 0:
                validation_errors.append(
                    prefix + f"return code is {result['return_code']!r}"
                )
            validation_errors.extend(
                prefix + error for error in result["parse_errors"]
            )

    expected_correctness: dict[str, str] = {}
    for result in runs:
        for name, value in result["correctness"].items():
            if name not in expected_correctness:
                expected_correctness[name] = value
            elif expected_correctness[name] != value:
                validation_errors.append(
                    f"{result['variant']} iteration {result['iteration']}: "
                    f"correctness field {name} differs"
                )
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "fastllm_feature_command_ab",
        "name": config["name"],
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gpu": _gpu_metadata(),
            "executable": _fingerprint_executable(config["command"][0]),
            "user_metadata": dict(config.get("metadata", {})),
        },
        "configuration": {
            "command": list(config["command"]),
            "cwd": config.get("cwd"),
            "warmups": warmups,
            "runs": runs_count,
            "variants": [
                {"name": variant["name"], "env": dict(variant.get("env", {}))}
                for variant in variants
            ],
            "metrics": config.get("metrics", {}),
            "correctness": config.get("correctness", {}),
        },
        "runs": runs,
        "summary": _command_summary(
            runs, variants, config.get("metrics", {})
        ),
        "validation": {
            "passed": not validation_errors,
            "errors": validation_errors,
        },
    }


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        json.dump(value, temporary, ensure_ascii=False, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def expand_request_spec(
    parsed: dict[str, Any], source: str
) -> dict[str, Any]:
    repeat = parsed.get("repeat")
    if repeat is None:
        if isinstance(parsed.get("body"), dict):
            result = dict(parsed)
            result.setdefault("id", Path(source).stem)
            return result
        return {"id": Path(source).stem, "body": parsed}
    if not isinstance(parsed.get("body"), dict) or not isinstance(repeat, dict):
        raise ValueError(f"{source}: repeat requires body and repeat objects")
    field = repeat.get("field")
    prefix = repeat.get("prefix", "")
    unit = repeat.get("unit")
    suffix = repeat.get("suffix", "")
    count = repeat.get("count")
    if not isinstance(field, str) or not field:
        raise ValueError(f"{source}: repeat field must be a non-empty string")
    if (
        not isinstance(prefix, str)
        or not isinstance(unit, str)
        or not isinstance(suffix, str)
    ):
        raise ValueError(f"{source}: repeat prefix, unit, and suffix must be strings")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"{source}: repeat count must be a non-negative integer")
    body = dict(parsed["body"])
    if field in body:
        raise ValueError(f"{source}: repeated field {field!r} already exists in body")
    expanded_bytes = (
        len(prefix.encode("utf-8"))
        + len(unit.encode("utf-8")) * count
        + len(suffix.encode("utf-8"))
    )
    if expanded_bytes > 64 * 1024 * 1024:
        raise ValueError(f"{source}: repeated field exceeds 64 MiB")
    body[field] = prefix + unit * count + suffix
    result = {key: value for key, value in parsed.items() if key != "repeat"}
    result["body"] = body
    result.setdefault("id", Path(source).stem)
    return result


def _load_request(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"{path}: request must be a JSON object")
    return expand_request_spec(parsed, str(path))


def _parse_metadata(values: Iterable[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key:
            raise ValueError(f"invalid metadata {value!r}; expected KEY=VALUE")
        metadata[key] = item
    return metadata

def _parse_prop_deltas(values: Iterable[str]) -> dict[str, float]:
    required: dict[str, float] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key:
            raise ValueError(
                f"invalid prop delta {value!r}; expected KEY=NUMBER"
            )
        minimum = float(item)
        if not math.isfinite(minimum):
            raise ValueError(f"invalid prop delta {value!r}; number must be finite")
        required[key] = minimum
    return required


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    http = subparsers.add_parser("http", help="benchmark OpenAI SSE requests")
    http.add_argument("--endpoint", required=True)
    http.add_argument("--request", action="append", required=True, type=Path)
    http.add_argument("--runs", type=int, default=1)
    http.add_argument("--concurrency", type=int, default=1)
    http.add_argument("--timeout", type=float, default=1800.0)
    http.add_argument("--api-key-env")
    http.add_argument("--metadata", action="append", default=[])
    http.add_argument(
        "--require-prop-delta",
        action="append",
        default=[],
        metavar="KEY=MINIMUM",
    )
    http.add_argument(
        "--max-prop-delta",
        action="append",
        default=[],
        metavar="KEY=MAXIMUM",
    )
    http.add_argument("--reference", type=Path)
    http.add_argument("--output", required=True, type=Path)
    command = subparsers.add_parser(
        "command", help="run an interleaved command A/B benchmark"
    )
    command.add_argument("--config", required=True, type=Path)
    command.add_argument("--metadata", action="append", default=[])
    command.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "command":
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("command config must be a JSON object")
        config_metadata = config.setdefault("metadata", {})
        if not isinstance(config_metadata, dict):
            raise ValueError("command config metadata must be a JSON object")
        config_metadata.update(_parse_metadata(args.metadata))
        result = run_command_benchmark(config)
        _atomic_write_json(args.output, result)
        return 0 if result["validation"]["passed"] else 2

    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ValueError(f"{args.api_key_env} is empty")
    result = run_http_benchmark(
        endpoint=args.endpoint,
        requests=[_load_request(path) for path in args.request],
        runs=args.runs,
        concurrency=args.concurrency,
        timeout=args.timeout,
        api_key=api_key,
        metadata=_parse_metadata(args.metadata),
        required_prop_deltas=_parse_prop_deltas(args.require_prop_delta),
        maximum_prop_deltas=_parse_prop_deltas(args.max_prop_delta),
    )
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        result["reference_comparison"] = compare_with_reference(result, reference)
    _atomic_write_json(args.output, result)
    reference_passed = result.get("reference_comparison", {}).get("match", True)
    return 0 if result["validation"]["passed"] and reference_passed else 2


if __name__ == "__main__":
    sys.exit(main())
