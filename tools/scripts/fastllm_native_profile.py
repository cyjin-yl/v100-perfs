#!/usr/bin/env python3
"""Strict manifest manager for native FastLLM GGUF/mmproj profiles.

This intentionally does not use the Python ftllm profile loader. Native C++
apiserver has a different command line and readiness contract.
"""
from __future__ import annotations

import argparse
import contextlib
import errno
import fcntl
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_MANIFEST = (
    REPO_ROOT / "benchmarks" / "fastllm" / "configs" /
    "thinkingcap_native_profiles.json"
)

TOP_KEYS = {
    "schema_version", "kind", "created_at_utc", "state_dir", "binary",
    "common", "manager", "profiles",
}
COMMON_KEYS = {
    "mmproj", "mmproj_size_bytes", "mmproj_sha256", "threads", "atype",
    "kv_cache_dtype", "batch", "tokens", "default_max_tokens", "device",
}
MANAGER_KEYS = {
    "startup_timeout_seconds", "stop_grace_seconds", "minimum_free_vram_mib",
    "single_gpu", "rollback_on_failure", "proxy_ownership",
}
PROFILE_KEYS = {
    "name", "status", "port", "model_name", "model_path", "model_size_bytes",
    "model_sha256", "artifact_variant", "cutover_status", "production_eligible",
    "availability_reason", "env", "argv", "readiness", "evidence",
}
READINESS_KEYS = {"host", "health_path", "props_path", "expected_model"}
PATH_OPTIONS = {"--path", "--mmproj"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class NativeProfileError(RuntimeError):
    """A user-actionable manifest or lifecycle error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _fail(message: str) -> None:
    raise NativeProfileError(message)


def _exact_keys(value: Any, required: set[str], allowed: set[str], where: str) -> None:
    if not isinstance(value, dict):
        _fail(f"{where} must be an object")
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - allowed)
    if missing:
        _fail(f"{where} missing keys: {', '.join(missing)}")
    if unknown:
        _fail(f"{where} has unknown keys: {', '.join(unknown)}")


def _string(value: Any, where: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        _fail(f"{where} must be a non-empty string")
    return value


def _absolute_path(value: Any, where: str) -> Path:
    path = Path(_string(value, where))
    if not path.is_absolute():
        _fail(f"{where} must be absolute: {path}")
    return path


def _integer(value: Any, where: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{where} must be an integer")
    if minimum is not None and value < minimum:
        _fail(f"{where} must be >= {minimum}")
    return value


def _argv_value(argv: list[str], option: str, where: str) -> str:
    try:
        index = argv.index(option)
    except ValueError:
        _fail(f"{where} missing {option}")
    if index + 1 >= len(argv):
        _fail(f"{where} has no value after {option}")
    return _string(argv[index + 1], f"{where}[{index + 1}]")


def _argv_unique_option(argv: list[str], option: str, where: str) -> str:
    if argv.count(option) != 1:
        _fail(f"{where} must contain exactly one {option}")
    return _argv_value(argv, option, where)


def _validate_profile(profile: dict[str, Any], index: int, manifest: dict[str, Any]) -> None:
    where = f"profiles[{index}]"
    _exact_keys(profile, PROFILE_KEYS, PROFILE_KEYS, where)
    name = _string(profile["name"], f"{where}.name")
    if not NAME_RE.fullmatch(name):
        _fail(f"{where}.name has unsafe format: {name!r}")
    status = _string(profile["status"], f"{where}.status")
    if status not in {"available", "unavailable"}:
        _fail(f"{where}.status must be available or unavailable")
    port = _integer(profile["port"], f"{where}.port", minimum=1)
    if port > 65535:
        _fail(f"{where}.port is outside TCP range")
    model_name = _string(profile["model_name"], f"{where}.model_name")
    model_path = _absolute_path(profile["model_path"], f"{where}.model_path")
    size = profile["model_size_bytes"]
    if size is not None:
        _integer(size, f"{where}.model_size_bytes", minimum=1)
    digest = profile["model_sha256"]
    if digest is not None and (not isinstance(digest, str) or not SHA256_RE.fullmatch(digest)):
        _fail(f"{where}.model_sha256 must be null or a lowercase SHA-256")
    if not isinstance(profile["production_eligible"], bool):
        _fail(f"{where}.production_eligible must be boolean")
    if profile["production_eligible"] and status != "available":
        _fail(f"{where}.production_eligible cannot be true for unavailable profile")
    reason = profile["availability_reason"]
    if status == "unavailable" and not isinstance(reason, str):
        _fail(f"{where}.availability_reason is required for unavailable profile")
    if status == "available" and reason is not None and not isinstance(reason, str):
        _fail(f"{where}.availability_reason must be null or string")
    _string(profile["artifact_variant"], f"{where}.artifact_variant")
    _string(profile["cutover_status"], f"{where}.cutover_status")

    env = profile["env"]
    if not isinstance(env, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in env.items()
    ):
        _fail(f"{where}.env must map strings to strings")
    for key in env:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            _fail(f"{where}.env has unsafe variable name: {key!r}")
    if env.get("FASTLLM_OWNED", "0").strip() not in {"", "0"}:
        _fail(f"{where}.env FASTLLM_OWNED must be 0 for external proxy ownership")

    argv = profile["argv"]
    if not isinstance(argv, list) or not argv or any(
        not isinstance(item, str) or not item for item in argv
    ):
        _fail(f"{where}.argv must be a non-empty string array")
    executable = _absolute_path(argv[0], f"{where}.argv[0]")
    binary = _absolute_path(manifest["binary"], "binary")
    if executable != binary:
        _fail(f"{where}.argv[0] must equal manifest binary")
    for option in PATH_OPTIONS:
        value = _argv_unique_option(argv, option, where)
        if not Path(value).is_absolute():
            _fail(f"{where}.{option} value must be absolute: {value}")
    if Path(_argv_value(argv, "--path", where)) != model_path:
        _fail(f"{where}.argv --path does not match model_path")
    mmproj = _absolute_path(manifest["common"]["mmproj"], "common.mmproj")
    if Path(_argv_value(argv, "--mmproj", where)) != mmproj:
        _fail(f"{where}.argv --mmproj does not match common.mmproj")
    port_arg = _argv_unique_option(argv, "--port", where)
    try:
        if int(port_arg) != port:
            _fail(f"{where}.argv --port does not match profile port")
    except ValueError as exc:
        raise NativeProfileError(f"{where}.argv --port must be an integer") from exc
    if _argv_unique_option(argv, "--model_name", where) != model_name:
        _fail(f"{where}.argv --model_name does not match model_name")

    readiness = profile["readiness"]
    _exact_keys(readiness, READINESS_KEYS, READINESS_KEYS, f"{where}.readiness")
    host = _string(readiness["host"], f"{where}.readiness.host")
    health_path = _string(readiness["health_path"], f"{where}.readiness.health_path")
    props_path = _string(readiness["props_path"], f"{where}.readiness.props_path")
    if host in {"", "0.0.0.0", "::"}:
        _fail(f"{where}.readiness.host must be a loopback/explicit probe host")
    if not health_path.startswith("/") or not props_path.startswith("/"):
        _fail(f"{where}.readiness paths must start with /")
    if _string(readiness["expected_model"], f"{where}.readiness.expected_model") != model_name:
        _fail(f"{where}.readiness.expected_model must equal model_name")
    if not isinstance(profile["evidence"], dict):
        _fail(f"{where}.evidence must be an object")
    # An available artifact must have enough information for a cheap preflight.
    if status == "available":
        if size is None or digest is None:
            _fail(f"{where} available profiles require model size and SHA-256")
    else:
        if profile["production_eligible"]:
            _fail(f"{where} unavailable profiles cannot be production eligible")


def load_manifest(path: str | os.PathLike[str] = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeProfileError(f"cannot read manifest {manifest_path}: {exc}") from exc
    _exact_keys(data, TOP_KEYS, TOP_KEYS, "manifest")
    _string(data["created_at_utc"], "created_at_utc")
    if data["schema_version"] != 1:
        _fail("manifest.schema_version must be 1")
    if data["kind"] != "fastllm-native-profile-manifest":
        _fail("manifest.kind is not fastllm-native-profile-manifest")
    _absolute_path(data["state_dir"], "state_dir")
    _absolute_path(data["binary"], "binary")
    common = data["common"]
    _exact_keys(common, COMMON_KEYS, COMMON_KEYS, "common")
    _absolute_path(common["mmproj"], "common.mmproj")
    _integer(common["mmproj_size_bytes"], "common.mmproj_size_bytes", minimum=1)
    if not isinstance(common["mmproj_sha256"], str) or not SHA256_RE.fullmatch(common["mmproj_sha256"]):
        _fail("common.mmproj_sha256 must be a lowercase SHA-256")
    for key in ("threads", "atype", "kv_cache_dtype", "batch", "tokens", "default_max_tokens", "device"):
        _string(common[key], f"common.{key}")
    manager = data["manager"]
    _exact_keys(manager, MANAGER_KEYS, MANAGER_KEYS, "manager")
    for key in ("startup_timeout_seconds", "stop_grace_seconds", "minimum_free_vram_mib"):
        _integer(manager[key], f"manager.{key}", minimum=1)
    for key in ("single_gpu", "rollback_on_failure"):
        if not isinstance(manager[key], bool):
            _fail(f"manager.{key} must be boolean")
    if manager["proxy_ownership"] != "external_only":
        _fail("manager.proxy_ownership must be external_only")
    profiles = data["profiles"]
    if not isinstance(profiles, list) or not profiles:
        _fail("profiles must be a non-empty array")
    names: set[str] = set()
    ports: set[int] = set()
    for index, profile in enumerate(profiles):
        _validate_profile(profile, index, data)
        if profile["name"] in names:
            _fail(f"duplicate profile name: {profile['name']}")
        if profile["port"] in ports:
            _fail(f"duplicate profile port: {profile['port']}")
        names.add(profile["name"])
        ports.add(profile["port"])
    if len(profiles) != 6:
        _fail(f"manifest must contain six nominal profiles, got {len(profiles)}")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_report(manifest: dict[str, Any], verify_sha256: bool = False) -> dict[str, Any]:
    binary = Path(manifest["binary"])
    common = manifest["common"]
    projector = Path(common["mmproj"])
    projector_exists = projector.is_file()
    projector_size = projector.stat().st_size if projector_exists else None
    projector_item: dict[str, Any] = {
        "path": str(projector),
        "exists": projector_exists,
        "size_bytes": projector_size,
        "expected_size_bytes": common["mmproj_size_bytes"],
        "size_match": projector_exists and projector_size == common["mmproj_size_bytes"],
        "sha256_checked": False,
    }
    report: dict[str, Any] = {
        "valid": True,
        "binary": {
            "path": str(binary),
            "exists": binary.is_file(),
            "executable": os.access(binary, os.X_OK),
        },
        "mmproj": projector_item,
        "profiles": {},
    }
    if not report["binary"]["exists"] or not report["binary"]["executable"]:
        report["valid"] = False
        report["errors"] = ["manifest binary is missing or not executable"]
    if not projector_exists or not projector_item["size_match"]:
        report["valid"] = False
        report.setdefault("errors", []).append("common mmproj is missing or size mismatched")
    if verify_sha256 and projector_exists:
        actual_projector_sha = _sha256(projector)
        projector_item["sha256_checked"] = True
        projector_item["sha256"] = actual_projector_sha
        projector_item["sha256_match"] = actual_projector_sha == common["mmproj_sha256"]
        if not projector_item["sha256_match"]:
            report["valid"] = False
            report.setdefault("errors", []).append("common mmproj SHA-256 mismatch")
    for profile in manifest["profiles"]:
        path = Path(profile["model_path"])
        item: dict[str, Any] = {
            "status": profile["status"],
            "path": str(path),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "expected_size_bytes": profile["model_size_bytes"],
            "sha256_checked": False,
        }
        if profile["status"] == "available":
            item["size_match"] = item["exists"] and item["size_bytes"] == profile["model_size_bytes"]
            if not item["exists"] or not item["size_match"]:
                report["valid"] = False
                report.setdefault("errors", []).append(f"{profile['name']} model artifact missing or size mismatch")
            if verify_sha256 and item["exists"]:
                actual = _sha256(path)
                item["sha256_checked"] = True
                item["sha256"] = actual
                item["sha256_match"] = actual == profile["model_sha256"]
                if not item["sha256_match"]:
                    report["valid"] = False
                    report.setdefault("errors", []).append(f"{profile['name']} model SHA-256 mismatch")
        else:
            item["availability_reason"] = profile["availability_reason"]
            item["accepted_missing"] = not item["exists"]
        report["profiles"][profile["name"]] = item
    return report


def _argv_fingerprint(argv: list[str]) -> str:
    payload = json.dumps(argv, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class NativeProfileManager:
    def __init__(self, manifest_path: str | os.PathLike[str] = DEFAULT_MANIFEST):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.manifest = load_manifest(self.manifest_path)
        self.state_dir = Path(self.manifest["state_dir"])
        self.state_path = self.state_dir / "runtime.json"
        self.lock_path = self.state_dir / "profile.lock"
        self.log_dir = self.state_dir / "logs"
        self._lock_handle: Any = None

    def _ensure_dirs(self) -> None:
        self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.state_dir, 0o700)
        os.chmod(self.log_dir, 0o700)

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        self._ensure_dirs()
        handle = self.lock_path.open("a+", encoding="utf-8")
        os.chmod(self.lock_path, 0o600)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            self._lock_handle = handle
            yield
        finally:
            self._lock_handle = None
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def load_state(self) -> dict[str, Any]:
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"status": "stopped", "active": None}
        return state if isinstance(state, dict) else {"status": "stopped", "active": None}

    def save_state(self, state: dict[str, Any]) -> None:
        self._ensure_dirs()
        fd, temp_name = tempfile.mkstemp(prefix="runtime.", suffix=".tmp", dir=self.state_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_name, 0o600)
            os.replace(temp_name, self.state_path)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    def profiles(self) -> dict[str, dict[str, Any]]:
        return {profile["name"]: profile for profile in self.manifest["profiles"]}

    @staticmethod
    def _proc_start_time(pid: int) -> str | None:
        try:
            raw = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        except (OSError, ValueError):
            return None
        end = raw.rfind(")")
        if end < 0:
            return None
        fields = raw[end + 2 :].split()
        return fields[19] if len(fields) > 19 else None

    @staticmethod
    def _proc_exe(pid: int) -> str | None:
        try:
            return os.path.realpath(f"/proc/{int(pid)}/exe")
        except (OSError, ValueError):
            return None

    @staticmethod
    def _proc_argv(pid: int) -> list[str] | None:
        try:
            raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
        except (OSError, ValueError):
            return None
        if not raw:
            return []
        return [item.decode("utf-8", "replace") for item in raw.rstrip(b"\0").split(b"\0")]

    def process_matches(self, state: dict[str, Any]) -> bool:
        pid = state.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            return False
        if self._proc_start_time(pid) != str(state.get("proc_start_time", "")):
            return False
        if self._proc_exe(pid) != os.path.realpath(str(state.get("exe", ""))):
            return False
        argv = self._proc_argv(pid)
        return argv is not None and _argv_fingerprint(argv) == state.get("argv_sha256")

    def _profile(self, name: str) -> dict[str, Any]:
        try:
            return self.profiles()[name]
        except KeyError as exc:
            raise NativeProfileError(f"unknown profile: {name}") from exc

    @staticmethod
    def _free_vram_mib() -> int:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise NativeProfileError(f"cannot query free VRAM with nvidia-smi: {exc}") from exc
        values = []
        for line in result.stdout.splitlines():
            match = re.search(r"\d+", line)
            if match:
                values.append(int(match.group(0)))
        if not values:
            raise NativeProfileError("nvidia-smi returned no GPU memory values")
        return min(values)

    def _assert_external_proxy(self) -> None:
        if self.manifest["manager"]["proxy_ownership"] != "external_only":
            raise NativeProfileError("native manager requires external_only proxy ownership")
        inherited = os.environ.get("FASTLLM_OWNED", "").strip()
        if inherited not in {"", "0"}:
            raise NativeProfileError(
                "native manager refuses FASTLLM_OWNED != 0; run the proxy in external mode"
            )

    def _assert_port_unoccupied(self, profile: dict[str, Any]) -> None:
        host = self._host_for_probe(profile)
        try:
            with socket.create_connection((host, profile["port"]), timeout=0.25):
                pass
        except OSError as exc:
            if exc.errno == errno.ECONNREFUSED:
                return
            raise NativeProfileError(
                f"pre-start port gate could not prove {host}:{profile['port']} is free: {exc}"
            ) from exc
        raise NativeProfileError(
            f"pre-start port gate failed: {host}:{profile['port']} already accepts connections"
        )

    def _preflight(self, profile: dict[str, Any], verify_sha256: bool = False) -> int:
        self._assert_external_proxy()
        if profile["status"] != "available":
            raise NativeProfileError(
                f"profile {profile['name']} is unavailable: {profile['availability_reason']}"
            )
        report = artifact_report(self.manifest, verify_sha256=verify_sha256)
        if not report["binary"]["exists"] or not report["binary"]["executable"]:
            raise NativeProfileError("artifact preflight failed: binary is missing or not executable")
        projector = report["mmproj"]
        if not projector["exists"] or not projector["size_match"]:
            raise NativeProfileError("artifact preflight failed: common mmproj is missing or size mismatched")
        if verify_sha256 and not projector.get("sha256_match"):
            raise NativeProfileError("artifact preflight failed: common mmproj SHA-256 mismatch")
        item = report["profiles"][profile["name"]]
        if not item.get("exists") or not item.get("size_match"):
            raise NativeProfileError(f"profile {profile['name']} model artifact is not ready")
        if verify_sha256 and not item.get("sha256_match"):
            raise NativeProfileError(f"profile {profile['name']} model SHA-256 mismatch")
        self._assert_port_unoccupied(profile)
        free_vram = self._free_vram_mib()
        required = int(self.manifest["manager"]["minimum_free_vram_mib"])
        if free_vram < required:
            raise NativeProfileError(
                f"pre-start VRAM gate failed: {free_vram} MiB free, {required} MiB required"
            )
        return free_vram

    @staticmethod
    def _host_for_probe(profile: dict[str, Any]) -> str:
        host = profile["readiness"]["host"]
        return "127.0.0.1" if host in {"0.0.0.0", "::"} else host

    @staticmethod
    def _http_json(host: str, port: int, route: str, timeout: float) -> dict[str, Any]:
        url = f"http://{host}:{port}{route}"
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=timeout) as response:
            if response.status != 200:
                raise NativeProfileError(f"readiness HTTP {response.status}: {url}")
            body = json.loads(response.read().decode("utf-8"))
        if not isinstance(body, dict):
            raise NativeProfileError(f"readiness response is not an object: {url}")
        return body

    def _probe_ready(self, profile: dict[str, Any]) -> tuple[bool, str]:
        host = self._host_for_probe(profile)
        port = profile["port"]
        try:
            with socket.create_connection((host, port), timeout=1.0):
                pass
            readiness = profile["readiness"]
            health = self._http_json(host, port, readiness["health_path"], 2.0)
            props = self._http_json(host, port, readiness["props_path"], 2.0)
        except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError, NativeProfileError) as exc:
            return False, str(exc)
        expected = readiness["expected_model"]
        if health.get("ready") is not True or health.get("model") != expected:
            return False, "health.ready/model mismatch"
        if props.get("model") != expected:
            return False, "props.model mismatch"
        if props.get("multimodal_projector_loaded") is not True:
            return False, "props.multimodal_projector_loaded is not true"
        return True, "ready"

    def _identity_state(self, process: subprocess.Popen[Any], profile: dict[str, Any], log_path: Path) -> dict[str, Any]:
        pid = process.pid
        start_time = self._proc_start_time(pid)
        if start_time is None:
            raise NativeProfileError("cannot read child proc_start_time")
        argv = profile["argv"]
        return {
            "status": "starting",
            "active": profile["name"],
            "pid": pid,
            "pgid": pid,
            "proc_start_time": start_time,
            "exe": os.path.realpath(argv[0]),
            "argv": argv,
            "argv_sha256": _argv_fingerprint(argv),
            "port": profile["port"],
            "model_name": profile["model_name"],
            "started_at_utc": utc_now(),
            "log_path": str(log_path),
            "manifest": str(self.manifest_path),
        }

    def _log_path(self, name: str) -> Path:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        return self.log_dir / f"{safe}-{stamp}-{time.time_ns() % 1_000_000_000:09d}.log"

    def _stop_state_locked(self, state: dict[str, Any], grace: float) -> bool:
        if not self.process_matches(state):
            return False
        pid = int(state["pid"])
        try:
            os.killpg(int(state.get("pgid", pid)), signal.SIGTERM)
        except ProcessLookupError:
            return False
        deadline = time.monotonic() + max(0.1, grace)
        while time.monotonic() < deadline:
            if not self.process_matches(state):
                return True
            time.sleep(0.1)
        if self.process_matches(state):
            try:
                os.killpg(int(state.get("pgid", pid)), signal.SIGKILL)
            except ProcessLookupError:
                pass
            for _ in range(30):
                if not self.process_matches(state):
                    return True
                time.sleep(0.1)
        return not self.process_matches(state)

    def _wait_ready_locked(self, process: subprocess.Popen[Any], state: dict[str, Any], profile: dict[str, Any], timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        last_error = "not probed"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise NativeProfileError(f"child exited during startup, code={process.returncode}")
            if not self.process_matches(state):
                raise NativeProfileError("child identity changed during startup")
            ready, detail = self._probe_ready(profile)
            last_error = detail
            if ready:
                if process.poll() is not None or not self.process_matches(state):
                    raise NativeProfileError("child exited or identity changed before readiness confirmation")
                free_vram = self._free_vram_mib()
                required = int(self.manifest["manager"]["minimum_free_vram_mib"])
                if free_vram < required:
                    raise NativeProfileError(
                        f"post-load VRAM gate failed: {free_vram} MiB free, {required} MiB required"
                    )
                return {
                    "ready": True,
                    "checked_at_utc": utc_now(),
                    "detail": detail,
                    "free_vram_mib": free_vram,
                    "minimum_free_vram_mib": required,
                }
            time.sleep(0.25)
        raise NativeProfileError(f"startup readiness timeout after {timeout:g}s: {last_error}")

    def _start_locked(self, profile: dict[str, Any], timeout: float) -> dict[str, Any]:
        preflight_free_vram = self._preflight(profile)
        current = self.load_state()
        if current.get("active") and self.process_matches(current):
            raise NativeProfileError(f"managed profile already running: {current['active']}; use switch")
        log_path = self._log_path(profile["name"])
        self._ensure_dirs()
        log_handle = log_path.open("ab", buffering=0)
        os.chmod(log_path, 0o600)
        env = os.environ.copy()
        env.update(profile["env"])
        env["FASTLLM_OWNED"] = "0"
        env["FASTLLM_NATIVE_PROFILE_NAME"] = profile["name"]
        cwd = Path(profile["argv"][0]).resolve().parent.parent
        try:
            process = subprocess.Popen(
                profile["argv"], stdin=subprocess.DEVNULL, stdout=log_handle,
                stderr=subprocess.STDOUT, env=env, cwd=cwd,
                start_new_session=True, close_fds=True,
            )
        except BaseException:
            log_handle.close()
            raise
        try:
            state = self._identity_state(process, profile, log_path)
        except BaseException:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    pass
            log_handle.close()
            raise
        state["preflight_free_vram_mib"] = preflight_free_vram
        self.save_state(state)
        try:
            readiness = self._wait_ready_locked(process, state, profile, timeout)
        except BaseException as exc:
            self._stop_state_locked(state, min(10.0, float(self.manifest["manager"]["stop_grace_seconds"])))
            state.update({"status": "failed", "active": None, "error": str(exc), "failed_at_utc": utc_now()})
            self.save_state(state)
            log_handle.close()
            raise
        state.update({"status": "running", "readiness": readiness, "ready_at_utc": utc_now()})
        self.save_state(state)
        log_handle.close()
        return state

    def start(self, name: str, timeout: float | None = None) -> dict[str, Any]:
        profile = self._profile(name)
        with self.lock():
            state = self.load_state()
            if state.get("active") and self.process_matches(state):
                if state.get("active") == name:
                    ready, detail = self._probe_ready(profile)
                    if ready:
                        state["readiness"] = {"ready": True, "checked_at_utc": utc_now(), "detail": detail}
                        self.save_state(state)
                        return state
                raise NativeProfileError(f"managed profile already running: {state['active']}; use switch")
            if state.get("active"):
                self.save_state({"status": "stopped", "active": None, "stale_state": state})
            return self._start_locked(
                profile,
                float(timeout if timeout is not None else self.manifest["manager"]["startup_timeout_seconds"]),
            )

    def stop(self, grace: float | None = None) -> dict[str, Any]:
        with self.lock():
            state = self.load_state()
            stopped = self._stop_state_locked(
                state,
                float(grace if grace is not None else self.manifest["manager"]["stop_grace_seconds"]),
            ) if state.get("active") else True
            result = {"status": "stopped", "active": None, "stopped_process": stopped}
            if state.get("active") and not stopped:
                result["error"] = "managed child did not exit or identity check failed"
            self.save_state(result)
            if result.get("error"):
                raise NativeProfileError(result["error"])
            return result

    def switch(self, name: str, timeout: float | None = None, grace: float | None = None) -> dict[str, Any]:
        target = self._profile(name)
        if target["status"] != "available":
            raise NativeProfileError(f"profile {name} is unavailable: {target['availability_reason']}")
        with self.lock():
            old = self.load_state()
            old_name = old.get("active") if self.process_matches(old) else None
            if old_name == name:
                ready, detail = self._probe_ready(target)
                if ready:
                    old["status"] = "running"
                    old["readiness"] = {"ready": True, "checked_at_utc": utc_now(), "detail": detail}
                    self.save_state(old)
                    return old
                raise NativeProfileError(f"profile {name} is running but not ready: {detail}")
            if old_name:
                if not self._stop_state_locked(old, float(grace if grace is not None else self.manifest["manager"]["stop_grace_seconds"])):
                    raise NativeProfileError(f"failed to stop old profile {old_name}")
            self.save_state({"status": "stopped", "active": None, "previous": old_name})
            try:
                return self._start_locked(
                    target,
                    float(timeout if timeout is not None else self.manifest["manager"]["startup_timeout_seconds"]),
                )
            except BaseException as target_error:
                rollback_error: str | None = None
                if old_name and self.manifest["manager"]["rollback_on_failure"]:
                    try:
                        rollback = self._start_locked(
                            self._profile(old_name),
                            float(timeout if timeout is not None else self.manifest["manager"]["startup_timeout_seconds"]),
                        )
                    except BaseException as rollback_exc:
                        rollback_error = str(rollback_exc)
                    else:
                        rollback["rollback_after_failure"] = str(target_error)
                        self.save_state(rollback)
                        raise NativeProfileError(
                            f"switch to {name} failed; rolled back to {old_name}: {target_error}"
                        ) from target_error
                detail = f"switch to {name} failed: {target_error}"
                if rollback_error:
                    detail += f"; rollback failed: {rollback_error}"
                raise NativeProfileError(detail) from target_error

    def status(self) -> dict[str, Any]:
        with self.lock():
            state = self.load_state()
            running = bool(state.get("active") and self.process_matches(state))
            state["running"] = running
            if running:
                profile = self._profile(str(state["active"]))
                ready, detail = self._probe_ready(profile)
                state["ready"] = ready
                state["readiness_detail"] = detail
            elif state.get("status") == "running":
                state["status"] = "stale"
            return state


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="absolute or repo-relative manifest path")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="strictly validate manifest and local artifacts")
    validate.add_argument("--verify-sha256", action="store_true")
    list_parser = sub.add_parser("list", help="list all nominal profiles")
    list_parser.add_argument("--json", action="store_true", dest="as_json")
    command = sub.add_parser("command", help="render one profile command")
    command.add_argument("name")
    command.add_argument("--shell", action="store_true")
    start = sub.add_parser("start", help="start one available profile")
    start.add_argument("name")
    start.add_argument("--timeout", type=float)
    switch = sub.add_parser("switch", help="stop old profile, start target, rollback on failure")
    switch.add_argument("name")
    switch.add_argument("--timeout", type=float)
    switch.add_argument("--grace", type=float)
    stop = sub.add_parser("stop", help="stop the managed native child")
    stop.add_argument("--grace", type=float)
    status = sub.add_parser("status", help="show managed child state")
    status.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            manifest = load_manifest(args.manifest)
            report = artifact_report(manifest, verify_sha256=args.verify_sha256)
            _print(report)
            return 0 if report["valid"] else 1
        manifest = load_manifest(args.manifest)
        if args.command == "list":
            rows = [
                {
                    "name": item["name"],
                    "status": item["status"],
                    "port": item["port"],
                    "cutover_status": item["cutover_status"],
                    "production_eligible": item["production_eligible"],
                }
                for item in manifest["profiles"]
            ]
            if args.as_json:
                _print(rows)
            else:
                for row in rows:
                    print("{name}\t{status}\t{port}\t{cutover_status}".format(**row))
            return 0
        manager = NativeProfileManager(args.manifest)
        if args.command == "command":
            profile = manager._profile(args.name)
            if profile["status"] != "available":
                raise NativeProfileError(
                    f"profile {args.name} is unavailable: {profile['availability_reason']}"
                )
            rendered = {"argv": profile["argv"], "env": profile["env"], "port": profile["port"]}
            if args.shell:
                env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in profile["env"].items())
                print((env_prefix + " " if env_prefix else "") + shlex.join(profile["argv"]))
            else:
                _print(rendered)
            return 0
        if args.command == "start":
            _print(manager.start(args.name, timeout=args.timeout))
            return 0
        if args.command == "switch":
            _print(manager.switch(args.name, timeout=args.timeout, grace=args.grace))
            return 0
        if args.command == "stop":
            _print(manager.stop(grace=args.grace))
            return 0
        if args.command == "status":
            state = manager.status()
            if args.as_json:
                _print(state)
            else:
                print(f"{state.get('status', 'unknown')}\t{state.get('active') or '-'}\t{state.get('pid') or '-'}")
            return 0
        raise NativeProfileError(f"unsupported command: {args.command}")
    except NativeProfileError as exc:
        print(f"fastllm-native-profile: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("fastllm-native-profile: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
