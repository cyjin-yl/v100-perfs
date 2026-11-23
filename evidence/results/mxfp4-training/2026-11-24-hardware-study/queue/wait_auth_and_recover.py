#!/usr/bin/env python3
"""Wait for the existing Crater account, then invoke one hash-bound launcher.

No login, token-file read, account change, job cancellation, or automatic retry
of a launcher is performed. A zero launcher exit means only that its own audit
completed; that launch-audit remains the authority for any created job.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time


CLI = "/home/ezra/verdant-dev4-transfer/tools/crater-1.4.1"
OWNER = "jincy26"
PLATFORM = "https://gpu.act.buaa.edu.cn"
ORIGINAL_JOB = "sg-jincy26-261123-e5bb3"
ORIGINAL_CREATED = "2026-11-23T15:57:51Z"
DEADLINE = datetime(2026, 11, 24, 4, 52, 24, tzinfo=timezone.utc)
POLL_SECONDS = 60


class Stop(RuntimeError):
    pass


class WaitingForAuth(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc)


def atomic_json(path, value, *, exclusive=False):
    """Publish a complete, fsynced JSON file; a claim never replaces a file."""
    fd, name = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def error_kind(result):
    """Return only a classification/code; never persist raw authentication data."""
    texts, codes, statuses = [], [], []

    def visit(value):
        if not isinstance(value, dict):
            return
        for key in ("code", "message", "reason", "error", "detail", "details", "context"):
            item = value.get(key)
            if isinstance(item, str):
                texts.append(item)
                if key == "code" and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", item):
                    codes.append(item)
            elif isinstance(item, dict):
                visit(item)
        for key in ("status", "status_code", "http_status"):
            if value.get(key) in (401, "401"):
                statuses.append(401)

    for stream in (result.stderr, result.stdout):
        try:
            visit(json.loads(stream))
        except (ValueError, TypeError):
            # Some CLI versions emit a plain diagnostic on authentication expiry.
            if isinstance(stream, str):
                texts.append(stream)
    joined = " ".join(texts)
    expired = bool(statuses) or any("UNAUTHORIZED_401" in code.upper() or
        ("EXPIRED" in code.upper() and any(word in code.upper() for word in ("TOKEN", "JWT"))) for code in codes)
    expired |= bool(re.search(r"\b(?:token|jwt)\s+(?:(?:has|is)\s+)?expired\b|\b(?:HTTP(?:/\d(?:\.\d)?)?\s+)?401\s+Unauthorized\b", joined, re.I))
    return expired, codes[0] if codes else "unclassified_error"


def execute_launcher(argv, *, cwd, stdout, stderr, timeout):
    """On interruption stop only this newly created local process group."""
    process = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL,
                               stdout=stdout, stderr=stderr, start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except BaseException:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=3)
        raise


class Recovery:
    def __init__(self, args, *, read=subprocess.run, launch=execute_launcher,
                 now=utc_now, monotonic=time.monotonic, sleep=time.sleep):
        self.args, self.read, self.launch = args, read, launch
        self.now, self.monotonic, self.sleep = now, monotonic, sleep
        self.outdir = args.outdir.resolve()
        self.launcher = args.launcher.resolve(strict=True)
        self.state_path = self.outdir / "auth-wait-state.json"
        self.lock_path = self.outdir / "auth-wait-lock.json"
        self.monotonic_deadline = monotonic() + max(0, (DEADLINE - now()).total_seconds())
        self.state = {"schema_version": 1, "pid": os.getpid(), "started_at_utc": now().isoformat(),
                      "deadline_utc": DEADLINE.isoformat(), "owner": OWNER, "platform": PLATFORM,
                      "original_job": ORIGINAL_JOB, "cli": CLI, "poll_seconds": POLL_SECONDS,
                      "launcher": str(self.launcher), "expected_launcher_sha256": args.expected_launcher_sha256,
                      "launcher_attempted": False, "launcher_attempt_count": 0,
                      "launcher_returncode": None, "login_attempted": False, "account_changed": False,
                      "token_read": False, "waiting_for_auth_count": 0}

    def remaining(self):
        return min((DEADLINE - self.now()).total_seconds(), self.monotonic_deadline - self.monotonic())

    def save(self, phase, **fields):
        self.state.update(fields, phase=phase, updated_at_utc=self.now().isoformat())
        atomic_json(self.state_path, self.state)

    def verified_launcher_bytes(self):
        if not self.launcher.is_file() or self.launcher.stat().st_size > 1024 * 1024:
            raise Stop("Launcher must be a regular Python file of at most 1 MiB")
        source = self.launcher.read_bytes()
        if hashlib.sha256(source).hexdigest() != self.args.expected_launcher_sha256:
            raise Stop("Launcher SHA differs from the explicitly reviewed SHA")
        return source

    def api(self, *arguments):
        left = self.remaining()
        if left <= 0:
            raise Stop("Original deadline reached")
        result = self.read([CLI, *arguments, "--json", "--no-interactive"],
                           capture_output=True, text=True, check=False, timeout=min(25, left))
        if result.returncode != 0:
            expired, code = error_kind(result)
            if expired:
                raise WaitingForAuth(code)
            raise Stop(f"Crater read failed: code={code}, exit={result.returncode}")
        if len(result.stdout) > 1024 * 1024:
            raise Stop("Crater read exceeded 1 MiB")
        value = json.loads(result.stdout)
        if not isinstance(value, dict) or not isinstance(value.get("data"), dict):
            raise Stop("Crater read did not return a valid data object")
        return value["data"]

    def authenticated_context(self):
        context = self.api("auth", "ls").get("active_context")
        if (not isinstance(context, dict) or context.get("username") != OWNER
                or not isinstance(context.get("platform_url"), str)
                or context["platform_url"].rstrip("/") != PLATFORM):
            raise Stop("Active Crater account/platform differs from the authorized context")
        job = self.api("job", "get", ORIGINAL_JOB).get("job")
        if not isinstance(job, dict) or any(job.get(key) != value for key, value in (
            ("jobName", ORIGINAL_JOB), ("username", OWNER), ("namespace", "crater-workspace"),
            ("createdAt", ORIGINAL_CREATED), ("status", "Failed"))):
            raise Stop("Original owned job identity or terminal Failed status differs")
        return {"username": OWNER, "platform_url": PLATFORM, "original_job_status": "Failed",
                "original_job_created_at": ORIGINAL_CREATED}

    def run(self):
        self.outdir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.state_path.exists() or self.state_path.is_symlink():
            raise Stop("Independent state already exists; inspect it, do not replay this launcher")
        if self.launcher == self.outdir / "reviewed-launcher.py":
            raise Stop("Launcher input must differ from the retained review snapshot")
        atomic_json(self.lock_path, self.state, exclusive=True)
        self.save("starting")
        try:
            self.verified_launcher_bytes()
            while self.remaining() > 0:
                poll_started = self.monotonic()
                try:
                    identity = self.authenticated_context()
                except WaitingForAuth as error:
                    self.save("waiting_for_auth", reason=str(error),
                              waiting_for_auth_count=self.state["waiting_for_auth_count"] + 1)
                    delay = min(POLL_SECONDS - (self.monotonic() - poll_started), self.remaining())
                    if delay > 0:
                        self.sleep(delay)
                    continue
                source = self.verified_launcher_bytes()
                snapshot_path = self.outdir / "reviewed-launcher.py"
                with snapshot_path.open("xb") as snapshot:
                    os.chmod(snapshot.name, 0o600)
                    snapshot.write(source)
                    snapshot.flush()
                    os.fsync(snapshot.fileno())
                if self.remaining() <= 0:
                    raise Stop("Original deadline reached before launcher invocation")
                # Persist this before spawning: even a crash or an uncertain create
                # cannot be mistaken for permission to repeat the launcher.
                self.save("launching", authenticated_context=identity, launcher_attempted=True,
                          launcher_attempt_count=1, launcher_started_at_utc=self.now().isoformat(),
                          executed_launcher=str(snapshot_path), executed_launcher_sha256=hashlib.sha256(source).hexdigest())
                try:
                    with (self.outdir / "launcher.stdout").open("xb") as out, (self.outdir / "launcher.stderr").open("xb") as err:
                        os.chmod(out.name, 0o600)
                        os.chmod(err.name, 0o600)
                        left = self.remaining()
                        if left <= 0:
                            self.save("deadline_reached_before_spawn", launcher_spawn_attempted=False, automatic_retry=False)
                            return 1
                        code = self.launch([sys.executable, str(snapshot_path)], cwd=self.launcher.parent,
                                           stdout=out, stderr=err, timeout=min(300, left))
                except (OSError, subprocess.TimeoutExpired, KeyboardInterrupt, InterruptedError) as error:
                    self.save("launcher_outcome_unknown", error_type=type(error).__name__,
                              automatic_retry=False, outcome_authority="launcher launch-audit; inspect before any recovery")
                    return 1
                self.save("launcher_returned_zero" if code == 0 else "launcher_nonzero",
                          launcher_returncode=code, automatic_retry=False,
                          outcome_authority="launcher launch-audit; zero is not a GPU-result claim")
                return 0 if code == 0 else 1
            self.save("deadline_reached", automatic_retry=False)
            return 1
        except (Stop, OSError, ValueError, subprocess.TimeoutExpired, KeyboardInterrupt, InterruptedError) as error:
            self.save("failed", error=str(error), automatic_retry=False)
            return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True, help="New independent directory; never reuse an old watcher state directory")
    parser.add_argument("--launcher", type=Path, required=True, help="The reviewed v3 submission script on this host")
    parser.add_argument("--expected-launcher-sha256", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_launcher_sha256):
        parser.error("expected launcher SHA must be 64 lowercase hex characters")
    if not os.access(CLI, os.X_OK):
        parser.error("the previously approved Crater CLI path is unavailable")
    def terminated(signum, frame):
        raise InterruptedError("Auth recovery waiter interrupted")
    signal.signal(signal.SIGTERM, terminated)
    try:
        return Recovery(args).run()
    except (Stop, OSError, ValueError) as error:
        print(json.dumps({"phase": "setup_failed", "error": str(error), "launcher_attempted_this_process": False}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
