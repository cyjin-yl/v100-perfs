#!/usr/bin/env python3
"""Refresh one unfrozen watcher's published inputs; never change its job/state.

Usage: queue_refresh.py CONFIG.json [--idempotent] | --self-test
CONFIG requires pid, starttime (Linux /proc field 22), euid, argv (exact list),
cwd, job, owner, run_id, outdir, state_path, backup_dir, audit_path, old_commit,
new_commit, and files={archive,payload,ready}. Each file entry requires absolute
resolved published/new_source paths and full old_sha256/new_sha256 strings.
The source/payload/ready paths must match the running watcher's CLI arguments.
Use a new backup directory and audit path. Successful audit reuse is opt-in.

All copies are verified/fsynced before SIGSTOP. Original backups are retained
read-only; separate prewritten rollback files permit atomic restoration.
SIGTERM/HUP/INT trigger cleanup; SIGKILL or host failure cannot run finally.
An unverifiable rollback leaves the watcher stopped with recovery_required and
retains all recovery files; the operator must repair before verified SIGCONT.
Unrecoverable rollback/resume errors are explicit failures, never success.
--self-test uses fake process handles/signals and temporary files only.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import select
import secrets
import signal
import stat
import tempfile
import time
from datetime import datetime, timezone

ORDER = ("archive", "payload", "ready")
SNAPSHOTS = ("source.tar.gz", "crater_remote_run.py", "source-publication.json")


class Skip(Exception):
    pass


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def opened(path, uid):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != uid:
        os.close(fd)
        raise RuntimeError("Input is not an owned regular file: " + str(path))
    return os.fdopen(fd, "rb")


def digest(path, uid):
    check = hashlib.sha256()
    with opened(path, uid) as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            check.update(chunk)
    return check.hexdigest()


def read_json(path, uid):
    with opened(path, uid) as source:
        if os.fstat(source.fileno()).st_size > 65536:
            raise RuntimeError("JSON exceeds 64 KiB: " + str(path))
        return json.load(source)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value, claim=False):
    temp = path.with_name(path.name + ".tmp-" + secrets.token_hex(8))
    created = False
    try:
        with temp.open("x", encoding="utf-8") as target:
            created = True
            os.fchmod(target.fileno(), 0o600)
            json.dump(value, target, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        if claim:
            os.link(temp, path)  # Atomically claim a NEW audit; never clobber a concurrent helper.
        else:
            os.replace(temp, path)
        sync_directory(path.parent)
    finally:
        if created:
            temp.unlink(missing_ok=True)


def copied(source, target, expected, uid, mode=0o600):
    check = hashlib.sha256()
    created = False
    try:
        with opened(source, uid) as src, target.open("xb") as dst:
            created = True
            for chunk in iter(lambda: src.read(1024 * 1024), b""):
                dst.write(chunk)
                check.update(chunk)
            dst.flush()
            os.fchmod(dst.fileno(), mode)
            os.fsync(dst.fileno())
        if check.hexdigest() != expected:
            raise RuntimeError("Source hash mismatch: " + str(source))
        sync_directory(target.parent)
        return target
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise


def option(argv, name, default=None):
    found = [argv[i + 1] for i, value in enumerate(argv[:-1]) if value == name]
    found += [value[len(name) + 1:] for value in argv if value.startswith(name + "=")]
    if len(found) > 1:
        raise RuntimeError("Duplicate watcher argument: " + name)
    return found[0] if found else default


def configuration(raw):
    c = dict(raw)
    if type(c["pid"]) is not int or c["pid"] <= 1 or type(c["starttime"]) is not int:
        raise ValueError("Invalid expected process identity")
    if type(c["euid"]) is not int or c["euid"] != os.geteuid():
        raise ValueError("Only a watcher owned by the invoking effective UID may be refreshed")
    if not isinstance(c["argv"], list) or not c["argv"] or not all(isinstance(x, str) for x in c["argv"]):
        raise ValueError("argv must be the complete observed argument list")
    def path(value):
        p = Path(value)
        if not p.is_absolute() or p.resolve() != p:
            raise ValueError("Paths must already be absolute and resolved: " + str(p))
        return p
    for key in ("cwd", "outdir", "state_path", "backup_dir", "audit_path"):
        c[key] = path(c[key])
    c["files"] = {key: dict(c["files"][key]) for key in ORDER}
    for item in c["files"].values():
        for key in ("published", "new_source"):
            item[key] = path(item[key])
        for key in ("old_sha256", "new_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", item[key]):
                raise ValueError("All source hashes must be complete lowercase SHA-256 values")
    for key in ("old_commit", "new_commit"):
        if not re.fullmatch(r"[0-9a-f]{40}", c[key]):
            raise ValueError("Source commits must be full lowercase Git object IDs")
    def arg_path(name, default=None):
        value = option(c["argv"], name, default)
        if value is None or str(value).startswith("~"):
            raise ValueError("Missing/unsupported watcher path argument: " + name)
        p = Path(value)
        return (p if p.is_absolute() else c["cwd"] / p).resolve()
    if option(c["argv"], "--job") != c["job"] or option(c["argv"], "--owner") != c["owner"]:
        raise ValueError("Watcher argv does not identify the configured job/owner")
    if arg_path("--outdir") != c["outdir"]:
        raise ValueError("Watcher outdir mismatch")
    state_arg = option(c["argv"], "--destination-state", str(c["outdir"] / "state.json"))
    if arg_path("--state-json", state_arg) != c["state_path"]:
        raise ValueError("Watcher state path mismatch")
    for key, flag in zip(ORDER, ("--source-archive", "--remote-script", "--ready-json")):
        if arg_path(flag) != c["files"][key]["published"]:
            raise ValueError("Publication path differs from the running watcher: " + flag)
    protected = {c["state_path"], *(c["outdir"] / n for n in (*SNAPSHOTS, "runner-lock.json", "events.jsonl", "endpoint.json", "known_hosts"))}
    targets = [item["published"] for item in c["files"].values()]
    if len(set(targets)) != 3 or set(targets) & protected or c["audit_path"] in protected | set(targets):
        raise ValueError("Publication/audit paths overlap protected watcher files")
    if c["audit_path"] in {item["new_source"] for item in c["files"].values()}:
        raise ValueError("Audit path overlaps a new source")
    if c["audit_path"] == c["backup_dir"] or c["backup_dir"] in c["audit_path"].parents:
        raise ValueError("Audit must not overwrite the immutable backup directory")
    c["config_sha256"] = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return c


class Process:
    def __init__(self, c):
        self.pid, self.fd = c["pid"], os.pidfd_open(c["pid"], 0)
        self.poll = select.poll()
        self.poll.register(self.fd, select.POLLIN)

    def identity(self):
        if self.poll.poll(0):
            raise RuntimeError("Bound process has exited")
        root = Path("/proc") / str(self.pid)
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        uid_line = next(line for line in (root / "status").read_text().splitlines() if line.startswith("Uid:"))
        parts = (root / "cmdline").read_bytes().split(b"\0")
        if parts and parts[-1] == b"":
            parts.pop()
        value = dict(pid=self.pid, starttime=int(fields[19]), euid=int(uid_line.split()[2]),
                     argv=[os.fsdecode(x) for x in parts], cwd=Path(os.readlink(root / "cwd")), state=fields[0])
        if self.poll.poll(0):
            raise RuntimeError("Bound process exited while identity was read")
        return value

    def stop(self):
        signal.pidfd_send_signal(self.fd, signal.SIGSTOP)

    def resume(self):
        signal.pidfd_send_signal(self.fd, signal.SIGCONT)

    def close(self):
        os.close(self.fd)


def verify(process, c, initial=False):
    identity = process.identity()
    if any(identity[key] != c[key] for key in ("pid", "starttime", "euid", "argv", "cwd")):
        raise RuntimeError("Bound process identity differs from the expected watcher")
    if initial and identity["state"] in ("T", "t"):
        raise RuntimeError("Watcher was already stopped; refusing to resume another stop")
    return identity["state"]


def waiting(c):
    state = read_json(c["state_path"], c["euid"])
    lock = read_json(c["outdir"] / "runner-lock.json", c["euid"])
    if any(state.get(key) != c[key] for key in ("job", "owner", "run_id")):
        raise RuntimeError("Watcher state identity mismatch")
    if any(lock.get(key) != c[key] for key in ("pid", "job", "run_id")):
        raise RuntimeError("Watcher lock identity mismatch")
    if (state.get("phase") != "waiting" or state.get("remote_payload_attempted") is not False
            or any(os.path.lexists(c["outdir"] / name) for name in SNAPSHOTS)):
        raise Skip("Watcher phase changed or inputs are already frozen")


def hashes(c, which):
    return {key: item[which + "_sha256"] for key, item in c["files"].items()}


def check_published(c, which):
    if any(digest(item["published"], c["euid"]) != item[which + "_sha256"] for item in c["files"].values()):
        raise RuntimeError("Published triplet differs from expected " + which + " hashes")


def check_ready(path, c, which):
    ready = read_json(path, c["euid"])
    expected = dict(schema_version=1, ready=True, job_name=c["job"], source_commit=c[which + "_commit"],
                    source_sha256=c["files"]["archive"][which + "_sha256"],
                    remote_script_sha256=c["files"]["payload"][which + "_sha256"])
    created = datetime.fromisoformat(ready["created_at_utc"].replace("Z", "+00:00"))
    if (type(ready.get("schema_version")) is not int or ready.get("ready") is not True
            or any(ready.get(key) != value for key, value in expected.items())
            or created.utcoffset() is None or created.utcoffset().total_seconds() != 0):
        raise RuntimeError("Invalid or inconsistent " + which + " publication gate")


def refresh(raw, idempotent=False, process_factory=Process, replace=os.replace):
    c = configuration(raw)
    if os.path.lexists(c["audit_path"]):
        previous = read_json(c["audit_path"], c["euid"])
        if not (idempotent and previous.get("config_sha256") == c["config_sha256"]
                and previous.get("status") == "refreshed" and previous.get("resumed") is True
                and previous.get("committed") is True):
            raise RuntimeError("Audit already exists; refusing to overwrite or repeat this operation")
        check_published(c, "new")
        return dict(previous, status="already_refreshed")
    audit = {key: c[key] for key in ("pid", "starttime", "euid", "job", "owner", "run_id", "config_sha256", "old_commit", "new_commit")}
    audit.update(status="preparing", started_at=now(), old_hashes=hashes(c, "old"), new_hashes=hashes(c, "new"),
                 published_paths={key: str(item["published"]) for key, item in c["files"].items()},
                 cwd=str(c["cwd"]), argv_sha256=hashlib.sha256(json.dumps(c["argv"]).encode()).hexdigest(),
                 backup_dir=str(c["backup_dir"]), stop_attempted=False, stop_sent=False,
                 resumed=False, committed=False, gpu_executed=False)
    process, lockfile, publishing = None, None, False
    staged, rollback, leftovers = {}, {}, []
    atomic_json(c["audit_path"], audit, claim=True)
    try:
        process = process_factory(c)  # pidfd opens BEFORE any process identity read.
        # Serialize every helper for this watcher, even with different audit
        # paths. The existing lock file is opened READ-ONLY; contents never change.
        lockfile = opened(c["outdir"] / "runner-lock.json", c["euid"])
        fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(process, c, initial=True)
        waiting(c)
        c["backup_dir"].mkdir(mode=0o700)
        sync_directory(c["backup_dir"].parent)
        nonce = secrets.token_hex(8)
        for key in ORDER:
            item = c["files"][key]
            backup = copied(item["published"], c["backup_dir"] / key, item["old_sha256"], c["euid"], 0o400)
            for collection, source, expected, suffix in ((rollback, backup, item["old_sha256"], "old"),
                                                        (staged, item["new_source"], item["new_sha256"], "new")):
                target = item["published"].with_name("." + item["published"].name + ".refresh-" + nonce + "-" + suffix)
                collection[key] = copied(source, target, expected, c["euid"])
                leftovers.append(target)
        check_ready(c["backup_dir"] / "ready", c, "old")
        check_ready(staged["ready"], c, "new")
        os.chmod(c["backup_dir"], 0o500)
        sync_directory(c["backup_dir"])
        audit.update(status="prepared", prepared_at=now(),
                     backup_paths={key: str(c["backup_dir"] / key) for key in ORDER},
                     staged_paths={key: str(path) for key, path in staged.items()},
                     rollback_paths={key: str(path) for key, path in rollback.items()},
                     new_source_paths={key: str(item["new_source"]) for key, item in c["files"].items()})
        atomic_json(c["audit_path"], audit)
        verify(process, c, initial=True)
        # Arm cleanup BEFORE the syscall: Python may deliver an interruption
        # after the signal was sent but before the following assignment.
        audit["stop_attempted"] = True
        process.stop()
        audit["stop_sent"] = True
        deadline = time.monotonic() + 3.0
        while verify(process, c) != "T":
            if time.monotonic() >= deadline:
                raise TimeoutError("Watcher did not confirm SIGSTOP within 3 seconds")
            time.sleep(0.01)
        audit.update(status="stopped", stopped_at=now())
        atomic_json(c["audit_path"], audit)
        waiting(c)
        check_published(c, "old")
        publishing = True
        for key in ORDER:  # The publication gate is replaced LAST.
            replace(staged[key], c["files"][key]["published"])
            sync_directory(c["files"][key]["published"].parent)
        check_published(c, "new")
        audit.update(status="refreshed", committed=True, committed_at=now())
    except Skip as error:
        audit.update(status="skipped", reason=str(error))
    except BaseException as error:
        audit.update(status="error", error=type(error).__name__ + ": " + str(error))
        if publishing and not audit["committed"]:
            failures = []
            for key in ORDER:
                item = c["files"][key]
                try:
                    try:
                        unchanged = digest(item["published"], c["euid"]) == item["old_sha256"]
                    except OSError:
                        unchanged = False
                    if not unchanged:
                        replace(rollback[key], item["published"])
                        sync_directory(item["published"].parent)
                except BaseException as failure:
                    failures.append(type(failure).__name__ + ": " + str(failure))
            try:
                check_published(c, "old")
                audit["rolled_back"] = True
            except BaseException as failure:
                failures.append(str(failure))
            if failures:
                audit["rollback_errors"] = failures
    finally:
        safe_to_resume = not publishing or audit["committed"] or audit.get("rolled_back") is True
        if audit["stop_attempted"] and not safe_to_resume:
            audit.update(status="recovery_required", recovery_required=True,
                         resume_skipped="Published triplet could not be verified; repair before SIGCONT")
        elif audit["stop_attempted"]:
            try:
                process.resume()  # Always the SAME pidfd; never numeric kill().
                audit.update(resumed=True, resumed_at=now())
            except BaseException as error:
                audit.update(status="error", resume_error=type(error).__name__ + ": " + str(error))
        if process is not None:
            try:
                process.close()
            except BaseException as error:
                audit.update(status="error", close_error=str(error))
        if lockfile is not None:
            lockfile.close()
        if not audit.get("rollback_errors") and not audit.get("recovery_required"):
            for path in leftovers:
                try:
                    path.unlink(missing_ok=True)
                except OSError as error:
                    audit.setdefault("cleanup_errors", []).append(str(error))
        audit["finished_at"] = now()
        try:
            atomic_json(c["audit_path"], audit)
        except BaseException as error:
            audit.update(status="error", audit_error=str(error))
    return audit


def self_test():
    checks = 0
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        for scenario in ("success", "bad_identity", "already_stopped", "lock_busy", "phase_changed", "snapshot", "stop_interrupt",
                         "no_stop_ack", "fail_first", "fail_second", "fail_ready", "fail_after_ready", "rollback_failed"):
            base = root / scenario
            base.mkdir()
            out = base / "watch"
            out.mkdir()
            cfg = dict(pid=12345, starttime=67890, euid=os.geteuid(), cwd=str(base), job="test-job", owner="test-owner",
                       run_id="test-run", outdir=str(out), state_path=str(out / "state.json"), backup_dir=str(base / "backup"),
                       audit_path=str(base / "audit.json"), old_commit="1" * 40, new_commit="2" * 40, files={})
            for key in ORDER:
                old, new = base / (key + ".old"), base / (key + ".new")
                old.write_bytes(("old-" + key).encode())
                new.write_bytes(("new-" + key).encode())
                cfg["files"][key] = dict(published=str(old), new_source=str(new), old_sha256=digest(old, cfg["euid"]), new_sha256=digest(new, cfg["euid"]))
            for which, path_key in (("old", "published"), ("new", "new_source")):
                gate = dict(schema_version=1, ready=True, job_name=cfg["job"], created_at_utc=now(), source_commit=cfg[which + "_commit"],
                            source_sha256=cfg["files"]["archive"][which + "_sha256"], remote_script_sha256=cfg["files"]["payload"][which + "_sha256"])
                path = Path(cfg["files"]["ready"][path_key])
                path.write_text(json.dumps(gate))
                cfg["files"]["ready"][which + "_sha256"] = digest(path, cfg["euid"])
            cfg["argv"] = ["python3", str(base / "crater_queue_runner.py"), "--job", cfg["job"], "--owner", cfg["owner"], "--outdir", str(out)]
            for key, flag in zip(ORDER, ("--source-archive", "--remote-script", "--ready-json")):
                cfg["argv"] += [flag, cfg["files"][key]["published"]]
            state = dict(job=cfg["job"], owner=cfg["owner"], run_id=cfg["run_id"], phase="waiting", remote_payload_attempted=False)
            atomic_json(out / "state.json", state)
            atomic_json(out / "runner-lock.json", {key: cfg[key] for key in ("pid", "job", "run_id")})
            calls = []
            class FakeProcess:
                def __init__(self, c):
                    calls.append("open")
                    self.c, self.state = c, "T" if scenario == "already_stopped" else "S"
                def identity(self):
                    value = {key: self.c[key] for key in ("pid", "starttime", "euid", "argv", "cwd")}
                    value["state"] = self.state
                    if scenario == "bad_identity":
                        value["starttime"] += 1
                    return value
                def stop(self):
                    calls.append("STOP")
                    self.state = "S" if scenario == "no_stop_ack" else "T"
                    if scenario == "stop_interrupt":
                        raise InterruptedError("injected interruption after successful STOP")
                    if scenario == "phase_changed":
                        atomic_json(out / "state.json", dict(state, phase="endpoint_ready"))
                    if scenario == "snapshot":
                        (out / "source.tar.gz").write_bytes(b"frozen")
                def resume(self):
                    calls.append("CONT")
                    check_published(configuration(cfg), "new" if scenario == "success" else "old")
                    self.state = "S"
                def close(self):
                    calls.append("close")
            attempt = [0]
            fail_at = {"fail_first": 1, "fail_second": 2, "fail_ready": 3, "fail_after_ready": 3, "rollback_failed": 2}.get(scenario)
            def replace(src, dst):
                attempt[0] += 1
                if attempt[0] == fail_at or (scenario == "rollback_failed" and attempt[0] == 3):
                    if scenario == "fail_after_ready":
                        os.replace(src, dst)
                    raise OSError("injected one-time publication failure")
                os.replace(src, dst)
            busy_lock = opened(out / "runner-lock.json", cfg["euid"]) if scenario == "lock_busy" else None
            try:
                if busy_lock is not None:
                    fcntl.flock(busy_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = refresh(cfg, process_factory=FakeProcess, replace=replace)
            finally:
                if busy_lock is not None:
                    busy_lock.close()
            expected = ("refreshed" if scenario == "success" else "recovery_required" if scenario == "rollback_failed"
                        else "skipped" if scenario in ("phase_changed", "snapshot") else "error")
            assert result["status"] == expected, (scenario, result)
            if scenario == "rollback_failed":
                assert result["recovery_required"] and result["rollback_errors"]
                assert digest(cfg["files"]["archive"]["published"], cfg["euid"]) == cfg["files"]["archive"]["new_sha256"]
                for key in ORDER:
                    assert Path(result["rollback_paths"][key]).is_file()
                    assert digest(result["backup_paths"][key], cfg["euid"]) == cfg["files"][key]["old_sha256"]
            else:
                check_published(configuration(cfg), "new" if scenario == "success" else "old")
            stops = 0 if scenario in ("bad_identity", "already_stopped", "lock_busy") else 1
            assert calls.count("STOP") == stops, (scenario, calls)
            assert calls.count("CONT") == (0 if scenario == "rollback_failed" else stops), (scenario, calls)
            assert result["resumed"] == ("CONT" in calls), (scenario, result)
            if scenario.startswith("fail_"):
                assert result["rolled_back"] is True
            if scenario == "success":
                before = list(calls)
                assert refresh(cfg, True, process_factory=FakeProcess)["status"] == "already_refreshed"
                assert calls == before
            checks += 1
        claimed = root / "exclusive-audit.json"
        atomic_json(claimed, {"preserve": True}, claim=True)
        try:
            atomic_json(claimed, {"overwrite": True}, claim=True)
            raise AssertionError("Concurrent audit claim was not rejected")
        except FileExistsError:
            assert read_json(claimed, os.geteuid()) == {"preserve": True}
        checks += 1
    return dict(passed=True, scenarios=checks, real_signals_sent=False, gpu_executed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path)
    parser.add_argument("--idempotent", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True))
        return 0
    if args.config is None:
        parser.error("CONFIG.json or --self-test is required")
    interrupted = [False]
    def interrupt(signum, frame):
        if not interrupted[0]:
            interrupted[0] = True
            raise InterruptedError("Operator signal " + str(signum))
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupt)
    try:
        result = refresh(read_json(args.config, os.geteuid()), args.idempotent)
    except BaseException as error:
        result = dict(status="error", error=type(error).__name__ + ": " + str(error), gpu_executed=False)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in ("refreshed", "already_refreshed", "skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
