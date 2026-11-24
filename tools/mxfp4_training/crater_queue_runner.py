#!/usr/bin/env python3
"""Wait for one owned Crater job, bind its SSH key, and run a bounded payload.

This runner never submits, changes, stops, or deletes a platform job or service.
It uses an already authenticated Crater CLI and an existing 2222 NodePort. Its
only remote writes are in a newly created directory below /workspace/user.
There is no login, ssh-keyscan, host-key acceptance prompt, or agent forwarding.

The caller supplies a reviewed Python payload. It is invoked as:

    python3 crater_remote_run.py SOURCE_ARCHIVE REMOTE_RUN_DIRECTORY

with REMOTE_RUN_DIRECTORY as cwd. Put retrievable evidence in its ``results/``
directory. Allocation, GPU selection, build/test/benchmark limits, and optional
PyTorch checks belong to that payload, which also creates results.tar.gz.
--ready-json is an INPUT publication gate, written only after the source commit,
archive, and payload are complete. It contains schema_version=1, ready=true,
job_name, source_sha256, remote_script_sha256, source_commit, created_at_utc.
Success here means the payload exited
zero; GPU correctness/performance claims must come from its recorded evidence.
The source archive and payload are snapshotted locally and SHA-256 verified on
upload. A remote GNU timeout bounds the payload; that payload must clean up its
child process groups on termination if it creates separate sessions.
The runner never automatically repeats a payload after an ambiguous SSH failure.

Crater v1.4.1 schemas were checked at commit
8232a995908027a65cab2e7a22851ac091ee1a29:
https://github.com/raids-lab/crater/blob/v1.4.1/cli/cmd/job_logs.go
https://github.com/raids-lab/crater/blob/v1.4.1/cli/cmd/pod.go
https://github.com/raids-lab/crater/blob/v1.4.1/cli/internal/api/node.go
https://github.com/raids-lab/crater/blob/v1.4.1/backend/internal/handler/tool/podcontainer.go

``job logs --json`` has data.logs=[{namespace,pod,container,content}], with
decoded content. ``pod nodeports --json`` has data.nodeports.nodeports=null or
[{name,containerPort,address,nodePort,ServiceName}]. NodePort.address is the
pod's HostIP; node.get.address is only the node's first address and can differ.
The bootstrap must print MXFP4_DEV_USER=<unix-user> and its public ed25519 host
key as a complete ``ssh-ed25519 BASE64 [comment]`` line in that pod's logs.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any


NAME = re.compile(r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?\Z")
UNIX_USER = re.compile(r"[a-z_][a-z0-9_-]{0,31}\$?\Z")
TERMINAL = {"failed", "aborted", "terminated", "deleted", "completed", "succeeded"}


class Fatal(RuntimeError):
    """An identity, schema, or execution failure that must not be retried."""


class NotReady(RuntimeError):
    """An owned job is still queued, booting, or temporarily unreachable."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + secrets.token_hex(4))
    try:
        with temporary.open("x", encoding="utf-8") as target:
            os.chmod(temporary, 0o600)
            json.dump(value, target, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Fatal(f"Malformed {label}: expected an object")
    return value


def validate_job(data: dict[str, Any], job: str, owner: str, namespace: str) -> dict[str, Any]:
    record = mapping(data.get("job"), "data.job")
    # jobName is the platform identifier; name is the editable display name.
    if record.get("jobName") != job or record.get("username") != owner:
        raise Fatal("Job name/owner does not match the explicitly authorized job")
    if record.get("namespace") not in (None, "", namespace):
        raise Fatal("Job namespace changed")
    if not isinstance(record.get("status"), str):
        raise Fatal("Job status is missing")
    if not isinstance(record.get("createdAt"), str) or not record["createdAt"].strip():
        raise Fatal("Job creation time is missing; cannot freeze its identity")
    if record["status"].lower() in TERMINAL:
        raise Fatal(f"Owned job reached terminal status {record['status']}")
    return record


def select_pod(data: dict[str, Any], job: str, namespace: str, requested: str | None) -> dict[str, Any]:
    pods = data.get("pods")
    if not isinstance(pods, list):
        raise Fatal("Malformed data.pods")
    if not pods:
        raise NotReady("Owned job has no pod yet")
    for pod in pods:
        mapping(pod, "pod")
        if pod.get("namespace") != namespace or not isinstance(pod.get("name"), str):
            raise Fatal("Owned job returned a pod outside the expected namespace")
        if not NAME.fullmatch(pod["name"]) or not pod["name"].startswith(job + "-"):
            raise Fatal("Pod name is inconsistent with the authorized job")
    candidates = [pod for pod in pods if requested is None or pod["name"] == requested]
    if not candidates:
        raise NotReady("The explicitly selected pod has not appeared")
    if len(candidates) != 1:
        raise Fatal("Multiple pods returned; select exactly one with --pod")
    pod = candidates[0]
    if str(pod.get("phase", "")).lower() in TERMINAL:
        raise Fatal("Selected pod reached a terminal phase")
    if pod.get("phase") != "Running" or not pod.get("nodename"):
        raise NotReady(f"Selected pod phase is {pod.get('phase', 'unknown')}")
    if not isinstance(pod["nodename"], str) or not NAME.fullmatch(pod["nodename"]):
        raise Fatal("Invalid platform node name")
    return pod


def select_nodeport(data: dict[str, Any], container_port: int) -> dict[str, Any]:
    rules = mapping(data.get("nodeports"), "data.nodeports").get("nodeports")
    if rules is None or rules == []:
        raise NotReady("Owned SSH NodePort has not appeared")
    if not isinstance(rules, list):
        raise Fatal("Malformed data.nodeports.nodeports")
    matches = []
    for rule in rules:
        mapping(rule, "NodePort rule")
        if type(rule.get("containerPort")) is int and rule["containerPort"] == container_port:
            matches.append(rule)
    if not matches:
        raise NotReady(f"No existing NodePort targets container port {container_port}")
    if len(matches) != 1:
        raise Fatal("Ambiguous NodePort rules for the SSH container port")
    rule = matches[0]
    if type(rule.get("nodePort")) is not int or not 1 <= rule["nodePort"] <= 65535:
        raise Fatal("Invalid platform NodePort number")
    if not isinstance(rule.get("address"), str):
        raise Fatal("NodePort HostIP must be a string")
    try:
        address = ipaddress.IPv4Address(rule.get("address", ""))
    except (ipaddress.AddressValueError, TypeError) as error:
        raise Fatal("NodePort must supply a literal IPv4 HostIP for this proxy") from error
    if address.is_loopback or address.is_unspecified or address.is_multicast or address.is_link_local:
        raise Fatal("NodePort HostIP is not a routable node address")
    return {"host": str(address), "port": rule["nodePort"], "container_port": container_port}


def parse_bootstrap(data: dict[str, Any], namespace: str, pod: str, container: str | None) -> dict[str, str]:
    logs = data.get("logs")
    if not isinstance(logs, list) or not logs:
        raise NotReady("Authenticated pod logs are not available")
    identities = []
    for log in logs:
        mapping(log, "job log record")
        if log.get("namespace") != namespace or log.get("pod") != pod:
            raise Fatal("Log record belongs to a different namespace or pod")
        if container is not None and log.get("container") != container:
            raise Fatal("Log record belongs to a different container")
        if not isinstance(log.get("container"), str) or not NAME.fullmatch(log["container"]):
            raise Fatal("Log record has an invalid container name")
        content = log.get("content")
        if not isinstance(content, str):
            raise Fatal("Decoded log content is missing")
        users = set(re.findall(r"^MXFP4_DEV_USER=([^\r\n]+)\r?$", content, re.MULTILINE))
        keys = set(re.findall(r"^ssh-ed25519 ([A-Za-z0-9+/]+={0,2})(?: [^\r\n]*)?\r?$", content, re.MULTILINE))
        if not users and not keys:
            continue
        if len(users) > 1 or len(keys) > 1:
            raise Fatal("Conflicting user or host-key bootstrap markers")
        if not users or not keys:
            raise NotReady("SSH bootstrap markers are incomplete")
        user, key = next(iter(users)), next(iter(keys))
        if not UNIX_USER.fullmatch(user):
            raise Fatal("Invalid SSH bootstrap user")
        try:
            blob = base64.b64decode(key, validate=True)
        except ValueError as error:
            raise Fatal("Invalid ed25519 public host key") from error
        if len(blob) != 51 or blob[:19] != b"\0\0\0\x0bssh-ed25519\0\0\0\x20":
            raise Fatal("Bootstrap key is not a canonical ed25519 SSH key")
        fingerprint = base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")
        identities.append({"user": user, "host_key": "ssh-ed25519 " + key,
                           "host_key_sha256": "SHA256:" + fingerprint, "container": log["container"]})
    if not identities:
        raise NotReady("SSH bootstrap markers have not appeared")
    if len(identities) != 1:
        raise Fatal("Multiple containers supplied bootstrap identities")
    return identities[0]


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.started = time.monotonic()
        self.deadline = self.started + args.max_wait_hours * 3600
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + secrets.token_hex(4)
        self.outdir = args.outdir.resolve()
        self.outdir.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (self.outdir / "runner-lock.json").open("x", encoding="utf-8") as lock:
            json.dump({"pid": os.getpid(), "run_id": self.run_id, "job": args.job}, lock)
        self.state_path = (args.state_json or self.outdir / "state.json").resolve()
        self.ready_path = args.ready_json.resolve()
        self.known_hosts = self.outdir / "known_hosts"
        self.remote_dir = f"/workspace/user/mxfp4-{args.job[:80]}-{self.run_id}"
        self.state = {"schema_version": 1, "started_at": utc_now(), "job": args.job,
                      "owner": args.owner, "namespace": args.namespace, "run_id": self.run_id,
                      "remote_dir": self.remote_dir, "remote_payload_attempted": False,
                      "remote_payload_completed": False, "gpu_executed": None}
        self.ssh_prefix: list[str] = []
        self.job_identity: tuple[Any, ...] | None = None
        self.ready: dict[str, Any] | None = None

    def update(self, phase: str, **values: Any) -> None:
        self.state.update(values, phase=phase, updated_at=utc_now())
        write_json(self.state_path, self.state)
        event = {"at": self.state["updated_at"], "phase": phase, **values}
        with (self.outdir / "events.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(event, sort_keys=True) + "\n")
        print(json.dumps(event, sort_keys=True), flush=True)

    def cli(self, *arguments: str) -> dict[str, Any]:
        command = [str(self.args.cli), *arguments, "--json", "--no-interactive"]
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Bounded Crater queue wait expired")
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=min(30, remaining), check=False)
        except subprocess.TimeoutExpired as error:
            raise NotReady("Crater read timed out") from error
        if result.returncode:
            try:
                detail = json.loads(result.stderr)
            except (json.JSONDecodeError, TypeError):
                detail = {}
            if not isinstance(detail, dict):
                detail = {}
            code = str(detail.get("code", "unknown"))
            if any(word in code for word in ("UNAUTHORIZED", "FORBIDDEN", "AUTH", "TOKEN")) or detail.get("category") == "usage_error":
                raise Fatal(f"Crater read failed ({code}); no login or account change attempted")
            raise NotReady(f"Crater read failed (exit {result.returncode}, code {code})")
        if len(result.stdout) > 16 * 1024 * 1024:
            raise Fatal("Crater response exceeds the bounded diagnostic size")
        try:
            envelope = mapping(json.loads(result.stdout), "Crater JSON envelope")
        except json.JSONDecodeError as error:
            raise Fatal("Crater did not return clean JSON") from error
        return mapping(envelope.get("data"), "Crater data")

    def discover(self) -> dict[str, Any]:
        a = self.args
        job = validate_job(self.cli("job", "get", a.job), a.job, a.owner, a.namespace)
        job_identity = (job["jobName"], job["username"], job.get("createdAt"))
        if self.job_identity is not None and job_identity != self.job_identity:
            raise Fatal("Authorized job identity/creation time changed during the wait")
        self.job_identity = job_identity
        if job["status"].lower() != "running":
            raise NotReady(f"Owned job status is {job['status']}")
        pod = select_pod(self.cli("job", "pods", a.job, "--all-pages"), a.job, a.namespace, a.pod)
        endpoint = select_nodeport(self.cli("pod", "nodeports", pod["name"], "--namespace", a.namespace), a.container_port)
        log_arguments = ["job", "logs", a.job, "--pod", pod["name"], "--tail", "0"]
        if a.container:
            log_arguments += ["--container", a.container]
        identity = parse_bootstrap(self.cli(*log_arguments), a.namespace, pod["name"], a.container)
        return {"job": a.job, "owner": a.owner, "namespace": a.namespace, "job_created_at": job.get("createdAt"),
                "pod": pod["name"], "node": pod["nodename"], **endpoint, **identity}

    def bind_host(self, endpoint: dict[str, Any]) -> None:
        alias = "mxfp4-" + self.args.job + "-" + self.run_id
        expected = f"{alias} {endpoint['host_key']}\n"
        if self.known_hosts.exists():
            if self.known_hosts.read_text(encoding="ascii") != expected:
                raise Fatal("Authenticated host identity changed after it was bound")
        else:
            with self.known_hosts.open("x", encoding="ascii") as target:
                os.chmod(self.known_hosts, 0o600)
                target.write(expected)
        known_hosts_option = str(self.known_hosts).replace("\\", "\\\\").replace('"', '\\"')
        options = ["BatchMode=yes", "StrictHostKeyChecking=yes", "GlobalKnownHostsFile=/dev/null",
                   f'UserKnownHostsFile="{known_hosts_option}"', "UpdateHostKeys=no", "CheckHostIP=no",
                   f"HostKeyAlias={alias}", "HostKeyAlgorithms=ssh-ed25519", "ForwardAgent=no", "ClearAllForwardings=yes",
                   "PasswordAuthentication=no", "KbdInteractiveAuthentication=no", "ConnectTimeout=10",
                   "ConnectionAttempts=1", "ServerAliveInterval=10", "ServerAliveCountMax=3"]
        if self.args.proxy_command:
            options.append("ProxyCommand=" + self.args.proxy_command)
        if self.args.identity_file is not None:
            options += ["IdentitiesOnly=yes", "IdentityAgent=none"]
        self.ssh_prefix = ["ssh", "-T", "-F", "/dev/null"]
        for option in options:
            self.ssh_prefix += ["-o", option]
        if self.args.identity_file is not None:
            self.ssh_prefix += ["-i", str(self.args.identity_file.resolve())]
        self.ssh_prefix += ["-p", str(endpoint["port"]), "-l", endpoint["user"], "--", endpoint["host"]]

    def ssh(self, command: str, **kwargs: Any) -> subprocess.CompletedProcess:
        return subprocess.run([*self.ssh_prefix, command], check=False, **kwargs)

    def read_ready(self) -> dict[str, Any]:
        try:
            if self.ready_path.stat().st_size > 65536:
                raise Fatal("Ready publication exceeds 64 KiB")
            ready = mapping(json.loads(self.ready_path.read_text(encoding="utf-8")), "ready publication")
        except FileNotFoundError as error:
            raise NotReady("Source publication is not ready") from error
        except json.JSONDecodeError as error:
            raise NotReady("Source publication JSON is incomplete; publish it atomically") from error
        if ready.get("ready") is not True:
            raise NotReady("Source publication is not marked ready")
        if type(ready.get("schema_version")) is not int or ready["schema_version"] != 1 or ready.get("job_name") != self.args.job:
            raise Fatal("Ready publication does not identify the authorized job/schema")
        for field, length in (("source_sha256", 64), ("remote_script_sha256", 64), ("source_commit", 40)):
            if not isinstance(ready.get(field), str) or not re.fullmatch(r"[0-9a-f]{" + str(length) + "}", ready[field]):
                raise Fatal(f"Ready publication has invalid {field}")
        try:
            created = datetime.fromisoformat(ready["created_at_utc"].replace("Z", "+00:00"))
            if created.utcoffset() is None or created.utcoffset().total_seconds() != 0:
                raise ValueError("not UTC")
        except (KeyError, AttributeError, TypeError, ValueError) as error:
            raise Fatal("Ready publication must contain an ISO UTC created_at_utc") from error
        if self.ready is not None and self.ready != ready:
            raise Fatal("Ready publication changed after the source was frozen")
        return ready

    def freeze_inputs(self) -> tuple[Path, Path]:
        ready = self.read_ready()
        archive, payload = self.outdir / "source.tar.gz", self.outdir / "crater_remote_run.py"
        for original, snapshot, field in ((self.args.source_archive, archive, "source_sha256"),
                                           (self.args.remote_script, payload, "remote_script_sha256")):
            if not original.is_file() or sha256(original) != ready[field]:
                raise Fatal(f"Published {field} does not match its local input")
            with original.open("rb") as source, snapshot.open("xb") as target:
                os.chmod(snapshot, 0o600)
                shutil.copyfileobj(source, target)
            if sha256(snapshot) != ready[field]:
                raise Fatal(f"Input changed while snapshotting {field}")
        self.ready = ready
        write_json(self.outdir / "source-publication.json", ready)
        self.update("source_ready", source_archive_sha256=ready["source_sha256"],
                    remote_script_sha256=ready["remote_script_sha256"], source_commit=ready["source_commit"])
        return archive, payload

    def ssh_to_files(self, command: str, stdout_path: Path, stderr_path: Path, timeout: int) -> int:
        """Bound wall time and the combined byte count while draining both pipes."""
        limit = self.args.max_artifact_mib * 1024 * 1024
        with stdout_path.open("xb") as out, stderr_path.open("xb") as err:
            process = subprocess.Popen([*self.ssh_prefix, command], stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, start_new_session=True)
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ, out)
            selector.register(process.stderr, selectors.EVENT_READ, err)
            deadline, received = time.monotonic() + timeout, 0
            try:
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired("bounded SSH", timeout)
                    for key, _ in selector.select(min(remaining, 1.0)):
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        received += len(chunk)
                        if received > limit:
                            raise Fatal(f"SSH output exceeded the {self.args.max_artifact_mib} MiB bound")
                        key.data.write(chunk)
                return process.wait(timeout=max(1.0, deadline - time.monotonic()))
            finally:
                selector.close()
                if process.poll() is None:
                    # This kills only this runner's SSH/proxy process group. The
                    # remote workload remains subject to its own GNU timeout.
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                process.stdout.close()
                process.stderr.close()

    def wait_ready(self) -> dict[str, Any]:
        while time.monotonic() < self.deadline:
            poll_started = time.monotonic()
            try:
                endpoint = self.discover()
                self.read_ready()
                # Re-read ownership, membership, route, and key immediately before SSH.
                if self.discover() != endpoint:
                    raise NotReady("Owned endpoint changed while being validated")
                self.bind_host(endpoint)
                remaining = self.deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Bounded Crater queue wait expired")
                result = self.ssh("id -un", capture_output=True, text=True, timeout=min(30, remaining))
                if result.returncode:
                    if any(message in result.stderr for message in ("REMOTE HOST IDENTIFICATION HAS CHANGED", "Host key verification failed", "Permission denied")):
                        raise Fatal("Strict SSH identity/authentication verification failed")
                    raise NotReady(f"Verified endpoint SSH is not ready (exit {result.returncode})")
                if result.stdout.strip() != endpoint["user"]:
                    raise Fatal("Remote authenticated Unix user differs from the bootstrap user")
                ready = {"schema_version": 1, "verified_at": utc_now(), "run_id": self.run_id,
                         "remote_dir": self.remote_dir, "host_key_source": "authenticated_owned_job_logs",
                         "identity_scope": "jobName/username/createdAt plus returned pod/namespace and key; DTOs do not expose pod UID, EndpointSlice, or NodePort protocol",
                         "known_hosts": str(self.known_hosts), **endpoint}
                write_json(self.outdir / "endpoint.json", ready)
                self.update("endpoint_ready", endpoint={key: value for key, value in endpoint.items() if key != "host_key"})
                return endpoint
            except (NotReady, subprocess.TimeoutExpired) as error:
                self.update("waiting", reason=str(error))
            delay = min(self.args.poll_seconds - (time.monotonic() - poll_started), self.deadline - time.monotonic())
            if delay > 0:
                time.sleep(delay)
        raise TimeoutError("Bounded Crater queue wait expired")

    def upload(self, local: Path, remote_name: str) -> None:
        destination = self.remote_dir + "/" + remote_name
        temporary = destination + ".partial"
        command = (f"set -eu; umask 077; cat > {shlex.quote(temporary)}; "
                   f"test \"$(sha256sum {shlex.quote(temporary)} | cut -d' ' -f1)\" = {sha256(local)}; "
                   f"mv -- {shlex.quote(temporary)} {shlex.quote(destination)}")
        with local.open("rb") as source:
            result = self.ssh(command, stdin=source, capture_output=True, timeout=180)
        if result.returncode:
            raise Fatal(f"Verified upload failed for {remote_name} (exit {result.returncode})")

    def revalidate_endpoint(self, expected: dict[str, Any]) -> None:
        try:
            current = self.discover()
        except NotReady as error:
            raise Fatal("Previously verified owned endpoint is no longer ready") from error
        if current != expected:
            raise Fatal("Owned endpoint changed before a remote write or payload submission")

    def run_payload(self, endpoint: dict[str, Any], archive: Path, payload: Path) -> None:
        published = self.read_ready()
        if sha256(archive) != published["source_sha256"] or sha256(payload) != published["remote_script_sha256"]:
            raise Fatal("Frozen input hash differs from the published source")
        self.revalidate_endpoint(endpoint)
        qdir = shlex.quote(self.remote_dir)
        self.update("uploading")
        setup = ("set -eu; umask 077; test -d /workspace/user; "
                 "for tool in bash python3 timeout sha256sum tee stat; do command -v \"$tool\" >/dev/null; done; "
                 f"mkdir -- {qdir}")
        result = self.ssh(setup, capture_output=True, timeout=30)
        if result.returncode:
            raise Fatal(f"Own persistent directory creation failed (exit {result.returncode})")
        self.upload(archive, "source.tar.gz")
        self.upload(payload, "crater_remote_run.py")
        manifest = {"schema_version": 1, "run_id": self.run_id, "job": self.args.job,
                    "owner": self.args.owner, "source_archive_sha256": sha256(archive),
                    "remote_script_sha256": sha256(payload), "source_commit": published["source_commit"],
                    "remote_dir": self.remote_dir,
                    "endpoint": {key: value for key, value in endpoint.items() if key != "host_key"}}
        manifest_path = self.outdir / "run-manifest.json"
        write_json(manifest_path, manifest)
        self.upload(manifest_path, "run-manifest.json")
        self.revalidate_endpoint(endpoint)
        invocation = shlex.join(["python3", "crater_remote_run.py", self.remote_dir + "/source.tar.gz", self.remote_dir])
        completion_marker = "MXFP4_RUNNER_EXIT_" + self.run_id + "="
        command = (f"cd {qdir} && bash -c " + shlex.quote(
            f"set -o pipefail; timeout --signal=TERM --kill-after=15s {self.args.payload_timeout_seconds}s "
            f"{invocation} 2>&1 | tee remote-run.log; statuses=(\"${{PIPESTATUS[@]}}\"); "
            "printf '%s\\n' \"${statuses[0]}\" > runner-payload-exit-code; "
            f"printf '{completion_marker}%s\\n' \"${{statuses[0]}}\"; "
            "if [ \"${statuses[1]}\" -ne 0 ]; then exit 74; fi; exit \"${statuses[0]}\""))
        self.update("running", remote_payload_attempted=True,
                    remote_payload_timeout_seconds=self.args.payload_timeout_seconds)
        try:
            returncode = self.ssh_to_files(command, self.outdir / "remote-run.stdout.log",
                                           self.outdir / "remote-run.stderr.log",
                                           self.args.payload_timeout_seconds + 60)
        except (Fatal, subprocess.TimeoutExpired, KeyboardInterrupt):
            self.update("remote_outcome_unknown", reason="SSH interrupted; remote timeout still bounds the one submitted payload")
            raise
        with (self.outdir / "remote-run.stdout.log").open("rb") as log:
            log.seek(max(0, log.seek(0, 2) - 4096))
            tail = log.read().decode("utf-8", errors="replace")
        matches = re.findall(r"^" + re.escape(completion_marker) + r"(\d{1,3})$", tail, re.MULTILINE)
        payload_code = int(matches[0]) if len(matches) == 1 and int(matches[0]) <= 255 else None
        self.update("collecting", payload_ssh_returncode=returncode, payload_exit_code=payload_code,
                    remote_payload_completed=payload_code is not None)
        artifact = self.outdir / "results.tar.gz"
        limit = self.args.max_artifact_mib * 1024 * 1024
        collect = (f"cd {qdir} && test -f results.tar.gz "
                   f"&& test \"$(stat -c %s results.tar.gz)\" -le {limit} && cat results.tar.gz")
        collected = self.ssh_to_files(collect, artifact, self.outdir / "collect.stderr", 180)
        if collected or artifact.stat().st_size > limit:
            artifact.rename(self.outdir / "results.incomplete.tar.gz")
            raise Fatal(f"Result collection failed (exit {collected}); persistent remote directory retained")
        successful = returncode == 0 and payload_code == 0
        self.update("succeeded" if successful else "payload_failed",
                    result_archive=str(artifact), result_archive_sha256=sha256(artifact),
                    payload_ssh_returncode=returncode)
        if not successful:
            raise Fatal(f"One payload attempt returned SSH exit {returncode}, payload exit {payload_code}; evidence collected, no automatic retry")

    def run(self) -> None:
        self.update("starting")
        version = mapping(self.cli("version").get("version"), "CLI version")
        if str(version.get("product_version", "")).lstrip("v") != "1.4.1":
            raise Fatal("This runner's JSON schema contract requires Crater CLI v1.4.1")
        self.update("waiting", cli_version=version.get("product_version"),
                    cli_commit=version.get("commit_sha"))
        endpoint = self.wait_ready()
        archive, payload = self.freeze_inputs()
        self.run_payload(endpoint, archive, payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--job", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--namespace", default="crater-workspace")
    parser.add_argument("--pod", help="Require this exact pod; otherwise the owned job must have exactly one pod")
    parser.add_argument("--container", help="Select the bootstrap container when the pod has multiple containers")
    parser.add_argument("--container-port", type=int, default=2222)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--remote-script", required=True, type=Path)
    parser.add_argument("--identity-file", type=Path, help="Optional explicit key; otherwise use the existing local SSH agent/default identities")
    parser.add_argument("--proxy-command", required=True, help="Explicit trusted proxy command; supply an empty string for a direct route")
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--ready-json", required=True, type=Path, help="Input source-publication gate (see module contract)")
    parser.add_argument("--state-json", "--destination-state", dest="state_json", type=Path, help="Atomically updated state JSON (default OUTDIR/state.json)")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-wait-hours", type=float, default=12)
    parser.add_argument("--payload-timeout-seconds", type=int, default=3000)
    parser.add_argument("--max-artifact-mib", type=int, default=128)
    args = parser.parse_args(argv)
    for label in ("job", "owner", "namespace", "pod", "container"):
        value = getattr(args, label)
        if value is not None and not NAME.fullmatch(value):
            parser.error(f"invalid --{label}")
    if args.container_port != 2222:
        parser.error("this bounded runner is scoped to the authorized SSH container port 2222")
    if not 0 < args.max_wait_hours <= 12 or not 60 <= args.poll_seconds <= 3600:
        parser.error("wait must be within 12 hours, and poll interval must be 60..3600 seconds")
    if not 1 <= args.payload_timeout_seconds <= 3600 or not 1 <= args.max_artifact_mib <= 1024:
        parser.error("payload timeout must be 1..3600 seconds and artifact cap 1..1024 MiB")
    for label in ("cli", "source_archive", "remote_script", "identity_file", "ready_json"):
        original = getattr(args, label)
        if original is None:
            continue
        value = original.expanduser().resolve()
        if label in ("cli", "identity_file") and not value.is_file():
            parser.error(f"--{label.replace('_', '-')} must name an existing regular file")
        setattr(args, label, value)
    if not os.access(args.cli, os.X_OK):
        parser.error("--cli is not executable")
    if args.identity_file is not None and args.identity_file.stat().st_mode & 0o077:
        parser.error("--identity-file must not be readable/writable by group or others")
    if args.state_json is not None and args.state_json.expanduser().resolve() == args.ready_json:
        parser.error("--state-json must differ from the input --ready-json")
    return args


def main(argv: list[str] | None = None) -> int:
    runner = None
    try:
        runner = Runner(parse_args(argv))
        runner.run()
        return 0
    except KeyboardInterrupt:
        if runner is not None:
            runner.update("interrupted", reason="Operator interrupted the local runner; no job or service was stopped")
        return 130
    except (Fatal, NotReady, TimeoutError, subprocess.TimeoutExpired, OSError) as error:
        if runner is not None:
            runner.update("failed", error=str(error))
        else:
            print(f"Runner setup failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
