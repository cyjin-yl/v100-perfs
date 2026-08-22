#!/usr/bin/env python3
"""Context-fidelity probe: exact-recall corruption vs context length,
with parallel GPU memory monitoring.

Filler = real OMP agentic session text. Needles = path/uuid/sha records
(the token classes observed to corrupt at 146K). Sweep context sizes in
ascending order so the server prefix cache makes each run incremental.

Usage: context_fidelity_probe.py <base_url> <auth_token> <out.jsonl> [max_size]
"""
import json
import random
import re
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

SESSION_JSONL = "/home/ezra/.omp/agent/sessions/-Documents-Proto-UI/2026-08-16T04-48-31-167Z_01a008e6-61bf-7000-ad86-68aa22f70b6d.jsonl"
SIZES = [32000, 64000, 96000, 128000, 146000, 176000, 208000]
N_NEEDLES = 12

PROJECTS = ["Kestrel-Dashboard", "Nomad-UI", "Vertex-Pipeline", "Helix-Gateway",
            "Quartz-Scheduler", "Drift-Console", "Meridian-API", "Falcon-Metrics",
            "Cinder-Storage", "Atlas-Renderer", "Beacon-Relay", "Ember-Worker"]
SUBDIRS = ["services/api-gateway", "pkg/auth/oauth", "cmd/worker", "internal/cache/redis",
           "web/components/Table", "scripts/deploy", "libs/parser", "tools/migrate",
           "configs/nginx", "docs/architecture", "tests/integration", "build/cmake"]


class GPUMonitor(threading.Thread):
    """Poll nvidia-smi in parallel with requests; keep per-phase samples."""

    def __init__(self, interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._phase = None
        self.samples = {}  # phase -> [(t, used_mib)]

    def set_phase(self, name):
        with self._lock:
            self._phase = name
            self.samples.setdefault(name, [])

    def _read_used(self):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10).stdout
            return int(out.strip().splitlines()[0])
        except Exception:
            return None

    def run(self):
        while not self._stop.is_set():
            used = self._read_used()
            if used is not None:
                with self._lock:
                    if self._phase is not None:
                        self.samples.setdefault(self._phase, []).append((time.time(), used))
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()

    def stats(self, phase):
        with self._lock:
            vals = [u for _, u in self.samples.get(phase, [])]
        if not vals:
            return {}
        return {"gpu_base_mib": vals[0], "gpu_peak_mib": max(vals),
                "gpu_end_mib": vals[-1], "gpu_samples": len(vals)}


def extract_filler(limit_chars):
    """Stream text fragments out of the session JSONL (never load wholesale)."""
    chunks, total = [], 0
    with open(SESSION_JSONL, encoding="utf-8", errors="replace") as f:
        for line in f:
            if total >= limit_chars:
                break
            for m in re.finditer(r'"(?:text|thinking|command|content)":\s*"((?:[^"\\]|\\.){80,4000})"', line):
                try:
                    s = json.loads('"' + m.group(1) + '"')
                except Exception:
                    continue
                if len(s) < 80:
                    continue
                chunks.append(s)
                total += len(s)
                if total >= limit_chars:
                    break
    return chunks


def build_needles(rng):
    needles = []
    for i in range(N_NEEDLES):
        path = f"/home/ezra/Documents/{PROJECTS[i]}/{SUBDIRS[i]}"
        sha = "".join(rng.choice("0123456789abcdef") for _ in range(40))
        ticket = f"T-{uuid.UUID(int=rng.getrandbits(128), version=4)}"
        block = (f'\n<tool_result name="bash" record="R{i+1:02d}">\n'
                 f"$ git -C {path} rev-parse HEAD\n{sha}\n"
                 f"$ cat {path}/.deploy/ticket\n{ticket}\n</tool_result>\n")
        needles.append({"id": f"R{i+1:02d}", "path": path, "sha": sha,
                        "ticket": ticket, "block": block})
    return needles


def assemble(filler, needles, target_chars):
    """Filler with needles spliced at evenly spaced absolute positions."""
    body, fc, ni = [], 0, 0
    interval = max(1, target_chars // (N_NEEDLES + 1))
    next_at = interval
    for chunk in filler:
        while fc + len(chunk) >= next_at and ni < len(needles):
            cut = next_at - fc
            body.append(chunk[:cut])
            body.append(needles[ni]["block"])
            ni += 1
            fc += cut + len(needles[ni - 1]["block"])
            next_at += interval
            chunk = chunk[cut:]
            if fc >= target_chars:
                break
        if fc >= target_chars:
            break
        body.append(chunk)
        fc += len(chunk)
    while ni < len(needles):  # corpus ran short; append remaining needles
        body.append(needles[ni]["block"])
        ni += 1
    return "".join(body)


QUERY = ("\n\n---\nAbove is a session log containing tool_result records R01..R"
         + f"{N_NEEDLES}".rjust(2, "0") + ". For EACH record, output one line exactly in this form:\n"
         "Rxx|<path>|<sha>|<ticket>\n"
         "Copy the path, sha and ticket character-for-character. "
         "No explanation, no extra text.\n")


def chat(base, token, content, max_tokens=5000):
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps({
            "model": "qwen3.8-27b", "stream": False, "temperature": 0,
            "max_tokens": max_tokens, "enable_thinking": False,
            "messages": [{"role": "user", "content": content}],
        }).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + token})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r), time.time() - t0


def score(out_text, needles):
    rows = {}
    for line in out_text.splitlines():
        m = re.match(r"\s*(R\d\d)\|([^|]*)\|([^|]*)\|([^|]*)", line)
        if m:
            rows[m.group(1)] = (m.group(2).strip(), m.group(3).strip(), m.group(4).strip())
    per = []
    for n in needles:
        got = rows.get(n["id"])
        if not got:
            per.append({"id": n["id"], "field": "missing", "ok": False})
            continue
        for fi, key in enumerate(("path", "sha", "ticket")):
            ok = got[fi] == n[key]
            rec = {"id": n["id"], "field": key, "ok": ok}
            if not ok:
                rec.update({"want": n[key], "got": got[fi]})
            per.append(rec)
    bad = [p for p in per if not p["ok"]]
    return len(per) - len(bad), len(per), bad


def wait_ready(base, token, timeout=1500):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            req = urllib.request.Request(base.rstrip("/") + "/health",
                                         headers={"Authorization": "Bearer " + token})
            with urllib.request.urlopen(req, timeout=10) as r:
                h = json.load(r)
            if h.get("ready"):
                return True
        except Exception:
            pass
        time.sleep(10)
    return False


def main():
    base, token, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    max_size = int(sys.argv[4]) if len(sys.argv) > 4 else max(SIZES)
    rng = random.Random(20260816)
    needles = build_needles(rng)
    mon = GPUMonitor(interval=1.0)
    mon.start()
    mon.set_phase("startup")
    print("waiting for backend ready...", flush=True)
    if not wait_ready(base, token):
        print("backend not ready in time", flush=True)
        sys.exit(2)
    print("ready; extracting filler...", flush=True)
    filler = extract_filler(4_000_000)
    print(f"filler chars={sum(len(c) for c in filler)}", flush=True)
    ratio = 3.2  # chars per token, calibrated after first request
    out = open(out_path, "a", encoding="utf-8")
    for size in SIZES:
        if size > max_size:
            break
        phase = f"size_{size}"
        mon.set_phase(phase)
        target_chars = int(size * ratio)
        content = assemble(filler, needles, target_chars) + QUERY
        try:
            resp, dt = chat(base, token, content)
        except Exception as e:
            rec = {"size": size, "error": str(e)[:300]}
            rec.update(mon.stats(phase))
            out.write(json.dumps(rec) + "\n"); out.flush()
            print(rec, flush=True)
            break
        pt = resp.get("usage", {}).get("prompt_tokens", 0)
        if pt > 0:
            ratio = 0.7 * ratio + 0.3 * (len(content) / pt)
        msg = resp["choices"][0]["message"]
        text = msg.get("content") or ""
        rc = msg.get("reasoning_content") or ""
        with open(f"/tmp/probe_raw_{size}.txt", "w", encoding="utf-8") as rf:
            rf.write("=== reasoning (%d chars) ===\n%s\n=== content ===\n%s\n" % (len(rc), rc, text))
        ok, tot, bad = score(text, needles)
        rec = {"size": size, "prompt_tokens": pt, "wall_s": round(dt, 1),
               "ok_fields": ok, "total_fields": tot,
               "bad": bad[:6], "finish": resp["choices"][0].get("finish_reason")}
        rec.update(mon.stats(phase))
        out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
        print(json.dumps(rec, ensure_ascii=False)[:500], flush=True)
    mon.set_phase("done")
    mon.stop()
    out.close()


if __name__ == "__main__":
    main()
