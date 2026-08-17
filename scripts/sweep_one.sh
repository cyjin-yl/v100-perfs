#!/usr/bin/env bash
# 跑完一个 sweep 档位: 起后端 → 冷启计时 → 核心验收套件 → 采显存峰值 → 汇总一行结果。
#
# 一块 V100 只能串行, 所以这个脚本是矩阵的执行单元; 上层循环调它。
#
# Usage: sweep_one.sh <profile.env>
# Env:   CORE=1 只跑核心套件(默认); FULL=1 additionally 跑 262K 与并发加压
set -uo pipefail

PROFILE="${1:?usage: $0 <profile.env>}"
TAG="$(basename "$PROFILE" .env)"
ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM
PERF="$ROOT/v100-perfs"
OUTDIR=/run/media/ezra/13D010B6FDBC1A06/projects/EzraVastLLM/reports
LOGDIR=/run/media/ezra/13D010B6FDBC1A06/projects/EzraVastLLM/logs
PY=/home/ezra/.conda/envs/tsenv/bin/python
BACKEND_LOG="$PERF/runtime/fastllm-native-profiles/logs/backend-$TAG.log"
mkdir -p "$OUTDIR" "$LOGDIR"
RUNLOG="$LOGDIR/$TAG.log"

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$RUNLOG"; }

say "=== $TAG 开始 ==="

# --- 冷启 ---
t0=$(date +%s)
WAIT_READY=1800 bash "$PERF/scripts/sweep_launch.sh" "$PROFILE" fastllm-prod proxy-8000 \
  >>"$RUNLOG" 2>&1
rc=$?
cold=$(( $(date +%s) - t0 ))
if [[ $rc -ne 0 ]]; then
  say "启动失败(rc=$rc), 跳过该档"
  echo "{\"tag\":\"$TAG\",\"status\":\"launch_failed\",\"cold_start_s\":$cold}" \
    > "$OUTDIR/$TAG.json"
  exit 1
fi
say "冷启 ${cold}s"

# --- 显存采样(后台) ---
VRAMLOG="$LOGDIR/$TAG.vram"
: > "$VRAMLOG"
( while :; do
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
      | awk -F', ' '{gsub(/ MiB/,"",$2); s+=$2} END {print s+0}' >> "$VRAMLOG"
    sleep 5
  done ) &
VRAMPID=$!
trap 'kill $VRAMPID 2>/dev/null' EXIT

cd "$PERF"
set -a; . "$ROOT/.env"; set +a

run_suite() {
  local name="$1"; shift
  say "-- suite $name"
  timeout 3600 $PY scripts/chain_acceptance.py "$@" >>"$RUNLOG" 2>&1
  say "-- suite $name rc=$?"
}

run_suite garble  --suite garble --repeat 3 \
  --json "$OUTDIR/$TAG.garble.json" --dump-dir "$OUTDIR/$TAG.dumps"
run_suite matrix  --suite matrix --repeat 2 \
  --json "$OUTDIR/$TAG.matrix.json" --dump-dir "$OUTDIR/$TAG.dumps"
run_suite toolloop --suite toolloop --rounds 4 \
  --json "$OUTDIR/$TAG.toolloop.json"
run_suite bench   --suite bench --bench-sizes 200,8000,32000,131072 \
  --log-path "$BACKEND_LOG" --json "$OUTDIR/$TAG.bench.json"
run_suite conc    --suite concurrency --concurrency 6 --rounds 2 \
  --json "$OUTDIR/$TAG.conc.json"

if [[ "${FULL:-0}" == "1" ]]; then
  run_suite longctx --suite longctx --ctx-tokens 250000 \
    --json "$OUTDIR/$TAG.longctx.json"
  run_suite conc10 --suite concurrency --concurrency 10 --rounds 2 \
    --json "$OUTDIR/$TAG.conc10.json"
fi

kill $VRAMPID 2>/dev/null

# --- 汇总 ---
$PY - "$TAG" "$OUTDIR" "$BACKEND_LOG" "$VRAMLOG" "$cold" "$PROFILE" <<'PY' \
  > "$OUTDIR/$TAG.json"
import json, os, re, sys
tag, outdir, backend_log, vramlog, cold, profile = sys.argv[1:7]

def load(suffix):
    p = os.path.join(outdir, f"{tag}.{suffix}.json")
    try:
        return json.load(open(p))
    except Exception:
        return None

def fails(rep, suite):
    if not rep:
        return None, None
    rows = (rep.get("suites", {}).get(suite, {}) or {}).get("rows", [])
    if suite in ("toolloop", "concurrency"):
        bad = [r for r in rows if r.get("failures")]
    else:
        bad = [r for r in rows if not r.get("ok")]
    return len(bad), len(rows)

blob = ""
if os.path.exists(backend_log):
    blob = open(backend_log, errors="replace").read()
acc = re.findall(r"pos_accept_rate=\[([\d.]+)%, ([\d.]+)%\]", blob)
mtp_enabled = "[Qwen3.5 MTP] enabled:" in blob
mtp_off_lines = blob.count("[Qwen3.5 MTP] not enabled")
hits = re.findall(r"prefix-cache HIT\(([a-z-]+)\): (\d+)/(\d+) tok", blob)
hit_ratio = None
if hits:
    got = sum(int(h[1]) for h in hits)
    tot = sum(int(h[2]) for h in hits)
    hit_ratio = round(got / tot, 3) if tot else None
peak = 0
if os.path.exists(vramlog):
    vals = [int(x) for x in open(vramlog).read().split() if x.isdigit()]
    peak = max(vals) if vals else 0

mat_f, mat_n = fails(load("matrix"), "matrix")
gar_f, gar_n = fails(load("garble"), "garble")
tl_f, tl_n = fails(load("toolloop"), "toolloop")
cc_f, cc_n = fails(load("concurrency"), "concurrency")
bench = load("bench")
bench_rows = (bench or {}).get("suites", {}).get("bench", {}).get("rows", [])

print(json.dumps({
    "tag": tag, "profile": os.path.basename(profile), "status": "ok",
    "cold_start_s": int(cold), "vram_peak_mib": peak,
    "mtp_enabled": mtp_enabled, "mtp_not_enabled_lines": mtp_off_lines,
    "mtp_accept_rate_last": acc[-1] if acc else None,
    "prefix_hit_ratio": hit_ratio,
    "matrix_fail": mat_f, "matrix_total": mat_n,
    "garble_fail": gar_f, "garble_total": gar_n,
    "toolloop_fail": tl_f, "toolloop_total": tl_n,
    "conc_fail": cc_f, "conc_total": cc_n,
    "bench": bench_rows,
}, ensure_ascii=False, indent=2))
PY

say "=== $TAG 完成, 摘要写入 $OUTDIR/$TAG.json ==="
grep -E '"(cold_start_s|vram_peak_mib|mtp_enabled|mtp_accept_rate_last|prefix_hit_ratio|matrix_fail|garble_fail|toolloop_fail|conc_fail)"' \
  "$OUTDIR/$TAG.json" | tee -a "$RUNLOG"
