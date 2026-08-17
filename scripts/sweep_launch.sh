#!/usr/bin/env bash
# 无头 profile 启动器 —— 给 模型 x KV量化 x MTP 的扫描实验用。
#
# 与 launch_proxy_tmux.sh 的区别(也是它不能用于自动化的原因):
#   - launch_proxy_tmux.sh 结尾是 `tmux attach-session`, 无 TTY 时失败;
#     而它又挂了 `trap cleanup EXIT`, 于是 attach 失败 -> cleanup ->
#     把刚创建的 session 杀掉。非交互环境下必然自毁。
#   - 这里不 attach, 不挂 EXIT trap; 起完就 poll /health 直到 ready 后返回。
#
# Usage: sweep_launch.sh <profile.env> [session] [window]
# Env:   WAIT_READY=900   ready 轮询上限秒数(0=不等)
set -euo pipefail

ENV_FILE="${1:?usage: $0 <profile.env> [session] [window]}"
SESSION="${2:-fastllm-prod}"
WINDOW="${3:-proxy-8000}"
WAIT_READY="${WAIT_READY:-900}"
TMUX_BIN="/usr/bin/tmux"
ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM

set -a
. "$ROOT/.env"
. "$ENV_FILE"
set +a
unset TMUX

PROXY_PORT="${PROXY_PORT:-8000}"

stop_all() {
  if "$TMUX_BIN" has-session -t "$SESSION" 2>/dev/null; then
    "$TMUX_BIN" send-keys -t "$SESSION:$WINDOW" C-c 2>/dev/null || true
    for _ in $(seq 1 30); do
      "$TMUX_BIN" has-session -t "$SESSION" 2>/dev/null || break
      pane_dead=$("$TMUX_BIN" list-panes -t "$SESSION:$WINDOW" \
        -F '#{pane_dead}' 2>/dev/null | head -1 || printf '1')
      [[ "$pane_dead" == "1" ]] && break
      sleep 1
    done
    "$TMUX_BIN" kill-session -t "$SESSION" 2>/dev/null || true
  fi
  # 扫描期每轮换模型, 所以任何存活的 apiserver 都是上一轮的残留:
  # 它会一直占着 20GB+ 显存, 让下一轮 OOM。
  local leftovers
  leftovers=$(pgrep -f "apiserver --path" || true)
  if [[ -n "$leftovers" ]]; then
    printf '[sweep] terminating stale apiserver: %s\n' "$leftovers" >&2
    kill $leftovers 2>/dev/null || true
    for _ in $(seq 1 25); do
      leftovers=$(pgrep -f "apiserver --path" || true)
      [[ -z "$leftovers" ]] && break
      sleep 1
    done
    leftovers=$(pgrep -f "apiserver --path" || true)
    [[ -n "$leftovers" ]] && kill -9 $leftovers 2>/dev/null || true
    sleep 2
  fi
}

stop_all

mkdir -p "$(dirname "$PROXY_LOG_FILE")"
: > "$PROXY_LOG_FILE"
if [[ -n "${FASTLLM_BACKEND_LOG:-}" ]]; then
  mkdir -p "$(dirname "$FASTLLM_BACKEND_LOG")"
  : > "$FASTLLM_BACKEND_LOG"     # 每轮清空, 保证日志里的指标只属于本轮
fi

PROXY_SHELL="set -o pipefail; set -a; . $ROOT/.env; . $ENV_FILE; set +a; \
LD_LIBRARY_PATH=/home/ezra/.conda/envs/tsenv/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH} \
/home/ezra/.conda/envs/tsenv/bin/python $ROOT/thinking_proxy.py 2>&1 \
| /usr/bin/tee -a $PROXY_LOG_FILE; rc=\${PIPESTATUS[0]}; \
printf '\\n[tmux] thinking proxy exited rc=%s\\n' \"\$rc\"; exec sleep 86400"

"$TMUX_BIN" new-session -d -s "$SESSION" -n "$WINDOW" \
  -c "$ROOT/v100-perfs" /bin/bash -lc "$PROXY_SHELL"
"$TMUX_BIN" set-option -t "$SESSION" history-limit 200000 >/dev/null
"$TMUX_BIN" set-option -t "$SESSION" remain-on-exit on >/dev/null

if [[ -n "${FASTLLM_BACKEND_LOG:-}" ]]; then
  "$TMUX_BIN" split-window -t "$SESSION:$WINDOW" -v -d \
    -c "$ROOT/v100-perfs" /usr/bin/tail -n 40 -F "$FASTLLM_BACKEND_LOG"
fi

printf '[sweep] started %s in %s:%s (proxy port %s)\n' \
  "$(basename "$ENV_FILE")" "$SESSION" "$WINDOW" "$PROXY_PORT"

if [[ "$WAIT_READY" == "0" ]]; then
  exit 0
fi

t0=$(date +%s)
while :; do
  body=$(curl -s -m 8 "http://127.0.0.1:$PROXY_PORT/health" || true)
  case "$body" in
    *'"ready":true'*)
      printf '[sweep] READY after %ss\n' "$(( $(date +%s) - t0 ))"
      exit 0;;
  esac
  if (( $(date +%s) - t0 > WAIT_READY )); then
    printf '[sweep] TIMEOUT after %ss; last health: %s\n' \
      "$WAIT_READY" "${body:0:200}" >&2
    exit 1
  fi
  sleep 5
done
