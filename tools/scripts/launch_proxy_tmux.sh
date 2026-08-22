#!/usr/bin/env bash
# Launch the Thinking Proxy in a tmux pane, owning the FastLLM backend.
# Usage: launch_proxy_tmux.sh <profile.env> [session-name] [window-name]
set -euo pipefail

ENV_FILE="${1:?usage: $0 <profile.env> [session] [window]}"
SESSION="${2:-fastllm-prod}"
WINDOW="${3:-proxy-8000}"
TMUX_BIN="/usr/bin/tmux"

set -a
. /run/media/ezra/13D010B6FDBC1A06/1CatVLLM/.env
. "$ENV_FILE"
set +a
unset TMUX

stop_session() {
  if ! "$TMUX_BIN" has-session -t "$SESSION" 2>/dev/null; then
    return 0
  fi
  "$TMUX_BIN" send-keys -t "$SESSION:$WINDOW" C-c 2>/dev/null || true
  for _ in $(seq 1 45); do
    local pane_dead
    pane_dead=$("$TMUX_BIN" list-panes -t "$SESSION:$WINDOW" -F '#{pane_dead}' 2>/dev/null || printf '1')
    [[ "$pane_dead" == "1" ]] && break
    sleep 1
  done
  "$TMUX_BIN" kill-session -t "$SESSION" 2>/dev/null || true
  # An owned FastLLM child that ignores SIGTERM (e.g. still loading) can
  # outlive a hard session kill and keep the GPU busy; sweep it up so the
  # next start does not OOM on a stale instance. Matches both .gguf files
  # and safetensors directories (anything after --path).
  local model_path
  model_path=$(sed -n "s/^FASTLLM_BACKEND_COMMAND=.*--path \([^ ']*\).*/\1/p" "$ENV_FILE" 2>/dev/null | head -1)
  if [[ -n "$model_path" ]]; then
    local leftovers
    leftovers=$(pgrep -f "apiserver --path $model_path" || true)
    if [[ -n "$leftovers" ]]; then
      printf '[launcher] killing orphan FastLLM: %s\n' "$leftovers" >&2
      kill $leftovers 2>/dev/null || true
      sleep 3
      leftovers=$(pgrep -f "apiserver --path $model_path" || true)
      [[ -n "$leftovers" ]] && kill -9 $leftovers 2>/dev/null || true
    fi
  fi
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  stop_session
  exit "$rc"
}

trap cleanup EXIT INT TERM

stop_session


mkdir -p "$(dirname "$PROXY_LOG_FILE")"
: > "$PROXY_LOG_FILE"

PROXY_SHELL="set -o pipefail; set -a; . /run/media/ezra/13D010B6FDBC1A06/1CatVLLM/.env; . $ENV_FILE; set +a; LD_LIBRARY_PATH=/home/ezra/.conda/envs/tsenv/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH} /home/ezra/.conda/envs/tsenv/bin/python /run/media/ezra/13D010B6FDBC1A06/1CatVLLM/thinking_proxy.py 2>&1 | /usr/bin/tee -a $PROXY_LOG_FILE; rc=\${PIPESTATUS[0]}; printf '\\n[tmux] thinking proxy exited rc=%s\\n' \"\$rc\"; exit \"\$rc\""

"$TMUX_BIN" new-session -d -s "$SESSION" -n "$WINDOW" \
  -c /run/media/ezra/13D010B6FDBC1A06/1CatVLLM/v100-perfs \
  /bin/bash -lc "$PROXY_SHELL"
"$TMUX_BIN" set-option -t "$SESSION" history-limit 100000
"$TMUX_BIN" set-option -t "$SESSION" remain-on-exit on

# Backend pane: the owned FastLLM child writes to FASTLLM_BACKEND_LOG, which
# is otherwise invisible in tmux. Follow it in a second pane below the proxy.
BACKEND_LOG_PATH="${FASTLLM_BACKEND_LOG:-}"
if [[ -n "$BACKEND_LOG_PATH" ]]; then
  mkdir -p "$(dirname "$BACKEND_LOG_PATH")"
  : >> "$BACKEND_LOG_PATH"
  "$TMUX_BIN" split-window -t "$SESSION:$WINDOW" -v -d \
    -c /run/media/ezra/13D010B6FDBC1A06/1CatVLLM/v100-perfs \
    /usr/bin/tail -n 40 -F "$BACKEND_LOG_PATH"
fi

"$TMUX_BIN" attach-session -t "$SESSION:$WINDOW"
