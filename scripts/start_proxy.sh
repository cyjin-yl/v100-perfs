#!/usr/bin/env bash
# Launch the Thinking Proxy against whichever local backend is configured.
#
# Default (llama mode): the proxy spawns and manages llama-server itself on
# BACKEND_PORT (default 8001) and serves the public proxy on PROXY_PORT (8000).
#
# FastLLM mode: set FASTLLM_BACKEND_URL for an external endpoint. To let the
# proxy cold-start and evict a FastLLM child, also set FASTLLM_OWNED=1 and an
# absolute FASTLLM_BACKEND_COMMAND. External mode remains the safe default.
#
# Env knobs (all optional):
#   PROXY_PORT          proxy listen port         (default 8000)
#   PROXY_HOST          proxy bind address        (default 0.0.0.0)
#   BACKEND_PORT        llama-server port         (default 8001, llama mode only)
#   AUTH_TOKEN          bearer/x-api-key token    (unset = INSECURE, no auth)
#   VENV                python interpreter        (default ../.venv-1cat/bin/python)
#   LD_LIBRARY_PATH     prepended as-is; default adds lib dirs if present
#   FASTLLM_BACKEND_URL  set to enable FastLLM mode (e.g. http://127.0.0.1:8002)
#   FASTLLM_MODEL_SLUG   backend slug              (default qwen3.6-fastllm)
#   FASTLLM_PUBLIC_ALIASES comma-separated public model aliases
#   FASTLLM_OWNED        1 = proxy owns child lifecycle (default 0)
#   FASTLLM_BACKEND_COMMAND absolute child command (required when owned)
#   FASTLLM_IDLE_TIMEOUT idle seconds before unload (owned default 300)
#   FASTLLM_VRAM_HIGH_WATERMARK used-VRAM eviction ratio (owned default 0.92)
#   FASTLLM_VRAM_RESUME_WATERMARK restart ratio      (owned default 0.85)
#   FASTLLM_PREFIX_CACHE_PERSIST 1 = checkpoint/restore owned child caches
#   FASTLLM_PREFIX_CACHE_PERSIST_KEY stable model/cache-layout identity
#   FASTLLM_PREFIX_CACHE_DISK_DIR atomic generation root directory
#   FASTLLM_PREFIX_CACHE_CHECKPOINT_TIMEOUT checkpoint HTTP timeout (default 300)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# The root thinking_proxy.py is the single canonical fair-routing proxy.
cd "$PROJECT_ROOT"
PROXY_LOG_FILE="${PROXY_LOG_FILE:-${PROJECT_DIR:-$PROJECT_ROOT/v100-perfs}/runtime/fastllm-native-profiles/logs/proxy-8000-live.log}"

VENV="${VENV:-$PROJECT_ROOT/.venv-1cat/bin/python}"

# Optional conda/venv lib path. Only prepend the default if that dir exists; in
# FastLLM mode you may instead run inside the fastllm/conda env via VENV.
CONDA_LIB="${CONDA_LIB:-$HOME/.conda/envs/tsenv/lib}"
if [[ -d "$CONDA_LIB" ]]; then
  export LD_LIBRARY_PATH="$CONDA_LIB:${LD_LIBRARY_PATH:-}"
fi

export AUTH_TOKEN="${AUTH_TOKEN:-}"
export BACKEND_PORT="${BACKEND_PORT:-8001}"
export PROXY_PORT="${PROXY_PORT:-8000}"
export PROXY_HOST="${PROXY_HOST:-0.0.0.0}"

# FastLLM mode is OFF unless FASTLLM_BACKEND_URL is set and non-empty.
export FASTLLM_BACKEND_URL="${FASTLLM_BACKEND_URL:-}"
export FASTLLM_MODEL_SLUG="${FASTLLM_MODEL_SLUG:-qwen3.6-fastllm}"
export FASTLLM_PUBLIC_ALIASES="${FASTLLM_PUBLIC_ALIASES:-qwen3.6-27b-heretic,qwen3.6-27b-awq}"
export FASTLLM_OWNED="${FASTLLM_OWNED:-0}"
export FASTLLM_BACKEND_COMMAND="${FASTLLM_BACKEND_COMMAND:-}"
export FASTLLM_PREFIX_CACHE_PERSIST="${FASTLLM_PREFIX_CACHE_PERSIST:-0}"
export FASTLLM_PREFIX_CACHE_PERSIST_KEY="${FASTLLM_PREFIX_CACHE_PERSIST_KEY:-}"
export FASTLLM_PREFIX_CACHE_DISK_DIR="${FASTLLM_PREFIX_CACHE_DISK_DIR:-}"
export FASTLLM_PREFIX_CACHE_CHECKPOINT_TIMEOUT="${FASTLLM_PREFIX_CACHE_CHECKPOINT_TIMEOUT:-300}"
if [[ "$FASTLLM_OWNED" == "1" ]]; then
  export FASTLLM_IDLE_TIMEOUT="${FASTLLM_IDLE_TIMEOUT:-300}"
  export FASTLLM_VRAM_HIGH_WATERMARK="${FASTLLM_VRAM_HIGH_WATERMARK:-0.92}"
  export FASTLLM_VRAM_RESUME_WATERMARK="${FASTLLM_VRAM_RESUME_WATERMARK:-0.85}"
fi

if [[ -n "${FASTLLM_BACKEND_URL:-}" ]]; then
  if [[ "$FASTLLM_OWNED" == "1" ]]; then
    [[ -n "$FASTLLM_BACKEND_COMMAND" ]] || {
      echo "[start_proxy] FASTLLM_OWNED=1 requires FASTLLM_BACKEND_COMMAND" >&2
      exit 2
    }
    echo "[start_proxy] owned FastLLM cold-start -> ${FASTLLM_BACKEND_URL}"
  else
    echo "[start_proxy] external FastLLM -> ${FASTLLM_BACKEND_URL}"
  fi
else
  echo "[start_proxy] llama mode -> llama-server on port ${BACKEND_PORT}"
fi

if [[ -n "${PROXY_LOG_FILE:-}" ]]; then
  mkdir -p "$(dirname "$PROXY_LOG_FILE")"
  echo "[start_proxy] logging stdout/stderr -> $PROXY_LOG_FILE"
  "$VENV" "$PROJECT_ROOT/thinking_proxy.py" 2>&1 | /usr/bin/tee -a "$PROXY_LOG_FILE"
  exit "${PIPESTATUS[0]}"
fi
exec "$VENV" "$PROJECT_ROOT/thinking_proxy.py"
