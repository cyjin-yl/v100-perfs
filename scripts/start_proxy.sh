#!/usr/bin/env bash
# Launch the Thinking Proxy against whichever local backend is configured.
#
# Default (llama mode): the proxy spawns and manages llama-server itself on
# BACKEND_PORT (default 8001) and serves the public proxy on PROXY_PORT (8000).
#
# FastLLM mode: set FASTLLM_BACKEND_URL (e.g. http://127.0.0.1:8002) to point the
# proxy at an EXTERNAL FastLLM OpenAI endpoint that you manage out-of-band (e.g.
# in a tmux session). The proxy will NOT spawn llama-server; it only health-checks
# the FastLLM endpoint and rewrites public model aliases to FASTLLM_MODEL_SLUG.
#
# Env knobs (all optional):
#   PROXY_PORT          proxy listen port         (default 8000)
#   PROXY_HOST          proxy bind address        (default 0.0.0.0)
#   BACKEND_PORT        llama-server port         (default 8001, llama mode only)
#   AUTH_TOKEN          bearer/x-api-key token    (unset = INSECURE, no auth)
#   VENV                python interpreter        (default ./.venv-1cat/bin/python)
#   LD_LIBRARY_PATH     prepended as-is; default adds lib dirs if present
#   FASTLLM_BACKEND_URL  set to enable FastLLM mode (e.g. http://127.0.0.1:8002)
#   FASTLLM_MODEL_SLUG   backend slug              (default qwen3.6-fastllm)
#   FASTLLM_PUBLIC_ALIASES comma-separated public  (default qwen3.6-27b-heretic,qwen3.6-27b-awq)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Proxy lives in the repo root (parent of scripts/).
cd "$SCRIPT_DIR/.."

VENV="${VENV:-.venv-1cat/bin/python}"

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

if [[ -n "${FASTLLM_BACKEND_URL:-}" ]]; then
  echo "[start_proxy] FastLLM mode -> ${FASTLLM_BACKEND_URL} (slug ${FASTLLM_MODEL_SLUG})"
else
  echo "[start_proxy] llama mode -> llama-server on port ${BACKEND_PORT}"
fi

exec "$VENV" "$SCRIPT_DIR/../thinking_proxy.py"
