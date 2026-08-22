#!/bin/bash
# Production proxy runner for tmux pane 0 (plain ThinkingCap Q6_K-MTP 262K).
set -o pipefail
set -a
. /run/media/ezra/13D010B6FDBC1A06/1CatVLLM/.env
. /run/media/ezra/13D010B6FDBC1A06/1CatVLLM/v100-perfs/runtime/fastllm-native-profiles/q6-plain-262k-mtp2.env
set +a
export LD_LIBRARY_PATH=/home/ezra/.conda/envs/tsenv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
/home/ezra/.conda/envs/tsenv/bin/python /run/media/ezra/13D010B6FDBC1A06/1CatVLLM/thinking_proxy.py 2>&1 | tee -a "$PROXY_LOG_FILE"
rc=${PIPESTATUS[0]}
printf '\n[tmux] thinking proxy exited rc=%s\n' "$rc"
exit "$rc"
