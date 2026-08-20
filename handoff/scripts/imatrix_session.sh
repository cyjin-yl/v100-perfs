#!/usr/bin/env bash
# 一次完整的 imatrix 校准会话: 停生产 -> 独占 GPU 跑校准 -> 无论成败都恢复生产。
#
# 为什么要做成一个脚本而不是手工三步:
#   imatrix 要独占显存(见 run_imatrix.sh 里的说明), 必须停生产。手工操作一旦
#   中途失联/超时, 生产就会一直躺着 —— 而 tmux 里有真实 agent 在等它。
#   这里用 trap 保证**任何退出路径**(成功/失败/被 Ctrl-C/被 kill)都会拉起生产。
#
# 停机窗口: 校准本身 + 两次模型加载(各约 8 分钟)。
set -uo pipefail

ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM
START_PROD=~/projects/EzraVastLLM/scripts/start_prod.sh
RUN_IMATRIX=~/projects/EzraVastLLM/scripts/run_imatrix.sh
LOGDIR=$ROOT/imatrix/logs
mkdir -p "$LOGDIR"

restore_production() {
  echo "[session] 恢复生产 ..."
  tmux respawn-pane -k -t fastllm-prod:0.0 2>/dev/null
  sleep 3
  tmux send-keys -t fastllm-prod:0.0 "$START_PROD" Enter 2>/dev/null
  echo "[session] 已在 tmux pane fastllm-prod:0.0 重新拉起(权重加载约 8 分钟)"
}
trap restore_production EXIT

echo "[session] 停止生产以独占 GPU"
for pid in $(pgrep -f 'thinking_proxy\.py' || true); do kill "$pid" 2>/dev/null; done
sleep 3
for pid in $(pgrep -f 'build-rw/apiserver' || true); do
  kill "$pid" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
done
sleep 3
echo "[session] 空闲显存: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"

echo "[session] 开始校准 (日志 $LOGDIR/imatrix-run.log)"
STARTED=$(date +%s)
"$RUN_IMATRIX" 2>&1 | tee "$LOGDIR/imatrix-run.log" | grep -E "^\[imatrix\]|compute_imatrix|Final estimate|ETA|error|failed" 
RC=${PIPESTATUS[0]}
echo "[session] 校准退出码=$RC 用时 $(( ($(date +%s) - STARTED) / 60 )) 分钟"
ls -la "$ROOT/imatrix/" 2>/dev/null | grep -E 'imatrix.*\.(gguf|dat)' || echo "[session] 未生成 imatrix 产物"
