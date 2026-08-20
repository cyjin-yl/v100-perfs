#!/usr/bin/env bash
# 给融合注意力 kernel 让出一个短的 GPU 独占窗口, 跑完自动归还生产。
#
# 为什么要独占: 要比的是"带宽受限"regime 下三条路径的差异(8K/32K/128K),
#   共享 GPU 时 SM 竞争会把绝对值全部拉长, 而且没法把"省物化"和"省 kernel 启动"
#   分开 —— 这正是 kvLen=2048 冒烟测(8.7~10.7x)不能当结论的原因。
#
# trap 保证任何退出路径都把生产拉回来: tmux 里 proto-ui / z3rm 两个真实 agent
#   在等这个后端。
set -uo pipefail
ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM
SCR=/home/ezra/projects/EzraVastLLM/scripts
PROF=$ROOT/v100-perfs/runtime/fastllm-native-profiles/q38-PROD-cyber-iq4xs-imatrix-mtp2-sm70.env
LOG=$ROOT/imatrix/logs/gpu-bench-window.log
: > "$LOG"; exec > >(tee -a "$LOG") 2>&1

restored=0
restore() {
  [ "$restored" = 1 ] && return; restored=1
  echo; echo "===== 归还生产 ====="
  tmux respawn-pane -k -t fastllm-prod:0.0 "$SCR/start_prod.sh $PROF" 2>/dev/null
  echo "  已拉起(IQ4_XS+BATCH_GQA, 加载约 6 分钟)"
}
trap restore EXIT INT TERM

echo "===== 停生产, 让出 GPU ====="
pkill -f 'thinking_proxy\.py' 2>/dev/null; sleep 2
for pid in $(pgrep -f 'build-rw/apiserver' || true); do
  kill "$pid" 2>/dev/null
  for _ in $(seq 1 25); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
done
sleep 3
echo "  空闲显存 $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"

echo
echo "===== 跑 bench(8K/32K/128K x qoLen{1,3} x 三条路径) ====="
cd "$ROOT/fastllm/build" || exit 2
timeout 900 ./testTurboPagedAttention --bench 2>&1 | sed 's/^/  /'
echo "  bench 退出码=$?"
exit 0
