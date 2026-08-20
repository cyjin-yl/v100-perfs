#!/usr/bin/env bash
# 把生产切到 IQ4_XS+imatrix+MTP, 切完立刻验收, 不过就自动回滚到 Q5。
#
# 为什么要带自动回滚: tmux 里 proto-ui / z3rm 两个 agent 要靠这个后端干活。
#   留一个"起来了但输出是坏的"的中间态, 比直接不切更糟 —— agent 会把坏输出
#   当成事实继续往下做, 而且会读回自己写坏的内容, 判定"我在被 prompt injection"
#   然后卡死。所以判据不通过就必须自动退回已知可用的那份。
#
# 回滚判据只认**模型层面**的失败(逐字抄写不满分)。前置/套件失败(exit 2)不回滚,
#   因为那说明我们没测出来, 而不是模型坏了 —— 上一轮闸门就是把自己的鉴权 bug
#   报成"IQ4_XS 不得上生产", 差点用一个 harness bug 挡掉一个好模型。
set -uo pipefail
ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM
SCR=/home/ezra/projects/EzraVastLLM/scripts
D=$ROOT/v100-perfs/runtime/fastllm-native-profiles
NEW=$D/q38-PROD-cyber-iq4xs-imatrix-mtp2-sm70.env
OLD=$D/q38-PROD-cyber-q5-mtp2-turbo4-sm70.env
LOG=$ROOT/imatrix/logs/switch-iq4xs.log
: > "$LOG"; exec > >(tee -a "$LOG") 2>&1

W=$(grep -oE '\-\-path [^ ]+' "$NEW" | awk '{print $2}')
[ -f "$W" ] || { echo "[中止] 目标权重不存在: $W"; exit 2; }
echo "===== 切换到 IQ4_XS ====="
echo "  权重 $(basename "$W")  ($(( $(stat -c %s "$W")/1048576 )) MB)"
echo "  页池 $(grep -oE '^FASTLLM_PAGED_POOL_MAX_MB=.*' "$NEW" | cut -d= -f2) MB (Q5 档是 6600)"

start_with () {
  pkill -f 'thinking_proxy\.py' 2>/dev/null; sleep 2
  for pid in $(pgrep -f 'build-rw/apiserver' || true); do
    kill "$pid" 2>/dev/null
    for _ in $(seq 1 25); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
  done
  sleep 3
  # respawn-pane **必须显式带命令**: 不带命令时它会重跑 pane 原本的命令
  # (本 pane 原命令是不带参数的 start_prod.sh, 即 Q5), 于是 profile 参数
  # 完全不生效, 而随后 send-keys 的那行只会被打进已在运行的进程的 stdin。
  # 这个坑同样存在于 gpu_verify_window.sh 和 iq4xs_gate.sh 的恢复逻辑里 ——
  # 它们"看起来正常"只是因为 pane 原命令恰好就是要恢复的那份。
  tmux respawn-pane -k -t fastllm-prod:0.0 "$SCR/start_prod.sh $1" 2>/dev/null
}

start_with "$NEW"
echo "  已在 fastllm-prod:0.0 拉起, 等待真就绪(权重 15.4GB, 机械盘约 5~9 分钟)"

echo
echo "===== 验收 ====="
TAG=prod-iq4xs READY_TIMEOUT=1500 "$SCR/probe_suite.sh"
RC=$?

echo
if [ "$RC" = 0 ]; then
  echo "===== 结论: 通过, 生产保持在 IQ4_XS ====="
elif [ "$RC" = 2 ]; then
  echo "===== 结论: 套件前置失败(不是模型的问题), **不回滚** ====="
  echo "  生产仍在 IQ4_XS。需要人工看 $LOG 决定。"
else
  echo "===== 结论: 模型层面未通过 -> 自动回滚到 Q5_K_M+MTP ====="
  start_with "$OLD"
  echo "  已回滚。证据见 $LOG"
fi
echo "  完整日志 $LOG"
exit "$RC"
