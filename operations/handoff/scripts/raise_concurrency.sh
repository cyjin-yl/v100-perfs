#!/usr/bin/env bash
# 把生产并发从 1 提到 4, 并验证。
#
# 为什么现在做: 验收目标里的"5~10 并发 subagent 无报错"在 --batch 1 下**结构上
#   不可能达成** —— 后端 maxActivateQueryNumber = max(1, min(256, batch)),
#   batch=1 就是严格串行。proxy 侧还有第二道 PROXY_STREAM_SLOTS=2。
#   两道串联, 实际并发恒为 1。副作用是任何探针都会被 agent 的长请求饿死
#   (抄写保真探针实测排队 13 分钟没轮到一个用例)。
#
# 为什么现在才敢做: IQ4_XS 把权重从 20.81 降到 15.96 GiB, 融合注意力 kernel
#   让 decode 的注意力便宜 5.26x, 显存与时间都腾出了余量。
#
# 风险与兜底: 并发上去后激活显存和常驻 KV 都变多。兜底有三层 ——
#   1) FASTLLM_VRAM_MIN_FREE_GIB=1.5 的水位保护;
#   2) 页池 Grow 已包 try/catch + 退避(不会再把异常穿到 MTPLoop 把在飞请求全 abort);
#   3) 本脚本自己的回滚: 验收不过就退回 batch=1 / slots=2。
set -uo pipefail
ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM
SCR=/home/ezra/projects/EzraVastLLM/scripts
PROF=$ROOT/v100-perfs/runtime/fastllm-native-profiles/q38-PROD-cyber-iq4xs-imatrix-mtp2-sm70.env
BATCH="${BATCH:-4}"
SLOTS="${SLOTS:-4}"
LOG=$ROOT/imatrix/logs/raise-concurrency.log
: > "$LOG"; exec > >(tee -a "$LOG") 2>&1

cp -f "$PROF" "$PROF.bak-concurrency"
echo "===== 1/4 改配置 batch=1->$BATCH  slots=2->$SLOTS ====="
sed -i "s/--batch 1 /--batch $BATCH /" "$PROF"
if grep -q "^PROXY_STREAM_SLOTS=" "$PROF"; then
  sed -i "s/^PROXY_STREAM_SLOTS=.*/PROXY_STREAM_SLOTS=$SLOTS/" "$PROF"
else
  cat >> "$PROF" <<EOS

# 2026-08-20 并发从 1 提到 $BATCH。
# 两道闸门必须一起提, 只提一道没用:
#   PROXY_STREAM_SLOTS  proxy 的信号量槽位(thinking_proxy.py:1374)
#   --batch             后端 maxActivateQueryNumber(apiserver.cpp:2326)
# 目标是验收里的"5~10 并发 subagent 无报错"。先到 $BATCH, 稳了再往上。
PROXY_STREAM_SLOTS=$SLOTS
EOS
fi
grep -oE '\-\-batch [0-9]+' "$PROF" | sed 's/^/  后端 /'
grep -E '^PROXY_STREAM_SLOTS=' "$PROF" | sed 's/^/  proxy /'

echo
echo "===== 2/4 重启 ====="
tmux respawn-pane -k -t fastllm-prod:0.0 "$SCR/start_prod.sh $PROF" 2>/dev/null
echo "  已拉起, 等真就绪"

echo
echo "===== 3/4 验收: 先单请求, 再并发 ====="
TAG=conc$BATCH READY_TIMEOUT=1500 "$SCR/probe_suite.sh"
RC=$?

A=$(grep -m1 -oE '^AUTH_TOKEN=.*' "$ROOT/.env" | cut -d= -f2- | tr -d "\"' ")
echo
echo "  ── 并发组($BATCH 路同时) ──"
python3 "$SCR/goal_acceptance.py" --token "$A" --only tools --concurrency "$BATCH" 2>&1 | tail -20 | sed 's/^/    /'
CONC=${PIPESTATUS[0]}

echo
echo "===== 4/4 结论 ====="
echo "  显存: $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader)"
grep -oE "vram=[0-9]+/[0-9]+MB" "$ROOT/v100-perfs/runtime/fastllm-native-profiles/logs/backend-PROD-cyber-iq4xs.log" | tail -1 | sed 's/^/  /'
echo "  探针 exit=$RC  并发 exit=$CONC"
if [ "$RC" = 0 ] && [ "$CONC" = 0 ]; then
  echo "  >> 并发 $BATCH 稳定, 保持"
  exit 0
fi
if [ "$RC" = 2 ]; then
  echo "  >> 套件前置失败(不是模型问题), **不回滚**, 需人工补验收"
  exit 2
fi
echo "  >> 未达标 -> 回滚到 batch=1 / slots=2"
cp -f "$PROF.bak-concurrency" "$PROF"
tmux respawn-pane -k -t fastllm-prod:0.0 "$SCR/start_prod.sh $PROF" 2>/dev/null
echo "  已回滚"
exit 1
