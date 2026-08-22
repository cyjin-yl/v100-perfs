#!/usr/bin/env bash
# 在**纯 CPU** 上验证 IQ4_XS 量化产物, 零 GPU 占用, 不影响生产 agent。
#
# 为什么必须纯 CPU:
#   生产正在 GPU 上以 ~97% 利用率给真实 agent 服务, 用户明确要求"验证通过前
#   别切换, 不然会耽误 agent 干活"。而 llama.cpp 只要编进了 CUDA, **即使
#   -ngl 0 也会初始化 CUDA 后端并占显存**(今天已踩过: 把空闲显存压到 0.27GiB,
#   触发代理的显存压力保护把生产整个卸载)。所以必须 CUDA_VISIBLE_DEVICES=
#   把 GPU 彻底藏掉, 而不是只靠 -ngl 0。
#
# 为什么要同条件跑 Q5_K_M 做对照:
#   单看 IQ4_XS 的输出无法判断"怪"是量化掉的还是模型本来如此。同 prompt、
#   同采样参数、同在 CPU 上跑一遍现役的 Q5_K_M, 差异才可归因。
#   注意: 这里测的是**权重质量**, 不是 fastllm 引擎行为 —— 引擎侧(预分词、
#   工具名归一化、语法约束)已由 goal_acceptance.py 在生产上验过。
#
# 代价: 27B 模型在 8 核 CPU 上约 1~3 tok/s, 每个 prompt 几十秒;
#   加上两次模型加载(15~21GB, 机械盘约 95MB/s, 各 3~4 分钟)。全程约 40~60 分钟。
set -uo pipefail

ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM
NV=/home/ezra/.local/lib/python3.13/site-packages/nvidia
LD=""
for d in cuda_runtime cublas nvjitlink; do
  [ -d "$NV/$d/lib" ] && LD="$LD:$NV/$d/lib"
done
export LD_LIBRARY_PATH="${LD#:}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=""        # 关键: 彻底藏掉 GPU

BIN=$ROOT/llama.cpp/build/bin/llama-cli
QDIR=$ROOT/models/Qwen3.8-27B-Uncensored-Cyber-GGUF
IQ4="${IQ4:-$QDIR/imatrix/Qwen3.8-27B-Uncensored-Cyber-IQ4_XS-imatrix-fromq8-plus-mtp.gguf}"
Q5="${Q5:-$QDIR/Qwen3.8-27B-Uncensored-Cyber-Q5_K_M-plus-mtp.gguf}"
THREADS="${THREADS:-4}"               # 留核给生产与其它任务
OUT=$ROOT/imatrix/logs/verify-iq4xs-cpu.log

# 没 graft 的产物也允许测(只是没 MTP), 但要提示
[ -f "$IQ4" ] || {
  ALT="$QDIR/imatrix/Qwen3.8-27B-Uncensored-Cyber-IQ4_XS-imatrix-fromq8.gguf"
  [ -f "$ALT" ] && { echo "[warn] graft 后的产物不存在, 改用未 graft 版(无 MTP, 不影响权重质量判断)"; IQ4="$ALT"; }
}
for f in "$BIN" "$IQ4" "$Q5"; do
  [ -f "$f" ] || { echo "缺文件: $f" >&2; exit 1; }
done
mkdir -p "$(dirname "$OUT")"; : > "$OUT"

# 判据用的 prompt: 每条都能**自动判定**, 不靠人眼看"像不像话"
run_case () {
  local model="$1" label="$2" name="$3" prompt="$4" expect="$5" ntok="$6"
  local t0=$(date +%s)
  local raw
  raw=$(timeout 900 "$BIN" -m "$model" -p "$prompt" -n "$ntok" -t "$THREADS" \
          --temp 0 --no-warmup -no-cnv 2>/dev/null | tail -c 2000)
  local dt=$(( $(date +%s) - t0 ))
  local ok="✗"
  if [ -n "$expect" ]; then
    echo "$raw" | grep -qF -- "$expect" && ok="✓"
  else
    [ -n "$(echo "$raw" | tr -d '[:space:]')" ] && ok="✓"
  fi
  printf "  %s %-22s %-14s %3ds\n" "$ok" "$name" "$label" "$dt"
  { echo "=== $label / $name (${dt}s) ==="; echo "$raw"; echo; } >> "$OUT"
}

sweep () {
  local model="$1" label="$2"
  echo "── $label  ($(du -h "$model" | cut -f1)) ──"
  run_case "$model" "$label" "能加载并输出"   "请用一句话说明什么是快速排序。" "" 48
  run_case "$model" "$label" "逐字抄写-长路径" \
    "把下面这行原样重复一遍, 不要加任何其它内容:
/home/ezra/Documents/Proto-UI/packages/prototypes/brutalist/src/card/index.ts" \
    "/home/ezra/Documents/Proto-UI/packages/prototypes/brutalist/src/card/index.ts" 64
  run_case "$model" "$label" "逐字抄写-设备UUID" \
    "把下面这行原样重复一遍, 不要加任何其它内容:
13D010B6FDBC1A06" "13D010B6FDBC1A06" 32
  run_case "$model" "$label" "逐字抄写-commit" \
    "把下面这行原样重复一遍, 不要加任何其它内容:
3f9a2c7e15b8d046" "3f9a2c7e15b8d046" 32
  run_case "$model" "$label" "中英混排" \
    "把下面这行原样重复一遍, 不要加任何其它内容:
在 /run/media/ezra/13D010B6FDBC1A06 上跑 Qwen3.8-27B" \
    "/run/media/ezra/13D010B6FDBC1A06" 48
  run_case "$model" "$label" "算术(能力未塌)" "13 乘以 17 等于多少? 只回答数字。" "221" 24
  echo
}

echo "════ IQ4_XS vs Q5_K_M 纯 CPU 同条件对比 (threads=$THREADS, GPU 已藏) ════"
echo "详细输出: $OUT"
echo
sweep "$IQ4" "IQ4_XS+imatrix"
sweep "$Q5"  "Q5_K_M(现役)"
echo "════ 判读 ════"
echo "  逐字抄写四条是关键: 那正是我们一整天在打的故障类型。"
echo "  若 IQ4_XS 在这几条上明显差于 Q5_K_M, 就不该切换 —— 省 5.95GB 显存"
echo "  不值得用'路径抄歪'去换, 因为 agent 会读回自己写坏的内容并卡死。"
