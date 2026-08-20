#!/usr/bin/env bash
# 一次完整的 GPU 验证窗口: 等量化 -> 停生产 -> 三方 PPL + 生成探针 -> 恢复生产。
#
# 为什么用 GPU 而不是 CPU: CPU 方案约 40~60 分钟(27B 在 8 核上 1~3 tok/s +
#   两次 15~21GB 的机械盘加载), GPU 约 15~20 分钟。用户确认可以短暂借卡。
#
# 为什么必须三方而不是只测 IQ4_XS:
#   只看一个模型的输出无法归因"怪"是量化掉的还是模型本来如此。
#     A IQ4_XS+imatrix   候选
#     B Q5_K_M+imatrix   同 imatrix、只差比特数 -> 隔离比特数的代价
#     C 官方 Q5_K_M+mtp  现状基线
#   A vs B 回答"降位宽的代价", B vs C 回答"我们的 imatrix 相对官方配方如何"。
#
# 为什么可以用未 graft 的 IQ4_XS: graft 是逐字节拷贝、不重量化, 权重质量一致。
#
# trap 保证任何退出路径都恢复生产 —— tmux 里有真实 agent 在等。
set -uo pipefail
ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM
NV=/home/ezra/.local/lib/python3.13/site-packages/nvidia
LD=""
for d in cuda_runtime cublas cudnn cufft curand cusolver cusparse nvjitlink cuda_cupti nccl; do
  [ -d "$NV/$d/lib" ] && LD="$LD:$NV/$d/lib"
done
export LD_LIBRARY_PATH="${LD#:}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PPL=$ROOT/llama.cpp/build/bin/llama-perplexity
CLI=$ROOT/llama.cpp/build/bin/llama-cli
QDIR=$ROOT/models/Qwen3.8-27B-Uncensored-Cyber-GGUF
HOLD=$ROOT/imatrix/holdout-agentic.txt
LOG=$ROOT/imatrix/logs/gpu-verify.log
CHUNKS="${CHUNKS:-60}"
mkdir -p "$(dirname "$LOG")"; : > "$LOG"

A="$QDIR/imatrix/Qwen3.8-27B-Uncensored-Cyber-IQ4_XS-imatrix-fromq8.gguf"
B="$QDIR/imatrix/Qwen3.8-27B-Uncensored-Cyber-Q5_K_M-imatrix-fromq8.gguf"
C="$QDIR/Qwen3.8-27B-Uncensored-Cyber-Q5_K_M-plus-mtp.gguf"

restore() {
  echo "[window] 恢复生产 ..."
  # 显式带命令: 不带命令的 respawn-pane 会重跑 pane 原命令, profile 参数无效
  tmux respawn-pane -k -t fastllm-prod:0.0 '~/projects/EzraVastLLM/scripts/start_prod.sh' 2>/dev/null
  echo "[window] 已拉起(权重加载约 8 分钟)"
}
trap restore EXIT

echo "===== 1/4 等两个候选产物就绪 ====="
# 不能用"没有 llama-quantize 在跑"作为条件 —— 流水线在 Q5_K_M 之后还要跑
# 无 imatrix 对照组, 那样会正好撞在两次量化的间隙启动。
# 改为等**文件存在且大小连续两次不变**(说明写完了)。
# PPL 是 GPU 密集、量化是 CPU 密集, 两者可以并存, 不必互相等。
wait_stable () {
  local f="$1" last=-1 cur
  for i in $(seq 1 120); do
    [ -f "$f" ] || { sleep 15; continue; }
    cur=$(stat -c %s "$f")
    if [ "$cur" = "$last" ] && [ "$cur" -gt 1000000000 ]; then
      echo "  就绪 $(basename "$f") ($(echo "$cur/1e9"|bc -l|cut -c1-5) GB)"; return 0
    fi
    last=$cur; sleep 15
  done
  echo "  超时: $(basename "$f") 未就绪" >&2; return 1
}
wait_stable "$A" || exit 1
wait_stable "$B" || exit 1
for f in "$A" "$B" "$C" "$HOLD" "$PPL"; do
  [ -f "$f" ] || { echo "缺文件: $f" >&2; exit 1; }
done
ls -la --block-size=M "$A" "$B" "$C" | awk '{n=split($NF,a,"/");printf "  %7s  %s\n",$5,a[n]}'

echo
echo "===== 2/4 停生产, 独占 GPU ====="
for pid in $(pgrep -f 'thinking_proxy\.py' || true); do kill "$pid" 2>/dev/null; done
sleep 3
for pid in $(pgrep -f 'build-rw/apiserver' || true); do
  kill "$pid" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
done
sleep 3
echo "  空闲显存: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"

echo
echo "===== 3/4 留出集困惑度(同一份留出集, 同一个 --chunks=$CHUNKS) ====="
echo "  留出集与两份校准语料的 512 字节窗口重叠 0.00%/0.11%, 是干净评测集"
for pair in "A|IQ4_XS+imatrix|$A" "B|Q5_K_M+imatrix|$B" "C|官方Q5_K_M+mtp|$C"; do
  IFS='|' read -r tag label model <<< "$pair"
  echo "  ── $tag $label ──"
  t0=$(date +%s)
  R=$(timeout 2400 "$PPL" -m "$model" -f "$HOLD" -ngl 99 -c 512 -t 6 --chunks "$CHUNKS" 2>&1 | tee -a "$LOG" \
        | grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' | tail -1)
  echo "     ${R:-<未取到>}   用时 $(( ($(date +%s)-t0)/60 )) 分钟"
  echo "$tag $label  ${R:-N/A}" >> "$LOG.summary"
done

echo
echo "===== 4/4 IQ4_XS 生成探针(逐字抄写 —— 今天在打的故障类型) ====="
probe () {
  local name="$1" prompt="$2" expect="$3"
  local raw
  raw=$(timeout 300 "$CLI" -m "$A" -p "$prompt" -n 64 -ngl 99 -t 6 --temp 0 -no-cnv 2>/dev/null | tail -c 800)
  if echo "$raw" | grep -qF -- "$expect"; then echo "  ✓ $name"
  else echo "  ✗ $name   实得: $(echo "$raw" | tr -d '\n' | tail -c 120)"; fi
  { echo "=== $name ==="; echo "$raw"; echo; } >> "$LOG"
}
probe "长路径" "把下面这行原样重复一遍, 不要加任何其它内容:
/home/ezra/Documents/Proto-UI/packages/prototypes/brutalist/src/card/index.ts" \
  "/home/ezra/Documents/Proto-UI/packages/prototypes/brutalist/src/card/index.ts"
probe "设备UUID" "把下面这行原样重复一遍, 不要加任何其它内容:
13D010B6FDBC1A06" "13D010B6FDBC1A06"
probe "commit哈希" "把下面这行原样重复一遍, 不要加任何其它内容:
3f9a2c7e15b8d046" "3f9a2c7e15b8d046"
probe "中英混排" "把下面这行原样重复一遍, 不要加任何其它内容:
在 /run/media/ezra/13D010B6FDBC1A06 上跑 Qwen3.8-27B" "/run/media/ezra/13D010B6FDBC1A06"
probe "算术未塌" "13 乘以 17 等于多少? 只回答数字。" "221"

echo
echo "===== 汇总 (PPL 越低越好) ====="
cat "$LOG.summary" 2>/dev/null
echo
echo "  A vs B = 降位宽的代价(同 imatrix)"
echo "  B vs C = 我们的 imatrix 相对官方配方"
echo "  注意: 官方 Q5_K_M 保留了 296 个 Q8_0 张量(我们只有 2 个), 是个"偏胖"变体,"
echo "        所以 C 的体积 21.7GB 也高于 B。这点在判读时要考虑进去。"
echo "===== 窗口结束 $(date) ====="
