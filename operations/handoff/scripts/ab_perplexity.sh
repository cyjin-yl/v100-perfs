#!/usr/bin/env bash
# 在**留出集**上对比「带 imatrix」与「不带 imatrix」两版量化的困惑度。
#
# 为什么必须用留出集:
#   imatrix 是在 calib-agentic.txt 上算出来的。如果拿同一份语料测 PPL,
#   等于用训练集当测试集 —— imatrix 版必然更好, 但这个"更好"毫无意义,
#   它只证明了统计量拟合了自己的输入。
#   holdout-agentic.txt 来自另外 14 个会话, 实测与校准集 512 字节窗口
#   重叠 0.00%, 是干净的评测集。
#
# 为什么两版都要自己量化:
#   官方发布的 Q5_K_M 可能用了不同的 llama.cpp 版本和 tensor-type 覆盖,
#   拿它当基线就分不清差异来自 imatrix 还是版本。对照组必须同源同档同参数,
#   唯一差别是有没有 --imatrix。
#
# 显存: 两个模型各约 21GB, 与生产(约 28GB)冲突。默认拒绝在生产运行时启动;
#   要在生产旁边跑就设 CUDA_VISIBLE_DEVICES= 强制 CPU(会慢很多)。
set -uo pipefail

ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM
NV=/home/ezra/.local/lib/python3.13/site-packages/nvidia
LD=""
for d in cuda_runtime cublas cudnn cufft curand cusolver cusparse nvjitlink cuda_cupti nccl; do
  [ -d "$NV/$d/lib" ] && LD="$LD:$NV/$d/lib"
done
export LD_LIBRARY_PATH="${LD#:}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

BIN=$ROOT/llama.cpp/build/bin/llama-perplexity
QDIR="${QDIR:-$ROOT/models/Qwen3.8-27B-Uncensored-Cyber-GGUF/imatrix}"
HOLD="${HOLD:-$ROOT/imatrix/holdout-agentic.txt}"
NGL="${NGL:-99}"
CTX="${CTX:-512}"
THREADS="${THREADS:-6}"
CHUNKS="${CHUNKS:-120}"     # 限制块数以控制时长; 两版必须用同一个值才可比
OUT=$ROOT/imatrix/logs/ab-perplexity.txt

if pgrep -f 'build-rw/apiserver' >/dev/null && [ -n "${CUDA_VISIBLE_DEVICES-unset}" ]; then
  if [ "${ALLOW_WITH_BACKEND:-0}" != "1" ]; then
    echo "拒绝运行: 生产后端在跑, 两者显存会打架。" >&2
    echo "  停生产后再跑, 或 CUDA_VISIBLE_DEVICES= ALLOW_WITH_BACKEND=1 走 CPU。" >&2
    exit 1
  fi
fi
[ -f "$BIN" ]  || { echo "缺 llama-perplexity" >&2; exit 1; }
[ -f "$HOLD" ] || { echo "缺留出集 $HOLD" >&2; exit 1; }

run_ppl() {
  local model="$1" label="$2"
  [ -f "$model" ] || { echo "  跳过(文件不存在): $(basename "$model")"; return 1; }
  echo "== $label =="
  local t0=$(date +%s)
  local ppl
  ppl=$("$BIN" -m "$model" -f "$HOLD" -ngl "$NGL" -c "$CTX" -t "$THREADS" \
          --chunks "$CHUNKS" 2>&1 | tee -a "$OUT" | grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' | tail -1)
  echo "  $label  ${ppl:-<未取到 PPL>}   用时 $(( ($(date +%s)-t0)/60 )) 分钟"
  echo "$label  ${ppl:-N/A}" >> "$OUT.summary"
}

mkdir -p "$(dirname "$OUT")"
: > "$OUT"; : > "$OUT.summary"
echo "留出集: $(basename "$HOLD") ($(du -h "$HOLD" | cut -f1)), chunks=$CHUNKS ctx=$CTX"
echo

# 三方对比。两两之间只差一个变量, 才能把差异归因清楚:
#   A vs C  -> imatrix 的效果      (同为 Q5_K_M, 只差 --imatrix)
#   A vs B  -> 比特数的代价        (同带 imatrix, 5.33bpw vs 4.25bpw)
# 三者必须用同一个 --chunks 与同一份留出集, 否则数字不可比。
run_ppl "$QDIR/Qwen3.8-27B-Uncensored-Cyber-Q5_K_M-imatrix-fromq8.gguf"   "A Q5_K_M+imatrix  "
run_ppl "$QDIR/Qwen3.8-27B-Uncensored-Cyber-IQ4_XS-imatrix-fromq8.gguf"   "B IQ4_XS+imatrix  "
run_ppl "$QDIR/Qwen3.8-27B-Uncensored-Cyber-Q5_K_M-noimatrix-fromq8.gguf" "C Q5_K_M 无imatrix"

echo
echo "===== 汇总 (PPL 越低越好) ====="
cat "$OUT.summary"
echo
echo "----- 模型体积(IQ4_XS 的主要卖点是省显存, 必须和 PPL 一起看) -----"
for f in "$QDIR"/*-imatrix-*.gguf "$QDIR"/*-noimatrix-*.gguf; do
  [ -f "$f" ] && printf "  %8.2f GB  %s\n" "$(stat -c %s "$f" | awk '{print $1/1e9}')" "$(basename "$f")"
done
echo
echo "注意: PPL 只反映"预测下一个 token 的平均难度", 不直接等于"工具调用对不对"、"路径抄没抄歪"。"
echo "行为层面还要跑 copy_fidelity.py / toolname_probe.py / goal_acceptance.py。"
