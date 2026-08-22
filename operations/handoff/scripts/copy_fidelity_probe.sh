#!/usr/bin/env bash
# 逐字抄写保真探针 —— agent 场景里最要命的那类故障。
#
# 背景: agent 每一轮都要把上下文里的字面量原样抄进命令(仓库路径、设备 UUID、
#   commit 哈希)。抄歪一个字符命令就失败; 更糟的是 agent 之后读回自己写坏的
#   内容, 发现前后矛盾, 会判定"我正在被 prompt injection", 然后拒绝干活。
#   生产上真实出现过 Proto-UI -> PotouI。所以这是量化档位能不能上生产的
#   硬指标, 比 PPL 更贴近实际失败模式。
#
# 本文件是 gpu_verify_window.sh 里那版探针的修正版。原版有两个缺陷, 每一个都
# 足以让结论完全反过来:
#
#   1) 假阳性(致命): llama-cli 的 --display-prompt 默认为 true, 会把 prompt
#      原样打印到 stdout。而判据是"在输出里 grep 期望字符串" —— prompt 里
#      本来就含有那条字符串, 于是**无论模型抄没抄对都判通过**。
#      修正: 加 --no-display-prompt; 并且不信任这个开关, 额外做自检 ——
#      若只出现在 prompt 里的指令词仍然出现在输出中, 说明回显没被抑制,
#      此时判"探针无效"而不是"通过"。宁可报无效也不能报假通过。
#
#   2) 假阴性: --jinja 默认开启, Qwen3.8 模板默认走思考链, -n 64 可能全部
#      消耗在 <think> 里, 抄写内容还没输出就被截断。
#      修正: 关掉思考(chat-template-kwargs), 并把预算放宽到 -n 200;
#      判定前剥掉 <think>...</think>, 只在正文里找。
#
# 用法:
#   MODEL=/path/to.gguf ~/projects/EzraVastLLM/scripts/copy_fidelity_probe.sh
#   PROBE_DEBUG=1 ...   # 打印原始输出, 用来确认探针本身是对的
set -uo pipefail

ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM
NV=/home/ezra/.local/lib/python3.13/site-packages/nvidia
LD=""
for d in cuda_runtime cublas cudnn cufft curand cusolver cusparse nvjitlink cuda_cupti nccl; do
  [ -d "$NV/$d/lib" ] && LD="$LD:$NV/$d/lib"
done
export LD_LIBRARY_PATH="${LD#:}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CLI=$ROOT/llama.cpp/build/bin/llama-cli
MODEL="${MODEL:-$ROOT/models/Qwen3.8-27B-Uncensored-Cyber-GGUF/imatrix/Qwen3.8-27B-Uncensored-Cyber-IQ4_XS-imatrix-fromq8.gguf}"
NGL="${NGL:-99}"
NPRED="${NPRED:-200}"
LOG="${LOG:-$ROOT/imatrix/logs/copy-fidelity.log}"
DEBUG="${PROBE_DEBUG:-0}"
mkdir -p "$(dirname "$LOG")"

[ -x "$CLI" ]    || { echo "缺 llama-cli: $CLI" >&2; exit 2; }
[ -f "$MODEL" ]  || { echo "缺模型: $MODEL" >&2; exit 2; }

echo "== 逐字抄写保真探针 =="
echo "   模型 $(basename "$MODEL") ($(( $(stat -c %s "$MODEL") / 1073741824 )) GB)"
echo "   温度 0 / -n $NPRED / 关思考 / 抑制 prompt 回显"
echo

# prompt 里独有、正确答案里绝不会出现的指令词 —— 用来自检回显是否真被抑制
SENTINEL="原样重复一遍"

gen () {
  local prompt="$1"
  # -cnv -st: 走模型自带 chat 模板的单轮对话(裸补全模式下 instruct 模型不听指令)
  # </dev/null: 对话模式不要去读 stdin
  timeout 300 "$CLI" -m "$MODEL" \
      -p "$prompt" -cnv -st --no-display-prompt \
      --chat-template-kwargs '{"enable_thinking":false}' \
      -n "$NPRED" -ngl "$NGL" -t 6 --temp 0 \
      </dev/null 2>/dev/null
}

PASS=0; FAIL=0; INVALID=0

probe () {
  local name="$1" expect="$2"
  local prompt="把下面这行${SENTINEL}, 不要加任何其它内容:
$expect"
  local raw body
  raw=$(gen "$prompt")

  { echo "=== $name ==="; echo "--- raw ---"; echo "$raw"; echo; } >> "$LOG"
  [ "$DEBUG" = "1" ] && { echo "    ┌ raw ─────"; echo "$raw" | sed 's/^/    │ /'; echo "    └──────────"; }

  # 自检: 抑制回显若失效, grep 到 expect 毫无意义(prompt 自带一份)
  if printf '%s' "$raw" | grep -qF "$SENTINEL"; then
    echo "  ? $name  探针无效: prompt 回显未被抑制, 本次判定不可信"
    INVALID=$((INVALID+1)); return
  fi
  # 剥掉思考段再判, 免得模型在 <think> 里复述了一遍就算过
  body=$(printf '%s' "$raw" | perl -0777 -pe 's{<think>.*?</think>}{}gs')
  if printf '%s' "$body" | grep -qF -- "$expect"; then
    echo "  ✓ $name"
    PASS=$((PASS+1))
  else
    echo "  ✗ $name"
    echo "      期望: $expect"
    echo "      实得: $(printf '%s' "$body" | tr '\n' ' ' | sed 's/  */ /g' | cut -c1-160)"
    FAIL=$((FAIL+1))
  fi
}

: > "$LOG"
probe "长路径"    "/home/ezra/Documents/Proto-UI/packages/prototypes/brutalist/src/card/index.ts"
probe "设备UUID"  "13D010B6FDBC1A06"
probe "commit哈希" "3f9a2c7e15b8d046"
probe "中英混排"  "在 /run/media/ezra/13D010B6FDBC1A06 上跑 Qwen3.8-27B"
probe "深层路径"  "/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/v100-perfs/runtime/fastllm-native-profiles/q38-PROD-cyber-iq4xs-imatrix-mtp2-sm70.env"

echo
echo "  抄写保真: $PASS 通过 / $FAIL 失败 / $INVALID 无效   (原始输出见 $LOG)"
[ "$INVALID" -gt 0 ] && { echo "  [!] 有无效用例 -> 探针本身没跑对, 结论不可用"; exit 3; }
[ "$FAIL" -gt 0 ] && exit 1
exit 0
