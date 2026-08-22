#!/usr/bin/env bash
# 用 imatrix 做加权量化, 并同时产出一份**不带 imatrix 的同源同档对照组**。
#
# 为什么必须自己造对照组:
#   官方发布的 Q5_K_M 可能用了不同的 llama.cpp 版本、不同的 tensor-type 覆盖,
#   拿它当基线, 测出来的差异就分不清是 imatrix 的功劳还是版本差异。
#   只有"同源、同档、同参数, 唯一差别是有没有 --imatrix"才是干净对照。
#
# 两条量化源, 各有取舍(默认 Q8_0, 因为它已经在手):
#   BF16 原版权重  —— 正确, 单次量化。但要下 54GB safetensors + 转 54GB GGUF。
#   Q8_0 GGUF      —— 二次量化, 需要 --allow-requantize。Q8_0 约 8.5bit 近乎无损,
#                     误差主要由 Q5 这一步主导, Q8 引入的增量很小; 胜在立刻能跑。
#   建议两条都出, 用同一套探针对比, 让数据说话, 别赌。
#
# --output-tensor-type / --token-embedding-type 抬到 Q8_0:
#   lm_head 与 embedding 对"逐字抄写"类错误特别敏感 —— 输出层的量化误差会直接
#   变成选错 token, 而生产实测的破坏样本(路径、包名、commit hash 抄歪)正是这类。
#   代价约 +1.5GB 显存。显存吃紧就去掉这两行。
#
# RECIPE=<file> 走 --tensor-type-file, 对指定张量类别单独指定档位。
#   备好的配方在 scripts/quant-recipes/, 依据与实测数据见那里的 README-recipes.md。
#   一句话: 上游 GGUF 里那 288 个 Q8_0 张量不是"有意保留高精度", 而是根本没被
#   量化(与 Q8_0 源逐字节相同), 所以不值得照抄; 真正划算的只有 ssm_alpha/
#   ssm_beta/attn_k/attn_v/attn_output 这几类小张量, 合计 +121 MiB。
set -uo pipefail

ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM
NV=/home/ezra/.local/lib/python3.13/site-packages/nvidia
LD=""
for d in cuda_runtime cublas nvjitlink; do
  [ -d "$NV/$d/lib" ] && LD="$LD:$NV/$d/lib"
done
export LD_LIBRARY_PATH="${LD#:}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

BIN=$ROOT/llama.cpp/build/bin/llama-quantize
SRC="${SRC:-$ROOT/models/Qwen3.8-27B-Uncensored-Cyber-GGUF/new-2026-08-19/Qwen3.8-27B-Uncensored-Cyber-Q8_0.gguf}"
IMAT="${IMAT:-$ROOT/imatrix/imatrix-agentic-v2.gguf}"
OUTDIR="${OUTDIR:-$ROOT/models/Qwen3.8-27B-Uncensored-Cyber-GGUF/imatrix}"
TYPE="${TYPE:-Q5_K_M}"
THREADS="${THREADS:-6}"
TAG="${TAG:-fromq8}"          # 产物命名里标明量化源, 免得两条路线的产物混淆

mkdir -p "$OUTDIR"
[ -f "$BIN" ]  || { echo "缺 llama-quantize" >&2; exit 1; }
[ -f "$SRC" ]  || { echo "缺量化源 $SRC" >&2; exit 1; }
[ -f "$IMAT" ] || { echo "缺 imatrix $IMAT" >&2; exit 1; }

# 源本身已是量化格式时必须显式允许重量化, 否则 llama-quantize 直接拒绝。
EXTRA=()
case "$(basename "$SRC")" in
  *BF16*|*bf16*|*F16*|*f16*|*F32*|*f32*) ;;
  *) EXTRA+=(--allow-requantize)
     echo "[quantize] 源是量化格式, 已加 --allow-requantize(二次量化)" ;;
esac

# 可选的逐类别升档配方。默认不启用 —— 默认配方就是 llama.cpp 的标准混合。
RECIPE="${RECIPE:-}"
if [ -n "$RECIPE" ]; then
  [ -f "$RECIPE" ] || { echo "缺配方文件 $RECIPE" >&2; exit 1; }
  EXTRA+=(--tensor-type-file "$RECIPE")
  echo "[quantize] 升档配方: $(basename "$RECIPE") -> $(tr '\n' ' ' < "$RECIPE")"
fi

run_one() {
  local out="$1"; shift
  echo "[quantize] -> $(basename "$out")"
  local t0=$(date +%s)
  "$BIN" "${EXTRA[@]}" "$@" \
    --output-tensor-type q8_0 --token-embedding-type q8_0 \
    "$SRC" "$out" "$TYPE" "$THREADS" \
    && echo "[quantize] 完成 $(du -h "$out" | cut -f1)  用时 $(( ($(date +%s)-t0)/60 )) 分钟" \
    || { echo "[quantize] 失败: $(basename "$out")" >&2; return 1; }
}

echo "[quantize] 源  : $(basename "$SRC") ($(du -h "$SRC" | cut -f1))"
echo "[quantize] imatrix: $(basename "$IMAT")"

# 一次产出三个候选, 让数据决定用哪个, 而不是先验拍板:
#
#   Q5_K_M + imatrix    主候选(当前生产就是 Q5_K_M, 便于对齐)
#   Q5_K_M 无 imatrix   干净对照 —— 同源同档同参数, 唯一差别是有没有 --imatrix。
#                       没有它就无法把差异归因给 imatrix(官方发布的 Q5_K_M 可能
#                       用了不同 llama.cpp 版本/参数, 不是干净对照)。
#   IQ4_XS + imatrix    省显存候选。4.25bpw vs 5.33bpw, 模型约 16~17GB vs 20.8GB,
#                       多出约 4GB 给 KV 与前缀缓存 —— 而我们今天所有的麻烦
#                       (pool 顶 6590/6600MB、Grow 抛异常打挂整批请求、
#                        前缀缓存吃不饱)根子都是显存不够。
#                       且 IQ 系列本就是为 imatrix 设计的, 收益比 Q5_K_M 大。
#                       风险: 比特数实打实更低, 而我们正在打的就是质量问题;
#                       另外 fastllm 的 SM70 IQ4_XS MMQ 路径(DP4A, 默认开启,
#                       FASTLLM_CUDA_SM70_IQ4XS_MMQ=0 可关)在本机没测过速度。
#                       所以是"多量化一档去测", 不是"替换"。
TYPES="${TYPES:-IQ4_XS Q5_K_M}"   # IQ4_XS 排首: 它是部署候选, 先产出先能测
for T in $TYPES; do
  TYPE="$T"
  run_one "$OUTDIR/Qwen3.8-27B-Uncensored-Cyber-${T}-imatrix-${TAG}.gguf" --imatrix "$IMAT"
done

if [ "${MAKE_BASELINE:-1}" = "1" ]; then
  TYPE="${BASELINE_TYPE:-Q5_K_M}"
  echo "[quantize] 对照组(同源同档, 唯一差别是没有 --imatrix): $TYPE"
  run_one "$OUTDIR/Qwen3.8-27B-Uncensored-Cyber-${TYPE}-noimatrix-${TAG}.gguf"
fi
