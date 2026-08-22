#!/usr/bin/env bash
# 用领域匹配语料计算 imatrix(重要性矩阵), 供 llama-quantize 做加权量化。
#
# imatrix 是什么: 拿一批校准文本跑前向, 统计每个权重通道上激活的二阶量。
#   量化时按这个统计给通道分配精度 —— 重要的少丢, 不重要的多丢。同样是
#   Q5_K_M, 带 imatrix 通常明显更好, 且**在校准分布上**尤其好。所以校准语料
#   选什么直接决定收益落在哪里, 见 build_imatrix_corpus.py。
#
# 为什么用 Q8_0 当校准源而不是 BF16:
#   统计量只需要"足够准的激活", Q8_0 近乎无损, 业界(bartowski/mradermacher)
#   对大模型普遍就这么做。Q8_0 约 28.6GB 能整个装进 V100 的 32GB 显存 ->
#   校准跑满 GPU; BF16 约 54GB 装不下, 只能 CPU, 慢一个量级。
#   最终**量化**仍然从 BF16 出, 不做二次量化, 见 quantize_with_imatrix.sh。
#
# ⚠ 必须独占 GPU —— 这条是踩坑换来的:
#   2026-08-20 我以为 `-ngl 0` 就是纯 CPU, 于是和生产一起跑。结果 llama.cpp
#   只要编进了 CUDA, **即使 ngl=0 也会初始化 CUDA 后端并占显存**, 把空闲显存
#   压到 0.27GiB, 代理的显存压力保护随即把生产后端整个卸载了
#   ([lifecycle] VRAM pressure sustained >60s, unloading)。
#   所以下面加了硬守卫: 后端活着就直接拒绝运行。真要 CPU 跑, 设
#   CUDA_VISIBLE_DEVICES= 把 GPU 藏掉, 并显式 ALLOW_WITH_BACKEND=1。
#
# 环境: llama.cpp 的二进制依赖 libcudart.so.12, 而系统只有 CUDA 13,
#   所以从 pip 的 nvidia-* 包借 CUDA 12 运行库。
#
# 输出格式: 新版 llama-imatrix 默认存 **GGUF 格式**的 imatrix(即使文件名是
#   .dat 也一样, 只会警告)。llama-quantize --imatrix 同版本可直接读。
#   需要老格式就加 --output-format dat。
set -uo pipefail

ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM
NV=/home/ezra/.local/lib/python3.13/site-packages/nvidia
LD=""
for d in cuda_runtime cublas cudnn cufft curand cusolver cusparse nvjitlink cuda_cupti nccl; do
  [ -d "$NV/$d/lib" ] && LD="$LD:$NV/$d/lib"
done
export LD_LIBRARY_PATH="${LD#:}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

BIN=$ROOT/llama.cpp/build/bin/llama-imatrix
MODEL="${MODEL:-$ROOT/models/Qwen3.8-27B-Uncensored-Cyber-GGUF/new-2026-08-19/Qwen3.8-27B-Uncensored-Cyber-Q8_0.gguf}"
CORPUS="${CORPUS:-$ROOT/imatrix/calib-agentic-large.txt}"
OUT="${OUT:-$ROOT/imatrix/imatrix-agentic-v2.gguf}"
NGL="${NGL:-99}"          # 显存装不下就调小, 例如 55
CTX="${CTX:-512}"         # imatrix 惯例块长
THREADS="${THREADS:-6}"

if pgrep -f 'build-rw/apiserver' >/dev/null && [ "${ALLOW_WITH_BACKEND:-0}" != "1" ]; then
  echo "拒绝运行: 生产后端 (build-rw/apiserver) 正在跑。" >&2
  echo "  imatrix 会占显存, 会触发代理的显存压力保护把生产卸载。" >&2
  echo "  先停生产, 或设 ALLOW_WITH_BACKEND=1 且 CUDA_VISIBLE_DEVICES= 强制 CPU。" >&2
  exit 1
fi

for f in "$BIN" "$MODEL" "$CORPUS"; do
  [ -f "$f" ] || { echo "缺文件: $f" >&2; exit 1; }
done

echo "[imatrix] 模型  : $(basename "$MODEL") ($(du -h "$MODEL" | cut -f1))"
echo "[imatrix] 语料  : $(basename "$CORPUS") ($(du -h "$CORPUS" | cut -f1))"
echo "[imatrix] 输出  : $OUT"
echo "[imatrix] ngl=$NGL ctx=$CTX threads=$THREADS"
echo "[imatrix] 空闲显存: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"

CHUNKS="${CHUNKS:-0}"     # >0 时限制块数, 用来把停机窗口控制在可接受范围
EXTRA=()
[ "$CHUNKS" -gt 0 ] && EXTRA+=(--chunks "$CHUNKS")

# --parse-special: 让 <|im_start|> / <tool_call> 这类标记按**真正的特殊 token**
#   参与统计, 而不是被当字面文本切成 "<" "|" "im" "_start"...。
#   第一轮漏了它 —— 而我们的语料正是用 chat template 标记渲染的, 也就是说
#   "让特殊 token 进入校准分布"这个设计目标第一轮根本没达成。
#   llama.cpp 官方 README: "Useful for models with custom tokenizers."
#
# ctx 保持 512 而不是拉大: 社区实测 512 通常优于 4096 —— 固定 token 预算下
#   块越小样本越多、上下文越多样, 统计量条件数更好; 大块把预算集中在少数
#   长段落上。别被"我们跑 262K 所以要长 ctx"的直觉带偏(我就被带偏过)。
exec "$BIN" -m "$MODEL" -f "$CORPUS" -o "$OUT" -ngl "$NGL" -c "$CTX" -t "$THREADS" \
     --parse-special "${EXTRA[@]}"
