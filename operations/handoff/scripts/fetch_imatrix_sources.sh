#!/usr/bin/env bash
# 下载 imatrix 量化所需的高精度源。
#
# 为什么需要两份:
#   imatrix 源  = Q8_0 GGUF(约 29GB)。近乎无损, 且停服后能整个装进 V100 的
#                 32GB 显存 -> 校准跑 GPU, 比 CPU 快一个量级。
#   量化源      = BF16 safetensors -> 转 GGUF(约 54GB)。从 BF16 直接出
#                 Q5_K_M, 避免"Q8/Q6 再量化"的误差叠加。
#
# 为什么必须走镜像:
#   本机直连 huggingface.co 超时(实测 curl 12s 无响应, HTTP 000), 而
#   hf-mirror.com 0.56s 返回 200。之前的下载停摆就是因为没设 HF_ENDPOINT ——
#   现象是"下了一半没进程了", 不报错, 很容易误判成下载完成。
set -uo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
HF=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/.venv-1cat/bin/hf
ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/models
LOG=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/imatrix/logs

# 1) imatrix 源: Q8_0
"$HF" download philbert440/Qwen3.8-27B-Uncensored-Cyber-GGUF \
  Qwen3.8-27B-Uncensored-Cyber-Q8_0.gguf \
  --local-dir "$ROOT/Qwen3.8-27B-Uncensored-Cyber-GGUF/new-2026-08-19" \
  >> "$LOG/dl-q8.log" 2>&1
echo "[fetch] Q8_0 完成 rc=$?"

# 2) 对照基线: 官方 Q5_K_M(之前只下了一半)
"$HF" download philbert440/Qwen3.8-27B-Uncensored-Cyber-GGUF \
  Qwen3.8-27B-Uncensored-Cyber-Q5_K_M.gguf \
  --local-dir "$ROOT/Qwen3.8-27B-Uncensored-Cyber-GGUF/new-2026-08-19" \
  >> "$LOG/dl-q5.log" 2>&1
echo "[fetch] Q5_K_M 完成 rc=$?"

# 3) 量化源: BF16 原版权重
"$HF" download philbert440/Qwen3.8-27B-Uncensored-Cyber \
  --local-dir "$ROOT/Qwen3.8-27B-Uncensored-Cyber-BF16" \
  >> "$LOG/dl-bf16.log" 2>&1
echo "[fetch] BF16 完成 rc=$?"
