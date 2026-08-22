#!/usr/bin/env bash
# 校准之后的完整流水线: 等 imatrix 产物 -> 量化(带/不带 imatrix 两版) -> 上传 HF。
# 放在 tmux 里跑, 断线不影响。
set -uo pipefail
ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM
IMAT=$ROOT/imatrix/imatrix-agentic.gguf
S=~/projects/EzraVastLLM/scripts

echo "===== 1/3 等待 imatrix 校准产物 ====="
for i in $(seq 1 180); do
  if [ -f "$IMAT" ] && ! pgrep -f llama-imatrix >/dev/null; then
    echo "  校准完成: $(ls -la "$IMAT" | awk '{printf "%.1f MB", $5/1e6}')"
    break
  fi
  if [ $((i % 10)) -eq 0 ]; then
    echo "  [$(date +%H:%M:%S)] 等待中 ... $(tail -1 $ROOT/imatrix/logs/imatrix-run.log 2>/dev/null | tail -c 80)"
  fi
  sleep 20
done
[ -f "$IMAT" ] || { echo "imatrix 产物未生成, 中止"; exit 1; }

echo
echo "===== 2/3 量化(线程留余量给生产) ====="
# 生产此时应已自动恢复; 量化是纯 CPU, 用 5 线程给后端留出余地
THREADS=5 TAG=fromq8 MAKE_BASELINE=1 $S/quantize_with_imatrix.sh
echo "量化退出码=$?"

echo
echo "===== 3/3 上传 HuggingFace ====="
$S/upload_to_hf.sh
echo "上传退出码=$?"
echo
echo "===== 流水线结束 $(date) ====="
