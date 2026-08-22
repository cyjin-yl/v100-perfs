#!/usr/bin/env bash
# 校准之后的自动流水线: 等 v2 imatrix -> 量化(IQ4_XS 优先) -> graft MTP -> 三方 PPL。
# 不做上传(上传有 CONFIRM_UPLOAD 闸门, 等数据出来人工放行)。
set -uo pipefail
ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM
S=~/projects/EzraVastLLM/scripts
IMAT=$ROOT/imatrix/imatrix-agentic-v2.gguf

echo "===== 1/4 等 v2 校准完成 ====="
for i in $(seq 1 240); do
  if [ -f "$IMAT" ] && ! pgrep -f llama-imatrix >/dev/null; then
    echo "  完成: $(ls -la "$IMAT" | awk '{printf "%.1f MB", $5/1e6}')  $(date +%H:%M:%S)"
    break
  fi
  [ $((i % 10)) -eq 0 ] && echo "  [$(date +%H:%M:%S)] $(tail -1 $ROOT/imatrix/logs/imatrix-run.log 2>/dev/null | tail -c 70)"
  sleep 20
done
[ -f "$IMAT" ] || { echo "imatrix 未生成, 中止"; exit 1; }

echo
echo "===== 2/4 量化(IQ4_XS 优先; 线程留余量给已恢复的生产) ====="
# 等生产先加载完再开量化, 避免抢 CPU 拖慢它的启动
for i in $(seq 1 40); do
  curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8002/health 2>/dev/null | grep -q '^2' && { echo "  生产已就绪, 开始量化"; break; }
  sleep 15
done
THREADS=5 TAG=fromq8 MAKE_BASELINE=1 $S/quantize_with_imatrix.sh
echo "量化退出码=$?"

echo
echo "===== 3/4 graft MTP(否则等于静默关掉 MTP) ====="
$S/graft_mtp.sh
echo "graft 退出码=$?"

echo
echo "===== 4/4 产物清单 ====="
ls -la --block-size=M $ROOT/models/Qwen3.8-27B-Uncensored-Cyber-GGUF/imatrix/*.gguf 2>/dev/null \
  | awk '{n=split($NF,a,"/"); printf "  %8s  %s\n", $5, a[n]}'
echo
echo "下一步(需要停生产独占 GPU, 人工决定时机):"
echo "  $S/ab_perplexity.sh          # 三方留出集 PPL"
echo "  CONFIRM_UPLOAD=1 $S/upload_to_hf.sh   # 数据满意后再放行上传"
echo "===== 流水线结束 $(date) ====="
