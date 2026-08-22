#!/usr/bin/env bash
# 把 MTP draft heads 移植进量化好的 GGUF。
#
# 为什么必须做: 生产跑的是 "...-Q5_K_M-plus-mtp.gguf" —— MTP heads 是 graft
# 进去的, 官方发布的主模型 GGUF **不含** MTP。新量化出来的 IQ4_XS 同样不含,
# 直接上生产等于静默关掉 MTP。而 MTP 是硬需求(实测 pos_accept_rate 约
# 70%/48%, 关掉吞吐直接掉一档)。
#
# MTP 用 Q8_0 而不是 Q4_0(1.68GB, 省 1.5GB):
#   draft 的质量直接决定接受率, 接受率掉了 MTP 就白开。为省 1.5GB 去赌接受率
#   不划算 —— 真正的显存收益应该从主模型(IQ4_XS 省约 4GB)拿, 不是从 draft 拿。
set -uo pipefail
ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM
GRAFT=$ROOT/v100-perfs/scripts/graft_gguf_mtp.py
MTP="${MTP:-$ROOT/models/Qwen3.8-27B-Uncensored-Cyber-GGUF/new-2026-08-19/mtp-Qwen3.8-27B-Uncensored-Cyber-Q8_0.gguf}"
QDIR="${QDIR:-$ROOT/models/Qwen3.8-27B-Uncensored-Cyber-GGUF/imatrix}"
PY=$ROOT/.venv-1cat/bin/python

[ -f "$GRAFT" ] || { echo "缺 graft 脚本 $GRAFT" >&2; exit 1; }
[ -f "$MTP" ]   || { echo "缺 MTP 源 $MTP" >&2; exit 1; }

shopt -s nullglob
for f in "$QDIR"/*-imatrix-*.gguf "$QDIR"/*-noimatrix-*.gguf; do
  case "$f" in *-plus-mtp.gguf) continue;; esac
  out="${f%.gguf}-plus-mtp.gguf"
  [ -f "$out" ] && { echo "[graft] 已存在, 跳过 $(basename "$out")"; continue; }
  echo "[graft] $(basename "$f")  ->  $(basename "$out")"
  t0=$(date +%s)
  if "$PY" "$GRAFT" --target "$f" --mtp-source "$MTP" --output "$out"; then
    echo "[graft] 完成 $(du -h "$out" | cut -f1)  用时 $(( ($(date +%s)-t0)/60 )) 分钟"
  else
    echo "[graft] 失败: $(basename "$f")" >&2
  fi
done
