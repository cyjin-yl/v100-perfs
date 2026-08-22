#!/usr/bin/env bash
# 顺序跑多个扫描档位。一块 V100 只能串行, 所以这里是整张矩阵的外层循环。
#
# 顺序按"信息量优先"排: 先跑能立刻改变结论的档位(缓存调优 / MTP 开关 / KV 量化),
# 再跑需要等下载或 graft 的档位。模型文件不存在的档位自动跳过并记录。
#
# Usage: sweep_driver.sh [id ...]     # 不给参数就跑默认顺序
set -uo pipefail

PERF=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/v100-perfs
SWEEP="$PERF/runtime/fastllm-native-profiles/sweep"
PROJ=/run/media/ezra/13D010B6FDBC1A06/projects/EzraVastLLM
LOG="$PROJ/logs/sweep-driver.log"

DEFAULT_ORDER=(n5 n4 n2 n3 n7 c1 n6 c6 n9 c5 c3 n8 c4)
ORDER=("$@")
[ ${#ORDER[@]} -eq 0 ] && ORDER=("${DEFAULT_ORDER[@]}")

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

say "=== sweep driver 启动, 计划: ${ORDER[*]} ==="
for id in "${ORDER[@]}"; do
  prof=$(ls "$SWEEP"/q38-"$id"-*.env 2>/dev/null | head -1)
  if [ -z "$prof" ]; then
    say "!! 找不到 $id 的 profile, 跳过"
    continue
  fi
  tag=$(basename "$prof" .env)
  if [ -f "$PROJ/reports/$tag.json" ]; then
    say "-- $id 已有报告, 跳过(要重跑先删 reports/$tag.json)"
    continue
  fi
  # profile 里的 --path 指向的模型必须存在(Cyber graft / Q6 下载可能还没完成)
  model=$(sed -n "s/.*--path \([^ ]*\).*/\1/p" "$prof" | head -1)
  if [ ! -e "$model" ]; then
    say "!! $id 的模型还没就绪($model), 跳过"
    continue
  fi
  say ">> 开始 $id ($tag)"
  # 候选档位额外跑降智/拒答对照(Cyber 与普通版的部署决策靠它)
  case "$id" in
    n1|n7|c1|c3) intel=1 ;;
    *) intel=0 ;;
  esac
  INTEL=$intel bash "$PERF/scripts/sweep_one2.sh" "$prof" >>"$LOG" 2>&1
  say "<< 完成 $id rc=$?"
done
say "=== sweep driver 结束 ==="
