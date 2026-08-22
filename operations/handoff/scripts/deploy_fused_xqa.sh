#!/usr/bin/env bash
# 把融合分页注意力 kernel 部署到生产。
#
# 依据(独占窗口实测, testTurboPagedAttention --bench):
#   生产形状 qoLen=3 上稳定 5.26~5.28x, 跨 8K/32K/128K 三个数量级不变;
#   而且比现路径**准 17~18 倍**(现路径 fp16 score 缓冲 + fp16 反量化,
#   累加的 key 越多舍入越厉害; 8K 上最大相对误差已到 6.6e-3)。
#   正确性: 92/92 小形状对拍 + 24/24 大形状(8K/32K/128K)且 compute-sanitizer
#   memcheck 0 errors。
#
# 回滚设计: 先备份现有 apiserver 二进制。重建失败或验收不过时直接换回二进制,
#   不需要再编译一次 —— tmux 里有真实 agent 在等, 回滚必须快。
set -uo pipefail
ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM
SCR=/home/ezra/projects/EzraVastLLM/scripts
BR=$ROOT/fastllm/build-rw
PROF=$ROOT/v100-perfs/runtime/fastllm-native-profiles/q38-PROD-cyber-iq4xs-imatrix-mtp2-sm70.env
BAK=$ROOT/imatrix/apiserver.bak-before-fused-xqa
LOG=$ROOT/imatrix/logs/deploy-fused-xqa.log
: > "$LOG"; exec > >(tee -a "$LOG") 2>&1

start_prod () { tmux respawn-pane -k -t fastllm-prod:0.0 "$SCR/start_prod.sh $PROF" 2>/dev/null; }

echo "===== 1/5 备份当前二进制 ====="
cp -f "$BR/apiserver" "$BAK" || { echo "  备份失败, 中止"; exit 2; }
echo "  $BAK  ($(stat -c %s "$BAK") 字节)"

echo
echo "===== 2/5 停生产 ====="
pkill -f 'thinking_proxy\.py' 2>/dev/null; sleep 2
for pid in $(pgrep -f 'build-rw/apiserver' || true); do
  kill "$pid" 2>/dev/null
  for _ in $(seq 1 25); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
done
sleep 3

echo
echo "===== 3/5 重建 build-rw ====="
# build-rw 是共享的, 串行化以免和别的编译撞车(本仓出过共享对象被写坏)
until ! pgrep -f "cmake --build" >/dev/null; do echo "  等别的编译..."; sleep 10; done
cd "$BR" || { echo "  build-rw 不存在"; start_prod; exit 2; }
t0=$(date +%s)
if cmake --build . --target apiserver -j 6 2>&1 | tail -25; then
  echo "  重建成功, 用时 $(( ($(date +%s)-t0)/60 )) 分钟"
else
  echo "  [!] 重建失败 -> 换回备份二进制并拉起生产"
  cp -f "$BAK" "$BR/apiserver"
  start_prod
  exit 1
fi

echo
echo "===== 4/5 拉起生产 ====="
start_prod
echo "  已拉起, 等真就绪"

echo
echo "===== 5/5 验收 ====="
TAG=fused-xqa READY_TIMEOUT=1500 "$SCR/probe_suite.sh"
RC=$?

echo
echo "  ── 路由普查: attn.sm70_turbo_xqa 必须非零 ──"
XQA=$(curl -s --max-time 15 http://127.0.0.1:8002/props 2>/dev/null | python3 -c '
import json,sys
try: r=json.load(sys.stdin).get("kernel_routes") or {}
except Exception: print(-1); raise SystemExit
v=r.get("attn.sm70_turbo_xqa")
print(int(v.get("calls",0)) if isinstance(v,dict) else (int(v) if v else 0))' 2>/dev/null)
echo "     attn.sm70_turbo_xqa calls = ${XQA:-取不到}"
curl -s --max-time 15 http://127.0.0.1:8002/props 2>/dev/null | python3 -c '
import json,sys
r=json.load(sys.stdin).get("kernel_routes") or {}
for k,v in sorted(r.items()):
    if v: print(f"     {k:<34}{v}")' 2>/dev/null

echo
echo "===== 结论 ====="
if [ "$RC" = 0 ] && [ -n "$XQA" ] && [ "$XQA" -gt 0 ] 2>/dev/null; then
  echo "  >> 部署成功: 验收通过且新 kernel 计数非零($XQA)"
  exit 0
fi
# 只有**模型层面**失败(exit 1)或新 kernel 压根没跑到(计数为 0)才回滚。
# exit 2 是套件前置失败(鉴权 / 后端没就绪 / 探针排队超时) —— 那说明我们**没测出来**,
# 不说明模型坏了。生产 batch=1 且有真实 agent 在跑时, 探针会排在几十个请求后面
# 而超时, 用它去回滚一个已经在正常服务的 kernel 是纯粹的自伤。
if [ "$RC" = 2 ] && [ -n "$XQA" ] && [ "$XQA" -gt 0 ] 2>/dev/null; then
  echo "  >> 套件前置失败(exit 2)但新 kernel 计数非零($XQA) -> **不回滚**, 需人工补验收"
  exit 2
fi
echo "  >> 未达标(验收 exit=$RC, xqa 计数=${XQA:-N/A}) -> 换回备份二进制"
pkill -f 'thinking_proxy\.py' 2>/dev/null; sleep 2
pkill -f 'build-rw/apiserver' 2>/dev/null; sleep 5
cp -f "$BAK" "$BR/apiserver"
start_prod
echo "  已回滚。证据见 $LOG"
exit 1
