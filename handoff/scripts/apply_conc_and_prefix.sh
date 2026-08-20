#!/usr/bin/env bash
# 一次重启同时上两项: 并发 1->4, 以及前缀缓存的多模态路由修复。
#
# 【并发】--batch 1 时后端 maxActivateQueryNumber=1, 严格串行; proxy 侧还有
#   PROXY_STREAM_SLOTS=2。两道串联使"5~10 并发 subagent"结构上不可达, 且任何探针
#   都会被 agent 的长请求饿死(实测抄写探针排队 13 分钟未轮到一个用例)。
#   上一轮实测 batch=4 是健康的: 抄写 7/7、工具名 3/3、每条用例 0.9~2.6s、
#   显存 25373/32494MB。它被回滚纯粹是因为我给 goal_acceptance.py 传了
#   不存在的组名 "tools"(合法组名是 toolcall), 拿套件自己的错去回滚了配置。
#
# 【前缀缓存】命中率 0% 的根因: 多模态前向走 ForwardFromHiddenStates, 里面把
#   AllocatePagedCacheManager 的下标硬编码成 i*2(base 0); 而文本前向走 ForwardGPU,
#   用的是 threadTpPagedCacheBase(3000000+), 查询侧也只看后者。
#   (3000000+layer)*2 != layer*2, 两组池子完全不相交 —— 多模态请求记录到了
#   没人查询的管理器里。生产日志佐证: 57 个请求 5 次命中, **全是纯文本**
#   (最高 hit=1024/1095 = 94%); 每个带 vision 的请求都是 hit=0。
#   而两个 agent 每轮都发 images=5, 所以 100% 真实流量都在坏掉那条路上。
#   副作用: 页池被分配两次(吃掉一半预算), L1trie 对所有管理器求和 ——
#   于是"缓存看起来是满的、命中恒为 0"这个矛盾读数得到解释。
#
# 回滚: 备份二进制 + 备份 profile, 任一验收失败就同时退回。
# **只在模型层面失败时回滚**; 套件前置失败(exit 2)不回滚 —— 那说明我们没测出来,
# 不说明配置坏了。这条今天已经踩过两次。
set -uo pipefail
ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM
SCR=/home/ezra/projects/EzraVastLLM/scripts
BR=$ROOT/fastllm/build-rw
PROF=$ROOT/v100-perfs/runtime/fastllm-native-profiles/q38-PROD-cyber-iq4xs-imatrix-mtp2-sm70.env
BAKBIN=$ROOT/imatrix/apiserver.bak-before-prefixfix
BAKPROF=$PROF.bak-before-conc4
LOG=$ROOT/imatrix/logs/apply-conc-prefix.log
: > "$LOG"; exec > >(tee -a "$LOG") 2>&1

cp -f "$BR/apiserver" "$BAKBIN" && cp -f "$PROF" "$BAKPROF"
echo "===== 1/5 备份 ====="
echo "  二进制 $(stat -c %s "$BAKBIN") 字节   profile 已备份"

echo
echo "===== 2/5 改并发 batch=4 / slots=4 ====="
sed -i "s/--batch [0-9]\+ /--batch 4 /" "$PROF"
if grep -q "^PROXY_STREAM_SLOTS=" "$PROF"; then
  sed -i "s/^PROXY_STREAM_SLOTS=.*/PROXY_STREAM_SLOTS=4/" "$PROF"
else
  printf '\n# 两道闸门必须一起提: proxy 信号量 + 后端 maxActivateQueryNumber\nPROXY_STREAM_SLOTS=4\n' >> "$PROF"
fi
grep -oE '\-\-batch [0-9]+' "$PROF" | head -1 | sed 's/^/  后端 /'
grep -E '^PROXY_STREAM_SLOTS=' "$PROF" | sed 's/^/  proxy /'

echo
echo "===== 3/5 停生产并重建(带前缀缓存多模态修复) ====="
pkill -f 'thinking_proxy\.py' 2>/dev/null; sleep 2
for pid in $(pgrep -f 'build-rw/apiserver' || true); do
  kill "$pid" 2>/dev/null
  for _ in $(seq 1 25); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
done
sleep 3
until ! pgrep -f "cmake --build" >/dev/null; do echo "  等别的编译..."; sleep 10; done
cd "$BR" || exit 2
t0=$(date +%s)
if cmake --build . --target apiserver -j 6 2>&1 | grep -E "error|Error|\[100%\]|\[ 9[0-9]%\]" | tail -12; then
  echo "  重建用时 $(( ($(date +%s)-t0)/60 )) 分钟"
else
  echo "  [!] 重建失败 -> 回滚"
  cp -f "$BAKBIN" "$BR/apiserver"; cp -f "$BAKPROF" "$PROF"
  tmux respawn-pane -k -t fastllm-prod:0.0 "$SCR/start_prod.sh $PROF" 2>/dev/null
  exit 1
fi
[ -x "$BR/apiserver" ] || { echo "  [!] 二进制不存在"; cp -f "$BAKBIN" "$BR/apiserver"; exit 1; }

echo
echo "===== 4/5 拉起并验收 ====="
tmux respawn-pane -k -t fastllm-prod:0.0 "$SCR/start_prod.sh $PROF" 2>/dev/null
TAG=conc4-prefixfix READY_TIMEOUT=1500 "$SCR/probe_suite.sh"
RC=$?
A=$(grep -m1 -oE '^AUTH_TOKEN=.*' "$ROOT/.env" | cut -d= -f2- | tr -d "\"' ")
echo
echo "  ── 并发组(4 路) + 工具调用组 ──"
python3 "$SCR/goal_acceptance.py" --token "$A" --only toolcall --concurrency 4 2>&1 | tail -22 | sed 's/^/    /'
CONC=${PIPESTATUS[0]}

echo
echo "===== 5/5 结论 ====="
echo "  显存 $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader)"
echo "  探针 exit=$RC   并发/工具 exit=$CONC"
if [ "$RC" = 0 ] && [ "$CONC" = 0 ]; then
  echo "  >> 全过, 保持 batch=4 + 前缀缓存修复"
  exit 0
fi
if [ "$RC" = 2 ] || [ "$CONC" = 2 ]; then
  echo "  >> 套件前置失败(exit 2), **不回滚**, 需人工补验收"
  exit 2
fi
echo "  >> 模型层面未达标 -> 回滚二进制与 profile"
pkill -f 'thinking_proxy\.py' 2>/dev/null; sleep 2
pkill -f 'build-rw/apiserver' 2>/dev/null; sleep 5
cp -f "$BAKBIN" "$BR/apiserver"; cp -f "$BAKPROF" "$PROF"
tmux respawn-pane -k -t fastllm-prod:0.0 "$SCR/start_prod.sh $PROF" 2>/dev/null
echo "  已回滚"
exit 1
