#!/usr/bin/env bash
# 一次窗口部署四项, 并把多模态前缀缓存翻成开启。
#
#   F9   多模态 decode 切回 ForwardGPU, 让 MTP 对 vision 生效
#        (实证: 19 个 vision 窗口 / 8734 decode token / MTP 验证记录 0 条,
#         而两个 agent 每轮都发 images=5 => MTP 对 100% 真实流量从未生效)
#   F10  (a) 命中路径上补算 mrope_position_delta
#        (b) 残段含图像 token 时拒绝复用, 计 miss=remainder-has-image
#        (单元测试: 修复前 delta 缺失导致整段偏移恰好 |delta|;
#         夹具 32 个图像 token -> 24, 生产约 3700 个 -> 数千量级)
#   25bbb360  bind 提到加载模型之前(端口冲突 787ms 退出而不是 23 分钟僵尸)
#   proxy 信号量回收纪元(修 streams=-3/4 的双重释放)
#
# 本脚本把今天踩过的坑都写进了流程:
#   * 就绪判定 = 新 pid + 环境变量正确 + 真发一次补全; **不看 /health**
#     (/health 会命中旧进程, 今天误判过两次; 且权重要到首次真实请求才上卡)
#   * 暂停 agent 要循环按到状态真的变成 Loop paused
#     (omp 在流式中按 Esc 只中断当轮, loop 随即进入下一轮)
#   * 前置失败(exit 2)与模型失败(exit 1)分开: 前者不回滚, 因为那说明**没测出来**
#   * 备份二进制与 profile, 回滚不需要重新编译
set -uo pipefail
ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM
SCR=/home/ezra/projects/EzraVastLLM/scripts
BR=$ROOT/fastllm/build-rw
PROF=$ROOT/v100-perfs/runtime/fastllm-native-profiles/q38-PROD-cyber-iq4xs-imatrix-mtp2-sm70.env
BAKBIN=$ROOT/imatrix/apiserver.bak-before-bundle
BAKPROF=$PROF.bak-before-bundle
LOG=$ROOT/imatrix/logs/deploy-bundle.log
BLOG=$ROOT/v100-perfs/runtime/fastllm-native-profiles/logs/backend-PROD-cyber-iq4xs.log
: > "$LOG"; exec > >(tee -a "$LOG") 2>&1

agent_state () { tmux capture-pane -p -J -t "$1" -S -3 2>/dev/null | grep -oE 'Loop (running|paused)' | tail -1; }

resume_agents () {
  tmux send-keys -t proto-ui -l '请一直严格 review 所有新增提交、新增 pr、新增 issues 的情况并以严格、严谨的方法和语气进行回复。【如果没有人回复过的话】你可以用 gh-cli 作为我完成工作。请一直继续。'
  sleep 1; tmux send-keys -t proto-ui Enter; sleep 3
  tmux send-keys -t z3rm -l '以维护者的身份处理、决策你的行动。请一直严格 review 所有 z3rm 的新增提交、新增 pr、新增 issues 的情况并以严格、严谨的方法和语气进行回复。【如果没有人回复过的话】你可以用 gh-cli 作为我完成工作。请一直继续。'
  sleep 1; tmux send-keys -t z3rm Enter; sleep 3
  for S in proto-ui z3rm; do echo "  $S: $(agent_state $S)"; done
}

echo "===== 1/6 暂停两个 agent(循环按到真的 paused) ====="
# omp 的 Esc: **流式中**只中断当轮, **非流式**才 pauseLoop(dist/cli.js)。
# 而中断后 loop 立刻进入下一轮又开始流式 —— 所以"按一次、等几秒、再按"永远
# 刚好错过那个非流式窗口(实测按 10 次都没成功)。
# 正确手法是**快速连按**: 第一次中断流, 紧接着的一次落在非流式状态上。
# 实测 0.35 秒间隔的三连, 一组就能进入 Loop paused。
for S in proto-ui z3rm; do
  for k in $(seq 1 15); do
    [ "$(agent_state $S)" = "Loop paused" ] && break
    tmux send-keys -t "$S" Escape 2>/dev/null; sleep 0.35
    tmux send-keys -t "$S" Escape 2>/dev/null; sleep 0.35
    tmux send-keys -t "$S" Escape 2>/dev/null; sleep 2
  done
  echo "  $S: $(agent_state $S)"
done
if [ "$(agent_state proto-ui)" != "Loop paused" ] || [ "$(agent_state z3rm)" != "Loop paused" ]; then
  echo "  [!] 有 agent 没暂停成功。不在有 agent 跑的情况下翻 MULTIMODAL —— 中止。"
  exit 2
fi

echo
echo "===== 2/6 备份 ====="
cp -f "$BR/apiserver" "$BAKBIN" && cp -f "$PROF" "$BAKPROF"
echo "  二进制 $(stat -c %s "$BAKBIN") 字节, profile 已备份"

echo
echo "===== 3/6 停生产并重建 ====="
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
if ! cmake --build . --target apiserver -j 6 2>&1 | grep -E "error|Error|\[100%\]" | tail -8; then
  echo "  [!] 重建失败 -> 还原并拉起"
  cp -f "$BAKBIN" "$BR/apiserver"; cp -f "$BAKPROF" "$PROF"
  tmux respawn-pane -k -t fastllm-prod:0.0 "$SCR/start_prod.sh $PROF" 2>/dev/null
  resume_agents; exit 1
fi
echo "  重建用时 $(( ($(date +%s)-t0)/60 )) 分钟"

echo
echo "===== 4/6 翻开 MULTIMODAL 并拉起 ====="
sed -i 's/^FASTLLM_PREFIX_CACHE_MULTIMODAL=.*/FASTLLM_PREFIX_CACHE_MULTIMODAL=1/' "$PROF"
grep -n "^FASTLLM_PREFIX_CACHE_MULTIMODAL=" "$PROF" | sed 's/^/  /'
OLD=$(pgrep -f build-rw/apiserver | head -1)
tmux respawn-pane -k -t fastllm-prod:0.0 "$SCR/start_prod.sh $PROF" 2>/dev/null
NEW=""
for i in $(seq 1 120); do
  P=$(pgrep -f build-rw/apiserver | head -1)
  if [ -n "$P" ] && [ "$P" != "$OLD" ]; then
    V=$(tr "\0" "\n" < /proc/$P/environ 2>/dev/null | grep -oE "MULTIMODAL=[01]")
    [ "$V" = "MULTIMODAL=1" ] && { NEW=$P; echo "  新 pid=$P $V (第 $((i*5)) 秒)"; break; }
    [ -n "$V" ] && { echo "  [!] 新进程是 $V, 中止"; exit 2; }
  fi
  sleep 5
done
[ -n "$NEW" ] || { echo "  [!] 没等到 MULTIMODAL=1 的新进程"; exit 2; }
A=$(grep -m1 -oE '^AUTH_TOKEN=.*' "$ROOT/.env" | cut -d= -f2- | tr -d "\"' ")
echo "  等真就绪(发一次补全, 不看 /health)"
for i in $(seq 1 150); do
  c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 120 --noproxy '*' \
      -H "Authorization: Bearer $A" -H 'Content-Type: application/json' \
      -X POST http://127.0.0.1:8000/v1/chat/completions \
      -d '{"model":"qwen3.8-27b","max_tokens":1,"temperature":0,"stream":false,"messages":[{"role":"user","content":"hi"}]}' 2>/dev/null)
  [ "$c" = "200" ] && { echo "  就绪(第 $((i*10)) 秒)"; break; }
  sleep 10
done

echo
echo "===== 5/6 窗口验收 ====="
MARK=$(wc -c < "$BLOG")
python3 "$SCR/mm_cache_experiment.py"
RC=$?
echo
echo "  ── 路由与计数器 ──"
tail -c +$MARK "$BLOG" | grep -oE "mgr-lookup-mismatch=[0-9]+" | tail -1 | sed 's/^/    /'
tail -c +$MARK "$BLOG" | grep -oE "miss\{[^}]*\}" | tail -1 | tr ' ' '\n' | grep -E "remainder|mm-|delta" | sed 's/^/    /'
tail -c +$MARK "$BLOG" | grep -oE "pos_accept_rate=\[[^]]*\]" | tail -1 | sed 's/^/    MTP /'
tail -c +$MARK "$BLOG" | grep -oE "req#[0-9]+ total=[0-9]+ hit=[0-9]+ layer=[^ ]* miss=[a-z-]*" | tail -5 | sed 's/^/    /'

echo
echo "===== 6/6 结论 ====="
if [ "$RC" = 0 ]; then
  echo "  >> 通过: 保持 MULTIMODAL=1"
elif [ "$RC" = 2 ]; then
  echo "  >> 实验无效(未命中) —— 保守起见退回 MULTIMODAL=0, 二进制保留"
  sed -i 's/^FASTLLM_PREFIX_CACHE_MULTIMODAL=.*/FASTLLM_PREFIX_CACHE_MULTIMODAL=0/' "$PROF"
  tmux respawn-pane -k -t fastllm-prod:0.0 "$SCR/start_prod.sh $PROF" 2>/dev/null; sleep 30
else
  echo "  >> 不通过: 退回 MULTIMODAL=0(二进制保留, 其余三项修复是独立的)"
  sed -i 's/^FASTLLM_PREFIX_CACHE_MULTIMODAL=.*/FASTLLM_PREFIX_CACHE_MULTIMODAL=0/' "$PROF"
  tmux respawn-pane -k -t fastllm-prod:0.0 "$SCR/start_prod.sh $PROF" 2>/dev/null; sleep 30
fi
echo
echo "  恢复 agent"
resume_agents
echo "  最终 MULTIMODAL=$(grep -oE '^FASTLLM_PREFIX_CACHE_MULTIMODAL=[01]' "$PROF" | cut -d= -f2)"
exit "$RC"
