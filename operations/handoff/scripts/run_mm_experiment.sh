#!/usr/bin/env bash
# 受控实验: 打开多模态前缀缓存, 验证命中后的输出是否仍然正确。
#
# 为什么要暂停 agent: 两个理由。
#   1. 不暂停就排不上队 —— 实测一个几百 token 的请求排在两个 75K 重算后面
#      要 19 分钟以上。
#   2. 万一那两个疑似问题成立(位置整体错位 / 图像特征丢失), agent 的降智输出
#      会被直接发到 GitHub 上。这是对外的、不可撤回的。
#
# 结果处理: 不通过就立刻改回 MULTIMODAL=0 并重启, 再恢复 agent。
set -uo pipefail
ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM
SCR=/home/ezra/projects/EzraVastLLM/scripts
PROF=$ROOT/v100-perfs/runtime/fastllm-native-profiles/q38-PROD-cyber-iq4xs-imatrix-mtp2-sm70.env
LOG=$ROOT/imatrix/logs/mm-experiment.log
: > "$LOG"; exec > >(tee -a "$LOG") 2>&1

echo "===== 1/5 停掉残留探针, 暂停两个 agent ====="
pkill -f one_vision.py 2>/dev/null; pkill -f copy_fidelity_vision.py 2>/dev/null
# omp 的 Esc 语义(dist/cli.js): loop 开启时, **正在流式输出**则中断流;
# **不在流式**才 pauseLoop。所以流式中按一次只会中断当轮, loop 随即进入下一轮。
# 要真暂停就得多按几次, 每次之间留出它回到非流式状态的时间。
for S in proto-ui z3rm; do
  for k in 1 2 3 4 5 6; do
    st=$(tmux capture-pane -p -J -t "$S" -S -3 2>/dev/null | grep -oE 'Loop (running|paused)' | tail -1)
    [ "$st" = "Loop paused" ] && break
    tmux send-keys -t "$S" Escape 2>/dev/null
    sleep 4
  done
  echo "  $S: $(tmux capture-pane -p -J -t "$S" -S -3 2>/dev/null | grep -oE 'Loop (running|paused)' | tail -1)"
done
if tmux capture-pane -p -J -t proto-ui -S -3 2>/dev/null | grep -q "Loop running" ||
   tmux capture-pane -p -J -t z3rm -S -3 2>/dev/null | grep -q "Loop running"; then
  echo "  [!] 有 agent 没暂停成功 —— 不在有 agent 跑的情况下做这个实验, 中止"
  exit 2
fi

echo
echo "===== 2/5 打开 MULTIMODAL 并重启 ====="
sed -i 's/^FASTLLM_PREFIX_CACHE_MULTIMODAL=.*/FASTLLM_PREFIX_CACHE_MULTIMODAL=1/' "$PROF"
grep -n "^FASTLLM_PREFIX_CACHE_MULTIMODAL=" "$PROF" | sed 's/^/  /'
OLDPID=$(pgrep -f build-rw/apiserver | head -1)
echo "  旧后端 pid=$OLDPID"
tmux respawn-pane -k -t fastllm-prod:0.0 "$SCR/start_prod.sh $PROF" 2>/dev/null
# 就绪判定必须认"新进程 + 环境变量正确 + 能真出一次补全"。
# 只探 :8002/health 会命中**旧进程**(respawn 需要时间), 上一轮就是这么误判的;
# 只等 /health 也不够 —— 权重要到第一次真实请求才上卡。
echo "  等新进程"
NEWPID=""
for i in $(seq 1 120); do
  P=$(pgrep -f build-rw/apiserver | head -1)
  if [ -n "$P" ] && [ "$P" != "$OLDPID" ]; then
    V=$(tr "\0" "\n" < /proc/$P/environ 2>/dev/null | grep -oE "MULTIMODAL=[01]")
    [ "$V" = "MULTIMODAL=1" ] && { NEWPID=$P; echo "  新后端 pid=$P  $V  (第 $((i*5)) 秒)"; break; }
    [ -n "$V" ] && { echo "  [!] 新进程环境是 $V, 开关没生效, 中止"; exit 2; }
  fi
  sleep 5
done
[ -n "$NEWPID" ] || { echo "  [!] 没等到带 MULTIMODAL=1 的新进程, 中止"; exit 2; }
echo "  等权重加载"
for i in $(seq 1 100); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 6 http://127.0.0.1:8002/health)" = "200" ] && { echo "  后端应答(第 $((i*10)) 秒)"; break; }
  sleep 10
done

echo
echo "===== 3/5 跑实验 ====="
python3 "$SCR/mm_cache_experiment.py"
RC=$?

echo
echo "===== 4/5 处理结果 ====="
if [ "$RC" = 0 ]; then
  echo "  >> 通过: 多模态前缀缓存保持开启"
elif [ "$RC" = 2 ]; then
  echo "  >> 实验无效(B 未命中), 保守起见改回 0"
  sed -i 's/^FASTLLM_PREFIX_CACHE_MULTIMODAL=.*/FASTLLM_PREFIX_CACHE_MULTIMODAL=0/' "$PROF"
  tmux respawn-pane -k -t fastllm-prod:0.0 "$SCR/start_prod.sh $PROF" 2>/dev/null
  for i in $(seq 1 80); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 6 http://127.0.0.1:8002/health)" = "200" ] && break
    sleep 10
  done
else
  echo "  >> 不通过: 改回 MULTIMODAL=0 并重启"
  sed -i 's/^FASTLLM_PREFIX_CACHE_MULTIMODAL=.*/FASTLLM_PREFIX_CACHE_MULTIMODAL=0/' "$PROF"
  tmux respawn-pane -k -t fastllm-prod:0.0 "$SCR/start_prod.sh $PROF" 2>/dev/null
  for i in $(seq 1 80); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 6 http://127.0.0.1:8002/health)" = "200" ] && break
    sleep 10
  done
fi

echo
echo "===== 5/5 恢复两个 agent ====="
tmux send-keys -t proto-ui -l '请一直严格 review 所有新增提交、新增 pr、新增 issues 的情况并以严格、严谨的方法和语气进行回复。【如果没有人回复过的话】你可以用 gh-cli 作为我完成工作。请一直继续。'
sleep 1; tmux send-keys -t proto-ui Enter; sleep 3
tmux send-keys -t z3rm -l '以维护者的身份处理、决策你的行动。请一直严格 review 所有 z3rm 的新增提交、新增 pr、新增 issues 的情况并以严格、严谨的方法和语气进行回复。【如果没有人回复过的话】你可以用 gh-cli 作为我完成工作。请一直继续。'
sleep 1; tmux send-keys -t z3rm Enter; sleep 3
for S in proto-ui z3rm; do
  echo "  $S: $(tmux capture-pane -p -J -t "$S" -S -3 2>/dev/null | grep -oE 'Loop (running|paused)' | tail -1)"
done
echo "  最终 MULTIMODAL=$(grep -oE '^FASTLLM_PREFIX_CACHE_MULTIMODAL=[01]' "$PROF" | cut -d= -f2)"
exit "$RC"
