#!/usr/bin/env bash
# 部署 MTP 归因埋点, 并把多模态前缀缓存翻成开启。
#
# 与上一次部署的区别: **不做输出质量实验, 只读计数器**。
# 因为"开 MULTIMODAL=1 安不安全"已经从结构上关闭了 ——
#   delta = maxPos + 1 - seqLen; 最后一张图之后 mRoPE 每 token 前进 1,
#   即 position[i] = i + C(C 为此前图像块累积的压缩量); 而 maxPos 必在
#   最后一个 token 取到(每张图后 currentPos += max(gridH,gridW)/merge,
#   所以图后的文本位置总超过该图自身)。故 delta == C 恒成立, 与几何无关。
#   守卫又把允许的情形限制在"残段纯文本", 正是该论证覆盖的范围。
# 剩下的问题只是"有没有用", 那没有正确性风险。
#
# **不暂停 agent** —— 这次恰恰需要它们产生 vision 流量, 否则计数器读不出东西。
#
# 判据(全部读 /props, 不做时序推断):
#   mtp_steps_multimodal != 0            F9: MTP 真的在多模态上跑了
#   vision 请求 hit > 0                  F10/F7: 前缀真的复用上了
#   mgr-lookup-mismatch == 0             管理器路由没错配
#   mm-remainder-img                     读数: 守卫是不是把每一轮都挡了
#                                        (若每轮都挡 => 收益为零但无害)
set -uo pipefail
ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM
SCR=/home/ezra/projects/EzraVastLLM/scripts
BR=$ROOT/fastllm/build-rw
PROF=$ROOT/v100-perfs/runtime/fastllm-native-profiles/q38-PROD-cyber-iq4xs-imatrix-mtp2-sm70.env
BAKBIN=$ROOT/imatrix/apiserver.bak-before-mtpattr
BAKPROF=$PROF.bak-before-mtpattr
LOG=$ROOT/imatrix/logs/deploy-mtp-attr.log
: > "$LOG"; exec > >(tee -a "$LOG") 2>&1

echo "===== 1/4 备份并重建 ====="
cp -f "$BR/apiserver" "$BAKBIN" && cp -f "$PROF" "$BAKPROF"
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
if ! cmake --build . --target apiserver -j 6 2>&1 | grep -E "error|Error|\[100%\]" | tail -6; then
  echo "  [!] 重建失败 -> 还原"
  cp -f "$BAKBIN" "$BR/apiserver"
  tmux respawn-pane -k -t fastllm-prod:0.0 "$SCR/start_prod.sh $PROF" 2>/dev/null
  exit 1
fi
echo "  重建用时 $(( ($(date +%s)-t0)/60 )) 分钟"

echo
echo "===== 2/4 翻开 MULTIMODAL 并拉起 ====="
sed -i 's/^FASTLLM_PREFIX_CACHE_MULTIMODAL=.*/FASTLLM_PREFIX_CACHE_MULTIMODAL=1/' "$PROF"
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
for i in $(seq 1 150); do
  c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 120 --noproxy '*' \
      -H "Authorization: Bearer $A" -H 'Content-Type: application/json' \
      -X POST http://127.0.0.1:8000/v1/chat/completions \
      -d '{"model":"qwen3.8-27b","max_tokens":1,"temperature":0,"stream":false,"messages":[{"role":"user","content":"hi"}]}' 2>/dev/null)
  [ "$c" = "200" ] && { echo "  就绪(第 $((i*10)) 秒)"; break; }
  sleep 10
done

echo
echo "===== 3/4 等 agent 跑几轮真实 vision 流量(最多 25 分钟) ====="
for i in $(seq 1 100); do
  n=$(curl -s --max-time 10 --noproxy '*' http://127.0.0.1:8002/props 2>/dev/null | \
      python3 -c 'import json,sys
try: d=json.load(sys.stdin)
except Exception: print(0); raise SystemExit
print(int((d.get("mtp_attribution") or d).get("mtp_steps_multimodal", 0) or 0))' 2>/dev/null)
  [ -n "$n" ] && [ "$n" -gt 0 ] 2>/dev/null && { echo "  mtp_steps_multimodal=$n (第 $((i*15)) 秒)"; break; }
  [ $((i % 8)) -eq 0 ] && echo "  [$((i*15))s] mtp_steps_multimodal=${n:-?}"
  sleep 15
done

echo
echo "===== 4/4 读判据 ====="
curl -s --max-time 20 --noproxy '*' http://127.0.0.1:8002/props -o /tmp/props.json 2>/dev/null
python3 - /tmp/props.json <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"  取 /props 失败: {e}"); raise SystemExit(2)
def find(k):
    stack=[d]
    while stack:
        o=stack.pop()
        if isinstance(o,dict):
            if k in o: return o[k]
            stack.extend(o.values())
        elif isinstance(o,list): stack.extend(o)
    return None
for k in ("mtp_steps","mtp_steps_multimodal","mtp_multimodal_prefill_done",
          "mtp_multimodal_decode_steps"):
    print(f"  {k:<32}{find(k)}")
r = find("kernel_routes") or {}
for k,v in sorted(r.items()):
    if v and k.startswith("attn"): print(f"  {k:<32}{v}")
PY
L=$ROOT/v100-perfs/runtime/fastllm-native-profiles/logs/backend-PROD-cyber-iq4xs.log
N=$(grep -n "Server\] ready" "$L" | tail -1 | cut -d: -f1)
echo "  ── 日志侧 ──"
awk -v n="$N" 'NR>n' "$L" | grep -oE "mgr-lookup-mismatch=[0-9]+" | tail -1 | sed 's/^/    /'
awk -v n="$N" 'NR>n' "$L" | grep -oE "miss\{[^}]*\}" | tail -1 | tr ' ' '\n' | grep -E "remainder|mm-|delta" | sed 's/^/    /'
awk -v n="$N" 'NR>n' "$L" | grep -oE "req#[0-9]+ total=[0-9]+ hit=[0-9]+ layer=[^ ]* miss=[a-z-]*" | tail -6 | sed 's/^/    /'
awk -v n="$N" 'NR>n' "$L" | grep -oE "\[Qwen3.5 MTP\] mm=[01][^|]*" | tail -2 | cut -c1-150 | sed 's/^/    /'
echo "  错误 $(awk -v n="$N" 'NR>n' "$L" | grep -ciE 'FastLLM Error|Assert|illegal|bind error') 条"
exit 0
