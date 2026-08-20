#!/usr/bin/env bash
# 对**当前正在服务的**模型跑一遍验收探针套件。
#
# 拆成独立脚本是因为上一版把"验收"和"换模型/占卡"焊在一起, 结果一个鉴权 bug
# 让整套跑空还报出"IQ4_XS 不得上生产"的结论 —— 闸门分不清"模型没过"和
# "闸门自己坏了", 这比没有闸门更危险。拆开之后可以先拿**已知没问题的生产模型**
# 验证套件本身, 顺便拿到基线数字, 再去占卡测候选。
#
# 前置条件按"硬件失败"处理并以 exit 2 退出, 与"模型没通过"(exit 1)严格区分。
set -uo pipefail
ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM
SCR=/home/ezra/projects/EzraVastLLM/scripts
BASE="${BASE:-http://127.0.0.1:8000}"
BACKEND="${BACKEND:-http://127.0.0.1:8002}"
TAG="${TAG:-run}"
OUTDIR="${OUTDIR:-$ROOT/imatrix/logs}"
READY_TIMEOUT="${READY_TIMEOUT:-1500}"     # 秒; 15GB 机械盘加载约 5~9 分钟
mkdir -p "$OUTDIR"

# ---- token: .env 里带引号, 必须清洗; 且清洗完要**实测**能用 ----------------
AUTH=$(grep -m1 -oE '^AUTH_TOKEN=.*' "$ROOT/.env" 2>/dev/null | cut -d= -f2- | tr -d "\"' ")
[ -n "$AUTH" ] || { echo "[前置失败] .env 里读不到 AUTH_TOKEN" >&2; exit 2; }

echo "== 验收探针套件  tag=$TAG  base=$BASE =="

# ---- 就绪判定: proxy 的 /health 一起来就 200, **不能**当作后端就绪 ---------
# 上一版正是栽在这: 20 秒就判"就绪", 而权重要 8 分钟。这里改成两级 ——
#   1) 后端自己的 /health(:8002) 通
#   2) 真发一次 1 token 的补全并拿到 200 + 非空内容
# 只有第二条成立才算真就绪。
echo "-- 等待后端真正就绪(最多 $((READY_TIMEOUT/60)) 分钟) --"
t0=$(date +%s); ready=0
while [ $(( $(date +%s) - t0 )) -lt "$READY_TIMEOUT" ]; do
  bh=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$BACKEND/health" 2>/dev/null)
  if [ "$bh" = "200" ]; then
    code=$(curl -s -o /tmp/ready-$TAG.json -w '%{http_code}' --max-time 120 \
      -H "Authorization: Bearer $AUTH" -H 'Content-Type: application/json' \
      -X POST "$BASE/v1/chat/completions" -d '{"model":"qwen3.8-fastllm","max_tokens":1,
        "temperature":0,"stream":false,"reasoning_effort":"none",
        "chat_template_kwargs":{"enable_thinking":false},
        "messages":[{"role":"user","content":"hi"}]}' 2>/dev/null)
    if [ "$code" = "200" ]; then
      ready=1; echo "   就绪, 用时 $(( ($(date +%s)-t0)/60 )) 分 $(( ($(date +%s)-t0)%60 )) 秒"; break
    fi
    [ "$code" = "401" ] && { echo "[前置失败] 鉴权被拒(401) —— 是套件的问题, 不是模型的问题" >&2; exit 2; }
  fi
  sleep 10
done
[ "$ready" = 1 ] || { echo "[前置失败] 后端在 $READY_TIMEOUT 秒内没就绪" >&2; exit 2; }

# ---- 当前服务的到底是哪份权重(避免拿错模型测出对的结论) -------------------
echo
echo "-- 在测的模型 --"
ps -eo args | grep "[b]uild-rw/apiserver" | grep -oE '\-\-path [^ ]+' | awk '{n=split($2,a,"/");print "   权重 "a[n]}'
ps -eo args | grep "[b]uild-rw/apiserver" | grep -oE '\-\-kv_cache_dtype [^ ]+|\-\-batch [^ ]+|\-\-tokens [^ ]+' | sed 's/^/   /'

# ---- 算子路由普查 ---------------------------------------------------------
echo
echo "-- 算子路由普查(非零计数=真的走了那条 kernel) --"
curl -s --max-time 15 "$BACKEND/props" -o "$OUTDIR/props-$TAG.json" 2>/dev/null
python3 - "$OUTDIR/props-$TAG.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"   取 /props 失败: {e}"); raise SystemExit
r = d.get("kernel_routes") or d.get("kernelRoutes") or {}
if not r:
    print("   /props 里没有 kernel_routes 字段"); raise SystemExit
for k, v in sorted(r.items()):
    print(f"   {k:<38}{v}")
PY

# ---- 验收 -----------------------------------------------------------------
echo
echo "-- 逐字抄写(最贴近 agent 的真实失败模式) --"
python3 "$SCR/copy_fidelity_http.py" --base-url "$BASE" --token "$AUTH" \
  --json "$OUTDIR/copy-fidelity-$TAG.json" 2>&1 | sed 's/^/   /'
COPY=${PIPESTATUS[0]}

echo
echo "-- 工具名保真(三种命名风格) --"
python3 "$SCR/toolname_probe.py" --base-url "$BASE" --token "$AUTH" \
  --json "$OUTDIR/toolname-$TAG.json" 2>&1 | sed 's/^/   /'
TOOL=${PIPESTATUS[0]}

echo
echo "-- decode 吞吐 --"
S=$(date +%s%N)
curl -s --max-time 900 -o "$OUTDIR/decode-$TAG.json" \
  -H "Authorization: Bearer $AUTH" -H 'Content-Type: application/json' \
  -X POST "$BASE/v1/chat/completions" -d '{"model":"qwen3.8-fastllm","max_tokens":300,
    "temperature":0,"stream":false,"reasoning_effort":"none",
    "chat_template_kwargs":{"enable_thinking":false},
    "messages":[{"role":"user","content":"从 1 数到 150, 用逗号分隔, 只输出数字。"}]}' 2>/dev/null
E=$(date +%s%N)
python3 - "$OUTDIR/decode-$TAG.json" "$(( (E-S)/1000000 ))" <<'PY'
import json, sys
ms = int(sys.argv[2])
try:
    d = json.load(open(sys.argv[1])); u = d.get("usage") or {}
    ct = u.get("completion_tokens") or 0
    print(f"   completion={ct} tok  prompt={u.get('prompt_tokens')} tok  "
          f"墙钟={ms/1000:.1f}s  端到端={ct/(ms/1000):.2f} tok/s"
          if ct else f"   没拿到 usage: {str(d)[:200]}")
except Exception as e:
    print(f"   解析失败: {e}")
PY

echo
echo "-- 汇总 --"
echo "   抄写 exit=$COPY   工具名 exit=$TOOL   (0=全过)"
[ "$COPY" = 0 ] && [ "$TOOL" = 0 ] && { echo "   >> 全部通过"; exit 0; }
echo "   >> 有未通过项(模型层面)"
exit 1
