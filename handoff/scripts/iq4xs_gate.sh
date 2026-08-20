#!/usr/bin/env bash
# IQ4_XS 上生产的准入闸门。
#
# 约束(用户明确要求): 在验证通过之前**绝不**把生产切到 IQ4_XS —— 切坏了会
# 直接耽误 tmux 里那几个真实 agent 干活。所以本脚本只做"验证 + 归还生产",
# 任何情况下都不改生产 profile。切换是人看过结论之后的单独决定。
#
# 它接管 gpu_verify_window.sh 的第 4 步。为什么要接管:
#   原窗口第 4 步用 llama-cli 做逐字抄写探针, 有两个致命缺陷 ——
#     a) llama-cli 的 --display-prompt 默认 true, 会把 prompt 原样打印,
#        而判据是在输出里 grep 期望串, prompt 自带一份 => **恒为通过**。
#        这比没有验证更糟: 它会给出"IQ4_XS 抄写没问题"的假结论。
#     b) 测的是 llama.cpp, 而生产跑 fastllm —— 分词器、预分词正则、采样、
#        MTP 草稿接受逻辑全不同, 结论不能外推。
#   所以: PPL 那三组照常跑完(那部分是对的且很便宜), 到第 4 步就拦下来,
#   改成把 IQ4_XS 用 fastllm 拉起来, 走真实 HTTP 栈验收。
#
# 为什么能"拦": 窗口的 trap 会在退出时恢复生产, 而我们此刻还要用 GPU。
#   所以用 SIGKILL 跳过它的 trap, 由本脚本自己的 trap 负责归还生产 ——
#   归还这件事只能有一个负责人。
set -uo pipefail

ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM
SCR=/home/ezra/projects/EzraVastLLM/scripts
PROF_DIR=$ROOT/v100-perfs/runtime/fastllm-native-profiles
PROD=$PROF_DIR/q38-PROD-cyber-q5-mtp2-turbo4-sm70.env
VERIFY=$PROF_DIR/q38-VERIFY-cyber-iq4xs-imatrix-nomtp-sm70.env
WLOG=$ROOT/imatrix/logs/gpu-window.log
OUT=$ROOT/imatrix/logs/iq4xs-gate.log
mkdir -p "$(dirname "$OUT")"; : > "$OUT"
exec > >(tee -a "$OUT") 2>&1

AUTH=$(grep -oE '^AUTH_TOKEN=.*' "$ROOT/.env" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"'"'"' ')

restored=0
restore_prod() {
  [ "$restored" = 1 ] && return; restored=1
  echo
  echo "===== 归还生产 (Q5_K_M+MTP, 已验证过的那份) ====="
  pkill -f 'thinking_proxy\.py' 2>/dev/null; sleep 2
  for pid in $(pgrep -f 'build-rw/apiserver' || true); do
    kill "$pid" 2>/dev/null
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
  done
  sleep 3
  # 显式带命令: 不带命令的 respawn-pane 会重跑 pane 原命令, profile 参数无效
  tmux respawn-pane -k -t fastllm-prod:0.0 "$SCR/start_prod.sh" 2>/dev/null
  echo "  已在 fastllm-prod:0.0 拉起生产(权重加载约 8 分钟)"
}
trap restore_prod EXIT INT TERM

# ---------------------------------------------------------------- 1 拦截
echo "===== 1/5 等 PPL 三组跑完, 在探针阶段之前拦截 ====="
for i in $(seq 1 240); do          # 最多 80 分钟
  if grep -q "4/4" "$WLOG" 2>/dev/null; then
    echo "  窗口已进入第 4 步(llama-cli 探针) -> 拦截"; break
  fi
  if ! pgrep -f "gpu_verify_window.sh" >/dev/null 2>&1; then
    echo "  窗口进程已退出"; break
  fi
  sleep 20
done
echo "  ── 已拿到的 PPL ──"
grep -E "Final estimate: PPL" "$ROOT/imatrix/logs/gpu-verify.log" 2>/dev/null | sed 's/^/    /'
cat "$ROOT/imatrix/logs/gpu-verify.log.summary" 2>/dev/null | sed 's/^/    /'

# SIGKILL 跳过窗口自己的 trap —— 归还生产只能有一个负责人(本脚本)
pkill -9 -f "gpu_verify_window.sh" 2>/dev/null
pkill -9 -f "llama-perplexity"     2>/dev/null
pkill -9 -f "llama-cli"            2>/dev/null
sleep 3
echo "  空闲显存 $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"

# ---------------------------------------------------------------- 2 起验证后端
echo
echo "===== 2/5 用 fastllm 拉起 IQ4_XS(未 graft MTP, 验证专用) ====="
pkill -f 'thinking_proxy\.py' 2>/dev/null; sleep 2
pkill -f 'build-rw/apiserver' 2>/dev/null; sleep 5
nohup "$SCR/start_prod.sh" "$VERIFY" >> "$ROOT/imatrix/logs/iq4xs-verify-boot.log" 2>&1 &
echo "  等待就绪(15.0GB 权重从机械盘加载, 预计 5~9 分钟)"
up=0
for i in $(seq 1 120); do          # 最多 20 分钟
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8000/health 2>/dev/null)
  if [ "$code" = "200" ]; then up=1; echo "  就绪, 用时 $((i*10)) 秒"; break; fi
  sleep 10
done
[ "$up" = 1 ] || { echo "  [!] 后端未起来, 放弃验证并归还生产"; exit 2; }

# ---------------------------------------------------------------- 3 路由普查
echo
echo "===== 3/5 算子路由普查(确认真的走了 SM70 IQ4_XS MMQ 快路径) ====="
# profile 注释里写清楚了: IQ4_XS 理论上应该比 Q5_K_M **更快**(带宽瓶颈,
# 少读 26% 权重)。如果实测更慢, 结论是"没走到快路径"而不是"IQ4_XS 不行"。
curl -s --max-time 10 http://127.0.0.1:8002/props 2>/dev/null \
  | python3 -c 'import json,sys
try: d=json.load(sys.stdin)
except Exception as e: print("  取 /props 失败:",e); raise SystemExit
r=d.get("kernel_routes") or d.get("kernelRoutes") or {}
if not r: print("  /props 没有 kernel_routes 字段"); raise SystemExit
for k,v in sorted(r.items()):
    if v: print(f"    {k:<34} {v}")' || echo "  (跳过)"

# ---------------------------------------------------------------- 4 验收
echo
echo "===== 4/5 验收 ====="
echo "  ── 4a 逐字抄写(最贴近真实失败模式) ──"
python3 "$SCR/copy_fidelity_http.py" --token "$AUTH" \
  --json "$ROOT/imatrix/logs/iq4xs-copy-fidelity.json" 2>&1 | sed 's/^/    /'
COPY=$?

echo "  ── 4b 工具名保真(三种命名风格) ──"
python3 "$SCR/toolname_probe.py" --token "$AUTH" \
  --json "$ROOT/imatrix/logs/iq4xs-toolname.json" 2>&1 | sed 's/^/    /'
TOOL=$?

echo "  ── 4c 目标验收表(工具/乱码/思考/流式) ──"
python3 "$SCR/goal_acceptance.py" --token "$AUTH" 2>&1 | tail -40 | sed 's/^/    /'
GOAL=$?

echo "  ── 4d decode 吞吐(用户明确说过没有吞吐就不可用) ──"
curl -s --max-time 900 -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $AUTH" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-fastllm","max_tokens":256,"temperature":0,"stream":false,
       "reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":false},
       "messages":[{"role":"user","content":"从 1 数到 120, 用逗号分隔, 只输出数字。"}]}' \
  -w '\n    HTTP=%{http_code} 总耗时=%{time_total}s\n' -o /tmp/iq4xs-decode.json 2>/dev/null
python3 -c 'import json
try:
    d=json.load(open("/tmp/iq4xs-decode.json"))
    u=d.get("usage") or {}
    print(f"    completion_tokens={u.get(\"completion_tokens\")} prompt_tokens={u.get(\"prompt_tokens\")}")
except Exception as e: print("    解析失败:",e)'

# ---------------------------------------------------------------- 5 结论
echo
echo "===== 5/5 结论 ====="
echo "  抄写保真 exit=$COPY   工具名 exit=$TOOL   验收表 exit=$GOAL   (0=全过)"
if [ "$COPY" = 0 ] && [ "$TOOL" = 0 ] && [ "$GOAL" = 0 ]; then
  echo "  >> IQ4_XS 三项全过。是否切生产由人决定, 本脚本不切。"
else
  echo "  >> 有未通过项, IQ4_XS **不得**上生产。证据见 $OUT"
fi
echo "  完整日志 $OUT"
exit 0
