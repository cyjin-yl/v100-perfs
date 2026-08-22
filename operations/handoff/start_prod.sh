#!/usr/bin/env bash
# 生产重启: thinking_proxy(:8000) + 它自己拉起的 fastllm apiserver(:8002)。
#
# 为什么不能只重启代理:
#   代理以 FASTLLM_OWNED=1 拉起后端。但 `tmux respawn-pane` 只杀掉代理进程,
#   后端会变成**孤儿**继续占着 27~31GiB 显存。下一次启动时新后端申请不到显存,
#   表现成"重启起不来", 且日志里看不出原因。所以必须先显式清理孤儿后端。
#
# 用法: start_prod.sh [profile路径]   环境变量 MTP=0|2 可覆盖 profile 里的设置
set -uo pipefail

PROFILE="${1:-/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/v100-perfs/runtime/fastllm-native-profiles/q38-PROD-cyber-q5-mtp2-turbo4-sm70.env}"
ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM
PY=/home/ezra/.conda/envs/tsenv/bin/python  # 与线上一致; .venv-1cat 缺 uvicorn/fastapi

# 1) 先停代理(它会尝试带走自己的后端), 再补杀孤儿后端
for pid in $(pgrep -f 'thinking_proxy\.py' || true); do
  echo "[start_prod] 停止代理 pid=$pid"; kill "$pid" 2>/dev/null
done
sleep 3

for pid in $(pgrep -f 'build-rw/apiserver' || true); do
  echo "[start_prod] 清理残留后端 pid=$pid"
  kill "$pid" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then echo "[start_prod] 未能优雅退出, 强杀"; kill -9 "$pid" 2>/dev/null; fi
done
sleep 2
echo "[start_prod] 显存: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

# 1.5) 载入 .env 里的 AUTH_TOKEN。
# 以前它是靠"启动这个脚本的 shell 恰好 source 过 .env"传进来的隐式状态。
# 一旦换个方式拉起(例如 tmux respawn-pane 显式指定命令、nohup、cron),
# 这个环境就没了, proxy 会打 "auth token: EMPTY - INSECURE" 静默地把鉴权关掉 ——
# 服务照常跑, 只是不再需要令牌, 从日志正文里看不出来。
# 所以改成脚本自己读, 不依赖调用者的环境。
if [ -f "$ROOT/.env" ]; then
  set -a; . "$ROOT/.env"; set +a
fi
if [ -n "${AUTH_TOKEN:-}" ]; then
  echo "[start_prod] AUTH_TOKEN: 已载入(${#AUTH_TOKEN} 字符)"
else
  echo "[start_prod] [!] AUTH_TOKEN 为空 —— proxy 将以无鉴权模式运行"
fi

# 2) 载入 profile。profile 里带引号的 FASTLLM_BACKEND_COMMAND 需要 source 才能正确解析
echo "[start_prod] profile=$PROFILE"
set -a; . "$PROFILE"; set +a

# 3) MTP 覆盖(排障用; 平时不要关 —— 没有 MTP 吞吐不够用)
if [ -n "${MTP:-}" ]; then export FASTLLM_QWEN35_ENABLE_MTP="$MTP"; fi
echo "[start_prod] MTP=${FASTLLM_QWEN35_ENABLE_MTP:-未设置}  LOCK_DEBUG=${FASTLLM_MTP_LOCK_DEBUG:-0}"

cd "$ROOT"
exec "$PY" "$ROOT/thinking_proxy.py"
