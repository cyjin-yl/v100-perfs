#!/usr/bin/env bash
# 一直跟随**当前**后端进程的 stdout 日志。
#
# 为什么不直接 tail -f 某个固定文件: 每个 profile 写自己的日志
# (backend-PROD-cyber-q5.log / -iq4xs.log ...), 一换模型, 固定的 tail 就永远
# 停在旧文件上 —— 看起来像"服务卡住了", 其实只是在看一份已经不再写入的日志。
# 这里改成从 /proc/<apiserver-pid>/fd/1 反查它真正在写哪个文件, 后端一换就跟过去。
while true; do
  P=$(pgrep -f 'build-rw/apiserver' | head -1)
  if [ -z "$P" ]; then
    echo "[follow] 等待后端进程 ..."; sleep 5; continue
  fi
  F=$(readlink -f /proc/$P/fd/1 2>/dev/null)
  case "$F" in
    *.log) : ;;
    *) echo "[follow] pid=$P 的 stdout 不是文件($F), 5 秒后重试"; sleep 5; continue ;;
  esac
  echo "[follow] === 跟随 pid=$P -> $(basename "$F") ==="
  # --pid=$P: 后端一退出 tail 就结束, 外层循环立刻去找新的
  tail -n 40 -F --pid="$P" "$F" 2>/dev/null
  echo "[follow] === 后端 pid=$P 已退出, 找新的 ==="
  sleep 2
done
