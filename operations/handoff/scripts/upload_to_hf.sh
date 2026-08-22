#!/usr/bin/env bash
# 把 imatrix 量化产物上传到 HuggingFace。
#
# 为什么要走代理:
#   这台机器**直连 huggingface.co 超时**(实测站点与 API 都是 HTTP 000 / 15s)。
#   下载能用 HF_ENDPOINT=https://hf-mirror.com 绕过, 但镜像是只读的, 传不上去。
#   本机 127.0.0.1:10808 有 SOCKS5/HTTP 两用代理, 实测经它访问 huggingface.co
#   返回 200(1.9s), whoami 也能过 —— 所以上传必须显式带上代理环境变量,
#   而且**不能**设 HF_ENDPOINT(那会把上传打到镜像上)。
#
# 不上传校准语料:
#   calib-agentic.txt 抽自真实 omp 会话日志, 含真实仓库路径、shell 命令和对话
#   内容, 属于私人工作数据。只发布 imatrix 结果文件, 方法可复现、原始数据不外泄。
set -uo pipefail

ROOT=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM
HF=$ROOT/.venv-1cat/bin/hf
REPO="${REPO:-cyjin-yl/Qwen3.8-27B-Uncensored-Cyber-agentic-imatrix-GGUF}"
QDIR="${QDIR:-$ROOT/models/Qwen3.8-27B-Uncensored-Cyber-GGUF/imatrix}"
# 必须是 v2。v1(imatrix-agentic.gguf)漏了 --parse-special, 语料里的
# <|im_start|>/<tool_call> 被当字面文本切碎, 特殊 token 从未进入统计 —— 它
# 留在磁盘上只为将来做消融对比, **不能发布**。
IMAT="${IMAT:-$ROOT/imatrix/imatrix-agentic-v2.gguf}"
CARD="${CARD:-$HOME/projects/EzraVastLLM/hf-upload/README.md}"

# 关键: 走代理, 且确保 HF_ENDPOINT 不生效(镜像只读, 会导致上传失败)
export HTTPS_PROXY=http://127.0.0.1:10808
export HTTP_PROXY=http://127.0.0.1:10808
unset HF_ENDPOINT

# 关掉 xet 后端。
# 现象: 传到 19% 卡死 132 分钟, rchar/read_bytes 完全不动, 唯一的连接是
#   CLOSE-WAIT -> 代理。xet 日志里全是
#     Retryable Client Error: cas::upload_xorb api call failed
#     Request error ... url: "https://cas-server.xethub.hf.co/..."
#   重试到第 5 次、并发降到 1, 然后彻底静默。
# 原因: xet 走的是 **另一个域名** cas-server.xethub.hf.co, 与 huggingface.co
#   不是一条链路 —— 普通 API(README 那次)经代理完全正常, 只有这个域名不通。
# 关掉后走经典分片上传, 全部流量都在 huggingface.co 上, 与已验证可用的路径一致。
export HF_HUB_DISABLE_XET=1

# 上传闸门: 必须显式 CONFIRM_UPLOAD=1 才会真的传。
#
# 为什么加这道门: v1 的 imatrix 是在**漏了 --parse-special** 的情况下算的 ——
# 语料用 chat template 标记(<|im_start|>/<tool_call>)渲染, 没这个开关时
# llama.cpp 把它们当字面文本切成 "<" "|" "im" "_start"..., 也就是说
# "让特殊 token 进入校准分布"这个核心设计目标在 v1 根本没达成。
# 这种半成品不该公开发布。流水线会自动走到上传这一步, 所以在这里拦住,
# 等 v2(大语料 + --parse-special)出来、三方对比有数字之后再显式放行。
if [ "${CONFIRM_UPLOAD:-0}" != "1" ]; then
  echo "上传已拦截: 需要 CONFIRM_UPLOAD=1 才会执行。" >&2
  echo "  这是一道防手滑的闸门 —— 发布是对外的、不可撤回的动作。" >&2
  echo "  放行前确认: (1) 用的是 v2 imatrix(带 --parse-special);" >&2
  echo "              (2) 大文件要发布前, 三方留出集对比已有数字。" >&2
  echo "    CONFIRM_UPLOAD=1 ~/projects/EzraVastLLM/scripts/upload_to_hf.sh" >&2
  exit 0
fi

echo "== 身份 =="
$HF auth whoami || { echo "未登录, 中止" >&2; exit 1; }

echo
echo "== 待上传文件 =="
FILES=()
# 注意通配写法: "*imatrix*" 会把 "-noimatrix-" 那个文件也匹配进来, 于是对照组
# 被列两次、白传 21GB。用 "-imatrix-" 才安全 —— "-noimatrix-" 里 imatrix 前面
# 是 o 不是 -, 不会误匹配。下面再加一层按 basename 去重兜底。
# ONLY: 只上传 basename 匹配该正则的文件(imatrix 结果文件永远传)。
# 加它的原因: 全量是 6 个 GGUF 约 106GB, 经代理传要很久, 而其中
# noimatrix 那两个只是我们自己 A/B 的对照组 —— 任何人用 stock llama.cpp
# 都能复现, 没有发布价值。默认仍是全量, 需要分批时用 ONLY 收窄。
ONLY="${ONLY:-.}"
declare -A SEEN=()
for f in "$QDIR"/*-imatrix-*.gguf "$QDIR"/*-noimatrix-*.gguf "$IMAT"; do
  [ -f "$f" ] || continue
  b=$(basename "$f")
  if [ "$f" != "$IMAT" ] && ! echo "$b" | grep -qE "$ONLY"; then continue; fi
  [ -n "${SEEN[$b]:-}" ] && continue
  SEEN[$b]=1
  FILES+=("$f")
  printf "  %8.2f GB  %s\n" "$(stat -c %s "$f" | awk '{print $1/1e9}')" "$b"
done
[ -f "$CARD" ] && echo "           README.md"
if [ ${#FILES[@]} -eq 0 ]; then
  echo "没有可上传的产物 —— 量化还没跑完?" >&2
  exit 1
fi

echo
echo "== 创建仓库(已存在则跳过) =="
$HF repo create "$REPO" --repo-type model 2>&1 | tail -2 || true

echo
echo "== 先传 README(小文件, 快速验证通道) =="
[ -f "$CARD" ] && $HF upload "$REPO" "$CARD" README.md --repo-type model --commit-message "模型卡: 说明校准语料构成与量化参数" 2>&1 | tail -3

for f in "${FILES[@]}"; do
  echo
  echo "== 上传 $(basename "$f")  ($(du -h "$f" | cut -f1)) =="
  START=$(date +%s)
  if $HF upload "$REPO" "$f" "$(basename "$f")" --repo-type model \
        --commit-message "上传 $(basename "$f")" 2>&1 | tail -4; then
    echo "  完成, 用时 $(( ($(date +%s)-START)/60 )) 分钟"
  else
    echo "  失败: $(basename "$f") —— 可单独重跑本脚本续传" >&2
  fi
done

echo
echo "== 完成: https://huggingface.co/$REPO =="
