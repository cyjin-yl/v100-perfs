#!/usr/bin/env bash
# 校验下载的 GGUF 是否完整 —— 按**仓库真实字节数**比对, 不看 GGUF 头。
#
# 为什么不能看头: GGUF 的魔数、版本、张量数都在文件最开头, 一个下到一半的
# 文件照样能解析出 "tensors=851" 看起来完全正常。2026-08-20 实测踩过:
# Qwen3.8-27B-Uncensored-Cyber-Q5_K_M.gguf 只下了 11.87GB / 应有 21.27GB,
# 头部检查全绿, 差点被当成完好的基线拿去做量化 A/B。
#
# 根因提醒: 本机直连 huggingface.co 超时(实测 HTTP 000), 必须
# export HF_ENDPOINT=https://hf-mirror.com。没设镜像时 hf download 会**静默**
# 停住 —— 没有进程、没有报错、文件停在半截, 极易误判成"下完了"。
set -uo pipefail
REPO="${REPO:-philbert440/Qwen3.8-27B-Uncensored-Cyber-GGUF}"
DIR="${1:?用法: verify_downloads.sh <本地目录>}"
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
curl -s -m 40 "https://hf-mirror.com/api/models/$REPO/tree/main?recursive=1" -o "$TMP"
python3 - "$DIR" "$TMP" <<'PY'
import json, os, sys
d, meta = sys.argv[1], sys.argv[2]
try:
    items = json.load(open(meta, encoding='utf-8'))
except Exception as e:
    print(f"  取仓库文件列表失败: {e}"); raise SystemExit(2)
remote = {f['path']: f.get('size', 0) for f in items
          if isinstance(f, dict) and f.get('path', '').endswith('.gguf')}
if not remote:
    print("  仓库里没找到 .gguf(接口返回格式可能变了)"); raise SystemExit(2)
bad = 0
for name in sorted(os.listdir(d)):
    if not name.endswith('.gguf'):
        continue
    local = os.path.getsize(os.path.join(d, name))
    exp = remote.get(name)
    if exp is None:
        print(f"  ?  {name}: 仓库里没有同名文件"); continue
    if local == exp:
        print(f"  ok {name}  {local/1e9:.2f} GB")
    else:
        bad += 1
        print(f"  ✗  {name}  本地 {local/1e9:.2f} != 仓库 {exp/1e9:.2f} GB "
              f"({100.0*local/exp:.0f}%)")
missing = [n for n in remote if not os.path.exists(os.path.join(d, n))]
if missing:
    print(f"  (仓库有而本地没有: {len(missing)} 个)")
raise SystemExit(1 if bad else 0)
PY
