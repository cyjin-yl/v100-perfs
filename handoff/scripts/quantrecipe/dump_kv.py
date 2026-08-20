#!/usr/bin/env python3
"""导出 GGUF 的 KV 元数据（只打印感兴趣的键）。"""
import sys
from gguf import GGUFReader

KEYS_HINT = ("general.", "quantize", "qwen35.expert", "qwen35.attention", "qwen35.block", "split.")

def val(f):
    try:
        return f.contents()
    except Exception:
        return "<unreadable>"

path = sys.argv[1]
r = GGUFReader(path, "r")
print(f"### {path}")
for k, f in r.fields.items():
    if any(k.startswith(p) or p in k for p in KEYS_HINT):
        v = val(f)
        s = str(v)
        if len(s) > 200:
            s = s[:200] + "..."
        print(f"  {k} = {s}")
