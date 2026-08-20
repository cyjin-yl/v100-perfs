#!/usr/bin/env python3
"""对指定张量做 sha256，判断不同档位文件里该张量是否逐字节相同。

用法: hash_tensor.py <gguf> <张量名> [张量名...]
"""
import sys, hashlib
from gguf import GGUFReader

path = sys.argv[1]
names = set(sys.argv[2:])
r = GGUFReader(path, "r")
for t in r.tensors:
    if t.name in names:
        b = t.data.tobytes()
        h = hashlib.sha256(b).hexdigest()[:16]
        print(f"{t.name}\t{t.tensor_type.name}\t{len(b)}\t{h}")
