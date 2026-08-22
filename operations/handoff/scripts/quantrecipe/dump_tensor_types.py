#!/usr/bin/env python3
"""导出 GGUF 的张量清单（名字 / 类型 / 形状 / 元素数 / 字节数）为 TSV。

用法: dump_tensor_types.py <in.gguf> <out.tsv>
只读元数据 + 张量表，不加载权重数据。
"""
import sys
from gguf import GGUFReader

def main():
    path, out = sys.argv[1], sys.argv[2]
    r = GGUFReader(path, "r")
    with open(out, "w") as f:
        f.write("name\ttype\tshape\tn_elements\tn_bytes\n")
        for t in r.tensors:
            shape = ",".join(str(int(x)) for x in t.shape)
            f.write(f"{t.name}\t{t.tensor_type.name}\t{shape}\t{int(t.n_elements)}\t{int(t.n_bytes)}\n")
    print(f"{path} -> {out}: {len(r.tensors)} 张量")

if __name__ == "__main__":
    main()
