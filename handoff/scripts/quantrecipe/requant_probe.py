#!/usr/bin/env python3
"""单张量重量化探针: 直接调 libggml 的 ggml_quantize_chunk, 不跑整模型量化。

为什么这么做:
  想知道"把某类张量换个档位, 误差会怎么变", 唯一诚实的办法是真量化一遍。
  但整模型量化要 30 分钟 + 16GB 磁盘。而 llama.cpp 的量化内核就是
  ggml_quantize_chunk(type, f32_src, dst, start, nrows, n_per_row, imatrix),
  llama-quant.cpp 里对每个张量调的就是它。直接 ctypes 调同一个符号,
  传同一份 imatrix(= in_sum2/counts, 与 quantize.cpp 的归一化一致),
  得到的就是**逐比特相同**的结果, 单张量只要几秒。

  可信度不靠嘴说: --validate 会拿已落地的 GGUF 里同名张量做交叉校验,
  harness 算出的误差必须与真实产物一致, 不一致就别信这脚本的任何结论。
"""
import argparse, ctypes, re, sys
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
from gguf.constants import GGMLQuantizationType as QT

GGML_TYPE = {"F32":0, "Q8_0":8, "Q5_K":13, "Q6_K":14, "IQ4_NL":20, "IQ4_XS":23}
BLK = {"Q5_K":(256,176), "Q6_K":(256,210), "Q8_0":(32,34), "IQ4_XS":(256,136), "IQ4_NL":(32,18)}

def load_lib(path):
    lib = ctypes.CDLL(path)
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                                        ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                        ctypes.c_int64, ctypes.POINTER(ctypes.c_float)]
    lib.ggml_quantize_init.argtypes = [ctypes.c_int]
    return lib

def quantize(lib, src_f32, nrows, n_per_row, tname, imat):
    """调 ggml 内核量化, 返回原始字节。"""
    t = GGML_TYPE[tname]
    lib.ggml_quantize_init(t)
    n, b = BLK[tname]
    dst = (ctypes.c_char * (nrows * (n_per_row // n) * b))()
    src = np.ascontiguousarray(src_f32, dtype=np.float32)
    im = np.ascontiguousarray(imat, dtype=np.float32)
    got = lib.ggml_quantize_chunk(t,
            src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(dst, ctypes.c_void_p), 0, nrows, n_per_row,
            im.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    assert got == len(dst), f"字节数不符: {got} != {len(dst)}"
    return np.frombuffer(bytes(dst), dtype=np.uint8)

def wrel(ref, got, m):
    """imatrix 加权相对误差, 与 tensor_error.py 同一口径。"""
    d = (ref - got).astype(np.float64)
    num = float(np.einsum('ij,j->', d*d, m))
    den = float(np.einsum('ij,j->', ref.astype(np.float64)**2, m))
    return num/den

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--tensors", required=True, help="逗号分隔的张量全名")
    ap.add_argument("--types", required=True, help="逗号分隔的目标类型")
    ap.add_argument("--validate", default=None, help="用这个已落地 GGUF 交叉校验 harness")
    a = ap.parse_args()

    lib = load_lib(a.lib)
    ri = GGUFReader(a.imatrix, "r")
    sums = {t.name[:-8]: np.array(t.data, dtype=np.float64) for t in ri.tensors if t.name.endswith(".in_sum2")}
    cnts = {t.name[:-7]: float(np.array(t.data).reshape(-1)[0]) for t in ri.tensors if t.name.endswith(".counts")}
    imat = {k: v/(cnts.get(k,1.0) or 1.0) for k, v in sums.items()}

    rs = GGUFReader(a.src, "r"); S = {t.name: t for t in rs.tensors}
    V = None
    if a.validate:
        rv = GGUFReader(a.validate, "r"); V = {t.name: t for t in rv.tensors}

    names = a.tensors.split(",")
    types = a.types.split(",")
    print(f"{'张量':<30}{'目标':<9}{'加权相对误差':>15}{'落地产物':>15}{'校验':>8}")
    for name in names:
        ts = S[name]
        ne0, ne1 = int(ts.shape[0]), int(ts.shape[1])
        ref = dequantize(ts.data, ts.tensor_type).astype(np.float32).reshape(ne1, ne0)
        m = imat[name]
        for tn in types:
            raw = quantize(lib, ref, ne1, ne0, tn, m)
            got = dequantize(raw, QT[tn]).astype(np.float32).reshape(ne1, ne0)
            e = wrel(ref, got, m)
            ref_s, mark = "-", ""
            if V and name in V and V[name].tensor_type.name == tn:
                gv = dequantize(V[name].data, V[name].tensor_type).astype(np.float32).reshape(ne1, ne0)
                ev = wrel(ref, gv, m)
                ref_s = f"{ev:.4e}"
                mark = "一致" if abs(e-ev)/max(ev,1e-30) < 1e-6 else f"差{abs(e-ev)/ev:.1%}"
            print(f"{name:<30}{tn:<9}{e:>15.4e}{ref_s:>15}{mark:>8}")

main()
