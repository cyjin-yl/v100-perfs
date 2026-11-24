"""Reproduce source guard counterexamples in CPU IEEE float16.

This is not a substitute for a real V100 kernel test.
"""
import json
import numpy as np

def bits32(x):
    return hex(int(np.array([x], dtype=np.float32).view(np.uint32)[0]))

def main():
    x = np.float32(129 * 2.0**-37)
    scale = np.float32(2048)
    restored = np.float32(np.float16(x * scale)) / scale
    report = {
        "execution": "CPU numpy.float16, not CUDA",
        "gemm_operand": {
            "max_abs": 1,
            "min_abs": float(x),
            "ratio": 1 / float(x),
            "guard_limit": 1073217536.0,
            "passes_guard": 1 / float(x) <= 1073217536.0,
            "scale": float(scale),
            "scaled_fp32": float(x * scale),
            "scaled_fp16": float(np.float16(x * scale)),
            "restored": float(restored),
            "relative_error": float((restored - x) / x),
            "input_bits": bits32(x),
            "restored_bits": bits32(restored),
        },
    }
    y = np.float32(2.0**-37)
    report["attention_operand"] = {
        "max_abs": 1,
        "min_abs": float(y),
        "passes_guard": 1 / float(y) <= 2.0**38,
        "scaled_fp32": float(y * scale),
        "scaled_fp16": float(np.float16(y * scale)),
    }
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
