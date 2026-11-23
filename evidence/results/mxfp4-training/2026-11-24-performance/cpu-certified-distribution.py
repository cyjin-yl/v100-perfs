"""Reproduce the 30 synthetic CPU certificate trials; this is not a GPU test.

Run from the repository root with Python and NumPy installed.
The saved JSONL contains the original session's results, archived without rerun.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from tools.mxfp4_training.reference import quantize_mx, unpack_q


def certificate(x):
    tensor = quantize_mx(x.astype(np.float32))
    scales = tensor.scales.astype(np.int64)
    row_min = scales.min(axis=1)
    delta = scales - row_min[:, None]
    result = {
        "range": [int(scales.min()), int(scales.max())],
        "max_span": int(delta.max()),
        "energy": None,
    }
    if result["max_span"] > 12:
        return result
    q = unpack_q(tensor).astype(np.int64).reshape(*scales.shape, 32)
    # This experiment's maximum K=16384 and delta<=12 fit in signed int64.
    energy = ((q * q).sum(axis=-1) * np.left_shift(np.int64(1), 2 * delta)).sum(axis=1)
    result["energy"] = int(energy.max())
    return result


def draw(rng, kind, shape):
    if kind in ("gaussian", "gradient_1e-5"):
        return rng.normal(size=shape) * (1e-5 if kind == "gradient_1e-5" else 1)
    if kind == "student_t3":
        return rng.standard_t(3, size=shape) / np.sqrt(3)
    if kind == "outlier_100_p0005":
        x = rng.normal(size=shape)
        x[rng.random(size=shape) < 0.0005] *= 100
        return x
    x = rng.normal(size=(shape[0], shape[1] // 32, 32))
    return np.ldexp(x, rng.integers(-8, 9, size=x.shape[:2])[..., None]).reshape(shape)


for k in (4096, 16384):
    for kind in ("gaussian", "gradient_1e-5", "student_t3",
                 "outlier_100_p0005", "block_range_16"):
        for seed in range(1000, 1003):
            rng = np.random.default_rng(seed)
            a = certificate(draw(rng, kind, (256, k)))
            b = certificate(draw(rng, kind, (256, k)))
            product = None if a["energy"] is None or b["energy"] is None else a["energy"] * b["energy"]
            print(json.dumps({
                "kind": kind, "rows": 256, "K": k, "seed": seed,
                "pass": product is not None and product <= 2**48,
                "bound_fraction": None if product is None else round(math.sqrt(product) / 2**24, 4),
                "A": a, "B": b,
            }), flush=True)
