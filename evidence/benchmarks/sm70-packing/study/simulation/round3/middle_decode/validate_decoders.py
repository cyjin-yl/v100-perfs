#!/usr/bin/env python3
"""CPU decoder identities over a conservative independent coefficient box.

Only exactly represented FP32 integer z values are admitted. This is not a
V100 execution, timing simulation, or a new BFMA4 numerical-model validation.
The BFMA4 error bound and restricted q-domain are reviewed in README.md.
"""
import json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent


def main():
    target = np.arange(-216, 217, dtype=np.int64)[:, None]
    low_error = np.arange(-126, 127, dtype=np.int64)[None, :]
    inner = 512 * target + low_error
    counts = dict(admitted=0, nonrepresentable_rejected=0, shift_errors=0,
                  bfe_errors=0, float_errors=0)
    for high in range(-108, 109):
        z = high * 262144 + inner
        zf = z.astype(np.float32)
        exact = zf.astype(np.int64) == z
        counts['admitted'] += int(exact.sum())
        counts['nonrepresentable_rejected'] += int((~exact).sum())
        zi = zf.astype(np.int32)
        biased = zi + np.int32(256)
        bitfield = ((biased.astype(np.int64) >> 9) & 511)
        bfe = np.where(bitfield >= 256, bitfield - 512, bitfield)
        shift = ((biased.astype(np.uint32) << np.uint32(14))
                 .view(np.int32) >> np.int32(23))
        h = np.rint(zf * np.float32(2.0**-18))
        # Under this box both the product and this residual are exactly FP32.
        residual = zf - h * np.float32(262144)
        f = np.rint(residual * np.float32(2.0**-9))
        counts['bfe_errors'] += int(((bfe != target) & exact).sum())
        counts['shift_errors'] += int(((shift != target) & exact).sum())
        counts['float_errors'] += int(((f != target) & exact).sum())
    assert not sum(counts[k] for k in ('bfe_errors', 'shift_errors', 'float_errors'))
    result = dict(scope='CPU decoder identities, conservative coefficient box',
                  gpu_executed=False, cycle_simulation=False,
                  coefficient_ranges=dict(high=[-108,108], target=[-216,216],
                                          low_plus_error=[-126,126]),
                  fp32_integer_exact_input_required=True,
                  full_cartesian_combinations=217*433*253,
                  **counts, passed=True)
    (BASE/'cpu-decoder-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
