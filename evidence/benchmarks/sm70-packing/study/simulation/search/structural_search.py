#!/usr/bin/env python3
"""Conservative exact-domain search for linear FP16 packing on V100.

This is a mathematical bound search, not a hardware benchmark. Numerical
behavior is independently exercised by ../numerics using the published V100
model. It only searches the stated linear radix family; a failed bound is not
an impossibility theorem for other encodings or even a counterexample.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

Q_FULL = (0, -1, 1, -2, 2, -3, 3, -4, 4, -6, 6, -8, 8, -12, 12)


def evaluate(qmax: int, radix: int, k: int, two_sided: bool) -> dict:
    alphabet = tuple(q for q in Q_FULL if abs(q) <= qmax)
    source = np.array([a + radix * b for a, b in itertools.product(alphabet, repeat=2)], dtype=np.int64)
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = source.astype(np.float16).astype(np.float64)
    operand_errors = int(np.count_nonzero(rounded != source))
    coefficient_max = k * qmax * qmax
    if two_sided:
        accumulator_bound = k * (qmax * (radix + 1)) ** 2
        high_safe = 2 * coefficient_max * (2 * radix + 1) < radix * radix
    else:
        accumulator_bound = coefficient_max * (radix + 1)
        high_safe = True
    low_safe = 2 * coefficient_max < radix
    exact_integer_safe = accumulator_bound < 2**24
    safe = not operand_errors and high_safe and low_safe and exact_integer_safe
    return dict(qmax=qmax, radix=radix, k=k, two_sided=two_sided,
                operand_pairs=len(source), operand_pairs_rounded=operand_errors,
                low_coefficient_bound=coefficient_max,
                cross_coefficient_bound=2 * coefficient_max if two_sided else None,
                sum_absolute_products_bound=accumulator_bound,
                high_extraction_proven=high_safe, low_extraction_proven=low_safe,
                integer_arithmetic_proven=exact_integer_safe, sufficient_conditions_pass=safe,
                fits_radix256_magic_decoder=bool(two_sided and safe and radix == 256 and accumulator_bound < 2**22))


def main() -> None:
    rows = []
    summary = []
    for two_sided in (False, True):
        for qmax in (1, 2, 3, 4, 6, 8, 12):
            candidates = [evaluate(qmax, 1 << shift, k, two_sided)
                          for shift in range(1, 13) for k in range(1, 129)]
            rows.extend(candidates)
            safe = [row for row in candidates if row["sufficient_conditions_pass"]]
            native_safe = [row for row in safe if row["k"] % 4 == 0]
            best = max(native_safe, key=lambda row: (row["k"], -row["radix"])) if native_safe else None
            summary.append(dict(mode="two_sided" if two_sided else "shared_rhs", qmax=qmax,
                                alphabet=[q for q in Q_FULL if abs(q) <= qmax],
                                best_native_k_up_to_128=best,
                                max_mathematical_k_up_to_128=max((row["k"] for row in safe), default=0)))
    result = {
        "kind": "exact_integer_sufficient_bound_search",
        "family": "shared: (a0+R*a1)*b; two-sided: (a0+R*a1)*(b0+R*b1)",
        "grid": {"qmax": [1, 2, 3, 4, 6, 8, 12], "radix_powers_of_two": [1, 12], "k": [1, 128]},
        "candidates_checked": len(rows),
        "assumptions": [
            "All input q are in the stated symmetric E2M1-derived integer alphabet (q=2*x).",
            "Scale is factored out before packing and applied to each decoded dot product afterwards.",
            "Packing must round no operand for any pair in the alphabet.",
            "Every partial sum is bounded by sum of absolute products < 2^24.",
            "V100 alignment exponent is <=23 in these bounded integer cases; alignment therefore cannot discard a nonzero integer bit.",
            "Nearest-integer coefficient extraction uses strict half-radix inequalities; ties excluded.",
            "K divisible by four denotes native arithmetic granularity, not proof of a fast end-to-end kernel.",
            "A failed sufficient bound does NOT prove the candidate impossible. No universal encoding search is claimed.",
        ],
        "summary": summary,
        "selected": {"two_sided_q3_R256_K4": evaluate(3,256,4,True),
                     "two_sided_q2_R256_K8": evaluate(2,256,8,True),
                     "two_sided_q1_R256_K32": evaluate(1,256,32,True)},
        "candidates": rows,
    }
    output = Path(__file__).with_name("structural-search-results.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    compact = [{"mode": s["mode"], "qmax": s["qmax"],
                "max_k": s["max_mathematical_k_up_to_128"],
                "native_k": (s["best_native_k_up_to_128"] or {}).get("k"),
                "radix": (s["best_native_k_up_to_128"] or {}).get("radix")}
               for s in summary]
    print(json.dumps({"candidates": len(rows), "summary": compact, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
