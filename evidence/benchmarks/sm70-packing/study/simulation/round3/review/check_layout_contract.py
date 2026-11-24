#!/usr/bin/env python3
"""Cross-check the actual C++ fragment header and rectangular BLAS contract.

No CUDA kernel or cuBLAS call is executed. The C++ header is compiled by a CPU
compiler, then compared with the independent Python fragment-map fixture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("layout-contract-results.json"))
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    directory = args.cuda_dir.resolve()
    compiler = shutil.which(args.cxx)
    require(compiler is not None, "C++ compiler is unavailable")
    source = r'''#include "fragment_mapping.hpp"
#include <iostream>
int main() {
  for(int lane=0;lane<32;++lane) {
    std::cout<<lane<<' '<<mma_group(lane)<<' '<<a_row(lane)<<' '<<b_col(lane);
    for(int i=0;i<8;++i)std::cout<<' '<<c_row(lane,i)<<' '<<c_col(lane,i);
    std::cout<<'\n';
  }
}
'''
    with tempfile.TemporaryDirectory(prefix="v100-fragment-header-") as raw:
        tmp = Path(raw)
        (tmp / "check.cpp").write_text(source)
        subprocess.run([compiler, "-O2", "-std=c++17", "-I" + str(directory),
                        str(tmp / "check.cpp"), "-o", str(tmp / "check")], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        lines = subprocess.check_output([str(tmp / "check")], text=True).splitlines()
    expected = json.loads((directory / "fragment-map.json").read_text())
    require(len(lines) == len(expected) == 32, "Fragment lane count mismatch")
    records = []
    for line, target in zip(lines, expected):
        values = list(map(int, line.split()))
        require(values[:4] == [target["lane"], target["group"], target["a_row"], target["b_col"]],
                "C++ input fragment map differs from Python map")
        pairs = list(zip(values[4::2], values[5::2]))
        require(pairs == [(x["row"], x["col"]) for x in target["outputs"]],
                "C++ accumulator map differs from Python map")
        for i, pair in enumerate(pairs):
            row_index, col_index = (i >> 1) & 1, (i & 1) | ((i >> 1) & 2)
            require(pairs[2 * row_index][0] == pair[0], "Two-row scale reuse index mismatch")
            inverse_col_i = (col_index & 1) + ((col_index & 2) << 1)
            require(pairs[inverse_col_i][1] == pair[1], "Four-column scale reuse index mismatch")
        for i in range(0, 8, 2):
            require(pairs[i][0] == pairs[i + 1][0] and pairs[i + 1][1] == pairs[i][1] + 1,
                    "float2 output pair is not adjacent in the same row")
            require(pairs[i][1] % 2 == 0, "float2 output pair lacks 8-byte row-major alignment")
        records.extend(pairs)
    require(len(set(records)) == 256, "Fragment output map aliases or misses an element")

    # Verify all active/retired warp tiles for non-square, partial-CTA grids.
    grids = []
    for m, n in [(16, 48), (32, 16), (48, 48), (80, 32)]:
        covered = []
        for block_x in range((m + 63) // 64):
            for block_y in range(n // 16):
                for warp in range(4):
                    row0, col0 = (block_x * 4 + warp) * 16, block_y * 16
                    if row0 >= m:
                        continue
                    covered.extend((row0 + r, col0 + c) for r, c in records)
        require(len(covered) == len(set(covered)) == m * n, "Grid tiles overlap or miss outputs")
        require(all(0 <= r < m and 0 <= c < n for r, c in covered), "Grid output out of bounds")
        grids.append({"M": m, "N": n, "covered_outputs": len(covered)})

    # The cuBLAS call contract is n,m,k; opT on stored B[K,N] and opN on
    # A's column-major view [K,M], resulting in C[N,M] == (A*B)^T.
    # Rectangular dimensions ensure a mistaken transpose is not hidden.
    rng = np.random.default_rng(20261122)
    blas_cases = []
    for m, n, k in [(16, 48, 4), (48, 16, 36), (32, 80, 64)]:
        rows, sb = m + n, (k + 31) // 32
        q = rng.integers(-3, 4, size=(2, rows, k))
        exp = rng.integers(-2, 3, size=(2, rows, sb))
        physical = q.astype(np.float64) * 0.5 * np.exp2(exp[:, :, np.arange(k) // 32])
        half = physical.astype(np.float16)
        require(np.array_equal(half.astype(np.float64), physical), "Half dequantization lost data")
        raw_half = half.ravel()
        for stream in range(2):
            begin, stride = stream * rows * k, rows * k
            a_storage = raw_half[begin:begin + m * k]
            b_storage = raw_half[begin + m * k:begin + stride]
            blas_a = b_storage.reshape((k, n), order="F").astype(np.float64).T
            blas_b = a_storage.reshape((k, m), order="F").astype(np.float64)
            result_memory = (blas_a @ blas_b).ravel(order="F")
            reference = physical[stream, :m] @ physical[stream, m:].T
            require(np.array_equal(result_memory.reshape((m, n)), reference), "Rectangular BLAS storage contract mismatch")
        blas_cases.append({"M": m, "N": n, "K": k, "independent_lanes": 2,
                           "variable_scale_exponents": [-2, 2], "mismatches": 0})

    result = {"all_checks_passed": True, "gpu_executed": False, "actual_cublas_executed": False,
              "cpp_header_executed_on_cpu": True, "cpp_python_lane_mapping_mismatches": 0,
              "accumulator_coordinates_checked": 256, "scale_reuse_indices_checked": 256,
              "float2_pairs_checked": 128, "partial_cta_grids": grids,
              "rectangular_cublas_storage_contract_cases": blas_cases,
              "sources_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                 for name in ["fragment_mapping.hpp", "mapped_packing.cu", "fragment-map.json"]},
              "limits": ["No CUDA instruction was executed.",
                         "The rectangular check reconstructs the reviewed cuBLAS parameter contract; it does not dynamically intercept a cuBLAS call.",
                         "This validates layout/scale bookkeeping, not arbitrary floating-point rounding or GPU speed."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"all_checks_passed": True, "cpp_python_lane_mapping_mismatches": 0,
                      "rectangular_contract_cases": len(blas_cases), "gpu_executed": False}, indent=2))


if __name__ == "__main__":
    main()
