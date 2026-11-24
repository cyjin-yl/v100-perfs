#!/usr/bin/env python3
"""Build both CUDA backends and standalone checks; never execute a GPU kernel.

Example:
  python3 tools/mxfp4_training/build_cuda.py --output-dir /tmp/mxfp4-cuda-build
  /tmp/mxfp4-cuda-build/check_mxfp4 --device 0 --correctness-only

CUDA 12.x with sm_70 code generation is required. The default output includes
SM70 cubin and compute_70 PTX; there is no dependency on PyTorch headers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nvcc", help="CUDA 12.x nvcc path (default: CUDA_HOME/bin/nvcc or PATH)")
    parser.add_argument("--library-only", action="store_true",
                        help="omit standalone correctness executables")
    parser.add_argument("--with-benchmark", action="store_true",
                        help="also build the packed-input/cuBLAS benchmark (requires cuBLAS)")
    parser.add_argument("--with-fused-benchmark", action="store_true",
                        help="also build the bounded packed-only fused benchmark (requires certified >=3)")
    parser.add_argument("--with-certified", action="store_true",
                        help="include the conditional exact Tensor Core backend and its checks")
    parser.add_argument("--with-quantizer-rne", action="store_true",
                        help="include the opt-in native FP32 MX32 RNE quantizer in the shared library")
    parser.add_argument("--certified-prepare-raw-cache", action="store_true",
                        help="experimental C2 preparation: cache raw MX32 blocks in shared memory "
                             "for 0<K<=8192 (requires --with-certified --certified-variant 2)")
    parser.add_argument("--tiled-variant", type=int, choices=(0, 1, 2), default=1,
                        help="0: MX32; 1: MX32 with normal-scale gate; 2: two MX32 blocks per stage")
    parser.add_argument("--certified-variant", type=int, choices=tuple(range(11)), default=2,
                        help="0..2: cuBLAS preparation/epilogue variants; "
                             "3: fused 64/BK64; 4: fused 128/BK32; 5: fused 128/BK64; "
                             "6: fused 128/BK32 with register prefetch; "
                             "7: prefetched 128/BK32 with 64x64 warp tiles; "
                             "8: half-row vector raw prefetch with variant-6 fallback; "
                             "9: half-row fixed-stride cursor prefetch; "
                             "10: half-row exact half2 normalization")
    args = parser.parse_args()
    if args.with_fused_benchmark and (not args.with_certified or args.certified_variant < 3):
        parser.error("--with-fused-benchmark requires --with-certified and --certified-variant >=3")
    if args.certified_prepare_raw_cache and (not args.with_certified or args.certified_variant != 2):
        parser.error("--certified-prepare-raw-cache requires --with-certified --certified-variant 2")
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = args.nvcc or (str(Path(cuda_home) / "bin/nvcc") if cuda_home else shutil.which("nvcc"))
    if not nvcc:
        parser.error("nvcc not found; pass --nvcc /usr/local/cuda-12.x/bin/nvcc")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent / "csrc"
    mapping = source.parents[2] / "evidence/benchmarks/sm70-packing/study/simulation/round3/cuda/fragment_mapping.hpp"
    version = subprocess.run([nvcc, "--version"], check=True, capture_output=True, text=True).stdout
    common = [nvcc, "-O3", "-std=c++17", "--ftz=false", "-lineinfo", "-Xptxas=-v",
              "-gencode=arch=compute_70,code=[sm_70,compute_70]",
              f"-DMXFP4_TILED_VARIANT={args.tiled_variant}",
              f"-DMXFP4_CERTIFIED_VARIANT={args.certified_variant}"]
    if args.certified_prepare_raw_cache:
        common.append("-DMXFP4_CERTIFIED_PREPARE_RAW_CACHE=1")
    kernels = [str(source / "mxfp4_gemm.cu"), str(source / "mxfp4_gemm_tiled.cu")]
    certified_sources = [str(source / "mxfp4_gemm_certified.cu")] if args.with_certified else []
    if args.with_certified and args.certified_variant >= 3:
        certified_sources.append(str(source / "mxfp4_gemm_fused.cu"))
    certified_links = ["-lcublas"] if args.with_certified else []
    quantizer_sources = [str(source / "mxfp4_quantize_rne.cu")] if args.with_quantizer_rne else []
    jobs = [("shared_library", common + ["-shared", "-Xcompiler=-fPIC"] + kernels +
             certified_sources + quantizer_sources + certified_links +
             ["-o", str(output / "libmxfp4_sm70.so")])]
    if not args.library_only:
        jobs.append(("standalone_harness", common + [str(source / "mxfp4_gemm.cu"),
                                                    str(source / "check_mxfp4.cu"), "-o",
                                                    str(output / "check_mxfp4")]))
        jobs.append(("tiled_harness", common + [
            "-Dmxfp4_gemm_cuda=mxfp4_gemm_tiled_cuda",
            "-Dmxfp4_cuda_abi_version=mxfp4_tiled_cuda_abi_version",
            "-Dmxfp4_cuda_backend=mxfp4_tiled_cuda_backend",
            str(source / "mxfp4_gemm_tiled.cu"), str(source / "check_mxfp4.cu"),
            "-o", str(output / "check_mxfp4_tiled")]))
        if args.with_certified:
            jobs.append(("certified_harness", common + [str(source / "mxfp4_gemm_tiled.cu")] +
                         certified_sources + [str(source / "check_mxfp4_certified.cu"), "-lcublas",
                                              "-o", str(output / "check_mxfp4_certified")]))
    if args.with_benchmark:
        jobs.append(("performance_benchmark", common + ["-DMXFP4_BENCH_TILED"] + kernels +
                     (["-DMXFP4_BENCH_CERTIFIED"] if args.with_certified else []) + certified_sources +
                     [str(source / "bench_mxfp4.cu"), "-lcublas", "-ldl", "-o",
                      str(output / "bench_mxfp4")]))
    if args.with_fused_benchmark:
        jobs.append(("fused_packed_benchmark", common +
                     ["-DMXFP4_BENCH_TILED", "-DMXFP4_BENCH_CERTIFIED"] + kernels +
                     certified_sources + [str(source / "bench_mxfp4_fused.cu"),
                     "-lcublas", "-ldl", "-o", str(output / "bench_mxfp4_fused")]))
    record = {
        "schema_version": 2, "evidence_type": "cuda_compilation", "gpu_executed": False,
        "backends": ["sm70_mx32_baseline", "sm70_mx32_tiled"] +
                    ([("sm70_mx32_certified_fused_hmma_or_tiled" if args.certified_variant >= 3
                       else "sm70_mx32_certified_cublas_or_tiled")] if args.with_certified else []),
        "tiled_variant": args.tiled_variant,
        "certified_enabled": args.with_certified,
        "quantizer_rne_enabled": args.with_quantizer_rne,
        "quantizer_rne_backend": "sm70_mx32_rne_warp_fp32_v1" if args.with_quantizer_rne else None,
        "certified_variant": args.certified_variant if args.with_certified else None,
        "certified_prepare_raw_cache_enabled": args.certified_prepare_raw_cache,
        "certified_prepare_raw_cache_backend": (
            "sm70_c2_row_raw_cache_k8192_v1" if args.certified_prepare_raw_cache else None),
        "certified_prepare_raw_cache_max_k": 8192 if args.certified_prepare_raw_cache else None,
        "certified_prepare_raw_cache_bytes_per_mx32": 20 if args.certified_prepare_raw_cache else None,
        "fused_packed_benchmark_enabled": args.with_fused_benchmark,
        "full_training_executed": False, "nvcc_version": version.strip(),
        "source_sha256": {str(p.relative_to(source.parent)): sha256(p)
                          for p in sorted(source.glob("*")) if p.is_file()},
        "fragment_mapping_sha256": sha256(mapping), "jobs": [], "artifacts": {},
    }
    for label, command in jobs:
        start = time.monotonic()
        result = subprocess.run(command, capture_output=True, text=True)
        log = output / f"{label}.log"
        log.write_text(result.stdout + result.stderr)
        record["jobs"].append({"name": label, "argv": command, "returncode": result.returncode,
                               "wall_seconds": time.monotonic() - start, "log": log.name,
                               "log_sha256": sha256(log)})
        if result.returncode:
            record["passed"] = False
            (output / "compile-evidence.json").write_text(json.dumps(record, indent=2) + "\n")
            print(result.stdout + result.stderr, file=sys.stderr)
            return result.returncode
        artifact = Path(command[-1])
        record["artifacts"][artifact.name] = {"sha256": sha256(artifact), "bytes": artifact.stat().st_size}
    record["passed"] = True
    (output / "compile-evidence.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"passed": True, "gpu_executed": False, "output_dir": str(output),
                      "artifacts": sorted(record["artifacts"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
