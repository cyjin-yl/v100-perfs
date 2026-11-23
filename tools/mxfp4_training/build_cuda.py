#!/usr/bin/env python3
"""Build the CUDA ABI and tiny standalone checks; never execute a GPU kernel.

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
    parser.add_argument("--library-only", action="store_true")
    args = parser.parse_args()
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
              "-gencode=arch=compute_70,code=[sm_70,compute_70]"]
    jobs = [("shared_library", common + ["-shared", "-Xcompiler=-fPIC", str(source / "mxfp4_gemm.cu"),
                                       "-o", str(output / "libmxfp4_sm70.so")])]
    if not args.library_only:
        jobs.append(("standalone_harness", common + [str(source / "mxfp4_gemm.cu"),
                                                    str(source / "check_mxfp4.cu"), "-o",
                                                    str(output / "check_mxfp4")]))
    record = {
        "schema_version": 1, "evidence_type": "cuda_compilation", "gpu_executed": False,
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
