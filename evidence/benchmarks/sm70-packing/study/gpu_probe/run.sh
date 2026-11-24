#!/usr/bin/env bash
set -euo pipefail
probe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$probe_dir"
mkdir -p build results
if ! command -v nvcc >/dev/null 2>&1; then
  printf '%s\n' 'nvcc is unavailable. Use an existing CUDA 12.x development environment with sm_70 support.' >&2
  exit 2
fi
nvcc --version > build/nvcc-version.txt
if ! nvcc --list-gpu-code | grep -qx 'sm_70'; then
  printf '%s\n' 'This nvcc does not support sm_70. CUDA 12.x is required for the V100 build; no installation was attempted.' >&2
  exit 2
fi
nvcc -O3 -std=c++17 -lineinfo \
  -gencode arch=compute_70,code=sm_70 \
  -gencode arch=compute_70,code=compute_70 \
  v100_packing_probe.cu -lcublas -o build/v100_packing_probe \
  2>build/compiler-stderr.txt
if command -v cuobjdump >/dev/null 2>&1; then
  cuobjdump --dump-sass build/v100_packing_probe > build/sass.txt
  if ! grep -q 'HMMA' build/sass.txt; then
    printf '%s\n' 'No HMMA instruction found in SASS; refusing to label this a Tensor Core probe.' >&2
    exit 2
  fi
fi
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,uuid,driver_version,pstate,temperature.gpu,utilization.gpu,memory.used,memory.total \
    --format=csv > "results/$stamp-nvidia-smi-before.csv"
fi
sha256sum v100_packing_probe.cu run.sh build/v100_packing_probe > "results/$stamp-sha256.txt"
./build/v100_packing_probe "$@" 2> >(tee "results/$stamp-stderr.txt" >&2) \
  | tee "results/$stamp-results.jsonl"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,uuid,driver_version,pstate,temperature.gpu,utilization.gpu,memory.used,memory.total \
    --format=csv > "results/$stamp-nvidia-smi-after.csv"
fi
