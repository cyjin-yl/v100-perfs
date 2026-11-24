#!/usr/bin/env bash
set -euo pipefail
probe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$probe_dir"
mkdir -p build results
command -v nvcc >/dev/null || { printf '%s\n' 'Use an existing CUDA 12.x development environment with sm_70 support.' >&2; exit 2; }
nvcc --list-gpu-code | grep -qx sm_70 || { printf '%s\n' 'This compiler does not support sm_70.' >&2; exit 2; }
nvcc -O3 -std=c++17 -lineinfo -Xptxas -v \
  -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70 \
  mapped_packing.cu -lcublas -o build/mapped_packing 2>build/compiler.stderr
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
sha256sum mapped_packing.cu fragment_mapping.hpp build/mapped_packing > "results/$stamp-hashes.txt"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,uuid,driver_version,pstate,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv > "results/$stamp-gpu-before.csv"
fi
./build/mapped_packing "$@" 2> >(tee "results/$stamp-stderr.txt" >&2) | tee "results/$stamp.jsonl"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,uuid,driver_version,pstate,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv > "results/$stamp-gpu-after.csv"
fi
