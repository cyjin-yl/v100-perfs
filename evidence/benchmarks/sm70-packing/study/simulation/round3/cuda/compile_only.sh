#!/usr/bin/env bash
set -euo pipefail
probe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_root="$(cd -- "$probe_dir/../../.." && pwd)"
task_sdk="${V100_COST_TOOLCHAIN:-$task_root/toolchain}"
task_nvcc="$task_sdk/cuda_nvcc-linux-x86_64-12.8.93-archive/bin/nvcc"
task_dump="$task_sdk/cuda_cuobjdump-linux-x86_64-12.8.90-archive/bin/cuobjdump"
task_disasm="$task_sdk/cuda_nvdisasm-linux-x86_64-12.8.90-archive/bin"
"$task_nvcc" --version > "$probe_dir/nvcc-version.txt"
"$task_nvcc" -O3 -std=c++17 -lineinfo -Xptxas -v \
  -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70 \
  "-I$task_sdk/cuda_cudart-linux-x86_64-12.8.90-archive/include" \
  "-I$task_sdk/cuda_cccl-linux-x86_64-12.8.90-archive/include" \
  "-I$task_sdk/cublas-dev/targets/x86_64-linux/include" \
  -c "$probe_dir/mapped_packing.cu" -o "$probe_dir/mapped_packing.o" \
  > "$probe_dir/compile.stdout" 2> "$probe_dir/compile.stderr"
NVDISASM_PATH="$task_disasm" "$task_dump" --dump-sass "$probe_dir/mapped_packing.o" > "$probe_dir/sm70-sass.txt"
"$task_dump" --dump-resource-usage "$probe_dir/mapped_packing.o" > "$probe_dir/resources.txt"
python3 "$probe_dir/summarize_sass.py"
