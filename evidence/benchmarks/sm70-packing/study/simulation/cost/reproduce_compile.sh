#!/usr/bin/env bash
set -euo pipefail
cost_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_root="$(cd -- "$cost_dir/../.." && pwd)"
task_sdk="${V100_COST_TOOLCHAIN:-$task_root/toolchain}"
task_nvcc="$task_sdk/cuda_nvcc-linux-x86_64-12.8.93-archive/bin/nvcc"
task_dump="$task_sdk/cuda_cuobjdump-linux-x86_64-12.8.90-archive/bin/cuobjdump"
task_disasm="$task_sdk/cuda_nvdisasm-linux-x86_64-12.8.90-archive/bin"
task_includes=(
  "-I$task_sdk/cuda_cudart-linux-x86_64-12.8.90-archive/include"
  "-I$task_sdk/cuda_cccl-linux-x86_64-12.8.90-archive/include"
)
test -x "$task_nvcc"
test -x "$task_dump"
"$task_nvcc" --version > "$cost_dir/nvcc-version.txt"
"$task_nvcc" -O3 -std=c++17 -lineinfo -Xptxas -v \
  -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70 \
  "${task_includes[@]}" -c "$cost_dir/packing_cost_kernels.cu" \
  -o "$cost_dir/packing_cost_kernels.o" > "$cost_dir/compile.stdout" 2> "$cost_dir/compile.stderr"
NVDISASM_PATH="$task_disasm" "$task_dump" --dump-sass "$cost_dir/packing_cost_kernels.o" > "$cost_dir/sm70-sass.txt"
"$task_dump" --dump-resource-usage "$cost_dir/packing_cost_kernels.o" > "$cost_dir/resources.txt"
"$task_nvcc" -O3 -std=c++17 -lineinfo -Xptxas -v \
  -gencode arch=compute_70,code=sm_70 "${task_includes[@]}" \
  -c "$cost_dir/m8n8k4_probe.cu" -o "$cost_dir/m8n8k4_probe.o" \
  > "$cost_dir/m8n8k4-compile.stdout" 2> "$cost_dir/m8n8k4-compile.stderr"
NVDISASM_PATH="$task_disasm" "$task_dump" --dump-sass "$cost_dir/m8n8k4_probe.o" > "$cost_dir/m8n8k4-sass.txt"
python3 "$cost_dir/validate_magic_decode.py"
python3 "$cost_dir/analyze_cost.py"
