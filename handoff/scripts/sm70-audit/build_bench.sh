#!/usr/bin/env bash
# 编译 SM70 审计用的三个小工具, 链接 build-rw 里已编好的 fastllm 目标文件
# (fastllm 是 CMake OBJECT 库, 没有 .so 可链, 只能把 .o 全带上; 链接选项抄自
#  build-rw/CMakeFiles/testKernelRouteCensus.dir/link.txt)。
#
#   bench_devprops   量 cudaGetDevice / cudaGetDeviceProperties / cudaDeviceGetAttribute
#                    的单次调用成本(不依赖 fastllm)
#   bench_mmq_trial  量 FastllmCudaTrySm70Iq4XsMmq() 在 decode 形状(n=1)下每次调用的
#                    开销, 并与修复前的实现同口径对比
#   probe_routes     用真实 IQ4_XS 权重跑 fastllm::Linear, 打印算子路由普查,
#                    直接回答"IQ4_XS 到底走没走 MMQ"
#
# 注意: 这些工具会建 CUDA 上下文(约 300MB 显存), 跑之前确认显存有余量。
set -euo pipefail
R=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/fastllm
D=/home/ezra/projects/EzraVastLLM/scripts/sm70-audit
NVCC=/home/ezra/.conda/envs/tsenv/bin/nvcc
HOSTCXX=/home/ezra/.conda/envs/tsenv/bin/x86_64-conda-linux-gnu-g++
OBJS=$(find "$R/build-rw/CMakeFiles/fastllm.dir" -name '*.o' | tr '\n' ' ')
INC=(-I "$R/include" -I "$R/include/utils" -I "$R/include/models" -I "$R/include/blocks"
     -I "$R/include/devices/cpu" -I "$R/include/devices/disk" -I "$R/include/devices/cuda"
     -I "$R/third_party/json11" -I "$R/third_party/gguf" -I "$R/third_party/flashinfer")
LIBS=(-L/home/ezra/.conda/envs/tsenv/targets/x86_64-linux/lib
      -L/home/ezra/.conda/envs/tsenv/targets/x86_64-linux/lib/stubs
      -lcublas -lcuda -lcudadevrt -lcudart_static -lnccl -ldl -lrt -lpthread
      /usr/lib64/libzstd.so)

# 不依赖 fastllm
"$NVCC" -ccbin "$HOSTCXX" -O2 -arch=sm_70 -Wno-deprecated-gpu-targets \
  -o "$D/bench_devprops" "$D/bench_devprops.cu"

for t in bench_mmq_trial probe_routes; do
  "$NVCC" -ccbin "$HOSTCXX" -O2 -std=c++17 -arch=sm_70 -Wno-deprecated-gpu-targets \
    "${INC[@]}" -o "$D/$t" "$D/$t.cu" $OBJS "${LIBS[@]}"
  echo "built: $D/$t"
done
echo "built: $D/bench_devprops"
