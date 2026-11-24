# Volta 架构模拟：实际运行记录与边界

日期：2026-11-22。这里记录的是本地 CPU 上成功构建、运行的 **Accel-Sim / GPGPU-Sim 周期模型**，不是 V100 实机测速。没有访问真实 GPU，没有跑完本项目的 CUDA/cuBLAS 程序，没有执行 MXFP4 训练。

## 已完成的三类实验

1. **上游真实采集 trace 的重放**：从 Accel-Sim 官方发布站点的 `tesla-v100` 目录下载 Rodinia 小归档，取其中 LUD 64 的 10 个 kernel，使用未修改的上游 `SM7_QV100` 配置成功模拟。这验证了本地 trace 解析、调度、缓存/内存和退出链路能工作；这个 LUD trace 没有 HMMA，不能给本项目的 packing kernel 提供性能结论。
2. **人工 HMMA 依赖图的周期实验**：自行生成明确标注 `SYNTHETIC_NOT_HARDWARE` 的 trace，改变独立累加链、warp 数量和模型的 HMMA latency。没有输入数值、矩阵结果或训练步骤。这验证了本模型下依赖关系会影响周期数。
3. **PTX 数值路径的局部算式检查**：发现固定版本的 `mma_impl` 在中间结果中使用 half 累加；用同一份 `half.hpp` 独立复现了这一算式的舍入/溢出差异。因此没有把该功能仿真路径用作 FP16 输入、FP32 累加 packing 的数值真值来源。此检查不是完整 PTX kernel 的执行。

## 固定版本与构建

| 组件 | 固定提交 |
|---|---|
| [accel-sim-framework](https://github.com/accel-sim/accel-sim-framework) | `7ba54ded60c4ba0083410c37d22d719166acd6dc` |
| [Accel-Sim 的 gpgpu-sim_distribution](https://github.com/accel-sim/gpgpu-sim_distribution) | `4195dd4f229cbf03f4f38c49139ae94d01d23f03` |
| [pybind11](https://github.com/pybind/pybind11) | `b783ee9464f216dd8444c1902933bb941363676e` |

以上源码工作树均无修改。构建环境是 Linux x86-64、Debian 13、GCC/G++ 14.2.0、CMake 4.4.5，使用本地 CUDA 12.8 头文件及 nvcc 工具；不需要 NVIDIA GPU 才能运行 trace 模拟器。构建依赖 bison、flex、m4、OpenGL 开发头/链接文件均只解压到本目录的 `build-deps`，没有修改系统软件包。`packages-selected.json` 保留候选包元数据；实际解压的包不包含其中列出的 Boost 候选。

`configure.log`、`build.log`、`build-resume.log` 保留原始过程。第一次编译缺少 `cudaProfiler.h`，补入 NVIDIA 官方 `cuda_profiler_api` 12.8.90 头文件后成功；没有修改模拟器源码规避编译。其下载地址与校验值见 `cuda_profiler_api-source.json`。

构建产物 `build/accel-sim.out` 为约 4.4 MiB 的本机可执行文件，SHA-256：

`5de36dc76df77270f05893cc4c37c0f6249b20d3fee89a2c508a49def83f1711`

该二进制只链接系统 libc、libm、libstdc++、libgcc、zlib、zstd；它没有链接 NVIDIA 驱动。跨主机运行仍需兼容的 Linux x86-64 系统库。源码固定版本和构建命令同时提供，避免把这个二进制误当成便携 CUDA 模拟器。

## 模型究竟是什么

上游 `SM7_QV100/gpgpusim.config` 自称模拟 **Volta Quadro V100**，对应产品名 Quadro GV100。它不是 Tesla V100 PCIe 的逐项校准配置。

| 项目 | 本次实际配置 / 官方资料 |
|---|---|
| 模型架构 | Compute capability 7.0；80 个 SM；每 SM 4 个调度器；启用 sub-core model |
| 模型时钟 | core/interconnect/L2 = 1132 MHz；DRAM = 850 MHz |
| 模型内存参数 | 32 个通道 × 16 B × DDR × 850 MHz，即参数给出的名义 870.4 GB/s；这不是实测带宽 |
| 官方 Quadro GV100 | 32 GB HBM2，870 GB/s，640 Tensor Cores |
| 官方 Tesla V100 PCIe | 16/32 GB HBM2，900 GB/s，640 Tensor Cores；官方 Tensor 峰值 112 TFLOPS |
| 启动固定开销 | 每 kernel 5000 个模型周期，TB 启动开销 0 |

Quadro 和 Tesla 的数据来源分别是 [NVIDIA Quadro GV100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/productspage/quadro/quadro-desktop/quadro-volta-gv100-a4-nvidia-704619-r3-web.pdf)、[Tesla V100 datasheet](https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet.pdf) 及 [Tesla PCIe product brief](https://images.nvidia.com/content/tesla/pdf/Tesla-V100-PCIe-Product-Brief.pdf)。模型参数直接来自随附原样配置；配置中的部分自然语言注释（例如 HBM 堆栈/通道数）与数值项不同，此处以实际数值项为准。

### HMMA latency 和吞吐参数

这两个层次不能混用：

- PTX 配置中的 `-ptx_opcode_latency_tesnor 64` / `-ptx_opcode_initiation_tensor 64` 是另一条模型路径的参数。
- 本次 SASS trace 中的 `HMMA` 在 `gpu-simulator/ISA_Def/volta_opcode.h` 映射到 `SPECIALIZED_UNIT_3_OP`，实际由 `trace.config` 的 `-trace_opcode_latency_initiation_spec_op_3 2,2` 控制；两个数分别是模型 latency 与 initiation interval。`trace_config::set_latency` 在 specialized unit 分支赋值。
- `-specialized_unit_3 1,4,8,4,4,TENSOR` 的 4 个模拟执行单元是调度/流水线抽象，不能直接解释成芯片只有 4 个物理 Tensor Core，也不能把 `2,2` 直接声称为任何 Volta HMMA 指令在所有条件下的实测延迟。
- 上游配置注释引用 Volta 白皮书、GTC 微基准等，但本次没有在真实 V100 上重新标定这些参数；没有逐项证实其微基准来源。将 latency 改为 8、initiation 保持 2 的结果只是敏感性分析，不是一个更“准确”的硬件模型。

## 结果一：官方 LUD 64 trace

归档 URL：

`https://engineering.purdue.edu/tgrogers/accel-sim/traces/tesla-v100/latest/rodinia_2.0-ft.tgz`

该路径含 `latest`，因此固定内容需以哈希为准：21,011,152 字节，SHA-256 `aeb7de478785856e4ac834d12be0e71ab0df297f43fc02650e2ca90dea66d8b1`。下载时 `Last-Modified` 为 2020-07-07。只提取 `rodinia_2.0-ft/9.1/lud-rodinia-2.0-ft/_v__b__i___data_64_dat/traces/`，未运行归档中的 shell 脚本。trace 头的 binary version 是 70。

模拟器成功退出，10 个 kernel 合计 **127309 个模型周期、418884 条 thread instructions**，日志报告模拟器宿主耗时 12 秒。12 秒是 CPU 模拟程序的运行时间，绝不是 V100 运行时间；thread instructions 也不是 FLOPs。

| kernel 序号 | kernel | 模型周期 | thread instructions |
|---|---|---:|---:|
| 1 | lud_diagonal | 20908 | 19673 |
| 2 | lud_perimeter | 8768 | 48240 |
| 3 | lud_internal | 5832 | 156672 |
| 4 | lud_diagonal | 20908 | 19673 |
| 5 | lud_perimeter | 8747 | 32160 |
| 6 | lud_internal | 5804 | 69632 |
| 7 | lud_diagonal | 20908 | 19673 |
| 8 | lud_perimeter | 8744 | 16080 |
| 9 | lud_internal | 5782 | 17408 |
| 10 | lud_diagonal | 20908 | 19673 |

机器可读结果为 `official-lud64-results.csv` 和 `.json`，原始日志为 `logs/official-lud64-qv100.log`。全部 10 个 trace 和其上游统计文件一并提供；不必再次下载整份归档才能重放。

## 结果二：人工 HMMA 依赖图

每 warp 执行 256 条 HMMA，除空对照外；每个 warp 均有相同的 10 条 MOV 初始化及 1 条 EXIT。只有一个 CTA，无加载、存储、地址、packing、unpacking、block scale、通信或训练。trace 中的寄存器是上游 tracer 的 operand-base 抽象，**不是逐 lane 的 HMMA 数值模型**。

| 模型 | warp 数 | 独立累加链/warp | HMMA/warp | 总模型周期 | 相同空对照周期 |
|---|---:|---:|---:|---:|---:|
| 原样 QV100 | 1 | 1 | 256 | 6818 | 5028 |
| 原样 QV100 | 1 | 4 | 256 | 5543 | 5028 |
| 原样 QV100 | 1 | 8 | 256 | 5658 | 5028 |
| 原样 QV100 | 4 | 1 | 256 | 6819 | — |
| 原样 QV100 | 16 | 1 | 256 | 7134 | — |
| latency=8, initiation=2 假设 | 1 | 1 | 256 | 8098 | 5028 |
| latency=8, initiation=2 假设 | 1 | 4 | 256 | 5800 | 5028 |
| latency=8, initiation=2 假设 | 1 | 8 | 256 | 5778 | 5028 |

含固定 5000 周期启动项，不能用这些总周期直接倒算 Tensor FLOPS。一个 warp 的 256 条 HMMA 加初始化/退出对应 `267 × 32 = 8544` 条 thread instructions；4 和 16 warp 的工作量也相应不同，不能把这些行当作同工作量加速比。

可以从这组实验得出：**在这份模型里，依赖链和可调度的并行工作影响执行周期；仅数 HMMA 条数不足以预测延迟。** 4 条与 8 条独立链没有单调关系，体现调度、寄存器收集等模型因素，未做 stall 归因。不能从它得出“packing 已加速几倍”或“MXFP4 训练可达到某 TFLOPS”。

机器可读结果为 `synthetic-results.csv` / `.json`；每次准确命令、退出码、日志 SHA-256、宿主秒数都在 JSON 中；trace 和全部日志均保留。

## 结果三：PTX 数值路径的限制

固定 GPGPU-Sim 源码的 `src/cuda-sim/instructions.cc`，`mma_impl`（约 1903–2064 行）先把 `matrix_d[i][j].f16` 置零，并将 16 项乘加反复写回 `.f16`，随后才在 FP32 输出分支转换并加 C。这一算式不能当作 FP16 输入、FP32 累加的精确参考。

`ptx_accumulator_formula_check.cpp` 直接包含该版本的 `half.hpp`，只复现这一中间表达式。实际运行结果：

| 测试 | 该 half 中间算式 | FP32 可精确表示的整数结果 |
|---|---:|---:|
| 一项 `1539 × 3` | 4616 | 4617 |
| 16 项同样乘积相加 | `+Inf` | 73872 |

结果见 `ptx_accumulator_formula_check.json`。这证明局部算式不适合作为本任务的数值 oracle；**没有宣称完整 PTX 程序已经运行失败**。我们的数值正确性证据应来自父任务的独立 CPU/Octave 数值分析，以及将来真实 V100 输出比对。

## 对本项目可以怎样使用这些证据

编译器生成的 sm70 SASS 和静态成本比较由同项目 `../cost/` 单独保存。本目录不把人工 HMMA trace 伪装成这些 kernel 的动态 trace。要从实际 SASS 构造可靠的纯算术片段，需要从输出 STG 的数据源做 SSA backward slice，区分寻址与 magic decode 中的 LEA，处理 LDG 造成的寄存器复用、HMMA 隐含寄存器对、活跃 lane 和输入就绪假设；直接删掉内存指令会引入假依赖，简单按 opcode 筛选也会漏算解码。这次没有做这个未经验证的变换。

Accel-Sim 的 trace frontend 接收的是已知动态指令/地址流。它不会凭一个 CUDA 程序自动产生所有缺失的执行结果，不模拟数值舍入，更不能判断某种 packing 的碰撞或训练收敛。原始 [Accel-Sim ISCA 2020 论文](https://engineering.purdue.edu/tgrogers/publication/khairy-isca-2020/khairy-isca-2020.pdf) 对 trace-driven 和 execution-driven frontend 的区别也有说明。

因此当前成立的交付是：**可复跑的 Volta 架构模型链路 + 公开 trace 重放 + 明示假设的 HMMA 调度敏感性 + 数值路径的局部反例**。本目录没有真实 V100 的 packing 吞吐、端到端训练速度或收敛结果。

## 复跑

有兼容系统库时，直接运行包含的二进制：

```bash
cd simulation/architecture
bash run_existing_binary.sh
```

该脚本重新生成 synthetic trace、重放官方 LUD 和全部人工实验，并更新结果摘要；不需要 GPU、网络或 CUDA runtime。默认会重写本目录对应日志/CSV，原始交付副本可先另行解压保存。

在本次工作目录重新构建：

```bash
source ./environment.sh
"$SIM_CMAKE" -S accel-sim-framework/gpu-simulator -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DOPENGL_gl_LIBRARY=/usr/lib/x86_64-linux-gnu/libGL.so.1 \
  -DOPENGL_INCLUDE_DIR="$ARCH_ROOT/build-deps/debian/usr/include"
"$SIM_CMAKE" --build build --target accel-sim.out -j2
```

外部机器需检出上面三个固定提交，准备 CMake、GCC、flex、bison、m4、zlib/zstd/OpenGL 开发包和 CUDA 12.8 头文件。`environment.sh` 是当前工作区的本地依赖叠加环境；证据归档不包含数百 MB 的完整源码/工具链，也不假装解压后可以凭它独立重建。使用完整源码时可按官方构建说明安装依赖，再采用上述相同 CMake 参数。

证据包中包含模拟器二进制、许可证、固定版本/provenance、脚本、原样配置、全部实际使用的官方 trace、人工 trace、原始日志、CSV/JSON 和局部数值检查源码；不包含 21 MB 整份 Rodinia 下载包和无关 trace、编译中间目标或系统软件包。整份下载包的 URL/哈希/成员清单仍保留。
