# V100 MXFP4：保持原算术的性能优化

实验在 2026-11-23 UTC 进行；归档日期使用 Asia/Shanghai 的 2026-11-24。原实现与训练接口见 [基础算子说明](../../../../tools/mxfp4_training/README.md)，此前正确性和 CPU 训练结果保留在 [2026-11-23](../2026-11-23/README.md)。本轮新增 shared-memory tiled kernel、带整数精度证书的 cuBLAS 快速路径、独立正确性检查与分层计时。

**已在真实 Tesla V100-PCIE-32GB 上测得：2048³、已打包 MXFP4 输入到 FP32 输出，选定快速路径为 0.33718 ms / 50.95 TFLOPS。** 同轮原内核为 2.91 TFLOPS，完整范围 tiled 内核为 23.79 TFLOPS。这是一次输入通过精度证书的合成矩阵实验，不包含 FP32→MXFP4 量化、H2D 或 workspace 分配，不是完整训练吞吐。

## 1. 同一轮的计时比较

原始记录：[variant 2 完整 JSONL](verified/build-perf-c2/varied-2048.jsonl)。三条 MXFP4 路径使用相同 codes/scales、逻辑 M=N=K=2048，执行目标均为 `C=A @ B_rows.T`、FP32 输出。每条路径预热 3 次，每轮 10 次，共 5 轮；每轮轮转后端次序。以下统一采用**同步墙钟中位数**，包括本次调用的主机调度与完成同步；原始文件也保留 CUDA event 时间。

| 路径 | 计时范围 | 中位数 ms | 有效 TFLOPS |
|---|---|---:|---:|
| 原 MX32 Tensor Core 内核 | packed 输入、解码、逐 MX32 缩放/累加、FP32 输出 | 5.8975124 | 2.9131 |
| shared-memory tiled，variant 1 | 同一逐块合同，复用解码与数据加载 | 0.7220190 | 23.7942 |
| 精度证书快速路径，variant 2 | 每次重新检查证书、归一化、cuBLAS、条件缩放及回退调度 | **0.3371798** | **50.9517** |
| cuBLAS compute 对照 | 预先物化的 FP16 输入 → FP32 输出 | 0.2393850 | 71.7667 |
| 反量化 + cuBLAS 对照 | 每次把 packed 输入物化为 FP16，再计算 FP32 输出 | 0.3386379 | 50.7323 |
| 单独反量化 | 两个 packed 输入 → 两份 FP16 矩阵 | 0.1029671 | 不适用 |

有效工作量为 `2*M*N*logical_K`，所以 `TFLOPS=2*M*N*K/(ms*1e9)`。它不把整数解码、证书、额外回退或准备工作的操作数计入分子。相对同轮原内核，tiled 约快 **8.17×**，证书路径约快 **17.49×**；这是对原来低效基线的改进。证书路径与“反量化 + cuBLAS”对照只差约 0.4%，不足以宣称统计意义上的优势。

所有计时排除主机输入生成、H2D、高精度量化、内存分配和正确性 D2H。workspace 在调用之间复用，但**每次都重新生成归一化操作数并重新计算证书**，没有缓存输入的准入结果。通过/拒绝检查在 GPU 上完成；计时区间不逐次读回 flag。拒绝时仍执行一次无用 cuBLAS，再由 tiled fallback 覆盖，因此该成本不能在拒绝输入的 total 中隐去。本轮 2048³ 表中的证书通过，不能把其速度套用于被拒绝的数据。

输入是固定种子的完整 E2M1 nibble 均匀分布，每个 MX32 的尺度指数随机取 `[-2,2]`；并非真实模型张量。三轮 variant 对照的输入校验值均为 `17e7aed60a9341e3`。该受控输入也能精确反量化为 FP16，因此可使用普通 cuBLAS 作为性能参照；一般 E8M0 输入不能无条件走这条参照路径。

## 2. 编译与实际运行环境

GPU 为 Tesla V100-PCIE-32GB，80 SM、计算能力 7.0、ECC 开启。CUDA driver API 版本为 12040，构建与 runtime 为 CUDA 12.9（nvcc 12.9.86），cuBLAS 版本 120902。编译使用 `-O3 --ftz=false`，产生 SM70 cubin 和 compute_70 PTX。

这是保留原服务的**非独占实验**。两张卡都有其他服务占用约 32 GiB 显存；本轮 benchmark 初始化 context/cuBLAS 后只剩 `136,511,488` 字节。使用单份输出缓冲区串行运行各后端，将本例总分配限制为 `59,048,008` 字节，并显式保留至少 64 MiB、限制不超过当前空闲显存的 75%。没有停止服务、调节功率或锁频。各轮前后的 SM clock、memory clock 和功率记录保留在 JSONL；这些离散采样不能证明整段测量均无其他 GPU 活动。

环境和排队状态的后续快照见 [allocation-and-environment.json](allocation-and-environment.json)。它是测后采样，不替代 benchmark 自身 telemetry。

NVIDIA [V100 官方数据表](https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf) 给出的 Tensor Performance 是 PCIe **112 TFLOPS**、SXM2 **125 TFLOPS**。当前实测的是 PCIe 型号；125 TFLOPS 不是它的标称普通 dense Tensor Core 峰值。达到更高的有效吞吐需要新的工作量压缩、其他硬件形态或另外验证的算法，不能把 FP4 存储位宽直接换算成 V100 的原生 FP4 算力。本轮没有声称达到 125 TFLOPS。

## 3. 数值合同与加速方法

基础合同保持为：完整 E2M1 编码先转成整数 `q=2*value`，每个 MX32 计算精确整数 dot，再按 `ldexpf(qsum, sa+sb-256)` 缩放，并沿 K 顺序执行 FP32 加法。NaN scale 即使遇到零 dot 也传播 NaN；次正规数、上下溢出及跨块消去仍依原顺序处理。

### 通用 tiled 内核

[mxfp4_gemm_tiled.cu](../../../../tools/mxfp4_training/csrc/mxfp4_gemm_tiled.cu) 使用 64×64 输出 tile、128 个线程、共享内存输入复用，以及 PRMT 直接生成 half 位模式。选定 variant 1 仍以 MX32 为阶段，用 CTA 范围的正常尺度判断减少每个结果的分支。在证明乘积可精确表示的尺度范围内使用 FMA，其他输入保留完整范围处理。尾部、奇字节输入地址与宽索引均有对应路径。

实验还实现 BK64 variant 2，在同一 stage 中装入两个 MX32，但保持它们的独立缩放和原累加次序。它在本轮 1024³/1536³ 比较中没有胜过选定 variant 1。较大的 stage 不自动意味着更快；不能只按减少 barrier 的数量选择实现。

### 带证书的整 K 累加

[数值证明](../../../../docs/research/performance/sm70-packing/mxfp4-certified-fastpath.md) 给出充分条件。每行提出非零块最小 scale，将元素变成精确的 FP16 整数；精确整数能量 U、V 满足 `U*V<=2^48`，行内跨度不超过 12，最终缩放指数位于 `[-126,103]`。这样任何子集的归一化点积部分和绝对值不超过 `2^24`，所有归约顺序都精确，最后一次缩放与原逐 MX32 合同一致。

实现用整数安全比较、能量饱和和 NaN 检查决定准入。cuBLAS 使用 FP16 输入、FP32 输出与 `CUBLAS_COMPUTE_32F`，显式禁止 reduced-precision reduction。未通过时回退通用 tiled 内核。整个 pipeline 使用同一 stream；接受结果的零显式写为 canonical `+0`。

证书是保守充分条件，**拒绝不表示输入无法计算**。所有输入仍可走原精确路径。此方法也没有消除 MXFP4 对量化前 FP32 数据的量化误差：它保证的是既定 packed MXFP4 运算的结果。

### 向量化准备与输出

相同 tiled variant 1 配合三个 certified variant，分别编译并完整执行：

| certified variant | 改动 | total ms | total TFLOPS |
|---|---|---:|---:|
| 0 | 标量归一化准备与标量输出处理 | 0.3921876 | 43.8052 |
| 1 | 四元素 PRMT/DP4A 准备、向量化 half 写入 | 0.3410741 | 50.3699 |
| **2** | 在 variant 1 上增加按行的 float4 输出缩放 | **0.3371798** | **50.9517** |

variant 2 相比 variant 0 约减少 14.0% 延迟。非对齐输入/输出、非四倍数 N 及超过 65535 的 M 保留安全路径。所有变体共用同一准入证书，未通过调整精度门槛获取速度。

## 4. 正确性证据

### 独立全范围 oracle

选定 tiled 内核的 [46 例记录](iterations/build-final-v1/correctness-v2.jsonl) 检查 **175,429 个输出，0 mismatch**。覆盖完整编码组合、全部 scale 字节配对、NaN、Inf、正负次正规数、相反溢出块、K=0、非方阵/多 CTA 边界与奇字节输入地址。

certified variants 0、1、2 各自执行：

- 上述 46 例完整范围 oracle，175,429 个输出。
- 18 个附加 fixture，196,658 个输出：能量界上/界外、跨度 12/13、缩放指数界上/界外、±49152 归一化值、能量饱和、低尺度零块、负零、零块 NaN、全零、强消去反例、非对齐输出及 M=65537 的行循环。
- 同一 context 的接受→NaN 拒绝→再次接受，共 3 次、45 个输出；使用异步调用、不读取每次 flag，并检查工作区复用和提交次数。

因此**每个 variant 共 67 次 pipeline 调用、372,132 个输出，全部通过**；21 次接受、46 次回退，其中包括一次 K=0；另检查了 400 个接受路径零结果的正零位模式。数值合同不区分 NaN payload；一般结果比较不区分零的符号，接受路径的 canonical +0 则单独检查。

原始结果分别在 [c0](verified/build-perf-c0/correctness.jsonl)、[c1](verified/build-perf-c1/correctness.jsonl)、[c2](verified/build-perf-c2/correctness.jsonl)。这些 fixture 验证具体实现，不能把有限测试表述成对所有程序行为的形式化验证；证书的数值充分性另有上述推导。

### 实际计时输出的验证

每个 2048³ variant 在计时前，将 tiled/certified 的 **4,194,304 个输出全部与原内核比较，0 mismatch**，并额外检查 256 个独立 CPU oracle 输出。计时后再次比较实际最后一次计时调用的全部 certified 输出，同时检查证书元数据、调用次数和 workspace 复用。不是只验证另一次未计时调用。

### 框架层状态

本轮 CPU NumPy/PyTorch 单元测试仍为 **43 项通过**。`cuda_tiled` 已接入 PyTorch `MXRecipe`；原 `cuda` 保留旧后端。certified workspace API 目前仅为 native C API，**没有宣称已接入 MXLinear 训练路径**。

新 [check_torch_cuda.py](../../../../tools/mxfp4_training/check_torch_cuda.py) 准备了 119 项真实 SM70 框架检查，覆盖非默认 stream、GPU 量化、两种 native 后端及 F/D/W 开关。新 [bench_torch_cuda.py](../../../../tools/mxfp4_training/bench_torch_cuda.py) 将 FP32、预转换 FP16、转换+FP16、仅量化、量化+baseline、量化+tiled 分开计时，要求此前的 GPU 检查报告与当前源码、库和设备匹配。其普通 Torch FP16 对照输出仍为 FP16，不能与这里 FP32 输出的 native cuBLAS primitive 混为一谈。

这两项框架脚本已通过本地语法和 host 逻辑检查；在 CPU-only 环境正确报告 blocked。**截至本次归档，尚无它们的 GPU 通过记录，也未执行本轮 GPU decoder smoke。** 已实际提交 Crater 单张 V100、4 CPU、16 GiB 的任务，当前在队列中。等待脚本和远程 payload 的运行状态另记，不把“已排队”“脚本已写好”写成“GPU 训练已执行”。

[等待进程启动快照](queue-watcher-start.json) 记录了真实远端 tmux 中存活的 Python 进程，以及 `Pending` 状态。它每 60 秒检查一次，最多等待 12 小时；实际执行前要求发布的源码 commit 与归档/脚本哈希相符，并从认证后的当前作业日志固定 SSH host key。远端 payload 内限时 2700 秒、外层限时 3000 秒，按 native correctness → native benchmark → 框架 correctness → 量化成本及两条后端 smoke 的顺序运行，失败后保存证据。该启动快照早于源码 ready 标记发布；不是 GPU 运行结果。等待超时不会擅自取消已经提交的作业。

## 5. 证据组织与复现

| 目录/文件 | 内容 |
|---|---|
| `iterations/` | 初始共享内存、BK64、最终 tiled 0/1 的编译及小型测量；保留探索过程 |
| `verified/build-certified-v1/` | 第一版证书路径的构建、检查与测量；不是最终选定变体 |
| `verified/build-perf-c0/`、`c1/`、`c2/` | 固定最终源文件、不同宏配置的完整检查与 2048³ 对照 |
| `verified/run_vector_comparison.py` | 实际运行三变体比较的受限驱动脚本 |
| `verified/vector-comparison-complete.json` | 三次编译与六次 GPU 命令全部退出 0 |
| `comparison-summary.json` | 从已保存 JSONL 提取的指标及源哈希对应关系 |
| `cpu-certified-distribution.*` | 30 次 CPU 合成分布证书实验及独立复跑脚本；不是训练命中率 |

最终构建记录包含逐文件 SHA-256、完整 nvcc 参数、编译日志哈希和二进制哈希。所有 native 源文件与最终 c2 构建所记录的 SHA-256 匹配。最终默认选择 tiled variant 1 / certified variant 2；构建工具后来仅更新默认参数和后端 metadata，原始历史记录没有改写。

在有 CUDA 12.x 和真实 V100 的新输出目录中复现：

```bash
python tools/mxfp4_training/build_cuda.py \
  --nvcc /usr/local/cuda-12.9/bin/nvcc \
  --output-dir /tmp/mxfp4-perf-new \
  --with-benchmark --with-certified \
  --tiled-variant 1 --certified-variant 2

/tmp/mxfp4-perf-new/check_mxfp4 --correctness-only
/tmp/mxfp4-perf-new/check_mxfp4_tiled --correctness-only
/tmp/mxfp4-perf-new/check_mxfp4_certified --correctness-only

/tmp/mxfp4-perf-new/bench_mxfp4 \
  --m 2048 --n 2048 --k 2048 --mode random_block_scales \
  --iters 10 --rounds 5 --warmup 3 \
  --output /tmp/mxfp4-perf-new/benchmark.jsonl
```

正常节点使用默认显存余量。此次拥挤设备上的 native benchmark 显式使用 `--max-mib 64 --reserve-mib 64`；小型正确性检查也使用 64 MiB 余量。不要把这项实验特定的紧预算设为训练默认值。

## 6. 仍需完成的性能工作

首先在排队节点上执行框架正确性与 GPU smoke，并测量量化和布局生成的真实成本；再记录真实训练 F/D/W 张量的证书接受率。本实验之后的融合归一化解码、块能量缓存和预取已另行实现并验证，见 [融合实验报告](../2026-11-24-fused/README.md)。新增结果没有改写本页的历史测量，默认仍为 certified variant 2。

完整训练需要另外比较相同模型、token 预算、优化器、量化配方、总 step 时间、峰值显存及 loss。当前结果证明已有原生 MXFP4 运算路径明显提速，尚不证明真实 LLM 训练加速、收敛或显存收益。
