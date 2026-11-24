# V100 / SM70：低精度运算打包研究

研究日期：2026-11-22；训练基础续进：2026-11-23；性能优化：2026-11-24（Asia/Shanghai）。本文是独立研究记录；当前生产状态仍以 [production.md](../../../operations/current-state/production.md) 为准。

## 2026-11-24：真实 V100 上的正确性与性能优化

在完整 E2M1 / E8M0 算术上新增共享内存 tiled 内核，并以 [整数精度证书](mxfp4-certified-fastpath.md) 决定何时可将整个 K 维交给 cuBLAS Tensor Core 连续累加。未通过证书的输入回退完整范围 tiled 内核，保持原 MX32 缩放与 FP32 累加合同。

在与原服务共存的 Tesla V100-PCIE-32GB 上，固定合成输入 `2048³` 的同步墙钟结果如下：

| 路径 | 有效 TFLOPS | 范围 |
|---|---:|---|
| 原始 MXFP4 内核 | 2.91 | packed 输入 → FP32 输出 |
| shared-memory tiled | 23.79 | 同一完整范围逐块合同 |
| 证书快速路径 | **50.95** | 含每次证书、归一化、cuBLAS、输出缩放和回退调度 |
| cuBLAS compute 对照 | 71.77 | 预物化 FP16 输入 → FP32 输出 |
| 反量化 + cuBLAS 对照 | 50.73 | 本轮可精确转成 FP16 的 packed 输入 → FP32 输出 |

每个最终 certified variant 都完成 **67 次调用、372,132 个输出、0 mismatch**；计时前后另对 `2048³` 的 4,194,304 个输出作完整比较。通用 tiled 通过 46 例、175,429 个独立 oracle 输出检查。原始数据、三种变体、计时口径及边界见 [性能证据](../../../../evidence/results/mxfp4-training/2026-11-24-performance/README.md)。

这些时间从已打包输入开始，不包括 FP32→MXFP4 量化、H2D 或分配；接受证书的合成数据不能代替真实训练命中率。本轮没有达到 125 TFLOPS，也没有完整训练加速结论。`cuda_tiled` 已接入训练接口，certified API 目前仍为 native 路径。框架 GPU 检查、包含量化的 benchmark 和 GPU decoder smoke 已准备，单张 V100 的 Crater 任务已实际提交排队，尚未把这些未执行阶段记为通过。

## 2026-11-23：完整 MXFP4 算子与训练接口已落地

本节保留早期基线状态；后续 tiled 与性能进展以上节为准。

新增 [MXFP4 训练基础实现](../../../../tools/mxfp4_training/README.md)，使用完整 E2M1 编码、MX32 E8M0 尺度和 SM70 Tensor Core，实现单输出 `A @ B.T`。它为受限 packing 提供可核对的通用基线；目前不宣称 packing 加速、FP16 基线加速或训练显存节约。

在用户的 **Tesla V100-PCIE-32GB** 上，真实 CUDA 12.9 编译及执行已通过：**42 个 case、137,792 个输出，零不一致**，包括全部尺度字节配对、NaN、溢出、次正规数及非方阵尾块。编译使用 64 个寄存器，无 spill。原始编译、SASS 与执行记录见 [2026-11-23 证据](../../../../evidence/results/mxfp4-training/2026-11-23/README.md)。

新增 NumPy 独立 oracle、PyTorch `MXLinear` 一阶 surrogate backward、独立控制的 forward/dgrad/wgrad 和 RNE/SR 舍入。**43 项 CPU 测试全部通过**。一个 1 层、宽度 32 的因果 decoder 在合成 next-token 数据上完成五组各 40 步 AdamW 对照，均能反传、更新 FP32 参数和降低损失。

**V100 上确认的是独立原生算子的正确性；五组训练对照运行于 CPU oracle。** PyTorch CUDA 训练链路、真实文本预训练收敛及端到端速度比较仍待验证。两卡正在运行 vLLM，独立 CUDA 上下文建立后仅余约 144 MiB；本轮未改变服务。量化暂由多步 PyTorch 运算组成，kernel 尚未采用 shared-memory tiling，原始 FP32 activation 与 master weight 仍保留。

## 2026-11-22 packing 研究：结论与完成状态

以下记录保留原研究范围；其中“GPU 执行待完成”针对受限 packing 候选，不包括上述新加入的通用 MXFP4 基线。

**FP16 Tensor Core 的输入尾数较短，并不能直接证明所有低精度 packing 都不可行。受限数值域内已经构造出可恢复的编码，并通过 V100 专用数值模型检验。** 本轮进一步发现不对称编码和只恢复同一点积中间系数的方案，改善了支持的输入域或解包频率。

这些结果尚未证明 V100 上的端到端加速，也没有得到任意完整 MXFP4 × MXFP4 的通用双吞吐算子。真实 V100 执行、训练吞吐和收敛均未完成。所有 `hardware_time_results` 保持空值。

| 证据层级 | 已完成 | 能支持的结论 |
|---|---|---|
| 数学与 CPU 枚举 | 原始搜索、本轮扩展搜索、充分界与反例 | 指定域与解码器的可行性、失败边界 |
| V100 数值模型 | 公开模型移植，5,000 个硬件 fixture，原版 Octave 交叉验证 | 在该公开模型下的数值行为；fixture 是上游测量 |
| CUDA / SM70 编译 | 原探针和本轮完整布局 harness 的对象、SASS、资源报告 | 编译器接受、实际指令和寄存器成本 |
| Accel-Sim 周期模型 | 公开 LUD trace、人工 HMMA 依赖图 | 对模型参数与依赖关系的敏感性 |
| 当前候选的 GPU 执行 | **待完成** | 尚无实际 V100 正确性、耗时或加速比 |
| 完整 MXFP4 训练 | **未实现、未测量** | 尚无训练质量或经济性结论 |

完整可读材料在 [study](../../../../evidence/benchmarks/sm70-packing/study/)，两个此前交付 ZIP 在 [历史归档](../../../../evidence/results/sm70-packing/2026-11-22/)。归档按原字节保存，包含最初的 Soft-BF16 审阅、CPU 反例、CUDA 探针、数值模型与周期模拟，不用本轮结论覆盖旧记录。

## 1. 先确定运算目标和数据域

令 `q = 2 × E2M1元素值`。完整的数值集合为：

```text
q ∈ {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}
```

这一步把 E2M1 的半整数变成小整数，实际乘积恢复时再乘 `scale_A × scale_B / 4`。MXFP4 的块尺度是独立的信息，不能因为元素值域很小就省略它；这里按 32 项块处理。[OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)

本研究区分三类目标：共享 B 的两个点积、两个完全独立的点积、同一个点积的两段之和。第三类只需要一个结果，允许丢弃的中间信息更多。把它的正确性误写成“两路结果都可恢复”会夸大结论。

## 2. 新方案一：不对称基数扩大一侧范围

两个独立输出可以使用不同基数：

```text
P = a0 + 512 a1
Q = b0 + 128 b1
Z = Σ P Q = L + 128 X01 + 512 X10 + 65536 H

L = Σ a0 b0
H = Σ a1 b1
```

在 A 的两路 `q ∈ {-1,0,1}`、B 的两路使用完整 E2M1 集合、`K=4` 时：

- 所有 P、Q 均能被 FP16 精确表示。
- `|L|, |H| ≤ 48`，低结果可由中心化模 128 提取，因为 `48 < 64`。
- 高位干扰绝对值不超过 `48 × (1+128+512) = 30,768 < 32,768`，因此高结果可由 `round(Z/65536)` 恢复。
- 总绝对乘积界为 `4 × 513 × 1548 = 3,176,496 < 2^22`，也满足本轮位模式解码的范围条件。

这个方案把一侧的支持域扩展到完整 E2M1，另一侧仍必须是三值。它不等于任意 MXFP4 GEMM。相同编码扩展到 K=8 已有精确碰撞反例；把 A 域扩大到含 ±2，K=4 也有碰撞。具体输入和冲突结果保存在 [本轮数学结果](../../../../evidence/benchmarks/sm70-packing/study/simulation/round3/math/round3_math_results.json)。

如果两个输出共享同一个 B，约束还可以放宽：`P=a0+512a1`、B 使用完整 E2M1 时，A 的 q=-3…3、K=4 已验证 2,810 例；A 三值、基数 1024 时可到 K=32，已验证 2,210 例。共享 B 方案只打包 A，不具有任意两个独立 B 的适用范围。本轮五组候选合计 18,236 个数值模型检查全部通过，另有 2,049 个原始 MX32 配对流水线检查。

## 3. 新方案二：只恢复同一点积的中间系数

对同一个点积的相邻两项进行反向打包：

```text
P = a0 + 512 a1
Q = b1 + 512 b0
Z = L + 512 C + 512² H

C = Σ(a0 b0 + a1 b1)   ← 需要恢复的原始点积
L = Σ a0 b1            ← 可丢弃
H = Σ a1 b0            ← 可丢弃
```

这里两边均限定 `q ∈ {-3,-2,-1,0,1,2,3}`。在压缩后 K=8 或 K=12 时，用下面的解码恢复 C：

```text
h = round(Z / 512²)
C = round((Z - 512² h) / 512)
```

关键变化是：**不再要求整个 packed 累加值逐位精确，只要求其误差不越过中间系数的判决边界。** 对压缩 K=12，外侧系数绝对值界为 108，中间系数界为 216；公开 V100 模型下的保守 packed 累加误差界为 18。`108+18 < 256`，同时高位分离仍有余量，因此低位允许损失而 C 仍能精确恢复。完整推导与模型假设见 [数学报告](../../../../evidence/benchmarks/sm70-packing/study/simulation/round3/math/README.md)。

这一点已在模型中实际出现：压缩 K=12 的 4,531 个检查中，有 81 个 packed 总和不精确，观察到的最大误差为 4，但中间系数全部正确。压缩 K=8 的 4,531 个检查同样全部正确，其中一例 packed 总和损失 1。

对一个原始 K=32 的 MX 块，已验证的安排是：

1. 相邻项配对，使原始 32 项变为 16 个 packed 项。
2. 按 `8+8` 分成两个 packed 子段；每段执行两组 BFMA4，然后解码一次 C。
3. 两个 C 相加后乘这个原始 MX 块的尺度乘积。

在算术计数上，这把原始 8 组 BFMA4 变成 4 组，增加两次中间系数解码。**这是乘加工作量的变化，不是已测得的 2 倍加速。** 数据准备、解码、访存、寄存器和流水线代价尚需独立 kernel 测量。本轮硬件 harness 的目标仍是两个独立 GEMM，没有把这个单输出方案套进错误的基线。

压缩 K=16 的全极值输入会进位失败；不同尺度乘积的项也不能直接共用这个中间系数。两类反例均随结果提供。该方案适合在原始同尺度块内配对，不能跨任意 MX 块合并。

### 新解码器：直接提取有符号中间位

还能利用 `Z` 在当前模型下始终是 int32 范围内的 FP32 整数这一条件，直接提取中间的九位，省去先求 H 再相减的步骤：

```cpp
unsigned u = (unsigned)(__float2int_rz(z) + 256);
int middle = ((int)(u << 14)) >> 23;
```

无符号左移按模 2^32 处理，随后进行有符号右移。在当前 CUDA 12.8.93 的 SM70 编译中，偏置加法与左移融合成 LEA。独立小探针的公平比较为：

| 解码器 | 相同 float 输出端点的 SASS | 指令数 | 整个小探针的寄存器 |
|---|---|---:|---:|
| 原浮点公式 | FMUL、FRND、FFMA、FMUL、FRND | 5 | 12 |
| 直接 PTX BFE | F2I、IADD3、SHF、SGXT、I2F | 5 | 10 |
| unsigned shift / LEA | F2I、LEA、SHF、I2F | 4 | 10 |

若保留整数端点、先合并两个子段再做尺度转换，LEA 版本的核心解码为三条指令。这个端点单独统计，没有与 float 输出混比。五个小探针实际编译通过，均零 spill；指令条数不代表实测延迟或吞吐。

另对保守系数盒子的 23,772,133 个组合做了 CPU 枚举。其中 18,916,471 个满足“输入 z 为精确表示的 FP32 整数”，三种公式均零错误；其余任意构造组合明确列为不满足输入前提，没有把这些组合伪装为实际 BFMA4 输出。证明、源码、编译产物和结果见 [middle_decode](../../../../evidence/benchmarks/sm70-packing/study/simulation/round3/middle_decode/README.md)。

## 4. 本轮 CUDA 实现的具体进展

[mapped_packing.cu](../../../../evidence/benchmarks/sm70-packing/study/simulation/round3/cuda/mapped_packing.cu) 将以前只检查寄存器和指令的探针补成完整的受限域硬件测试程序：真实 E2M1 nibble 输入、两路独立 A/B 的 E8M0 块尺度、数据准备、矩阵输出、CPU 参考和同目标 cuBLAS 基线。

| 候选 | A 域 | B 域 | 解码间隔 |
|---|---|---|---:|
| `ternary_k32` | 三值 | 三值 | K=32 |
| `seven_k4` | q=-3…3 | q=-3…3 | K=4 |
| `ternary_full_k4` | 三值 | 完整 E2M1 | K=4 |

三条 packed 路径都明确拒绝完整 E2M1 × 完整 E2M1，没有裁剪输入。harness 生成的尺度指数限定在 [-2,2]；这不是对 E8M0 全范围或所有异常值的支持承诺。

实现根据官方 `m8n8k4` 定义，把一 warp 的四组 8×8 运算映射到 16×16 输出 tile。主机 C++ 编译实际共享头文件，再与独立 Python 映射比较。1,024 组 impulse 覆盖每个输出位置及 K4 的每个乘积位置；44 组配置分别用三种解码器验证，67,584 个输出值比较全部通过。[官方 fragment 定义](https://docs.nvidia.com/cuda/archive/11.4.0/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-884-f16)

CUDA 12.8.93 的 SM70 对象编译产生 24 个 mapped kernel 和 2 个准备 kernel，均无 spill。实际 SASS 确认：相邻结果改用 float2 后，输出存储指令从 16 条 STG32 变为 8 条 STG64，写入字节数相同。位模式解码版本不含 FRND、F2I、I2F，但增加的整数运算仍有成本，不能仅凭删掉某些 opcode 宣称最快。

通用 K、独立尺度和动态地址也有代价：magic 版本三值 K32 使用 62/64 个寄存器（尺度全一/独立块尺度），另两类 K4 为 64/72，均零 spill。不能把这些完整 kernel 与旧的固定 K、40 寄存器探针直接作性能比较。源码、对象、SASS 和资源报告的哈希见 [编译回执](../../../../evidence/benchmarks/sm70-packing/study/simulation/round3/cuda/compile-evidence.json)。

## 5. 对通用 MXFP4 和训练的判断

完整 E2M1 的元素有多个指数层级，受限整数域的正确性不会自动扩展到它。当前搜索覆盖线性基数家族和充分条件，不涵盖所有可能编码；未找到某类参数不构成信息论不可能证明。

把窄域作为普通 MXFP4 的自动 fast path 也需要谨慎：常见按块最大值选尺度会让块内至少一项处在较大幅值，不能假设整块自然落在 q≤3 或三值范围。本轮补充了合成分布准入检查与残差修复成本；它们用于说明条件，不代表真实模型的激活或梯度分布，详见数学报告。

例如，每个分布生成 2,048 个长度 32 的块，用同一固定 max-scaling 量化器处理，再检查共享 B 的两个 K4 行片段是否都可进入 q≤3 路径。精确提取片段公共 2 的幂次因子之后，准入率有变化：

| 合成输入 | 原始准入率 | 公共 2 的幂次因子提出后 |
|---|---:|---:|
| Gaussian | 1.45% | 5.43% |
| Laplace | 6.71% | 11.45% |
| Student-t，3 自由度 | 11.36% | 16.89% |
| 90% 零的 Gaussian | 55.32% | 95.04% |
| 三值正控制 | 0% | 100% |

三值正控制的原始准入为零，是该量化器把其非零值编码到较大的 q，而公共因子可精确消去；这不代表把任意多值输入变成了三值。表格只支持后续优先检查匹配的数据结构，不能推导真实模型的命中率或端到端收益。

简单加一条 dense 残差修复也没有免费收益：两个共享 B 输出原来需要两份 dot，packed 核心需要一份，但若每个输出再各做一份 dense 残差 dot，总数变成三份，即基线的 1.5 倍算术工作量。按 E2M1 指数分成三个窄域桶同样可能需要三份 packed 工作。稀疏修复另有索引、收集和调度成本，本轮只记录补乘数量，没有把它们等同于 Tensor Core 周期。

训练中还需要权重、激活、梯度各自的量化与尺度管理、转置布局、数据准备成本和收敛实验。本轮没有做这些验证。因此可以说“某些编码在 V100 模型下成立，值得对匹配工作负载测量”，尚不能说“V100 已具备原生 MXFP4 训练能力”，也不能仅凭猜测宣布训练一定无用。

## 6. 前两轮材料的保留与解释

### Soft-BF16-V100 审阅

审阅固定在 `S-i-l-v-e-t/Soft-BF16-V100@3f7bf2cef67cc7b0d37966b5c85fed9ba4c350c5`，见 [原始审阅](../../../../evidence/benchmarks/sm70-packing/study/audit/audit-notes.md)。代码确实包含缩放、FP16 输入 / FP32 累加的库调用、scratch 管理和 fallback，但不能由此推断有自写 HMMA、完整训练验证或固定倍数加速。二维与批量 fallback 范围不同；SDPA 和卷积路径也不能概括为同一种精确算法。

CPU 反例表明范围 guard 不等价于逐值无损；例如半精度次正规数舍入会使允许范围内的较小值变化。审阅记录、复现脚本和结果全部保留，未用宣传性评价替代性能证据。[固定版本源码](https://github.com/S-i-l-v-e-t/Soft-BF16-V100/tree/3f7bf2cef67cc7b0d37966b5c85fed9ba4c350c5)

### 数值模型

使用 [Accurate Models of NVIDIA Tensor Cores](https://arxiv.org/abs/2512.07004v4) 的公开 V100 模型，作者仓库固定在 `6fd0520850164bdd1c37f901954e27ec63eada56`。Python 移植对 5,000 个公开硬件 fixture 的逐位差异为零；原版模型在 Octave 中也通过 5,000 个 fixture 和 885 个选定 packing/负控制样例。六类受限方案共 30,940 次原始数值检查通过。

这些公开样本集中在有限的正常值范围，不能称为对所有 FP16、非正规数、NaN/Inf 的硬件等价证明。普通逐项 IEEE FP32 累加并不等价于 V100 Tensor Core 的实际数值行为，所以本研究保留 V100 专用模型及其来源验证，而不以 NumPy matmul 冒充硬件。

### 周期模型

归档包含实际构建的 [Accel-Sim](https://github.com/accel-sim/accel-sim-framework) 和固定版本配置。公开 LUD64 trace 重放得到 10 个 kernel、127,309 个模型周期、418,884 条线程指令；该 LUD trace 不含本研究 packing kernel，也不用于证明 Tensor Core 候选的加速。

人工 HMMA trace 用于比较依赖链数量和延迟参数：一条链与多条链的模型耗时不同，说明单看 HMMA 数量不足以推断性能。Quadro GV100 配置、人工 trace 和原始日志均保留。当前没有候选的真实执行 trace，不能把周期模型输出换算成用户 Tesla V100 的可信跑分。

## 7. 复跑和证据入口

在仓库根目录执行：

```bash
python3 tools/sm70_packing/run_cpu_validation.py \
  --require-round3 --random-cases 4096 \
  --output-dir /tmp/v100-packing-validation
```

依赖为 Python 3.10+、NumPy、C++17 编译器；输出目录须尚不存在。runner 在隔离副本中重跑，检查源证据哈希未变，逐项声明哪些是新 CPU 计算、哪些只是解析历史编译或周期模型记录。`--require-round3` 会在本轮检查缺失时失败，不会只用旧测试报告本轮全通过。

原版 Octave 和周期模型为显式可选项：`--octave-bin /path/to/octave-cli`、`--simulate`。CUDA 重编译方法和真实 GPU 运行入口见 [harness 说明](../../../../evidence/benchmarks/sm70-packing/study/simulation/round3/cuda/README.md)。

整合到仓库最终目录之后，已从仓库根目录实际复跑上述 16 项检查，全部通过；68 个受保护输入及历史结果哈希未变。完整摘要和原始日志见 [整合验证回执](../../../../evidence/results/sm70-packing/2026-11-22/integrated-validation/)。

```bash
cd evidence/benchmarks/sm70-packing/study/simulation/round3/cuda
./run_hardware.sh --device 0 --correctness-only
./run_hardware.sh --device 0 --warmup 10 --iters 100
```

GPU 程序分别记录同一压缩输入的准备时间、预处理后计算时间、准备加计算的端到端时间。基线为一次 cuBLAS strided-batched 调用完成两个独立 GEMM；所有输出先与独立 CPU 参考比较。这里的端到端从 nibble/scale 输入开始，不包含训练框架、原始浮点量化、优化器或通信。尺寸是小规模正确性/微基准范围，不是峰值吞吐承诺。

## 来源与许可证

公开模型最小快照保留作者许可证及 11 个文件的固定哈希；模拟器、half 类型头文件和 pybind11 相关许可证位于各自证据目录。未纳入完整 CUDA SDK、Octave 安装或上游 Git 工作树。实验性派生代码与第三方原文件通过目录和来源清单区分。历史 ZIP 哈希见 `archives.json`，可读实验树哈希见 `study/SHA256SUMS`。
