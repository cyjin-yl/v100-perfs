# V100 / SM70 打包计算：数值仿真与结构反推实验

日期：2026-11-22。本文是上一轮 FP16 packing 实验的补充；这一轮实际在本地运行了 V100 数值模型、作者原版 Octave 模型和 Accel-Sim 周期模型，并编译了新的 SM70 CUDA 候选。

## 结论

**在受限数值域内，把两份低精度乘加装进一次 FP16 Tensor Core 计算是可行的。本轮把这个结论从普通 CPU 浮点运算推进到了经过公开 V100 硬件样本校验的数值模型，并找到了 K=4 的新实现入口。完整范围 MXFP4 的通用加速训练算子仍未实现。**

最有用的新结果是双侧 `R=256, K=4, q∈{-3,…,3}`：两个操作数都能打包，最大累加值 2,377,764，模型正确，能使用便宜的位模式解包，且 SM70 原生 `mma.m8n8k4` 已编译成功。`q=2x`，所以它覆盖 E2M1 中 `x∈{0,±0.5,±1,±1.5}`，不是 E2M1 的全部数值。

另一条更适合优先评估性能的路线仍是 `R=256, K=32, q∈{-1,0,1}`。其解包频率较低。本地编译把同等两份输出的 HMMA 指令从 64 条减到 32 条；但完整直线指令只从 233 条降至 214/215 条。因此本报告不把 HMMA 减半换算成训练加速倍数。

## 1. 本次究竟执行了什么

| 工作 | 实际执行结果 | 证据范围 |
|---|---|---|
| V100 数值模型的 Python 整数移植 | 5,000 个作者公开硬件输出样本，逐 FP32 bit 零差异 | 数值模型与已公开样本的一致性 |
| 未修改的作者原版模型，在 GNU Octave 10.3.0 中运行 | 同样 5,000/5,000 硬件样本一致；885/885 本次选定 packing/control 样本与移植版一致 | 避免仅靠自行移植得到结论 |
| 六种 packing 参数的模型测试 | 30,940 组随机、常量极值、交替符号样本，零解包错误 | 这些域内的计算结果；另附充分条件证明 |
| 结构约束搜索 | 21,504 组参数通过/失败判定 | 仅线性 radix family 的充分条件，不是所有编码的穷举 |
| 新位模式 decoder | 三个系数边界集合共 3,133,667 组，零错误 | CPU FP32/整数位模式算法 |
| SM70 编译 | CUDA 12.8.93 编译成功，实际反汇编包含 HMMA.884 | 编译、指令数、寄存器和 spill 信息 |
| Accel-Sim 构建与运行 | 上游 Tesla-V100 LUD64 trace 的 10 个 kernel 完成重放；合成 HMMA 探针完成周期仿真 | 模拟器可运行，及给定模型内的调度敏感性 |
| 用户 V100 上的新执行 | 本次没有 | 仍不能提供本机实测吞吐或训练收敛结果 |

所有实验脚本、原始日志、输入、结果 JSON、源码来源与 SHA256 随本报告提供。模型中的“hardware fixture”是作者以前在硬件上生成并公开的数据，不是我们这次在用户电脑上测得的数据。

## 2. 采用的公开设计与模型

### 官方架构与 ISA

NVIDIA 的 Volta 白皮书、Volta Tuning Guide 和 PTX ISA 提供 SM 结构、调度器、寄存器资源、Tensor Core 支持的数据类型和 MMA fragment 规则。它们足够支持这次约束搜索和目标指令编译；不是完整 V100 RTL 或门级网表。

特别有用的是 `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`。官方文档说明该形状针对 SM70 优化。一个 warp 执行的是四份 8×8×4 MMA，各线程持有规定的 fragment；不能把它误读成 32 线程共同计算一份普通 8×8。我们的最小编译探针产生四条 `HMMA.884.F32.F32.STEP0…STEP3`，22 个寄存器，零 spill。

### 经过硬件行为测量的数值模型

使用 Faizan A. Khattak、Mantas Mikaitis 的 *Accurate Models of NVIDIA Tensor Cores* 及作者公开仓库：

- 论文：<https://arxiv.org/abs/2512.07004>
- 代码：<https://github.com/north-numerical-computing/MATLAB-tensor-core>
- 固定提交：`6fd0520850164bdd1c37f901954e27ec63eada56`

本次实现 FP16 输入、FP32 输出的 V100 分支：每组四个乘积，乘积保留未规格化形式，与 C 一起全局对齐；对齐阶段没有额外 guard/sticky 位，低位直接截断，最后按模型的 RZ 规则生成 FP32。

这是有意义的区别。对同一批 5,000 个公开硬件样本：

| 计算方法 | 与硬件输出不同的样本数 |
|---|---:|
| 作者原版 V100 模型 | 0 |
| 本次 Python 位模型 | 0 |
| 普通 FP32 顺序累加 | 2,104 |
| FP64 求和后一次舍入成 FP32 | 1,885 |

公开 fixture 全部为有限正常值；它不是全 FP16/FP32 输入空间的穷举，不能用这个数字声称所有 NaN、Inf、subnormal、不同 kernel 累加顺序都已在硬件上验证。本次另有极端值和子正规数的控制测试与原版模型交叉验证，但那仍是模型交叉验证。

### 周期模型

实际构建并运行：

- Accel-Sim：`7ba54ded60c4ba0083410c37d22d719166acd6dc`
- GPGPU-Sim fork：`4195dd4f229cbf03f4f38c49139ae94d01d23f03`

采用上游 `SM7_QV100` 配置，80 SM、core 1,132 MHz、HBM 850 MHz。它是 Quadro GV100 的配置，不是对用户 Tesla V100 PCIe/SXM 的逐项校准。周期数据以 model cycles 报告，不换算成用户卡的 TFLOPS。

没有采用 GPGPU-Sim 的 PTX `mma_impl` 功能路径作为数值真值。所检查版本的该路径存在 half 类型中间表达式；对其同款 half 运算的独立复现中，`1539×3` 为 4,616，而 FP32 正确值为 4,617，16 项还会溢出成 Inf。这不足以代表本任务的 FP32 累加语义。这里把数值模型和时序模型分别校验。

## 3. 从结构反推得到的可行域

对两个独立 dot product，令：

\[
A_k=a_{0k}+R a_{1k},\qquad B_k=b_{0k}+R b_{1k}.
\]

一次 packed dot 的输出为：

\[
Z=\sum_k A_kB_k=L+RX+R^2H,
\]

其中 `L=Σa0b0`、`H=Σa1b1` 是目标，`X=Σ(a0b1+a1b0)` 是交叉项。交叉项不必为零，只要有足够间隔且舍入不破坏需要的位，就能把两份目标恢复。

若所有 q 的绝对值不超过 Q，下列是本次使用的充分条件：

1. 对输入字母表中每一对值，`a0+R*a1` 都必须能被 FP16 精确表示。
2. `2*K*Q² < R`，使低位系数可恢复。
3. `2*K*Q²*(2R+1) < R²`，使最高位系数不受低位和交叉项污染。
4. `K*Q²*(R+1)² < 2²⁴`，使任意部分和保持在 FP32 精确整数范围内。输入及乘积均为整数，这些候选的 V100 对齐网格不会截掉非零整数字位。

这不是必要条件集合；不通过仅表示此证明不适用。完整非线性映射、近似计算或额外修复通道不在这次搜索中。

实际数值模型测试结果：

| 输入 q 字母表 | 方式 | R | 每次解包的 K | 测试数 | 解包错误 |
|---|---|---:|---:|---:|---:|
| {-3,…,3} | 共享 B，A 两路打包 | 512 | 16 | 4,466 | 0 |
| {-1,0,1} | A、B 都各打包两路 | 128 | 16 | 4,258 | 0 |
| {-1,0,1} | A、B 都各打包两路 | 256 | 32 | 4,258 | 0 |
| {-3,…,3} | A、B 都各打包两路 | 512 | 4 | 6,578 | 0 |
| {-2,…,2} | A、B 都各打包两路 | 256 | 8 | 4,802 | 0 |
| {-3,…,3} | A、B 都各打包两路 | **256** | **4** | **6,578** | **0** |

每一行还检查了 packed 累加值与精确整数是否相同，以及 V100 模型对齐阶段是否丢掉非零位，均为零异常。测试是随机与构造样本的组合，并非穷举所有长度 K 的输入向量。

### 新 K4 候选为什么成立

取 Q=3、R=256、K=4：

- packed 输入绝对值至多 `3+256×3=771`，FP16 精确。
- `|L|,|H|≤36`，`|X|≤72`。
- 最高位的干扰至多 `256×72+36=18,468 < 65,536/2`。
- 所有部分和的绝对值受 `4×771²=2,377,764 < 2²²` 约束。

因此高位 `H=round(Z/65536)` 可恢复，低位 L 可用平衡余数提取。这个域比 ternary 更宽；K4 指令使“每四项就解码一次”有实际 ISA 入口。

代价也直接可见：K4 时解码相对乘加更频繁，数值可行不保证这个候选比普通 FP16 更快。它适合扩展正确性和布局实验；K32 ternary 更适合先检验性能收益。

## 4. 模型确实暴露了失效边界

### 仅仅扩大 radix 不能解决问题

在一个 ternary R1024/K16 构造案例中：

- 精确整数结果：16,783,363。
- 普通 FP32 顺序累加：16,783,364。
- V100 数值模型：16,783,362。
- 解包低位目标是 3；普通 FP32 得 4，V100 模型得 2。

这里不是“模型都一样，只是换了名字”：实际舍入路径会改变低位损失。

另一个 q3/R512/K8 案例，`A=B=[1537,1536,…,1536]`，精确结果 18,877,441，V100 模型得到 18,877,440，目标 `(L,H)=(1,72)` 被解为 `(0,72)`。在 K4 安全不代表延长到 K8 仍然安全。

### 完整 E2M1 的旧碰撞仍然成立

完整 E2M1 的整数化字母表为 `q∈{0,±1,±2,±3,±4,±6,±8,±12}`。共享 B、R128 的单次乘积可以借助查表区分，但累加后存在严格信息碰撞：

| B | A0 | A1 | 目标 (ΣA0B,ΣA1B) | packed 输出 |
|---|---|---|---|---:|
| [8,1] | [8,0] | [0,0] | (64,0) | 64 |
| [8,1] | [-8,0] | [0,1] | (-64,1) | 64 |

补零到 K4/K16 不改变碰撞。这个例子在 V100 模型中也完全相同。它否定的是这套特定编码和单一结果恢复方案，不是否定所有可能的低精度编码。

## 5. 指令和解包成本

公平对照使用相同四 warp CTA、相同两份输出、相同 K32；输入已预先打包，缩放均设为 1，结果按 q=2x 乘 1/4。

| 版本 | HMMA | 完整直线指令数 | 寄存器/线程 | spill |
|---|---:|---:|---:|---:|
| 两份普通 FP16 MMA | 64 | 233 | 56 | 0 |
| 打包，固定常数浮点解码 | 32 | 214 | 40 | 0 |
| 打包，转换到整数再解码 | 32 | 230 | 38 | 0 |
| 打包，FP32 位模式解码 | 32 | 215 | 40 | 0 |

完整指令数包含地址计算、读写、解码，排除了 EXIT 后填充与恒假启动指令。这些是机器码静态统计，不是实际耗时。寄存器资源预算改善也不是测得 occupancy。

位模式解码利用 `|Z|<2²²`，把 `Z+1.5×2²³` 放入 FP32 相邻整数对应连续位模式的区间。它把原来的 `32 FMUL + 16 FRND + 16 FFMA` 变成 `8 FADD + 16 FFMA`，消除了 F2I/I2F/FRND，但增加整数指令。总数没有超过固定常数浮点版本的优势，值得硬件比较其管线行为。

本轮精确穷举的解码系数超集为 K32 ternary 的 545,025 组、K48 ternary 的 1,815,937 组、q3/K4 的 772,705 组，全部零错误。K48 只能在尺度兼容时考虑，不能直接越过标准 MX32 的尺度边界。

此外，radix packed half 每个原始数平均占 8 bit，不等于 MXFP4 的每元素 4 bit 存储。在训练或大矩阵比较中，应以“压缩 FP4 输入、片上解码/反量化”的实现为重要基线，不能靠放大基线内存格式得到虚假的收益。

## 6. 实际周期仿真结果

先用 Accel-Sim 上游公开的 Tesla-V100 LUD64 trace 跑通 10 个 kernel：`gpu_tot_sim_cycle=127,309`、`gpu_tot_sim_insn=418,884`。这是模拟链自检，不是本次 packing kernel 的性能。

然后生成不含全局内存访问的合成 HMMA 依赖图。每 warp 256 条 HMMA，给定上游 QV100 配置，实际得到：

| 情形 | 模型总 cycles | 减去匹配空探针后的增量 |
|---|---:|---:|
| 1 warp，1 条依赖链 | 6,818 | 1,790 |
| 1 warp，4 条独立链 | 5,543 | 515 |
| 1 warp，8 条独立链 | 5,658 | 630 |

匹配空探针是 5,028 cycles，其中配置明确含 5,000 cycles 的 kernel launch 模型。每种单 warp 情形使用相同的寄存器预算和初始化指令。表中增量只是相减后的模型量，不是一个可直接部署内核的延迟。更多链改变寄存器依赖与调度；四链和八链结果并不单调，这里没有进一步归因其 stall，不能拿这些数字当作纯吞吐公式。

把 HMMA 模型 latency 从 2 改为 8、initiation 保持 2 的敏感性测试中，三种情况变为 8,098、5,800、5,778 cycles。它支持一个设计方向：多条独立累加链和合理 warp 数量比一条长串依赖更能隐藏该模型中的等待。它没有测量真实 V100 的 HMMA latency，也没有给出 packing 的加速倍数。

本轮没有把静态 SASS 简单删掉 load/store 后伪造成候选 kernel 的动态 trace。寄存器复用、HMMA 隐含寄存器对、LEA 的多种用途，会使这种删减产生错误依赖。真正的候选 kernel 周期估计仍需要可靠的数据流提取或真实 NVBit trace。

## 7. 对 MXFP4 训练的实际含义

OCP MXFP4 包含完整 E2M1 元素、每 32 项共享的 E8M0 尺度。这里的整数域 q=2x 只研究块内规范化数值。

把尺度放到解包之后处理，是合理的实现方向：先算两个块的规范化 dot，再分别乘回各自 `scale_A * scale_B / 4`，最后累加。不同块的尺度不能在 packed 累加器里随意混合。这个处理顺序既不自动覆盖完整 E2M1，也不保证其舍入与另一套 MXFP4 硬件逐位相同。

当前已建立：可行的受限整数域、V100 数值行为下的正确性证据、原生 K4 编译入口、解码候选和成本分析。尚需实现或验证：完整码域处理、量化和尺度选择、矩阵 fragment 布局的 GPU 正确性、长 K 融合累加、forward/dgrad/wgrad 性能及训练收敛。

所以此前“FP16 只有 10 位尾数，任何 packing 都不可能”的断言站不住。另一端的“已经实现全范围 MXFP4 训练翻倍”也没有证据。已有结果足够支持继续做一个受限域的 SM70 内核实验，而不需要靠猜测来论证可行性。

## 8. 文件和复现入口

- `numerics/README.md`：作者模型、数值测试、原版 Octave 交叉验证及覆盖范围。
- `numerics/run_numerical_study.py`：跑 Python 模型与公开 fixture；`numerical_results.json` 为本次结果。
- `numerics/run_original_octave.m`：运行未修改原版模型；`original_octave_results.json` 为本次结果。
- `search/structural_search.py`：重跑 21,504 组充分条件参数搜索。
- `cost/README.md`：完整指令、寄存器、解码和带宽成本；`reproduce_compile.sh` 复现编译。
- `architecture/README.md`：Accel-Sim 的构建、来源、trace、配置、周期与局限。
- `RUN_STATUS.json`：机器可读的执行状态与证据边界。
- `SHA256SUMS`：交付文件校验。

基础 CPU 重跑命令（Python 3 + NumPy；在此包的 simulation 目录中）：

```bash
python3 numerics/run_numerical_study.py --random-cases 4096
python3 search/structural_search.py
python3 cost/validate_magic_decode.py
python3 cost/analyze_cost.py
```

Octave、CUDA 12.x 和 Accel-Sim 的运行依赖及路径见各子报告。没有打包整个 Octave/CUDA 环境；CUDA 编译用的私有 SDK 曾为 glibc 2.41 修正四条未使用的 sinpi/cospi 声明，补丁及上一轮编译证据随包附带，不涉及用户系统的更改。

## 主要来源

1. NVIDIA Volta Tuning Guide：<https://docs.nvidia.com/cuda/volta-tuning-guide/>
2. NVIDIA Tesla V100 Architecture Whitepaper：<https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf>
3. NVIDIA PTX ISA（mma.m8n8k4）：<https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html>
4. Khattak & Mikaitis, Accurate Models of NVIDIA Tensor Cores：<https://arxiv.org/abs/2512.07004>
5. 作者模型仓库：<https://github.com/north-numerical-computing/MATLAB-tensor-core>
6. Accel-Sim：<https://accel-sim.github.io/> 与 <https://github.com/accel-sim/accel-sim-framework>
7. OCP Microscaling Formats v1.0：<https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1.0-spec-final-pdf/>

本报告中的代码计数、错误计数、范围搜索及模拟周期来自本次实际执行，以上来源用于架构、格式和模型的定义。原模型及模拟器的许可分别保留，不把第三方工作称为本次原创。
