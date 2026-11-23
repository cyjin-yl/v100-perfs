# V100 MXFP4：从硬件与文献确定下一轮优化

本轮继续保持现有 full MX32 算术、整数证书、精确拒绝回退、FP32 输出与 canonical `+0` 契约。根据 V100 的资源和原始研究资料实现的C8，在充分预热的1536³配对中将完整算子吞吐从32.893提高到39.961 TFLOPS，约21.49%；共享设备的测量限制和全部原始输出见§5。[上一轮融合实验](../2026-11-24-fused/README.md)保留C3–C7历史；默认certified2不变。§1–4的资源公式与bank模型是静态推导，实际GPU证据另行标明。

## 1. 125 TFLOPS 对应哪种 V100

NVIDIA JAN20 数据表明确区分三种型号 [H1]：

| 官方规格 | V100 PCIe | V100 SXM2 | V100S PCIe |
|---|---:|---:|---:|
| Tensor Cores | 640 | 640 | 640 |
| FP32 CUDA cores | 5,120 | 5,120 | 5,120 |
| FP32 SIMT 峰值 | 14 TFLOPS | 15.7 TFLOPS | 16.4 TFLOPS |
| Tensor 峰值 | **112 TFLOPS** | **125 TFLOPS** | **130 TFLOPS** |
| HBM2 带宽 | 900 GB/s | 900 GB/s | 1,134 GB/s |
| 最大板卡功率 | 250 W | 300 W | 250 W |

当前设备是 `Tesla V100-PCIE-32GB`。PCIe Product Brief 给这个32GB型号的 base／maximum boost 为 **1230／1380 MHz**，HBM clock为877 MHz，总线4096-bit [H2]。不能将 SXM2 或 V100S 的规格直接套到本卡。

量产 V100 启用80个SM，每SM有8个Tensor Core；每TC每周期64个FMA，即128个FLOP [H3]。在所有SM始终维持1380 MHz、TC完全满载的理想假设下：

```text
80 * 8 * 64 * 2 * 1.38 GHz = 113.0496 TFLOPS
125e12 / (80 * 1024) = 1525.87890625 MHz
```

113.05是架构×时钟计算值，112是官方标称值。当前 C6 将一个MXFP4数值解码为一个normalized FP16数值，再执行dense HMMA；无padding/tail时，主计算的native FLOP与useful `2MNK`一致。沿用这套映射，125 TFLOPS已经超过本卡最大boost所对应的原生TC预算，完整pipeline还要支付证书、解码、同步和输出成本。

**这不是对任意算法 useful MXFP4 TFLOPS 的上限证明。** 如果新的精确编码能让一次原生运算完成更多逻辑工作，useful计数可以不同；但交叉项、carry、指数跨度及结果恢复都需要新的证明。仅把输入存成4-bit，不会自动让当前FP16 HMMA多完成几倍点积。

## 2. C6 静态资源揭示的优化边界

Volta每SM有65,536个32-bit寄存器、最多64个驻留warp、最多32个CTA，以及最多96 KiB shared；L1/shared合计128 KiB [H4]。寄存器按每warp 256个向上分配 [H6]。结合已有ptxas记录，可得以下资源上限；它不是实测 achieved occupancy：

| 路径 | threads/CTA | regs/thread | 分配regs/CTA | 寄存器允许CTA/SM | 对应warp/SM |
|---|---:|---:|---:|---:|---:|
| C6 int32 | 256 | 128 | 32,768 | 2 | 16 |
| C6若增加到129 regs | 256 | 按粒度等效136 | 34,816 | 1 | 8 |
| C7 int32无尾部 | 128 | 251，按粒度等效256 | 32,768 | 2 | 8 |

**128→129 regs是C6必须避免的驻留断点。** 没有spill不等于增加暂存没有代价；增加几个fragment或地址寄存器，可能使可驻留CTA减半。另一方面，降到120甚至88 regs仍只允许两个CTA；达到三个256-thread CTA大约要求≤80 regs/thread。已有64个FP32 accumulator，所以减少raw寄存器首先是给流水留空间，而非保证更高occupancy。C7在已测1536³的int32路径没有spill，其int64 spill只能作为另一个宽索引资源限制。

C6的shared为 `2*128*9*8+256=18,688 B/CTA`。若双缓冲输入、row bases仍保留一份，则为37,120 B/CTA；两CTA要74,240 B，需96 KiB shared配置，不能沿用64 KiB容量预算。更大的shared carveout会减少L1，必须与寄存器、实际active blocks和缓存行为一起评估 [H4]。

每个128×128×32 stage执行1,048,576 FLOP，distinct输入为4096 B packed codes加256 B scales，约240.94 FLOP/B。即使假设1536³的每CTA输入重新从HBM读取，再加一次prepare输入和FP32输出，简化流量为42,024,960 B、172.46 FLOP/B；900 GB/s对应155.22 TFLOPS的理想流量预算。这忽略事务放大、cache、指令发射、延迟及同步，不能据此证明compute-bound，但说明不能只盯着HBM字节数。

128×128 tile在1536³、2048³、4096³分别产生144、256、1024个CTA。前者仅平均1.8 CTA/SM，容易受到有限网格和流水启动／结束比例影响。V100有6 MiB L2；两个packed输入连scale分别约2.39、4.25、17 MiB，输出和其他进程还会争用缓存 [H3]。形状与负载窗口都需要保持可比较。

## 3. 文献里哪些技术可以迁移

| 一手资料 | 对本项目有用的部分 | 必须保留的边界 |
|---|---|---|
| NVIDIA Native Volta Tensor Cores with CUTLASS，GTC2019 [H8] | 原生`m8n8k4`、permuted shared layout、128-bit访问、按phase检查bank，以及Volta迭代器设计。 | 具体布局依据这份Volta原始讲稿；不从已经不列Volta的新版Functionality表推导。扩大load宽度须同时检查线程映射、对齐和shared事务。 |
| CUTLASS Efficient GEMM [H7] | CTA shared与warp fragments两层软件双缓冲，tile复用、rasterization和有限网格的权衡。 | 是设计方法，不保证某个本地candidate更快；双缓冲会占更多shared和寄存器。 |
| Jia等，Dissecting the NVIDIA Volta GPU Architecture，2018 [P1] | SASS、寄存器bank/reuse、HMMA映射和延迟的微基准分析方法。 | 文中寄存器重排示例是NVCC9.0的FFMA，不能把其收益比例套到CUDA12.9的当前HMMA。 |
| Ladder／Microsoft BitBLAS [P2/P3] | 联合选择global/shared/register布局和转换位置，压缩转换临时值生存期；论文包含V100实测。 | BitBLAS表中的V100 `FP16 × FP4_E2M1` 是FP16输出，非本项目双MXFP4、FP32 exact契约。 |
| Marlin [P4] | 较宽向量load、静态索引、decode与计算交错、权重及scale预排布。 | 官方要求SM80+；其全局归约暂时把FP32 accumulator转FP16，不能照搬到本项目。小batch W4A16推理的倍数不等于大GEMM训练收益。 |
| FLUTE [P5/P6] | 非均匀LUT量化的离线重排、unpack与LUT向量化／复制。 | 官方GPU表没有V100。当前C6已用寄存器PRMT解码，新增shared LUT未必有利；训练中activation、gradient和更新后权重的重排成本不能忽略。 |
| Tseng等，Training LLMs with MXFP4，AISTATS2025 [P7] | 随机舍入用于无偏梯度估计，随机Hadamard变换控制块内异常值导致的方差；为训练配方和收敛验证提供依据。 | 论文对支持MXFP4硬件的速度估计不能套到V100。训练配方近似误差与本项目packed GEMM实现的逐值正确性是两个需要分别验证的问题。 |

面向pretrain，最终还要计入activation／weight／gradient量化、变换与重排、前后向算子、optimizer和框架调度。当前native certified C API的吞吐不能写成完整训练TFLOPS；证书接受率也需要在真实训练张量上测量。不能将某篇论文的模型质量结论当成本项目已经完成的收敛实验。

SM70支持并特别优化FP16 `mma.sync.aligned.m8n8k4`；一个warp执行的是四个8×8×4 MMA。`ldmatrix`要求SM75，`cp.async`要求SM80，不能将较新架构的指令作为V100实现前提 [H5]。本卡需要普通global load到寄存器，再发布shared，以及shared到fragment的显式软件流水。

## 4. C6 已有流水与 C8 设计

C6已有下一MX32块的raw global预取，且**编译后的SASS已经将部分下一K4步的B fragment LDS交错到当前HMMA中**，不是完全没有fragment预取。[已有反汇编](../2026-11-24-fused/verified/v4/prefetch-c6-int32-no-tail.sass.txt)例如在`0x3ac0/0x3b10/0x3b60`读取下一vector的B，在其间继续当前HMMA；A0的下一vector加载`0x3ba0`距离首个消费者`0x3bc0`较近。这是特定产物的指令位置观察，不是stall实测，也不保证任意编译版本相同。

因此本次先缩小raw-loader开销，暂不直接堆更多fragment暂存。C8保留C6的128×128/BK32、256线程、warp32×64、shared布局、HMMA顺序、输出缩放与每stage两次barrier。新loader每线程每operand读取半行MX32，即8B packed `uint2`及一次scale；A+B理论raw暂存由16个降为6个32-bit寄存器，源级load由16个降为4个。最终指令宽度、总寄存器与收益以实际编译/设备证据为准。

新kernel只用于**int32安全索引、M/N均为128倍数、A和B均8B对齐**；其余路径显式使用原C6。Kp为32倍数，因此row_bytes为16倍数，`row*row_bytes + block*16 + half*8`保留8B对齐。K=0不读取raw/row_bases；末stage不读下一块；certificate拒绝仍在barrier与operand读取前统一返回。

线程到行的映射交换`linear_row=tid>>1`的bit2/bit3。固定半行内的copy，shared word bank起点为：

```text
(18*local_row + 8*half + 2*copy) % 32
```

按8B shared访问的16-thread phase计算，该置换使16个起点恰为所有偶数bank，高32-bit字覆盖所有奇数bank。独立主机算术审阅得到每operand的1024个half4位置恰写一次，64个store phase各覆盖32个bank；这是地址模型，不能冒充硬件bank计数。若未经行置换直接分配半行，即使最终shared内容不变，也会引入同bank不同word。

数值方面，整个half含非零q即可证明其MX32块参与row_min，已有证书保证delta在0..12；同half中的零half4可以使用这个有效delta，最终被原`normalized_four`掩码归为`+0`。全零half不形成delta，仍覆盖低scale零块和signed zero。证书门槛没有放宽。

后续128-bit shared加载仍值得研究，但当前72B row stride并非全行16B对齐；单纯改成`uint4`或把stride加到80B都不足以证明bank友好。应按16B访问的8-thread phase检查新布局 [H8]。同时，`asm volatile`只约束生成PTX时的相关优化，不承诺ptxas最终SASS顺序；隐藏shared读写、shared地址空间转换及真实依赖也必须正确 [H9]。

## 5. 本轮实际 GPU 验证

### V6：同源码编译与正确性

CUDA12.9在真实Tesla V100-PCIE-32GB节点对同一冻结源码分别构建C6/C8；共享库、三个correctness harness和packed-only benchmark均成功。两次构建各约100秒，随后在同一设备顺序执行检查。C8半行kernel为**119 registers/thread、0 stack、0 spill stores/loads、18,688 B shared**；同编译中的C6 int32/no-tail为128 registers。

C6和C8各自通过原69次调用、405,930个输出的完整检查，均为23次证书接受、46次精确回退、0 mismatch，包含2,950个canonical `+0`检查。C8随后独立通过新增9例、180,224个输出、0 mismatch；其中4例为非空新loader适用输入、3例测试C6兼容路径、1例K=0、1例scale255拒绝，另有37,961个canonical `+0`位模式检查。**C8合计78次调用、586,154个输出**；专用fixture的路径标记来自实参对齐和经过host验证的dispatch条件，没有冒充运行时kernel探针。

完整记录见 [C6 correctness](verified/v6/correctness-c6.jsonl)、[C8 correctness](verified/v6/correctness-c8.jsonl)和[独立汇总](verified/v6/comparison-summary.json)。主机端另有43项NumPy/PyTorch contract测试通过，见 [原始CPU日志](host/cpu-contracts.log)；CPU通过不代替待执行的GPU框架检查。

### 充分预热的配对复测：本轮主要性能结果

首轮发现具体的频率爬升风险后，使用**同一组已验证C6/C8二进制**，把每backend的普通工作负载预热从3次增加到100次，其余形状1536³、seed20261123、输入fingerprint `8729bc8ef4895a63`、5 rounds×10 iterations和C6→C8→C8→C6顺序保持不变。没有重编译或改动时钟/功耗。运行时间为北京时间03:18:03–03:18:06；全部4个进程退出0，完整初始和最后实际timed输出均精确通过，CPU oracle仍各抽样256个输出。

| 100次预热的独立窗口 | C6 | C8 |
|---|---:|---:|
| 第一次运行wall median，ms | 0.2236686 | 0.1816069 |
| 第二次运行wall median，ms | 0.2192175 | 0.1810037 |
| 全部10个round的pooled wall median，ms | **0.22034445** | **0.18136930** |
| 对应useful TFLOPS | **32.892852** | **39.961324** |
| 全部10个round的pooled CUDA event median，ms | 0.219238400459 | 0.180326402188 |

同步wall吞吐增加**21.4894%**、延迟降低**17.6883%**；CUDA event对应吞吐增加21.5786%。前后相邻两对运行分别支持约23.16%和21.11%的描述性收益。使用全部round，没有挑选最快一轮；round相关，因此不计算置信区间或显著性。长预热缩小了C6两次运行差异，但NVML整个窗口仍记录1230–1380 MHz，16点中的中位数1380 MHz、GPU0利用率0–100%，GPU1为0%；没有逐round时钟归因，**不能把该百分比解释为严格受控的单机制因果收益**。

原始文件与[独立复测审计](verified/v6/warm100/comparison-summary.json)保存在[独立目录](verified/v6/warm100/)。二进制SHA与原compile manifest逐项一致；69例及C8专用9例的独立suite来自前面的原始验证，本窗口没有重复执行该suite。benchmark仍重新验证实际输出，四次certified调用均接受151次。收集包为6768 B、SHA256 `c8fe16d98a478a48514d74a8d7c58eb0d2eddb6e60bd08c2bce3a1b779487500`，在本地解包前核验。

### 首轮配对：完整保留频率爬升结果

V6采用1536³、seed20261123、`random_block_scales`，输入fingerprint为`8729bc8ef4895a63`。按C6→C8→C8→C6顺序，每次3 warmup、5 rounds、每round10 iterations；每个程序内部轮换baseline/tiled/certified的计时顺序。全部初始和最后实际timed输出各2,359,296个值与baseline精确一致，baseline全量finite，另有256个独立CPU oracle采样。

| 首轮运行 | C6完整pipeline wall ms | C8完整pipeline wall ms |
|---|---:|---:|
| 各自第一次 | 0.2457706 | 0.1850910 |
| 各自第二次 | 0.2194114 | 0.1817888 |
| 每版本全部10个round的pooled median | **0.23382105** | **0.18252580** |
| 对应useful TFLOPS | **30.997027** | **39.708125** |

这组描述性统计为吞吐+28.1030%、延迟−21.9378%。**首个C6的计时前后SM频率为1230/1357 MHz，第一个C8为1357/1380 MHz，最后两个程序均为1380/1380 MHz。** NVML还记录GPU0利用率2–99%、GPU1为0%。所以该百分比同时受到频率／负载时段变化影响，不能将全部收益归因于loader。每个程序内的round并非独立试验，不报告置信区间或显著性。较充分预热的独立复测用于核对这个具体风险；不把两个窗口的round混合。

完整pipeline计时包含每次prepare、整数证书、融合GEMM及条件fallback调度，输出为FP32。同步wall包含完成同步；量化、H2D、分配、poison及其同步、D2H验证和诊断读取在计时外。显式分配11,996,232 B、workspace52,296 B，实际warmup/timing后free均50,528,256 B，满足原48 MiB reserve；没有降低内存门槛、停止服务或修改时钟/功耗。

当前结论是C8值得进入更大形状的同条件比较；**默认仍为C2**。这里没有C2控制组，不能与不同窗口的2048³ C2 50.95 TFLOPS直接计算加速比，也不能写成LLM训练吞吐。

### 编译来源与实际机器码

[冻结源码](verified/v6/source.tar.gz)为107,138 B，SHA256 `077953136dd2cf00faabde0ae4bd9910761cedb059f85c44473fe5ede838c251`；它从parent commit `8834f6a15b1e866b4b6fe67dcdb8a0d79c4b7e50`加本轮明确的工作树改动产生，实际各文件hash保存于[source-state](verified/v6/source-state.json)。收集包为193,538 B、SHA256 `c44e192358791afe44edb6a8dc2ff3ca9d48dcb9092c535e49d965ac3f0be628`，在本地提取前核验。

独立汇总逐项核对两份compile manifest、12个csrc源文件、build入口和fragment mapping，与归档及本地核心源码相符；10个编译日志的哈希也匹配。之后仅更新文档和排队payload时，不将它们冒充原编译快照。[C6 SASS](verified/v6/c6-int32-no-tail.sass.txt)和[C8 SASS](verified/v6/c8-halfrow.sass.txt)均提取自本轮实际`build-c8/bench_mxfp4_fused`，二进制哈希、cuobjdump版本与section哈希见[来源记录](verified/v6/sass-provenance.json)。这是编译/静态调度证据，不是profiler周期或stall计数。

[逐指令审阅](verified/v6/c8-sass-analysis.json)进一步核验了同一产物内两条路径：

| 实际SASS或编译资源 | C6 | C8 |
|---|---:|---:|
| registers/thread | 128 | **119** |
| stack／spill stores／spill loads，B | 0／0／0 | **0／0／0** |
| shared bytes/CTA | 18,688 | 18,688 |
| 非末stage每thread raw global load指令 | 16 | **4** |
| decode/publish区静态指令 | 383 | **293** |
| 循环内row-base `LDS.U8` | 8 | **0** |
| HMMA／fragment `LDS.64`／barrier每stage | 256／48／2 | 256／48／2 |
| fragment `LDS.64`到首个HMMA消费者的静态间隔min/median/max | 0／12／15 | 16／48／75 |

decode/publish区包含active/zero备选路径，383→293不能当成动态发射数量或周期。119regs仍允许至多两个256-thread CTA/SM的寄存器资源上限，未增加理论驻留CTA。实际store为`STS.64`，没有声称自动形成`STS.128`。

C8两条scale预取在第一条HMMA前；B packed的`LDG.64`位于`0x2560`，后面还有186条HMMA；A packed的`LDG.64`位于`0x3190`，后面只有18条HMMA。六个raw目的寄存器及宽加载隐式第二寄存器在backedge前没有使用或覆盖，下一轮消费；末块谓词防止越界。因此稳态预取确实与当前剩余计算交错，但**不是全部load都在第一条HMMA前，也不证明内存延迟已经隐藏**。重用barrier在`0x3200`，早于最后13条HMMA，晚于最后shared read `0x2ec0`；余下计算使用寄存器，shared重用保护保持。fragment加载提前说明编译器有更多调度空间，无法从静态记录分离其对总收益的贡献。

## 6. 排队作业与后续执行

原作业`sg-jincy26-261123-e5bb3`于UTC2026-11-23 15:57:51创建，申请1张V100、4 CPU、16GiB；保持原作业和创建时间。原watcher在北京时间02:46:47因Crater401退出，未尝试远端payload，见[失败状态](queue/previous-failed-state.json)。同一既有LDAP账号的读取随后恢复；本轮没有登录或切换账号。

北京时间**03:08:30**，在确认旧进程已结束、同一作业仍Pending/Inqueue、原archive/payload哈希正确后，启动新的有界等待进程，见[恢复审计](queue/recovery-audit.json)。[独立进程观察](queue/recovered-live-process.json)和[恢复后状态](queue/recovered-state-observation.json)确认新PID2630233、run id `20261123T190830Z-5f5e8cd5`处于waiting。仍每60秒轮询，只分配原截止时间剩余的等待时长，**截止不晚于北京时间12:52:24**；没有重建作业或改变排队优先级。

恢复时远端ready仍绑定旧源码commit `4d7ff3b95768ff0e4833a336931e3e80265c897d`。本地后续payload已加入C8，保留原native、Torch检查、量化benchmark、五组smoke和C2控制，再执行C3–C8完整检查及2048³/4096³同seed测量；正常路径36个stage共享原2700秒deadline。[10场景host mock](host/payload-host-simulation.json)验证正常、缺工具链/框架、失败和超时的编排，不是Crater/GPU执行。是否已发布新源码，以随后真实publication审计为准。

## 一手资料链接

- **H1** NVIDIA V100/V100S Datasheet，JAN20，第一页型号规格：<https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf>
- **H2** Tesla V100 PCIe Product Brief，Table1/2时钟与显存：<https://images.nvidia.com/content/tesla/pdf/Tesla-V100-PCIe-Product-Brief.pdf>
- **H3** Volta Architecture Whitepaper，80 SM、每周期FMA、6144 KiB L2：<https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf>
- **H4** CUDA12.9.1 Volta Tuning Guide，Occupancy及Unified Shared Memory/L1：<https://docs.nvidia.com/cuda/archive/12.9.1/volta-tuning-guide/index.html>
- **H5** PTX ISA8.8，`mma`／`ldmatrix`／`cp.async` Target ISA Notes：<https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html>
- **H6** CUDA12.9.1 Best Practices，register allocation及shared banks：<https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-best-practices-guide/index.html>
- **H7** NVIDIA CUTLASS Efficient GEMM，Pipelining与Parallelized Reductions：<https://docs.nvidia.com/cutlass/4.3.3/media/docs/cpp/efficient_gemm.html>
- **H8** Programming Tensor Cores: Native Volta Tensor Cores with CUTLASS，GTC2019，PDF页12–18、26–33、37–39：<https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9593-cutensor-high-performance-tensor-operations-in-cuda-v2.pdf>
- **H9** CUDA12.9.1 Inline PTX Assembly，§1.2.2／1.2.3：<https://docs.nvidia.com/cuda/archive/12.9.1/inline-ptx-assembly/index.html>
- **P1** Jia等，Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking，2018：<https://arxiv.org/pdf/1804.06826>
- **P2** Wang等，Ladder，OSDI2024：<https://www.usenix.org/system/files/osdi24-wang-lei.pdf>
- **P3** Microsoft BitBLAS，Supported matrix multiplication表：<https://github.com/microsoft/BitBLAS>；论文artifact：<https://github.com/microsoft/BitBLAS/tree/osdi24_ladder_artifact>
- **P4** Marlin作者仓库，Techniques及Requirements：<https://github.com/IST-DASLab/marlin>
- **P5** Guo等，Fast Matrix Multiplications for Lookup Table-Quantized LLMs，Findings of EMNLP2024：<https://aclanthology.org/2024.findings-emnlp.724/>
- **P6** FLUTE作者仓库，Kernel/GPU支持表：<https://github.com/HanGuo97/flute>
- **P7** Tseng、Yu、Park，Training LLMs with MXFP4，AISTATS2025，PMLR258：<https://proceedings.mlr.press/v258/tseng25a.html>；作者代码：<https://github.com/amazon-science/mxfp4-llm>
