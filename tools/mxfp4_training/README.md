# V100 MXFP4 训练基础算子

本目录提供 **完整 E2M1 数值域、MX32 块尺度的 SM70 GEMM、共享内存 tiled 实现、带精确证书与回退的 native cuBLAS／融合 HMMA 路径，以及可分别启用 Forward／Dgrad／Wgrad 的 PyTorch 线性层**。目标是为后续 LLM 预训练实验建立可核对的算术、反向传播和测量入口。

V100 执行的是 FP16 输入、FP32 累加的 Tensor Core 运算；MXFP4 由软件编码、解码和施加块尺度。当前版本没有原生 FP4 指令，也没有接入受限域的 middle-only packing 优化。native packed-input 性能、PyTorch 框架正确性和合成 GPU 训练 smoke 已分别执行；它们的输入、生命周期和计时范围不同，不能互相代替完整 LLM 预训练吞吐。

## 1. 已验证到什么程度

最新一组 native 对照在 **2026-11-23 UTC、Tesla V100-SXM2-32GB、CUDA 12.8** 上执行，完整结果见 [C2／C8／C9／C10 独立审计](../../evidence/results/mxfp4-training/2026-11-23-crater-c8/candidate-c9-c10/audit.md)。同节点的早期框架补跑见 [119 项检查、Torch benchmark 与 GPU smoke 审计](../../evidence/results/mxfp4-training/2026-11-23-crater-c8/framework-followup-procfix/independent-audit.md)。

基础正确性与 CPU smoke 保存在 [2026-11-23 实验目录](../../evidence/results/mxfp4-training/2026-11-23/)。历史 PCIe 卡测量保留在归档名为 2026-11-24 的 [50.95 TFLOPS 性能证据](../../evidence/results/mxfp4-training/2026-11-24-performance/README.md)、[融合实验](../../evidence/results/mxfp4-training/2026-11-24-fused/README.md)和 [C8 硬件研究报告](../../evidence/results/mxfp4-training/2026-11-24-hardware-study/README.md)；不跨卡型、形状或负载窗口计算加速比。

| 项目 | 已有结果 | 能支持的结论 |
|---|---|---|
| NumPy／PyTorch CPU 单元测试（早期归档） | **43 项通过** | 编码、尺度、尾部、量化、独立矩阵参考及一阶反向接口通过对应检查 |
| CUDA 编译 | **历史 CUDA 12.9、最新 CUDA 12.8 构建均有通过记录** | 最新 C2／C8／C9／C10 的源、命令、编译日志和产物 SHA 均绑定至冻结快照 |
| 真实 V100 tiled，选用 variant 1 | **46 例、175,429 个输出、0 mismatch** | 完整数值域、极端尺度、尾部和非默认 stream 通过独立 ordered-MX32 oracle 检查 |
| 历史 certified，variant 0／1／2 | **每个 variant 均为 67 次调用、372,132 个输出、0 mismatch** | 保留早期 21 次接受、46 次回退与 400 个 canonical `+0` 检查，不与扩展后的 suite 混计 |
| 最新 certified C2 | **69 次调用、405,930 个输出、0 mismatch** | 23 次接受、46 次精确回退、2,950 个 canonical `+0`；仍为默认 variant |
| 融合 certified C3–C10 | **原 69-call suite 均有通过记录；最新 C8 合计 78 次，C9／C10 各 79 次** | C8 为 586,154 个输出；C9／C10 各为 602,538 个输出。包含另计的 loader 与 cancellation fixtures，整数证书未放宽 |
| C10 half2 解码枚举 | **851,968 组、3,407,872 个 half 值、0 mismatch** | 非零位模式严格相同，内部零比较只归一化符号；最终 FP32 canonical `+0` 由 GEMM suite 另行验证 |
| 最新 native packed-input 性能 | **C2：4096³ block scales 为 75.355 TFLOPS；8192³ unit scale 为 89.290 TFLOPS** | 同窗口完整 pipeline 的同步 wall median；8192 block scales 另行检查后按证书回退，未达到 125 TFLOPS 目标 |
| 最新融合解码优化 C10 | **相对 C8，2048³／4096³／8192³ 吞吐分别提升 11.42%／9.23%／8.21%** | C9 的 cursor 方案反而慢约 3.67%–4.53%；同窗口 C2 仍快于全部融合候选 |
| 历史 PCIe 预取优化 | **1536³ C4→C6 约 +30.85%，C6→C8 长预热复测约 +21.49%** | 属于不同历史窗口；当时没有 C2 的同形状配对结果 |
| CPU tiny causal decoder（历史 smoke） | **5 组配方各训练 40 步，全部完成** | CPU 端前向、反向、FP32 AdamW 更新已接通 |
| PyTorch baseline／tiled | **119 项实卡检查通过，两个 backend 各完成五组配方 × 40 步 GPU smoke** | 覆盖量化、F/D/W、非连续输入、非默认 stream，以及有限 FP32 参数／AdamW 更新 |
| PyTorch certified | **显式 `cuda_certified` 已接入；C2／C8／C10 各 16 项实卡检查通过** | 每次调用创建并同步销毁 context，检查线程隔离、错误清理和 Graph capture 拒绝 |
| 显式 Torch workspace | **11 项实卡检查、16 次全输出比较、1,455 个值通过；4096³ 完整量化＋GEMM 重复调用为 56.473 TFLOPS** | 固定 shape／device／owner thread，1024³／4096³ 相对同窗口 per-call 分别为 7.872824×／2.474954×；创建与关闭另计，尚未接入 autograd／`MXLinear` |
| PyTorch 量化＋GEMM | **1024³／4096³ 的 per-call 与显式 workspace 均已有真实完整 wrapper 测量** | 包含两输入量化、验证、分配和 dispatch；计时范围见 §1.4–1.5，不能用 native packed-input TFLOPS 替代 |
| 显式 CUDA RNE 量化 | **135 项实卡检查通过；两输入完整量化 wrapper 观测到 6.203×／17.683×速度比** | 分别对应 1024²／4096²；默认仍为 Torch，`cuda_rne` 只替换 nearest 路径，SR 不变；不是 GEMM 或训练加速比 |
| C2 prepare raw-cache | **327 项微检查及 OFF／ON 完整检查通过；三组吞吐差异均小于 1%** | 仅是该窗口的描述性观察，不据此改变默认 OFF；workspace 实验也使用 OFF 库 |
| Crater 节点工作流 | **恢复后的 V100-SXM2 节点已执行 native、框架检查与 GPU smoke** | 原失败／blocked 报告保留，后续结果使用独立归档；节点或 SSH 成功本身不计为算子通过 |
| 真实语料 LLM 预训练 | **未运行** | tiny decoder smoke 不建立 LLM 收敛、训练吞吐、峰值显存收益或加速比结论 |

历史 tiled 记录见 [46-case 输出](../../evidence/results/mxfp4-training/2026-11-24-performance/iterations/build-final-v1/correctness-v2.jsonl)；certified 的早期检查分别见 [variant 0](../../evidence/results/mxfp4-training/2026-11-24-performance/verified/build-perf-c0/correctness.jsonl)、[variant 1](../../evidence/results/mxfp4-training/2026-11-24-performance/verified/build-perf-c1/correctness.jsonl)、[variant 2](../../evidence/results/mxfp4-training/2026-11-24-performance/verified/build-perf-c2/correctness.jsonl)。最早基线的 42 例、137,792 个输出也保留在原始实验目录中。

### 1.1 最新 SXM2 对照与历史 PCIe 性能

2026-11-23 UTC 的 SXM2 实验对每个形状执行 `C2→C8→C9→C10→C10→C9→C8→C2`，每个 job 预热 100 次，再测 5 轮、每轮 10 次。每个 variant 合并两次进程的全部 10 个 round，以下均为 **完整 packed-to-FP32 pipeline 的同步 wall median**：

| 形状／输入分布 | C2 TFLOPS | C8 TFLOPS | C9 TFLOPS | C10 TFLOPS | C10 相对 C8 吞吐 |
|---|---:|---:|---:|---:|---:|
| 2048³／random block scales | **54.863** | 44.022 | 42.407 | 49.047 | +11.42% |
| 4096³／random block scales | **75.355** | 54.402 | 51.940 | 59.421 | +9.23% |
| 8192³／unit scale | **89.290** | 59.194 | 56.718 | 64.054 | +8.21% |

数据及全部原始 round 见 [独立审计](../../evidence/results/mxfp4-training/2026-11-23-crater-c8/candidate-c9-c10/audit.json)。24 次 benchmark 的初始与最后一次实际 timed certified 输出均通过全量比较；同形状同分布的输入指纹一致，每次计时调用都重算证书，诊断计数为 `1→151`。进程内 round 有相关性，边界时钟快照为 1335–1530 MHz，不能据此声称十次独立实验或锁定频率。

8192³ `random_block_scales` 单独执行 correctness-only 探针，因能量条件得到 `reject_flags=4`，其 67,108,864 个 fallback 输出完整比较通过。该输入没有融合快路径计时，不能把上表 unit-scale 的吞吐外推给它。**默认 C2 保持不变；C10 是本组融合候选中最快者。**

另有 [C2 cuBLAS 算法筛选审计](../../evidence/results/mxfp4-training/2026-11-23-crater-c8/c2-algorithm-screen/audit.md)：4096³ 的 `DEFAULT_TENSOR_OP` 观测到 92.560 TFLOPS，8192³ unit-scale 的 `DEFAULT` 观测到 99.280 TFLOPS。这两项仅计 **normalized GEMM compute**，不含 prepare、证书、epilogue 或 fallback，且有频率变化；未据此切换 production 算法，也不把它们列为完整 MXFP4 吞吐。

以下保留归档名为 **2026-11-24 的历史 PCIe** 对照：

下表直接取自 [selected variant 2 的 2048³ 原始 JSONL](../../evidence/results/mxfp4-training/2026-11-24-performance/verified/build-perf-c2/varied-2048.jsonl)，全部使用**同步 wall median**，没有混用不同轮次的 CUDA event 与 wall 数据。

| 路径 | 计时输入与工作范围 | wall median（ms） | useful TFLOPS |
|---|---|---:|---:|
| 原始 MX32 baseline | packed codes/scales → 解码、逐块缩放、FP32 输出 | 5.897512 | 2.913 |
| tiled variant 1 | packed codes/scales → 共享内存 tiled GEMM、FP32 输出 | 0.722019 | 23.794 |
| certified variant 2 total | packed codes/scales → 证书、normalized FP16、cuBLAS、条件 epilogue／精确回退 | 0.337180 | 50.952 |
| FP16 cuBLAS compute | 已物化的 FP16 输入 → FP32 输出 | 0.239385 | 71.767 |
| dequant＋cuBLAS total | packed codes/scales → 两个 FP16 输入物化 → FP32 输出 | 0.338638 | 50.732 |

该例在 **Tesla V100-PCIE-32GB** 上运行，已有 vLLM 服务同驻，未停服务或修改时钟／功耗限制。输入是合成的完整 E2M1 nibble 数据，每个 MX32 block 的尺度指数随机取 `[-2,2]`；不是实际训练张量。这次证书通过，tiled 和 certified 对全部 4,194,304 个输出与原始基线精确比较通过，另有独立 CPU oracle 抽样检查。

每个 job warmup 3 次，测量 5 轮、每轮 10 次，轮换 backend 顺序。TFLOPS 分子为 `2*M*N*logical_K`，不把 padding 算作有用工作。**不包含高精度输入量化、H2D、初次分配或 workspace 创建**；certified 每次重算证书和 normalized operands，完整 GPU pipeline 在计时内。workspace 在调用间复用，诊断读取在计时外。FP16 cuBLAS compute 的起点是已解码输入，不能作为 packed-input 全路径成本。

certified total 与该次简单 dequant＋cuBLAS total 处于同一量级；这不能证明优于所有优化后的解码实现，也不能外推为训练加速。该历史窗口采用单个共享 GPU 输出缓冲区，显式使用 `--reserve-mib 64 --max-mib 64` 适应有限空闲显存；默认 reserve 仍为 512 MiB。当时因显存不足未执行框架阶段；后续 SXM2 节点已独立完成相应验证，原历史报告不作改写。

### 1.2 融合 HMMA 与预取实验

certified variants 3–10 共用原精度证书，把 packed 解码和归一化直接融合到 HMMA 共享内存加载中，移除全局 normalized FP16 输入和 cuBLAS workspace。C8 在 int32 索引、M/N 为 128 倍数且 A/B 均 8 字节对齐时，用半行向量读取 packed 数据；其余输入使用 C6 的融合 kernel。C9 保留这一合同并尝试 cursor 地址递增；C10 使用精确的 half2 二幂乘法替换部分整数解码。2048³ 的 native context 显式设备工作区从 C2 的 **21,037,128 B** 降至 **69,704 B**；这不包含输入、输出与驱动分配，也不是完整训练峰值显存。

最新 C8／C9／C10 的 SASS 完整函数分别为 1664／1664／1568 个静态指令槽，寄存器为 119／121／119，均无 spill。C10 的循环少 96 个静态指令槽，保留 HMMA、fragment LDS、shared store 和 barrier 数量。C9 的 packed load 确实提前，但实测变慢；静态指令减少或 load 提前均不能直接换算成动态速度，具体证据见 [审计中的 SASS 对照](../../evidence/results/mxfp4-training/2026-11-23-crater-c8/candidate-c9-c10/audit.md#sass-静态证据)。

历史 PCIe V6 实验中的候选为 **C8：128×128 CTA、BK32、半行宽加载及下一块 raw 预取**。首轮1536³ ABBA、每backend预热3次的描述性收益为28.10%，但首个C6存在1230→1357 MHz频率爬升。随后使用同一组已验证二进制，将每backend预热增至100次，其余输入和测量参数不变，独立重复ABBA。每版本保留全部10个round取中位数后，C6为0.22034445 ms／32.8929 TFLOPS，C8为0.1813693 ms／39.9613 TFLOPS，**吞吐增加21.49%，延迟下降17.69%**；CUDA event的对应收益为21.58%。全部初始及最后实际计时输出精确一致。长预热没有锁定频率，也未消除共享设备负载变化；两窗口分开汇总，不报告显著性或单一优化机制的因果收益。

该历史窗口同一二进制的SASS显示，C8每stage的raw全局加载由16条降为4条，decode/publish静态指令由383条降为293条，registers/thread从128降至119且无spill；256条HMMA、48条fragment LDS和两次barrier保持。指令间隔改善属于静态调度证据，不是profiler测得的stall。较早C4/C6和退步的C7结果仍完整保留，不与V6跨负载窗口计算加速比。

历史原始 JSONL、编译／SASS 证据、同驻负载采样及较慢实验都保留在 [融合实验报告](../../evidence/results/mxfp4-training/2026-11-24-fused/README.md)及 [C8研究报告](../../evidence/results/mxfp4-training/2026-11-24-hardware-study/README.md)。当时两张 GPU 均有既有服务；1536³ C8 与 2048³ C2 的历史数字不能直接比较。最新 SXM2 同条件对照已补上，结果见 §1.1。NVIDIA 标称 FP16 Tensor Core 指标中 SXM2 为 125 TFLOPS、PCIe 为 112 TFLOPS；新节点是 SXM2，历史卡是 PCIe。原生计算预算与未来算法打包的 useful 吞吐仍需分别讨论。

### 1.3 历史 CPU 训练连接 smoke

CPU smoke 使用固定种子的单层 causal decoder、32 维隐藏状态、4 个 attention heads、长度 16 的合成序列与词表 32。任务是可预测的循环递增 token 序列。五组使用相同初始参数，记录如下：

| 配方 | MXFP4 Forward | MXFP4 Dgrad | MXFP4 Wgrad | 梯度量化方式 | 验证 loss：开始 → 40 步后 |
|---|---:|---:|---:|---|---:|
| `fp32` | 否 | 否 | 否 | 不量化 | 3.645811 → 0.062209 |
| `mx_forward` | 是 | 否 | 否 | 不量化 | 3.608196 → 0.071482 |
| `mx_forward_dgrad` | 是 | 是 | 否 | RNE | 3.608196 → 0.073649 |
| `mx_forward_dgrad_wgrad` | 是 | 是 | 是 | RNE | 3.608196 → 0.076575 |
| `mx_all_stochastic_grad` | 是 | 是 | 是 | 对上游梯度使用 SR | 3.608196 → 0.076672 |

这些数值是**训练连接 smoke 的结果**。该任务很小，合成规则简单，验证序列与训练序列遵循同一规则；loss 下降不证明真实语料泛化、LLM 预训练收敛或配方优劣。脚本内的耗时也只是含大量检查的 smoke 墙钟时间。

### 1.4 最新 CUDA RNE 与完整 Torch 调用成本

2026-11-23 UTC 的同一 V100-SXM2 节点已完成 [RNE v2 独立审计](../../evidence/results/mxfp4-training/2026-11-23-crater-c8/candidate-cuda-rne-v2/audit.md)：量化 135 项、原 Torch 119 项、C2 certified 16 项，以及 `cuda_tiled`／`cuda_certified` 搭配 CUDA RNE 的五组配方 × 40 步 smoke 全部通过。135 项包含 147 个 codes／scales 逐字节比较、126 个 exact 数值比较和 180 个合法 FP32 allclose 对照；不能把后者也称为逐位相同。

两个已驻留 GPU 的 FP32 输入，完整量化 wrapper 的同步 wall median 为：

| 两个输入的形状 | 默认 Torch 量化（ms） | CUDA RNE（ms） | wall 速度比 |
|---|---:|---:|---:|
| 各 1024×1024 | 1.843861 | 0.297238 | **6.203×** |
| 各 4096×4096 | 11.972170 | 0.677044 | **17.683×** |

该项包含分配、验证、dispatch 和有限值标志同步；每 backend 预热 10 次、5 轮 × 10 次，轮内交替先后顺序，每轮最后实际 timed packed 输出全量对独立 NumPy oracle。每个尺寸只有一个进程，round 有相关性；24 个边界 NVML 样本的 SM 时钟为 1290–1530 MHz，未锁频，也不是连续负载追踪。量化单项不报告 TFLOPS。

另以 `Torch→CUDA RNE→CUDA RNE→Torch` 四个进程比较完整 Torch 量化＋C2 certified，每个 job 预热 3 次、5 轮 × 10 次；每种量化器合并全部 10 个 round 得到：

| GEMM 形状 | 量化器 | 完整 wall median（ms） | useful TFLOPS |
|---|---|---:|---:|
| 1024³ | Torch | 4.110380 | 0.522454 |
| 1024³ | CUDA RNE | 2.779606 | **0.772586** |
| 4096³ | Torch | 16.043665 | 8.566556 |
| 4096³ | CUDA RNE | 5.451042 | **25.213337** |

这里每次调用的 context 创建、workspace 分配、prepare、证书、GEMM、epilogue 和同步销毁都在计时内。即使量化已经变快，per-call 生命周期成本仍明显；1024³ 的 CUDA RNE＋tiled 在同组记录中为 0.392438 ms，比 per-call certified 更快。它们均不等于 native 复用 context 的吞吐。各进程使用相同 seed、形状、生成器和运行环境，但此完整 wrapper harness 没有保存跨进程输入字节指纹；已驻留输入在每个进程内复用，初始与最后实际 timed 输出全量检查通过。七条 job 的全部 wall／event 原始 round 均保留在 [audit.json](../../evidence/results/mxfp4-training/2026-11-23-crater-c8/candidate-cuda-rne-v2/audit.json)。

初始 RNE 实验在 lazy negative view 案例失败，原因是未改动的旧 Torch 对照无法通过 `x.new_tensor` 处理该 dispatch key；该失败及后续 blocked 阶段完整保留。`f338faaa` 仅修正 checker 中旧对照输入的 lazy view 解析，native wrapper／direct 仍接收原 view，kernel、Torch 接口和比较标准均未改。v2 的成功不改写初始结果。

### 1.5 显式 workspace：完整量化＋GEMM 的重复调用

同一 V100-SXM2 节点的后续实验使用默认 **C2、prepare-cache OFF** 库，对每个尺寸按 `per_call→workspace→workspace→per_call` 四个独立进程运行。两条路径都在每次调用内重新量化两个已驻留 GPU 的 FP32 输入，并执行验证、packed／输出分配、Python／ctypes dispatch、prepare、证书、GEMM 和最终缩放／回退。只在 workspace 路径把一次性的 context 创建与关闭单独计时；原 per-call 路径仍逐次创建、分配并同步销毁。结果见 [workspace 独立审计](../../evidence/results/mxfp4-training/2026-11-23-crater-c8/candidate-workspace/audit.md)。

| GEMM 形状 | per-call wall median（ms） | workspace wall median（ms） | 重复调用速度比 | workspace useful TFLOPS |
|---|---:|---:|---:|---:|
| 1024³ | 3.712270199 | 0.471529702 | **7.872824×** | **4.554291** |
| 4096³ | 6.023360652 | 2.433725900 | **2.474954×** | **56.472651** |

每个进程先验证 1 次、预热 10 次，再测量 5 轮 × 10 次；每种生命周期合并两进程的全部 10 个 round 取 median。同尺寸四进程的实际 FP32 输入 SHA 一致，输入在计时前后未变；初始和最后一次实际 timed 输出均全量逐位比较通过，零值为 canonical `+0`。workspace 的末次证书记录 61 次提交；它包含初始验证、预热和 50 次计时调用，不能当作 61 个独立测试或训练 step。原始 round 和证据绑定见 [audit.json](../../evidence/results/mxfp4-training/2026-11-23-crater-c8/candidate-workspace/audit.json)。

**一次性创建（含初始 ABI 查询）与最终关闭合计为 26.488478–32.648653 ms，未计入上表重复调用吞吐。** 关闭前的最终输出检查已同步 GPU，这个范围也不是任意环境的固定冷启动成本。只有少量调用时必须另计资源成本；不能把 56.472651 TFLOPS 用作单次冷调用、autograd 或完整训练 step 的速度。该实验没有连续 NVML 轨迹；每种生命周期只有两个独立进程，round 有相关性，per-call 的进程间耗时也有变化，因此倍率限于本次配对窗口。

独立 workspace checker 的 **11 个案例、16 次全输出比较、1,455 个值及 96 个 canonical `+0`** 已通过，覆盖尾部、变更输入后刷新证书、NaN／跨度拒绝回退、线程归属、busy stream 拒绝与完成后切换、queued tensor 生命周期、提交后错误清理、Graph capture 拒绝及恢复。当前 API 是显式 packed forward 接口，**未接入 autograd 或 `MXLinear`**；原 `gemm_packed(..., backend="cuda_certified")` 的 per-call 行为保持不变。

### 1.6 C2 prepare raw-cache：正确性通过，默认继续关闭

可选 `--certified-prepare-raw-cache` 为 C2 的每行准备阶段缓存 raw MX32 数据及能量／尺度，仅在 `0<Kp<=8192` 时启用，其余 K 走原准备路径。[独立审计](../../evidence/results/mxfp4-training/2026-11-23-crater-c8/candidate-c2-prepare-cache/independent-audit.md)核验了 26 个实际阶段全部通过：327 项微检查中的 normalized half 位模式、证书行数据及输出比较无 mismatch；OFF／ON 两库的原 native 检查和各 135 项 RNE 检查，以及 ON 库的 16 项 Torch certified 检查也通过。

在 `OFF→ON→ON→OFF` 配对中，每进程预热 100 次、5 轮 × 10 次，合并各设置的全部 10 个 round；2048³／4096³ block scales 和 8192³ unit scale 的 ON／OFF 吞吐比分别约为 **1.00379×／1.00828×／1.00709×**。全部观测差异小于 1%，仅是本次数据的描述性结果；两个进程的重复数和变化的边界时钟不足以建立稳定收益。**默认仍为 OFF，上节 workspace 结果使用同一个 OFF 库。** 这里测量的是 native packed-input 完整 pipeline，不含 FP32 量化、Python 或训练成本，不能与 workspace 的完整量化＋GEMM 表混算。

### 1.7 最后硬件诊断的边界

同一冻结准备内核的 327-case microchecker 又在 Compute Sanitizer memcheck 下实际执行通过，`ERROR SUMMARY: 0 errors`；这个 sanitizer 结论限于该检查器覆盖的内核。安装的 Nsight Compute 2025.1.1 实际尝试了预热后一次 4096³ C2 pipeline 的采集，但驱动返回 `ERR_NVGPUCTRPERM`。GPU 数值 probe 通过与硬件计数器采集受阻分别记录；本轮没有动态 Tensor Core 利用率或 stall 数据。首次全量 metric catalog 超过脚本上限的失败也保留，修复后的查询确认九个请求指标存在，复用原 memcheck 证据而未重复执行。详情见 [诊断与原始归档](../../evidence/results/mxfp4-training/2026-11-23-crater-c8/README.md#8-最后诊断memcheck-通过硬件计数器权限受限)。

## 2. 文件与依赖

| 文件 | 用途 |
|---|---|
| [reference.py](reference.py) | 独立 NumPy 编码、量化、解码和逐块整数 GEMM oracle |
| [torch_ops.py](torch_ops.py) | PyTorch 量化、packed GEMM、C ABI 加载、一阶 surrogate backward、`MXLinear` |
| [certified_workspace.py](certified_workspace.py) | 显式复用固定 shape／device 的 certified context；owner thread 管理，不改变原 per-call 接口 |
| [check_certified_workspace.py](check_certified_workspace.py)／[bench_certified_workspace.py](bench_certified_workspace.py) | workspace 生命周期及精确输出检查；完整两输入量化＋GEMM 的 per-call／workspace 对比，单独记录创建与关闭成本 |
| [csrc/mxfp4_gemm.cu](csrc/mxfp4_gemm.cu) | 完整 E2M1 × E2M1 的 SM70 Tensor Core GEMM |
| [csrc/mxfp4_gemm.h](csrc/mxfp4_gemm.h) | C ABI、内存、尺度和累计顺序契约 |
| [csrc/mxfp4_gemm_tiled.cu](csrc/mxfp4_gemm_tiled.cu) | 共享内存复用、PRMT 解码和数值门控的尺度路径；当前选用 variant 1 |
| [csrc/mxfp4_gemm_certified.cu](csrc/mxfp4_gemm_certified.cu)／[C API](csrc/mxfp4_gemm_certified.h) | GPU 精确证书、向量化 prepare／epilogue、normalized FP16 cuBLAS 与 exact tiled fallback；当前选用 variant 2 |
| [csrc/mxfp4_gemm_fused.cu](csrc/mxfp4_gemm_fused.cu) | variants 3–10 的共享内存归一化、SM70 HMMA 和最终缩放；最新同条件测量中 C10 为最快融合候选，C2 仍更快 |
| [csrc/check_mxfp4_half2_decode.cu](csrc/check_mxfp4_half2_decode.cu) | C10 解码与原 integer 解码的完整 packed16 × delta 枚举 |
| [csrc/mxfp4_quantize_rne.cu](csrc/mxfp4_quantize_rne.cu)／[C API](csrc/mxfp4_quantize_rne.h) | 显式启用的 CUDA warp-per-MX32 RNE 量化，保留现有字节合同 |
| [csrc/check_mxfp4.cu](csrc/check_mxfp4.cu) | 不依赖 PyTorch／cuBLAS 的真实 GPU 正确性 harness |
| [csrc/check_mxfp4_certified.cu](csrc/check_mxfp4_certified.cu) | 完整数值域、证书边界、拒绝回退和 context 重用检查 |
| [csrc/bench_mxfp4.cu](csrc/bench_mxfp4.cu) | packed-input baseline／tiled／certified 与 FP16 cuBLAS 分项及总成本比较 |
| [csrc/bench_mxfp4_fused.cu](csrc/bench_mxfp4_fused.cu) | 小显存 packed-only 比较，保留完整 certified pipeline；不含 C2／普通 cuBLAS 对照 |
| [build_cuda.py](build_cuda.py) | 编译共享库和 harness，并记录源码哈希、命令、日志和产物哈希 |
| [check_torch_cuda.py](check_torch_cuda.py) | 119 项框架级 GPU 检查：独立 NumPy 预期、所有 recipe 开关与非默认 stream |
| [check_torch_certified.py](check_torch_certified.py) | 单独的 16 项 certified Torch 检查：数值、F/D/W、context／stream／线程和错误清理 |
| [check_quantize_cuda.py](check_quantize_cuda.py)／[bench_quantize_cuda.py](bench_quantize_cuda.py) | CUDA RNE 的独立逐字节检查与两个输入完整量化 wrapper 的成本比较 |
| [bench_torch_cuda.py](bench_torch_cuda.py) | 通过外部 GPU check 后测量 Torch FP32、FP16、casts、量化与量化＋packed GEMM；可显式加入 certified 和 CUDA RNE |
| [pretrain_smoke.py](pretrain_smoke.py) | 五组配方的合成 causal next-token 训练 smoke |
| [crater_queue_runner.py](crater_queue_runner.py)／[crater_remote_run.py](crater_remote_run.py) | 已授权 Crater job 的排队跟进与有时限的远端验证 payload；保留源码绑定、失败和补跑证据 |
| [参考测试](../../tests/test_mxfp4_reference.py)／[PyTorch 测试](../../tests/test_mxfp4_torch.py) | 独立预期值、边界和反向传播检查 |

NumPy 参考路径只需要 NumPy；框架测试和 smoke 还需要 PyTorch。历史 CPU 验证环境为 Python 3.12.14、NumPy 2.3.5、PyTorch 2.5.1+cpu；最新 SXM2 框架环境为 Python 3.12.15、NumPy 2.3.5、PyTorch 2.10.0+cu128。这些是已测版本，不是唯一支持版本。CUDA 构建需要能生成 SM70 代码的 CUDA 12.x 工具链；历史 PCIe 使用 12.9，最新 SXM2 使用 12.8.93。共享库不依赖 PyTorch C++ 头文件；启用 certified 或 native benchmark 需要 cuBLAS。GPU 框架运行还需要支持本机 V100 的 CUDA 版 PyTorch。

下面所有命令都从仓库根目录执行。`python` 应指向装有相应依赖的环境。为每次运行选用新的构建目录和输出路径，保留旧证据；Python check／benchmark／smoke 和 native benchmark 的 `--output` 会拒绝覆盖已有报告。

## 3. 复跑

### 3.1 CPU 单元测试

```bash
python -m unittest tests.test_mxfp4_reference tests.test_mxfp4_torch -v
```

没有安装 PyTorch 时，对应测试会标记为 skipped；这不等于通过完整的 43 项检查。只重放 NumPy 参考可运行：

```bash
python -m unittest tests.test_mxfp4_reference -v
```

### 3.2 编译 CUDA 共享库和独立 harness

```bash
python tools/mxfp4_training/build_cuda.py \
  --nvcc /usr/local/cuda-12.8/bin/nvcc \
  --output-dir /tmp/mxfp4-cuda-build-20261124 \
  --with-certified --with-benchmark \
  --tiled-variant 1 --certified-variant 2
```

按安装位置调整 `--nvcc`；也可省略它，使用 `CUDA_HOME`／`CUDA_PATH` 或 `PATH` 中的工具链。当前默认就是 tiled variant 1、certified variant 2；命令显式写出版本以便复现。输出包含 `libmxfp4_sm70.so`、`check_mxfp4`、`check_mxfp4_tiled`、`check_mxfp4_certified`、`bench_mxfp4`、编译日志和 `compile-evidence.json`。只需 baseline／tiled 共享库时可省略两个 `--with-*` 并加 `--library-only`。

构建使用 `--ftz=false`，以保留本契约要求的 FP32 次正规数。certified variants 0–2 使用 cuBLAS，3–10 为融合候选，编译记录和 benchmark 环境记录保留所选版本。`--with-fused-benchmark` 只允许启用 certified 且 variant>=3。编译成功本身不证明 GPU 执行正确。Torch 通过 `backend="cuda_certified"` 显式使用共享库中选定的 variant；当前桥接每次调用创建、执行并同步销毁 context，其成本与 native benchmark 复用 context 的计时不同。

C8 的历史小显存实验可用独立目录构建，运行原 69 次调用 correctness suite、另计的 9 个 C8 专用 fixture 与 1536³ packed-only benchmark：

```bash
python tools/mxfp4_training/build_cuda.py \
  --nvcc /usr/local/cuda-12.8/bin/nvcc \
  --output-dir /tmp/mxfp4-cuda-build-c8 \
  --with-certified --with-fused-benchmark \
  --tiled-variant 1 --certified-variant 8

/tmp/mxfp4-cuda-build-c8/check_mxfp4_certified \
  --device 0 --correctness-only --small-memory-correctness --reserve-mib 48 \
  > /tmp/mxfp4-c8-correctness.jsonl \
  2> /tmp/mxfp4-c8-correctness.stderr

/tmp/mxfp4-cuda-build-c8/bench_mxfp4_fused \
  --device 0 --m 1536 --n 1536 --k 1536 \
  --mode random_block_scales --seed 20261123 \
  --rounds 5 --iters 10 --warmup 3 --samples 256 \
  --max-mib 12 --reserve-mib 48 \
  --output /tmp/mxfp4-c8-packed-1536.jsonl
```

先检查 correctness 的成功退出及 `certified_summary`、`c8_fixture_summary` 均通过，再运行 benchmark。C8 专用组含 180,224 个输出，分别标记 4 个非空新 loader 案例、3 个 C6 兼容路径案例、1 个空 K 案例与 1 个证书拒绝案例；覆盖 row bases、delta12、低尺度零块、K=32/96/160、输入和输出偏移。路径标记依据实参对齐及经主机检查的 dispatch 条件，未添加 kernel 运行时探针。单例显式分配最多 170,312 B，原 69 次调用的 3,014,876 B 峰值保持。

上述运行参数对应历史首轮 C8 检查与测量；当时 V6 driver 另按 C6→C8→C8→C6 配对，长预热复测将 `--warmup` 改为 `100` 并复用同一二进制。重新构建得到的是当前源码产物，不能冒充当时的二进制。显式小显存 correctness 保留全部 fixture，分配上限为 8 MiB；packed-only benchmark 分配上限为 12 MiB。最新 C9／C10 可分别指定 `--certified-variant 9`／`10` 构建，原 suite、半行 fixture 和额外 cancellation 检查都须通过。完整冻结命令与 C2 同条件对照见 [最新 native 审计](../../evidence/results/mxfp4-training/2026-11-23-crater-c8/candidate-c9-c10/audit.md)；默认构建仍选择 C2。

### 3.3 真实 V100 正确性检查

```bash
/tmp/mxfp4-cuda-build-20261124/check_mxfp4 \
  --device 0 --correctness-only \
  > /tmp/mxfp4-native-baseline-20261124.jsonl \
  2> /tmp/mxfp4-native-baseline-20261124.stderr

/tmp/mxfp4-cuda-build-20261124/check_mxfp4_tiled \
  --device 0 --correctness-only \
  > /tmp/mxfp4-native-tiled-20261124.jsonl \
  2> /tmp/mxfp4-native-tiled-20261124.stderr

/tmp/mxfp4-cuda-build-20261124/check_mxfp4_certified \
  --device 0 --correctness-only \
  > /tmp/mxfp4-native-certified-20261124.jsonl \
  2> /tmp/mxfp4-native-certified-20261124.stderr
```

默认要求真实 V100，检查在非默认 CUDA stream 上执行。完整域测试覆盖 16 × 16 个 E2M1 编码组合、两种非零 dot 下全部 65,536 个尺度字节组合、零 dot 的 NaN 传播、正负次正规数舍入、相反溢出块的 FP32 累计、非方阵／K 尾部及 `K=0`。certified 额外覆盖能量／span／lattice 边界、NaN scale 拒绝、非对齐与大行数路径、canonical `+0` 和 context 重用。

每个案例分配前检查显存，默认保留 128 MiB；受限共享 GPU 的已归档 certified 运行显式使用 `--reserve-mib 64`。程序不停止或调整现有服务。`--allow-other-gpu` 仅用于明确标记的对照，结果中的 `actual_V100` 必须保留。正确性 harness 内的旧 `--benchmark` 不提供完整比较；正式性能比较使用下面的 `bench_mxfp4`，certified harness 也明确拒绝该旧选项。

融合版完整 correctness 另支持显式 `--small-memory-correctness --reserve-mib 48`：仅允许 certified>=3，显式分配封顶 8 MiB，禁止 benchmark，保持全部 fixture。它不改变普通模式门槛，也不影响 Torch 检查。实际预算、适用范围和受限设备的运行证据见融合报告。

### 3.4 Native packed-input 性能

在上述正确性检查通过后，复现第 1.1 节受限显存的单 shape 测量：

```bash
/tmp/mxfp4-cuda-build-20261124/bench_mxfp4 \
  --device 0 --m 2048 --n 2048 --k 2048 \
  --mode random_block_scales \
  --iters 10 --rounds 5 --warmup 3 \
  --reserve-mib 64 --max-mib 64 \
  --output /tmp/mxfp4-packed-2048-20261124.jsonl
```

有充足显存时省略 `--reserve-mib`，保留默认 512 MiB；省略 M/N/K 可运行默认 5 组 shape。输入模式有 `zero`、`random_unit_scale`、`random_block_scales`，默认全部执行；`--quick --correctness-only` 可先做小规模比较。分配同时受 `--max-mib`、reserve 和当前 free memory 的 75% 上限约束。

benchmark 从真实 packed nibble／E8M0 数据开始，分别报告 baseline、tiled、certified total、预解码 cuBLAS compute、解码成本、解码＋cuBLAS total。所有 job 都输出 CUDA event 与同步 wall 时间；比较时统一选择同一种时钟。每个 case 先执行独立 CPU oracle 全量或抽样检查，并要求 tiled／certified 对 baseline 全输出精确相等；只有普通 cuBLAS 对照允许预设的重排误差容差。GPU 型号、variant、时钟／功耗、证书、显存预算、输入分布和完整 scope 保留在 JSONL 中。

### 3.5 PyTorch GPU integration check：119 项及独立 certified 检查

```bash
export MXFP4_CUDA_LIBRARY=/tmp/mxfp4-cuda-build-20261124/libmxfp4_sm70.so
python -m tools.mxfp4_training.check_torch_cuda \
  --device cuda:0 \
  --output /tmp/mxfp4-torch-check-20261124.json
```

此检查覆盖 `cuda` 和 `cuda_tiled`、packed 特殊值、RNE／SR、所有 8 种 F/D/W 开关、尾部／非连续输入、原始 FP32 转置重新量化以及非默认 stream，并用独立 NumPy 算术产生预期值。必须为实际 SM70，context 建立后至少有 512 MiB free；缺 CUDA 或显存不足会记录 blocked，不会回退 CPU 或写成通过。

这 119 项已在 SXM2 节点通过，但不覆盖 `cuda_certified`。certified 使用独立检查，`--expected-certified-variant` 必须与正在加载的共享库一致：

```bash
python -m tools.mxfp4_training.check_torch_certified \
  --device cuda:0 \
  --compile-evidence /tmp/mxfp4-cuda-build-20261124/compile-evidence.json \
  --expected-certified-variant 2 \
  --output /tmp/mxfp4-torch-certified-c2.json
```

16 项检查包括独立 packed oracle、三个训练矩阵方向、非默认 stream、两线程各自 context、提交后的错误清理与 CUDA Graph capture 拒绝。C8／C10 的报告随 native 归档核验，C2 的报告随 RNE v2 归档核验；复测 C8／C10 时须分别加载对应库并使用 `8`／`10`。源码、共享库或 variant 改变后，应重新生成相匹配的检查报告。

### 3.6 PyTorch 量化＋GEMM 成本

必须先在外部完成第 3.5 节，并保留完整通过的报告：

```bash
python -m tools.mxfp4_training.bench_torch_cuda \
  --device cuda:0 --m 1024 --n 1024 --k 1024 \
  --rounds 3 --iters 3 --warmup 1 \
  --check-report /tmp/mxfp4-torch-check-20261124.json \
  --output /tmp/mxfp4-torch-bench-20261124.json
```

脚本核对外部 check 的 source/library 哈希、Torch/CUDA build 与设备身份；源码或共享库改变后必须重跑 check。它要求实际 SM70 和至少 1 GiB free，并有保守临时显存预算。输入是固定种子的 GPU Gaussian FP32 张量。

默认六个 job 分别是原始 FP32 matmul、预转换 FP16 compute、FP32→FP16 casts＋matmul、两个输入的量化、量化＋`cuda`、量化＋`cuda_tiled`。量化总路径包含 validation 同步、临时张量与 packed/output 分配、Python／ctypes dispatch 和 GEMM；输入生成、初次 device context 初始化与正确性复制在计时外。报告 event／wall 数据和前后显存，量化单项不报告 TFLOPS。

**普通 Torch FP16 matmul 输出仍为 FP16；没有隐藏的 `.float()` 计时转换，也不是 native benchmark 的 FP32-output cuBLAS primitive。** 所有初始及最终计时结果须有限，MX backend 全输出须 exact，最终结果还须与各自初始结果保持同 dtype 一致。对原 FP32 输出的误差只作背景记录；MX backend 的 exact gate 独立执行。

显式增加 `--with-certified` 可测量 certified Torch wrapper，此时每次 `create→GEMM→destroy`、workspace 分配及销毁同步都在计时内。它不是 native 复用 context 的 GEMM 成本。若选择 `--quantizer-backend cuda_rne`，还必须传入同源同库的 `--quantizer-check-report`；先按 §3.10 构建并完成量化检查。

已独立核验的早期 1024³、默认 Torch 量化测量中，两输入量化为 **1.898 ms**，量化＋tiled 为 **2.047 ms**，原始 Torch FP32 matmul 为 **0.249 ms**，均为同一报告的 wall median。这说明当时的完整 wrapper 受量化与调用成本影响，不能用 native TFLOPS 推出训练接口加速；原始轮次、数据范围和限制见 [框架补跑审计](../../evidence/results/mxfp4-training/2026-11-23-crater-c8/framework-followup-procfix/independent-audit.md)。

### 3.7 CPU causal decoder smoke：历史阶段已执行

```bash
python -m tools.mxfp4_training.pretrain_smoke \
  --backend reference --device cpu \
  --steps 40 --seed 20261123 --lr 0.005 \
  --output /tmp/mxfp4-smoke-cpu-20261123.json
```

脚本比较全 FP32、仅 Forward、Forward+Dgrad、Forward+Dgrad+Wgrad，以及上游梯度 SR 五组；逐步检查 loss 和梯度有限、参数实际更新、AdamW 状态保持 FP32，并保存训练曲线、recipe、版本和源码哈希。

### 3.8 GPU causal decoder smoke：两条基础 backend 已完成

```bash
export MXFP4_CUDA_LIBRARY=/tmp/mxfp4-cuda-build-20261124/libmxfp4_sm70.so
python -m tools.mxfp4_training.pretrain_smoke \
  --backend cuda --device cuda:0 \
  --steps 40 --seed 20261123 --lr 0.005 \
  --output /tmp/mxfp4-smoke-cuda-20261124.json

python -m tools.mxfp4_training.pretrain_smoke \
  --backend cuda_tiled --device cuda:0 \
  --steps 40 --seed 20261123 --lr 0.005 \
  --output /tmp/mxfp4-smoke-cuda-tiled-20261124.json
```

先完成第 3.5 节的 GPU integration check。CUDA wrapper 仅接受计算能力 `(7,0)`，不会静默回退 CPU。smoke 在创建 context 后要求至少 512 MiB 空闲显存；该门槛是启动检查，不是对所有环境峰值占用的保证。上述两条 backend 已各完成五组配方 × 40 步，全部 FP32 参数、梯度与 AdamW 状态有限，参数实际更新；两者曲线在该实验中一致，见 [独立审计](../../evidence/results/mxfp4-training/2026-11-23-crater-c8/framework-followup-procfix/independent-audit.md)。

使用 certified 时显式指定 `--backend cuda_certified`，并先通过对应的 16 项检查；使用 CUDA RNE 时另加 `--quantizer-backend cuda_rne`，并先完成量化检查。RNE v2 中 `cuda_tiled` 和 C2 `cuda_certified` 也已各完成五组 × 40 步，两条 backend 的训练曲线一致，全部参数张量实际更新且 FP32 状态有限。这些 smoke 保留逐步验证，耗时不能作为训练吞吐比较。

### 3.9 Crater 执行入口与历史失败记录

[crater_queue_runner.py](crater_queue_runner.py) 跟进一个已授权、已存在的 Crater job；固定源码 archive 和 payload 通过 `--ready-json` 发布门禁后，才在分配到的节点上运行 [crater_remote_run.py](crater_remote_run.py)。runner 本身不提交、停止或删除平台 job。队列状态及后续结果入口见 [性能证据说明](../../evidence/results/mxfp4-training/2026-11-24-performance/README.md)。

payload 只探测和使用已安装的 CUDA 12.x、NumPy 与 CUDA 版 PyTorch，执行 native build/check、native benchmark、框架 check、量化成本 benchmark 与训练 smoke。每个候选须先通过自己的 correctness，后续独立实验保持冻结源码、明确的阶段依赖和总 deadline。前置检查不通过或运行环境缺失时，相应阶段记录 blocked／failed；结果和哈希保存在独立目录与 `results.tar.gz` 中。具体阶段、资源与截止时间以各次实际 receipt／manifest 为准。

原排队作业因镜像缺少 sshd 失败，后续启动与认证恢复过程也保留了失败记录。恢复后的 SXM2 节点已实际完成算子和框架工作。早期 B 归档中的三个框架阶段因 Python 解释器路径处理而 blocked；修正后使用原冻结库和源独立补跑，原 blocked 结论仍保留。后续 [native C9／C10 审计](../../evidence/results/mxfp4-training/2026-11-23-crater-c8/candidate-c9-c10/audit.md)与[框架补跑审计](../../evidence/results/mxfp4-training/2026-11-23-crater-c8/framework-followup-procfix/independent-audit.md)分别核验实际阶段内容，runner 启动、节点分配或 SSH 成功均不代替 GPU 证据。

119 项基础检查、16 项 certified 检查、量化逐字节检查和合成 smoke 各有独立范围。它们都不能替代真实 LLM 预训练的完整 step、收敛与峰值显存测量。

### 3.10 显式 CUDA RNE 量化

CUDA RNE 是可选构建项；不带该选项的默认库不包含对应 ABI：

```bash
python -m tools.mxfp4_training.build_cuda \
  --nvcc /usr/local/cuda-12.8/bin/nvcc \
  --output-dir /tmp/mxfp4-rne-build \
  --with-certified --with-benchmark --with-quantizer-rne \
  --tiled-variant 1 --certified-variant 2

export MXFP4_CUDA_LIBRARY=/tmp/mxfp4-rne-build/libmxfp4_sm70.so
python -m tools.mxfp4_training.check_quantize_cuda \
  --device cuda:0 \
  --compile-evidence /tmp/mxfp4-rne-build/compile-evidence.json \
  --output /tmp/mxfp4-rne-check.json

python -m tools.mxfp4_training.bench_quantize_cuda \
  --device cuda:0 --size 1024 --warmup 10 --rounds 5 --iters 10 \
  --compile-evidence /tmp/mxfp4-rne-build/compile-evidence.json \
  --check-report /tmp/mxfp4-rne-check.json \
  --output /tmp/mxfp4-rne-bench-1024.json
```

也可用 `--size 4096` 写入新的输出文件。benchmark 在输入已驻留 GPU 后比较两个完整量化 wrapper，包含分配、Python／ctypes dispatch、有限值检查和同步；每轮最后一次实际 timed codes／scales 都在计时终点后全量比较。它不包含 GEMM，也不报告 TFLOPS。要测量量化＋GEMM，另用 §3.6 的相匹配框架检查和参数。

训练配方显式设为 `MXRecipe(..., backend="cuda_tiled", quantizer_backend="cuda_rne")`，或选择已验证的 `cuda_certified` 库。只在该操作数使用 `nearest` 时调用 native RNE；stochastic 上游梯度继续走原 Torch SR 实现，保持 RNG 顺序。默认 `quantizer_backend="torch"` 不变，缺少 RNE ABI 时明确失败，不静默替换 backend。量化证明见 [CUDA RNE 目录](../../evidence/results/mxfp4-training/2026-11-23-crater-c8/candidate-cuda-rne/QUANTIZER_PROOF.md)，实际结果与初始失败说明见 [v2 独立审计](../../evidence/results/mxfp4-training/2026-11-23-crater-c8/candidate-cuda-rne-v2/audit.md)。135 项量化 checker 没有执行量化器自身的 Graph capture／缺库拒绝测试；GEMM 的 Graph capture 拒绝由独立 certified suite 覆盖，两者不能混计。

### 3.11 显式 workspace：构建、检查与完整调用对照

以下构建使用 C2、CUDA RNE 和默认关闭的 prepare raw-cache。先通过新库自己的 native、量化及 workspace 检查，benchmark 再校验检查报告与当前源／库的绑定：

```bash
python -m tools.mxfp4_training.build_cuda \
  --nvcc /usr/local/cuda-12.8/bin/nvcc \
  --output-dir /tmp/mxfp4-workspace-build \
  --with-certified --with-quantizer-rne \
  --tiled-variant 1 --certified-variant 2

export MXFP4_CUDA_LIBRARY=/tmp/mxfp4-workspace-build/libmxfp4_sm70.so
/tmp/mxfp4-workspace-build/check_mxfp4
/tmp/mxfp4-workspace-build/check_mxfp4_tiled
/tmp/mxfp4-workspace-build/check_mxfp4_certified

python -m tools.mxfp4_training.check_quantize_cuda \
  --device cuda:0 \
  --compile-evidence /tmp/mxfp4-workspace-build/compile-evidence.json \
  --output /tmp/mxfp4-workspace-rne-check.json

python -m tools.mxfp4_training.check_certified_workspace \
  --device cuda:0 --expected-certified-variant 2 \
  --compile-evidence /tmp/mxfp4-workspace-build/compile-evidence.json \
  --output /tmp/mxfp4-workspace-check.json
```

每个 benchmark 命令是独立进程。下面对两个尺寸各按 `per_call→workspace→workspace→per_call` 执行，保留不同输出文件；复跑时更换所有报告路径，避免覆盖：

```bash
for size in 1024 4096; do
  run=0
  for lifecycle in per_call workspace workspace per_call; do
    run=$((run + 1))
    python -m tools.mxfp4_training.bench_certified_workspace \
      --device cuda:0 --expected-certified-variant 2 \
      --compile-evidence /tmp/mxfp4-workspace-build/compile-evidence.json \
      --check-report /tmp/mxfp4-workspace-check.json \
      --quantizer-check-report /tmp/mxfp4-workspace-rne-check.json \
      --quantizer-backend cuda_rne --lifecycle "$lifecycle" \
      --m "$size" --n "$size" --k "$size" \
      --seed 20261123 --warmup 10 --rounds 5 --iters 10 \
      --output "/tmp/mxfp4-workspace-${size}-${run}-${lifecycle}.json" || exit
  done
done
```

输入在计时前驻留 GPU；每个 timed operation 仍重新执行两次完整量化和 GEMM。报告单独给出 workspace 创建／关闭耗时，不能将其丢弃后解释为单次调用成本。修改源码、库或设备后，须重新生成相匹配的检查报告，不能复用旧报告作为门禁。

在代码中显式使用固定逻辑 shape 的 workspace：

```python
import torch
from tools.mxfp4_training.certified_workspace import CertifiedWorkspace
from tools.mxfp4_training.torch_ops import quantize_mx_cuda_rne

m = n = k = 1024
device = torch.device("cuda:0")
a = torch.randn((m, k), device=device, dtype=torch.float32)
b_rows = torch.randn((n, k), device=device, dtype=torch.float32)

with CertifiedWorkspace(m, n, k, device=device) as workspace:
    for _ in range(10):
        qa = quantize_mx_cuda_rne(a)
        qb = quantize_mx_cuda_rne(b_rows)
        out = workspace.gemm(qa, qb)  # FP32 [m,n]，A @ B_rows.T
    info = workspace.info()          # 同步诊断；放在性能计时之外
# 退出 with 会等待 native 工作结束并释放 context。
```

构造函数也可显式接收 `library_path=...`，否则使用 `MXFP4_CUDA_LIBRARY`。只允许进入 context 一次；进入时绑定 host thread、SM70 device、`(M,N,logical_K)` 与库。资源必须在同一 host thread 上关闭，不依赖隐式 finalizer。`gemm` 接受对应形状的 `TorchMXTensor`，每次分配新的 FP32 输出并重算证书；复用 context 不会缓存上次输入的准入结果。

同一 current stream 上可连续提交；切换 stream 前，上一 stream 必须已经完成，否则抛出 `CertifiedWorkspaceBusy`，不会自动插入隐藏依赖。调用者仍负责跨 stream 的输入／输出数据依赖。成功提交不强制等待整个 GEMM，但量化的有限值检查与 packed padding 验证仍可能同步。`info()`、`certified_info={}` 诊断请求和 `close()` 的同步规则，以及错误后的资源处理见 §6。该接口不支持 CUDA Graph capture，也不自动参与 autograd 或 `MXLinear`。

复现 prepare-cache 独立实验时，可在新的 C2 构建目录显式增加 `--certified-prepare-raw-cache`；该选项要求 `--with-certified --certified-variant 2`。它不会由 workspace 自动开启，也未被选作默认；原始微检查、OFF／ON 构建和完整 pipeline 配对命令保存在 [cache 实验目录](../../evidence/results/mxfp4-training/2026-11-23-crater-c8/candidate-c2-prepare-cache/)。

## 4. 编码、量化与计算契约

### 4.1 存储布局

设原始矩阵为 `[R,K]`，`Kp = ceil(K/32) × 32`：

| 字段 | dtype／形状 | 含义 |
|---|---|---|
| `codes` | `uint8[R,Kp/2]`，连续 | 每字节低 nibble 存偶数 K、高 nibble 存奇数 K |
| `scales` | `uint8[R,Kp/32]`，连续 | 每行每 32 个元素独立的 E8M0 尺度 |
| `logical_shape` | `(R,K)` | 不含 K padding 的形状 |

E2M1 的正数编码 `0…7` 对应 `+0, 0.5, 1, 1.5, 2, 3, 4, 6`，最高位是符号位，保留负零。K padding 必须是位模式为零的规范正零。Python 接口拒绝空矩阵；baseline／tiled C ABI 另支持 `M/N=0` no-op 和 `Kp=0` 输出零。certified context 要求正的 M/N，并支持 `Kp=0`。

E8M0 字节 `e=0…254` 表示 `2^(e-127)`，**码 0 是 `2^-127`，没有零尺度**；码 255 表示整个块为 NaN。仅使用 `e << 23` 重解释为 FP32 会错误处理这些边界。格式依据 [OCP MX v1.0，§5.2–5.4](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf/)。

### 4.2 量化配方

本实现对每个非零块使用 `floor(log2(amax))-2` 的尺度指数，再夹到 `[-127,127]`；全零块使用尺度 1。实现用 `frexp` 提取二进制指数，避免 `log2` 在二的幂邻近位置产生错误分组。

归一化元素在 E2M1 范围内舍入，超过最大幅值 6 时饱和。支持：

- `nearest`：round-to-nearest, ties-to-even，按 E2M1 尾数最低位对应的偶数编码处理 midpoint。
- `stochastic`：按相邻可表示值的距离随机舍入；正负数对称处理。

**SR 只在未裁剪区间内无偏。** 此 max-scaling 配方可能使块内最大值落在归一化区间 `[6,8)`，饱和仍有偏；不能把整个量化器称为无偏。NumPy 接受有限 FP32／FP64 矩阵；PyTorch 量化路径只接受有限 FP32。直接读取 packed 输入时可以处理 NaN scale，但从高精度输入量化的入口会拒绝 NaN／Inf。

可选的 `quantize_mx_cuda_rne` 以每 warp 一个 MX32 块执行相同 nearest 配方；阈值处保留 ties-to-even，原始负零保留符号，padding 写 canonical `+0`，全零块尺度字节为 `127`。它支持同设备、非空二维 FP32 张量的非负 stride，包括零 stride 广播、storage offset 和 lazy negative view。有限值标志回读会同步当前 stream，CUDA Graph capture 不受支持。次正规数、二幂边界与 midpoint 的等价性论证见 [量化证明](../../evidence/results/mxfp4-training/2026-11-23-crater-c8/candidate-cuda-rne/QUANTIZER_PROOF.md)；这不改变默认 Torch 量化或 SR。

### 4.3 GEMM 的逐块结果

两个 row-wise 表示通过 `gemm(A, B_rows) = A @ B_rows.T` 相乘。每个输出、每个 MX32 块执行：

```text
qA = 2 × E2M1(A), qB = 2 × E2M1(B)
integer_dot = sum(qA[i] × qB[i], i=0…31)
block_result = FP32(ldexp(integer_dot, scaleA_byte + scaleB_byte - 256))
output = FP32(output + block_result)
```

`|integer_dot| ≤ 4608`，整数块内积可精确表示；块结果沿 K 从小到大按上述 FP32 顺序累计。NaN scale 即使遇到零 dot 也传播 NaN。溢出、下溢以及跨块相消遵循这一明确顺序；它不等价于先用无限精度算完整点积再只舍入一次。

baseline／tiled 的通用尺度路径通过 `ldexp` 一次施加在整个 dot 上：若先把尺度乘积单独物化为 FP32，可能提前产生 Inf 或零，丢失本来可表示的结果。它们不先把有尺度的元素解为 FP16，因为那会缩小有效动态范围；tiled 的尺度快路径也必须保持相同结果。完整编码与尺度支持不意味着与量化前 FP32 GEMM 相同，量化误差仍然存在。

NumPy oracle 用独立整数矩阵乘法构造预期值；CUDA 使用已有 [SM70 fragment 映射](../../evidence/benchmarks/sm70-packing/study/simulation/round3/cuda/fragment_mapping.hpp)，在 FP16 Tensor Core 中计算无尺度整数 q 的块点积。

certified 路径保留同一算术契约：GPU 先检查每行 active scale span、normalized integer 的行能量乘积上界和最终 lattice 范围；证书成立时，才可把完整 K 的计算重排为精确的 normalized FP16→FP32 cuBLAS，再一次缩放输出。证书不成立时输出来自 exact tiled fallback，NaN／Inf／次正规数等完整域语义仍保留。准入范围和整数精确性证明见 [certified fastpath 说明](../../docs/research/performance/sm70-packing/mxfp4-certified-fastpath.md)。这不是允许 cuBLAS 用误差容差替代 ordered-MX32 结果。

## 5. 训练接口与三个 reduction axis

令 `X[M,K]`、`W[N,K]`，输出 `Y = X @ W.T + bias`，上游梯度为 `G[M,N]`：

| 路径 | 数学目标 | 传给统一 GEMM 的 A／B_rows | MX32 分组轴 |
|---|---|---|---|
| Forward | `X @ W.T` | `X[M,K]`／`W[N,K]` | K |
| Dgrad | `G @ W` | `G[M,N]`／`W.T[K,N]` | N |
| Wgrad | `G.T @ X` | `G.T[N,M]`／`X.T[K,M]` | M |

每一项转置都**从原始 FP32 张量重新量化**，不能转置已量化 nibble 后继续沿用原来的 scales。分组轴改变会改变数值，双方向量化需要独立表示；相关动机也见 [Transformer Engine 的低精度训练说明](https://docs.nvidia.com/deeplearning/transformer-engine/examples/fp8_primer.html)。

`MXRecipe` 的 `forward`、`dgrad`、`wgrad` 是三个独立开关；关闭的路径使用普通 FP32 矩阵乘法。`rounding` 控制 Forward 的两个输入和反向中的权重／activation；`gradient_rounding` 只控制反向中作为左操作数的 `G` 或 `G.T`。当前没有 Hadamard 变换。

反向是配方指定的 **surrogate gradient**，不是舍入函数的数学导数，仅支持一阶自动微分。不能用针对光滑函数的有限差分 gradcheck 作为它的正确性定义；测试应比较所声明的 Dgrad／Wgrad 独立 oracle。

### 5.1 `MXLinear` 最小用例

```python
import torch
from tools.mxfp4_training.torch_ops import MXLinear, MXRecipe

torch.manual_seed(20261123)
recipe = MXRecipe(
    forward=True,
    dgrad=True,
    wgrad=True,
    rounding="nearest",
    gradient_rounding="nearest",
    backend="reference",
)
layer = MXLinear(33, 17, recipe=recipe)
optimizer = torch.optim.AdamW(layer.parameters(), lr=0.001)
x = torch.randn(4, 16, 33, dtype=torch.float32, requires_grad=True)

optimizer.zero_grad(set_to_none=True)
y = layer(x)                      # [4,16,17]，FP32
loss = y.square().mean()
loss.backward()
optimizer.step()                  # 参数与优化器状态保持 FP32
```

输入采用 `[...,K]`，至少二维；权重 `[N,K]`，bias `[N]`，全部为同设备 FP32。内部关闭 autocast，因此调用者仍需传入 FP32；不应对该层执行 `.half()` 或把 BF16 输入当作已支持。bias 不量化，bias gradient 是 FP32 的 `G.sum(dim=0)`。

GPU 用法是将 `backend` 设为 `cuda`、`cuda_tiled` 或显式的 `cuda_certified`，在构造时传 `device="cuda:0"`，并把 FP32 输入放在同一设备；先设置前述 `MXFP4_CUDA_LIBRARY`，完成对应检查。需要 CUDA RNE 时另设 `quantizer_backend="cuda_rne"`，且加载含该 ABI 的库。三个训练方向仍分别从原始 FP32 张量按各自 reduction axis 生成表示，量化器选择不复用错误分组的转置 packed 数据。

`cuda_certified` 在 Torch 调用中使用同步的 per-call context 生命周期，native benchmark 则复用 context；不能把 native certified 的速度写作这个训练接口的速度。§3.11 的 `CertifiedWorkspace` 已验证显式重复调用，但未接入本节的 autograd／`MXLinear`；设置 `backend="cuda_certified"` 不会自动启用它。

### 5.2 Checkpoint 必须包含 recipe 和 RNG

`recipe` 是 Python 属性，**不会自动写入 `state_dict()`**。仅恢复参数会遗漏量化开关、舍入方式和 `quantizer_backend`。单层示例：

```python
from dataclasses import asdict

torch.save({
    "model": layer.state_dict(),
    "optimizer": optimizer.state_dict(),
    "mxfp4_recipe": asdict(layer.recipe),
    "rng_cpu": torch.get_rng_state(),
    "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
}, "mxlinear-checkpoint.pt")
```

恢复时先用保存的配置构造 `MXRecipe` 和层，再加载模型／优化器状态与 RNG。多层模型应按模块名保存各层 recipe，并记录代码版本、训练步和数据进度。底层量化函数允许显式 NumPy RNG 或 `torch.Generator`；当前 `MXLinear` 的 SR 使用 PyTorch 全局 RNG，精确继续训练需要一并恢复它。

## 6. Stream、内存和当前性能范围

baseline／tiled C ABI 在调用者提供的 stream 上 enqueue，返回 CUDA 状态码，不自行同步。它要求连续、互不重叠、属于当前 CUDA device 的有效缓冲区；大小、设备和生命周期由调用者负责。PyTorch wrapper 在目标 device 的 current stream 上启动，并对所有输入／输出调用 `record_stream`。跨 stream 生产和消费仍需调用者遵守正常的 PyTorch stream 依赖规则。

certified C API 使用绑定 device 与 shape 的显式 context／workspace，同一 context 不可并发使用。`mxfp4_certified_gemm(..., used_fast=nullptr)` 提交完整异步 pipeline；传入非空 `used_fast` 或调用 `mxfp4_certified_get_info` 会同步，诊断必须放在异步性能计时之外。状态码中的 cuBLAS 错误有独立编码，应使用 `mxfp4_certified_error_string` 解释。具体资源、stream 与销毁规则见 [头文件契约](csrc/mxfp4_gemm_certified.h)。

原 `gemm_packed(..., backend="cuda_certified")` 桥接仍每次在调用线程内创建 context，在目标 device 的 current stream 上提交，然后在 `finally` 中销毁并等待该 context 的工作结束；错误路径也执行清理。输入／输出在提交前登记 `record_stream`，Graph capture 在创建资源前拒绝，不使用隐式共享 context 缓存。

新增 `CertifiedWorkspace` 把同一 shape 的 context 显式复用于多个调用，保留严格证书、输入验证和完整域回退。`gemm`／`info`／`close` 只能在进入 context 的 host thread 使用；同一 context 不可并发，stream 切换只在前一 stream 完成后接受。所有 packed 输入和输出都在 native 提交前登记 `record_stream`，成功调用可在同一 stream 排队；调用者仍负责张量之间的数据依赖。前置验证错误不消耗 context，native 提交尝试后的错误会同步关闭再抛出，`close()` 等待已有工作且不可重开；没有隐式 finalizer 或全局缓存。`info()` 或传入空字典 `certified_info={}` 请求诊断会同步，应放在性能计时之外。Graph capture 不受支持。§1.5 的真实结果仅支持这种显式重复调用场景，创建与关闭成本仍需另计，autograd 线程中的复用尚未集成。

当前已有的性能工作与剩余成本如下：

- 默认 Torch 量化使用多个 kernel 和与元素数量同阶的临时张量；有限值／padding 检查会产生主机同步。显式 CUDA RNE 将单个操作数的 nearest 量化合并到 native kernel，完整 wrapper 仍有验证、分配和同步；尚无 F/D/W 双方向联合生产 kernel。
- tiled 已实现共享内存 tile 复用、PRMT nibble 解码和受数值门控的尺度路径；当前选用 variant 1。原始 baseline 继续保留为完整算术对照。
- certified variant 2 使用向量化 prepare 与 epilogue，按 GPU 证书选择 cuBLAS 精确结果或 full-domain tiled 回退。为保持无主机分支同步，当前拒绝路径仍执行一遍安全的 normalized cuBLAS，再执行条件 fallback；该成本保留在总计时内。
- 融合 variants 3–10 在证书通过后直接从 packed 输入执行 HMMA；拒绝时融合 CTA 提前返回，随后 exact tiled 回退。C6 增加下一块数据预取；Kp<=32768 时准备阶段缓存块能量，减少第二次 packed 读取。C8 使用半行宽加载，C9 尝试 cursor，C10 使用精确 half2 解码。最新同条件比较中 C10 优于 C8，C9 退步，C2 仍快于这些融合候选。
- C2 的 cuBLAS 算法筛选已完整核验，但只覆盖 normalized compute；候选必须再经过完整 pipeline 配对才能改变默认。prepare raw-cache 已通过 327 项微检查与 OFF／ON 完整对照，观测吞吐差异全部小于 1%，不足以建立稳定收益，默认继续关闭。显式 workspace 的已验证收益来自 context 复用，其实验使用 OFF 库。
- 证书是否通过取决于数据尺度分布和能量；合成 case 的通过不能推出所有训练张量均能走 cuBLAS 路径。实际训练数据的准入率、回退成本及准备阶段占比仍需测量。
- 此前 `|q|≤3` 等受限域的 middle-only packing 尚未接入，也没有该 packing 的残差修复路径。certified 的精确证书是独立方案，不是对这项受限 packing 的启用声明。
- master weights、保存供反向使用的 activation 和 optimizer states 均保持 FP32。packed 缓冲区的低位存储不等于训练峰值显存下降。
- 训练 smoke 中有逐参数有限值和更新检查；其墙钟时间不可作为正式吞吐比较。

目前已有可核对 scope 的 native FP16／cuBLAS 比较，结论限于已归档的 packed-input、shape、分布和共享 GPU 条件。完整训练加速与显存收益仍未建立；需要把高精度输入量化、方向布局生成、GEMM、反向、优化器、峰值显存和完整 step 时间一起测量，并使用相同数学目标与训练配置的基线。

## 7. 后续验收顺序

1. **维持已通过的数值门并补齐明确缺项。** 源码或库改变后重跑相匹配的 RNE 135 项、基础 119 项、certified 16 项和显式 workspace 检查；另补量化器自身的 Graph capture／缺库拒绝用例。涉及 prepare-cache 时保留其逐位微检查。失败旧归档与修复后结果继续分开保留。
2. **把已验证的显式复用接入训练生命周期。** 当前 workspace 只有固定 shape、owner thread 的 packed forward API；下一步须为 F/D/W、autograd 工作线程、并发及异常清理设计明确归属，再在实际训练循环测量完整 step。不能直接让现有 `MXLinear` 跨线程共享同一个 context，也不能把重复调用倍率写成训练加速。
3. **只采用完整 pipeline 中可复现的 C2 改进。** raw-cache 已完成实卡对照，当前小于 1% 的差异不改变默认 OFF；后续优化应由代表性输入和准备阶段实际成本驱动。cuBLAS compute-only 筛选的优胜算法仍须经过同输入、充分预热的完整 pipeline 配对与全输出精确检查，不能直接替换默认。
4. **扩大代表性输入和训练成本测量。** 使用实际训练张量和更多 M/N/K，记录证书准入率、拒绝后的完整成本，以及量化、方向布局、反向、优化器、峰值显存和完整 step 时间。受限 packing 仍须独立的数据域／尺度条件与完整域回退；旧研究见 [SM70 packing 报告](../../docs/research/performance/sm70-packing/index.md)。
5. **进行真实语料的 LLM 预训练对照。** 使用明确的模型、token 预算与相同训练配置，比较 FP32／FP16 基线，分开 Forward／Dgrad／Wgrad、RNE／SR、尺度策略和 Hadamard 方案，记录 loss、梯度误差、收敛和吞吐。

可参考 [Training LLMs with MXFP4，AISTATS 2025](https://proceedings.mlr.press/v258/tseng25a.html) 的 SR／随机 Hadamard 反向配方，以及 [2026 原生 MXFP4 预训练研究 v4](https://arxiv.org/html/2605.09825v4) 对不同训练阶段和 Wgrad 的对照。两项工作的硬件与配方不同，不能把其速度或收敛结果直接迁移为 V100 本实现的结论。
