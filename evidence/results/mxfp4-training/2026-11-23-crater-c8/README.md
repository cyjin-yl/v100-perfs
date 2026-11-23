# V100-SXM2：完整原生比较、GPU 训练连接与后续优化

本目录保留 2026-11-23 的实际 Crater 执行证据。原生 C2–C10 比较、cuBLAS 算法筛选、框架补跑、CUDA RNE 量化、C2 准备缓存及显式 workspace 复用均已完成实卡运行和独立原始内容审计。每次实验使用独立冻结来源；失败、回退和较慢的候选都保留。

## 1. 已核验的结果与口径

设备为 **Tesla V100-SXM2-32GB**，CUDA 编译工具链为 **12.8.93**。这与前一轮共享的 V100-PCIE-32GB/CUDA 12.9 环境不同，不能把跨设备的绝对吞吐变化算成代码加速。

原生 B 和框架补跑建立的基础结果如下，后续优化及其独立计时范围见第 4–7 节：

- C2–C8 的原生正确性检查及 2048³/4096³ 完整 packed-input benchmark 全部通过。
- C2 在这两个尺寸上是已测完整路径中最快者，默认构建仍为 C2。
- 原生 B 有 32 个 passed、3 个 framework blocked；阻塞由虚拟环境解释器路径解析造成。独立补跑保留原 B 源码、动态库及结果不变，随后五个框架阶段实际全部通过。
- 框架补跑覆盖 `cuda` / `cuda_tiled`，包括 119 项 GPU 集成检查、1024³ Torch benchmark、两个后端各五组 40 步 synthetic decoder smoke。
- 这里的严格 packed GEMM 正确性针对仓库声明的逐 MX32 缩放、FP32 累计合同。它不表示量化前 FP32 模型与 MXFP4 模型完全相同，也不表示已经验证真实 LLM 预训练收敛。

## 2. 原生 B：同次运行的 C2–C8 完整比较

[独立汇总](attempt-b/comparison-summary.json)核验了[原始结果包](attempt-b/results.tar.gz)、所有成员哈希、编译日志和源码绑定。以下使用 `synchronized_wall` 中位数及 `2*M*N*logical_K` 有用 FLOP；每个测量都从相同 packed codes/scales 开始，计入每次准备、证书、所选 GEMM、最终缩放和条件回退调度。

| Certified variant | 2048³ wall ms | 2048³ useful TFLOPS | 4096³ wall ms | 4096³ useful TFLOPS |
|---|---:|---:|---:|---:|
| C2 | 0.3137511 | **54.7564** | 1.8125731 | **75.8253** |
| C3 | 0.9446707 | 18.1861 | 6.5775455 | 20.8952 |
| C4 | 0.6525657 | 26.3267 | 4.3027948 | 31.9418 |
| C5 | 0.7129747 | 24.0960 | 4.9423328 | 27.8085 |
| C6 | 0.5057271 | 33.9706 | 3.1436271 | 43.7199 |
| C7 | 0.5226432 | 32.8711 | 3.3322840 | 41.2447 |
| C8 | 0.4168314 | **41.2154** | 2.4983784 | **55.0113** |

输入为 `random_block_scales`、seed `20261123`。2048³ 的指纹为 `17e7aed60a9341e3`，4096³ 为 `6a76b74ce5d1d307`，同尺寸跨所有 variants 一致。14 项结果的真实证书全部 accepted，reject flags 为 0；每项调用数由 1 增至 54。初始与最后一次实际计时的 certified 输出均与精确 baseline 全量比较通过。

这是 C2→C8 固定顺序的一次 batch；每项 warmup 3、5 rounds × 10 iterations。2048³ 部分运行存在 1290→1530 MHz 的频率上升，4096³ 前后通常为 1530 MHz，C8 结束为 1515 MHz。因此 C8 相对 C6 的观测吞吐提升 21.33% / 25.83% 是描述性结果，不是本批次独立 ABBA 或锁频因果估计。C8 仍比 C2 慢，未改变默认版本。

计时排除了高精度量化、H2D、首次分配、workspace 创建、D2H 正确性检查和诊断读取。普通 cuBLAS compute 的起点是已物化 FP16，且其 handle math/workspace 设置与 C2 不完全相同；不能直接用两个总时长相减，声称得到 C2 prepare/epilogue 的实测分项。

### 正确性覆盖

各 C2–C8 的原始认证 suite 都包含 69 次调用、405,930 个输出，23 accepted / 46 fallback，零 mismatch。C8 再增加 9 次调用、180,224 个输出，包括 4 个非空新 loader 案例、3 个 C6 兼容路径案例、1 个 empty-K 和 1 个拒绝案例。合计 C8 为 78 次调用、586,154 个输出。

### 来源与失败保留

- 原始 A 的 CUDA 12.8 standalone fixture 编译失败完整保留于 [attempt-a](attempt-a/)，没有 GPU 正确性或性能通过结论。
- 修正仅把相同八个随机 shape 的 range-for 列表显式声明为 `const int[][3]`。B 原生算子数学、fixture、seed 未变。
- B 源码 commit 为 `80d3a7c2bb0bb551a34b28bd7fda59978b6304ef`，archive SHA256 为 `62378914e2b6ad7e34e3dd70736cb90aea07c962963494d97062d5f5738b0f57`。
- B result archive 为 140,605 B，SHA256 `f7267b0d24a4cae8888d211264473a2d9caab23da3e5fccbccf310ffa7a0e14b`；独立 audit 为 verified，issues 为空。
- B 初始框架探测把 `/opt/venv/bin/python3` 的 symlink resolve 到基础解释器，失去该 venv 的 NumPy/Torch。修复保留 lexical executable path；主机回归证据见 [host](host/)。

## 3. 框架补跑：119 项、实际训练 step 与瓶颈

[独立内容审计](framework-followup-procfix/independent-audit.md)及[机器可读结果](framework-followup-procfix/independent-audit.json)核验了完整[结果包](framework-followup-procfix/results.tar.gz)。包为 47,225 B，SHA256 `2de0fc7de6d7f3fb8e33a760c538fc2e36c37cf4f08b133960e9c7df9f778cd1`。

实际环境是 Python 3.12.15、NumPy 2.3.5、PyTorch 2.10.0+cu128。使用原 B 的 C2-built 动态库，SHA256 `eb3255a5b74756457c25cfa9422bd57398d642ea6db26577ee8d317c68789f07`。五个 stage 均实际 exit 0，原 B source/library/evidence 前后哈希一致。

119 项由 12 packed、9 RNE 量化、1 stochastic seed、96 linear、1 禁止 GPU-to-reference fallback 案例组成。内部 156 项 exact 数值 metric 零 mismatch，240 项普通 FP32 allclose metric 零 tolerance failure；其中 136 项存在允许的 FP32 位差，所以不能写成“119 项全部 bitwise 一样”。

### 包含量化的 1024³ Torch 测量

下表是三个原始 round 的同步 wall 中位数，每 round 三次调用、一次预热。FP16 对照产生 FP16 输出；packed GEMM 产生 FP32 输出。

| 工作 | wall median ms | useful TFLOPS |
|---|---:|---:|
| Torch FP32 matmul | 0.2486240 | 8.6375 |
| Torch FP16 compute | 0.0669127 | 32.0938 |
| FP32 casts + FP16 matmul | 0.0896507 | 23.9539 |
| 两个输入的 MXFP4 Torch 量化 | **1.8977567** | 不适用 |
| 量化 + `cuda` packed GEMM | 2.7897627 | 0.7698 |
| 量化 + `cuda_tiled` packed GEMM | **2.0466060** | **1.0493** |

量化后的两个整数输出首次和最后实际计时结果均完整核对；五个 GEMM 对照的最后实际计时输出均与各自初始结果相同。量化 totals 包含分配、validation、Python dispatch 和两个输入的量化。这个结果说明当前训练连接的主要成本在未融合的 Torch 量化，不能把 native packed-GEMM 的 TFLOPS 当成完整训练吞吐。

### 40 步 synthetic decoder smoke

两个 CUDA 后端使用相同种子与参数初始化，各自执行下表全部五组；两者的每步 loss 轨迹与初末验证 loss 完全一致。每组更新 17 个参数 tensor，master parameters 与 AdamW 状态保持 FP32，梯度和参数有限。

| 配方 | 初始 validation loss | 第 40 步后 loss |
|---|---:|---:|
| FP32 | 3.645810843 | 0.062208887 |
| MX forward | 3.608196259 | 0.071482189 |
| MX forward + dgrad | 3.608196259 | 0.073648632 |
| MX forward + dgrad + wgrad | 3.608196259 | 0.076575369 |
| MX all + stochastic gradient | 3.608196259 | 0.072527491 |

模型为一层、width 32、4 heads、11,168 参数；数据为长度 16、词表 32 的合成 modular token 序列。这是训练连接检查，尚不是 LLM 预训练质量结论。

初次独立补跑在任何 GPU 工作前被同 UID `sshd` 的 cwd 访问权限拦住，原始失败另存 [framework-followup-initial](framework-followup-initial/)。修复仅先按 argv 排除无关进程，再对可能属于 B 的进程检查 cwd；不确定的 B 候选仍停止执行。修正版与原版 helper 均保留于 [queue](queue/)。

## 4. C9/C10：完整路径的对称顺序比较

[独立审计](candidate-c9-c10/audit.md)实际核验了 25 个源文件、174 个结果文件、24 项计时，以及独立 SASS 记录；状态为 `verified`、issues 为空。冻结 commit 为 `ed39b0d715271e7b359672a081d758b4e4d5070f`，结果包 SHA256 为 `d6775a7149d64b175ff20af085e1f9788937531561f10a80edccd5efb09462b8`。41 个 stage 全部通过。

每个 shape 使用 `[2,8,9,10,10,9,8,2]` 顺序，每进程 warmup 100、5 rounds × 10 iterations。下表合并同 variant 两次进程的全部 10 个 wall round 取中位数；没有挑选最快一轮。进程内 rounds 有相关性，不能当作十次独立实验。

| 形状及输入 | C2 TFLOPS | C8 TFLOPS | C9 TFLOPS | C10 TFLOPS | C10 相对 C8 |
|---|---:|---:|---:|---:|---:|
| 2048³，block scales | **54.8631** | 44.0218 | 42.4072 | 49.0474 | **+11.416%** |
| 4096³，block scales | **75.3552** | 54.4022 | 51.9402 | 59.4211 | **+9.225%** |
| 8192³，unit scale | **89.2896** | 59.1940 | 56.7182 | 64.0541 | **+8.210%** |

计时范围为完整 packed codes/scales → FP32：每次准备、证书、GEMM、postscale 及条件回退调度都在内；量化、H2D、context 创建、首次分配、正确性拷贝和诊断读取在外。C2 仍然最快，默认构建没有改为 C10。

2048³/4096³ 指纹分别为 `17e7aed60a9341e3` / `6a76b74ce5d1d307`；8192³ unit-scale 为 `e7a8b86de92b433a`。24 项性能输入全部 accepted、flags=0，实际累计提交数从 1 增至 151；每个初始和最后一次实际计时输出都全量比较通过。

**8192³ block-scale 另有不计时边界探针**：指纹 `17da68c322a6382f`，U=25,710,077、V=26,753,427，因 energy bound 拒绝，flags=4。完整回退输出 67,108,864 项零 mismatch。这个输入不能用 unit-scale 的 89.29 TFLOPS 代替，也没有通过放宽证书取得快路径数字。

### 实际代码与 SASS

C9 保留解码和 HMMA 次序，尝试固定步长 packed/scale cursor。实际 load 调度提前了，但同条件完整吞吐反而下降 3.67% / 4.53% / 4.18%。这是保留的负结果。

C10 独立保留 C8 地址生成，使用 `mul.rn.f16x2` 实现精确二次幂缩放，减少整数解码操作。对于证书允许的 Δ=0..12，非零 normalized half 最大绝对值为 49152，可精确表示；最终 FP32 输出仍规范为正零。实际 half2 检查器枚举 65,536 个 packed patterns × 13 个 Δ，共 851,968 组、3,407,872 个 half 值：非零值逐位一致，内部零允许符号差异；完整 GEMM 的相消 fixture 另验最终正零。

| 实际编译产物 | C8 | C9 | C10 |
|---|---:|---:|---:|
| 完整函数静态指令槽 | 1664 | 1664 | **1568** |
| K-stage 循环静态指令槽 | 620 | 624 | **524** |
| registers/thread | 119 | 121 | 119 |
| shared bytes/CTA | 18,688 | 18,688 | 18,688 |
| stack / spill-store / spill-load，B | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

三个循环都含 256 条 HMMA、48 条 LDS、16 条 STS、两条 BAR。C10 加入 16 条 HMUL2，减少 36 条 IMAD、48 条 LOP3、32 条 SEL，增加 4 条 LEA，合计少 96 个静态指令槽。这支持“减少解码指令”的机制判断，但不是动态周期或 stall 测量。寄存器分配粒度下三者仍至多两个 256-thread CTA/SM，不能声称 occupancy 提升。

本轮原生 suite 共 397 次调用、2,548,018 个输出，零 mismatch；还通过旧 Torch 119 项及 C8/C10 各 16 项独立 certified 集成检查。边界 NVML 快照 SM 为 1335–1530 MHz，不是连续时钟或独占 GPU 证明。

## 5. C2 cuBLAS 算法筛选

[独立审计](c2-algorithm-screen/audit.md)已验证真实 ELF、源码、编译日志、前序数值门以及全部结果成员。结果归档为 1,028,506 B，SHA256 `b9bb395a2abaeaab11e08e44452361f931c7d710a0e16861d4305e899bfa88c6`。

这里固定实际 C2 normalized FP16 A/B、FP32 compute/output、math flags、4 MiB workspace 和 stream，筛选 DEFAULT、DEFAULT_TENSOR_OP 与 16 个显式 Tensor Op 算法。**计时仅为 normalized GEMM compute**，不包含准备、证书或 postscale，不作为完整算子吞吐。

| 输入 | DEFAULT_TENSOR_OP TFLOPS | DEFAULT TFLOPS | 最快观测算法 |
|---|---:|---:|---|
| 4096³ block scales | **92.5597** | 92.3868 | DEFAULT_TENSOR_OP |
| 8192³ unit scale | 95.4294 | **99.2804** | DEFAULT |

两种 accepted 输入各 18 种算法都通过初次与最后实际计时输出的全量位比较，均有限且零规范为正零；每算法 151 次成功调用。8192³ block scales 被证书拒绝，仅有 blocked 结果，没有任何算法时间。因此整体 `completed_with_blocked_cases` 是真实边界状态，不能写成三个 case 都 passed。

算法是筛选顺序执行，边界时钟有变化，8192 的较小差异尚未做完整路径独立 ABBA 确认。没有据此改动 production cuBLAS 算法。

## 6. CUDA RNE 量化：已验证的完整 wrapper 提速

显式 `quantize_mx_cuda_rne` 把 FP32 MX32 RNE 量化放入一个 CUDA kernel；构建需 `--with-quantizer-rne`。`MXRecipe(quantizer_backend="cuda_rne")` 只替换最近偶数舍入；stochastic rounding 继续使用原 Torch 路径，F/D/W 仍分别从原始 FP32 方向的输入编码。默认量化仍为 Torch。

源 commit `f338faaa5cbdcd900aa82ae95af00ed5fb770235`，29 个文件；[成功结果包](candidate-cuda-rne-v2/results.tar.gz)为 146,654 B，SHA256 `95f8691ff4f437d0c2c3f8e93cad326f28f98796274d1257a89cef961b93d45e`。20 个 stage 全部通过，独立审计重新核验原始内容和计时。

### 正确性与训练连接

135 个量化/路由病例包含 45 个有限输入病例的三路逐字节比较，以及 12 个非有限输入拒绝后的恢复；共 147 项 byte comparison。覆盖 E2M1 ties、subnormal、极端 E8M0 scale、signed zero、padding、transpose、strided、broadcast、offset 及 lazy negative view。72 个 F/D/W recipe 病例与 6 个 SR control 共 306 个数值记录：126 项 exact、180 项普通 FP32 allclose，全部符合各自合同。

此外通过原生 baseline/tiled 各 46 项、C2 certified 69 项、旧 Torch 119 项、C2 certified Torch 16 项；`cuda_tiled` 和 `cuda_certified` 各执行五个配方的 40 步 synthetic decoder smoke。它们验证路由、梯度和参数更新，并不证明真实 LLM 预训练的长期收敛。

初次量化实验保留于 [candidate-cuda-rne](candidate-cuda-rne/)：前 44 个量化病例通过，随后旧 Torch 对照在 lazy negative view 的 `x.new_tensor` 触发 DispatchKey 错误；性能阶段未执行。[单独诊断](candidate-cuda-rne/lazy-negative-diagnostic.json)确认 native wrapper 和原输入 `record_stream` 能工作。修复仅让**旧 Torch 对照**解析 lazy view，native wrapper/direct 仍接收原 view，NumPy 字节 oracle 不变。v2 源归档逐文件差异只有 checker；没有修改 CUDA kernel 或 `torch_ops.py` 来隐藏错误。

### 两输入量化完整耗时

每次计入两输入的分配、有限性检查、量化、Python/ctypes 和同步。每进程 warmup 10、5×10，轮次交替后端；下表由五个原始同步 wall round 取中位数。初始及最后实际计时的 codes/scales 全量对照 NumPy；实际 FP32 输入前后 SHA 一致。

| 两个输入的形状 | Torch ms | CUDA RNE ms | 速度比 |
|---|---:|---:|---:|
| 1024² | 1.8438614 | **0.2972383** | **6.2033×** |
| 4096² | 11.9721700 | **0.6770435** | **17.6830×** |

量化没有使用虚构的 GEMM FLOP 计数，故只报告 latency/speedup。这是单进程五轮测量，轮次不是独立样本。

### 量化＋GEMM 完整 Torch 调用

每个 shape 的进程顺序为 `torch,cuda_rne,cuda_rne,torch`，每进程 warmup 3、5×10；汇总每后端全部十个 wall round。分配、输入验证和 Python 调用都在内。`cuda_certified` 此时每次创建并同步销毁 native context，开销也在内。

| 形状 | GEMM 后端 | Torch 量化 ms / TFLOPS | CUDA RNE ms / TFLOPS |
|---|---|---:|---:|
| 1024³ | `cuda_tiled` | 1.9979388 / 1.07485 | **0.3924380 / 5.47216** |
| 1024³ | `cuda_certified`，每次创建销毁 | 4.1103804 / 0.52245 | **2.7796056 / 0.77259** |
| 4096³ | `cuda_tiled` | 16.8077350 / 8.17713 | **5.4767146 / 25.09515** |
| 4096³ | `cuda_certified`，每次创建销毁 | 16.0436649 / 8.56656 | **5.4510417 / 25.21334** |

这些输出首次和最后实际计时均全量核对。此旧完整 Torch harness 记录同 seed/config 和每进程 resident inputs 复用，没有保存跨进程 FP32 输入字节指纹；独立量化 harness 的双输入 SHA 覆盖不能替代这一限制。

量化优化后，上下文生命周期成为值得单独测量的成本。这里的 25.21 TFLOPS 与原生 C2 的 75–89 TFLOPS 使用不同计时起点，不能相互替代，也不能把跨实验时长差直接当作实测分项。

## 7. 准备缓存与显式 workspace：实际对照结果

### C2 raw preparation cache：不足 1% 的观测差异，默认仍关闭

冻结 commit 为 `c581389551be2233ba23404fa72c471cc4c6f961`；[独立审计](candidate-c2-prepare-cache/independent-audit.md)已核验 30 个源文件、108 个结果文件、同源 OFF/ON 编译产物及全部 26 个通过的阶段。[结果包](candidate-c2-prepare-cache/results.tar.gz)为 150,676 B，SHA256 `9bea8409d639e984fa7834a16681312f23b14158b4c6f568796165f7bda0c797`。

候选缓存每个 MX32 的 16-byte raw data 与 4-byte scale/energy，减少第二次 global read 与 64-bit energy 工作；K=0、K>8192 继续原准备路径，证书条件不变。实际新旧准备内核及独立 CPU oracle 检查器通过 **327 项**，比较 1,079,840 个 half、1,635 个 16-byte RowData、24 个 FP32 输出，全部零 mismatch；guard 和输入前后不变。OFF/ON 各自还通过 baseline 46、tiled 46、certified 69 及 CUDA RNE 135 项检查。

每个形状按 `OFF,ON,ON,OFF` 执行，每进程 warmup 100、5×10；同配置合并全部十个同步 wall round。范围为完整 packed codes/scales → FP32，不包括量化和 context 创建。

| 形状及输入 | OFF ms / TFLOPS | ON ms / TFLOPS | ON 吞吐观测差异 |
|---|---:|---:|---:|
| 2048³，block scales | 0.3134183 / 54.8145 | 0.3122342 / 55.0224 | +0.379% |
| 4096³，block scales | 1.8202881 / 75.5040 | 1.8053371 / 76.1292 | +0.828% |
| 8192³，unit scale | 12.3118048 / 89.3055 | 12.2251716 / 89.9383 | +0.709% |

不足 1% 的差异尚不建立稳定提速结论，默认保持 OFF。12 次 benchmark 记录 36 个 NVML 快照，其中 24 个 benchmark 边界快照的 SM 时钟按形状为 1530、1522–1530、1327–1522 MHz；显存均 877 MHz，报告功率上限 300 W。源码采集位置是全部六个 backend 预热之后，以及全部 backend 计时和 CPU 输出验证之后，不能代表单个 C2 计时区间的连续频率。早期派生审计漏提取这些字段，现已纠正；旧版审计作为 `independent-audit-v1.*` 保留，原始数据与性能结论未改。

### 显式 CertifiedWorkspace：减少重复创建与同步销毁

冻结 commit 为 `f74de5be95687d75758c1f636ffbd00d6b63d161`，仅新增三个 Python 模块，沿用上述 **cache OFF** 动态库以单独测量生命周期复用。[独立审计](candidate-workspace/audit.md)核验了 33 个源文件、58 个结果文件、实际输入与输出、计时范围和最终提交次数，10 个阶段全部通过。[结果包](candidate-workspace/results.tar.gz)为 111,196 B，SHA256 `ac5998366a536800bac705eb03c185e01b089b0e7b24b29d84aa8e5aa2b65c30`。

一个固定 shape/device/library 的 workspace 由同一 owner thread 显式创建、复用和关闭；同 stream 可以异步排队，busy 状态不允许直接切换到其他 stream。所有输入输出 tensor 在 native 提交前 `record_stream`，CUDA Graph capture 明确拒绝，部分 enqueue 后的异常会完成同步清理。没有全局 context cache，也没有把显式 workspace 自动接入 autograd/MXLinear。

11 个实卡案例包含 16 次全输出比较、1,455 个比较值和 96 个 canonical `+0`。覆盖线程约束、busy stream 切换拒绝、排队 tensor 生命周期、提交后异常清理、capture 拒绝与恢复、logical K=33 padding，以及相同 GPU storage 修改后 accepted→跨度回退→NaN 回退的证书刷新；全部通过。单卡环境没有执行第二块 CUDA device 的切换案例。

每个形状按 `per_call,workspace,workspace,per_call` 执行，每进程 warmup 10、5×10；每生命周期合并两个进程全部十个 wall round。FP32 输入在四个进程间逐字节 SHA 一致，前后未变。每次调用包含**两个输入的 CUDA RNE 量化、有限性及 packed 验证、分配、Python/ctypes、prepare、证书、GEMM、postscale 和条件回退调度**。

| 形状 | 每次创建销毁 ms / useful TFLOPS | 显式复用 ms / useful TFLOPS | 重复调用速度比 |
|---|---:|---:|---:|
| 1024³ | 3.7122702 / 0.57848 | **0.4715297 / 4.55429** | **7.87282×** |
| 4096³ | 6.0233607 / 22.81765 | **2.4337259 / 56.47265** | **2.47495×** |

两种路径的首次与最后实际 timed 输出均全量逐位核对通过，且有限、零规范为 `+0`。四个复用进程最后的实际证书计数均为 61＝1 初始验证＋10 预热＋50 timed，没有额外运行一次 GEMM 替代最后计时结果。per-call 的最后 context 已销毁，因此其最终诊断字段显式为空；最后实际输出仍经过检查。

复用路径把**一次性创建和关闭单独测量**，没有计入上述重复调用延迟；per-call 路径则每次承担创建和同步销毁成本。完整冷成本如下：

| 形状／ABBA 位置 | 创建 ms | 关闭 ms | 合计 ms |
|---|---:|---:|---:|
| 1024³／02 | 22.755462 | 9.893191 | 32.648653 |
| 1024³／03 | 29.044370 | 0.784066 | 29.828436 |
| 4096³／02 | 29.643057 | 2.384289 | 32.027346 |
| 4096³／03 | 25.250695 | 1.237783 | 26.488478 |

一次性成本为 26.49–32.65 ms，需通过实际复用次数摊销。原报告中除以 61 的描述性摊销包含验证和预热，不能解释成训练 step 或 50 次 timed 操作的均摊。每种生命周期只有两个独立进程，rounds 存在相关性；per-call 进程中位数有漂移，workspace benchmark 没有连续 NVML 记录。这里建立的是长期持有显式 workspace 的重复算子调用收益，**尚未建立完整 LLM 训练 step 的加速或收敛结论**，也不能与第 6 节不同测量窗口的绝对时长直接拼成速度比。

## 8. 最后诊断：memcheck 通过，硬件计数器权限受限

[第一次诊断归档](post-workspace-diagnostics/attempt-a/results.tar.gz)使用冻结 C2/cache/workspace 源和二进制，没有重新编译。安装版为 Nsight Compute 2025.1.1，属于 NVIDIA [明确支持 GV100 的版本](https://developer.nvidia.com/tools-overview/nsight-compute/get-started-2025_1)。全量 GV100 metric catalog 实际返回 39,912,522 字节，超过脚本的 32 MiB 检查上限，导致 catalog stage 失败、profile 尚未执行。该次失败原样保留。

同次 **Compute Sanitizer memcheck 实际执行成功**：运行原来的 327-case 准备内核 microchecker，`--leak-check full`、`--error-exitcode 99`，实际日志为 `ERROR SUMMARY: 0 errors`，原数值及 guard 检查也全部通过。该结论只覆盖实际执行的 microchecker 范围，不等于对所有训练代码做过 sanitizer 检查。

[第二次诊断归档](post-workspace-diagnostics/attempt-b/results.tar.gz)只修复查询范围：根据实际安装版 help，用 `suffix --metrics` 查询原定九个 base metrics，保留大小上限；实际输出 188,857 字节，九个指标都存在。它重新核验并复用第一次成功的 memcheck 原始归档，**没有再次启动 sanitizer**。

随后实际启动的 NCU profile 返回 exit 1，原日志明确为 **`ERR_NVGPUCTRPERM`**：当前用户无权读取该 GPU 的硬件性能计数器。因此最终状态为 `completed_with_blocked_diagnostics`，没有 Tensor Core 利用率、occupancy、DRAM 吞吐或 stall 的有效计数器结果，不能把缺失值写成零。没有修改驱动权限、GPU 时钟或系统配置。

被采集程序本身仍完成了 C2 工作；独立核验的两次全量比较各含 16,777,216 个输出，均零位差、有限、零规范为 `+0`，实际提交次数为 1→11→12，六份输入字节哈希前后不变。profile 范围为预热后的一次 4096³ packed-input C2 pipeline，量化、context 创建及正确性 D2H 在范围外。它的输出检查与 profiler 计数器采集是两个独立状态；不能因为 GPU probe 通过就写成 profile 成功，也不能把 profiling 进程的 wall time 当作算子性能。源码、实际命令、失败日志及完整来源链已经由 [attempt-a 独立审计](post-workspace-diagnostics/attempt-a/audit.md)和 [attempt-b 独立审计](post-workspace-diagnostics/attempt-b/audit.md)核验；审计通过表示证据一致，不会把原始 blocked 状态改成 passed。

## 9. 硬件、文献与性能上限

[NVIDIA V100 datasheet](https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf)列出 PCIe 112、SXM2 125、V100S PCIe 130 Tensor TFLOPS。本次是 SXM2；125 是硬件参照，尚未测得 125 TFLOPS 的完整 MXFP4 算子。

[Volta tuning guide](https://docs.nvidia.com/cuda/archive/12.8.1/volta-tuning-guide/index.html)规定 Tensor Core 为 FP16 矩阵输入、可进行 FP32 累积。MXFP4 在这里是压缩存储和软件精确解码，不具备现代 GPU 原生 FP4 的硬件吞吐倍数。64K registers/SM、96 KiB shared/SM、独立线程调度与同步要求共同约束内核；静态减少寄存器或提前加载并不自动带来更高吞吐。

[Jia 等的 Volta 微基准研究](https://arxiv.org/abs/1804.06826)采用微基准与反汇编研究硬件。本轮对应做法是实际 SASS、完整数值检查及端到端对照；C9 的负结果也说明源码直觉需要实测验证。[Marlin 作者实现](https://github.com/IST-DASLab/marlin)提供向量加载、减少地址计算、布局与软件流水的设计参考，但它是 FP16×INT4 推理，要求 compute capability≥8.0；其 4× headline 不能移用于 V100 的 MXFP4 训练。

[CUDA 12.8.1 cuBLAS 文档](https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#cublasgemmalgo-t)支持本轮 SM70 算法筛选。当前证据使下一步重点落在准备、量化和上下文开销；单独 GEMM 的 99.28 TFLOPS 已与完整算子 89.29 TFLOPS 明确分开。真实 LLM 的 F/D/W 分布、证书接受率、长程 loss 和训练 step 吞吐仍需独立实验。

训练配方还需要单独验证。[Tseng、Yu、Park，AISTATS 2025](https://proceedings.mlr.press/v258/tseng25a.html)研究 stochastic rounding 与随机 Hadamard 对量化梯度方差的控制，并在最高 6.7B 的 GPT 训练中报告较小质量损失；文中吞吐估计依赖支持低精度的硬件，不能转写成 V100 实测。[2026 年原生 FP4 预训练研究 v4](https://arxiv.org/html/2605.09825v4)在 MI355X、Llama 3.1–8B/C4 的特定设置中发现 Wgrad 更敏感，确定性 Hadamard 能稳定其完整流程，而所测随机方法没有稳定同一流程。模型、硬件与协议不同，不能把两个结果合并成普适配方。本仓库继续保留 F/D/W 分别启用及 SR 对照；目前的 40 步小模型检查没有实现或复现上述论文的长期训练结论。
