# V100 MXFP4：融合解码与整 K HMMA 候选

后续硬件与文献调研、C8向量加载实验，以及原Crater等待进程的恢复记录见 [下一轮研究](../2026-11-24-hardware-study/README.md)。本页保留V1–V5及02:37发布时的历史证据。

本轮延续 [此前 50.95 TFLOPS 的证书路径](../2026-11-24-performance/README.md)，在同一数值合同下减少全局 FP16 工作区和调用阶段。**目前保留 C6 作为融合候选：1536³ 的 V4 ABBA 比较中，相对 C4 的完整 pipeline 吞吐提高 30.85%；进一步扩大 warp tile 的 C7 在 V5 ABBA 中比 C6 低 5.23%，保留为负面结果。** 两次比较发生在不同设备负载窗口，不能用其绝对吞吐计算跨轮加速比。默认仍为 tiled variant 1／certified variant 2；尚无同形状、同条件 C2 对照，也未达到 125 TFLOPS。日期按 Asia/Shanghai 归档；驱动日志记录 UTC，NVML CSV 保留设备本地时间。

## 1. 实现变化

[融合内核](../../../../tools/mxfp4_training/csrc/mxfp4_gemm_fused.cu) 共用原有整数精度证书。通过证书后，将 packed E2M1 与 E8M0 解码、归一化到共享内存，跨整 K 使用 FP16 输入／FP32 累加的 SM70 HMMA，在最终写回时施加行对尺度并规范化零。拒绝时整个融合 CTA 在输入访问与 barrier 前一致返回，原 exact tiled 后端计算结果。

| certified variant | CTA 输出 tile | BK | 线程数 | warp 输出 tile | 每线程 FP32 累加器 | 静态共享内存 |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 64×64 | 64 | 128 | 32×32 | 32 | 17,536 B |
| 4 | 128×128 | 32 | 256 | 32×64 | 64 | 18,688 B |
| 5 | 128×128 | 64 | 256 | 32×64 | 64 | 35,072 B |
| 6：下一 MX32 原始数据预取 | 128×128 | 32 | 256 | 32×64 | 64 | 18,688 B |
| 7：更大 warp tile 的预取候选 | 128×128 | 32 | 128 | 64×64 | 128 | 18,688 B |

原 cuBLAS variants 0／1／2 物化两份 normalized FP16 输入，并有 4 MiB 显式 cuBLAS workspace。融合 variants 3–7 的显式设备工作区是 `17*(M+N)+72` 字节；2048³ 时为 **69,704 B**，原 variant 2 为 **21,037,128 B**。输入、输出、驱动、主机诊断元数据和 SM 共享内存不包含在这些数字中。这是 native context 的分配差异，不是完整训练峰值显存结论。

C6／C7 使用独立预取 kernel，原 C3–5 路径保留。每个 stage 将当前块解码至共享内存、同步，再把下一块 packed 数据与 scale 读入寄存器，然后计算当前 HMMA，最后同步后重用共享内存。每线程暂存原始数据分别为 16／32 个 uint 寄存器。K=0 和最后一个 stage 均不会读取不存在的下一块；非零判断仍先于 scale delta 计算。

融合版还在 `Kp<=32768` 时缓存每个 MX32 的原始整数能量与 scale，减少第二遍 packed 全局读取和重复能量移位；2048 的 K 只需每 CTA 256 B 动态共享内存。更长 K 继续使用原来的元数据准备路径。每次调用均重新读取当前输入并计算证书，未跨调用缓存准入结果。

完整推导见 [整数证书及融合路径证明](../../../../docs/research/performance/sm70-packing/mxfp4-certified-fastpath.md)。能量、跨度、NaN 与尺度范围门槛均未放宽；融合候选同样保留任意 M/N 尾部、BK64 的 MX32 尾块、奇字节输入、仅 float 对齐的输出和 K=0。

## 2. 编译与指令证据

初版 `iterations/v1/` 保留实际源码归档、驱动脚本、四个变体的编译记录和未执行原因。CUDA 12.9 的 variants 2／3／4／5 均完成共享库、三个 correctness harness 和完整 benchmark 构建。

`iterations/v1/resource-summary.json` 从实际 ptxas 日志提取 12 个融合 kernel 特化：**90–112 个寄存器／线程，全部 0 spill stores、0 spill loads、0 stack bytes**。常见 int32、无 M/N tail 的 64/BK64、128/BK32、128/BK64 路径分别为 91、97、105 个寄存器。`sass-summary.json` 绑定共享库 SHA256，保留 cuobjdump 的 HMMA 指令计数和摘录；这是实际编译／反汇编证据，不代表 GPU 执行或实测速度。

后续 C6 的 [SASS 分析](verified/v4/prefetch-c6-sass-analysis.json)、[完整 int32 无尾部特化反汇编](verified/v4/prefetch-c6-int32-no-tail.sass.txt) 与 [产物来源](verified/v4/sass-provenance.json) 检查了稳态循环：当前 shared store 后的 barrier 在 `0x3110`，下一块的对齐全局加载位于 `0x3350–0x38f0`，当前块 HMMA 位于 `0x3960–0x4c00`，共享内存重用 barrier 位于 `0x4c20`。原始寄存器 `R105–R120` 穿过当前计算及回跳，供下一轮消费；最后一块有跳过预取的分支。这证明下一阶段加载确实排在当前 HMMA 前，且并非只有循环前的首次加载。它不测量实际重叠时间或 stall，也不能把全部实测收益唯一归因于预取；该反汇编结论限定为所检查的 C6 int32 无尾部特化。

最新 [V5 ptxas 原始日志](verified/v5/build-c7/shared_library.log) 的资源记录如下；数字为每线程，shared 均为 18,688 B：

| variant／索引特化 | 寄存器 | stack bytes | spill stores bytes | spill loads bytes |
|---|---:|---:|---:|---:|
| C6 int32，无尾部／有尾部 | 128／128 | 0／0 | 0／0 | 0／0 |
| C7 int32，无尾部／有尾部 | 251／243 | 0／0 | 0／0 | 0／0 |
| C6 int64，无尾部／有尾部 | 166／168 | 0／0 | 0／0 | 0／0 |
| C7 int64，无尾部／有尾部 | 255／255 | 152／112 | 168／132 | 168／216 |

**本次 1536³ 性能比较使用 int32、无尾部特化，C7 在该路径没有 spill。** int64 spill 是单独的宽索引资源限制，不能用来解释本次 C7 的实测退步。寄存器数增加是编译事实；其对吞吐的具体影响仍需独立 profiler 证据。

## 3. 有限显存上的验证安排

本轮现有 Tesla V100-PCIE-32GB 在 CUDA 初始化后实际只剩 **65,208,320 B**。初版全部正确性命令按普通 64 MiB 最低 reserve 拒绝运行，记录明确为 `gpu_executed=false`、0 completed cases；这些失败没有改写成通过。未停止既有服务、修改时钟或功耗。

进一步逐 fixture 核算后，完整融合 correctness suite 的最大显式设备分配是 tall case `65537×3×32` 的 **3,014,876 B**。因此增加只对 certified>=3 有效的显式 `--small-memory-correctness --reserve-mib 48`：保留至少 48 MiB，另设 **8 MiB 显式分配上限**，禁止 benchmark。普通模式仍为默认 128 MiB／最低 64 MiB；原完整性能程序的显存门槛也未改变。

对观测到的空闲量，扣除上述 peak 后仍有 62,193,444 B，超过 48 MiB 保留量 11,861,796 B。驱动加载和分配粒度仍以实际运行的 `cudaMemGetInfo` 为准，预算本身不是运行通过证明。

带块能量缓存的 CUDA 源码在 `verified/v2/` 重新编译；后续预取候选在 V4／V5 再构建验证。**certified variants 3–7 均在真实 V100 上通过各自完整检查：每套 69 次调用、405,930 个输出、0 mismatch；23 次接受、46 次精确回退，另有 2,950 个接受路径 canonical `+0` 位模式检查。** 计数为普通 46 cases／175,429 values，加 20 个 certified fixtures／230,456 values，再加同一 context 的 3 次调用／45 values。

覆盖 E2M1 码值、E8M0 scale、subnormal 舍入、相反符号溢出块、整数能量及 lattice 边界、scale span 12／13、scale 255 与零码组合、低尺度零块、跨 CTA／BK64 尾部、不同 row bases、奇字节输入、仅 4 B 对齐的输出、K=0、K=262144 的非缓存准备路径，以及同一 context 的接受→NaN 拒绝→再接受。

原始日志分别见 [c3](verified/v2/build-c3/correctness.jsonl)、[c4](verified/v2/build-c4/correctness.jsonl)、[c5](verified/v2/build-c5/correctness.jsonl)。[汇总](verified/v2/correctness-summary.json) 从实际日志逐条计算；构建记录绑定源文件与产物哈希。V2 驱动的首次启动曾因目录替换错误在任何编译／kernel 前退出，错误日志也保留；修正路径后才进行了上述真实运行。

后续完整套件见 V4 [C4](verified/v4/correctness-c4.jsonl)／[C6](verified/v4/correctness-c6.jsonl) 和 V5 [C6](verified/v5/correctness-c6.jsonl)／[C7](verified/v5/correctness-c7.jsonl)。上述 69／405,930／2,950 是每套计数，不是把多个版本合并后的总数。

## 4. 主机验证

`host/` 将算法、编排和 GPU 执行证据分开记录：

- `fused-host-checks.json`：**3,407,872 个 normalized-half 位模式比较，0 mismatch**；还检查三种 tile 的输出恰覆盖一次、float2 相邻列关系与共享内存银行映射。脚本为 `fused_host_checks.py`，明确标注 GPU 未执行。
- `comparison_stdout.json` 与 `check_energy_cache.py`：所有 65,536 个四-nibble 模式的能量对照，以及 **6,798 个 row case** 的原分组、缓存块聚合和独立大整数参考比较，覆盖所有 scale、跨度、饱和边界、零／NaN、缓存 K 边界及长 K 回退。来源哈希见 `provenance.json`。
- `payload-host-simulation.json`：**10 个完全 mock 的 host 编排场景**，覆盖正常顺序、缺工具链／Torch、构建／正确性失败、超时和 deadline 用尽；不冒充任何 Crater／CUDA 运行。
- `harness-host-checks.json`：**146 项 host 检查**，覆盖小显存显式 opt-in、参数顺序、分配／reserve 精确边界、packed-only 预算，以及共享 NaN／Inf 未写孔的故障注入。

### 1536³ 初轮完整 pipeline 测量

[独立 packed-only benchmark](../../../../tools/mxfp4_training/csrc/bench_mxfp4_fused.cu) 复用原输入生成器与 oracle，但不创建普通 cuBLAS 对照句柄或 FP16 缓冲区。1536³ 的显式设备分配是 **11,996,232 B**，显式 cap 12 MiB／reserve 48 MiB。这里移除的是比较程序的额外缓冲区；被测 certified pipeline 仍完整包含每次准备、证书、融合 GEMM 和条件 fallback 调度。

实际 [c3](verified/v3/build-c3/packed-1536.jsonl)、[c4](verified/v3/build-c4/packed-1536.jsonl)、[c5](verified/v3/build-c5/packed-1536.jsonl) 使用同一输入校验值 `8729bc8ef4895a63`、seed 20261123、3 warmup、5 rounds、每轮 10 iterations；每轮轮换三个后端顺序。统一以**同步 wall median**比较，同时保留 CUDA event 数据。

| certified variant | 同轮 tiled wall TFLOPS | fused total wall ms | fused total wall TFLOPS | fused total event TFLOPS |
|---|---:|---:|---:|---:|
| 3：64/BK64 | 8.7508 | 1.2238593 | 5.9221 | 7.3529 |
| 4：128/BK32 | 8.8389 | 0.7772003 | 9.3255 | 13.2248 |
| 5：128/BK64 | 8.8479 | 0.7968124 | 9.0959 | 12.8876 |

三组证书均通过；每个候选初始和最后实际 timed 输出的 **2,359,296 个值全部与 baseline 精确一致**，另有 CPU oracle 采样。程序在计时外 poison 输出并同步，检查初始 baseline 全量有限，避免共享输出让 no-op 或未写孔假通过。编译记录与前述完整 correctness suite 的 CUDA／helper 源码哈希相同，驱动在运行前验证匹配。

**这些是非独占设备上的本轮实测，不能当作 kernel 的独占上限，也不能与不同形状／时段的旧 2048³ C2 数字直接计算加速比。** 该程序明确记录 `c2_or_cublas_comparison_performed=false`。128 tile 在本轮胜过 64 tile；BK64 没有胜过 BK32。结果不足以替换现有默认 certified2，亦未达到 125 TFLOPS。

### V4：C4 → C6 → C6 → C4

V4 在 UTC 2026-11-23 18:06:26–18:07:49 完成构建、验证与 ABBA 驱动流程。四次 benchmark 使用相同 1536³ 输入、seed、fingerprint、warmup 与迭代设置。将每个版本两次进程运行中的全部五个 wall round 合并，再取 **10 个 round 的中位数**；未丢弃任何 round，也不把同一进程内的 round 当作独立实验。

| variant | pooled wall median ms | 完整 pipeline wall TFLOPS | pooled event median ms | event TFLOPS |
|---|---:|---:|---:|---:|
| C4 | 0.28706705 | 25.247611 | 0.2859008074 | 25.350601 |
| C6 | 0.21938200 | 33.037156 | 0.2181119919 | 33.229522 |

C6 相对 C4 的 wall 吞吐增加 **30.8526%**，wall 延迟下降 **23.5781%**。吞吐统一按 `2*M*N*K / wall_seconds / 1e12` 计算。全部原始 rounds 与计算方法见 [V4 comparison-summary](verified/v4/comparison-summary.json)，原始顺序为 [00-C4](verified/v4/abba-00-c4-1536.jsonl)、[01-C6](verified/v4/abba-01-c6-1536.jsonl)、[02-C6](verified/v4/abba-02-c6-1536.jsonl)、[03-C4](verified/v4/abba-03-c4-1536.jsonl)。这是该窗口内两个完整候选的比较，SASS 支持预取实现存在，但没有完成能把全部收益分解到单一机制的实验。

### V5：C6 → C7 → C7 → C6，保留退步结果

V5 驱动流程在 UTC 2026-11-23 18:18:02–18:19:46 完成，沿用上述输入与计时方法。C7 将 warp tile 扩大为 64×64、每 CTA 减少到 128 线程。合并两次运行的全部 10 个 round 后：

| variant | pooled wall median ms | 完整 pipeline wall TFLOPS | pooled event median ms | event TFLOPS |
|---|---:|---:|---:|---:|
| C6 | 0.71883590 | 10.082631 | 0.4807168007 | 15.076979 |
| C7 | 0.75850490 | 9.555320 | 0.5202944040 | 13.930108 |

**C7 wall 吞吐下降 5.2299%，延迟增加 5.5185%，因此没有晋升为首选融合候选。** 两种百分比的分母不同，不能混用。C7 的较快 round `0.6886834 ms` 同样保留。计算、原始数组、正确性与 NVML 汇总见 [V5 comparison-summary](verified/v5/comparison-summary.json)；原始顺序为 [00-C6](verified/v5/abba-00-c6-1536.jsonl)、[01-C7](verified/v5/abba-01-c7-1536.jsonl)、[02-C7](verified/v5/abba-02-c7-1536.jsonl)、[03-C6](verified/v5/abba-03-c6-1536.jsonl)。

V4／V5 的每次 benchmark 均有证书接受、54 次已提交调用（1 次初始检查、3 次 warmup、50 次 timed）、baseline 全量 nonfinite=0、CPU oracle 256 个采样，以及每个后端最后一次实际 timed 输出的 2,359,296 个值全量比较、0 mismatch。各轮完整 pipeline 都重新准备 metadata 和证书；分配、输出 poison 及其随后的同步、D2H 验证、诊断查询均在计时区间之外；wall 计时包含被测调用结束的完成同步。1536³ 的 certified workspace 为 52,296 B，总显式分配为 11,996,232 B；实际 warmup／timing 后剩余 50,528,256 B，比 48 MiB reserve 高 196,608 B，显存 guard 均通过。

保留的 [V4 NVML](verified/v4/background-gpu-samples.csv) 中，GPU0 利用率 1–61%、GPU1 为 0%；[V5 NVML](verified/v5/background-gpu-samples.csv) 有 34 条记录、每卡 17 条，两卡利用率均为 99–100%，SM 时钟均为 1380 MHz。NVML 包含测量程序自身的活动，不足以分离外部干扰或确定退步原因。**V3、V4、V5 是不同的非独占负载窗口，全部保留；不能把 V4 的 C6 33.04 TFLOPS 与 V5 的 10.08 TFLOPS 当作代码回归比例。** 这些比较都未包含 C2／cuBLAS 控制组，也不是完整训练吞吐或收敛实验。

### V5 来源绑定

- 冻结 [源码归档](verified/v5/source.tar.gz) SHA256：`5a3010b8c2ed97e203d545453d7992af6e4e82e73fe6b671c4b2cdcecb1fb88e`。
- 远端收集归档 SHA256：`97e41d00562535a33250b9ac26ca284bde50ca4c88708edbd86ba3a00e28149f`；在提取前核验，未将派生汇总冒充归档中的原始文件。
- [最终源码绑定](host/final-source-bindings.json) 核对 V5 C6／C7 的 12 个 csrc 源文件哈希与当前工作树，并逐项绑定五类实际编译日志。V5 派生汇总同时记录输入 JSONL、NVML、manifest、ptxas 日志和源码归档的哈希；该汇总自身没有执行 GPU。

## 5. 排队后的同条件比较

已授权的 Crater 单 V100 作业 `sg-jincy26-261123-e5bb3` 继续排队，原框架检查与 smoke 保留。**2026-11-24 02:37:08（Asia/Shanghai）已向原 watcher 发布源码 commit `4d7ff3b95768ff0e4833a336931e3e80265c897d`**，见 [真实发布审计](queue/refresh-audit.json)、[发布配置](queue/refresh-config.json) 和 [源码 ready 记录](queue/source-ready.json)。helper 在核对进程身份、waiting 状态和输入尚未冻结后更新 archive／payload／ready，随后恢复同一 PID；未重建作业或重启 watcher。原每 60 秒轮询、最长等待 12 小时的限制不变，等待期限为当日 12:52:24（Asia/Shanghai）。

[发布后独立观察](queue/post-refresh-observation.json) 核对相同 PID／starttime、进程已恢复、新文件哈希和原 Crater 作业的 Pending／Inqueue 状态。作业实际执行的固定源码是上述 commit；随后记录发布事实的文档与证据提交不改变算子或 payload。发布成功不表示 GPU 阶段已经执行，框架检查、量化成本及 GPU smoke 仍待节点分配。

扩展 payload 在原流程后，对 certified2 控制组与 certified3–7 分别执行 2048³、4096³，统一 seed `20261123`、`random_block_scales`、3 warmup、5 rounds、每轮 10 iterations。每个融合候选必须先通过自己的完整 certified correctness。实际执行与完成情况以 watcher 和各 GPU 阶段原始记录为准。

主框架始终使用原 certified2 库；融合 C API 仍未接入 MXRecipe。新增阶段共用既有 2700 秒总 deadline，不能执行的阶段明确记 `blocked`，各候选构建日志、源码／产物哈希和实际命令单独保存。这里描述的是待执行流程，不是新增吞吐、框架正确性或预训练收敛结论。

## 6. 复现

```bash
python3 tools/mxfp4_training/build_cuda.py \
  --nvcc /usr/local/cuda-12.9/bin/nvcc \
  --output-dir /tmp/mxfp4-fused-new \
  --with-certified --with-benchmark --with-fused-benchmark \
  --tiled-variant 1 --certified-variant 6

/tmp/mxfp4-fused-new/check_mxfp4_certified \
  --correctness-only --small-memory-correctness --reserve-mib 48

/tmp/mxfp4-fused-new/bench_mxfp4_fused \
  --m 1536 --n 1536 --k 1536 --seed 20261123 \
  --mode random_block_scales --iters 10 --rounds 5 --warmup 3 \
  --max-mib 12 --reserve-mib 48
```

上述两条执行命令分别显式启用已核算的小显存 correctness profile 与 packed-only benchmark；每次仍须通过实际显存 guard。在显存充足的节点，可去掉 correctness 的小显存选项，并使用 `bench_mxfp4 --m 2048 --n 2048 --k 2048 --seed 20261123 --mode random_block_scales --iters 10 --rounds 5 --warmup 3` 运行原完整比较程序。它的普通 reserve 门槛未改变，也不能让缺少显存的训练任务绕过框架检查。要得到 C2 同条件加速比，仍需以相同输入、形状与计时约定另行构建并运行 certified2 控制组。
