# 显式 certified workspace 原始归档独立审计

审计：**verified**；实际实验：**passed**。

本地审计没有执行 GPU、编译器、远端命令或归档代码，也没有修改原始证据。

## 来源与正确性

源 commit `f74de5be95687d75758c1f636ffbd00d6b63d161`，33 个源文件；相对已核验 cache 源仅增加三个 workspace Python 模块。
本轮复用 cache `off` 库，未重新编译；四个 ABBA 进程使用同一个实际库。
11 个独立检查案例、16 次全输出精确比较、1,455 个比较值通过；覆盖线程、stream、输入变更后的证书刷新、NaN回退、canonical +0、capture拒绝与异常清理。

## 完整两输入量化＋certified GEMM

每形状按per_call→workspace→workspace→per_call，逐进程1次初始验证、10次预热、5轮×10次计时。每种生命周期合并全部10个round。四个进程实际FP32输入SHA完全一致。

| 形状 | 生命周期 | pooled wall ms | useful TFLOPS |
|---|---|---:|---:|
| 1024³ | per_call | 3.712270199 | 0.578482581 |
| 1024³ | workspace | 0.471529702 | 4.554291359 |
| 4096³ | per_call | 6.023360652 | 22.817653037 |
| 4096³ | workspace | 2.433725900 | 56.472651026 |

workspace重复调用计时排除一次性创建与关闭；per_call每次创建、分配和同步销毁仍在计时内。两种路径都包含两输入量化、验证、分配、dispatch、prepare、证书、GEMM和最终缩放／回退。

workspace末次实际timed输出的证书计数为61，结果全量逐位匹配且零值均为+0；per_call最后context已销毁，其缺失诊断被明确保留。没有用重新执行GEMM替换最后计时结果。

## 一次性资源成本必须另计

每个workspace进程的首次创建（含初始ABI查询）与最终关闭实际观察如下。关闭前的最终结果检查已同步GPU；这些观察不代表任意环境的固定冷启动成本。

| 形状／ABBA位置 | 创建 ms | 关闭 ms | 总计 ms |
|---|---:|---:|---:|
| 1024³／02 | 22.755462 | 9.893191 | 32.648653 |
| 1024³／03 | 29.044370 | 0.784066 | 29.828436 |
| 4096³／02 | 29.643057 | 2.384289 | 32.027346 |
| 4096³／03 | 25.250695 | 1.237783 | 26.488478 |

本次总计范围为 **26.488478–32.648653 ms**，未包含于上表重复调用TFLOPS。原报告另保留除以61次完整操作的描述性摊销；61含初始验证和warmup，不能偷换为50次计时调用或训练step的平均成本。

per_call两个进程各自的wall median：

- 1024³：3.520449／4.270481 ms。
- 4096³：6.876227／5.957346 ms。

报告保留全部round并按既定ABBA顺序汇总；未锁定频率、没有连续NVML轨迹，不据此声称固定普遍倍率或统计显著性。

## 限制

- This API is explicit, single-shape, owner-thread reuse; autograd and MXLinear do not use it, and existing default interfaces are unchanged.
- Repeated-call timing includes both full quantizations, validation, allocations, Python/ctypes, preparation, certificate, GEMM and epilogue/fallback.
- Workspace creation and close are measured separately; per-call creation and synchronous destruction remain inside each timed operation.
- The separate amortization divisor 61 includes one initial validation, ten warmups and fifty timed operations, not sixty-one independent cases or training steps.
- Each lifecycle has two independent processes per size and ten correlated round observations; no confidence interval or statistical significance is established.
- The individual per-call process medians vary; ABBA reduces ordering bias but does not establish a fixed universal speedup or isolate every source of variance.
- Only the final actual timed GEMM output is fully validated per process; intermediate timed outputs are not individually copied back.
- Per-call final native contexts are destroyed before the timer returns, so no final certificate is available; the initial per-call diagnostic and final output checks are retained.
- The large Gaussian output oracle is the unchanged exact tiled path; independent NumPy checks are provided by the workspace and prerequisite suites, not full large-matrix CPU GEMM.
- CUDA event intervals include host submission gaps and wrapper synchronizations; they are not isolated kernel timings.
- This runner has no continuous NVML clock/load trace and does not assert exclusive device activity; selected cache build is held identical across both lifecycles.
- Only node-recorded ELF hashes are available locally, cross-bound to build manifests and before/after execution records.
- No real-corpus LLM pretraining, backward/autograd integration, optimizer step, training throughput or peak-memory benefit is established here.
