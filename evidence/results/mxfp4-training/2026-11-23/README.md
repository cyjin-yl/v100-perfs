# MXFP4 训练基础：2026-11-23 实际验证记录

本轮在用户 `doescompute` 的 Tesla V100-PCIE-32GB 上执行了完整数值域 MXFP4 原生 GEMM；PyTorch 训练接口与 tiny decoder 对照运行于 CPU。代码与复跑命令见 [tools/mxfp4_training](../../../../tools/mxfp4_training/README.md)。

## 结果与证据范围

| 项目 | 实际结果 | 范围 |
|---|---|---|
| NumPy 独立参考与 PyTorch 接口 | **43/43 通过，0 skip** | CPU；含完整编码/尺度、全部八种 F/D/W 开关、surrogate 梯度与 FP32 AdamW 状态 |
| CUDA 12.9 / SM70 编译 | **通过**；64 registers、0 stack、0 spill | 实际编译共享库与独立 harness，不能据此推导速度 |
| V100 原生 kernel | **42 cases，137,792 values，0 mismatch** | 完整 E2M1 与全部 256×256 尺度字节配对、非方阵/尾块、异常数和非默认 stream |
| 五组因果 decoder smoke | **五组各 40 步通过** | CPU oracle、合成 token 数据、1 层/宽度32；不构成真实 LLM 收敛证据 |
| PyTorch CUDA 完整训练链路 | **未执行** | 当前两卡被既有 vLLM 服务占用；原生 CUDA context 后仅约144 MiB空闲 |
| cuBLAS/FP16 公平速度对照 | **未执行** | 无加速比、tokens/s 或训练显存节约结论 |
| 受限域 middle-only packing | **未整合进本内核** | 2026-11-22 的数学/编译结果保持原证据范围 |

硬件最终运行时间为 `2026-11-23T15:26:16Z`，即北京时间 **2026-11-23 23:26**。GPU capability 7.0，driver API 12.4，runtime 12.9。原生上下文建立后的实际空闲显存为 **151,191,552 bytes**；程序在每次分配前检查所需容量与128 MiB余量。结束后两张卡仍分别由原服务占用32044 MiB，本轮没有停止或调整服务。

原生检查包含120,506个有限结果、1,026个NaN、16,260个无穷结果；其中5,388个为次正规数。NaN/Inf来自刻意的全尺度边界用例，不是把非有限训练结果认定为成功。训练smoke单独要求loss、参数和梯度有限。

## CPU 合成训练对照

共同设置：1层因果 self-attention decoder，宽度32、4 heads、词表32、序列16、batch4；FP32 master parameters / AdamW states，learning rate 0.005，40步，固定seed 20261123。attention、LayerNorm、softmax和loss保持FP32。各组从相同参数开始，量化方案具有自己的初始loss。

| 线性层配方 | 初始验证 loss | 40 步后验证 loss |
|---|---:|---:|
| FP32 | 3.645811 | 0.062209 |
| MXFP4 Forward | 3.608196 | 0.071482 |
| MXFP4 Forward + Dgrad | 3.608196 | 0.073649 |
| MXFP4 Forward + Dgrad + Wgrad | 3.608196 | 0.076575 |
| 三条路径 MXFP4，梯度使用 SR | 3.608196 | 0.076672 |

这只证明该明确配方可连通前向、反向和优化器。数据是可预测的模32序列，非自然语言语料；不把loss下降、RNE/SR间的小差异或CPU wall time解释成真实预训练质量或性能。

## 文件

- `cpu-validation.json`：43项测试的环境、最终源码哈希与日志哈希；`cpu-tests.txt`：逐项结果。
- `cpu-pretrain-smoke.json`：五个实验臂、完整40步loss、参数/梯度状态与源码哈希。`gpu_executed=false`。
- `compile-evidence.json`：真实nvcc版本、完整编译命令、源码/共享库/harness哈希；`shared_library.log`、`standalone_harness.log`：原始ptxas输出。
- `hardware-run.json`：真实执行时间、GPU环境、returncode与汇总。
- `v100-correctness.jsonl`：42项原生GPU用例的逐项输出；`v100-correctness.stderr.txt`为空。
- `mxfp4-sm70.sass.txt`、`sass-inspection.json`：实际SM70反汇编与静态opcode计数，使用已安装的CUDA12.4 cuobjdump读取CUDA12.9生成的共享库；静态计数不等于动态周期。
- `gpu-after-check.txt`：测试进程退出后的GPU占用情况。
- `SHA256SUMS`：本目录证据文件的字节哈希（清单自身除外）。

共享库和可执行文件可由源码重新生成；仓库仅保存其哈希与编译/执行证据，不提交二进制。原始日志中的 `/tmp/v100-mxfp4-3cc0562b0106` 是此次隔离验证路径。

## 已明确保留的限制

PyTorch CUDA wrapper虽已实现current-stream启动和输入生命周期管理，但尚未进行完整GPU训练验收。当前实现保留原始FP32 activation，量化使用多个PyTorch操作并进行finite/padding同步检查，CUDA内核仍是无shared-memory tiling的正确性基线；不存在已验证的FP16加速或总体显存节约。

下一步在显存充足时先执行相同五组GPU训练验收，再分别测量量化、GEMM、反向和优化器的端到端成本；随后才在真实文本与更长训练中对比RNE/SR、Wgrad配方以及有域检查的packing优化。
