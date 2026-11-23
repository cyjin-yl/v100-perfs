# V100 MXFP4 训练基础算子

本目录提供 **完整 E2M1 数值域、MX32 块尺度的 SM70 GEMM、共享内存 tiled 实现、带精确证书与回退的 native cuBLAS／融合 HMMA 路径，以及可分别启用 Forward／Dgrad／Wgrad 的 PyTorch 线性层**。目标是为后续 LLM 预训练实验建立可核对的算术、反向传播和测量入口。

V100 执行的是 FP16 输入、FP32 累加的 Tensor Core 运算；MXFP4 由软件编码、解码和施加块尺度。当前版本没有原生 FP4 指令，也没有接入受限域的 middle-only packing 优化。下面的 native 性能来自已打包输入；PyTorch GPU 训练与高精度输入量化成本仍待实际运行验证。

## 1. 已验证到什么程度

基础正确性与 CPU smoke 保存在 [2026-11-23 实验目录](../../evidence/results/mxfp4-training/2026-11-23/)；native 默认选择见 [50.95 TFLOPS 性能证据](../../evidence/results/mxfp4-training/2026-11-24-performance/README.md)，新增融合解码、预取候选和 Crater 扩展流程见 [融合实验](../../evidence/results/mxfp4-training/2026-11-24-fused/README.md)。

| 项目 | 本轮结果 | 能支持的结论 |
|---|---|---|
| NumPy／PyTorch CPU 单元测试 | **43 项通过** | 编码、尺度、尾部、量化、独立矩阵参考及一阶反向接口通过对应检查 |
| CUDA 编译 | **CUDA 12.9 编译通过** | native 共享库、三种正确性 harness 和 cuBLAS benchmark 可构建；编译证据含命令、variant 与哈希 |
| 真实 V100 tiled，选用 variant 1 | **46 例、175,429 个输出、0 mismatch** | 完整数值域、极端尺度、尾部和非默认 stream 通过独立 ordered-MX32 oracle 检查 |
| 真实 V100 certified，variant 0／1／2 | **每个 variant 均为 67 次调用、372,132 个输出、0 mismatch** | 每组含 21 次证书准入、46 次精确回退、context 重用及 400 个 canonical `+0` 检查；当前选用 variant 2 |
| 真实 V100 fused certified，variants 3–7 | **每个 variant 均通过扩展后的 69 次调用、405,930 个输出，0 mismatch** | 每组 23 次接受、46 次精确回退及 2,950 个 canonical `+0` 检查；原整数证书未放宽 |
| Native packed-input 性能 | **2048³ 的 certified 全路径为 50.952 TFLOPS** | 这是下表同次运行的同步 wall 测量，尚未达到 125 TFLOPS 目标 |
| 融合路径预取优化 | **1536³ 配对 C4→C6→C6→C4 测量中，吞吐提升约 30.85%** | 完整 pipeline 的 pooled wall median 从 0.287067 ms 降到 0.219382 ms；尚无与 C2 的同形状同轮比较，默认仍为 C2 |
| CPU tiny causal decoder（历史 smoke） | **5 组配方各训练 40 步，全部完成** | CPU 端前向、反向、FP32 AdamW 更新已接通 |
| PyTorch CUDA wrapper | **119 项 GPU 检查已准备，尚未在 GPU 上运行** | `cuda`／`cuda_tiled`、量化与 F/D/W 梯度的框架级验证仍待完成 |
| PyTorch 量化＋GEMM 测量 | **脚本已准备，GPU 结果待执行** | 不能从 native packed-input 数字推算高精度输入端到端吞吐 |
| Crater 临时节点工作流 | **单 V100 job 已提交，有时限 waiter 正在运行；GPU 阶段仍待执行** | 60 秒轮询、最长等待 12 小时；排队状态不代表 GPU 检查或训练通过 |
| 完整 GPU 训练、真实语料预训练 | **未运行** | 尚无 LLM 收敛、训练吞吐、显存收益或加速比结论 |

最新 tiled 记录见 [46-case 输出](../../evidence/results/mxfp4-training/2026-11-24-performance/iterations/build-final-v1/correctness-v2.jsonl)；certified 的三组完整检查分别见 [variant 0](../../evidence/results/mxfp4-training/2026-11-24-performance/verified/build-perf-c0/correctness.jsonl)、[variant 1](../../evidence/results/mxfp4-training/2026-11-24-performance/verified/build-perf-c1/correctness.jsonl)、[variant 2](../../evidence/results/mxfp4-training/2026-11-24-performance/verified/build-perf-c2/correctness.jsonl)。早期基线的 42 例、137,792 个输出保留在原始实验目录中。

### 1.1 同次运行的 native 性能

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

certified total 与本次简单 dequant＋cuBLAS total 处于同一量级；这不能证明优于所有优化后的解码实现，也不能外推为训练加速。此轮采用单个共享 GPU 输出缓冲区，显式使用 `--reserve-mib 64 --max-mib 64` 适应有限空闲显存；默认 reserve 仍为 512 MiB。PyTorch GPU check／训练 smoke 需要至少 512 MiB free，Torch benchmark 需要至少 1 GiB free，因此这些框架级阶段仍未执行。

### 1.2 融合 HMMA 与预取实验

certified variants 3–7 共用原精度证书，把 packed 解码和归一化直接融合到 HMMA 共享内存加载中，移除全局 normalized FP16 输入和 cuBLAS workspace。2048³ 的 native context 显式设备工作区从 C2 的 **21,037,128 B** 降至 **69,704 B**；这不包含输入、输出与驱动分配，也不是完整训练峰值显存。

当前融合候选为 **C6：128×128 CTA、BK32、下一块 packed／scale 寄存器预取**。在同一 1536³ 输入、交错 C4/C6 的真实 V100 测量中，C6 两次运行分别为 33.0306／33.0390 wall TFLOPS，C4 为 25.2303／25.2637。全部初始及最后实际计时输出精确一致。较大的 64×64 warp tile 候选 C7 在另一组配对实验中比 C6 慢，64 位索引编译路径还出现 spill，未选用。

原始 JSONL、编译／SASS 证据、同驻负载采样及较慢的实验都保留在 [融合实验报告](../../evidence/results/mxfp4-training/2026-11-24-fused/README.md)。两张 GPU 上仍有既有服务；不能把不同时段或不同形状的吞吐直接算作加速比。**C6 的 1536³ 数据没有证明胜过 C2 的 2048³ 50.95 TFLOPS，默认 certified variant 2 保持不变。**

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

## 2. 文件与依赖

| 文件 | 用途 |
|---|---|
| [reference.py](reference.py) | 独立 NumPy 编码、量化、解码和逐块整数 GEMM oracle |
| [torch_ops.py](torch_ops.py) | PyTorch 量化、packed GEMM、C ABI 加载、一阶 surrogate backward、`MXLinear` |
| [csrc/mxfp4_gemm.cu](csrc/mxfp4_gemm.cu) | 完整 E2M1 × E2M1 的 SM70 Tensor Core GEMM |
| [csrc/mxfp4_gemm.h](csrc/mxfp4_gemm.h) | C ABI、内存、尺度和累计顺序契约 |
| [csrc/mxfp4_gemm_tiled.cu](csrc/mxfp4_gemm_tiled.cu) | 共享内存复用、PRMT 解码和数值门控的尺度路径；当前选用 variant 1 |
| [csrc/mxfp4_gemm_certified.cu](csrc/mxfp4_gemm_certified.cu)／[C API](csrc/mxfp4_gemm_certified.h) | GPU 精确证书、向量化 prepare／epilogue、normalized FP16 cuBLAS 与 exact tiled fallback；当前选用 variant 2 |
| [csrc/mxfp4_gemm_fused.cu](csrc/mxfp4_gemm_fused.cu) | variants 3–7 的共享内存归一化、SM70 HMMA 和最终缩放；融合候选选 C6，未改变全局默认 |
| [csrc/check_mxfp4.cu](csrc/check_mxfp4.cu) | 不依赖 PyTorch／cuBLAS 的真实 GPU 正确性 harness |
| [csrc/check_mxfp4_certified.cu](csrc/check_mxfp4_certified.cu) | 完整数值域、证书边界、拒绝回退和 context 重用检查 |
| [csrc/bench_mxfp4.cu](csrc/bench_mxfp4.cu) | packed-input baseline／tiled／certified 与 FP16 cuBLAS 分项及总成本比较 |
| [csrc/bench_mxfp4_fused.cu](csrc/bench_mxfp4_fused.cu) | 小显存 packed-only 比较，保留完整 certified pipeline；不含 C2／普通 cuBLAS 对照 |
| [build_cuda.py](build_cuda.py) | 编译共享库和 harness，并记录源码哈希、命令、日志和产物哈希 |
| [check_torch_cuda.py](check_torch_cuda.py) | 119 项框架级 GPU 检查：独立 NumPy 预期、所有 recipe 开关与非默认 stream |
| [bench_torch_cuda.py](bench_torch_cuda.py) | 通过外部 GPU check 后测量 Torch FP32、FP16、casts、量化与量化＋packed GEMM |
| [pretrain_smoke.py](pretrain_smoke.py) | 五组配方的合成 causal next-token 训练 smoke |
| [crater_queue_runner.py](crater_queue_runner.py)／[crater_remote_run.py](crater_remote_run.py) | 已授权 Crater job 的排队跟进与有时限的远端验证 payload；结果仍待执行 |
| [参考测试](../../tests/test_mxfp4_reference.py)／[PyTorch 测试](../../tests/test_mxfp4_torch.py) | 独立预期值、边界和反向传播检查 |

NumPy 参考路径只需要 NumPy；框架测试和 smoke 还需要 PyTorch。历史 CPU 验证环境为 Python 3.12.14、NumPy 2.3.5、PyTorch 2.5.1+cpu，这是一组已测版本，不是唯一支持版本。CUDA 构建需要能生成 SM70 代码的 CUDA 12.x 工具链；本轮使用 12.9。共享库不依赖 PyTorch C++ 头文件；启用 certified 或 native benchmark 需要 cuBLAS。GPU 框架运行还需要支持本机 V100 的 CUDA 版 PyTorch。

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
  --nvcc /usr/local/cuda-12.9/bin/nvcc \
  --output-dir /tmp/mxfp4-cuda-build-20261124 \
  --with-certified --with-benchmark \
  --tiled-variant 1 --certified-variant 2
```

按安装位置调整 `--nvcc`；也可省略它，使用 `CUDA_HOME`／`CUDA_PATH` 或 `PATH` 中的工具链。当前默认就是 tiled variant 1、certified variant 2；命令显式写出版本以便复现。输出包含 `libmxfp4_sm70.so`、`check_mxfp4`、`check_mxfp4_tiled`、`check_mxfp4_certified`、`bench_mxfp4`、编译日志和 `compile-evidence.json`。只需 baseline／tiled 共享库时可省略两个 `--with-*` 并加 `--library-only`。

构建使用 `--ftz=false`，以保留本契约要求的 FP32 次正规数。certified variants 0–2 使用 cuBLAS，3–7 为融合候选，编译记录和 benchmark 环境记录保留所选版本。C6 的复现命令见 [融合报告](../../evidence/results/mxfp4-training/2026-11-24-fused/README.md)；`--with-fused-benchmark` 只允许启用 certified 且 variant>=3。编译成功本身不证明 GPU 执行正确。**certified 目前只提供 native C API，没有 `MXRecipe`／`gemm_packed` 的 Torch backend；`cuda_tiled` 已有 Torch 接口。**

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

### 3.5 PyTorch GPU integration check：119 项已准备，尚未执行

```bash
export MXFP4_CUDA_LIBRARY=/tmp/mxfp4-cuda-build-20261124/libmxfp4_sm70.so
python -m tools.mxfp4_training.check_torch_cuda \
  --device cuda:0 \
  --output /tmp/mxfp4-torch-check-20261124.json
```

此检查覆盖 `cuda` 和 `cuda_tiled`、packed 特殊值、RNE／SR、所有 8 种 F/D/W 开关、尾部／非连续输入、原始 FP32 转置重新量化以及非默认 stream，并用独立 NumPy 算术产生预期值。必须为实际 SM70，context 建立后至少有 512 MiB free；缺 CUDA 或显存不足会记录 blocked，不会回退 CPU 或写成通过。

### 3.6 PyTorch 量化＋GEMM 成本：命令已准备，GPU 结果待执行

必须先在外部完成第 3.5 节，并保留完整通过的报告：

```bash
python -m tools.mxfp4_training.bench_torch_cuda \
  --device cuda:0 --m 1024 --n 1024 --k 1024 \
  --rounds 3 --iters 3 --warmup 1 \
  --check-report /tmp/mxfp4-torch-check-20261124.json \
  --output /tmp/mxfp4-torch-bench-20261124.json
```

脚本核对外部 check 的 source/library 哈希、Torch/CUDA build 与设备身份；源码或共享库改变后必须重跑 check。它要求实际 SM70 和至少 1 GiB free，并有保守临时显存预算。输入是固定种子的 GPU Gaussian FP32 张量。

六个 job 分别是原始 FP32 matmul、预转换 FP16 compute、FP32→FP16 casts＋matmul、两个输入的量化、量化＋`cuda`、量化＋`cuda_tiled`。量化总路径包含现有 validation 同步、临时张量与 packed/output 分配、Python／ctypes dispatch 和 GEMM；输入生成、初次 context 初始化与正确性复制在计时外。报告 event／wall 数据和前后显存，量化单项不报告 TFLOPS。

**普通 Torch FP16 matmul 输出仍为 FP16；没有隐藏的 `.float()` 计时转换，也不是 native benchmark 的 FP32-output cuBLAS primitive。** 所有初始及最终计时结果须有限，MX 两条 backend 全输出须 exact，最终结果还须与各自初始结果保持同 dtype 一致。对原 FP32 输出的误差只作背景记录；MX backend 的 exact gate 独立执行。此脚本没有测量 certified C API，也没有执行完整训练。

### 3.7 CPU causal decoder smoke：历史阶段已执行

```bash
python -m tools.mxfp4_training.pretrain_smoke \
  --backend reference --device cpu \
  --steps 40 --seed 20261123 --lr 0.005 \
  --output /tmp/mxfp4-smoke-cpu-20261123.json
```

脚本比较全 FP32、仅 Forward、Forward+Dgrad、Forward+Dgrad+Wgrad，以及上游梯度 SR 五组；逐步检查 loss 和梯度有限、参数实际更新、AdamW 状态保持 FP32，并保存训练曲线、recipe、版本和源码哈希。

### 3.8 GPU causal decoder smoke：命令已准备，尚未执行

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

先完成第 3.5 节的 GPU integration check。CUDA wrapper 仅接受计算能力 `(7,0)`，不会静默回退 CPU。smoke 在创建 context 后要求至少 512 MiB 空闲显存；该门槛是启动检查，不是对所有环境峰值占用的保证。现有共享 V100 的余量不足，所以上表只把 CPU 训练标为已接通。

### 3.9 Crater 排队执行入口：waiter 运行中，GPU 阶段待执行

[crater_queue_runner.py](crater_queue_runner.py) 跟进一个已授权、已存在的 Crater job；固定源码 archive 和 payload 通过 `--ready-json` 发布门禁后，才在分配到的节点上运行 [crater_remote_run.py](crater_remote_run.py)。runner 本身不提交、停止或删除平台 job。队列状态及后续结果入口见 [性能证据说明](../../evidence/results/mxfp4-training/2026-11-24-performance/README.md)。

payload 只探测和使用已安装的 CUDA 12.x、NumPy 与 CUDA 版 PyTorch，依次执行 native build/check、native benchmark、119 项 Torch check、量化成本 benchmark，以及两条 Torch backend 的训练 smoke。扩展流程随后对 C2 控制组与 C3–C7 逐个比较 2048³／4096³；每个融合候选须先通过自己的完整 correctness。各阶段共用原 2700 秒总 deadline，主框架库保持 C2。前置检查不通过或运行环境缺失时，相应阶段记录 blocked／failed；结果和哈希保存在独立目录与 `results.tar.gz` 中。队列发布状态与新增候选证据见 [融合报告](../../evidence/results/mxfp4-training/2026-11-24-fused/README.md)。

**单 V100 job 已提交，有时限 waiter 正在按 60 秒轮询、最长 12 小时等待；GPU check、benchmark 和训练阶段仍待执行。** runner 启动、节点分配或 SSH 成功均不能代替各阶段实际 GPU 证据；候选运行环境仍须由 payload 实际探测和验证。

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

GPU 用法是将 `backend` 设为 `cuda` 或 `cuda_tiled`，在构造时传 `device="cuda:0"`，并把 FP32 输入放在同一设备；先设置前述 `MXFP4_CUDA_LIBRARY`。这两个框架组合的 GPU 运行验证仍待完成。certified 尚未接入 Torch wrapper，不能把 native certified 的速度写作这个训练接口的速度。

### 5.2 Checkpoint 必须包含 recipe 和 RNG

`recipe` 是 Python 属性，**不会自动写入 `state_dict()`**。仅恢复参数会遗漏量化开关和舍入方式。单层示例：

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

当前已有的性能工作与剩余成本如下：

- 量化使用多个 PyTorch kernel 和与元素数量同阶的临时张量；有限值／padding 检查会产生主机同步，尚无融合量化或双方向生产 kernel。
- tiled 已实现共享内存 tile 复用、PRMT nibble 解码和受数值门控的尺度路径；当前选用 variant 1。原始 baseline 继续保留为完整算术对照。
- certified variant 2 使用向量化 prepare 与 epilogue，按 GPU 证书选择 cuBLAS 精确结果或 full-domain tiled 回退。为保持无主机分支同步，当前拒绝路径仍执行一遍安全的 normalized cuBLAS，再执行条件 fallback；该成本保留在总计时内。
- 融合 variants 3–7 在证书通过后直接从 packed 输入执行 HMMA；拒绝时融合 CTA 提前返回，随后 exact tiled 回退。C6 增加下一块数据预取；Kp<=32768 时准备阶段缓存块能量，减少第二次 packed 读取。默认仍为 C2，融合候选尚待与 C2 同条件完整比较。
- 证书是否通过取决于数据尺度分布和能量；合成 case 的通过不能推出所有训练张量均能走 cuBLAS 路径。实际训练数据的准入率、回退成本及准备阶段占比仍需测量。
- 此前 `|q|≤3` 等受限域的 middle-only packing 尚未接入，也没有该 packing 的残差修复路径。certified 的精确证书是独立方案，不是对这项受限 packing 的启用声明。
- master weights、保存供反向使用的 activation 和 optimizer states 均保持 FP32。packed 缓冲区的低位存储不等于训练峰值显存下降。
- 训练 smoke 中有逐参数有限值和更新检查；其墙钟时间不可作为正式吞吐比较。

目前已有可核对 scope 的 native FP16／cuBLAS 比较，结论限于已归档的 packed-input、shape、分布和共享 GPU 条件。完整训练加速与显存收益仍未建立；需要把高精度输入量化、方向布局生成、GEMM、反向、优化器、峰值显存和完整 step 时间一起测量，并使用相同数学目标与训练配置的基线。

## 7. 后续验收顺序

1. **完成 119 项 PyTorch CUDA wrapper 验证。** Crater 分配到有足够显存的节点后，对两条 Torch backend 检查非默认 stream、生命周期、独立 NumPy oracle 和三个训练矩阵方向。
2. **测量高精度输入成本并执行 GPU causal decoder smoke。** 先用 matching check report 解锁 Torch benchmark，再运行 `cuda`／`cuda_tiled` 五组训练配方；保留 CPU 历史对照，分别记录 native、wrapper 与训练是否真正执行。
3. **扩大已有 native 性能基线。** 使用更多代表性 M/N/K、真实训练张量和可控 GPU 负载，测量 certified 准入率、拒绝后的完整成本及与 FP16 cuBLAS 的差距；现有 2048³ 合成测量不能代表完整 LLM 训练。
4. **按端到端瓶颈继续优化。** 优先核对融合量化、双方向布局生产和准备阶段成本，再评估受限 packing；后者仍需严格的数据域／尺度／安全分段条件和完整域回退。旧研究见 [SM70 packing 报告](../../docs/research/performance/sm70-packing/index.md)。
5. **扩大训练配方实验。** 使用真实数据、明确模型与 token 预算，继续分开 Forward／Dgrad／Wgrad，比较 RNE、SR、尺度策略与 Hadamard 方案，记录 loss、梯度误差和训练稳定性。

可参考 [Training LLMs with MXFP4，AISTATS 2025](https://proceedings.mlr.press/v258/tseng25a.html) 的 SR／随机 Hadamard 反向配方，以及 [2026 原生 MXFP4 预训练研究 v4](https://arxiv.org/html/2605.09825v4) 对不同训练阶段和 Wgrad 的对照。两项工作的硬件与配方不同，不能把其速度或收敛结果直接迁移为 V100 本实现的结论。
