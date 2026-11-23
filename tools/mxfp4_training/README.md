# V100 MXFP4 训练基础算子

本目录提供 **完整 E2M1 数值域、MX32 块尺度的 SM70 GEMM 正确性基线，以及可分别启用 Forward／Dgrad／Wgrad 的 PyTorch 线性层**。目标是为后续 LLM 预训练实验建立可核对的算术、反向传播和测量入口。

V100 执行的是 FP16 输入、FP32 累加的 Tensor Core 运算；MXFP4 由软件编码、解码和施加块尺度。当前版本没有原生 FP4 指令，也没有接入受限域的 middle-only packing 优化。完整 E2M1 支持不代表相对 FP16 GEMM 已获得加速。

## 1. 已验证到什么程度

截至 2026-11-23，本轮记录见 [实验结果目录](../../evidence/results/mxfp4-training/2026-11-23/)。

| 项目 | 本轮结果 | 能支持的结论 |
|---|---|---|
| NumPy／PyTorch CPU 单元测试 | **43 项通过** | 编码、尺度、尾部、量化、独立矩阵参考及一阶反向接口通过对应检查 |
| CUDA 编译 | **CUDA 12.9 编译通过** | 共享库和独立 harness 可构建为 SM70 cubin／compute_70 PTX |
| 真实 V100 独立 native harness | **42 例、137,792 个输出、0 mismatch** | 在真实 SM70 上通过本轮完整编码、尺度、边界和非默认 stream 检查 |
| CPU tiny causal decoder | **5 组配方各训练 40 步，全部完成** | CPU 端前向、反向、FP32 AdamW 更新已接通 |
| PyTorch CUDA wrapper | **未在 GPU 上运行** | current-stream／生命周期逻辑已有源码检查，仍缺框架级 GPU 验证 |
| 完整 GPU 训练、真实语料预训练 | **未运行** | 尚无 LLM 收敛、训练吞吐、显存收益或加速比结论 |

本轮 V100 初始化 CUDA context 后仅剩约 **144 MiB** 空闲显存，因此只运行小型 native 正确性检查。已有 vLLM 服务保持运行。GPU 训练 smoke 要求 context 建立后至少有 512 MiB 可用显存，本轮没有执行它；不能将 native kernel 通过写成 PyTorch GPU 训练已经通过。

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
| [csrc/check_mxfp4.cu](csrc/check_mxfp4.cu) | 不依赖 PyTorch／cuBLAS 的真实 GPU 正确性 harness |
| [build_cuda.py](build_cuda.py) | 编译共享库和 harness，并记录源码哈希、命令、日志和产物哈希 |
| [pretrain_smoke.py](pretrain_smoke.py) | 五组配方的合成 causal next-token 训练 smoke |
| [参考测试](../../tests/test_mxfp4_reference.py)／[PyTorch 测试](../../tests/test_mxfp4_torch.py) | 独立预期值、边界和反向传播检查 |

NumPy 参考路径只需要 NumPy；框架测试和 smoke 还需要 PyTorch。本轮 CPU 验证环境为 Python 3.12.14、NumPy 2.3.5、PyTorch 2.5.1+cpu，这是一组已测版本，不是唯一支持版本。CUDA 构建需要能生成 SM70 代码的 CUDA 12.x 工具链；本轮使用 12.9。共享库不依赖 PyTorch C++ 头文件。GPU 框架运行还需要支持本机 V100 的 CUDA 版 PyTorch。

下面所有命令都从仓库根目录执行。`python` 应指向装有相应依赖的环境。为每次运行选用新的输出路径，保留旧证据；smoke 会拒绝覆盖已有报告。

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
  --output-dir /tmp/mxfp4-cuda-build-20261123
```

按安装位置调整 `--nvcc`；也可省略它，使用 `CUDA_HOME`／`CUDA_PATH` 或 `PATH` 中的工具链。输出包含 `libmxfp4_sm70.so`、`check_mxfp4`、各编译日志和 `compile-evidence.json`。仅需要共享库时加 `--library-only`。

构建使用 `--ftz=false`，以保留本契约要求的 FP32 次正规数。编译成功本身不证明 GPU 执行正确。

### 3.3 真实 V100 正确性检查

```bash
/tmp/mxfp4-cuda-build-20261123/check_mxfp4 \
  --device 0 --correctness-only \
  > /tmp/mxfp4-native-20261123.jsonl \
  2> /tmp/mxfp4-native-20261123.stderr
```

默认要求真实 V100，检查在非默认 CUDA stream 上执行。测试覆盖 16 × 16 个 E2M1 编码组合、两种非零 dot 下全部 65,536 个尺度字节组合、零 dot 的 NaN 传播、正负次正规数舍入、相反溢出块的 FP32 累计、非方阵／K 尾部及 `K=0`。每个案例分配前检查显存并保留 128 MiB 余量；程序不停止或调整现有服务。

`--allow-other-gpu` 仅用于明确标记的对照，结果中的 `actual_V100` 必须保留。可选的 `--benchmark` 只测预先打包输入的 kernel 时间，没有 cuBLAS／完整训练基线，输出的 `speedup` 保持空值；本轮报告不据此作加速结论。

### 3.4 CPU causal decoder smoke：本轮已执行

```bash
python -m tools.mxfp4_training.pretrain_smoke \
  --backend reference --device cpu \
  --steps 40 --seed 20261123 --lr 0.005 \
  --output /tmp/mxfp4-smoke-cpu-20261123.json
```

脚本比较全 FP32、仅 Forward、Forward+Dgrad、Forward+Dgrad+Wgrad，以及上游梯度 SR 五组；逐步检查 loss 和梯度有限、参数实际更新、AdamW 状态保持 FP32，并保存训练曲线、recipe、版本和源码哈希。

### 3.5 GPU causal decoder smoke：命令已准备，本轮未执行

```bash
export MXFP4_CUDA_LIBRARY=/tmp/mxfp4-cuda-build-20261123/libmxfp4_sm70.so
python -m tools.mxfp4_training.pretrain_smoke \
  --backend cuda --device cuda:0 \
  --steps 40 --seed 20261123 --lr 0.005 \
  --output /tmp/mxfp4-smoke-cuda-20261123.json
```

CUDA wrapper 仅接受计算能力 `(7,0)`，不会静默回退 CPU。脚本在创建 context 后要求至少 512 MiB 空闲显存；该门槛是启动检查，不是对所有环境峰值占用的保证。本轮约 144 MiB 余量不足，所以上表只把 CPU 训练标为已接通。

## 4. 编码、量化与计算契约

### 4.1 存储布局

设原始矩阵为 `[R,K]`，`Kp = ceil(K/32) × 32`：

| 字段 | dtype／形状 | 含义 |
|---|---|---|
| `codes` | `uint8[R,Kp/2]`，连续 | 每字节低 nibble 存偶数 K、高 nibble 存奇数 K |
| `scales` | `uint8[R,Kp/32]`，连续 | 每行每 32 个元素独立的 E8M0 尺度 |
| `logical_shape` | `(R,K)` | 不含 K padding 的形状 |

E2M1 的正数编码 `0…7` 对应 `+0, 0.5, 1, 1.5, 2, 3, 4, 6`，最高位是符号位，保留负零。K padding 必须是位模式为零的规范正零。Python 接口拒绝空矩阵；C ABI 另支持 `M/N=0` no-op 和 `Kp=0` 输出零。

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

尺度通过 `ldexp` 一次施加在整个 dot 上：若先把尺度乘积单独物化为 FP32，可能提前产生 Inf 或零，丢失本来可表示的结果。也不先把有尺度的元素解为 FP16，因为那会缩小有效动态范围。完整编码与尺度支持不意味着与量化前 FP32 GEMM 相同，量化误差仍然存在。

NumPy oracle 用独立整数矩阵乘法构造预期值；CUDA 使用已有 [SM70 fragment 映射](../../evidence/benchmarks/sm70-packing/study/simulation/round3/cuda/fragment_mapping.hpp)，在 FP16 Tensor Core 中计算无尺度整数 q 的块点积。

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

GPU 用法是将 `backend` 设为 `cuda`，在构造时传 `device="cuda:0"`，并把 FP32 输入放在同一设备；先设置前述 `MXFP4_CUDA_LIBRARY`。此组合的运行验证在本轮仍待完成。

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

## 6. Stream、内存和当前性能限制

C ABI 在调用者提供的 stream 上 enqueue，返回 CUDA 状态码，不自行同步。它要求连续、互不重叠、属于当前 CUDA device 的有效缓冲区；大小、设备和生命周期由调用者负责。PyTorch wrapper 在目标 device 的 current stream 上启动，并对所有输入／输出调用 `record_stream`。跨 stream 生产和消费仍需调用者遵守正常的 PyTorch stream 依赖规则。

当前实现优先建立正确性，性能限制具体包括：

- 量化使用多个 PyTorch kernel 和与元素数量同阶的临时张量；有限值／padding 检查会产生主机同步，尚无融合量化或双方向生产 kernel。
- CUDA GEMM 在输出 tile 间重复加载、解码输入，未做 shared-memory tiling、流水线或复用优化。
- 尚未把此前 `|q|≤3` 等受限域的 middle-only packing 接入本内核，也没有自动准入或残差修复路径。
- master weights、保存供反向使用的 activation 和 optimizer states 均保持 FP32。packed 缓冲区的低位存储不等于训练峰值显存下降。
- 训练 smoke 中有逐参数有限值和更新检查；其墙钟时间不可作为正式吞吐比较。

因此目前不提供相对 FP16／cuBLAS 的加速比，也不承诺训练显存收益。正式性能评估需要同时记录量化、方向布局生成、GEMM、反向、优化器、峰值显存和完整 step 时间，并使用相同数学目标与训练配置的基线。

## 7. 后续验收顺序

1. **完成 PyTorch CUDA wrapper 验证。** 在有足够可用显存时检查非默认 stream、输入生命周期、FP32输出与 NumPy oracle，以及三个训练矩阵方向。
2. **执行 GPU causal decoder smoke。** 保留 CPU 五组配置作为对照，分别记录 native kernel、wrapper 和训练是否真正执行。
3. **建立代表性 LLM 线性层的性能基线。** 按实际 M/N/K 测量 FP16 cuBLAS、当前通用 MXFP4 路径和量化准备成本，再确定融合／tile 优化的优先级。
4. **验证受限 packing 的实际收益。** 只在检查过的数据域、同一尺度产品和安全分段内启用；其余输入走完整域基线。旧研究见 [SM70 packing 报告](../../docs/research/performance/sm70-packing/index.md)。
5. **扩大训练配方实验。** 使用真实数据、明确模型与 token 预算，继续分开 Forward／Dgrad／Wgrad，比较 RNE、SR、尺度策略与 Hadamard 方案，记录 loss、梯度误差和训练稳定性。

可参考 [Training LLMs with MXFP4，AISTATS 2025](https://proceedings.mlr.press/v258/tseng25a.html) 的 SR／随机 Hadamard 反向配方，以及 [2026 原生 MXFP4 预训练研究 v4](https://arxiv.org/html/2605.09825v4) 对不同训练阶段和 Wgrad 的对照。两项工作的硬件与配方不同，不能把其速度或收敛结果直接迁移为 V100 本实现的结论。
