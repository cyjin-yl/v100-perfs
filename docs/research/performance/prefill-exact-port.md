# 「大矩阵 exact 方案」调研与移植报告

调研对象：1Cat-vLLM v1.3.0 的 long-prefill exact-dense 路径
移植目标：fastllm/VastLLM GGUF 后端的 prefill 速度
硬件：单卡 Tesla V100-PCIE-32GB（SM70），CUDA 12.9（conda `tsenv`），fastllm build `Release`, `CUDA_ARCH=70`
生产模型：`Qwen3.8-27B-Uncensored-Cyber-Q5_K_M-plus-mtp.gguf`（arch `qwen35`，65 blocks，hidden 5120，ffn 17408，`full_attention_interval=4`，48 层线性注意力 + 16 层全注意力 + 1 层 MTP）
日期：2026-08-18

---

## 0. 三行结论

1. **「大矩阵 exact 方案」= 1Cat-vLLM 1.3.0 的 long-prefill exact-dense：大 M（prefill）时放弃融合量化 GEMM，把整块权重一次性反量化进一块共享 fp16 workspace，然后跑普通 cuBLAS fp16 GEMM，反量化公式必须与原融合内核逐位一致。**
2. **这个方案 fastllm 早就有了**（`MMVQ_MAX_BATCH_SIZE=8` 以上就走 dequant→`cublasGemmEx`，还带持久 scratch），所以**没有"内核可抄"**；真正没吃到的红利有两条：反量化内核本身被 L2 write-allocate 卡在 ~410 GB/s，以及生产 `--chunked_prefill_size 512` 让这笔固定开销的摊薄比 vLLM（M=4096）差 8 倍。
3. **已实现并验证**：把 fp16 权重展开改成"每 lane 8 个连续输出 + streaming store"，逐位相同，整模型每 chunk 展开时间 154.9 ms → 90.7 ms（1.71x）；配合 chunk 512→2048，量化投影段每 token 成本 966.5 µs → 654.4 µs（**1.48x**），换算到 8K/32K/128K 端到端 prefill 约 **+24% / +19% / +9.5%**。128K 的主要瓶颈不在这里，在注意力。

---

## 1. 「大矩阵 exact 方案」到底是什么

### 1.1 定位（1Cat-vLLM v1.3.0，本地 checkout 在 `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/1Cat-vLLM`，已 `git fetch` 到 tag `v1.3.0`）

本地工作树是 v1.2.2（`3ec0c68c6`），**v1.3.0 才有这套东西**。设计文档两篇（都是 1.3.0 新增）：

| 文件 | 行数 | 内容 |
|---|---|---|
| `docs/design/sm70_awq_long_prefill_exact_dense.md` | 165 | AWQ int4，Qwen3.6-27B TP4，M=4096 |
| `docs/design/sm70_fp8_long_prefill_exact_dense.md` | 113 | FP8 W8A16，Qwen3.8-27B TP4，M>=3920 |

实现代码（行号均为 `git show v1.3.0:<path>`）：

| 位置 | 内容 |
|---|---|
| `csrc/sm70_turbomind/ops/awq_sm70_gemm.cu:2286` | `awq_prefill_physical_nibble()` — 还原 TurboMind K8 片段的 `[0,2,4,6,1,3,5,7]` 物理顺序 |
| `csrc/sm70_turbomind/ops/awq_sm70_gemm.cu:2292` | `awq_sm70_dequantize_kernel()` — **核心**：把 TurboMind K8/N32 打包权重展开成连续 `K x N` fp16 |
| `csrc/sm70_turbomind/ops/awq_sm70_gemm.cu:2327` | `awq_sm70_dequantize_out()` — host 包装 + 形状/对齐校验（强制 `group_size==128`） |
| `csrc/sm70_turbomind/ops/awq_sm70_gemm.cu:2375` | `fp8_sm70_dequantize_kernel()` — FP8 版同构实现 |
| `csrc/sm70_turbomind/ops/awq_sm70_gemm.cu:3521` | `fp8_gemm_sm70_prefill_dispatch_out()` — **M 判定放在这里**（不是 Python） |
| `csrc/sm70_turbomind/ops/awq_sm70_gemm.cu:4785 / 4878` | torch binding shim |
| `csrc/ops.h:177` | `fp8_gemm_sm70_prefill_dispatch_out` 声明 |
| `vllm/model_executor/layers/quantization/awq.py:40` | `_SM70_AWQ_PREFILL_DENSE_M = 4096` |
| `vllm/model_executor/layers/quantization/awq.py:79` | `_awq_exact_f16_weight()` — 逐位精确展开的参考实现 |
| `vllm/model_executor/layers/quantization/awq.py:110` | `_get_sm70_awq_prefill_exact_dense_workspace()` — 有界 85 MiB workspace（`WeakValueDictionary` 按 device 复用） |
| `vllm/model_executor/layers/quantization/awq.py:469 / 512 / 589-608` | 启用判定 / 建 workspace / runtime 分支 |
| `vllm/model_executor/layers/quantization/fp8.py:102` | `_SM70_FP8_PREFILL_DENSE_MIN_M = 3920` |
| `vllm/model_executor/layers/quantization/fp8.py:807 / 920` | 调用 dispatch op |
| `vllm/envs.py:140,158,1595,1601` | `VLLM_SM70_AWQ_PREFILL_EXACT_DENSE` / `VLLM_SM70_FP8_PREFILL_EXACT_DENSE`，默认 `True` |

> 注：仓库里另有 `csrc/sm70_turbomind/.../kernel/sm70_884_4.cu:17-31` 的 `ExactMKernelImpl` / `ExactMnkKernelImpl`，那是**小 M**（M=1 / MTP verifier M=5）的 exact-shape 路由，跟"大矩阵"无关，别混淆。`csrc/quantization/marlin/sm70_marlin_splitk.cuh`（1.3.0 新增）是窄 GEMM 的 split-K，也不是这套。

### 1.2 核心思想

vLLM 用 NCU profile 了主导形状 `M4096,N8704,K5120` 的 gate/up 融合 AWQ GEMM，结论（`sm70_awq_long_prefill_exact_dense.md` "Reason For The Path"）：

- 248 registers/thread、65.55 KB shared memory
- **1 CTA/SM，achieved occupancy 12.5%**
- **62.1% 的调度周期没有可发射 warp**
- **DRAM throughput 只有 9.36%** — 完全不是带宽瓶颈
- stall 集中在 online AWQ unpack 周围的 dependency / barrier / math / shared-load

而 registry autotuning 已经选出了同样的 `CTA128x256x16`，**形状调优这条路已经走死**。所以做**结构替换**：

> 解量化一次 -> 写进一块共享 fp16 `K x N` workspace -> 让 cuBLAS 跑这个 compute-dense 的 M=4096 GEMM。
> decode 和小 M 仍然留在融合量化内核上。

### 1.3 四个关键工程细节（这才是 "exact" 的含义）

1. **数值契约必须逐位一致。** TurboMind 的反量化是
   ```
   bias   = fp16(-zero * scale)
   weight = fp16_fma(q, scale, bias)
   ```
   用数学等价的 `(q - zero) * scale` **不行**——首次探针就出现 `0.008789` 的输出误差。必须保留 fp16 bias 的舍入和单条 FMA 的顺序。验收门槛是"所有 TP shard 上输出逐位相等 + 全模型 token hash 相同"。
2. **有界 workspace，不是常驻权重。** 初版把展开后的权重常驻，每 rank 多占 10.60 GiB（6.50 -> 17.36 GiB），KV 容量从 10.01x 掉到 5.05x。最终版只留**一块 85 MiB 的 `K x N` buffer**，所有合格层共用，靠同一 CUDA stream 的顺序性保证复用安全，代价降到 0.09 GiB/rank（省掉 99.2%）。分配失败就打 warning 退回 TurboMind。
3. **直接 `K x N` 布局，不要 `N x K` + transpose。** 初版存连续 `N x K` 然后每次 `torch.mm` 转置；改成直接物化连续 `K x N` 再调同一个 cuBLAS，额外拿到 2.39%-3.01%（几何均值 2.58%），nsys 上 3824 次调用聚合时间 38.511 s -> 36.959 s。
4. **M 判定必须放在 C++ opaque op 里。** 第一版把 `M >= 3920` 写成 Python 分支，被 `torch.compile` 的 dynamic 编译区间在 tracing 时折叠掉了，实测请求仍然发了 3648 次 TurboMind 调用、零次 dense GEMM。另外把 workspace 当 Tensor 参数传会被 Inductor 提升进 compiled graph，1K/256 单请求吞吐从 ~60 掉到 19 tok/s——最终 ABI 传的是**稳定 CUDA 地址的 int64**。（这条对我们不适用，fastllm 没有 torch.compile。）

### 1.4 vLLM 侧收益（引用其文档）

AWQ TP4，算子级：

| projection | K | N | layers | AWQ 融合 | exact dense | 每 chunk 省 |
|---|---:|---:|---:|---:|---:|---:|
| MLP gate/up | 5120 | 8704 | 63 | 7.089 ms | 4.781 ms | 145.40 ms |
| MLP down | 4352 | 5120 | 63 | 3.371 ms | 2.296 ms | 67.76 ms |
| linear-attn QKVZ | 5120 | 4096 | 47 | 3.376 ms | 2.377 ms | 46.96 ms |
| linear-attn out | 1536 | 5120 | 47 | 1.279 ms | 0.899 ms | 17.86 ms |
| full-attn out | 1536 | 5120 | 16 | 1.276 ms | 0.900 ms | 6.01 ms |

端到端（64K 输入，固定 1530 MHz）：prefill 24.048 s -> 20.338 s（**-15.43%**），TPOT 无变化，token 逐位相同。
FP8 版：32K prefill +22.36%，128K +12.94%，256K +7.25%（**注意这条衰减曲线——越长收益越小**）。

---

## 2. SM70 可移植性清单

| 组件 | 依赖 | V100/SM70 | 结论 |
|---|---|---|---|
| exact-dense 的**结构**（一次展开 + cuBLAS） | 无特殊指令 | 可 | **可移植 — 而且 fastllm 已经有了**（见 3） |
| 有界共享 workspace + 同 stream 复用 | 无 | 可 | 可移植；fastllm 已有 `FastllmBorrowDequantScratch` |
| 逐位一致的反量化数值契约 | 无 | 可 | 可移植；我们的实现已按此标准做（见 5） |
| `K x N` 直接布局（省 transpose） | 无 | 可 | fastllm 已经是 `CUBLAS_OP_T` 的 K-major TN 形态，本来就是 V100 上 fp16 TC 的原生布局，**这条没有额外红利** |
| `awq_sm70_dequantize_kernel` 本体 | AWQ int4 + TurboMind K8/N32 打包 | 需改写 | 内核在 SM70 能跑（vLLM 本来就是 SM70 专用），但**布局完全不同**：我们跑的是 GGUF Q5_K/Q8_0/Q6_K super-block，不是 AWQ。要抄的是**思路不是代码** |
| `dequantize_s4_to_fp16x2` LOP3 快速反量化（`csrc/libtorch_stable/quantization/awq/dequantize.cuh`） | `lop3.b32`（sm_50+） | 指令可用但**用不上** | LOP3 技巧针对 int4 对称量化的 `q -> fp16` 位拼接；GGUF Q5_K 是 6-bit 子块 scale/min + 5-bit（4+1 位拆分）量化，`d1*(ql\|qh<<4) - m1` 无法用位拼接表达 |
| TurboMind SM70 s884 GEMM（`config_sm70_s884.h`, `mainloop_sm70.h`, `iterator_sm70.h`） | `mma.m8n8k4` FP16 TC | 可跑但没必要 | 这就是 exact-dense 要绕过的那条融合路；且 fastllm 的 AWQ SM70 内核（`src/devices/cuda/awq_sm70/`）已经是从这里抄的 |
| `cp.async`（`cp_async.h`、`iterator_sm80.h`、`mainloop_sm80_v2.h`） | sm_80+ | **不可用** | V100 只能 `LDG -> STS` 两段式，这也是 TurboMind SM70 路占寄存器/shared 多、occupancy 只有 12.5% 的根因之一 |
| `mma.m16n8k16` / `config_sm75_s16816.h` / `config_sm80_s16816.h` | sm_75+ | **不可用** | — |
| machete（`csrc/quantization/machete`） | sm_90 CUTLASS 3.x + TMA | **不可用** | — |
| marlin / marlin split-K（`csrc/quantization/marlin/sm70_marlin_*.cu`, `sm70_marlin_splitk.cuh`） | 该分支有 SM70 特化版 | 可跑但正交 | 目标是**窄 N / 小 M** 的 split-K，跟大矩阵 prefill 不是一回事 |
| `st.global.cs` / `__stcs` streaming store | sm_20+ | 可 | **可用，而且这是本次最大的实际收益来源**（见 4.3） |
| `mmq.cuh` vs `mmvq.cuh`（GGUF 融合量化 GEMM） | DP4A（sm_61+） | 可 | fastllm 只对 IQ4_XS 做了 SM70 MMQ（`fastllm-iq4xs-sm70.cu`）；Q5_K 没有。**但按 vLLM 的结论，融合 MMQ 正是要被替换掉的那条路，不建议补** |

**一句话**：vLLM 那套里，**方法论 100% 可移植，代码 0% 可直接复用**（量化格式不同），而方法论我们已经在跑了。

---

## 3. fastllm 的现状：exact-dense 早就在跑

关键证据（`/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/fastllm`）：

- `src/devices/cuda/fastllm-ggml-cuda.cu:781-783`
  ```c
  // Keep speculative verification with one anchor plus seven draft tokens on the
  // quantized path.  Falling back at batch 8 dequantizes every GGUF linear weight
  // to BF16 before GEMM, which multiplies target-model memory traffic.
  #define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.
  ```
- `src/devices/cuda/fastllm-ggml-cuda.cu:2816-2861`（fp16 输入路径）
  ```c
  } else if (!sm70Iq4XsDone && (n > MMVQ_MAX_BATCH_SIZE || !has_vec_dot) && dequant != nullptr) {
      half *cudaFp16Weight = (half *) FastllmBorrowDequantScratch(needBytes, &wsBytes, &ownScratch);
      ...
      dequant((const char *)weight.cudaData, cudaFp16Weight, k, m, stream);   // 一次性展开
      ...
      cublasGemmEx(..., CUBLAS_OP_T, CUBLAS_OP_N, k, nCount, m,
                   &h_alpha, cudaFp16Weight, CUDA_R_16F, m, ...);             // 普通 fp16 GEMM
      FastllmReleaseDequantScratch(cudaFp16Weight, ownScratch);
  }
  ```
- `src/devices/cuda/attention/fastllm-attention.cu:2525` — `FastllmBorrowDequantScratch()` 就是 vLLM 的"有界共享 workspace"，底层是一个持久缓存 holder（不是每次 cudaMalloc）。

对照 vLLM 的 checklist：

| vLLM exact-dense 要素 | fastllm 状态 |
|---|---|
| 大 M 走 dequant + 普通 GEMM | 已有（阈值 M>8） |
| 小 M / decode 留在融合量化内核 | 已有（MMVQ，M<=8） |
| 共享有界 workspace，不常驻权重 | 已有（`FastllmBorrowDequantScratch`） |
| M 判定在 C++ 里 | 天然如此（无 torch.compile） |
| K-major 直接布局 | `CUBLAS_OP_T` + lda=m 就是 TN，V100 fp16 TC 原生 |
| 逐位一致 | 反量化内核是 llama.cpp 原版，无近似 |

**所以"把 exact 方案移植过来"这个任务本身，答案是：没有东西可移。** 剩下的价值在两个 fastllm 特有的浪费上，见 4/5。

---

## 4. 我们 prefill 的定量瓶颈

### 4.1 生产配置

```
apiserver ... --atype float16 --kv_cache_dtype turbo3 --batch 2
              --tokens 262144 --chunked_prefill_size 512 --device cuda
```

**`--chunked_prefill_size 512`** => 每个 prefill chunk 的 GEMM 只有 **M=512**。vLLM 的 exact-dense 是 M=4096（AWQ）/ M>=3920（FP8）。**同样一次权重展开，我们只摊到 1/8 的 token 上。**

`Qwen3_5Model::GetChunkedPrefillSize()`（`src/models/qwen3_5.cpp:9203-9213`）：
```cpp
int base = basellm::GetChunkedPrefillSize();               // 来自 CLI = 512
int interval = Qwen35LinearPrefixSnapshotIntervalTokens();  // 64 pages * 128 = 8192
return std::max(pageLen, std::min(base, interval));         // = min(512, 8192) = 512
```
=> **这个 clamp 只会往下压，把 CLI 值提到 2048（甚至 8192）会被完整采纳。** 唯一副作用是 chunk 变大后 decode 交织粒度变粗（TPOT 抖动），以及注意力 scratch 从 ~8 MB 涨到 ~32 MB。

### 4.2 权重形状（GGUF 实测，非估算）

从 GGUF header 直接读出（`n_tensors=866`，`file_type=17` 即 Q5_K_M）：

| tensor | dims (in -> out) | 量化 | 层数 |
|---|---|---|---|
| `ffn_gate` / `ffn_up` | 5120 -> 17408 | **Q5_K** | 64 each |
| `ffn_down` | 17408 -> 5120 | **Q5_K** | 64 |
| `attn_qkv`（线性注意力） | 5120 -> 10240 | Q8_0 | 48 |
| `attn_gate`（线性注意力 z） | 5120 -> 6144 | Q8_0 | 48 |
| `ssm_out` | 6144 -> 5120 | Q8_0 | 48 |
| `attn_q` / `attn_k` / `attn_v`（全注意力） | 5120 -> 12288 / 1024 / 1024 | Q8_0 | 16 |
| `attn_output`（全注意力） | 6144 -> 5120 | **Q6_K** | 16 |

量化投影总参数约 24.3 G。**每个 chunk 要把这 24.3 G 全部展开成 fp16（约 48.6 GB 写 + 约 21 GB 读，约 70 GB HBM 流量），不管这个 chunk 是 512 个 token 还是 4096 个。**

（注：fastllm 载入时会把 `gate/up` 合成 `gateup_proj {34816,5120}`、`q/k/v` 合成 `mergeqkv {14336,5120}`、线性注意力 `qkv+z` 合成 `in_proj_qkvz {16384,5120}`。合并不改变总参数量与总流量，下面的 roll-up 按未合并形状统计，二者等价。）

### 4.3 microbenchmark（`work/fastllm-prefill/tools/prefill_dequant_bench/bench.cu`）

离线内核级基准（生产占着 GPU 时不可能跑整模型：`nvidia-smi` 显示 apiserver 常驻 30.6 GB / 32.7 GB，最低时只剩 211 MiB）。基准做法：
- 反量化 baseline 内核**逐字**从 `fastllm-ggml-cuda.cu` 抄过来（`base_q5_K` <- 行 1615，`base_q8_0` <- 行 1423，`base_q6_K` <- 行 1649）
- cuBLAS 调用与 `fastllm-ggml-cuda.cu:2836` **完全同参**（`CUBLAS_OP_T/N`，compute type `CUDA_R_16F`）
- min-of-N 计时；下面这组是 **GPU 完全空闲**（free 31.7 GiB, util 0%）时 min-of-60 的结果

**反量化（权重展开）：**

| shape | q | base ms | base GB/s | 优化后 ms | 优化后 GB/s | 加速 | 逐位 |
|---|---|---:|---:|---:|---:|---:|---|
| `ffn_gate` 17408x5120 | Q5_K | 0.6133 | 390.5 | **0.3225** | **742.7** | **1.90x** | EQ |
| `ffn_down` 5120x17408 | Q5_K | 0.5612 | 426.9 | **0.3205** | **747.3** | **1.75x** | EQ |
| `in_proj_qkv` 10240x5120 | Q8_0 | 0.2898 | 554.1 | **0.2089** | **768.6** | **1.39x** | EQ |
| `attn_q` 12288x5120 | Q8_0 | 0.3492 | 551.8 | **0.2499** | **771.1** | **1.40x** | EQ |
| `attn_o_proj` 5120x6144 | Q6_K | 0.1966 | 451.2 | **0.1239** | **716.0** | **1.59x** | EQ |

**roofline 对照（同样的字节搬运量，去掉全部算术）：**

```
plain store : 0.5816 ms  (411.8 GB/s)
cs    store : 0.3205 ms  (747.3 GB/s)
```

> **这是本次调研最重要的一条定量证据**：一个不做任何真实数学、只搬同样字节数的内核，用普通 store 也只能跑到 411.8 GB/s，用 streaming store 立刻 747.3 GB/s。也就是说 **Q5_K 反量化的瓶颈既不是算术也不是 store 宽度，而是 L2 的 write-allocate 策略**——展开出来的权重有 100~350 MB，被 cuBLAS 读回恰好一次，在 6 MB L2 里毫无复用，默认 write-back 只是在反复冲刷 L2。
>
> 附带结论：第一版"只做 16 字节向量化 store"的改动实测是 **0.92x（更慢）**，因为 llama.cpp 原版内核的 store 本来就已经是满 64 B sector 合并的。**如果只凭直觉去做"向量化"，会做出负优化。**

**cuBLAS fp16 GEMM（TFLOP/s，空闲 GPU）：**

| shape | M=256 | M=512 | M=1024 | M=2048 |
|---|---:|---:|---:|---:|
| `ffn_gate` 17408x5120 | 52.4 | 76.4 | 83.8 | **83.8** |
| `ffn_down` 5120x17408 | 58.9 | 77.2 | **92.7** | 72.2 |
| `in_proj_qkv` 10240x5120 | 62.3 | 59.0 | 82.2 | **80.9** |
| `ssm_out_proj` 5120x6144 | 69.0 | 90.4 | **93.3** | 81.7 |
| `attn_k` 1024x5120 | 36.9 | 46.4 | 69.0 | **70.1** |

M=512 时 cuBLAS 只有 46~90 TFLOP/s（V100 fp16 峰值 112），M>=1024 才普遍上到 80~93。**小 chunk 不但摊不薄反量化，GEMM 自己的 tile 效率也差。**

### 4.4 整模型 roll-up（176 个量化投影，一个 chunk）

| chunk M | 展开(base) | 展开(优化) | GEMM | 合计(base) | **展开占比** | 合计(优化) | 优化增益 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 154.87 | 90.72 | 225.89 | 380.76 | **40.7%** | 316.60 | 16.85% |
| **512（生产）** | 154.87 | 90.72 | 339.95 | **494.82** | **31.3%** | 430.67 | **12.97%** |
| 1024 | 154.87 | 90.72 | 591.03 | 745.91 | 20.8% | 681.75 | 8.60% |
| 2048 | 154.87 | 90.72 | 1249.43 | 1404.30 | 11.0% | 1340.15 | 4.57% |

**每 token 成本（量化投影段）：**

| chunk M | base us/token | 优化后 us/token | vs 512 (base) |
|---:|---:|---:|---:|
| 256 | 1487.34 | 1236.73 | -53.9% |
| **512（生产）** | **966.46** | 841.15 | 0% |
| 1024 | 728.43 | 665.77 | **+24.6%** |
| 2048 | **685.70** | **654.37** | **+29.1%** |

**两个杠杆叠加：966.46 -> 654.37 us/token = 投影段 1.48x。**

### 4.5 端到端归因

已知实测 prefill：8K 775 tok/s（512 tok/chunk 约 660.6 ms）、32K 610 tok/s（约 839 ms）、128K 305 tok/s（约 1679 ms）。
上表给出 chunk=512 时量化投影段 = 494.8 ms。相减即为"注意力 + GDN + norm/激活 + 调度"：

| 上下文 | 每 512-token chunk 总耗时 | 量化投影 | 其余（注意力/GDN/其他） | 投影占比 |
|---|---:|---:|---:|---:|
| 8K | 660.6 ms | 494.8 ms | 165.8 ms | **74.9%** |
| 32K | 838.9 ms | 494.8 ms | 344.1 ms | **59.0%** |
| 128K | 1678.7 ms | 494.8 ms | 1183.9 ms | **29.5%** |

其中"权重展开"这一项固定 154.9 ms：占 8K 的 23.4%、32K 的 18.5%、128K 的 9.2%。

**=> prefill 随长度衰减的元凶不是反量化也不是 GEMM（两者对 token 数是线性的），而是那块随上下文平方增长的注意力项**（16 层全注意力；48 层 GDN 线性注意力对长度线性）。128K 时它已经吃掉 70.5%。这正好对应 vLLM FP8 文档里 exact-dense 收益从 32K 的 +22.36% 衰减到 256K 的 +7.25% 的同一条曲线。

---

## 5. worktree 里做了什么

- 分支：**`feat/prefill-exact-port`**
- worktree：`/run/media/ezra/13D010B6FDBC1A06/projects/EzraVastLLM/work/fastllm-prefill`
- commit：**`aeb9ad20`** — `cuda(gguf): streaming vectorised fp16 weight expansion for prefill`
- 改动：`src/devices/cuda/fastllm-ggml-cuda.cu` +206/-1，新增 `tools/prefill_dequant_bench/bench.cu`
- 独立 build 目录 `build-prefill/`（**没有碰 `fastllm/build-rw/apiserver`**）；`fastllm-ggml-cuda.cu.o` 编译通过（4.8 MB，零 error）

### 改了什么

针对 fastllm 的 exact-dense 展开步骤，新增三个 fp16 专用内核（`dequantize_block_q5_K_vec_half` / `_q8_0_` / `_q6_K_`）：

1. **每 lane 产出 8 个连续输出**，发一条 16 字节 store，而不是 2~4 条散落的 2 字节 store；
2. **streaming store**（`__stcs` -> `st.global.cs`，evict-first）。这是主要收益来源，理由见 4.3 roofline。

通过模板特化接入，只改 fp16 表，fp32/bf16 路径原样不动：

```cpp
template<> void dequantize_row_q5_K_cuda<half>(...)          // :1974
template<> void dequantize_row_q8_0_dispatch_cuda<half>(...) // :1994
template<> void dequantize_row_q6_K_cuda<half>(...)          // :2094
ggml_get_to_fp16_cuda(): case GGML_TYPE_Q8_0 -> dequantize_row_q8_0_dispatch_cuda  // :2635
```

守卫：`FastllmGGUFCanVecDequant()` = 环境开关 && 输出 16 字节对齐 && 元素数整除 block 长度，否则原路退回。

**回滚开关：`FASTLLM_GGUF_VEC_DEQUANT=0`。**

### 正确性

算术、算子顺序、`__float2half_rn` 舍入全部未变，**权重展开结果与原内核逐位相同**。benchmark 对 Qwen3.8-27B 的全部 10 种投影形状做了 `memcmp` 校验，**10/10 BITWISE-EQUAL**（Q5_K / Q8_0 / Q6_K 各变体均验证）。这与 vLLM 的验收标准一致（他们那篇文档专门强调 `(q-zero)*scale` 是错的）。

### 前后数据

| 指标 | before | after | 加速 |
|---|---:|---:|---:|
| Q5_K `ffn_gate` 17408x5120 | 0.6133 ms / 390.5 GB/s | **0.3225 ms / 742.7 GB/s** | **1.90x** |
| Q8_0 `attn_q` 12288x5120 | 0.3492 ms / 551.8 GB/s | **0.2499 ms / 771.1 GB/s** | **1.40x** |
| Q6_K `attn_o_proj` 5120x6144 | 0.1966 ms / 451.2 GB/s | **0.1239 ms / 716.0 GB/s** | **1.59x** |
| **整模型每 chunk 权重展开** | **154.87 ms** | **90.72 ms** | **1.71x** |
| 量化投影段 @M=512 | 494.82 ms | 430.67 ms | 1.149x |
| 量化投影段每 token @M=512 | 966.46 us | 841.15 us | 1.149x |

原始输出：`reports/bench_clean.txt`（空闲 GPU，min-of-60，主数据）、`reports/bench_contended.txt`（生产占用时，用于确认 ratio 稳定）。

**未做（明确说明）：** 没有在 worktree 里起临时后端跑整模型基准。原因是生产 apiserver 常驻 30.6 GB / 32.7 GB，剩余显存最低时只有 211 MiB，起第二个 27B 实例既不可能也会挤爆生产。所有数据都是离线内核级的。全程未重启 proxy/后端、未碰 8000/8002、未改 `thinking_proxy.py` 与任何 `runtime/fastllm-native-profiles/*.env`、未在 `fastllm` 主工作树写任何文件。

---

## 6. 下一步建议

### 值不值得继续："移植 vLLM 内核"不值得，"补齐摊薄"很值得

| 动作 | 预期收益（8K / 32K / 128K prefill） | 成本 | 风险 |
|---|---|---|---|
| **A. 合并 `feat/prefill-exact-port`** | +9.7% / +7.6% / +3.8% | 已完成，206 行 | 低。逐位一致，有 env 回滚 |
| **B. `--chunked_prefill_size` 512 -> 2048** | +21.8% / +17.1% / +8.6% | **零代码** | 中：decode 交织粒度变粗（TPOT 抖动上升）、注意力 scratch 8->32 MB。`GetChunkedPrefillSize()` 的 clamp 上限是 8192，2048 会被完整采纳 |
| **A+B 叠加** | **+24.2% / +19.0% / +9.5%**（775->1022 / 610->726 / 305->334 tok/s） | | |
| C. 长上下文注意力 | 128K 的大头（70.5%）在这里 | 大 | 见下 |
| D. 抄 vLLM 的 exact-dense 内核 | **约 0** | 大 | fastllm 已经在跑同一套结构 |
| E. 给 Q5_K 补 SM70 融合 MMQ | **可能是负的** | 大 | vLLM 的 NCU 结论就是融合路 occupancy 12.5%、是要被替换掉的那条 |

**建议顺序：先 B（零成本，收益最大），再 A（已在分支上），C 单独立项。**

关于 B 的验证方法：可以先只改临时 profile 用 8007 端口验一轮 A/B，确认 TPOT 退化在可接受范围内再上生产。

### 关于 C（长上下文注意力）

128K 时注意力占 70.5%，这才是"prefill 随长度衰减"的真正原因。vLLM 那边对应的是另外几篇文档，不是本次调研的 exact-dense：
- `docs/design/sm70_flash_v100_prefill_operator_optimization.md`（1134 行）
- `docs/design/sm70_fa2_d256_prefill_pipeline.md`（439 行，1.3.0 新增）
- `docs/design/sm70_sm120_long_context_attention_parity.md`

这几篇是下一轮调研的正确入口。另外 fastllm 侧已经有两个**实现了但默认关闭**的开关值得先 A/B：
`FASTLLM_CUDA_PAGED_CUBLAS_BATCH_GQA=1`、`FASTLLM_CUDA_PAGED_CUBLAS_FUSED_STATE_COMMIT=1`。

### 2TP 场景会不一样吗？会，而且收益更大

1. **权重展开是纯 per-rank 固定成本，TP 切分后按 N 维减半，但 GEMM 的 M 不变。** 2TP 下每 rank 展开约 154.87/2 = 77.4 ms，GEMM 约 339.95/2 = 170 ms，**展开占比仍然是 31.3%**（分子分母同除 2）。所以第 5 节的 1.71x 加速在 2TP 下**完全保留**。
2. **chunk 摊薄的杠杆更重要。** 2TP 时每 rank 的 GEMM 变小（N 减半），cuBLAS 在 M=512 下 tile 效率更差（参考 `attn_k` 1024x5120 这种窄形状 M=512 只有 46.4 TFLOP/s，M=2048 才 70.1），而展开成本占比不降。vLLM 的数据点正好在 TP4 上：他们在 M=4096 才拿到 15~22%。**=> 2TP 上建议把 chunk 提到 2048~4096，收益会比 1TP 更明显。**
3. **新增变量：all-reduce。** 2TP 每层 2 次 all-reduce，prefill 时是 `M x 5120 x 2` 字节。M=512 时每 chunk 64 层 x 2 x 5.24 MB = 671 MB over PCIe（V100-PCIE 无 NVLink，约 10 GB/s 双向）约 67 ms/chunk，**跟展开开销同一量级**。这条在 1TP 上完全不存在，2TP 立项时必须先测它，否则前面所有优化都会被 PCIe 吃掉。

---

## 7. 复现方式

```bash
# microbenchmark（约 700 MB 显存，需 GPU 空闲才有干净数字）
cd /run/media/ezra/13D010B6FDBC1A06/projects/EzraVastLLM/work/fastllm-prefill/tools/prefill_dequant_bench
/home/ezra/.conda/envs/tsenv/bin/nvcc -O3 -arch=sm_70 -std=c++17 -Wno-deprecated-gpu-targets \
  -ccbin /home/ezra/.conda/envs/tsenv/bin/x86_64-conda-linux-gnu-g++ bench.cu -o bench -lcublas
./bench 60

# 隔离构建（不要覆盖 fastllm/build-rw/apiserver）
cd /run/media/ezra/13D010B6FDBC1A06/projects/EzraVastLLM/work/fastllm-prefill/build-prefill
make fastllm -j 6      # 需先加 -DFASTLLM_ZSTD_INCLUDE_DIR=/usr/include \
                       #        -DFASTLLM_ZSTD_LIBRARY=/usr/lib64/libzstd.so 重新 configure
```

产出文件：
- `reports/prefill-exact-port.md`（本文）
- `reports/bench_clean.txt` — 空闲 GPU，min-of-60，**主数据**
- `reports/bench_contended.txt` — 生产占用时的对照
- `work/fastllm-prefill/` — 分支 `feat/prefill-exact-port`，commit `aeb9ad20`
