# 用整数证书允许 MXFP4 整段 K 的 Tensor Core 累加

CPU 实验日期：2026-11-23 UTC；证据归档目录使用 Asia/Shanghai 的 2026-11-24。对应实现现已获得真实 V100 的正确性验证和 packed-input pipeline 性能测量，详细结果、计时范围及限制统一见 [GPU 证据说明](../../../../evidence/results/mxfp4-training/2026-11-24-performance/README.md)。本文负责数值证明和 CPU 合成分布实验，不据此推断真实训练命中率或收敛。

目标是在证书通过时，以归一化 FP16 输入、FP32 累加和最后一次缩放，精确实现现有 MX32 算术；其余输入继续使用完整范围的逐 MX32 后端。此前 cuBLAS 实现的独立审查对应 `mxfp4_gemm_certified.cu` 的 SHA-256 `489786a18363bf1314c5de8ff8b3cab92ca1a29c3c186f593f051f2fc33e20ba`。所选 variant 2 将四个 q 的能量合组计算，并按行向量化最终缩放。新增融合候选的实现、构建哈希与验证状态另见 [融合实验](../../../../evidence/results/mxfp4-training/2026-11-24-fused/README.md)；它们共用下述整数证书、正常数边界和完整范围回退。

## 现有算术及为什么需要证书

[独立参考实现](../../../../tools/mxfp4_training/reference.py) 定义 `q = 2 * E2M1`，完整数值集合为 `0, ±1, ±2, ±3, ±4, ±6, ±8, ±12`。一个输入元素为

\[
x_{ik}=q_{ik}2^{s_{i,b(k)}-128},\qquad b(k)=\lfloor k/32\rfloor.
\]

每个输出先精确计算一个 MX32 的整数点积，再用 `ldexpf(qsum, sa + sb - 256)` 缩放，最后按 K 顺序进行 FP32 加法。scale 字节 `255` 必须传播 NaN，即使该块整数点积为零。

**仅证明输入能被 FP16 精确表示不够。** 令 K=96，三个 MX32 各仅一个非零乘积：两个操作数分别为 `[4096, 1, -4096]` 和 `[4096, 1, 4096]`，其余位置为零。它们均为精确的正常 FP16 数，可用 q 值 `[2, 1, -2]`、`[2, 1, 2]` 和 scales `[139, 128, 139]` 表示。三个块的贡献为 `[2^24, 1, -2^24]`。现有 FP32 顺序结果为 `0`，先消去两个大数再相加则为 `1`。本页证书会拒绝此例。

## 保守而充分的证书

对每个 A 行 i 和 B 行 j：

1. 先检查每一个 scale；发现 `255` 即拒绝快路径，不能因该块为零而跳过。
2. 取非零 q 块的最小 scale 为 `m_i`，最大 scale 为 `h_i`。也可把零块纳入 min/max：这只会更保守。所有 q 均为零的行可令 `m_i=127`、能量为零。
3. 要求每个非零行 `h_i-m_i <= 12`，并定义归一化元素 `a'_ik = q_ik * 2^(s_i,b(k)-m_i)`；B 同理。
4. 精确计算整数行能量，并取各操作数最大值：

\[
U=\max_i\sum_k (a'_{ik})^2,\qquad
V=\max_j\sum_k (b'_{jk})^2.
\]

5. 要求 `U * V <= 2^48`。
6. 要求所有需要计算的行对满足 `-126 <= m_A,i + m_B,j - 256 <= 103`。

第 6 条不需要遍历 M×N：分别求 A、B 的行最小 scale 的最小值和最大值，即可检查两个端点。零行可以纳入检查而保守拒绝；若排除零行，必须先明确记录其能量为零。若整个 A 或 B 的能量为零，在完成所有 NaN scale 检查之后，可以直接返回 canonical `+0`。

### 证明

**FP16 输入精确。** 每个非零归一化元素都是 q 乘以 `2^d`，其中 `0 <= d <= 12`。q 至多有三个有效二进制位，故这些值均能由 FP16 精确表示，绝对值介于 `1` 与 `12*2^12=49152`，没有 FP16 下溢、次正规数或溢出。

**任何点积部分和精确。** Cauchy–Schwarz 不等式给出

\[
\sum_k |a'_{ik} b'_{jk}|
\leq \sqrt{\sum_k(a'_{ik})^2\sum_k(b'_{jk})^2}
\leq \sqrt{UV}\leq 2^{24}.
\]

每个乘积和每个任意子集的部分和均为绝对值不超过 `2^24` 的整数，因而可由 FP32 精确表示。对这些相同乘积进行任何顺序或树形归约，都没有舍入误差；这同时覆盖 split-K 的 FP32 部分结果与归约。证书依赖的是绝对和界，不能用最终有符号和小于 `2^24` 代替。

**与原 MX32 顺序算术一致。** 对固定行对，令 `lambda = m_A,i + m_B,j - 256`。原点积中的每一项都是归一化整数乘积乘以同一个 `2^lambda`。任何块内或块间部分和的非零绝对值至少为 `2^-126`，最多为 `2^24 * 2^103 = 2^127`；且仍至多有 24 个有效位。因此原实现的每次块缩放和顺序 FP32 加法也都精确、正常且有限。整 K 的归一化结果最后缩放一次，与原参考逐位一致。零结果在最后明确写为 `+0`，不依赖库实现对有符号零的选择。

这是覆盖完整 E2M1 q 集合的**条件快速路径**，不是限制 q 为更窄字母表，也不是允许任意 E8M0 输入无条件进入 cuBLAS。绝对 scale 不必在 FP16 原值范围内：归一化把它们移到了输出的一次缩放中。未满足条件的输入始终回退。

## 实现边界

- cuBLAS 使用 FP16 A/B、`CUBLAS_COMPUTE_32F`、FP32 输出、`alpha=1`、`beta=0`。不要用 FP16 输出或 `FAST_16F` 模式替代。可显式启用 `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`；不要为此选择会阻止 Tensor Core 路径的 pedantic 模式。证明覆盖同一组乘积的 FP32 归约，应固定并验证实际选用的 SM70/cuBLAS 后端。
- 能量必须用整数精确计算。每块可先求 `sum(q*q)`，再左移 `2*(s-m)`。若忽略零块取 m，**零块的 `s-m` 可能为负数**：必须先检查块能量为零并跳过，不能先移位再乘零。逐元素反量化也应先把 q=0 写为 half `+0`。
- 使用 `uint64_t` 时防止移位和加法先溢出。把能量饱和到 `2^48+1` 足以安全拒绝所有不可能通过的非零对。比较用 `U <= (1ULL<<48)/V`，先处理 V=0，避免 `U*V` 溢出。不能先转 FP32 计算能量门禁。
- 最后先处理零，再用正确的 power-of-two 缩放；不要先构造可能不适合的半精度 scale。本文刻意采用正常数区间 `[-126,103]`，不依赖 Tensor Core 的次正规数行为。
- 若将证书缓存，必须绑定到输入 codes、scales 和布局的真实版本；权重更新及重新量化后失效。F、dgrad、wgrad 的不同归约轴分别验证。
- 证书、归一化反量化、cuBLAS、最终缩放和失败回退都应计入端到端计时。可在同一 stream 中依次执行 GPU 证书、无条件 cuBLAS、通过时的最终缩放、拒绝时的精确 tiled fallback，避免每次把 flag 读回 host 后再调度。拒绝的归一化输入必须安全地写成有限值；其无用 cuBLAS 结果由 fallback 覆盖。fallback 的 device guard 必须在任何 barrier 前由整个 CTA 一致执行；拒绝时额外执行一次 cuBLAS 的成本不能从 total 隐去。额外 FP16 workspace 为约 `2*(M+N)*padded_K` 字节；这条路径本身不构成省显存结论。
- 相同证书也支持融合反量化的 HMMA 主循环：跨整 K 累加，最终只缩放一次，避免全局 FP16 workspace。variants 3–10 实现此路径；[后续 SXM2 完整对照](../../../../evidence/results/mxfp4-training/2026-11-23-crater-c8/README.md)中 C10 比 C8 更快，所选 C2 cuBLAS 路径仍最快。不能从减少 workspace 直接推断吞吐优势。

### 融合路径与块能量缓存

融合实现把 `q*2^(s-m)` 直接写入共享内存，用 `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32` 跨整 K 累加，在写 FP32 输出时施加一次行对尺度并将零规范化为 `+0`。GPU 证书拒绝时，整个 CTA 在读取操作数和任何 barrier 之前一致返回，随后原 exact tiled 后端计算输出。cuBLAS variants 0／1／2 的拒绝成本仍包含无用 cuBLAS；融合 variants 3–10 没有该项成本，报告须区分。

C6／C7 在计算当前 BK32 stage 前，把下一块 packed codes 和 scale 预取到寄存器，下一次迭代才解码并写共享内存。现有两个 CTA barrier 分别保护当前共享数据发布与共享 stage 复用；所有线程按统一的 K 循环执行。预取不改变归一化数值、证书门槛或最终缩放，零块仍在任何可能为负的尺度差移位前返回零。C7 改用 64×64 warp tile；这只改变已由整数证书允许的 HMMA 归约分组，性能是否有利另由测量判定。

C8 保留 C6 的计算顺序和共享 stage，只改变符合条件的输入加载。int32 安全索引、M/N 为128倍数且 A/B 均8B对齐时，每线程为每个操作数读取半个MX32块，形成一个 `uint2` packed 与一个 scale。行字节数为16倍数，行内偏移为 `block*16+half*8`，因此宽加载保持对齐且不跨块；其他输入显式使用 C6。交换线程行编号的bit2/bit3只是双射，所有共享half4位置仍恰写一次，HMMA读取的逻辑元素不变。

若一个half的任一q非零，其所属MX32块必已参与row_min；通过原证书后，整个half可共用有效的 `0<=delta<=12`。其中即使某个half4全为零，原归一化掩码也会将其写为canonical `+0`。全零half在形成delta前直接发布零，避免低scale零块的负移位。预取只在下一块存在时发起，K=0不读取packed、scale或row_bases；拒绝仍在任何barrier前一致返回。以上不改变证书准入条件，也不引入FP16部分和。硬件依据、源码检查和实际验证状态见 [C8硬件与文献研究](../../../../evidence/results/mxfp4-training/2026-11-24-hardware-study/README.md)。

当 `padded_K<=32768`，融合版准备阶段先为每个 MX32 计算整数 `Q=sum(q*q)`，把它与 scale 存在一个 32-bit 共享字中。`0<=Q<=4608`，因此低 16 位足够；scale 放在高 16 位。取得行最小尺度后，只需将每个非零且 scale 有限的 Q 左移 `2*(s-m)`，按原规则饱和累加。此时非零块已经参与最小值计算，故移位非负；`delta<=24` 时最大中间值 `4608*2^48<2^61`，仍在 uint64 范围内。非负整数饱和加法满足结合律，因此块聚合与原先四元素分组得到相同能量和拒绝标志。更长 K 保留不缓存的准备路径。

缓存只在当前调用内部保存原始块能量，下一次调用仍重新读取输入、重算尺度和证书；它不是跨调用的证书缓存。最大动态共享内存为 4096 字节，2048 的 K 使用 256 字节。全局 workspace 则只有每行 16-byte 元数据、1-byte 行尺度和 72-byte 证书，即 `17*(M+N)+72` 字节。这里的 workspace 数字不包括输入、输出、驱动初始化、主机元数据及每个 SM 的共享内存资源。

后续 C2 可选 raw preparation cache 是另一个实现：仅在 `0<padded_K<=8192` 缓存每块 16-byte packed data 加 4-byte scale/energy，最大动态共享内存为 5120 字节，仍保留 C2 全局 FP16 workspace。其独立[证明与实测](../../../../evidence/results/mxfp4-training/2026-11-23-crater-c8/candidate-c2-prepare-cache/independent-audit.md)覆盖新旧准备内核和 327 项检查；完整路径观测差异不足 1%，默认关闭。显式 Python workspace 只复用分配与 handle，每次仍重新计算证书，不改变上述准入条件。

### 仅用 scale 的更便宜初筛

设两边所有行的最大 scale 跨度分别为 `Delta_A`、`Delta_B`，则

\[
144 K 2^{\Delta_A+\Delta_B}\leq 2^{24}
\]

也是绝对和界的充分条件。若没有独立验证尾部 padding 为零，K 使用实际遍历的 padded_K。该条件通过即可跳过数据感知能量计算；不通过不代表真实能量证书不能通过。K=4096、8192、16384 分别允许跨度和不超过 4、3、2。

## CPU 合成分布实验

使用当前 `quantize_mx`，两边各 256 行，种子 `1000,1001,1002`，RNE；分别令 K=4096 和 16384。以下统计采用**包含零块的保守行 min/max**。表中的比值是 `sqrt(U*V)/2^24`，仅作展示；是否通过使用整数比较。所有测试中原 lambda 区间均满足上述门禁。仅测试 3 个种子，不是命中概率估计。

| 合成分布 | K=4096：通过数、界比值 | K=16384：通过数、界比值 |
|---|---:|---:|
| 标准高斯 | 3/3；0.0310–0.0316 | 3/3；0.1254–0.2454 |
| 高斯乘 `1e-5` | 3/3；0.0274–0.0279 | 3/3；0.1075–0.1078 |
| Student-t(3) 除以 `sqrt(3)` | 3/3；0.0862–0.1574 | 3/3；0.3600–0.6763 |
| 高斯中以 0.05% 概率将元素乘 100 | 3/3；0.5449–0.6072 | 0/3；1.0533–1.3991 |
| 每个 MX32 高斯块乘随机 `2^e`，e∈[-8,8] | 0/3；行跨度 18 | 0/3；行跨度 18 |

这些结果说明证书可以接受非 unit-scale 的完整 E2M1 量化输入，也会拒绝强离群值或大块间跨度。它们不能证明真实 LLM 的 F/D/W 命中率，更不证明训练稳定性。下一步应在真实、未经选择的训练 tensor 上记录各方向的门禁通过率、拒绝原因和完整耗时。

原实验的 [30 条逐次结果](../../../../evidence/results/mxfp4-training/2026-11-24-performance/cpu-certified-distribution.jsonl)、[来源、环境和 SHA-256](../../../../evidence/results/mxfp4-training/2026-11-24-performance/cpu-certified-distribution-provenance.json) 与 [独立复跑脚本](../../../../evidence/results/mxfp4-training/2026-11-24-performance/cpu-certified-distribution.py) 已归档。归档从本次已执行命令的 stdout 保存，没有为归档重跑实验。

在仓库根目录、安装 NumPy 后可复跑：

```bash
python - <<'PY'
import json, math
import numpy as np
from tools.mxfp4_training.reference import quantize_mx, unpack_q

def certificate(x):
    t = quantize_mx(x.astype(np.float32))
    s = t.scales.astype(np.int64)
    m = s.min(axis=1)
    d = s - m[:, None]
    result = dict(scale_range=[int(s.min()), int(s.max())],
                  max_span=int(d.max()), energy=None)
    if result['max_span'] > 12:
        return result
    q = unpack_q(t).astype(np.int64).reshape(*s.shape, 32)
    # For these K <= 16384 experiments the complete sum fits in int64.
    energy = ((q*q).sum(axis=-1) * (np.int64(1) << (2*d))).sum(axis=1)
    result['energy'] = int(energy.max())
    return result

def draw(rng, kind, shape):
    if kind in ('gaussian', 'gradient_1e-5'):
        return rng.normal(size=shape) * (1e-5 if kind == 'gradient_1e-5' else 1)
    if kind == 'student_t3':
        return rng.standard_t(3, size=shape) / np.sqrt(3)
    if kind == 'outlier_100_p0005':
        x = rng.normal(size=shape)
        x[rng.random(size=shape) < 0.0005] *= 100
        return x
    x = rng.normal(size=(shape[0], shape[1]//32, 32))
    return np.ldexp(x, rng.integers(-8, 9, size=x.shape[:2])[..., None]).reshape(shape)

for k in (4096, 16384):
    for kind in ('gaussian', 'gradient_1e-5', 'student_t3',
                 'outlier_100_p0005', 'block_range_16'):
        trials = []
        for seed in range(1000, 1003):
            rng = np.random.default_rng(seed)
            a = certificate(draw(rng, kind, (256, k)))
            b = certificate(draw(rng, kind, (256, k)))
            p = None if a['energy'] is None or b['energy'] is None else a['energy']*b['energy']
            trials.append(dict(energy_pass=p is not None and p <= 2**48,
                               bound_fraction=None if p is None else round(math.sqrt(p)/2**24, 4),
                               A=a, B=b))
        print(json.dumps(dict(kind=kind, rows=256, K=k, trials=trials)), flush=True)
PY
```

## 必须独立验证的用例

- 证书通过时，与原 MX32 oracle 逐位比较，包括完整 16 个 nibble 编码、不同非均匀 scales、K 尾部、非对齐输入视图及全部 F/D/W 布局。
- `U*V=2^48` 的边界、刚超过边界、能量累加饱和、跨度 12/13、lambda -126/103 以及外侧一格。
- 一个无需 mock 的能量边界：M=N=1、K=262144，两边全 q=8、统一 scale，则 U=V=`2^24`，归一化点积也为 `2^24`。再增加一个 q=1 的非零位置，两边能量成为 `2^24+1`，必须拒绝。两边统一 scales 65/65、179/180 分别覆盖 lambda=-126、103。
- `q=0` 和负零编码 8；零块 scale 小于 active-row min；全零行；零块上的 scale255；非零块上的 scale255。
- 上述 `[2^24,1,-2^24]` 反例必须回退；能量不通过不能通过放宽误差容忍来绕过。
- 报告实际通过/回退次数、证书与回退的全部时间、额外显存、cuBLAS 版本和输出比较方式。CPU 分布实验不能替代 V100 kernel 验证。

## 官方一次来源

- [NVIDIA PTX ISA 8.8 / CUDA 12.9](https://docs.nvidia.com/cuda/archive/12.9.0/parallel-thread-execution/index.html)：`wmma.mma` / `mma` 的 FP16 乘法至少采用 single precision，FP32 accumulator 的中间累加至少采用 single precision；累加顺序、舍入和次正规输入处理未指定。本证书通过整数精确性与正常数区间避开这些自由度。
- [NVIDIA cuBLAS 12.9](https://docs.nvidia.com/cuda/archive/12.9.0/cublas/index.html)：`cublasComputeType_t` 中 `CUBLAS_COMPUTE_32F` 的计算和中间存储精度约定，`cublasMath_t` 的 reduced-precision-reduction 控制，以及 Tensor Core 使用条件。
- [OCP Microscaling Formats MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf/)：E2M1 与 E8M0 存储格式。本文的量化配方、MX32 顺序舍入合同和快速路径证书是本项目显式定义，并非 OCP 格式自动规定。
