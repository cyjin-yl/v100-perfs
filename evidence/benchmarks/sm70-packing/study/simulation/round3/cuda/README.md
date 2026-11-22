# Round 3：明确 fragment 布局的 V100 packing harness

## 当前状态

**CPU 布局与数值检查通过，实际 SM70 对象编译通过；V100 执行仍待完成。**
这里没有硬件时间或加速比。只有在真实 GPU 上运行 `run_hardware.sh` 后，程序才会生成计时字段。默认拒绝非 V100；通过 `--allow-other-gpu` 显式比较其他 GPU 时也保留 `actual_V100=false`。

这轮把此前的寄存器编译占位推进成了完整的输入、映射、缩放、输出及基线闭环。它仍是受限值域的实验实现，未覆盖全部 MXFP4 数据范围或训练收敛。

## 三个候选与运算目标

所有候选的目标都是两个独立乘积 `C0=A0*B0`、`C1=A1*B1`。它们不要求 B0=B1，也不把两个结果偷偷相加。

令 `q=2×E2M1值`：

| 候选 | A 两路 q 范围 | B 两路 q 范围 | A packing | B packing | 每次解码 K |
|---|---|---|---|---|---:|
| ternary_k32 | {-1,0,1} | {-1,0,1} | a0+256a1 | b0+256b1 | 32 |
| seven_k4 | {-3,…,3} | {-3,…,3} | a0+256a1 | b0+256b1 | 4 |
| ternary_full_k4 | {-1,0,1} | 完整 E2M1 q 集合 | a0+512a1 | b0+128b1 | 4 |

完整 q 集合是 `{0,±1,±2,±3,±4,±6,±8,±12}`。两路符号零按数值零处理。

**完整 E2M1 × 完整 E2M1 在这三个 packed 路径中都明确拒绝；没有裁剪、静默改数据或未经验证的 fallback。** 硬件 harness 只生成、验证声明域中的输入。单独调用 CUDA kernel 的开发者必须保持相同前置条件。

后两个 K4 候选复用 `mma.sync.aligned.m8n8k4`。不对称候选的低结果取中心化的 7 个低位，高结果按 16 位提取；中间交叉项不需要分别恢复。

## 输入是真正的 nibble 与块尺度

输入在内存中的布局为：

```text
codes  : uint8 [2 streams][M+N rows][K/2 bytes]
scales : uint8 [2 streams][M+N rows][ceil(K/32)]
```

每个字节的低、高 nibble 分别对应同一 stream 中相邻的两个 K 元素。前 M 行是 A 的行；后 N 行是 B 的列，因此 A 使用 row-major，B 使用 column-major，二者沿 K 都连续。E2M1 代码通过整数运算解码，没有假设常量查表在随机索引下免费。

E8M0 字节使用 bias=127。两路 A、B 的 scale 分别生成，可互不相同。每 32 项的缩放沿各自 K 轴：

1. packed half 只装原始 q，先不乘两路不同的 scale。
2. K4 候选每四项解码，再在当前 32 项 MX 块内累加两个结果。
3. 块结束后分别乘 `scale_A0*scale_B0`、`scale_A1*scale_B1`。
4. 两个结果分别累加到最终矩阵。

这避免了跨块尺度混淆。测试尺度指数明确限制在 **[-2,2]**，并检查基线半精度反量化完全精确、有限。没有声称覆盖 E8M0 全范围、NaN scale、溢出或下溢。K=4 的逻辑短尾只使用块内前四项，其他项视为零；其数学尺度沿用 32 项块定义，不宣称这种紧凑尾块是标准文件序列化格式。

## m8n8k4 的明确映射

官方定义中，一 warp 并行执行四份 8×8×4 MMA。每组包含 lane 0–3 与 16–19，随后是 4–7 与 20–23，以此类推。这里把四组放在一个 16×16 输出 tile 的四个象限。

`fragment_mapping.hpp` 保存真正 CUDA kernel 使用的映射。核心公式为：

```text
group = (lane >> 2) & 3
quadrant_row = (group >> 1) * 8
quadrant_col = (group & 1) * 8

A local row = (lane & 3) + (lane >= 16 ? 4 : 0)
B local col = (lane & 3) + (lane >= 16 ? 4 : 0)

C local row(i) = (lane & 1) + (i & 2) + (lane >= 16 ? 4 : 0)
C local col(i) = (i & 4) + (lane & 2) + (i & 1)
```

每线程 A、B 各读一个连续 `uint2`，即四个 half。输出八个 FP32 槽通过上述坐标写回。`fragment-map.json` 包含全部 32 lane 的具体坐标。[NVIDIA PTX ISA 的 m8n8k4 FP16 fragment 定义](https://docs.nvidia.com/cuda/archive/11.4.0/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-884-f16)

CPU 检查不是只做加载/存储往返：它把各 lane fragment 重建为四份独立 A/B 矩阵，计算乘积，再按输出映射写回。另以主机 C++ 编译实际 header，与独立 Python 公式逐项比较，防止测试模型和 CUDA 头文件发生漂移。

## 这一轮有编译证据的改进

### 相邻输出用 float2 写回

输出槽 `i` 与 `i+1`（i=0,2,4,6）在同一行、相邻列。N 是 16 的倍数，因此地址满足 float2 对齐。

实际 SM70 SASS 已确认：

| 同一个 mapped kernel 的版本 | 每 warp 输出指令 |
|---|---:|
| 标量写回 | 16×STG.E.SYS |
| float2 写回 | 8×STG.E.64.SYS |

输出字节数相同。这个结果证明存储指令减半，**没有证明总时间减半**。硬件 harness 会把两版纳入同一批计时。

### 三种精确解码使用同一输入、同一目标

- mode 0：固定常数浮点解码，包含取整和交叉项消除。
- mode 1：F2I 后移位/掩码，再转换回浮点。
- mode 2：限定范围内的 FP32 位模式解码，绕开转换和 FRND。

实际编译的全部 mode 2 内核都没有 FRND、F2I、I2F。整数和地址指令依然有代价，当前没有提前把任一版本评为最快。

### 不重复处理块尺度

K4 路径先合并块内结果再乘 scale；每个 lane 的八个输出只涉及两行、四列，源码预取两路共 12 个 scale 字节并复用，避免源码层面做 32 次重复加载。编译器与缓存实际实现仍需硬件分析。

通用长 K、scale 和动态指针也增加寄存器成本。当前 magic 版本：

| 候选 | scale 全一 | 独立块尺度 | spill |
|---|---:|---:|---:|
| ternary K32 | 62 registers/thread | 64 | 0 |
| seven K4 | 64 | 72 | 0 |
| asymmetric K4 | 64 | 72 | 0 |

不要把它与前一轮固定 K 的 40 寄存器小探针直接作速度比较。

## 验证结果

`validate_fragments.py` 已实际执行并通过：

- 1024 组 impulse 乘积，覆盖每个 16×16 输出位置和 K4 中的每个位置。
- 44 组配置，包含三个候选、长/短 K、全一/不同块尺度、随机/极值输入。
- 每组分别验证 float、integer、magic 三种解码，合计 67,584 个输出值比较，零错误。
- nibble 编码/解码、半精度输入精确性、基线有限性与精确性检查。
- 三个候选均拒绝 full E2M1 × full E2M1。
- 真实 CUDA header 经主机 C++ 编译后的映射与 Python 模型一致。

这些是 CPU 布局/有界算术证据。选择的整数和块尺度结果在 FP32 可精确表示范围内，未声称模拟了所有 Volta 数值边缘行为。GPU 指令是否按预期工作仍有专门硬件入口。

CUDA 12.8.93 的 `sm_70` 编译已完成，共 24 个 mapped kernel 和 2 个数据准备 kernel，全部零 spill。`compile-evidence.json` 包含源码、对象、SASS 的哈希和 opcode 信息。带循环的 SASS 静态行数不是动态指令数或运行周期。

## 硬件基线与计时边界

基线从同一 nibble/scale 输入反量化到 half，再通过**一次 cuBLAS strided-batched 调用完成两份独立 GEMM**。输出数量与候选一致。C 使用 row-major，cuBLAS 按 Cᵀ 计算，参数为 `opT(B_colmajor) × opN(A_rowmajor视作K×M)`。

候选从同一输入准备 packed half，再执行融合解码/缩放/输出 kernel。程序分别记录：

```text
baseline dequant preparation
baseline cuBLAS compute
baseline dequant + cuBLAS total
candidate pack preparation
candidate compute
candidate pack + compute total
```

所有输出先逐项与独立 CPU 参考比较；cuBLAS 也必须对同一参考通过。错误时退出，不保留有效加速宣称。

令 R=M+N、S=2R×ceil(K/32)，只按每个矩阵读写一次计算最低逻辑流量：

| 从相同 nibble 输入开始 | 最低逻辑字节数 |
|---|---:|
| 物化 half 后调用 cuBLAS | 9RK+S+8MN |
| 物化 packed half 后融合计算 | 5RK+S+8MN |

这解释了为什么要把数据准备放入总时间。实际缓存复用、重复加载和 memory transaction 会改变结果。更先进的“直接在算子内部反量化压缩输入”基线不需要物化 half，也不在当前 cuBLAS 比较范围内。packed half 仍比原始 nibble 输入大两倍，不能把它说成真正原生 FP4 内存格式。

## 运行

CPU 验证需要 NumPy 与主机 C++ 编译器：

```bash
python3 validate_fragments.py
```

本地已有私有 CUDA SDK 时可只编译，不执行 GPU：

```bash
V100_COST_TOOLCHAIN=/path/to/toolchain ./compile_only.sh
```

普通完整 CUDA 12.x 环境可直接编译对象：

```bash
nvcc -O3 -std=c++17 -lineinfo -Xptxas -v \
  -gencode arch=compute_70,code=sm_70 \
  -c mapped_packing.cu -o mapped_packing.o
```

在真实 V100 上：

```bash
./run_hardware.sh --device 0 --correctness-only
./run_hardware.sh --device 0 --warmup 10 --iters 100
```

脚本不安装环境、不停止服务、不修改时钟。默认只用一个 GPU，基准尺寸 M=N=128、K=32/256，显存规模很小；请在空闲卡上采集可解释的时间。

## 与中间系数路线的区别

数学搜索还提出把同一个 dot 的两个 K 位置合并，只保留多项式中间系数。这可能更接近常规 GEMM，然而其输出目标、有效 K、尺度配对和基线都与本文不同。这里没有把那条路线的结果混进两个独立 GEMM 的比较，也没有把中间系数可恢复误写成两路结果都可恢复。
