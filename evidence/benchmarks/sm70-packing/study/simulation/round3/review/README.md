# Round 3 独立审查与隔离 CPU 验证

本目录包含对新数学方案、SM70 fragment/基线契约、输入准备成本及入库复跑方式的独立审查。所有新结果均来自 CPU；没有新的 V100 执行、CUDA/cuBLAS 调用、训练实验或实测速度。

## 1. 新方案的数学复核

### A 为 ternary，B 为完整 E2M1：两个独立 dot

令 `q=2x`，A 的两路 q 均取 `{-1,0,1}`，B 的两路 q 可取完整 E2M1 数值集合，采用：

`P=a0+512*a1`，`Q=b0+128*b1`，每 4 项解码一次。

输入绝对值至多 513 和 1548，均为可精确表示的 FP16 整数。两路目标 `L=Σa0*b0`、`H=Σa1*b1` 的绝对值不超过 48；高位干扰不超过 `48*(1+128+512)=30768<32768`，低位绝对值小于 64。因此取高 16 位系数与低 7 位平衡余数可恢复两路目标。所有绝对乘积之和不超过 3,176,496，适用于此前验证的精确整数域。

两路输出恢复后可各自乘回不同的块尺度。这扩大了某一侧的支持范围，但仍要求 A 为 ternary；没有实现两侧完整 E2M1 的通用打包。

### 只提取中间系数：同一个 dot 的两段 K

对 q 在 `{-3,…,3}` 的情况，反向打包 B：

`P=a0+512*a1`，`Q=b1+512*b0`。

目标变成中间系数 `T=Σ(a0*b0+a1*b1)`，两端系数只是干扰。它与“保留两个独立 dot”不同。

对 packed K=12，绝对乘积和不超过 28,422,252，小于 `2^25`。在作者 V100 BFMA4 的递归模型中，对齐网格至多为 2；每个有符号整数项对齐最多损失 1，包含 C 的五项至多 5，最后 RZ 再至多损失 1。因此每组误差保守界为 6，三组为 18。负号只改变误差方向，绝对界仍成立；这些输入是整数，不涉及 FP16 subnormal。

低端干扰和误差合计至多 `108+18=126<256`；高端提取的剩余量至多 `512*216+108+18=110718<131072`。所以即使 packed 结果的低位已经不精确，T 仍可正确恢复。这是针对所选模型和四项递归顺序的结论。

进一步无需先恢复高系数：

```text
u = int32(z) + 256
T = sign_extend_9((u >> 9) & 511)
```

这里右移须为有符号算术右移，或用相应 signed bit-field extraction。由于 `|T|≤216`，截低 9 位后符号扩展可恢复 T；z 在 int32 范围内且是整数，转换不引入新误差。返回 FP32 输出仍需转换与尺度运算，不能只把 BFE 单条指令当全部解码成本。

同事随后提供了独立 `../middle_decode/` 编译探针。review 核对了实际 SASS：同样返回 float 的原解码是 5 条算术指令、BFE 写法也是 5 条；unsigned 左移/有符号右移写法让编译器将偏置加法与左移融合为 LEA，变为 `F2I, LEA, SHF, I2F` 共 4 条。整数输出端点单独报告为 3 条，不能与 float 输出端点混用。系数盒枚举明确只接受能被 FP32 精确表示的整数 z：18,916,471 组全部三种解码零错误，另 4,855,662 组是违反输入表示前提的构造值，未假称已验证。这是具体的编译改进，仍没有测得硬件加速。

### 跨尺度限制有具体碰撞

`preparation_review.py` 检查了两组输入 `(a0,a1,b0,b1)=(1,0,1,0)` 与 `(0,1,0,1)`。两者上述 reverse-B packed 输出都为 512，中间系数都为 1；若两段的尺度乘积分别为 2、4，真实 E2M1 输出分别为 0.5、1。即使保留整个单一 packed 数值，也无法区分这个加权目标。

因此 middle-only 必须在相同尺度乘积内配对，或保留额外信息。当前 math 目录的 MX32 演示采用同一尺度块内两次 packed K8 解码；当前 CUDA harness 仍采用两独立输出方案，没有把 middle-only 冒充成已实现硬件内核。

## 2. CUDA 数据布局与基线审查

独立检查通过的内容：

- 用 host C++ 编译实际 `fragment_mapping.hpp`，32 个 lane、256 个输出坐标与 Python mapping 一致。
- 核对每 lane 两行/四列尺度复用的 256 个索引。
- 128 个 `float2` 输出对均在同一行、列相邻并满足 8 字节对齐。
- 对 M,N 为 `(16,48)`、`(32,16)`、`(48,48)`、`(80,32)` 的部分 CTA 网格验证无漏写、重复和越界。
- 使用非方阵 `(M,N,K)=(16,48,4),(48,16,36),(32,80,64)`，含两路独立变化尺度，重建 cuBLAS 存储契约并得到相同结果。

具体契约为：A 以 row-major `M×K` 存储，视为 column-major `K×M`；B 以 column-major `K×N` 存储。cuBLAS 使用 `(n,m,k)`、`opT(B)`、`opN(A)`，得到 `(AB)^T` 的 column-major 输出，即目标 `AB` 的 row-major 内存布局。两路 batch stride 为 `(M+N)*K`，C stride 为 `M*N`。

这验证的是代码所用的布局约定，不是动态执行真实 cuBLAS。CUDA device 函数是实验 harness 的内部组件，不是接受任意数值、尺度、尺寸的通用 SDK API。host 测试域显式限制候选字母表、shape，并只生成 E8M0 指数 `[-2,2]`，检查基线 FP16 反量化精确、有限。绕过 host 前置条件直接启动内核不在保证范围内。

`float2` 将对应 store 指令数量从 16 条变为 8 条，但每条存储从 32 bit 变为 64 bit，**字节数相同**。静态 SASS 数量不能代替吞吐或延迟。

## 3. 从真实 4-bit 输入开始算准备成本

`preparation_review.py` 对 A ternary/R512、B full E2M1/R128 分别构造 256 项 pair-to-half 表，所有允许项精确，每张表 512 字节，若使用 bit-packed 有效性掩码再加 32 字节。它穷举了 131,072 组物理字节对：A 的 65,536 组中只有 256 组满足 ternary 前提，B 的 65,536 组全部允许；所有允许组的 even/odd K 对应关系零错误。其余 A 编码由有效性掩码拒绝，没有视为可用输入。

当前布局中一个 byte 存同一路的相邻 K 位置；LUT key 却需组合两路同一 K 的 code。因此需要从两路各取 byte，再交叉拼出 even/odd 两个 key，不能声称“直接按原始 byte 一次查表就完成两路 packing”。该 LUT 是精确编码候选，不是实测优化；GPU 常量内存中的不同地址访问可能串行化，当前 harness 继续使用算术解码。

对全部 16 个 E2M1 编码、255 个有限 E8M0 指数的独立枚举还确认：**若要求所有 E2M1 值乘该尺度后写 FP16 都精确且有限，尺度指数范围恰为 `[-23,13]`。** 这只是单个输入反量化的范围，不证明完整 dot 的输出范围，也不覆盖 E8M0 NaN。采用 `[-2,2]` 的当前硬件实验前置条件处于该范围内。

令 `F=(M+N)*K` 为一路 A+B 元素数，S 为两路尺度字节数，O=`8*M*N` 为两份 FP32 输出，理想最少流量为：

| 实现边界 | 最少流量 |
|---|---:|
| 原始压缩输入→两路 FP16 临时矩阵→cuBLAS | `9F+S+O` |
| 原始压缩输入→一路 packed FP16 临时矩阵→候选 | `5F+S+O` |
| 假设融合读取压缩输入、片上反量化的基线 | `F+S+O` |

例如 M=N=128、K=256，前两者字节比约 1.566；但第三者与候选之比约 0.434。它们只是不同实现边界的字节模型。临时 packed half 为每原始元素平均 8 bit，仍大于原始 E2M1 的 4 bit；数据准备的优势依赖所比较的基线，不能从物化 FP16 基线的开销推导通用加速。实际 tiling 重读、缓存、scale 访问、量化和训练开销未计时。

## 4. 入库与可复跑性审查

| 发现 | 整合方式 |
|---|---|
| 多数历史脚本把 JSON/CSV/log 写到 `__file__` 同目录 | 只在隔离副本执行；保留历史原件 |
| `math/packing_exhaustive.py` 有 `--output`，其他脚本未统一 | wrapper 复制精确白名单，避免为改 I/O 而修改历史代码 |
| 历史结构搜索 JSON 约 10.24 MB，可机械重生 | 保留历史 ZIP；展开目录放脚本，小摘要与重跑日志即可 |
| `analyze_cost.py` 读取 SASS/resources 和 `.o` 做证据哈希 | 保留真实的小对象文件；不制造占位对象跳过检查 |
| 原架构构建环境依赖当地解压 SDK、绝对目标 symlink | 默认 CI 不构建模拟器；固定提交/命令留存，实际重放显式启用 |
| 本机模拟器二进制依赖 Linux 系统库 ABI | 可用 `--simulator-bin` 指定兼容构建；默认只检查生成器和已记录日志一致性 |
| 旧结果中的绝对 `argv` 是当时命令记录 | 保持不可变；新脚本执行不依赖这些路径 |
| 原模型与模拟器包含第三方代码 | 保留 upstream snapshot 的 `LICENCE`、`SNAPSHOT.json` 以及架构 licenses；不把第三方源码标成原创 |
| 单纯历史检查全绿不能证明新方案通过 | 本轮 manifest 显式加入新数学、fragment、SASS、准备、非方阵契约及 middle decoder 等七项检查 |

没有必要将整个 CUDA/Octave 安装、完整模拟器源码 clone、下载缓存或编译临时文件提交仓库。历史归档和实际用到的 trace 已足够保留证据链。原 `Soft-BF16-V100` 的入库内容仅为审计脚本/笔记和定位证据，没有重新发布其整个源码工作树。

## 5. 统一验证入口

最终仓库中的统一入口为 `tools/sm70_packing/run_cpu_validation.py`，不在本证据目录重复保存。默认只需 Python 3.10+、NumPy 和 C++17 编译器，实际测试环境为 Python 3.12、NumPy 2.3.5、g++ 14.2。默认无网络、无 CUDA/GPU、无 Octave，不会把日志解析称为新周期仿真。

```bash
python3 tools/sm70_packing/run_cpu_validation.py \
  --study-root evidence/benchmarks/sm70-packing/study \
  --extra-checks-manifest evidence/benchmarks/sm70-packing/study/simulation/round3/review/ci-checks.json \
  --require-round3 \
  --output-dir /tmp/v100-packing-validation-new
```

输出目录必须不存在，避免覆盖先前证据。省略 `--output-dir` 使用自动清理的临时目录，只在控制台返回摘要；需要检查完整失败日志时请指定独立输出目录。wrapper 顺序执行各项以控制峰值内存，保存日志和命令，最终逐项核对输入与历史结果的哈希没有变化。

`ci-checks.json` 声明本轮新增检查的输入、命令和 JSON 断言。`--require-round3` 在缺少 manifest 时失败，因此历史脚本通过不会把未验证的新研究标成全绿。可以通过增加显式 check 扩展后续实验，不能用空列表绕过。

可选 `--octave-bin /path/to/octave-cli` 执行原版模型，`--simulate [--simulator-bin /path/to/accel-sim.out]` 实际重跑已有 CPU 周期模拟器。它们均不安装依赖、不访问 GPU；只有显式启用后摘要才会标记相应步骤实际执行。

最终隔离执行的 16 项检查全部通过，其中 7 项为本轮新检查；宿主耗时约 10.20 秒，69 个源输入/历史结果文件哈希保持不变。该时间只是 CPU 验证流程耗时。独立数据准备及布局验证结果分别见 `preparation-review-results.json`、`layout-contract-results.json`。统一 runner 的执行摘要在 `validation-summary.json`，日志在 `logs/`；不需把隔离副本的 10 MB 可再生搜索 JSON 再提交一份。

## 来源与证据

- [OCP MX Formats v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf/)：E2M1/E8M0 定义与按块尺度组织 dot。
- `../math/round3_math.py`：新增充分条件、模型测试、预期失败例和 synthetic admission/repair 研究。
- `../cuda/fragment_mapping.hpp`、`mapped_packing.cu`、`compile-evidence.json`：实际编译目标、布局和静态指令证据。
- 历史 `simulation/numerics/upstream-snapshot`：固定提交与文件哈希校验的 V100 数值模型和公开 fixture。

本审查不为任意模型以外的 V100 数值路径作保证，也不从合成 trace、静态 SASS 或字节公式推算真实训练速度。
