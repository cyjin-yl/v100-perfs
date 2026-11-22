# V100 / SM70 packed low-precision arithmetic study

研究入口：[结论、适用范围与复跑方法](../../../../docs/research/performance/sm70-packing/index.md)。本目录提供可读源码和实验记录，两个原始交付 ZIP 位于 [evidence/results](../../../results/sm70-packing/2026-11-22/)。

已执行的工作包括 CPU 数学实验、公开 V100 硬件样本重验、原版 Octave 数值模型交叉验证、CUDA 编译和 Accel-Sim 周期模型。**尚未在用户 V100 上执行新的 kernel，也没有训练速度或收敛结果。**

## 从仓库根目录复跑

需要 Python 3.10+、NumPy 和 C++17 编译器。以下入口会复制必要输入到新的输出目录，保护仓库内的历史证据；默认不联网、不安装工具、不执行 GPU 或模拟器二进制：

```bash
python3 tools/sm70_packing/run_cpu_validation.py \
  --require-round3 --random-cases 4096 \
  --output-dir /tmp/v100-packing-validation
```

输出目录必须尚不存在。`summary.json` 逐项记录输入哈希、运行命令、结果和证据类型。数值模型的原版最小快照和硬件 fixture 已包含，无需 Git 或下载。

如已有 GNU Octave，可额外传入 `--octave-bin /path/to/octave-cli`。如要在兼容的 Linux x86-64 系统执行归档的 Accel-Sim，额外传入 `--simulate`；也可用 `--simulator-bin /path/to/accel-sim.out` 指定自己编译的兼容版本。这两项与默认 CPU 检查分开记录。

新的真实 V100 测试入口是 `simulation/round3/cuda/run_hardware.sh`。它需要支持 `sm_70` 的 CUDA 12.x 开发环境及 cuBLAS；具体输入域、尺度限制、计时范围和命令见该目录 README。编译通过不代表已执行。

## 内容和历史边界

| 路径 | 内容 |
|---|---|
| `initial-assessment.zh-CN.md`、`math/`、`audit/` | 第一轮数学搜索、Soft-BF16-V100 固定版本审阅及 CPU 反例 |
| `simulation/README.zh-CN.md` | 第二轮原始报告，保留当时的结论与命令记录 |
| `simulation/numerics/` | V100 模型、5,000 个公开硬件 fixture、原版模型快照及交叉验证 |
| `simulation/search/` | 21,504 组充分条件筛选的可重跑脚本 |
| `simulation/cost/` | 旧候选的 CUDA 编译、SASS、位模式解包测试 |
| `simulation/architecture/` | 固定版本 Accel-Sim、公开 LUD trace、人工 HMMA trace、配置、日志、许可证 |
| `simulation/round3/` | 本轮新增候选、显式 fragment 布局、MX 块尺度处理和独立审阅 |
| `gpu_probe/`、`compile-evidence/` | 第一轮硬件探针、工具链兼容补丁和编译记录 |

历史报告中的工作区绝对路径是实验记录。便携入口以上述仓库根目录命令为准。直接运行旧脚本可能覆盖其同目录结果，因此建议使用隔离 runner。

10 MB 的 `structural-search-results.json` 保存在原始第二个 ZIP 中，本可读目录不重复展开；其脚本可重建该文件。没有包含完整 CUDA 或 Octave 安装，也没有复制整个上游 Git 工作树。上游模型、公开数据和模拟器的固定提交及许可证保留在对应目录中。

`SHA256SUMS` 记录本可读实验目录的文件哈希；两个历史 ZIP 的独立哈希见 `archives.json`。
