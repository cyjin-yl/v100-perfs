# 上游 cherry-pick 评估（2026-08-22，merge-base af23441b）

状态：HEAD=831476ab（本地领先 43 / 上游新提交 133... 实测 upstream-only = **43**，本地未推送提交另计）。
上游 fetch 时间：2026-08-22 06:53（当日已 fetch，无需再拉）。

## 上游 43 个新提交的主题分布

| 主题 | 数量 | 与本栈相关性 |
|---|---:|---|
| DFlash（新投机解码框架，15 连） | 15 | 中——思想可借鉴，代码绑 DFlash 模型路径 |
| Dots3/Dots3-Note（新模型+DSA 稀疏注意力） | 14 | 低——`dots3_note.cpp/-kernels.cu` 是独立模型文件 |
| NUMA FP8 混合 MoE | 7 | 低——我们单路 CPU 无 NUMA 设备 |
| DeepSeek-V4 CUDA Graph/双卡修复 | 3 | 低——V4 模型专属 |
| 杂项（YaRN 回归测试、Qwen3.6 双卡文档、八令牌 MTP 快照、FP8 投影调度、V4 路由缓存） | 7 | 见下表 |

## 干跑 cherry-pick 冲突评估（git merge-tree，全部干净）

| 提交 | 内容 | touched files | 建议 |
|---|---|---|---|
| `161962de` 八令牌投机验证快照 | MTP fast-seq 上限 6→8；conv1d/GDN 内核加 snap5/snap6 参数 | fastllm-cuda.cu(+15/-5), qwen3_5.cpp(2) | **值得 pick**：与我们的 MTP=2 路径直接相关，干跑无冲突。但注意本地 qwen3_5.cpp 有 +6199/-670 魔改、fastllm-cuda.cu +1189/-137——文本干净不代表语义干净，pick 后必须跑工具调用探针 + MTP 接受率对账 |
| `d7c5672e` 六令牌 FP8 投影调度 | linear-fp8.cu +16 | 独立 | 可 pick，低风险低收益（我们不用 FP8 投影） |
| `df38398b` YaRN 数值回归测试 | regressionOps.cpp +16/-4 | 测试 | 可 pick，零风险 |
| `05e55618` V4 分段 CUDA Graph 解码 | deepseekv4 专属 +626 行 | V4 | 不 pick（无此模型）；CUDA Graph 分段思想留给未来 decode 优化 [INFERENCE] |
| `5909ced0` V4 路由缓存析构开销 | fastllm.h/fastllm.cpp 各 +4 | 半通用 | 缓：改的是 V4 分支内部逻辑，收益不覆盖 Qwen3.8 |
| `e939c9b8` V4 双卡串行解码修复 | deepseekv4.cpp | V4 | 不 pick，但其"双卡串行解码"教训对我们 TP2 规划有参考价值 |
| `6136ba45` Qwen3.6 双卡性能分析文档 | docs +159 行 | 文档 | **pick**——正是我们 TP2 规划需要的实测数据 |

## DFlash/Dots3/NUMA 的处理策略

不整体 pick。理由：
1. DFlash 15 连是围绕一个新投机解码框架的整链改动（KV 压缩融合、门控融合、滑窗注意力），依赖其模型布局；
2. 但其中**通用算子**（如 `融合 KV 物化`、`直写 KV 缓存`）的思路与我们 turbo3/turbo4 KV 直读方向一致，值得作为 SM70 算子改进的设计输入，而非代码移植；
3. NUMA FP8 系列（852a96d2 等 7 个）动 executor.cpp/numasdevice.cpp，我们无对应硬件。

## 执行顺序建议

1. 先 pick `6136ba45`（纯文档，读 TP2 实测数据再定双卡 profile）
2. 再 pick `df38398b`（回归测试加固）
3. 最后 `161962de`（MTP 快照扩展），pick 后必须：编译 → 工具调用探针（逐字抄写+工具名）→ MTP 接受率对比（基线：95.1/86.8%）
4. DFlash/Dots3 标记为「设计参考」，在算子改进时人工借鉴
