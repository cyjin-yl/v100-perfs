# Qwen3.8-27B 工具调用 bench 对比（2026-08-22）

修复「工具调用约束故障族」（A 参数值退化/空值、B 工具名被改写、C 调度器死锁）并上线
字符级语法推进器后，对三个量化档 + Claude Code 客户端做 `file_roundtrip` agentic bench。

## 结果

| 模型 / 客户端 | runner | 权重 | 加载耗时* | 任务执行 wall_s | 冷启动总耗时* | verified | error_marks |
|---|---|---|---|---|---|---|---|
| **ud-q5**（Unsloth UD-Q5_K_M） | opencode | 19.27 GiB | 219s | 186.7 | ~406s | ✅ | 8 |
| **iq4xs**（自家 imatrix IQ4_XS） | opencode | 15.96 GiB | 261s | **116.1** | ~377s | ✅ | 8 |
| **cyber-q5**（外部 Q5_K_M） | opencode | 20.81 GiB | 300s | 201.7 | ~502s | ✅ | 8 |
| **claude**（ud-q5） | claude | 19.27 GiB | 219s | **85.6** | ~305s | ✅ | **0** |

\* 加载耗时为**温页缓存**下的权重加载时间（模型之前已载过）；完全冷启动会更久。
「冷启动总耗时」= 加载 + 任务执行，供"从零拉起一个可用后端"的参考。

## 观察

- **任务执行最快是 claude(ud-q5) 85.6s**，且 `error_marks=0`——Claude Code 用 `Write`
  写 `42` 时没有触发 opencode 那个「write 把 `42` 解析成数字 → schema 报错 → 改走
  bash `printf`」的重试（opencode 三个模型都因此带 8 个 error_marks）。这是**客户端
  工具语义差异**，不是模型差异。
- **opencode 三档里 iq4xs 执行最快（116.1s）**，cyber-q5 最慢（201.7s）。三者
  error_marks 都是 8（同一个 write 重试），差异主要来自模型生成/重试节奏。
- 加载耗时随权重体积大致单调（iq4xs 最小却因冷页未命中比 ud-q5 略久，属磁盘
  I/O 抖动）。

## 本轮同时验证

- **Claude Code 归因头剥除（真实客户端端到端）**：bench 用临时空 `CLAUDE_CONFIG_DIR`
  把 claude 指到本网关（不加载/不修改用户 `~/.claude` 里指向外部端点的配置）。
  不设 `CLAUDE_CODE_ATTRIBUTION_HEADER=0`，claude 默认会发那行每请求都变的
  `x-anthropic-billing-header`；网关在渲染前剥掉，`/health` 的
  `cc_attribution_stripped` 由 0 增到 6 —— 证明真实 Claude Code 流量下前缀缓存
  前缀保持稳定。（用户真实 `~/.claude/settings.json` 已设
  `CLAUDE_CODE_ATTRIBUTION_HEADER=0`，日常本就关闭归因头。）
- **A/B/C 修复回归**：`verify_abc.sh` 双端点（OpenAI + Anthropic）PASS=9 FAIL=0；
  真实 opencode 会话并行双 bash 成功；真实 Claude Code 会话完成工具任务。

## 复现

```
cd /run/media/ezra/13D010B6FDBC1A06/1CatVLLM
# 单个模型（先加载该模型的 profile，再）
python3 v100-perfs/benchmarks/fastllm/agentic_toolcall_bench.py \
    --sandbox /tmp/agentbench_<name> \
    --runner opencode --model catvllm/qwen3.8-27b \
    --tasks file_roundtrip --timeout 600 \
    --output v100-perfs/results/toolcall_bench_2026-08-22/<name>_file_roundtrip.json
# Claude Code（runner=claude；会用临时空 CLAUDE_CONFIG_DIR 指向本网关）
python3 ... --runner claude --model qwen3.8-27b --tasks file_roundtrip ...
```

## 备注

- 本机 `omp` CLI 无模型配置（`~/.omp` 无 models.yml），故未用 `omp` runner 跑；
  Anthropic 协议路径由 `verify_abc.sh`（curl 打 `/v1/messages`）与 Claude Code 覆盖。
- 详细单模型原始结果见同目录各 `*_file_roundtrip.json`。
