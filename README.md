# V100-32GB LLM 推理性能实测

在单块 Tesla V100-PCIE-32GB (SM70) 上对 **Qwen3.6-27B** 混合注意力模型进行的多引擎性能基准测试。

覆盖 vLLM (1Cat-vLLM 1.2.0/1.2.1)、llama.cpp（上游 + TurboQuant fork）、FastLLM 原生 TurboQuant/paged KV、推测解码（DFlash/MTP）、多种量化级别和 KV cache 压缩方案。

## 硬件

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA Tesla V100-PCIE-32GB (compute capability 7.0 / SM70) |
| 驱动 | 580.159.04 |
| VRAM | 32 GB HBM2 |
| 系统 | Fedora Linux 44, kernel 6.19.14 |

## 模型

| 项目 | 值 |
|------|-----|
| 模型 | Qwen3.6-27B (z-lab) |
| 架构 | 64 层 = 16 full_attention + 48 linear_attention/GDN (混合注意力) |
| 原生上下文 | 262,144 tokens |
| MTP 层 | 内置 1 层 (`mtp_num_hidden_layers: 1`) |

测试过的权重格式:

| 格式 | 文件 | 大小 |
|------|------|------|
| AWQ 4-bit | `Qwen3.6-27B-AWQ` | 21 GB |
| GGUF Q3_K_M | `Qwen3.6-27B-Q3_K_M.gguf` | 13.6 GB |
| GGUF Q6_K | `Qwen3.6-27B-Q6_K.gguf` | 22.5 GB |
| GGUF IQ4_XS (含 MTP) | `Qwen3.6-27B-IQ4_XS.gguf` | 12.2 GB |
| GGUF IQ2_M | `Qwen3.6-27B-IQ2_M.gguf` | 10.9 GB |
| DFlash draft | `dflash.gguf` | 3.46 GB |
| mmproj (视觉) | `mmproj-BF16.gguf` | 0.93 GB |

## TL;DR 结论

| 场景 | 最佳配置 | 单 agent tok/s | 上下文 | 多模态 |
|------|---------|---------------|--------|--------|
| **速度优先 (短上下文)** | llama.cpp TurboQuant + IQ4_XS + MTP | **41.5** | 65K | yes |
| **平衡 (中等上下文)** | llama.cpp Q3_K_M + DFlash + q8_0 KV | **48–51** | 65K | yes |
| **4 agent × 140K 上下文 (生产)** | llama.cpp TurboQuant turbo3, 4 slots | **~30** (1-2 活跃) / 5.5 (4 并发) | 4×140K | yes |
| **vLLM 兼容** | 1Cat-vLLM 1.2.1 AWQ + fp8 KV | **21.4** | 98K | yes |
| **vLLM 高并发** | 1Cat-vLLM 1.2.0 AWQ | **60.5** (5 并发合计) | 20K | no |
| **FastLLM 短上下文混合调度** | IQ4_XS + 单请求 MTP2 / 多请求 plain batch + FP8 KV | **50.85** (单请求) / **81.13** (4 并发合计) | 262K 共享池 | 未验证 |
| **FastLLM FP8 多并发容量** | IQ4_XS + 8K chunk + long-prefill interleave | — (4×约 52K，合计 207,807 prompt tokens) | 262K 共享池 | 未验证 |
| **FastLLM 单序列 exact 256K 容量** | FastLLM IQ4_XS + MTP9 + FP8 E4M3 KV | — (冷 261,888-token prefill) | 256K | 未验证 |
| **FastLLM 双序列 exact 256K 容量** | FastLLM 原生 Turbo3：IQ4_XS + K Q8_0 / V Turbo3 + MTP0 | —（双路冷 exact-window 容量验证） | **2×256K** | 未验证 |

> 4×140K 配置在真实 agent 负载中, agent 通常是轮流工作而非同时生成。1-2 个 agent 活跃时单 agent 约 30 tok/s; 只有 4 个 agent 同时在 140K context 上并发解码时才会降到 5.5 tok/s (attention 计算量随 context 平方增长)。这个配置的核心价值是 "4 个 agent 各自拥有 140K 上下文", 这是单 V100-32GB 上能达到的最大并发 × 上下文组合。

> FastLLM这一行是exact-window容量验证,不是与多轮steady-state tok/s的直接排名:256K请求以261,888-token冷prompt + 256-token decode完成,E2E 745.467 s,采样显存峰值31,088 MiB。完整worktree/git历史审计显示:既有native split-KV是5月已进入上游的FastLLM baseline,不是本次FlashInfer-SM70移植;当前PR真正落地的是`b693ad8`的Flash-Attention-V100式GQA6 XQA,其单层microbenchmark比原route快2.22×/3.37×/4.03× (8K/32K/128K KV)。FlashInfer-SM70 backend/prefill kernel未进入当前worktree,不声明其FastLLM性能。长上下文artifact早于最后的XQA commit,最终分支仍需重跑完整exact-window矩阵;详见[`docs/fastllm_benchmark.md`](docs/fastllm_benchmark.md)。

> FastLLM 的 **FP8 速度 profile** 保留单请求 MTP2，并用 `FASTLLM_QWEN35_BATCHED_MTP=0` 让多请求走 plain batched decode；短 decode 的 4 并发 aggregate 从 batched MTP2 的 47.57 tok/s 提升到 81.13 tok/s，同时单请求保持 50.85 tok/s。`--batch 5` 同时传给 HTTP worker 和模型 scheduler；运行时 route 为 `chunk=8192, decode_lanes=4, resident_lanes=4`。当前 `:8002` 生产服务则是下述 Turbo3 容量 profile。

> FastLLM 的 262K 只指 FP8 速度配置的总共享池，不是每请求容量：正式 C4 结果为 4×约 52K，合计 207,807 prompt tokens，峰值 29,150 MiB；更高的 4×约 60K 仅保留“服务端四请求完成、无 OOM/SIG”的上界证据。FastLLM 原生 Turbo3 容量配置改为 524,288-token 共享池，并已从干净进程用两个不同首 page 的 prompt 完成 **2× exact 256K**：每路 usage 均为 261,888 prompt + 256 completion = 262,144，batch wall 1,711.95s，峰值 28,798 MiB。Turbo4 的 llama.cpp 256K 多并发结果实际使用 K=`q8_0`、V=`turbo4`。

> FastLLM 当前生产链路为 Thinking Proxy `:8000` → FastLLM `:8002`，运行 Turbo3 容量 profile：`--kv_cache_dtype turbo3 --tokens 524288 --batch 2`，并显式设置 `FASTLLM_QWEN35_TURBO3_KV=1`。16 个 full-attention 层使用 K=`Q8_0_KV`、V=`TURBO3_KV`；48 个 linear-attention/GDN 层保持 FP16 recurrent state。首版 packed attention 按 bounded chunk 解压到 FP16 后复用 cuBLAS，因此这是容量/正确性结果，不是低比特直接计算的吞吐声明。

### V100 上的核心限制

**AWQ 4-bit 权重在 SM70 上实际占用 ~27 GB** (而非理论 4-bit 的 ~14 GB), 因为 V100 没有原生 4-bit 计算, 权重以半解包形式驻留 VRAM。这导致:

- AWQ + fp8 KV: 仅剩 ~3 GB 给 KV cache, 单序列上限 ~98K
- AWQ + DFlash draft (3.5 GB): 直接 OOM
- AWQ + MTP draft (~2 GB): KV cache 仅剩 1 GB, 上下文降到 ~3K

**GGUF 路线绕过了这个问题**: Q3_K_M 权重仅 ~14 GB, 给 KV cache 和 draft model 留出 ~16 GB。llama.cpp + GGUF 是 V100 上的最优路径。

## 目录结构

```
.
├── README.md                           # 本文件
├── thinking_proxy.py                   # Thinking Proxy: API 翻译 + 认证 + 进程管理
├── docs/
│   ├── performance.md                  # 完整性能报告 (19 个章节, 含所有实测数据)
│   ├── fastllm_benchmark.md            # FastLLM 混合调度、长上下文、MTP 与 SM70 XQA 中文报告
│   ├── HANDOFF_fastllm.md              # FastLLM 上游 PR 与部署交接状态
│   └── EXPERIENCE.md                   # 经验总结: SM70 踩坑、构建、调优
├── scripts/
│   ├── start_proxy.sh                  # Thinking Proxy 启动 (管理 llama-server)
│   ├── start.sh                        # vLLM 1.2.0 生产 (51K ctx, multimodal)
│   ├── start_120.sh                    # vLLM 1.2.0 纯文本 (20K ctx)
│   ├── start_121.sh                    # vLLM 1.2.1 (98K ctx, fp8 KV)
│   ├── start_llama.sh                  # llama.cpp + DFlash 生产 (65K, multimodal)
│   ├── start_llama_nospec.sh           # llama.cpp 无推测 (65K, multimodal)
│   ├── start_llama_turboquant.sh       # TurboQuant 通用启动器 (env 可配置)
│   └── start_llama_turboquant_4x140k.sh # TurboQuant 4×140K 生产配置
├── benchmarks/
│   ├── agent_bench.py                  # 多 agent 基准测试 harness
│   ├── run_agent_bench.sh             # 串行多引擎测试 runner
│   ├── vllm-120/                       # 1.2.0 结果 (1/2/4 agents)
│   ├── vllm-121/                       # 1.2.1 结果 (1/2/4 agents)
│   ├── llama-nospec/                   # llama.cpp no-spec 结果
│   ├── fastllm/
│   │   └── results/                    # FastLLM exact-window 与 XQA JSON artifact
│   └── turboquant/
│       ├── results/                    # 所有 TurboQuant JSON 结果
│       └── logs/                       # 所有 TurboQuant 服务器日志
└── chat_templates/
    └── qwen3.6_merged.jinja            # 合并的 Qwen3.6 chat template (支持 thinking 开关)
```

## 如何复现

### 前提条件

1. **V100-32GB** (或其他 SM70 GPU), 驱动 >= 550
2. **CUDA 12.8/12.9 工具链** (系统 CUDA 13.x 不支持 SM70)
3. 模型权重 (按上方表格自行下载)
4. llama.cpp 编译 (见 `docs/EXPERIENCE.md` 的构建章节)

### 运行基准测试

```bash
# 1. 启动要测试的引擎, 例如 TurboQuant 4×140K
./scripts/start_llama_turboquant_4x140k.sh &

# 2. 等待 health
curl http://127.0.0.1:8000/health

# 3. 运行多 agent 基准测试
python benchmarks/agent_bench.py \
  --endpoint http://127.0.0.1:8000 \
  --model qwen3.6-27b-awq \
  --num-agents 4 \
  --max-model-len 143360 \
  --shared-prefix-tokens 64512 \
  --turns 25 \
  --max-tokens 256 \
  --compact-ratio 0.6 \
  --warmup-turns 1 \
  --output results.json
```

### 运行完整串行对比

```bash
# 依次启动 vLLM 1.2.0, 1.2.1, llama.cpp no-spec 并测试 1/2/4 agents
./benchmarks/run_agent_bench.sh /tmp/results
```

> **重要:** 必须串行测试, 每次只跑一个引擎。并发测试会导致 GPU 资源争抢, 数据失真。

## 指标说明

| 指标 | 定义 |
|------|------|
| **decode tok/s** | `completion_tokens / (last_token_time - first_token_time)`, 仅 token 生成阶段 |
| **e2e tok/s** | `completion_tokens / total_latency`, 含 TTFT + 网络开销 |
| **total throughput** | `total_completion_tokens / wall_time`, 所有 agent 合计 |

> `decode tok/s` 不应与浏览器端 OpenAI 流吞吐直接比较。V100 首次请求有 JIT warmup, 因此每个 agent 的第 1 轮被丢弃, 仅统计 steady-state。

## License

测试数据和脚本供社区参考, 按 MIT 许可发布。模型权重版权归各自所有者。
