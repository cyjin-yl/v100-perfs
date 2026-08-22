# V100 32G PCIe · FastLLM 本地推理性能与运维

> 主力部署：**Qwen3.8-27B**（混合注意力，16 full-attention + 48 GDN/linear + MTP），
> FastLLM C++ 推理引擎 + thinking_proxy 网关，单张 **Tesla V100-PCIE-32GB (SM70)**。
> 历史评测覆盖 Qwen3.5 / 3.6（vLLM、llama.cpp、FastLLM 多引擎对比），已归档至
> [`docs/archive/qwen36/performance/`](docs/archive/qwen36/performance/) 与 [`docs/archive/qwen35/`](docs/archive/qwen35/)。

## 硬件

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA Tesla V100-PCIE-32GB (SM70), HBM2 |
| CPU | Intel i3-12100 (Alder Lake) |
| 主板 | ASUS TUF GAMING B660M-PLUS WIFI D4 |
| 系统 | Fedora Linux 44 |

**硬件路线图**：评估 PCIe 双卡 V100 TP=2（prefill 吞吐与并发容量翻倍，
decode 受 PCIe all-reduce 限制）；评估傲腾 NVMe 作 L3 prefix-cache 盘
（256G ≈ 369M tok，轮转 TTFT 从全冷 22.5s → 亚秒，见
[可行性测算](docs/research/storage/optane-l3.md)）。PMem DIMM 形态
B660 不支持，仅考虑 NVMe 形态。

## 当前生产配置 (Qwen3.8-27B)

| 项目 | 值 |
|------|-----|
| 权重 | Unsloth UD-Q5_K_M (19.3 GiB) / Cyber IQ4_XS imatrix (16.0 GiB) |
| 引擎 | FastLLM (`build-rw/apiserver`, SM70 CUDA) |
| KV cache | K=q8_0, V=turbo3 (packed, 三级缓存 L1/L2/L3) |
| MTP | drafts_per_step=2 |
| 上下文 | 262,144 tok 共享页池 |
| 并发 | batch=4, PROXY_STREAM_SLOTS=4 |
| max_tokens 默认 | 131072 |
| 网关 | `src/gateway/thinking_proxy.py` :8000 (Anthropic + OpenAI 双协议) |
| 管理 | apiserver 内嵌 WebUI `/admin` (AUTH_TOKEN 认证) |

关键实测（在线）：prefill ≈ 667 tok/s（128K 上下文），decode ≈ 14.8 tok/s
单流；前缀缓存命中率 >90%（agent 轮转场景）。

## 文档导航

```
docs/
├── concepts/              # 核心概念与工具调用约束
├── operations/            # 运维与排障
├── performance/           # 当前性能报告与方法
├── reference/             # API、配置与模型参考
├── research/              # 性能与存储研究
└── archive/               # 历史设计、交接与评测
evidence/
├── benchmarks/            # 基准脚本及其捕获结果
└── results/               # 原始扫描与工具调用证据
operations/handoff/        # 生产启动脚本 + profile + 运维脚本集
tools/scripts/             # 通用运行与探针脚本
runtime/                   # 本机运行时(profile/logs/prefix-cache, gitignore)
site/                      # 文档站(Astro + Proto-UI, GitHub Pages)
```

## 快速链接

| 主题 | 文档 |
|------|------|
| 工具调用约束(三级缓存+语法推进器) | [docs/concepts/tool-calling/constraint-architecture.md](docs/concepts/tool-calling/constraint-architecture.md) |
| 短 prefill 优化计划(qLen 512–2048) | [docs/research/performance/short-prefill-plan.md](docs/research/performance/short-prefill-plan.md) |
| 傲腾 L3 可行性 | [docs/research/storage/optane-l3.md](docs/research/storage/optane-l3.md) |
| 上游 cherry-pick 评估 | [docs/archive/assessments/2026-08-22-upstream-cherry-pick.md](docs/archive/assessments/2026-08-22-upstream-cherry-pick.md) |
| Qwen3.8 扫描矩阵 | [docs/performance/qwen38/sweep-matrix.md](docs/performance/qwen38/sweep-matrix.md) |
| Claude Code 归因头剥离 + claude runner | [EXPERIENCE.md §15.54+](docs/archive/chronicles/v100-fastllm-experience.md) |
| 历史多引擎评测(qwen3.5/3.6) | [docs/archive/qwen36/performance/](docs/archive/qwen36/performance/) · [docs/archive/qwen35/](docs/archive/qwen35/) |

## 如何复现

```bash
# 生产启动(proxy :8000 + owned backend :8002)
./operations/handoff/start_prod.sh <profile.env 绝对路径>

# 或用 TUI/管理面
浏览器访问 http://<host>:8000/admin   # C++ 内嵌管理 WebUI
# 浏览器访问 http://<host>:8000/admin
```

profile 位于 `runtime/fastllm-native-profiles/*.env`（本机 gitignore），
门面模板 `FASTLLM_CHAT_TEMPLATE` 与加速参数均可在管理 WebUI 在线编辑。

## License

MIT（见各上游仓库）
