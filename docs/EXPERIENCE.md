# V100-32GB (SM70) LLM 推理经验总结

> 本文记录了在 Tesla V100-PCIE-32GB 上运行 Qwen3.6-27B (混合注意力架构) 过程中遇到的所有坑、解决方案和调优经验。适用于任何在 SM70 GPU 上部署大模型的场景。

---

## 1. SM70/V100 的根本限制

### 1.1 没有原生 4-bit 计算

V100 (SM70) 硬件不支持 4-bit 整数运算。AWQ/GPTQ 等 4-bit 量化权重在推理时必须 **半解包**到 16-bit 才能参与矩阵乘法, 这意味着:

- 理论上 27B × 4-bit = ~14 GB 权重
- V100 上实际占用 **~27 GB** (几乎翻倍)
- 剩余给 KV cache 的空间极小 (~3 GB)

**这是 V100 上一切显存问题的根源。** A100/H100 有原生 4-bit 支持, 不会有这个问题。

### 1.2 CUDA 13.x 不支持 SM70

NVIDIA CUDA Toolkit 13.0+ 移除了对 compute capability 7.0 的支持。必须使用 **CUDA 12.8 或 12.9** 工具链。

```bash
# 检查 nvcc 支持的架构
nvcc --list-gpu-arch
# 如果看不到 sm_70, 说明 CUDA 版本太新

# 解决方案: 用 conda 安装 CUDA 12.8
conda install -c nvidia cuda-toolkit=12.8
# 或 12.9 (turboquant fork 实测可用)
conda install -c nvidia cuda-toolkit=12.9
```

### 1.3 没有 Flash Attention 2/3 原生支持

V100 支持 FA1 但不支持 FA2/FA3 的某些优化路径。1Cat-vLLM 提供了 `FLASH_ATTN_V100` 后端作为替代, 但功能受限 (例如 sliding window 支持不完整)。

---

## 2. 引擎选择决策树

```
你的首要需求是什么?
│
├─ 单 agent 速度最快
│   └─ llama.cpp + IQ4_XS GGUF + MTP (41.5 tok/s, 65K ctx)
│       └─ 需要带 MTP heads 的 GGUF
│       └─ MTP 与长上下文互斥 (VRAM 不够)
│
├─ 平衡速度 + 上下文 + 多模态
│   └─ llama.cpp + Q3_K_M + DFlash + q8_0 KV (48-51 tok/s, 65K ctx)
│       └─ DFlash 需要 llama.cpp 主线 (TurboQuant fork 不支持)
│
├─ 多 agent × 长上下文
│   └─ llama.cpp TurboQuant turbo3, 4 slots × 140K
│       └─ 单 agent 满负载时仅 5.5 tok/s (attention 平方复杂度)
│       └─ 但 4 个 agent 各自独享 140K, 这是 V100 极限
│
├─ 必须 vLLM (工具调用/兼容性)
│   ├─ 1-2 并发 → 1.2.1 (单请求快 3.8×)
│   └─ 5+ 并发 → 1.2.0 (高并发更稳定)
│
└─ 需要最大单序列上下文
    └─ TurboQuant turbo3, 1 slot × 256K + YaRN
        └─ 可扩展到 314K, 但只剩 ~1 GB VRAM 余量
```

---

## 3. KV Cache 量化: 最关键的调优杠杆

### 3.1 为什么 KV cache 量化如此重要

Qwen3.6-27B 有 16 层 full_attention (有 KV cache) + 48 层 GDN (递归状态, 不占 token 级 KV)。

- fp16 KV: 每 token 64 KB → 65K context 需要 4 GB
- q8_0 KV: 每 token 32 KB → 65K context 需要 2 GB
- turbo4 KV: ~4.25 bit/val → 65K context 需要 ~1.1 GB
- turbo3 KV: ~3.25 bit/val → 65K context 需要 ~0.85 GB

在 V100 32GB 上, 权重固定占 14-27 GB, KV cache 的每一 GB 都很珍贵。

### 3.2 量化级别选择

| KV cache 类型 | 速度损失 | VRAM 节省 | 质量影响 | 建议 |
|--------------|---------|----------|---------|------|
| fp16 | 基线 | — | 无 | 仅短上下文可用 |
| q8_0 | ~10% | 50% | 极小 | llama.cpp 生产推荐 |
| turbo4 | ~10% | ~70% | 小 | TurboQuant 默认 |
| turbo3 | ~30% | ~80% | 可感知 | 需要极限上下文时 |

**经验法则:** 先用 turbo4/q8_0, 只有需要 140K+ 单 slot 上下文时才降到 turbo3。

### 3.3 turbo3 在 GQA 模型上的自动升级

Qwen3.6 使用 GQA (grouped query attention), KV head 数与 attention head 数比例为 6:1。TurboQuant fork 在检测到 GQA 时, 会自动将 K cache 升级为 q8_0 (因为 K 的精度对 attention score 影响更大), 只对 V 使用 turbo3。这是一个合理的设计, 不需要手动干预。

---

## 4. 推测解码 (Speculative Decoding) 在 V100 上的现实

### 4.1 AWQ + 任何 draft model = OOM

27B-AWQ 在 V100 上占 27 GB, 剩余 3 GB 不够任何 draft model:
- MTP draft: ~2 GB → KV cache 仅剩 1 GB, 上下文降到 ~3K
- DFlash draft: 3.5 GB → 直接 OOM

**结论: AWQ + speculative decoding 在单 V100-32GB 上不可行。**

### 4.2 GGUF Q3_K_M + DFlash: 可行且有效

Q3_K_M 仅 14 GB, 给 DFlash 留出足够空间:

```
权重 14 GB + DFlash 3.5 GB + CUDA/激活 4 GB = 21.5 GB
剩余 ~10 GB 给 KV cache → 65K context (q8_0 KV)
```

实测加速 ~5% (短 prompt) 到 2.7× (100 token prompt)。DFlash 的优势在短上下文场景最明显。

### 4.3 GGUF IQ4_XS + MTP: 最快但上下文受限

IQ4_XS (12.2 GB) + MTP draft, 单 slot 65K context:
- steady decode: **41.5 tok/s** (acceptance 86.4%)
- 但 MTP + 2×256K context 会 OOM

**MTP 和长上下文不可兼得**, 根据场景二选一。

### 4.4 TurboQuant fork 不支持 DFlash

TheTom 的 TurboQuant fork (`feature/turboquant-kv-cache` 分支) 没有 DFlash 支持 (`unknown model architecture: 'dflash'`)。如果需要 DFlash, 必须用 llama.cpp 主线; 如果需要 TurboQuant KV 压缩, 必须用 fork。两者不可兼得。

---

## 5. 构建指南

### 5.1 llama.cpp 主线 (SM70)

```bash
# 需要 CUDA 12.8 (CUDA 13 不支持 SM70)
conda activate tsenv  # 或其他有 CUDA 12.8 的环境

git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build --config Release -j$(nproc)
# 二进制: build/bin/llama-server
```

### 5.2 TurboQuant fork (SM70)

```bash
git clone -b feature/turboquant-kv-cache https://github.com/TheTom/llama-cpp-turboquant.git
cd llama-cpp-turboquant

# 需要兼容的 gcc 和 nvcc
# 实测组合: gcc 14.3 + nvcc 12.9 (conda tsenv)
conda install -c nvidia cuda-toolkit=12.9
conda install -c conda-forge gcc=14.3 gxx=14.3

cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_C_COMPILER=gcc-14.3 \
  -DCMAKE_CXX_COMPILER=g++-14.3
cmake --build build --config Release -j$(nproc)
```

### 5.3 运行时库依赖

TurboQuant 二进制链接的是 conda tsenv 的 CUDA 12.9 运行时库, 必须在启动脚本中设置:

```bash
export LD_LIBRARY_PATH="/home/$USER/.conda/envs/tsenv/lib:${LD_LIBRARY_PATH:-}"
```

如果忘了设置, 会报 `libcudart.so.12: cannot open shared object file`。

### 5.4 GGUF 加载器在 vLLM/transformers 中不可用

1Cat-vLLM 内置了 GGUF 加载器, 但 transformers 的 `load_gguf_checkpoint()` 不支持 `qwen35` 架构。因此 **不能** 通过 vLLM 加载 GGUF 格式的 Qwen3.6, 只能用 llama.cpp。

---

## 6. 1Cat-vLLM 1.2.0 vs 1.2.1 的并发回归

### 6.1 问题描述

GitHub issues 报告 1.2.1 在高并发下性能下降:
- #78: 1.2.1 双并发 ~43 tok/s, 1.2.0 可达 90+ tok/s
- #89: 3-6 并发时降到 ~1 tok/s
- #90: 5 并发 prefix cache hit rate 仅 21.3%

### 6.2 实测确认

| 版本 | 单请求 | 2 并发 | 5 并发 |
|------|--------|--------|--------|
| 1.2.0 | 4.8 tok/s | 24.1 tok/s | **60.5 tok/s** |
| 1.2.1 | **18.1 tok/s** | **40.1 tok/s** | 34.2 tok/s |

1.2.1 单请求快 3.8×, 但 5 并发时 1.2.0 反超 77%。P95 延迟在 1.2.1 的 4-agent 测试中飙到 166 秒。

### 6.3 选择建议

- **1-2 并发**: 用 1.2.1
- **5+ 高并发**: 用 1.2.0 (但单请求慢, 且无 fp8 KV)
- **最佳方案**: 换 llama.cpp, 两者都优于 vLLM

---

## 7. 1.2.0 多模态修复

### 7.1 缺失的 CUDA 扩展

1.2.0 wheel 缺少 `vllm.vllm_flash_attn` 的 CUDA 扩展 `_vllm_fa2_C.abi3.so`, 加载视觉编码器时报:

```
ImportError: vllm.vllm_flash_attn requires the CUDA flash attention extensions
```

**修复:** 从 1.2.1 venv 复制:

```bash
cp .venv-1cat/lib64/python3.12/site-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so \
   .venv-1cat-120/lib64/python3.12/site-packages/vllm/vllm_flash_attn/
```

复制后多模态正常启动, 视觉编码器 fallback 到 `TORCH_SDPA`。

### 7.2 关闭默认 MTP

1.2.0 自动为 SM70 启用 MTP4:

```
Applied 1Cat SM70 MTP defaults: speculative_config=mtp4, ...
```

MTP4 吃掉 ~2+ GiB, 导致 KV cache 极小。关闭:

```bash
export VLLM_1CAT_DISABLE_SM70_MTP_DEFAULTS=1
```

### 7.3 fp8 KV cache 不生效

1.2.0 的 `scaled_fp8_quant` CUDA kernel 未实现 `Float8_e5m2`/`Float8_e4m3fn`。1.2.1 日志显示 `kv_cache_dtype=auto` 即使传了 `--kv-cache-dtype fp8_e5m2` 也没生效。

---

## 8. Qwen3.6 思考模式 (Thinking) 控制

### 8.1 问题

Qwen3.6 默认在每次回复前生成 `<think>` 推理过程。在 OpenAI 兼容 API 中:
- 推理内容进入 `reasoning_content` 字段
- `content` 字段为空

大多数 OpenAI 客户端只读 `content`, 导致看起来 "模型没有回复"。

### 8.2 解决方案

**方法 1 (推荐):** llama-server 启动时加 `--reasoning off`:

```bash
./llama-server ... --reasoning off --jinja
```

客户端按需开启:
```json
{"chat_template_kwargs": {"enable_thinking": true}}
```

**方法 2:** 加载支持 thinking 开关的 chat template:

```bash
--chat-template-file ./chat_templates/qwen3.6_merged.jinja
```

我们使用的 merged template 来自 fakezeta/allanchan339/froggeric 的合并版本, 支持 `enable_thinking` / `think_off` / `think_on` 三种控制方式。

**注意:** `--chat-template-kwargs '{"enable_thinking":false}'` 已被废弃, 推荐用 `--reasoning off`。如果必须传 JSON, 注意 shell 转义 — 用单引号包裹, 不要用 `\\\"`。

---

## 9. 显存极限探索

### 9.1 安全余量

| 配置 | 启动 VRAM | 余量 | 安全性 |
|------|----------|------|--------|
| turbo3 4×140K | 30.0 GB | 2.5 GB | 安全 |
| turbo3 4×150K | 31.1 GB | 1.4 GB | 紧张 |
| turbo3 4×155K | 31.6 GB | 0.8 GB | 极易 OOM |

**经验法则: 启动 VRAM 不要超过 31 GB, 保留 ≥1.5 GB 给运行时尖峰。**

### 9.2 运行时尖峰来源

- Image encoder 临时 buffer (多模态)
- Prefill batch 内存分配
- CUDA malloc 碎片

在 4×140K 的 4-agent stress 测试中 (含 20 次多模态), 尖峰仅 +0.2 GB (30.0 → 30.2)。但更大 context 或更多图片时尖峰会增加。

### 9.3 nvidia-smi 显示的 "Free" 不可全用

nvidia-smi 看到的 3 GB "Free" 包含 PyTorch 预留但未分配内存 + CUDA 驱动保留。vLLM 的 `--gpu-memory-utilization` 不会超分配。**不要用 nvidia-smi Free 来计算可用 KV cache, 要看引擎日志报告的实际 KV cache 大小。**

---

## 10. 基准测试方法论

### 10.1 必须串行测试

同时跑两个引擎会导致 GPU 资源争抢, 数据完全失真。每次只启动一个引擎, 测完再换。

### 10.2 Warmup 必须丢弃

V100 首次请求有 JIT 编译 / CUDA kernel 缓存开销。每个 agent 的第 1 轮结果必须丢弃, 只统计 steady-state。

### 10.3 decode tok/s vs e2e tok/s

- `decode tok/s`: 仅 token 生成阶段, 严格解码速度
- `e2e tok/s`: 含 TTFT + 网络, 浏览器/客户端实际体验

两者不能混用。发布基线时必须注明使用了哪个指标。

### 10.4 Context 长度对 decode 速度的影响巨大

| Context 长度 | decode tok/s (turbo3, 4 agents) |
|-------------|-------------------------------|
| 32K/slot | ~10 |
| 65K/slot | ~8 |
| 140K/slot | ~5.5 |

Attention 的计算量随 context 平方增长。**不能把短上下文的 tok/s 外推到长上下文场景。**

### 10.5 发布基线的检查清单

每次发布基线数据时, 必须包含:

- [ ] 完整启动命令
- [ ] GPU 型号
- [ ] 驱动版本
- [ ] CUDA 运行时版本
- [ ] 模型检查点 (量化级别/文件名)
- [ ] 采样参数 (temperature, max_tokens)
- [ ] Prompt 长度
- [ ] Decode 长度
- [ ] 使用的吞吐指标定义 (decode vs e2e)

---

## 11. 其他引擎测试结论

| 引擎 | 结论 |
|------|------|
| **ik_llama.cpp** | Q3_K_M 上与 llama.cpp 速度相同 (~26 tok/s), 无提升。CPU+GPU 混合推理极慢 (5 tok/s)。 |
| **exllamav3** | 依赖冲突 (pydantic v1 vs v2); Q4 量化在 layer 0 后静默崩溃。不可用。 |
| **exllamav2** | 不支持 `Qwen3_5ForConditionalGeneration` 架构, 加载失败。 |
| **GGUF via vLLM** | transformers `load_gguf_checkpoint()` 不支持 `qwen35` 架构。不可用。 |
| **DFlash via TurboQuant fork** | TurboQuant fork 不识别 `dflash` 架构。需用 llama.cpp 主线。 |

---

## 12. 工具和脚本说明

### 12.1 agent_bench.py

多 agent 基准测试 harness, 模拟真实 agent 工作负载:
- N 个 agent 并发对话, 各自维护独立上下文
- 共享前缀 (模拟 system prompt / RAG context)
- 上下文增长 + 60% 阈值 compaction
- 每 5 轮插入图片 (多模态)
- warmup 分离 (前 N 轮丢弃)
- 输出 JSON 含 steady-state 分区统计

### 12.2 start_llama_turboquant.sh

通用 TurboQuant 启动器, 支持环境变量覆盖:

```bash
# 基本用法
./start_llama_turboquant.sh

# 自定义配置
TURBO_CTX=262144 TURBO_SLOTS=5 \
TURBO_CACHE_K=turbo4 TURBO_CACHE_V=turbo4 \
./start_llama_turboquant.sh

# 启用 YaRN 扩展上下文
USE_YARN=1 YARN_SCALE=2 ./start_llama_turboquant.sh

# 启用 MTP (需要 IQ4_XS MTP GGUF)
USE_MTP=1 MTP_DRAFT_N_MAX=2 \
TURBO_MODEL=./models/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-IQ4_XS.gguf \
./start_llama_turboquant.sh
```

### 12.3 chat_templates/qwen3.6_merged.jinja

合并的 Qwen3.6 chat template, 支持三种 thinking 控制方式:
- `enable_thinking` (true/false) — 通过 chat_template_kwargs 传入
- `<|think_off|>` / `<|think_on|>` — 通过 system message 控制
- `--reasoning off/on` — 通过 llama-server CLI 控制

---

## 13. Thinking Proxy: 让标准 API 客户端控制思考模式

### 13.1 问题

llama-server 的 `enable_thinking` 只能通过 `chat_template_kwargs` 控制, 这是 llama.cpp 专有扩展。标准 OpenAI 和 Anthropic SDK 不会发送这个字段。

具体来说, llama-server 的请求解析 (`server-common.cpp:1069-1096`) 只查找:
- `reasoning_format` (请求体, 支持逐请求覆盖)
- `chat_template_kwargs.enable_thinking` (请求体, 合并 CLI 默认值后覆盖)

它完全不识别:
- OpenAI 的 `reasoning_effort` ("low"/"medium"/"high")
- OpenAI 的 `reasoning` 对象
- Anthropic 的 `thinking` (`{"type":"enabled","budget_tokens":N}`)
- Cherry Studio 的 `enable_thinking` (顶层字段, 非标准)

而且 Anthropic SDK 使用 `/v1/messages` 端点, 请求/响应/SSE 格式与 OpenAI 完全不同。llama-server 没有这个端点。

### 13.2 解决方案: thinking_proxy.py

单文件 FastAPI 代理, 运行在 llama-server 前面。架构:

```
Client (OpenAI SDK / Cherry Studio)  ─┐
                                      ├──► Proxy (port 8000)
Client (Anthropic SDK)                ─┘    ├─ reasoning_effort / enable_thinking → chat_template_kwargs
                                             ├─ Anthropic /v1/messages 格式转换
                                             ├─ IP 认证 (Tailscale/localhost 绕过)
                                             └─ llama-server 进程管理 (崩溃自动重启)
                                                      │
                                                      ▼
                                             llama-server (port 8001)
```

### 13.3 客户端兼容性 (踩过的坑)

不同客户端发送 thinking 控制参数的方式完全不同。代理必须全部拦截:

| 客户端 | 参数名 | 位置 | 格式 |
|--------|--------|------|------|
| **OpenAI SDK** (o1/o3 模式) | `reasoning_effort` | 请求体顶层 | `"high"` / `"low"` / `"none"` |
| **OpenAI SDK** (替代格式) | `reasoning` | 请求体顶层 | `{"effort":"high"}` 或 `"on"` |
| **Cherry Studio** | `enable_thinking` | 请求体顶层 | `true` / `false` |
| **Cherry Studio** | `thinking_budget` | 请求体顶层 | 数字 (token 预算) |
| **Anthropic SDK** | `thinking` | 请求体顶层 | `{"type":"enabled","budget_tokens":N}` |
| **llama.cpp 原生** | `chat_template_kwargs.enable_thinking` | 嵌套对象 | `true` / `false` |

代理的策略: 从请求体中 pop 出所有已知变体, 统一映射到 `chat_template_kwargs.enable_thinking`, 然后转发给 llama-server。

**关键教训:** Cherry Studio 是最隐蔽的坑。它不发 `reasoning_effort`, 而是在请求体顶层放 `enable_thinking: true` + `thinking_budget: 65536`。llama-server 完全忽略这两个字段 (不在它的解析范围内), 导致思考模式始终关闭。只有通过代理日志 dump 完整请求体才发现这个问题。

### 13.4 Anthropic 格式转换

参考 vLLM 的 `vllm/entrypoints/anthropic/serving.py` 实现, 主要转换点:

**请求 (Anthropic → OpenAI):**
- 顶层 `system` → 第一条 `{"role":"system"}` 消息
- `content` 数组中的 `text` / `image` / `thinking` / `tool_use` / `tool_result` 块 → OpenAI 对应格式
- `thinking.type == "enabled"` → `chat_template_kwargs.enable_thinking = true`
- `max_tokens` / `temperature` / `top_p` / `top_k` / `stop_sequences` → 直接映射
- `tools` 数组 (Anthropic `input_schema`) → OpenAI `function.parameters`

**响应 (OpenAI → Anthropic):**
- `choices[0].message.reasoning_content` → `content[].type = "thinking"` (附随机 `signature`)
- `choices[0].message.content` → `content[].type = "text"`
- `choices[0].message.tool_calls` → `content[].type = "tool_use"`
- `finish_reason` 映射: `stop → end_turn`, `length → max_tokens`, `tool_calls → tool_use`
- `usage.prompt_tokens → input_tokens`, `usage.completion_tokens → output_tokens`

**流式 SSE (OpenAI → Anthropic):**
- OpenAI 的 `data: {"choices":[{"delta":{"reasoning_content":"..."}}]}` → Anthropic 的 `event: content_block_delta` + `{"delta":{"type":"thinking_delta","thinking":"..."}}`
- OpenAI 的 `data: {"choices":[{"delta":{"content":"..."}}]}` → Anthropic 的 `event: content_block_delta` + `{"delta":{"type":"text_delta","text":"..."}}`
- 需要跟踪当前 block 类型 (thinking/text), 在切换时发出 `content_block_stop` + `content_block_start`
- 生命周期: `message_start` → `content_block_start` → `content_block_delta` × N → `content_block_stop` → `message_delta` → `message_stop`

### 13.5 认证设计

代理同时接受 OpenAI 和 Anthropic 的认证方式:

| 来源 | 认证方式 | 绕过条件 |
|------|---------|---------|
| Tailscale (100.64.0.0/10) | 无需认证 | IP 在 Tailscale 网段 |
| Localhost | 无需认证 | IP 是 loopback |
| FRP / 公网 | `Authorization: Bearer <token>` | — |
| FRP / 公网 | `x-api-key: <token>` | — |

默认 token: `AUTH_TOKEN` 环境变量 (建议通过环境变量设置, 不要硬编码)。

### 13.6 进程管理

代理内部管理 llama-server 子进程:
1. 启动时 spawn llama-server, 等待 `/health` 返回 200
2. 后台 monitor task 每 15 秒健康检查
3. 进程退出或健康检查失败 → kill + 重启, 指数退避 (5s → 10s → 20s → ... → 60s)
4. 代理关闭时 terminate llama-server

**模型加载时间:** 4x140K TurboQuant 配置从启动到 health ready 约 60-90 秒。重启后客户端会经历这段时间的 503。如果需要更快的恢复, 可以考虑:
- 降低 context 配置 (如 4x65K, 加载更快)
- 预加载模型到系统 page cache
- 使用 systemd 的 watchdog 机制替代代理内部监控

### 13.7 部署

```bash
# 在 tmux 中启动 (推荐)
tmux new-session -d -s proxy
tmux send-keys -t proxy "cd /path/to/project && ./start_proxy.sh" Enter

# 或直接运行
AUTH_TOKEN=your_token ./start_proxy.sh

# 客户端连接
# OpenAI: base_url=http://<host>:8000/v1, api_key=CHANGE_ME
# Anthropic: base_url=http://<host>:8000, api_key=CHANGE_ME
```

所有日志 (代理 + llama-server) 输出到 tmux 同一窗口, 可以实时观察。

---

## 7. Fedora 44 + CUDA 12.9 构建问题

### 7.1 GCC 16 与 nvcc 不兼容

Fedora 44 自带 **GCC 16.1.1**，但 CUDA 12.9 的 nvcc 最高只支持 GCC 14。直接使用会报：

```
#error -- unsupported GNU version! gcc versions later than 14 are not supported!
```

即使传 `--allow-unsupported-compiler` 也不行——GCC 14+ `<type_traits>` 使用了 `__is_array`、`__is_pointer` 等新关键字，nvcc 的 host 编译器前端无法解析：

```
/usr/include/c++/16/type_traits: __call_is_nothrow<__invoke_result<...>>: error: type name is not allowed
```

### 7.2 解决方案：conda GCC 12

安装 conda 的 GCC 12 并让 nvcc 通过 `-ccbin` 使用它：

```bash
conda install -n tsenv gxx_linux-64=12
```

然后用 nvcc wrapper 脚本让 CMake 使用正确的 host 编译器：

```bash
#!/bin/bash
exec /home/ezra/.conda/envs/tsenv/bin/nvcc \
    -ccbin=/home/ezra/.conda/envs/tsenv/bin/x86_64-conda-linux-gnu-g++ "$@"
```

CMake 配置（fastllm 示例）：
```bash
export PATH="/home/ezra/.conda/envs/tsenv/bin:$PATH"
export CC=/home/ezra/.conda/envs/tsenv/bin/x86_64-conda-linux-gnu-gcc
export CXX=/home/ezra/.conda/envs/tsenv/bin/x86_64-conda-linux-gnu-g++

cmake .. -DUSE_CUDA=ON \
         -DCMAKE_CUDA_COMPILER=/tmp/nvcc-wrapper \
         -DCUDA_ARCH="70" \
         -DCMAKE_C_COMPILER="$CC" \
         -DCMAKE_CXX_COMPILER="$CXX"
```

### 7.3 fastllm 第三方依赖

fastllm 的 `third_party/` 需要以下子仓库（通过 `git clone --recursive` 获取或手动补全）：

| 目录 | 来源 | 用途 |
|---|---|---|
| `third_party/pybind11` | Git submodule | Python binding |
| `third_party/json11` | dropbox/json11 | JSON 解析 |
| `third_party/gguf` | 独立仓库 | GGUF 格式读取 |
| `third_party/flashinfer` | 独立仓库 | FlashInfer 算子 |
| `third_party/turbomind` | turbomind 内核 | SM x84 优化 |
| `third_party/cutlass` | Nvidia CUTLASS | FP8 Marlin Linear 等 |

如果直接 `git clone --depth 1` 没有子模块，需要逐一克隆或运行 `git submodule update --init --recursive`。

### 7.4 SM70 上的 fastllm 注意事项

- V100 (SM70) 不支持 FP8——CUTLASS FP8 kernels 自动关闭
- fastllm 的 AWQ 算子有 SM70 专用路径 (`src/devices/cuda/awq_sm70/`)
- 对于缺乏原生 4-bit 支持的 SM70，fastllm 通过 FP16 半解包方式运行量化模型
- fastllm 支持 SM70 的 `--cache-ram` 上下文缓存机制（不同于 llama.cpp 的 checkpoint 方案）

## 14. FastLLM 后端模板误用：工具调用空参数/拼错工具名

### 14.1 症状（Cherry Studio 实测）

- 模型调用 `web_search` 但 `arguments` 恒为 `{}`，Cherry Studio 校验失败：
  `Invalid input for tool web_search: Type validation failed: Value: {}. expected: string`
- 多轮后模型开始拼错工具名：`web_feetch` → `wweb_fe tc h`，死循环重试
- 有时直接输出 Qwen 老式文本工具格式 `<tool_call> <web_search> <parameter=query> ... </tool_call>`，客户端无法解析

### 14.2 根因

thinking_proxy.py 的模板默认路径基于 `PROJECT_DIR` 拼接，而 **profile env 把 `PROJECT_DIR` 指向了 `v100-perfs/`**：

```
FASTLLM_CHAT_TEMPLATE = PROJECT_DIR / "chat_templates" / "qwen3.6_gguf_original.jinja"
# PROJECT_DIR=/run/media/ezra/.../v100-perfs
# → 实际加载 v100-perfs/chat_templates/qwen3.6_gguf_original.jinja
```

`qwen3.6_gguf_original.jinja` 对 tools 的处理是 `{{- tool | tojson }}` —— 把整个工具定义**原样 dump 成 JSON blob** 塞进 system。模型看到的是 `{"function": {"description": "...", "parameters": {...}}}` 巨型 JSON，没有可读的签名（`name(param: type)`）、没有 required 标注、没有输出格式示例，工具调用能力直接退化。

对比 `qwen3.6_merged.jinja`（`1CatVLLM/chat_templates/` 下的改进版）：
- `render_tool_signature` 输出 `- name(param: type, req_param: type) — description`，required 参数不带 `?`
- 带完整 XML 输出格式示例 + "Include every required parameter" 指令

### 14.3 修复

两个 profile（`q5-262k-mtp2.env` / `q5-262k-mtp1.env`）显式指定改进版模板：

```
FASTLLM_CHAT_TEMPLATE=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/chat_templates/qwen3.6_merged.jinja
```

### 14.4 验证

- Cherry Studio 标准工具集（fs_read/web_fetch/web_search）原样重放：原版模板 5/5 输出 `{}`，切换后 3/3 输出 `{"query":"MC 迷你世界"}`，finish=tool_calls
- `test_fastllm_adapter.py` 13 pass、`test_fastllm_feature_bench.py` 10 pass

### 14.5 教训

1. **profile env 里的 `PROJECT_DIR` 会改变 proxy 的默认模板解析路径**——显式设置 `FASTLLM_CHAT_TEMPLATE`，不要依赖默认拼接
2. 工具调用的"模型能力"问题先查模板渲染的 tools 段（抓 proxy→FastLLM 的实际请求体对比），再怀疑模型
3. 同症状还可能是：`/v1/models` 缺能力标注（Cherry Studio 不注入 tools）、`route_for_model` 把本地别名路由到外部 NIM（工具格式不匹配）

### 14.6 遗留（未同源修复）

emoji 乱码（`\xf0\x9f\x92` 缺第 4 字节）在 merged 模板下依旧存在——坏字节产生于 FastLLM 输出侧（token 生成），与模板输入无关，需单独排查 GGUF vocab / FastLLM 输出路径。

## 15. 图像请求 CUDA OOM：paged KV 页池全量预留 + 临时缓冲共享竞态

### 15.1 症状

- Cherry Studio 发图像请求 → 流中断（`ERR_INCOMPLETE_CHUNKED_ENCODING`、`AI_TypeValidationError`）
- 后端日志：`CUDA error when allocating ... gpuFree: 7 MB / 32494 MB`，FastLLM 崩（code=1/6）后 reload
- 图像请求峰值 ~32.4 GiB（free 0.3 GiB），与分辨率无关（64px 与 1024px 相同）；文本请求 25.2 GiB 安全

### 15.2 根因（两个独立 bug）

**A. PagedCacheManager 全量预留页池（图像 prefill 运行时分配 ~7 GB 撞显存峰值）**

- 多模态/图像 prefill 走 non-fused paged KV 路径（turbo3 下 CUDA direct runner 禁用，走 `ForwardFromHiddenStates` 的 `AllocatePagedCacheManager(i*2)`）
- `AllocatePagedCacheManager` 按 `maxPages = GetMaxTokens()/pageLen` **全量预留**（262K 上下文合计 ~7 GiB），且 map 按层懒创建
- 首次图像请求（或累积页不足时）在**显存已满载**时再分配 4.8–7 GiB → CUDA OOM（gpuFree 剩几 MB）
- 文本 fused prefill 不建页池，所以纯文本 25.2 GiB 安全；MTP0/1/2 差异仅 0.15 GiB，不改变结论

**B. `FastllmBorrowCudaTempBuffer` 扩容释放共享临时缓冲（MTP conv use-after-free）**

- 设备临时缓冲是每设备单例（`FastllmCudaTempDeviceBuffer holder`），MTP 校验/conv 与主 forward 并行借用
- 某借用者需要更大缓冲时：`DirectFree(holder.data)` + 重新分配 → 仍持有旧指针的借用者（MTP conv kernel 参数）读已释放内存 → `cudaErrorInvalidValue`（"batched multi-token MTP conv"），`Qwen35MtpBatchFastPathUnavailable` → abort
- 图像+长文本组合（img+long）可复现；纯文本/纯图像不触发

### 15.3 修复（fastllm 源码，`fastllm/build/apiserver` 已重建）

1. **页池懒分配 + 按需增长**（`fastllm.cpp`）：
   - `AllocatePagedCacheManager`：物理初始池 128 页（~0.8 GiB 总量），`maxPages` 字段保留逻辑预算（调度预留检查用）
   - 新增 `PagedCacheManager::Grow(newMaxPages)`：分配更大池 → 拷贝旧页 → 退役旧池（`retiredCudaData` 列表，析构统一释放，避免并发 use-after-free）；`dims[0]` 为物理页数，`maxPages` 为预算
   - `CudaAppendPagedCacheOp`：页不足时 Grow（不再 ErrorInFastLLM）
   - `GetUnusedPageIndex`：无页时预算内自动 Grow 重试（覆盖 decode/MTP CoW/prefix restore/swap 所有取页方）
   - 调度检查（`pageNeedsFitWithReservations`/`hasPagedManagerShortage`，qwen3_5.cpp + basellm.cpp）：物理不足但需求在预算内 → Grow；busy/total 页数改用 `dims[0]`（物理）而非 `maxPages`
   - prefix-cache 快照导出/恢复：兼容懒池（按物理页数校验，恢复前 Grow 到所需页数）
2. **临时缓冲扩容不释放旧缓冲**（`fastllm-attention.cu` `FastllmBorrowCudaTempBuffer`）：只增不减，消除共享缓冲 use-after-free
3. 防御：Grow 拷贝前 `FastllmCudaValidatePointerRange`，旧指针无效则清零新池并跳过退役（不 double-free）

### 15.4 配置调整

- `q5-262k-mtp2.env`（及 mtp1）：`FASTLLM_CPU_REQUEST_SWAP=0`（swap 与图像路径冲突，且懒页池已取代其显存优化价值）；`--tokens` 恢复 **262144**（262K 原生上下文，懒页池下与 240K 无额外风险）
- VRAM 阈值保持 `FASTLLM_VRAM_MIN_FREE_GIB=0.5` / `RESUME=1.0`（长对话页池累积时 proxy 保护性 unload）

### 15.5 验证（Tailscale IP 100.94.73.9:8000）

| 场景 | 修复前 | 修复后 |
|---|---|---|
| 1024px 图像 | 峰值 32.4 GiB，随机崩（4-6 chunks） | 峰值 23.2 GiB，198 chunks 完整 |
| 64px 图像 | 峰值 31.7 GiB（free 1.1） | 峰值 23.2 GiB，198 chunks 完整 |
| 19800 token 长 prefill | — | 峰值 25.0 GiB，完整流 |
| 图像+3000字长文本 | 4 chunks 后 abort（MTP conv invalid） | 70 chunks 完整 |
| 连续图像 | 第二个请求 503/崩 | 稳定，after 24.9 GiB used / 7.5 GiB free |
| lifecycle | 多次 exited code=1/6 + reload | 单 generation，无 reload |

### 15.6 遗留

- **YARN rope scaling 未接入**（FastLLM kernel 有 `YarnRope`（cudadevice.cpp yarnFactor/yarnAttentionFactor/yarnCorrectionLow/High），但 Qwen3.5 模型路径只解析 `rope_scaling.type=linear|dynamic`，未解析 `yarn`，mrope 应用无 yarn 分支）——超过 262K（原生 256K）上下文需要它
- emoji 乱码（§14.6）依旧
- `doesworkstation` nginx IPv6/502 未修（Cherry 用 http://100.94.73.9 正常）

### 15.7 YaRN 接入（代码完成，生产未启用）

- FastLLM 已有 YarnRope kernel（`FastllmYarnInvFreq` + `YarnRopeEncoding`），但 Qwen3.5 模型路径未接线。本次接入：
  - `Qwen35InterleavedRopeKernel`（3 个 dtype）加 useYarn/yarnFactor/yarnAttentionFactor/yarnCorrectionLow/High 参数（mrope + YaRN）
  - `FastllmCudaQwen35InterleavedRope`、`CudaQwen35InterleavedRopeOp`、`fastllm.cpp Qwen35InterleavedRope` 透传（默认关闭，向后兼容）
  - `InitParams`：解析 `rope_scaling.type=yarn`（+factor/attention_factor/beta_fast/beta_slow/original_max_position_embeddings），支持 env `FASTLLM_QWEN35_YARN=1` + `FASTLLM_QWEN35_YARN_FACTOR` 等覆盖；预计算 correction 区间
  - `ApplyMultimodalRotary`：YARN 时 mrope→yarn interleaved、普通→`YarnRopeEncoding`
- 验证：factor=2 + tokens 524288 下 19800 tokens 长文本 + 图像全部正常（YaRN enabled 日志、correction=[14,22]、KV 池 3.6GB、图像后 free 7.5GB）
- **未启用原因**：>262K 的单请求（245K tokens prefill）在 YARN 开（0.2s 快速失败）与关（25s 完整流但请求后清理阶段 `CudaAppendPagedCacheOp failed to launch multi-page copy` abort）都不稳定——超长单请求的既有边界问题（multi-page copy launch invalid），需单独调试；生产保持 262K 无 YARN

### 15.8 超长请求稳定性：MtpKvCache double-free + swap 多模态排除

**症状**：245K 单请求完整流后 abort（`CudaAppendPagedCacheOp failed to launch multi-page copy`）；swap=1 时图像+长文本组合卡 800s+。

**根因链（A：MtpKvCache double-free）**：
- `Data` 无自定义拷贝构造（裸指针浅拷贝），`MtpKvCache{Data key, value}` 放 `unordered_map`——rehash/erase 时浅拷贝链使同一 cudaData 析构多次（cudaFree 同地址 4 次）
- 释放的地址被 cudaMalloc 复用给 paged KV 池 → 页池被"再释放"（Grow 时报 `old pool already freed`）→ 后续 multi-page copy 用坏指针（launch invalid）
- 修复：`MtpKvCache` 禁拷贝（拷贝构造/赋值 delete，移动 default），`mtpCaches` 值改 `std::unique_ptr<MtpKvCache>`，新增 `GetMtpCache()` 辅助（调用者须持 `mtpCacheMutex`）；`emplace` 改 `make_unique`，所有 `it->second.x` 改 `it->second->x`
- 验证：245K 200 完整不崩；cudaFree 无重复；`old pool already freed` 归零（本 gen）

**根因（B：swap 与多模态卡死）**：
- `CanSuspendResponseContextToCpu` 未排除多模态请求——图像+长文本的 CPU 交换恢复卡 800s+
- 修复：多模态（`context->multimodalInput` 非空）不挂起（返回 false + error）
- 验证：img+long 4.7s 完整（swap=1 保持，zstd 磁盘交换对纯文本长 prefill 生效）

**残留**：仍有少量 `old pool already freed`（Grow 降级 memset 0，影响已退役空闲池，不崩、输出正常）——另一处浅拷贝释放者未定位（低影响，后续可查）。

### 15.9 低频字乱码根因确认（§14.6 续）

- GGUF vocab 全部合法（248320 token 无无效 UTF-8）；与采样无关（temp=0/1.0 输出逐字节相同）、与 MTP 无关（MTP0 同样 14 处坏字节）
- 根因：Q5_K_M 量化的字节级 token 预测——低频汉字/emoji 的整字 token logits 被量化噪声压低，模型确定性改走字节 fallback token（`<0xE6><0xA7>` 等），字节序列断裂（`\xe6\xa7` 缺第 3 字节、孤立 continuation）
- 无法在服务层修复（缺字节不可恢复）；换更高精度权重（Q8/F16）超出 32GB 显存。缓解：proxy 层可丢弃无效字节（字消失而非 �），未做（保持透传）

### 15.10 换 abliterated 模型（mradermacher Q6_K）+ MTP 档位对比

**模型切换**（2026-08-11）：
- 原 `ThinkingCap-Qwen3.6-27B-heretic-Q5_K_M-plus-mtp.gguf` → **`Huihui-ThinkingCap-Qwen3.6-27B-abliterated.i1-Q6_K.gguf`**（mradermacher，Huihui abliterated，**真去拒绝**：核弹步骤等敏感内容直接生成；此前 protoLabsAI 的 abliterated 仓库是假的（discussion 1 确认，已弃用））
- **MTP 头自带**（`blk.64.*` 15 个 + `nextn.*` 4 个——与 FastLLM 的 GGUF 规则兼容，**无需移植**；`scripts/merge_mtp_head.py` 保留备用（未来若遇无 MTP 的量化可直接合并）
- mmproj 换 `mmproj-ThinkingCap-Qwen3.6-27B-f16.gguf`（companion 仓库）；前缀缓存 key 换 `qwen3.6-mrad-q6-mtp2`
- Q6_K 加载后 23.4GB used / 9.1GB free；**首次图像慢 ~260s**（冷视觉加载），后续 8s

**MTP 档位对比**（262K + mrad Q6，200 tokens decode）：
| MTP 档 | decode tok/s | 备注 |
|---|---|---|
| 0（关） | 4.2 | 基线 |
| 1 | 4.8 | 单 draft 加速小 |
| **2** | **30.1** | **最优（7 倍），保持生产** |
| 3 | 10.8 | 3 draft 验证开销/接受率下降 |
| 4 | 5.2 | 继续下降 |

prefill（80K 字符）0.1-0.2s，MTP 档位影响可忽略（MTP 只加速 decode）。

**长上下文压力**（262K 上限）：
- 150K 字符（~19.5 万 tokens）prefill+decode 20.9s 不崩（峰值 29.9GB/free 2.9GB）
- 200K 字符（~26 万 tokens，超 262K）后端 502 拒绝（正确行为，不崩）
- **512K（--tokens 524288）反复崩溃**（`FastllmCudaCopyFromDeviceToDevice` invalid argument 5495，无明确根因、非 OOM）——已回 262K，**待后续排查**（诊断 backtrace 已加在 5495 处）

**乱码确认**：Q6_K 同样 14 坏字节（与 Q5 相同）——字节 token 断裂是量化模型的固有行为，Q6 不改善；换更高精度（Q8 28GB）超出显存，接受。

### 15.11 multi-page copy 容错 + 恢复

- 预热/多请求场景偶发 `CudaAppendPagedCacheOp failed to launch multi-page copy`（并发 Grow 竞态，诊断 backtrace 未捕获具体调用者；gdb 下单请求正常，proxy 多请求复现）
- 修复：`CudaAppendPagedCacheOp` 的 multi-page 失败改为**降级逐页复制**（`FastllmCudaPackedKVCacheCopy` 循环），不再 abort
- 诊断代码（copy backtrace / pool-miss 跟踪 / TurboKV 打印 / Grow 打印）已清理，保留 `[TurboKV]` 入口/launch 诊断（供未来定位）
- 恢复后：text 200（70 chunks 1.2s）、img 200（132 chunks 7.6s）、READY gen 1；`old pool already freed` 仍有零星（降级 memset 0，不崩）
- 后续定位到 Turbo3 共用 kernel 的 row-bound 回归：single-page/batch 路径把局部 `rows + 1` 当成全局页池 row 上限，物理页号大于 0 时合法写入被当作 OOB 丢弃；device `printf("%zu")` 又把后续诊断字段打印乱序。修复为只在拿到真实页池 geometry 的 multi-page 路径启用全局上限，legacy single-page/batch 路径传 `-1` 禁用该检查，并用 `%llu` 输出 `size_t`。`turbo3_kv` 回归通过，新 generation 的文本+三次图像请求后无 `ROW OOB`/copy/CUDA 错误。
- **当前生产（2026-08-12）**：Huihui abliterated i1-Q6_K + 262K + MTP2 + swap/zstd + 前缀缓存；启动后 22.8GB，代理冷启动请求 HTTP 200（335.8s）。KV 修复后三次 1×1 PNG 均 HTTP 200（7.31/4.41/4.40s），三次都正确识别 red，服务保持 READY；强制工具请求仍未产生 `tool_calls`，因此视觉稳定性/语义已通过，工具调用能力仍未通过。

### 15.12 模型级 RAM suspend/resume（托管 host cache）

- `HostCacheBudget` 统一限制模型权重与前缀 KV 的宿主内存占用；模型挂起按 `must-cache → derived → ordinary` 分级，12GiB 权重留 RAM，其余按 GGUF materialization recipe 从源文件重建。
- 实机 i1-Q6_K：READY 23.7GB VRAM；memory suspend 17.45s，缓存 12,863,627,264 bytes、驱逐 8,513,433,600 bytes，GPU 降至 2.2GB；memory resume 111.89s，cache hit ratio 0.601749，重建 8,513,433,600 bytes，恢复至 23.7GB。
- 首次实机恢复暴露 partial reload 只还原 CPU payload、未还原原 CUDA placement；修复为记录每个驱逐 tensor 的原 device id，GGUF 重建后显式 H2D，并在任一阶段失败时自动回退完整 disk reload。修复后 `/admin/resume` 返回 `tier=memory`，随后推理 HTTP 200、`/health` 为 `ready=true`。
- 代理空闲分级：12h 内 memory tier，超过 12h disk tier；后端进程常驻，首个请求自动 resume。生产 profile 已启用 `FASTLLM_HOST_SUSPEND_CACHE=1`、12GiB host 权重预算及 3GiB prefix RAM 预算。

### 15.13 直接后台部署 + 恢复链实机复验（2026-08-12 晚）

- 生产改为**直接后台进程**（`setsid` + 日志重定向，禁 hub/tmux）：`thinking_proxy.py` owned 冷启动，首个请求触发 spawn generation 1（pid 1071105），冷启动请求 HTTP 200（321.9s，前缀缓存从磁盘恢复 generation 5，652MB）。
- 共享 host 预算实机上限：profile `FASTLLM_HOST_SUSPEND_CACHE_MAX_BYTES=17179869184`（16GiB，按用户要求从 12GiB 上调），prefix RAM 预算 3GiB。
- `/admin/suspend`（memory tier）31.2s：cached 17,117,118,464 bytes（≈15.9GiB，16GiB 硬上限内）、source evicted 4,259,942,400 bytes、hit ratio 0.8007；GPU 22356→860 MiB，进程同 pid 常驻（VmRSS 21.7GB，MemAvailable 25.2GB）。
- `/admin/resume` 89.3s（冷启动的 1/3.6）：hit ratio 0.8007，rebuilt 4,259,942,400 bytes，GPU 恢复 22292 MiB，同 pid；随后推理 HTTP 200（1.13s）、前缀持久化 `loaded_generation=5` 保持。
- 带图稳定性（cat.png，5 连发）：全部 200，4.30-6.87s，GPU 恒定 23788 MiB 无泄漏；1024-token 请求 finish=stop，文本正确（"浅紫色背景可爱卡通黑猫图标…"）。日志零错误模式（packed copy FAILED / old pool / multi-page failed / CUDA error / abort 全为 0）。
- 回归：`testHostCacheBudget`、`testHostOffloadLifecycle`、`testHostOffloadControl`、`regressionOps`（persistent_prefix_cache + turbo3_kv）全部 PASS。fastllm HEAD `6219778b`、v100-perfs HEAD `2be2adf`，两仓干净。

### 15.14 换普通 ThinkingCap（弃 abliterated）+ 大图 8MiB 缓冲修复（2026-08-13）

- **模型替换**：abliterated i1-Q6_K 工具调用频繁失败 → 生产切到 `ThinkingCap-Qwen3.6-27B-MTP-GGUF/ThinkingCap-Qwen3.6-27B-Q6_K-MTP.gguf`（原版非 heretic，tensor 名含 `nextn.eh_proj/enorm/hnorm/shared_head_norm` + blk.64 = MTP 头，保留 MTP2）。新 profile `q6-plain-262k-mtp2.env`（前缀缓存 key 改为 `qwen3.6-plain-q6-mtp2`，旧模型 KV 不复用）。运行于 tmux `fastllm-prod` 会话双 pane（pane0 proxy + pane1 `tail -F` 后端日志）。
- **验证**：冷启动 READY 280s；文本 200；**工具调用 `finish=tool_calls`（get_weather {"city":"北京"}）→ tool 结果 → 最终回答 "北京现在的天气是晴，气温 24 摄氏度。"（finish=stop）多轮 agent 闭环通过**；cat.png 图像 200、描述正确（finish=stop）。
- **大图 8MiB 请求缓冲 bug**：后端 `char buff[8MiB]` 导致 4K 图（尤其 JPEG 经 proxy 重编码 PNG 膨胀 2.7× 后 base64 9.1MB）请求体读不满 → 15s socket 超时断连 → proxy ReadError → 503。修复：缓冲 8→64MiB（`a7db89d7`）；proxy `_convert_images` 仅在 EXIF 转置/非 RGB 时重编码，否则保留原字节（`42c839a`）。验证：4K JPEG 直连+proxy 全链路 200。
- **吞吐登记**（`benchmarks/fastllm/results/fastllm_iq6_prefill_decode_throughput_20260812.json`，abliterated 版本测得，架构相同可参考）：prefill 635.8 tok/s（32K tokens）、decode 26.0 tok/s（MTP2）。
- stable tag `stable-20260812`：fastllm `a7db89d7`、v100-perfs `42c839a`。

### 15.15 omp/opencode 实机工具调用 + agentic benchmark（2026-08-13 晚）

- **根因链**：omp 走 OpenAI 兼容 API 时 tool call 作为文本返回 = 三层问题：(1) merged chat template 的 `<tools>` 段渲染"紧凑签名"而非官方完整 JSON，模型指令跟随退化；(2) 后端/proxy 解析器只认 `<function=`/`<parameter=` 前缀；(3) 模型回落原生格式（裸 `<bash>`、`<prompt>` 参数标签、Qwen JSON `{"name":...}` 形式）。
- **修复**：模板对齐官方（`tool | tojson` + 官方指令，v100-perfs `4df3231`）；proxy `parse_fastllm_tool_calls` 接受裸名/裸参数/JSON 形式（`65e7935`）；后端 `ParseBlock` 同样三级兼容（fastllm `27370f3e`）。验证：omp 目录任务、opencode Sisyphus（空参→SchemaError→重试→mkdir→Write→Read→最终回答）全通。
- **agentic benchmark**（`benchmarks/fastllm/agentic_toolcall_bench.py`，tmux+omp/opencode 跑 4 个可验证副作用任务，记录并行率/前缀缓存差值/工具行为）：
  - 并行：peak_active=1、mean 0.12-0.86、peak_pending=0——agentic 流量天然串行，瓶颈是每轮 thinking decode（26 tok/s）；batch=1 下无并行空间，多用户并发未测。
  - 前缀缓存：每任务 GPU hit pages +19K~+83K（多轮长 system prompt 重入累积命中）；磁盘命中 0（未达 disk tier 门槛 min-hits 2 + 64K tokens）。
  - 任务结果：file_roundtrip / fib_run / multi_file 通过；count_py（建 5 文件+统计+写结果）3 次全超时——暴露工具名漂移（"writ"、"function_calls"）、参数缺失、格式漂移（`<function=write@/path>`）。JSON 形式解析已修，工具名模糊匹配/参数强制未做（记为边界）。


### 15.16 agentic 负载评估（1-2 主 agent + 5-10 子 agent）+ turbo4 移植

- **turbo4 移植**（fastllm `TURBO4_KV`，4bit PolarQuant，上游 TURBO4_0 表）：K=q8_0 + V=turbo4 上线验证 READY 280s。KV recall A/B：100K 两者 10/10 持平；**200K turbo4 被 VRAM 水位拦截（503）**（V +32% 显存顶到保护线）→ 生产保持 turbo3，turbo4 作为可选档。
- **agentic load benchmark**（10 agent × 4 轮共享前缀，全 python）：40/40 成功、tool_call_rate=1.0、args 全合法；mean_active=1.94（GPU decode 饱和）。
- **batch 扫描**：b2s2w2 total 85.5s/p50 12.65/p95 60.9；b4s4w4 total 108.7s/p50 20.75/p95 53.6——**batch=2 是最优**：decode 是串行计算瓶颈，batch 4 只重新分配延迟不增吞吐。结论：GPU decode（MTP2 26 tok/s）是唯一硬约束，MTP3/4 更慢无解。
- **优化落地**：`FASTLLM_PREFIX_CACHE_DISK_MIN_TOKENS=4096`（agentic 系统提示跨重启磁盘持久化）；生产配置 batch=2 + slot=2 + workers=2。
- 交叉会话前缀缓存验证：同前缀 100K 上下文 10 查询，首个 302.5s（全量 prefill），其余 2.4-2.5s（全命中）。
