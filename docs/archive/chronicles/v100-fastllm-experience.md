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


### 15.17 VRAM 水位保护线实测（0.5 GiB vs 0.1 GiB）

- 用 turbo4 + 200K 上下文复测水位:0.5 GiB 时干净拦截(503,后端存活);降到 0.1 GiB 后请求放行,运行中显存最低 138 MiB,后端在池增长尖峰处 **CUDA OOM 崩溃** → 全部 502 + 5 分钟重载。
- **结论:0.5 GiB 不是保守,是正确值**——后端瞬时分配尖峰需要 ~0.5 GiB 余量,138 MiB 不够。


### 15.18 Hermes 微信会话错误工具调用诊断 + 修复 + benchmark

- **现象**：Hermes 微信 DM 会话（20260704_201000_4181eaeb，78K tokens）出现 "funct"/"writ"/空 args 等完全错误的工具调用。
- **诊断**：会话主要跑在 NVIDIA fallback（glm-5.2 ×29、minimax-m3 ×137）——8/13-14 我们 proxy 频繁重启导致 Hermes 落 fallback；长上下文多轮下 fallback 模型产出截断工具名。单轮复现两个 NVIDIA 模型均正常 → 是长上下文退化，非解析器 bug。
- **修复**（hermes 侧）：`agent/agent_runtime_helpers.py` repair_tool_call 增加唯一前缀匹配（"writ"→"write_file"、"exec"→"execute_code"；"funct" 无匹配→正确拒绝+重试）。已备份 + systemctl 重启 hermes-gateway 生效。
- **benchmark**（`benchmarks/fastllm/hermes_toolcall_bench.py`）：重放微信会话真实用户消息 × 3 provider 打分：
  - fastllm-proxy：0 错误，12/12 名称合法、12/12 args JSON 合法、必填字段 2/12（模型偶发漏必填，靠 Hermes 校验+重试兜底）
  - nvidia-glm52：2×429，4/4 全合法
  - nvidia-minimax3：3×429，1/1 合法
  - 结论：主链路（我们的 proxy）名称/格式 100% 可靠；fallback 有速率限制与长上下文退化。修复后 Hermes 对截断名自愈。


### 15.19 Qwen3.8-27B UD-Q4_K_XL + 集成 MTP 上线（2026-08-15）

- **模型与模板**：生产切到 `Qwen3.8-27B-UD-Q4_K_XL.gguf` + `mmproj-F16.gguf`，GGUF 为 `qwen35` 65/64+1 MTP 布局，最大上下文 262144。`chat_templates/qwen3.8_merged.jinja` 直接采用 GGUF 内嵌的 9993-byte 官方模板（SHA-256 前缀 `12827f24b742ea4e`），没有再叠加 Qwen3.6 模板约定。
- **FastLLM 兼容性根因**：UD-Q4_K_XL 的 65 层 dense MLP 全部是混合量化对（`ffn_gate=IQ4_XS`、`ffn_up=Q5_K`）。加载器会正确保留 split `gate_proj` / `up_proj`，但优化后的 `ForwardSingleGPU`、CUDA graph、MTP 和通用 fallback 只识别合并 `gateup_proj`，随后误判为 MoE 并报 `neither dense MLP nor router gate weight`。修复为在所有执行与 TP 准备路径支持 `silu(gate(x)) * up(x)` 的 split dense MLP，不反量化、不复制权重；全量构建及 `FASTLLM_REGRESSION_ONLY=qwen35_gguf` 通过。
- **集成 MTP 实测**：日志确认 `layers=1, drafts_per_step=2, acceptance=exact`；256-token 请求 8.59s，端到端 29.81 tok/s，位置接受率观测约 `[83.59–92.71%, 71.09–82.81%]`。READY 空载 GPU 约 19009 MiB，256-token 测试后 20523 MiB，V100 仍余 11972 MiB。
- **canary 覆盖**：文本返回 `Q38 READY`；工具调用为 `finish_reason=tool_calls`、`get_weather {\"city\":\"北京\"}`；1×1 红图回答 `Red`；66060-token passkey recall 精确返回 `V100-Q38-739251`（136.43s）。
- **reasoning effort**：OpenAI `reasoning_effort=none|low|medium|high|xhigh` 全链路 HTTP 200。官方模板中 low 注入简短思考指令，medium 使用基线，high 映射到官方 xhigh 指令；确定性算术测试的隐藏推理长度分别为 0/87/80/164/164 字符。该控制是模板提示，不是硬 token 配额；`none` 能关闭 `reasoning_content`，但不能保证最终答案一定简短。
- **生产与回滚**：生产 profile 为 `qwen38-udq4-262k-mtp2-turbo4.env`，tmux `fastllm-prod`，公网别名 `qwen3.8-27b` / `qwen3.8-27b-heretic`，`/health` 为 READY generation 1，两个别名短请求均为 HTTP 200。Qwen3.6 Q6 回滚 profile `q6-ablit-262k-mtp2.env` 保留不变；切回时仍使用同一个 `launch_proxy_tmux.sh`。

### 15.20 Qwen3.8 生产路由、持久前缀缓存与最终性能

- **统一公网模型 ID**：新增 `auto`，由本机 FastLLM、NVIDIA NIM、OpenRouter、OpenCode Zen 按就绪状态、并发、前缀命中、近期失败和历史成功率评分；固定 `qwen3.8-27b` 与 `qwen3.8-27b-heretic` 仍强制走本机，便于保底和诊断。Hermes、OMP、OpenCode 均改用 `auto`，服务端响应恢复调用方请求的公开 ID，不泄露内部 provider slug。
- **真实流式转发**：OpenAI SSE 按上游 chunk 原样增量转发并保留 `[DONE]`；Anthropic `/v1/messages` 将增量 OpenAI SSE 转成唯一的 `message_start`/`message_stop` 序列，文本、thinking 和 tool-use block 均逐块输出。生产实测两个协议均 HTTP 200、零 JSON 解析错误并完整终止。
- **持久缓存崩溃根因**：懒页池的逻辑容量为 2048 pages、初始物理容量仅 128 pages，但旧持久化代码用逻辑容量计算单页字节数，并在恢复时把 `0..2047` 全部暴露为物理空闲页。重启后的首次前缀命中因此拿到 page 2043 等越界页，触发 `cudaErrorIllegalAddress`。修复后所有页字节、空闲页和物化边界都以 `dims[0]` 物理页数计算；序列化版本升到 2，旧缓存自动 fail-open。`persistent_prefix_cache` 回归覆盖 130 logical / 128 physical 的懒池跨进程恢复。
- **最终生产实测**：12776-token 唯一前缀、1-token 输出端到端 17.21s，约 **742.4 prompt tok/s**；256-token 连续解码 TTFT 0.054s、增量区间 5.243s，约 **48.6 tok/s**。本机固定别名返回 `PERSIST-FIX-OK` 后 `/health` 保持 READY；后端常驻显存 18826 MiB。

### 15.21 OMP 多轮工具历史导致 proxy HTTP 500（2026-08-15）

- **现象与边界**：普通聊天请求正常，但 OMP/Hermes 带历史工具调用的请求持续返回 HTTP 500；请求尚未进入 FastLLM 推理阶段。
- **根因**：OpenAI 工具调用历史允许 assistant 消息的 `content` 为 `null` 或省略。Qwen3.8 官方模板在 Jinja `StrictUndefined` 下直接读取 `message.content`，触发 `UndefinedError: 'dict object' has no attribute 'content'`。同一请求显式设置 `"content": ""` 后可返回 HTTP 200。
- **修复**：`fastllm_adapter` 仅对含 `tool_calls` 且缺少文本内容的 assistant 消息补空字符串，并同时规范化模板渲染副本和发往后端的消息；普通消息及后端所需的字符串形式 `function.arguments` 保持不变。
- **验证**：回归测试先复现缺失 `content` 的失败，再验证完整 adapter 套件 16/16 通过。生产重启后原始复现请求返回 HTTP 200，Hermes 客户端随后连续 3 次真实 `/v1/chat/completions` 请求均为 HTTP 200，`/health` 保持 READY 且 `last_error=null`。

### 15.22 多图会话绕过单图像素上限导致 CUDA OOM（2026-08-16）

- **崩溃签名**：后端在第 381 个请求处理期间尝试把 CUDA 临时缓冲从 170 MiB 扩到 1265.2 MiB；随后 `FastllmCudaPermute` 分配失败，设备仅余 51 MiB，再申请 200 MiB 的 `[1,10240,10243]` FP16 张量时触发 fatal `cudaErrorMemoryAllocation`，进程以 code 1 退出并进入自动重载。
- **根因**：Qwen3.5/3.8 图像预处理只限制每张图最多 `28×28×1280 = 1,003,520` pixels，没有限制一个多轮请求内所有历史图片的总量。图片数量增长时，vision patch 和 merger 中间张量线性累加；约 17 张达到单图上限的图片足以产生 1.3 GiB 的单个 permute scratch，并把 V100 推到 OOM。
- **修复**：保留原有单图缩放，再将一次请求的总目标像素限制为单图上限的 4 倍（4,014,080 pixels）。超限时统一按面积比例缩小所有图片并保持图片数量与宽高比；极端长宽比无法落入预算时在进入 GPU 前返回明确错误，不再让 CUDA 分配失败杀死后端。
- **验证**：新增 5 图 aggregate-cap 回归，先确认旧实现以 20,480 pixels 超过 16,384 测试预算，再验证缩放到 5,120 并通过 `qwen35_gguf` 套件。生产用 17×1024² PNG 重放：日志确认从 16,729,088 缩到 3,916,800 pixels，请求 HTTP 200（416.64s），后端保持 READY generation 1、无新 CUDA OOM；请求后 GPU 使用 22,311 MiB、剩余 10,184 MiB。
- **复发根因 1——多模态绕过 chunked prefill**：聚合像素限制生效后，生产 5 图请求仍形成 51,997 个文本+视觉 token。调度器先匹配 `ForwardMultimodal`，再判断 `chunked_prefill_size`，因此直接分配 `[1,10240,51997]` 的 1,015 MiB dense activation；相同请求在 generation 1/2/3 均复现 OOM。修复后先合并视觉 embedding 与三轴 mRoPE，再把 CPU hidden states 按 512 token 切片送入语言模型，KV/线性注意力状态跨片累计；完整 embedding、图像和视频 feature 在切片前立即释放。
- **复发根因 2——页池跨 512 页时倍增并永久保留旧池**：chunked prefill 首次跑到第 516 页时，每层 K/V pool 从 512 一次翻倍至 1,024 页；`Grow` 为回避异步 use-after-free 将每个 512 页旧池留到析构，导致增长中显存再次打满并以 `PagedCacheManager::Grow: failed to allocate larger page pool` 退出。现在 `Grow` 在罕见容量迁移前同步整个 CUDA device，完成 D2D 拷贝后立即 direct-free 旧池；兜底扩容从几何倍增改为每次最多增加 128 页。
- **最终复测**：CUDA 回归先确认旧实现仍保留 superseded pool，且 256 页会直接翻倍到 512；修复后旧指针失效且目标为 384，`paged_cache_grow` 通过。生产连续两次重放 99,985-token、5×1024² PNG 请求，均 HTTP 200（286.37s / 258.48s），峰值分别 30,807 / 30,933 MiB、最少剩余 1,688 / 1,562 MiB；随后短请求 0.83s 返回 HTTP 200。日志均显示 `chunk=512`、`chunks=196`，无新 CUDA error，`/health` 保持 READY generation 1、`last_error=null`。
- **复发根因 3——算子层仍把 1,024 页翻倍到 2,048 页**：上述修复只覆盖兜底取页路径，`CudaAppendPagedCacheOp::Reshape` 仍按 `maxPages*2` 扩容。146K-token 请求在池已接近满载时需要约 1,141 页，却为单个 K pool 申请完整 272 MiB 的 2,048 页新池；旧池在迁移期间仍占用，最终以 `cudaErrorMemoryAllocation` / code -6 退出。现在所有调度器、append、decode、MTP CoW 和兜底取页共用 `GetPagedCacheGrowthTarget`：补足实际缺口并最多额外预留 128 页，不再几何倍增。
- **CUDA graph 与 CPU 页池边界**：`Grow` 释放旧 CUDA 地址前通过全局存储锁与 graph launch 串行，并递增 storage version；Qwen3.5/3.8 graph 在 replay 前发现版本变化会销毁旧 graph、eager warmup 后重捕获，避免 graph 内核重放已释放的 pool pointer。CPU pool 同步实现 realloc/copy，增长后不再只改 `dims/freePages` 而越界访问旧缓冲。
- **最终 146K 生产复测**：自动重放的 146,019-token 多模态请求完成，日志为 `chunk=512` / `chunks=286`，后端保持 READY generation 1、`last_error=null`；GPU 峰值 30,497 MiB、最少剩余 1,998 MiB。池尺寸按 `128→256→384→…→1,152` 增长（实际需求池为 1,141 页），未再出现旧路径的 2,048 页池或任何 CUDA error。随后相同 146K 请求再次 HTTP 200（104.97s，峰值 30,163 MiB、剩余 2,332 MiB），短请求 0.91s 返回 HTTP 200；`paged_cache_grow` 与 `qwen35_gguf` 聚焦回归均通过。
### 15.23 前缀缓存 HIT 后续 prefill 全序列一把梭导致 OOM 自杀（2026-08-17）

- **现象**:proto-ui/z3rm 的 100K+ 带图长会话周期性"挂掉";后端进程每 16-19 分钟消失一次,watchdog 拉起;期间请求表现为 0.00s 假完成(1 token)或连接断开。
- **定位链**:先误判为 hang(`running=2` 零吞吐,实为 metrics 只统计完成的 prefill,进行中不可见);在 OOM fatal 路径加 `backtrace_symbols_fd`(apiserver 链接 `ENABLE_EXPORTS=ON` 即 `-rdynamic`)后 85 秒抓到铁证栈:`Qwen35MTPLoop → ForwardMultimodal → Forward → ForwardV2 → CudaRepeatOp::Run → MallocSpace`,分配 `[1, 61004, 16, 3, 128]`(=GDN kRepeat,16 k-heads×3 扩至 48 v-heads;61004=61516-512 HIT)。
- **根因**:`ForwardMultimodal` 开头 `pastKeyValues 非空 → early return 全序列 Forward`。首次带图 prefill 有 vision chunked(512)保护;**前缀缓存 HIT 后的续 prefill(pastKeyValues 已被恢复引用填充)绕过一切分片**,GDN Repeat/ragged pack 按全剩余序列申请显存。显存够就活(134K 剩 120K 撞 3.9GB 活),不够就 `std::_Exit`(138K 撞 free 1.7GB 死)。
- **修复**:续 prefill 分支按 `GetChunkedPrefillSize()` 分块循环(中间块传 nullptr logits),与首次 vision chunked 同机制;`NeedAttentionMask` 恒 false,分块无 mask 错位风险。同时加 `[req N] prefilling: x/y tok` 5s 节流进度打印。
- **验证**:修复前 16-19 分钟必崩;修复后 28+ 分钟零崩溃,135K prefill 完成(363s),生产 HIT 频繁(14%~95%)正常。提交 `ccde84af`。

### 15.24 视频管线上线:三个连环 bug(2026-08-17)

- **管线**:apiserver `video_url` → `video_loader`(data URL base64 → mkstemp 临时文件 → ffprobe 量尺寸 → ffmpeg 抽帧 rawvideo rgb24,`FASTLLM_VIDEO_FPS=2`/`MAX_FRAMES=32`/`MAX_EDGE=768`)→ `PrepareMultimodalVideoInputs`(尺寸 32 对齐+像素上下限、video_grid_thw、占位符展开、video_frames 张量)→ vision encoder → mrope。
- **bug 1——ffprobe 莫名失败**:错误只有 "exited abnormally"(stderr 被丢进 /dev/null)。stderr 并入捕获管道后真相大白:`symbol lookup error: /lib64/libpangoft2-1.0.so.0: undefined symbol: FcConfigSetDefaultSubstitute`——apiserver 继承的 `LD_LIBRARY_PATH=~/.conda/envs/tsenv/lib` 让系统 ffprobe 加载 conda 的不兼容库。修复:fork 子进程 exec 前 `unsetenv("LD_LIBRARY_PATH")`(只影响子进程)。
- **bug 2——占位符展开格式与引擎不一致**:Prepare 把一个 `<|video_pad|>` 展开成**一大段** T×(H/32)×(W/32) 个 pad;引擎 mrope 侧却把 video_grid_thw 按 T **逐帧重复**(`repeatedVideoGridThwList`),校验每段 == 单帧 token 数——一大段永远对不上,`AssertInFastLLM` 抛异常无人 catch → terminate 杀进程。修复:Prepare 展开为 T 段,段间插 `<|vision_end|><|vision_start|>` 打断 mmTypes 连续扫描;时间维靠段间 currentPos 推进(max(H,W)/2 每段)表达,与引擎设计一致。
- **验证**:`data:video/mp4` base64(2s testsrc2 彩条)→ 200,模型正确描述画面;prompt 167 tok(4 帧 320×240→gridT=2 段)。提交 `e4056b36`。
- **遗留**:引擎 AssertInFastLLM 异常在请求路径无 catch(任何 assert=terminate 全进程),后续应包一层请求级 catch 降级为 500。

### 15.25 Jinja 引擎增强: default 过滤器 + 元组/列表字面量 + in-array (2026-08-17)

- **症状**: Cyber 模型自带 chat_template.jinja 渲染失败退回 MakeInput(图像/视频占位符与
  reasoning 文案全丢): `Jinja Error: unsupport function default`, 修复后变为 `expression error
  near token [] type=33 stack=1`(Filter token 缺操作数)。
- **根因 1——带参过滤器**: `x|default('y')` 词法上 ID 后随 `(` 被转成 FUNC token, 但前面的
  Filter(`|`)token 残留, Shunting-yard 产出的 RPN 里 Filter 求值时栈上只有 1 个元素。
  修复: tokenizer 在 ID→FUNC 转换时吸收前一个 Filter token(带参过滤器语义 = 函数调用,
  被过滤值天然是栈上第一参数)。
- **根因 2——元组字面量**: 模板 48 行 `x not in ('a','b','c')` 的逗号走 Namespace(kwargs)
  分支断言失败。引擎原本连 `[1,2,3]` 列表字面量都不支持(RMB 求值只有下标语义)。
  修复: 开括号入 ops 时标记角色(call=函数参数表 / sub=下标 / grp@n=分组 / list@n=列表,
  n=进入时 suffixExp 深度); 逗号按"是否 name="分走 kwargs 或元素分隔; 闭合括号按角色
  生成 BuildArray(n) 指令; 求值段弹 n 元素组 JinjaArray; `in` 增加 array 线性成员测试。
- **in 左操作数**: 原代码对 In 跳过 local 解析(变量名当 dict key), 违反 Jinja 语义且使
  `var in (tuple)` 永假; 已统一解析(字面量 key `'image' in item` 不受影响)。
- **default 实现坑**: FUNC 求值传入的 args 容器 type=JinjaNone(元素在 arrayValue),
  不能按 JinjaArray 判断; `x|default` 无参形式退化为恒等。
- **验证**: 离线测试程序(/tmp/jt2.cpp 直接编 template.cpp)6 用例全绿 + 生产模板
  带 tools 完整渲染; 线上 warmup 后 prefill 258/298 tok(渲染后长度)且无 fallback 行。
- **诊断**: 5 处 expression error 带 token type/stack 深度; `JINJA_DEBUG=1` dump suffixExp。
- **提交**: fastllm bf9c94ad。

### 15.26 生产切换 unsloth UD-Q4_K_XL + 长上下文"假崩溃"真因: proxy ReadTimeout (2026-08-17)

- **切换**: 生产从 cyber AWQ(W4A16) 切到 unsloth UD-Q4_K_XL(GGUF 17.9GB + mmproj-F16
  927MB 分离加载 `--mmproj`)。动机: Q5_K_M/Q6_K 的乱码(14 坏字节,低频字字节 token 断裂)
  在 Q4_K_XL 上完全消失(生僻字拼音全对, bad_bytes=0); 加载 150-280s(vs cyber 冷启 745s);
  decode 22 tok/s(cyber MTP2 为 30 tok/s, 慢 27% 换来无乱码)。
- **vision**: unsloth GGUF 是纯语言模型, 必须挂 mmproj-F16.gguf, 否则 400
  "no usable vision projector"。图像理解实测正常(大色块构图描述精确; 32px 细条纹是
  patch-merge 池化极限, 非故障)。
- **SKIP_WARMUP**: ComfyUI 常驻 8.8GB 时, 权重(19GB)+mmproj+warmup forward 激活撞 32.8GB
  天花板 OOM(cudaErrorMemoryAllocation 45MB at AutoWarmup→ForwardGPU);
  FASTLLM_SKIP_WARMUP=1 后由首个真实请求自然预热(首请求 ~14s 完成权重上传)。
- **262K 占满"崩溃"真相**: proxy 下 196.9K-token prefill 返回 503 backend_reloading,
  但后端日志显示 prefilling 95% 仍在正常推进——**是 proxy→后端 httpx 读超时**
  (QUEUE_TIMEOUT 600 + 60 = 660s < 实际 prefill ~790s)。干净环境直连(无 proxy)
  同请求 200/787s 完美完成。修复: 生产 env QUEUE_TIMEOUT=1800(覆盖 prefill+decode 余量)。
  VRAM pressure watchdog 不背锅: active>0 时 pressure_stale 不 unload, 设计正确。
- **压力测试全过**: 图像 5 连发(1024px×3+双图)峰值 21.2GB; 196.9K token 占满 prefill
  峰值 24.4GB(+ComfyUI 8.8 = 33.2GB? 实测 24.4 含全部, free ~8.3GB)——余量健康。
- **遗留**: GGUF skip-warmup 下权重延迟到首请求上传, "model loaded" 时 nvidia-smi 仅
  1.2GB 属正常, 勿误判为加载失败。

### 15.27 工具调用空参数根因与修复: 约束生成 + 解析容错 (2026-08-18)

- **症状**(proto-ui/omp): read/bash 工具调用间歇性 `arguments: {}` 空参数
  (Validation failed 连发), 或格式漂移成裸文本(`<path>`/`<fuction=` 拼写错误/
  `<prefix>` 编造参数名)。
- **排查**(omp CLI tmux 复现 + 会话 jsonl 取证):
  1. **温度不是根因**: models.yml extraBody temperature: 0 改为 1 后缓解(~50% 仍空),
     但 temp=0 下 Q4_K_XL 干脆不调工具(官方对齐"我没有文件访问权限")。
  2. **KV 量化不是根因**: turbo4 vs fp8_e4m3(8-bit 全量 K/V)对照, 空参数/漂移同样出现。
  3. **权重档位不是根因**: Q4_K_XL(3.8) 与 Q5_K_M(3.6 heretic) 同样漂移
     (Q5 甚至编造参数名 <prefix>/<max_depth>)。
  4. **真根因**: FastLLM 工具名约束(<function= 后只允许合法工具名)之外的参数块
     **完全自由生成**; 量化模型在 `<parameter=` / 闭合标签等长尾 token 序列上
     概率分布塌缩 -> 跳过参数块 / 拼错标签。模板无锅: Qwen3.8 官方模板就是
     `<function=name><parameter=key>` 格式, merged.jinja 与之一致。
- **修复**(commit ae7151b8):
  1. apiserver 从 tools schema 提取参数名 -> `tool_call_parameter_name_constraint`
     (`<parameter=` 后只放行 schema 参数名 token);
  2. 新增 required 参数块强制: `</function>` 前已闭合 parameter 块数 < required 数时,
     约束只允许 `<parameter=` 前缀链(allowedIds 空时采样层自动不 mask, 无卡死风险);
  3. OpenAIOutputParser.Flush 对未闭合 `<tool_call>` 块尝试 ParseBlock
     (漏 `</tool_call>` 的完整调用不再丢成裸文本); 缺 `</parameter>` 时值延伸容错。
- **验证**: omp CLI temp=1 同任务实测: 修复前 3 连空 + 漂移; 修复后 6+ 调用
  0 空 0 漂移, 探索行为连贯(read path 准确引用 ls 输出里的真实文件)。
- **附带**: models.yml vllm provider temperature 0 -> 1(本地 omp);
  Qwen3.8-27B-Q5_K_M.gguf 已下载备用(models/Qwen3.8-27B-Q5KM/, 19.8GB)。

### 15.28 真根因订正: qwen3_5 投机解码路径从未接线工具约束 (2026-08-18)

- **订正 15.27 结论**: ae7151b8 的约束实现本身正确, 但**对 Qwen3.5 模型从未生效**。
  Qwen3.5 走 qwen3_5.cpp 独立 decode 循环(投机/非投机统一), 该循环:
  1. 装配 generationConfigs 时不调 `PrepareToolCallConstraint`
  2. 接受 token 时不调 `UpdateToolCallConstraintState`
  -> `toolCallConstraintGeneratedText` 恒空 -> 约束分支全部静默 return
  -> `allowedIds` 恒空 -> `Qwen35MtpSupportsGenerationConfig` 的 mask 检查失效
  -> 约束+禁用 MTP 的保护链整体不存在, 量化漂移裸奔。
- **为何之前误判"修复生效"**: T5 验证只跑了 ~6 次调用, 对 ~12% 的间歇空参数率
  是小样本幸存者偏差; 且测试走 mini-proxy 直连(非生产链), 变量不唯一。
- **确诊路径**: dump proxy 抓 proto-ui 真实流量 -> 简化 system prompt 8/8 好,
  完整 35K omp prompt 非流式 8/8 好, **流式 1/8 空** -> 流式专属只是表象,
  实质是约束全无效的随机漂移, 样本量上去就现形。
- **修复**(commit cd2c0488): qwen3_5.cpp 装配点调 Prepare + 接受循环逐 token
  Update + selectedNeedLastTokens 补 allowedIds 分支。生产链路流式 16/16 参数完整。
- **附带发现**: `Qwen35MtpSupportsGenerationConfig` 在 repeat_penalty≠1 时也禁用
  MTP; 生产 proxy 注入 repeat_penalty=1.05 -> **生产所有请求 MTP 实际禁用**
  (30 tok/s 的 MTP2 基准是直连无 penalty 测的)。若要让生产吃到 MTP 加速,
  需要把约束 mask 引入 MTP 接受路径, 或评估 penalty 是否可放宽。

### 15.29 MTP 投机路径接线工具约束 + repeat_penalty 放开 (2026-08-18, commit d9d0aec5)

- **背景**: 15.28 留下两条——(a) 约束激活边界处的投机漂移(draft 提议跨越
  `<parameter=`→`>` 等边界时 verify 沿用旧空 mask, 漂移 token 被接受,
  proto-ui 实机复现: 多余 `<parameter=` + 转义 `</function>`); (b) 生产
  repeat_penalty=1.05 导致 MTP 全禁(decode 30→4.8 tok/s)。
- **(a) 约束感知截断**: 新公开方法 `basellm::EvaluateToolCallConstraintText`
  (从 PrepareToolCallConstraint 提取的纯文本版, 不依赖 ResponseContext)。
  单请求 `countAcceptedDrafts` 与 batch 接受 while 循环对**每个候选 token**
  做假设性预检: 临时文本推进约束状态机, 若约束即将激活且候选不在
  allowedIds → 拒绝并截断投机块; 下一步 Prepare 产生正确 mask, 走非投机。
  KV cache 只提交到截断点, 无回滚问题。
- **(b) repeat_penalty 下启用 MTP**:
  - gate `Qwen35MtpSupportsGenerationConfig` 不再拒绝 repeat_penalty
    (mask 非空仍禁——工具调用窗口短, 走非投机更简单可靠)。
  - verify 前向传真实 `LastTokensManager`(单请求 context->tokens,
    batch 逐 request contexts[b]->tokens; 此前恒传空 manager)。
  - `FastllmCudaTopKTopPSamplingWithTypicalAcceptance` 增加可选
    penaltyIds/penaltyFactors/penaltyTokens 参数, 前置
    `FastllmRepeatPenaltyFactorsKernel` 就地作用于 logits——typical
    acceptance 的 posterior 基于惩罚后分布, 与 CPU LLMSampling 同语义
    (last_n>0 时 pow(penalty, count))。
  - 采样调用侧把 per-request penalty 集展开成 per-row 数组(投机 verify
    的行 = request × seqLen)。
- **近似与取舍**: verify 各行共用请求级历史, 不把块内 draft token 计入
  penalty(逐位置精确历史成本高; llama.cpp 等同款近似)。mask 场景
  (allowedIds 非空)仍禁 MTP, 窗口仅工具调用内 ~30 token。
- **验证**: 见 inbox-fastllm.md 三条验收信号(生产链路流式 16 次工具调用 /
  penalty 下无 MTP-not-enabled 日志 / decode ~30 tok/s)。

### 15.30 生产链路端到端排查: 三个独立根因 + 一个静默路由陷阱 (2026-08-18)

排查者: macOS 端 Claude(与 fastllm pane 的 K3 分工协作)。口径: **一切以 omp 实战为准**,
不以单项指标为准 —— 15.28 已经证明小样本探针会产生幸存者偏差。

#### 0. 静默云端 fallback: 让此前所有"链路测量"都可能无效

- **现象**: 20 格协议矩阵全部 `no_tool_call`, `prompt_tokens` 只有 77,
  模型在 reasoning 里写"工具是什么? 大概能用 bash 吧"。
- **根因**: `thinking_proxy.py` 只有当 model 名匹配 `FASTLLM_PUBLIC_ALIASES`(子串)时才
  `_must_route_local`; 否则走 `_pick_backend` → OpenRouter / NIM / Zen **云端**, 且不报错。
  当时后端被挂成 `q5-262k-mtp2.env`(ThinkingCap-Qwen3.6-27B-heretic-Q5_K_M),
  别名只有 `qwen3.6-*`, 而我们打的是 `qwen3.8-27b`。
  proxy 日志实锤: `[route] routing to or` → OpenRouter 403 → `[route] routing to nim`。
- **修复**: (a) 所有实验/生产 profile 设 `FALLBACK_ENABLED=0`, 路由错配显性失败;
  (b) `scripts/chain_acceptance.py` 开头强制 preflight —— 目标模型不在 `/v1/models`
  就 `SystemExit`; (c) 换 profile 后先 `curl /v1/models` 确认别名。
- **教训**: "模型忽然变傻/不调工具"先查 `/v1/models` 与后端 `--path`, 再怀疑模型。

#### 1. 根因 A: 流式路径的 read timeout 硬编码 600s

- `_open_backend_stream(url, body, timeout=600)` 用 `httpx.AsyncClient(timeout=600)`。
  httpx 的 read timeout 是**两个 chunk 之间的上限**, 不是整条流的上限。长 prefill 期间
  后端一个字节都不发: 196.9K token 实测 ~790s ⇒ **必然断流** → `local_failed=True`
  → 甚至 fallback 云端。非流式走 `QUEUE_TIMEOUT`(生产 1800)所以从未暴露。
  **omp 永远流式**, 所以这是"长上下文一上 omp 就寄"的一个独立根因。
- **修复**: 改 `httpx.Timeout(read=STREAM_READ_TIMEOUT, connect=15, write=120)`,
  `STREAM_READ_TIMEOUT` 默认跟随 `QUEUE_TIMEOUT`。

#### 2. 根因 B: required 参数约束是"按块数"计数, 不是"按参数名"

- `include/fastllm.h:183` 是 `std::map<std::string,int> tool_call_required_parameter_counts`
  —— 只存**个数**。判定是"已闭合 parameter 块数 ≥ required 数就放行 `</function>`"。
  于是 `grep` 的 required=["pattern"](count=1)时, 模型输出一个 `<parameter=case>` 块
  就满足 ≥1 → 允许闭合 → **必填参数缺失**。
- **实证(omp 真实负载, 旧二进制基线)**: 14 次工具结果 / **3 次** validation 失败,
  全是 grep 漏 `pattern`, 参数分别是 `{case:true}`、`{path:...,skip:0}`、
  `{original:{case:"False",path:...}}`。模型自己在 thinking 里写
  "I keep forgetting to include the pattern."
- **协议矩阵佐证**: 40 格里 3 例失败**全部落在 medium effort 档**(3/8 = 37.5%),
  其余 32 格 0 失败, 均为 `missing_required_arg:path`。
- **修复**(K3): 改成缺失名集合判定 —— `CollectClosedToolCallParameterNames` +
  `MissingRequiredToolCallParameters`, 参数名位置 required-first(交集为空回退全量,
  allowedIds 空时采样层不 mask, 无卡死)。投机路径经 `EvaluateToolCallConstraintText`
  自动继承。
- **意义**: ae7151b8 的"required 参数块强制"从来没真正生效; cd2c0488/d9d0aec5 把它接进
  投机路径, 接的是一个本身就漏的约束。这就是"指标上修好了但一上 omp 还是寄"的另一半。

#### 3. 根因 C: 长上下文下工具调用**结构破损**(需要完整语法约束, 不是继续打补丁)

- **现场**(TB14P 上 omp 打生产 endpoint, 上下文 ~31K/240K):
  ```
  <tool_call>
  <function=read><parameter=path>i</parameter><parameter=
  </function>
  </tool_call>
  ```
  值是垃圾(`i`)、第二个参数名没写完就换行、参数块没闭合就 `</function>`。
- **为什么打补丁治不住**: 现有约束是"在几个点位打点"(函数名、`<parameter=` 名字、
  required 计数)。模型只要在**任何没被打点的位置**跑偏(值结束/标签闭合/块边界)就破损;
  上下文越长、量化越狠越容易命中。
- **方案**: 把 `<tool_call>…</tool_call>` 整段做成受约束的小语法状态机
  (S0 `<function=` → S1 函数名 → S2 参数块之间(必填齐了才允许 `</function>`)
  → S3 参数名(required-first) → S4 值(屏蔽会开始 `</function>`/`</tool_call>` 闭合序列
  的 token, 只有 `</parameter>` 能离开) → S5 只允许 `</tool_call>`),
  从构造上让破损不可能出现; 整套可 env 关掉便于 A/B。
- **配套诊断**: `FASTLLM_TOOLCALL_TRACE=1` 打状态机转移 + allowedIds 规模 +
  破损时 dump token ids; `/props` 暴露 `toolcall_blocks_total` /
  `toolcall_malformed_total` / `toolcall_repaired_total` /
  `toolcall_constraint_masked_tokens`; **解析器走"修复"分支也要计数打日志**
  —— 静默修复等于我们看不见问题。

#### 4. 已排除的嫌疑: chat template 与 tokenizer

- 我们在用的 `chat_template.jinja` 与 `Qwen/Qwen3.8-27B` 官方**逐字一致**
  (md5 `519239a4908bb1f805bbce5fa8c8a242`), `tokenizer_config.json` 同样一致
  (md5 `e843642217637a5738bc9b86021a3eef`), cyber AWQ 仓库的那份也是同一个 md5。
  ⇒ 模板/tokenizer_config 这条线可以关掉了。
- **但**官方 `generation_config.json` 是 `temperature 1.0 / top_k 20 / top_p 0.95`,
  而 `fastllm_adapter.prepare_fastllm_body` 注入的是 `0.6 / 0.8 / 20` —— 偏离官方,
  已列入采样档对比实验。

#### 5. Anthropic 侧 thinking effort 此前完全不可控

- `/v1/messages` 的转换器只写 `chat_template_kwargs.enable_thinking`, **从不写
  `reasoning_effort`** ⇒ 模板恒取默认 `xhigh`; 且硬写 `temperature=1.0/top_p=1.0`
  覆盖了 Qwen 推荐采样。
- **修复**: 从 `output_config.effort` / `thinking.type=="max"` / `thinking.budget_tokens`
  推导 effort(≤2048→low, ≤4096→medium, 否则 xhigh); `effort=="none"` 真正关思考;
  采样参数只在客户端显式给出时才透传。
- **顺带确认**: thinking **本来就是逐块流式**下发的 —— OpenAI 路径 124 个
  `delta.reasoning_content` 小块(每块 1–3 字), Anthropic 路径 132 个 `thinking_delta`。
  客户端看不到流式 thinking 是客户端侧的展示/兼容配置问题, 不是 API 不支持。

#### 6. MTP 在 repeat_penalty 下恢复(K3 的 d9d0aec5 实测有效)

- 修复前后端日志: `[Qwen3.5 MTP] not enabled: ... repeat_penalty=1.0500 ...`
  —— `fastllm_adapter` 默认注入 `frequency_penalty=1.05`(apiserver 直传 repeat_penalty),
  而 gate 在 penalty≠1 时直接禁用投机 ⇒ **生产所有请求 MTP 全禁**。
- 修复后同一生产链路: `[Qwen3.5 MTP] enabled: layers=1, drafts_per_step=2,
  acceptance=typical(0.09/0.30)`, `pos_accept_rate=[100.00%, 92.19%]`,
  decode **23–31 tok/s**(修复前 17–22)。
- `fastllm_adapter` 的默认值已改为读 `FASTLLM_DEFAULT_FREQUENCY_PENALTY`(默认仍 1.05),
  便于 A/B 证明"MTP 在 penalty 下真生效"而不是靠绕开 penalty。

#### 7. 前缀缓存: 三级机制都在, 但默认值让 agent 负载**永远够不着门槛**

- 实测 agent 轮次: `prefix-cache HIT(mem-trie): 4096/24641 tok (17%)`,
  `L2disk=0.0MB`, `hits=0`, `kv_pool` 被两路长请求打满到 99% 后上一轮前缀即被逐出。
- 查到的默认值: `FASTLLM_PREFIX_CACHE_CPU_TIER` 默认 **false**(RAM 层根本没启用);
  `FASTLLM_PREFIX_CACHE_MIN_TOKENS` 与 `DISK_MIN_TOKENS` 默认 **65536**
  —— agent 每轮前缀才 20~30K, **永远达不到准入门槛**;
  生产 profile 又只给 `DISK_MAX_BYTES=2GiB`(磁盘实际余 2.2TB)。
  另有 `RECOMPUTE_TPS`(默认 800)、`CPU_READ_MBPS`(10000)、`ZSTD`(默认开, level 1)
  等代价模型输入。
- ⇒ 磁盘 offload 的本意(200K 上下文重 prefill 要 ~200s+, 而 ZSTD 压缩后从盘上读回
  即使 300MB/s 也远快于重算)在当前配置下**完全没被触发**。
- 已生成 `cachetuned` 扫描档: `CPU_TIER=1` / `CPU_MAX_BYTES=16GiB` /
  `MIN_TOKENS=DISK_MIN_TOKENS=4096` / `MIN_HITS=1` / `DISK_MAX_BYTES=200GiB` /
  `ZSTD_LEVEL=3` / `SNAPSHOT_INTERVAL_PAGES=8`, 进矩阵对比。

#### 8. 工具链: 为什么原来的启动器不能用于自动化

- `scripts/launch_proxy_tmux.sh` 结尾是 `tmux attach-session`, 且挂了
  `trap cleanup EXIT`(cleanup 会 `stop_session`)。**无 TTY 时 attach 失败 → 触发 cleanup
  → 把刚创建的 session 杀掉**, 非交互环境下必然自毁。
- 新增 `scripts/sweep_launch.sh`: 不 attach、不挂 EXIT trap, 起完轮询 `/health` 到 ready
  才返回; 并且每轮清空后端日志、扫掉任何存活的 `apiserver --path`(换模型时上一轮残留
  会一直占 20GB+ 显存)。

#### 9. 新增工具(都在 v100-perfs)

- `scripts/chain_acceptance.py`: 生产链路验收探针。suites =
  `matrix`(双协议 × 流式/非流式 × 五档 effort 的工具调用保真度 + thinking 开关)、
  `toolloop`(多轮工具历史往返)、`concurrency`(K 路并发)、`longctx`(大上下文 + 重放命中)、
  `garble`(低频字/多字节保真)、`bench`(prefill/decode/**e2e** 三个速度, 优先取后端日志里
  引擎自己打的 `[req N] done:` 拆分)。自带失败样本原始 SSE 转储与解码循环检测。
- `scripts/sweep_profiles.py`: 表驱动生成 16 份扫描 profile
  (模型 × KV量化 turbo4/turbo3/fp8_e4m3 × MTP 2/0 × SM70 算子 × 缓存策略)。
  注意 `--kv_cache_dtype turbo3/turbo4` **还必须配 `FASTLLM_QWEN35_TURBO3_KV/TURBO4_KV=1`**,
  否则后端启动即抛异常。
- `scripts/sweep_one.sh`: 单档执行器(冷启计时 → 套件 → 显存峰值采样 → 汇总一行 JSON)。

### 15.31 工具调用约束的四层剥洋葱 + 语法状态机首版回归 (2026-08-18 上午)

这一节记录"工具调用为什么一直修不干净"的完整剥离过程。每一层都是**独立**缺陷,
修掉上一层才会露出下一层 —— 这也解释了为什么此前每次都"指标上修好了, 一上 omp 还是寄"。

| 层 | 现象 | 根因 | 状态 |
|---|---|---|---|
| L1 | 参数块整个不出现 / 标签拼错 | 约束只管工具名, 参数块自由生成 | ae7151b8 加参数名约束 |
| L2 | 修了却没生效 | Qwen3.5 走 `qwen3_5.cpp` 独立 decode 循环, 从未接线约束 | cd2c0488 / d9d0aec5 |
| L3 | **必填参数缺失**(`grep{case:true}`) | `tool_call_required_parameter_counts` 是 `map<string,int>`, **只记个数不记名字**; 发一个可选参数块就满足"块数≥required 数" | a13833cc 改为缺失名集合 + required-first |
| L4 | **空值**(`{"path":""}`) | 名字有了, 但 `<parameter=path>` 之后可以立刻闭合 —— 没有"值非空"约束 | 待修(已提设计) |
| L5 | 长上下文**结构破损**(`<parameter=path>i</parameter><parameter=` 然后 `</function>`) | 打点式约束覆盖不到值结束/标签闭合/块边界 | 语法状态机 409412d6(**首版有回归, 见下**) |

#### L3 的实证(旧二进制基线)
omp 真实负载 14 次工具结果 / **3 次** validation 失败, 全是 grep 漏 `pattern`:
`{case:true}`、`{path:...,skip:0}`、`{original:{case:"False",path:...}}`;
模型自己在 thinking 里写 "I keep forgetting to include the pattern."

#### L4 的实证(a13833cc 之后)
n5 档 40 格矩阵 5 例失败, **全部落在 medium effort**:
```
tool_calls: [{"name":"list_dir","arguments":"{\"path\":\"\"}"}]
```
名字在、值是空串。为什么集中在 medium: 官方模板里 `reasoning_effort=medium` 对应的
`reasoning_instructions` 是**空字符串**(模板只给 xhigh 与 low 写了指令文本),
medium 的系统提示直接从 `# Tools` 开始, 思考更短更"赶"。
⇒ 修法(源头): 语法状态机的 S4 在产出至少 1 个非空白 token 之前, 屏蔽 `</parameter>` 的起始 token。

#### 语法状态机首版(409412d6)的回归 —— 必须记录
n4 档(该二进制第一次上生产链路)实测:
```
openai/block/off : decode_loop_in_text:'\n</parameter><parameter=pattern>\n\n      '  text=6278c lat=219.3s
openai/block/high: decode_loop_in_text:'paath>\n/etc\n</parameter><parameter=maax_'   text=3084c lat=157.9s
引擎侧: [ToolCall] MALFORMED unterminated block (6278B) / (3084B)
```
三个要命细节:
1. 放行了**不在本次 schema 里**的参数名(`pattern`; 当次工具只有 `list_dir(path*, max_depth)` /
   `read_file(path*)`);
2. 出现 `paath` / `maax_` 这类**被破坏的名字** —— 像 mask 后重采样把 token 拼错;
3. **永远关不掉** `</function>`, 在 `</parameter><parameter=` 之间循环到 max_tokens,
   一次请求烧 219s。
⇒ 怀疑 S3 的 allowedIds 与"已闭合参数名集合"解析在**多参数块之后**错位。
建议的不变量测试: 造一个已写完 `<parameter=path>/etc</parameter>` 的前缀, 断言 allowedIds
必须同时包含 `</function>` 与 `max_depth` 的起始 token, 且**不含**任何非 schema 名字的起始 token。
⇒ 处置: 扫描矩阵全部 profile 加 `FASTLLM_TOOLCALL_GRAMMAR=0` 退回打点式, 保证性能数据不被污染;
修好后单独做 `GRAMMAR=1/0` 的 A/B(口径: `--suite matrix --repeat 4` 160 格 0 破损 0 循环 +
`--efforts medium --repeat 8` 0 空参数)。

#### 测量纪律(这一轮踩到的)
- **探针必须识别 SSE 里的 `{"error":...}` 事件**: 后端在懒加载权重/显存水位窗口会返回
  503 `backend_reloading` 然后 `[DONE]`。不识别就会把空响应记成"模型把字丢了"——
  n1/n2/n4/n5 的 garble 全是这么误报的, 数据作废。
- **suite 顺序有讲究**: 冷启后头一两分钟后端还在铺权重(`SKIP_WARMUP=1`), 又有外部流量
  抢显存。把耗时最长的 matrix 放最前面当"热机", 再跑对空响应敏感的 garble。
- **preflight warmup 要用有长度的请求**: 8 token 的探针会"通过"但后端下一刻仍会 503。
- **换 profile 后核对二进制 mtime 与后端启动时间**: n1 档 06:25 起, 而修复版二进制 06:33
  才编出来 —— 那一档的工具调用数据属于修复前, 不能混进结论。
- **外部流量会污染 bench**: 日志里出现 `req#70 total=154949` 这种 15 万 token 的请求
  (真实使用), 会把 KV pool 打满、改变前缀缓存命中。受影响档位需复测。

#### 前缀缓存: 主因是 no-record, 不是逐出也不是门槛
`FASTLLM_PREFIX_CACHE_STATS=1`(256c0af3)在 n5 档给出:
```
periodic: reqs=64 hitReqs=8 hitTok=65024/151455 (mem=65024 cpu=0 disk=0)
  miss{no-record=55 evicted=0 below-thresh=0 gen=0 restore-fail=0 other=1}
  record{ok=0 rej-min=0 rej-cap=0 rej-space=0 rej-other=0}
  resident{mem=149MB cpu=0MB disk=0MB}
```
- `record{ok=0}` 且所有 rej 计数为 0 ⇒ 记录路径**根本没走到计数点**,
  这推翻了"MIN_TOKENS=65536 门槛挡住了"的先验猜测(调低门槛并未让记录发生)。
- 二三级从未落数据(即使 `CPU_TIER=1`), 一级只常驻 149MB(装不下一个 32K 前缀)。
- 有记录时命中极好: `req#66 total=9768 hit=9216 (94%)`。
⇒ 下一步是给 `TryRecordPagedCache` 的每个 early return 打原因计数, 定位卡在哪一条。

### 15.32 扫描矩阵结果与生产选型: 解限版 Cyber Q5 胜出 (2026-08-18)

口径: 全部走生产链路 `nginx:80 → thinking_proxy:8000 → fastllm:8002`;
prefill/decode/e2e 取后端日志里引擎自己打的 `[req N] done:` 拆分;
工具/乱码/多轮/并发列是 `失败数/总数`; 探针 `scripts/chain_acceptance.py`。

#### 解限版 (Cyber, philbert440/Qwen3.8-27B-Uncensored-Cyber-GGUF + graft MTP 头)

| 档 | 模型 | KV | MTP | SM70 | 冷启 | 显存峰值 | 接受率 | 前缀命中 | prefill@8K/32K | decode@8K | 工具 | 乱码 | 多轮 | 并发 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c1 | Cyber Q5_K_M+mtp | turbo4 | 2 | 关 | 337s | 25.6G | 76.3/56.2% | 0.935 | 775/612 | 26.0 | **0/40** | **0/3** | 0/2 | **0/6** |
| c6 | Cyber Q5_K_M+mtp | turbo4 | 2 | 开 | 335s | 25.6G | 95.1/86.8% | 0.935 | 775/610 | 26.1 | **0/40** | **0/3** | 0/2 | **0/6** |

#### 普通版 (官方 Qwen3.8-27B)

| 档 | 模型 | KV | MTP | SM70 | 冷启 | 显存峰值 | 接受率 | 前缀命中 | prefill@8K/32K | decode@8K | 工具 | 多轮 | 并发 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n1 | UD-Q4_K_XL | turbo4 | 2 | 关 | 342s | 24.5G | 84.3/70.8% | 0.979 | 600/503* | 28.4 | 2/40 | 0/2 | - |
| n4 | UD-Q4_K_XL | turbo4 | **0** | 关 | 306s | 23.8G | - | 0.980 | 751/595 | **13.5** | 2/40 | **2/2** | **4/6** |
| n5 | UD-Q4_K_XL | turbo4 | 2 | 关 | 506s | 23.8G | 86.7/72.8% | **0.869** | 694/546 | 28.4 | 5/40 | 0/2 | 0/6 |
| n6 | UD-Q4_K_XL | turbo4 | 2 | 开 | 310s | 22.2G | 78.7/58.2% | 0.935 | 756/604 | 28.4 | 3/40 | 0/2 | 2/6 |
| n7 | Q5_K_M | turbo4 | 2 | 关 | 342s | 27.2G | 78.9/59.5% | 0.968 | 701/567 | 26.6 | 1/40 | 0/2 | 0/6 |

`*` n1 是当晚第一档, 与 Cyber GGUF 的 graft(21GB 读写)并发跑在同一块机械盘上,
prefill 数字偏低, **不能与后面几档直接比**(见下文"被污染的测量")。

#### 长上下文(仅 n1/n7/c1 测了 128K)
| 档 | prefill@128K | decode@128K | 冷 TTFT | 命中重放 TTFT |
|---|---|---|---|---|
| n1 UD-Q4_K_XL | 286 | 5.0 | 543s | 44s |
| n7 Q5_K_M | 306 | 4.6 | 506s | 104s |
| c1 Cyber Q5 | 305 | 4.9 | 510s | - |

⇒ **decode 随上下文塌陷**: 46→29→15→5 tok/s(200/8K/32K/128K)。262K 的真实瓶颈不是显存
而是 attention 成本。**建议 agent 侧把工作上下文压在 ~64K 以内**(靠 compaction),
把 262K 当硬上限而不是日常工作区。

#### 降智/拒答对照(scripts/intel_probe.py: 5 道可自动判分的能力题 + 5 道网络安全防御题)
| 档 | 能力题 | 安全题实质作答 | 拒答 |
|---|---|---|---|
| n7 普通版 Q5 | 5/5 | 4/5 | 0 |
| **c1 解限版 Cyber Q5** | **5/5** | **5/5** | **0** |

⇒ **Cyber 版没有可测到的降智**: 能力题持平, 网络安全防御类题目反而全部实质作答。

#### 结论与选型
1. **量化档位**: Q5 明显优于 Q4 —— 工具调用参数漂移(空值)随量化误差上升:
   Cyber Q5 **0/40**, 普通版 Q5 1/40, 而 Q4_K_XL 各档 2~5/40。
   Q4_K_M 已知有隐患(未进候选), Q6 显存不够跑 262K。
2. **MTP 必须开**: 关掉后 decode 从 28.4 掉到 13.5(约 2 倍差距), 而且**多轮工具调用
   开始掉线**(toolloop 2/2 失败、并发 4/6 失败)。
3. **KV 量化**: turbo4 可用; fp8 档的数据这轮无效(见下), 待重测。
4. **SM70 算子**: c1(关) 与 c6(开)在 prefill/decode 上**没有可测差异**;
   之前"SM70 提升 prefill 26%"的印象来自与被污染的 n1 对比, 不成立。
   ~~但 c6 的 MTP 接受率显著更高(95.1/86.8% vs 76.3/56.2%), 且全绿, 故生产选择开启。~~

   **【2026-08-20 订正: 上面这条归因不成立】** c1 与 c6 跑的是**同一条 kernel**,
   接受率差异不可能来自 SM70 开关。源码依据:
   - `FASTLLM_CUDA_SM70_PAGED_XQA` -> `FastllmCudaTrySm70PagedAttentionDecode`,
     入口要求分页 K **和** V 都是 `FLOAT16`
     (`src/devices/cuda/attention/paged/fastllm-paged-attention-native.cu:2030-2031`)
   - `FASTLLM_CUDA_SM70_FLASH_ATTN` -> `FastllmCudaTrySm70FlashAttentionPrefill`,
     要求 `FP8_E4M3` 分页 KV 且 `kvLen<=512`、每请求 `qLen 2..10`

   而 c1/c6 两组的 KV 都是 **turbo4**, 两个条件都不满足 —— 开关取 0 还是 1
   **完全等价**。实证: 两组的后端日志**逐行相同**; 全部 backend 日志里
   grep `XQA` / `SM70 flash` **零命中**。

   那么 95.1/86.8% vs 76.3/56.2% 的差异来自哪里? **未知** —— 只能归为未受控的
   轮次差异(负载、缓存状态、请求内容)。这提醒: **接受率是个噪声较大的指标,
   单轮对比不足以支撑选型结论**。

   另: `FLASH_ATTN=0` 有**独立的正当理由**, 见 `fastllm_benchmark.md:345,353` ——
   该实验路线端到端输出与 native **不等价**(token 数与 SHA-256 都不同)且更慢
   (1 agent steady 10.00 -> 4.64 tok/s)。即使将来换到 fp8 KV 也不该直接打开。
   而且 `EnvEnabled()` 未设时返回 false, 所以写 `FLASH_ATTN=0` 等于没写。

   现已在代码里加了 `FastllmReportSm70AttentionRouteOnce()`: 只要显式设过这两个
   变量, 启动时打印一次"该开关在当前 KV dtype 下不起作用", 避免以后再有人
   误以为在调优。(commit 614845d1)
5. **缓存"调优"是负优化**: `CPU_TIER=1 + MIN_TOKENS=4096 + DISK_MAX=200GiB` 反而把
   前缀命中从 0.979 打到 0.869, 且 `record{ok=0}`。缓存要按 15.31 的思路先修记录路径。

**生产部署**: `runtime/fastllm-native-profiles/q38-PROD-cyber-q5-mtp2-turbo4-sm70.env`
= Cyber Q5_K_M+mtp / turbo4 / MTP2 / batch2 / 262K / SM70 开 / 默认缓存 /
`FALLBACK_ENABLED=0`(禁止静默转云端) / `FASTLLM_TOOLCALL_GRAMMAR=0`(语法状态机修好前退回打点式)。
部署后生产链路复验 **matrix 20/20**。

#### 被污染的测量(记录下来避免以后重蹈)
- **n1**: 与 21GB 的 GGUF graft 抢同一块机械盘, prefill 偏低。
- **n3(fp8)**: 后端在 10:47 起来了, 但我在 10:49 另起了一个 driver 去重跑 c1,
  `sweep_launch` 的 stale-apiserver 清理把 n3 的后端杀了 ⇒ **n3 的 suite 实际跑在
  Cyber Q5 的后端上**, 那份 "0/40" 是 Cyber Q5 的成绩, `mtp_enabled=False` 也只是
  因为从被杀后端的空日志取值。⇒ **同一时刻只能有一个 driver**; 否则探针照样拿 200,
  测量会**静默串档**。
- **garble 探针**: 整晚 0/3 全是假失败 —— 流式路径里后端 503 事件被当成"字丢了"。
  改非流式后 Cyber Q5 3/3 通过; Q4_K_XL 直连也逐字复现
  `饕餮 魑魅魍魉 齉齾 靐龘 兲丆 瓛璩 黼黻 龏龗 / αβγ ∑∫∂ ✓✗ 🌸🔧🧪 ①②③ ｱｲｳ` 零坏字节。
- **外部流量**: haru 自己在 TB14P 上用生产 endpoint 跑 omp(见到 15 万 token 的请求),
  会打满 KV pool 并改变命中率。受影响档位需复测。

### 15.33 前缀缓存 no-record 根因: 混合注意力双门槛 + 262K 逻辑上限超配物理显存 (2026-08-18)

**现象**: agent 场景 req#2 total=133558 hit=0 miss=no-record, L1trie=0 恒空;
每轮 130K+ 全量重算 ~7min (128K prefill ≈305 tok/s), 双槽占死, 客户端超时重试放大成几十路并发。

**根因链 (Qwen3.5 hybrid 结构性 bug)**:
1. `TryRecordPagedCache` 主路径对含 linear 层的模型整体 gate 在 `TryRecordPagedPrefixCacheExtra` 返回 true 上;
   extra false → skip-linear-bounded → 主路径 full-attn 页也不记 → L1 trie 恒空。
2. extra 的两个 early-return 把 agent 请求全挡死:
   - **页对齐**: 要求 `currentLen % pageLen == 0`。生成结束记录时 currentLen 含生成 token, 几乎从不 128 对齐 → skip-len。
   - **MTP 严格同步**: 要求 mtpCaches 存在且 `tokens == currentLen`。prefill 期 MTP cache 未建 (decode 才懒建) → skip-mtp;
     结束时 MTP tokens = 未对齐总长 ≠ 对齐前缀 → 也挂。
3. bench 能命中 (0.979) 是因为其记录在非 MTP/对齐场景碰巧通过, 与 agent 路径不同。

**修复 (qwen3_5.cpp + fastllm.cpp, 编译 rc=0)**:
- 记录侧: currentLen 向下页对齐 (记对齐前缀, 不整条丢弃); MTP cache 缺失/不同步 → mtpValid=false 继续记 linear+full-attn。
- 查询/恢复侧: 两轮 find (MTP 快照优先, 无 MTP 兜底命中); 恢复时无 MTP → mtpCaches 留空, 首个 draft 步自动重新 seed (该步无投机, 功能正确)。
- 打点: reqSeq%64→%8; extra-mtp-degraded 并入 mtp= 计数。
- **验证**: gen3 req#2 total=31451 **hit=512 layer=mem-trie** —— L1 trie 首次命中, 修复生效。

**第二个独立问题 — 262K 逻辑上限 vs 32GB 物理**:
- turbo3 KV ≈65KB/token → 262K 满装 ~17GB; 权重 21.7GB + 运行时 ~1.5GB → 物理容量仅 ~140K token。
- 懒页池按需 Grow, 日常 <<100K 从未触顶; agent 单请求 148K prefill 首次顶穿 (kv_pool 98%, free 0.4GB) → gen1 abort(-6), gen2 segv(-11)。
- **前缀缓存修复正好治本**: 后续轮次共享前缀页, 双槽总占用 ≈ 共享前缀 + 增量 ≈ 140K+ε。
- 用户决策: 切 **turbo4** (KV 更高压缩 → 280K+ 物理可行), 同二进制 env 切换 (TURBO4_KV=1, --kv_cache_dtype turbo4)。

**口径修正**: 记录 snapshot interval 默认 64 页=8192 tok (首次记录不节流); 结束记录若不对齐 8192 被 skip-interval,
影响仅末段 <8K 未记, 前段命中仍省 ~99% prefill。

### 15.33 多 agent 服务的容量事故 + prefill 优化调研 (2026-08-18 下午)

#### A. 事故: "几十路并发把后端搞爆" 的真实机制

现象: z3rm / proto-ui 被唤醒后, 出现几十路并发, 疑似不停 retry; 重启无效。

根因链(三环叠加, 第 3 环是这次改出来的):
1. **真并发只有 2**: 后端 `--batch 2`, proxy 流式槽位也是 2(刻意对齐)。
2. **单轮可以跑十几分钟**: 常驻 agent 用 `thinking high`(经 adapter 映射成 xhigh),
   加上 `--default_max_tokens 16384`, 一轮近万 token ≈ 9~10 分钟。两个这样的 agent
   就能把 2 个槽位长期占死。
3. **流式改成"排队等"后队列无界**(为修"显存压力直接 503"引入): 后续请求全堆在
   `acquire_stream`, 客户端各自超时重试, 每次重试再追加一份 ⇒ 表面几十路"并发",
   实际绝大多数在空等。proxy metrics 形态: `streams=2/2 inflight=[315s~0tok,314s~0tok] queued=N`。

⇒ **重启无效**, 因为重启只清队列, 三环都还在。

**但更底层的原因是前缀缓存完全没记录**:
```
[PrefixCache] req#2 total=133558 hit=0 layer=- miss=no-record
L1trie=0 pg (~0 tok)
```
13.3 万 token 的请求命中为零 ⇒ agent 每轮把整个上下文从头重算(128K prefill ≈ 305 tok/s,
一轮 ~7 分钟)。对比 bench 场景(同前缀立刻重放)命中 0.979 —— 差别在于 agent 的前缀是
**逐轮增长**的。

根因(在 `qwen3_5.cpp::TryRecordPagedPrefixCacheExtra`): 记录要求
`currentLen % pageLen == 0`, 而 generation-end 的调用包含生成 token, 长度几乎不可能页对齐;
整个记录路径又以这个 extra 返回 true 为前提 ⇒ **L1 trie 永远是空的**。
修法: 记录**页对齐前缀**而不是拒绝未对齐尾巴; MTP 快照缺失时降级而不是丢弃整条记录。
(部署后观察到首次出现 `hit=512 layer=mem-trie`, 但仍以 `miss=other` 为主, **尚未到位**,
继续排查中。)

#### B. proxy 侧三个修复(都已上生产)

1. **准入控制/熔断**: `MAX_QUEUE_WAITERS`(默认 **24**) —— 注意阈值故意设得很高,
   **不做常规限流**: omp 被 429 打多了会直接 "Retry budget exhausted" 停摆(proto-ui 就这么死过),
   所以正常拥塞靠排队 + 快速周转解决, 熔断只兜底。超限返回 503 + `Retry-After: 10`;
   在 async generator 里则以 SSE error 事件 + `[DONE]` 收尾(不能 return 响应对象)。
2. **并发槽位泄漏自愈**: 不能只看流式计数器 —— 非流式 `_worker` 是**直接** `self._slot.acquire()` 的。
   改成比对信号量真实余量, 并用 `_INFLIGHT` / `active` / `pending()` 三个信号确认"确实没人在用",
   持续 90s 才回收。同类问题的后端租约(`active` 卡住不降)也加了 `reap_stale_leases()`。
3. **`_CONTEXT_LIMIT` 在 FastLLM 模式下退回 32768**(它只解析 llama-server 的 `-c`) ⇒
   >16K 的请求被判超限: `FALLBACK_ENABLED=1` 时**静默转发到云端**, =0 时直接 429。
   **这就是 proto-ui 那串 429 的来源**。改为优先读 `CONTEXT_LIMIT`, 否则从
   `FASTLLM_BACKEND_COMMAND` 的 `--tokens` 解析(262144)。

另: `--default_max_tokens` 16384 → **8192**(实测 `finish_reason=length` 为 0, 无截断副作用)。
**thinking 档位不动, 保持 high** —— 实测非 high 档这个模型明显变蠢。

#### C. prefill 优化调研(subagent, 有 roofline 实测)

**「大矩阵 exact 方案」= 1Cat-vLLM v1.3.0 的 long-prefill exact-dense**
(`docs/design/sm70_{awq,fp8}_long_prefill_exact_dense.md`,
实现在 `csrc/sm70_turbomind/ops/awq_sm70_gemm.cu:2292/2327/3521` + `awq.py:40/79/110/589`):
大 M 时放弃融合量化 GEMM(NCU: occupancy 仅 12.5%、62% 周期无可发射 warp、DRAM 才 9.36%),
改成一次性把权重展开进有界共享 fp16 `K×N` workspace 再跑普通 cuBLAS。

**结论: 不值得移植** —— fastllm 早就在跑同一套结构(`fastllm-ggml-cuda.cu:783`,
`MMVQ_MAX_BATCH_SIZE=8` 以上即 dequant→`cublasGemmEx`, 配持久 scratch、TN 布局)。
vLLM 那边 SM70 能用的部分我们全有; 用不上的(`cp.async`/`m16n8k16`/machete)本来就不支持;
LOP3 反量化对 Q5_K 的 6-bit-scale + 4/1-bit 拆分结构**无法表达**。

**真正的红利在两个 fastllm 特有的浪费**:
1. **L2 write-allocate**: roofline 内核证明 Q5_K 展开的瓶颈既不是算术也不是 store 宽度 ——
   同样字节量, 普通 store **411.8 GB/s** vs streaming store **747.3 GB/s**。
   改成 streaming store 后整模型每 chunk 展开 **154.9 → 90.7 ms(1.71x, 10/10 形状逐位相同)**。
2. **`--chunked_prefill_size` 512→2048**(零代码, `GetChunkedPrefillSize()` clamp 上限 8192):
   量化投影每 token **966.5 → 654.4 µs(1.48x)**。
合计端到端 prefill 约 **+24% @8K / +19% @32K / +9.5% @128K**。

**128K 的大头不在这里**: 按每 512-token chunk 拆解, 量化投影只占 **29.5%**,
attention+GDN 占 **70.5%**(8K 时反过来, 投影占 74.9%)。这与 vLLM FP8 文档里收益从 32K 的
+22.4% 衰减到 256K 的 +7.25% 是同一条曲线。下一轮该看
`sm70_flash_v100_prefill_operator_optimization.md` / `sm70_fa2_d256_prefill_pipeline.md`;
fastllm 侧有两个**已实现但默认关闭**的开关值得先 A/B:
`FASTLLM_CUDA_PAGED_CUBLAS_BATCH_GQA=1`、`FASTLLM_CUDA_PAGED_CUBLAS_FUSED_STATE_COMMIT=1`。

**待决策**: `chunked_prefill_size` 512→2048 是收益最大的单项(+21.8% @8K)且零代码,
但会让 decode 交织粒度变粗(TPOT 抖动↑)、注意力 scratch 8→32 MB, 上线前应 A/B 验 TPOT 退化。
**2TP 展望**: 展开加速 1.71x 完全保留, chunk 摊薄的杠杆更重要(每 rank N 减半后窄形状
在 M=512 的 cuBLAS 效率更差); 但要先测 all-reduce —— M=512 时每 chunk 671 MB over PCIe
≈ 67 ms, 与展开开销同量级, 1TP 上完全不存在这一项。

**一个值得记的教训**: 第一版只做「16 字节向量化 store」实测是 **0.92x(负优化)**,
因为原内核的 store 本来就已满 64B sector 合并。是 roofline 对照实验才定位到真瓶颈是 L2 写策略。

#### D. 运维: 别再被本机网络带走

本机 Tailscale 曾在长任务中途停掉 ⇒ SSH 全断、**没 nohup 的远端进程被 SIGHUP 带走**
(一轮验证白跑)。现已固化两条路:
- `ssh dw` —— Tailscale 直连(MagicDNS), 最快;
- `ssh dw-jump` —— 经 hermes 公网跳板(`ezra@60.205.210.140`, 取自 tailnet `CurAddr`)
  ProxyJump 进 tailnet, **Tailscale 停掉也能用**。
规矩: 远端长任务一律 `nohup`; 直连失败先试跳板, 别急着判定远端挂了。

### 15.34 静默僵死 + 前缀缓存整层失效 + 分词器最后一块 (2026-08-20)

一天里挖出四个互相独立的根因，其中三个的共同特征是**静默**：不报错、不崩溃、
指标看起来还在跳，但功能已经废了。记录判据和手法，比记录结论更有用。

#### 15.34.1 后端"进程还在但一个请求都不完成" —— `forwardLocker` 漏解锁

**判据**（满足这几条基本就是这一类，不用猜）：

```
[metrics] running=1 pending=2 (total=21) | prefill 0 tok (0.0 tok/s)
          | decode 0 tok (0.0 tok/s) | done 0 req | vram=27914/32494MB
```

- `done 0 req` 是**最强判据**：自启动以来零完成，不是"慢"是"停"
- prefill 与 decode 同时 0 tok/s 但 `running` 非零
- GPU 利用率 0%，显存却照占；代理侧 `queued` 单调上涨永不回落

反例：`running=1 pending=1` 且 `decode 442 tok (29.5 tok/s)` 是正常繁忙，别误判。

**三条命令定位到具体锁**（详见 `EzraVastLLM/docs/analysis/silent-hang-diagnosis.md`）：

1. `gdb -p PID -batch -ex "thread apply all bt 8"` —— 看有没有线程停在
   `__lll_lock_wait`，以及**有没有任何线程在跑模型前向**。若其余线程全在
   `__syscall_cancel_arch` / `pthread_cond_wait` / `accept`，就是死锁不是慢。
2. glibc 的 `pthread_mutex_t` 前 5 个 int 是 `__lock/__count/__owner/__nusers/__kind`，
   而 `__lll_lock_wait` 的第一个参数（`$rdi`）就是锁地址：
   ```
   -ex "printf \"MUTEX=%p owner=%d kind=%d\n\", $rdi, *(int*)($rdi+8), *(int*)($rdi+16)"
   ```
   **`owner` 等于该线程自己的 LWP ⇒ 自死锁，实锤**，不需要再推理。
   `kind=0` ⇒ 普通非递归锁，同线程二次加锁必然永久阻塞。
3. 没有调试信息时，用"该函数里只有几个 `std::mutex::lock()` 调用点"收敛：
   `disassemble <mangled_name> | grep "call.*_ZNSt5mutex4lockEv"`，
   再用 `info symbol <返回地址>` 拿到 `函数名+偏移` 对上。

   **能死锁的一定是裸 `std::mutex` 的 `.lock()`**——`std::unique_lock::lock()`
   在已持有时会抛 `resource_deadlock_would_occur` 而不是挂死。这条能一下排除一半候选。

**根因**：

```cpp
auto &forwardLocker = model->forwardLocker;   // 裸引用, 无 RAII
forwardLocker.lock();
    ... 批前向 ...        // 内部 PagedCacheManager::Grow 显存不足时抛异常
forwardLocker.unlock();   // 异常展开跳过这一行
```

异常一抛锁永久不释放；外层 catch 打印 "process survives" 后 `while` 继续，
下一轮再 `lock()` 即自死锁，且线程是**攥着锁**死的，所有客户端线程堵在
`FetchResponseTokens` 上陪葬。

**间歇性来自状态**：只有异常恰好落在 lock/unlock 窗口内才死。当天日志里
`Grow` 抛了 6 次，前 5 次都活了下来，第 6 次才僵死。**不要因为"上次没复现"
就放过**。修法是 `std::unique_lock<std::mutex> x(m, std::defer_lock)` ——
使用点写法完全不变，但异常展开会析构释放。

同一模式在 `basellm.cpp`(x2) 和 `deepseekv4.cpp` 各有一份，一并修掉。

#### 15.34.2 为什么拖了一小时才发现 —— 健康检查被推理堵死

`maxActivateQueryNumber = min(256, --batch)`，生产 `--batch 1` ⇒ **1**，
而派发闸门对**所有路由**一视同仁。只要有一个请求在生成，`/health` `/version`
就一直排队到客户端超时 ⇒ 上游代理**永远分不清"后端在忙"和"后端已僵死"**，
死锁期间代理始终显示 `backend=READY`。

修法：这些元数据路由走独立 `lightQ` 与独立并发计数。`/admin/*` **不能**放进来
（会 suspend/resume 模型，必须串行）。

> **教训**：健康检查如果会被业务负载阻塞，它就不是健康检查。

顺带一个坑：这个后端**没有实现 `/v1/models`**（路由表只有 `/health` `/version`
`/props` `/config.json` `/generate` `/v1/chat/completions` `/admin/*`），
拿它探活只会得到 HTTP 000，与"僵死"无法区分。**探活请用 `/health`**。

#### 15.34.3 `L1trie=0` / `hits=0` —— 页字节数除错了对象

现象：`kv_pool=53920/65536 pg (82%) L1trie=0 pg (~0 tok) ... hits=0`，
偶尔跳到 `L1trie=2080` 随即归零。

**不是"缺少下放路径"**（`EvictOneColdPageLocked` 一直有调 `PageOutTrieNode`），
而是：

```cpp
pageBytes = GetBytes() / maxPages    // 错: maxPages 是逻辑预算
pageBytes = GetBytes() / dims[0]     // 对: GetBytes() 覆盖的就是 dims[0]
```

`dims[0]`（已分配物理页）`<= maxPages`（逻辑预算）恒成立：页池懒分配
（`initialPages = min(128, maxPages)`），靠 `Grow()` 追赶，而 `Grow` 一旦被
`FASTLLM_PAGED_POOL_MAX_MB` 挡住就永远追不上。日志实证
`dims0=514 -> 707 maxPages=2048`。

后果分两支，**生产命中的是更隐蔽的那支**：
- 不整除 → 判成 `manager-state` 直接拒绝下放；
- **恰好整除**（生产如此，pageBytes 含 2^7 因子）→ pageBytes 被算小约 4 倍，
  从错误 offset 抠一段存下去，**"下放成功"但内容是错的**；而上提路径
  `MaterializeTrieNode` 用的是正确的 `dims[0]`，尺寸永远对不上 →
  **上提 100% 失败**。

这精确解释了 "`L1trie` 跳一下就归零 + `hits=0` 恒成立"。同一换算错误另有 3 处：
CPU 请求换出、页号边界检查、**linear-attention 的借用显存指针**（stride 错会
直接指到别的请求的页上，属于静默数据串扰）。

**顺带修掉触发源**：`GetUnusedPageIndex` 里裸调 `Grow()` 做投机性水位扩容，
而此时 `pageIndex` 通常**已经拿到手了**；`Grow` 抛异常后穿到 MTPLoop 的 catch，
把所有在飞请求 `isAbort`。实证 `prefill 63231 tok 87.94s | decode 0 tok` ——
跑完 88 秒 prefill 一个 token 没吐就被打掉。与 15.34.1 是同一故障的两端，
两处都修才算根治。

#### 15.34.4 分词器最后一块 —— 从未读 `tokenizer.ggml.pre`

`Tokenizer::Encode` 只按 special token 切开输入，剩下的**整段**交给
`BytePairEncode`；而 GGUF 里 `tokenizer.ggml.pre = "qwen35"` 从来没被读过。
没有预分词正则，BPE 会跨词/跨数字/跨标点任意合并，产生 HF 与 llama.cpp
**绝不会产生**的 token 序列。

与 llama.cpp `llama-tokenize` 端到端对拍（两个独立 oracle：Python `regex` 跑
原始正则 + llama.cpp 出 token id）：

| 判据 | 关掉预分词 | 启用 |
|---|---|---|
| 内建 10 条用例整句 exact match | 7/10 | **10/10** |
| 45 篇语料批量对拍 | 33/45 | **45/45** |
| 累计 token 数差 | −3 | **0** |

对不上的 12 篇里 **3 篇长真实文本全中招**（agent system prompt 512 tok、
长代码 2727 tok、长日志 4244 tok）——**短文本看不出差距，长文本必错**，
这正是生产表现。

意外收获：启用后**略快**（1465530 → 1511867 tok/s），因为 BPE 优先队列合并
对段长超线性，先切块反而更省。

`\p{N}` 是**单独成块**的 —— 数字逐位切分，这点最容易漏，对拍长数字会立刻暴露。
未知/缺失的 `pre` 值保持不切分并打印 Warning，不静默改变老模型行为。

#### 15.34.5 imatrix 校准的方法论坑

- **`--parse-special` 必须加**。语料若用 chat template 标记渲染
  （`<|im_start|>` / `<tool_call>`），不加这个开关 llama.cpp 会把它们当**字面
  文本**切成 `<` `|` `im` `_start`…，"让特殊 token 进入校准分布"这个目标直接
  落空。旁证：加上后同一类语料的 token/byte 从 0.342 降到 0.299（−12.7%）。
- **ctx 不要想当然拉大**。直觉是"生产跑 262K 所以校准也该长"，但社区实测
  **512 通常优于 4096**：固定 token 预算下块越小样本越多、上下文越多样，
  统计量条件数更好。llama.cpp 的老默认就是 512。
- **必须留出集**。imatrix 在语料 A 上算，就不能拿 A 测 PPL——那是拿训练集当
  测试集。做法：同一 seed 洗牌会话文件后按索引区间切分，实测两边 512 字节
  窗口重叠 0.00%（扩充语料后为 0.11%，属跨会话样板文本，噪声量级）。
- **对照组必须自己造**。官方发布的同档 GGUF 可能用了不同 llama.cpp 版本或
  tensor-type 覆盖，拿它当基线就分不清差异来自 imatrix 还是版本。对照组要
  同源、同档、同参数，**唯一差别是有没有 `--imatrix`**。
- **Q5_K_M 在收益边界上**。社区共识是 imatrix 主要惠及 **Q5_K_M 以下**
  （Q3/Q4 区间）。目标档越高，预期收益越温和。
- 新版 `llama-imatrix` 默认存 **GGUF 格式**的 imatrix（即使文件名是 `.dat`），
  要老格式加 `--output-format dat`。

#### 15.34.6 运维陷阱清单（都是当天实际踩到的）

| 陷阱 | 现象 | 正确做法 |
|---|---|---|
| llama.cpp 编了 CUDA 时 **`-ngl 0` 也占显存** | 与生产同跑把空闲显存压到 0.27 GiB，代理的显存压力保护把后端整个卸载 | 纯 CPU 跑要 `CUDA_VISIBLE_DEVICES=`；脚本里加"生产在跑就拒绝启动"的硬守卫 |
| **下到一半的 GGUF 头部检查全绿** | `Q5_K_M.gguf` 只有 11.87 GB（应 21.27 GB），魔数/版本/张量数都能正常解析出 851 | 按**仓库真实字节数**比对，别看 GGUF 头（见 `verify_downloads.sh`） |
| 直连 `huggingface.co` 超时 | `hf download` **静默**停住：没有进程、没有报错、文件停在半截 | 下载设 `HF_ENDPOINT=https://hf-mirror.com`；**上传不能走镜像**（只读），需走本机 `127.0.0.1:10808` 代理并 `unset HF_ENDPOINT` |
| 同一 build 目录**并发 `cmake --build`** | 两个进程写同一批 `fastllm.dir/*.o`，可能产出半截目标文件 → 莫名链接错误 | 编译前 `until ! pgrep -f "cmake --build"; do sleep 20; done` |
| 代理认证：**带错 token 比不带更糟** | 带 `Bearer x` → 401；不带 → localhost 放行 | token 在 `1CatVLLM/.env` 的 `AUTH_TOKEN` |
| 前缀缓存磁盘配额 | `shutdown checkpoint failed: disk byte limit exceeded`，每次重启缓存全丢 | 活跃 root 1884 MB / 配额 2048 MB，余量仅 163 MB 装不下 262K 的 checkpoint。已提到 32 GiB（磁盘实际剩 2.0 TB）。注意 `FASTLLM_PREFIX_CACHE_DISK_DIR` 下可能有**换 key 方案遗留的孤儿 root**，它不计入活跃 root 的配额，但会长期占盘且无回收出口 |
| `tmux respawn-pane` 只杀代理 | 后端变孤儿继续占 27~31 GiB，下次启动申请不到显存，表现成"重启起不来" | 重启脚本必须显式清理孤儿 `apiserver` |


---

## 15.35 2026-08-20 IQ4_XS 上生产, 以及六个"会给出错误结论"的坑

这一轮的主线是把 imatrix 校准过的 IQ4_XS 推上生产。结论先放:

| | IQ4_XS+imatrix | Q5_K_M+imatrix | 生产原用 Q5_K_M+mtp |
|---|---|---|---|
| 留出集 PPL | 2.6989 ± 0.0471 | 2.6877 ± 0.0470 | 2.7799 ± 0.0503 |
| 权重体积 | 15.96 GiB | 18.6 GiB | 20.81 GiB |
| 逐字抄写 | 7/7 | — | 7/7 |
| 工具名保真 | 3/3 | — | — |
| MTP 接受率 | 92.19% / 85.94% | — | — |

但**真正的结论要靠配对检验才读得出来**, 见下面第 1 条。

### 1. 比两个"独立置信区间"会严重低估判别力 —— 要做配对

`llama-perplexity` 只报边际区间 `PPL = 2.6989 +/- 0.0471`。拿两组的区间去比是错的:
两组评的是**同一份文本的同一批分块**, "这一块是密集代码还是自然语言"对两个模型
是共同的, 这部分方差在配对后会被完全抵消。

按独立区间读: 差值 0.0112 远小于任一区间宽度 -> "完全分不出来"。
按配对读:

```
IQ4_XS vs Q5_K_M   ΔPPL = +0.417% ± 0.187%   t = 2.22   60 块中 39 块更差
```

不确定度从 ±1.74% 压到 ±0.187%, 约 **9 倍**。结论也随之改变: 从"分不出来"变成
"**差异真实、方向一致, 但只有 0.42%**" —— 后者同时排除了"其实掉了 3% 只是被噪声
盖住"这种可能, 这才是选型需要的信息。

实现: llama.cpp 打印的 `[n]v` 是**前 n 块的累计** PPL, 可以逐块还原
`nll_n = n·ln(PPL_n) − (n−1)·ln(PPL_{n−1})`, 再对差值做配对检验。
工具 `EzraVastLLM/scripts/ppl_paired.py`, 零额外算力, 数据本来就在日志里。

### 2. 基线选错: graft 过 MTP 的 GGUF 不能用 llama.cpp 当基线

`-plus-mtp.gguf` 声明 `block_count=65`(比原模型多一层), llama.cpp 会照着建 65 层,
**把 MTP 层当成一个普通 transformer 层执行** —— 只有 fastllm 知道要特殊处理它。

它的指纹很好认: 对该组的**配对标准误是对正常组的 17 倍**(3.22% vs 0.187%),
符号检验 30/60 正好是掷硬币。如果两个模型只是量化档不同, 逐块差应该又小又稳;
方差爆炸说明它在行为上是**另一个模型**。

要做"我们的配方 vs 官方配方"的对比, 必须用未 graft 的 851 张量版本。

### 3. 探针恒为通过: llama-cli 默认回显 prompt

原来的逐字抄写探针是 `llama-cli -p "<指令>\n<期望串>" | grep -F "<期望串>"`。
`--display-prompt` **默认为 true**, prompt 会被原样打印, 而它本身就含有期望串 ——
于是**无论模型抄没抄对都判通过**。这比没有验证更糟: 它会主动给出"没问题"的假结论。

两条修法, 都要:
- `--no-display-prompt`;
- **不信任这个开关**, 再加自检 —— 若只出现在 prompt 里的指令词仍出现在输出中,
  判"探针无效"而不是"通过"。宁可报无效, 绝不报假通过。

同类问题: `--jinja` 默认开启会走思考模板, `-n 64` 可能全花在 `<think>` 里,
抄写内容还没输出就被截断 -> 假**阴**性。

### 4. 闸门必须能区分"模型没过"和"闸门自己坏了"

第一版准入闸门跑出了 ">> 有未通过项, IQ4_XS **不得**上生产"。实际原因是它自己
从 `.env` 取 token 的那行 shell 转义被 heredoc 嚼坏, 全部请求 401。
差一点用一个鉴权 bug 挡掉一个合格的模型。

现在的口径: 前置/套件失败 `exit 2`(不回滚, 因为**我们没测出来**),
模型层面失败 `exit 1`(回滚)。同一套逻辑还要求先拿**已知没问题的生产模型**
跑一遍套件, 验证套件本身并拿到基线, 再去测候选。

### 5. 三个"看起来正常"的运维陷阱

**a) `/health` 返回 200 不等于就绪。** proxy 一起来就 200, 而权重要从机械盘读
8 分钟。第一版闸门因此在 20 秒时判"就绪"。更隐蔽的是: fastllm 读完权重后先放在
主机内存(RSS 已 18 GB), 要到**第一次真实请求**才上卡(GPU 从 724 MiB 跳到 22 GB)。
可靠的就绪判据只有一个: 真发一次补全并拿到 200。

**b) `tmux respawn-pane -k` 不带命令时会重跑 pane 的原命令。**
`respawn-pane -k` 之后再 `send-keys "start_prod.sh <profile>"`, 那行文本只会被打进
已经在运行的进程的 stdin。表现是"切换脚本跑完了但模型没换"。
正确写法是把命令显式传给 respawn-pane: `tmux respawn-pane -k -t <pane> "<cmd>"`。
这个坑还会**伪装成正常**: 若 pane 原命令恰好就是要恢复的那份, 恢复逻辑看起来完全没问题。

**c) 固定的 `tail -f <某个日志>` 会在换模型后永远停在旧文件上。**
每个 profile 写自己的 `backend-PROD-cyber-{q5,iq4xs}.log`。看板卡在旧日志上时,
现象是"服务像是卡住了", 实际只是在看一份不再写入的文件 —— 我据此差点判断
IQ4_XS 加载了 20 GB(那是 Q5 日志里的行)。
修法: 从 `/proc/<pid>/fd/1` 反查当前后端真正在写哪个文件并跟随, 见
`EzraVastLLM/scripts/follow_backend_log.sh`。

### 6. 隐式环境导致鉴权被静默关掉

`start_prod.sh` 从不 source `.env`, `AUTH_TOKEN` 一直是靠"启动它的那个 shell 恰好
source 过"传进来的。换一种拉起方式(respawn-pane 显式命令 / nohup / cron)这个环境
就没了, proxy 打一行 `auth token: EMPTY - INSECURE` 然后**照常服务, 只是不再需要令牌**。
从功能上完全看不出来。已改为脚本自己读 `.env`, 并显式打印是否载入。

### 7. 上游 Q5_K_M 里那 296 个 Q8_0 张量: 不是保留精度, 是压根没量化

同名张量在该作者的 `Q4_K_M / Q5_K_M / Q6_K / Q8_0` 四个文件里 **sha256 逐字节相同**
(抽样 67 个, 0 例外)。一个 Q6_K 文件里 `attn_q` 是 Q8_0, 只可能是没被重新量化。
上游只量化了 `ffn_*` + `attn_output` + `token_embd` + `output`, 注意力和 SSM 的权重
矩阵一个没动。

由此产生两个必须校准的认知:
- 生产原先在跑的"Q5_K_M"实测 **6.33 bpw**, 不是名义上的 5.33。切到 IQ4_XS(4.78 bpw)
  的落差比"Q5→IQ4"听起来大 1.55 bpw。
- "照抄它的高精度张量"这个想法前提不成立。真按加权误差排序, 值得升档的只有
  `ssm_alpha`/`ssm_beta`(加权误差最高, 各只要 6 MiB); 而占照抄成本 70% 的
  `attn_gate`/`attn_q`/`attn_qkv` 恰恰是加权误差最低的几个。
  全抄要 +3.17 GB, 会直接吃掉给 KV 页池腾出的显存。

配套发现: `llama-quantize --dry-run` 可以在 1 分钟内给任何配方定价, 不必赌着跑 30 分钟。

### 8. 已结案(2026-08-20 复测): "IQ4_XS 的 decode 反而更慢" 不成立

原记录是: 约 5.9 tok/s vs Q5_K_M 的 6.9~12.1, 并据此推断"SM70 MMQ 快路径在
decode 上没被调用, 应该去优化 `mmvq` 的 IQ4_XS 分支"。**两句都错。**
受控复测后: IQ4_XS 比 Q5_K_M **快 1.20x**, 而 `mmvq` 已经接近带宽上限, 没有可优化空间。

**原对比同时混了三个变量**(这才是教训本身):

1. **上下文长度**。IQ4_XS 那几个慢数字都是 46K~74K token 的请求;
   拿来对比的 Q5 12.1 tok/s 是百级上下文。decode 在 262K 档位上是注意力主导, 不是权重主导。
2. **profile 不只换了权重**。`q38-PROD-cyber-iq4xs-*.env` 相对 q5 档还多了
   `FASTLLM_CUDA_PAGED_CUBLAS_BATCH_GQA=1` 与 `FASTLLM_PAGED_POOL_MAX_MB` 6600->10600。
3. **二进制不同**。量 5.9 那一版的 `apiserver` 里, `FastllmCudaTrySm70Iq4XsMmq`
   每次调用都真的去查 `cudaGetDeviceProperties`(36.5us)。反汇编可验:
   `objdump -d build-rw/apiserver.bak-preroutecensus` 里该函数含 2 次
   `cudaGetDeviceProperties_v2`, 而修复后的 `build-rw/apiserver` 含 **0** 次(改调缓存的
   `s70_device_caps`)。这一项只在 IQ4_XS 档位出现(调用点就写着
   `if (ggufType == GGML_TYPE_IQ4_XS)`), 天然伪装成"换了 IQ4_XS 才变慢"。

#### 证据 1: 8 组完全匹配的端到端样本 —— IQ4_XS 快 1.20x

两份后端日志里各跑过同一套验收探针, prefill/decode 的 token 数逐条相同:

| prefill/decode tok | Q5_K_M | IQ4_XS | 比 |
|---|---|---|---|
| 59 / 21 | 38.6 | 47.1 | 1.22 |
| 52 / 14 | 42.9 | 53.5 | 1.25 |
| 54 / 16 | 41.1 | 49.6 | 1.21 |
| 72 / 34 | 43.8 | 53.6 | 1.22 |
| 98 / 60 | 38.0 | 44.0 | 1.16 |
| 55 / 17 | 44.9 | 52.8 | 1.18 |
| 55 / 17 | 44.4 | 52.8 | 1.19 |
| 33 / 20 | 44.7 | 53.7 | 1.20 |

**均值 1.201x, 中位数 1.204x, 8/8 同向**。而且方向是向下偏的 —— 日志里那八条
Q5 都是 `running=0 pending=0` 跑的, IQ4_XS 那八条是 `running=1`。
工具: `EzraVastLLM/scripts/sm70-mmvq-bench/match_logs.py`(零算力, 数据本来就在日志里)。

#### 证据 2: 算子级 microbench(空闲 GPU) —— IQ4_XS 的 mmvq 每一组都更快

把 `fastllm-ggml-cuda.cu` 的 mmvq 字节不改地抽出来(行 28..1086 -> `mmvq_body.inc`),
在 12 个生产形状 x n=1/2/3 上对拍(V100 空闲, 8 轮取最小):

```
单层合计(n=2):  IQ4_XS 0.4098 ms   Q5_K 0.6260 ms   -> 1.53x
x64 层:          IQ4_XS 26.2 ms/tok  Q5_K 40.1 ms/tok
30 组形状 x n:   IQ4_XS/Q5_K 耗时比 0.63~0.92, **没有一组反向**
峰值带宽:        735 GB/s = V100 900 GB/s 的 82%
```

带宽比(136/176 = 0.773)反而比实测耗时比(~0.73)差, 因为 IQ4_XS 的 vec_dot 每个权重值
只花 0.25 次 dp4a, 而 Q5_K 要 0.5 次(其中一半是算 min 的 `0x01010101` 点积)。
即 IQ4_XS 在带宽和 ALU 两头都赢。

#### 证据 3: `mmvq` 对 IQ4_XS 本来就是量化点积, 不是"反量化再乘"

当时担心的是"每 token 多写读一遍 fp16 权重"。源码上不成立:

- `fastllm-ggml-cuda.cu:800` `get_has_vec_dot_q_cuda(GGML_TYPE_IQ4_XS) -> true`,
  因此 n<=8 时分派器的前两个 `dequantFp32/dequantFp16` 分支都进不去;
- `fastllm-ggml-cuda.cu:815` `get_vec_dot_q_cuda(IQ4_XS) -> vec_dot_iq4_xs_q8_1`(行 478),
  它直接读 `bq4->qs` 的 nibble, 经 `get_int_from_table_16`(`__byte_perm` 查表)展开后
  做 `dp4a`。**全程不写回任何反量化中间缓冲**。
- 对照组 `vec_dot_q5_K_q8_1`(行 649)同样是量化点积。两边性质相同, 所以能对比。

#### 证据 4: MMQ 的 `n in [8,64]` 门槛是对的 —— 把下界降到 1 会更慢

把 `Sm70Iq4XsMmqShapeRejectReason` 的 minN 变成环境变量, 强行让 n=1..8 进 MMQ,
和 mmvq 同形状对拍(V100 空闲, ms):

```
             n =      1       2       3       4       5       6       7       8
attn_qkv  mmvq   0.0421  0.0476  0.0616  0.0730  0.0873  0.1011  0.1143  0.1281
          MMQ    0.1121  0.1116  0.1040  0.1039  0.1038  0.1039  0.1031  0.0997
ffn_down  mmvq   0.0703  0.0723  0.1007  0.1281  0.1630  0.1891  0.2195  0.2563
          MMQ    0.2996  0.2997  0.2999  0.3004  0.3000  0.3002  0.3007  0.2904
```

MMQ 的耗时对 n **几乎不变**(小 n 会被 pad 到 tile 宽度 mmq_x=8), 而 mmvq 随 n 线性涨。
交叉点落在 **n≈6~8**: n=1..3 上 MMQ 比 mmvq 慢 **1.6~4.3倍**。
所以 `kSm70Iq4XsMmqMinN = 8` 不是拍脑袋的经验值, 实测支持它; 若要挑剔, 只能说它
**略偏低** —— attn_gate/attn_output/ffn_down 在 n=8 上 MMQ 还慢 13~34%。

#### 复现

```
EzraVastLLM/scripts/sm70-mmvq-bench/
  mmvq_bench.cu      # IQ4_XS vs Q5_K 的 mmvq, 生产形状 x n=1/2/3
  mmq_vs_mmvq.cu     # SM70 MMQ vs mmvq, n=1..8 找交叉点
  match_logs.py      # 从两份后端日志里拉完全匹配的端到端样本对
  decode_ab.py       # 受控端到端探针(固定 prompt/ctx/max_tokens, 只算稳态 tok/s,
                     #   并快照 /props 路由差分 —— 没有路由差分的 tok/s 不算证据)
```
编译用 build 里同一个 nvcc(`/home/ezra/.conda/envs/tsenv/bin/nvcc`, CUDA 12.9, sm_70)。

#### 结论

`gguf.sm70_iq4xs_mmq` 在 decode 上恒为 0 是**设计如此且正确**, 不是"没走到快路径"。
decode 应该走 `gguf.mmvq`, 而它已经跑到 V100 带宽的 82%。IQ4_XS 不需要新算子。

(顺带: MTP 是正常工作的, 接受率 92.19%/85.94%。日志里那条
`[Qwen3.5 MTP] not enabled` 是 `mtpSkipLogPrinted.exchange(true)` 的**一次性**打印,
只说明有过一个被拒的请求, 不能据此判断 MTP 没启用 —— 我一开始就误读了这条。)

---

## 15.36 被文档判过死刑但没记测试形状的开关 (2026-08-20)

今天第三次撞上同一个模式了(前两次是 SM70 开关归因、llama-cli 假阳性)。这次的样本
是 `FASTLLM_CUDA_PAGED_CUBLAS_BATCH_GQA`, 值得把**判据**单独写下来, 因为它可以直接
复用到剩下那一批默认关闭的 gate 上。

### 现象: 三处文档, 三种口径, 而 A/B 从来没做过

同一个开关在本仓的三处记录:

| 出处 | 说法 |
|---|---|
| `fastllm/docs/qwen35_v100_local_stack.md:204` | 「**不启用负结果。** … GQA batched cuBLAS、async gather、fused preprocess 和 persistent scratch 也**没有稳定收益**」, 并在 215 行列进「当前实验 gate 均默认关闭」 |
| `v100-perfs/docs/EXPERIENCE.md:1426` | 「fastllm 侧有两个**已实现但默认关闭**的开关**值得先 A/B**」 |
| `v100-perfs/docs/analysis/prefill-exact-port-investigation.md:344` | 同上, 逐字重复 |

一处判了死刑, 两处说值得试。**而那个 A/B 到今天为止一次都没跑过。**

代码侧同样没有线索: 它随 commit `8b6b1ec4`(2026-08-02, "feat: add V100 cache tiers
and service controls")进来, 那是个大 squash, message 只写了
"Integrate SM70 paged/GDN/IQ4_XS experiments with **safe gates**"。
`FastllmPagedCublasBatchGqaEnabled()` 函数体旁边**一行注释都没有**。

### 关键缺陷: "没有稳定收益"这句话没有记录测试形状

这才是问题所在, 不是"结论错了", 而是**结论不可证伪**。

判死刑那一节的标题是「算子优化优先级」, 同一条里并列的是 SM70 fused H/O、SM70 WMMA
chunk GDN prefill —— 全是 **cold prefill** 算子; 整节的收尾是「不代表 cold prefill
算子本身变快」。所以合理推测当时量的是 prefill 形状, 但**文档没写**, 我也不能替它断言。

而这个开关的作用域跨了两个完全不同的 regime:

```
prefill (qoLen 数百~2048):  qk 是真 GEMM, A 矩阵在单次 GEMM 的 tile 内就被复用
                             -> 批不批几乎没区别, "没有稳定收益"完全可信
decode / MTP 校验 (qoLen 1~3): qk 退化成 GEMV, N=1..3
                             -> 逐 head 循环会把 4 MB 的 kChunk/vChunk 各流 6 遍
                             -> 这才是全部成本
```

一句不带形状的"没有稳定收益", 把**两个 regime 一起判了**。

### 这次补上的是"为什么关得没道理", 不是"没有注释"

开之前只确认了"没有注释说明为什么关"是不够的 —— 那只说明信息缺失, 不说明可以开。
真正的安全依据是**等价性证据**: 两条路必须算的是同一个东西, 否则性能对比毫无意义,
而且会静默改变输出。

新增 `fastllm/test/ops/turboPagedAttentionTest.cpp` 做了这件事(fp64 参考: 直接从打包
字节按格式定义在 double 下反量化, 再在 double 下做注意力):

- 覆盖 qoLen = 1/2/3/4 × 页表 `{2}`(尾 40) / `{2,0,3}`(乱序, 尾 17) / `{1,4}`(整页), 共 11 组
- **逐位一致 = 否**(全部 11 组)
- 两条路径互差 `7.6e-6 ~ 4.9e-4`, 输出幅度 `0.13 ~ 0.36`。fp16 在 0.35 附近 ULP =
  2.44e-4、0.13 附近 = 6.1e-5 —— **互差就是 1~2 个 fp16 ULP**, 纯输出舍入
- 对 fp64 参考: 默认 RMS `1.6e-5~5.8e-5`, BATCH_GQA RMS `1.9e-5~5.2e-5`,
  **11 组里谁更准各占一半, 无系统性差异**

结论: 两条路算的是同一个东西, 差异只来自 cuBLAS 内核选择带来的累加次序。
另外核对了索引映射(`qk` 布局 (g,t,j) → `o = g*qoLen + t`;
`SoftmaxWithCausalMaskBatchGqa` 的 `token = o % qoLen` 与非批版同式; state 数组、
`AttnBlockUpdateFloat` 的 grid 与 `outFloatScratch` 行布局一致), 且**页表与
lastPageLen 根本不经过这个分支** —— gather 在 `if (batchGqa)` 之前就完成了, 两条路
共用同一次 `FastllmCudaPagedCacheGatherHeadRangeToHalf`。

基于这份证据, 2026-08-20 把 `FASTLLM_CUDA_PAGED_CUBLAS_BATCH_GQA=1` 加进
`q38-PROD-cyber-iq4xs-imatrix-mtp2-sm70.env` 并上生产。

### 性能 A/B 结果: 收益完全由 qoLen 决定, 而生产形状上等于没开

2026-08-20 两次独占窗口实测(batch=1, headDim=256, GQA6, K=q8_0/V=turbo3)。
`默认` = 逐 query head 循环调 group 次 cublasHgemm; `BATCH_GQA` = 一次 StridedBatched。

| kvLen | qoLen | 默认 | BATCH_GQA=1 | 加速 | 收益 |
|---|---|---|---|---|---|
| 8192 | 1 | 1.55679 ms | 0.905318 ms | **1.720x** | 72.0% |
| 8192 | 3 | 1.58208 ms | 1.53293 ms | 1.032x | 3.2% |
| 32768 | 1 | 5.92276 ms | 3.38555 ms | **1.749x** | 74.9% |
| 32768 | 3 | 5.96065 ms | 5.83849 ms | 1.021x | 2.1% |
| 131072 | 1 | 23.2398 ms | 13.2488 ms | **1.754x** | 75.4% |
| 131072 | 3 | 23.4164 ms | 23.0466 ms | 1.016x | 1.6% |

**生产稳态是 qoLen=3** —— 路由普查实测 `attn.native_fallback` 的
tokens/calls = 53648/18352 = **2.92**(MTP `drafts_per_step=2` 让 target 前向 seqLen 恒为 2~3)。
所以 **BATCH_GQA 在生产的主导形状上基本等于没开**。

这直接否掉了一条已经写下的归因: 当时观察到生产 decode 从 5.9 涨到 9.0 tok/s(+53%),
曾归因给刚打开的 BATCH_GQA。**一个在主导形状上只值 1.6~3.2% 的开关解释不了 53% 的吞吐提升**,
该归因已收回。(真实原因另查, 大概率在 IQ4_XS 权重那条 GEMV 路上。)

### 一个自我订正: 别用"收益之比"这种数, 当分母在噪声里的时候

这个开关在两个 regime 上差多少倍? 先后出现过三个数字, 而**三个都不该说**:

| 来源 | 说法 |
|---|---|
| 目测 | "差 25 倍" |
| 第一次窗口(仅 8K) | 68.6% / 6.6% = **10.4 倍** |
| 第二次窗口 | 8K 22.4 倍 / 32K 35.8 倍 / 128K **47.0 倍** |

我当时用第一次窗口的数去订正那个目测的 25 倍, 说"是 10.4 倍不是 25 倍"。
**这个订正本身也是错的** —— 不是因为算错, 而是因为**这个比值根本不稳定**:

- 分子(qoLen=1 的收益)很稳: 68.6% → 72.0% / 74.9% / 75.4%, 两次窗口、三档上下文都一致;
- 分母(qoLen=3 的收益)在噪声里: **同一个 8K 形状, 一次量到 6.6%, 一次量到 3.2%**,
  两次差 2 倍; 整体范围 1.6%~6.6%。

分母只有几个百分点、且自身重复性只有 2 倍时, 比值必然在 10~47 之间乱跳,
**报任何一个都是在报噪声**。

正确的报法是**分别报, 各带范围**:
```
qoLen=1: 稳定 1.72~1.75x   (两次窗口 x 三档上下文, 一致)
qoLen=3: 1.6%~3.2%(窗口2) / 6.6%(窗口1), 与运行间波动同量级 -> 视作"无收益"
```

教训: **当一个结论要用"A 比 B 大多少倍"表达时, 先看 B 有没有离开噪声底。**
没离开就别做除法 —— 除法会把噪声放大成一个看起来很精确、很有冲击力的数字,
而这正是本节开头那句"没有稳定收益"当年可能踩过的坑的**镜像**:
一边是没记形状, 一边是拿噪声做分母, 两种都会生产出不可复现的"结论"。

### 双向教训

- 文档那句不带形状的**"没有稳定收益"**, 在 qoLen=1 上是**错的**(实测稳定 1.72~1.75x);
- 而如果只测 qoLen=1 就宣布"BATCH_GQA 值 1.75x 应当上生产", 在生产形状上又是**错的**(1.6~3.2%)。

**同一个开关, 两句话都对一半, 而两句话都没写形状。** 这就是为什么形状必须进结论。


## 15.37 一次被"粘性 CUDA 错误"带偏的定位, 以及"测了很多次≠测过" (2026-08-20)

融合分页注意力 kernel 的 bench 在 `kvLen=32768` 上崩了, 日志长这样:

```
kvLen=8192 qoLen=3   <- 这一组跑完并打印了结果
[TurboKV] Copy sync error: 700 (an illegal memory access was encountered)
Error: CUDA error when copy from memory to GPU!
  CUDA error = 700, cudaErrorIllegalAddress at src/devices/cuda/fastllm-cuda.cu:5452
[TurboKV] Copy meta H2D error: 700   (之后每一次 CUDA 调用都报同一个错)
...段错误
```

第一反应是"新 kernel 在大 kvLen 上越界了", 怀疑方向列了一串: grid.z 溢出、
页表固定缓冲、workspace 二分与 kernel 假设不一致、`numPages*pageLen*headDim` 溢出 int。
**全都不是。**

### 根因: 测试夹具直接写物理页号, 而页池是懒分配的

`src/fastllm.cpp` 的 `AllocatePagedCacheManager`:

```cpp
const int initialPages = preallocateMax ? maxPages
                                        : std::max(1, std::min(128, maxPages));
Resize({initialPages, pageLen, numHeads, headDim});
Allocate();
```

**默认只物理分配 `min(128, maxPages)` 页**, 其余靠运行期 `GetUnusedPageIndex()` → `Grow()`
按需追加。生产路径永远从取页接口拿页号, 所以页号必然 < `dims[0]`, 一辈子碰不到这条。

而 bench 夹具为了构造确定的页表, **直接往物理页号 0..numPages-1 上写**:

| kvLen | 需要页数 | 实际分配 | 结果 |
|---|---|---|---|
| 8192 | 64 | 64 | 正常 |
| 32768 | 256 | **128** | 页号 128..255 越界写 -> 700 |

阈值恰好是 **128 页 = 16384 token**, 正好卡在"8192 过、32768 崩"之间。
修复: 夹具里 `ScopedEnvVar("FASTLLM_PAGED_CACHE_PREALLOCATE_MAX", "1")`, 并加硬断言
`dims[0] >= numPages` —— 宁可当场抛异常, 也不要再把"池子比页号小"变成显存踩踏。
修完 `kvLen=8192/32768` 全绿 (16/16), 融合 kernel 本身一行没改。

### 教训一: 粘性错误让"崩溃点"和"故障点"必然不在一起

`cudaErrorIllegalAddress` 会**毒化整个 context**: 出错之后每一次 CUDA 调用都返回同一个错,
直到重建 context。所以日志里报出来的位置, 是**故障之后第一次同步的地方**, 不是故障发生的地方。
这次报在一次 H2D 拷贝(`fastllm-cuda.cu:5452`)上, 真正越界的是更早的一个量化 kernel。

可用的判据:
- **看第一条错误, 别看最响的那条。** 上面日志里 `Copy sync error` 在前, 后面几十行
  `Copy meta H2D error` 全是它的回声。回声多不代表那里是现场。
- **阈值形状比错误信息有用。** "8192 过 / 32768 崩"这个二分, 十分钟就能把范围压到
  16384 附近; 而 16384 = 128 × 128 一眼就指向一个**固定常量**, 不是算术溢出
  (溢出的阈值通常在 2^31 附近, 不会这么早)。**遇到干净的 2 的幂阈值, 先找 clamp 常量。**
- **异步执行意味着"上一组跑完并打印了结果"不等于"上一组没出错"。** 计时用的
  `cudaEventSynchronize` 返回值如果没检查, 一个已经中毒的 context 照样会给出
  看起来合理的耗时数字。

### 教训二: "92/92 通过 + sanitizer 0 errors" 也可以是**零覆盖**

这个 kernel 当时的证据是: 11 组形状的 fp64 对拍全绿(92 条断言)、compute-sanitizer
memcheck 0 errors。听起来很扎实。但那些形状的 `kvLen <= 273` ——
**kernel 从来没在超过 273 个 token 上跑过**, 而生产是 262144。

更糟的是这个覆盖缺口**恰好被夹具的 bug 保护着**: 想测大形状就会崩在夹具上,
于是"测不了"很容易被当成"待办", 而不是"当前结论的适用范围只到 273"。

写覆盖率的时候, **断言条数是没有意义的数字**, 要写的是**被遍历的参数区间**:

```
覆盖: kvLen ∈ {40, 256, 273}      <- 真实情况, 一眼看出问题
覆盖: 11 组形状 / 92 条断言        <- 听起来很多, 什么也没说
```

补上大形状档之后(`--large`, kvLen ∈ {8192, 32768, 131072}, 逐 kv head 流式 fp64 参考,
host 峰值内存压在 ~134 MB @32K), 全部通过, 且 compute-sanitizer memcheck 在这三档上
`ERROR SUMMARY: 0 errors`。顺带看到一个此前看不见的现象:

| kvLen | 现路径 RMS | 融合 kernel RMS | 融合更准 |
|---|---|---|---|
| 273 | 2.1e-05 | 7.6e-06 | 2.8x |
| 8192 | 1.2e-04 | 6.7e-06 | 18x |
| 32768 | 1.1e-04 | 6.6e-06 | 17x |
| 131072 | 1.0e-04 | 5.8e-06 | 17x |

**精度差距随上下文变大而拉开**(2.8x → 17x)。原因是现路径的 score 缓冲 `qk` 是 fp16、
反量化结果也落 fp16, 累加的 key 越多, 这两处舍入积累得越厉害; 融合 kernel 全程 fp32。
只测 kvLen<=273 的话, 这个结论根本看不到 —— 又一个"小形状测不出来"的例子。

---

## 15.38 我们一直在长上下文上**静默**损失精度 (2026-08-20)

这一条是做融合注意力 kernel 时顺带发现的, 但它的重要性**高于**那个 kernel 的加速比,
所以单独成节。

### 结论

生产在跑的 `gather + chunked cuBLAS` 分页注意力路径, 在长上下文上带着约 **0.5~0.7% 的
最大相对误差**, 而**这个误差不是量化本身造成的** —— 同一份 turbo3/q8_0 KV 缓存,
换一条实现就能把它降到 3e-4。也就是说这部分精度是**实现白丢的**, 不是格式的代价。

| kvLen | 现路径 RMS | 现路径最大相对误差 | 融合 kernel RMS | 白丢了多少 |
|---|---|---|---|---|
| 273 | 2.1e-05 | 6.0e-04 | 7.6e-06 | 2.8x |
| 8192 | 1.2e-04 | **6.7e-03** | 6.7e-06 | **18x** |
| 32768 | 1.1e-04 | 5.9e-03 | 6.6e-06 | **17x** |
| 131072 | 1.0e-04 | **6.6e-03** | 5.8e-06 | **17x** |

参考值是 fp64: **直接从打包字节按格式定义在 double 下反量化**(q8_0 是 `scale*int8`,
turbo3 是 `InverseWht128(centroid[idx]*correctedNorm)`), 再在 double 下做注意力。
所以"给定这份量化 KV, 正确答案是什么"是唯一确定的, 上表量的是**两条实现各自离它多远**,
与量化误差无关。

### 机理: fp16 中间量的误差随累加长度增长

现路径有两处 fp16 中间量, 而且都在**长度随 kvLen 增长**的累加链上:

1. 反量化结果 `__float2half_rn` 落成 fp16 连续缓冲, 再喂 cuBLAS;
2. **score 缓冲 `qk` 本身就是 `half`** —— `cublasHgemm` 算完 q·K 直接写 fp16,
   softmax 再从 fp16 读回来做 `expf`。

融合 kernel 全程 fp32(它压根不物化中间缓冲, 所以也就没有落 fp16 的机会)。

误差从 kvLen=273 的 2.8x 涨到 8192 的 18x, 之后**basically 持平**(17~18x)。
持平的推测原因是 softmax 会把概率质量集中到少数 key 上, 有效参与累加的项数会饱和 ——
这条是推测, 没有单独验证, 但"273 -> 8192 涨 6 倍、8192 以后不动"这个形状是实测的。

### 为什么一直没人发现

三个原因叠在一起, 每一个单独看都很合理:

1. **它不报错。** 不崩、不 NaN、不出乱码, 只是输出"稍微差一点"。
2. **长上下文本来就被认为会变差。** 模型在 128K 上不如在 8K 上聪明, 这是所有人都接受的
   常识, 于是实现层面白丢的 0.7% 被归进"长上下文就这样"这个筐里, 没人去分。
3. **所有对拍都在小形状上做。** 这是最关键的一条 —— 见 15.37: 该 kernel 原先的
   11 组对拍 `kvLen <= 273`, 而在 273 上现路径只差 2.8 倍, **看起来完全可以接受**。
   把 kvLen 拉到 8192 才第一次看见 18 倍。

### 判据

**凡是含 fp16(或更低精度)中间量、且累加链长度由运行期参数决定的路径, 误差量级必须
在生产规模上量, 不能在单测规模上量。** 具体做法:

- 参考值要**与实现无关**(fp64 从原始字节重算), 不能拿另一条实现当参考 ——
  两条实现共享同一个 fp16 中间量时, 互相比是全绿的;
- 误差要按 **kvLen 扫一遍**看趋势, 单点数值没有信息量;
- 报"最大相对误差"而不只是 RMS —— RMS 1e-4 听起来很小, 但对应的最大相对误差是 6.7e-3,
  差 60 倍, 而影响输出 token 的是尾部不是均值。

### 附带的一个反直觉点

融合 kernel **既更快又更准**, 这在数值计算里不常见 —— 通常是拿精度换速度。
这里能同时拿到, 是因为两者的来源是同一个: **不物化中间缓冲**。
不物化 => 不用写/读 1024 B/行 => 更快; 不物化 => 没有落 fp16 的地方 => 更准。
所以看到"又快又准"时不必怀疑, 但要问清楚**是不是同一个原因导致的** —— 如果不是,
大概率有一边测错了。

---

## 15.39 融合分页注意力: 实测收益, 以及**还没吃到的两块余量** (2026-08-20)

把 turbo3/q8_0 的反量化融进注意力 kernel(不再物化 fp16 中间缓冲)之后的实测。
形状是生产形状: batch=1, 24 Q head / 4 KV head(GQA6), headDim=256, pageLen=128,
K=q8_0, V=turbo3。

### 实测

| kvLen | qoLen | 现路径(默认) | 融合 turbo-XQA | 加速 |
|---|---|---|---|---|
| 8192 | 1 | 1.55679 ms | 0.148992 ms | **10.45x** |
| 8192 | **3** | 1.58208 ms | 0.299366 ms | **5.28x** |
| 32768 | 1 | 5.92276 ms | 0.441395 ms | **13.42x** |
| 32768 | **3** | 5.96065 ms | 1.12968 ms | **5.28x** |
| 131072 | 1 | 23.2398 ms | 1.44497 ms | **16.08x** |
| 131072 | **3** | 23.4164 ms | 4.45389 ms | **5.26x** |

(qoLen=3 是生产稳态形状。)

### 为什么 qoLen=1 随上下文越来越快, 而 qoLen=3 纹丝不动

第一眼会以为"qoLen=3 卡在带宽墙上了"。**不是。** 把时间折算成每
(token, kvHead) 行的等效访存量(@750 GB/s), 再和理论下限对照:

| kvLen | 现路径 | 融合 qo=1 | 融合 qo=3 | qo3/qo1 |
|---|---|---|---|---|
| 8192 | 35632 B/行 | 3410 | 6852 | 2.01 |
| 32768 | 33890 B/行 | 2526 | 6464 | 2.56 |
| 131072 | 33245 B/行 | 2067 | 6371 | **3.08** |

理论下限是 **372 B/行**(q8_0 的 K 272 B + turbo3 的 V 100 B, 各读一次)。

`qo3/qo1` 在 128K 上收敛到 **3.08 ≈ 3**, 这个 3 不是巧合, 是**设计里明写的**:
qoLen>1 时为了把寄存器里活跃的 `(m, l, acc[8])` 组数压在 6 以内, 把 group 拆进了 grid ——
qoLen=3 用 `GROUP_CHUNK=2`, 即 6 个 query head 分成 3 个 chunk, **K/V 每行被读 3 遍**。
小 kvLen 上比值不到 3, 是因为固定开销(Q 载入、combine、kernel 启动)稀释了。

所以 qoLen=3 的 5.26x 之所以三个数量级纹丝不动, 是因为它的成本结构里
**"设计上的 3 倍重读"是常数项**, 而 qoLen=1 的现路径成本里"逐 head 重读 6 遍"
会随上下文越来越占主导 —— 一个分母在涨, 一个分子分母同涨。

### 余量一: qoLen>1 的 3 倍重读(约 3x)

上面那个 3 是可以去掉的: 让一个 block 承担全部 6 个 head x 3 个 token = 18 组状态,
K/V 就只读一遍。障碍是寄存器 —— 18 组 x 10 个 float = 180 个/lane, 放不下。
可行方向是**把 K/V 行先 stage 进共享内存**, 让"组数"和"全局 load 次数"解耦,
代价是多一轮 shared 读写和 `__syncthreads()`。**未实现, 未验证。**

### 余量二: 达成带宽只有峰值的 15%

只统计**必须读的** 372 B/行, 折算实际达成带宽:

| kvLen | qo=1 | qo=3 |
|---|---|---|
| 8192 | 81.8 GB/s | 122.2 GB/s |
| 32768 | 110.5 GB/s | 129.5 GB/s |
| 131072 | **135.0 GB/s** | 131.4 GB/s |

V100 峰值 898 GB/s, 即 **达成率约 15%**。注意 qo=1 和 qo=3 收敛到同一个
~130 GB/s —— 两者效率相同, qo=3 只是搬了 3 倍的字节, 这**反过来印证了上面
"3 倍重读"的解释**(如果 qo=3 是撞到了别的墙, 两者的达成带宽不会相等)。

15% 说明这条 kernel **不是带宽受限**, 还有余量。可能的原因(未逐一验证):
- q8_0 的行内布局是 `{fp16 scale; int8 values[32]}` = 34 B/块, lane 的 8 字节起点
  落在 2/4/6/0 mod 8 上, **没法用对齐的 64 位 load**, 只能发 8 条 `LDG.U8`;
  turbo3 侧另有 3 条小 load。每 token 每 lane 11 条访存指令, LSU 发射受限;
- 占用率 25%(4 block/SM x 128 线程, 受 126 寄存器约束), 在途访存请求可能不足以
  掩盖 HBM 延迟 —— 这正是当年 fp16 XQA 用 `__launch_bounds__` 从 2 block/SM 调到
  5 block/SM 拿到收益的同一个杠杆。

**两块余量相互独立**, 且都作用在生产形状(qoLen=3)上。

### 一个方法论点: "等效 B/行"这一列值得每个访存类微基准都加

上面几乎所有判断都来自这一列, 而它只是 `ms -> ms * BW / 行数` 的换算。它的作用是
**把绝对耗时变成可以和理论下限直接比的量纲**:

- kvLen=2048 的冒烟测出 7954 B/行 vs 下限 372 -> 差 21 倍, 一眼看出**这个尺寸根本没跑到
  带宽墙**, 测到的主要是 kernel 启动数之差(默认路 ~96 次启动, 融合路 2 次)。
  没有这一列, 很容易把那组 8.7x 直接写成"省掉物化的收益";
- `qo3/qo1 = 3.08` 一眼指向设计里的 `GROUP_CHUNK`, 不用去猜;
- 达成带宽 15% 说明"还没到墙", 于是"5.26x 已经很好了"就不能作为收工的理由。

## 15.40 从函数中间开始读 (2026-08-20)

同一天里同一个病犯了两次, 一次在上游一次在我这, 所以值得把**判据**单独固化下来。

### 事故

前缀缓存的 vision 半边修完之后, 我报了一条根因:

> 带图请求一旦命中前缀, `ForwardMultimodal` 会拿到被截断的 `inputIds`,
> `EncodeVisualItems` 仍然编码全部图片, 于是 `BuildMultimodalPositionData` 末尾的
> `AssertInFastLLM(imageIndex == imageGridThwList.size(), ...)` 抛错 -> **确定性 500**。

这条结论**是错的**, 而且它已经被拿去论证了一次生产重启(把
`FASTLLM_PREFIX_CACHE_MULTIMODAL` 关掉)。

真相是 `ForwardMultimodal` 的**入口**有一道守卫:

```cpp
std::vector<int> Qwen3_5Model::ForwardMultimodal(...) {
    ...
    if (pastKeyValues.size() > 0 && pastKeyValues[0].second.dims.size() > 0) {
        // 续 prefill(典型为前缀缓存 HIT 之后)  <-- 注释是原来就有的
        AdjustPositionIdsWithDelta(positionIds, *deltaIt->second[0], adjustedPositionIds);
        ... Forward(...)   // 普通文本前向, 不做视觉编码
        return ret;
    }
    // ↓↓↓ 我是从这里开始读的 ↓↓↓
    AssertInFastLLM(inputIds.dims.size() == 2 && inputIds.dims[0] == 1, ...);
    ... EncodeVisualItems / BuildMultimodalPositionData ...
}
```

命中恢复时 `RestorePagedPrefixCacheExtra` 会给线性层(含 layer 0, 在
`full_attention_interval=4` 下它是 GDN 层)写上 `dims`, 所以那个条件成立 ——
**我想分析的那条路径, 根本不会走到我分析的那段代码。** 那道守卫的存在意义,
恰恰就是把它导向别处。

### 判据(可直接执行)

> **在建立"某条路径会怎样"的机制假设之前, 必须先从函数入口读到你关心的那段,
> 把沿途所有 early return / guard 列出来, 并逐个回答"我这条输入会不会被它拦掉"。**

配套的三条:

1. **先定位入口, 再定位机制。** 顺序反了就会变成"先有假设、再去找支持它的代码"——
   而任何足够大的函数都能找到一段支持任意假设的代码。
2. **注释是证据, 尤其是别人写的注释。** 上面那句"续 prefill(典型为前缀缓存 HIT
   之后)"是原来就在的, 它直接说明这条路**被设计过**。我如果读到了它, 就不会
   得出"没人考虑过这种情况"的结论。
3. **降级用词, 不要降级检查。** 拿不准时写"读代码得出, 尚未在运行时复现",
   而不是写"确定性 500"。前者不会让人去重启生产。

### 这和 §15.36 是同一类

§15.36 是"结论没记测试形状" —— 用一次特定形状的测量推出全局结论。
这一条是"结论没记入口守卫" —— 用一段特定代码推出全局路径行为。
**共同点: 用局部事实推全局结论, 而且省掉的恰恰是决定结论适用范围的那部分上下文。**

同一天的第三个样本: 把 `req#20~30` 那串 99% 命中当成"vision 前缀缓存修好了"的证据——
那些请求无法确认是 vision, 而纯文本在修复前就能命中 94%, 所以那串数字**不能区分
修复前后**。同样是局部读数推全局结论。

### 附带产物: 一个假阳性探针

查这件事的过程中发现 `EzraVastLLM/scripts/probe_prefix_multimodal.py` 的 image
轮次全部命中 `Jinja Error: expression error at end (stack=3)` ->
`using MakeInput fallback`, 图片内容被丢掉, 实际发出去的是纯文本。它报的
`hit=512` / `hit=896` 看着像"vision 命中了", 其实一次都没测到 vision。

**探针必须自证它测到了目标**: 没有在链路里观察到 vision 标记, 就判"探针无效",
而不是"通过"。这条规矩今天已经救过一次(llama-cli 把 prompt 回显当成模型抄写正确)。

---

## 15.41 本机 HTTP 探针被静默绕道代理 (2026-08-20)

### 现象

带图抄写探针挂了 79 分钟不返回, 而**后端日志里明明写着它早就处理完了**:

```
[req 2] prefill done: 3989 tok in 6.29s (634.6 tok/s)
```

`ss -tnp` 一看就清楚了 —— 探针进程的连接是:

```
ESTAB  ->  127.0.0.1:10808
```

10808 是给 HuggingFace 上传用的那个 HTTP 代理端口, **不是** `:8000`。

### 原因

`~/.zshrc` 里有:

```sh
export http_proxy="http://127.0.0.1:10808"
export https_proxy="http://127.0.0.1:10808"
```

而 tmux 新开的窗口默认用 zsh 登录 shell, 所以继承了它们。
**Python 的 `urllib` 在没有 `no_proxy` 时不会自动把 localhost 排除在代理之外**
(很多人以为会)。于是打 `http://127.0.0.1:8000` 的请求先送到 10808, 由代理再
转回本机 —— 请求确实送达(后端处理了), 但响应卡在代理回程上。

### 为什么这条比"探针挂住"严重得多

**它会污染时延测量, 而且完全没有征兆。**

挂住至少是显性失败。真正危险的是"能返回、只是多了一段代理开销"的情形 ——
任何本地 HTTP 基准(TTFT、decode tok/s、端到端墙钟)都会被悄悄加上一笔,
而所有人都会把它当成模型的开销。本仓大量结论建立在这类测量上。

### 判据与修法

判据(一行就能查):

```sh
ss -tnp | grep "pid=<探针pid>"       # 目的地必须是 :8000, 不是代理端口
tr '\0' '\n' < /proc/<pid>/environ | grep -i proxy
```

修法是在**脚本内部**清除, 而不是指望调用者:

```python
for _v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
           "all_proxy", "ALL_PROXY"):
    os.environ.pop(_v, None)
os.environ["no_proxy"] = "127.0.0.1,localhost"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
```

原因同 §15.36 里 `start_prod.sh` 不读 `.env` 那条: **不要把正确性寄托在
"启动我的那个 shell 恰好有/没有某个环境变量"上**。同一个脚本换个拉起方式
(tmux / nohup / cron / systemd)行为就变了, 而且变化不报错。

已给 7 个探针脚本加上: copy_fidelity_vision / copy_fidelity_http /
toolname_probe / goal_acceptance / probe_prefix_multimodal / runtime_metrics /
verify_prefix_cache。

---

## 15.42 bind 失败后进程不退出: 握着 20GB 权重却不服务 (2026-08-20)

### 现象

生产静默不可用 23 分钟。后端日志:

```
[Load] model loaded, workers started
[Server] socket ready!
bind error!
```

之后**进程继续运行**: RSS 20GB(权重已读进主机内存)、GPU 只有 724MB
(从未上卡, 因为没有请求触发懒上传)、proxy 侧永远 `backend=STARTING`,
客户端拿到 `timed out` 和 `502 Bad Gateway`。

诱因: 重启时上一个 apiserver 还占着 8002, 新进程 bind 失败。

### 两个独立缺陷

**(a) bind 失败必须让进程退出。**
现在它只打一行 `bind error!` 就往下跑。于是生命周期管理器看到的是"进程还活着,
应该是在启动中", 而不是"启动失败, 该重试" —— **永远不会重试**。
正确做法: bind 失败 -> 非零退出码。
代价对比: 白占 20GB 主机内存 + 一次 16GB 的机械盘加载(约 3 分钟), 期间全部
请求超时; 而正确退出的话管理器几秒内就能重试。

**(b) proxy 的信号量槽位泄漏。**
日志里出现 `streams=-3/4` —— 计数变成**负数**, 说明 release 被多调用过。
后果不只是数字难看: 并发闸门失效(实际放行的比配额多), 且
`reap_leaked_streams` 的水位判断 (`held = max_concurrent - _slot._value`)
会算出错误的持有数。

### 为什么没被更早发现

因为**它长得像"正在加载"**。16GB 权重从机械盘读本来就要 3~9 分钟, 所以
`backend=STARTING` 持续几分钟是完全正常的现象。区分点在于:

```
正常加载中: weights_load 百分比在涨, RSS 在涨
bind 卡死  : weights_load 已 100%, RSS 稳定在 20GB, 但 GPU 仍是 724MB
```

**"GPU 只有几百 MB 而 RSS 已满"是这个故障的指纹** —— fastllm 读完权重先放主机
内存, 到第一次真实请求才上卡(见 §15.35 第 5a 条), 所以 bind 失败时权重永远
停在主机侧。

### 规矩

- 服务进程的任何**绑定/监听失败**都必须是致命错误, 不能降级成一行日志。
- 拉起前先探测端口占用并等待释放。`start_prod.sh` 里本来就有这个等待循环
  (kill 后最多等 25 秒再强杀), 但**这次的重启不是它发起的** —— 是 proxy 的
  lifecycle 管理器自己拉起的, 那条路径上没有同样的等待。
  **同一件事有两条拉起路径时, 保护要加在两条上。**

## 15.43 串行下成立的分析方法, 并发之后会静默失效 (2026-08-20)

### 事故

要回答"MTP 到底有没有在带图请求上生效", 用的方法是**按请求窗口归属日志行**:
在日志里找 `[Qwen3.5 vision]` 标记, 把它到该请求结束之间出现的
`[Qwen3.5 MTP] pos_accept_rate=...` 都算作这个请求的。

`--batch 1` 时这个方法是对的, 并且给出过一个很硬的结论:

```
vision 窗口 19 个 / decode 8734 token / 窗口内 pos_accept_rate 行 0
text   窗口 47 个 / decode 11404 token / 窗口内 pos_accept_rate 行 21
```

`--batch 4` 上线之后同一套方法继续跑, 又给出了一个"vision 窗口内 0 条"的结论 ——
**这次是错的**。因为最多 4 个请求交错, "该请求结束"这个边界被另一个先完成的
短请求提前关掉了(日志里 L16 起、L17 就结束的那条显然不是 6 万 token 的 vision
请求)。窗口划错, 结论照旧输出, 没有任何报错。

### 判据

> **一个依赖"事件先后顺序"的分析方法, 其前提是"同一时刻只有一个主体在产生事件"。
> 并发度从 1 变成 N 的那一刻, 这个前提就没了, 而方法本身不会报错 —— 它会继续
> 输出看起来同样确定的结论。**

所以并发度是**分析方法的前置条件**, 和被测对象的配置一样要记进结论里。
`--batch 1` 下得出的"按窗口归属"结论, 不能直接搬到 `--batch 4`。

### 修法: 把推断换成记账

不要在日志上做归属推断, 让产生事件的地方**自己带上归属**:

* 日志行本身带标识: `pos_accept_rate` 那行补上 `mm=`(这一路请求是否多模态);
* 更硬的是**直接记数**, 完全绕开时序: `mtp_steps_multimodal != 0` 就是
  "MTP 在带图请求上跑过", 无需知道任何一条日志属于谁。
  (`MtpAttributionStats`, 暴露在 `/props`。)

判据从"某个窗口里有没有出现某行"变成"某个计数是不是 0"。后者不受并发影响。

### 同一类的其它例子

- §15.40「从函数中间开始读」: 用局部代码推全局路径行为;
- §15.36「没记测试形状的结论」: 用一次特定形状的测量推全局结论;
- 本条: 用串行下成立的归属方法推并发下的结论。

**共同点: 结论本身没错, 错的是"它成立的前提"没跟着结论一起记下来。**

## 15.44 新增并行代码路径时, 漏掉的永远是"顺带做的事" (2026-08-20)

一天之内, **同一种不对称出现了四次**, 全都是"多模态路径是后加的, 主逻辑抄对了,
但原路径上那些**顺带做的事**一条都没跟过来":

| # | 文本路径顺带做的事 | 多模态路径 | 后果 |
|---|---|---|---|
| 1 | 分页 manager 用 `threadTpPagedCacheBase + i` 当全局层号 | 硬编码 `i * 2` | 两套互不相交的 manager, 前缀命中**恒为 0**, 页池还被分配两份 |
| 2 | decode 走 `ForwardGPU`(能吃 MTP 草稿) | `multimodalInput` 只在析构里清, 于是每个 decode token 都走 `ForwardMultimodal` | MTP 对带图流量**从未生效** |
| 3 | 分块 prefill 在块边界调 `TryRecordPagedCache`(全文件 10 处) | 分块循环里 **0 处** | 整段 prompt 只有一个深度有快照, 任何部分匹配被整条丢弃 |
| 4 | `ForwardSingleGPU` 里 `speculativeHiddenStates.CopyFrom(...)` | `ForwardFromHiddenStates` 里 **0 处引用** | MTP 草稿缓存没法 seed(即使 2 修好了) |

四条的共同点: **主计算是对的, 输出也是对的**, 所以任何"跑一下看结果对不对"的
验证都发现不了。全部只在**别的子系统的指标**上体现(命中率、接受率、显存),
而且都要绕好几层才能追回来。

### 判据

> **新增一条与已有路径并行的代码路径时, 不要只对照"主逻辑"。要把原路径从入口
> 到出口逐行过一遍, 把所有"顺带做的事"列成清单 —— 注册、记账、埋点、缓存回填、
> 状态捕获、边界钩子 —— 然后逐条回答: 新路径要不要做? 谁来做?**

可操作的做法有两个:

1. **对照计数**: `grep -c` 原路径上那个调用在全文件出现多少次, 再看新路径里出现
   几次。上表第 3 行就是这么发现的(10 vs 0), 第 4 行同理(有 vs 0)。
   这比通读代码快, 而且不依赖"我记得原路径还做了什么"。
2. **让两条路径共用那段顺带逻辑**, 而不是各写一份。基址就是这么修的
   (`PagedCacheLayerBase()`) —— 共用之后, 想漏都漏不掉。

### 与 15.40 / 15.43 的关系

- 15.40: 从函数中间开始读 -> 漏掉入口守卫;
- 15.43: 串行下的归属方法 -> 漏掉并发前提;
- 本条: 新并行路径 -> 漏掉原路径的副作用。

三条都是**"漏掉了不在视线正中的那部分"**。区别只在漏的是守卫、前提, 还是副作用。

## 15.45 动态 grammar 不能用一张 mask 验证整个 MTP 草稿块 (2026-08-21)

旧路径在 `tool_call_allowed_token_ids` / `tool_call_blocked_token_ids` 非空时直接禁用
MTP。仅删除这道守卫也不正确：verify 的第 0 行使用当前 grammar 状态，第 1 行必须先
假设接受 draft 0 再重算 mask，后续行同理；函数名、参数名和闭合标签的边界都可能落在
同一个草稿块内。

当前实现由 `AppendToolCallConstraintRowConfigs` 按 draft 前缀逐行推进状态，
`Qwen35ApplyCudaTokenConstraints` 在完整 FP32 CUDA logits 上同时应用 allow-list 与
block-list，再进行 greedy/top-k 采样。全灭行保持原有“退回自由采样”语义。

验证不是只看最终文本：

- CUDA mask 与 PyTorch `torch.where` 在 V100 上对拍 18 组、6,756,168 个 float，
  逐 bit、argmax、top-k 全部一致；
- 合成 grammar 检查证明四行依次允许 `list`、`_dir`、`>`、`<parameter=`；
- 同一生产二进制、同一强制 `get_weather` 请求：plain decode 与 MTP 都是 37 token，
  工具名和参数 JSON 完全一致；
- MTP 请求计数增量：`mtp_constrained_verify_steps=9`、
  `mtp_constrained_verify_rows=26`，`toolcall_mask_emptied=0`。

判据：动态约束下“输出看起来合法”不够；必须同时证明逐行 mask 正确、MTP 没有静默
回退、以及与同二进制 plain decode 的可观察结果一致。

## 15.46 “backend reloading” 曾经是流错误的错误总称 (2026-08-21)

`_local_stream` 原来吞掉任意 `Exception`，统一回传
`FastLLM backend is reloading`。因此健康的 READY 后端遇到 404、客户端断开或网络读
错误时也会显示成 reload。

改为记录异常类型和阶段；只有 lifecycle 非 READY 才返回 `backend_reloading`，
READY 状态的真实中断返回 `backend_stream_interrupted`。客户端断开不再惩罚本地后端，
已经输出本地 token 后也不再拼接另一个 provider 的回答。

这次新日志把现场钉成了：

`backend returned 404: The model auto does not exist`

根因是生产 profile 没声明 `FASTLLM_AUTO_ROUTE_ALIASES=auto`，导致公平路由选中 local
后没有把公开名 `auto` 改写成内部 slug `qwen3.8-fastllm`。默认值和生产 profile
现都显式包含 `auto`；Python 流探针验证 `200 + [DONE]`，无
`model_not_found` / `backend_stream_interrupted`。

随后第二个现场是正文和 thinking 都已完整显示，尾部却追加
`backend_stream_interrupted`。新日志给出
`yielded=1, UnicodeDecodeError: invalid continuation byte`：byte-level tokenizer
生成了无法组成 UTF-8 的低频 token 字节，而 proxy 的 strict incremental decoder 把
单个不可恢复字节升级成了整流失败。

现在仍用 incremental decoder 跨网络 chunk 保留正常多字节字符，但以
`surrogateescape` 精确标出后端的非法字节，转发前只把这些字节替换为 `U+FFFD`，继续
发送后续 SSE 与 `[DONE]`；同时记录 replacement 数量。缺失字符本身无法从服务层恢复，
但不能因此把已经完成的回答标成失败。

## 15.47 SM70 的旧 XQA/Flash 开关与 packed turbo-XQA 不是同一路 (2026-08-21)

`FASTLLM_CUDA_SM70_PAGED_XQA` 只接受 FP16 K/V；
`FASTLLM_CUDA_SM70_FLASH_ATTN` 只接受 FP8_E4M3 K/V。turbo3 profile 下两者为 no-op，
但这不代表融合注意力没有生效。

生产实际走独立入口 `FASTLLM_CUDA_SM70_TURBO_XQA=1`：
K=`q8_0_kv`、V=`turbo3`、qLen 1–4。运行期 `/props` 已观测
`attn.sm70_turbo_xqa` 432 次、624 token、`max_n=3`；fp64 对拍 92/92 通过。

边界必须写清：当前 packed 融合 kernel **只支持 turbo3 decode/短 query**。
Turbo4 已有打包和反量化，但没有融合 XQA；packed turbo3/4 的 prefill Flash 也尚未
实现。启动日志现在分别报告真实 turbo-XQA 与两个旧入口，避免再把 no-op 日志误读成
“移植未生效”。

## 15.48 Byte-level tokenizer 必须跨 token 组装 UTF-8 后再发 SSE (2026-08-21)

proxy 把非法字节替换成 `U+FFFD` 只能避免整流失败，不能恢复字符。生产强制模型只输出
`👋` 时，旧后端返回 `����`。直接抓 `:8002` 原始 SSE 得到决定性证据：

- 同一个 emoji 的 `F0 9F 91` 在 event 11，`8B` 在 event 12；
- 另一处拆成 `F0 9F` / `91` / `8B` 三个 event；
- 整份后端 SSE 本身无法用 strict UTF-8 解码。

根因在 apiserver：每次 `FetchResponseTokens` 后只用一个 token 调
`Tokenizer::Decode`，随即把返回的原始 byte fragment 放进 JSON。`byteAsChar`
tokenizer 的单 token 不保证是完整 Unicode 字符。

`Utf8StreamAssembler` 现在跨 token 保留最多 3 个未完成尾字节，完整 code point 才交给
stop matcher / output parser / JSON；真正非法、overlong、surrogate 或超出 U+10FFFF 的
序列才替换。生产同一探针结果为 `👋`、0 replacement、合法 raw SSE、正常 `[DONE]`。

## 15.49 名字规范化不能污染工具 XML 结构；OpenWebUI 必须走 native (2026-08-21)

为兼容 `Bash` ↔ `bash`，`ToolCallCanonicalKey` 会忽略大小写、`_`、`-`、空格和点。
它曾被错误复用于 `<function=` / `<parameter=` 等结构前缀，于是 `<_` 规范化后等于
`<`，下划线、空格和点都成为“合法但零进度”的 token。MTP 与 plain decode 用同一
grammar，所以两者都能稳定复现 `<tool_call><_  _  _ ...`。

现规则：

- S1 工具名、S3 参数名允许 canonical alias，但规范化后必须有进度；
- XML 标签逐字节匹配；
- 声明本身正确位置的 `_` 仍可精确生成；
- 前导/重复 `_`、空格、点拒绝。

生产 MTP2 两工具探针返回标准 `delta.tool_calls`：
`read_file({\"path\":\"/app/pyproject.toml\"})`，finish=`tool_calls`，
`mtp_constrained_verify_steps +11`、rows `+32`、mask-empty/malformed/repair 均为 0。

OpenWebUI 原样显示 XML 是另一层：请求后的所有 constraint/MTP 计数增量均为 0，
证明它没有发送 OpenAI `tools` 数组。其 SQLite 中活动模型 `auto` / `qwen3.8-27b`
的 params 原为 `{}`；已按 v0.9.6 官方契约改为
`{\"function_calling\":\"native\"}` 并重启。legacy/default 模式只向 prompt 注入工具
说明，无法执行 FastLLM 的原生 `tool_calls`。

## 15.50 工具 guidance 必须由活动 Jinja 编译，而不是手抄规则 (2026-08-21)

生产模板用 sentinel tool/两参数做 dry-run，编译出 function/parameter/value/close 的
实际布局；模型加载时编译失败则 tools 请求拒绝服务。S1/S3 只允许请求 schema 的精确
名字，别名映射只留最终输出防御层，每个真实调用最多记录一次。

MTP 行构造还会在 proposal 违反当前 allow/block 时剪掉后续不可达行；旧实现继续把
非法 draft 追加到临时文本，产生 `partial='ash'` 的假 mask-exhausted，虽然该分支永远
不会 commit。生产 `bash(ls -la)` 探针：原生 tool_calls、MTP constrained steps>0，
mask-empty/malformed/repair=0，新进程无 exhausted 日志。

提交：`c91580a2`、`f8290701`。

## 15.51 packed Turbo3/Turbo4 prefill 的安全收益区间 (2026-08-21)

新增 SM70 packed prefill：K=q8_0_kv，V=turbo3/turbo4，在线 softmax、延后一次逆 WHT。
qLen=8/16/32 对独立 fp64 参考全部通过；Turbo4 的未对齐 uint32 nibble load 已改为
byte-wise load。

实测 qLen=8 相对 BATCH_GQA 约 5.6–7.5x，qLen=32 约 2.34x；逐-query qLen=512
只有 0.17x。8-query 寄存器 tile 负实验更差：40.03ms vs 1.63ms，已回退。因此默认
只路由 qLen=5–32，长 prefill 保持 BATCH_GQA。后续 shared-memory/single-launch
query tile 记录在 VastLLM issue #1。

提交：`dd035cf1`。

## 15.52 参数值的裸 `<` 不能被提前判成闭合标签 (2026-08-21)

OpenWebUI `write_file` 两次稳定把 Python 正则截在 `re.findall(...([^`。直接对生产
OpenAI API 复现后，非流式响应已经是合法 JSON 且 `finish_reason=tool_calls`，证明
不是 SSE/OpenWebUI 丢字节：预期下一个正文字符 `<` 被 S4 grammar 当成
`</parameter>` 前缀，白名单强迫模型提前闭合参数。

修复后 S4 不再对白名单提交闭合前缀；普通 `<`、HTML 和 `r"[^<]"` 均保持自由。
空值保护改为只屏蔽**实际完成**全空白 `</parameter>` 的 token，不能屏蔽 `<` 或
`</pa` 前缀。C++ 状态机测试、Python 镜像不变量、非流式和 42-chunk 流式 API
均通过；重组内容包含完整 `([^<]+)</span>`，落盘后 `py_compile` 通过。

提交：`b716145b`。

## 15.53 长 qLen 要共享 KV，但必须保留 tensor-core 矩阵并行 (2026-08-21)

`2 query × 3 GQA head` 的 scalar shared-memory kernel 虽然把 global KV 读取降到
每 tile 一次，却在 qLen512/kv8192 跑到 54.70ms（BATCH_GQA 约 3.86ms）。
Nsight Compute 2024.3.2 实测：DRAM 0.39%、L2 hit 98.84%、L1/TEX 69.60%、
56 registers/thread、32.83KB shared/block、occupancy 18.58%。瓶颈是 shared
重读、barrier 和 scalar QK/PV，不是 DRAM 或 spill；该 kernel 已删除。

保留实现是 packed all-head tensor path：每 chunk 只反量化4个 Q8/Turbo3或Turbo4
KV head，24个 query head 用 batched tensor-core QK/PV，一次 softmax 覆盖全部
head。它同时共享 KV、保留矩阵并行，并减少逐 KV head 的 GEMM/softmax/store launch。

CUDA-event 20轮实测（新/旧 BATCH_GQA）：

- Turbo3 kv8192：q512 3.29/3.73ms（1.13x），q2048 10.64/11.12ms（1.05x）；
- Turbo3 kv32768：q512 12.77/14.25ms（1.12x），q2048 42.47/44.31ms（1.04x）；
- Turbo4 kv8192：q512 3.36/3.75ms（1.12x），q2048 10.62/11.14ms（1.05x）；
- q64 kv8192：1.08/1.69ms（1.57x）。

独立 CPU/fp64 抽样对拍覆盖 Turbo3/4、qLen512/2048、乱序页、非满尾页和
kvLen10003 partial multi-chunk，287/287 PASS；XQA 也加入 Turbo4，qLen1/3
相对 fallback 分别约 10.97x/10.30x。路由名：
`attn.sm70_turbo_all_head_gqa`。

Volta profiler 固定使用用户级
`~/.local/opt/nsight-compute/2024.3.2.3`；`~/.local/bin/ncu` 由用户级
`alternatives --altdir ~/.local/alternatives --admindir ~/.local/var/lib/alternatives`
切换，系统 2025.4.1 保留共存。

提交：`d3561684`。

## 15.54 finally 里抛的异常会静默杀死 worker，整个调度器变砖 (2026-08-22)

proxy 的 BackendScheduler._worker 在 finally 里做租约释放。当
reap_stale_leases 把"正在跑的非流式长请求"误判成泄漏并归零 active 后，
worker 的 finally 再 release 就下溢抛 BackendLifecycleError —— 而 finally
里抛的异常没有任何 try 接得住，协程当场死亡，且 create_task 的异常无人
await ⇒ 完全静默。四个 worker 逐个死光后 metrics 显示
workers=0/4 sched_active=4 queued=4 reqs=0，/health 仍报 READY。

两个致命细节:
1. 误判的根源是泄漏判据看不见非流式请求 —— _INFLIGHT 只登记流式,
   pending() 只算队列, 已被 worker 取走正在跑的请求两者都不算。
2. 同一个 codebase 早用"纪元法"修过流式槽位的同款问题, 租约这条路漏了。

修复: 租约加纪元(reap 归零时递增, 旧租约 release 变 no-op)、_release 不
再抛异常、reap 判据补 scheduler.active、finally 全程异常安全 +
respawn_dead_workers 自愈 + DEADLOCK 显式报警。四个不变量回归全过。

提交: thinking_proxy.py (f6007acd)。

## 15.55 逐状态白名单前缀匹配拒绝跨边界 token，约束从护栏变成改写器 (2026-08-22)

工具调用约束的旧实现是"每状态维护一个候选字符串集合, 逐状态独立做前缀
匹配"。一个 token 若跨过状态边界 —— 同时补完结构前缀并开启下一状态内容
—— 必然被拒。逐 token 轨迹实测: 模型想调 bash, 自然选的 token 是
'=b'(id 21402, '=' 补完 functionPrefix + 'b' 开启名字), 被判定
"\n<function=b" 不是 "\n<function=" 的前缀而拒绝, 只放行光秃秃的 '='。
模型被迫改走自己概率更低的拼法, 隐状态偏离, S1 就选了 todo。

规模量化(一次工具调用): 90 步约束中 17 步放行集仅 1 个 token(完全没得选)、
13 步直接否决模型的 argmax。这就是"强制"而非"纠正"。

修复对齐 llama.cpp 的 llama_grammar_accept_str: 一个 token 合法当且仅当
其全部字符能被语法从当前状态连续消费。逐字符推进, 跨边界天然支持,
转移规则全部取自 CompileToolCallGrammarLayout 编译出的 layout, 零模型专有
字面量。起点由"从块开头逐字符重放"构造, 与 LocateToolCallGrammarCursor
互校验(WALKER MISMATCH trace), 重放卡住自动回退旧路径。

验证: grammar 开/关对照 3 轮零例外(开=关=bash, 此前开=todo);
mask_overrode_argmax 从 13 降到 0。

提交: c550535b。

## 15.56 全词表 blocked 会触发 CUDA 兜底全放开，约束静默失效 (2026-08-22)

S4 空值守卫的判定是 "tail + tokenText 是否构成完整 </parameter>"。
当 tail 本身已含完整闭合标签时, 追加任何 token 都仍然匹配 ⇒ 整个词表
248320 个 id 全被塞进 blocked(实测 503 次)。而 CUDA 掩码对"全禁"的兜底是
memset(row,1) 全放开 —— 约束在最该生效的一步彻底失效, 且
toolcall_mask_emptied 恒为 0(那个计数器统计的是 LLMSampling 里的另一条
兜底, 生产路径根本不走)。**完全静默**。

现场: <parameter=path>\n\n</parameter> 空值参数直接发给客户端, path=""
导致 EISDIR, 客户端重复 5 次后挂死。

修复: 守卫在 tail 已含完整闭合标签时直接返回; CUDA "全禁->全放开"兜底
加计数器 toolcall_cuda_mask_all_blocked, 杜绝静默。

教训: "计数器为 0 证明没发生"只对覆盖了全部路径的计数器成立。
同一语义有多条兜底分支时, 每条都要有自己的计数器。

提交: c550535b。

## 15.57 四个假设先后被自己的数据推翻：先确认代码路径被执行 (2026-08-22)

排查"模型想调 bash 却发出 todo"时, 四个看起来很有说服力的假设全部被
证伪: (1) 模板 NO suffix 禁止第二调用 —— 放开后照样 1 个, 且官方与
Unsloth 两版逐字节相同; (2) 掩码屏蔽 bash —— 词表复现显示 bash 就在
S1 的 14 个放行 token 里; (3) MTP 投机行错位 —— top_k=51 关 MTP 输出
逐字节相同; (4) 掩码只能筛 top-k —— 掩码在全词表先于 top-k 生效。

方法论教训:
1. 先确认代码路径真的被执行, 再分析它。两轮诊断加错位置
   (LLMSampling/LLMSamplingOnly 生产根本不走), 靠"trace 输出 0 条"才发现。
2. A/B 必须排除混杂。prompt_tok 两边都是 12808 才证明 prompt 真的一致
   (raw_prompt=true 生效)。
3. 日志是追加式的, 全文件 grep "加载失败"会命中旧代内容, 必须按生成分界切片。
4. 并发流量污染 trace: 用户的 OMP 会话(12 工具集)曾被误当成自己探针
   (9 工具集)的输出, 需用 allow_n/allowed_values 当判别器分离。
5. 模型"意识到自己不对"是真实信号: thinking 全程清醒点名 bash, 是约束
   把输出改掉了 —— 用户的直觉比四个技术假设都准。

## 15.58 Claude Code 连本地网关: 配置隔离 + 归因头剥除 (2026-08-22)

用 Claude Code 打本网关做 bench 时, 内联 `ANTHROPIC_BASE_URL=http://127.0.0.1:8000`
**会被 `~/.claude/settings.json` 的 `env` 盖掉**(那里把 base_url 指向外部端点, 且
优先级更高), 现象是 "Model not exist" 且请求根本没进 8000 的日志。解法: 用临时空
`CLAUDE_CONFIG_DIR`(`mktemp -d` + `echo '{}' > settings.json`)隔离, 不加载/不改用户
真实配置, 内联环境变量才生效。

归因头: Claude Code 每请求在 system prompt 最前面塞一行会变的
`x-anthropic-billing-header: cc_version=...; cch=...`, 落在前缀第 0 位 -> 前缀缓存每轮
全失效(本地慢 ~90%)。`thinking_proxy.py` 已加 `_strip_claude_code_attribution` 在渲染前
剥掉(只吃开头, 不碰正文), `/health` 暴露 `cc_attribution_stripped` 计数。验证: 真实
claude 流量下该计数 0->6。注意用户真实 `~/.claude/settings.json` 已设
`CLAUDE_CODE_ATTRIBUTION_HEADER=0`, 日常本就关闭; 测试时**不要**设 0 才能看到剥除。

教训: 别在诊断输出里回显 `~/.claude/settings.json` 的 `env` 值——里面有真实密钥;
只查键名。

## 15.59 launcher 的 profile 必须用绝对路径 (2026-08-22)

`launch_proxy_tmux.sh` 的 `PROXY_SHELL` 在 tmux pane 里运行, pane 的 cwd 被固定为
`.../1CatVLLM/v100-perfs`; pane 内会再次 `. $ENV_FILE`。若传相对路径(如
`v100-perfs/runtime/...`), 会解析成 `v100-perfs/v100-perfs/runtime/...`(不存在),
`. $ENV_FILE` 静默失败 -> `FASTLLM_BACKEND_URL` 为空 -> `FASTLLM_ENABLED=False` ->
proxy 退回 llama 模式(日志 `starting llama-server on port 8001` + llama-server 二进制
缺失的 FileNotFoundError), 8002 的 FastLLM 后端永远起不来。调用 launcher 或写切换模型
的脚本时, profile 一律用绝对路径。

### 15.60 Cyber thinking 退化、图片格式穿透与 C++ 管理面（2026-08-22）

#### A. Cyber IQ4_XS 的长周期 thinking 退化

OpenWebUI 现场：模型在设计 SVG 的鹈鹕嘴袋路径时，将
`OK final pouch: ... Let me just write: M478 239 ... C577?`
整段约 100B 的思考片段重复 30 次以上。它发生在 thinking 段，不在工具参数值
S4；现有 `ToolCallValueLoopPeriod` 只管 S4 且最大周期 96B，不能也不该靠工具
grammar 修模型内部思考。生产已切回 Unsloth UD-Q5_K_M；Cyber 保留作按需 profile。

#### B. `unsupported image format` 的真根因

后端 `image_loader.cpp` 只解 PNG/JPEG。proxy 原本虽然用 Pillow 归一化图片，
但只有「非 RGB 或 EXIF 需要旋转」时才转 PNG；RGB WebP/GIF 会原样穿透，
于是后端确定性 400，客户端又将流式 400 当空回答重试。

修复：
- data URL 与 HTTP(S) 图片统一在 proxy 异步读取；
- 实际格式为 PNG/JPEG、RGB、无旋转、非动画时保留原字节；
- 其余 Pillow 可解格式（WebP/GIF/BMP/TIFF/ICO/PPM 等）统一转 RGB PNG；
- GIF 取首帧；>32MiB/不可解格式在 proxy 直接明确 400；
- Pillow 解码/编码走 `asyncio.to_thread`，不阻塞 FastAPI 事件循环。

单测：WebP/GIF/BMP 均转 PNG；RGB PNG/JPEG 保留原字节。AVIF/SVG 栅格器本机
未安装，仍返回明确错误，不静默穿透。

#### C. TUI 删除，改为 FastLLM C++ 内嵌管理 WebUI

删除 `fastllm/tools/apiserver_tui.py`，新增 C++ 管理面：
- `GET /admin`：无外部依赖的单页；
- Bearer `AUTH_TOKEN` 双层认证（proxy + apiserver，后端 fail-closed）；
- 引擎内直接读取 VRAM 对账、物理页池/L1 trie、L2 RAM、L3 磁盘、
  命中/逐出、工具 grammar 指标；
- 展示 backend + thinking_proxy 日志尾部；
- 列出 24 个 profile，在线切换/停止；
- profile 编辑：`0/1/true/false/yes/no/on/off` 自动渲染 checkbox，其余文本框；
  原子写回并保留原注释；profile 名严格白名单防目录穿越。

验证：C++ 增量编译通过；浏览器实际登录后可见 ud-q5 READY、262K/batch4、
页池/显存/cache 数据与日志；无 token 为 401；profile 读取 56 个键、
其中 21 个 checkbox；临时 profile 原子修改两项并校验成功。

### 15.61 傲腾可行性、逐出静态审计、restore 永久失败根因 (2026-08-22 下午)

#### A. 256G 傲腾(@690 元)能不能用 —— 主板已查明

主板 ASUS TUF GAMING B660M-PLUS WIFI D4 (BIOS 3212) + i3-12100:

- **傲腾 DIMM(PMem): 不支持**。PMem 需要 ACPI NFIT 表, 本机 ACPI 只有
  APIC/DMAR/TPM2 等, 无 NFIT; 也无 /dev/pmem*。B660 消费级平台本来就不支持
  Optane Persistent Memory(那是 C620A/C740 芯片组 + Xeon/特定酷睿的能力)。
- **傲腾 NVMe SSD(Q系列/P系列): 可以用**, 但收益取决于插哪:
  - M.2_1 由 CPU 直连 x4(当前 Kioxia NVMe 占用, Gen4 x4);
  - 其余 M.2 走 B660 chipset PCH, 共享 DMI3 (~3.9GB/s);
  - V100 已占 CPU x16 controller #1 的第一条 x16(GPU 01:00.0)。
- **结论**: 买傲腾 NVMe 当 L3/prefill-cache 盘是划算的(顺序读写远超机械盘,
  低 QD 随机延迟 <100µs vs HDD 12ms seek); 但别期待 PMem 级字节寻址。
- 注意系统盘 btrfs 已用 91%(剩 179G), 新盘应独立挂载, 别塞进根分区。

#### B. 容量压力淘汰链路 —— 静态审计(未动生产)

L1→L2→L3 全链路代码级确认(fastllm.cpp):

- L1 取页缺货 → EvictColdPagesLocked 批量淘汰(32页滞回),
  LFU+LRU+最小驻留期(1s), 两阶段兜底保证不会取不到页;
- 受害页走 PageOutTrieNode: 先试 L2(CPU 层 CAS 计数器 + HostCacheBudget
  共享预算), 满则 RotateCpuTierToDiskLocked 把 L2 最冷载荷轮转到 L3;
  L2 放不下且轮转不动才直接写 L3; L3 也写不下才真丢(hard-drop 有计数);
- L2 内存压力 → HostCacheBudget 回调 EvictCpuTierPayloads:
  先"只搬家不丢"(allowDrop=false), 再允许丢(allowDrop=true);
- 主动路径 DemoteColdTriePages 在调度期把冷页提前下沉, 避开前向关键路径。

**判定: 压力驱动的容量淘汰链路完整可用**(与历史 bug 修复后的版本一致,
testPrefixCacheTier 29/29 通过)。仍然缺的只有两样: 时间 TTL 和跨 root 孤儿清理
(代码注释明确承认孤儿无回收出口; 实测 ud-q5 root 7.86GB/gen-1 正常,
prefix-cache 下另有 qwen3.6 两个旧 root 共 0.6GB)。

#### C. "每次 load 自动恢复持久化 cache" —— 已经存在, 但有一个致命 bug

机制本来就在: apiserver 启动 → PrepareServerPersistentPrefixCache →
PreparePersistentPrefixCacheFromEnv(按模型指纹自动派生 cacheKey) →
LoadPersistentPrefixCacheGeneration 读 CURRENT 指向的 gen-N manifest →
MODEL_EXTRA(linear snapshot) 即时导入, paged trie/page 按 manager 懒恢复。

但生产日志连续多次 `restore skipped; cold start: persistent Qwen3.5 prefix
snapshot is incompatible`, gen-1(5.25GB) 成了只写不可读的死数据。

**静态取证定位根因**(对 gen-1 的 19 个 linear_snapshot 逐字节解析验证):
- 外层(manifest)FNV 与内层(snapshot 尾部)FNV 全部匹配;
- 结构遍历 19 个全部 OK, layers=64=block_cnt;
- 唯一失败点: mtpKey/mtpValue 的 ReadTensor 字节数校验。
  dims=[4,32896,256] fp16 紧凑应为 67,371,008B, 实际写出了 67,629,056B
  (= 行距 257 元素的 padded 布局)。WriteTensor 直接写 GetBytes()(含 stride
  padding), ReadTensor 用 prod(dims)*unitSize 校验 —— 写读不对称,
  **checkpoint 必然成功、restore 必然失败**。

**修复**(qwen3_5.cpp WriteTensor): strides 为紧凑布局时零拷贝直写;
非紧凑时先把 GetBytes() 镜像 append 进缓冲, 再按 N 维索引 gather 覆盖成紧凑
数据。编译通过; testPrefixCacheTier/testQwen35MultimodalPrefixPosition/
testPagedKvBudget/testPrefixCacheRouting 全绿。已提交 8ebfe39a。

遗留: 磁盘上现存的坏 gen-1 无法被新代码读取(格式仍是 padded), 下一次
checkpoint 会写出可读的 gen-2 并由 prune 收掉 gen-1; 或手动删除该 root 让其
重建。11:20 那次中断的 .staging-* (2.6GB) 属 RemoveStaleStaging 清理范围,
下次提交时自动清。
