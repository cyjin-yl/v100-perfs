# Qwen3.8-27B 短 prefill（qLen 512–2048）优化计划

日期：2026-08-22 · 生产档：`q38-PROD-cyber-iq4xs-imatrix-mtp2-sm70.env`（iq4xs Cyber，turbo3 KV，MTP=2，batch=4，262K ctx）
目标：生产 qLen 512–2048 的 prefill 延迟与吞吐。所有结论有实测/代码依据，标注 `[INFERENCE]` 的除外。

---

## 一、基线（实测，两轮一致）

| prompt_tok | TTFT | prefill tok/s |
|---:|---:|---:|
| ~505 | 0.70–0.78 s | 643–723 |
| ~1003 | 1.15–1.17 s | 860–870 |
| ~2050 | 2.14–2.32 s | 890–953 |

交叉印证：引擎在线测得 `prefix_cache_recompute_tokens_per_second` ≈ 695–809，与上表同量级。
注意：**agent 流量的续写型 prefill 大多命中前缀缓存**（历史：80K 续写 114s → 0.8s），本基线是冷前缀的最坏情况。

## 二、当前内核路由图（qLen 维度，全部来自代码 + /props 实测计数）

```
GEMM 侧 (gguf.cu):
  n = 每次前向的 token 数
  n ≤ 8            → gguf.mmvq          (量化点积, decode 用)
  9 ≤ n ≤ 12(实测) → gguf.sm70_iq4xs_mmq (DP4A MMQ, 仅 IQ4_XS; 上限 n≤64)
  n > 8            → gguf.dequant_fp16_gemm (每 chunk 反量化权重→cuBLAS HGEMM)
                     ★ 生产 120.8M tokens 走这条路 = prefill 主路径
                     ★ 每 chunk 固定开销: 权重展开 154.87ms/chunk (占 M=512 chunk 时间 31%)

Attention 侧 (paged attention native 入口):
  qoLen ≤ 4        → attn.sm70_turbo_xqa      (融合 XQA, K=q8_0+V=turbo3 直读) — decode
  4 < qoLen ≤ 32   → attn.sm70_turbo_prefill  (融合打包 prefill; 生产计数为 0!)
                      准入条件: FASTLLM_CUDA_SM70_TURBO_PREFILL_MAX_Q 默认 32,
                      且要求 pageIndicesGpuIn 非空等
  qoLen ≥ 64 且 ALL_HEAD_BATCH 开 → attn.sm70_turbo_all_head_gqa (打包 all-head batch GQA;
                      生产 5,040 次, n∈[159,1024], 4.99M tokens)
  其余全部         → attn.native_fallback     (物化 fp16 K/V + chunked cuBLAS)
                      ★ 生产 360,258 次 / 6.57M tokens —— 短 prefill 的实际落点
```

关键调度事实：
- `defaultChunkedPrefillSize=8192`，但 `Qwen3_5Model::GetChunkedPrefillSize()` 对含线性注意力层的 Qwen3.8 **二次收窄到 1024**（`FASTLLM_QWEN35_PREFILL_CHUNK_CAP` 默认 1024；目的是让线性层 snapshot 边界落在 chunk 内）。生产未显式传 `--chunked_prefill_size` ⇒ 生效 chunk=1024。
- 因此 qLen≤1024 单次整体前向；qLen∈(1024,2048] 走 long-prefill 分子（1024+余量）。
- `GetBatchedPrefillTokenLimit()` 未显式设置时 = max(chunk, defaultChunk)=…（qwen3_5.cpp:9668）多请求合批上限。

## 三、瓶颈定性（为什么短 prefill 只有 ~700–950 tok/s）

P-A【GEMM 效率·主因】n=512–1024 时 cuBLAS HGEMM 处于 46–90 TFLOP/s 区间（峰值 112），且每 chunk 支付 154.87ms 权重反量化固定开销。已有实测：M=512 时 966µs/tok vs M=2048 时 654µs/tok（`docs/prefill_exact_port_investigation.md §4.4`）——大 chunk 摊薄固定开销。

P-B【chunk 过小】1024-token chunk 让 2K prefill 变成 2 个串行 forward，每次都重付权重展开固定开销 + per-forward 调度开销。investigation §6.B 预估 chunk 提到 2048 有 +21.8%@8K / +17.1%@32K 收益 [INFERENCE：该预估基于长上下文场景，对 2K 场景方向一致但幅度未测]。

P-C【attention 物化税】native_fallback 每行成本 ≈372B 读量化 K/V + 1024B 写 fp16 + cuBLAS 回读（BATCH_GQA 已开，读一遍而非 6 遍）。qoLen≥64 且 turbo KV 时走 all_head_gqa 打包路（省物化），但 **qoLen∈[33,63] 与非打包形状仍付全税**。生产 360K 次 fallback 中大量是短 prefill 的逐层调用。

P-D【融合 prefill 内核空转】`sm70_turbo_prefill`（直读打包 KV、零物化）准入上限默认 qoLen≤32，生产计数恒 0。它正是为消除 P-C 设计的，但 32 的上限把整个 prefill 区间让给了 fallback。

P-E【缓存准入门槛】L2/L3 录取要求 `MIN_TOKENS=8192` 默认值——512–2048 的新会话前缀**永远不够格**进 L2/L3，压力下被硬丢。对短会话工作负载这是命中率天花板。

## 四、实验矩阵（按预期收益/风险排序；全部 env/参数级，无需改码）

| # | 实验 | 操作 | 预期 | 风险 |
|---|---|---|---|---|
| E1 | 提高 prefill chunk 到 2048 | profile 加 `--chunked_prefill_size 2048`（或 `FASTLLM_QWEN35_PREFILL_CHUNK_CAP=2048`） | 2K prefill 少一次固定开销摊销，~10–20% TTFT ↓ [INFERENCE] | attention scratch 显存 ↑；线性层 snapshot 边界仍对齐（2048 是 512 interval 的倍数）；需盯 OOM |
| E2 | 放开融合 prefill 上限 | `FASTLLM_CUDA_SM70_TURBO_PREFILL_MAX_Q=1024` | qoLen 33–1024 全部改走零物化融合核，fallback 计数应大跌 | 数值等价未在 qoLen>32 验证过；须先跑逐字抄写/工具名探针比对输出 SHA |
| E3 | 降低 L2/L3 录取门槛 | `FASTLLM_PREFIX_CACHE_MIN_TOKENS=512`、`FASTLLM_PREFIX_CACHE_MIN_HITS=1`（已是默认 1） | 短会话第二轮起可命中 L2，TTFT 显著↓ | L2 占用↑（16GiB cap 内）；观察 evict_cpu_to_disk |
| E4 | A/B 注意力路由对照 | 分别关 `FASTLLM_CUDA_PAGED_CUBLAS_BATCH_GQA` / ALL_HEAD_BATCH 复测三档 | 量化 BATCH_GQA 对 qoLen≥64 的真实贡献 | 无，纯观测 |

E2 是唯一需要数值验证的门；其余三个都是纯旋钮。每个实验跑完用 `/props` 的 kernel_routes + prefix_cache_stats 对账。

## 五、双 V100 (TP2) 与傲腾评估结论

### PCIe 双卡 V100 32G TP=2
- 权重切分：IQ4_XS 16.5GB ÷ 2 = 8.3GB/卡，每卡腾出 ~24GB → 页池可翻倍以上，262K ctx 的 KV 压力解除。
- 通信代价：TP 每层 2 次 all-reduce（attn+mlp out），PCIe 3.0 x16 单向 ~12GB/s，27B 每层 hidden 8192×fp16=16MB × 48 层 × 2 次 ≈ 1.5GB/step 载体 [INFERENCE]，decode 小消息延迟主导，NCCL over P2P 不可用时退化明显。V100 无 NVLink（PCIe 型号）⇒ decode tok/s 预计只小幅提升甚至持平，但 **prefill 受益于算力翻倍**（cuBLAS 是计算受限段），且 batch 并发容量近似翻倍。
- 结论：TP2 主要买「并发容量 + prefill 吞吐」，不买单流 decode 延迟。profile 应预留：`CUDA_VISIBLE_DEVICES=0,1`、TP 相关 env、页池预算按 2 卡重算、前缀缓存 manager 按 gpuId 分池现状确认。

### 傲腾（PMem/NVMe 企业盘）作 prefill cache 盘
- 数学：一页 128tok KV ≈ 372KB（q8_0+turbo3）。恢复成本 = 介质读 + zstd 解压(~370µs) + H2D(~50µs)。DDR4≈0.5ms、企业 NVMe(P5800X 级)≈0.55ms、PMem≈0.6ms —— 三者都在 0.5–0.7ms 档，而重算一页要 ~160ms（800tok/s）。**介质不是瓶颈，解压+互斥才是**；PMem 相比 DDR4 无速度收益，只有容量收益。
- 当前部署的真实问题：所有磁盘层目录在外置 7200rpm HDD 上（seek 12ms，8 并发恢复 263ms > 重算 199ms ⇒ L3 在并发下净亏损）。**把 L3/swap 目录迁到任意企业 NVMe 即可翻转这个不等式**，收益远大于买傲腾。
- 建议：不买傲腾。若已有 PMem，用作 L2 容量扩展（HostCacheBudget 16GiB 已与 suspend-cache 共享，存在挤压证据）；否则一块二手企业 NVMe（如 P1600X/P5800X）+ 目录迁移是性价比最高的动作。

## 六、缓存过期释放（用户问询的答案，已单独汇报过）

- L1/L2/L3 全部容量驱动淘汰（LFU/LRU+驻留滞回 / HostCacheBudget 回调 / 配额 prune keep-2），**无时间 TTL**。
- 缺口：跨模型孤儿 root 不记账（换模型后旧 root 永久留存，本次手动清了 36G）；无年龄过期。
- 修复方向：启动+checkpoint 后扫 DISK_DIR 下全部 root 做 mtime-LRU 清理；加 `FASTLLM_PREFIX_CACHE_MAX_AGE_HOURS`；/props 暴露 per-root 字节。

## 七、附带发现的生产缺陷（待修）

1. 僵尸流槽位：客户端断开后 proxy 侧 stream 不回收（inflight=[1660s~0tok] 挂死，占满 PROXY_STREAM_SLOTS=4 直到超时）。
2. 错误掩蔽：上游客户端用错误模型名请求时后端回 404，proxy 把它吞成 200 OK 空响应（yielded=0），客户端陷入 2 秒一次的重试风暴。

## 八、后续

- UpstreamReview scout 因上游配额未完成，配额窗口（08-28 UTC 21:05 后）重派。
- SM70 参考实现对比 scout 空返回，其核心结论已由主线自查覆盖（Volta 无 mma.sync fp16 tile、融合核是我们自己的最优路径、参考价值主要在 llama.cpp 的跨边界语法语义与 xgrammar 的 rollback——均已落地）。
