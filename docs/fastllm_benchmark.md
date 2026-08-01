# FastLLM 在 V100 上运行 Qwen3.6 Fable Fusion 的性能报告

**更新日期：** 2026-08-01
**硬件：** NVIDIA Tesla V100-PCIE-32GB（SM70，32 GiB VRAM），64 GiB 系统内存
**软件：** Fedora 44、CUDA 12.9、GCC 16.1.1（系统）/ 12.4.0（conda）
**模型：** `Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-LOW-MTP-IQ4_XS.gguf`（64 个主干层 + 1 个 MTP 层）

## 结论摘要

FastLLM 已能在单块 V100-32GB 上完整运行这份 Qwen3.6 27B GGUF，包括 IQ4_XS 权重、混合 full-attention/GDN、FP8 E4M3 KV、FastLLM 原生 Turbo3 packed KV、SM70 原生 paged XQA 和 MTP。实测需要把目标拆成两个 profile：

- **FP8 速度 profile**：FP8 E4M3 KV、单请求 MTP2、多请求 plain batch、CUDA embedding、8,192-token prefill chunk；短 decode 为 C1 **50.85 tok/s**、C4 **81.13 tok/s aggregate**；
- **Turbo3 缓存/当前生产 profile**：full-attention K=`Q8_0_KV`、V=`TURBO3_KV`，MTP2、2,048-token prefill quantum、262,144-token GPU pool 与 `--batch 5`；每个请求的逻辑上限仍为 262,144 tokens；
- **Turbo3 双路容量档（已验证，非当前生产参数）**：MTP0、CPU embedding、524,288-token 共享池与 `--batch 2`；已完成单路 exact 256K，以及首 page 不同、不可共享 prefix 的双路各 exact 256K；
- **分层状态与 prefix cache**：pool 内请求可组成 resident CUDA batch；超池请求在安全 quantum 边界完整换出到 raw/zstd CPU snapshot，并可 spill 到 session-local 磁盘；page-aligned prefix trie 另有 GPU→CPU/NVMe 冷层、checksum 和实测成本门控；
- FP8 与 Turbo3 的 KV、embedding、chunk 和容量配置不同；旧 FP8 速度数据不能直接归因给 Turbo3，双路容量验收也不能替代当前 MTP2 单池的性能验收。

| 场景 | 配置 | 实测结果 | 结论 |
|---|---|---:|---|
| 短 decode，单请求 | MTP2 | 50.85 tok/s | 保留 MTP2 |
| 短 decode，四并发 | batched MTP2 | 47.57 tok/s aggregate | recurrent-state snapshot/rollback 开销过高 |
| 短 decode，四并发 | plain batch | 81.06 tok/s aggregate | V100 的底层 batch kernel 有效 |
| 短 decode，四并发 | 单请求 MTP2 + 多请求 plain batch | **81.13 tok/s aggregate** | 当前最优混合策略 |
| 32K 冷 prefill，单请求 | prefix cache 开、8K snapshot interval | TTFT 34.40s，约 954 prompt tok/s | 基本追平 cache-off 基线 |
| 32K 冷 prefill，单请求 | Turbo3 + MTP2、2K quantum，稳态两次 | TTFT 40.38–40.57s，约 807–811 prompt tok/s | 比旧 FP8 参考约慢 18%，仍需同二进制同 prompt A/B |
| FP8 共享池，四并发 | 4×约 52K prompt | 总 207,807 prompt tokens，峰值 29,150 MiB | 完整协议与容量验收通过 |
| 单请求 exact 256K | MTP9 + FP8 KV（历史矩阵） | E2E 745.47s，峰值 31,088 MiB | 容量通过，早于最终 XQA commit |
| 单请求 exact 256K | FastLLM 原生 Turbo3 容量 profile | E2E 890.21s，峰值 28,596 MiB | exact usage、SSE 与输出正确 |
| 双请求各 exact 256K | FastLLM 原生 Turbo3 容量 profile | batch wall 1,711.95s，峰值 28,798 MiB | 两路均 262,144 tokens，干净进程、独立 prefix |
| Turbo3 resident C2 | 2×约 32K prompt，262,144-token pool，2 runs | 两路 TTFT median 78.96/78.98s；0 disk spill | pool 内保持 resident，协议与重复输出校验通过 |
| resident decode 实验 | C2，各 128 completion | gate-on wall 106.14s；gate-off 89.39s | resident plain decode batch 为负收益，默认关闭 |
| Turbo3 CPU/disk swap L3 | 3×约 32K，32,768-token pool | 1 次增量 spill/restore；39,172,183 / 48,758,784 B；输出与单路 control 全匹配 | 超池完整状态轮转与 zstd/disk 恢复通过 |
| Turbo3 page-aligned prefix tier | 12,128-token prompt，16,384-token pool | cold 12.414s；GPU hit 2.009s；NVMe restore 2.840s | partial hit、压力淘汰、磁盘恢复、输出一致性通过 |

## 当前生产配置

生产链路为 Thinking Proxy `:8000` → FastLLM `:8002`。当前上线的是 Turbo3 缓存/速度折中 profile；已验证的双 256K 容量档保留为独立命名 profile，不与当前参数混用。

```bash
FASTLLM_QWEN35_TURBO3_KV=1 \
FASTLLM_QWEN35_ENABLE_MTP=2 \
FASTLLM_QWEN35_INTERLEAVE_LONG_PREFILL=1 \
FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES=16 \
FASTLLM_CUDA_SM70_PAGED_XQA=1 \
FASTLLM_CUDA_SM70_FLASH_ATTN=0 \
./apiserver \
  -p /path/to/Qwen3.6-27B-IQ4_XS-MTP.gguf \
  -t 2 -l --atype float16 --kv_cache_dtype turbo3 \
  --batch 5 --tokens 262144 --default_max_tokens 16384 \
  --model_name qwen3.6-fastllm \
  --port 8002 --device cuda
```

Turbo3 必须由 CLI 和环境变量双重显式启用；只传 `--kv_cache_dtype turbo3` 会被门控拒绝。262,144-token GPU pool 表示同时驻留的物理容量，不把每个请求的逻辑上下文截短到更小值。完整 CPU 后备状态、粗时间片换入换出、zstd snapshot 和 session-local prefix NVMe tier 已实现，但都由独立环境变量显式启用；不需要超池会话或冷 prefix 时不承担对应后台与主机内存成本。

FP8 速度 profile 仍保留为短上下文对照：FP8 E4M3、MTP2、`FASTLLM_QWEN35_BATCHED_MTP=0`、CUDA embedding、8K chunk、`--batch 5 --tokens 262144`。它的 50.85/81.13 tok/s 与 34.40s 32K TTFT 不能直接外推到当前 Turbo3 profile。

2026-07-31 最终探针确认：FastLLM `:8002` native 流式请求为 HTTP 200、0 malformed、1 个 `[DONE]`、usage 完整且无 marker 泄漏；经 Thinking Proxy `:8000` 的非流式与流式请求均为 HTTP 200，最终 content 为 `4`，reasoning boundary、usage、`finish_reason=stop` 正确，流式为 21 个 event、0 malformed、1 个 `[DONE]`。

## 2026-07-31 调度与稳定性改动

### apiserver batch 参数传播

此前 `--batch 5` 只限制 HTTP worker 的并发数，没有写入 `model->maxBatch`。Qwen3.5 模型 scheduler 因而看到默认值并退回单 lane；表面上请求并发进入，GPU 模型层仍基本串行。

现在 clamp 后的 batch 值同时设置：

- `workQueue.maxActivateQueryNumber`；
- `workQueue.model->maxBatch`。

这与 pytools 入口已有的 `model->maxBatch = batch` 语义一致。运行时 scheduler 因此得到 4 个安全 decode lane（单 GPU snapshot 上限为 4）。

### long-prefill 崩溃修复

最初的 interleave 路径在 `Split(fullInputIds, ...)` 后直接读取 `selectedInputIds.cpuData`。真实 V100 崩溃现场的指令为 `vmovss (%rsi)`，且 `rsi=0`：`Split` 产物可供 GPU 使用，但不保证存在 CPU buffer。

修复后：

- token ids 直接从 `FillLLMInputs` 的原始 CPU tensor 按 offset 读取；
- position ids 按最后一维逐行复制；
- input/position slice 都增加边界断言。

修复后的 32K C1、32K C2、208K C4 和约 240K 服务端压力轮次均未再出现 SIGSEGV。

### prefill 驻留数与 decode 批宽解耦

long-prefill 每个请求拥有独立的 KV、GDN recurrent state、MTP cache、cursor、ticket 和 page reservation。显式开启 interleave 后，prefill resident lane 可以在 page reservation 约束下独立于 batched MTP fast-path；每次 GPU forward 仍只执行一个 prefill quantum，decode 批宽也没有被不安全地放宽。

### 单请求 MTP、多请求 plain batch

V100 上 batched MTP 每步需要为大量 linear-attention 层和多个请求备份/恢复 recurrent state，还要执行 batched draft；这些复制与 rollback 成本超过了 target batch 的收益。

新增 `FASTLLM_QWEN35_BATCHED_MTP` 门控：

- 默认开启，保持兼容；
- 显式设为 `0/false/no/off` 时，多请求跳过 `Qwen35MTPBatchForward`，走已有 plain batched fallback；
- 单请求仍走原 `Qwen35MTPForward`，因此保留 MTP2 加速。

## 多请求 decode 吞吐

workload 为约 108–114 prompt tokens、每请求固定 256 completion tokens、`temperature=0`。每条请求均为 HTTP 200、`finish_reason=length`、0 malformed、1 个 `[DONE]`，usage 与目标 token 数一致。

| 路线 | C1 aggregate | C4 aggregate | C4/C1 | C4 单请求速度 |
|---|---:|---:|---:|---:|
| batched MTP2 | 49.22 tok/s | 47.57 tok/s | 0.97× | 约 11.9 tok/s |
| plain MTP0 | 35.88 tok/s | 81.06 tok/s | 2.26× | 约 20.28 tok/s |
| **混合：C1 MTP2 / C4 plain batch** | **50.85 tok/s** | **81.13 tok/s** | **1.60×** | **约 20.30 tok/s** |

混合路线的 C4 相对 batched MTP2 C4 提升 **70.5%**，同时没有牺牲单请求速度。机器结果：

- `benchmarks/fastllm/results/fastllm_mtp2_short_decode_batch_propagated_c1_c4.json`
- `benchmarks/fastllm/results/fastllm_hybrid_mtp2_plain_batch_short_decode_c1_c4.json`

## 冷 cache-miss prefill

Qwen3.5 linear-prefix snapshot 原默认间隔为 16 pages，即 2,048 tokens。它会把正常的 8,192-token prefill chunk 压成 2,048，并且每请求默认只保留 4 个 snapshot，覆盖范围只有约 8K。已有 A/B 表明这会显著拖慢完全冷 miss。

默认间隔现改为 64 pages，即 8,192 tokens；环境变量 `FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES` 仍可覆盖。prefix cache 仍然开启，并没有为了跑分而关闭。

| 路线 | 32K C1 TTFT | 冷 prefill 速率 | 峰值显存 |
|---|---:|---:|---:|
| snapshot interval 2K（cache-on） | 40.18s | 816 tok/s | 24,746 MiB |
| cache-off、8K chunk | 34.20s | 959 tok/s | 28,090 MiB |
| **cache-on、8K snapshot interval** | **34.40s** | **954 tok/s** | **28,092 MiB** |

TTFT 从 40.18s 降至 34.40s，改善 **14.4%**，并基本追平 cache-off 基线。机器结果：

- `benchmarks/fastllm/results/fastllm_mtp2_cold_prefill_c1_32k_http_native_fp8_interleave_8k.json`

## Turbo3 + MTP2 32K prefill 与公共 GDN 实验

同一 Turbo3+MTP2 服务在第一次模型冷触摸后，两次 32K 稳态请求的真实 TTFT 为 40.381s 和 40.574s；请求 prompt 均为 32,733 tokens，对应约 807–811 prompt tok/s。首次请求的 78.82s 包含一次性权重/内核/allocator 冷启动和 41-token decode，不能当作 Turbo3 稳态 prefill。

默认关闭的 `FASTLLM_QWEN35_PREFILL_PROFILE` 仅用于阶段归因。稳态每个 2K quantum 中，48 层 GDN 约 0.72s、MLP 约 1.25s；full attention 从短上下文约 0.10s 增至长上下文约 0.82s；packed append 约 8.5ms。因此 append 不是主要瓶颈，正式 wall time 也不使用 profile-on 数据。

`ChunkGatedDeltaRulePrefill` 已是 FastLLM 公共 operator。针对真实 `B1/H48/32×64/K128/V128/FP16` shape 做了四项通用 CUDA backend 实验，全部 env gate 默认关闭且不支持时保留原 cuBLAS fallback：

- SM70 FP16 fused H/O：数值在 `2e-3` 绝对/相对容差内通过，但为 12.58–12.80ms，原 cuBLAS 为 2.04–2.87ms，慢约 4.4–6.2 倍；
- per-thread stream async gather：bit-exact，但不同轮次收益方向相反，没有稳定收益；
- fused state/K preprocess：bit-exact，但没有跨轮稳定收益；
- per-thread/per-device persistent scratch：bit-exact，实测 3.78ms 对 baseline 2.87ms，为负收益。

生产 profile 不开启以上任何实验 gate。它们作为失败安全、可复现的通用层实验保留，防止后续重复走同一路线。机器结果：`benchmarks/fastllm/results/fastllm_sm70_chunk_gdn_prefill_experiments.json`。

## long-prefill interleave 的公平性边界

32K C2 在 batch 传播与 resident-lane 修复后得到：

- TTFT：80.27s / 80.61s；
- TTFT spread：0.34s；
- batch wall：87.57s；
- 峰值显存：24,862 MiB；
- 两请求均 HTTP 200、64 completion tokens、0 malformed、1 个 `[DONE]`。

旧 gate-off C2 的 TTFT 为 34.19s / 71.34s，呈明显阶梯；interleave 让两个长请求在 8K/2K quantum 边界轮转，消除了后排请求的长期饥饿。它不会减少单 GPU 上 cold prefill 的总 FLOPs，所以不能把公平性改善写成总吞吐提升。

机器结果：

- `benchmarks/fastllm/results/fastllm_mtp2_cold_prefill_c2_32k_http_native_fp8_interleave_batch5.json`

## FP8 共享池多并发容量

`--tokens 262144` 是整个服务的 token/KV 共享池，不是每请求上限。正式完整 C4 验收使用 4×约 52K 的独立 cold prompt：

| 指标 | 结果 |
|---|---:|
| prompt tokens 合计 | 207,807 |
| completion tokens 合计 | 256 |
| batch wall | 262.25s |
| TTFT 范围 | 225.94–239.73s |
| TTFT spread | 13.80s |
| 峰值显存 | 29,150 MiB |
| 协议 | 4× HTTP 200；usage 正确；0 malformed；每请求 1 个 `[DONE]` |

另一次 4×约 60K（约 240K prompt pool）压力轮次在服务日志中四请求全部 `Response client finish`，没有 OOM、SIGSEGV 或 CUDA error；但客户端执行器中止，未保留完整 usage/TTFT artifact，因此只作为容量上界佐证，不替代 208K 正式结果。

机器结果：

- `benchmarks/fastllm/results/fastllm_mtp2_fp8_208k_pool_c4_interleave_8k.json`

## FastLLM 原生 Turbo3 双 exact 256K

这条路线不使用 llama.cpp 后端。FastLLM 把 packed KV 作为 paged cache 的物理 dtype：16 个 full-attention 层的 K 使用 `Q8_0_KV`，V 使用 `TURBO3_KV`；48 个 linear-attention/GDN 层保留普通 FP16 recurrent state，因为它们是固定状态而不是随 token 增长的 paged KV。

head_dim=256 时，每个 K row 为 272 B，每个 V row 为 100 B。每 token 的 full-attention packed KV 为：

```text
16 layers × 4 KV heads × (272 + 100) bytes = 23,808 bytes/token
```

相同部分使用 FP16 K/V 时为 65,536 B/token，因此 packed/FP16 比例为 36.328%。524,288-token pool 的 packed KV 约为 11,904 MiB；相同容量的 FP16 KV 单是 full-attention 部分就需要 32,768 MiB。

| 指标 | 单路 exact 256K | 双路各 exact 256K |
|---|---:|---:|
| pool | 524,288 tokens / 4,096 pages | 524,288 tokens / 4,096 pages |
| 每路 usage | 261,888 prompt + 256 completion | 261,888 prompt + 256 completion |
| 每路总窗口 | 262,144 | 262,144 |
| E2E / batch wall | 890.21s | 1,711.95s |
| SSE | 258 JSON events、0 malformed、1 `[DONE]` | 每路 258 JSON events、0 malformed、1 `[DONE]` |
| finish | `length` | 两路均 `length` |
| 输出 | 从 4 连续递增，无 marker | 两路均从 4 连续递增到 68，无 marker |
| 采样峰值 | 28,596 MiB | 28,798 MiB |
| CUDA/OOM/SIG 错误 | 0 | 0 |

双路验证在一个全新 FastLLM 进程上直接并发发起，没有先跑校准或单路 exact。A prompt 使用单 token 单元 `" a"`，B 使用 `" b"`；两者固定开销都校准为 43 tokens，但从首个 128-token page 起就不同。因此 prefix trie 不可能通过共享旧 page “放水”，两个请求物理上必须各占 2,048 pages。

首版 packed attention 是正确性/容量实现：按 bounded chunk 将 Q8/Turbo3 K/V 解压到 FP16，再复用 cuBLAS attention；不会 materialize 整个 256K 窗口，也尚未做低比特直接计算 kernel。Turbo3 自动绕过当前不可 capture 的 CUDA graph 路径。本文不把它描述为 decode 吞吐优化。

容量边界也做过反例验证：525,312-token pool + MTP2 + CUDA embedding + 8K chunk 会在临时激活分配时 OOM；关闭 MTP/CUDA embedding 但仍用 525,312 pool + 8K chunk，也会在延迟的大权重 materialization 时因连续显存不足失败。最终稳定配置是精确 524,288-token pool、MTP0、CPU embedding 和 2K chunk。

机器结果：`benchmarks/fastllm/results/fastllm_turbo3_c1_c2_exact_256k.json`。

## Resident batch、CPU swap 与 prefix 三级缓存

这三条路径解决不同边界：

- resident scheduler 只在总 page reservation 不超过 GPU pool 时让多个请求同时驻留；单请求仍可用 MTP2，多请求 plain decode batch 由 `FASTLLM_QWEN35_RESIDENT_PLAIN_BATCH` 单独门控；
- `FASTLLM_CPU_REQUEST_SWAP=1` 在 committed prefill quantum 或安全 decode 边界保存完整请求状态；snapshot 包含 packed K/V、GDN recurrent/conv、MTP、token/history 和调度游标，恢复失败会回滚目标页；
- `FASTLLM_PREFIX_CACHE_CPU_TIER=1` 给 page-aligned prefix trie 增加 CPU/NVMe page-out；Qwen 的 linear/GDN/MTP snapshot 与 paged full-attention 页按同一 cached length 恢复。

### Resident 与 CPU/disk swap

262,144-token pool 的 C2 resident 验收跑两轮，共四条请求；两路 TTFT median 为 78.957s 和 78.979s，所有 HTTP/SSE/usage 校验通过，`cpu_request_swap_disk_spills` 增量为 0。128-completion decode A/B 显示 resident plain batch gate 开启后 wall 为 106.143s，关闭时为 89.394s，因此该实验保留但默认关闭。

32,768-token pool 的三路 32K L3 验收强制超池轮转。正式轮次相对 warm control 增加 1 次 disk spill 和 1 次 restore；zstd 将 48,758,784 B 降至 39,172,183 B（stored ratio 0.80339），压缩 0.0672s / 691.77 MiB/s，解压 0.0384s / 1,189.86 MiB/s。三路 content/reasoning hash、usage 和 finish reason 与逐路 control 全匹配，结束时 suspended=0。

### GPU→CPU/NVMe prefix tier

prefix tier 使用 token page 的完整 token 序列校验，不只依赖 hash；查询返回最长 page-aligned prefix。LRU/frequency 保护热页，只有命中数或 prefix 长度达到阈值的候选才 page-out。运行时记录实际 prefill token/s、disk read MiB/s 和 zstd decompress MiB/s，估算 restore 时间不优于重算时直接 miss。磁盘 extent 有 checksum、容量和最低剩余空间约束，并用进程独立目录与锁避免实例互相清理。

最终 16,384-token pool 验收：

| 指标 | 值 |
|---|---:|
| cold seed TTFT | 12.414s |
| GPU shared-prefix partial-hit TTFT | 2.009s |
| 压力淘汰后 NVMe restore TTFT | 2.840s |
| GPU / disk hits | 3,584 / 1,536 pages |
| disk write / read | 233,442,550 / 116,542,691 B |
| prefix zstd stored ratio | 0.79795 |
| prefix zstd compress | 0.427s，652.86 MiB/s |
| prefix zstd decompress | 0.117s，1,097.31 MiB/s |

seed 与 restore 的 content/reasoning SHA-256、usage、`finish_reason`、0 malformed 和单一 `[DONE]` 完全一致。早期两个负结果也保留：非 page-aligned 12,000-token fixture 无法命中；只探测 `kvCacheId=0` 在 hybrid Qwen 上拿不到 paged manager。修复后 fixture 对齐到 12,082 个重复单元（实际 prompt 12,128 tokens），运行时从可用 full-attention layer 选择 probe manager。

## 新功能统一可复现 benchmark

`benchmarks/fastllm/run_cache_benchmark.py` 统一三类容量/缓存验收；它调用同一 `feature_bench.py` HTTP/SSE runner，固定收集 `/props` before/after、请求与响应 SHA-256、usage、`finish_reason`、malformed event、`[DONE]`、TTFT、wall、平台和用户 metadata。

```bash
python benchmarks/fastllm/run_cache_benchmark.py \
  --mode resident --endpoint http://127.0.0.1:8002 \
  --output benchmarks/fastllm/results/resident.json

python benchmarks/fastllm/run_cache_benchmark.py \
  --mode l3 --endpoint http://127.0.0.1:8002 \
  --output benchmarks/fastllm/results/l3.json

python benchmarks/fastllm/run_cache_benchmark.py \
  --mode prefix --endpoint http://127.0.0.1:8002 \
  --output benchmarks/fastllm/results/prefix.json
```

| mode | 必需 profile | 自动验收 |
|---|---|---|
| `resident` | Turbo3、262,144-token pool、CPU swap enabled | C2×2 runs；重复输出一致；disk spill 增量为 0 |
| `l3` | Turbo3、32,768-token pool、CPU swap + zstd + disk | C3；先跑逐路 control；spill/restore/read/write 与 zstd counter 均增加；三路输出逐字段一致 |
| `prefix` | Turbo3、16,384-token pool、prefix CPU/disk tier | warmup 后依次 seed/shared/pressure/restore；GPU hit、disk write/read/hit、prefix zstd counter 增加；restore 等于 seed 且快于重算 |

runner 在 profile 不匹配时先失败，不产生貌似有效的错配置成绩。所有 prop gate 使用 counter delta，而不是绝对值，避免 warmup 污染。`test_fastllm_feature_bench.py` 与 `test_fastllm_cache_benchmark.py` 覆盖 fixture 展开、SSE 多行/分块解析、属性上下限、重复输出、control comparison 和 zstd 汇总公式。

## 历史 exact-window 长上下文矩阵

这些结果使用 C++ apiserver、IQ4_XS、FP16 activation、FastLLM 原生 paged attention 和 MTP9。每个请求为 `context - 256` prompt tokens + 256 completion tokens。

| 总窗口 | KV dtype | TTFT | E2E | 采样峰值 | 结果 |
|---:|---|---:|---:|---:|---|
| 128K | FP16 | 未可靠采集 | 318.76s | 28,049 MiB | HTTP 200，exact usage |
| 160K | FP16 | 344.53s | 381.13s | 30,529 MiB | HTTP 200，exact usage |
| 256K | FP16 | — | — | 31,642 MiB 后失败 | 申请额外 874,086,400-byte `lm_head.weight` 时 OOM |
| 256K | FP8 E4M3 | 未可靠采集 | 745.47s | 31,088 MiB | HTTP 200，exact usage |

最终生产 FP8 binary 还完成了 128K/160K/256K 三档 64-token 输出矩阵：E2E 分别 240.35/330.78/695.89s，峰值 27,968/28,220/28,436 MiB；全部 HTTP 200、66 个有效 event、0 malformed、1 个 `[DONE]`。

机器结果：

- `benchmarks/fastllm/results/fastllm_mtp9_128k_exact.json`
- `benchmarks/fastllm/results/fastllm_mtp9_160k_exact.json`
- `benchmarks/fastllm/results/fastllm_mtp9_256k_exact.json`
- `benchmarks/fastllm/results/fastllm_mtp9_fp8_longctx_final.json`

历史 exact-window artifact 早于最终 `b693ad8` XQA commit，因此它们验证 native stack 容量，不构成最终 XQA 的长上下文端到端 A/B。

## V100 attention 路线与来源边界

| 路线 | 当前状态 | 来源 | 已验证范围 |
|---|---|---|---|
| FastLLM 原生 paged attention | 既有 baseline | `38b4b8b8`、`adb30fd0`、`bd9a24ad` | exact-window 矩阵、paged metadata/CUDA graph 回归 |
| FlashInfer-SM70 | **未进入当前 worktree/PR** | 没有独立 backend/planner/Volta prefill kernel | 不声明 FastLLM 性能 |
| Flash-Attention-V100 风格 SM70 XQA | `b693ad8` 落地 | 面向 page128/Q24/KV4/D256/GQA6 的 FastLLM-native 重写 | qLen1 decode microbenchmark、回归与短 smoke |

真实 V100 prefill 日志为 `Native paged prefill uses chunked cublas attention`。因此本文不把 FlashInfer-SM70 或 Flash-Attention-V100 BM32/`ALL_P` prefill 的性能归因给 FastLLM。

### SM70 paged XQA 单层 microbenchmark

这是单层 attention microbenchmark，不是端到端模型吞吐。配置为 batch1、FP16 Q/K/V、page128、Q24/KV4、D256，5 次 warmup + 30 次 CUDA event 计时。

| KV tokens | 原 per-Q-head native | SM70 XQA | 加速 |
|---:|---:|---:|---:|
| 8,192 | 0.397 ms | 0.179 ms | **2.22×** |
| 32,768 | 1.739 ms | 0.516 ms | **3.37×** |
| 131,072 | 5.774 ms | 1.433 ms | **4.03×** |

机器结果：`benchmarks/fastllm/results/fastllm_sm70_paged_xqa.json`。

## qLen2..10 SM70 FP8 paged-prefill 实验

`FASTLLM_CUDA_SM70_FLASH_ATTN=1` 是本地实验性 Volta WMMA 路线，不是 FlashInfer。单层 microbenchmark 在小 KV 上有收益，但端到端输出与 native 不等价，steady decode 也退化：

| Agents | Native steady aggregate | 实验路线 steady aggregate | Native avg decode | 实验路线 avg decode |
|---:|---:|---:|---:|---:|
| 1 | 10.00 tok/s | 4.64 tok/s | 11.24 tok/s | 8.81 tok/s |
| 2 | 10.28 tok/s | 7.41 tok/s | 11.93 tok/s | 9.50 tok/s |
| 5 | 10.84 tok/s | 8.41 tok/s | 12.57 tok/s | 9.81 tok/s |

固定 greedy prompt 在 native 重复运行中稳定，但实验路线产生不同 token 数和 SHA-256，因此该路线保持默认关闭，生产显式设 `FASTLLM_CUDA_SM70_FLASH_ATTN=0`。

## TurboQuant 256K 共享池参考

TurboQuant 的“256K”同样是总 262,144-token pool，不是每请求 256K。实际 KV 为 K=`q8_0`、V=`turbo4`。并发 C1/C2/C4/C5 aggregate 分别为：

| 并发 | aggregate throughput |
|---:|---:|
| 1 | 20.714 tok/s |
| 2 | 29.732 tok/s |
| 4 | 35.689 tok/s |
| 5 | 38.652 tok/s |

这些结果的引擎、KV 格式、prompt reuse 和 workload 与 FastLLM cold exact-window 不同，只作为部署容量参考，不做直接 A/B。

## 主要补丁与功能状态

### Qwen3.5/3.6 GGUF 与 runtime

- `qwen35` → `qwen3_5` architecture alias；
- 65 blocks 纠正为 64 trunk + 1 MTP；
- `attention.key_length`、`ssm.*` 与 linear-attention metadata 映射；
- V-head tiled/grouped 逆置换与 `FASTLLM_QWEN35_GGUF_VHEAD_TILED=0` override；
- GGUF magic dispatch、KV metadata、TP out-proj scheme 与 MTP profile。

### SM70 IQ4_XS 与 attention

- SM70 DP4A IQ4_XS MMQ，窄 projection 回退 legacy MMVQ；
- 原生 split-KV/chunked-cuBLAS attention；
- page128/Q24/KV4/D256/GQA6 qLen1 XQA；
- unsupported dtype/layout/shape 和 qLen>1 安全回退。

### apiserver 与 OpenAI 协议

- tokenizer buffer UAF 修复；
- 无 body GET/readiness 不再死锁；
- full socket write 与正确 chunked/SSE 终止；
- 多行 SSE event；
- request-driven stop string/array 与多-token stop sequence；
- reasoning boundary、stream/non-stream tool call；
- abort handle generation 防复用竞态；
- proxy-side 原 GGUF Jinja2 rendering。

## 已知限制

- GGUF 内嵌模板使用 `{% macro %}`，FastLLM 自带 Jinja 子集不能直接解析；生产由 Thinking Proxy 用完整 Jinja2 渲染后发送 `raw_prompt=true`。
- 当前模型只有 1 个 MTP layer；`nextn_predict_layers>1` 会拒绝加载。
- 仅验证单块 V100，没有验证多 GPU tensor parallel。
- FP16 KV 的 exact 256K 在 32GB 上确定 OOM；FP8 E4M3 与 FastLLM 原生 Turbo3 packed KV 都已完成 exact 256K。
- Turbo3 当前只为 CUDA Qwen3.5/3.6、head_dim=256 显式门控；普通 FP16/BF16/FP8、非 Qwen 模型和默认配置不走 packed 路径。
- 当前 Turbo3 attention 会即时解压到 FP16/cuBLAS，容量收益已经验证，但不能据此宣称低比特直接计算的性能收益。
- 双 256K 生产 profile 关闭 MTP 与 CUDA embedding，并使用 2K prefill quantum；需要短上下文吞吐时应切回独立的 FP8 速度 profile。
- prefix NVMe store 是服务进程生命周期内的冷层，正常退出会删除 session 目录；当前不提供跨重启持久索引。
- XQA 是 qLen1 decode specialization，不是通用 prefill kernel。
- interleave 改善公平性，不减少 cold prefill 的总计算量。
- 生产混合策略是 V100 实测选择；其他 GPU 应重新做 batched MTP 与 plain batch A/B。

## 验证状态

以下验证均通过：

- `apiserver` 与 `regressionOps` 编译；
- `FASTLLM_REGRESSION_ONLY=turbo3_kv ./build/regressionOps`，覆盖 packed 行字节、门控、异构 per-layer dtype、跨页 append/gather/dequantize、C1/C2 layout 和 legacy attention；
- `testApiServerSocket` 与 usage accounting；
- Turbo3 + MTP2 短请求端到端二轮 speculative validation；
- Turbo3 单路 exact 256K、干净进程双路各 exact 256K；
- 32K C1/C2、短 decode C1/C4、FP8 208K C4 端到端 HTTP/SSE/usage；
- resident C2 两轮在池内无 spill；三路 32K/32K pool 触发完整 zstd+disk swap 并与单路 control 输出一致；
- page-aligned shared-prefix GPU hit、压力淘汰、NVMe restore、动态成本指标和 prefix zstd 压缩/解压计数；
- 8000→8002 最终生产非流式与流式探针。

仅因硬件条件跳过双 GPU 相关测试；Triton chunk GDN 测试在未启用对应环境时按预期跳过。

## 协议修复与部署生命周期（v2 分支增量）

在 `fix/qwen35-gguf-sm70-v2` 分支上，基于 `origin/master@26a65a44` 干净 rebase 后，新增以下协议与部署能力：

### 可配置缺省输出 token 上限

- C++ 与 Python server 均支持 `--default_max_tokens` / `--default-max-tokens`（默认 16384）。
- 客户端显式正整数 `max_tokens` 严格尊重；零、负数、浮点、字符串、布尔值在 launch 前拒绝。
- `finish_reason=length` 基于实际生效上限。
- 新增 `example/apiserver/output_token_limit.h` 与 `test/api/test_output_token_limit.py`。

### HTTP 协议完整性

- C++ apiserver 新增 `/health`、`/version`、`/props` GET 端点（JSON 响应）。
- 未知路由返回 JSON 404；方法不匹配返回 JSON 405 + `Allow` 头。
- 新增 `example/apiserver/http_response.h` 统一固定响应构造器。
- Python server 新增 `/health`、`/version`、`/props` 端点。
- Thinking Proxy 对所有 `httpx.TransportError` 返回结构化 503。

### 原生 raw_prompt 路径修复

- 发现 Python server 的 `raw_prompt` 路径使用 `launch_response_llm_model`（`add_special_tokens=false`），
  而 C++ apiserver 与 Python 普通聊天路径均使用 `add_special_tokens=true`。
  在 Qwen3.5 Turbo3 + SM70 GPU 调度器下，`add_special_tokens=false` 导致首次 prefill 的
  `FillLLMInputs` 触发 host 端 `__memmove` 段错误。
- 修复：新增 `launch_raw_prompt_llm_model` C 导出，使用 `make_config(..., true)` 与 C++ apiserver 对齐。
- Python `llm.py` raw_prompt 分支改调新函数；顺手修正 `launch_response_str_llm_model.argtype` → `argtypes` 拼写错误。
- 新增 `test/api/test_raw_prompt.py` 覆盖编码一次、绕过模板、stop token 传递、无效请求拒绝。

### Profile 生命周期 CLI

- `ftllm profile list|validate|start|stop|status` 子命令。
- 命名 profile（`qwen36-v100-capacity`、`qwen36-v100-speed`）互斥门控。
- 切换时自动健康验证与失败回滚。
- 锁文件 + PID 日志管理。

### 当前阻塞项

以下任务仍需要独占 V100 的服务切换窗口：

- Python `ftllm raw_prompt` 修复的 GPU 端到端验收；
- capacity 命名 profile 的单路/双路 exact 256K 重验。

Turbo3+MTP2 32K baseline、prefill 分层归因、公共 GDN operator 实验、resident batch、完整 CPU/disk swap、zstd 冷层和 prefix NVMe tier 均已完成，不再属于阻塞项。


## 机器结果索引

- `benchmarks/fastllm/results/fastllm_hybrid_mtp2_plain_batch_short_decode_c1_c4.json`
- `benchmarks/fastllm/results/fastllm_mtp2_short_decode_batch_propagated_c1_c4.json`
- `benchmarks/fastllm/results/fastllm_mtp2_cold_prefill_c1_32k_http_native_fp8_interleave_8k.json`
- `benchmarks/fastllm/results/fastllm_mtp2_cold_prefill_c2_32k_http_native_fp8_interleave_batch5.json`
- `benchmarks/fastllm/results/fastllm_mtp2_fp8_208k_pool_c4_interleave_8k.json`
- `benchmarks/fastllm/results/fastllm_mtp9_fp8_longctx_final.json`
- `benchmarks/fastllm/results/fastllm_sm70_paged_xqa.json`
- `benchmarks/fastllm/results/fastllm_turbo3_c1_c2_exact_256k.json`
- `benchmarks/fastllm/results/fastllm_sm70_chunk_gdn_prefill_experiments.json`
- `benchmarks/fastllm/results/fastllm_turbo3_mtp2_resident_batch_acceptance.json`
- `benchmarks/fastllm/results/fastllm_turbo3_mtp2_resident_plain_batch_off_steady.json`
- `benchmarks/fastllm/results/fastllm_turbo3_mtp2_resident_decode_batch_fixed_hot.json`
- `benchmarks/fastllm/results/fastllm_turbo3_mtp2_cpu_swap_l3_acceptance.json`
- `benchmarks/fastllm/results/fastllm_turbo3_prefix_tier_16k_acceptance.json`
- `benchmarks/fastllm/results/fastllm_turbo3_prefix_tier_16k_unaligned_negative.json`
- `benchmarks/fastllm/results/fastllm_turbo3_prefix_tier_16k_probe_layer_negative.json`
- `benchmarks/turboquant/results/llama_tq4_256k_agents{1,2,4,5}.json`

## 构建说明

参考 `docs/HANDOFF_fastllm.md` 的构建章节。关键前提：

- 支持 SM70 的 CUDA toolkit；
- 与 CUDA 版本兼容的 host compiler；
- 启用 NUMA 时安装 `libnuma` 开发头文件；
- 大型 CUDA translation unit 建议限制构建并行度，避免系统 swap 抖动。
