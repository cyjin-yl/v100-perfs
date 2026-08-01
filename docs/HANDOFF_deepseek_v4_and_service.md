# Handoff — FastLLM 双主线：Qwen V100 服务硬化 + DeepSeek-V4 GGUF bring-up

**更新日期：** 2026-08-01（活文档，每完成一步回写「当前步骤 / 下一步」）
**唯一授权工作树：** `/home/ezra/.config/superpowers/worktrees/fastllm/qwen35-gguf-sm70-v2`
**只读旧树（禁止修改）：** `/home/ezra/.config/superpowers/worktrees/fastllm/qwen35-gguf-sm70`
**计划文件：** `/home/ezra/.claude/plans/mutable-conjuring-cake.md`

本文自包含。接手者无需读历史会话日志即可继续。旧的 07-28 `HANDOFF_fastllm.md` 记录的是 `1CatVLLM/fastllm` 目录的早期状态，与当前工作树不同，不要用它替代本文。

---

## 0. 硬性约束（逐字绑定，违反即事故）

- **生产端口 8002 冻结。** 禁止对 PID 543269 / 端口 8002 发信号、停止、重启、切换、探针、回滚或以任何方式访问。当前生产进程：
  - PID 543269，启动时间 2026-08-01 13:50:42，父 PID 3841462（`bun cli.js __omp_worker_daemon_broker`）。
  - 命令：`build-v2/apiserver -p .../Qwen3.6-27B-...-LOW-MTP-IQ4_XS.gguf -t 2 -l --atype float16 --kv_cache_dtype turbo3 --batch 2 --default_max_tokens 16384 --tokens 524288 --model_name qwen3.6-fastllm --port 8002 --device cuda`。
  - 门控：`FASTLLM_CUDA_SM70_FLASH_ATTN=0`、`FASTLLM_CUDA_SM70_PAGED_XQA=1`、`FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES=16`、`FASTLLM_QWEN35_ENABLE_MTP=0`、`FASTLLM_QWEN35_INTERLEAVE_LONG_PREFILL=1`、`FASTLLM_QWEN35_TURBO3_KV=1`。
  - **只有** Python server SIGSEGV 与生命周期验收门通过、且用户重新明确授权后，才允许任何 8002 动作。
- **DeepSeek 只用独立实验端口**，永不接触 8002。
- **下载只用 ModelScope**；cache 与最终文件都必须在 `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/models`。
- **绝不下载整个多量化仓库**，只按显式 allow-list 拉文件。
- **绝不** 硬编码、打印、复制、记录、暴露凭据或 token；**绝不** 读原始 shell history（此前一次搜索曾在输出里暴露过 ModelScope token，该 token 永不复述）。
- 保留用户改动与未跟踪 benchmark/debug artifact；不自动 commit/push。
- **保留已停止的 REAP partial 文件**，不续传、不删除（REAP K160 删除 37.5% experts，已废弃为执行方案）。
- **下载阻塞门**：Q2 payload 三个 split 在所有 preflight（adapter、expert allow-list、type-39、精确 offset、descriptor 生命周期、有界 Disk MoE）通过前禁止下载；DSpark shards 46–48 在 hybrid-loader fixture 通过前禁止下载。

## 1. 两条主线

### 主线 A — Qwen V100 服务硬化（Task #2）
修原生 C++ HTTP、request-owned `raw_prompt`、输出上限 16384、`ftllm profile` 一等生命周期、双 profile（Turbo3 capacity / FP8+MTP2 speed）安全切换与回滚、Python `ftllm server` 首次生成 SIGSEGV、SM70 cold 32K prefill 约 1250 tok/s。状态见 §4。

### 主线 B — DeepSeek-V4-Flash-0731 GGUF bring-up（Task #3/#4/#5/#7）
小量化 GGUF + Disk MoE/swap + selected-expert staging 先跑通 43 层 backbone；token 正确性与确定性是门槛，吞吐不是。先证 backbone，再用官方 safetensors shards 46–48 叠加 DSpark（3 个 MTP stage）。状态见 §4。

确认的 DeepSeek 架构：hidden 4096、64 attn head、1 KV head、head dim 512、q LoRA rank 1024、output LoRA rank 1024、output groups 8、256 routed experts、top-6、1 shared expert、MoE inter 2048、前 3 层 hash routing、`sqrtsoftplus`、`noaux_tc`、43 backbone 层、最大 context 1,048,576。A13B 指激活量不是存储量。GGUF 只含 backbone，不含 DSpark。

选定首个量化：**UD-Q2_K_XL**。选择依据是实际 tensor 格式而非 Q2_K 标签：routed gate/up 主用 **IQ2_XS**，routed down 主用 **IQ3_XXS**，另有 2 个 routed-down 为标准 **GGUF type 39 = MXFP4**；dense tensors 还混用 Q4_K/Q5_K/Q6_K/Q8_0/BF16/F32/I32。

## 2. 构建与测试（只读 / 离线可用）

构建目录：`build-v2`（`UNIT_TEST=ON`，Unix Makefiles）。测试可执行文件需要包内 CUDA/NCCL 库，用一次性 `LD_LIBRARY_PATH`，不改系统 loader：

```bash
LD_LIBRARY_PATH="/home/ezra/.local/lib/python3.13/site-packages/nvidia/cublas/lib:/home/ezra/.local/lib/python3.13/site-packages/nvidia/nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  build-v2/<test-binary>
```

聚焦 DeepSeek 测试目标：

```bash
cmake --build build-v2 --target testDeepSeekV4QuantizedDiskMoe testDeepSeekV4GGUFAdapter testDeepSeekV4ExpertAllowList -j2
```

注意：构建用 `-DNDEBUG` 会禁用 `assert`，测试目标已加 `-UNDEBUG`。不要跑完整 `regressionOps`（会占用当前 GPU，干扰 8002）。

## 3. 关键文件（绝对路径，工作树前缀 `WT=` 代指唯一授权工作树）

| 用途 | 路径 |
|---|---|
| 原生 HTTP 响应 helper | `WT/example/apiserver/http_response.h` |
| 输出上限 | `WT/example/apiserver/output_token_limit.h` |
| apiserver | `WT/example/apiserver/apiserver.cpp` |
| socket 测试 | `WT/test/apiserver/socketWriteTest.cpp` |
| GGUF 头（含 MXFP4/IQ 定义） | `WT/third_party/gguf/gguf.h` |
| 反量化（含 MXFP4/IQ2/IQ3） | `WT/third_party/gguf/ggml-dequantize.cpp` |
| 类型注册 | `WT/third_party/gguf/gguf.cpp` |
| DeepSeek4 adapter | `WT/third_party/gguf/gguf-adapter.cpp` |
| Data 类（生命周期焦点） | `WT/include/fastllm.h`、`WT/src/fastllm.cpp` |
| Disk MoE（缓存焦点） | `WT/src/devices/disk/diskdevice.cpp`、`WT/include/devices/disk/diskdevice.h` |
| 有界 expert 缓存 | `WT/include/devices/disk/disk_expert_cache.h` |
| 缓存 LRU 测试 | `WT/test/gguf/deepseek4DiskExpertCacheTest.cpp` |
| CPU MergeMOE | `WT/src/devices/cpu/cpudevice.cpp` |
| DeepSeekV4 模型（allow-list、BuildMoERoutingData） | `WT/src/models/deepseekv4.cpp`、`WT/include/models/deepseekv4.h` |
| adapter 测试 | `WT/test/gguf/deepseek4AdapterTest.cpp` |
| 量化 Disk MoE 测试 | `WT/test/gguf/deepseek4QuantizedDiskMoeTest.cpp` |
| allow-list 测试 | `WT/test/models/deepseek4ExpertAllowListTest.cpp` |
| profile 生命周期 | `WT/tools/fastllm_pytools/profile.py`、`deploy.py` |
| Qwen benchmark 报告（中文） | `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/v100-perfs/docs/fastllm_benchmark.md` |

## 4. 任务状态（活）

| # | 任务 | 状态 | 备注 |
|---|---|---|---|
| 1 | 审计工作树 | ✅ 完成 | |
| 2 | Qwen 服务/profile | 🔄 进行中 | 归 omp-1catvllm；8002 冻结 |
| 3 | DeepSeek adapter | ✅ 完成 | 41 规则、1328 源、34223 目标、exact-one |
| 4 | 量化 Disk MoE | 🔄 进行中 | MXFP4/IQ 已验证；剩 descriptor 生命周期 + 有界缓存 |
| 5 | DSpark | ⏸ 待定 | 被 #4 阻塞 |
| 6 | 验收与报告 | ⏸ 待定 | 被 #2/#4/#5 阻塞 |
| 7 | expert allow-list | ✅ 完成 | 严格解析、hash remap、pre-I/O 拒绝 |
| 8 | handoff/docs | 🔄 进行中 | 本文件 |
| 9 | Data descriptor 生命周期 | ✅ 完成 | scoped deleter，见 §6 |
| 10 | 有界 selected-expert 缓存 | ✅ 完成 | 线程安全 LRU，见 §6 |
| 11 | BuildMoERoutingData 测试 seam | 🔄 待办 | reviewer 建议 |

## 5. 当前步骤 / 下一步（每次回写这里）

**当前步骤：** Task #11 — BuildMoERoutingData 测试 seam（reviewer 建议）。

**下一步：**
1. Task #11 — 按 reviewer：把 file-local `BuildMoERoutingData`（`src/models/deepseekv4.cpp:2442`）用窄 test-only wrapper 暴露；表驱动测试覆盖 learned CPU、hash CPU（CUDA 禁用）、hash CUDA（`USE_CUDA` 下），每次 copy `expertIndex` 到 host 并 `ValidateDeepSeekV4AllowedExpertIndices`，可选喂入现有 disk fixture。
2. 重编译 + 重跑聚焦测试；`git diff --check`。
3. Task #4 完成后向 team-lead 汇报，并解除 #5 阻塞。

## 6. 已验证事实（截至本次更新）

- `testDeepSeekV4QuantizedDiskMoe` 通过：MXFP4/IQ2_XS/IQ3_XXS 已知向量、packed expert 精确 offset、1-byte 短读拒绝。
- `testDeepSeekV4GGUFAdapter` 通过：type 40（真正未知类型）在 stride 计算前被拒绝。
- `testDeepSeekV4ExpertAllowList` 通过：denied expert 在 disk I/O 前拒绝。
- `git diff --check` 通过。
- MXFP4 = GGUF type 39，`QK_MXFP4=32`，`block_mxfp4` 17 字节，E8M0-half 在指数 0/1 保留两个非正规特例，低 nibble→输出 0–15、高 nibble→输出 16–31。
- **descriptor 生命周期已修（Task #9）**：`diskdevice.cpp` 新增 file-local `ReleaseOwnedGGUFDescriptor` + `DiskTempWeightPtr`（RAII deleter）。四个泄漏点全部覆盖：`releaseOwnedWeights`（DiskMergeMOE 每调用）、DiskLinear 两处 `unique_ptr`、`DiskKimiK3RoutedExpertsOp::ownedWeights`。借用 descriptor（`RunDiskLinearChunk` 的栈 `ggmlChunk`）由 `isFake` 守卫，绝不释放。故意不改 `~Data`（`Data` 无自定义 `operator=`，隐式浅拷贝 `void* ggmlTensor` 会 double-free）。三测试重编重跑通过。
- **有界 selected-expert 缓存已落地（Task #10）**：新增公共头 `include/devices/disk/disk_expert_cache.h` + `diskdevice.cpp` 内的 `DiskExpertWeightCache`（线程安全 LRU，默认 8 GiB，`FASTLLM_DISK_EXPERT_CACHE_BYTES` 覆盖，0 禁用）。**只缓存 `DATA_GGUF_FORMAT` 权重**：`CudaSupportsDiskMoeWeight` 拒绝该类型，故 GGUF 权重永不进入非幂等 CUDA prepare（`CrossSwigluReorderWeightInPlace`），缓存副本在 prefill/decode 间恒有效；FP8/NVFP4/float 仍每次新载新释。key = `MakeDiskWeightCacheKey`（name+dt+gt+dims+首个非 scale payload 的 `file@offset`），跨 split 同名 expert 不冲突。`DiskMergeMOE::Run` 先按 key Lookup 命中（cache-owned），未命中才 disk read；GGUF 未命中经 `Insert`（超额单权重不缓存、原样返回由调用方持有；重复 key 释放冗余来件返回既有指针）。allow-list 校验仍在 disk I/O 前（§7 未变）。`releaseOwnedWeights` 简化为 `ReleaseDiskExpertWeight`。新增 `testDeepSeekV4DiskExpertCache`（key 同一性/跨 split、hit-miss-stats、LRU eviction+recency、oversized 不缓存、duplicate 释放来件、disabled）。四测试全过，`git diff --check` clean。

## 7. 未解决项（不阻塞 Task #4）

- WorkQueue mutex 饥饿（capacity loop 持锁 `continue`）。
- 原生 `/v1/models` 缺失。
- Python 首次生成 SIGSEGV（FillLLMInputs/memmove 附近）未解。
- profile 生命周期剩余硬化（provisional PID、strict name、stale/reused-PID no-killpg 测试、redaction 审计、TUI 直接 import）。
- V100 prefill 优化与最终中文报告。
- OMP 8002 两次替换事故说明未收到确认。
