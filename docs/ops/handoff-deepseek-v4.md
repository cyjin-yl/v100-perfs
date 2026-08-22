# Handoff — FastLLM 双主线：Qwen V100 服务硬化 + DeepSeek-V4 GGUF bring-up

**更新日期：** 2026-08-09（活文档，每完成一步回写「当前步骤 / 下一步」）
**唯一授权工作树：** `/home/ezra/.config/superpowers/worktrees/fastllm/qwen35-gguf-sm70-v2`
**只读旧树（禁止修改）：** `/home/ezra/.config/superpowers/worktrees/fastllm/qwen35-gguf-sm70`
**计划文件：** `/home/ezra/.claude/plans/mutable-conjuring-cake.md`

本文自包含。接手者无需读历史会话日志即可继续。旧的 07-28 `HANDOFF_fastllm.md` 记录的是 `1CatVLLM/fastllm` 目录的早期状态，与当前工作树不同，不要用它替代本文。

---

## 0. 硬性约束（逐字绑定，违反即事故）

- **生产端口 8002 冻结。** 禁止对当前 8002 服务进程发信号、停止、重启、切换、探针、回滚或以任何方式访问。记录的原进程 PID 543269 已不存在；2026-08-09 清点发现 8002 现由 omp 会话的新进程服务（被动 ps 清点，未做任何接触）：
  - PID 1765590（约 2026-08-09 上午开始，清点时 etime 43 分钟），父 PID 3341914。
  - 命令：`/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/fastllm/build/apiserver --path .../ThinkingCap-Qwen3.6-27B-Q4_K_M.gguf --mmproj .../mmproj-ThinkingCap-Qwen3.6-27B-f16.gguf --threads 2 --atype float16 --kv_cache_dtype turbo3 --batch 5 --default_max_tokens 16384 --tokens 65536 --model_name qwen3.6-fastllm --port 8002 --device cuda`。
  - 注意二进制来自主目录 `1CatVLLM/fastllm/build/`（不是工作树 build-v2），模型换成 Q4_K_M + mmproj。
  - **只有** Python server SIGSEGV 与生命周期验收门通过、且用户重新明确授权后，才允许任何 8002 动作。冻结语义绑定端口与服务本身，不因 PID 更换而解除。
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
cmake --build build-v2 --target testDeepSeekV4QuantizedDiskMoe testDeepSeekV4GGUFAdapter testDeepSeekV4ExpertAllowList testDeepSeekV4DiskExpertCache testDeepSeekV4RouteSeam -j2
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
| routing seam 测试 | `WT/test/models/deepseek4RouteSeamTest.cpp` |
| CPU MergeMOE | `WT/src/devices/cpu/cpudevice.cpp` |
| DeepSeekV4 模型（allow-list、BuildMoERoutingData） | `WT/src/models/deepseekv4.cpp`、`WT/include/models/deepseekv4.h` |
| adapter 测试 | `WT/test/gguf/deepseek4AdapterTest.cpp` |
| 量化 Disk MoE 测试 | `WT/test/gguf/deepseek4QuantizedDiskMoeTest.cpp` |
| allow-list 测试 | `WT/test/models/deepseek4ExpertAllowListTest.cpp` |
| profile 生命周期 | `WT/tools/fastllm_pytools/profile.py`、`deploy.py` |
| Qwen benchmark 报告（中文） | `/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/v100-perfs/docs/benchmarks/fastllm-qwen36-legacy.md` |

## 4. 任务状态（活）

| # | 任务 | 状态 | 备注 |
|---|---|---|---|
| 1 | 审计工作树 | ✅ 完成 | |
| 2 | Qwen 服务/profile | 🔄 进行中 | 归 omp-1catvllm；8002 冻结 |
| 3 | DeepSeek adapter | ✅ 完成 | 41 规则、1328 源、34223 目标、exact-one |
| 4 | 量化 Disk MoE | ✅ 完成 | MXFP4/IQ、descriptor 生命周期、有界缓存、routing seam 全部落地 |
| 5 | DSpark | 🔄 进行中 | loader+config 注入+**MXFP4/MXFP8 dtype 路由**完成并已推送至 origin（e41023d9）；13 项 loader 测试+5 项回归全过；下一步 = backbone bring-up 与 DSpark runtime，见 §5/§6 |
| 6 | 验收与报告 | ⏸ 待定 | 被 #2/#4/#5 阻塞 |
| 7 | expert allow-list | ✅ 完成 | 严格解析、hash remap、pre-I/O 拒绝 |
| 8 | handoff/docs | 🔄 进行中 | 本文件 |
| 9 | Data descriptor 生命周期 | ✅ 完成 | scoped deleter，见 §6 |
| 10 | 有界 selected-expert 缓存 | ✅ 完成 | 线程安全 LRU，见 §6 |
| 11 | BuildMoERoutingData 测试 seam | ✅ 完成 | learned/hash CPU 已过；hash CUDA 留 opt-in，见 §6 |
| 12 | UD-Q2_K_XL split 2/3 下载校验 | ✅ 完成 | 三 split 大小精确；completed header 与预下载 range 对 1328 个 tensor 的 name/type/dims/offset 逐项一致；见 §6 与 `deepseek_q2_split_verification.json` |
| 13 | 合并 upstream/master（内置 DSpark） | ✅ 完成 | e70fb7f3 已推送 origin；旧 worktree 已清理，见 §6 |

## 5. 当前步骤 / 下一步（每次回写这里）

**当前步骤：** DeepSeek backbone bring-up 前置工作。**已完成**：① shards 46–48 下载校验通过（§6）；② Q2 三个 split 下载及逐 tensor 校验完成：split2/3 的 completed header 与预下载 header-range 对 1328 个 tensor 的 name/type/dims/offset 逐项一致，688+640、零重复/缺失/多余、padded offset 与文件尾精确（§6）；③ `InjectDeepSeekV4DSparkConfig` 落地并接入 InitParams 前；④ **hybrid loader MXFP4/MXFP8 dtype 路由已完成**：I8 打包 FP4→`NVFP4`（eager = weight+尾随原始 E8M0 inline 布局；disk-lazy = compact scale disk part）、F8_E4M3+F8_E8M0→`FP8_E4M3`+F32 scales（复用 `CreateBuffer` 既有 E8M0→F32 转换），两条声明路径（顺序 + GGUF 并行）packed dim×2；新增 4 项测试（MXFP4/MXFP8 × eager/disk），13 项 loader 测试 + 5 项 DeepSeek 回归全过（见 §6「DSpark dtype 路由」）；⑤ 927ae97e/e41023d9 已推送 `origin/fix/qwen35-gguf-sm70-v2`。

**下一步：**
1. **DeepSeek backbone bring-up（DSpark off，先行）**：实验端口 + 独立 state/log，Q2 GGUF + Disk MoE + allow-list，最保守配置；tokenizer/单 token/固定 greedy 确定性/参考对比。吞吐不作门槛。
2. **DSpark runtime**：按官方 `inference/model.py`（target layers {40,41,42} hidden 捕获 mean→concat [b,s,12288]、stage-0 main_proj/main_norm、三个 DSparkBlock attention+MoE、forward_head Markov/confidence、verification 状态机）；loader 已能装载真实 MXFP4/MXFP8 权重；先 batch-1 greedy 正确路径；最终门槛 = DSpark-on/off greedy token 序列完全一致。
3. Task #2 剩余项归 omp-1catvllm（Python SIGSEGV、profile 硬化）；**本会话不触碰 8002**。2026-08-09 本次续接检查时 8002 无 listener、8000 `/health` 为 `loading/ready=false`；由 Task #2 所有者处理，DeepSeek 仅使用独立实验端口。

## 6. 已验证事实（截至本次更新）

- `testDeepSeekV4QuantizedDiskMoe` 通过：MXFP4/IQ2_XS/IQ3_XXS 已知向量、packed expert 精确 offset、1-byte 短读拒绝。
- `testDeepSeekV4GGUFAdapter` 通过：type 40（真正未知类型）在 stride 计算前被拒绝。
- `testDeepSeekV4ExpertAllowList` 通过：denied expert 在 disk I/O 前拒绝。
- `git diff --check` 通过。
- MXFP4 = GGUF type 39，`QK_MXFP4=32`，`block_mxfp4` 17 字节，E8M0-half 在指数 0/1 保留两个非正规特例，低 nibble→输出 0–15、高 nibble→输出 16–31。
- **descriptor 生命周期已修（Task #9）**：`diskdevice.cpp` 新增 file-local `ReleaseOwnedGGUFDescriptor` + `DiskTempWeightPtr`（RAII deleter）。四个泄漏点全部覆盖：`releaseOwnedWeights`（DiskMergeMOE 每调用）、DiskLinear 两处 `unique_ptr`、`DiskKimiK3RoutedExpertsOp::ownedWeights`。借用 descriptor（`RunDiskLinearChunk` 的栈 `ggmlChunk`）由 `isFake` 守卫，绝不释放。故意不改 `~Data`（`Data` 无自定义 `operator=`，隐式浅拷贝 `void* ggmlTensor` 会 double-free）。三测试重编重跑通过。
- **有界 selected-expert 缓存已落地（Task #10）**：新增公共头 `include/devices/disk/disk_expert_cache.h` + `diskdevice.cpp` 内的 `DiskExpertWeightCache`（线程安全 LRU，默认 8 GiB，`FASTLLM_DISK_EXPERT_CACHE_BYTES` 覆盖，0 禁用）。**只缓存 `DATA_GGUF_FORMAT` 权重**：`CudaSupportsDiskMoeWeight` 拒绝该类型，故 GGUF 权重永不进入非幂等 CUDA prepare（`CrossSwigluReorderWeightInPlace`），缓存副本在 prefill/decode 间恒有效；FP8/NVFP4/float 仍每次新载新释。key = `MakeDiskWeightCacheKey`（name+dt+gt+dims+首个非 scale payload 的 `file@offset`），跨 split 同名 expert 不冲突。`DiskMergeMOE::Run` 先按 key Lookup 命中（cache-owned），未命中才 disk read；GGUF 未命中经 `Insert`（超额单权重不缓存、原样返回由调用方持有；重复 key 释放冗余来件返回既有指针）。allow-list 校验仍在 disk I/O 前（§7 未变）。`releaseOwnedWeights` 简化为 `ReleaseDiskExpertWeight`。新增 `testDeepSeekV4DiskExpertCache`（key 同一性/跨 split、hit-miss-stats、LRU eviction+recency、oversized 不缓存、duplicate 释放来件、disabled）。四测试全过，`git diff --check` clean。
- **BuildMoERoutingData 测试 seam 已落地（Task #11，reviewer 建议）**：file-local（匿名 namespace）`BuildMoERoutingData` 经窄 forwarder `DeepSeekV4BuildMoERoutingDataForTesting`（`deepseekv4.h` 声明、`deepseekv4.cpp` 中 `ValidateDeepSeekV4AllowedExpertIndices` 之后定义）暴露。新增 `testDeepSeekV4RouteSeam`：**learned CPU**（无 tid2eid，denied expert 3 的 logit=10 最高，policy 掩蔽后 top-2 = {2,1}，softmax scores 0.665/0.245）与 **hash CPU**（`FASTLLM_DSV4_DISABLE_CUDA_ROUTE=1`，remapped tid2eid 行直接公布 {2,0}）默认运行；每次 copy `expertIndex` 到 host 并过 `ValidateDeepSeekV4AllowedExpertIndices`。**hash CUDA** 分支编译进 `USE_CUDA` 但默认跳过（`FASTLLM_DSV4_TEST_CUDA_ROUTE=1` 才运行）——默认运行绝不在冻结生产 GPU 上创建 CUDA 计算。关键坑：`Data` 拷贝构造是深拷贝而隐式 `operator=` 是浅拷贝，测试里向 `WeightMap::weight` 填权重必须 `map[key]` 取引用后 in-place `Resize/Allocate/memcpy`，不能 `map[key] = local`（会悬挂）。五测试全过。
- **upstream 已实现完整内置 DSpark（483b3594，已核实）**：CLI `--dspark N` → env `FASTLLM_DSPARK_TOKENS`；硬编码 `dsparkLayers=3`、target layers `{40,41,42}`；从 config dict 读 `dspark_noise_token_id/dspark_markov_rank/dspark_block_size`；`FASTLLM_DSPARK_CONFIDENCE_THRESHOLD` ∈ [0,1]；启用时 `canDoConcurrentForward=false`。**但 upstream GGUF adapter 对 mtp 零支持**（`third_party/gguf/gguf-adapter.cpp` grep mtp = 0）→ GGUF backbone + safetensors mtp 的 hybrid loader 仍是 Task #5 缺口。upstream linearNames 仍含过时的 `mtp.*.e_proj/h_proj`（官方 0731 namespace 实际不存在，需按 §6 官方清单修正）。
- **合并冲突解决要点（Task #13）**：`MergeMOE/MergeMOEBlock` 尾部参数序 = `…, swigluLimit, deepSeekV4Mode, allowedExpertMask, pairedReduceInput`（我方 mask 在前、upstream pairedReduceInput 在后），全调用点已审计一致。qwen3_5.cpp 保住 HEAD-only 特性：`FASTLLM_CUDA_SM70_GDN_PRECORE` 包装（新增 `!logicalRagged` 门控 + precore 路径强制 `fuseOutputDecayMask=false`，因 ragged/fused 变体晚于 precore 设计）、TURBO3_KV CUDA-graph 排除、`GetKVCacheDataTypes(i)`、`hasCombinedQkvzProjection` 双路 GDN split；并入 upstream：logicalRagged/packedRagged GDN（`FastllmCudaMappedGdnKkt`、varlen prefill）、`directOutputQk`（opt-in）、CUDA serving warmup 重构、graph-safe singleRow 视图、多 handle GPU token handoff（`forcedGpuTokenHandOffHandles` 向量取代 HEAD 单 handle，`forceDecode` 条件同步改写）、`gpuTokenHandoffAllRunnable`、`dictCV.notify_all()`。8/2 合并曾静默丢失 5 个 testDeepSeekV4* CMake 目标与 `-UNDEBUG` 块，已恢复并补回 UNIT_TEST 的 `endif()`。
- **Task #13 收尾事实（2026-08-09）**：合并提交 e70fb7f3 已推送 `origin/fix/qwen35-gguf-sm70-v2`（推送区间 f1856225..e70fb7f3）。构建通过，5 个 DeepSeek 测试（adapter/allow-list/quantized-disk-moe/expert-cache/route-seam）全过，`git diff --check` clean。**fastdiv.cuh 保留 HEAD 自包含实现**：upstream 版改用 `cuda::fast_mod_div`（libcu++，CCCL 3.0 / CUDA 13+ 才有；本机 conda 工具链为 CUDA 12.9，`cuda/cmath` 无该符号，编译报 6 错），HEAD 版（milakov magic-number 实现）API 等价（默认构造、`operator unsigned int`、`divmod`、`operator/`、`operator%`）且已在 SM70 路径验证；全树唯一其他 `fast_mod_div` 引用在 `trtllm/fmha/kernelParams.h`（SM90+/trtllm 门控路径，SM70 构建不编译）。
- **worktree 清理事实（2026-08-09）**：旧 worktree `qwen35-gguf-sm70` 已 `git worktree remove`（移除前工作树 0 改动）。注意 `git branch --merged` 事实确认**未通过**——旧分支 813c2e06 的 13 个提交以不同 SHA 重做进 v2，非字面祖先；删除依据是分支已推送 `origin/fix/qwen35-gguf-sm70`（历史在远端，无丢失）且用户授权清理。本地分支 `fix/qwen35-gguf-sm70` 保留。主目录 `1CatVLLM/fastllm`（main, faac12c4）**不可清理**：当前 8002 生产进程（PID 1765590）正从它的 `build/apiserver` 运行。
- **生产进程更换（2026-08-09 被动清点）**：原记录的 PID 543269 已不存在；8002 现由 PID 1765590 服务（约 2026-08-09 上午起），二进制 `1CatVLLM/fastllm/build/apiserver`，模型 ThinkingCap-Qwen3.6-27B-Q4_K_M.gguf + mmproj，`--batch 5 --tokens 65536`。未读取其 environ（遵守冻结字面）。更换应为 omp 会话 Task #2 动作，未收到事故说明。
- **DSpark hybrid loader 已完成（Task #5 第一阶段，563d1b57 已推送，2026-08-09）**：
  - 结构：`PlanDeepSeekV4DSparkShards`（index 或单文件，只收 mtp.*，backbone 条目跳过、单文件混入 backbone 拒绝、缺 shard 拒绝、空计划拒绝）+ `ImportDeepSeekV4DSparkWeights`（顺序路径）；生产 GGUF 路径在并行装载循环内复用同一 `ImportDeepSeekV4DSparkTensor` 核心，env 门控 = `FASTLLM_DSPARK_MODEL_PATH` + `FASTLLM_DSPARK_TOKENS>0`。支持 BF16/F16/F32/F8_E4M3+scale 与 disk-lazy metadata。
  - 严格命名 seam：新增 `basellm::IsRecognizedWeightName`（默认宽松，不回归既有路径）；`DeepSeekV4Model` override 按官方 0731 namespace 封闭校验 mtp.*（stage/expert 数字解析、commonSuffixes 集合、stage-0 专属 main_proj/main_norm、末 stage 专属 norm/HC/Markov/confidence heads）。**弃用 GetTensorMap 判别**（basellm 版映射一切名称，不健全）。过时 `mtp.*.e_proj/h_proj` 已从 linearNames 删除。
  - `DeepSeekV4Model` 构造函数固定 `num_experts=256`/`num_experts_per_tok=6`（basellm 两成员无类内初始化器，InitParams 前读取原是 UB）。
  - util.py 新增 `--dspark_checkpoint_path`：与 `--dspark N` 联合校验、与 `--speculative_draft_model_path` 互斥、目录与 index/单文件存在性检查后设 `FASTLLM_DSPARK_MODEL_PATH`。deploy.py `extra_args` 直通，profile 无需 schema 变更。
  - 测试 `testDeepSeekV4DsparkHybridLoader`（9 项全过）：分片计划过滤 backbone、backbone-only/缺 shard/混入 backbone/空目录拒绝、BF16 字节精确导入、22 接受/17 拒绝命名矩阵、未识别名称与碰撞拒绝、FP8 标量 scale 配对（blockK=1/blockM=rows、raw bytes 不变、scale 不成为独立权重）、缺 scale 拒绝、Disk MoE 懒加载 metadata（手工 moeLinears+merge rule+specialWeights+moeDeviceMap 复现选路链，不依赖 InitParams）。
  - **关键坑（fixture 调试发现）**：本仓 json11 fork 中 `ll_value()` 仅 `JsonLL` 实现，`JsonDouble`/`JsonInt` 落到默认 `return 0LL`（json11.cpp:351）；`dump(double)` 输出 `%.12f` 定点格式。fixture writer 推 `(double)` offset → 回读为 0 → `bytes=0` → fread 读 0 字节、buffer 残留堆垃圾。修复 = fixture 推 `(long long)`。真实 safetensors offset 恒为整数、parser 整数走 `atoll`→JsonLL，上游加载路径不受影响；但**任何向 json11 推数值再取 ll_value 的新代码必须用 long long**。
  - 1 维权重 `CalcWeightSum` 的 `dims[1]` 越界读与上游主 safetensors loader（model.cpp:4563）行为一致，不另行处理。
  - 推送状态：本机 TLS 对 github.com 偶发 `unexpected eof`；2026-08-09 本次续接以 `git -c http.version=HTTP/1.1 push` 成功推送 927ae97e/e41023d9，远端 `origin/fix/qwen35-gguf-sm70-v2` 已到 e41023d9。
  - **gate.bias 命名补丁（927ae97e，已推送）**：真实 index 含 `mtp.{0,1,2}.ffn.gate.bias`（backbone MoE 层 `layers.{3..42}.ffn.gate.bias` 同理有 40 个），封闭命名集合原缺 → 会拒绝真实 tensor。已补进 `IsRecognizedWeightName` commonSuffixes 与测试矩阵（接受 3 / 拒绝 `.extra` 后缀）。**bias 不进 linearNames**（linearNames 只喂 GetWeightType 的 LINEAR glob 匹配；runtime 已在 deepseekv4.cpp:3606/3638 读 `prefix + ".gate.bias"`）。
- **官方 index/config 事实（下载前已核实，2026-08-09）**：`model.safetensors.index.json` 共 72,317 tensors；mtp.* 仅出现在 model-0004{6,7,8}（1568/1565/1572，合计 4,705），与 backbone shards 1–45 零重叠。顶层 `config.json`：`n_routed_experts=256`、`num_experts_per_tok=6`、`num_hidden_layers=43`、vocab 129280、`dspark_block_size=5`、`dspark_markov_rank=256`、`dspark_noise_token_id=128799`、`dspark_target_layer_ids=[40,41,42]`、`num_nextn_predict_layers=1`（legacy 误导项，实际 3 MTP 层）、`model_type=deepseek_v4`。REAP 目录 `inference/config.json` 的 `n_routed_experts=160` 为 REAP 剪枝专用，backbone 实为 256，不采信。
- **官方 DSpark 参考语义（`inference/model.py` 961 行已通读，runtime 实现基准）**：
  - `Transformer.forward`：embed → unsqueeze(2).repeat(hc_mult=4) → 43 Blocks；在 target layers {40,41,42} 捕获 `h.mean(dim=2)` → `main_hidden = cat(..., dim=-1)` → [b,s,12288]；收尾 `hc_head(hc_head_fn[4,16384], scale[1], base[4])` + `head(norm(h))`。
  - `forward_spec`：`h, main_x = mtp[0].forward_embed(main_hidden, input_ids)` → 逐层 `h = layer(h, start_pos, input_ids, main_x)`；`start_pos==0` 只做 KV 初始化直接返回；否则 `mtp[-1].forward_head(h, input_ids)` → output_ids [b, block_size+1]、logits、confidence。
  - `DSparkBlock.forward_embed`：`main_x = main_norm(main_proj(main_hidden))`（main_proj 12288→4096）；draft ids = full([b,5], 128799) 且 `[:,0]=input_ids`；`x = embed(ids).unsqueeze(2).repeat(1,1,4,1)`。
  - `DSparkBlock.forward`：start_pos>0 走完整 Block（HC pre/post + attn + MoE）；start_pos==0 只 `self.attn(x, 0, main_x)` 建 KV（`main_kv = kv_norm(wkv(main_x))`，rope 作用于 [-rd:]，非 rope 段 act_quant fp8-sim，按 `cutoff=seqlen%win` 分段写入 window cache）。
  - `DSparkAttention` decode：q = wq_b(q_norm(wq_a(x)))，64 head×512 head_dim；`q *= rsqrt(mean(q²)+eps)`；rope q[-rd:]；`topk_idxs = cat([arange(min(128, start_pos+1)), 128+arange(5)])`；`kv_cache[start_pos%128] = main_kv` 后 `kv = cat([kv_cache, kv])`；`sparse_attn(q, kv, attn_sink, topk_idxs, 512^-0.5)`；输出做**逆 rope** `apply_rotary_emb(o[...,-rd:], freqs, True)`；`o.view(b,s,8 groups,-1)` 与 `wo_a.weight.view(8,1024,-1)` 做 `einsum("bsgd,grd->bsgr")` → `wo_b(flatten)`。
  - `forward_head`：`hc_head` 后 `head(norm(x), full_logits=True)` [b,5,vocab]；output_ids[:,0]=input_ids；i∈0..4 逐步 `logits_bias, markov_embed = markov_head(output_ids[:,i])`（markov_w1 embed vocab→256，markov_w2 [vocab,256]）→ `logits[:,i] += logits_bias` → `output_ids[:,i+1] = sample(logits[:,i])`；`confidence = proj(cat([x, stacked_markov_embeds], -1).float()).squeeze(-1)`（proj 4352→1，fp32）。
  - HC 机制：mix_hc=(2+4)*4=24 行、hc_dim=4*4096=16384；hc_pre：flatten(2).float() → `rsqrt(mean(x²)+eps)` → `mixes = F.linear(x, hc_fn)*rsqrt` → `hc_split_sinkhorn(mixes, scale[3], base[24], 4, 20 iters, eps)` → pre/post/comb；`y = sum(pre.unsqueeze(-1)*x, dim=2)`。hc_post 对称复原。hc_head：`pre = sigmoid(mixes*hc_scale + hc_base) + hc_eps` → `y = sum(pre*x)`。
  - Gate：`scores = linear(x.float(), weight.float())`；sqrtsoftplus = `F.softplus(scores).sqrt()`；**bias 只平移选择分数**，`weights = original_scores.gather(1, indices)`，非 softmax、按 sum 归一、`×route_scale(1.5)`。hash 层（前 3 层）用 tid2eid 且无 bias。
  - Expert SwiGLU：gate/up 在 float32；swiglu_limit=10（`up=clamp(up,-10,10)`、`gate=clamp(gate, max=10)`）；`silu(gate)*up`；shared expert 单独累加。
  - `sample()`：temperature==0 → argmax（greedy）；否则 Gumbel-max。
- **DSpark 官方 checkpoint 量化格式（2026-08-09 对真实 safetensors header + inference/model.py 逐 tensor 核实）**——这是 hybrid loader runtime 阶段的硬阻塞，loader 现有 asserts 会拒绝真实权重：
  - dtype 分布（4705 mtp.*）：**I8=2304**（expert w1/w2/w3）、**F8_E4M3=25**（dense/attn/shared/main_proj）、**F8_E8M0=2329**（全部 .scale）、**BF16=20**、**F32=27**。
  - **expert routed = MXFP4**：存为 "I8" `[out, in//2]`（2 个 fp4 e2m1 打包进 1 byte），scale "F8_E8M0" `[out, in//32]`（沿 K block 32）。例 `mtp.0.ffn.experts.0.w1.weight` I8 (2048,2048)+scale (2048,128) → 逻辑 [2048,4096]，4096/32=128。**不是 int8**——`CreateBufferWithScale` 里 `isPackedFp4 = (dtype=="I8"||"U8")` 即此意。
  - **dense/attn/shared/main_proj = MXFP8**：`F8_E4M3 [out,in]`，scale `F8_E8M0 [ceil(out/128), ceil(in/128)]`（block 128×128）。例 `wq_a` F8_E4M3 (1024,4096)+scale (8,32)=(1024/128,4096/128)。
  - **BF16/F32 无 scale**：`ffn.gate.weight`(BF16)、`ffn.gate.bias`(F32)、各 norm(BF16)、`attn.attn_sink`(F32)、`hc_*`(F32)、`markov_head.markov_w1/w2.weight`(BF16)、`confidence_head.proj.weight`(BF16→runtime 提升 FP32)。
  - 参考语义（model.py `Linear`/`linear`）：按 weight dtype 分派 `fp4_gemm`/`fp8_gemm`/`F.linear`；激活经 `act_quant`（block 128、`scale_fmt="ue8m0"`、power-of-2 E8M0 scale）；FP4 block 32、FP8 block 128；scale 均 `float8_e8m0fnu`。kernel.py：`block_quant` FP8 block_size=128、FP4 block_size=32、`fast_round_scale`=pow2。
  - **loader 缺口清单**：`DeepSeekV4DSparkSourceDataType` 不映射 "I8"（直接 ErrorInFastLLM）；eager 路径 `AssertInFastLLM(oriDataType == FP8_E4M3)` 拒绝 I8 expert；`AssertInFastLLM(scale.dtype == "F32"||"BF16")` 拒绝 F8_E8M0；disk-lazy 路径 `AssertInFastLLM(diskDataType == FP8_E4M3)` 同理。FastLLM 已有 `NVFP4_BLOCK_32_E8M0`(=1009) 与 FP8_E4M3+E8M0 的 `CreateBufferWithScale` 机制，需把 mtp 权重正确路由过去（或 eager 反量化到 BF16），实现须逐 block 对照参考。
  - stage-0 专属张量实为 3 个：`main_proj.weight`+`main_proj.scale`+`main_norm.weight`（早前记 2 个，漏了 main_proj.scale；scale 在识别前被跳过，命名矩阵不受影响）。
- **DSpark dtype 路由已完成（2026-08-09，上一节阻塞项解除）**——真实 MXFP4/MXFP8 mtp 权重全部落入 FastLLM 既有能力，未新增量化格式：
  - 映射：`DeepSeekV4DSparkSourceDataType` I8/U8→`NVFP4`；expert routed（I8+F8_E8M0，block 32）eager 走 `CreateBufferWithScale(NVFP4)`（buffer = packed weight + 尾随原始 E8M0 bytes，`GetBytes`/`MallocSpace` 的 NVFP4 分支天然含 scale，`GetNVFP4ScaleData` 消费），disk-lazy 走 `SetDiskWeightMeta` compact scale part（第二 `DiskWeightPart`，`isScalePart`，diskdevice 既有支持）；dense（F8_E4M3+F8_E8M0 2D 128×128）eager/disk 均先经 `SafeTensorItem::CreateBuffer` 既有 F8_E8M0→F32（`FP8E8M0ToFloat`=2^(v-127)，torch e8m0fnu 语义）转 F32 scalesBuffer/常驻 scales，再走 FP8_E4M3 块状 scale 既有路径（blockK/blockM 由 shape 整除推出，真实 dense 恰为 128/128）。两条声明路径（`ImportDeepSeekV4DSparkWeights` 顺序 + GGUF 并行循环）对 packed FP4 做 `dims.back() *= 2` 逻辑维展开。
  - runtime 执行侧已存在：CPU `DeepSeekV4MoeLinearTaskStorage` nvfp4/fp8/bf16 任务分派（`GetNVFP4ScaleData` + `LinearNVFP4_Base_Run` AVX2/通用 fallback）、diskdevice compact NVFP4 DiskLinear。**本机 CPU 仅 AVX2**（无 AVX512/BF16/AMX），走 AVX2/generic 路径；V100 SM70 无 fp4/fp8 硬件，CUDA 侧行为待 bring-up 实测（可能回落 CPU）。
  - E8M0 边界字节核查：真实 shards 抽样 151 MB scale bytes（25 个 dense scale 全量 + expert scale 每块前 64 KB）**0 个 0x00、0 个 0xFF** → `NVFP4E8M0ScaleToFloat(0)=2^-126` 与 torch `2^-127` 的语义差异对本 checkpoint 无影响。
  - 测试：新增 `TestMxfp4EagerImport`/`TestMxfp8EagerImport`/`TestMxfp4DiskLazy`/`TestMxfp8DiskLazy`（inline 字节精确、E8M0 0x80→2.0 换算、blockK/blockM、compact scale part 元数据、`.scale` 与 `.weight_scale` 两种配对命名）；合计 13 项 loader 测试 + 5 项 DeepSeek 回归（adapter/allow-list/quantized-disk-moe/expert-cache/route-seam）全过。
  - 运行测试二进制需 `LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama/mlx_cuda_v13`（系统 ldconfig 只有 CUDA 13；二进制 NEEDED libcublas.so.12/libnccl.so.2）。
- **UD-Q2_K_XL split 逐 tensor 校验完成（Task #12，2026-08-09）**：split1 SHA-256=`de65c5bb660817b95cc281bf54935f9d1bcc3b22535f2eae63b35a14c7c5724b`、大小 5,257,664；split2/3 大小 49,437,013,568 / 47,390,237,120。completed split2/3 header 与下载前保存的 8 MiB header-range 对每个 tensor 的 `(name,dims,type,offset)` 完全一致且 descriptor 顺序相同；目录 688+640=1328，跨 split 零重叠、对 canonical 1328 清单零缺失零多余；GGUF data start 默认 32-byte alignment，逐 tensor padded offset progression 与最终文件尾精确。类型直方图：F32 662、Q8_0 318、Q5_K 127、IQ2_XS 84、Q6_K 45、BF16 43、IQ3_XXS 43、I32 3、MXFP4 2、Q4_K 1；MXFP4 仅 `blk.26/42.ffn_down_exps.weight`。machine-readable 证据：`v100-perfs/deepseek_q2_split_verification.json`（含 header-range SHA-256、验证脚本 SHA-256、file sizes；ModelScope `.msc` 仅提供 revision `b323ace6efd127442ae02b80d671c9ed95aca9a7`，没有内容 hash，未伪称远端 hash 已核对）。

## 7. 未解决项（不阻塞 Task #4）

- WorkQueue mutex 饥饿（capacity loop 持锁 `continue`）。
- 原生 `/v1/models` 缺失。
- Python 首次生成 SIGSEGV（FillLLMInputs/memmove 附近）未解。
- profile 生命周期剩余硬化（provisional PID、strict name、stale/reused-PID no-killpg 测试、redaction 审计、TUI 直接 import）。
- V100 prefill 优化与最终中文报告。
- OMP 8002 两次替换事故说明未收到确认。
