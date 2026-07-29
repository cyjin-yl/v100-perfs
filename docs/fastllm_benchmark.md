# FastLLM on V100 — Qwen3.6 Fable Fusion Performance

**Date:** 2026-07-28  
**Hardware:** NVIDIA V100 (SM70, 32GB VRAM, 64GB system RAM)
**Software:** Fedora 44, GCC 16.1.1 (system), GCC 12.4.0 (conda), CUDA 12.9  
**Model:** Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-LOW-MTP-IQ4_XS.gguf (IQ4_XS, 64 trunk layers + 1 MTP head)  

## Outcome

FastLLM now **runs full inference** (trunk + MTP9) on this model with the upstream-PR build — see §Patches. Exact-window requests completed at 128K and 160K with FP16 KV, and at 256K with FP8 E4M3 KV. MTP9 remained active through all three runs. Those exact-window artifacts validate FastLLM's pre-existing V100 native paged-attention stack, SM70 IQ4_XS MMQ, and the Qwen3.5 transposed-recurrent GDN kernels. The later `b693ad8` branch tip adds a Flash-Attention-V100-inspired exact-shape SM70 paged XQA specialization for `qLen=1` decode; its post-commit evidence is the focused/full regression, kernel microbenchmark, and short real-model smoke described below, not a rerun of the complete exact-window matrix.

## What works

1. **Model load** — all 65 blocks + tokenizer + IQ4_XS dequant → GPU in ~2m30s.
2. **`/generate` raw prompt** — HTTP 200, correct output, streaming and non-stream OpenAI SSE.
3. **SM70 IQ4_XS MMQ** — auto-selected for `k>=128`, F16 activations, DP4A accumulation. Fallback to legacy MMVQ for narrow projections.
4. **MTP9 speculative decode** — 9 drafts/step, exact acceptance, single-GPU copy-validation path.
5. **GDN (linear attention)** — `cuda-transposed-recurrent` with QKVZ+BA fused layout.
6. **V100 native paged attention + SM70 XQA** — FastLLM's existing native split-KV fallback predates this work. `b693ad8` adds a Qwen3.6 GQA6 XQA specialization that adapts the KV-head reuse strategy exercised in Flash-Attention-V100 to FastLLM's page-128 layout. Exact FP16 `page128/Q24/KV4/D256/GQA6/qLen1` decode uses XQA by default; prefill remains the existing chunked-cuBLAS native path.

## Known limitations

- **Chat template:** The embedded GGUF Jinja template uses `{% macro %}`, which FastLLM's Jinja parser does not support. Use raw `/generate` with pre-formatted ChatML, or deploy the Thinking Proxy with proxy-side Jinja2 rendering.
- **MTP layer count:** This runtime supports the model's single MTP layer and rejects `nextn_predict_layers>1` at load time.
- **Multi-GPU TP:** Single V100 only — not validated with tensor parallelism.
- **Long-context capacity:** FP16 KV completes at 128K and 160K but cannot fit an exact 256K request on this 32GB card: the process reaches 31,642 MiB and then fails an additional 874,086,400-byte `lm_head.weight` CUDA allocation. FP8 E4M3 KV completes exact 256K with a 31,088 MiB sampled peak.
- **XQA scope:** The SM70 XQA specialization is a `qLen=1` decode kernel, not a general prefill kernel. MTP9 target validation uses `qLen=10` and intentionally remains on the native causal prefill path; greedy/seed `qLen=1` decode can use XQA.
- **Deployment state (audited after the benchmark):** These are completed benchmark results, not evidence of a currently running service. Ports 8000/8001/8002 were not listening and no GPU inference process was active at the latest audit.

## V100 attention provenance and test boundary

The complete Git history and the surviving `.config/superpowers` worktree establish three distinct states:

| Code path / workstream | FastLLM status | Provenance | Verified result |
|------------------------|----------------|------------|-----------------|
| Existing native paged attention | Already upstream before this work | `38b4b8b8` added the low-compute-capability native fallback on 2026-05-31; `adb30fd0` optimized SM70 memory bandwidth; `bd9a24ad` added bounded chunked-cuBLAS prefill | Exact-window 128K/160K/256K service matrix completed on this baseline; paged metadata/CUDA graph regressions pass |
| FlashInfer-SM70 | **Not landed in the current FastLLM worktree or PR #705** | No independent backend, planner, Volta MMA compatibility layer, BM32/`ALL_P` prefill kernel, or corresponding FastLLM benchmark artifact is present. Bundled upstream FlashInfer remains disabled on CC 7.0 | No FastLLM performance result is claimed |
| Flash-Attention-V100-style XQA | Landed in `b693ad8` as `FastllmSm70PagedXqaSplitKernel` plus the existing combine kernel | Adapts work-by-KV-head and K/V reuse across six Q heads to FastLLM's page-128 Q24/KV4/D256 layout; it is a FastLLM-native rewrite, not a byte-for-byte copy | 2.22×/3.37×/4.03× single-layer decode speedup at 8K/32K/128K KV, plus focused/full regressions and short 27B smoke |

Real V100 prefill logs `Native paged prefill uses chunked cublas attention`; therefore no FlashInfer-SM70 or Flash-Attention-V100 BM32/`ALL_P` prefill speedup is attributed to FastLLM. The existing native fallback's split-KV/state-combine structure is similar to mechanisms explored in FlashInfer-SM70, but full history proves it was added months earlier and is not a port from that later workstream.

The exact-window JSON files were produced before `b693ad8`; they validate the existing native stack but not the final GQA6 XQA increment. After `b693ad8`, the XQA increment was validated by focused/full regressions, a three-shape microbenchmark, and short greedy/MTP9 real-model smoke. Repeating the 128K/160K/256K matrix on the final branch remains outstanding.

## Patches applied

The isolated upstream branch for PR [ztxz16/fastllm#705](https://github.com/ztxz16/fastllm/pull/705) contains the Qwen3.5 GGUF/runtime, MTP9, SM70 IQ4_XS, native SM70 paged XQA, apiserver, and regression changes described below.

### 1. Qwen3.5/3.6 GGUF adapter (`src/model.cpp`, `third_party/gguf/*`)
- Architecture alias `qwen35`→`qwen3_5`
- `block_count` correction (65→64 trunk + 1 MTP)
- `attention.key_length` → `head_dim`, `ssm.*` → linear-attention metadata
- V-head tiled→grouped inverse permutation at load time (`GGUFWeightReplaceUntileVHeads`)
- Norm pre-offset recognition independent of V-head layout
- `FASTLLM_QWEN35_GGUF_VHEAD_TILED=0` override for pre-grouped GGUFs
- `CreateLLMModelFromFile()` magic-based GGUF dispatch (fixes `std::bad_alloc`)

### 2. Qwen3.5 runtime (`src/models/qwen3_5.cpp`, `include/models/qwen3_5.h`)
- KV metadata initialization (`num_key_value_heads` shadowing fix, linear-attention metadata)
- Grouped vs tiled TP out-proj scheme selection via `ggufOutProjColumnsTiled`
- MTP logging/profile infrastructure

### 3. SM70 IQ4_XS MMQ (`src/devices/cuda/fastllm-iq4xs-sm70.cu`)
- DP4A matrix-multiply-quantized kernel for SM70/V100
- F16/F32/BF16 activations, Q8_1 D4 quantization, per-stream persistent scratch
- env-gated (`FASTLLM_CUDA_SM70_IQ4XS_MMQ=0` to disable)

### 4. Existing V100 native attention and new SM70 paged XQA (`src/devices/cuda/attention/paged/*`)
- Upstream commits `38b4b8b8`, `adb30fd0`, and `bd9a24ad` supply the existing native split-KV/graph-stable state management and chunked-cuBLAS prefill; these are baseline FastLLM code, not part of our port
- `b693ad8` adds the exact SM70 decode route, adapting the GQA6 K/V-reuse strategy exercised in Flash-Attention-V100 to page 128, Q24/KV4, D256, and `qLen=1`
- One four-warp split-KV block reuses each K/V load across six Q heads; the existing second kernel combines split online-softmax state
- Stable per-thread-stream scratch supports CUDA graph capture/replay and concurrent PTDS workers
- Enabled by default for the exact shape; `FASTLLM_CUDA_SM70_PAGED_XQA=0` restores the original native route
- Ineligible dtypes/layouts/shapes, non-contiguous output views, and `qLen>1` safely fall back before launch

### 5. CPU rounding fix (`src/devices/cpu/cpudevice.cpp`)
- `_mm256_cvtps_epi32` → `_mm256_cvttps_epi32` (truncation, not round-to-nearest, matching scalar path).

### 6. Apiserver hardening (`example/apiserver/apiserver.cpp`)
- Tokenizer buffer UAF fix (direct init instead of assign-after-declare)
- Readiness probe no longer deadlocks
- HTTPS/OpenAI SSE chat completions + streaming

### 7. Parallel regression test suite (`test/ops/regressionOps.cpp`)
All focused regressions pass: GGUF config aliases, V-head layout, grouped override, embedding/direct memory, projection layout resolution, TP out-proj scheme, CPU embedding low-mem AWQ Linear, MTP9 snapshots/greedy, CUDA graph ownership, paged batches, etc.

## Performance

### Exact-window service results

Long-context testing used the C++ apiserver with `--atype float16`, MTP9, IQ4_XS MMQ, and FastLLM's existing native paged-attention stack. Exact raw-prompt token counts were confirmed by the server's final `usage`. Each request contains `context - 256` prompt tokens and requests 256 completion tokens. The output continued the terminal `Count upward: 1, 2, 3,` instruction from 4 in every successful run. These artifacts predate the final `b693ad8` GQA6 XQA specialization.

The final traces began at 21–152 MiB, and the previously observed ComfyUI PID was no longer running. These results are therefore **exclusive-GPU capacity/performance measurements**, not coexistence measurements.

| Context | KV dtype | Prompt + decode | TTFT | E2E | Sampled VRAM peak | Result |
|---------|----------|-----------------|------|-----|-------------------|--------|
| 128K | FP16 | 130,816 + 256 | Not captured | 318.76s | 28,049 MiB | HTTP 200; exact usage 131,072 |
| 160K | FP16 | 163,584 + 256 | 344.53s | 381.13s | 30,529 MiB | HTTP 200; exact usage 163,840 |
| 256K | FP16 | 261,888 + 256 | — | — | 31,642 MiB before failure | OOM while allocating 874,086,400-byte `lm_head.weight`; disabling CUDA embedding repeats the same failure |
| 256K | FP8 E4M3 | 261,888 + 256 | Not reliably captured | 745.47s | 31,088 MiB | HTTP 200; exact usage 262,144 |

For the 160K run, the measured prefill rate to the first content chunk was 474.81 prompt tok/s. The remaining 256-token stream took 36.61s (6.99 tok/s). No TTFT claim is made for 128K because that client was non-streaming. No TTFT claim is made for 256K because the measurement client timestamped the initial OpenAI `role=assistant` metadata chunk rather than the first non-empty content token.

MTP profile counters confirm actual speculative validation rather than a no-op flag. The final cumulative profiles recorded:

| Context | Speculative validations | Full / partial / reject-0 | Avg commit | Avg matched draft |
|---------|-------------------------|---------------------------|------------|-------------------|
| 128K | 17 | 10 / 6 / 1 | 1.74 | 0.74 |
| 160K | 18 | 9 / 8 / 1 | 1.74 | 0.74 |
| 256K FP8 | 22 | 8 / 12 / 2 | 1.72 | 0.72 |

The earlier long-context timeouts were caused by two corrected runtime issues: post-prefill MTP cache replay and exact-window single-token page-budget fallback. The official `ftllm bench` attempt is still excluded because its internal FP32 GGUF warmup entered a different path and aborted with a cuBLAS error.

Validation chronology: the three exact-window artifacts were written at approximately 06:50 on 2026-07-29; `b693ad8` was committed at 09:19, and the XQA summary artifact was written at approximately 09:34. Do not interpret the table above as a post-XQA end-to-end A/B.

### Post-`b693ad8` SM70 paged XQA decode microbenchmark

This is a **single-layer attention microbenchmark**, not end-to-end model throughput. It measures the final FastLLM-native XQA specialization: work is grouped by KV head and each K/V load is reused across six Q heads, adapting the strategy exercised in Flash-Attention-V100 to page 128. The pre-existing split scheduling/state reduction is FastLLM baseline code and is not attributed to FlashInfer-SM70. The comparison is against FastLLM's original per-Q-head native split decode on the same V100, with batch 1, FP16 Q/K/V, page size 128, Q24/KV4, D256, five warmup iterations, and 30 CUDA-event-timed iterations. The paged cache is resident on the GPU and the full two-kernel XQA sequence is timed.

| KV tokens | Original per-Q-head native | SM70 XQA | Speedup |
|----------:|---------------------------:|---------:|--------:|
| 8,192 | 0.397 ms | 0.179 ms | **2.22×** |
| 32,768 | 1.739 ms | 0.516 ms | **3.37×** |
| 131,072 | 5.774 ms | 1.433 ms | **4.03×** |

The final phase-1 cubin uses 136 registers and 30,912 bytes of shared memory, with zero local memory/spill. Focused coverage includes non-sequential physical pages, partial final pages, batch 2 with nonzero `pageStart`, CUDA graph capture/instantiate/replay, two concurrent per-thread default streams, unsupported groups, and non-contiguous output rejection. The complete `regressionOps` suite also passes.

The retained smoke summary records HTTP 200 for the 27B IQ4_XS greedy request and the production route string `SM70 paged XQA enabled`. A separate MTP9 smoke returned HTTP 200 with speculative full/partial acceptance; its `qLen=10` target-validation calls correctly stayed on the existing native path. The raw post-`b693ad8` service log was not retained, so the machine-readable JSON is summary evidence rather than a replayable raw-log artifact.

Machine-readable result: `benchmarks/fastllm/results/fastllm_sm70_paged_xqa.json`.

## TurboQuant 256K reference

The retained TurboQuant result uses two agents with 262,144 tokens per slot, 117,964 shared-prefix tokens, and 256 output tokens. Steady-state aggregate throughput was 15.65 tok/s (7.78 tok/s per request), with 32.92s average latency. Its workload, prefix reuse, concurrency, and cache format differ from the cold exact-window FastLLM measurement, so the figures are retained as deployment evidence rather than presented as a direct A/B.

Result artifacts:
- `benchmarks/fastllm/results/fastllm_mtp9_128k_exact.json`
- `benchmarks/fastllm/results/fastllm_mtp9_160k_exact.json`
- `benchmarks/fastllm/results/fastllm_mtp9_256k_exact.json`
- `benchmarks/fastllm/results/fastllm_sm70_paged_xqa.json`
- `benchmarks/turboquant/results/llama_tq3_iq4_256k_2slot_short.json`

## Build notes

See `docs/HANDOFF_fastllm.md` (§3) for the reference build procedure. Key prerequisites:
- CUDA toolkit with an SM70-compatible compiler
- A host compiler version supported by that CUDA toolkit
- `libnuma` development headers when NUMA support is enabled
- Build parallelism: `-j4` max (swap thrashing at `-j8`)

## Service deployment

```bash
# tmux session: fastllm
ulimit -c 0
cd /path/to/fastllm/build
FASTLLM_QWEN35_ENABLE_MTP=9 ./apiserver \
  -p /path/to/model.gguf \
  -t 2 -l --atype float16 --kv_cache_dtype fp8_e4m3 \
  --batch 1 --tokens 262144 --model_name qwen3.6-fastllm \
  --port 8002 --device cuda --cuda_embedding
```

Proxy adapter code can render the macro-heavy template with Jinja2 and send `raw_prompt=true`, but the checked-in `thinking_proxy.py` refactor is not yet the required independent fifth-backend architecture: its current FastLLM mode replaces the llama backend. At the latest audit neither the proxy nor FastLLM service was running on ports 8000/8002.