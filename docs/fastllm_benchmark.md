# FastLLM on V100 — Qwen3.6 Fable Fusion Performance

**Date:** 2026-07-28  
**Hardware:** NVIDIA V100 (SM70, 32GB VRAM, 64GB system RAM)
**Software:** Fedora 44, GCC 16.1.1 (system), GCC 12.4.0 (conda), CUDA 12.9  
**Model:** Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-LOW-MTP-IQ4_XS.gguf (IQ4_XS, 64 trunk layers + 1 MTP head)  

## Outcome

FastLLM now **runs full inference** (trunk + MTP9) on this model with a patched build — see §Patches. Exact-window requests completed at 128K and 160K with FP16 KV, and at 256K with FP8 E4M3 KV. MTP9 remained active through all three runs. The measured path uses native paged attention on V100, SM70 IQ4_XS MMQ matmul, and the Qwen3.5 transposed-recurrent GDN kernels.

## What works

1. **Model load** — all 65 blocks + tokenizer + IQ4_XS dequant → GPU in ~2m30s.
2. **`/generate` raw prompt** — HTTP 200, correct output, streaming and non-stream OpenAI SSE.
3. **SM70 IQ4_XS MMQ** — auto-selected for `k>=128`, F16 activations, DP4A accumulation. Fallback to legacy MMVQ for narrow projections.
4. **MTP9 speculative decode** — 9 drafts/step, exact acceptance, single-GPU copy-validation path.
5. **GDN (linear attention)** — `cuda-transposed-recurrent` with QKVZ+BA fused layout.
6. **Native paged attention** — FlashInfer disabled on CC 7.0 → chunked cublas prefill + native paged decode.

## Known limitations

- **Chat template:** The embedded GGUF Jinja template uses `{% macro %}`, which FastLLM's Jinja parser does not support. Use raw `/generate` with pre-formatted ChatML, or deploy the Thinking Proxy with proxy-side Jinja2 rendering.
- **MTP layer count:** This runtime supports the model's single MTP layer and rejects `nextn_predict_layers>1` at load time.
- **Multi-GPU TP:** Single V100 only — not validated with tensor parallelism.
- **Long-context capacity:** FP16 KV completes at 128K and 160K but cannot fit an exact 256K request on this 32GB card: the process reaches 31,642 MiB and then fails an additional 874,086,400-byte `lm_head.weight` CUDA allocation. FP8 E4M3 KV completes exact 256K with a 31,088 MiB sampled peak.

## Patches applied

The isolated upstream branch contains the Qwen3.5 GGUF/runtime, MTP9, SM70 IQ4_XS, apiserver, and regression changes described below.

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

### 4. CPU rounding fix (`src/devices/cpu/cpudevice.cpp`)
- `_mm256_cvtps_epi32` → `_mm256_cvttps_epi32` (truncation, not round-to-nearest, matching scalar path).

### 5. Apiserver hardening (`example/apiserver/apiserver.cpp`)
- Tokenizer buffer UAF fix (direct init instead of assign-after-declare)
- Readiness probe no longer deadlocks
- HTTPS/OpenAI SSE chat completions + streaming

### 6. Parallel regression test suite (`test/ops/regressionOps.cpp`, ~+1498 lines)
All focused regressions pass: GGUF config aliases, V-head layout, grouped override, embedding/direct memory, projection layout resolution, TP out-proj scheme, CPU embedding low-mem AWQ Linear, MTP9 snapshots/greedy, CUDA graph ownership, paged batches, etc.

## Performance

Long-context testing used the C++ apiserver with `--atype float16`, MTP9, IQ4_XS MMQ, and exact raw-prompt token counts confirmed by the server's final `usage`. Each request contains `context - 256` prompt tokens and requests 256 completion tokens. The output continued the terminal `Count upward: 1, 2, 3,` instruction from 4 in every successful run.

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

## TurboQuant 256K reference

The retained TurboQuant result uses two agents with 262,144 tokens per slot, 117,964 shared-prefix tokens, and 256 output tokens. Steady-state aggregate throughput was 15.65 tok/s (7.78 tok/s per request), with 32.92s average latency. Its workload, prefix reuse, concurrency, and cache format differ from the cold exact-window FastLLM measurement, so the figures are retained as deployment evidence rather than presented as a direct A/B.

Result artifacts:
- `benchmarks/fastllm/results/fastllm_mtp9_128k_exact.json`
- `benchmarks/fastllm/results/fastllm_mtp9_160k_exact.json`
- `benchmarks/fastllm/results/fastllm_mtp9_256k_exact.json`
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

Proxy: `FASTLLM_BACKEND_URL=http://127.0.0.1:8002` plus proxy-side Jinja2 template render.