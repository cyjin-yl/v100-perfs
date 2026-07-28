# FastLLM on V100 — Qwen3.6 Fable Fusion Performance

**Date:** 2026-07-28  
**Hardware:** NVIDIA V100 (SM70, 32GB VRAM, 16GB SM, 64GB system RAM)  
**Software:** Fedora 44, GCC 16.1.1 (system), GCC 12.4.0 (conda), CUDA 12.9  
**Model:** Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-LOW-MTP-IQ4_XS.gguf (IQ4_XS, 64 trunk layers + 1 MTP head)  

## Outcome

FastLLM now **runs full inference** (trunk + MTP9) on this model with a patched build — see §Patches. The first real token was produced at approximately 87s (cold startup with `--low` lazy weight placement). MTP9 yields partial acceptance (~1.75 avg matched drafts per step at commit~2.75 tokens). All ops are on CUDA: `Fastllm paged prefill` (chunked cublas), SM70 IQ4_XS MMQ matmul, and GDN transposed-recurrent decode.

## What works

1. **Model load** — all 65 blocks + tokenizer + IQ4_XS dequant → GPU in ~2m30s.
2. **`/generate` raw prompt** — HTTP 200, correct output, streaming and non-stream OpenAI SSE.
3. **SM70 IQ4_XS MMQ** — auto-selected for `k>=128`, F16 activations, DP4A accumulation. Fallback to legacy MMVQ for narrow projections.
4. **MTP9 speculative decode** — 9 drafts/step, exact acceptance, single-GPU copy-validation path.
5. **GDN (linear attention)** — `cuda-transposed-recurrent` with QKVZ+BA fused layout.
6. **Native paged attention** — FlashInfer disabled on CC 7.0 → chunked cublas prefill + native paged decode.

## Known limitations

- **Chat template:** The embedded GGUF Jinja template uses `{% macro %}`, which FastLLM's Jinja parser does not support. Use raw `/generate` with pre-formatted ChatML, or deploy the Thinking Proxy with proxy-side Jinja2 rendering.
- **MTP >1 layer:** The GGUF has `nextn_predict_layers=1`. Multi-layer MTP is untested.
- **Multi-GPU TP:** Single V100 only — not validated with tensor parallelism.
- **Long context:** 128K, 160K, and 256K KV capacities all initialize, but exact full-window raw prompts did not complete within the benchmark deadlines; see §Performance.

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

Short production-path inference remains functional after the final fixes: MTP9 returned HTTP 200 with `OK`; the profile recorded partial speculative acceptance. Long-context testing used the same C++ apiserver path with `--atype float16`, MTP9, and exact raw-prompt token counts calibrated from server `usage.prompt_tokens` (`"x " × N` produces $N+1$ tokens).

| Context | Input target | Decode target | Capacity startup | Request result | Observed VRAM |
|---------|--------------|---------------|------------------|----------------|---------------|
| 128K | 130,816 | 256 | PASS | Did not complete in 3,600s; GPU initially saturated, then fell to 0% while request remained pending | 27,067 MiB idle; 27,899 MiB peak sampled |
| 160K | 163,584 | 1 prefill probe | PASS | Did not complete in 900s | Not separately sampled |
| 256K | 261,888 | 1 prefill probe | PASS | Did not complete in 900s | Not separately sampled |

The official `ftllm bench` runner was also attempted at 128K with 130,816 input and 256 output tokens. It aborted during its internal warmup with a cuBLAS error (exit 134) because that Python path entered FP32 GGUF matmul, unlike the production F16 apiserver path. It is therefore not used for throughput claims.

These are failed performance runs, not successful throughput measurements. They prove KV-capacity initialization but do not establish usable long-context prefill/decode rates.

## TurboQuant 256K reference

The existing completed TurboQuant result uses two agents with 262,144 tokens per slot, 117,964 shared-prefix tokens, and 256 output tokens. Steady-state aggregate throughput was 15.65 tok/s (7.78 tok/s per request), with 32.92s average latency. The historical VRAM trace reached 32,199 MiB transiently and about 30,692 MiB in steady decode. A same-session rerun was intentionally skipped because ComfyUI remained resident and the combined requirement exceeds the 32GB V100; the existing result and server log are retained as the verified 256K evidence.

Result artifacts:
- `benchmarks/fastllm/results/fastllm_mtp9_128k_timeout.json`
- `benchmarks/fastllm/results/fastllm_mtp9_160k_timeout.json`
- `benchmarks/fastllm/results/fastllm_mtp9_256k_http.json`
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
  -t 2 -l --atype float16 --batch 1 --tokens 8192 \
  --model_name qwen3.6-fastllm --port 8002 \
  --device cuda --cuda_embedding
```

Proxy: `FASTLLM_BACKEND_URL=http://127.0.0.1:8002` plus proxy-side Jinja2 template render.