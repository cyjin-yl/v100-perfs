# Qwen3.6 Q4_K_M on V100 (SM70) optimization design

Date: 2026-08-09
Scope: single Tesla V100-PCIE-32GB, FastLLM native GGUF stack, Qwen3.6-27B class
(Q4_K_M trunk + nextn MTP block). All experimental gates default-off or
semantically equivalent, fail-safe fallback, per project convention.

## 1. Goal and success target

Raise effective decode throughput of the Q4_K_M artifact on SM70 to at least
the IQ4_XS reference (llama.cpp IQ4_XS+MTP: 41.5 tok/s steady; FastLLM IQ4_XS
FP8 profile C1 50.85 tok/s) without changing Q4_K_M weight quality, and
without regressing the production Turbo3 profile.

## 2. Current evidence

- FastLLM apiserver, ThinkingCap-Qwen3.6-27B-Q4_K_M.gguf, `--atype float16
  --kv_cache_dtype turbo3 --batch 5 --tokens 65536`, temperature 0, 2026-08-09:
  text decode 41.1-45.2 tok/s (median over two runs of five fixtures),
  vision 23.5-24.0 tok/s, TTFT <=1.4 s warm.
  (`benchmarks/fastllm/results/thinkingcap_q4_fastllm_mtp0.json`)
- Bandwidth arithmetic: 16.81 GB weights x 45 tok/s = 0.76 TB/s, i.e.
  ~84% of V100 HBM peak (~0.9 TB/s). Decode is memory-bandwidth bound; the
  dequant/GEMM kernel is no longer the bottleneck.
- SM70 ChunkGDN prefill experiment matrix
  (`benchmarks/fastllm/results/fastllm_sm70_chunk_gdn_prefill_experiments.json`):
  fused_h_o 4.4-6.2x slower (conclusive negative), sm70_wmma 63% slower
  (conclusive negative), async_gather / fused_preprocess / persistent_scratch
  within noise (negatives). Micro-optimization around existing kernels has
  repeatedly failed to beat cuBLAS + stock dequant on SM70.

## 3. Candidate approaches

### A. New SM70 Q4_K DP4A/MMQ kernel
Port the IQ4_XS SM70 MMQ pattern to the Q4_K superblock layout.
- Expected gain: bounded by remaining HBM headroom, at best +10-20% at batch 1.
- Cost: new kernel + regression matrix; risk of subtle scale-decode bugs.
- Verdict: not recommended. Evidence §2 shows <20% headroom, and §2 experiment
  history shows custom SM70 kernels losing to stock paths twice.

### B. Offline requant to IQ4_XS, reuse existing SM70 MMQ path
- Weight 16.8 -> ~12.2 GB (-27%); theoretical decode ~+35% (llama.cpp IQ4_XS
  no-MTP 36 tok/s vs Q3_K_M 21 tok/s shows the bandwidth lever is real).
- Changes the artifact and its quality profile (imatrix required); no longer
  "Q4_K_M". Quality evaluation mandatory (perplexity + fixture parity).
- Verdict: viable deployment alternative, not an optimization of Q4_K_M itself.
  Keep as an option when VRAM/bandwidth dominate and quality delta is accepted.

### C. Enable native MTP on the existing Q4_K_M GGUF (recommended)
The Q4_K_M artifact already carries the full nextn MTP block (15 tensors,
263 MB; FastLLM maps blk.64.nextn.* -> mtp.* and performs speculative
validation). Effective throughput multiplies by the acceptance rate at the
cost of draft KV/VRAM and validation complexity.
- llama.cpp precedent on the same model family: IQ4_XS 36 -> 41.5 tok/s
  (+15%) at 86.4% acceptance; FastLLM MTP9 already validated on IQ4_XS.
- Zero new kernels; fail-safe: `--mtp 0` restores the measured baseline;
  gates (`FASTLLM_QWEN35_BATCHED_MTP` etc.) keep production semantics.
- Cost: +draft KV VRAM (bounded; ~2.4 GB at dual-256K in llama.cpp, much less
  at the 65K production window), needs acceptance + parity verification.
- Verdict: recommended primary path.

### D. Further SM70 micro-optimizations (gather streams, preprocess fusion,
scratch arenas, WMMA prototypes)
- Verdict: prohibited by evidence. The GDN experiment matrix (§2) concluded
  these are noise or conclusive negatives on this GPU.

## 4. Recommendation

1. Primary: run Q4_K_M with FastLLM native MTP (`--mtp 2` default candidate),
   verify greedy parity vs `--mtp 0`, acceptance >=70%, no VRAM regression in
   the Turbo3 262K pool profile. Ship as an explicit profile, env-gated.
2. Alternative: offer IQ4_XS requant as a separate artifact for
   bandwidth-bound deployments (option B), with its own quality note.
3. Do not invest in option A/D kernels.

## 5. Verification matrix (when GPU time permits)

- `--mtp 0` vs `--mtp 2` vs `--mtp 5` on the five feature fixtures
  (arithmetic/exact/reasoning/tool/vision), temperature 0: identical greedy
  token streams required.
- Acceptance from `[Qwen3.5 MTP]` logs; decode tok/s median of 2 runs.
- Peak VRAM via CUDA runtime (NVML is broken on this host: driver 580.159
  kernel vs 580.173 userspace), must stay >=512 MiB below the 32 GB ceiling
  with the 65K pool.
- Results recorded under `benchmarks/fastllm/results/` with schema v1.
