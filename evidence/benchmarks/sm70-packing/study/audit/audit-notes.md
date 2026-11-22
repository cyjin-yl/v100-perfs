# Soft-BF16-V100 audit, 2026-11-22

Snapshot: `S-i-l-v-e-t/Soft-BF16-V100`, `3f7bf2cef67cc7b0d37966b5c85fed9ba4c350c5`.
Source checkout: `audit/Soft-BF16-V100`.

## Assessment

This is a useful compact compatibility/optimization prototype with concrete engineering work in operand packing, coalesced transposition, dimension padding, restoration and fallback selection. It is not evidence of a new hardware BF16 mode, nor evidence of production-quality general BF16 training support. The actual multiply is cuBLAS FP16 Tensor Core GEMM with FP32 accumulation, not a newly written HMMA microkernel. Training support is explicitly present, so dismissing training categorically is unsupported.

There is no README, test directory, benchmark script, CI configuration, or published benchmark result artifact in this snapshot. The six commits in the visible GitHub history include source/package/license updates; the author's performance claims are in source comments and one convolution commit, with no included complete timing harness.

## Implementation evidence

| Capability | Source | Observation |
|---|---|---|
| BF16 bit widening | `src/kernels/scaled_common.cuh:13-15` | BF16 bits shifted into an FP32 word. |
| Power-of-two scaling | `scaled_common.cuh:24-29` | Magnitudes scale near 1024–2048. |
| Warp/CTA reduction | `rowcol_scaled_gemm_bf16.cu:38-122` | Computes min/max/nonfinite flags and packs FP16. |
| Coalesced column reduction/transpose | `rowcol_scaled_gemm_bf16.cu:383-543` | Separate column stats, finish and tiled transpose kernels. |
| Main GEMM | `rowcol_scaled_gemm_bf16.cu:372` | `cublasGemmEx`, FP16 operands, FP32 output/compute, Tensor Op algorithm. |
| Batched GEMM | `rowcol_scaled_gemm_bf16.cu:309-312` | `cublasGemmStridedBatchedEx` with the same numerical types. |
| Restore | `rowcol_scaled_gemm_bf16.cu:324-342` | Reads FP32 accumulator, uses double for scale division/bias addition, rounds output once. Separate kernel rather than cuBLAS-fused epilogue. |
| Scratch splitting | `bf16_online.py:77-81`, CUDA `358-379` | Splits output rows, leaving reduction dimension intact. |
| Selective FP32 repair | `bf16_online.py:125-149` | Bad rows/columns get FP32 recomputation if within thresholds; too many triggers whole-matrix FP32. |
| Batched fallback | `bf16_online.py:206-208` | Any unsafe pack triggers whole batched FP32 path, unlike 2D selective repair. |
| Autograd | `bf16_online.py:153-169,235-270` | First-order backward computes dA/dB through the same scaled routines; `once_differentiable` prohibits general second derivatives. |
| FP8 weights | `bf16_online.py:273-286,306-314` | E4M3 weights supported only for inference. No FP4 path. |
| SDPA | `bf16_attn.py` | Patches `F.scaled_dot_product_attention`; default `fp16_scaled` despite stale docstring saying FP32. Uses existing PyTorch efficient-attention backend. Early-bound imports unaffected. |
| Convolution | `bf16_conv.py` | Patches six F.conv/transpose functions. Default mode FP32, scaled FP16 opt-in. Small operations retained as native. |

Shape bypass thresholds (`m*k*n < 2e9`, batched total `<8e8`) and host syncs via `.item()/.tolist()` are real overheads and relevant to small operations/graph capture.

## Public performance claims and limits

- Attention source comment: S=4096, H=10, D=64: FP16 1.54 ms vs FP32 5.09 ms. This is 3.305x, but is only a claim in a comment with incomplete environment and no harness.
- Convolution commit `5163b8fc02d76d62470bbacd7b5ba79efa86a141`: on FLUX VAE decoder shapes, native BF16 2.8–3.9x slower than FP32; scaled FP16 another 1.2–1.6x for >=4e9 MACs; 1.42x whole VAE decode. These are author claims, not independently reproduced in this audit.
- No source evidence for the conversation's specific “80–90 TFLOPS”, “5–6x for BF16 overall”, “training 7–8x slower”, “only LoRA worthwhile”, or “gradients frequently force fallback”. These require workloads and real measurements.

## Numerical counterexamples (CPU IEEE binary16 simulation, not GPU observations)

`cpu_numerical_repros.py` reproduces both examples; values are exactly representable BF16 inputs.

1. **GEMM operand pack is not always lossless inside its default guard.** A row with maximum 1 and x=129×2^-37 has ratio 1,065,418,243.97, below the 1,073,217,536 guard. Scaling by 2048 and converting to FP16 changes x from 9.38598532229662e-10 to 9.313225746154785e-10, relative error -0.77519%. Its BF16 upper bits change 0x3081→0x3080. This refutes universal lossless operand preservation; it does not measure overall GEMM error.
2. **Attention's allowed range can still zero small entries.** For max 1 and x=2^-37, ratio 2^37 passes `_MAX_SAFE_RATIO=2^38`, but half(x×2048) rounds to zero. The source comment “True if the fp16 round trip would lose entries this tensor carries” overstates this guard.

Static issue worth GPU reproduction: 2D `matmul_bt` dispatches through `_Gemm` only if a or w requires gradients, overlooking bias-only training. In the fast CUDA path, only bias requiring grad may fail to retain its autograd dependency; small FP32-bypass cases do not have the same issue. No GPU reproduction was performed by this auditing subagent.

## Official hardware comparison

NVIDIA V100 official specifications distinguish PCIe 112 Tensor TFLOPS / 14 FP32 TFLOPS, SXM/NVLink 125 / 15.7, and V100S PCIe 130 / 16.4. V100/V100S bandwidths are 900 / 1134 GB/s respectively. Thus “all V100s peak at 125” is inaccurate.

H100 whitepaper distinguishes PCIe BF16 dense 756 and SXM dense 989.4 TFLOPS, doubled with advertised structured sparsity. Comparing 756 with 125 is about 6.05x in raw peaks; comparing H100 SXM989.4 with V100 SXM125 is about7.92x. This does not predict end-to-end speed.

NVIDIA explicitly documents Volta FP16 input multiplication with FP32 accumulation. Consequently the conversation's argument that only the FP16 10 stored fraction bits exist at the output is inapplicable to that path. Normal FP16 significands also have an implicit leading bit (11 effective bits). Neither fact by itself proves useful arbitrary two-lane packing; input encoding, cross terms, cumulative range and extraction costs still need testing.

## Source URLs

- Pinned main source: https://github.com/S-i-l-v-e-t/Soft-BF16-V100/tree/3f7bf2cef67cc7b0d37966b5c85fed9ba4c350c5
- Author convolution claim: https://github.com/S-i-l-v-e-t/Soft-BF16-V100/commit/5163b8fc02d76d62470bbacd7b5ba79efa86a141
- V100 datasheet: https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf
- V100 specifications: https://www.nvidia.com/en-us/data-center/v100/
- Volta architecture: https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
- Tensor Core precisions: https://www.nvidia.com/en-eu/data-center/tensorcore/
- NVIDIA mixed precision: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/
- H100 whitepaper: https://resources.nvidia.com/en-us-hopper-architecture

