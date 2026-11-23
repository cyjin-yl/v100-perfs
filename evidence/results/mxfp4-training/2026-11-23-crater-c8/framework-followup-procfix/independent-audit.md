# Framework follow-up: independent audit

The real procfix follow-up is **verified**: all five stages exited zero, the 119-case integration suite passed its declared contracts, and both five-arm synthetic smokes completed 40 steps per arm. This audit performed no new GPU or remote work.

## Provenance

- Result archive SHA256: `2de0fc7de6d7f3fb8e33a760c538fc2e36c37cf4f08b133960e9c7df9f778cd1` (47225 bytes).
- All 21 non-completion regular members match completion SHA256 entries; completion itself matches the receipt.
- Source B `80d3a7c2bb0bb551a34b28bd7fda59978b6304ef` / archive `62378914e2b6ad7e34e3dd70736cb90aea07c962963494d97062d5f5738b0f57` and the original B build/result manifests are byte/hash bound.
- Executed helper SHA256: `2046ca0b52d0f52fa9baaf565be92fe899a5ac165da3970eaa583902eee0197b`. Library SHA256: `eb3255a5b74756457c25cfa9422bd57398d642ea6db26577ee8d317c68789f07`.
- Device: Tesla V100-SXM2-32GB; Python 3.12.15 via `/opt/venv/bin/python3`; NumPy 2.3.5; PyTorch 2.10.0+cu128.

## Integration correctness

- 12 packed fixtures + 9 RNE quantization cases + 1 stochastic-seed case + 96 linear cases + 1 rejected GPU-to-reference fallback = **119**.
- 156 exact numerical metrics have zero mismatches. All 240 ordinary FP32 allclose metrics have zero tolerance failures at atol=rtol=3e-5; 136 of those metrics have permitted bitwise differences.
- The 96 linear cases cover six fixture groups × two backends × eight forward/dgrad/wgrad recipes, including noncontiguous input and original-FP32 transpose checks.

## 1024³ Torch benchmark

Wall medians are recomputed from all three raw rounds, each averaging three invocations after one warmup. These are repeated measurements in one process.

| Job | Wall median (ms) | Useful TFLOPS | Output |
|---|---:|---:|---|
| torch_fp32_matmul | 0.248623992 | 8.637476 | torch.float32 |
| torch_fp16_compute | 0.066912665 | 32.093829 | torch.float16 |
| torch_fp16_casts_and_matmul | 0.089650663 | 23.953907 | torch.float16 |
| mxfp4_quantize_both | 1.897756670 | — | uint8_codes_and_scales |
| mxfp4_cuda_quantize_and_gemm | 2.789762667 | 0.769773 | torch.float32 |
| mxfp4_cuda_tiled_quantize_and_gemm | 2.046605999 | 1.049290 | torch.float32 |

Packed cuda/tiled initial and final timed outputs match at all 1,048,576 positions; all five final timed GEMM outputs match their own initial outputs; final RNE codes/scales remain exact. FP16 comparators produce FP16 output. Quantized totals include both input quantizations, temporary/output allocations, validation and Python dispatch.

## Forty-step synthetic smoke

All arms update 17 parameter tensors, retain finite FP32 master parameters and AdamW state, and reduce validation loss. Mean step time is `1000 × instrumented training wall seconds / 40`; it is not an isolated kernel or LLM training benchmark.

| Backend | Arm | Initial loss | Final loss | Total training wall (s) | Mean step (ms) |
|---|---|---:|---:|---:|---:|
| cuda | fp32 | 3.645810843 | 0.062208887 | 0.803111643 | 20.077791 |
| cuda | mx_forward | 3.608196259 | 0.071482189 | 0.978099212 | 24.452480 |
| cuda | mx_forward_dgrad | 3.608196259 | 0.073648632 | 2.043076970 | 51.076924 |
| cuda | mx_forward_dgrad_wgrad | 3.608196259 | 0.076575369 | 2.308564933 | 57.714123 |
| cuda | mx_all_stochastic_grad | 3.608196259 | 0.072527491 | 2.309955114 | 57.748878 |
| cuda_tiled | fp32 | 3.645810843 | 0.062208887 | 0.722624293 | 18.065607 |
| cuda_tiled | mx_forward | 3.608196259 | 0.071482189 | 1.018932235 | 25.473306 |
| cuda_tiled | mx_forward_dgrad | 3.608196259 | 0.073648632 | 2.095009978 | 52.375249 |
| cuda_tiled | mx_forward_dgrad_wgrad | 3.608196259 | 0.076575369 | 2.415708282 | 60.392707 |
| cuda_tiled | mx_all_stochastic_grad | 3.608196259 | 0.072527491 | 2.415425379 | 60.385634 |

The two backends have identical per-arm training-loss trajectories and initial/final validation losses for this seed. The model is one layer, width32, four heads, 11,168 parameters, and trains on synthetic length-16 modular sequences with vocabulary32.

## Limits and diagnostics

These results validate baseline/tiled Torch integration from the C2-built library; they do not measure the certified/C8 Torch path. The tiny synthetic smoke establishes finite updates and decreasing loss for this fixture, not LLM pretraining quality or convergence.

The integration stderr contains a PyTorch warning that the autograd cuBLAS path set a previously absent current CUDA context. The actual suite subsequently synchronized and passed every declared gate; the warning is retained verbatim in the audit JSON.

The original B report remains blocked for its original framework attempt. This independent successful follow-up does not rewrite that history.
