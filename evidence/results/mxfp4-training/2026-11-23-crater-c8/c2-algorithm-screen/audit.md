# C2 cuBLAS 算法筛选：独立原始证据审计

状态：`verified`。本审计仅处理真实本地归档，未执行 GPU、编译器、远端操作或归档代码。

**这里全部性能数字仅为 normalized GEMM compute，未测完整 MXFP4 算子流水线，也未改变 production 算法。**

结果 tar SHA-256 `b9bb395a2abaeaab11e08e44452361f931c7d710a0e16861d4305e899bfa88c6`；冻结生产源提交 `ed39b0d715271e7b359672a081d758b4e4d5070f`，源 tar SHA-256 `3fb8cc55336a187f3c863eb98ae4d5089f57aabd6b8b585db85c0d09e03c1bd0`。

实际 ELF 已在本地重新计算 SHA-256：`85f1168922d8329f2e488741a1eb2ef022a4e097e4a8d3b2a7a082b84877bb9a`；编译 manifest/log、候选源码、生产源码、全部归档成员及前序 native 正确性来源均一致。

## 观测到的筛选结果

| 输入 | 最快观测算法 | Wall compute TFLOPS | 默认 DEFAULT_TENSOR_OP | 默认 DEFAULT |
|---|---|---:|---:|---:|
| 4096³ random_block_scales | CUBLAS_GEMM_DEFAULT_TENSOR_OP | 92.559661 | 92.559661 | 92.386799 |
| 8192³ random_unit_scale | CUBLAS_GEMM_DEFAULT | 99.280369 | 95.429419 | 99.280369 |

两个 accepted 输入各 18 种算法全部完成初次与最后实际 timed 输出的全量位比较。每种算法 151 次成功 GEMM，100 次 warmup，5×10 次计时；无 nonfinite、负零或位不一致。中位数直接由全部五个原始 wall/event round 重算。

8192³ random_block_scales 的证书因 energy bound 拒绝（flags=4），没有任何算法计时或虚构吞吐。上层 `completed_with_blocked_cases` 是此预期边界结果，不能改写成三个 case 均 passed。

## 4096³ random_block_scales 完整排序

| 算法 | Wall ms | Wall compute TFLOPS | Event ms | Event compute TFLOPS |
|---|---:|---:|---:|---:|
| CUBLAS_GEMM_DEFAULT_TENSOR_OP | 1.484868800 | 92.559661 | 1.484006405 | 92.613450 |
| CUBLAS_GEMM_DEFAULT | 1.487647100 | 92.386799 | 1.486486435 | 92.458936 |
| CUBLAS_GEMM_ALGO3_TENSOR_OP | 1.600849500 | 85.853763 | 1.599993587 | 85.899690 |
| CUBLAS_GEMM_ALGO4_TENSOR_OP | 1.625737900 | 84.539429 | 1.624857545 | 84.585233 |
| CUBLAS_GEMM_ALGO6_TENSOR_OP | 1.630835600 | 84.275174 | 1.629980803 | 84.319369 |
| CUBLAS_GEMM_ALGO5_TENSOR_OP | 1.723846400 | 79.728074 | 1.723020792 | 79.766277 |
| CUBLAS_GEMM_ALGO7_TENSOR_OP | 1.883831800 | 72.957126 | 1.883008003 | 72.989044 |
| CUBLAS_GEMM_ALGO1_TENSOR_OP | 1.966002200 | 69.907833 | 1.965167999 | 69.937508 |
| CUBLAS_GEMM_ALGO2_TENSOR_OP | 2.032829400 | 67.609684 | 2.031964779 | 67.638453 |
| CUBLAS_GEMM_ALGO0_TENSOR_OP | 2.432818900 | 56.493705 | 2.431952000 | 56.513843 |
| CUBLAS_GEMM_ALGO11_TENSOR_OP | 2.656877400 | 51.729505 | 2.656028748 | 51.746034 |
| CUBLAS_GEMM_ALGO14_TENSOR_OP | 2.746880600 | 50.034557 | 2.746073484 | 50.049263 |
| CUBLAS_GEMM_ALGO12_TENSOR_OP | 2.758418400 | 49.825274 | 2.757535934 | 49.841219 |
| CUBLAS_GEMM_ALGO13_TENSOR_OP | 2.823186600 | 48.682207 | 2.822384119 | 48.696048 |
| CUBLAS_GEMM_ALGO15_TENSOR_OP | 3.028523000 | 45.381512 | 3.027699232 | 45.393859 |
| CUBLAS_GEMM_ALGO9_TENSOR_OP | 3.179702000 | 43.223847 | 3.178825617 | 43.235764 |
| CUBLAS_GEMM_ALGO10_TENSOR_OP | 3.530515200 | 38.928866 | 3.529577732 | 38.939206 |
| CUBLAS_GEMM_ALGO8_TENSOR_OP | 3.609693100 | 38.074969 | 3.608649492 | 38.085980 |

边界 NVML 读数范围：`{"sm_clock_mhz": {"min": 1290, "max": 1530}, "memory_clock_mhz": {"min": 877, "max": 877}, "power_w": {"min": 63.588, "max": 301.735}, "power_limit_w": {"min": 300, "max": 300}}`。这些快照不是各算法执行期间的连续频率测量。

## 8192³ random_unit_scale 完整排序

| 算法 | Wall ms | Wall compute TFLOPS | Event ms | Event compute TFLOPS |
|---|---:|---:|---:|---:|
| CUBLAS_GEMM_DEFAULT | 11.074814100 | 99.280369 | 11.073762894 | 99.289793 |
| CUBLAS_GEMM_DEFAULT_TENSOR_OP | 11.521726100 | 95.429419 | 11.520917892 | 95.436114 |
| CUBLAS_GEMM_ALGO6_TENSOR_OP | 12.034468100 | 91.363542 | 12.033641815 | 91.369815 |
| CUBLAS_GEMM_ALGO4_TENSOR_OP | 12.157146700 | 90.441586 | 12.156361580 | 90.447427 |
| CUBLAS_GEMM_ALGO3_TENSOR_OP | 14.074715100 | 78.119637 | 14.073971748 | 78.123763 |
| CUBLAS_GEMM_ALGO7_TENSOR_OP | 14.356688600 | 76.585323 | 14.355868340 | 76.589698 |
| CUBLAS_GEMM_ALGO1_TENSOR_OP | 17.618736000 | 62.405818 | 17.617914200 | 62.408729 |
| CUBLAS_GEMM_ALGO5_TENSOR_OP | 19.232174500 | 57.170427 | 19.230764389 | 57.174619 |
| CUBLAS_GEMM_ALGO12_TENSOR_OP | 23.033569700 | 47.735181 | 23.032878876 | 47.736613 |
| CUBLAS_GEMM_ALGO14_TENSOR_OP | 23.374677200 | 47.038580 | 23.373954773 | 47.040034 |
| CUBLAS_GEMM_ALGO11_TENSOR_OP | 23.839611700 | 46.121205 | 23.838914871 | 46.122554 |
| CUBLAS_GEMM_ALGO13_TENSOR_OP | 24.669111100 | 44.570379 | 24.668407440 | 44.571650 |
| CUBLAS_GEMM_ALGO15_TENSOR_OP | 25.836646800 | 42.556282 | 25.835906982 | 42.557501 |
| CUBLAS_GEMM_ALGO0_TENSOR_OP | 28.892945200 | 38.054675 | 28.892282486 | 38.055547 |
| CUBLAS_GEMM_ALGO2_TENSOR_OP | 29.385526400 | 37.416775 | 29.384826660 | 37.417666 |
| CUBLAS_GEMM_ALGO9_TENSOR_OP | 30.225314200 | 36.377178 | 30.224630356 | 36.378001 |
| CUBLAS_GEMM_ALGO10_TENSOR_OP | 31.334372800 | 35.089633 | 31.333568573 | 35.090533 |
| CUBLAS_GEMM_ALGO8_TENSOR_OP | 44.417271000 | 24.754146 | 44.416625977 | 24.754506 |

边界 NVML 读数范围：`{"sm_clock_mhz": {"min": 1350, "max": 1530}, "memory_clock_mhz": {"min": 877, "max": 877}, "power_w": {"min": 73.854, "max": 288.565}, "power_limit_w": {"min": 300, "max": 300}}`。这些快照不是各算法执行期间的连续频率测量。

## 结论边界

- Only normalized FP16 GEMM compute is timed; certificate, packed decode/normalization, original epilogue, fallback, allocations, quantization and copies are excluded.
- Every actual final timed output receives the production C2 epilogue only after both endpoints, then full exact comparison against the preserved C2 reference and transitively exact tiled output; no replacement GEMM is used.
- The certificate is computed by production C2 once, then its actual normalized buffers/handle/4MiB workspace are reused. This differs from production full-pipeline timing, which recomputes the certificate/normalization each call.
- One process per accepted shape/mode and five correlated rounds per algorithm; this is a screen, not independent ABBA validation or a stable production selection.
- Boundary NVML samples can vary and are not per-algorithm continuous frequency traces. Clock/power changes were not requested; exclusive GPU use is not established.
- The 8192 block-scale case legitimately rejects on energy with no algorithm timing. Its numerical fallback output was verified in the separate same-input native probe, not copied/compared by this blocked screen.
- 4096 block-scale and 8192 unit-scale distributions are distinct. These compute figures must not replace the independent full-pipeline numbers or be presented as complete MXFP4 training throughput.
- No production algorithm was changed. Any proposed winner still requires independent full-pipeline validation under paired comparable conditions.
