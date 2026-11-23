# Independent C2 preparation-cache audit

Audit: **verified**.

Actual experiment: **passed**; stage counts: {'passed': 26, 'failed': 0, 'blocked': 0}.

| Shape | Packed input mode | OFF wall ms / TFLOPS | ON wall ms / TFLOPS | ON/OFF throughput |
|---|---|---:|---:|---:|
| 2048³ | random_block_scales | 0.313418 / 54.8145 | 0.312234 / 55.0224 | 1.00379× |
| 4096³ | random_block_scales | 1.820288 / 75.5040 | 1.805337 / 76.1292 | 1.00828× |
| 8192³ | random_unit_scale | 12.311805 / 89.3055 | 12.225172 / 89.9383 | 1.00709× |

All raw per-process rounds, full correctness records, certificate submissions and source/binary bindings are retained in the JSON.

- Two process runs per tag/shape, five correlated rounds each; ten pooled rounds are not ten independent trials.
- 8192 uses random_unit_scale; 2048/4096 use random_block_scales.
- This native packed-input benchmark excludes FP32 quantization, Python overhead and training.
- No NVML samples are embedded by this runner; independent collection is required for clock/load interpretation.
- Kernel-path instrumentation is absent; public preparation routing is supported by the shared host guard and actual source binding.
