# Independent C2 preparation-cache audit

Audit: **verified**.

Actual experiment: **passed**; stage counts: {'passed': 26, 'failed': 0, 'blocked': 0}.

| Shape | Packed input mode | OFF wall ms / TFLOPS | ON wall ms / TFLOPS | ON/OFF throughput |
|---|---|---:|---:|---:|
| 2048³ | random_block_scales | 0.313418 / 54.8145 | 0.312234 / 55.0224 | 1.00379× |
| 4096³ | random_block_scales | 1.820288 / 75.5040 | 1.805337 / 76.1292 | 1.00828× |
| 8192³ | random_unit_scale | 12.311805 / 89.3055 | 12.225172 / 89.9383 | 1.00709× |

All raw per-process rounds, full correctness records, certificate submissions and source/binary bindings are retained in the JSON.

## Recorded NVML boundaries

The native records contain 36 snapshots: 12 environment snapshots and 24 before/after benchmark snapshots. All are retained in the JSON.

| Shape | Before/after SM clock MHz | Memory clock MHz | Power snapshot W | Reported limit W |
|---|---:|---:|---:|---:|
| 2048³ | 1530–1530 | 877–877 | 87.29–209.997 | 300–300 |
| 4096³ | 1522–1530 | 877–877 | 73.805–273.619 | 300–300 |
| 8192³ | 1327–1522 | 877–877 | 67.634–315.749 | 300–300 |

NVML snapshots cover each benchmark process; before is after all six backends warm up, after is after all backend timings and CPU output validation. They are not per-C2-interval measurements.
Observed snapshot extrema do not establish exclusive GPU use, continuous clocks, or active-kernel average power.

## Limits

- Two process runs per tag/shape, five correlated rounds each; ten pooled rounds are not ten independent trials.
- 8192 uses random_unit_scale; 2048/4096 use random_block_scales.
- This native packed-input benchmark excludes FP32 quantization, Python overhead and training.
- Native benchmark records contain environment and before/after NVML snapshots; no continuous clock, utilization or temperature trace is recorded. The after snapshot follows all backend timings and CPU validation, so snapshot power is not active-C2 average power.
- All three observed ON/OFF throughput gains are below 1%; with two process runs per tag/shape and changing 8192 boundary clocks, they are descriptive observations and do not justify changing the default OFF setting.
- Kernel-path instrumentation is absent; public preparation routing is supported by the shared host guard and actual source binding.
