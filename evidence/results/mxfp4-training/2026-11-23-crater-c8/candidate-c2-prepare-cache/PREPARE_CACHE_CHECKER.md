# C2 preparation cache microchecker

This standalone correctness executable includes the actual candidate
`mxfp4_gemm_certified.cu` into one translation unit. It directly invokes both
`prepare_rows_raw_cached_c2` and the unchanged `prepare_rows<false>`. The
certified source must not also be compiled as a separate translation unit.
There is no replacement kernel or duplicated CUDA decoder in the checker.

## Independent oracle and coverage

A scalar host oracle enumerates the 16 E2M1 signed integer q values. It computes
normalization minima over nonzero finite-scale blocks, counts every scale-255
block even if zero, computes capped integer energy mathematically with `ldexp`,
and converts exact `q * 2^delta` values through the host `__float2half_rn` routine.
All accepted half values are exact binary integers in the half range. The energy
cap is 2^48+1; integer sums below it are exactly representable even in binary64.
The maximum exponent used by host `ldexp` is 508, within binary64 range. The
oracle never calls the actual device decode/PRMT/DP4A/normalization helpers.

Every output byte in both compared kernels is initialized to the bitwise inverse
of its independent CPU expected byte. Thus even identically unwritten positions
in both kernels fail. Normalized half bits and all 16 RowData bytes must match
both the CPU oracle and each other. Static sizeof/offsetof assertions establish
that the compared RowData has no padding. Input bytes and 16-byte allocation
guards must remain unchanged. Work runs on a nonblocking CUDA stream.

The fixed suite has **323 direct cases plus four public submissions = 327 cases**:

| Group | Cases | Coverage |
|---|---:|---|
| All codes / uniform scales | 256 | Every E2M1 code under every scale byte 0..255, including invalid 255 |
| Row spans | 10 | 0, 1, 11, 12, 13, 23, 24, 25, 127, 254; finite-half, invalid-row and saturated-energy boundaries |
| K and raw alignment | 49 | K=32,64,128,4096,4128,8160,8192; A/B offsets (0,0),(1,2),(2,4),(4,8),(8,1),(0,1),(1,0) |
| Special rows | 8 | Signed zero, zero+NaN scale, active+NaN scale, zero blocks below row minimum, delta12 maximum half, only NaN blocks active, differing row bases, sparse active blocks, and all 256 packed-byte patterns (some share a fixture) |
| Public submissions | 4 | K=0,32,8192,8224; actual public Context workspace checked against original preparation and CPU oracle, plus exact closed-form unit-input FP32 output |

M=3 and N=2 throughout. The suite compares **1,079,840 normalized half values**,
**1,635 RowData rows**, and **24 public FP32 outputs**. These counts describe the
logical outputs; each is independently compared against both implementations.
Other data-dependent counts (zero halves, NaN-scale rows, saturated rows) are
reported only from actual execution.

Seven host checks call the same `use_c2_prepare_raw_cache` helper used by public
dispatch: K=0,32,64,4096,8192,8224,16384. Public K=0 and 8224 are explicitly
reported as cache-ineligible preparation fallback. Public K=32 and 8192 are
eligible. Kernel-path instrumentation is absent and is not claimed. Direct
invocations are expressly reported as actual cached-kernel invocations.

## Allocation and failure behavior

The ordinary admission rule requires 1 GiB free after CUDA context initialization,
with a fixed 512 MiB reserve. Explicit device allocations are capped at 8 MiB.
The largest direct fixture uses 186,032 bytes. The public K=8224 fixture is the
peak: **4,381,122 bytes** including the C2 4 MiB cuBLAS workspace, normalized
Context inputs, row metadata, certificate metadata, guarded inputs, original
preparation reference buffers, and one guarded FP32 output. The API workspace
query is used at runtime; peak allocation is tracked rather than inferred in the
JSON result. The reserve covers library/driver allocations outside explicit API
accounting. Post-create memory-query or reserve failures destroy the native
Context before unwinding; cleanup errors retain the original failure reason.

No timings or throughput claims are produced. The ordinary 69-call native
certified suite must also run independently on the opt-in build. This checker
is additional evidence, not a replacement for that suite.

## Compile and run

From a complete frozen source tree with this checker and the candidate applied:

```bash
/usr/local/cuda-12.8/bin/nvcc -O3 -std=c++17 --ftz=false -lineinfo -Xptxas=-v \
  '-gencode=arch=compute_70,code=[sm_70,compute_70]' \
  -DMXFP4_CERTIFIED_VARIANT=2 -DMXFP4_TILED_VARIANT=1 \
  -DMXFP4_CERTIFIED_PREPARE_RAW_CACHE=1 \
  tools/mxfp4_training/csrc/check_mxfp4_prepare_cache.cu \
  tools/mxfp4_training/csrc/mxfp4_gemm_tiled.cu \
  -lcublas -o /new/build/check_mxfp4_prepare_cache
/new/build/check_mxfp4_prepare_cache --device 0 --correctness-only
```

JSONL output is on stdout. Summary type is `prepare_cache_summary`; a passing
summary reports `cases=327`, `direct_cached_vs_original_cases=323`,
`public_dispatch_cases=4`, `public_fallback_preparation_cases=2`, zero mismatch
fields, and actual peak allocation. Missing/blocked GPU execution is never a
passing result. CUDA compilation and all GPU tests are delegated to root; none
were performed by the checker author locally.
