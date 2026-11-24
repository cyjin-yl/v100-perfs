# Bounded post-workspace diagnostics

These are proposed diagnostics, not executed GPU results. The root agent owns
all remote execution and the exact original B GPU serialization lock.

## Installed-tool compatibility

Root's read-only version query observed Nsight Compute 2025.1.1.0 (build
35528883) and nvprof 12.8.90 (21) at `/usr/local/cuda/bin`.
NVIDIA's version-specific 2025.1 page lists GV100 as supported and documents
the 2025.1 update's CUDA 12.8 Update 1 support:

https://developer.nvidia.com/tools-overview/nsight-compute/get-started-2025_1

Nsight Compute 2025.3 explicitly removed Volta support. Current generic
Nsight Compute support tables therefore cannot be applied to installed
2025.1.1:

https://developer.nvidia.com/tools-overview/nsight-compute/get-started-2025_3

CUDA 12.8.1's profiler guide lists Volta as the last architecture fully
supported by nvprof. It can provide a legacy fallback in principle, but this
diagnostic controller uses the confirmed-compatible NCU and does not retry
hardware counters through another tool after a permission rejection:

https://docs.nvidia.com/cuda/archive/12.8.1/profiler-users-guide/index.html

## Exact range and unchanged native code

The isolated probe imports the immutable f74 workspace source, binds the
existing cache-OFF shared library and compile manifest, and uses the same
4096-square FP32 inputs and seed as the actual completed wrapper benchmark.
Both FP32 and packed inputs are hashed before and after. Initial and final
profiled outputs undergo full FP32 bit comparison against the validated tiled
implementation. Initial one call + ten warmup + one profiled call must produce
native submitted count 12 and an accepted zero-flag certificate.

Only one `workspace.gemm` and its same-stream completion wait occur between
CUDA profiler start and stop. Quantization, input preparation on the host,
reference GEMM, correctness copies, diagnostics and context destruction are
outside that range. Normal wrapper packed-shape validation and output
allocation remain in the called wrapper; they launch no quantization kernel.

PyTorch's official CUDA profiler source uses checked `cudart` calls for start
and stop:

https://github.com/pytorch/pytorch/blob/main/torch/cuda/profiler.py

The probe does not assume that main-branch code describes installed Torch
2.10. It reads the actual installed start/stop function source, checks its AST
against `check_error(cudart().cudaProfilerStart())` / `cudaProfilerStop()`,
and stores function and installed-module hashes before using those APIs.

## Minimal metrics and interpretation

The installed NCU GV100 metric catalog is captured first. A bounded list of
nine candidate metrics is intersected with actual available full names.
Tensor pipeline activity, active-warp occupancy and DRAM utilization are
required; issue activity, eligible warps, read/write bandwidth, SM frequency
and kernel duration are optional when absent from the actual catalog.

The CUDA 12.8.1 CUPTI manual's metric mapping table explicitly covers compute
capability 7.0 and supplies these Tensor Core/occupancy/DRAM names:

https://docs.nvidia.com/cupti/12.8.1/main/main.html#metrics-mapping-table

The command explicitly uses `--clock-control none`, `--cache-control none`,
`--profile-from-start off`, kernel replay, device zero, and application-only
profiling. It has a hard launch cap of ten; a result reaching that cap is not
accepted as proof of a complete pipeline. The actual CSV must include
prepare, certificate finalization, cuBLAS GEMM, postscale and conditional
tiled fallback, and must contain no quantization kernel.

NVIDIA describes the CLI's range, cache, clock, replay and configuration
semantics here:

https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html

Clock `none` is essential because profiler defaults may change clocks.
Cache `none` preserves normal application state but can make replay passes
see different cache states. All profiler times therefore remain diagnostics
and are excluded from unprofiled performance comparisons. Tensor pipeline
activity is not itself useful GEMM TFLOPS, and low occupancy alone does not
prove a slow kernel. High tensor activity with low DRAM utilization points
toward compute pressure; low issue/eligibility suggests latency or scheduling
questions that would require more evidence before causal claims.

The frozen controller requires `--config-file` to appear in the captured
installed help, then passes `--config-file off` to both metric catalog and
profile commands. It passes `--rename-kernels off` to the profile command
only when that option is advertised by the installed help. Missing config
isolation support blocks profiling. This avoids external section lists or
aliases changing the bounded measurement without writing any configuration
file.

## Separate memory correctness diagnostic

The actual cache-ON preparation microchecker binary can be run once under
Compute Sanitizer memcheck, with a 300-second owned-process deadline,
`--error-exitcode 99`, and full leak checking. Its original 327 numerical and
guard checks must still pass, in addition to the sanitizer's zero-error
summary. This addresses the concrete dynamic-shared-memory/unaligned-input
bounds risk introduced by the cache candidate.

https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html

## Permissions and evidence

An actual `ERR_NVGPUCTRPERM` or CUPTI insufficient-privilege message is recorded
as a blocked profiler diagnostic. No driver configuration, privilege, clocks,
power, service or package changes are attempted. The memory diagnostic remains
separate. NVIDIA documents this performance-counter access restriction:

https://developer.nvidia.com/ERR_NVGPUCTRPERM

The diagnostics archive records separate profile/memcheck outcomes, includes
tool versions, installed help, metric catalog, raw CSV/stderr, probe JSON,
NCU report, sanitizer log, source/helper/binary hashes and all stage receipts.
It never rewrites the earlier unprofiled performance evidence.
