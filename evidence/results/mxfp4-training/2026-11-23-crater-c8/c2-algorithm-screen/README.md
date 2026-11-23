# C2 cuBLAS algorithm screening candidate

This is an independent experiment, not a production change or a full-pipeline
speed result. No GPU execution has been performed by the candidate author.

The CUDA TU includes the actual `bench_mxfp4.cu` (renaming its main) and actual
`mxfp4_gemm_certified.cu`, with certified variant 2 and tiled variant 1. It
reuses the production Input generator, normalization, certificate, cuBLAS
Context/handle, workspace and exact postscale kernel. The baseline and tiled
translation units are linked separately. There is no separate compilation of
the included certified TU, no reconstructed preparation, and no extra FP16
input allocation.

## Build and run on the verified node

Use a new directory for each build/run. Supply the frozen native C9/C10 source
directory (its C2 numerical implementation is unchanged); do not modify B or
the running native source. Building is CPU-only and can precede native completion.

```bash
python3 candidate.py build \
  --source-root /workspace/user/mxfp4-c9-c10-20261123T2128/source \
  --source-binding /workspace/user/mxfp4-c9-c10-inputs-20261123T2118/source-binding.json \
  --source-archive /workspace/user/mxfp4-c9-c10-inputs-20261123T2118/source.tar.gz \
  --nvcc /path/to/cuda-12.8/bin/nvcc \
  --output-directory /path/to/new-c2-algorithm-build

python3 candidate.py run \
  --build-directory /path/to/new-c2-algorithm-build \
  --run-directory /path/to/new-c2-algorithm-results \
  --prior-native-run /workspace/user/mxfp4-c9-c10-20261123T2128
```

The wrapper serially runs three separately labelled cases, each with seed
20261123: 4096³ random_block_scales, 8192³ random_block_scales, and 8192³
random_unit_scale. The 4096 block-scales fingerprint must match B's
`6a76b74ce5d1d307`. A real rejected certificate produces
`blocked_certificate`, exit 77, and no algorithm timings. In particular, the
8192 block-scales case can exceed the existing energy bound; no threshold or
input is silently changed. The unit-scale run is an independent experiment.

## Measurements and correctness gates

- Candidate list: DEFAULT, DEFAULT_TENSOR_OP and ALGO0_TENSOR_OP through
  ALGO15_TENSOR_OP. Optional `run --include-ordinary` adds ALGO0..23 and may
  need substantially more time; unsupported algorithms are explicitly
  non-passing, untimed records.
- Original C2 handle settings are checked: FP16 A/B, FP32 C, COMPUTE_32F,
  DEFAULT_MATH | DISALLOW_REDUCED_PRECISION_REDUCTION, host alpha=1/beta=0,
  TN layout, the original stream, and 4 MiB user workspace.
- The production C2 API first creates actual normalized buffers and runs the
  real certificate. Accepted/reject_flags=0 is mandatory for screening.
- Preserve the complete original C2 output; compare selected 256 values with
  the existing CPU ordered-MX32 oracle, and every output against the actual
  exact tiled kernel. Every candidate initial and final output is compared
  against this full reference. Equality to tiled follows through that full
  comparison, not through a sample.
- Each supported algorithm has one correctness probe, 100 warmups, then five
  rounds of ten GEMMs. Algorithm order rotates each round. Both CUDA events
  and synchronized wall samples are retained; wall time is primary.
- Before every timing round, poison and synchronize the shared output outside
  both endpoints. After the final timed invocation, apply the original C2
  epilogue, then compare every actual output by bits, allowing signed-zero
  equivalence in the comparison and rejecting all nonfinite values. Separately,
  both the full C2 reference and every postscaled candidate must contain no
  negative zero, preserving C2's canonical positive-zero contract. The tiled
  reference retains its existing signed-zero equivalence. No replacement GEMM is run
  to generate the checked final output.
- The only measured scope is `normalize_gemm_compute`: GEMM on existing C2
  normalized FP16 arrays. Normalization, certificate, epilogue, fallback,
  allocation, transfers, poison and correctness are excluded. No packed-input
  total/pipeline throughput claim is made. A winner needs a new production
  candidate, full correctness gates, and an independent full-pipeline ABBA.

## Resource and evidence limits

Explicit peak device buffers are capped at 1 GiB; admission preserves 512 MiB
and at least 25% initially free memory. Free memory is checked again after
context/allocation and warmup/final validation. These counts include the C2
workspace and one shared device output, not duplicated ordinary-cuBLAS inputs.
Opaque CUDA/cuBLAS allocations are covered by runtime free-memory checks.
The run wrapper uses a 1800-second global budget and at most 600 seconds per
case, with cleanup of only its own process groups. It changes no services,
clocks, power settings, packages or production source. Root must schedule it
after other owned GPU work. Before GPU execution, the wrapper checks the
prior native run's final archive/completion/receipt and repeated baseline,
tiled and C2 full-suite gates; it then holds the same original-B framework
flock throughout its run and archival. If that lock is busy or final evidence
is absent, it fails without running GPU work. It does not claim exclusivity
against unrelated GPU users.

Compilation records actual source, candidate, compiler version/command,
compile logs and binary hashes. The run wrapper rechecks those bindings before
and after execution, validates report contents as well as return codes, and
archives its own binary, sources, provenance, raw JSONL/stdout/stderr,
completion and file hashes. An expected blocked case remains blocked in the
overall `completed_with_blocked_cases` result (exit 2).

## Primary API basis

- CUDA 12.8 cuBLAS enum/math/compute contracts:
  https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#cublasgemmalgo-t
- Workspace and stream reset behavior:
  https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#cublassetworkspace
- Split-K numerical behavior:
  https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#gemm-algorithms-numerical-behavior
- NVIDIA algorithm selection examples:
  https://github.com/NVIDIA/CUDALibrarySamples/blob/main/cuBLASLt/README.md

The explicit algorithm enum affects SM70 in CUDA 12.8. Its tensor-op values
are deprecated but still defined in this version. No CUDA 13 autotune enum,
TF32, FP8/block-scaling API, or reduced-precision accumulation is introduced.
