#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

#define MXFP4_CERTIFIED_ABI_VERSION 1
#define MXFP4_CERTIFIED_CUBLAS_ERROR_BASE 10000
#define MXFP4_CERTIFIED_ENERGY_BOUND UINT64_C(281474976710656)
#define MXFP4_CERTIFIED_ENERGY_CAP UINT64_C(281474976710657)

// Compile-time experiments, all with the same certificate and exact fallback:
// MXFP4_CERTIFIED_VARIANT=0: verified scalar preparation / scalar epilogue.
// MXFP4_CERTIFIED_VARIANT=1: four-nibble PRMT/DP4A preparation, original epilogue.
// MXFP4_CERTIFIED_VARIANT=2: variant 1 plus fixed-row float4/masked epilogue.
// MXFP4_CERTIFIED_VARIANT=3: fused normalized HMMA, 64x64 CTA / BK64.
// MXFP4_CERTIFIED_VARIANT=4: fused normalized HMMA, 128x128 CTA / BK32.
// MXFP4_CERTIFIED_VARIANT=5: fused normalized HMMA, 128x128 CTA / BK64.
// MXFP4_CERTIFIED_VARIANT=6: 128x128 CTA / BK32 with raw-input register prefetch.
// MXFP4_CERTIFIED_VARIANT=7: variant 6 with 128 threads and 64x64 warp tiles.
// build_cuda.py defaults to 2; direct compilation without a define uses 0.
// Runtime autotuning is deliberately absent.

enum MXFP4CertifiedReject {
  MXFP4_CERT_REJECT_NAN_SCALE = 1u,
  MXFP4_CERT_REJECT_SCALE_SPAN = 2u,
  MXFP4_CERT_REJECT_ENERGY_BOUND = 4u,
  MXFP4_CERT_REJECT_LATTICE_RANGE = 8u
};

typedef struct MXFP4CertifiedInfo {
  uint32_t abi_version;
  int32_t device;
  int64_t m, n, padded_k;
  // Explicit cudaMalloc allocations. Variants 0..2 include normalized FP16
  // operands and a 4 MiB cuBLAS workspace; variants 3..7 own only row metadata,
  // row bases and the certificate, with no global normalized operand storage.
  // Excludes caller inputs/output and library/driver initialization overhead.
  uint64_t workspace_bytes;
  uint64_t max_energy_a, max_energy_b;
  uint32_t accepted, reject_flags;
  // Min/max of the PER-ROW normalization minima, including all-zero rows
  // assigned base 127. A row minimum ignores wholly q-zero MX32 blocks.
  // Thus lambda_min=min_scale_a+min_scale_b-256, and likewise lambda_max.
  int32_t min_scale_a, max_scale_a, min_scale_b, max_scale_b;
  // Largest within-row span among blocks containing at least one nonzero q.
  // All-zero rows have span zero. E8M0=255 is counted/rejected independently.
  uint32_t max_span_a, max_span_b;
  uint32_t nan_scales_a, nan_scales_b;
  uint32_t zero_rows_a, zero_rows_b;
  // If set, max_energy is the lower bound ENERGY_CAP, not an exact sum.
  uint32_t energy_saturated_a, energy_saturated_b;
  uint32_t metadata_valid;
  uint32_t reserved;
  uint64_t submitted_calls;
} MXFP4CertifiedInfo;

#ifdef __cplusplus
extern "C" {
#endif

// Status: 0 success; cudaError_t otherwise; a cuBLAS failure is encoded as
// MXFP4_CERTIFIED_CUBLAS_ERROR_BASE + cublasStatus_t. No exceptions cross ABI.
const char* mxfp4_certified_error_string(int status);

// Pure host calculation; performs no CUDA initialization or allocations.
// M/N must be positive, K nonnegative and a multiple of 32; dimensions must
// fit cuBLAS's int interface. Empty-M/N calls do not need a GEMM context.
int mxfp4_certified_workspace_bytes(int64_t m, int64_t n, int64_t padded_k,
                                   uint64_t* bytes);

// The context is bound to the current device and fixed shape, and owns row
// minima/energies, GPU certificate metadata and pinned diagnostic metadata.
// Variants 0..2 also own two normalized-half matrices, a cuBLAS handle and
// explicit cuBLAS workspace; variants 3..7 instead own compact row bases.
// Creation/destruction are outside per-call timing. This prototype binds a
// context to its CREATING HOST THREAD as well as device, so per-thread-default
// stream tokens cannot silently change meaning. Do not use it concurrently.
// A call may change streams only after the previous call
// has completed; callers are responsible for tensor lifetime/stream ordering.
// The last-used stream must remain valid until a later call changes it or the
// context is destroyed. A failing call may already have enqueued GPU work;
// destruction still waits for that stream before releasing owned buffers.
// CUDA Graph capture/replay is not supported by this context API yet.
int mxfp4_certified_create(int64_t m, int64_t n, int64_t padded_k, void** context);

// Same uint8 MXFP4 storage and exact MX32 arithmetic contract as the tiled
// backend. The certificate checks row scale span <=12, normalized integer
// row energies U,V with U*V <= 2^48, and common lattice exponent [-126,103].
// E8M0=255 always rejects, including zero blocks. Rejection dispatches the
// exact tiled path; it NEVER returns an uncertified fast-path value.
//
// With used_fast=nullptr this enqueues the complete pipeline asynchronously:
// GPU preparation/certificate, conditional fast path, conditional fallback.
// Variants 0..2 execute cuBLAS even on rejected calls using bounded finite
// safe inputs, then conditionally postscale; this unused work remains inside
// fallback timing. Variants 3..7 normalize directly into shared memory in a
// fused HMMA kernel, and rejected calls skip that kernel before input reads.
// With used_fast!=nullptr the function explicitly synchronizes and returns
// accepted as 0/1 as well.
int mxfp4_certified_gemm(void* context,
                        const uint8_t* a, const uint8_t* a_scales,
                        const uint8_t* b, const uint8_t* b_scales, float* out,
                        cudaStream_t stream, int* used_fast);

// Diagnostic query: copies a small GPU certificate and synchronizes the last
// used stream. Keep it OUTSIDE asynchronous performance timing. Values are
// for the most recently submitted call. Before any call metadata_valid=0.
int mxfp4_certified_get_info(void* context, MXFP4CertifiedInfo* info);

// Waits for the last used stream before freeing any owned resource. Call on
// the context's device, after ending all concurrent host use of the context.
int mxfp4_certified_destroy(void* context);

#ifdef __cplusplus
}
#endif
