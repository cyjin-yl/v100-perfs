#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Conditional SM70 Tensor Core GEMM with on-chip MXFP4 normalization.
//
// This is a BUILDING BLOCK for the certified backend, not an independent
// full-domain MXFP4 API. Before launch, the caller must establish the exact
// certificate described in mxfp4_gemm_certified.h for these immutable inputs:
// no NaN scales, active-row span <=12, normalized row-energy product <=2^48,
// and every output lattice exponent in [-126,103]. row_bases contains the
// same per-row normalization minima used in that certificate, A[M] then
// B_rows[N]. Minima ignore wholly q-zero blocks; all-zero rows use base127.
//
// accepted is a device-resident int finalized earlier on the same stream.
// Every CTA returns before any barrier or operand access when it is zero.
// The caller must enqueue its exact fallback for rejected inputs. This call
// does NOT generate the certificate or modify its metadata.
//
// Accepted inputs are normalized directly into shared memory, continuously
// accumulated over all K in FP32 HMMA, then scaled once on output. No global
// FP16 matrix or intermediate FP32 output is materialized. q-zero values
// become half +0 before exponent manipulation; output zero is canonical +0.
// Numeric results equal the ordered MX32 contract under the certificate.
//
// Storage is unchanged: low nibble = even K, uint8 scales per row/MX32,
// Kp%32==0 and logical tail positions contain canonical positive-zero codes.
// A/B byte offsets need not be aligned; output needs only float alignment.
// M/N may have tails. Kp==0 writes +0 for accepted calls without touching
// input codes/scales/row_bases, which may then be null. M/N==0 is a no-op.
// Buffers and accepted/row_bases must remain valid and immutable until stream
// completion, be on the current device, and not overlap output storage.
//
// Bounded runtime experiments (no automatic tuning or silent substitution):
//   variant0: CTA64x64,   BK64, 128 threads, warp32x32.
//   variant1: CTA128x128, BK32, 256 threads, warp32x64.
//   variant2: CTA128x128, BK64, 256 threads, warp32x64.
//   variant3: variant1 geometry, next-stage packed/scale register prefetch.
//             One shared stage and the same two barriers per K stage.
//   variant4: CTA128x128, BK32, 128 threads, warp64x64, raw register prefetch.
//             128 FP32 accumulators/thread, same shared stage and barriers.
// All variants implement the same certificate-dependent arithmetic.
// Returns cudaError_t as int after enqueue; does not synchronize.
int mxfp4_gemm_fused_cuda(const uint8_t* a, const uint8_t* a_scales,
                         const uint8_t* b, const uint8_t* b_scales, float* out,
                         int64_t m, int64_t n, int64_t padded_k,
                         const uint8_t* row_bases, const int* accepted,
                         int variant, cudaStream_t stream);

int mxfp4_fused_cuda_abi_version(void);
const char* mxfp4_fused_cuda_backend(int variant);

#ifdef __cplusplus
}
#endif
