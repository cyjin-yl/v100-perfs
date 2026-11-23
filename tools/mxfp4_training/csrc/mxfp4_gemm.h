#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Packed row matrices A[M, padded_k] and B[N, padded_k] produce C=A*B^T.
// Each byte stores the even K element in its low nibble and the odd element
// in its high nibble. Each row has padded_k/32 independent E8M0 scale bytes.
// padded_k must be a multiple of 32. Logical K tails must already be zero.
//
// Each MX32 integer dot of q=2*E2M1 is exact, then scaled/rounded to FP32 by
// ldexpf(qsum, scale_a + scale_b - 256). FP32 block results accumulate in K
// order. Scale 0 means 2^-127; scale 255 propagates NaN even for a zero dot.
// Overflow/underflow and block cancellation follow this explicit FP32 order.
//
// Returns a cudaError_t value and enqueues on stream without synchronizing.
// Buffers must be contiguous, non-overlapping CUDA memory on the current
// device and remain alive until stream completes. Byte lengths/device/alias
// ownership are the caller's responsibility; the C ABI cannot infer them.
// M or N zero is a no-op. padded_k zero fills a nonempty C with zero, and the
// four input pointers may then be null. Otherwise every pointer is required.
int mxfp4_gemm_cuda(const uint8_t* a, const uint8_t* a_scales,
                   const uint8_t* b, const uint8_t* b_scales, float* out,
                   int64_t m, int64_t n, int64_t padded_k,
                   cudaStream_t stream);

int mxfp4_cuda_abi_version(void);
const char* mxfp4_cuda_backend(void);

#ifdef __cplusplus
}
#endif
