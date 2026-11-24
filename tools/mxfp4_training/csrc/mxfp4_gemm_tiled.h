#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

#ifdef __cplusplus
#include <cmath>
#include <cstring>

#if defined(__CUDACC__)
#define MXFP4_TILED_HD __host__ __device__ __forceinline__
#else
#define MXFP4_TILED_HD inline
#endif

// This helper is deliberately exposed for an independent arithmetic oracle.
// Precondition: qsum is an exact integer in [-4608,4608] (or signed zero).
// Every possible 32-element full-E2M1 q dot satisfies this precondition.
MXFP4_TILED_HD float mxfp4_tiled_float_bits(uint32_t bits) {
#if defined(__CUDA_ARCH__)
  return __uint_as_float(bits);
#else
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
#endif
}

MXFP4_TILED_HD float mxfp4_tiled_scaled_qsum(float qsum, uint8_t sa, uint8_t sb) {
  if (sa == 255 || sb == 255) return mxfp4_tiled_float_bits(0x7fc00000u);
  const int exponent = int(sa) + int(sb) - 256;
  if (unsigned(exponent + 126) <= 253u) {
    // 2^exponent is normal and finite for exponent in [-126,127].
    // Multiplication by that exact power of two agrees with ldexpf,
    // including correct overflow and signed zero. No exponent product is
    // materialized outside this proven interval.
    const float factor = mxfp4_tiled_float_bits(uint32_t(exponent + 127) << 23);
#if defined(__CUDA_ARCH__)
    return __fmul_rn(qsum, factor);
#else
    return qsum * factor;
#endif
  }
#if defined(__CUDA_ARCH__)
  return ldexpf(qsum, exponent);
#else
  return std::ldexp(qsum, exponent);
#endif
}

MXFP4_TILED_HD float mxfp4_tiled_accumulate(float sum, float qsum,
                                         uint8_t sa, uint8_t sb) {
  const int exponent = int(sa) + int(sb) - 256;
  if (sa != 255 && sb != 255 && unsigned(exponent + 126) <= 241u) {
    // For exponent in [-126,115], each nonzero integer qsum has unbiased
    // exponent 0..12, so qsum*2^exponent is normal, EXACT, and finite.
    // In this interval ONLY, FMA has exactly the same single final rounding
    // as the specified rounded block product followed by FP32 addition.
    // Extending this FMA interval to +127 would violate the contract when
    // the block product overflows before cancellation with a finite sum.
    const float factor = mxfp4_tiled_float_bits(uint32_t(exponent + 127) << 23);
#if defined(__CUDA_ARCH__)
    return __fmaf_rn(qsum, factor, sum);
#else
    return std::fma(qsum, factor, sum);
#endif
  }
  const float value = mxfp4_tiled_scaled_qsum(qsum, sa, sb);
#if defined(__CUDA_ARCH__)
  return __fadd_rn(sum, value);
#else
  return sum + value;
#endif
}

#undef MXFP4_TILED_HD
extern "C" {
#endif

// Same storage, arithmetic, stream, lifetime and error contract as
// mxfp4_gemm_cuda. This version cooperatively decodes one MX32 block into
// shared memory and reuses its half-valued INTEGER q operands across a
// 64x64 output tile. It does not materialize scaled FP16 input matrices.
// Compile-time experiment selection, with no runtime autotuning:
//   -DMXFP4_TILED_VARIANT=0: BK32, independently checked scale helpers (default)
//   -DMXFP4_TILED_VARIANT=1: BK32, block-wide normal-scale gate and FMA path
//   -DMXFP4_TILED_VARIANT=2: BK64 staging, same gate, two ORDERED MX32 products
// All variants retain the complete E2M1/E8M0 fallback and arithmetic contract.
int mxfp4_gemm_tiled_cuda(const uint8_t* a, const uint8_t* a_scales,
                         const uint8_t* b, const uint8_t* b_scales, float* out,
                         int64_t m, int64_t n, int64_t padded_k,
                         cudaStream_t stream);

// Conditional exact fallback for a GPU-certified caller. Each CTA reads the
// device-resident guard before ANY shared-memory barrier and returns if it is
// nonzero. The guard must be finalized earlier on the same stream, remain
// unchanged through completion, and not alias outputs. Original entry above
// is separately specialized without any guard read or conditional work.
int mxfp4_gemm_tiled_if_cuda(const uint8_t* a, const uint8_t* a_scales,
                            const uint8_t* b, const uint8_t* b_scales, float* out,
                            int64_t m, int64_t n, int64_t padded_k,
                            cudaStream_t stream, const int* skip_if_nonzero);

// Selective exact fallback for 128x128 full-K certificates. A 64x64 CTA
// returns when global_accepted is nonzero, or when its containing 128x128
// tile is accepted. Both device guards are finalized earlier on this stream.
// The tile mask has ceil(M/128)*ceil(N/128) entries in row-major order.
int mxfp4_gemm_tiled_tiles_if_cuda(const uint8_t* a, const uint8_t* a_scales,
                                  const uint8_t* b, const uint8_t* b_scales, float* out,
                                  int64_t m, int64_t n, int64_t padded_k,
                                  cudaStream_t stream, const int* global_accepted,
                                  const int* tile_accepted);

int mxfp4_tiled_cuda_abi_version(void);
const char* mxfp4_tiled_cuda_backend(void);

#ifdef __cplusplus
}
#endif
