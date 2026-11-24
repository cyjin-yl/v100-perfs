// Shared-memory full-domain MXFP4 SM70 GEMM, preserving the MX32 oracle.
// Performance must be measured on hardware; compilation is not a speed claim.
#include "mxfp4_gemm_tiled.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "../../../evidence/benchmarks/sm70-packing/study/simulation/round3/cuda/fragment_mapping.hpp"

#ifndef MXFP4_TILED_VARIANT
#define MXFP4_TILED_VARIANT 0
#endif
#if MXFP4_TILED_VARIANT < 0 || MXFP4_TILED_VARIANT > 2
#error "MXFP4_TILED_VARIANT must be 0 (BK32), 1 (BK32 gate), or 2 (BK64 gate)"
#endif

namespace {

constexpr int kTile = 64;
constexpr int kThreads = 128;
constexpr int kVectorsPerRow = 8;
constexpr int kStages = MXFP4_TILED_VARIANT == 2 ? 2 : 1;
constexpr bool kNormalGate = MXFP4_TILED_VARIANT != 0;

__device__ __forceinline__ unsigned decode_q_pair(unsigned byte) {
  // Half encodings of nonnegative INTEGER q are
  // 0000,3c00,4000,4200,4400,4600,4800,4a00. Their low bytes are all zero.
  // PRMT fills the upper byte of each half from this exact 8-entry LUT;
  // the other output bytes select LUT entry 0. No int-to-half conversion.
  const unsigned selector = ((byte & 0x07u) << 4) | ((byte & 0x70u) << 8);
  const unsigned magnitudes = __byte_perm(0x42403c00u, 0x4a484644u, selector);
  const unsigned signs = ((byte & 0x08u) << 12) | ((byte & 0x80u) << 24);
  return magnitudes | signs;
}

__device__ __forceinline__ uint2 decode_q_four(uint16_t packed) {
  return make_uint2(decode_q_pair(unsigned(packed) & 255u),
                    decode_q_pair(unsigned(packed) >> 8));
}

__device__ __forceinline__ void mma_four(float (&d)[8], uint2 a, uint2 b) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
}

template<class Index, bool Tails, int Stages, bool NormalGate, bool Conditional>
__global__ __launch_bounds__(kThreads) void gemm_tiled_mx32(
    const uint8_t* __restrict__ a, const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ b, const uint8_t* __restrict__ b_scales,
    float* __restrict__ out,
    Index m, Index n, Index padded_k, Index tiles_n, const int* skip_if_nonzero) {
  if constexpr (Conditional) {
    if (*skip_if_nonzero != 0) return; // Uniform for the entire CTA, pre-barrier.
  }
  // One padding vector gives row strides of 72/136 bytes, rotating by
  // 18/2 banks. The 16 distinct 8-byte fragment row loads cover all banks.
  constexpr int Vectors = kVectorsPerRow * Stages;
  constexpr int SharedStride = Vectors + 1;
  __shared__ __align__(16) uint2 shared_a[kTile][SharedStride];
  __shared__ __align__(16) uint2 shared_b[kTile][SharedStride];
  __shared__ uint8_t shared_as[Stages][kTile];
  __shared__ uint8_t shared_bs[Stages][kTile];

  const int tid = int(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const Index tile_row = (Index(blockIdx.x) / tiles_n) * kTile;
  const Index tile_col = (Index(blockIdx.x) % tiles_n) * kTile;
  const int warp_row = (warp >> 1) * 32;
  const int warp_col = (warp & 1) * 32;
  const Index row_bytes = padded_k / 2;
  const Index blocks_k = padded_k / 32;
  const bool aligned_a = (reinterpret_cast<uintptr_t>(a) & 1u) == 0;
  const bool aligned_b = (reinterpret_cast<uintptr_t>(b) & 1u) == 0;
  float sum[2][2][8] = {};

  for (Index first_block = 0; first_block < blocks_k; first_block += Stages) {
    // Each packed pair is fetched and decoded once per cooperative tile.
    // BK32 loads four 16-byte row segments per warp; BK64 loads two 32-byte
    // segments. All writes use aligned 64-bit shared-memory vectors.
    #pragma unroll
    for (int index = tid; index < 2 * kTile * Vectors; index += kThreads) {
      const int logical_row = index / Vectors;
      const int vector = index % Vectors;
      const bool use_a = logical_row < kTile;
      const int local_row = use_a ? logical_row : logical_row - kTile;
      const Index row = (use_a ? tile_row : tile_col) + local_row;
      uint2 decoded = make_uint2(0, 0);
      if ((!Tails || row < (use_a ? m : n))
          && first_block + vector / kVectorsPerRow < blocks_k) {
        const uint8_t* source = use_a ? a : b;
        const Index address = row * row_bytes + first_block * 16 + vector * 2;
        // cudaMalloc/PyTorch storage takes the aligned single-load path.
        // Preserve the original uint8 ABI for a contiguous odd-offset view.
        const uint16_t packed = (use_a ? aligned_a : aligned_b)
            ? *reinterpret_cast<const uint16_t*>(source + address)
            : uint16_t(unsigned(source[address]) | (unsigned(source[address + 1]) << 8));
        decoded = decode_q_four(packed);
      }
      if (use_a) shared_a[local_row][vector] = decoded;
      else shared_b[local_row][vector] = decoded;
    }
    bool local_slow_scale = false;
    if (tid < kTile) {
      const Index ar = tile_row + tid;
      const Index br = tile_col + tid;
      #pragma unroll
      for (int stage = 0; stage < Stages; ++stage) {
        const Index block = first_block + stage;
        const uint8_t sa = ((!Tails || ar < m) && block < blocks_k)
            ? a_scales[ar * blocks_k + block] : 127;
        const uint8_t sb = ((!Tails || br < n) && block < blocks_k)
            ? b_scales[br * blocks_k + block] : 127;
        shared_as[stage][tid] = sa;
        shared_bs[stage][tid] = sb;
        if constexpr (NormalGate)
          local_slow_scale |= unsigned(int(sa) - 65) > 120u
              || unsigned(int(sb) - 65) > 120u;
      }
    }
    bool normal_scales = false;
    if constexpr (NormalGate) {
      // ALL threads execute this block-wide barrier/reduction, including
      // masked rows and the inactive half of the scale-loader threads.
      // The sufficient interval [65,185] implies pair exponent [-126,114].
      normal_scales = __syncthreads_or(int(local_slow_scale)) == 0;
    } else {
      __syncthreads();
    }

    // Keep one copy of the uncommon full-range scale fallback in BK64.
    #pragma unroll 1
    for (int stage = 0; stage < Stages; ++stage) {
      if (first_block + stage >= blocks_k) continue; // Block-uniform tail.
      // Preserve EACH MX32 dot -> scale -> sum in original K order.
      // Stages=2 shares only memory staging/barriers, never arithmetic.
      float dot[2][2][8] = {};
      #pragma unroll
      for (int kvector = 0; kvector < kVectorsPerRow; ++kvector) {
        const int vector = stage * kVectorsPerRow + kvector;
        const uint2 av0 = shared_a[warp_row + a_row(lane)][vector];
        const uint2 av1 = shared_a[warp_row + 16 + a_row(lane)][vector];
        const uint2 bv0 = shared_b[warp_col + b_col(lane)][vector];
        const uint2 bv1 = shared_b[warp_col + 16 + b_col(lane)][vector];
        mma_four(dot[0][0], av0, bv0);
        mma_four(dot[0][1], av0, bv1);
        mma_four(dot[1][0], av1, bv0);
        mma_four(dot[1][1], av1, bv1);
      }

      // A lane has four distinct output rows and eight distinct columns.
      // Keep 32-bit registers so the normal path can reuse shifted exponents.
      uint32_t sa[2][2], sb[2][4];
      #pragma unroll
      for (int fragment = 0; fragment < 2; ++fragment) {
        #pragma unroll
        for (int r = 0; r < 2; ++r)
          sa[fragment][r] = shared_as[stage][warp_row + fragment * 16 + c_row(lane, r * 2)];
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int i = (c & 1) + ((c & 2) << 1);
          sb[fragment][c] = shared_bs[stage][warp_col + fragment * 16 + c_col(lane, i)];
        }
      }
      if (NormalGate && normal_scales) {
        // Integer arithmetic is unsigned: negative exponent offsets wrap
        // modulo 2^32, then combine to a valid positive normal FP32 factor.
        #pragma unroll
        for (int f = 0; f < 2; ++f) {
          #pragma unroll
          for (int r = 0; r < 2; ++r) sa[f][r] = (sa[f][r] - 129u) << 23;
          #pragma unroll
          for (int c = 0; c < 4; ++c) sb[f][c] <<= 23;
        }
        #pragma unroll
        for (int fr = 0; fr < 2; ++fr) {
          #pragma unroll
          for (int fc = 0; fc < 2; ++fc) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
              const int r = (i >> 1) & 1;
              const int c = (i & 1) | ((i >> 1) & 2);
              const float factor = __uint_as_float(sa[fr][r] + sb[fc][c]);
              sum[fr][fc][i] = __fmaf_rn(dot[fr][fc][i], factor, sum[fr][fc][i]);
            }
          }
        }
      } else {
        #pragma unroll
        for (int fr = 0; fr < 2; ++fr) {
          #pragma unroll
          for (int fc = 0; fc < 2; ++fc) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
              const int r = (i >> 1) & 1;
              const int c = (i & 1) | ((i >> 1) & 2);
              sum[fr][fc][i] = mxfp4_tiled_accumulate(
                  sum[fr][fc][i], dot[fr][fc][i], uint8_t(sa[fr][r]), uint8_t(sb[fc][c]));
            }
          }
        }
      }
    }
    // No warp may overwrite the shared MX32 block while another reads it.
    __syncthreads();
  }

  #pragma unroll
  for (int fr = 0; fr < 2; ++fr) {
    #pragma unroll
    for (int fc = 0; fc < 2; ++fc) {
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        const Index row = tile_row + warp_row + fr * 16 + c_row(lane, i);
        const Index col = tile_col + warp_col + fc * 16 + c_col(lane, i);
        if (!Tails || (row < m && col < n)) out[row * n + col] = sum[fr][fc][i];
      }
    }
  }
}

bool product_fits(int64_t a, int64_t b, size_t element_size) {
  const uint64_t limit = uint64_t(std::numeric_limits<ptrdiff_t>::max());
  return a == 0 || uint64_t(b) <= limit / element_size / uint64_t(a);
}

template<class Index, bool Conditional>
void launch(const uint8_t* a, const uint8_t* as, const uint8_t* b,
            const uint8_t* bs, float* out, int64_t m, int64_t n, int64_t k,
            int64_t tiles_n, unsigned grid, cudaStream_t stream, const int* skip_if_nonzero) {
  if (m % kTile == 0 && n % kTile == 0)
    gemm_tiled_mx32<Index, false, kStages, kNormalGate, Conditional><<<grid, kThreads, 0, stream>>>(
        a, as, b, bs, out, Index(m), Index(n), Index(k), Index(tiles_n), skip_if_nonzero);
  else
    gemm_tiled_mx32<Index, true, kStages, kNormalGate, Conditional><<<grid, kThreads, 0, stream>>>(
        a, as, b, bs, out, Index(m), Index(n), Index(k), Index(tiles_n), skip_if_nonzero);
}

int dispatch_tiled(
    const uint8_t* a, const uint8_t* a_scales,
    const uint8_t* b, const uint8_t* b_scales, float* out,
    int64_t m, int64_t n, int64_t padded_k, cudaStream_t stream, const int* skip_if_nonzero) {
  if (m < 0 || n < 0 || padded_k < 0 || padded_k % 32 != 0)
    return int(cudaErrorInvalidValue);
  if (m == 0 || n == 0) return int(cudaSuccess);
  if (!out || (padded_k && (!a || !a_scales || !b || !b_scales)))
    return int(cudaErrorInvalidValue);
  if (!product_fits(m, n, sizeof(float)) ||
      !product_fits(m, padded_k / 2, 1) ||
      !product_fits(n, padded_k / 2, 1))
    return int(cudaErrorInvalidValue);
  const int64_t tiles_m = m / kTile + (m % kTile != 0);
  const int64_t tiles_n = n / kTile + (n % kTile != 0);
  constexpr int64_t max_grid = 2147483647;
  if (tiles_m > max_grid / tiles_n) return int(cudaErrorInvalidValue);
  const unsigned grid = unsigned(tiles_m * tiles_n);
  // Common pretraining shapes fit signed 32-bit element offsets. Keep that
  // fast indexing path, without narrowing valid large caller dimensions.
  const int64_t int_limit = std::numeric_limits<int>::max();
  const bool small = m <= int_limit && n <= int_limit && padded_k <= int_limit
      && m <= int_limit / n
      && (padded_k == 0 || (m <= int_limit / (padded_k / 2)
                            && n <= int_limit / (padded_k / 2)));
  if (skip_if_nonzero) {
    if (small)
      launch<int, true>(a, a_scales, b, b_scales, out, m, n, padded_k, tiles_n, grid, stream, skip_if_nonzero);
    else
      launch<int64_t, true>(a, a_scales, b, b_scales, out, m, n, padded_k, tiles_n, grid, stream, skip_if_nonzero);
  } else {
    if (small)
      launch<int, false>(a, a_scales, b, b_scales, out, m, n, padded_k, tiles_n, grid, stream, nullptr);
    else
      launch<int64_t, false>(a, a_scales, b, b_scales, out, m, n, padded_k, tiles_n, grid, stream, nullptr);
  }
  return int(cudaGetLastError());
}

}  // namespace

extern "C" int mxfp4_gemm_tiled_cuda(
    const uint8_t* a, const uint8_t* a_scales,
    const uint8_t* b, const uint8_t* b_scales, float* out,
    int64_t m, int64_t n, int64_t padded_k, cudaStream_t stream) {
  return dispatch_tiled(a, a_scales, b, b_scales, out, m, n, padded_k, stream, nullptr);
}

extern "C" int mxfp4_gemm_tiled_if_cuda(
    const uint8_t* a, const uint8_t* a_scales,
    const uint8_t* b, const uint8_t* b_scales, float* out,
    int64_t m, int64_t n, int64_t padded_k, cudaStream_t stream, const int* skip_if_nonzero) {
  if (!skip_if_nonzero) return int(cudaErrorInvalidValue);
  return dispatch_tiled(a, a_scales, b, b_scales, out, m, n, padded_k, stream, skip_if_nonzero);
}

extern "C" int mxfp4_tiled_cuda_abi_version(void) { return 1; }
extern "C" const char* mxfp4_tiled_cuda_backend(void) {
#if MXFP4_TILED_VARIANT == 0
  return "sm70_e2m1_mx32_shared64_hmma_fp32_v1";
#elif MXFP4_TILED_VARIANT == 1
  return "sm70_e2m1_mx32_shared64_bk32_scalegate_v1";
#else
  return "sm70_e2m1_mx32_shared64_bk64_scalegate_v1";
#endif
}
