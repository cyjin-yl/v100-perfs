// Full E2M1 nibble-domain software MXFP4 GEMM for Volta Tensor Cores.
// This is a correctness baseline, not a native FP4 or packed-throughput claim.
#include "mxfp4_gemm.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

// Reuse the fragment mapping already exercised by the round-three impulse
// tests instead of maintaining an independent CUDA mapping implementation.
#include "../../../evidence/benchmarks/sm70-packing/study/simulation/round3/cuda/fragment_mapping.hpp"

namespace {

__device__ __forceinline__ int decode_q(unsigned code) {
  const unsigned magnitude = code & 7u;
  const int q = magnitude < 2u ? int(magnitude)
      : int(2u + (magnitude & 1u)) << (int(magnitude / 2u) - 1);
  return (code & 8u) ? -q : q;
}

// Four consecutive integer q values are exactly representable in half.
__device__ __forceinline__ uint2 load_four_q(const uint8_t* row,
                                            int64_t k, bool valid) {
  if (!valid) return make_uint2(0u, 0u);
  const unsigned lo = row[k / 2];
  const unsigned hi = row[k / 2 + 1];
  const half2 a = __halves2half2(__int2half_rn(decode_q(lo & 15u)),
                               __int2half_rn(decode_q(lo >> 4)));
  const half2 b = __halves2half2(__int2half_rn(decode_q(hi & 15u)),
                               __int2half_rn(decode_q(hi >> 4)));
  const __half2_raw ar = a;
  const __half2_raw br = b;
  return make_uint2(unsigned(ar.x) | (unsigned(ar.y) << 16),
                    unsigned(br.x) | (unsigned(br.y) << 16));
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

// Four full warps compute four 16x16 tiles. All lanes in an active warp
// execute every MMA, including masked rows/columns on boundary tiles.
__global__ __launch_bounds__(128) void gemm_mx32(
    const uint8_t* a, const uint8_t* a_scales,
    const uint8_t* b, const uint8_t* b_scales, float* out,
    int64_t m, int64_t n, int64_t padded_k,
    int64_t tiles_n, int64_t total_tiles) {
  const int lane = int(threadIdx.x) & 31;
  const int64_t tile = int64_t(blockIdx.x) * 4 + (int(threadIdx.x) >> 5);
  if (tile >= total_tiles) return;  // Warp-uniform return.
  const int64_t tile_row = (tile / tiles_n) * 16;
  const int64_t tile_col = (tile % tiles_n) * 16;
  const int64_t a_input_row = tile_row + a_row(lane);
  const int64_t b_input_row = tile_col + b_col(lane);
  const int64_t row_bytes = padded_k / 2;
  const int64_t blocks_k = padded_k / 32;
  // Do not form an out-of-bounds row pointer for masked lanes.
  const uint8_t* ap = a;
  const uint8_t* bp = b;
  if (a_input_row < m && padded_k) ap += a_input_row * row_bytes;
  if (b_input_row < n && padded_k) bp += b_input_row * row_bytes;
  float sum[8] = {};

  for (int64_t block = 0; block < blocks_k; ++block) {
    float integer_dot[8] = {};
    #pragma unroll
    for (int offset = 0; offset < 32; offset += 4) {
      const int64_t k = block * 32 + offset;
      const uint2 av = load_four_q(ap, k, a_input_row < m);
      const uint2 bv = load_four_q(bp, k, b_input_row < n);
      mma_four(integer_dot, av, bv);
    }
    // |qsum| <= 32*12*12 = 4608: integer HMMA products and every partial
    // sum are exactly representable in FP32 for the entire E2M1 domain.
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int64_t row = tile_row + c_row(lane, i);
      const int64_t col = tile_col + c_col(lane, i);
      if (row < m && col < n) {
        const int sa = a_scales[row * blocks_k + block];
        const int sb = b_scales[col * blocks_k + block];
        const float value = (sa == 255 || sb == 255)
            ? __int_as_float(0x7fc00000)
            : ldexpf(integer_dot[i], sa + sb - 256);
        sum[i] = __fadd_rn(sum[i], value);
      }
    }
  }
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int64_t row = tile_row + c_row(lane, i);
    const int64_t col = tile_col + c_col(lane, i);
    if (row < m && col < n) out[row * n + col] = sum[i];
  }
}

bool product_fits(int64_t a, int64_t b, size_t element_size) {
  // Bound pointer offsets by PTRDIFF_MAX as well as allocation byte count.
  const uint64_t limit = uint64_t(std::numeric_limits<ptrdiff_t>::max());
  return a == 0 || uint64_t(b) <= limit / element_size / uint64_t(a);
}

}  // namespace

extern "C" int mxfp4_gemm_cuda(
    const uint8_t* a, const uint8_t* a_scales,
    const uint8_t* b, const uint8_t* b_scales, float* out,
    int64_t m, int64_t n, int64_t padded_k, cudaStream_t stream) {
  if (m < 0 || n < 0 || padded_k < 0 || padded_k % 32 != 0)
    return int(cudaErrorInvalidValue);
  if (m == 0 || n == 0) return int(cudaSuccess);
  if (!out || (padded_k && (!a || !a_scales || !b || !b_scales)))
    return int(cudaErrorInvalidValue);
  if (!product_fits(m, n, sizeof(float)) ||
      !product_fits(m, padded_k / 2, 1) ||
      !product_fits(n, padded_k / 2, 1))
    return int(cudaErrorInvalidValue);
  // Division form avoids overflow when dimensions approach INT64_MAX.
  const int64_t tiles_m = m / 16 + (m % 16 != 0);
  const int64_t tiles_n = n / 16 + (n % 16 != 0);
  constexpr int64_t max_tiles = int64_t(2147483647) * 4;
  if (tiles_m > max_tiles / tiles_n) return int(cudaErrorInvalidValue);
  const int64_t total_tiles = tiles_m * tiles_n;
  const unsigned grid = unsigned(total_tiles / 4 + (total_tiles % 4 != 0));
  gemm_mx32<<<grid, 128, 0, stream>>>(
      a, a_scales, b, b_scales, out, m, n, padded_k, tiles_n, total_tiles);
  return int(cudaGetLastError());
}

extern "C" int mxfp4_cuda_abi_version(void) { return 1; }
extern "C" const char* mxfp4_cuda_backend(void) {
  return "sm70_e2m1_mx32_hmma_fp32_v1";
}
