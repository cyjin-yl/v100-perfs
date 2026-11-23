// Certificate-dependent fused MXFP4 normalization + all-K SM70 HMMA.
// Rejected calls return before reading inputs; caller enqueues exact fallback.
#include "mxfp4_gemm_fused.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "../../../evidence/benchmarks/sm70-packing/study/simulation/round3/cuda/fragment_mapping.hpp"

namespace {

__device__ __forceinline__ unsigned q_half_pair(unsigned packed) {
  const unsigned selector = ((packed & 7u) << 4) | ((packed & 0x70u) << 8);
  const unsigned magnitude = __byte_perm(0x42403c00u, 0x4a484644u, selector);
  return magnitude | ((packed & 8u) << 12) | ((packed & 0x80u) << 24);
}

__device__ __forceinline__ uint2 normalized_four(unsigned packed, int delta) {
  // The caller has established at least one nonzero q in this group, hence
  // its active block participated in row_min: 0<=delta<=12. Add exponent bits
  // only after that guard. Individual zero nibbles are masked to canonical+0.
  const unsigned shift = unsigned(delta) << 10;
  const unsigned both = shift | (shift << 16);
  unsigned lo = q_half_pair(packed & 255u) + both;
  unsigned hi = q_half_pair(packed >> 8) + both;
  const unsigned lo_mask = ((packed & 0x0007u) ? 0xffffu : 0u)
      | ((packed & 0x0070u) ? 0xffff0000u : 0u);
  const unsigned hi_mask = ((packed & 0x0700u) ? 0xffffu : 0u)
      | ((packed & 0x7000u) ? 0xffff0000u : 0u);
  return make_uint2(lo & lo_mask, hi & hi_mask);
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

__device__ __forceinline__ float scale_output(float value, unsigned a_base,
                                             unsigned b_base) {
  // The accepted certificate proves exponent[-126,103] and |value|<=2^24.
  const unsigned exponent_bits = (a_base + b_base - 129u) << 23;
  return value == 0.0f ? 0.0f : __fmul_rn(value, __uint_as_float(exponent_bits));
}

template<class Index, bool Tails, int TileM, int TileN, int BlockK,
         int Threads, int WarpM, int WarpN>
__global__ __launch_bounds__(Threads) void fused_certified(
    const uint8_t* __restrict__ a, const uint8_t* __restrict__ as,
    const uint8_t* __restrict__ b, const uint8_t* __restrict__ bs,
    float* __restrict__ out, Index m, Index n, Index k,
    const uint8_t* __restrict__ row_bases, const int* accepted, Index tiles_n) {
  if (*accepted == 0) return; // Uniform CTA decision BEFORE all barriers.
  constexpr int Vectors = BlockK / 4;
  constexpr int SharedStride = Vectors + 1;
  constexpr int FragM = WarpM / 16, FragN = WarpN / 16;
  constexpr int WarpsN = TileN / WarpN;
  static_assert((TileM / WarpM) * WarpsN * 32 == Threads, "warp tile coverage");
  static_assert(BlockK % 32 == 0, "MX32-aligned stage");
  // One uint2 padding vector per row rotates the half4 fragment loads across
  // all 32 SM70 shared-memory banks (row strides72/136bytes for BK32/BK64).
  __shared__ __align__(16) uint2 shared_a[TileM][SharedStride];
  __shared__ __align__(16) uint2 shared_b[TileN][SharedStride];
  __shared__ uint8_t base_a[TileM], base_b[TileN];
  const int tid = int(threadIdx.x), lane = tid & 31, warp = tid >> 5;
  const int warp_row = (warp / WarpsN) * WarpM;
  const int warp_col = (warp % WarpsN) * WarpN;
  const Index tile_row = (Index(blockIdx.x) / tiles_n) * TileM;
  const Index tile_col = (Index(blockIdx.x) % tiles_n) * TileN;
  const Index row_bytes = k / 2, blocks = k / 32;
  const bool aligned_a = (reinterpret_cast<uintptr_t>(a) & 1u) == 0;
  const bool aligned_b = (reinterpret_cast<uintptr_t>(b) & 1u) == 0;
  if (tid < TileM)
    base_a[tid] = k && (!Tails || tile_row + tid < m) ? row_bases[tile_row + tid] : 127;
  if (tid < TileN)
    base_b[tid] = k && (!Tails || tile_col + tid < n) ? row_bases[m + tile_col + tid] : 127;
  __syncthreads();

  float accum[FragM][FragN][8] = {};
  // Counting MX32 blocks avoids an int32 k+=BK overflow at the largest
  // legal Kp. Each source byte offset is separately bounded by dispatch.
  for (Index block = 0; block < blocks; block += BlockK / 32) {
    #pragma unroll
    for (int index = tid; index < TileM * Vectors; index += Threads) {
      const int local_row = index / Vectors, vector = index % Vectors;
      const Index row = tile_row + local_row;
      const Index mx_block = block + vector / 8;
      uint2 value = make_uint2(0, 0);
      if ((!Tails || row < m) && mx_block < blocks) {
        const Index address = row * row_bytes + block * 16 + vector * 2;
        const unsigned packed = aligned_a ? *reinterpret_cast<const uint16_t*>(a + address)
            : unsigned(a[address]) | (unsigned(a[address + 1]) << 8);
        if (packed & 0x7777u) {
          const int delta = int(as[row * blocks + mx_block]) - int(base_a[local_row]);
          value = normalized_four(packed, delta);
        }
      }
      shared_a[local_row][vector] = value;
    }
    #pragma unroll
    for (int index = tid; index < TileN * Vectors; index += Threads) {
      const int local_row = index / Vectors, vector = index % Vectors;
      const Index row = tile_col + local_row;
      const Index mx_block = block + vector / 8;
      uint2 value = make_uint2(0, 0);
      if ((!Tails || row < n) && mx_block < blocks) {
        const Index address = row * row_bytes + block * 16 + vector * 2;
        const unsigned packed = aligned_b ? *reinterpret_cast<const uint16_t*>(b + address)
            : unsigned(b[address]) | (unsigned(b[address + 1]) << 8);
        if (packed & 0x7777u) {
          const int delta = int(bs[row * blocks + mx_block]) - int(base_b[local_row]);
          value = normalized_four(packed, delta);
        }
      }
      shared_b[local_row][vector] = value;
    }
    __syncthreads();
    #pragma unroll
    for (int vector = 0; vector < Vectors; ++vector) {
      uint2 av[FragM], bv[FragN];
      #pragma unroll
      for (int fr = 0; fr < FragM; ++fr)
        av[fr] = shared_a[warp_row + fr * 16 + a_row(lane)][vector];
      #pragma unroll
      for (int fc = 0; fc < FragN; ++fc)
        bv[fc] = shared_b[warp_col + fc * 16 + b_col(lane)][vector];
      #pragma unroll
      for (int fr = 0; fr < FragM; ++fr) {
        #pragma unroll
        for (int fc = 0; fc < FragN; ++fc)
          mma_four(accum[fr][fc], av[fr], bv[fc]);
      }
    }
    // No warp overwrites an input stage while another still consumes it.
    __syncthreads();
  }

  const bool vector_output = (reinterpret_cast<uintptr_t>(out) & 7u) == 0 && n % 2 == 0;
  #pragma unroll
  for (int fr = 0; fr < FragM; ++fr) {
    #pragma unroll
    for (int fc = 0; fc < FragN; ++fc) {
      #pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        const int i = pair * 2;
        const int local_row = warp_row + fr * 16 + c_row(lane, i);
        const int local_col = warp_col + fc * 16 + c_col(lane, i);
        const Index row = tile_row + local_row, col = tile_col + local_col;
        if (!Tails || (row < m && col < n)) {
          const float x = scale_output(accum[fr][fc][i], base_a[local_row], base_b[local_col]);
          if (!Tails || col + 1 < n) {
            const float y = scale_output(accum[fr][fc][i + 1], base_a[local_row], base_b[local_col + 1]);
            if (vector_output)
              *reinterpret_cast<float2*>(out + row * n + col) = make_float2(x, y);
            else {
              out[row * n + col] = x;
              out[row * n + col + 1] = y;
            }
          } else {
            out[row * n + col] = x;
          }
        }
      }
    }
  }
}

// Each BK32 copy maps four half4 groups per operand to a thread. Keep the
// packed bits and raw scale in separate registers, so issuing the scale load
// does not depend on the packed load completing. No normalization happens in
// this prefetch: zero groups can have scales below their row's base.
template<class Index, bool Tails>
__device__ __forceinline__ void prefetch_raw32(
    const uint8_t* codes, const uint8_t* scales, Index tile_row, Index rows,
    Index row_bytes, Index blocks, Index block, int tid, bool aligned,
    uint2 (&raw)[4]) {
  #pragma unroll
  for (int copy = 0; copy < 4; ++copy) {
    const int index = tid + copy * 256;
    const int local_row = index / 8, vector = index % 8;
    const Index row = tile_row + local_row;
    unsigned packed = 0, scale = 0;
    if ((!Tails || row < rows) && block < blocks) {
      const Index address = row * row_bytes + block * 16 + vector * 2;
      packed = aligned ? *reinterpret_cast<const uint16_t*>(codes + address)
          : unsigned(codes[address]) | (unsigned(codes[address + 1]) << 8);
      scale = unsigned(scales[row * blocks + block]);
    }
    raw[copy] = make_uint2(packed, scale);
  }
}

// A separate kernel keeps the existing variants' source paths unchanged.
// Only raw packed/scale data are prefetched. The single shared-memory stage,
// HMMA order, and two barriers per K stage are identical to variant1.
template<class Index, bool Tails>
__global__ __launch_bounds__(256) void fused_certified_prefetch32(
    const uint8_t* __restrict__ a, const uint8_t* __restrict__ as,
    const uint8_t* __restrict__ b, const uint8_t* __restrict__ bs,
    float* __restrict__ out, Index m, Index n, Index k,
    const uint8_t* __restrict__ row_bases, const int* accepted, Index tiles_n) {
  if (*accepted == 0) return;
  __shared__ __align__(16) uint2 shared_a[128][9], shared_b[128][9];
  __shared__ uint8_t base_a[128], base_b[128];
  const int tid = int(threadIdx.x), lane = tid & 31, warp = tid >> 5;
  const int warp_row = (warp / 2) * 32, warp_col = (warp % 2) * 64;
  const Index tile_row = (Index(blockIdx.x) / tiles_n) * 128;
  const Index tile_col = (Index(blockIdx.x) % tiles_n) * 128;
  const Index row_bytes = k / 2, blocks = k / 32;
  const bool aligned_a = (reinterpret_cast<uintptr_t>(a) & 1u) == 0;
  const bool aligned_b = (reinterpret_cast<uintptr_t>(b) & 1u) == 0;
  if (tid < 128) {
    base_a[tid] = k && (!Tails || tile_row + tid < m) ? row_bases[tile_row + tid] : 127;
    base_b[tid] = k && (!Tails || tile_col + tid < n) ? row_bases[m + tile_col + tid] : 127;
  }
  __syncthreads();
  float accum[2][4][8] = {};
  uint2 raw_a[4], raw_b[4];
  if (blocks) {
    prefetch_raw32<Index, Tails>(a, as, tile_row, m, row_bytes, blocks, Index(0), tid, aligned_a, raw_a);
    prefetch_raw32<Index, Tails>(b, bs, tile_col, n, row_bytes, blocks, Index(0), tid, aligned_b, raw_b);
  }
  for (Index block = 0; block < blocks; ++block) {
    #pragma unroll
    for (int copy = 0; copy < 4; ++copy) {
      const int index = tid + copy * 256;
      const int local_row = index / 8, vector = index % 8;
      uint2 value = make_uint2(0, 0);
      if (raw_a[copy].x & 0x7777u)
        value = normalized_four(raw_a[copy].x, int(raw_a[copy].y) - int(base_a[local_row]));
      shared_a[local_row][vector] = value;
    }
    #pragma unroll
    for (int copy = 0; copy < 4; ++copy) {
      const int index = tid + copy * 256;
      const int local_row = index / 8, vector = index % 8;
      uint2 value = make_uint2(0, 0);
      if (raw_b[copy].x & 0x7777u)
        value = normalized_four(raw_b[copy].x, int(raw_b[copy].y) - int(base_b[local_row]));
      shared_b[local_row][vector] = value;
    }
    __syncthreads();
    // These reads write only thread-private raw registers. Shared still holds
    // the current stage, so its HMMA can execute while the next loads travel.
    if (block + 1 < blocks) {
      prefetch_raw32<Index, Tails>(a, as, tile_row, m, row_bytes, blocks, block + 1, tid, aligned_a, raw_a);
      prefetch_raw32<Index, Tails>(b, bs, tile_col, n, row_bytes, blocks, block + 1, tid, aligned_b, raw_b);
    }
    #pragma unroll
    for (int vector = 0; vector < 8; ++vector) {
      uint2 av[2], bv[4];
      #pragma unroll
      for (int fr = 0; fr < 2; ++fr)
        av[fr] = shared_a[warp_row + fr * 16 + a_row(lane)][vector];
      #pragma unroll
      for (int fc = 0; fc < 4; ++fc)
        bv[fc] = shared_b[warp_col + fc * 16 + b_col(lane)][vector];
      #pragma unroll
      for (int fr = 0; fr < 2; ++fr) {
        #pragma unroll
        for (int fc = 0; fc < 4; ++fc)
          mma_four(accum[fr][fc], av[fr], bv[fc]);
      }
    }
    // All warps must finish current-stage reads before next-stage stores.
    __syncthreads();
  }

  const bool vector_output = (reinterpret_cast<uintptr_t>(out) & 7u) == 0 && n % 2 == 0;
  #pragma unroll
  for (int fr = 0; fr < 2; ++fr) {
    #pragma unroll
    for (int fc = 0; fc < 4; ++fc) {
      #pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        const int i = pair * 2;
        const int local_row = warp_row + fr * 16 + c_row(lane, i);
        const int local_col = warp_col + fc * 16 + c_col(lane, i);
        const Index row = tile_row + local_row, col = tile_col + local_col;
        if (!Tails || (row < m && col < n)) {
          const float x = scale_output(accum[fr][fc][i], base_a[local_row], base_b[local_col]);
          if (!Tails || col + 1 < n) {
            const float y = scale_output(accum[fr][fc][i + 1], base_a[local_row], base_b[local_col + 1]);
            if (vector_output)
              *reinterpret_cast<float2*>(out + row * n + col) = make_float2(x, y);
            else {
              out[row * n + col] = x;
              out[row * n + col + 1] = y;
            }
          } else {
            out[row * n + col] = x;
          }
        }
      }
    }
  }
}

// Each BK32 copy maps eight half4 groups per operand to a thread. Keep the
// packed bits and raw scale in separate registers, so issuing the scale load
// does not depend on the packed load completing. No normalization happens in
// this prefetch: zero groups can have scales below their row's base.
template<class Index, bool Tails>
__device__ __forceinline__ void prefetch_raw32_warp64(
    const uint8_t* codes, const uint8_t* scales, Index tile_row, Index rows,
    Index row_bytes, Index blocks, Index block, int tid, bool aligned,
    uint2 (&raw)[8]) {
  #pragma unroll
  for (int copy = 0; copy < 8; ++copy) {
    const int index = tid + copy * 128;
    const int local_row = index / 8, vector = index % 8;
    const Index row = tile_row + local_row;
    unsigned packed = 0, scale = 0;
    if ((!Tails || row < rows) && block < blocks) {
      const Index address = row * row_bytes + block * 16 + vector * 2;
      packed = aligned ? *reinterpret_cast<const uint16_t*>(codes + address)
          : unsigned(codes[address]) | (unsigned(codes[address + 1]) << 8);
      scale = unsigned(scales[row * blocks + block]);
    }
    raw[copy] = make_uint2(packed, scale);
  }
}

// Final bounded candidate: four 64x64 warp tiles per 128x128 CTA.
// Each thread owns 128 FP32 accumulators and 32 raw prefetch registers.
// The shared stage, increasing-K arithmetic per output, and two barriers
// per K stage retain the same contract as the 256-thread prefetch kernel.
template<class Index, bool Tails>
__global__ __launch_bounds__(128) void fused_certified_prefetch32_warp64(
    const uint8_t* __restrict__ a, const uint8_t* __restrict__ as,
    const uint8_t* __restrict__ b, const uint8_t* __restrict__ bs,
    float* __restrict__ out, Index m, Index n, Index k,
    const uint8_t* __restrict__ row_bases, const int* accepted, Index tiles_n) {
  if (*accepted == 0) return;
  __shared__ __align__(16) uint2 shared_a[128][9], shared_b[128][9];
  __shared__ uint8_t base_a[128], base_b[128];
  const int tid = int(threadIdx.x), lane = tid & 31, warp = tid >> 5;
  const int warp_row = (warp / 2) * 64, warp_col = (warp % 2) * 64;
  const Index tile_row = (Index(blockIdx.x) / tiles_n) * 128;
  const Index tile_col = (Index(blockIdx.x) % tiles_n) * 128;
  const Index row_bytes = k / 2, blocks = k / 32;
  const bool aligned_a = (reinterpret_cast<uintptr_t>(a) & 1u) == 0;
  const bool aligned_b = (reinterpret_cast<uintptr_t>(b) & 1u) == 0;
  if (tid < 128) {
    base_a[tid] = k && (!Tails || tile_row + tid < m) ? row_bases[tile_row + tid] : 127;
    base_b[tid] = k && (!Tails || tile_col + tid < n) ? row_bases[m + tile_col + tid] : 127;
  }
  __syncthreads();
  float accum[4][4][8] = {};
  uint2 raw_a[8], raw_b[8];
  if (blocks) {
    prefetch_raw32_warp64<Index, Tails>(a, as, tile_row, m, row_bytes, blocks, Index(0), tid, aligned_a, raw_a);
    prefetch_raw32_warp64<Index, Tails>(b, bs, tile_col, n, row_bytes, blocks, Index(0), tid, aligned_b, raw_b);
  }
  for (Index block = 0; block < blocks; ++block) {
    #pragma unroll
    for (int copy = 0; copy < 8; ++copy) {
      const int index = tid + copy * 128;
      const int local_row = index / 8, vector = index % 8;
      uint2 value = make_uint2(0, 0);
      if (raw_a[copy].x & 0x7777u)
        value = normalized_four(raw_a[copy].x, int(raw_a[copy].y) - int(base_a[local_row]));
      shared_a[local_row][vector] = value;
    }
    #pragma unroll
    for (int copy = 0; copy < 8; ++copy) {
      const int index = tid + copy * 128;
      const int local_row = index / 8, vector = index % 8;
      uint2 value = make_uint2(0, 0);
      if (raw_b[copy].x & 0x7777u)
        value = normalized_four(raw_b[copy].x, int(raw_b[copy].y) - int(base_b[local_row]));
      shared_b[local_row][vector] = value;
    }
    __syncthreads();
    // These reads write only thread-private raw registers. Shared still holds
    // the current stage, so its HMMA can execute while the next loads travel.
    if (block + 1 < blocks) {
      prefetch_raw32_warp64<Index, Tails>(a, as, tile_row, m, row_bytes, blocks, block + 1, tid, aligned_a, raw_a);
      prefetch_raw32_warp64<Index, Tails>(b, bs, tile_col, n, row_bytes, blocks, block + 1, tid, aligned_b, raw_b);
    }
    #pragma unroll
    for (int vector = 0; vector < 8; ++vector) {
      uint2 av[4], bv[4];
      #pragma unroll
      for (int fr = 0; fr < 4; ++fr)
        av[fr] = shared_a[warp_row + fr * 16 + a_row(lane)][vector];
      #pragma unroll
      for (int fc = 0; fc < 4; ++fc)
        bv[fc] = shared_b[warp_col + fc * 16 + b_col(lane)][vector];
      #pragma unroll
      for (int fr = 0; fr < 4; ++fr) {
        #pragma unroll
        for (int fc = 0; fc < 4; ++fc)
          mma_four(accum[fr][fc], av[fr], bv[fc]);
      }
    }
    // All warps must finish current-stage reads before next-stage stores.
    __syncthreads();
  }

  const bool vector_output = (reinterpret_cast<uintptr_t>(out) & 7u) == 0 && n % 2 == 0;
  #pragma unroll
  for (int fr = 0; fr < 4; ++fr) {
    #pragma unroll
    for (int fc = 0; fc < 4; ++fc) {
      #pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        const int i = pair * 2;
        const int local_row = warp_row + fr * 16 + c_row(lane, i);
        const int local_col = warp_col + fc * 16 + c_col(lane, i);
        const Index row = tile_row + local_row, col = tile_col + local_col;
        if (!Tails || (row < m && col < n)) {
          const float x = scale_output(accum[fr][fc][i], base_a[local_row], base_b[local_col]);
          if (!Tails || col + 1 < n) {
            const float y = scale_output(accum[fr][fc][i + 1], base_a[local_row], base_b[local_col + 1]);
            if (vector_output)
              *reinterpret_cast<float2*>(out + row * n + col) = make_float2(x, y);
            else {
              out[row * n + col] = x;
              out[row * n + col + 1] = y;
            }
          } else {
            out[row * n + col] = x;
          }
        }
      }
    }
  }
}

bool product_fits(int64_t a, int64_t b, size_t element_size) {
  const uint64_t limit = uint64_t(std::numeric_limits<ptrdiff_t>::max());
  return a == 0 || uint64_t(b) <= limit / element_size / uint64_t(a);
}

template<class Index, int TileM, int TileN, int BlockK, int Threads, int WarpM, int WarpN>
void launch(const uint8_t* a, const uint8_t* as, const uint8_t* b, const uint8_t* bs,
            float* out, int64_t m, int64_t n, int64_t k, const uint8_t* bases,
            const int* accepted, int64_t tiles_n, unsigned grid, cudaStream_t stream) {
  if (m % TileM == 0 && n % TileN == 0)
    fused_certified<Index, false, TileM, TileN, BlockK, Threads, WarpM, WarpN><<<grid, Threads, 0, stream>>>(
        a, as, b, bs, out, Index(m), Index(n), Index(k), bases, accepted, Index(tiles_n));
  else
    fused_certified<Index, true, TileM, TileN, BlockK, Threads, WarpM, WarpN><<<grid, Threads, 0, stream>>>(
        a, as, b, bs, out, Index(m), Index(n), Index(k), bases, accepted, Index(tiles_n));
}

template<class Index>
void dispatch_variant(const uint8_t* a, const uint8_t* as, const uint8_t* b, const uint8_t* bs,
                      float* out, int64_t m, int64_t n, int64_t k, const uint8_t* bases,
                      const int* accepted, int variant, int64_t tiles_n, unsigned grid,
                      cudaStream_t stream) {
  if (variant == 0)
    launch<Index, 64, 64, 64, 128, 32, 32>(a, as, b, bs, out, m, n, k, bases, accepted, tiles_n, grid, stream);
  else if (variant == 1)
    launch<Index, 128, 128, 32, 256, 32, 64>(a, as, b, bs, out, m, n, k, bases, accepted, tiles_n, grid, stream);
  else if (variant == 2)
    launch<Index, 128, 128, 64, 256, 32, 64>(a, as, b, bs, out, m, n, k, bases, accepted, tiles_n, grid, stream);
  else if (variant == 3 && m % 128 == 0 && n % 128 == 0)
    fused_certified_prefetch32<Index, false><<<grid, 256, 0, stream>>>(
        a, as, b, bs, out, Index(m), Index(n), Index(k), bases, accepted, Index(tiles_n));
  else if (variant == 3)
    fused_certified_prefetch32<Index, true><<<grid, 256, 0, stream>>>(
        a, as, b, bs, out, Index(m), Index(n), Index(k), bases, accepted, Index(tiles_n));
  else if (m % 128 == 0 && n % 128 == 0)
    fused_certified_prefetch32_warp64<Index, false><<<grid, 128, 0, stream>>>(
        a, as, b, bs, out, Index(m), Index(n), Index(k), bases, accepted, Index(tiles_n));
  else
    fused_certified_prefetch32_warp64<Index, true><<<grid, 128, 0, stream>>>(
        a, as, b, bs, out, Index(m), Index(n), Index(k), bases, accepted, Index(tiles_n));
}

}  // namespace

extern "C" int mxfp4_gemm_fused_cuda(
    const uint8_t* a, const uint8_t* a_scales, const uint8_t* b, const uint8_t* b_scales,
    float* out, int64_t m, int64_t n, int64_t padded_k,
    const uint8_t* row_bases, const int* accepted, int variant, cudaStream_t stream) {
  if (m < 0 || n < 0 || padded_k < 0 || padded_k % 32 != 0 || variant < 0 || variant > 4)
    return int(cudaErrorInvalidValue);
  if (m == 0 || n == 0) return int(cudaSuccess);
  if (!out || !accepted || (padded_k && (!a || !a_scales || !b || !b_scales || !row_bases)))
    return int(cudaErrorInvalidValue);
  const int64_t index_limit = std::numeric_limits<ptrdiff_t>::max();
  if (m > index_limit - n || !product_fits(m, n, sizeof(float)) ||
      !product_fits(m, padded_k / 2, 1) || !product_fits(n, padded_k / 2, 1))
    return int(cudaErrorInvalidValue);
  const int tile = variant == 0 ? 64 : 128;
  const int64_t tiles_m = m / tile + (m % tile != 0);
  const int64_t tiles_n = n / tile + (n % tile != 0);
  constexpr int64_t max_grid = 2147483647;
  if (tiles_m > max_grid / tiles_n) return int(cudaErrorInvalidValue);
  const unsigned grid = unsigned(tiles_m * tiles_n);
  const int64_t int_limit = std::numeric_limits<int>::max();
  const bool small = m + n <= int_limit && padded_k <= int_limit
      && m <= int_limit / n
      && (padded_k == 0 || (m <= int_limit / (padded_k / 2)
                            && n <= int_limit / (padded_k / 2)));
  if (small)
    dispatch_variant<int>(a, a_scales, b, b_scales, out, m, n, padded_k, row_bases,
                          accepted, variant, tiles_n, grid, stream);
  else
    dispatch_variant<int64_t>(a, a_scales, b, b_scales, out, m, n, padded_k, row_bases,
                              accepted, variant, tiles_n, grid, stream);
  return int(cudaGetLastError());
}

extern "C" int mxfp4_fused_cuda_abi_version(void) { return 1; }
extern "C" const char* mxfp4_fused_cuda_backend(int variant) {
  switch (variant) {
    case 0: return "sm70_certified_fused64_bk64_warp32x32_v1";
    case 1: return "sm70_certified_fused128_bk32_warp32x64_v1";
    case 2: return "sm70_certified_fused128_bk64_warp32x64_v1";
    case 3: return "sm70_certified_fused128_bk32_prefetch_warp32x64_v1";
    case 4: return "sm70_certified_fused128_bk32_prefetch_warp64x64_v1";
    default: return "invalid_fused_variant";
  }
}
