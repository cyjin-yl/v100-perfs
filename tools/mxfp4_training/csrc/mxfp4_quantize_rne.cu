// Opt-in FP32 -> MX32 RNE quantization; no change to stochastic rounding.
#include "mxfp4_quantize_rne.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {
constexpr unsigned kWarpMask = 0xffffffffu;
constexpr int kWarpsPerCTA = 8;
constexpr int kThreads = 32 * kWarpsPerCTA;

__device__ __forceinline__ unsigned rne_magnitude_code(float value) {
  // Midpoint ties alternate between the lower and upper even E2M1 code.
  // Values beyond the final threshold saturate to magnitude code 7 (6.0).
  return unsigned(value > .25f) + unsigned(value >= .75f)
      + unsigned(value > 1.25f) + unsigned(value >= 1.75f)
      + unsigned(value > 2.5f) + unsigned(value >= 3.5f)
      + unsigned(value > 5.f);
}

__device__ __forceinline__ int scale_exponent(unsigned maximum_abs_bits) {
  if (maximum_abs_bits == 0u) return 0;  // Both source zero signs count as zero.
  const int exponent_bits = int(maximum_abs_bits >> 23);
  // Every nonzero subnormal, and E=1 normal, clamps at the minimum E8M0.
  // E>=2 normal: floor(log2(amax))-2 = (E-127)-2 = E-129.
  return exponent_bits == 0 ? -127 : max(-127, min(127, exponent_bits - 129));
}

template <typename Index>
__global__ __launch_bounds__(kThreads) void quantize_mx32_rne(
    const float* __restrict__ x, uint8_t* __restrict__ codes,
    uint8_t* __restrict__ scales, int* __restrict__ error_flag,
    Index logical_k, Index stride_row, Index stride_col,
    Index blocks_per_row, Index total_blocks) {
  const int lane = int(threadIdx.x) & 31;
  const Index first = Index(blockIdx.x) * kWarpsPerCTA + (int(threadIdx.x) >> 5);
  const Index warp_stride = Index(gridDim.x) * kWarpsPerCTA;
  // The bound and loop condition are warp-uniform. All active warp lanes take
  // each shuffle, including K-tail lanes which explicitly contribute +0.
  for (Index block = first; block < total_blocks; block += warp_stride) {
    const Index row = block / blocks_per_row;
    const Index column = (block - row * blocks_per_row) * 32 + lane;
    unsigned bits = 0u;
    if (column < logical_k)
      bits = __float_as_uint(x[row * stride_row + column * stride_col]);
    const unsigned absolute_bits = bits & 0x7fffffffu;
    const bool nonfinite = absolute_bits >= 0x7f800000u;
    const unsigned invalid_lanes = __ballot_sync(kWarpMask, nonfinite);
    if (lane == 0 && invalid_lanes) atomicExch(error_flag, 1);
    // Do not perform floating arithmetic on nonfinite payloads. Such a call is
    // rejected through error_flag; its output representation is unspecified.
    unsigned maximum = nonfinite ? 0u : absolute_bits;
    #pragma unroll
    for (int distance = 16; distance > 0; distance >>= 1)
      maximum = max(maximum, __shfl_xor_sync(kWarpMask, maximum, distance));
    const int exponent = scale_exponent(maximum);
    const float magnitude = __uint_as_float(nonfinite ? 0u : absolute_bits);
    // --ftz=false is required. Every value in a threshold neighborhood scales
    // exactly into a normal FP32 value. Any underflow far below .25 can only
    // select magnitude zero, so it cannot change the RNE code.
    const float normalized = ldexpf(magnitude, -exponent);
    const unsigned code = rne_magnitude_code(normalized) | ((bits >> 28) & 8u);
    const unsigned next_code = __shfl_down_sync(kWarpMask, code, 1);
    if (!(lane & 1)) codes[block * 16 + lane / 2] = uint8_t(code | (next_code << 4));
    if (lane == 0) scales[block] = uint8_t(exponent + 127);
  }
}

bool checked_extent(int64_t rows, int64_t logical_k,
                    int64_t stride_row, int64_t stride_col, uint64_t* last_element) {
  const uint64_t limit = uint64_t(std::numeric_limits<ptrdiff_t>::max()) / sizeof(float);
  const uint64_t nr = uint64_t(rows - 1), nk = uint64_t(logical_k - 1);
  if ((nr && uint64_t(stride_row) > limit / nr)
      || (nk && uint64_t(stride_col) > limit / nk)) return false;
  const uint64_t r = nr * uint64_t(stride_row), k = nk * uint64_t(stride_col);
  if (r > limit - k) return false;
  *last_element = r + k;
  return true;
}
}  // namespace

extern "C" int mxfp4_quantize_rne_cuda(
    const float* x, uint8_t* codes, uint8_t* scales, int* error_flag,
    int64_t rows, int64_t logical_k, int64_t stride_row, int64_t stride_col,
    cudaStream_t stream) {
  if (rows <= 0 || logical_k <= 0 || stride_row < 0 || stride_col < 0
      || !x || !codes || !scales || !error_flag
      || (reinterpret_cast<uintptr_t>(x) % alignof(float))
      || (reinterpret_cast<uintptr_t>(error_flag) % alignof(int)))
    return int(cudaErrorInvalidValue);
  if (logical_k > std::numeric_limits<int64_t>::max() - 31)
    return int(cudaErrorInvalidValue);
  const uint64_t blocks_per_row = uint64_t((logical_k + 31) / 32);
  const uint64_t offset_limit = uint64_t(std::numeric_limits<ptrdiff_t>::max());
  // This also bounds total_blocks and every codes/scales index and prevents
  // overflow in the final grid-stride increment for both Index specializations.
  if (uint64_t(rows) > offset_limit / 16u / blocks_per_row)
    return int(cudaErrorInvalidValue);
  const uint64_t total_blocks = uint64_t(rows) * blocks_per_row;
  uint64_t last_element = 0;
  if (!checked_extent(rows, logical_k, stride_row, stride_col, &last_element))
    return int(cudaErrorInvalidValue);
  cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
  cudaError_t status = cudaStreamIsCapturing(stream, &capturing);
  if (status != cudaSuccess) return int(status);
  if (capturing != cudaStreamCaptureStatusNone) return int(cudaErrorStreamCaptureUnsupported);
  status = cudaMemsetAsync(error_flag, 0, sizeof(int), stream);
  if (status != cudaSuccess) return int(status);
  const unsigned grid = unsigned((total_blocks + kWarpsPerCTA - 1) / kWarpsPerCTA > 65535u
      ? 65535u : (total_blocks + kWarpsPerCTA - 1) / kWarpsPerCTA);
  const uint64_t small_limit = uint64_t(std::numeric_limits<int32_t>::max());
  const bool small = uint64_t(logical_k) <= small_limit && uint64_t(stride_row) <= small_limit
      && uint64_t(stride_col) <= small_limit && last_element <= small_limit
      && total_blocks * 16u <= small_limit;
  if (small) {
    quantize_mx32_rne<int32_t><<<grid, kThreads, 0, stream>>>(x, codes, scales, error_flag,
        int32_t(logical_k), int32_t(stride_row), int32_t(stride_col),
        int32_t(blocks_per_row), int32_t(total_blocks));
  } else {
    quantize_mx32_rne<int64_t><<<grid, kThreads, 0, stream>>>(x, codes, scales, error_flag,
        logical_k, stride_row, stride_col, int64_t(blocks_per_row), int64_t(total_blocks));
  }
  return int(cudaGetLastError());
}

extern "C" int mxfp4_quantize_rne_cuda_abi_version(void) { return 1; }
extern "C" const char* mxfp4_quantize_rne_cuda_backend(void) {
  return "sm70_mx32_rne_warp_fp32_v1";
}
