// Compile with -I <patched-checkout>/tools/mxfp4_training/csrc.
// Include the actual candidate translation unit: this tests the real helpers,
// not a separately maintained approximation or copy of the decoder.
#include "mxfp4_gemm_fused.cu"

#include <cstdio>

namespace {
constexpr unsigned kPatterns = 65536;
constexpr unsigned kDeltas = 13;
constexpr unsigned kCases = kPatterns * kDeltas;
constexpr unsigned kHalfValues = kCases * 4;
constexpr unsigned kExpectedZeroHalves = kHalfValues / 8;

struct DecodeCounters {
  unsigned cases;
  unsigned mismatched_half_values;
  unsigned expected_zero_halves;
  unsigned negative_zero_halves;
  unsigned first_case;
  unsigned first_half;
  unsigned first_expected_bits;
  unsigned first_actual_bits;
};

__global__ void check_half2_decode(DecodeCounters* counters) {
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kCases) return;
  const unsigned packed = index & 0xffffu;
  const int delta = int(index >> 16);
  // delta is independently in [0,12] even for an all-zero standalone group.
  const uint2 expected = normalized_four(packed, delta);
  const unsigned factor = normalized_half2_factor(delta);
  const uint2 actual = normalized_four_half2_mul(packed, factor);
  unsigned mismatches = 0, zeros = 0, negative_zeros = 0;
  unsigned first_half = 0, first_expected = 0, first_actual = 0;
  #pragma unroll
  for (unsigned half = 0; half < 4; ++half) {
    const unsigned shift = (half & 1u) * 16;
    const unsigned expected_bits = ((half < 2 ? expected.x : expected.y) >> shift) & 0xffffu;
    const unsigned actual_bits = ((half < 2 ? actual.x : actual.y) >> shift) & 0xffffu;
    // Clear only the sign of an exact zero. Nonzero values, including their
    // sign and all exponent/mantissa bits, are compared without modification.
    const unsigned canonical_actual = (actual_bits & 0x7fffu) == 0 ? 0u : actual_bits;
    if ((expected_bits & 0x7fffu) == 0) ++zeros;
    if (actual_bits == 0x8000u) ++negative_zeros;
    if (canonical_actual != expected_bits) {
      if (mismatches == 0) {
        first_half = half;
        first_expected = expected_bits;
        first_actual = actual_bits;
      }
      ++mismatches;
    }
  }
  atomicAdd(&counters->cases, 1u);
  if (zeros) atomicAdd(&counters->expected_zero_halves, zeros);
  if (negative_zeros) atomicAdd(&counters->negative_zero_halves, negative_zeros);
  if (mismatches) {
    atomicAdd(&counters->mismatched_half_values, mismatches);
    if (atomicCAS(&counters->first_case, 0xffffffffu, index) == 0xffffffffu) {
      counters->first_half = first_half;
      counters->first_expected_bits = first_expected;
      counters->first_actual_bits = first_actual;
    }
  }
}

int report_error(const char* stage, cudaError_t error, bool attempted, bool completed) {
  std::printf("{\"kind\":\"mxfp4_half2_decode_microcheck\",\"passed\":false,"
              "\"status\":\"%s\",\"stage\":\"%s\",\"cuda_error\":%d,"
              "\"kernel_launch_attempted\":%s,\"gpu_executed\":%s}\n",
              attempted ? "error" : "blocked", stage, int(error),
              attempted ? "true" : "false", completed ? "true" : (attempted ? "null" : "false"));
  return attempted ? 1 : 77;
}
}  // namespace

int main() {
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  if (error != cudaSuccess || device_count == 0)
    return report_error("cuda_device", error, false, false);
  int device = 0;
  error = cudaGetDevice(&device);
  if (error != cudaSuccess) return report_error("current_device", error, false, false);
  cudaDeviceProp properties{};
  error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess) return report_error("device_properties", error, false, false);
  if (properties.major != 7 || properties.minor != 0) {
    std::printf("{\"kind\":\"mxfp4_half2_decode_microcheck\",\"passed\":false,"
                "\"status\":\"blocked\",\"reason\":\"requires_sm70\","
                "\"observed_major\":%d,\"observed_minor\":%d,\"gpu_executed\":false}\n",
                properties.major, properties.minor);
    return 77;
  }
  DecodeCounters result{};
  result.first_case = 0xffffffffu;
  DecodeCounters* device_result = nullptr;
  error = cudaMalloc(reinterpret_cast<void**>(&device_result), sizeof(result));
  if (error != cudaSuccess) return report_error("allocation", error, false, false);
  error = cudaMemcpy(device_result, &result, sizeof(result), cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    cudaFree(device_result);
    return report_error("initialize", error, false, false);
  }
  check_half2_decode<<<(kCases + 255) / 256, 256>>>(device_result);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    cudaFree(device_result);
    return report_error("launch", error, true, false);
  }
  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    cudaFree(device_result);
    return report_error("synchronize", error, true, false);
  }
  error = cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    cudaFree(device_result);
    return report_error("result_copy", error, true, true);
  }
  error = cudaFree(device_result);
  if (error != cudaSuccess) return report_error("release", error, true, true);
  const bool passed = result.cases == kCases && result.mismatched_half_values == 0
      && result.expected_zero_halves == kExpectedZeroHalves && result.first_case == 0xffffffffu;
  std::printf("{\"kind\":\"mxfp4_half2_decode_microcheck\",\"passed\":%s,"
              "\"status\":\"%s\",\"gpu_executed\":true,\"sm\":70,"
              "\"patterns\":%u,\"deltas\":%u,\"expected_cases\":%u,\"cases\":%u,"
              "\"half_values\":%u,\"nonzero_half_bitwise_comparisons\":%u,"
              "\"zero_half_sign_normalized_comparisons\":%u,"
              "\"negative_zero_halves_observed\":%u,\"mismatched_half_values\":%u,"
              "\"device_result_bytes\":%zu,\"first_mismatch\":",
              passed ? "true" : "false", passed ? "passed" : "failed", kPatterns, kDeltas,
              kCases, result.cases, result.cases * 4, result.cases * 4 - result.expected_zero_halves,
              result.expected_zero_halves, result.negative_zero_halves, result.mismatched_half_values,
              sizeof(result));
  if (result.first_case == 0xffffffffu) std::printf("null");
  else std::printf("{\"packed\":%u,\"delta\":%u,\"half\":%u,"
                   "\"expected_bits\":\"0x%04x\",\"actual_bits\":\"0x%04x\"}",
                   result.first_case & 0xffffu, result.first_case >> 16, result.first_half,
                   result.first_expected_bits, result.first_actual_bits);
  std::printf(",\"golden\":\"actual_C8_normalized_four\","
              "\"candidate\":\"actual_normalized_half2_factor_and_normalized_four_half2_mul\","
              "\"comparison\":\"integer_bits_no_float_tolerance\","
              "\"final_gemm_fp32_zero_checked_here\":false}\n");
  return passed ? 0 : 1;
}
