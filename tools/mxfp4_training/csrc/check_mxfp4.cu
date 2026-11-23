// Standalone CUDA checks: no PyTorch/cuBLAS allocation and no service changes.
#include "mxfp4_gemm.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int reserve_mib = 128;
#ifndef MXFP4_CHECK_EXTRA_BYTES
#define MXFP4_CHECK_EXTRA_BYTES(m, n, k) uint64_t(0)
#endif

void checked(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
#define CUDA_CHECK(expression) checked((expression), #expression)

std::string json_string(const std::string& value) {
  std::string result = "\"";
  for (unsigned char c : value) {
    if (c == '\\' || c == '"') { result += '\\'; result += char(c); }
    else if (c == '\n') result += "\\n";
    else if (c == '\r') result += "\\r";
    else if (c == '\t') result += "\\t";
    else if (c < 32) result += '?';
    else result += char(c);
  }
  return result + '"';
}

template<class T> struct Buffer {
  T* data = nullptr;
  explicit Buffer(size_t count) {
    if (count) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
  }
  ~Buffer() { if (data) cudaFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};

struct Stream {
  cudaStream_t value = nullptr;
  Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { if (value) cudaStreamDestroy(value); }
};

// Literal host lookup is intentionally independent of the CUDA bit decoder.
constexpr int qtable[16] = {0, 1, 2, 3, 4, 6, 8, 12,
                           0, -1, -2, -3, -4, -6, -8, -12};

struct Case {
  int input_byte_offset = 0;
  int output_float_offset = 0;
  std::string name;
  int m, n, logical_k, padded_k, blocks;
  std::vector<uint8_t> a, as, b, bs;
  Case(std::string title, int rows, int cols, int k)
      : name(std::move(title)), m(rows), n(cols), logical_k(k),
        padded_k((k + 31) / 32 * 32), blocks(padded_k / 32),
        a(size_t(m) * padded_k / 2, 0), as(size_t(m) * blocks, 127),
        b(size_t(n) * padded_k / 2, 0), bs(size_t(n) * blocks, 127) {}
};

void set_code(std::vector<uint8_t>& bytes, int row, int k, int stride, unsigned code) {
  const size_t index = size_t(row) * stride / 2 + k / 2;
  const unsigned shift = unsigned(k & 1) * 4;
  bytes[index] = uint8_t((bytes[index] & ~(15u << shift)) | ((code & 15u) << shift));
}

int get_q(const std::vector<uint8_t>& bytes, int row, int k, int stride) {
  const unsigned code = (bytes[size_t(row) * stride / 2 + k / 2] >> ((k & 1) * 4)) & 15u;
  return qtable[code];
}

std::vector<float> reference(const Case& c) {
  std::vector<float> result(size_t(c.m) * c.n);
  for (int row = 0; row < c.m; ++row) for (int col = 0; col < c.n; ++col) {
    float sum = 0.0f;
    for (int block = 0; block < c.blocks; ++block) {
      int integer_sum = 0;
      for (int k = block * 32; k < (block + 1) * 32; ++k)
        integer_sum += get_q(c.a, row, k, c.padded_k) * get_q(c.b, col, k, c.padded_k);
      const int sa = c.as[size_t(row) * c.blocks + block];
      const int sb = c.bs[size_t(col) * c.blocks + block];
      const float value = (sa == 255 || sb == 255)
          ? std::numeric_limits<float>::quiet_NaN()
          : std::ldexp(float(integer_sum), sa + sb - 256);
      // Keep the declared FP32 block-rounding and ordered addition contract.
      volatile float rounded = sum + value;
      sum = rounded;
    }
    result[size_t(row) * c.n + col] = sum;
  }
  return result;
}

bool equal(float actual, float expected) {
  if (std::isnan(expected)) return std::isnan(actual);
  return actual == expected; // Includes signed infinities, subnormals and zeros.
}

uint64_t required_bytes(const Case& c) {
  return uint64_t(c.a.size()) + c.as.size() + c.b.size() + c.bs.size()
      + uint64_t(c.m) * c.n * sizeof(float) + 2 * c.input_byte_offset
      + sizeof(float) * c.output_float_offset
      + MXFP4_CHECK_EXTRA_BYTES(c.m, c.n, c.padded_k);
}

void require_free_memory(uint64_t bytes) {
  size_t available = 0, total = 0;
  CUDA_CHECK(cudaMemGetInfo(&available, &total));
  const uint64_t reserve = uint64_t(reserve_mib) << 20;
  if (available < reserve || bytes > uint64_t(available) - reserve)
    throw std::runtime_error("Insufficient free VRAM: need case allocations plus "
        + std::to_string(reserve_mib) + " MiB reserve; no services are stopped");
}

struct Totals {
  size_t cases = 0, values = 0, finite = 0, nan = 0, infinity = 0, subnormal = 0;
  // Track execution independently from correctness: the first launched case
  // may fail synchronization or numerical comparison before cases increments.
  bool kernel_launch_attempted = false;
  bool kernel_launch_submitted = false;
  bool kernel_execution_completed = false;
};

void run_case(const Case& c, cudaStream_t stream, Totals& totals,
              bool benchmark = false, int warmup = 2, int iters = 10) {
  require_free_memory(required_bytes(c));
  const std::vector<float> expected = reference(c);
  std::vector<float> actual(expected.size());
  Buffer<uint8_t> a(c.a.size() + c.input_byte_offset), as(c.as.size()),
                  b(c.b.size() + c.input_byte_offset), bs(c.bs.size());
  uint8_t* a_codes = a.data ? a.data + c.input_byte_offset : nullptr;
  uint8_t* b_codes = b.data ? b.data + c.input_byte_offset : nullptr;
  Buffer<float> out(expected.size() + c.output_float_offset);
  float* output = out.data + c.output_float_offset;
  auto copy = [&](uint8_t* target, const std::vector<uint8_t>& source) {
    if (!source.empty()) CUDA_CHECK(cudaMemcpyAsync(target, source.data(), source.size(), cudaMemcpyHostToDevice, stream));
  };
  copy(a_codes, c.a); copy(as.data, c.as); copy(b_codes, c.b); copy(bs.data, c.bs);
  auto launch = [&]() {
    totals.kernel_launch_attempted = true;
    CUDA_CHECK(cudaError_t(mxfp4_gemm_cuda(a_codes, as.data, b_codes, bs.data, output,
                                         c.m, c.n, c.padded_k, stream)));
    totals.kernel_launch_submitted = true;
  };
  launch();
  CUDA_CHECK(cudaMemcpyAsync(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  totals.kernel_execution_completed = true;
  size_t mismatches = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!equal(actual[i], expected[i])) {
      if (mismatches < 4) std::cerr << c.name << " index " << i << " expected "
                                   << expected[i] << " actual " << actual[i] << '\n';
      ++mismatches;
    }
    if (std::isnan(expected[i])) ++totals.nan;
    else if (std::isinf(expected[i])) ++totals.infinity;
    else {
      ++totals.finite;
      if (std::fpclassify(expected[i]) == FP_SUBNORMAL) ++totals.subnormal;
    }
  }
  if (mismatches) throw std::runtime_error(c.name + ": " + std::to_string(mismatches) + " mismatches");
  ++totals.cases;
  totals.values += actual.size();
  std::cout << "{\"type\":\"" << (benchmark ? "benchmark" : "correctness")
            << "\",\"case\":" << json_string(c.name)
            << ",\"M\":" << c.m << ",\"N\":" << c.n << ",\"logical_K\":" << c.logical_k
            << ",\"padded_K\":" << c.padded_k << ",\"values\":" << actual.size()
            << ",\"mismatches\":0,\"allocation_bytes\":" << required_bytes(c)
            << ",\"packed_input_byte_offset\":" << c.input_byte_offset
            << ",\"output_float_offset\":" << c.output_float_offset
            << ",\"reserve_mib\":" << reserve_mib
            << ",\"gpu_executed\":true,\"nondefault_stream\":true";
  if (benchmark) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t start = nullptr, end = nullptr;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(end, stream)); CUDA_CHECK(cudaEventSynchronize(end));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(end));
    std::cout << ",\"kernel_ms\":" << std::setprecision(9) << ms / iters
              << ",\"warmup\":" << warmup << ",\"iterations\":" << iters
              << ",\"timing_scope\":\"kernel_only_prepacked_inputs\",\"speedup\":null";
  }
  std::cout << "}\n";
}

Case random_case(int m, int n, int k, unsigned seed) {
  Case c("random_tails_" + std::to_string(m) + "_" + std::to_string(n)
             + "_" + std::to_string(k), m, n, k);
  std::mt19937 rng(seed);
  for (int r = 0; r < m; ++r) for (int j = 0; j < k; ++j)
    set_code(c.a, r, j, c.padded_k, rng() % 16);
  for (int r = 0; r < n; ++r) for (int j = 0; j < k; ++j)
    set_code(c.b, r, j, c.padded_k, rng() % 16);
  for (auto& s : c.as) s = uint8_t(107 + rng() % 41);
  for (auto& s : c.bs) s = uint8_t(107 + rng() % 41);
  return c;
}

void check_api_contract() {
  if (mxfp4_cuda_abi_version() != 1) throw std::runtime_error("Unexpected CUDA ABI");
  auto probe = [](int64_t m, int64_t n, int64_t k) {
    return mxfp4_gemm_cuda(nullptr, nullptr, nullptr, nullptr, nullptr, m, n, k, nullptr);
  };
  if (probe(-1, 1, 32) != cudaErrorInvalidValue ||
      probe(1, -1, 32) != cudaErrorInvalidValue ||
      probe(1, 1, -32) != cudaErrorInvalidValue ||
      probe(1, 1, 31) != cudaErrorInvalidValue ||
      probe(1, 1, 32) != cudaErrorInvalidValue ||
      probe(0, 7, 32) != cudaSuccess || probe(7, 0, 32) != cudaSuccess)
    throw std::runtime_error("CUDA ABI argument validation failed");
}

struct Options {
  int device = 0, warmup = 2, iters = 10, m = 64, n = 64, k = 97;
  bool allow_other_gpu = false, benchmark = false;
};

Options parse(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--allow-other-gpu") options.allow_other_gpu = true;
    else if (arg == "--benchmark") options.benchmark = true;
    else if (arg == "--correctness-only") options.benchmark = false;
    else if (arg == "--help") {
      std::cout << "check_mxfp4 [--device N] [--correctness-only] [--allow-other-gpu]\n"
                   "             [--reserve-mib N (default 128, minimum 64)]\n"
                   "             [--benchmark --warmup N --iters N --m N --n N --k N]\n"
                   "Default: bounded correctness cases, including multiple CTA tiles and odd-address inputs.\n";
      std::exit(0);
    } else {
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
      const std::string text = argv[++i];
      size_t consumed = 0;
      const long long value = std::stoll(text, &consumed);
      if (consumed != text.size() || value < 0 || value > 1048576)
        throw std::runtime_error("Invalid nonnegative bounded integer for " + arg);
      if (arg == "--device") options.device = int(value);
      else if (arg == "--reserve-mib") {
        if (value < 64) throw std::runtime_error("reserve-mib must be at least 64");
        reserve_mib = int(value);
      }
      else if (arg == "--warmup") options.warmup = int(value);
      else if (arg == "--iters") options.iters = int(value);
      else if (arg == "--m") options.m = int(value);
      else if (arg == "--n") options.n = int(value);
      else if (arg == "--k") options.k = int(value);
      else throw std::runtime_error("Unknown option " + arg);
    }
  }
  if (options.iters == 0 || options.m == 0 || options.n == 0)
    throw std::runtime_error("iters, M and N must be positive");
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  Totals totals;
  bool actual_v100 = false;
  try {
    const Options options = parse(argc, argv);
    CUDA_CHECK(cudaSetDevice(options.device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, options.device));
    actual_v100 = properties.major == 7 && properties.minor == 0
        && std::string(properties.name).find("V100") != std::string::npos;
    if (!actual_v100 && !options.allow_other_gpu)
      throw std::runtime_error("A real V100 is required by default; use --allow-other-gpu to label another GPU explicitly");
    if (properties.major < 7) throw std::runtime_error("Tensor Core compute capability >= 7.0 is required");
    CUDA_CHECK(cudaFree(nullptr));  // Initialize context before measuring free VRAM.
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    int driver_version = 0, runtime_version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    std::cout << "{\"type\":\"environment\",\"device\":" << options.device
              << ",\"name\":" << json_string(properties.name)
              << ",\"compute_capability\":\"" << properties.major << '.' << properties.minor
              << "\",\"actual_V100\":" << (actual_v100 ? "true" : "false")
              << ",\"driver_version\":" << driver_version << ",\"runtime_version\":" << runtime_version
              << ",\"free_bytes_after_context\":" << free_bytes << ",\"total_bytes\":" << total_bytes
              << ",\"backend\":" << json_string(mxfp4_cuda_backend())
              << ",\"gpu_executed\":false}\n";
    require_free_memory(65536);
    check_api_contract();
    Stream stream;

    Case code_pairs("all_16_by_16_e2m1_codes", 16, 16, 32);
    for (int r = 0; r < 16; ++r) for (int k = 0; k < 32; ++k) {
      set_code(code_pairs.a, r, k, 32, r);
      set_code(code_pairs.b, r, k, 32, r);
    }
    run_case(code_pairs, stream.value, totals);

    // Exhaust all 65,536 scale pairs in small 64x64 launches. Two nonzero
    // dot magnitudes expose both premature scale overflow and underflow.
    for (unsigned code : {1u, 4u}) for (int abase = 0; abase < 256; abase += 64)
      for (int bbase = 0; bbase < 256; bbase += 64) {
        Case c("all_scale_pairs_q" + std::to_string(qtable[code]) + "_"
                   + std::to_string(abase) + "_" + std::to_string(bbase), 64, 64, 1);
        for (int r = 0; r < 64; ++r) {
          set_code(c.a, r, 0, 32, code); c.as[r] = uint8_t(abase + r);
          set_code(c.b, r, 0, 32, code); c.bs[r] = uint8_t(bbase + r);
        }
        run_case(c, stream.value, totals);
      }

    Case zero_nan("zero_dot_nan_scale_propagation", 2, 2, 32);
    zero_nan.as = {0, 255}; zero_nan.bs = {254, 255};
    run_case(zero_nan, stream.value, totals);
    Case subnormal_ties("subnormal_positive_negative_round_to_even", 8, 1, 2);
    std::fill(subnormal_ties.as.begin(), subnormal_ties.as.end(), 53);
    subnormal_ties.bs[0] = 53; // qsum * 2^-150, halfway between FP32 subnormals.
    set_code(subnormal_ties.b, 0, 0, 32, 1);
    set_code(subnormal_ties.b, 0, 1, 32, 1);
    const unsigned low_codes[4] = {1, 3, 1, 1};
    const unsigned high_codes[4] = {0, 0, 4, 5}; // qsum = 1,3,5,7, then negatives.
    for (int row = 0; row < 8; ++row) {
      const unsigned sign = row >= 4 ? 8u : 0u;
      set_code(subnormal_ties.a, row, 0, 32, low_codes[row % 4] | sign);
      set_code(subnormal_ties.a, row, 1, 32, high_codes[row % 4] | sign);
    }
    run_case(subnormal_ties, stream.value, totals);
    Case overflow_cancel("opposite_overflow_blocks_follow_fp32_order", 1, 1, 64);
    overflow_cancel.as = {254, 254}; overflow_cancel.bs = {254, 254};
    set_code(overflow_cancel.a, 0, 0, 64, 7);
    set_code(overflow_cancel.b, 0, 0, 64, 7);
    set_code(overflow_cancel.a, 0, 32, 64, 15);
    set_code(overflow_cancel.b, 0, 32, 64, 7);
    run_case(overflow_cancel, stream.value, totals);

    for (const auto shape : {std::vector<int>{1, 1, 1}, {7, 13, 31}, {17, 19, 33},
                             {33, 49, 65}, {64, 64, 97}, {65, 97, 65},
                             {129, 67, 97}, {128, 128, 128}})
      run_case(random_case(shape[0], shape[1], shape[2], 20261123u + unsigned(shape[2])),
               stream.value, totals);
    Case odd_address = random_case(65, 97, 65, 20261124);
    odd_address.name = "odd_address_multitile_tail_inputs";
    odd_address.input_byte_offset = 1;
    run_case(odd_address, stream.value, totals);
    run_case(Case("zero_K_fills_output", 17, 19, 0), stream.value, totals);
    if (options.benchmark)
      run_case(random_case(options.m, options.n, options.k, 20261123), stream.value,
               totals, true, options.warmup, options.iters);

    std::cout << "{\"type\":\"summary\",\"passed\":true,\"gpu_executed\":true"
              << ",\"kernel_launch_attempted\":true,\"kernel_launch_submitted\":true"
              << ",\"kernel_execution_completed\":true,\"actual_V100\":"
              << (actual_v100 ? "true" : "false") << ",\"cases\":" << totals.cases
              << ",\"values\":" << totals.values << ",\"finite_values\":" << totals.finite
              << ",\"nan_values\":" << totals.nan << ",\"infinite_values\":" << totals.infinity
              << ",\"subnormal_values\":" << totals.subnormal
              << ",\"mismatches\":0,\"full_training_executed\":false,\"speedup\":null}\n";
    return 0;
  } catch (const std::exception& error) {
    // A successful enqueue without confirmed stream completion is unknown,
    // not evidence that the GPU did not execute. A later mismatch cannot
    // turn a successfully synchronized execution into gpu_executed=false.
    const char* gpu_executed = totals.kernel_execution_completed ? "true"
        : (totals.kernel_launch_submitted ? "null" : "false");
    std::cout << "{\"type\":\"summary\",\"passed\":false,\"gpu_executed\":"
              << gpu_executed
              << ",\"kernel_launch_attempted\":" << (totals.kernel_launch_attempted ? "true" : "false")
              << ",\"kernel_launch_submitted\":" << (totals.kernel_launch_submitted ? "true" : "false")
              << ",\"kernel_execution_completed\":" << (totals.kernel_execution_completed ? "true" : "false")
              << ",\"actual_V100\":"
              << (actual_v100 ? "true" : "false") << ",\"completed_cases\":" << totals.cases
              << ",\"error\":" << json_string(error.what()) << "}\n";
    std::cerr << error.what() << '\n';
    return 1;
  }
}
