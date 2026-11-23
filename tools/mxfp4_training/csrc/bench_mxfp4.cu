// Fair packed-input MXFP4 benchmark; no service, power, clock or GPU changes.
// Build (CUDA 12.x): nvcc -O3 -std=c++17 --ftz=false -arch=sm_70
//   mxfp4_gemm.cu bench_mxfp4.cu -lcublas -ldl -o bench_mxfp4
// Add the tiled translation unit and -DMXFP4_BENCH_TILED to compare that API.
// MXFP4_BENCH_CERTIFIED additionally measures the explicit certified workspace
// path, including preparation, GPU certificate, its selected compute backend,
// output scaling and fallback. Variants >=3 fuse normalized decode with HMMA.
// Both CUDA event intervals and synchronized host wall time are reported.
// Every TFLOPS numerator is 2*M*N*logical_K, never padded_K or packed lanes.
#include "mxfp4_gemm.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef MXFP4_BENCH_TILED
#include "mxfp4_gemm_tiled.h"
#endif
#ifdef MXFP4_BENCH_CERTIFIED
#include "mxfp4_gemm_certified.h"
#ifndef MXFP4_CERTIFIED_VARIANT
#define MXFP4_CERTIFIED_VARIANT 0
#endif
#endif

namespace {

void cuda_checked(cudaError_t status, const char* expression) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(status));
}
void cublas_checked(cublasStatus_t status, const char* expression) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(expression) + ": cuBLAS status " + std::to_string(int(status)));
}
#define CUDA(expression) cuda_checked((expression), #expression)
#define BLAS(expression) cublas_checked((expression), #expression)
#ifdef MXFP4_BENCH_CERTIFIED
constexpr bool kCertifiedFused = MXFP4_CERTIFIED_VARIANT >= 3;
constexpr const char* kCertifiedAcceptedPath = kCertifiedFused
    ? "fused_normalized_hmma" : "normalized_integer_cublas";
constexpr const char* kCertifiedTotalScope = kCertifiedFused
    ? "packed_to_fp32_gpu_certificate_fused_normalized_hmma_and_conditional_fallback"
    : "packed_to_fp32_gpu_certificate_normalization_cublas_conditional_epilogue_and_fallback";

void certified_checked(int status, const char* expression) {
  if (status != 0)
    throw std::runtime_error(std::string(expression) + ": " + mxfp4_certified_error_string(status));
}
#define CERTIFIED(expression) certified_checked((expression), #expression)

struct CertifiedWorkspace {
  void* value = nullptr;
  uint64_t bytes = 0;
  CertifiedWorkspace(int64_t m, int64_t n, int64_t kp) {
    CERTIFIED(mxfp4_certified_workspace_bytes(m, n, kp, &bytes));
    const int status = mxfp4_certified_create(m, n, kp, &value);
    if (status != 0) {
      if (value) mxfp4_certified_destroy(value);
      value = nullptr;
      certified_checked(status, "mxfp4_certified_create");
    }
  }
  ~CertifiedWorkspace() { if (value) mxfp4_certified_destroy(value); }
  CertifiedWorkspace(const CertifiedWorkspace&) = delete;
  CertifiedWorkspace& operator=(const CertifiedWorkspace&) = delete;
};
#endif

std::string quote(const std::string& text) {
  std::string out = "\"";
  for (unsigned char c : text) {
    if (c == '\\' || c == '"') { out += '\\'; out += char(c); }
    else if (c == '\n') out += "\\n";
    else if (c == '\r') out += "\\r";
    else if (c == '\t') out += "\\t";
    else if (c < 32) out += '?';
    else out += char(c);
  }
  return out + '"';
}

std::string number(double value) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream out;
  out << std::setprecision(12) << value;
  return out.str();
}

class Reporter {
  std::ofstream file_;
 public:
  void open(const std::string& path) {
    if (path.empty()) return;
    if (std::ifstream(path).good()) throw std::runtime_error("Refusing to overwrite evidence file " + path);
    file_.open(path);
    if (!file_) throw std::runtime_error("Cannot create JSONL output " + path);
  }
  void emit(const std::string& json) {
    std::cout << json << std::endl;
    if (file_.is_open()) {
      file_ << json << std::endl;
      if (!file_) throw std::runtime_error("Cannot write JSONL evidence");
    }
  }
};

template<class T> struct Buffer {
  T* data = nullptr;
  explicit Buffer(size_t count) {
    CUDA(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
  }
  ~Buffer() { if (data) cudaFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};

struct Stream {
  cudaStream_t value = nullptr;
  Stream() { CUDA(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() { if (value) cudaStreamDestroy(value); }
};

struct Events {
  cudaEvent_t start = nullptr, stop = nullptr;
  Events() { CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&stop)); }
  ~Events() { if (start) cudaEventDestroy(start); if (stop) cudaEventDestroy(stop); }
};

struct Blas {
  cublasHandle_t handle = nullptr;
  explicit Blas(cudaStream_t stream) {
    BLAS(cublasCreate(&handle));
    BLAS(cublasSetStream(handle, stream));
    BLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
  }
  ~Blas() { if (handle) cublasDestroy(handle); }
};

// Dynamically load read-only NVML telemetry: no SDK headers or link dependency.
// Missing or unsupported sensors are explicit JSON nulls, not guessed values.
class Telemetry {
  void* library_ = nullptr;
  void* device_ = nullptr;
  bool initialized_ = false;
  using Init = int (*)();
  using Shutdown = int (*)();
  using GetByPci = int (*)(const char*, void**);
  using Clock = int (*)(void*, unsigned, unsigned*);
  using Scalar = int (*)(void*, unsigned*);
  Clock clock_ = nullptr;
  Scalar power_ = nullptr, limit_ = nullptr;
  Shutdown shutdown_ = nullptr;
  std::string error_;
 public:
  explicit Telemetry(const char* pci) {
    library_ = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!library_) { error_ = "NVML library unavailable"; return; }
    auto init = reinterpret_cast<Init>(dlsym(library_, "nvmlInit_v2"));
    shutdown_ = reinterpret_cast<Shutdown>(dlsym(library_, "nvmlShutdown"));
    auto get = reinterpret_cast<GetByPci>(dlsym(library_, "nvmlDeviceGetHandleByPciBusId_v2"));
    if (!init || !shutdown_ || !get) { error_ = "Required NVML symbols unavailable"; return; }
    const int status = init();
    if (status != 0) { error_ = "NVML init status " + std::to_string(status); return; }
    initialized_ = true;
    const int device_status = get(pci, &device_);
    if (device_status != 0) {
      device_ = nullptr;
      error_ = "NVML device status " + std::to_string(device_status);
      return;
    }
    clock_ = reinterpret_cast<Clock>(dlsym(library_, "nvmlDeviceGetClockInfo"));
    power_ = reinterpret_cast<Scalar>(dlsym(library_, "nvmlDeviceGetPowerUsage"));
    limit_ = reinterpret_cast<Scalar>(dlsym(library_, "nvmlDeviceGetPowerManagementLimit"));
  }
  ~Telemetry() {
    if (initialized_ && shutdown_) shutdown_();
    if (library_) dlclose(library_);
  }
  std::string snapshot() const {
    unsigned sm = 0, mem = 0, power = 0, limit = 0;
    const bool sm_ok = device_ && clock_ && clock_(device_, 1, &sm) == 0;
    const bool mem_ok = device_ && clock_ && clock_(device_, 2, &mem) == 0;
    const bool power_ok = device_ && power_ && power_(device_, &power) == 0;
    const bool limit_ok = device_ && limit_ && limit_(device_, &limit) == 0;
    return "{\"nvml_available\":" + std::string(device_ ? "true" : "false")
        + ",\"sm_clock_mhz\":" + (sm_ok ? std::to_string(sm) : "null")
        + ",\"memory_clock_mhz\":" + (mem_ok ? std::to_string(mem) : "null")
        + ",\"power_w\":" + (power_ok ? number(power / 1000.0) : "null")
        + ",\"power_limit_w\":" + (limit_ok ? number(limit / 1000.0) : "null")
        + ",\"error\":" + (error_.empty() ? "null" : quote(error_)) + "}";
  }
};

constexpr int host_q[16] = {0, 1, 2, 3, 4, 6, 8, 12,
                            0, -1, -2, -3, -4, -6, -8, -12};
__device__ __constant__ int device_q[16] = {0, 1, 2, 3, 4, 6, 8, 12,
                                         0, -1, -2, -3, -4, -6, -8, -12};

// Independent straightforward LUT decode for the cuBLAS baseline. Its stores
// materialize both complete FP16 inputs and are included in total input cost.
__global__ void dequantize_both(const uint8_t* a, const uint8_t* as,
                               const uint8_t* b, const uint8_t* bs,
                               half* ah, half* bh, int m, int n, int kp) {
  const int64_t count = int64_t(m + n) * kp;
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += int64_t(gridDim.x) * blockDim.x) {
    const bool is_a = index < int64_t(m) * kp;
    const int64_t local = is_a ? index : index - int64_t(m) * kp;
    const int row = int(local / kp), k = int(local % kp);
    const uint8_t* packed = is_a ? a : b;
    const uint8_t* scales = is_a ? as : bs;
    half* output = is_a ? ah : bh;
    const unsigned code = (packed[local / 2] >> ((k & 1) * 4)) & 15u;
    const int scale = scales[int64_t(row) * (kp / 32) + k / 32];
    output[local] = __float2half_rn(ldexpf(float(device_q[code]), scale - 128));
  }
}

struct Options {
  int device = 0, warmup = 3, iters = 10, rounds = 5, samples = 256;
  int m = 0, n = 0, k = 0, max_mib = 1024, reserve_mib = 512;
  uint32_t seed = 20261123;
  bool quick = false, correctness_only = false, allow_other_gpu = false;
  std::string mode = "all", output;
};

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--quick") o.quick = true;
    else if (arg == "--correctness-only") o.correctness_only = true;
    else if (arg == "--allow-other-gpu") o.allow_other_gpu = true;
    else if (arg == "--help") {
      std::cout << "bench_mxfp4 [--quick] [--correctness-only] [--device N]\n"
          "  [--m M --n N --k K] [--iters N --rounds N --warmup N]\n"
          "  [--mode all|zero|random_unit_scale|random_block_scales]\n"
          "  [--samples N] [--seed N] [--max-mib N] [--reserve-mib N]\n"
          "  [--output new-file.jsonl]\n"
          "  [--allow-other-gpu]\n"
          "Defaults: 5 bounded shapes, 3 labeled input modes, 5 timing rounds.\n"
          "VRAM reserve defaults to 512 MiB; explicit --reserve-mib is at least 64.\n"
          "Allocation also obeys --max-mib and cannot exceed 75% of free VRAM.\n"
          "All compared backends start from the same packed nibble/scale data.\n"
          "TFLOPS counts useful logical K; quantization and H2D are excluded.\n"
          "Existing GPU services, clocks and power limits are never changed.\n";
      std::exit(0);
    } else {
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
      const std::string text = argv[++i];
      if (arg == "--mode") { o.mode = text; continue; }
      if (arg == "--output") { o.output = text; continue; }
      size_t used = 0;
      const long long value = std::stoll(text, &used);
      if (used != text.size() || value < 0 || value > 2147483647)
        throw std::runtime_error("Invalid nonnegative integer for " + arg);
      if (arg == "--device") o.device = int(value);
      else if (arg == "--warmup") o.warmup = int(value);
      else if (arg == "--iters") o.iters = int(value);
      else if (arg == "--rounds") o.rounds = int(value);
      else if (arg == "--samples") o.samples = int(value);
      else if (arg == "--m") o.m = int(value);
      else if (arg == "--n") o.n = int(value);
      else if (arg == "--k") o.k = int(value);
      else if (arg == "--max-mib") o.max_mib = int(value);
      else if (arg == "--reserve-mib") o.reserve_mib = int(value);
      else if (arg == "--seed") o.seed = uint32_t(value);
      else throw std::runtime_error("Unknown argument " + arg);
    }
  }
  if (o.iters <= 0 || o.iters > 10000 || o.rounds < 3 || o.rounds > 100
      || o.warmup > 10000 || o.samples <= 0 || o.samples > 65536 || o.max_mib <= 0)
    throw std::runtime_error("Require iters=1..10000, rounds=3..100, samples=1..65536 and positive max-mib");
  if (o.reserve_mib < 64)
    throw std::runtime_error("reserve-mib must be at least 64; default is 512 MiB");
  if ((o.m || o.n || o.k) && !(o.m > 0 && o.n > 0 && o.k > 0))
    throw std::runtime_error("Custom shapes require --m, --n and --k together");
  if (o.m > 65536 || o.n > 65536 || o.k > 65536)
    throw std::runtime_error("Each custom dimension is bounded to 65536");
  if (o.mode != "all" && o.mode != "zero" && o.mode != "random_unit_scale"
      && o.mode != "random_block_scales")
    throw std::runtime_error("Unknown input mode " + o.mode);
  return o;
}

struct Input {
  int m, n, k, kp, blocks;
  uint32_t seed;
  std::string mode;
  std::vector<uint8_t> a, as, b, bs;
  Input(int m_, int n_, int k_, uint32_t seed_, std::string mode_)
      : m(m_), n(n_), k(k_), kp((k + 31) / 32 * 32), blocks(kp / 32), seed(seed_),
        mode(std::move(mode_)), a(size_t(m) * kp / 2), as(size_t(m) * blocks, 127),
        b(size_t(n) * kp / 2), bs(size_t(n) * blocks, 127) {
    std::mt19937 rng(seed);
    auto fill = [&](std::vector<uint8_t>& data, int rows) {
      if (mode == "zero") return;
      for (int row = 0; row < rows; ++row) for (int col = 0; col < k; ++col)
        data[size_t(row) * kp / 2 + col / 2] |= uint8_t((rng() & 15u) << ((col & 1) * 4));
    };
    fill(a, m); fill(b, n);
    if (mode != "random_unit_scale") {
      for (auto& s : as) s = uint8_t(125 + rng() % 5);
      for (auto& s : bs) s = uint8_t(125 + rng() % 5);
    }
  }
  std::string fingerprint() const {
    uint64_t value = 14695981039346656037ull;
    for (const auto* v : {&a, &as, &b, &bs})
      for (uint8_t byte : *v) { value ^= byte; value *= 1099511628211ull; }
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
  }
};

uint64_t allocation_bytes(int m, int n, int kp) {
  const uint64_t inputs = uint64_t(m + n) * (kp / 2 + kp / 32 + uint64_t(kp) * sizeof(half));
  // Every backend executes serially on the same stream. One device output is
  // enough; correctness results are copied to independent host arrays before
  // the next backend overwrites it. The ordinary cuBLAS half inputs stay
  // separate from any certified workspace. Fused certified variants do not
  // allocate global normalized-half matrices; the C API reports their bytes.
  uint64_t bytes = inputs + uint64_t(m) * n * sizeof(float);
#ifdef MXFP4_BENCH_CERTIFIED
  uint64_t workspace = 0;
  CERTIFIED(mxfp4_certified_workspace_bytes(m, n, kp, &workspace));
  bytes += workspace;
#endif
  return bytes;
}

void memory_guard(int m, int n, int kp, const Options& options) {
  size_t free = 0, total = 0;
  CUDA(cudaMemGetInfo(&free, &total));
  const uint64_t requested = allocation_bytes(m, n, kp);
  const uint64_t reserve = uint64_t(options.reserve_mib) << 20;
  if (requested > uint64_t(options.max_mib) * (1ull << 20)
      || free < reserve || requested > uint64_t(free) - reserve
      || requested > uint64_t(free) * 3 / 4)
    throw std::runtime_error("VRAM guard refused " + std::to_string(requested)
        + " allocation bytes; preserves " + std::to_string(options.reserve_mib)
        + " MiB, 25% free VRAM, and --max-mib cap; no services are stopped");
}

int decoded_q(const std::vector<uint8_t>& input, int row, int k, int kp) {
  return host_q[(input[size_t(row) * kp / 2 + k / 2] >> ((k & 1) * 4)) & 15u];
}

float cpu_reference(const Input& in, size_t index) {
  const int row = int(index / in.n), col = int(index % in.n);
  float sum = 0;
  for (int block = 0; block < in.blocks; ++block) {
    int dot = 0;
    for (int k = block * 32; k < (block + 1) * 32; ++k)
      dot += decoded_q(in.a, row, k, in.kp) * decoded_q(in.b, col, k, in.kp);
    const int sa = in.as[size_t(row) * in.blocks + block];
    const int sb = in.bs[size_t(col) * in.blocks + block];
    const float value = std::ldexp(float(dot), sa + sb - 256);
    volatile float rounded = sum + value;
    sum = rounded;
  }
  return sum;
}

std::vector<size_t> reference_indices(const Input& in, int requested, bool& exhaustive) {
  const size_t outputs = size_t(in.m) * in.n;
  exhaustive = uint64_t(outputs) * in.k <= 2000000;
  if (exhaustive) {
    std::vector<size_t> indices(outputs);
    for (size_t i = 0; i < outputs; ++i) indices[i] = i;
    return indices;
  }
  std::set<size_t> selected;
  for (int row : {0, 1, 15, 16, in.m - 1}) for (int col : {0, 1, 15, 16, in.n - 1})
    if (row >= 0 && row < in.m && col >= 0 && col < in.n)
      selected.insert(size_t(row) * in.n + col);
  const size_t target = std::min(outputs, std::max(size_t(requested), selected.size()));
  std::mt19937_64 rng(uint64_t(in.seed) ^ 0x9e3779b97f4a7c15ull);
  while (selected.size() < target) selected.insert(size_t(rng() % outputs));
  return {selected.begin(), selected.end()};
}

bool exact_equal(float a, float b) {
  return a == b || (std::isnan(a) && std::isnan(b)); // Only NaN and signed-zero exceptions.
}

struct ErrorMetrics {
  size_t values = 0, exact_mismatches = 0, tolerance_failures = 0, nonfinite_mismatches = 0;
  double max_abs = 0, max_relative = 0, squared_error = 0, squared_reference = 0;
  void add(float actual, float expected, double atol, double rtol) {
    ++values;
    if (!exact_equal(actual, expected)) ++exact_mismatches;
    if (!std::isfinite(actual) || !std::isfinite(expected)) {
      if (!exact_equal(actual, expected)) { ++nonfinite_mismatches; ++tolerance_failures; }
      return;
    }
    const double error = std::abs(double(actual) - expected);
    max_abs = std::max(max_abs, error);
    max_relative = std::max(max_relative, error / std::max(std::abs(double(expected)), 1e-12));
    squared_error += error * error;
    squared_reference += double(expected) * expected;
    if (error > atol + rtol * std::abs(double(expected))) ++tolerance_failures;
  }
  std::string json() const {
    const double nrmse = squared_reference > 0 ? std::sqrt(squared_error / squared_reference)
        : (squared_error == 0 ? 0 : std::numeric_limits<double>::infinity());
    return "{\"values\":" + std::to_string(values)
        + ",\"exact_mismatches\":" + std::to_string(exact_mismatches)
        + ",\"tolerance_failures\":" + std::to_string(tolerance_failures)
        + ",\"nonfinite_mismatches\":" + std::to_string(nonfinite_mismatches)
        + ",\"max_abs\":" + number(max_abs) + ",\"max_relative\":" + number(max_relative)
        + ",\"nrmse\":" + number(nrmse) + "}";
  }
};

struct Job {
  std::string name, scope;
  std::function<void()> launch;
  std::vector<double> milliseconds;
  std::vector<double> wall_milliseconds;
  bool primary_wall_clock = false;
};

struct ExecutionState {
  bool attempted = false, submitted = false, completed = false, pending = false;
  bool submission_uncertain = false;
  void begin() { attempted = true; }
  void begin_multi_stage() {
    attempted = true;
    // A multi-stage API can fail after submitting earlier internal kernels.
    // Its error code cannot prove that no work remains queued.
    submission_uncertain = true;
  }
  void enqueue_succeeded() {
    submitted = true;
    pending = true;
    submission_uncertain = false;
  }
  void synchronization_succeeded() {
    completed = completed || submitted;
    pending = false;
    submission_uncertain = false;
  }
  const char* gpu_executed() const {
    // Enqueued work followed by a failed synchronization has unknown execution
    // status, not false. A later failure cannot erase an earlier confirmed run.
    return completed ? "true" : ((submitted || submission_uncertain) ? "null" : "false");
  }
  std::string fields() const {
    return "\"gpu_executed\":" + std::string(gpu_executed())
        + ",\"kernel_launch_attempted\":" + (attempted ? "true" : "false")
        + ",\"kernel_launch_submitted\":" + (submitted ? "true" : (submission_uncertain ? "null" : "false"))
        + ",\"kernel_submission_uncertain\":" + (submission_uncertain ? "true" : "false")
        + ",\"kernel_execution_completed\":" + (completed ? "true" : "false")
        + ",\"all_submitted_work_completed\":" + (pending ? "false" : (submission_uncertain ? "null" : "true"));
  }
};

double median_ms(const std::vector<double>& values) {
  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  const size_t n = sorted.size();
  if (n == 0) throw std::runtime_error("Timing sample set is empty");
  return n % 2 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) * .5;
}

std::string timing_json(const Job& job, double flops, bool count_flops) {
  const auto& primary = job.primary_wall_clock ? job.wall_milliseconds : job.milliseconds;
  const double median = median_ms(primary);
  const double event_median = median_ms(job.milliseconds);
  const double wall_median = median_ms(job.wall_milliseconds);
  std::ostringstream out;
  out << "{\"name\":" << quote(job.name) << ",\"scope\":" << quote(job.scope)
      << ",\"primary_clock\":" << quote(job.primary_wall_clock ? "synchronized_steady_clock" : "cuda_event_elapsed")
      << ",\"median_ms\":" << number(median)
      << ",\"min_ms\":" << number(*std::min_element(primary.begin(), primary.end()))
      << ",\"max_ms\":" << number(*std::max_element(primary.begin(), primary.end()))
      << ",\"useful_tflops\":" << (count_flops ? number(flops / (median * 1e9)) : "null")
      << ",\"cuda_event_median_ms\":" << number(event_median)
      << ",\"cuda_event_useful_tflops\":" << (count_flops ? number(flops / (event_median * 1e9)) : "null")
      << ",\"synchronized_wall_median_ms\":" << number(wall_median)
      << ",\"synchronized_wall_useful_tflops\":" << (count_flops ? number(flops / (wall_median * 1e9)) : "null")
      << ",\"round_ms\":[";
  for (size_t i = 0; i < primary.size(); ++i) {
    if (i) out << ',';
    out << number(primary[i]);
  }
  out << "],\"cuda_event_round_ms\":[";
  for (size_t i = 0; i < job.milliseconds.size(); ++i) {
    if (i) out << ',';
    out << number(job.milliseconds[i]);
  }
  out << "],\"synchronized_wall_round_ms\":[";
  for (size_t i = 0; i < job.wall_milliseconds.size(); ++i) {
    if (i) out << ',';
    out << number(job.wall_milliseconds[i]);
  }
  out << "]}";
  return out.str();
}

std::string case_prefix(const Input& in, const Options& options) {
  return "\"M\":" + std::to_string(in.m) + ",\"N\":" + std::to_string(in.n)
      + ",\"logical_K\":" + std::to_string(in.k) + ",\"padded_K\":" + std::to_string(in.kp)
      + ",\"input_mode\":" + quote(in.mode) + ",\"seed\":" + std::to_string(in.seed)
      + ",\"input_fnv1a64\":" + quote(in.fingerprint())
      + ",\"scale_exponent_range\":" + (in.mode == "random_unit_scale" ? "[0,0]" : "[-2,2]")
      + ",\"input_distribution\":" + quote(in.mode == "zero" ? "all_zero_codes" : "uniform_full_e2m1_nibbles")
      + ",\"real_training_tensor\":false,\"fp16_dequant_exact\":true"
      + ",\"reserve_mib\":" + std::to_string(options.reserve_mib)
      + ",\"max_allocation_mib\":" + std::to_string(options.max_mib)
      + ",\"shared_device_output_count\":1,\"correctness_host_outputs_independent\":true"
      + ",\"allocation_bytes\":" + std::to_string(allocation_bytes(in.m, in.n, in.kp));
}

#ifdef MXFP4_BENCH_CERTIFIED
void validate_certificate_info(const MXFP4CertifiedInfo& info, const Input& in,
                               uint64_t expected_workspace) {
  int device = -1;
  CUDA(cudaGetDevice(&device));
  if (!info.metadata_valid || info.abi_version != MXFP4_CERTIFIED_ABI_VERSION
      || info.device != device || info.m != in.m || info.n != in.n
      || info.padded_k != in.kp || info.workspace_bytes != expected_workspace
      || info.accepted > 1 || (info.reject_flags & ~uint32_t(15))
      || bool(info.accepted) != (info.reject_flags == 0)
      || info.zero_rows_a > uint32_t(in.m) || info.zero_rows_b > uint32_t(in.n))
    throw std::runtime_error("Certified backend returned invalid/inconsistent diagnostic metadata");
}

std::string certificate_json(const MXFP4CertifiedInfo& info) {
  const bool energy_pass = info.max_energy_a == 0 || info.max_energy_b == 0
      || info.max_energy_a <= MXFP4_CERTIFIED_ENERGY_BOUND / info.max_energy_b;
  std::ostringstream out;
  out << "{\"abi_version\":" << info.abi_version
      << ",\"metadata_valid\":" << (info.metadata_valid ? "true" : "false")
      << ",\"accepted\":" << (info.accepted ? "true" : "false")
      << ",\"path\":" << quote(info.accepted ? kCertifiedAcceptedPath : "exact_tiled_fallback")
      << ",\"reject_flags\":" << info.reject_flags
      << ",\"reject_flag_bits\":{\"nan_scale\":1,\"span\":2,\"energy\":4,\"lattice\":8}"
      << ",\"workspace_bytes\":" << info.workspace_bytes
      << ",\"workspace_reused_across_calls\":true,\"workspace_allocation_timed\":false"
      << ",\"certificate_recomputed_each_call\":true"
      << ",\"normalized_operands_recomputed_each_call\":" << (!kCertifiedFused || info.accepted ? "true" : "false")
      << ",\"normalization_executes_on_rejected_calls\":" << (kCertifiedFused ? "false" : "true")
      << ",\"normalized_operands_materialized\":" << (kCertifiedFused ? "false" : "true")
      << ",\"normalization_location\":" << quote(kCertifiedFused ? "fused_gemm" : "prepare_global_fp16")
      << ",\"output_scaling_location\":" << quote(kCertifiedFused ? "fused_gemm_epilogue" : "separate_epilogue")
      << ",\"host_gate_synchronization_per_call\":false"
      << ",\"diagnostic_get_info_in_timed_region\":false"
      << ",\"cublas_executes_on_rejected_calls\":" << (kCertifiedFused ? "false" : "true")
      << ",\"max_energy_a\":" << info.max_energy_a
      << ",\"max_energy_b\":" << info.max_energy_b
      << ",\"energy_a_representation\":" << quote(info.energy_saturated_a ? "lower_bound_at_cap" : "exact")
      << ",\"energy_b_representation\":" << quote(info.energy_saturated_b ? "lower_bound_at_cap" : "exact")
      << ",\"energy_product_bound\":" << MXFP4_CERTIFIED_ENERGY_BOUND
      << ",\"energy_cap\":" << MXFP4_CERTIFIED_ENERGY_CAP
      << ",\"energy_gate_comparison\":\"U==0 || V==0 || U<=2^48/V (integer division)\""
      << ",\"energy_gate_pass_from_reported_bounds\":" << (energy_pass ? "true" : "false")
      << ",\"max_span_a\":" << info.max_span_a
      << ",\"max_span_b\":" << info.max_span_b
      << ",\"span_limit\":12"
      << ",\"span_scope\":\"maximum_within_row_over_blocks_with_nonzero_q\""
      << ",\"min_scale_a\":" << info.min_scale_a
      << ",\"max_scale_a\":" << info.max_scale_a
      << ",\"min_scale_b\":" << info.min_scale_b
      << ",\"max_scale_b\":" << info.max_scale_b
      << ",\"scale_extrema_scope\":\"per_row_normalization_minima_including_zero_rows_at_127\""
      << ",\"lattice_exponent_min\":" << info.min_scale_a + info.min_scale_b - 256
      << ",\"lattice_exponent_max\":" << info.max_scale_a + info.max_scale_b - 256
      << ",\"lattice_exponent_allowed_range\":[-126,103]"
      << ",\"nan_scales_a\":" << info.nan_scales_a
      << ",\"nan_scales_b\":" << info.nan_scales_b
      << ",\"zero_rows_a\":" << info.zero_rows_a
      << ",\"zero_rows_b\":" << info.zero_rows_b
      << ",\"nan_scale_gate_pass\":" << ((info.reject_flags & MXFP4_CERT_REJECT_NAN_SCALE) ? "false" : "true")
      << ",\"span_gate_pass\":" << ((info.reject_flags & MXFP4_CERT_REJECT_SCALE_SPAN) ? "false" : "true")
      << ",\"energy_gate_pass\":" << ((info.reject_flags & MXFP4_CERT_REJECT_ENERGY_BOUND) ? "false" : "true")
      << ",\"lattice_gate_pass\":" << ((info.reject_flags & MXFP4_CERT_REJECT_LATTICE_RANGE) ? "false" : "true")
      << ",\"submitted_calls\":" << info.submitted_calls << '}';
  return out.str();
}

bool same_certificate(const MXFP4CertifiedInfo& a, const MXFP4CertifiedInfo& b) {
  // Ignore only the monotonically increasing call count and reserved/padding.
  return a.accepted == b.accepted && a.reject_flags == b.reject_flags
      && a.max_energy_a == b.max_energy_a && a.max_energy_b == b.max_energy_b
      && a.min_scale_a == b.min_scale_a && a.max_scale_a == b.max_scale_a
      && a.min_scale_b == b.min_scale_b && a.max_scale_b == b.max_scale_b
      && a.max_span_a == b.max_span_a && a.max_span_b == b.max_span_b
      && a.nan_scales_a == b.nan_scales_a && a.nan_scales_b == b.nan_scales_b
      && a.zero_rows_a == b.zero_rows_a && a.zero_rows_b == b.zero_rows_b
      && a.energy_saturated_a == b.energy_saturated_a && a.energy_saturated_b == b.energy_saturated_b;
}
#endif

void run_case(const Input& in, const Options& options, cudaStream_t stream,
              cublasHandle_t blas, Telemetry& telemetry, Reporter& report,
              ExecutionState& execution, bool actual_v100) {
  const size_t output_count = size_t(in.m) * in.n;
  Buffer<uint8_t> da(in.a.size()), das(in.as.size()), db(in.b.size()), dbs(in.bs.size());
  Buffer<half> ah(size_t(in.m) * in.kp), bh(size_t(in.n) * in.kp);
  Buffer<float> output(output_count);
#ifdef MXFP4_BENCH_CERTIFIED
  CertifiedWorkspace certified_workspace(in.m, in.n, in.kp);
#endif
  auto copy_in = [&](uint8_t* dst, const std::vector<uint8_t>& src) {
    CUDA(cudaMemcpyAsync(dst, src.data(), src.size(), cudaMemcpyHostToDevice, stream));
  };
  copy_in(da.data, in.a); copy_in(das.data, in.as);
  copy_in(db.data, in.b); copy_in(dbs.data, in.bs);
  auto baseline_call = [&]() {
    execution.begin();
    CUDA(cudaError_t(mxfp4_gemm_cuda(da.data, das.data, db.data, dbs.data, output.data,
                                    in.m, in.n, in.kp, stream)));
    execution.enqueue_succeeded();
  };
#ifdef MXFP4_BENCH_TILED
  auto tiled_call = [&]() {
    execution.begin();
    CUDA(cudaError_t(mxfp4_gemm_tiled_cuda(da.data, das.data, db.data, dbs.data, output.data,
                                          in.m, in.n, in.kp, stream)));
    execution.enqueue_succeeded();
  };
#endif
#ifdef MXFP4_BENCH_CERTIFIED
  auto certified_call = [&]() {
    execution.begin_multi_stage();
    // A null used_fast deliberately chooses the fully asynchronous API.
    // Every call recomputes the device certificate. The selected backend
    // either materializes normalized FP16 for cuBLAS (including rejected
    // calls), or decodes inside a device-gated fused HMMA kernel. Conditional
    // tiled fallback and all preparation remain in the measured pipeline.
    CERTIFIED(mxfp4_certified_gemm(certified_workspace.value, da.data, das.data,
                                  db.data, dbs.data, output.data, stream, nullptr));
    execution.enqueue_succeeded();
  };
#endif
  auto dequant_call = [&]() {
    const unsigned grid = unsigned(std::min<int64_t>(65535, (int64_t(in.m + in.n) * in.kp + 255) / 256));
    execution.begin();
    dequantize_both<<<grid, 256, 0, stream>>>(da.data, das.data, db.data, dbs.data,
                                           ah.data, bh.data, in.m, in.n, in.kp);
    CUDA(cudaGetLastError());
    execution.enqueue_succeeded();
  };
  auto cublas_call = [&]() {
    const float alpha = 1, beta = 0;
    // Row-major C[M,N] is column-major C^T[N,M]. Both inputs are row-major
    // K-contiguous, hence C^T = transpose(B_as_K_by_N) * A_as_K_by_M.
    execution.begin();
    BLAS(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, in.n, in.m, in.kp,
                      &alpha, bh.data, CUDA_R_16F, in.kp,
                      ah.data, CUDA_R_16F, in.kp,
                      &beta, output.data, CUDA_R_32F, in.n,
                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    execution.enqueue_succeeded();
  };
  // Serialize correctness and preserve every complete result on the host
  // before another backend reuses the one device output allocation.
  auto copy_completed_output = [&](std::vector<float>& host_output) {
    CUDA(cudaStreamSynchronize(stream));
    execution.synchronization_succeeded();
    CUDA(cudaMemcpy(host_output.data(), output.data, output_count * sizeof(float), cudaMemcpyDeviceToHost));
  };
  std::vector<float> cpu_baseline(output_count), cpu_cublas(output_count);
  baseline_call();
  copy_completed_output(cpu_baseline);
#ifdef MXFP4_BENCH_TILED
  std::vector<float> cpu_tiled(output_count);
  tiled_call();
  copy_completed_output(cpu_tiled);
#endif
#ifdef MXFP4_BENCH_CERTIFIED
  std::vector<float> cpu_certified(output_count);
  certified_call();
  copy_completed_output(cpu_certified);
  MXFP4CertifiedInfo certified_info{};
  CERTIFIED(mxfp4_certified_get_info(certified_workspace.value, &certified_info));
  validate_certificate_info(certified_info, in, certified_workspace.bytes);
  if (certified_info.submitted_calls != 1)
    throw std::runtime_error("Unexpected initial certified call count");
#endif
  dequant_call(); cublas_call();
  copy_completed_output(cpu_cublas);
  bool exhaustive = false;
  const auto indices = reference_indices(in, options.samples, exhaustive);
  ErrorMetrics baseline_oracle, cublas_vs_baseline, cublas_oracle;
#ifdef MXFP4_BENCH_TILED
  ErrorMetrics tiled_oracle, tiled_vs_baseline;
#endif
#ifdef MXFP4_BENCH_CERTIFIED
  ErrorMetrics certified_oracle, certified_vs_baseline;
#endif
  // These tolerances are fixed before execution. They allow only cuBLAS's
  // different FP32 accumulation order, never a relaxed tiled correctness gate.
  constexpr double cublas_atol = 2e-3, cublas_rtol = 2e-4;
  for (size_t index : indices) {
    const float expected = cpu_reference(in, index);
    baseline_oracle.add(cpu_baseline[index], expected, 0, 0);
    cublas_oracle.add(cpu_cublas[index], expected, cublas_atol, cublas_rtol);
#ifdef MXFP4_BENCH_TILED
    tiled_oracle.add(cpu_tiled[index], expected, 0, 0);
#endif
#ifdef MXFP4_BENCH_CERTIFIED
    certified_oracle.add(cpu_certified[index], expected, 0, 0);
#endif
  }
  for (size_t i = 0; i < output_count; ++i) {
    cublas_vs_baseline.add(cpu_cublas[i], cpu_baseline[i], cublas_atol, cublas_rtol);
#ifdef MXFP4_BENCH_TILED
    tiled_vs_baseline.add(cpu_tiled[i], cpu_baseline[i], 0, 0);
#endif
#ifdef MXFP4_BENCH_CERTIFIED
    certified_vs_baseline.add(cpu_certified[i], cpu_baseline[i], 0, 0);
#endif
  }
  bool passed = !baseline_oracle.exact_mismatches && !cublas_oracle.tolerance_failures
      && !cublas_vs_baseline.tolerance_failures;
#ifdef MXFP4_BENCH_TILED
  passed = passed && !tiled_oracle.exact_mismatches && !tiled_vs_baseline.exact_mismatches;
#endif
#ifdef MXFP4_BENCH_CERTIFIED
  passed = passed && !certified_oracle.exact_mismatches && !certified_vs_baseline.exact_mismatches;
#endif
  std::string validation = "{\"type\":\"correctness\"," + case_prefix(in, options)
      + "," + execution.fields() + ",\"actual_V100\":" + (actual_v100 ? "true" : "false")
      + ",\"reference_scope\":" + quote(exhaustive ? "cpu_ordered_mx32_all_outputs" : "cpu_ordered_mx32_selected_outputs")
      + ",\"cpu_reference_values\":" + std::to_string(indices.size())
      + ",\"packed_equality\":\"bitwise_except_nan_payload_and_signed_zero\""
      + ",\"relative_error_denominator_floor\":1e-12"
      + ",\"nrmse_definition\":\"sqrt(sum(error^2)/sum(reference^2))\""
      + ",\"baseline_vs_cpu\":" + baseline_oracle.json()
      + ",\"cublas_vs_cpu\":" + cublas_oracle.json()
      + ",\"cublas_vs_baseline_all_outputs\":" + cublas_vs_baseline.json()
      + ",\"cublas_atol\":" + number(cublas_atol) + ",\"cublas_rtol\":" + number(cublas_rtol);
#ifdef MXFP4_BENCH_TILED
  validation += ",\"tiled_vs_cpu\":" + tiled_oracle.json()
      + ",\"tiled_vs_baseline_all_outputs\":" + tiled_vs_baseline.json();
#endif
#ifdef MXFP4_BENCH_CERTIFIED
  validation += ",\"certified_vs_cpu\":" + certified_oracle.json()
      + ",\"certified_vs_baseline_all_outputs\":" + certified_vs_baseline.json()
      + ",\"certificate\":" + certificate_json(certified_info);
#endif
  validation += ",\"passed\":" + std::string(passed ? "true" : "false") + "}";
  report.emit(validation);
  if (!passed) throw std::runtime_error("Numerical comparison failed; no timing or speed claim for this case");
  if (options.correctness_only) return;

  std::vector<Job> jobs;
  jobs.push_back({"mxfp4_baseline", "packed_to_fp32_including_decode_and_scale", baseline_call, {}});
#ifdef MXFP4_BENCH_TILED
  jobs.push_back({"mxfp4_tiled", "packed_to_fp32_including_decode_and_scale", tiled_call, {}});
#endif
#ifdef MXFP4_BENCH_CERTIFIED
  jobs.push_back({"mxfp4_certified_total",
                 kCertifiedTotalScope,
                 certified_call, {}, {}, true});
#endif
  jobs.push_back({"cublas_compute", "predequantized_fp16_to_fp32", cublas_call, {}});
  jobs.push_back({"cublas_dequant", "packed_to_materialized_fp16_both_operands", dequant_call, {}});
  jobs.push_back({"cublas_total", "packed_to_fp32_dequant_then_cublas", [&]() { dequant_call(); cublas_call(); }, {}});
  for (int i = 0; i < options.warmup; ++i) for (auto& job : jobs) job.launch();
  CUDA(cudaStreamSynchronize(stream));
  execution.synchronization_succeeded();
  const std::string telemetry_before = telemetry.snapshot();
  Events events;
#ifdef MXFP4_BENCH_CERTIFIED
  bool certified_final_output_captured = false;
#endif
  // Rotate backend order each round, avoiding systematic cold/thermal order.
  for (int round = 0; round < options.rounds; ++round) {
    for (size_t offset = 0; offset < jobs.size(); ++offset) {
      auto& job = jobs[(size_t(round) + offset) % jobs.size()];
      const auto wall_start = std::chrono::steady_clock::now();
      CUDA(cudaEventRecord(events.start, stream));
      for (int iteration = 0; iteration < options.iters; ++iteration) job.launch();
      CUDA(cudaEventRecord(events.stop, stream));
      CUDA(cudaEventSynchronize(events.stop));
      const auto wall_stop = std::chrono::steady_clock::now();
      execution.synchronization_succeeded();
      float total_ms = 0;
      CUDA(cudaEventElapsedTime(&total_ms, events.start, events.stop));
      if (!(total_ms > 0) || !std::isfinite(total_ms))
        throw std::runtime_error("Invalid CUDA event timing");
      job.milliseconds.push_back(double(total_ms) / options.iters);
      const double wall_ms = std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
      if (!(wall_ms > 0) || !std::isfinite(wall_ms))
        throw std::runtime_error("Invalid synchronized wall timing");
      job.wall_milliseconds.push_back(wall_ms / options.iters);
#ifdef MXFP4_BENCH_CERTIFIED
      if (round == options.rounds - 1 && job.name == "mxfp4_certified_total") {
        // Keep the actual final timed certified result before the next job
        // reuses output. This one D2H copy is after BOTH timing endpoints.
        CUDA(cudaMemcpy(cpu_certified.data(), output.data, output_count * sizeof(float), cudaMemcpyDeviceToHost));
        certified_final_output_captured = true;
      }
#endif
    }
  }
#ifdef MXFP4_BENCH_CERTIFIED
  // Stateful workspace reuse is checked again AFTER all timed repetitions.
  // Diagnostic synchronization/copies are outside the reported intervals.
  MXFP4CertifiedInfo certified_after{};
  CERTIFIED(mxfp4_certified_get_info(certified_workspace.value, &certified_after));
  validate_certificate_info(certified_after, in, certified_workspace.bytes);
  const uint64_t expected_certified_calls = certified_info.submitted_calls
      + uint64_t(options.warmup) + uint64_t(options.iters) * options.rounds;
  if (!same_certificate(certified_info, certified_after)
      || certified_after.submitted_calls != expected_certified_calls)
    throw std::runtime_error("Certificate changed for unchanged inputs or a timed invocation was omitted");
  if (!certified_final_output_captured)
    throw std::runtime_error("Final timed certified result was not preserved for validation");
  ErrorMetrics certified_after_vs_baseline;
  for (size_t i = 0; i < output_count; ++i)
    certified_after_vs_baseline.add(cpu_certified[i], cpu_baseline[i], 0, 0);
  if (certified_after_vs_baseline.exact_mismatches)
    throw std::runtime_error("Certified output changed after timed workspace reuse");
#endif
  const double useful_flops = 2.0 * in.m * in.n * in.k;
  std::string timing = "{\"type\":\"benchmark\"," + case_prefix(in, options)
      + "," + execution.fields() + ",\"actual_V100\":" + (actual_v100 ? "true" : "false")
      + ",\"correctness_passed\":true,\"useful_flops\":" + number(useful_flops)
      + ",\"tflops_formula\":\"2*M*N*logical_K/(median_ms*1e9)\""
      + ",\"warmup_per_job\":" + std::to_string(options.warmup)
      + ",\"iterations_per_round\":" + std::to_string(options.iters)
      + ",\"rounds\":" + std::to_string(options.rounds)
      + ",\"backend_order\":\"rotating_each_round\",\"same_resident_inputs_reused\":true"
      + ",\"excluded_from_all_times\":[\"host_input_generation\",\"H2D\",\"high_precision_quantization\",\"allocation\",\"correctness_copy\"]"
      + ",\"telemetry_before\":" + telemetry_before
#ifdef MXFP4_BENCH_CERTIFIED
      + ",\"certificate_after_timing\":" + certificate_json(certified_after)
      + ",\"certified_post_timing_vs_baseline_all_outputs\":" + certified_after_vs_baseline.json()
#endif
      + ",\"telemetry_after\":" + telemetry.snapshot() + ",\"timings\":[";
  for (size_t i = 0; i < jobs.size(); ++i) {
    if (i) timing += ',';
    timing += timing_json(jobs[i], useful_flops, jobs[i].name != "cublas_dequant");
  }
  report.emit(timing + "]}");
}

}  // namespace

int main(int argc, char** argv) {
  Reporter report;
  ExecutionState execution;
  bool actual_v100 = false;
  size_t completed_cases = 0;
  try {
    const Options options = parse(argc, argv);
    report.open(options.output);
    CUDA(cudaSetDevice(options.device));
    cudaDeviceProp properties{};
    CUDA(cudaGetDeviceProperties(&properties, options.device));
    actual_v100 = properties.major == 7 && properties.minor == 0
        && std::string(properties.name).find("V100") != std::string::npos;
    if (!actual_v100 && !options.allow_other_gpu)
      throw std::runtime_error("Default benchmark requires a real V100; use --allow-other-gpu to label other hardware");
    if (properties.major < 7) throw std::runtime_error("Tensor Cores require compute capability >= 7.0");
    if (mxfp4_cuda_abi_version() != 1) throw std::runtime_error("Unsupported MXFP4 CUDA ABI");
#ifdef MXFP4_BENCH_TILED
    if (mxfp4_tiled_cuda_abi_version() != 1) throw std::runtime_error("Unsupported tiled MXFP4 CUDA ABI");
#endif
    CUDA(cudaFree(nullptr));
    char pci[64]{};
    CUDA(cudaDeviceGetPCIBusId(pci, sizeof(pci), options.device));
    int driver = 0, runtime = 0;
    CUDA(cudaDriverGetVersion(&driver)); CUDA(cudaRuntimeGetVersion(&runtime));
    Stream stream;
    Blas blas(stream.value);
    int cublas_version = 0;
    BLAS(cublasGetVersion(blas.handle, &cublas_version));
    size_t free = 0, total = 0;
    CUDA(cudaMemGetInfo(&free, &total));
    Telemetry telemetry(pci);
    std::ostringstream environment;
    environment << "{\"type\":\"environment\",\"schema_version\":3,\"name\":" << quote(properties.name)
        << ",\"device\":" << options.device << ",\"pci_bus_id\":" << quote(pci)
        << ",\"compute_capability\":\"" << properties.major << '.' << properties.minor << '"'
        << ",\"actual_V100\":" << (actual_v100 ? "true" : "false")
        << ",\"sm_count\":" << properties.multiProcessorCount
        << ",\"nominal_sm_clock_khz\":" << properties.clockRate
        << ",\"nominal_memory_clock_khz\":" << properties.memoryClockRate
        << ",\"memory_bus_bits\":" << properties.memoryBusWidth
        << ",\"ecc_enabled\":" << (properties.ECCEnabled ? "true" : "false")
        << ",\"driver_version\":" << driver << ",\"runtime_version\":" << runtime
        << ",\"cublas_version\":" << cublas_version
        << ",\"cublas_compute_type\":\"CUBLAS_COMPUTE_32F\""
        << ",\"cublas_algorithm\":\"CUBLAS_GEMM_DEFAULT_TENSOR_OP\""
        << ",\"cublas_math_mode\":\"CUBLAS_TENSOR_OP_MATH\""
        << ",\"free_bytes_after_context_and_cublas\":" << free << ",\"total_bytes\":" << total
        << ",\"reserve_mib\":" << options.reserve_mib
        << ",\"max_allocation_mib\":" << options.max_mib
        << ",\"maximum_fraction_of_free_memory\":0.75"
        << ",\"baseline_backend\":" << quote(mxfp4_cuda_backend())
        << ",\"baseline_abi_version\":" << mxfp4_cuda_abi_version()
#ifdef MXFP4_BENCH_TILED
        << ",\"tiled_compiled\":true"
        << ",\"tiled_backend\":" << quote(mxfp4_tiled_cuda_backend())
        << ",\"tiled_abi_version\":" << mxfp4_tiled_cuda_abi_version()
#else
        << ",\"tiled_compiled\":false"
        << ",\"tiled_backend\":null,\"tiled_abi_version\":null"
#endif
#ifdef MXFP4_BENCH_CERTIFIED
        << ",\"certified_compiled\":true"
        << ",\"certified_abi_version\":" << MXFP4_CERTIFIED_ABI_VERSION
        << ",\"certified_variant\":" << MXFP4_CERTIFIED_VARIANT
        << ",\"certified_accepted_backend\":" << quote(kCertifiedAcceptedPath)
        << ",\"certified_global_normalized_fp16_workspace\":" << (kCertifiedFused ? "false" : "true")
        << ",\"certified_total_timing_clock\":\"synchronized_steady_clock\""
        << ",\"certified_workspace_allocation_excluded\":true"
#else
        << ",\"certified_compiled\":false"
#endif
        << ",\"telemetry\":" << telemetry.snapshot()
        << ",\"clock_or_power_changes\":false,\"gpu_services_stopped\":false," << execution.fields() << "}";
    report.emit(environment.str());
    std::vector<std::array<int, 3>> shapes;
    if (options.m) shapes.push_back({options.m, options.n, options.k});
    else if (options.quick) shapes = {{17, 19, 65}, {128, 128, 128}};
    else shapes = {{17, 19, 65}, {1024, 1024, 1024}, {4096, 1024, 4096},
                   {1024, 4096, 4096}, {4096, 4096, 4096}};
    const std::vector<std::string> modes = options.mode == "all"
        ? std::vector<std::string>{"zero", "random_unit_scale", "random_block_scales"}
        : std::vector<std::string>{options.mode};
    for (size_t i = 0; i < shapes.size(); ++i) for (const auto& mode : modes) {
      const auto shape = shapes[i];
      memory_guard(shape[0], shape[1], (shape[2] + 31) / 32 * 32, options);
      const Input in(shape[0], shape[1], shape[2], options.seed + uint32_t(i) * 7919u, mode);
      run_case(in, options, stream.value, blas.handle, telemetry, report,
               execution, actual_v100);
      ++completed_cases;
    }
    report.emit("{\"type\":\"summary\",\"passed\":true," + execution.fields() + ",\"actual_V100\":"
        + std::string(actual_v100 ? "true" : "false") + ",\"completed_cases\":" + std::to_string(completed_cases)
        + ",\"correctness_only\":" + (options.correctness_only ? "true" : "false")
        + ",\"end_to_end_training_measured\":false}");
    return 0;
  } catch (const std::exception& error) {
    const std::string record = "{\"type\":\"summary\",\"passed\":false," + execution.fields() + ",\"actual_V100\":"
        + (actual_v100 ? "true" : "false") + ",\"completed_cases\":" + std::to_string(completed_cases)
        + ",\"error\":" + quote(error.what()) + "}";
    try { report.emit(record); } catch (...) { std::cerr << record << '\n'; }
    return 1;
  }
}
