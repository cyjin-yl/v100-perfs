// Standalone C2 algorithm screen. Compile against the frozen production csrc.
// No copied quantizer, prepare kernel, certificate, epilogue, or input generator.
#ifndef MXFP4_CERTIFIED_VARIANT
#define MXFP4_CERTIFIED_VARIANT 2
#endif
#if MXFP4_CERTIFIED_VARIANT != 2
#error "This screen requires the unchanged certified2 implementation"
#endif
#define MXFP4_BENCH_TILED 1
#define MXFP4_BENCH_CERTIFIED 1
#define main unused_original_bench_mxfp4_main
#include "bench_mxfp4.cu"
#undef main
// Inspected both anonymous namespaces: their defined member names are distinct.
// Including the production TU exposes Context and the exact postscale kernel.
#include "mxfp4_gemm_certified.cu"

#include <cstring>

namespace c2_algorithm_screen {

constexpr uint64_t kMaximumBytes = 1ull << 30;
constexpr uint64_t kReserveBytes = 512ull << 20;
constexpr int kWarmup = 100, kRounds = 5, kIterations = 10;

struct ScreenOptions {
  int size = 4096, device = 0;
  bool ordinary = false;
  std::string mode = "random_block_scales", output, build_binding;
};

ScreenOptions parse_screen(int argc, char** argv) {
  ScreenOptions value;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--include-ordinary") { value.ordinary = true; continue; }
    if (++i >= argc) throw std::runtime_error("Missing argument after " + key);
    const std::string argument = argv[i];
    if (key == "--size") {
      if (argument != "4096" && argument != "8192") throw std::runtime_error("size must be 4096 or 8192");
      value.size = std::stoi(argument);
    } else if (key == "--device") {
      if (argument != "0") throw std::runtime_error("This screen explicitly uses device 0");
    } else if (key == "--mode") value.mode = argument;
    else if (key == "--output") value.output = argument;
    else if (key == "--build-binding-sha256") value.build_binding = argument;
    else throw std::runtime_error("Unknown option " + key);
  }
  if (value.output.empty() || value.build_binding.size() != 64 ||
      value.build_binding.find_first_not_of("0123456789abcdef") != std::string::npos)
    throw std::runtime_error("A new --output and verified --build-binding-sha256 are required; use candidate.py run");
  if (value.mode != "random_block_scales" && value.mode != "random_unit_scale")
    throw std::runtime_error("Only the two explicitly labelled production Input modes are permitted");
  return value;
}

struct Algorithm {
  cublasGemmAlgo_t id;
  Job timing;
  uint64_t successful_calls = 0;
  bool final_checked = false;
};

std::vector<Algorithm> algorithms(bool ordinary) {
  std::vector<Algorithm> result;
  auto append = [&](cublasGemmAlgo_t id, std::string name) {
    Algorithm a;
    a.id = id;
    a.timing.name = std::move(name);
    a.timing.scope = "normalize_gemm_compute";
    a.timing.primary_wall_clock = true;
    result.push_back(std::move(a));
  };
  append(CUBLAS_GEMM_DEFAULT, "CUBLAS_GEMM_DEFAULT");
  append(CUBLAS_GEMM_DEFAULT_TENSOR_OP, "CUBLAS_GEMM_DEFAULT_TENSOR_OP");
  // These explicit names deliberately fail compilation if an API drops them.
  const cublasGemmAlgo_t tc[] = {
    CUBLAS_GEMM_ALGO0_TENSOR_OP, CUBLAS_GEMM_ALGO1_TENSOR_OP,
    CUBLAS_GEMM_ALGO2_TENSOR_OP, CUBLAS_GEMM_ALGO3_TENSOR_OP,
    CUBLAS_GEMM_ALGO4_TENSOR_OP, CUBLAS_GEMM_ALGO5_TENSOR_OP,
    CUBLAS_GEMM_ALGO6_TENSOR_OP, CUBLAS_GEMM_ALGO7_TENSOR_OP,
    CUBLAS_GEMM_ALGO8_TENSOR_OP, CUBLAS_GEMM_ALGO9_TENSOR_OP,
    CUBLAS_GEMM_ALGO10_TENSOR_OP, CUBLAS_GEMM_ALGO11_TENSOR_OP,
    CUBLAS_GEMM_ALGO12_TENSOR_OP, CUBLAS_GEMM_ALGO13_TENSOR_OP,
    CUBLAS_GEMM_ALGO14_TENSOR_OP, CUBLAS_GEMM_ALGO15_TENSOR_OP};
  for (int i = 0; i < 16; ++i) append(tc[i], "CUBLAS_GEMM_ALGO" + std::to_string(i) + "_TENSOR_OP");
  if (ordinary) {
    const cublasGemmAlgo_t regular[] = {
      CUBLAS_GEMM_ALGO0, CUBLAS_GEMM_ALGO1, CUBLAS_GEMM_ALGO2, CUBLAS_GEMM_ALGO3,
      CUBLAS_GEMM_ALGO4, CUBLAS_GEMM_ALGO5, CUBLAS_GEMM_ALGO6, CUBLAS_GEMM_ALGO7,
      CUBLAS_GEMM_ALGO8, CUBLAS_GEMM_ALGO9, CUBLAS_GEMM_ALGO10, CUBLAS_GEMM_ALGO11,
      CUBLAS_GEMM_ALGO12, CUBLAS_GEMM_ALGO13, CUBLAS_GEMM_ALGO14, CUBLAS_GEMM_ALGO15,
      CUBLAS_GEMM_ALGO16, CUBLAS_GEMM_ALGO17, CUBLAS_GEMM_ALGO18, CUBLAS_GEMM_ALGO19,
      CUBLAS_GEMM_ALGO20, CUBLAS_GEMM_ALGO21, CUBLAS_GEMM_ALGO22, CUBLAS_GEMM_ALGO23};
    for (int i = 0; i < 24; ++i) append(regular[i], "CUBLAS_GEMM_ALGO" + std::to_string(i));
  }
  return result;
}

struct Comparison {
  uint64_t values = 0, mismatches = 0, nonfinite = 0, negative_zero = 0;
  size_t first_bad = std::numeric_limits<size_t>::max();
  void add(float actual, float expected, size_t index) {
    ++values;
    uint32_t a, b;
    std::memcpy(&a, &actual, 4); std::memcpy(&b, &expected, 4);
    if (!std::isfinite(actual) || !std::isfinite(expected)) ++nonfinite;
    if (a == 0x80000000u) ++negative_zero;
    const bool equal = a == b || ((a & 0x7fffffffu) == 0 && (b & 0x7fffffffu) == 0);
    if (!equal || !std::isfinite(actual) || !std::isfinite(expected)) {
      ++mismatches; if (first_bad == std::numeric_limits<size_t>::max()) first_bad = index;
    }
  }
  std::string json() const {
    return "{\"values\":" + std::to_string(values) + ",\"exact_mismatches\":" + std::to_string(mismatches)
      + ",\"nonfinite_values\":" + std::to_string(nonfinite) + ",\"negative_zero_values\":" + std::to_string(negative_zero)
      + ",\"first_bad_index\":" + (first_bad == std::numeric_limits<size_t>::max() ? "null" : std::to_string(first_bad))
      + ",\"comparison\":\"bitwise_except_signed_zero; nonfinite always fails\"}";
  }
};

void require_comparison(const Comparison& c, const char* operation, bool canonical_positive_zero = true) {
  if (c.mismatches || c.nonfinite || (canonical_positive_zero && c.negative_zero))
    throw std::runtime_error(std::string(operation) + ": " + c.json());
}

Comparison compare_device(const float* device, const std::vector<float>& reference, cudaStream_t stream) {
  std::vector<float> chunk(std::min<size_t>(reference.size(), 1u << 20));
  Comparison result;
  for (size_t offset = 0; offset < reference.size(); offset += chunk.size()) {
    const size_t count = std::min(chunk.size(), reference.size() - offset);
    CUDA(cudaMemcpyAsync(chunk.data(), device + offset, count * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < count; ++i) result.add(chunk[i], reference[offset + i], offset + i);
  }
  return result;
}

void epilogue(Context* c, float* output, cudaStream_t stream) {
  const dim3 grid(unsigned((uint64_t(c->n) + 1023) / 1024), unsigned(c->m < 65535 ? c->m : 65535));
  // Exact production variant2 epilogue; cudaMalloc + the allowed sizes imply its aligned branch.
  if ((reinterpret_cast<uintptr_t>(output) & 15u) == 0 && c->n % 4 == 0)
    postscale_rows_if_certified<true><<<grid, 256, 0, stream>>>(output, c->rows, c->metadata, c->m, c->n);
  else
    postscale_rows_if_certified<false><<<grid, 256, 0, stream>>>(output, c->rows, c->metadata, c->m, c->n);
  CUDA(cudaGetLastError());
  CUDA(cudaStreamSynchronize(stream));
}

size_t free_memory_guard(const char* phase, uint64_t initial_requested = 0) {
  size_t free = 0, total = 0;
  CUDA(cudaMemGetInfo(&free, &total));
  if (free < kReserveBytes || initial_requested > kMaximumBytes ||
      initial_requested > uint64_t(free) - kReserveBytes || initial_requested > uint64_t(free) * 3 / 4)
    throw std::runtime_error(std::string("Memory guard: ") + phase + ", free=" + std::to_string(free)
                             + ", explicit_requested=" + std::to_string(initial_requested));
  return free;
}

int screen(const ScreenOptions& options, Reporter& reporter, ExecutionState& execution) {
  CUDA(cudaSetDevice(options.device));
  cudaDeviceProp properties{};
  CUDA(cudaGetDeviceProperties(&properties, options.device));
  if (properties.major != 7 || properties.minor != 0 || std::string(properties.name).find("V100") == std::string::npos)
    throw std::runtime_error("Requires an actual V100 SM70; no fallback device");
  CUDA(cudaFree(nullptr));
  const int n = options.size;
  uint64_t workspace_bytes = 0;
  CERTIFIED(mxfp4_certified_workspace_bytes(n, n, n, &workspace_bytes));
  const uint64_t packed_bytes = uint64_t(n) * 2 * (n / 2 + n / 32);
  const uint64_t output_bytes = uint64_t(n) * n * sizeof(float);
  const uint64_t allocation_bytes = packed_bytes + output_bytes + workspace_bytes;
  const size_t free_before = free_memory_guard("before allocations", allocation_bytes);
  const Input input(n, n, n, 20261123u, options.mode);
  const std::string fingerprint = input.fingerprint();
  if (n == 4096 && options.mode == "random_block_scales" && fingerprint != "6a76b74ce5d1d307")
    throw std::runtime_error("Production Input 4096 fingerprint differs from B");
  Stream stream;
  Buffer<uint8_t> da(input.a.size()), das(input.as.size()), db(input.b.size()), dbs(input.bs.size());
  Buffer<float> output(size_t(n) * n);
  CertifiedWorkspace owned(n, n, n);
  Context* c = static_cast<Context*>(owned.value);
  const size_t free_after = free_memory_guard("after allocation/context");
  auto copy = [&](uint8_t* dst, const std::vector<uint8_t>& src) {
    CUDA(cudaMemcpyAsync(dst, src.data(), src.size(), cudaMemcpyHostToDevice, stream.value));
  };
  copy(da.data, input.a); copy(das.data, input.as); copy(db.data, input.b); copy(dbs.data, input.bs);
  auto poison = [&]() {
    CUDA(cudaMemsetAsync(output.data, 0x7f, output_bytes, stream.value));
    CUDA(cudaStreamSynchronize(stream.value));
  };
  poison();
  execution.begin_multi_stage();
  CERTIFIED(mxfp4_certified_gemm(c, da.data, das.data, db.data, dbs.data, output.data, stream.value, nullptr));
  execution.enqueue_succeeded();
  CUDA(cudaStreamSynchronize(stream.value)); execution.synchronization_succeeded();
  MXFP4CertifiedInfo info{};
  CERTIFIED(mxfp4_certified_get_info(c, &info));
  validate_certificate_info(info, input, workspace_bytes);
  if (info.submitted_calls != 1) throw std::runtime_error("Expected one original C2 prerequisite call");
  int driver = 0, runtime = 0, blas_version = 0;
  CUDA(cudaDriverGetVersion(&driver)); CUDA(cudaRuntimeGetVersion(&runtime));
  BLAS(cublasGetVersion(c->blas, &blas_version));
  cublasMath_t math;
  cublasPointerMode_t pointer_mode;
  BLAS(cublasGetMathMode(c->blas, &math)); BLAS(cublasGetPointerMode(c->blas, &pointer_mode));
  if (math != cublasMath_t(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)
      || pointer_mode != CUBLAS_POINTER_MODE_HOST || kBlasWorkspace != 4u * 1024u * 1024u)
    throw std::runtime_error("C2 handle contract differs from the intended screening configuration");
  // The original C2 call has already bound its stream and user workspace.
  // Do not reset the stream (which would reset the workspace) inside timing.
  cudaStream_t bound_stream = nullptr;
  BLAS(cublasGetStream(c->blas, &bound_stream));
  if (bound_stream != stream.value) throw std::runtime_error("Unexpected C2 stream binding");
  char pci[64]{};
  CUDA(cudaDeviceGetPCIBusId(pci, sizeof(pci), options.device));
  Telemetry telemetry(pci);
  const std::string identity = "\"M\":" + std::to_string(n) + ",\"N\":" + std::to_string(n)
    + ",\"K\":" + std::to_string(n) + ",\"mode\":" + quote(options.mode)
    + ",\"seed\":20261123,\"input_fnv1a64\":" + quote(fingerprint);
  reporter.emit("{\"type\":\"screen_environment\",\"schema_version\":1," + identity
    + ",\"gpu_name\":" + quote(properties.name) + ",\"actual_SM70\":true,\"actual_V100\":true"
    + ",\"driver_version\":" + std::to_string(driver) + ",\"runtime_version\":" + std::to_string(runtime)
    + ",\"cublas_version\":" + std::to_string(blas_version) + ",\"cublas_math_flags\":" + std::to_string(int(math))
    + ",\"A_B_type\":\"CUDA_R_16F\",\"C_type\":\"CUDA_R_32F\",\"compute_type\":\"CUBLAS_COMPUTE_32F\""
    + ",\"alpha\":1,\"beta\":0,\"pointer_mode\":\"HOST\",\"transa\":\"T\",\"transb\":\"N\""
    + ",\"explicit_allocation_bytes\":" + std::to_string(allocation_bytes)
    + ",\"workspace_bytes\":" + std::to_string(workspace_bytes) + ",\"blas_workspace_bytes\":" + std::to_string(kBlasWorkspace)
    + ",\"max_allocation_bytes\":" + std::to_string(kMaximumBytes) + ",\"reserve_bytes\":" + std::to_string(kReserveBytes)
    + ",\"free_bytes_before\":" + std::to_string(free_before) + ",\"free_bytes_after_allocations\":" + std::to_string(free_after)
    + ",\"build_binding_sha256\":" + quote(options.build_binding)
    + ",\"timescope\":\"normalize_gemm_compute\",\"normalization_timed\":false,\"certificate_timed\":false"
    + ",\"epilogue_timed\":false,\"fullpipeline_performance_measured\":false,\"warmup_per_supported_algorithm\":100"
    + ",\"rounds\":5,\"iterations_per_round\":10,\"order\":\"rotate_supported_algorithms_each_round\""
    + ",\"services_stopped\":false,\"clock_or_power_changes\":false,\"exclusive_gpu_claim\":false"
    + ",\"excluded\":[\"production_prepare_and_certificate\",\"production_epilogue\",\"fallback_gate\",\"input_generation\",\"allocations\",\"H2D_D2H\",\"correctness\",\"poison_and_sync\"]"
    + ",\"certificate\":" + certificate_json(info) + ",\"telemetry\":" + telemetry.snapshot() + "}");
  if (!info.accepted || info.reject_flags != 0) {
    reporter.emit("{\"type\":\"screen_summary\",\"status\":\"blocked_certificate\",\"passed\":false," + identity
      + ",\"algorithm_timing_executed\":false,\"fullpipeline_performance_measured\":false,\"throughput\":null,"
      + execution.fields() + ",\"certificate\":" + certificate_json(info) + "}");
    return 77;
  }
  // Preserve the entire original C2 output before any other backend can overwrite it.
  std::vector<float> reference(size_t(n) * n);
  CUDA(cudaMemcpyAsync(reference.data(), output.data, output_bytes, cudaMemcpyDeviceToHost, stream.value));
  CUDA(cudaStreamSynchronize(stream.value));
  uint64_t finite = 0, reference_negative_zero = 0;
  for (float value : reference) {
    finite += std::isfinite(value);
    reference_negative_zero += value == 0.0f && std::signbit(value);
  }
  if (finite != reference.size() || reference_negative_zero)
    throw std::runtime_error("Original C2 full output violates finite/canonical-positive-zero contract");
  bool exhaustive = false;
  const auto selected = reference_indices(input, 256, exhaustive);
  Comparison oracle;
  for (size_t index : selected) oracle.add(reference[index], cpu_reference(input, index), index);
  require_comparison(oracle, "Original C2 versus production CPU ordered-MX32 oracle");
  poison();
  execution.begin();
  CUDA(cudaError_t(mxfp4_gemm_tiled_cuda(da.data, das.data, db.data, dbs.data, output.data, n, n, n, stream.value)));
  execution.enqueue_succeeded();
  CUDA(cudaStreamSynchronize(stream.value)); execution.synchronization_succeeded();
  const Comparison tiled = compare_device(output.data, reference, stream.value);
  // Tiled's existing equality contract permits signed zero. C2 and every
  // postscaled candidate separately require canonical positive zero.
  require_comparison(tiled, "Full original C2 versus exact tiled", false);
  reporter.emit("{\"type\":\"screen_prerequisite\",\"passed\":true," + identity
    + ",\"original_c2_finite_outputs\":" + std::to_string(finite)
    + ",\"original_c2_negative_zero_outputs\":" + std::to_string(reference_negative_zero)
    + ",\"original_c2_vs_cpu_selected\":" + oracle.json() + ",\"cpu_reference_exhaustive\":false"
    + ",\"original_c2_vs_tiled_all_outputs\":" + tiled.json()
    + ",\"candidate_vs_tiled_method\":\"exact transitivity through complete original C2 array\"}");

  auto requested = algorithms(options.ordinary);
  std::vector<Algorithm> supported;
  size_t not_supported = 0;
  auto raw_gemm = [&](Algorithm& algorithm) {
    const float alpha = 1, beta = 0;
    execution.begin_multi_stage();
    const cublasStatus_t status = cublasGemmEx(c->blas, CUBLAS_OP_T, CUBLAS_OP_N,
      n, n, n, &alpha, c->normalized + size_t(n) * n, CUDA_R_16F, n,
      c->normalized, CUDA_R_16F, n, &beta, output.data, CUDA_R_32F, n,
      CUBLAS_COMPUTE_32F, algorithm.id);
    if (status == CUBLAS_STATUS_SUCCESS) {
      ++algorithm.successful_calls;
      execution.enqueue_succeeded();
    }
    return status;
  };
  Events events;
  CUDA(cudaEventRecord(events.start, stream.value)); CUDA(cudaEventRecord(events.stop, stream.value));
  CUDA(cudaEventSynchronize(events.stop));
  for (auto& algorithm : requested) {
    poison();
    const cublasStatus_t status = raw_gemm(algorithm);
    CUDA(cudaStreamSynchronize(stream.value)); execution.synchronization_succeeded();
    if (status == CUBLAS_STATUS_NOT_SUPPORTED) {
      ++not_supported;
      reporter.emit("{\"type\":\"algorithm_result\",\"status\":\"not_supported\",\"passed\":false," + identity
        + ",\"algorithm\":" + quote(algorithm.timing.name) + ",\"algorithm_id\":" + std::to_string(int(algorithm.id))
        + ",\"cublas_status\":" + std::to_string(int(status)) + ",\"timing\":null,\"correctness_verified\":false}");
      continue;
    }
    if (status != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error(algorithm.timing.name + ": cuBLAS probe status " + std::to_string(int(status)));
    epilogue(c, output.data, stream.value);
    const Comparison initial = compare_device(output.data, reference, stream.value);
    require_comparison(initial, algorithm.timing.name.c_str());
    for (int i = 0; i < kWarmup; ++i) BLAS(raw_gemm(algorithm));
    CUDA(cudaStreamSynchronize(stream.value)); execution.synchronization_succeeded();
    free_memory_guard("after algorithm warmup");
    reporter.emit("{\"type\":\"algorithm_initial_check\",\"passed\":true," + identity
      + ",\"algorithm\":" + quote(algorithm.timing.name) + ",\"algorithm_id\":" + std::to_string(int(algorithm.id))
      + ",\"initial_scaled_output_vs_original_c2_all_outputs\":" + initial.json()
      + ",\"warmup_completed\":100}");
    supported.push_back(std::move(algorithm));
  }
  if (supported.empty()) throw std::runtime_error("No supported and verified GEMM algorithm");
  const double flops = 2.0 * n * n * n;
  for (int round = 0; round < kRounds; ++round) {
    reporter.emit("{\"type\":\"round_environment\",\"round\":" + std::to_string(round)
      + ",\"telemetry\":" + telemetry.snapshot() + "}");
    for (size_t j = 0; j < supported.size(); ++j) {
      Algorithm& algorithm = supported[(j + size_t(round)) % supported.size()];
      poison(); // Strictly outside both endpoints, including the final round.
      const auto begin = std::chrono::steady_clock::now();
      CUDA(cudaEventRecord(events.start, stream.value));
      for (int i = 0; i < kIterations; ++i) BLAS(raw_gemm(algorithm));
      CUDA(cudaEventRecord(events.stop, stream.value));
      CUDA(cudaEventSynchronize(events.stop));
      const auto end = std::chrono::steady_clock::now();
      execution.synchronization_succeeded();
      float elapsed = 0;
      CUDA(cudaEventElapsedTime(&elapsed, events.start, events.stop));
      const double event_ms = elapsed / kIterations;
      const double wall_ms = std::chrono::duration<double, std::milli>(end - begin).count() / kIterations;
      if (!(event_ms > 0 && wall_ms > 0 && std::isfinite(event_ms) && std::isfinite(wall_ms)))
        throw std::runtime_error("Invalid measured interval");
      algorithm.timing.milliseconds.push_back(event_ms);
      algorithm.timing.wall_milliseconds.push_back(wall_ms);
      if (round + 1 == kRounds) {
        // These are the LAST ACTUAL TIMED outputs. Do not rerun GEMM here.
        epilogue(c, output.data, stream.value);
        const Comparison final = compare_device(output.data, reference, stream.value);
        require_comparison(final, algorithm.timing.name.c_str());
        algorithm.final_checked = true;
        free_memory_guard("after final timed output validation");
        reporter.emit("{\"type\":\"algorithm_result\",\"status\":\"passed\",\"passed\":true," + identity
          + ",\"algorithm\":" + quote(algorithm.timing.name) + ",\"algorithm_id\":" + std::to_string(int(algorithm.id))
          + ",\"cublas_status\":0,\"timescope\":\"normalize_gemm_compute\",\"fullpipeline_performance_measured\":false"
          + ",\"successful_gemm_calls\":" + std::to_string(algorithm.successful_calls)
          + ",\"final_actual_timed_output_checked\":true,\"epilogue_applied_after_timer\":true"
          + ",\"final_scaled_output_vs_original_c2_all_outputs\":" + final.json()
          + ",\"final_scaled_output_vs_tiled_verified_by_full_reference_transitivity\":true"
          + ",\"timing\":" + timing_json(algorithm.timing, flops, true) + "}");
      }
    }
  }
  MXFP4CertifiedInfo after{};
  CERTIFIED(mxfp4_certified_get_info(c, &after));
  if (!same_certificate(info, after) || after.submitted_calls != 1)
    throw std::runtime_error("Original certificate/source context changed during screening");
  for (const auto& a : supported)
    if (!a.final_checked || a.timing.milliseconds.size() != kRounds || a.successful_calls != 1 + kWarmup + kRounds * kIterations)
      throw std::runtime_error("Incomplete algorithm run");
  const auto best = std::min_element(supported.begin(), supported.end(), [](const Algorithm& a, const Algorithm& b) {
    return median_ms(a.timing.wall_milliseconds) < median_ms(b.timing.wall_milliseconds);
  });
  reporter.emit("{\"type\":\"screen_summary\",\"status\":\"passed\",\"passed\":true," + identity
    + ",\"requested_algorithms\":" + std::to_string(requested.size())
    + ",\"supported_verified_algorithms\":" + std::to_string(supported.size())
    + ",\"not_supported_algorithms\":" + std::to_string(not_supported)
    + ",\"timescope\":\"normalize_gemm_compute\",\"fullpipeline_performance_measured\":false"
    + ",\"best_screened_wall_algorithm\":" + quote(best->timing.name)
    + ",\"selection_requires_independent_fullpipeline_ABBA\":true,\"round_samples_are_correlated\":true,"
    + execution.fields() + ",\"certificate_unchanged\":true,\"telemetry\":" + telemetry.snapshot() + "}");
  // Direct compute calls used the same thread/device/stream as the owning C2
  // context; every call has finished before the production destructor runs.
  return 0;
}

} // namespace c2_algorithm_screen

int main(int argc, char** argv) {
  Reporter reporter;
  ExecutionState execution;
  try {
    const auto options = c2_algorithm_screen::parse_screen(argc, argv);
    reporter.open(options.output);
    return c2_algorithm_screen::screen(options, reporter, execution);
  } catch (const std::exception& error) {
    reporter.emit("{\"type\":\"screen_summary\",\"status\":\"failed\",\"passed\":false,"
      + execution.fields() + ",\"fullpipeline_performance_measured\":false,\"error\":" + quote(error.what()) + "}");
    return 1;
  }
}
