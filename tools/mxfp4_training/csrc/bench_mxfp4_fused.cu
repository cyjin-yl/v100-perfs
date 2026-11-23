// Bounded packed-only comparison for the fused certified variants.
// Compile with the same kernel translation units and variant defines as the
// certified benchmark, plus -lcublas -ldl. The shared helper translation unit
// references cuBLAS, but this entry point never constructs a cuBLAS handle or
// allocates global FP16 operands. Its explicit memory budget is independent
// of the unchanged packed/cuBLAS benchmark's budget.
#if !defined(MXFP4_CERTIFIED_VARIANT) || MXFP4_CERTIFIED_VARIANT < 3
#error "bench_mxfp4_fused requires a compiled certified variant >= 3"
#endif
#ifndef MXFP4_BENCH_TILED
#define MXFP4_BENCH_TILED
#endif
#ifndef MXFP4_BENCH_CERTIFIED
#define MXFP4_BENCH_CERTIFIED
#endif
#define main unused_full_comparison_main
#include "bench_mxfp4.cu"
#undef main
#include "mxfp4_gemm_fused.h"

// Optional build-supplied identities, never hashes of possibly changed files
// read at runtime. An external compilation manifest can supply these instead.
#ifndef MXFP4_PACKED_BENCH_SOURCE_SHA256
#define MXFP4_PACKED_BENCH_SOURCE_SHA256 ""
#endif
#ifndef MXFP4_PACKED_HELPERS_SOURCE_SHA256
#define MXFP4_PACKED_HELPERS_SOURCE_SHA256 ""
#endif

namespace {

constexpr int kPackedMaxMiB = 12;
constexpr int kPackedMinimumReserveMiB = 48;

struct PackedBudget {
  uint64_t inputs = 0, output = 0, workspace = 0;
  uint64_t total() const { return inputs + output + workspace; }
  std::string json() const {
    return "{\"packed_nibbles_and_scales\":" + std::to_string(inputs)
        + ",\"single_fp32_output\":" + std::to_string(output)
        + ",\"certified_workspace\":" + std::to_string(workspace)
        + ",\"total_explicit_gpu_bytes\":" + std::to_string(total()) + "}";
  }
};

PackedBudget packed_budget(int m, int n, int kp) {
  PackedBudget budget;
  budget.inputs = uint64_t(m + n) * (kp / 2 + kp / 32);
  budget.output = uint64_t(m) * n * sizeof(float);
  CERTIFIED(mxfp4_certified_workspace_bytes(m, n, kp, &budget.workspace));
  return budget;
}

struct PackedMemory {
  uint64_t free = 0, total = 0;
  std::string json() const {
    return "{\"free_bytes\":" + std::to_string(free)
        + ",\"total_bytes\":" + std::to_string(total) + "}";
  }
};

PackedMemory packed_memory_guard(const PackedBudget& budget, const Options& options,
                                 bool resident, const char* stage) {
  const uint64_t cap = uint64_t(options.max_mib) << 20;
  const uint64_t reserve = uint64_t(options.reserve_mib) << 20;
  if (options.max_mib <= 0 || options.max_mib > kPackedMaxMiB
      || options.reserve_mib < kPackedMinimumReserveMiB || budget.total() > cap)
    throw std::runtime_error(std::string(stage) + ": packed-only allocation cap refused "
        + std::to_string(budget.total()) + " explicit GPU bytes; cap is at most 12 MiB");
  size_t free = 0, total = 0;
  CUDA(cudaMemGetInfo(&free, &total));
  // Once arrays and workspace are resident, their bytes have already been
  // subtracted from free memory. Do not subtract the same allocation twice.
  if (free < reserve || (!resident && (budget.total() > uint64_t(free) - reserve
                                     || budget.total() > uint64_t(free) * 3 / 4)))
    throw std::runtime_error(std::string(stage) + ": insufficient free VRAM for packed-only "
        "allocations and the explicit " + std::to_string(options.reserve_mib)
        + " MiB reserve; no services are stopped");
  return {uint64_t(free), uint64_t(total)};
}

Options parse_packed(int argc, char** argv) {
  Options options;
  options.mode = "random_block_scales";
  bool explicit_reserve = false, explicit_cap = false, explicit_shape = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--correctness-only") options.correctness_only = true;
    else if (arg == "--allow-other-gpu") options.allow_other_gpu = true;
    else if (arg == "--help") {
      std::cout << "bench_mxfp4_fused --max-mib N --reserve-mib N\n"
          "  [--m M --n N --k K] [--device N] [--iters N --rounds N --warmup N]\n"
          "  [--mode all|zero|random_unit_scale|random_block_scales]\n"
          "  [--samples N] [--seed N] [--output new-file.jsonl]\n"
          "  [--correctness-only] [--allow-other-gpu]\n"
          "Requires an explicit allocation cap of 1..12 MiB and reserve of at least 48 MiB.\n"
          "Defaults: 1536 cubed, random_block_scales, 5 rounds, 10 iterations, 3 warmups.\n"
          "Compares packed baseline/tiled/fused pipelines; no ordinary cuBLAS handle or FP16 arrays.\n"
          "Both event and synchronized wall times include the complete certified pipeline.\n";
      std::exit(0);
    } else {
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
      const std::string text = argv[++i];
      if (arg == "--mode") { options.mode = text; continue; }
      if (arg == "--output") { options.output = text; continue; }
      size_t used = 0;
      const long long value = std::stoll(text, &used);
      if (used != text.size() || value < 0 || value > 2147483647)
        throw std::runtime_error("Invalid nonnegative integer for " + arg);
      if (arg == "--device") options.device = int(value);
      else if (arg == "--warmup") options.warmup = int(value);
      else if (arg == "--iters") options.iters = int(value);
      else if (arg == "--rounds") options.rounds = int(value);
      else if (arg == "--samples") options.samples = int(value);
      else if (arg == "--m") { options.m = int(value); explicit_shape = true; }
      else if (arg == "--n") { options.n = int(value); explicit_shape = true; }
      else if (arg == "--k") { options.k = int(value); explicit_shape = true; }
      else if (arg == "--max-mib") { options.max_mib = int(value); explicit_cap = true; }
      else if (arg == "--reserve-mib") { options.reserve_mib = int(value); explicit_reserve = true; }
      else if (arg == "--seed") options.seed = uint32_t(value);
      else throw std::runtime_error("Unknown argument " + arg);
    }
  }
  if (!explicit_cap || !explicit_reserve)
    throw std::runtime_error("Packed-only benchmark requires explicit --max-mib and --reserve-mib");
  if (options.max_mib <= 0 || options.max_mib > kPackedMaxMiB)
    throw std::runtime_error("Packed-only max-mib must be 1..12");
  if (options.reserve_mib < kPackedMinimumReserveMiB)
    throw std::runtime_error("Packed-only reserve-mib must be at least 48");
  if (options.iters <= 0 || options.iters > 10000 || options.rounds < 3 || options.rounds > 100
      || options.warmup > 10000 || options.samples <= 0 || options.samples > 65536)
    throw std::runtime_error("Require iters=1..10000, rounds=3..100, samples=1..65536 and warmup<=10000");
  if (explicit_shape && !(options.m > 0 && options.n > 0 && options.k > 0))
    throw std::runtime_error("Custom shapes require positive --m, --n and --k together");
  if (!explicit_shape) options.m = options.n = options.k = 1536;
  if (options.m > 65536 || options.n > 65536 || options.k > 65536)
    throw std::runtime_error("Each custom dimension is bounded to 65536");
  if (options.mode != "all" && options.mode != "zero" && options.mode != "random_unit_scale"
      && options.mode != "random_block_scales")
    throw std::runtime_error("Unknown input mode " + options.mode);
  return options;
}

std::string packed_source_fields() {
  const std::string source_hash = MXFP4_PACKED_BENCH_SOURCE_SHA256;
  const std::string helper_hash = MXFP4_PACKED_HELPERS_SOURCE_SHA256;
  return "\"source_file\":" + quote(__FILE__)
      + ",\"shared_helpers_source_file\":\"bench_mxfp4.cu\""
      + ",\"compiled_source_sha256\":" + (source_hash.empty() ? "null" : quote(source_hash))
      + ",\"compiled_shared_helpers_sha256\":" + (helper_hash.empty() ? "null" : quote(helper_hash));
}

std::string packed_case_prefix(const Input& in, const Options& options, const PackedBudget& budget) {
  return "\"benchmark_family\":\"bounded_packed_only_fused\",\"M\":" + std::to_string(in.m)
      + ",\"N\":" + std::to_string(in.n) + ",\"logical_K\":" + std::to_string(in.k)
      + ",\"padded_K\":" + std::to_string(in.kp) + ",\"input_mode\":" + quote(in.mode)
      + ",\"seed\":" + std::to_string(in.seed) + ",\"input_fnv1a64\":" + quote(in.fingerprint())
      + ",\"scale_exponent_range\":" + (in.mode == "random_unit_scale" ? "[0,0]" : "[-2,2]")
      + ",\"input_distribution\":" + quote(in.mode == "zero" ? "all_zero_codes" : "uniform_full_e2m1_nibbles")
      + ",\"real_training_tensor\":false,\"certified_variant\":" + std::to_string(MXFP4_CERTIFIED_VARIANT)
      + ",\"reserve_mib\":" + std::to_string(options.reserve_mib)
      + ",\"max_allocation_mib\":" + std::to_string(options.max_mib)
      + ",\"shared_device_output_count\":1,\"global_fp16_operands_allocated\":false"
      + ",\"ordinary_cublas_handle_created\":false,\"allocation_bytes\":" + std::to_string(budget.total())
      + ",\"allocation_breakdown\":" + budget.json();
}

void run_packed_case(const Input& in, const Options& options, cudaStream_t stream,
                     Telemetry& telemetry, Reporter& report, ExecutionState& execution,
                     bool actual_v100, const PackedBudget& budget) {
  const PackedMemory before_alloc = packed_memory_guard(budget, options, false, "before allocation");
  const size_t output_count = size_t(in.m) * in.n;
  Buffer<uint8_t> a(in.a.size()), as(in.as.size()), b(in.b.size()), bs(in.bs.size());
  Buffer<float> output(output_count);
  CertifiedWorkspace workspace(in.m, in.n, in.kp);
  if (workspace.bytes != budget.workspace)
    throw std::runtime_error("Certified workspace changed after memory-budget calculation");
  auto copy_in = [&](uint8_t* dst, const std::vector<uint8_t>& src) {
    CUDA(cudaMemcpyAsync(dst, src.data(), src.size(), cudaMemcpyHostToDevice, stream));
  };
  copy_in(a.data, in.a); copy_in(as.data, in.as);
  copy_in(b.data, in.b); copy_in(bs.data, in.bs);
  auto baseline_call = [&]() {
    execution.begin();
    CUDA(cudaError_t(mxfp4_gemm_cuda(a.data, as.data, b.data, bs.data, output.data,
                                    in.m, in.n, in.kp, stream)));
    execution.enqueue_succeeded();
  };
  auto tiled_call = [&]() {
    execution.begin();
    CUDA(cudaError_t(mxfp4_gemm_tiled_cuda(a.data, as.data, b.data, bs.data, output.data,
                                          in.m, in.n, in.kp, stream)));
    execution.enqueue_succeeded();
  };
  auto certified_call = [&]() {
    execution.begin_multi_stage();
    // No host decision or diagnostic read inside a measured invocation.
    // The API recomputes normalization metadata/certificate, launches gated
    // fused HMMA and the conditional exact fallback, all on this stream.
    CERTIFIED(mxfp4_certified_gemm(workspace.value, a.data, as.data, b.data,
                                  bs.data, output.data, stream, nullptr));
    execution.enqueue_succeeded();
  };
  auto copy_completed_output = [&](std::vector<float>& host_output) {
    CUDA(cudaStreamSynchronize(stream));
    execution.synchronization_succeeded();
    CUDA(cudaMemcpy(host_output.data(), output.data, output_count * sizeof(float), cudaMemcpyDeviceToHost));
  };
  auto poison_output = [&]() {
    // These finite generated inputs cannot legitimately produce this NaN
    // sentinel. A backend that leaves an earlier result untouched must fail.
    CUDA(cudaMemsetAsync(output.data, 0xff, output_count * sizeof(float), stream));
    CUDA(cudaStreamSynchronize(stream));
  };
  std::vector<float> baseline(output_count), tiled(output_count), certified(output_count);
  poison_output(); baseline_call(); copy_completed_output(baseline);
  poison_output(); tiled_call(); copy_completed_output(tiled);
  poison_output(); certified_call(); copy_completed_output(certified);
  MXFP4CertifiedInfo initial_info{};
  CERTIFIED(mxfp4_certified_get_info(workspace.value, &initial_info));
  validate_certificate_info(initial_info, in, budget.workspace);
  if (initial_info.submitted_calls != 1)
    throw std::runtime_error("Unexpected initial certified call count");

  bool exhaustive = false;
  const auto indices = reference_indices(in, options.samples, exhaustive);
  ErrorMetrics baseline_cpu, tiled_cpu, certified_cpu, tiled_baseline, certified_baseline;
  size_t baseline_nonfinite_values = 0;
  for (size_t index : indices) {
    const float expected = cpu_reference(in, index);
    baseline_cpu.add(baseline[index], expected, 0, 0);
    tiled_cpu.add(tiled[index], expected, 0, 0);
    certified_cpu.add(certified[index], expected, 0, 0);
  }
  for (size_t i = 0; i < output_count; ++i) {
    // Every generated workload has finite outputs. Check all baseline values
    // so matching NaN poison outside the CPU sample cannot mask unwritten holes.
    if (!std::isfinite(baseline[i])) ++baseline_nonfinite_values;
    tiled_baseline.add(tiled[i], baseline[i], 0, 0);
    certified_baseline.add(certified[i], baseline[i], 0, 0);
  }
  const bool passed = !baseline_nonfinite_values
      && !baseline_cpu.exact_mismatches && !tiled_cpu.exact_mismatches
      && !certified_cpu.exact_mismatches && !tiled_baseline.exact_mismatches
      && !certified_baseline.exact_mismatches;
  const std::string prefix = packed_case_prefix(in, options, budget);
  report.emit("{\"type\":\"correctness\"," + prefix + "," + execution.fields()
      + ",\"actual_V100\":" + (actual_v100 ? "true" : "false")
      + ",\"reference_scope\":" + quote(exhaustive ? "cpu_ordered_mx32_all_outputs" : "cpu_ordered_mx32_selected_outputs")
      + ",\"cpu_reference_values\":" + std::to_string(indices.size())
      + ",\"baseline_nonfinite_values_all_outputs\":" + std::to_string(baseline_nonfinite_values)
      + ",\"packed_equality\":\"bitwise_except_nan_payload_and_signed_zero\""
      + ",\"baseline_vs_cpu\":" + baseline_cpu.json() + ",\"tiled_vs_cpu\":" + tiled_cpu.json()
      + ",\"certified_vs_cpu\":" + certified_cpu.json()
      + ",\"tiled_vs_baseline_all_outputs\":" + tiled_baseline.json()
      + ",\"certified_vs_baseline_all_outputs\":" + certified_baseline.json()
      + ",\"certificate\":" + certificate_json(initial_info)
      + ",\"memory_before_allocation\":" + before_alloc.json()
      + ",\"passed\":" + (passed ? "true" : "false") + "}");
  if (!passed) throw std::runtime_error("Exact numerical comparison failed; timing is not attempted");
  packed_memory_guard(budget, options, true, "after correctness and kernel initialization");
  if (options.correctness_only) return;

  std::vector<Job> jobs = {
      {"mxfp4_baseline", "packed_to_fp32_including_decode_and_scale", baseline_call, {}, {}, true},
      {"mxfp4_tiled", "packed_to_fp32_including_decode_and_scale", tiled_call, {}, {}, true},
      {"mxfp4_certified_total", kCertifiedTotalScope, certified_call, {}, {}, true}};
  Events events;
  for (int i = 0; i < options.warmup; ++i) for (auto& job : jobs) job.launch();
  CUDA(cudaStreamSynchronize(stream));
  execution.synchronization_succeeded();
  const PackedMemory after_warmup = packed_memory_guard(budget, options, true, "after warmup");
  const std::string telemetry_before = telemetry.snapshot();
  std::vector<float> final_baseline(output_count);
  const std::array<std::vector<float>*, 3> final_outputs = {&final_baseline, &tiled, &certified};
  std::array<bool, 3> captured = {false, false, false};
  for (int round = 0; round < options.rounds; ++round) {
    for (size_t offset = 0; offset < jobs.size(); ++offset) {
      const size_t job_index = (size_t(round) + offset) % jobs.size();
      auto& job = jobs[job_index];
      // Final-result verification must not accept a previous backend's
      // unchanged output. Poisoning and its synchronization are untimed.
      if (round == options.rounds - 1) poison_output();
      const auto wall_start = std::chrono::steady_clock::now();
      CUDA(cudaEventRecord(events.start, stream));
      for (int iteration = 0; iteration < options.iters; ++iteration) job.launch();
      CUDA(cudaEventRecord(events.stop, stream));
      CUDA(cudaEventSynchronize(events.stop));
      const auto wall_stop = std::chrono::steady_clock::now();
      execution.synchronization_succeeded();
      float event_ms = 0;
      CUDA(cudaEventElapsedTime(&event_ms, events.start, events.stop));
      const double wall_ms = std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
      if (!(event_ms > 0) || !std::isfinite(event_ms) || !(wall_ms > 0) || !std::isfinite(wall_ms))
        throw std::runtime_error("Invalid event or synchronized wall timing");
      job.milliseconds.push_back(double(event_ms) / options.iters);
      job.wall_milliseconds.push_back(wall_ms / options.iters);
      if (round == options.rounds - 1) {
        // Capture each backend's actual LAST TIMED output before the next
        // backend reuses it. Both timing endpoints precede this D2H copy.
        CUDA(cudaMemcpy(final_outputs[job_index]->data(), output.data,
                        output_count * sizeof(float), cudaMemcpyDeviceToHost));
        captured[job_index] = true;
      }
    }
  }
  MXFP4CertifiedInfo after_info{};
  CERTIFIED(mxfp4_certified_get_info(workspace.value, &after_info));
  validate_certificate_info(after_info, in, budget.workspace);
  const uint64_t expected_calls = 1 + uint64_t(options.warmup)
      + uint64_t(options.iters) * options.rounds;
  if (!same_certificate(initial_info, after_info) || after_info.submitted_calls != expected_calls)
    throw std::runtime_error("Certificate changed or a measured invocation was omitted");
  std::array<ErrorMetrics, 3> final_errors;
  for (size_t job = 0; job < jobs.size(); ++job) {
    if (!captured[job]) throw std::runtime_error("A final timed output was not captured");
    for (size_t i = 0; i < output_count; ++i)
      final_errors[job].add((*final_outputs[job])[i], baseline[i], 0, 0);
    if (final_errors[job].exact_mismatches)
      throw std::runtime_error(jobs[job].name + ": final timed output differs from the validated baseline");
  }
  const PackedMemory after_timing = packed_memory_guard(budget, options, true, "after timing");
  const double useful_flops = 2.0 * in.m * in.n * in.k;
  std::string timing = "{\"type\":\"benchmark\"," + prefix + "," + execution.fields()
      + ",\"actual_V100\":" + (actual_v100 ? "true" : "false")
      + ",\"correctness_passed\":true,\"all_final_timed_outputs_checked\":true"
      + ",\"useful_flops\":" + number(useful_flops)
      + ",\"tflops_formula\":\"2*M*N*logical_K/(median_ms*1e9)\""
      + ",\"primary_clock\":\"synchronized_steady_clock\""
      + ",\"warmup_per_job\":" + std::to_string(options.warmup)
      + ",\"iterations_per_round\":" + std::to_string(options.iters)
      + ",\"rounds\":" + std::to_string(options.rounds)
      + ",\"backend_order\":\"rotating_each_round\",\"same_resident_inputs_reused\":true"
      + ",\"excluded_from_all_times\":[\"host_input_generation\",\"H2D\",\"high_precision_quantization\",\"allocation\",\"correctness_copy\",\"diagnostic_get_info\",\"output_poison_and_sync\"]"
      + ",\"certificate_after_timing\":" + certificate_json(after_info)
      + ",\"baseline_final_timed_vs_initial_all_outputs\":" + final_errors[0].json()
      + ",\"tiled_final_timed_vs_baseline_all_outputs\":" + final_errors[1].json()
      + ",\"certified_final_timed_vs_baseline_all_outputs\":" + final_errors[2].json()
      + ",\"memory_before_allocation\":" + before_alloc.json()
      + ",\"memory_after_warmup\":" + after_warmup.json()
      + ",\"memory_after_timing\":" + after_timing.json()
      + ",\"telemetry_before\":" + telemetry_before + ",\"telemetry_after\":" + telemetry.snapshot()
      + ",\"timings\":[";
  for (size_t i = 0; i < jobs.size(); ++i) {
    if (i) timing += ',';
    timing += timing_json(jobs[i], useful_flops, true);
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
    const Options options = parse_packed(argc, argv);
    report.open(options.output);
    CUDA(cudaSetDevice(options.device));
    cudaDeviceProp properties{};
    CUDA(cudaGetDeviceProperties(&properties, options.device));
    actual_v100 = properties.major == 7 && properties.minor == 0
        && std::string(properties.name).find("V100") != std::string::npos;
    if (!actual_v100 && !options.allow_other_gpu)
      throw std::runtime_error("Default benchmark requires a real V100; use --allow-other-gpu to label other hardware");
    if (properties.major != 7 || properties.minor != 0)
      throw std::runtime_error("The certified backend requires SM70");
    if (mxfp4_cuda_abi_version() != 1 || mxfp4_tiled_cuda_abi_version() != 1
        || mxfp4_fused_cuda_abi_version() != 1)
      throw std::runtime_error("Unsupported packed kernel ABI");
    CUDA(cudaFree(nullptr));
    Stream stream;
    char pci[64]{};
    CUDA(cudaDeviceGetPCIBusId(pci, sizeof(pci), options.device));
    int driver = 0, runtime = 0;
    CUDA(cudaDriverGetVersion(&driver)); CUDA(cudaRuntimeGetVersion(&runtime));
    size_t free = 0, total = 0;
    CUDA(cudaMemGetInfo(&free, &total));
    Telemetry telemetry(pci);
    report.emit("{\"type\":\"environment\",\"schema_version\":1,\"benchmark_family\":\"bounded_packed_only_fused\","
        + packed_source_fields() + ",\"device\":" + std::to_string(options.device)
        + ",\"name\":" + quote(properties.name) + ",\"pci_bus_id\":" + quote(pci)
        + ",\"compute_capability\":\"7.0\",\"actual_V100\":" + (actual_v100 ? "true" : "false")
        + ",\"sm_count\":" + std::to_string(properties.multiProcessorCount)
        + ",\"driver_version\":" + std::to_string(driver) + ",\"runtime_version\":" + std::to_string(runtime)
        + ",\"free_bytes_after_context_and_stream\":" + std::to_string(free)
        + ",\"total_bytes\":" + std::to_string(total)
        + ",\"baseline_backend\":" + quote(mxfp4_cuda_backend())
        + ",\"tiled_backend\":" + quote(mxfp4_tiled_cuda_backend())
        + ",\"certified_variant\":" + std::to_string(MXFP4_CERTIFIED_VARIANT)
        + ",\"certified_accepted_backend\":" + quote(kCertifiedAcceptedPath)
        + ",\"fused_kernel_backend\":" + quote(mxfp4_fused_cuda_backend(MXFP4_CERTIFIED_VARIANT - 3))
        + ",\"certified_abi_version\":" + std::to_string(MXFP4_CERTIFIED_ABI_VERSION)
        + ",\"ordinary_cublas_handle_created\":false,\"global_fp16_operands_allocated\":false"
        + ",\"reserve_mib\":" + std::to_string(options.reserve_mib)
        + ",\"max_explicit_allocation_mib\":" + std::to_string(options.max_mib)
        + ",\"hard_max_explicit_allocation_mib\":12,\"minimum_reserve_mib\":48"
        + ",\"memory_guards\":[\"before_allocation\",\"after_correctness\",\"after_warmup\",\"after_timing\"]"
        + ",\"primary_timing_clock\":\"synchronized_steady_clock\""
        + ",\"certified_diagnostics_in_timed_region\":false"
        + ",\"clock_or_power_changes\":false,\"gpu_services_stopped\":false"
        + ",\"telemetry\":" + telemetry.snapshot() + "," + execution.fields() + "}");
    const int kp = (options.k + 31) / 32 * 32;
    const PackedBudget budget = packed_budget(options.m, options.n, kp);
    const std::vector<std::string> modes = options.mode == "all"
        ? std::vector<std::string>{"zero", "random_unit_scale", "random_block_scales"}
        : std::vector<std::string>{options.mode};
    for (const auto& mode : modes) {
      packed_memory_guard(budget, options, false, "before input construction");
      const Input in(options.m, options.n, options.k, options.seed, mode);
      run_packed_case(in, options, stream.value, telemetry, report, execution, actual_v100, budget);
      ++completed_cases;
    }
    report.emit("{\"type\":\"summary\",\"benchmark_family\":\"bounded_packed_only_fused\",\"passed\":true,"
        + execution.fields() + ",\"actual_V100\":" + (actual_v100 ? "true" : "false")
        + ",\"completed_cases\":" + std::to_string(completed_cases)
        + ",\"correctness_only\":" + (options.correctness_only ? "true" : "false")
        + ",\"c2_or_cublas_comparison_performed\":false,\"end_to_end_training_measured\":false}");
    return 0;
  } catch (const std::exception& error) {
    const std::string record = "{\"type\":\"summary\",\"benchmark_family\":\"bounded_packed_only_fused\",\"passed\":false,"
        + execution.fields() + ",\"actual_V100\":" + (actual_v100 ? "true" : "false")
        + ",\"completed_cases\":" + std::to_string(completed_cases) + ",\"error\":" + quote(error.what()) + "}";
    try { report.emit(record); } catch (...) { std::cerr << record << '\n'; }
    return 1;
  }
}
