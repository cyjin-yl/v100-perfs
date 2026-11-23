// Reuse the independent ordered-MX32 CPU oracle, then exercise certificate
// boundaries and accepted -> rejected -> accepted workspace reuse.
// This executable performs correctness checks only, never performance timing.
#include "mxfp4_gemm_certified.h"
#include "mxfp4_gemm_tiled.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef MXFP4_CERTIFIED_VARIANT
#define MXFP4_CERTIFIED_VARIANT 0
#endif

namespace certified_checks {
MXFP4CertifiedInfo last_info{};
uint64_t accepted = 0, rejected = 0, empty = 0;
uint64_t canonical_zero_values = 0;
bool any_completed = false, pipeline_attempted = false;
#if MXFP4_CERTIFIED_VARIANT == 8
bool c8_fixture_active = false;
unsigned c8_a_alignment = 0, c8_b_alignment = 0, c8_output_alignment = 0;
#endif

void checked(int status, const char* operation) {
  if (status) throw std::runtime_error(std::string(operation) + ": "
                                      + mxfp4_certified_error_string(status));
}

uint64_t extra_bytes(int64_t m, int64_t n, int64_t k) {
  uint64_t bytes = 0;
  checked(mxfp4_certified_workspace_bytes(m, n, k, &bytes), "workspace_bytes");
  return bytes;
}

int adapter(const uint8_t* a, const uint8_t* as, const uint8_t* b,
            const uint8_t* bs, float* out, int64_t m, int64_t n,
            int64_t k, cudaStream_t stream) {
  if (m < 0 || n < 0 || k < 0 || k % 32 != 0) return int(cudaErrorInvalidValue);
  if (m == 0 || n == 0) return int(cudaSuccess);
  if (!out || (k && (!a || !as || !b || !bs))) return int(cudaErrorInvalidValue);
  if (k == 0) ++empty;
  void* context = nullptr;
  int status = mxfp4_certified_create(m, n, k, &context);
  if (status) {
    std::cerr << "certified create: " << mxfp4_certified_error_string(status) << '\n';
    return status >= MXFP4_CERTIFIED_CUBLAS_ERROR_BASE ? int(cudaErrorUnknown) : status;
  }
  int used = -1;
#if MXFP4_CERTIFIED_VARIANT == 8
  if (c8_fixture_active) {
    c8_a_alignment = unsigned(reinterpret_cast<uintptr_t>(a) & 7u);
    c8_b_alignment = unsigned(reinterpret_cast<uintptr_t>(b) & 7u);
    c8_output_alignment = unsigned(reinterpret_cast<uintptr_t>(out) & 7u);
    // A finite sentinel differs from every dedicated fixture's finite output
    // and also exposes unwritten outputs where the reference expects NaN.
    status = int(cudaMemsetAsync(out, 0x7f, size_t(m) * n * sizeof(float), stream));
    if (status) {
      mxfp4_certified_destroy(context);
      return status;
    }
  }
#endif
  pipeline_attempted = true;
  status = mxfp4_certified_gemm(context, a, as, b, bs, out, stream, &used);
  if (!status) {
    any_completed = true;
    status = mxfp4_certified_get_info(context, &last_info);
  }
  if (!status && (last_info.abi_version != MXFP4_CERTIFIED_ABI_VERSION
                 || !last_info.metadata_valid || last_info.submitted_calls != 1
                 || used != int(last_info.accepted)))
    status = int(cudaErrorUnknown);
  if (!status) {
    if (used) ++accepted;
    else ++rejected;
  }
  if (!status && used) {
    std::vector<float> actual(size_t(m) * n);
    status = int(cudaMemcpyAsync(actual.data(), out, actual.size() * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
    if (!status) status = int(cudaStreamSynchronize(stream));
    if (!status) for (float value : actual) {
      if (value == 0.0f) {
        if (std::signbit(value)) { status = int(cudaErrorUnknown); break; }
        ++canonical_zero_values;
      }
    }
  }
  const int destroy_status = mxfp4_certified_destroy(context);
  if (!status) status = destroy_status;
  if (status >= MXFP4_CERTIFIED_CUBLAS_ERROR_BASE) {
    std::cerr << "certified adapter: " << mxfp4_certified_error_string(status) << '\n';
    return int(cudaErrorUnknown);
  }
  return status;
}
}  // namespace certified_checks

extern "C" int certified_check_gemm(
    const uint8_t* a, const uint8_t* as, const uint8_t* b, const uint8_t* bs,
    float* out, int64_t m, int64_t n, int64_t k, cudaStream_t stream) {
  return certified_checks::adapter(a, as, b, bs, out, m, n, k, stream);
}
extern "C" int certified_check_abi(void) { return 1; }
extern "C" const char* certified_check_backend(void) {
#if MXFP4_CERTIFIED_VARIANT >= 3
  return "sm70_mx32_certified_fused_hmma_or_exact_tiled_v1";
#else
  return "sm70_mx32_certified_cublas_or_exact_tiled_v1";
#endif
}

// Compile the same exhaustive E8M0 / E2M1 / tail oracle with this adapter.
// All explicit workspace allocations participate in its memory guard.
#define MXFP4_CHECK_EXTRA_BYTES(m, n, k) certified_checks::extra_bytes(m, n, k)
#define MXFP4_CHECK_SMALL_MEMORY_ALLOWED (MXFP4_CERTIFIED_VARIANT >= 3)
#define mxfp4_gemm_cuda certified_check_gemm
#define mxfp4_cuda_abi_version certified_check_abi
#define mxfp4_cuda_backend certified_check_backend
#define main ordered_mx32_checks_main
#include "check_mxfp4.cu"
#undef main
#undef mxfp4_gemm_cuda
#undef mxfp4_cuda_abi_version
#undef mxfp4_cuda_backend
#undef MXFP4_CHECK_EXTRA_BYTES
#undef MXFP4_CHECK_SMALL_MEMORY_ALLOWED

namespace {

void expect_certificate(bool accepted, uint32_t required_reject, const char* name) {
  const auto& info = certified_checks::last_info;
  if (bool(info.accepted) != accepted || (accepted && info.reject_flags)
      || (info.reject_flags & required_reject) != required_reject)
    throw std::runtime_error(std::string(name) + ": incorrect certificate decision");
  std::cout << "{\"type\":\"certificate_decision\",\"case\":" << json_string(name)
            << ",\"accepted\":" << (info.accepted ? "true" : "false")
            << ",\"reject_flags\":" << info.reject_flags
            << ",\"max_energy_a\":" << info.max_energy_a
            << ",\"max_energy_b\":" << info.max_energy_b
            << ",\"max_span_a\":" << info.max_span_a
            << ",\"max_span_b\":" << info.max_span_b
            << ",\"lambda_min\":" << info.min_scale_a + info.min_scale_b - 256
            << ",\"lambda_max\":" << info.max_scale_a + info.max_scale_b - 256
            << ",\"workspace_bytes\":" << info.workspace_bytes
            << ",\"passed\":true}" << '\n';
}

Case impulse(const char* name, int sa, int sb) {
  Case c(name, 1, 1, 32);
  c.as[0] = uint8_t(sa); c.bs[0] = uint8_t(sb);
  set_code(c.a, 0, 0, c.padded_k, 1);
  set_code(c.b, 0, 0, c.padded_k, 1);
  return c;
}

Case energy_boundary(const char* name, bool above, int sa, int sb) {
  const int k = 262144 + int(above);
  Case c(name, 1, 1, k);
  std::fill(c.as.begin(), c.as.end(), uint8_t(sa));
  std::fill(c.bs.begin(), c.bs.end(), uint8_t(sb));
  for (int i = 0; i < 262144; ++i) {
    set_code(c.a, 0, i, c.padded_k, 6); // q = 8; energy = 64*K = 2^24.
    set_code(c.b, 0, i, c.padded_k, 6);
  }
  if (above) {
    set_code(c.a, 0, 262144, c.padded_k, 1);
    set_code(c.b, 0, 262144, c.padded_k, 1);
  }
  return c;
}

Case multitile_row_bases(const char* name, int m, int n, int k, bool unaligned) {
  Case c(name, m, n, k);
  c.input_byte_offset = unaligned ? 1 : 0;
  c.output_float_offset = unaligned ? 1 : 0;
  auto fill = [&](std::vector<uint8_t>& codes, std::vector<uint8_t>& scales,
                  int rows, bool is_a) {
    for (int row = 0; row < rows; ++row) {
      const int base = is_a ? 100 + row % 9 : 150 + (row * 5) % 11;
      const bool zero_row = row % 31 == 0;
      for (int block = 0; block < c.blocks; ++block) {
        const bool zero_block = zero_row || (block == 0 && row % (is_a ? 5 : 7) == 0);
        // Every nonzero row has a base-scale active block at block 1.
        // Inactive blocks deliberately have scales below that active minimum.
        scales[size_t(row) * c.blocks + block] = uint8_t(zero_block ? 1
            : base + (block == 1 ? 0 : (block + row) % 4));
        for (int kk = block * 32; kk < (block + 1) * 32 && kk < k; ++kk) {
          const unsigned code = zero_block ? 8u // Signed zero, normalize to +0.
              : unsigned(is_a ? (row * 11 + kk * 7) % 16 : (row * 3 + kk * 5 + 1) % 16);
          set_code(codes, row, kk, c.padded_k, code);
        }
      }
    }
  };
  fill(c.a, c.as, c.m, true);
  fill(c.b, c.bs, c.n, false);
  return c;
}

void check_context_reuse(cudaStream_t stream) {
  Case c("reused_context", 3, 5, 64);
  for (int row = 0; row < c.m; ++row) for (int k = 0; k < c.logical_k; ++k)
    set_code(c.a, row, k, c.padded_k, unsigned((row + k) % 16));
  for (int row = 0; row < c.n; ++row) for (int k = 0; k < c.logical_k; ++k)
    set_code(c.b, row, k, c.padded_k, unsigned((row * 3 + k) % 16));
  require_free_memory(required_bytes(c));
  Buffer<uint8_t> a(c.a.size()), as(c.as.size()), b(c.b.size()), bs(c.bs.size());
  Buffer<float> out(size_t(c.m) * c.n);
  void* context = nullptr;
  certified_checks::checked(mxfp4_certified_create(c.m, c.n, c.padded_k, &context), "reuse create");
  try {
    MXFP4CertifiedInfo info{};
    certified_checks::checked(mxfp4_certified_get_info(context, &info), "fresh info");
    if (info.metadata_valid || info.submitted_calls)
      throw std::runtime_error("New context reported stale certificate metadata");
    for (int iteration = 0; iteration < 3; ++iteration) {
      std::fill(c.as.begin(), c.as.end(), uint8_t(126 + iteration));
      std::fill(c.bs.begin(), c.bs.end(), uint8_t(127));
      if (iteration == 1) c.as[0] = 255;
      const auto expected = reference(c);
      CUDA_CHECK(cudaMemcpyAsync(a.data, c.a.data(), c.a.size(), cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(as.data, c.as.data(), c.as.size(), cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(b.data, c.b.data(), c.b.size(), cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(bs.data, c.bs.data(), c.bs.size(), cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemsetAsync(out.data, 0xff, expected.size() * sizeof(float), stream));
      certified_checks::pipeline_attempted = true;
      // The performance API must enqueue correctly without a host decision.
      certified_checks::checked(mxfp4_certified_gemm(context, a.data, as.data, b.data,
                                                    bs.data, out.data, stream, nullptr), "reuse gemm");
      std::vector<float> actual(expected.size());
      CUDA_CHECK(cudaMemcpyAsync(actual.data(), out.data, actual.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      certified_checks::any_completed = true;
      certified_checks::checked(mxfp4_certified_get_info(context, &info), "reuse info");
      if (!info.metadata_valid || info.submitted_calls != uint64_t(iteration + 1)
          || bool(info.accepted) != (iteration != 1))
        throw std::runtime_error("Workspace reused an old certificate or call counter");
      for (size_t i = 0; i < actual.size(); ++i)
        if (!equal(actual[i], expected[i]))
          throw std::runtime_error("Asynchronous reused context output mismatch");
      if (info.accepted) ++certified_checks::accepted;
      else ++certified_checks::rejected;
      std::cout << "{\"type\":\"context_reuse\",\"iteration\":" << iteration
                << ",\"accepted\":" << (info.accepted ? "true" : "false")
                << ",\"submitted_calls\":" << info.submitted_calls
                << ",\"values\":" << actual.size()
                << ",\"mismatches\":0,\"nondefault_stream\":true,\"passed\":true}\n";
    }
  } catch (...) {
    mxfp4_certified_destroy(context);
    throw;
  }
  certified_checks::checked(mxfp4_certified_destroy(context), "reuse destroy");
}

#if MXFP4_CERTIFIED_VARIANT == 8
struct C8Fixture {
  Case data;
  bool accepted, vector_eligible;
  uint32_t reject_flag = 0;
};

Case c8_max_delta(const char* name, int m, int n, int output_offset) {
  Case c(name, m, n, 160);
  c.output_float_offset = output_offset;
  std::fill(c.a.begin(), c.a.end(), uint8_t(0x88));
  std::fill(c.b.begin(), c.b.end(), uint8_t(0x88));
  for (int row = 0; row < c.m; ++row) {
    const int base = 100 + row % 9;
    const bool zero_row = row % 31 == 0;
    for (int block = 0; block < c.blocks; ++block) {
      const bool zero_block = zero_row || block == 0;
      const int delta = block == 4 ? 12 : (block == 1 ? 0 : block % 3);
      c.as[size_t(row) * c.blocks + block] = uint8_t(zero_block ? 1 : base + delta);
      if (!zero_block) {
        const unsigned magnitude = block == 4 ? 7u : (block == 1 ? 1u : 3u);
        set_code(c.a, row, block * 32 + (row * 3) % 32, c.padded_k,
                 magnitude | (row & 1 ? 8u : 0u));
      }
    }
  }
  for (int row = 0; row < c.n; ++row) {
    const int base = 150 + row % 11;
    const bool zero_row = row % 29 == 0;
    for (int block = 0; block < c.blocks; ++block) {
      const bool zero_block = zero_row || block == 0;
      c.bs[size_t(row) * c.blocks + block] = uint8_t(zero_block ? 2 : base);
      if (!zero_block) for (int kk = block * 32; kk < (block + 1) * 32; ++kk)
        set_code(c.b, row, kk, c.padded_k, unsigned((row * 3 + kk * 5 + 1) % 16));
    }
  }
  return c;
}

std::vector<C8Fixture> c8_loader_fixtures() {
  std::vector<C8Fixture> cases;
  cases.push_back({Case("c8_empty_k_128x128", 128, 128, 0), true, true});
  cases.push_back({multitile_row_bases("c8_aligned_128x128x32", 128, 128, 32, false), true, true});
  cases.push_back({multitile_row_bases("c8_aligned_row_bases_256x128x96", 256, 128, 96, false), true, true});
  cases.push_back({c8_max_delta("c8_aligned_delta12_128x256x160", 128, 256, 0), true, true});
  cases.push_back({c8_max_delta("c8_delta12_output_offset4", 128, 128, 1), true, true});
  for (int offset : {1, 2, 4}) {
    Case c = multitile_row_bases("c8_input_offset_c6", 128, 128, 96, false);
    c.name += std::to_string(offset);
    c.input_byte_offset = offset;
    cases.push_back({std::move(c), true, false});
  }
  Case rejected = multitile_row_bases("c8_zero_block_nan_scale_rejected", 128, 128, 96, false);
  rejected.as[0] = 255;
  cases.push_back({std::move(rejected), false, true, MXFP4_CERT_REJECT_NAN_SCALE});
  return cases;
}

int check_c8_loader_fixtures(cudaStream_t stream) {
  Totals totals;
  uint64_t peak_bytes = 0;
  size_t nonempty_c8_calls = 0, c6_compatibility_calls = 0, empty_calls = 0, rejected_calls = 0;
  const auto accepted_before = certified_checks::accepted;
  const auto rejected_before = certified_checks::rejected;
  const auto zero_before = certified_checks::canonical_zero_values;
  certified_checks::c8_fixture_active = true;
  try {
    for (const auto& fixture : c8_loader_fixtures()) {
      const auto& c = fixture.data;
      peak_bytes = std::max(peak_bytes, required_bytes(c));
      run_case(c, stream, totals);
      expect_certificate(fixture.accepted, fixture.reject_flag, c.name.c_str());
      const auto& info = certified_checks::last_info;
      if (c.name.find("delta12") != std::string::npos && info.max_span_a != 12)
        throw std::runtime_error(c.name + ": expected exact span 12");
      if (certified_checks::c8_a_alignment != unsigned(c.input_byte_offset & 7)
          || certified_checks::c8_b_alignment != unsigned(c.input_byte_offset & 7)
          || certified_checks::c8_output_alignment != unsigned((c.output_float_offset * 4) & 7))
        throw std::runtime_error(c.name + ": unexpected allocated pointer alignment");
      // These bounded fixtures all fit int32 addressing. Combined with the
      // separately host-tested dispatch predicate, record which accepted
      // nonempty cases cover the new loader and which exercise C6 fallback.
      const bool vector_eligible = c.m % 128 == 0 && c.n % 128 == 0
          && certified_checks::c8_a_alignment == 0 && certified_checks::c8_b_alignment == 0;
      if (vector_eligible != fixture.vector_eligible)
        throw std::runtime_error(c.name + ": incorrect C8 dispatch fixture");
      const char* path = !fixture.accepted ? "exact_tiled_rejection"
          : c.padded_k == 0 ? "empty_k"
          : vector_eligible ? "c8_halfrow_loader" : "c6_compatible_fused";
      if (!fixture.accepted) ++rejected_calls;
      else if (c.padded_k == 0) ++empty_calls;
      else if (vector_eligible) ++nonempty_c8_calls;
      else ++c6_compatibility_calls;
      std::cout << "{\"type\":\"c8_fixture_dispatch\",\"case\":" << json_string(c.name)
                << ",\"expected_execution_path\":" << json_string(path)
                << ",\"vector_dispatch_eligible\":" << (vector_eligible ? "true" : "false")
                << ",\"pointer_alignment_observed\":true,\"kernel_path_instrumented\":false"
                << ",\"accepted\":" << (info.accepted ? "true" : "false")
                << ",\"passed\":true}\n";
    }
    certified_checks::c8_fixture_active = false;
    if (totals.cases != 9 || totals.values != 180224 || nonempty_c8_calls != 4
        || c6_compatibility_calls != 3 || empty_calls != 1 || rejected_calls != 1)
      throw std::runtime_error("C8 dedicated fixture coverage changed unexpectedly");
    std::cout << "{\"type\":\"c8_fixture_summary\",\"passed\":true,\"gpu_executed\":true"
              << ",\"cases\":" << totals.cases << ",\"values\":" << totals.values
              << ",\"nonempty_c8_loader_cases\":" << nonempty_c8_calls
              << ",\"c6_compatibility_cases\":" << c6_compatibility_calls
              << ",\"empty_K_cases\":" << empty_calls << ",\"rejected_cases\":" << rejected_calls
              << ",\"accepted_calls\":" << certified_checks::accepted - accepted_before
              << ",\"fallback_calls\":" << certified_checks::rejected - rejected_before
              << ",\"canonical_positive_zero_values\":" << certified_checks::canonical_zero_values - zero_before
              << ",\"peak_explicit_allocation_bytes\":" << peak_bytes
              << ",\"mismatches\":0,\"performance_measured\":false}\n";
    return 0;
  } catch (const std::exception& error) {
    certified_checks::c8_fixture_active = false;
    std::cout << "{\"type\":\"c8_fixture_summary\",\"passed\":false,\"gpu_executed\":"
              << (totals.kernel_execution_completed ? "true"
                  : (totals.kernel_launch_attempted ? "null" : "false"))
              << ",\"completed_cases\":" << totals.cases
              << ",\"error\":" << json_string(error.what()) << "}\n";
    return 1;
  }
}
#endif

}  // namespace

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--help") {
      std::cout << "check_mxfp4_certified [--device N] [--reserve-mib N]\n"
                   "Full ordered-MX32 oracle, exact certificate boundaries and context reuse.\n"
                   "Correctness only; default reserve 128 MiB, explicit minimum 64 MiB.\n";
#if MXFP4_CERTIFIED_VARIANT >= 3
      std::cout << "Optional: --small-memory-correctness --reserve-mib N\n"
                   "Requires an explicit reserve of at least 48 MiB; caps explicit GPU allocations at 8 MiB.\n";
#endif
      return 0;
    }
    if (std::string(argv[i]) == "--benchmark") {
      std::cerr << "Use bench_mxfp4 for correctly scoped performance measurements\n";
      return 2;
    }
  }
  const int full_domain_status = ordered_mx32_checks_main(argc, argv);
  if (full_domain_status) {
    std::cout << "{\"type\":\"certified_summary\",\"passed\":false,\"gpu_executed\":"
              << (certified_checks::any_completed ? "true"
                  : (certified_checks::pipeline_attempted ? "null" : "false"))
              << ",\"full_domain_suite_passed\":false,\"error\":\"ordered_mx32_suite_failed\"}\n";
    return full_domain_status;
  }
  try {
    Stream stream;
    Totals totals;
    auto fixture = [&](const Case& c, bool accepted, uint32_t flag = 0) {
      run_case(c, stream.value, totals);
      expect_certificate(accepted, flag, c.name.c_str());
    };
    fixture(impulse("lattice_minimum_accepted", 65, 65), true);
    fixture(impulse("lattice_below_minimum_rejected", 64, 65), false,
            MXFP4_CERT_REJECT_LATTICE_RANGE);
    fixture(impulse("lattice_maximum_accepted", 179, 180), true);
    fixture(impulse("lattice_above_maximum_rejected", 180, 180), false,
            MXFP4_CERT_REJECT_LATTICE_RANGE);

    for (int span : {12, 13}) {
      Case c("active_scale_span_" + std::to_string(span), 1, 1, 64);
      c.as = {127, uint8_t(127 + span)};
      set_code(c.a, 0, 0, c.padded_k, 1);
      set_code(c.a, 0, 32, c.padded_k, 1);
      set_code(c.b, 0, 0, c.padded_k, 1);
      fixture(c, span == 12, span == 13 ? MXFP4_CERT_REJECT_SCALE_SPAN : 0);
    }

    Case max_half("normalized_half_positive_negative_49152", 2, 1, 64);
    max_half.as = {127, 139, 127, 139};
    for (int row = 0; row < 2; ++row) {
      set_code(max_half.a, row, 0, max_half.padded_k, 1);
      set_code(max_half.a, row, 32, max_half.padded_k, row == 0 ? 7 : 15);
    }
    set_code(max_half.b, 0, 32, max_half.padded_k, 1);
    fixture(max_half, true);

    Case saturated("energy_saturation_is_conservative", 1, 1, 64);
    saturated.as = {127, 152};
    set_code(saturated.a, 0, 0, saturated.padded_k, 1);
    set_code(saturated.a, 0, 32, saturated.padded_k, 1);
    set_code(saturated.b, 0, 0, saturated.padded_k, 1);
    fixture(saturated, false, MXFP4_CERT_REJECT_ENERGY_BOUND | MXFP4_CERT_REJECT_SCALE_SPAN);
    if (certified_checks::last_info.max_energy_a != MXFP4_CERTIFIED_ENERGY_CAP
        || !certified_checks::last_info.energy_saturated_a)
      throw std::runtime_error("Saturated integer energy metadata is incorrect");

    Case zero_block("zero_block_lower_scale_than_active_row", 1, 1, 64);
    zero_block.as = {1, 130}; zero_block.bs = {2, 128};
    set_code(zero_block.a, 0, 0, zero_block.padded_k, 8); // Negative zero.
    set_code(zero_block.a, 0, 32, zero_block.padded_k, 2);
    set_code(zero_block.b, 0, 32, zero_block.padded_k, 2);
    zero_block.input_byte_offset = 1;
    fixture(zero_block, true);
    zero_block.name = "zero_block_nan_scale_requires_fallback";
    zero_block.as[0] = 255;
    fixture(zero_block, false, MXFP4_CERT_REJECT_NAN_SCALE);
    fixture(Case("all_zero_certificate", 3, 5, 65), true);

    Case unaligned_output("output_pointer_four_byte_offset", 2, 8, 64);
    unaligned_output.output_float_offset = 1;
    for (int row = 0; row < unaligned_output.m; ++row)
      set_code(unaligned_output.a, row, 0, unaligned_output.padded_k, row ? 15 : 7);
    for (int row = 0; row < unaligned_output.n; ++row)
      set_code(unaligned_output.b, row, 0, unaligned_output.padded_k, unsigned(row + 1));
    fixture(unaligned_output, true);

    Case tall("row_grid_stride_beyond_65535", 65537, 3, 32);
    for (int row = 0; row < tall.m; ++row)
      set_code(tall.a, row, 0, tall.padded_k, row & 1 ? 9 : 1);
    for (int row = 0; row < tall.n; ++row)
      set_code(tall.b, row, 0, tall.padded_k, unsigned(row + 1));
    fixture(tall, true);

    // Cross both 64/128 CTA tile boundaries with different A/B row bases.
    // K=161 spans three BK64 stages; Kp=288 for K=257 also exercises a final
    // half-stage. The second case offsets packed input and FP32 output.
    fixture(multitile_row_bases("multitile_row_bases_129x131x161", 129, 131, 161, false), true);
    fixture(multitile_row_bases("multitile_row_bases_unaligned_131x129x257", 131, 129, 257, true), true);

    Case cancellation("fp16_exact_does_not_imply_safe_reordering", 1, 1, 96);
    cancellation.as = {140, 128, 140}; cancellation.bs = {140, 128, 140};
    for (int k : {0, 32, 64}) {
      set_code(cancellation.a, 0, k, cancellation.padded_k, k == 64 ? 9 : 1);
      set_code(cancellation.b, 0, k, cancellation.padded_k, 1);
    }
    if (reference(cancellation)[0] != 0)
      throw std::runtime_error("Independent cancellation fixture is malformed");
    fixture(cancellation, false, MXFP4_CERT_REJECT_ENERGY_BOUND);

    fixture(energy_boundary("energy_bound_equal_2pow48", false, 127, 127), true);
    if (certified_checks::last_info.max_energy_a != (uint64_t(1) << 24)
        || certified_checks::last_info.max_energy_b != (uint64_t(1) << 24))
      throw std::runtime_error("Energy boundary certificate is not exact");
    fixture(energy_boundary("energy_bound_one_larger_per_row", true, 127, 127), false,
            MXFP4_CERT_REJECT_ENERGY_BOUND);
    if (certified_checks::last_info.max_energy_a != (uint64_t(1) << 24) + 1
        || certified_checks::last_info.max_energy_b != (uint64_t(1) << 24) + 1)
      throw std::runtime_error("Above-boundary energies are not exact");
    fixture(energy_boundary("energy_boundary_at_lambda_minimum", false, 65, 65), true);
    fixture(energy_boundary("energy_boundary_at_lambda_maximum", false, 179, 180), true);
    check_context_reuse(stream.value);
    if (!certified_checks::accepted || !certified_checks::rejected)
      throw std::runtime_error("Both dispatch paths must execute");
    std::cout << "{\"type\":\"certified_summary\",\"passed\":true,\"gpu_executed\":true"
              << ",\"certified_variant\":" << MXFP4_CERTIFIED_VARIANT
              << ",\"full_domain_suite_passed\":true,\"extra_fixture_cases\":" << totals.cases
              << ",\"extra_fixture_values\":" << totals.values
              << ",\"context_reuse_calls\":3,\"accepted_calls\":" << certified_checks::accepted
              << ",\"fallback_calls\":" << certified_checks::rejected
              << ",\"zero_K_calls\":" << certified_checks::empty
              << ",\"canonical_positive_zero_values\":" << certified_checks::canonical_zero_values
              << ",\"mismatches\":0,\"performance_measured\":false}\n";
#if MXFP4_CERTIFIED_VARIANT == 8
    // Preserve the existing 69-call suite's summary and report the C8-only
    // coverage separately. A failure in either suite returns a failure status.
    return check_c8_loader_fixtures(stream.value);
#endif
    return 0;
  } catch (const std::exception& error) {
    std::cout << "{\"type\":\"certified_summary\",\"passed\":false,\"gpu_executed\":"
              << (certified_checks::any_completed ? "true"
                  : (certified_checks::pipeline_attempted ? "null" : "false"))
              << ",\"error\":" << json_string(error.what()) << "}\n";
    return 1;
  }
}
