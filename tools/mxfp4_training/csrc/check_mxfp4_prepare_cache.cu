// Standalone correctness-only check; include the ACTUAL candidate implementation.
// Compile this TU once with tiled.cu. Do not also compile certified.cu separately.
#include "mxfp4_gemm_certified.cu"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if MXFP4_CERTIFIED_VARIANT != 2 || MXFP4_CERTIFIED_PREPARE_RAW_CACHE != 1
#error "This checker requires certified variant 2 and the opt-in raw cache"
#endif
static_assert(sizeof(RowData) == 16 && offsetof(RowData, energy) == 0 &&
    offsetof(RowData, nan_scales) == 8 && offsetof(RowData, min_scale) == 12 &&
    offsetof(RowData, span) == 13 && offsetof(RowData, zero) == 14 &&
    offsetof(RowData, saturated) == 15, "Compare all 16 RowData bytes without padding");

namespace cache_check {
constexpr size_t kReserve = size_t(512) * 1024 * 1024;
constexpr size_t kMinimumFree = size_t(1024) * 1024 * 1024;
constexpr size_t kAllocationCap = size_t(8) * 1024 * 1024;
constexpr size_t kGuard = 16;
constexpr uint8_t kGuardByte = 0x6d;
size_t allocated = 0, peak_allocated = 0;
bool attempted = false, completed = false;
uint64_t cases_done = 0, direct_cases = 0, public_cases = 0;
uint64_t half_values = 0, row_values = 0, public_output_values = 0;
uint64_t zero_half_values = 0, nan_scale_rows = 0, saturated_rows = 0;

void require(bool good, const std::string& why) {
  if (!good) throw std::runtime_error(why);
}
void checked(int status, const char* operation) {
  if (status) throw std::runtime_error(std::string(operation) + ": " + mxfp4_certified_error_string(status));
}
void account(size_t bytes) {
  require(bytes <= kAllocationCap - allocated, "Explicit device allocation cap exceeded");
  allocated += bytes; peak_allocated = std::max(peak_allocated, allocated);
}
std::string quote(const std::string& text) {
  std::string result = "\"";
  for (char c : text) {
    if (c == '\\' || c == '"') { result += '\\'; result += c; }
    else if (c == '\n') result += "\\n";
    else if (static_cast<unsigned char>(c) < 32) result += '?';
    else result += c;
  }
  return result + '"';
}
struct Stream {
  cudaStream_t value = nullptr;
  Stream() { checked(int(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)), "create stream"); }
  ~Stream() { if (value) cudaStreamDestroy(value); }
};
struct Buffer {
  uint8_t* allocation = nullptr;
  size_t payload, offset, bytes;
  std::vector<uint8_t> uploaded;
  Buffer(size_t payload_bytes, size_t displacement = 0)
      : payload(payload_bytes), offset(displacement), bytes(payload_bytes + displacement + 2*kGuard) {
    account(bytes);
    const int status = int(cudaMalloc(reinterpret_cast<void**>(&allocation), bytes));
    if (status) { allocated -= bytes; checked(status, "cudaMalloc check buffer"); }
  }
  ~Buffer() { if (allocation) { cudaFree(allocation); allocated -= bytes; } }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  uint8_t* data() const { return allocation + kGuard + offset; }
  void upload(const std::vector<uint8_t>& contents, cudaStream_t stream) {
    require(contents.size() == payload, "Wrong upload size");
    uploaded.assign(bytes, kGuardByte);
    if (payload) std::memcpy(uploaded.data() + kGuard + offset, contents.data(), payload);
    checked(int(cudaMemcpyAsync(allocation, uploaded.data(), bytes, cudaMemcpyHostToDevice, stream)), "upload guarded buffer");
  }
  std::vector<uint8_t> read(cudaStream_t stream, bool unchanged = false) const {
    std::vector<uint8_t> actual(bytes);
    checked(int(cudaMemcpyAsync(actual.data(), allocation, bytes, cudaMemcpyDeviceToHost, stream)), "download guarded buffer");
    checked(int(cudaStreamSynchronize(stream)), "synchronize buffer"); completed = true;
    for (size_t i = 0; i < kGuard + offset; ++i) require(actual[i] == kGuardByte, "Leading guard changed");
    for (size_t i = kGuard + offset + payload; i < bytes; ++i) require(actual[i] == kGuardByte, "Trailing guard changed");
    if (unchanged) require(actual == uploaded, "Input bytes were modified");
    return std::vector<uint8_t>(actual.begin() + kGuard + offset, actual.begin() + kGuard + offset + payload);
  }
};
struct PublicContext {
  void* handle = nullptr;
  uint64_t bytes = 0;
  PublicContext(int m, int n, int k) {
    checked(mxfp4_certified_workspace_bytes(m, n, k, &bytes), "context workspace query");
    account(size_t(bytes));
    int status = mxfp4_certified_create(m, n, k, &handle);
    if (status) { allocated -= size_t(bytes); checked(status, "public context create"); }
    try {
      size_t free = 0, total = 0;
      checked(int(cudaMemGetInfo(&free, &total)), "memory after public context");
      require(free >= kReserve, "Post-context memory reserve violated");
    } catch (const std::exception& error) {
      const std::string original = error.what();
      const int cleanup = mxfp4_certified_destroy(handle);
      handle = nullptr; allocated -= size_t(bytes);
      if (cleanup) throw std::runtime_error(original + "; context cleanup: " + mxfp4_certified_error_string(cleanup));
      throw;
    }
  }
  ~PublicContext() { if (handle) { mxfp4_certified_destroy(handle); allocated -= size_t(bytes); } }
  Context* internal() const { return static_cast<Context*>(handle); }
};
struct Fixture {
  std::string name;
  int m = 3, n = 2, k;
  size_t a_offset = 0, b_offset = 0;
  std::vector<uint8_t> codes, scales;
  Fixture(std::string label, int extent) : name(std::move(label)), k(extent),
      codes(size_t(m+n) * extent / 2), scales(size_t(m+n) * extent / 32, 127) {
    for (int row = 0; row < m+n; ++row)
      for (int kk = 0; kk < k; ++kk) set(row, kk, unsigned((kk + row * 7) & 15));
  }
  unsigned code(int row, int kk) const {
    return (codes[size_t(row)*k/2 + kk/2] >> ((kk&1)*4)) & 15u;
  }
  void set(int row, int kk, unsigned code) {
    uint8_t& byte = codes[size_t(row)*k/2 + kk/2];
    const unsigned shift = unsigned(kk&1)*4;
    byte = uint8_t((byte & ~(15u << shift)) | ((code & 15u) << shift));
  }
  void block(int row, int block_id, unsigned code) {
    for (int kk = block_id*32; kk < block_id*32+32; ++kk) set(row, kk, code);
  }
  uint8_t& scale(int row, int block_id) { return scales[size_t(row)*k/32 + block_id]; }
};
struct Oracle {
  std::vector<uint16_t> halves;
  std::vector<RowData> rows;
};
// Independent scalar mathematical oracle. It never calls the device decoder,
// DP4A/PRMT helpers, normalized_half*_bits, or either preparation kernel.
Oracle oracle(const Fixture& f) {
  const int q_table[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
  Oracle out; out.halves.assign(size_t(f.m+f.n)*f.k, 0); out.rows.resize(f.m+f.n);
  for (int row = 0; row < f.m+f.n; ++row) {
    RowData data{};
    int minimum = 255, maximum = 0; bool active = false;
    for (int block = 0; block < f.k/32; ++block) {
      const int scale = f.scales[size_t(row)*f.k/32+block];
      if (scale == 255) ++data.nan_scales;
      bool nonzero = false;
      for (int j = 0; j < 32; ++j) nonzero |= q_table[f.code(row, block*32+j)] != 0;
      active |= nonzero;
      if (nonzero && scale != 255) { minimum = std::min(minimum, scale); maximum = std::max(maximum, scale); }
    }
    data.min_scale = uint8_t(minimum == 255 ? 127 : minimum);
    data.span = uint8_t(minimum == 255 ? 0 : maximum-minimum);
    data.zero = uint8_t(!active);
    const bool safe = data.nan_scales == 0 && data.span <= 12;
    for (int kk = 0; kk < f.k; ++kk) {
      const int q = q_table[f.code(row, kk)];
      const int scale = f.scales[size_t(row)*f.k/32+kk/32];
      if (!q || scale == 255) continue;
      const int delta = scale - int(data.min_scale);
      require(delta >= 0, "CPU minimum invariant failed");
      // Below 2^48+1 all integer sums are exact even with binary64 precision.
      // ldexp also safely covers the full finite E8M0 span through 254.
      const long double total = static_cast<long double>(data.energy)
          + std::ldexp(static_cast<long double>(q*q), 2*delta);
      data.energy = total >= static_cast<long double>(MXFP4_CERTIFIED_ENERGY_CAP)
          ? MXFP4_CERTIFIED_ENERGY_CAP : static_cast<unsigned long long>(total);
      if (safe) {
        const __half value = __float2half_rn(std::ldexp(float(q), delta));
        uint16_t bits; std::memcpy(&bits, &value, sizeof(bits));
        require((bits & 0x7c00u) != 0x7c00u, "CPU normalized half must be finite");
        out.halves[size_t(row)*f.k+kk] = bits;
      }
    }
    data.saturated = uint8_t(data.energy == MXFP4_CERTIFIED_ENERGY_CAP);
    out.rows[row] = data;
  }
  return out;
}
template<class T> std::vector<uint8_t> bytes_of(const std::vector<T>& input) {
  std::vector<uint8_t> result(input.size()*sizeof(T));
  if (!result.empty()) std::memcpy(result.data(), input.data(), result.size());
  return result;
}
std::vector<uint8_t> invert(std::vector<uint8_t> input) {
  for (auto& byte : input) byte = uint8_t(~byte);
  return input;
}
std::vector<uint8_t> slice(const std::vector<uint8_t>& x, size_t first, size_t length) {
  return std::vector<uint8_t>(x.begin()+first, x.begin()+first+length);
}
void compare(const std::vector<uint8_t>& actual, const std::vector<uint8_t>& expected,
             const std::string& name) {
  require(actual.size() == expected.size(), "Comparison size differs: " + name);
  for (size_t i=0; i<actual.size(); ++i)
    if (actual[i] != expected[i]) throw std::runtime_error(name+" byte mismatch at "+std::to_string(i));
}

void run_case(const Fixture& f, cudaStream_t stream, bool public_dispatch = false) {
  const Oracle expected = oracle(f);
  const auto half_bytes = bytes_of(expected.halves), row_bytes = bytes_of(expected.rows);
  const auto half_poison = invert(half_bytes), row_poison = invert(row_bytes);
  Buffer a(size_t(f.m)*f.k/2, f.a_offset), b(size_t(f.n)*f.k/2, f.b_offset);
  Buffer as(size_t(f.m)*f.k/32, 1), bs(size_t(f.n)*f.k/32, 3);
  Buffer old_halves(half_bytes.size()), old_rows(row_bytes.size());
  a.upload(slice(f.codes, 0, a.payload), stream); b.upload(slice(f.codes, a.payload, b.payload), stream);
  as.upload(slice(f.scales, 0, as.payload), stream); bs.upload(slice(f.scales, as.payload, bs.payload), stream);
  old_halves.upload(half_poison, stream); old_rows.upload(row_poison, stream);
  require((reinterpret_cast<uintptr_t>(a.data()) & 15u) == f.a_offset &&
          (reinterpret_cast<uintptr_t>(b.data()) & 15u) == f.b_offset, "Actual input alignment differs");
  const uint8_t *ap = f.k ? a.data() : nullptr, *bp = f.k ? b.data() : nullptr;
  const uint8_t *asp = f.k ? as.data() : nullptr, *bsp = f.k ? bs.data() : nullptr;
  attempted = true;
  prepare_rows<false><<<unsigned(f.m+f.n), kPrepareThreads, 0, stream>>>(
      ap, asp, bp, bsp, reinterpret_cast<uint16_t*>(old_halves.data()),
      reinterpret_cast<RowData*>(old_rows.data()), nullptr, f.m, f.n, f.k);
  checked(int(cudaGetLastError()), "actual original preparation launch");
  auto old_h = old_halves.read(stream), old_r = old_rows.read(stream);
  compare(old_h, half_bytes, f.name+": original normalized vs CPU");
  compare(old_r, row_bytes, f.name+": original RowData vs CPU");
  bool eligible = use_c2_prepare_raw_cache(f.k);
  if (!public_dispatch) {
    require(eligible, "Direct cache invocation outside its public dispatch domain");
    Buffer new_halves(half_bytes.size()), new_rows(row_bytes.size());
    new_halves.upload(half_poison, stream); new_rows.upload(row_poison, stream);
    prepare_rows_raw_cached_c2<<<unsigned(f.m+f.n), kPrepareThreads, size_t(20)*(f.k/32), stream>>>(
        ap, asp, bp, bsp, reinterpret_cast<uint16_t*>(new_halves.data()),
        reinterpret_cast<RowData*>(new_rows.data()), f.m, f.k);
    checked(int(cudaGetLastError()), "actual cached preparation launch");
    const auto new_h = new_halves.read(stream), new_r = new_rows.read(stream);
    compare(new_h, half_bytes, f.name+": cached normalized vs CPU");
    compare(new_r, row_bytes, f.name+": cached RowData vs CPU");
    compare(new_h, old_h, f.name+": new-vs-original half bits");
    compare(new_r, old_r, f.name+": new-vs-original all RowData bytes");
    ++direct_cases;
  } else {
    Buffer output(size_t(f.m)*f.n*sizeof(float));
    std::vector<uint8_t> output_poison(output.payload, 0x7f); output.upload(output_poison, stream);
    PublicContext context(f.m, f.n, f.k);
    if (!half_poison.empty()) checked(int(cudaMemcpyAsync(context.internal()->normalized, half_poison.data(),
        half_poison.size(), cudaMemcpyHostToDevice, stream)), "poison public normalized workspace");
    checked(int(cudaMemcpyAsync(context.internal()->rows, row_poison.data(), row_poison.size(),
        cudaMemcpyHostToDevice, stream)), "poison public row workspace");
    int used = -1;
    checked(mxfp4_certified_gemm(context.handle, ap, asp, bp, bsp, reinterpret_cast<float*>(output.data()), stream, &used),
            "actual public certified submission");
    MXFP4CertifiedInfo info{};
    checked(mxfp4_certified_get_info(context.handle, &info), "public certificate info");
    require(info.abi_version == 1 && info.metadata_valid == 1 && info.submitted_calls == 1 &&
            info.accepted == 1 && info.reject_flags == 0 && used == 1, "Unit-input public certificate failed");
    std::vector<uint8_t> new_h(half_bytes.size()), new_r(row_bytes.size());
    if (!new_h.empty()) checked(int(cudaMemcpyAsync(new_h.data(), context.internal()->normalized, new_h.size(),
        cudaMemcpyDeviceToHost, stream)), "download public normalized workspace");
    checked(int(cudaMemcpyAsync(new_r.data(), context.internal()->rows, new_r.size(),
        cudaMemcpyDeviceToHost, stream)), "download public row workspace");
    checked(int(cudaStreamSynchronize(stream)), "public workspace synchronization"); completed = true;
    compare(new_h, half_bytes, f.name+": public normalized vs CPU");
    compare(new_r, row_bytes, f.name+": public RowData vs CPU");
    compare(new_h, old_h, f.name+": public-vs-original half bits");
    compare(new_r, old_r, f.name+": public-vs-original all RowData bytes");
    const auto actual_output = output.read(stream);
    const float value = float(f.k)*0.25f;
    std::vector<float> correct(size_t(f.m)*f.n, value);
    compare(actual_output, bytes_of(correct), f.name+": public FP32 exact closed-form unit dot");
    public_output_values += correct.size(); ++public_cases;
  }
  a.read(stream, true); b.read(stream, true); as.read(stream, true); bs.read(stream, true);
  for (uint16_t value : expected.halves) if (!value) ++zero_half_values;
  for (const RowData& row : expected.rows) {
    nan_scale_rows += row.nan_scales != 0; saturated_rows += row.saturated != 0;
  }
  half_values += expected.halves.size(); row_values += expected.rows.size(); ++cases_done;
  std::cout << "{\"type\":\"prepare_cache_correctness\",\"name\":" << quote(f.name)
      << ",\"passed\":true,\"gpu_executed\":true,\"M\":" << f.m << ",\"N\":" << f.n << ",\"K\":" << f.k
      << ",\"normalized_half_values\":" << expected.halves.size() << ",\"rowdata_rows\":" << expected.rows.size()
      << ",\"rowdata_bytes_per_row\":16,\"normalized_half_bit_mismatches\":0,\"rowdata_byte_mismatches\":0"
      << ",\"independent_cpu_oracle_mismatches\":0,\"input_bytes_unchanged\":true,\"guard_bytes_unchanged\":true"
      << ",\"a_base_alignment_mod16\":" << f.a_offset << ",\"b_base_alignment_mod16\":" << f.b_offset
      << ",\"actual_original_kernel_compared\":true,\"actual_cache_kernel_directly_invoked\":" << (public_dispatch ? "false" : "true")
      << ",\"public_dispatch_executed\":" << (public_dispatch ? "true" : "false")
      << ",\"shared_public_dispatch_helper_eligible\":" << (eligible ? "true" : "false")
      << ",\"kernel_path_instrumented\":false,\"poison\":\"bitwise_inverse_of_every_CPU_oracle_output_byte\"}\n";
}

std::vector<Fixture> fixtures() {
  std::vector<Fixture> out;
  for (int scale = 0; scale < 256; ++scale) {
    Fixture f("all_codes_uniform_scale_"+std::to_string(scale), 32);
    std::fill(f.scales.begin(), f.scales.end(), uint8_t(scale)); out.push_back(std::move(f));
  }
  const int spans[] = {0,1,11,12,13,23,24,25,127,254};
  for (int span : spans) {
    Fixture f("row_span_"+std::to_string(span), 64);
    for (int r=0; r<f.m+f.n; ++r) { f.scale(r,0)=0; f.scale(r,1)=uint8_t(span); }
    out.push_back(std::move(f));
  }
  const int extents[] = {32,64,128,4096,4128,8160,8192};
  const int offsets[][2] = {{0,0},{1,2},{2,4},{4,8},{8,1},{0,1},{1,0}};
  for (int k : extents) for (const auto& offset : offsets) {
    Fixture f("raw_alignment_k"+std::to_string(k)+"_a"+std::to_string(offset[0])+"_b"+std::to_string(offset[1]), k);
    f.a_offset=size_t(offset[0]); f.b_offset=size_t(offset[1]);
    uint32_t state=0x9e3779b9u+unsigned(k)+unsigned(offset[0]*31+offset[1]);
    for (auto& byte : f.codes) { state^=state<<13; state^=state>>17; state^=state<<5; byte=uint8_t(state); }
    for (int row=0;row<f.m+f.n;++row) for(int block=0;block<k/32;++block)
      f.scale(row,block)=uint8_t(120+row+(block*7)%9);
    out.push_back(std::move(f));
  }
  for (int nan=0;nan<2;++nan) {
    Fixture f(nan ? "signed_zero_with_nan_scale" : "signed_zero_finite_scales",128);
    for(int row=0;row<f.m+f.n;++row) for(int block=0;block<4;++block) {
      f.block(row,block,(row+block)&1 ? 8u : 0u);
      const int scales[]={0,126,127,254}; f.scale(row,block)=uint8_t(scales[block]);
      if(nan && block==3) f.scale(row,block)=255;
    }
    out.push_back(std::move(f));
  }
  {
    Fixture f("active_mixed_nan_scale",128);
    for(int row=0;row<f.m+f.n;++row) f.scale(row,3)=255;
    out.push_back(std::move(f));
  }
  {
    Fixture f("zero_lower_scales_and_delta12_max_half",128);
    const int bases[]={12,13,127,200,242};
    for(int row=0;row<f.m+f.n;++row) {
      f.block(row,0,8); f.scale(row,0)=0;
      f.scale(row,1)=uint8_t(bases[row]); f.scale(row,2)=uint8_t(bases[row]+12);
      f.block(row,3,0); f.scale(row,3)=254;
    }
    out.push_back(std::move(f));
  }
  {
    Fixture f("nan_only_active_with_finite_zero_blocks",64);
    for(int row=0;row<f.m+f.n;++row) {
      f.block(row,0,8); f.scale(row,0)=0; f.block(row,1,1); f.scale(row,1)=255;
    }
    out.push_back(std::move(f));
  }
  {
    Fixture f("different_row_normalization_bases",128);
    const int bases[]={0,1,127,242,254};
    for(int row=0;row<f.m+f.n;++row) for(int block=0;block<4;++block) f.scale(row,block)=uint8_t(bases[row]);
    out.push_back(std::move(f));
  }
  {
    Fixture f("sparse_active_blocks_with_zero_scale_extremes",8192);
    for(int row=0;row<f.m+f.n;++row) for(int block=0;block<f.k/32;++block) {
      f.block(row,block,block&1 ? 8u : 0u); f.scale(row,block)=uint8_t(block&1 ? 0 : 254);
    }
    for(int row=0;row<f.m+f.n;++row) {
      f.block(row,0,1); f.scale(row,0)=127; f.block(row,f.k/32-1,7); f.scale(row,f.k/32-1)=139;
    }
    out.push_back(std::move(f));
  }
  {
    Fixture f("all_256_packed_bytes_long_row",8192);
    for(size_t i=0;i<f.codes.size();++i) f.codes[i]=uint8_t(i);
    out.push_back(std::move(f));
  }
  return out;
}
} // namespace cache_check

int main(int argc,char** argv) {
  using namespace cache_check;
  int device=0;
  try {
    for(int i=1;i<argc;++i) {
      const std::string arg=argv[i];
      if(arg=="--device" && i+1<argc) { device=std::stoi(argv[++i]); require(device>=0,"Device must be nonnegative"); }
      else if(arg=="--correctness-only") {}
      else if(arg=="--help") { std::cout<<"check_mxfp4_prepare_cache [--device 0] [--correctness-only]\n"; return 0; }
      else throw std::runtime_error("Unknown/incomplete option: "+arg);
    }
    const int guard_k[]={0,32,64,4096,8192,8224,16384};
    const bool guard_expected[]={false,true,true,true,true,false,false};
    for(size_t i=0;i<sizeof(guard_k)/sizeof(guard_k[0]);++i)
      require(use_c2_prepare_raw_cache(guard_k[i])==guard_expected[i],"Actual shared dispatch helper failed");
    checked(int(cudaSetDevice(device)),"set device");
    cudaDeviceProp properties{}; checked(int(cudaGetDeviceProperties(&properties,device)),"device properties");
    require(properties.major==7 && properties.minor==0 && std::string(properties.name).find("V100")!=std::string::npos,
            "Require actual V100 SM70");
    checked(int(cudaFree(nullptr)),"initialize CUDA context");
    size_t free=0,total=0; checked(int(cudaMemGetInfo(&free,&total)),"initial memory");
    if(free<kMinimumFree || free<kReserve+kAllocationCap) {
      std::cout<<"{\"type\":\"prepare_cache_summary\",\"status\":\"blocked\",\"passed\":false,\"gpu_executed\":false,\"reason\":\"guarded_memory_unavailable\"}\n";
      return 77;
    }
    std::cout<<"{\"type\":\"prepare_cache_environment\",\"device\":"<<device<<",\"gpu_name\":"<<quote(properties.name)
        <<",\"actual_V100\":true,\"certified_variant\":2,\"certified_prepare_raw_cache\":true"
        <<",\"free_bytes_after_context\":"<<free<<",\"total_bytes\":"<<total<<",\"minimum_free_bytes\":"<<kMinimumFree
        <<",\"reserve_bytes\":"<<kReserve<<",\"explicit_allocation_cap_bytes\":"<<kAllocationCap
        <<",\"shared_dispatch_helper_checks\":7,\"performance_timing_executed\":false}\n";
    Stream stream;
    auto inputs=fixtures(); require(inputs.size()==323,"Unexpected fixed fixture count");
    for(const auto& f:inputs) run_case(f,stream.value);
    const int public_k[]={0,32,8192,8224};
    for(int k:public_k) {
      Fixture f("public_dispatch_k"+std::to_string(k),k);
      std::fill(f.codes.begin(),f.codes.end(),uint8_t(0x11));
      std::fill(f.scales.begin(),f.scales.end(),uint8_t(127));
      if(k==8224) { f.a_offset=1; f.b_offset=8; }
      run_case(f,stream.value,true);
    }
    checked(int(cudaStreamSynchronize(stream.value)),"final synchronize");
    require(allocated==0,"Explicit device buffer leak");
    std::cout<<"{\"type\":\"prepare_cache_summary\",\"status\":\"passed\",\"passed\":true,\"gpu_executed\":true"
        <<",\"all_submitted_work_completed\":true,\"actual_V100\":true,\"certified_variant\":2,\"certified_prepare_raw_cache\":true"
        <<",\"cases\":"<<cases_done<<",\"direct_cached_vs_original_cases\":"<<direct_cases<<",\"public_dispatch_cases\":"<<public_cases
        <<",\"public_fallback_preparation_cases\":2,\"public_cached_preparation_eligible_cases\":2"
        <<",\"normalized_half_values\":"<<half_values<<",\"rowdata_rows\":"<<row_values<<",\"public_FP32_output_values\":"<<public_output_values
        <<",\"canonical_positive_zero_half_values\":"<<zero_half_values<<",\"rows_with_nan_scales\":"<<nan_scale_rows
        <<",\"energy_saturated_rows\":"<<saturated_rows<<",\"normalized_half_bit_mismatches\":0,\"rowdata_byte_mismatches\":0"
        <<",\"independent_cpu_oracle_mismatches\":0,\"public_output_bit_mismatches\":0,\"guard_bytes_unchanged\":true,\"input_bytes_unchanged\":true"
        <<",\"peak_explicit_allocation_bytes\":"<<peak_allocated<<",\"kernel_path_instrumented\":false"
        <<",\"performance_timing_executed\":false,\"cpu_half_conversion\":\"ldexp(integer_q,delta)_then_host_float2half_rn\"}\n";
    return 0;
  } catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    std::cout<<"{\"type\":\"prepare_cache_summary\",\"status\":\"failed\",\"passed\":false,\"gpu_work_attempted\":"<<(attempted?"true":"false")
        <<",\"gpu_executed\":"<<(completed?"true":(attempted?"null":"false"))<<",\"completed_cases\":"<<cases_done
        <<",\"error\":"<<quote(error.what())<<"}\n";
    return 1;
  }
}
