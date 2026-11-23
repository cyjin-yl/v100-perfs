// Exact, data-certified FP16-input / FP32-output Tensor Core acceleration.
// A failed certificate always executes the exact tiled MX32 fallback.
#include "mxfp4_gemm_certified.h"
#include "mxfp4_gemm_tiled.h"
#include "mxfp4_gemm_fused.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <thread>

#ifndef MXFP4_CERTIFIED_VARIANT
#define MXFP4_CERTIFIED_VARIANT 0
#endif
#if MXFP4_CERTIFIED_VARIANT < 0 || MXFP4_CERTIFIED_VARIANT > 10
#error "MXFP4_CERTIFIED_VARIANT must be 0..2 (cuBLAS) or 3..10 (fused normalized HMMA)"
#endif
#ifndef MXFP4_CERTIFIED_PREPARE_RAW_CACHE
#define MXFP4_CERTIFIED_PREPARE_RAW_CACHE 0
#endif
#if MXFP4_CERTIFIED_PREPARE_RAW_CACHE != 0 && MXFP4_CERTIFIED_PREPARE_RAW_CACHE != 1
#error "MXFP4_CERTIFIED_PREPARE_RAW_CACHE must be 0 or 1"
#endif
#if MXFP4_CERTIFIED_PREPARE_RAW_CACHE && MXFP4_CERTIFIED_VARIANT != 2
#error "MXFP4_CERTIFIED_PREPARE_RAW_CACHE requires certified variant 2"
#endif

namespace {

constexpr unsigned long long kEnergyBound = MXFP4_CERTIFIED_ENERGY_BOUND;
constexpr unsigned long long kEnergyCap = MXFP4_CERTIFIED_ENERGY_CAP;
constexpr size_t kBlasWorkspace = 4u * 1024u * 1024u;
constexpr int kPrepareThreads = 128;
constexpr int kReduceThreads = 256;
constexpr int kCachedEnergyBlocks = 1024;
constexpr bool kFused = MXFP4_CERTIFIED_VARIANT >= 3;

#if MXFP4_CERTIFIED_PREPARE_RAW_CACHE
constexpr int kC2PrepareRawCacheMaxK = 8192;
constexpr bool use_c2_prepare_raw_cache(int k) {
  return k > 0 && k <= kC2PrepareRawCacheMaxK;
}
#endif

struct RowData {
  unsigned long long energy;
  unsigned nan_scales;
  uint8_t min_scale, span, zero, saturated;
};
static_assert(sizeof(RowData) == 16, "workspace accounting requires stable RowData");

struct DeviceMetadata {
  unsigned long long max_energy_a, max_energy_b;
  int min_scale_a, max_scale_a, min_scale_b, max_scale_b;
  unsigned max_span_a, max_span_b, nan_scales_a, nan_scales_b;
  unsigned zero_rows_a, zero_rows_b, energy_saturated_a, energy_saturated_b;
  unsigned reject_flags;
  int accepted;
};

struct Context {
  const std::thread::id owner_thread = std::this_thread::get_id();
  int device = -1;
  int m = 0, n = 0, k = 0;
  uint64_t workspace_bytes = 0;
  uint16_t* normalized = nullptr;
  uint8_t* row_bases = nullptr;
  RowData* rows = nullptr;
  DeviceMetadata* metadata = nullptr;
  DeviceMetadata* host_metadata = nullptr;
  void* blas_workspace = nullptr;
  cublasHandle_t blas = nullptr;
  cudaStream_t last_stream = nullptr;
  bool has_stream_work = false;
  bool metadata_valid = false;
  uint64_t submitted_calls = 0;
  std::atomic_flag busy = ATOMIC_FLAG_INIT;
};

struct Lock {
  Context* context;
  bool acquired;
  explicit Lock(Context* c) : context(c), acquired(!c->busy.test_and_set(std::memory_order_acquire)) {}
  ~Lock() { if (acquired) context->busy.clear(std::memory_order_release); }
};

__device__ __forceinline__ unsigned long long sat_add(unsigned long long a,
                                                      unsigned long long b) {
  return a >= kEnergyCap || b >= kEnergyCap || a > kEnergyCap - b ? kEnergyCap : a + b;
}

__device__ __forceinline__ unsigned count_add(unsigned a, unsigned b) {
  return a > 0xffffffffu - b ? 0xffffffffu : a + b;
}

__device__ __forceinline__ int q_from_code(unsigned code) {
  const unsigned magnitude = code & 7u;
  const int q = magnitude < 2u ? int(magnitude)
      : int(2u + (magnitude & 1u)) << (int(magnitude / 2u) - 1);
  return (code & 8u) ? -q : q;
}

__device__ __forceinline__ uint16_t normalized_half_bits(unsigned code, int delta) {
  const unsigned mag = code & 7u;
  if (mag == 0) return 0; // Never shift a zero block's possibly negative delta.
  const unsigned half_mag = mag >= 2 ? (mag + 30u) << 9 : 0x3c00u;
  return uint16_t(half_mag + (unsigned(delta) << 10) + ((code & 8u) << 12));
}

__device__ __forceinline__ bool block_has_nonzero_q(const uint8_t* p) {
  if ((reinterpret_cast<uintptr_t>(p) & 15u) == 0) {
    const uint4 words = *reinterpret_cast<const uint4*>(p);
    return ((words.x | words.y | words.z | words.w) & 0x77777777u) != 0;
  }
  unsigned bits = 0;
  #pragma unroll
  for (int i = 0; i < 16; ++i) bits |= p[i];
  return (bits & 0x77u) != 0;
}

__device__ __forceinline__ unsigned packed_word_q_energy(unsigned packed) {
  // Eight signed E2M1 codes: only magnitudes matter for the integer energy.
  const unsigned lo = __byte_perm(0x03020100u, 0x0c080604u, packed & 0x7777u);
  const unsigned hi = __byte_perm(0x03020100u, 0x0c080604u, (packed >> 16) & 0x7777u);
  return __dp4a(lo, lo, __dp4a(hi, hi, 0u));
}

__device__ __forceinline__ unsigned unaligned_packed_word(const uint8_t* p) {
  return unsigned(p[0]) | (unsigned(p[1]) << 8)
      | (unsigned(p[2]) << 16) | (unsigned(p[3]) << 24);
}

__device__ __forceinline__ unsigned block_q_energy(const uint8_t* p) {
  uint4 words;
  if ((reinterpret_cast<uintptr_t>(p) & 15u) == 0) {
    words = *reinterpret_cast<const uint4*>(p);
  } else {
    words = make_uint4(unaligned_packed_word(p), unaligned_packed_word(p + 4),
                       unaligned_packed_word(p + 8), unaligned_packed_word(p + 12));
  }
  // Four independent two-DP4A chains instead of a serial eight-DP4A chain.
  // The complete MX32 energy is at most 32*12^2=4608, fitting 16 bits.
  return (packed_word_q_energy(words.x) + packed_word_q_energy(words.y))
      + (packed_word_q_energy(words.z) + packed_word_q_energy(words.w));
}

__device__ __forceinline__ unsigned q_half_pair_bits(unsigned byte) {
  const unsigned selector = ((byte & 7u) << 4) | ((byte & 0x70u) << 8);
  const unsigned magnitude = __byte_perm(0x42403c00u, 0x4a484644u, selector);
  return magnitude | ((byte & 8u) << 12) | ((byte & 0x80u) << 24);
}

__device__ __forceinline__ uint2 normalized_half4_bits(unsigned packed, int delta) {
  // Called only after a nonzero group established delta>=0 and the row's
  // span<=12 established delta<=12. At most 0xca00+0x3000=0xfa00 in a half,
  // so adding packed exponent offsets cannot carry out of the low half.
  const unsigned shift = unsigned(delta) << 10;
  const unsigned both = shift | (shift << 16);
  unsigned lo = q_half_pair_bits(packed & 255u) + both;
  unsigned hi = q_half_pair_bits(packed >> 8) + both;
  // Both signed zero codes become canonical +0 BEFORE any normalized value
  // is consumed. They are never treated as scale-dependent half magnitudes.
  const unsigned lo_mask = ((packed & 0x0007u) ? 0x0000ffffu : 0u)
      | ((packed & 0x0070u) ? 0xffff0000u : 0u);
  const unsigned hi_mask = ((packed & 0x0700u) ? 0x0000ffffu : 0u)
      | ((packed & 0x7000u) ? 0xffff0000u : 0u);
  return make_uint2(lo & lo_mask, hi & hi_mask);
}

// One CTA per matrix row. Minima ignore wholly q-zero blocks, but ALL scale
// bytes are checked for 255 before that exclusion. Normalization and integer
// row energy then use these exact minima. cuBLAS variants emit finite safe
// zeros on invalid rows because cuBLAS precedes the conditional fallback.
// Fused variants emit only metadata and skip rejected GEMM input reads.
// Their short-K specialization caches one (scale, raw energy) word per MX32,
// avoiding a second global packed read and eight separate energy shifts.
template<bool CacheBlockEnergy>
__global__ __launch_bounds__(kPrepareThreads) void prepare_rows(
    const uint8_t* a, const uint8_t* as, const uint8_t* b, const uint8_t* bs,
    uint16_t* normalized, RowData* row_data, uint8_t* row_bases, int m, int n, int k) {
  static_assert(!CacheBlockEnergy || kFused, "energy cache is metadata-only");
  // Only the cached instantiation requests dynamic shared memory at launch.
  // Host dispatch bounds this allocation by kCachedEnergyBlocks*4 bytes.
  extern __shared__ unsigned block_energy_scale[];
  const int row = int(blockIdx.x);
  const bool is_a = row < m;
  const int local_row = is_a ? row : row - m;
  const int tid = int(threadIdx.x), lane = tid & 31, warp = tid >> 5;
  const int blocks = k / 32;
  const uint8_t* codes = is_a ? a : b;
  const uint8_t* scales = is_a ? as : bs;
  if (k) {
    codes += size_t(local_row) * (k / 2);
    scales += size_t(local_row) * blocks;
  }
  int minimum = 255, maximum = 0;
  unsigned nan_count = 0, active = 0;
  for (int block = tid; block < blocks; block += kPrepareThreads) {
    const unsigned scale = scales[block];
    if (scale == 255) ++nan_count;
    bool nonzero;
    if constexpr (CacheBlockEnergy) {
      const unsigned q_squared = block_q_energy(codes + size_t(block) * 16);
      block_energy_scale[block] = (scale << 16) | q_squared;
      nonzero = q_squared != 0;
    } else {
      nonzero = block_has_nonzero_q(codes + size_t(block) * 16);
    }
    active |= unsigned(nonzero);
    if (nonzero && scale != 255) {
      minimum = min(minimum, int(scale));
      maximum = max(maximum, int(scale));
    }
  }
  #pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    minimum = min(minimum, __shfl_down_sync(0xffffffffu, minimum, offset));
    maximum = max(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
    nan_count += __shfl_down_sync(0xffffffffu, nan_count, offset);
    active |= __shfl_down_sync(0xffffffffu, active, offset);
  }
  __shared__ int warp_min[4], warp_max[4];
  __shared__ unsigned warp_nan[4], warp_active[4];
  __shared__ unsigned long long warp_energy[4];
  __shared__ int row_min, row_span;
  __shared__ unsigned row_nan, row_active;
  if (lane == 0) {
    warp_min[warp] = minimum; warp_max[warp] = maximum;
    warp_nan[warp] = nan_count; warp_active[warp] = active;
  }
  __syncthreads();
  if (warp == 0) {
    minimum = lane < 4 ? warp_min[lane] : 255;
    maximum = lane < 4 ? warp_max[lane] : 0;
    nan_count = lane < 4 ? warp_nan[lane] : 0;
    active = lane < 4 ? warp_active[lane] : 0;
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
      minimum = min(minimum, __shfl_down_sync(0xffffffffu, minimum, offset));
      maximum = max(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
      nan_count += __shfl_down_sync(0xffffffffu, nan_count, offset);
      active |= __shfl_down_sync(0xffffffffu, active, offset);
    }
    if (lane == 0) {
      row_min = minimum == 255 ? 127 : minimum;
      row_span = minimum == 255 ? 0 : maximum - minimum;
      row_nan = nan_count;
      row_active = active;
    }
  }
  __syncthreads();
  const bool safe_half = row_nan == 0 && row_span <= 12;
  unsigned long long energy = 0;
  if constexpr (CacheBlockEnergy) {
    for (int block = tid; block < blocks; block += kPrepareThreads) {
      const unsigned cached = block_energy_scale[block];
      const unsigned q_squared = cached & 0xffffu;
      const unsigned scale = cached >> 16;
      // Zero blocks can have scales below row_min. Exclude them BEFORE
      // forming a shift; scale 255 is still counted/rejected independently.
      if (q_squared != 0 && scale != 255) {
        const int delta = int(scale) - row_min;
        // q_squared<=4608, so even a shift by 48 stays below 2^61.
        // Saturating sums of these nonnegative terms equal groupwise sums.
        const unsigned long long term = delta > 24 ? kEnergyCap
            : (static_cast<unsigned long long>(q_squared) << (2 * delta));
        energy = sat_add(energy, term);
      }
    }
  } else {
#if MXFP4_CERTIFIED_VARIANT == 0
  // Coalesced half writes; neighboring lanes decode the two nibbles of one
  // byte. All source nonzero q values remain exact, including delta=12/q=12.
  for (int64_t kk = tid; kk < k; kk += kPrepareThreads) {
    const unsigned packed = codes[kk / 2];
    const unsigned code = (packed >> ((kk & 1) * 4)) & 15u;
    const int q = q_from_code(code);
    const unsigned scale = scales[kk / 32];
    uint16_t output_bits = 0;
    if (q != 0 && scale != 255) {
      const int delta = int(scale) - row_min; // Nonzero finite block entered min.
      // Above 24, even q^2=1 exceeds ENERGY_BOUND. Clamp BEFORE shifting;
      // shifts through 48 and q^2<=144 remain within uint64_t.
      const unsigned long long term = delta > 24 ? kEnergyCap
          : (static_cast<unsigned long long>(q * q) << (2 * delta));
      energy = sat_add(energy, term);
      if (safe_half) output_bits = normalized_half_bits(code, delta);
    }
    normalized[size_t(row) * k + kk] = output_bits;
  }
#else
  // One thread handles four consecutive nibbles. All four share an MX32
  // scale, so one DP4A/energy shift replaces four scalar square/shift paths.
  // k/4<=536870904; adding 128 at loop exit still fits signed int32.
  const int groups = k / 4;
  const bool aligned_codes = (reinterpret_cast<uintptr_t>(codes) & 1u) == 0;
  uint2* normalized_vectors = reinterpret_cast<uint2*>(normalized);
  for (int group = tid; group < groups; group += kPrepareThreads) {
    const int offset = group * 2; // <=k/2, also safely int32.
    const unsigned packed = aligned_codes
        ? *reinterpret_cast<const uint16_t*>(codes + offset)
        : unsigned(codes[offset]) | (unsigned(codes[offset + 1]) << 8);
    // Unsigned int8 magnitudes {0,1,2,3,4,6,8,12}; each fits DP4A exactly.
    const unsigned magnitudes = __byte_perm(0x03020100u, 0x0c080604u, packed & 0x7777u);
    const unsigned q_squared = __dp4a(magnitudes, magnitudes, 0u);
    const unsigned scale = scales[group / 8];
    uint2 output_bits = make_uint2(0, 0);
    if (q_squared != 0 && scale != 255) {
      const int delta = int(scale) - row_min;
      // q_squared<=4*144=576. For delta<=24 the shift is <=48 bits and
      // the value is <2^58; above 24 a nonzero term already exceeds 2^48.
      const unsigned long long term = delta > 24 ? kEnergyCap
          : (static_cast<unsigned long long>(q_squared) << (2 * delta));
      energy = sat_add(energy, term);
      if constexpr (!kFused)
        if (safe_half) output_bits = normalized_half4_bits(packed, delta);
    }
    // cudaMalloc alignment and k%32==0 make every 64-bit output aligned.
    if constexpr (!kFused)
      normalized_vectors[size_t(row) * groups + group] = output_bits;
  }
#endif
  }
  #pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    energy = sat_add(energy, __shfl_down_sync(0xffffffffu, energy, offset));
  if (lane == 0) warp_energy[warp] = energy;
  __syncthreads();
  if (warp == 0) {
    energy = lane < 4 ? warp_energy[lane] : 0;
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
      energy = sat_add(energy, __shfl_down_sync(0xffffffffu, energy, offset));
    if (lane == 0) {
      RowData result;
      result.energy = energy; result.nan_scales = row_nan;
      result.min_scale = uint8_t(row_min); result.span = uint8_t(row_span);
      result.zero = uint8_t(row_active == 0); result.saturated = uint8_t(energy == kEnergyCap);
      row_data[row] = result;
      if constexpr (kFused) row_bases[row] = uint8_t(row_min);
    }
  }
}

#if MXFP4_CERTIFIED_PREPARE_RAW_CACHE
// Experimental C2 preparation only. Keep prepare_rows<> above untouched so
// default-off builds, K=0, and K>8192 retain the original implementation.
// One CTA caches a row's 16-byte raw block plus one packed (scale,Q) word.
// The SOA layout costs 20*(K/32) dynamic bytes, at most 5120 bytes per CTA.
__global__ __launch_bounds__(kPrepareThreads) void prepare_rows_raw_cached_c2(
    const uint8_t* a, const uint8_t* as, const uint8_t* b, const uint8_t* bs,
    uint16_t* normalized, RowData* row_data, int m, int k) {
  extern __shared__ uint4 raw_blocks[];
  const int blocks = k / 32;
  unsigned* energy_scale = reinterpret_cast<unsigned*>(raw_blocks + blocks);
  const int row = int(blockIdx.x);
  const bool is_a = row < m;
  const int local_row = is_a ? row : row - m;
  const int tid = int(threadIdx.x), lane = tid & 31, warp = tid >> 5;
  // Host dispatch guarantees 0<K<=8192. Each original input pointer is valid;
  // row addresses still use size_t because M and N retain the full ABI range.
  const uint8_t* codes = (is_a ? a : b) + size_t(local_row) * (k / 2);
  const uint8_t* scales = (is_a ? as : bs) + size_t(local_row) * blocks;
  const bool aligned_codes = (reinterpret_cast<uintptr_t>(codes) & 15u) == 0;
  int minimum = 255, maximum = 0;
  unsigned nan_count = 0, active = 0;
  for (int block = tid; block < blocks; block += kPrepareThreads) {
    const uint8_t* source = codes + block * 16;
    uint4 words;
    if (aligned_codes) {
      words = *reinterpret_cast<const uint4*>(source);
    } else {
      words = make_uint4(unaligned_packed_word(source), unaligned_packed_word(source + 4),
                         unaligned_packed_word(source + 8), unaligned_packed_word(source + 12));
    }
    // Exactly the block_q_energy expression, while retaining its loaded words.
    // Q<=32*12^2=4608 fits the low 16 bits; scale occupies bits 16..23.
    const unsigned q_squared = (packed_word_q_energy(words.x) + packed_word_q_energy(words.y))
        + (packed_word_q_energy(words.z) + packed_word_q_energy(words.w));
    const unsigned scale = scales[block];
    raw_blocks[block] = words;
    energy_scale[block] = (scale << 16) | q_squared;
    if (scale == 255) ++nan_count;
    const bool nonzero = q_squared != 0;
    active |= unsigned(nonzero);
    if (nonzero && scale != 255) {
      minimum = min(minimum, int(scale));
      maximum = max(maximum, int(scale));
    }
  }
  #pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    minimum = min(minimum, __shfl_down_sync(0xffffffffu, minimum, offset));
    maximum = max(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
    nan_count += __shfl_down_sync(0xffffffffu, nan_count, offset);
    active |= __shfl_down_sync(0xffffffffu, active, offset);
  }
  __shared__ int warp_min[4], warp_max[4];
  __shared__ unsigned warp_nan[4], warp_active[4];
  __shared__ unsigned long long warp_energy[4];
  __shared__ int row_min, row_span;
  __shared__ unsigned row_nan, row_active;
  if (lane == 0) {
    warp_min[warp] = minimum; warp_max[warp] = maximum;
    warp_nan[warp] = nan_count; warp_active[warp] = active;
  }
  // Publishes raw/energy caches as well as the four warp reductions.
  __syncthreads();
  if (warp == 0) {
    minimum = lane < 4 ? warp_min[lane] : 255;
    maximum = lane < 4 ? warp_max[lane] : 0;
    nan_count = lane < 4 ? warp_nan[lane] : 0;
    active = lane < 4 ? warp_active[lane] : 0;
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
      minimum = min(minimum, __shfl_down_sync(0xffffffffu, minimum, offset));
      maximum = max(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
      nan_count += __shfl_down_sync(0xffffffffu, nan_count, offset);
      active |= __shfl_down_sync(0xffffffffu, active, offset);
    }
    if (lane == 0) {
      row_min = minimum == 255 ? 127 : minimum;
      row_span = minimum == 255 ? 0 : maximum - minimum;
      row_nan = nan_count;
      row_active = active;
    }
  }
  __syncthreads();
  unsigned long long energy = 0;
  for (int block = tid; block < blocks; block += kPrepareThreads) {
    const unsigned cached = energy_scale[block];
    const unsigned q_squared = cached & 0xffffu;
    const unsigned scale = cached >> 16;
    // Whole q-zero blocks may have scale<row_min: exclude before delta/shift.
    // Scale 255 remains counted above, even when the block is entirely zero.
    if (q_squared != 0 && scale != 255) {
      const int delta = int(scale) - row_min;
      // Q<=4608 and 2*delta<=48 imply a term<2^61. Above delta 24,
      // every nonzero term exceeds the unchanged 2^48 energy bound.
      const unsigned long long term = delta > 24 ? kEnergyCap
          : (static_cast<unsigned long long>(q_squared) << (2 * delta));
      energy = sat_add(energy, term);
    }
  }
  const bool safe_half = row_nan == 0 && row_span <= 12;
  const int groups = k / 4;
  const unsigned* raw_words = reinterpret_cast<const unsigned*>(raw_blocks);
  uint2* normalized_vectors = reinterpret_cast<uint2*>(normalized);
  for (int group = tid; group < groups; group += kPrepareThreads) {
    // Adjacent threads read the two halves of one shared 32-bit word. This
    // preserves coalesced 64-bit global output stores and needs no second
    // global packed/scale read. q-zero subgroups are canonicalized to +0.
    const unsigned packed = (raw_words[group / 2] >> ((group & 1) * 16)) & 0xffffu;
    uint2 output_bits = make_uint2(0, 0);
    if (safe_half && (packed & 0x7777u) != 0) {
      const unsigned scale = energy_scale[group / 8] >> 16;
      // A nonzero subgroup implies an active finite block, hence delta>=0;
      // safe_half guarantees delta<=12. Reuse the unchanged exact decoder.
      output_bits = normalized_half4_bits(packed, int(scale) - row_min);
    }
    normalized_vectors[size_t(row) * groups + group] = output_bits;
  }
  #pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    energy = sat_add(energy, __shfl_down_sync(0xffffffffu, energy, offset));
  if (lane == 0) warp_energy[warp] = energy;
  __syncthreads();
  if (warp == 0) {
    energy = lane < 4 ? warp_energy[lane] : 0;
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
      energy = sat_add(energy, __shfl_down_sync(0xffffffffu, energy, offset));
    if (lane == 0) {
      RowData result;
      result.energy = energy; result.nan_scales = row_nan;
      result.min_scale = uint8_t(row_min); result.span = uint8_t(row_span);
      result.zero = uint8_t(row_active == 0); result.saturated = uint8_t(energy == kEnergyCap);
      row_data[row] = result;
    }
  }
}
#endif

__device__ __forceinline__ DeviceMetadata neutral_metadata() {
  DeviceMetadata value{};
  value.min_scale_a = value.min_scale_b = 255;
  return value;
}

__device__ __forceinline__ DeviceMetadata warp_reduce_metadata(DeviceMetadata v) {
  #pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    v.max_energy_a = max(v.max_energy_a, __shfl_down_sync(0xffffffffu, v.max_energy_a, offset));
    v.max_energy_b = max(v.max_energy_b, __shfl_down_sync(0xffffffffu, v.max_energy_b, offset));
    v.min_scale_a = min(v.min_scale_a, __shfl_down_sync(0xffffffffu, v.min_scale_a, offset));
    v.min_scale_b = min(v.min_scale_b, __shfl_down_sync(0xffffffffu, v.min_scale_b, offset));
    v.max_scale_a = max(v.max_scale_a, __shfl_down_sync(0xffffffffu, v.max_scale_a, offset));
    v.max_scale_b = max(v.max_scale_b, __shfl_down_sync(0xffffffffu, v.max_scale_b, offset));
    v.max_span_a = max(v.max_span_a, __shfl_down_sync(0xffffffffu, v.max_span_a, offset));
    v.max_span_b = max(v.max_span_b, __shfl_down_sync(0xffffffffu, v.max_span_b, offset));
    v.nan_scales_a = count_add(v.nan_scales_a, __shfl_down_sync(0xffffffffu, v.nan_scales_a, offset));
    v.nan_scales_b = count_add(v.nan_scales_b, __shfl_down_sync(0xffffffffu, v.nan_scales_b, offset));
    v.zero_rows_a += __shfl_down_sync(0xffffffffu, v.zero_rows_a, offset);
    v.zero_rows_b += __shfl_down_sync(0xffffffffu, v.zero_rows_b, offset);
    v.energy_saturated_a |= __shfl_down_sync(0xffffffffu, v.energy_saturated_a, offset);
    v.energy_saturated_b |= __shfl_down_sync(0xffffffffu, v.energy_saturated_b, offset);
  }
  return v;
}

__global__ __launch_bounds__(kReduceThreads) void finalize_certificate(
    const RowData* rows, DeviceMetadata* output, int m, int n) {
  const int tid = int(threadIdx.x), lane = tid & 31, warp = tid >> 5;
  DeviceMetadata v = neutral_metadata();
  for (int64_t row = tid; row < int64_t(m) + n; row += kReduceThreads) {
    const RowData item = rows[row];
    if (row < m) {
      v.max_energy_a = max(v.max_energy_a, item.energy);
      v.min_scale_a = min(v.min_scale_a, int(item.min_scale));
      v.max_scale_a = max(v.max_scale_a, int(item.min_scale));
      v.max_span_a = max(v.max_span_a, unsigned(item.span));
      v.nan_scales_a = count_add(v.nan_scales_a, item.nan_scales);
      v.zero_rows_a += item.zero; v.energy_saturated_a |= item.saturated;
    } else {
      v.max_energy_b = max(v.max_energy_b, item.energy);
      v.min_scale_b = min(v.min_scale_b, int(item.min_scale));
      v.max_scale_b = max(v.max_scale_b, int(item.min_scale));
      v.max_span_b = max(v.max_span_b, unsigned(item.span));
      v.nan_scales_b = count_add(v.nan_scales_b, item.nan_scales);
      v.zero_rows_b += item.zero; v.energy_saturated_b |= item.saturated;
    }
  }
  v = warp_reduce_metadata(v);
  __shared__ DeviceMetadata partial[kReduceThreads / 32];
  if (lane == 0) partial[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < kReduceThreads / 32 ? partial[lane] : neutral_metadata();
    v = warp_reduce_metadata(v);
    if (lane == 0) {
      unsigned flags = 0;
      if (v.nan_scales_a || v.nan_scales_b) flags |= MXFP4_CERT_REJECT_NAN_SCALE;
      if (v.max_span_a > 12 || v.max_span_b > 12) flags |= MXFP4_CERT_REJECT_SCALE_SPAN;
      // Every normalized energy is an integer. If both sides are nonzero,
      // saturation above 2^48 implies rejection. Never multiply two u64s.
      if (v.max_energy_a && v.max_energy_b &&
          (v.max_energy_b > kEnergyBound || v.max_energy_a > kEnergyBound / v.max_energy_b))
        flags |= MXFP4_CERT_REJECT_ENERGY_BOUND;
      const int lattice_min = v.min_scale_a + v.min_scale_b - 256;
      const int lattice_max = v.max_scale_a + v.max_scale_b - 256;
      if (lattice_min < -126 || lattice_max > 103) flags |= MXFP4_CERT_REJECT_LATTICE_RANGE;
      v.reject_flags = flags;
      v.accepted = flags == 0;
      *output = v;
    }
  }
}

__global__ void postscale_if_certified(float* out, const RowData* rows,
                                       const DeviceMetadata* metadata, int m, int n) {
  if (!metadata->accepted) return;
  const int64_t count = int64_t(m) * n;
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += int64_t(gridDim.x) * blockDim.x) {
    const int row = int(index / n), col = int(index % n);
    const int exponent = int(rows[row].min_scale) + int(rows[m + col].min_scale) - 256;
    const float factor = __uint_as_float(uint32_t(exponent + 127) << 23);
    const float value = out[index];
    // The certificate bounds |value|<=2^24, exponent in [-126,103], so
    // this exact finite multiplication equals ldexpf. Canonicalize zero.
    out[index] = value == 0.0f ? 0.0f : __fmul_rn(value, factor);
  }
}

__device__ __forceinline__ float scale_certified_value(float value, int a_base, int b_base) {
  const int exponent = a_base + b_base - 256;
  const float factor = __uint_as_float(uint32_t(exponent + 127) << 23);
  return value == 0.0f ? 0.0f : __fmul_rn(value, factor);
}

template<bool Aligned>
__global__ void postscale_rows_if_certified(float* out, const RowData* rows,
                                            const DeviceMetadata* metadata, int m, int n) {
  if (!metadata->accepted) return;
  // Fixed row / four adjacent columns: no 64-bit division per output, and
  // four times fewer CTAs than the original one-element-per-thread mapping.
  const int col = int(blockIdx.x) * 1024 + int(threadIdx.x) * 4;
  if (col >= n) return;
  // grid.y is bounded by 65535. int64 iteration handles larger legal M.
  for (int64_t row = blockIdx.y; row < m; row += gridDim.y) {
    const int a_base = rows[row].min_scale;
    float* destination = out + size_t(row) * n + col;
    if constexpr (Aligned) {
      // Host dispatch proved out%16==0 && N%4==0, so each row and group
      // are aligned; col<N then also implies col+3<N.
      float4 value = *reinterpret_cast<const float4*>(destination);
      value.x = scale_certified_value(value.x, a_base, rows[m + col].min_scale);
      value.y = scale_certified_value(value.y, a_base, rows[m + col + 1].min_scale);
      value.z = scale_certified_value(value.z, a_base, rows[m + col + 2].min_scale);
      value.w = scale_certified_value(value.w, a_base, rows[m + col + 3].min_scale);
      *reinterpret_cast<float4*>(destination) = value;
    } else {
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (col + i < n)
          destination[i] = scale_certified_value(destination[i], a_base, rows[m + col + i].min_scale);
      }
    }
  }
}

int validate_shape(int64_t m, int64_t n, int64_t k, uint64_t* bytes) {
  if (!bytes || m <= 0 || n <= 0 || k < 0 || k % 32 != 0 ||
      m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() || m + n > std::numeric_limits<int>::max())
    return int(cudaErrorInvalidValue);
  const uint64_t limit = uint64_t(std::numeric_limits<ptrdiff_t>::max());
  const uint64_t rows = uint64_t(m) + uint64_t(n);
  if (uint64_t(m) > limit / sizeof(float) / uint64_t(n)) return int(cudaErrorInvalidValue);
  const uint64_t fixed = (kFused ? 0 : kBlasWorkspace) + sizeof(DeviceMetadata)
      + rows * (sizeof(RowData) + (kFused ? sizeof(uint8_t) : 0));
  if (fixed > limit || (k && rows > (limit - fixed) / (2 * uint64_t(k))))
    return int(cudaErrorInvalidValue);
  *bytes = fixed + (kFused ? 0 : rows * uint64_t(k) * 2);
  return int(cudaSuccess);
}

int check_device(Context* context) {
  if (context->owner_thread != std::this_thread::get_id()) return int(cudaErrorInvalidResourceHandle);
  int device = -1;
  const cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return int(status);
  return device == context->device ? int(cudaSuccess) : int(cudaErrorInvalidDevice);
}

int blas_status(cublasStatus_t status) {
  return status == CUBLAS_STATUS_SUCCESS ? 0 : MXFP4_CERTIFIED_CUBLAS_ERROR_BASE + int(status);
}

int release(Context* c, bool synchronize) {
  int first = 0;
  auto remember = [&](int status) { if (!first && status) first = status; };
  if (synchronize && c->has_stream_work) remember(int(cudaStreamSynchronize(c->last_stream)));
  if (c->blas) remember(blas_status(cublasDestroy(c->blas)));
  if (c->normalized) remember(int(cudaFree(c->normalized)));
  if (c->row_bases) remember(int(cudaFree(c->row_bases)));
  if (c->rows) remember(int(cudaFree(c->rows)));
  if (c->metadata) remember(int(cudaFree(c->metadata)));
  if (c->blas_workspace) remember(int(cudaFree(c->blas_workspace)));
  if (c->host_metadata) remember(int(cudaFreeHost(c->host_metadata)));
  delete c;
  return first;
}

void static_info(const Context* c, MXFP4CertifiedInfo* info) {
  *info = MXFP4CertifiedInfo{};
  info->abi_version = MXFP4_CERTIFIED_ABI_VERSION;
  info->device = c->device; info->m = c->m; info->n = c->n; info->padded_k = c->k;
  info->workspace_bytes = c->workspace_bytes; info->submitted_calls = c->submitted_calls;
}

int query_info_locked(Context* c, MXFP4CertifiedInfo* info) {
  static_info(c, info);
  if (!c->metadata_valid) return 0;
  cudaError_t status = cudaMemcpyAsync(c->host_metadata, c->metadata, sizeof(DeviceMetadata),
                                        cudaMemcpyDeviceToHost, c->last_stream);
  if (status != cudaSuccess) return int(status);
  status = cudaStreamSynchronize(c->last_stream);
  if (status != cudaSuccess) return int(status);
  const DeviceMetadata& v = *c->host_metadata;
  info->max_energy_a = v.max_energy_a; info->max_energy_b = v.max_energy_b;
  info->accepted = unsigned(v.accepted); info->reject_flags = v.reject_flags;
  info->min_scale_a = v.min_scale_a; info->max_scale_a = v.max_scale_a;
  info->min_scale_b = v.min_scale_b; info->max_scale_b = v.max_scale_b;
  info->max_span_a = v.max_span_a; info->max_span_b = v.max_span_b;
  info->nan_scales_a = v.nan_scales_a; info->nan_scales_b = v.nan_scales_b;
  info->zero_rows_a = v.zero_rows_a; info->zero_rows_b = v.zero_rows_b;
  info->energy_saturated_a = v.energy_saturated_a;
  info->energy_saturated_b = v.energy_saturated_b;
  info->metadata_valid = 1;
  return 0;
}

}  // namespace

extern "C" const char* mxfp4_certified_error_string(int status) {
  if (status < MXFP4_CERTIFIED_CUBLAS_ERROR_BASE) return cudaGetErrorString(cudaError_t(status));
  switch (status - MXFP4_CERTIFIED_CUBLAS_ERROR_BASE) {
    case CUBLAS_STATUS_NOT_INITIALIZED: return "cuBLAS: not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED: return "cuBLAS: allocation failed";
    case CUBLAS_STATUS_INVALID_VALUE: return "cuBLAS: invalid value";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "cuBLAS: architecture mismatch";
    case CUBLAS_STATUS_MAPPING_ERROR: return "cuBLAS: mapping error";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "cuBLAS: execution failed";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "cuBLAS: internal error";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "cuBLAS: not supported";
    case CUBLAS_STATUS_LICENSE_ERROR: return "cuBLAS: license error";
    default: return "cuBLAS: unknown status";
  }
}

extern "C" int mxfp4_certified_workspace_bytes(int64_t m, int64_t n, int64_t k,
                                               uint64_t* bytes) {
  if (bytes) *bytes = 0;
  return validate_shape(m, n, k, bytes);
}

extern "C" int mxfp4_certified_create(int64_t m, int64_t n, int64_t k, void** context) {
  if (!context) return int(cudaErrorInvalidValue);
  *context = nullptr;
  uint64_t bytes = 0;
  int status = validate_shape(m, n, k, &bytes);
  if (status) return status;
  Context* c = new (std::nothrow) Context;
  if (!c) return int(cudaErrorMemoryAllocation);
  c->m = int(m); c->n = int(n); c->k = int(k); c->workspace_bytes = bytes;
  status = int(cudaGetDevice(&c->device));
  if (!status) {
    cudaDeviceProp properties{};
    status = int(cudaGetDeviceProperties(&properties, c->device));
    if (!status && (properties.major != 7 || properties.minor != 0)) status = int(cudaErrorNotSupported);
  }
  const size_t row_count = size_t(m) + size_t(n);
  if (!status && k && !kFused)
    status = int(cudaMalloc(reinterpret_cast<void**>(&c->normalized), row_count * size_t(k) * 2));
  if (!status && kFused)
    status = int(cudaMalloc(reinterpret_cast<void**>(&c->row_bases), row_count));
  if (!status) status = int(cudaMalloc(reinterpret_cast<void**>(&c->rows), row_count * sizeof(RowData)));
  if (!status) status = int(cudaMalloc(reinterpret_cast<void**>(&c->metadata), sizeof(DeviceMetadata)));
  if (!status && !kFused) status = int(cudaMalloc(&c->blas_workspace, kBlasWorkspace));
  if (!status) status = int(cudaMallocHost(reinterpret_cast<void**>(&c->host_metadata), sizeof(DeviceMetadata)));
  if (!status && !kFused) status = blas_status(cublasCreate(&c->blas));
  if (!status && !kFused) status = blas_status(cublasSetPointerMode(c->blas, CUBLAS_POINTER_MODE_HOST));
  if (!status && !kFused) status = blas_status(cublasSetMathMode(c->blas, cublasMath_t(
      CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
  if (status) { release(c, false); return status; }
  *context = c;
  return 0;
}

extern "C" int mxfp4_certified_gemm(
    void* context, const uint8_t* a, const uint8_t* as,
    const uint8_t* b, const uint8_t* bs, float* out,
    cudaStream_t stream, int* used_fast) {
  if (used_fast) *used_fast = -1;
  if (!context || !out) return int(cudaErrorInvalidValue);
  Context* c = static_cast<Context*>(context);
  Lock lock(c);
  if (!lock.acquired) return int(cudaErrorNotReady);
  int status = check_device(c);
  if (status) return status;
  if (c->k && (!a || !as || !b || !bs)) return int(cudaErrorInvalidValue);
  if (c->has_stream_work && stream != c->last_stream) {
    status = int(cudaStreamQuery(c->last_stream));
    if (status) return status; // No hidden wait when switching streams.
  }
  c->metadata_valid = false;
  c->last_stream = stream; c->has_stream_work = true;
#if MXFP4_CERTIFIED_PREPARE_RAW_CACHE
  if (use_c2_prepare_raw_cache(c->k)) {
    const size_t cache_bytes = size_t(c->k / 32) * (sizeof(uint4) + sizeof(unsigned));
    prepare_rows_raw_cached_c2<<<unsigned(c->m + c->n), kPrepareThreads, cache_bytes, stream>>>(
        a, as, b, bs, c->normalized, c->rows, c->m, c->k);
  } else
#endif
#if MXFP4_CERTIFIED_VARIANT >= 3
  if (c->k / 32 <= kCachedEnergyBlocks) {
    const size_t cache_bytes = size_t(c->k / 32) * sizeof(unsigned);
    prepare_rows<true><<<unsigned(c->m + c->n), kPrepareThreads, cache_bytes, stream>>>(
        a, as, b, bs, c->normalized, c->rows, c->row_bases, c->m, c->n, c->k);
  } else
#endif
  {
    prepare_rows<false><<<unsigned(c->m + c->n), kPrepareThreads, 0, stream>>>(
        a, as, b, bs, c->normalized, c->rows, c->row_bases, c->m, c->n, c->k);
  }
  status = int(cudaGetLastError());
  if (status) return status;
  finalize_certificate<<<1, kReduceThreads, 0, stream>>>(c->rows, c->metadata, c->m, c->n);
  status = int(cudaGetLastError());
  if (status) return status;
  const int* accepted = reinterpret_cast<const int*>(
      reinterpret_cast<const char*>(c->metadata) + offsetof(DeviceMetadata, accepted));
#if MXFP4_CERTIFIED_VARIANT >= 3
  status = mxfp4_gemm_fused_cuda(a, as, b, bs, out, c->m, c->n, c->k,
      c->row_bases, accepted, MXFP4_CERTIFIED_VARIANT - 3, stream);
  if (status) return status;
#else
  if (c->k) {
    status = blas_status(cublasSetStream(c->blas, stream));
    if (status) return status;
    // cublasSetStream unconditionally resets the workspace: reapply it here.
    status = blas_status(cublasSetWorkspace(c->blas, c->blas_workspace, kBlasWorkspace));
    if (status) return status;
    const float alpha = 1.0f, beta = 0.0f;
    const uint16_t* an = c->normalized;
    const uint16_t* bn = c->normalized + size_t(c->m) * c->k;
    status = blas_status(cublasGemmEx(c->blas, CUBLAS_OP_T, CUBLAS_OP_N,
        c->n, c->m, c->k, &alpha, bn, CUDA_R_16F, c->k, an, CUDA_R_16F, c->k,
        &beta, out, CUDA_R_32F, c->n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    if (status) return status;
  } else {
    status = int(cudaMemsetAsync(out, 0, size_t(c->m) * c->n * sizeof(float), stream));
    if (status) return status;
  }
#if MXFP4_CERTIFIED_VARIANT >= 2
  const dim3 grid(unsigned((uint64_t(c->n) + 1023) / 1024),
                  unsigned(c->m < 65535 ? c->m : 65535));
  if ((reinterpret_cast<uintptr_t>(out) & 15u) == 0 && c->n % 4 == 0)
    postscale_rows_if_certified<true><<<grid, 256, 0, stream>>>(out, c->rows, c->metadata, c->m, c->n);
  else
    postscale_rows_if_certified<false><<<grid, 256, 0, stream>>>(out, c->rows, c->metadata, c->m, c->n);
#else
  const uint64_t values = uint64_t(c->m) * c->n;
  const unsigned grid = unsigned((values + 255) / 256 < 65535 ? (values + 255) / 256 : 65535);
  postscale_if_certified<<<grid, 256, 0, stream>>>(out, c->rows, c->metadata, c->m, c->n);
#endif
  status = int(cudaGetLastError());
  if (status) return status;
#endif
  status = mxfp4_gemm_tiled_if_cuda(a, as, b, bs, out, c->m, c->n, c->k, stream, accepted);
  if (status) return status;
  ++c->submitted_calls;
  c->metadata_valid = true;
  if (used_fast) {
    MXFP4CertifiedInfo info;
    status = query_info_locked(c, &info);
    if (status) return status;
    *used_fast = int(info.accepted);
  }
  return 0;
}

extern "C" int mxfp4_certified_get_info(void* context, MXFP4CertifiedInfo* info) {
  if (!context || !info) return int(cudaErrorInvalidValue);
  Context* c = static_cast<Context*>(context);
  Lock lock(c);
  if (!lock.acquired) return int(cudaErrorNotReady);
  const int status = check_device(c);
  if (status) return status;
  return query_info_locked(c, info);
}

extern "C" int mxfp4_certified_destroy(void* context) {
  if (!context) return 0;
  Context* c = static_cast<Context*>(context);
  const int status = check_device(c);
  if (status) return status;
  if (c->busy.test_and_set(std::memory_order_acquire)) return int(cudaErrorNotReady);
  // Do not clear the lock after deletion; no concurrent host use is allowed.
  return release(c, true);
}
