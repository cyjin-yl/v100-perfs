// Compile-only SM70 decoder probe. No GPU execution or performance claim.
// Precondition, established separately under the V100 BFMA4 numerical model:
// z = L + 512*T + 262144*H + error, z is an exactly represented integer,
// |z| < 2^25, |T| <= 216, and |L + error| <= 126.
// These are the restricted q in [-3,3], packed-K<=12 middle-coefficient case.
#include <cuda_runtime.h>

__device__ __forceinline__ float middle_float(float z) {
  float high = rintf(z * (1.0f / 262144.0f));
  float residual = fmaf(-262144.0f, high, z);
  return rintf(residual * (1.0f / 512.0f));
}

__device__ __forceinline__ int middle_bfe(float z) {
  // Conversion is exact under the integer/range precondition, for both signs.
  int biased = __float2int_rz(z) + 256;
  int result;
  asm("bfe.s32 %0, %1, 9, 9;" : "=r"(result) : "r"(biased));
  return result;
}

__device__ __forceinline__ int middle_shift(float z) {
  unsigned biased = (unsigned)(__float2int_rz(z) + 256);
  // Unsigned left shift is defined modulo 2^32. The emitted SM70 arithmetic
  // right shift implements signed extension of this nine-bit field.
  return ((int)(biased << 14)) >> 23;
}

// Matched float-output kernels: same input, output type, launch, and memory.
extern "C" __global__ void middle_float_output(const float* z, float* out, int n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (unsigned)n) out[i] = middle_float(z[i]);
}

extern "C" __global__ void middle_bfe_float_output(const float* z, float* out, int n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (unsigned)n) out[i] = (float)middle_bfe(z[i]);
}

extern "C" __global__ void middle_shift_float_output(const float* z, float* out, int n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (unsigned)n) out[i] = (float)middle_shift(z[i]);
}

// Additional endpoint when integer partial sums can be kept until MX32 scaling.
// It is intentionally reported separately from the matched float comparison.
extern "C" __global__ void middle_bfe_integer_output(const float* z, int* out, int n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (unsigned)n) out[i] = middle_bfe(z[i]);
}

// The same integer endpoint expressed as shifts, avoiding signed left-shift UB.
// NVCC may choose a different lowering than the PTX bitfield extraction.
extern "C" __global__ void middle_shift_integer_output(const float* z, int* out, int n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (unsigned)n) out[i] = middle_shift(z[i]);
}
