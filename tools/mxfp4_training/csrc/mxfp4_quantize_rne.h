#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Experimental RNE MX32 encoder for finite FP32 x[rows, logical_k]. Input
// strides are nonnegative counts of float elements and may describe a transpose
// or a sliced/offset/broadcast view. rows and logical_k must be positive. Input values
// must be finite; original -0 has nibble 8. Padding to the next MX32 block is
// canonical +0, regardless of the source tensor's neighboring storage.
//
// Let amax be each block's maximum absolute FP32 value, including +0 padding.
// E8M0 exponent = 0 for amax==0, otherwise
// clamp(floor(log2(amax))-2, -127, 127). Finite FP32 inputs reach at most 125.
// Encode abs(x)*2^-exponent in E2M1 by nearest, ties to even magnitude code,
// with saturation at 6. Packed bytes contain even K in the low nibble.
// Output codes[rows, ceil(logical_k/32)*16] and
// scales[rows, ceil(logical_k/32)] are contiguous uint8, E8M0 bias 127.
//
// Returns cudaError_t and enqueues flag initialization followed by one kernel
// on stream. error_flag must point to a separate aligned device int: the kernel
// sets it to 1 if ANY input is NaN or infinity, otherwise it remains 0. On a
// nonfinite input the encoded outputs are unspecified and MUST NOT be used.
// Callers must wait for stream completion and read the flag before acceptance.
// CUDA Graph capture is rejected, matching the synchronous Python validation
// wrapper. The C ABI itself does not synchronize or copy data to the host.
//
// All buffers must belong to the current CUDA device, remain alive through
// completion, have the stated byte extents, and not overlap. The input pointer
// is float-aligned; codes and scales have no extra alignment requirement.
// Device/extent/alias ownership is the caller's responsibility.
int mxfp4_quantize_rne_cuda(const float* x, uint8_t* codes, uint8_t* scales,
                           int* error_flag, int64_t rows, int64_t logical_k,
                           int64_t stride_row, int64_t stride_col,
                           cudaStream_t stream);

int mxfp4_quantize_rne_cuda_abi_version(void);
const char* mxfp4_quantize_rne_cuda_backend(void);

#ifdef __cplusplus
}
#endif
