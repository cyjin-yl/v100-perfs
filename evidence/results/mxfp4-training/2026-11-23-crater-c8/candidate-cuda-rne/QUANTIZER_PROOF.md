# CUDA RNE MX32 candidate: contract and verification boundary

This is an opt-in candidate. No GPU execution or CUDA compilation is claimed by
this file. `quantize_mx_torch` and the stochastic rounding algorithm stay
unchanged. `MXRecipe.quantizer_backend` defaults to `torch`; `cuda_rne` selects
`quantize_mx_cuda_rne` only for an operand whose actual rounding is `nearest`.
Forward, Dgrad and Wgrad still re-encode their original FP32 operands in the
required direction. No packed transpose or quantized-weight reuse is introduced.

## Scale proof

Finite nonnegative IEEE binary32 bit patterns have the same order as their
numerical values. Masking the sign bit and applying integer max therefore finds
the exact block amax, including subnormals. A block containing only +0/-0 uses
scale exponent zero. For a nonzero normal with biased exponent E,
`floor(log2(amax))-2 = E-129`; clamp to [-127,127]. Every nonzero subnormal clamps
to -127. The largest finite FP32 biased exponent is 254, so the largest reachable
exponent is 125 and E8M0 byte 252. Scale byte 255 is never generated.

With this exponent, every finite normalized magnitude is strictly less than 8.
For the unclamped case the largest value is in [4,8); low-end clamping makes it
smaller. There is consequently no normalization overflow. A power-of-two scale
change is exact whenever its binary32 result is normal: it changes only the
exponent and preserves the available significand bits. Each of the seven RNE
thresholds is normal and exactly representable. Values and source nextafter
neighbors in any threshold neighborhood therefore reach their exact normalized
positions. Subnormal/underflow results lie far below the first threshold 0.25;
any rounding there still selects magnitude code zero. `--ftz=false` is retained.
CUDA 12.8.1's Programming Guide, Table 15, lists ldexpf with maximum error 0 ULP:
https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#mathematical-functions
The libdevice definition is a power-of-two scale operation:
https://docs.nvidia.com/cuda/archive/12.8.1/libdevice-users-guide/__nv_ldexpf.html

## RNE and sign proof

The ordered E2M1 magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4, 6. Summing these seven
predicates gives exactly the nearest magnitude code with ties to an even code:

| Boundary | Increment code when |
|---|---|
| 0.25 | greater |
| 0.75 | greater or equal |
| 1.25 | greater |
| 1.75 | greater or equal |
| 2.5 | greater |
| 3.5 | greater or equal |
| 5 | greater |

The final predicate also saturates every larger finite magnitude to code 7;
clipping explicitly to 6 would choose the same code. The input bit-31 sign is
copied to nibble bit 3, including original -0 and negative values rounded to zero.
Tail lanes begin with raw bits 0 and never read neighboring input storage. This
makes every padded nibble canonical +0. Even lanes combine their nibble with the
next odd lane and write one byte. Each output byte and each scale byte has one
writer.

## Warp, address and stream obligations

There are eight full warps per 256-thread CTA, with one warp per MX32. The
flattened block loop and its bound are warp-uniform, so full-mask ballot and
shuffles include all 32 lanes even at K tails. Grid-stride work extends beyond
the capped launch grid without a CTA barrier or shared memory. Input offsets use
nonnegative element strides; row/column broadcast strides of zero are valid
read-only aliases. `data_ptr()` already includes PyTorch's storage offset.
The C API checks K+31, output byte products and the maximum float offset before
launch. The int32 specialization additionally bounds dimensions, strides,
maximum input index and total packed bytes. Larger legal offsets use int64.

The native API first enqueues a same-stream zero of a private error flag, then
one kernel. Nonfinite inputs are detected by integer bits and excluded from the
amax arithmetic; a bad warp atomically sets the flag. Invalid encoded outputs
are unspecified. Python reads the flag with `.item()` before returning and
raises ValueError if nonzero, so rejected outputs never reach GEMM. That stream
wait is included in wrapper timing. Input, output and flag tensors are recorded
on the stream before native submission; optional lazy negative/conjugate views
are resolved on the same stream, with their original storage recorded first.
No context cache, cross-device fallback or CPU fallback is introduced. Graph
capture is rejected before Python allocation/submission and before native flag
initialization. The caller still owns producer-to-consumer stream dependencies.
PyTorch documents record_stream as allocator lifetime protection:
https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html

## Actual GPU gates required before acceptance

The independent checker must compare packed codes and E8M0 scales byte-for-byte
with the existing NumPy oracle for every reachable exponent -127..125, all seven
midpoints and their source nextafter neighbors with both signs, finite raw-bit
samples, extrema/subnormals/zeros, padding, strided/offset/transposed/broadcast
views, rejection of nonfinite input and a nondefault stream. MXLinear forward,
Dgrad and Wgrad must also be checked with the explicit new recipe. CPU model or
source review is not a substitute for this compiled-kernel check.

The benchmark compares two-input complete quantization wrappers on identical
resident FP32 inputs. Allocation, native finite validation and synchronization
remain inside the timed wrapper. Initial and actual final timed outputs require
bytewise checking. Quantization is reported in milliseconds, not FLOPS. The
width-32 smoke remains an integration/loss smoke; its geometry does not establish
128x128 halfrow fast-loader performance, including when the library is C10.
