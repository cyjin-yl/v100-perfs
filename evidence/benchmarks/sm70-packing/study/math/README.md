# FP16 packing: CPU search and mathematical bounds

## Scope

These results were actually computed on the CPU with NumPy 2.3.5. **No result in this directory is a V100 measurement**, and the script does not model the undocumented internal summation order of Volta Tensor Cores. The CUDA harness prepared by another agent must still be compiled and run on a real sm70 GPU.

Run:

```bash
python3 packing_exhaustive.py --output packing_results.json
```

Only NumPy is required. The JSON records its Python/NumPy versions, creation time, source hash, exact test counts, failures and counterexamples.

## What is being packed?

The 15 distinct finite E2M1 numerical values are

\[
x\in\{0,\pm0.5,\pm1,\pm1.5,\pm2,\pm3,\pm4,\pm6\}.
\]

We use `q=2*x`, making every element an integer. Positive and negative zero share one numerical value; this experiment does not preserve the sign bit of zero. A dot product of these integer `q` values is divided by four to recover the E2M1 element dot product. MX block scales are a further, separate multiplication.

The ordinary FP16 input / FP32 accumulator path does not restrict the output to a 10-bit significand. It is therefore invalid to dismiss *all* packing schemes merely by counting the 10 explicit fraction bits in an FP16 input. Exactness depends on the encoding, the output precision, and the length of the reduction.

## Positive result 1: two independent ternary dots

Let all four input sequences have values in `{-1,0,1}` and length `K=16`. Encode

\[
A_k=a_{0k}+128a_{1k},\qquad B_k=b_{0k}+128b_{1k}.
\]

One packed dot product is

\[
Z=\sum_k A_kB_k=L+128C+16384H,
\]

where

\[
L=\sum_k a_{0k}b_{0k},\quad
C=\sum_k(a_{0k}b_{1k}+a_{1k}b_{0k}),\quad
H=\sum_k a_{1k}b_{1k}.
\]

The desired independent results are `L` and `H`. The middle coefficient `C` is an unwanted cross term, but it can be separated and discarded:

```text
H = round(Z / 16384)
remainder = Z - 16384*H
C = round(remainder / 128)
L = remainder - 128*C
```

This is exact in the numerical model because:

- Every packed operand has magnitude at most 129 and is an exact FP16 integer.
- `|L|,|H| <= 16`, `|C| <= 32`.
- `|128*C+L| <= 4112 < 8192`, so extracting `H` does not mix coefficients.
- `|L| <= 16 < 64`, so extracting `C` does not mix coefficients.
- Every product and every partial sum is an integer with magnitude at most `16*129*129 = 266256`, well below the exact FP32 integer limit.

The script checks all 81 scalar input quadruples and all 70,785 integer coefficient triples in a conservative box containing every possible length-16 result: **zero errors**. The integer bounds address all legal input sequences, regardless of summation order; the enumeration is of the output superset, not the astronomically larger set of input sequences.

This is a direct counterexample to a universal assertion that two independent low-precision operations cannot be carried through a floating-point multiply-accumulate. It is a restricted ternary scheme, not a full-domain MXFP4 implementation, and no speedup has been measured.

### The K=32 variant

Using radix 256 instead of 128 extends the exact window to K=32, aligning with the length of an MX scaling block. Here `|L|,|H| <= 32`, `|C| <= 64`, and `|256*C+L| <= 16416 < 32768`. The largest possible partial accumulator is `32*257*257=2,113,568`, still exactly representable in FP32. The script checks all **545,025** integer coefficient triples in this conservative box and finds zero errors. The same bound is valid through K=63, although only K=16 and K=32 conservative boxes are exhaustively checked in this script.

## Positive result 2: two outputs sharing one RHS, with a wider domain

For `q` in `{-3,-2,-1,0,1,2,3}`, corresponding to E2M1 elements between -1.5 and +1.5, pack

\[
P_k=a_{0k}+512a_{1k},\quad
Z=\sum_{k=1}^{16}P_k b_k.
\]

Decode

\[
C_1=\operatorname{round}(Z/512),\qquad C_0=Z-512C_1.
\]

`|P_k| <= 1539`, so every packed operand is exactly representable in FP16. Each unpacked dot has magnitude at most `16*3*3=144`, below the half-radix guard of 256. Any partial packed sum has magnitude at most 73,872, also exact in FP32.

The script exhausts all 343 scalar triples, and a superset of all possible length-16 output pairs containing **83,521 pairs**: zero packing or decoding errors. This can represent two rows multiplied by a shared matrix on the right.

The same decoder is not valid for arbitrarily long reductions. At `K=32`, setting `a0=b=3` and `a1=0` gives `Z=288`; nearest-radix decoding produces `(-224,1)` instead of `(288,0)`. This failure is an exact carry collision, not floating-point error. Interestingly, all 4,096 random tests at K=32, 64 and 128 happened to pass with the fixed seed. That contrast demonstrates why random zero-error results cannot establish a safe domain.

## Positive result 3: full E2M1 scalar lookup with a shared RHS

For the complete 15-value `q` set, encode `P=a0+128*a1` and multiply by `b`. All 225 input pair encodings are exact in FP16. Exhausting all `15^3=3375` scalar combinations gives 849 distinct FP32 product keys and **no key with conflicting output targets `(a0*b,a1*b)`**.

Thus a lookup table can decode both full-domain scalar outputs. A simple nearest-radix decoder is insufficient: it fails on 300 of the 3,375 combinations. The successful assertion is specifically about a lookup decoder.

### Why this does not extend to a Tensor Core dot product

At length two, the same exact packed sum can represent different wanted answers:

| Case | `a0` | `a1` | Shared `b` | Packed dot `Z` | Desired integer dots |
| --- | --- | --- | --- | --- | --- |
| A | `[8,0]` | `[0,0]` | `[8,1]` | `64` | `(64,0)` |
| B | `[-8,0]` | `[0,1]` | `[8,1]` | `-64+128=64` | `(-64,1)` |

All values are legal scaled E2M1 elements; every operation is exactly representable. No decoder observing only `Z=64` can distinguish these two cases. Padding both inputs with zeros produces the same counterexample for a WMMA length-16 operation.

Increasing the radix loses the input low lane instead. At radix 512, `(a0,a1)=(0,12)` and `(1,12)` encode to intended integers 6144 and 6145, but both become FP16 value 6144. The loss occurs before multiplication.

The sweep covers power-of-two radices from 1 through 4096. It falsifies this particular direct fixed-radix family as a universal full-domain reduction scheme; it does **not** prove that all nonlinear encodings, auxiliary correction schemes, approximate methods or specialized data distributions are impossible.

## Appendix: nonlinear scalar information-theoretic counterexample

Assign each of the 225 ordered `q` pairs a different prime, using the first 225 primes. The largest is 1427, so all codes are exact FP16 integers; their products are at most 2,036,329, exact in FP32. The prime product identifies the unordered pair of original pairs. Swapping the two whole operands does not change either wanted per-lane product.

All `225^2=50,625` independent four-value combinations were exhausted, yielding 25,425 distinct product keys and zero conflicting target products. This construction is an existence proof only. Factoring/table decoding is costly, and arbitrary encoded products cannot be accumulated to recover the desired sums. It is not proposed as a fast GEMM algorithm.

## What would decide practical value?

The real GPU experiment should compare equal useful mathematical work and verify generated HMMA instructions. For a shared-RHS scheme, compare one conventional GEMM with twice as many output rows against one packed GEMM, rather than comparing against two artificially small launches. For independent matrices, compare against an appropriately batched or fused baseline.

Report both prepacked GEMM timing and the complete encode → GEMM → decode timing. Long `K` requires safe partial accumulation and extraction; that extra work must be timed. Full MXFP4 additionally requires packed 4-bit storage, 32-element scales, conversion cost and the correct block grouping for forward and backward matrix layouts. Training claims require convergence/error experiments and measurements of the complete training step.

## Primary sources checked

1. [NVIDIA PTX ISA 6.0 / CUDA 9.0, WMMA section](https://docs.nvidia.com/cuda/archive/9.0/parallel-thread-execution/index.html#warp-level-matrix-multiply-accumulate-instructions-wmma-mma). The specification describes at-least-single-precision multiplication, at-least-single-precision accumulation for FP32 accumulators, and unspecified accumulation order, rounding, and subnormal treatment. The documented target is sm70 or newer. These integer-valued candidate constructions avoid subnormals and stay far inside the FP32 exact-integer range; hardware validation remains necessary.
2. [OCP Microscaling Formats Specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf/). MXFP4 uses E2M1 elements, an E8M0 block scale and 32 elements per scaling block. The experiments above operate on the normalized element arithmetic rather than implementing the full storage/conversion format.
3. [ULPPACK, MLSys 2022](https://proceedings.mlsys.org/paper_files/paper/2022/file/e09d45e14e9ece7142217550ddd3c4d0-Paper.pdf). This is a genuine precedent for packing multiple small operations into a wider multiplier and controlling local accumulation length. Its evaluated implementation uses integer CPU SIMD, so its performance results are not evidence for V100 FP16 packing performance.
