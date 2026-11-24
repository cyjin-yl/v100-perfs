# Middle coefficient decoder: independent review and SM70 compile evidence

This is a small, separate compiler probe. **No GPU kernel has been executed.**
It does not modify the frozen two-output harness in `../cuda`, provide a new
end-to-end GEMM, or measure speed. It targets a different objective: combining
two K positions of the **same** dot product and retaining only its middle
coefficient.

## Reviewed numerical contract

Let all four integer operands lie in `[-3,3]`, corresponding to the restricted
E2M1 values `q/2`. Use `P=a0+512*a1` and `Q=b1+512*b0` for at most 12 packed
K terms. The packed sum is

```
Z = L + 512*T + 262144*H
L = sum(a0*b1)
T = sum(a0*b0 + a1*b1)       # required original dot product
H = sum(a1*b0)
```

All input packs have magnitude at most 1539 and are exactly FP16. The bounds
are `|L|,|H| <=108`, `|T|<=216`, and
`sum(abs(P*Q)) <= 12*1539^2 = 28,422,252 < 2^25`.

The independent review uses the previously ported, published **V100 BFMA4
numerical model**, implemented in `../../numerics/v100_model.py`: four exact
half products and C, global significand alignment with truncation of each
signed magnitude, and final rounding toward zero. This is not a substitute
for hardware validation of the model or the generated kernel.

For this integer domain every model result remains an integer. The largest
alignment/final grid is 2. In each BFMA4 group, each of the four products and C
can lose at most 1 absolute integer unit during alignment, and final rounding
can lose at most another 1. Three groups therefore give the conservative
bound `|error|<=18`, regardless of signs and cancellation. Alignment and RZ
never increase an individual magnitude, so the absolute partial-sum bound
continues to hold throughout the induction.

Two strict inequalities justify the original floating decoder:

```
|512*T + L + error| <= 110,718 < 131,072
|L + error| <= 126 < 256
```

Thus `high=round(z/262144)` recovers H and
`round((z-262144*high)/512)` recovers T. Low bits may be lost without affecting
the middle coefficient because the entire low residual plus error remains
inside its half-radix rounding interval. This argument covers both signs;
the bounds exclude halfway ties. It does not require the packed sum to be
exact.

## Integer extraction and the compiler improvement

The model output z is itself an exactly represented FP32 integer with
`abs(z)<2^25`. Consequently `__float2int_rz(z)` is exact in int32. Define

```
zi = int(z)
u  = zi + 256
T  = signed_extract_9_bits(u, start=9)
```

Indeed `floor((zi+256)/512) = T + 512*H`, because
`0 < L+error+256 <512`. Extracting and sign-extending the bottom nine bits
removes H and recovers T, since `|T|<=216<256`. This works for negative z as
well, using two's-complement arithmetic shift.

The direct PTX `bfe.s32` lowering was less compact than its PTX spelling
suggests. Expressing the extraction as an **unsigned** left shift followed
by arithmetic right shift allowed the compiler to fuse the bias and left
shift into `LEA`:

```cpp
unsigned u = (unsigned)(__float2int_rz(z) + 256);
int middle = ((int)(u << 14)) >> 23;
```

The unsigned left shift is modulo 2^32. The emitted SM70 `SHF.R.S32.HI`
confirms the required signed right shift. The PTX bitfield and shift forms
are included side by side in the source; no assumption is made that every
future compiler must choose the same lowering.

## Actual compiled evidence

CUDA 12.8.93 compiled all five kernels for `sm_70`, with zero spill stores,
zero spill loads, zero stack, and zero local memory. Registers are for the
whole small probe kernel, including its common indexing and load/store.

| Decoder and endpoint | Decoder SASS | Count | Registers |
| --- | --- | ---: | ---: |
| Original, float output | FMUL, FRND, FFMA, FMUL, FRND | 5 | 12 |
| PTX BFE, float output | F2I, IADD3, SHF, SGXT, I2F | 5 | 10 |
| Shift/LEA, float output | F2I, LEA, SHF, I2F | 4 | 10 |
| PTX BFE, integer output | F2I, IADD3, SHF, SGXT | 4 | 10 |
| Shift/LEA, integer output | F2I, LEA, SHF | 3 | 10 |

The float-output comparison has matching input/output types and memory.
The integer-output endpoint is useful if two partial integer sums are added
before one MX32 scale application, but it is reported separately. Fewer
instructions do **not** establish throughput or latency gains; these
instructions use different pipelines and dependency latencies, and an
integrated matrix kernel can have different register pressure.

CPU identity checks consider the conservative independent box
`H in [-108,108]`, `T in [-216,216]`, and `L+error in [-126,126]`.
There are 23,772,133 combinations; 18,916,471 produce exactly representable
FP32 integer z values and all three decoder formulas return T in every
admitted case. The remaining 4,855,662 arbitrary box combinations are
excluded because their constructed z is not exactly represented by FP32;
this is an explicit input precondition, not a suppressed decoder failure.
The BFMA4 model validation of reachable z values lives separately in
`../math/round3_math_results.json`.

## Scale and integration limits

For a 32-element original MX block, pair positions inside that same block
and split its 16 packed terms into 8+8, or 12+4, then sum the two decoded T
values. Apply the block's common `scaleA*scaleB/4` once. The two paired
positions must share the scale product. Combining arbitrary independent
scale products into T loses information. A single packed-K16 accumulation
is outside this proof and has a carry counterexample in the math results.

This candidate still requires actual input rearrangement/packing, MMA
fragment integration, scale handling, and a **single-output** GEMM baseline.
The double-output harness's timings cannot be reused for this objective.

## Reproduce

```bash
python3 validate_decoders.py
V100_COST_TOOLCHAIN=/path/to/private/toolchain bash compile_only.sh
```

With a complete installed CUDA SDK, the core compile command is:

```bash
nvcc -O3 -std=c++17 -lineinfo -Xptxas -v \
  -gencode arch=compute_70,code=sm_70 -c middle_decode.cu -o middle_decode.o
cuobjdump --dump-sass middle_decode.o
cuobjdump --dump-resource-usage middle_decode.o
```

`compile-evidence.json` records the actual static instructions, resources,
and source/object/SASS hashes. GPU execution and timing fields remain
false/null. The private SDK and its host-glibc compatibility changes are
the same as the parent project's documented compilation setup.
