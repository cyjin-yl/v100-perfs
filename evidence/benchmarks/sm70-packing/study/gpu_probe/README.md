# V100 FP16 packing / MXFP4-subset hardware probe

**Status: compiled for sm_70, but NOT executed on a GPU.**
The preparation container has no CUDA device. A private CUDA 12.8 toolchain was
downloaded and verified against NVIDIA's hashes; object compilation succeeded,
and disassembly contains 96 HMMA instruction lines. This was an object-file
compile, not a linked executable or a hardware run. Four unused sinpi/cospi
declarations in the private SDK needed a glibc 2.41 compatibility adjustment;
the patch and all compile receipts are retained in `../compile-evidence/`.
No system headers or user-machine environment were changed. A successful CPU
experiment or compiler check must not be described as a V100 result. Run the
program on the actual card and retain its output before claiming hardware
correctness or speedup.

This is an exploratory arithmetic and performance harness, **not a training
library**, a complete MXFP4 implementation, or a drop-in replacement for the
Soft-BF16-V100 repository.

## Run on an existing CUDA 12.x environment

```bash
chmod +x run.sh
./run.sh --device 0
```

Useful alternatives:

```bash
./run.sh --device 1 --correctness-only
./run.sh --device 0 --benchmark-only --warmup 10 --iters 100
```

The script compiles `sm_70` machine code and PTX, checks for HMMA instructions
when `cuobjdump` is installed, records GPU identity and temperatures, and writes
JSONL output, compiler information and hashes. It does not install packages,
stop services, change clocks, reserve both cards, or launch a training job.
The default benchmark matrices are at most 1024×512 output values, with a few
MiB of live data. Use an idle GPU for interpretable timing.

The binary checks the actual device name and compute capability. A run on a
different GPU requires `--allow-other-gpu`, and its `actual_V100` field remains
false. CUDA 13 removed the compiler target used here; use an existing CUDA 12.x
development environment rather than silently compiling for a newer GPU.

## What the arithmetic cases test

We express E2M1 values as integers `q = 2*x`. Full E2M1 has the distinct numeric
alphabet:

```text
{0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}
```

Positive and negative zero have the same arithmetic value. This probe ignores
zero's sign. It does not cover NaN scales, scale overflow, quantizer rounding,
or distinct MXFP4 scale factors. All external scales are one.

### A constructive shared-right-hand-side case

For `q ∈ {-3,-2,-1,0,1,2,3}` (physical E2M1 magnitudes no greater than 1.5),
pack two rows as:

```text
Apacked = A0 + 512*A1
Z = Apacked * B
C1 = round(Z / 512)
C0 = Z - 512*C1
```

For a reduction of at most 28 elements, the low dot is bounded by `9*K < 256`.
Every operand is an exactly representable FP16 integer; the intermediate
integer sums in the 16-element probe fit well inside FP32's exact integer
range. WMMA works in 16-element chunks here. The program tests K=16 and also
K=32,64,128,512 to expose carry outside the supported direct-decoding region.
Physical products are `C0/4` and `C1/4`.

The `wmma_decode_chunks` variant decodes after every 16 terms and then adds
each logical output separately. This gives a candidate route to long K, but
the decode instructions, tiled accumulation, and additional data handling
all cost time. Its intentionally simple WMMA kernel is not a competitive
implementation of GEMM.

A second positive control uses ternary q, radix 256 and K=64.

### A constructive two-sided case

For ternary values, radix 128 and K=16, both operands can be packed:

```text
(A0 + 128*A1) * (B0 + 128*B1)
  = L + 128*X + 16384*H
```

`L=A0*B0`, `H=A1*B1`, and `X` is the sum of the two cross terms.
The bounds `|L|≤16`, `|X|≤32`, `|H|≤16` leave space for all three
digits. A second positive case uses radix 256 and K=32, matching the MX
32-element block length: `|L|≤32`, `|X|≤64`, `|H|≤32`, and the maximum
integer accumulator magnitude is `32*257^2=2,113,568`, below `2^24`.
The program checks both target products and the cross term against
exact integer references on random and extreme inputs. The exhaustive
single-product two-sided tests then scan wider alphabets and radices. A
failed balanced-radix decoder is evidence about that decoder, not proof that
every possible encoding fails.

### An exact collision for full E2M1 at radix 128

The same B=[8,1] gives:

| A0 | A1 | Desired pair of integer dot products | Packed dot |
|---|---|---|---|
| [8,0] | [0,0] | (64,0) | 64 |
| [-8,0] | [0,1] | (-64,1) | 64 |

Every input and calculation is exact. Zero padding embeds this two-term
collision in an actual WMMA operation. Therefore Z alone cannot identify the
two outputs for this particular encoding. Extra side information, different
encoding families and additional operations are outside that result.

## Fair performance comparison

Both benchmark paths produce two M×N matrices with a shared B.

- Baseline: one cuBLAS GEMM with **2M rows**, rather than two artificially
  separated small calls.
- Candidate: pack A, one M-row GEMM, and unpack both outputs. Both pure GEMM
  time and **pack + GEMM + unpack** time are reported.
- Long-K exploratory candidate: packing followed by naive WMMA with decode
  after each 16 terms; baseline remains the stacked cuBLAS GEMM.

The program compares every output to cuBLAS and additionally checks 32
cuBLAS output positions against an exact integer reference. Any mismatch
invalidates acceleration claims and makes the process fail. The correctness
suite deliberately includes failing decoders, so its failure counts by
themselves are expected; the named constructive examples must pass.

Reported useful TFLOP/s counts the same `4*M*N*K` mathematical operations for
both outputs. It is an **effective workload rate**, not a claim that the chip
acquires native FP4 arithmetic units. Timing uses CUDA events, warmup and
synchronization. Default runs are short; repeat on an otherwise idle device
and compare distributions before making a performance claim. The script
does not use a profiler to prove which algorithm cuBLAS selected.

None of these measurements include MX quantization, training backward passes,
optimizer updates, global gradient statistics, convergence, or block-scale
handling. A positive microbenchmark would not establish a training speedup.

## Evidence files

`results/*-results.jsonl` contains machine-readable records. The first record
identifies the actual device; the final `completed` record appears only after
all selected tests finish. `results/*-sha256.txt` binds output to the source
and binary. `build/sass.txt`, when present, provides HMMA inspection evidence.
Keep these files with the source when reporting results.
