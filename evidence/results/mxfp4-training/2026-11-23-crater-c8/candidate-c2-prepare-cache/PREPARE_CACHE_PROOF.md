# Opt-in C2 row preparation cache candidate

This is an unmeasured candidate based on source commit
`0ff69f8e9a9522aced2f3be887a71ba318fdb867`. It changes only
`csrc/mxfp4_gemm_certified.cu` and `build_cuda.py`. The existing
`prepare_rows<false>` / `<true>` kernels and their helpers are unchanged.
There is no change to the public C ABI, certificate, cuBLAS algorithm,
normalization decoder, postscale, exact tiled fallback, or default variant 2.

## Explicit selection and evidence

Build with `--with-certified --certified-variant 2 --certified-prepare-raw-cache`.
The new flag defaults off. When present, it defines
`MXFP4_CERTIFIED_PREPARE_RAW_CACHE=1`; the source rejects values other than
0/1 and rejects enabling it for variants other than 2. The compilation
manifest records the flag, backend `sm70_c2_row_raw_cache_k8192_v1`, maximum
K=8192, and 20 dynamic shared bytes per MX32. Default-off compiler arguments
do not receive the new macro.

Public shape validation still establishes K>=0 and K%32==0. The shared host
helper `use_c2_prepare_raw_cache(k)` selects the new kernel only for
0<K<=8192. K=0 (including null input pointers) and K>8192 execute the old
prepare kernel. Context allocations and workspace byte accounting do not
change: this cache is CTA shared memory, not a new persistent allocation.

## Data layout and synchronization

Each 128-thread CTA owns one input row. For B, the local row remains
`blockIdx.x-M`. Input and output row offsets still use size_t and the existing
shape validator. There are at most 256 MX32 blocks; each thread handles at
most two blocks in the first and energy loops.

Dynamic shared memory is a 16-byte-aligned `uint4 raw_blocks[blocks]`
followed by `unsigned energy_scale[blocks]`. Every block has exactly one
writer. A source row aligned to 16 bytes uses one uint4 load per block;
otherwise four explicit safe byte-assembled words reconstruct its 16 bytes.
No access before or beyond the block is needed for offsets 1..15.
The metadata word is `(scale << 16) | Q`, where Q is the exact squared
integer-code energy and is at most 4608; the fields cannot overlap.

The normalized output still uses four consecutive half values per thread
and aligned uint2 stores, maintaining the original coalesced output layout.
Group g reads `raw_words[g/2]`, selecting its low or high 16 bits by g%2;
this is exactly source byte offset 2*g. Its metadata block is g/8.
For every valid K, the group loop writes each of the K half values once.

There are the same three CTA barriers as the original prepare kernel:
the first publishes raw cache, metadata cache, and warp min/max summaries;
the second publishes the final row minimum/span/NaN/active state;
the third publishes per-warp saturated energies. No thread exits early.
The cache is written once and read only afterwards.

## Row metadata equivalence

For each MX32 block, `Q = sum(q_i*q_i)` with q in
{0, +/-1, +/-2, +/-3, +/-4, +/-6, +/-8, +/-12}. Q is zero exactly when all
nibble magnitudes are zero, including either sign of zero. Therefore the
row minimum, maximum, active flag, and fallback minimum 127 are identical
to the old block_has_nonzero_q path. Scale 255 is counted before zero
exclusion, so even an entirely zero block with that scale still rejects.

For a finite active block, delta=scale-row_min is nonnegative. An entirely
zero block may have scale below row_min and is excluded before forming
delta or shifting. Scale-255 blocks are also excluded from energy, matching
the original implementation while preserving their separate rejection.

The old C2 energy sums eight terms per block, one per four-nibble group:
`Q_group << (2*delta)`. The new kernel sums one term:
`Q_block << (2*delta)`. For delta<=24, Q_block<=4608 implies this shift is
less than 2^61, so uint64 arithmetic cannot overflow before saturation.
For delta>24, any nonzero term is at least 2^50, above the unchanged 2^48
certificate bound, so both paths substitute ENERGY_CAP=2^48+1.

For nonnegative terms, capped addition is associative:
`min(cap, min(cap,a+b)+c) = min(cap,a+b+c)`.
Thus regrouping the eight group energies into one block energy, and
redistributing terms across threads/warps, preserves the exact final
saturated row energy. RowData fields and certificate finalization are
unchanged. The certificate is neither weakened nor bypassed.

## Normalized half equivalence

The same row-level condition `row_nan==0 && row_span<=12` controls both
implementations. Invalid rows emit finite positive zeros in the complete
normalized buffer before cuBLAS, which is important because the exact
fallback follows cuBLAS rather than suppressing that call.

For a safe row, a nonzero four-nibble subgroup implies an active finite
MX32 block. Hence its delta is in 0..12. The candidate calls the unchanged
`normalized_half4_bits` on the exact same two packed bytes and delta.
Zero subgroups directly emit uint2(0,0), and the decoder masks individual
positive/negative zero nibbles to canonical positive half zero. Every
normalized half bit therefore matches the old C2 preparation, not just its
floating-point value. The cuBLAS inputs, accumulated order, and final
canonical FP32 zero behavior are unchanged.

## Resource tradeoff and measurement gate

At K=2048/4096/8192, dynamic shared memory is respectively 1280/2560/5120
bytes per row CTA, in addition to the same small static reduction arrays.
The register count and occupancy must be read from the actual ptxas output.
The source introduces no extra CTA barrier. It removes the second global
packed/scale read and reduces 64-bit energy shifts/saturating additions
from eight per MX32 to one. It adds a shared raw store/read, a metadata
store/read, and a separate block-energy loop. Short-K underutilization and
the added shared instructions can offset its savings; no performance
benefit is assumed.

The shared source mapping has no distinct-address collision in the usual
transaction-sized phase: contiguous uint4 stores cover consecutive words;
normalization word loads have two adjacent threads read the same word;
metadata loads share one word among each eight adjacent threads. This is
an address-level model, not an observed bank-conflict or throughput metric.

Required actual-GPU acceptance is the unchanged 69-case certified suite
plus the independent checker comparing actual new/old preparation kernels
for every normalized half bit and every RowData field, including negative
zero, scale 255, span and energy rejection, odd input offsets, K=8192, and
public K=0/8224 fallback. Performance must compare flag-off/on C2 on the
same source/device, inputs, clocks, warmup and serialized timing protocol.
The packed full-chain comparison remains the decision metric; prepare
stage timings, if collected, are diagnostic rather than substituted for it.
