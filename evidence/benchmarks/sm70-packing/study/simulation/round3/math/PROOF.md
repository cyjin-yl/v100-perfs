# Round 3 packing proofs and their scope

These proofs refer to the finite FP16-input/FP32-output V100 BFMA4 model at upstream commit `6fd0520850164bdd1c37f901954e27ec63eada56`. They assume the recursive sequence of four-product groups used in our validated model. They do not assert a timing result or equivalence to an arbitrary future implementation of a larger dot product.

## 1. Asymmetric independent outputs: ternary A, full E2M1 B

Use the integer E2M1 representation `q=2*x`. Let each A lane belong to `{-1,0,1}`, each B lane belong to the full numerical alphabet `{0,±1,±2,±3,±4,±6,±8,±12}`, and K=4.

Define

\[
P_i=a_{0i}+512a_{1i},\qquad Q_i=b_{0i}+128b_{1i}.
\]

Then

\[
Z=L+128C_{01}+512C_{10}+65536H,
\]

where `L=sum(a0*b0)`, `H=sum(a1*b1)` are independent wanted outputs. All four coefficient magnitudes are at most `D=4*1*12=48`.

Every packed P is an integer of magnitude at most 513, every packed Q an integer of magnitude at most 1548. All input encodings are exact in FP16. The sum of absolute products is bounded by `4*513*1548=3,176,496<2^22`, so the V100 alignment grid and final rounding cannot discard an integer bit.

The high coefficient is recovered because

\[
|L+128C_{01}+512C_{10}|\le48(1+128+512)=30768<32768.
\]

The low coefficient is recovered by balanced modulo 128, since every other term is a multiple of 128 and `|L|≤48<64`.

For the exact integer `zi=int(Z)`:

```text
H = (zi + 32768) arithmetic_shift_right 16
L = ((zi + 64) & 127) - 64
```

Both formulas work for negative values. The strict coefficient bounds exclude half-way cases. After decoding, each output may receive its own product of A/B block scales; the four lanes need not share a scale with one another.

This changes the supported domain from the previous symmetric seven-value subset on both operands to a three-value A operand and a full 15-value E2M1 B operand. It is not full E2M1 on both sides.

## 2. Middle-only packing changes the required output

Let all `a0,a1,b0,b1` be integers in `[-3,3]`. Reverse the B packing:

\[
P_i=a_{0i}+512a_{1i},\quad Q_i=b_{1i}+512b_{0i}.
\]

The exact encoded dot is

\[
Z_{\rm exact}=G_0+512T+262144G_2,
\]

with

\[
G_0=\sum a_0b_1,\quad T=\sum(a_0b_0+a_1b_1),\quad G_2=\sum a_1b_0.
\]

Only `T` is wanted. It is the sum of two K subsequences of the **same dot product**. `G0` and `G2` are discarded. Unlike the previous independent-output scheme, losing a low bit of G0 is allowed.

### V100 error bound for compressed K=12

All packed operands are exact FP16 integers with magnitude at most 1539. The full sum of absolute products is at most

\[
B=12\cdot1539^2=28,422,252<2^{25}.
\]

At every BFMA4 stage, model output magnitude is bounded by the sum of absolute raw products processed so far: aligning each term truncates its magnitude, summing signed terms is bounded by the sum of their magnitudes, and final RZ does not increase magnitude. Hence the common alignment exponent is at most 24. Unnormalized product exponents are no larger than the exponent of the corresponding product, so that feature does not invalidate this bound.

The alignment grid is therefore at most 2. Because every input product and incoming model accumulator is an integer, each aligned term loses at most one unit in magnitude. Four products plus C lose at most five units. Final RZ loses at most one further unit, giving a conservative absolute error of at most six per BFMA4. Three groups give

\[
Z_{\rm model}=Z_{\rm exact}+e,\qquad |e|\le18.
\]

All signs are allowed: truncation direction changes with the sign, but the absolute error bound remains valid. This is a conservative bound; the measured maximum error among the recorded K=12 cases is four.

Now `|G0|≤108`, `|T|≤216`, and

\[
|G_0+e|\le126<256.
\]

Thus `round((Zmodel-262144*G2)/512)` recovers T exactly. Recovering G2 first is also safe because

\[
512\cdot216+108+18=110718<131072.
\]

For compressed K=8, the analogous conservative error bound is 12, `|G0|≤72`, and `|T|≤144`, which also satisfies the strict guards.

### A shorter signed bitfield decoder

The model output is an integer with magnitude below `2^25`, so it is exactly convertible to int32. Let `zi=int(Zmodel)`. Because `-256<G0+e<256`,

\[
\left\lfloor\frac{zi+256}{512}\right\rfloor=T+512G_2.
\]

Since `-256<T<256`, the low nine bits of this integer, interpreted as a signed nine-bit number, are exactly T. Therefore:

```text
u = int32(Zmodel) + 256
T = signed_bitfield_extract(u, offset=9, width=9)
```

Equivalently, in explicit unsigned-shift notation, extract bits 9…17 and sign-extend bit 17. Arithmetic right shift or signed `bfe.s32` is required where applicable. The strict guard excludes ties; negative values follow the same floor identity. The addition cannot overflow int32.

This decoder does not need to compute G2. Returning a physical FP32 value still requires conversion and scaling; packing, scheduling and memory costs are separate.

## 3. Correct MX32 pairing and operation counts

Within an original length-32 block sharing one scale product, pair adjacent entries along K. This creates 16 compressed entries. A single compressed-K16 decode is unsafe for the seven-value q domain.

The implemented model pipeline uses compressed `8+8`: each eight-entry section uses two BFMA4 operations followed by one middle decode. The original block therefore uses **four BFMA4 groups and two decodes**, compared with eight ordinary BFMA4 groups. Compressed `12+4` is another permissible partition, also requiring two decodes. These are arithmetic counts, not a speed prediction.

Pairing entries from different scale products is generally invalid. For example, `(a0,a1,b0,b1)=(1,0,1,0)` and `(0,1,0,1)` both encode to 512 and have T=1, but scale products 2 and 4 give physical E2M1 outputs 0.5 and 1. The encoded result cannot distinguish those differently weighted answers.

## 4. Counterexamples delimit the claims

- Asymmetric K=8: `a0=1`, `a1=0`; either `b0=12,b1=0` for all eight terms, or `b0=-4` for all terms and only the first b1=1. Both packed results are 96, but wanted L is 96 or -32.
- Signed INT2 A / full E2M1 B at K=4: a0=`[-2,-2,-2,-1]`, a1=0; suitable B inputs give the same encoded 64 for wanted L=64 and L=-64. Increasing A's permitted range cannot be assumed free.
- Middle-only compressed K=16: all four lanes equal 3 produces desired T=288 outside the signed nine-bit safe interval. The decoder returns a different value.

These falsify particular extensions of the tested families. They do not establish universal impossibility of other encodings, corrections, approximate schemes or arithmetic architectures.
