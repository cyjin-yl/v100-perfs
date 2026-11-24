# V100 full-K tile certificates and paired RNE experiment

## Current execution status

The Crater account was reauthenticated successfully on 2026-11-24. A fresh
one-V100 job, `sg-jincy26-261124-ae156`, was submitted at 12:59 UTC with 4 CPUs
and 16 GiB host memory. The latest saved observation is **Pending / Inqueue**;
no new CUDA compilation, GPU correctness result, or performance result has
been observed for this experiment yet.

The owned persistent watcher is running on doesvm. Its frozen source and
payload were published at 13:13 UTC. It will wait for the assigned card,
validate the owned endpoint and authenticated SSH host key, and execute the
experiment once. A source-ready publication is not an execution result.

The container bootstrap has a four-hour lifetime starting with container
execution. The watcher allows up to twelve hours for the queue; the GPU
experiment itself is bounded to 3000 seconds, including cleanup and evidence
collection. No previous jobs, other users' processes, clocks, or power settings
were changed.

## Frozen candidates

Source commit: `a3e65b5f7dd68f85233a107bda0d2acadf0b3c7f`.

Source archive SHA256:
`05cbaecbea4dbeadfa6d1aa63336268d983d9e5ed8bcbf0e30567eab4ae74a19`.

Payload SHA256:
`d67ed199ad795f8009b55fd876be42ca7569c72e498d3d09f131959c16b7e911`.

**C11** applies the existing full-K exactness certificate to each 128×128
output tile after a global rejection. A passing tile uses the normalized
Tensor Core path; a rejecting tile uses the existing ordered-MX32 fallback.
It does not split K, alter the floating-point accumulation contract, cache
decisions across mutations, or change the meaning of the original global
certificate diagnostics. The original 128-byte diagnostic ABI is preserved;
tile diagnostics have a separate 72-byte optional ABI.

**Paired CUDA RNE** exposes `quantize_pair_mx_cuda_rne(a, b)`. It invokes the
existing native quantizer for both operands, keeps two independent finite
flags, and reads both flags with one host wait. Both operands are validated
before submission, and either nonfinite operand rejects the whole pair.
The original single-input API, Torch quantizer, stochastic rounding, and
training defaults are unchanged.

Neither candidate is enabled by default. The default certified variant
remains C2. Workspace reuse is an explicit caller API and is not integrated
into autograd or an LLM training loop by this change.

## Validation and measurement plan

The original 24 CPU reference tests passed locally. Independent local review
covered the native dispatch/mask mapping, same-stream updates, error cleanup,
and paired flag lifetime. These checks do not substitute for NVCC and V100
execution.

The frozen payload has 55 planned stages:

| Stages | Count | Scope |
| --- | ---: | --- |
| Builds | 3 | CUDA 12.8; C2, C10, C11; native RNE and all native harnesses |
| Correctness | 10 | Native full-domain checks, independent tile checks, paired quantizer and workspace checks |
| Packed tile benchmarks | 26 | Matched 4096³ block/sparse-span/sparse-energy and 8192³ unit/block inputs |
| Quantizer benchmarks | 8 | 1024² and 4096²; singles/pair/pair/singles |
| Quantizer plus workspace | 8 | Same ABBA order, including repeated quantization and the full C2 wrapper |

All correctness stages must pass before performance is measured. The tile
benchmark compares the actual first timed probe and final timed output in
full against a GPU ordered-MX32 oracle, independently checked against NumPy
on small complete cases and sampled large outputs. The first single-call
probe is reported separately from repeated-batch medians. The workspace
benchmark retains the actual first and last outputs from its timed batch and
compares every packed byte and every finite FP32 output bit; NaN payload bits
are intentionally not contractual.

Each comparison uses identical device-input hashes. Timing includes all
operations declared by the corresponding harness; packed-only numbers must
not be relabeled as quantization-plus-GEMM or training throughput. Useful
GEMM FLOPs are `2*M*N*logical_K`. Context construction and destruction are
reported separately when excluded from repeated-call timing.

GPU UUID, process lists, and idle boundaries are checked around every
benchmark. These samples detect observed interference but do not prove
continuous exclusivity between samples. CUDA compilation logs, source and
binary hashes, exact diagnostic decisions, failure logs, and timed-output
checks are collected even when the run stops at a failed prerequisite.

## CPU certificate predictions only

For the new deterministic synthetic input seed `20261124`:

| Input | C11 fast tiles | C11 fallback tiles |
| --- | ---: | ---: |
| 4096³ block | 1024 | 0 |
| 4096³ sparse span | 961 | 63 |
| 4096³ sparse energy | 961 | 63 |
| 8192³ unit | 4096 | 0 |
| 8192³ block | 0 | 4096 |

These predictions motivate testing selective rejection. They are not GPU
performance measurements, training traces, or evidence that C11 improves the
already accepted C2 path. The all-rejected 8192³ block case remains in the
experiment as an explicit limit and regression control.

## Reproducible records

- [Submission and resource observation](queue/created-observation.json)
- [Source-ready publication](queue/source-ready.json)
- [Watcher launch](queue/watch-launch.json)
- [Saved queue observation](queue/queue-observation.json)
- [Bounded bootstrap](queue/bootstrap.sh)
- [One-shot submission helper](queue/submit.py)
- [Frozen GPU experiment payload](queue/remote_run.py)

The source-ready hash and the payload hash bind this particular run. Any
subsequent candidate requires a new frozen publication and new evidence.
