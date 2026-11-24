# Explicit certified workspace candidate

This candidate adds three Python modules to base commit
`f338faaa5cbdcd900aa82ae95af00ed5fb770235`. It does not edit existing sources,
the native ABI, quantization, certificate limits, CUDA kernels, default
`gemm_packed` behavior, `MXRecipe`, or autograd. It has passed AST parsing and
`git apply --check`; no GPU result is claimed by this candidate handoff.

## API and ownership argument

```python
from tools.mxfp4_training.certified_workspace import CertifiedWorkspace

with CertifiedWorkspace(m, n, logical_k, device="cuda:0") as workspace:
    output = workspace.gemm(packed_a, packed_b)
    # Repeat on this same host thread; changing operand bytes is supported.
    diagnostic = workspace.info()  # Synchronizes; outside measured work.
```

Construction binds one positive logical shape, an explicit CUDA device index,
and the resolved shared-library path. These bindings have read-only public
properties. Entering allocates exactly one native context and checks its ABI,
device, padded shape, zero submitted calls and invalid initial metadata. The
native implementation still owns and sizes its fixed workspace. No registry,
global context cache, weak reference, implicit finalizer or `__del__` exists.
The existing library loader cache contains shared-library handles only.

The entering Python `Thread` object is retained as the owner. Every `gemm`,
`info`, and `close` checks that exact object before any CUDA operation. This
also rejects another thread whose numeric thread identifier happens to be
reused. Native C++ thread checks remain in force. Each native operation runs
under the bound PyTorch device guard; wrong-device operands are rejected.
Entering on an SM other than 70 is rejected. Users must use the context
manager or explicitly close on the entering thread.

The wrapper preserves the native stream rule. Repeated submissions on the
same current stream are ordered by CUDA. Switching current streams performs
a nonblocking query of the previous stream and rejects a busy stream before
output allocation or native submission. After the previous stream completes,
the switch is permitted and remains subject to the native `cudaStreamQuery`
guard. This does not manufacture input-data dependencies: callers remain
responsible for tensors produced on other streams.

All five tensors—both code buffers, both scale buffers, and the new FP32
output—are registered with `record_stream` before native submission. The
Python locals retain them through the native call and through cleanup if a
call fails. The remembered stream object also remains alive until close.
Repeated asynchronous calls therefore preserve PyTorch allocator lifetimes
without retaining every historical operand in an unbounded Python list.

GEMM uses the existing packed validation, checks the exact fixed logical shape
and device, and then calls the unchanged certified C API. The C API recomputes
preparation and its certificate for every call. A successful wrapper call
increments its expected native count. Requested diagnostics validate the ABI,
shape, device, metadata-valid bit and cumulative submission count.

Preflight shape, device, owner, padding, diagnostic-dictionary, capture and
busy-stream errors do not consume the active context. An exception following
an attempted native submission or diagnostic call closes synchronously on
the owner thread before propagation; this handles partial enqueue failures.
Constructor diagnostic failure likewise rolls back a real created context.
Context-body exceptions invoke the same cleanup. Destruction can report an
error after deleting the native object; the wrapper never retries that
consumed pointer. Repeated `close` is idempotent and reuse after close is
rejected. No context can be reentered.

Capture detection occurs before padding validation, allocation, submission,
diagnostic synchronization, or destruction. An attempted close during capture
leaves the context active so it can be closed after capture ends. The checker
catches these expected errors inside a valid graph capture and verifies that
the context still works afterward.

## Independent real-device checker

`check_certified_workspace.py` requires a successful native compile manifest,
the actual library hash, every current native-source hash, and SM70 V100
identity. It reports blocked rather than passing when prerequisites are
missing. The existing per-call checker and native suites remain separate.

Its eleven mandatory cases are:

1. Explicit lifecycle, fixed public bindings, initial ABI, first output,
   idempotent close, use-after-close and reentry rejection.
2. Logical K=33 versus native K=64, full output and invalid padding rejection.
3. An error after actual create and actual initial-info calls; destroy exactly
   once, with the context marked closed.
4. Five submissions mutating the same GPU input buffers: ordinary values,
   changed scale, signed zero, scale-span fallback and NaN-scale fallback.
   Every output is compared to independent NumPy MX32 arithmetic, and every
   count and certificate transition is checked. NaN payload bits are excluded
   only for the NaN case; all output zeros must be canonical positive zero.
5. Wrong shape, CPU operand device and nonempty diagnostics, followed by
   successful recovery without an intervening submission.
6. A different host thread attempts GEMM, diagnostics and close. All must fail
   before touching the native context; owner-thread use afterward must work.
7. A real pending stream (bounded CUDA sleep), rejected switch without a wait,
   then a completed-stream switch with exact output.
8. Several queued calls with temporary operand references released; explicit
   close must finish the stream, and retained GPU consumers must be correct.
9. A controlled error returned only after a real native GEMM has enqueued;
   destruction must occur once and wait for actual work.
10. A context-body exception after real submission; cleanup must complete.
11. GEMM and close rejected during valid CUDA Graph capture, followed by
    successful ordinary execution and normal destruction.

A second-GPU operand test executes only when another CUDA device is visible;
otherwise the absence is explicitly reported. CPU wrong-device rejection
still executes on a one-GPU node. The checker is not an autograd integration
test and does not establish LLM convergence.

## Benchmark scope

`bench_certified_workspace.py` requires the passing workspace report, matching
sources/library/runtime, and for native RNE the independent quantizer report.
It supports separate-process `per_call` and `workspace` lifecycles with the
same seed and shape. Root should use ABBA order per shape (1024 and 4096).
Default warmup is 10 calls followed by five rounds of ten calls; raw event and
synchronized wall times are retained. Synchronized wall median is primary.

Every measured operation quantizes both resident original FP32 operands,
allocates packed tensors and output, validates inputs, calls Python/ctypes,
prepares and certifies data, performs GEMM, and applies scale/fallback. The
per-call arm additionally creates and destroys its native context inside
every measured operation. The explicit-workspace arm creates once; object
construction, create and initial ABI query are timed separately, as is close.
Those one-time costs are not disguised as repeated-call GEMM work. Amortized
create/close cost is reported with its exact denominator of all successful
calls, including validation and warmup; it is not a training-step rate.

Actual FP32 input bytes are hashed before and after timing. The untimed first
result and actual last timed result are compared over all FP32 output bits
against the tiled backend, whose arithmetic is independently checked by the
existing suites. Zeros must be positive and all Gaussian outputs finite.
For reuse, the final diagnostic is read from the actual last timed submission
and must show count `1 + warmup + rounds * iters`, accepted=1 and flags=0.
The per-call arm cannot query its already-destroyed final context without
changing the timed path; this limitation is explicit in its report. Its full
actual final output still receives the same bit comparison.

Input generation, correctness work, diagnostics, and library/event
initialization are outside measured repeated calls. Real quantizer validation
synchronization remains inside. No full training-step or autograd speedup is
claimed. Autograd can execute backward on a different host thread; this
candidate deliberately requires explicit direct packed-GEMM use on the
owner thread until a separate lifetime/worker integration is established.

## Commands after application

Use the same frozen source snapshot and compiled library as the parent run:

```bash
python -m tools.mxfp4_training.check_certified_workspace \
  --compile-evidence /path/compile-evidence.json \
  --expected-certified-variant 2 --output /new/workspace-check.json

python -m tools.mxfp4_training.bench_certified_workspace \
  --compile-evidence /path/compile-evidence.json \
  --check-report /new/workspace-check.json \
  --quantizer-check-report /path/quantizer-check.json \
  --lifecycle workspace --quantizer-backend cuda_rne \
  --m 4096 --n 4096 --k 4096 --warmup 10 --rounds 5 --iters 10 \
  --output /new/workspace-4096.json
```

Set `MXFP4_CUDA_LIBRARY` to that exact manifest-bound library. Use new output
paths and `--lifecycle per_call` for the control arm. The candidate-manifest
file binds the three new modules and patch by SHA-256.
