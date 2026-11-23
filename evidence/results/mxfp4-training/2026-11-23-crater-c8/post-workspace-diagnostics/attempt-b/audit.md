# Independent post-workspace diagnostics audit

Audit: **verified**.

Raw experiment status: `completed_with_blocked_diagnostics`; stage counts: `{'passed': 5, 'blocked': 1}`.

## Nsight Compute

Status: **blocked**.

Numeric requested counter rows: 0; unavailable or nonnumeric requested rows: 0.

GPU performance-counter permission was explicitly denied. No profiler metric is treated as a completed measurement.

> ==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on enabling permissions and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM

The target probe **did execute on the GPU and passed**. Its native submission counts are 1 initial, 11 after warmup, and 12 after the final ranged call. Both complete 4096² output comparisons have zero bit mismatches, all outputs are finite with canonical +0, and all six FP32/packed input SHA values remain unchanged. The two FP32 hashes also match the prior actual workspace benchmark.

Quantization and context creation are outside the requested profiler range according to the frozen probe and its execution report. Without a captured kernel trace, that scope is not independently confirmed by NCU kernel names.

The generic library_metadata helper reports its historical per-call label; actual probe source and native submission counters establish the explicit resident workspace execution. These native call counts are not hardware profiler counters.

## Compute Sanitizer

Status: **reused_passed**.

Actual checker cases: 327; reported memory errors: 0.

## Limits

- Profiler counters are diagnostic kernel-replay measurements, not unprofiled wall-time or a new throughput result.
- Unavailable, N/A and nonnumeric counter values are retained as null plus their raw text; zero is reported only when the raw metric is numeric zero.
- Counter permission denial is blocked, never a successful measurement; no retry/escalation is inferred.
- Library and installed-tool ELF bytes are not necessarily in the archive; recorded remote hashes are cross-bound to immutable prior build evidence and stage records.
- Memcheck covers the actual 327-case cache-ON preparation checker only. It does not imply racecheck, whole-training sanitization, or LLM convergence.
- The memcheck invocation belongs to immutable attempt-a and is reused once; no sanitizer GPU execution occurred in this focused profiler follow-up.
