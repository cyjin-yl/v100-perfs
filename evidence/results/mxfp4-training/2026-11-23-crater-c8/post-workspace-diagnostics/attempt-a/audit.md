# Independent post-workspace diagnostics audit

Audit: **verified**.

Raw experiment status: `failed`; stage counts: `{'passed': 5, 'failed': 1, 'blocked': 1}`.

## Nsight Compute

Status: **blocked**.

Numeric requested counter rows: 0; unavailable or nonnumeric requested rows: 0.

## Compute Sanitizer

Status: **passed**.

Actual checker cases: 327; reported memory errors: 0.

## Limits

- Profiler counters are diagnostic kernel-replay measurements, not unprofiled wall-time or a new throughput result.
- Unavailable, N/A and nonnumeric counter values are retained as null plus their raw text; zero is reported only when the raw metric is numeric zero.
- Counter permission denial is blocked, never a successful measurement; no retry/escalation is inferred.
- Library and installed-tool ELF bytes are not necessarily in the archive; recorded remote hashes are cross-bound to immutable prior build evidence and stage records.
- Memcheck covers the actual 327-case cache-ON preparation checker only. It does not imply racecheck, whole-training sanitization, or LLM convergence.
