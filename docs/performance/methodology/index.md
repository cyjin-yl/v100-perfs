# Performance methodology

Performance claims in this repository have three layers: measurement, summary, and historical benchmark. A number is only as authoritative as the layer and workload that produced it.

## Measurement

A measurement is a captured observation with enough context to reproduce or bound it. It should record:

- model and asset identity, including quantization and checksums when relevant;
- engine commit or binary identity;
- hardware and process topology;
- context length, input/output token counts, concurrency, and warm/cold state;
- sampling, template, KV dtype, MTP, cache, and scheduler settings;
- request and response hashes or equivalent integrity metadata;
- timing definitions such as TTFT, prefill rate, decode interval, and wall time;
- health/properties counters before and after when a kernel, cache, or MTP route is part of the claim;
- capture time and known confounders.

Raw logs, JSON, CSV, and benchmark output belong under `evidence/` or another manifest-classified evidence path. Evidence should not be silently edited into a cleaner story.

## Summary

A summary interprets one or more measurements. It must:

- link the exact evidence paths;
- state the workload and denominator;
- separate observation from inference;
- identify incomparable runs instead of ranking them;
- preserve failures and exclusions;
- avoid presenting a stale profile as current production.

Current Qwen3.8 summaries live under `docs/performance/qwen38/`. Their route is canonical for the document, but their individual results remain bounded by capture conditions.

## Historical benchmark

A historical benchmark is retained to explain prior engine/model choices or compare old experiments. It belongs in the archive or carries historical manifest status. It must not populate the homepage's current-state summary or override the immutable deployment snapshot.

Qwen3.5 and Qwen3.6 results often differ in model, engine, context, cache, and timing method. They are useful history, not a direct Qwen3.8 production baseline.

## Required rate calculations

State the interval used for every rate.

- Prefill rate uses counted input tokens divided by the measured prefill interval.
- Decode rate uses generated token intervals; specify whether the first token is excluded.
- End-to-end rate uses wall time and must not be labeled decode throughput.
- TTFT includes the client-to-first-content boundary selected by the harness; do not substitute a server-internal timestamp without labeling it.

If token counts come from different tokenizers or clients, report that difference.

## Cache and route attribution

A latency improvement is not a cache or kernel claim without route evidence. Capture `/props` before and after and compare the relevant counters. For prefix caching, record matched tokens/pages and the tier. For MTP, record draft/verification counters and acceptance mode. For tool-call constraints, record the grammar and masked-token counters.

Time-window log correlation alone is weak under concurrent traffic. Prefer monotonic counters attributable to the exercised path.

## Comparing runs

Treat two runs as directly comparable only when the variables that matter to the claim match. At minimum compare:

- weights and template;
- engine/binary;
- hardware and device count;
- KV dtype and token pool;
- MTP mode and acceptance;
- prompt and output shape;
- sampling;
- cache state;
- concurrency;
- measurement harness and timing definition.

If a variable differs, describe the comparison as directional or exploratory rather than a regression gate.

## Publication checklist

Before promoting a result into canonical guidance:

1. preserve the raw artifact;
2. compute and record integrity metadata;
3. link evidence from the summary;
4. label generation and capture date;
5. state whether the run was production, local non-production, or historical;
6. verify route/counter attribution;
7. keep current deployment claims in [Current production](../../operations/current-state/production.md).

The manifest-backed site exposes evidence and archive pages separately so readers can distinguish a measurement from a current operating claim.
