# Current production

This is the only canonical production-state page. Every value below comes from the immutable [Qwen3.8 deployment snapshot](../../_data/deployment-snapshots/2026-08-22T15-53-14Z-qwen38.json), captured at **2026-08-22 15:53:14 UTC**.

The capture is an observation, not a permanent live-state guarantee. It used only read-only health, properties, authenticated admin state, an explicit allowlist of process environment fields, and local checksums. It sent no generation request, changed no state, and restarted no process.

## Observed service

| Field | Value at capture | Snapshot provenance |
|---|---|---|
| Model slug | `qwen3.8-fastllm` | backend health, properties, admin state, allowlisted process environment |
| Public aliases | `qwen3.8-27b`, `qwen3.8-27b-q4kxl`, `qwen3.8-27b-heretic` | allowlisted process environment |
| Active profile | `q38-PROD-unsloth-ud-q6km-mtp3.env` | authenticated admin state |
| Proxy / backend ready | yes / yes | proxy and backend health |
| Lifecycle | `READY`, generation 1, not suspended | proxy and backend health |
| Accepting / active / queued | yes / 0 / 0 | backend health and admin state |

These readiness fields describe the capture instant only. Use the [proxy runbook](../runbooks/proxy.md) for a fresh read-only inspection.

## Model and template identity

| Asset | File identity | SHA-256 |
|---|---|---|
| Qwen3.8 weight | `Qwen3.8-27B-UD-Q6_K_M.gguf`, 23,088,409,504 bytes | `493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb` |
| Multimodal projector | `mmproj-F16.gguf`, 927,607,488 bytes | `cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e` |
| Chat template | `Qwen3.8-27B-UD-Q6_K_M.chat_template.jinja`, 9,993 bytes | `12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce` |

The backend reported the multimodal projector loaded and the chat template startup validation successful with no startup error. Full local paths remain in the snapshot as provenance; canonical operating instructions use repository-relative paths or explicit placeholders.

## Engine and runtime configuration

| Field | Value at capture | Snapshot provenance |
|---|---|---|
| Engine | FastLLM | backend properties |
| Engine commit / checkout state | unknown | not exposed by the allowlisted runtime sources |
| Running executable SHA-256 | `30c0da47021c8bce860a6841e3e3b7768a3cfd751bc2262b2c620646763d04d9` | local checksum of process executable |
| Activation dtype | `float16` | properties, admin state, allowlisted backend command environment |
| KV cache dtype | `turbo3` | properties, admin state, allowlisted environment |
| MTP drafts per step | 3 | allowlisted process environment |
| MTP acceptance | exact | allowlisted process environment |
| Token pool | 262,144 | properties, admin state, allowlisted backend command environment |
| Maximum batch | 4 | properties, admin state, allowlisted backend command environment |
| Backend default maximum tokens | 65,536 | properties and allowlisted backend command environment |

The allowlisted runtime sources did not expose an engine commit. The executable SHA-256 is the direct identity of the running binary recorded by this capture.

## Sampling defaults

The allowlisted process environment exposed `repeat_penalty=1.0`. It did not expose runtime values for temperature, top-p, top-k, or presence penalty, so the snapshot records those fields as **unknown**. They are not backfilled from an old README, handoff, profile, or historical report.

For the tracked adapter and model recommendation contract, read [Qwen3.8 sampling and template](../../reference/models/qwen38/sampling-and-template.md). That reference does not replace this deployment observation.

## Deployment boundary

This documentation capture did **not** deploy or restart the tracked reasoning-control compatibility change. The canonical resolver docs describe tracked source behavior; they do not prove that the observed production process is running that commit. A separate explicit deployment plus post-deploy checks is required before making that claim.

## Provenance audit

The snapshot stores provenance beside every deployment field, including source endpoint or environment key, capture policy, file byte lengths, checksums, and interpretation limits. Unknown values remain unknown by design.
