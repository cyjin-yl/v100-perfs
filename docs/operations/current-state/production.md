# Current production

This is the only canonical production-state page. The October11 restoration below supersedes the August22 capture as current deployment authority; the historical snapshot remains unchanged.

## 2026-10-11: doescompute production restored

Production was restored under the user's explicit production-first acceptance decision: the unreproduced historical private repetition incident used a different compute path and is no longer a restoration blocker. This is authorization to operate the admitted profile, not an exact replay, causal cure, fair engine winner or broader quality certification. The [production restoration receipt](../../../evidence/results/1cat-v100-2026-10-09/2026-10-11-production-restoration.json) records the handoff; the prior [c39 admission](../../../evidence/results/1cat-v100-2026-10-09/2026-10-11-current-main-c39-admission.json) and [local-repair patch](../../../operations/patches/1cat-vllm-c39f53ab-local-repairs.patch) retain the source/native provenance.

| Field | October11 production observation |
|---|---|
| Engine / source | 1Cat-vLLM, `c39f53abae51df6bf8ef122e34975a9f3596a93c` plus admitted local operand/FP8 namespace, real-head, quantized-growth and Responses continuation repairs; upstream serializer preserved |
| Matching native identity | 21 matching c39 artifacts, policy ABI67; no native-build or runtime admission transferred to fetched `8d4fd9401ce0fd713821be0f045dc1f1f718b57b` |
| Target / draft | W4A16 target revision `3e644f233eb7ebb294a5128d75ca79143987f661`, FP8 DFlash2, TP2, q7 |
| Runtime profile | Half activation dtype; target KV `fp8_e4m3`; four sequences; utilization0.90; batch8192; fair-prefill2048; mixed minimum512 tokens and250ms latency target |
| Engine / client budgets | Engine maximum model length262144; existing client context131072 and maximum output32768 unchanged |
| Engine endpoint / alias | `http://10.10.10.11:8000`; `swift-qwen3.8-27b-nvfp4` |
| LAN / Tailnet API | `http://10.10.10.12:18080/v1`; `https://doesvm.icefish-exponential.ts.net:8446/v1` |
| Identity / readiness | Tag `production-c39-w4-fp8-min512-20261011T0450Z`; identity verified, PID2749169/starttime84525844, health ready at2026-10-11 05:01:48 UTC |
| Startup | 426.972s; permanent launcher `/tank/hot/1cat-deploy/production-c39f53ab/launch.py`; observers and development mode disabled |
| Gateway handoff | LAN/Tailnet ready and models HTTP200; actual ordinary/dense/schema Responses generation3/3 passed, inputs26/26/28 and outputs3/3/12 |
| Admission / cancellation | Four streams HTTP200 with actual content; fifth HTTP429 `kv_slots_exhausted`; all four controlled streams cancelled, capacity4/in-use0/background-in-use0; subsequent Tailnet chat HTTP200 with correct answer |
| Existing contracts | Original authentication, gateway routing/upstream and client budgets unchanged; engine authentication, argv/environment and raw logs are not exported |
| Rollback provenance | Permanent launcher and prepared identity-guarded rollback provenance are retained with the restoration decision; private receipts are `production-c39f53ab/restoration-decision.json` and `production-c39f53ab/gateway-handoff.json` under `/tank/hot/1cat-deploy/` |

The existing c39 measured lower bound remains four130560-input+512-output windows, not a stable maximum or proof of full32768-token generation. Historical failures remain evidence; user permission does not convert them into successes. The newly completed production C1/C4 speed captures are in the [phase-separated performance section](../../performance/qwen38/engine-quantization-comparison-2026-10-09.md#2026-10-11-production-prefill-and-decode-speed) and restoration receipt; no older rate is substituted for them.

Life resumed on the same pane `%6`, PID26007, session prefix `01a0c738`. The temporary benchmark hold was released. After closing the native context inspector, the continuation was actually admitted at05:20:40 UTC and a successful assistant/tool-use event followed at05:21:12 UTC. Its current `yvxi/gpt-6.1-sol` model was preserved, not silently switched to Qwen; Qwen readiness is independently proved through the production gateway. The earlier aborted event remains recorded and is not counted as a successful response.

## Historical snapshot — 2026-08-22 FastLLM

Every value in the remaining August sections comes from the immutable [Qwen3.8 deployment snapshot](../../_data/deployment-snapshots/2026-08-22T15-53-14Z-qwen38.json), captured at **2026-08-22 15:53:14 UTC**. These sections describe that checkpoint only, not the October production engine. All snapshot identities, unknown fields and provenance are retained.

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
