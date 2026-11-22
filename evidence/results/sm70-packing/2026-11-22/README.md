# SM70 packing evidence — 2026-11-22

These are the exact two previously delivered experiment bundles. Their contents are preserved byte for byte; `archives.json` records their sizes and SHA-256 values. They contain CPU arithmetic experiments, a pinned Soft-BF16-V100 audit, the pending V100 hardware probe, the validated V100 numerical model, CUDA compilation evidence, and the Accel-Sim experiment record.

| Archive | Scope | New V100 execution |
|---|---|---|
| `v100-fp16-packing-study-20261122.zip` | Initial mathematical search, source audit, CPU reproductions, and CUDA probe | No |
| `v100-sm70-simulation-20261122.zip` | Original/ported V100 models, public hardware-fixture revalidation, SM70 compilation, and cycle simulation | No |

Use the [current research report](../../../../docs/research/performance/sm70-packing/index.md) for later findings and interpretation. The [readable study](../../../benchmarks/sm70-packing/study/) contains the runnable sources and compact evidence. The 10 MB structural-search JSON remains inside the immutable second archive and can be regenerated from its script.

Published hardware fixtures are measurements supplied by the upstream numerical-model authors. They are not fresh measurements on the user's GPU. The Quadro GV100 cycle model is not a calibrated measurement of the user's Tesla V100. No training speed or convergence is claimed.

The [integrated validation](integrated-validation/) directory records the actual 16-step isolated replay after the new sources were placed in their final repository paths. It is separate from the two immutable historical ZIPs.
