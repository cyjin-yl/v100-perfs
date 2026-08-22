# Getting started

This repository is an auditable operating record and documentation source for a FastLLM inference stack built around a 32 GB NVIDIA V100. Begin with the canonical contract, then descend into evidence or preserved history only when you need the underlying observation.

## What is supported

The tracked stack contains:

- an OpenAI- and Anthropic-compatible gateway in `src/gateway/`;
- a framework-independent reasoning-control resolver shared by both protocol adapters;
- `fastllm_adapter.py`, which prepares canonical requests for FastLLM;
- Qwen3.8 model, sampling, template, and performance references;
- deployment profiles, launch material, diagnostics, and handoff artifacts under `operations/`;
- evidence and benchmark artifacts that remain distinct from narrative summaries;
- a manifest-driven Astro site that exposes canonical docs, evidence, archive pages, and every final Git-index path.

The exact running production combination is deliberately not repeated here. Read [Current production](../operations/current-state/production.md), which names its immutable capture time and field provenance.

## What is not promised

- The repository does not claim that a tracked gateway change has been deployed.
- A reasoning budget is not a hard token cap. It is a compatibility hint used to choose a template tier when no named effort is present.
- Historical Qwen3.5, Qwen3.6, benchmark, incident, and handoff documents are not current authority.
- Research proposals are not production commitments.
- The static documentation site does not poll, mutate, suspend, resume, or restart a production service.

## Read the system before operating it

1. Follow the [system architecture](../concepts/architecture/system-overview.md).
2. Learn the [reasoning control model](../concepts/reasoning/control-model.md).
3. Compare the [current deployment snapshot](../operations/current-state/production.md) with the tracked contract.
4. Use the [proxy runbook](../operations/runbooks/proxy.md) for read-only inspection and explicit deployment boundaries.
5. Use [performance methodology](../performance/methodology/index.md) before quoting a result.

## Local documentation verification

From the repository root:

```bash
node site/scripts/generate-repo-index.mjs
node site/scripts/validate-content-manifest.mjs
```

The generator reads the final Git index, not an untracked working directory. The validator checks unique routes, full tracked-path coverage, baseline joins, authority metadata, and generated-file consistency.

Site compilation and browser checks are separate gates. A successful local static build proves that artifacts were generated; it does not prove deployed HTTPS status or production inference behavior.

## Next steps

- OpenWebUI users: [configure per-chat reasoning controls](../guides/openwebui/reasoning-controls.md).
- Client authors: [choose protocol fields](../guides/client-compatibility/reasoning-fields.md).
- Operators: [inspect the proxy boundary](../operations/runbooks/proxy.md).
- Reviewers: [verify the exact resolver contract](../reference/api/reasoning-controls.md).
