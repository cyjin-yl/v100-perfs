# V100 FastLLM inference repository

This repository documents and verifies a FastLLM inference stack for NVIDIA V100 hardware. It contains the gateway and adapter boundary, reasoning-control compatibility, operational material, model references, performance evidence, preserved historical work, and the static documentation site.

Start with the [canonical documentation index](docs/index.md). It separates current guidance, supporting reference, evidence, research, and archive material instead of treating the repository chronology as an operating manual.

## Scope

- OpenAI- and Anthropic-compatible gateway behavior
- one shared reasoning-control resolver
- FastLLM request adaptation and model-template boundaries
- Qwen3.8 sampling, template, operations, and performance methodology
- reproducible evidence and benchmark artifacts
- lossless historical archive and migration ledgers
- a Git-index-backed Astro documentation site

The repository does not claim that every tracked change is deployed. It does not represent a reasoning budget as a hard FastLLM token cap, and it does not use historical handoffs as current production authority.

## Documentation authority

The only canonical production-state page is [Current production](docs/operations/current-state/production.md). It is sourced field-by-field from the latest immutable file under:

```text
docs/_data/deployment-snapshots/
```

That page shows its capture time and interpretation limits. Missing runtime values remain unknown rather than being inferred from this README, a chronicle, a handoff, or a historical report.

For the tracked software contract, use:

- [System architecture](docs/concepts/architecture/system-overview.md)
- [Reasoning control model](docs/concepts/reasoning/control-model.md)
- [OpenWebUI reasoning controls](docs/guides/openwebui/reasoning-controls.md)
- [Client reasoning fields](docs/guides/client-compatibility/reasoning-fields.md)
- [Proxy runbook](docs/operations/runbooks/proxy.md)
- [Reasoning controls API](docs/reference/api/reasoning-controls.md)
- [Performance methodology](docs/performance/methodology/index.md)

## Repository map

```text
src/gateway/          canonical gateway and reasoning-control source
docs/                 canonical docs, supporting research, and archive
docs/_data/           immutable deployment observations
operations/           deployment material, snapshots, and migration ledgers
evidence/             captured benchmark and result artifacts
benchmarks/           benchmark runners and workload material
tools/                repository and operational tooling
tests/                contract and migration verification
site/                 manifest-driven Astro documentation site
fastllm_adapter.py    checksum-pinned FastLLM request adapter
```

Historical bodies live under `docs/archive/` and remain untouched. The site adds archive notices from manifest metadata rather than editing preserved content.

## Verification

Generate and validate the final Git-index content manifest:

```bash
node site/scripts/generate-repo-index.mjs
node site/scripts/validate-content-manifest.mjs
```

Run the focused repository and reasoning gates selected for the change:

```bash
python -m unittest tests.test_repository_layout -v
python -m unittest tests.test_reasoning_controls -v
python -m unittest tests.test_thinking_proxy_fastllm -v
```

Build the documentation site separately:

```bash
npm --prefix site run build
```

Local generation and build checks do not prove deployed HTTPS status or production inference behavior. Deployment, restart, and post-deploy probes are explicit operations described in the [proxy runbook](docs/operations/runbooks/proxy.md).

## License

MIT, subject to the licenses of referenced upstream projects and artifacts.
