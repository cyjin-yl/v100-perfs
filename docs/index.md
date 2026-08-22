# V100 FastLLM documentation

Use this documentation to understand, operate, and verify the repository's single-GPU FastLLM stack without treating historical field notes as current production truth.

## Choose a path

### Start here

1. [Repository overview](getting-started/overview.md) defines the supported surface and non-goals.
2. [System architecture](concepts/architecture/system-overview.md) follows a request from client to model and back.
3. [Current production](operations/current-state/production.md) is the only canonical production-state page.

### Configure reasoning

1. [Reasoning control model](concepts/reasoning/control-model.md) separates public effort, toggles, budget hints, and template inputs.
2. [OpenWebUI reasoning controls](guides/openwebui/reasoning-controls.md) covers OpenWebUI v0.9.6 settings and precedence.
3. [Client reasoning fields](guides/client-compatibility/reasoning-fields.md) compares the accepted OpenAI and Anthropic shapes with other ecosystem surfaces.
4. [Reasoning controls API](reference/api/reasoning-controls.md) is the exact resolver and error contract.

### Operate and verify

- [Proxy runbook](operations/runbooks/proxy.md) separates inspection, local verification, deployment, and rollback.
- [Qwen3.8 sampling and template](reference/models/qwen38/sampling-and-template.md) documents the model-facing defaults and template identity rules.
- [Performance methodology](performance/methodology/index.md) explains which artifacts are measurements, summaries, or historical benchmarks.

## Research

Research notes are supporting material, not production state. Start with [performance research](research/performance/short-prefill-plan.md) or [storage research](research/storage/optane-l3.md), then verify every conclusion against its linked evidence and capture date.

## Archive

The [archived design contract](archive/design-decisions/specs/2026-08-22-full-repository-docs-and-thinking-controls-design.md) leads into preserved chronicles, handoffs, assessments, earlier model generations, and design decisions. Archive pages can explain how a decision was reached, but they do not override canonical docs or the immutable deployment snapshot.

## Authority rules

- Current operational claims come from the latest immutable file in `docs/_data/deployment-snapshots/` and are rendered in [Current production](operations/current-state/production.md).
- Canonical pages describe the tracked repository contract. A tracked change is not deployed merely because it is documented.
- Evidence records observations. Performance summaries must link to measurement artifacts.
- Archive bodies remain historical and are labeled by the site layout rather than rewritten.
- If an allowlisted production source did not expose a value, the snapshot records it as unknown instead of inferring it from an old README or handoff.

## Repository index

The documentation site also provides a manifest-backed repository view for every tracked path. Binary and non-regular objects render as metadata; text is escaped or sanitized before display.
