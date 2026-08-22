# Reasoning Control Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make OpenWebUI, OpenAI-compatible and Anthropic-compatible thinking controls deterministic through one canonical resolver, without changing production inference or pretending FastLLM enforces a hard thinking-token budget.

**Architecture:** A framework-independent resolver converts typed effort, toggle and budget candidates into one immutable `ThinkingResolution` or stable `ReasoningControlError`. The moved canonical proxy source uses it for both OpenAI and Anthropic paths; HTTP adapters produce protocol-specific 400 envelopes. `fastllm_adapter.py` remains byte-identical.

**Tech Stack:** Python 3.14, FastAPI, unittest, Jinja2, OpenAI/Anthropic-compatible JSON.

**Spec:** `docs/superpowers/specs/2026-08-22-full-repository-docs-and-thinking-controls-design.md`

## Global Constraints

- Prerequisite: repository migration plan Task 3 is committed; canonical proxy path is `src/gateway/thinking_proxy.py`.
- Work only in the same isolated implementation worktree; do not edit the active workspace or sibling production proxy.
- Ownership is limited to `src/gateway/**`, `tests/test_thinking_proxy_fastllm.py`, and new resolver tests. Do not edit docs, site, workflows, README, `fastllm_adapter.py`, profiles or FastLLM engine code.
- Use TDD: each behavior requires a failing test observed before implementation.
- No production requests, sync or restart.
- Do not emit a backend `reasoning.budget` field: the current engine does not enforce it.
- Do not unpack literal nested `extra_body`.
- Conflicting explicit controls and invalid fields return stable HTTP 400; never silently fall back to xhigh.
- Reasoning and tool calls remain independent response channels.

---

### Task 1: Implement the pure canonical resolver

**Files:**
- Create: `src/gateway/reasoning_controls.py`
- Create: `tests/test_reasoning_controls.py`

**Interfaces:**
- Produces: `ReasoningSource` enum.
- Produces: immutable `ReasoningCandidates`, `ThinkingResolution`, `ReasoningControlError`.
- Produces: `resolve_reasoning_candidates(candidates: ReasoningCandidates) -> ThinkingResolution`.
- Produces: `chat_template_kwargs_for(resolution: ThinkingResolution) -> dict[str, object]`.

- [ ] **Step 1: Write failing public-tier and toggle tests**

```python
from src.gateway.reasoning_controls import (
    ReasoningCandidates,
    ReasoningControlError,
    ReasoningSource,
    resolve_reasoning_candidates,
    chat_template_kwargs_for,
)


def test_off_disables_thinking_instead_of_falling_to_xhigh():
    result = resolve_reasoning_candidates(ReasoningCandidates(
        efforts=((ReasoningSource.TOP_LEVEL_EFFORT, "off"),),
    ))
    assert result.enabled is False
    assert result.public_effort == "off"
    assert result.template_effort is None
    assert chat_template_kwargs_for(result) == {"enable_thinking": False}


def test_native_nested_false_is_preserved():
    result = resolve_reasoning_candidates(ReasoningCandidates(
        toggles=((ReasoningSource.TEMPLATE_TOGGLE, False),),
    ))
    assert chat_template_kwargs_for(result) == {"enable_thinking": False}


def test_minimal_and_low_conflict_at_public_tier():
    try:
        resolve_reasoning_candidates(ReasoningCandidates(efforts=(
            (ReasoningSource.TOP_LEVEL_EFFORT, "minimal"),
            (ReasoningSource.NESTED_EFFORT, "low"),
        )))
    except ReasoningControlError as exc:
        assert exc.code == "conflicting_controls"
    else:
        raise AssertionError("minimal + low must be rejected")
```

- [ ] **Step 2: Run tests and confirm RED**

```bash
python -m unittest tests.test_reasoning_controls -v
```

Expected: import failure because `src.gateway.reasoning_controls` does not exist.

- [ ] **Step 3: Define exact immutable types**

```python
class ReasoningSource(str, Enum):
    TEMPLATE_EFFORT = "template_effort"
    TOP_LEVEL_EFFORT = "top_level_effort"
    NESTED_EFFORT = "nested_effort"
    ANTHROPIC_OUTPUT_EFFORT = "anthropic_output_effort"
    TEMPLATE_TOGGLE = "template_toggle"
    TOP_LEVEL_TOGGLE = "top_level_toggle"
    ANTHROPIC_THINKING_TYPE = "anthropic_thinking_type"
    OPENAI_BUDGET = "openai_budget"
    ANTHROPIC_BUDGET = "anthropic_budget"


@dataclass(frozen=True)
class ReasoningCandidates:
    efforts: tuple[tuple[ReasoningSource, str], ...] = ()
    toggles: tuple[tuple[ReasoningSource, bool], ...] = ()
    openai_budget: int | None = None
    anthropic_thinking_type: str | None = None
    anthropic_budget: int | None = None


@dataclass(frozen=True)
class ThinkingResolution:
    enabled: bool
    public_effort: str | None
    template_effort: str | None
    explicit_budget_tokens: int | None
    unlimited: bool
    sources: tuple[ReasoningSource, ...]


class ReasoningControlError(ValueError):
    def __init__(self, code: str, field_path: str, message: str):
        self.code = code
        self.field_path = field_path
        super().__init__(message)
```

- [ ] **Step 4: Implement total resolution rules**

Implement the exact spec §8 table:

```python
PUBLIC_TO_TEMPLATE = {
    "off": None,
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "xhigh",
    "xhigh": "xhigh",
    "max": "xhigh",
}
```

Canonicalize `none` to `off`. Compare duplicates at public tier, not template tier. Toggle-only true returns enabled with null effort so Qwen3.8 uses template default xhigh. Budget-only requests infer low/medium/xhigh at 4096/8192 boundaries. Disabled+budget, max+finite-budget and contradictory toggles/efforts raise stable errors.

- [ ] **Step 5: Add field parser tests**

Test string trim/lowercase, null omission, empty string, unknown string, bool/object/array efforts, bool/float/zero/negative budgets, template/top-level toggle types, Anthropic enabled minimum 1024 and adaptive/disabled budget rejection.

- [ ] **Step 6: Implement typed extraction helpers**

Produce helpers used by Task 2/3:

```python
def extract_openai_candidates(body: Mapping[str, object]) -> ReasoningCandidates: ...
def extract_anthropic_candidates(body: Mapping[str, object]) -> ReasoningCandidates: ...
```

Extraction validates malformed containers before leaf fields and preserves deterministic source ordering. It must not mutate the body.

- [ ] **Step 7: Run resolver tests and confirm GREEN**

```bash
python -m unittest tests.test_reasoning_controls -v
```

Expected: all total-table, schema and error-ordering tests pass.

- [ ] **Step 8: Commit the pure resolver**

```bash
git add src/gateway/reasoning_controls.py tests/test_reasoning_controls.py
git commit -m "Add canonical reasoning control resolver"
```

---

### Task 2: Integrate OpenAI and OpenWebUI request normalization

**Files:**
- Modify: `src/gateway/thinking_proxy.py` symbols `_normalize_openai_reasoning_controls` and OpenAI chat endpoint
- Modify: `tests/test_thinking_proxy_fastllm.py`
- Test: `tests/test_reasoning_controls.py`

**Interfaces:**
- Consumes: `extract_openai_candidates`, `resolve_reasoning_candidates`, `chat_template_kwargs_for`.
- Produces: `apply_openai_reasoning_controls(body: dict[str, object]) -> ThinkingResolution`.
- Produces: OpenAI invalid-request error envelope.

- [ ] **Step 1: Add failing OpenWebUI-shaped request tests**

```python
def test_openwebui_reasoning_effort_off_closes_thinking():
    body = {
        "model": "qwen3.8-fastllm",
        "messages": [{"role": "user", "content": "Answer directly."}],
        "reasoning_effort": "off",
    }
    resolution = thinking_proxy._normalize_openai_reasoning_controls(body)
    assert resolution.enabled is False
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning" not in body


def test_openwebui_low_reaches_template_without_fake_budget():
    body = {"reasoning_effort": "low"}
    thinking_proxy._normalize_openai_reasoning_controls(body)
    assert body["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "low",
    }
    assert "reasoning" not in body
```

- [ ] **Step 2: Run focused tests and confirm RED**

```bash
python -m unittest \
  tests.test_thinking_proxy_fastllm.ThinkingProxyFastLLMTests.test_openwebui_reasoning_effort_off_closes_thinking \
  tests.test_thinking_proxy_fastllm.ThinkingProxyFastLLMTests.test_openwebui_low_reaches_template_without_fake_budget -v
```

Expected current failures: off produces enable=true/default xhigh; low injects `reasoning.budget`.

- [ ] **Step 3: Replace duplicated OpenAI maps with resolver calls**

`_normalize_openai_reasoning_controls(body)` must:

1. Extract candidates without mutating.
2. Resolve or raise `ReasoningControlError`.
3. Consume public control fields only after successful resolution.
4. Merge canonical kwargs into existing `chat_template_kwargs` without overriding a validated explicit toggle.
5. Remove old synthesized `reasoning.budget` behavior.
6. Return the immutable resolution for trace logging/tests.

Delete obsolete alias/default-budget tables once every caller uses the resolver. Do not retain parallel legacy behavior.

- [ ] **Step 4: Map resolver errors to exact OpenAI HTTP 400**

At the OpenAI chat endpoint catch only `ReasoningControlError` and return:

```python
JSONResponse(
    status_code=400,
    content={"error": {
        "message": str(exc),
        "type": "invalid_request_error",
        "param": exc.field_path,
        "code": exc.code,
    }},
)
```

Do not catch unrelated exceptions as invalid requests.

- [ ] **Step 5: Add full OpenAI matrix and error-envelope tests**

Cover unset, off, none, minimal, low, medium, high, xhigh, max; nested effort; nested false; equivalent duplicates; public-tier conflicts; malformed containers; invalid values; explicit budgets and exact serialized kwargs.

- [ ] **Step 6: Render the real Qwen3.8 template in tests**

Use the moved template path `config/chat-templates/qwen3.8_unsloth_ud.jinja` with unchanged root `fastllm_adapter.py`. Assert off ends with the closed empty think block and enabled tiers contain the exact low/medium/xhigh template state.

- [ ] **Step 7: Run focused OpenAI/OpenWebUI tests**

```bash
python -m unittest tests.test_reasoning_controls -v
python -m unittest tests.test_thinking_proxy_fastllm -v
```

Expected: no off→xhigh fallback, no nested false overwrite, no fake backend budget.

- [ ] **Step 8: Commit OpenAI integration**

```bash
git add src/gateway/thinking_proxy.py tests/test_thinking_proxy_fastllm.py
git commit -m "Normalize OpenWebUI reasoning effort controls"
```

---

### Task 3: Integrate Anthropic and OMP reasoning controls

**Files:**
- Modify: `src/gateway/thinking_proxy.py` symbols `_anthropic_to_openai` and Anthropic messages endpoint
- Modify: `tests/test_thinking_proxy_fastllm.py`

**Interfaces:**
- Consumes: `extract_anthropic_candidates`, canonical resolver.
- Produces: canonical OpenAI/backend body and Anthropic invalid-request envelope.

- [ ] **Step 1: Add failing adaptive/max and budget tests**

```python
def test_anthropic_adaptive_max_is_unlimited_without_4096_budget():
    body = {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "max"},
        "messages": [{"role": "user", "content": "Solve."}],
    }
    converted = thinking_proxy._anthropic_to_openai(body)
    assert converted["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "xhigh",
    }
    assert "reasoning" not in converted


def test_anthropic_disabled_with_budget_is_400_contract_error():
    with self.assertRaises(ReasoningControlError) as ctx:
        thinking_proxy._anthropic_to_openai({
            "thinking": {"type": "disabled", "budget_tokens": 4096}
        })
    assert ctx.exception.code == "conflicting_controls"
```

- [ ] **Step 2: Run tests and confirm RED**

Expected: adaptive/max receives fixed 4096 budget; invalid combinations are accepted or normalized inconsistently.

- [ ] **Step 3: Replace Anthropic-specific tier/budget logic**

`_anthropic_to_openai` must extract and resolve controls before converting messages. It uses the same `chat_template_kwargs_for()` result as OpenAI. Explicit budget is retained only in returned resolution/logging metadata, never written as an ignored backend `reasoning.budget`.

- [ ] **Step 4: Implement Anthropic 400 envelope**

At `/v1/messages`, map `ReasoningControlError` to:

```python
JSONResponse(
    status_code=400,
    content={
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": f"{exc.field_path}: {exc}",
        },
    },
)
```

- [ ] **Step 5: Add complete Anthropic matrix tests**

Cover enabled/disabled/adaptive; output_config effort off/none/minimal/low/medium/high/xhigh/max; 1 invalid, 1024, 4096, 4097, 8192, 8193, 16384 budgets; adaptive/disabled budget rejection; max+finite budget; exact template kwargs and prompt render.

- [ ] **Step 6: Run full reasoning tests**

```bash
python -m unittest tests.test_reasoning_controls -v
python -m unittest tests.test_thinking_proxy_fastllm -v
```

- [ ] **Step 7: Commit Anthropic integration**

```bash
git add src/gateway/thinking_proxy.py tests/test_thinking_proxy_fastllm.py
git commit -m "Unify Anthropic thinking controls with canonical resolver"
```

---

### Task 4: Preserve reasoning and tool-call response invariants

**Files:**
- Modify only if tests expose a bug: `src/gateway/thinking_proxy.py`
- Modify: `tests/test_thinking_proxy_fastllm.py`
- Create: `tests/fixtures/openwebui_reasoning_requests.json`

**Interfaces:**
- Consumes canonical request behavior.
- Verifies existing response channel semantics without changing `fastllm_adapter.py`.

- [ ] **Step 1: Add captured OpenWebUI request fixtures**

Fixture rows include flattened provider payloads for unset/off/minimal/low/medium/high/xhigh/max, nested custom params, malformed values and a model-level override example. Do not store auth tokens, endpoint secrets or runtime logs.

- [ ] **Step 2: Add response tests for reasoning plus tools**

```python
import fastllm_adapter


def test_reasoning_is_preserved_when_finish_reason_is_tool_calls():
    upstream = {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "reasoning_content": "I need both tools.",
                "content": "",
                "tool_calls": [
                    {"id": "call_weather", "type": "function",
                     "function": {"name": "weather", "arguments": "{}"}},
                    {"id": "call_time", "type": "function",
                     "function": {"name": "time", "arguments": "{}"}},
                ],
            },
        }]
    }
    response = fastllm_adapter.adapt_fastllm_response(upstream)
    message = response["choices"][0]["message"]
    assert message["reasoning_content"] == "I need both tools."
    assert len(message["tool_calls"]) == 2
```

Add streaming delimiter-fragmentation, reasoning-only, malformed/unclosed marker, disabled plain text, and Anthropic block ordering tests. Assert exactly one OpenAI reasoning field; do not emit both `reasoning` and `reasoning_content`.

- [ ] **Step 3: Run tests and inspect failures**

```bash
python -m unittest tests.test_thinking_proxy_fastllm -v
```

If existing behavior already passes, do not modify production code merely to create churn.

- [ ] **Step 4: Apply only evidence-backed response fixes**

Preserve parser ordering: reasoning first, then content/tool calls. Never clear reasoning because tool calls exist. Keep OpenWebUI-compatible `reasoning_content` on the current OpenAI proxy endpoint.

- [ ] **Step 5: Re-run all focused tests**

```bash
python -m unittest tests.test_reasoning_controls -v
python -m unittest tests.test_thinking_proxy_fastllm -v
```

- [ ] **Step 6: Commit fixtures and any required response fix**

```bash
git add tests src/gateway/thinking_proxy.py
git commit -m "Cover reasoning controls with tool-call responses"
```

---

### Task 5: Verify the undeployed compatibility change

**Files:**
- Verify: `src/gateway/reasoning_controls.py`
- Verify: `src/gateway/thinking_proxy.py`
- Verify: reasoning tests and fixtures
- Preserve byte-identical: `fastllm_adapter.py`

**Interfaces:**
- Produces verification results consumed by the repository migration plan Task 8.
- Produces a concise result message for the docs subagent; does not edit docs/site.

- [ ] **Step 1: Run all reasoning-control tests**

```bash
python -m unittest tests.test_reasoning_controls -v
python -m unittest tests.test_thinking_proxy_fastllm -v
```

- [ ] **Step 2: Run static control scans**

Assert old duplicated effort/budget maps and off/none-only branches are absent from the moved canonical proxy. Assert all OpenAI/Anthropic paths call the shared resolver.

- [ ] **Step 3: Verify the adapter safety pin**

```bash
sha256sum fastllm_adapter.py
```

Expected: exact baseline ledger checksum. Do not import or execute sibling production proxy.

- [ ] **Step 4: Exercise captured fixtures through local non-production handlers**

Use FastAPI test client or direct handler calls. No live URL requests. Assert exact 400 envelopes and exact canonical kwargs for every fixture.

- [ ] **Step 5: Send docs facts to the integration owner**

Return a message containing supported public tiers, conflict policy, budget-hint semantics, stable errors, OpenWebUI saved-param precedence and the fact that production deployment/restart was not performed. Do not write docs directly.

- [ ] **Step 6: Commit any final deterministic fixture update**

If verification changed no files, do not create an empty commit. Otherwise commit only owned test/fixture files.
