# Reasoning controls API

This page is the canonical contract for the tracked resolver in `src/gateway/reasoning_controls.py`. OpenAI and Anthropic adapters extract candidates, call this resolver, and serialize the same `ThinkingResolution`.

## Accepted inputs

### OpenAI and OpenWebUI

| Field path | Type | Normalization |
|---|---|---|
| `reasoning_effort` | string or null | ASCII trim, lowercase, public effort enum |
| `reasoning.effort` | string or null | same |
| `chat_template_kwargs.reasoning_effort` | string or null | same |
| `enable_thinking` | boolean or null | none |
| `chat_template_kwargs.enable_thinking` | boolean or null | none |
| `thinking_budget` | integer or null | minimum 1; booleans are not integers |

`reasoning` and `chat_template_kwargs` must each be an object or null. The extractor does not mutate the caller's request.

### Anthropic

| Field path | Type | Normalization |
|---|---|---|
| `output_config.effort` | string or null | ASCII trim, lowercase, public effort enum |
| `thinking.type` | string or null | ASCII trim, lowercase; `enabled`, `disabled`, or `adaptive` |
| `thinking.budget_tokens` | integer or null | minimum 1024; valid only with enabled thinking |

`output_config` and `thinking` must each be an object or null.

### Public effort mapping

| Public input | `enabled` | `public_effort` | `template_effort` | `unlimited` |
|---|---:|---|---|---:|
| omitted | true | null | null | false |
| `off` / `none` | false | `off` | null | false |
| `minimal` | true | `minimal` | `low` | false |
| `low` | true | `low` | `low` | false |
| `medium` | true | `medium` | `medium` | false |
| `high` | true | `high` | `xhigh` | false |
| `xhigh` | true | `xhigh` | `xhigh` | false |
| `max` | true | `max` | `xhigh` | true |

## Duplicate and conflict rules

- Equivalent effort duplicates are accepted. `off` and `none` are equivalent.
- Different public tiers conflict before template mapping. `minimal` plus `low`, and `high` plus `xhigh`, are conflicts.
- Equal toggle duplicates are accepted; unequal toggles conflict.
- A true toggle conflicts with `off`; a false toggle conflicts with any enabled effort.
- A finite budget conflicts with disabled thinking and with `max`.
- `thinking.budget_tokens` conflicts with `thinking.type=adaptive` or `disabled`.
- If both protocol budgets are supplied directly to the shared resolver, unequal values conflict.

The conflicting field is the later field in fixed schema order, and its message names the earlier field. Request insertion order does not change the result.

## Budget resolution

When no named effort is present:

| Finite budget | Template effort |
|---:|---|
| 1–4096 OpenAI, or 1024–4096 Anthropic | `low` |
| 4097–8192 | `medium` |
| above 8192 | `xhigh` |

When a non-`max` named effort is present, the named effort chooses the template tier. The budget remains `explicit_budget_tokens` metadata.

A budget is not forwarded as backend `reasoning.budget`, not mapped to `max_tokens`, and not documented as a FastLLM-enforced reasoning limit.

## Resolution record

A successful resolution contains:

```text
enabled: boolean
public_effort: off|minimal|low|medium|high|xhigh|max|null
template_effort: low|medium|xhigh|null
explicit_budget_tokens: integer|null
unlimited: boolean
sources: ordered tuple of candidate sources
```

Template serialization is intentionally narrower:

```json
{
  "enable_thinking": true,
  "reasoning_effort": "medium"
}
```

`enable_thinking` is always present. `reasoning_effort` is included only when thinking is enabled and `template_effort` is not null.

## Stable errors

The resolver uses three codes:

- `invalid_type`;
- `invalid_value`;
- `conflicting_controls`.

Error selection order is malformed container, leaf type, enum/range, named-effort conflict, toggle conflict, effort/toggle conflict, then budget conflict. Fields within a category follow schema order.

Representative stable messages include:

```text
reasoning_effort must be a string or null
chat_template_kwargs must be an object or null
thinking_budget must be at least 1
thinking.budget_tokens must be at least 1024
thinking.budget_tokens cannot be used when thinking.type is adaptive
thinking_budget cannot be used when thinking is disabled
thinking_budget cannot be used with max reasoning effort
```

Enum errors list accepted values in canonical order.

## HTTP envelopes

OpenAI-compatible HTTP 400:

```json
{
  "error": {
    "message": "<stable message>",
    "type": "invalid_request_error",
    "param": "<field_path>",
    "code": "<invalid_type|invalid_value|conflicting_controls>"
  }
}
```

Anthropic-compatible HTTP 400:

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "<field_path>: <stable message>"
  }
}
```

Endpoint adapters catch `ReasoningControlError` for these client errors. They do not relabel unrelated exceptions as invalid client controls.

## Endpoint coverage

- OpenAI `/v1/chat/completions` resolves controls before tool normalization or routing.
- Anthropic `/v1/messages` resolves before the Haiku fast path or other routing.
- Anthropic `/v1/messages/count_tokens` uses the same conversion and reasoning validation path.

See [Reasoning control model](../../concepts/reasoning/control-model.md) for concepts and [OpenWebUI reasoning controls](../../guides/openwebui/reasoning-controls.md) for UI precedence.
