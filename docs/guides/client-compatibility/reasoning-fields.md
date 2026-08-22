# Client reasoning fields

The gateway provides one deterministic compatibility boundary for OpenAI-shaped and Anthropic-shaped requests. Engine names elsewhere in the ecosystem do not imply identical native semantics.

## Compatibility matrix

| Client or surface | Fields accepted by this gateway | Normalization | Boundary |
|---|---|---|---|
| OpenAI-compatible SDK | `reasoning_effort`, `reasoning.effort`, `enable_thinking`, `thinking_budget` | Shared resolver | SDK `extra_body` must be flattened onto the wire |
| OpenWebUI v0.9.6 | top-level `reasoning_effort` plus OpenAI-compatible fields | Shared resolver after OpenWebUI precedence | Saved model params override per-chat params before transport |
| Anthropic-compatible client | `output_config.effort`, `thinking.type`, `thinking.budget_tokens` | Same shared resolver, Anthropic error envelope | Budget is valid only with `thinking.type=enabled` |
| Direct template integration | `chat_template_kwargs.reasoning_effort`, `chat_template_kwargs.enable_thinking` | Shared resolver preserves unrelated kwargs | Public and template candidates must agree |
| vLLM or SGLang client using this gateway | OpenAI-compatible wire fields above | Gateway contract only | No claim is made about the engine's native server fields or enforcement |
| TokenSpeed/SMG-style client using this gateway | OpenAI-compatible wire fields above | Gateway contract only | No native TokenSpeed/SMG alias is copied into FastLLM |

The last two rows are routing guidance, not a statement that those native servers share this resolver. When a client targets this gateway, inspect the final HTTP JSON rather than reasoning from the client library name.

## OpenAI-compatible examples

Named effort:

```json
{
  "reasoning_effort": "high"
}
```

Explicit disable through template kwargs:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": false,
    "unrelated_template_option": "preserved"
  }
}
```

Budget hint without a named effort:

```json
{
  "thinking_budget": 8192
}
```

The final example selects template `medium`; it does not cap hidden reasoning at 8192 tokens.

## Anthropic-compatible examples

Named effort with adaptive thinking:

```json
{
  "output_config": {"effort": "max"},
  "thinking": {"type": "adaptive"}
}
```

Finite budget with enabled thinking:

```json
{
  "thinking": {
    "type": "enabled",
    "budget_tokens": 4096
  }
}
```

`adaptive` plus `budget_tokens` conflicts. `disabled` plus a budget conflicts. A budget below 1024 is invalid.

## Equivalent and conflicting duplicates

Equivalent duplicates are accepted:

```json
{
  "reasoning_effort": "off",
  "reasoning": {"effort": "none"}
}
```

Different public tiers conflict even when they reach the same template tier. For example, `minimal` and `low` conflict; `high` and `xhigh` conflict.

Toggle conflicts are also explicit. `enable_thinking=false` cannot accompany an enabled effort, and `enable_thinking=true` cannot accompany `off`.

## Error envelopes

OpenAI-compatible routes return `error.code` and `error.param`. Anthropic-compatible routes return an Anthropic top-level `type=error`; the stable field path prefixes the message. Both select errors in schema order, not request insertion order.

See [Reasoning controls API](../../reference/api/reasoning-controls.md) for exact envelopes.

## Transport checklist

- Log or capture the provider-boundary body without credentials.
- Confirm SDK `extra_body` members are flattened.
- Avoid saving a fixed OpenWebUI model effort when per-chat selection is required.
- Do not forward compatibility budget metadata as backend `reasoning.budget`.
- Do not infer native vLLM, SGLang, TokenSpeed, or SMG behavior from this gateway's accepted fields.
