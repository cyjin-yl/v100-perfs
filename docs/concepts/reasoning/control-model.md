# Reasoning control model

Reasoning control has four distinct layers: public effort, enable/disable intent, finite budget hints, and model-template inputs. Keeping them separate is the core invariant.

## Public effort

The accepted public values are:

```text
off, none, minimal, low, medium, high, xhigh, max
```

Input strings are trimmed with ASCII whitespace rules and lowercased. `none` is an input alias for canonical public `off`. Public tiers remain distinct during conflict detection even when they map to the same template tier.

| Public input | Thinking | Canonical public effort | Template effort | Unlimited metadata |
|---|---:|---|---|---:|
| omitted | enabled | omitted | template default | no |
| `off` / `none` | disabled | `off` | omitted | no |
| `minimal` | enabled | `minimal` | `low` | no |
| `low` | enabled | `low` | `low` | no |
| `medium` | enabled | `medium` | `medium` | no |
| `high` | enabled | `high` | `xhigh` | no |
| `xhigh` | enabled | `xhigh` | `xhigh` | no |
| `max` | enabled | `max` | `xhigh` | yes |

Omitted controls preserve template-default thinking. Disabling always serializes `enable_thinking=false` and omits `reasoning_effort` from template kwargs.

## Toggles

Boolean controls express enable or disable intent without selecting a tier. Equal duplicates are valid; unequal duplicates conflict.

A true toggle conflicts with `off` or `none`. A false toggle conflicts with any enabled named effort. The resolver reports the conflict instead of choosing whichever field happened to appear last.

Anthropic `thinking.type` provides a typed toggle:

- `enabled` and `adaptive` enable thinking;
- `disabled` disables thinking;
- `adaptive` may be paired with a named effort when no finite Anthropic budget is present.

## Budget hints

A budget is compatibility metadata, not a hard reasoning-token cap.

- OpenAI `thinking_budget` is an integer of at least 1.
- Anthropic `thinking.budget_tokens` is an integer of at least 1024 and is valid only with `thinking.type=enabled`.
- Without a named effort, a finite budget selects `low` at 4096 or below, `medium` at 8192 or below, and `xhigh` above 8192.
- With a non-`max` named effort, the name chooses the template tier and the budget remains resolution/log metadata.
- `max` means unlimited metadata and conflicts with a finite budget.

The gateway does not map a reasoning budget to `max_tokens`, and it does not forward a backend `reasoning.budget`. FastLLM is not documented as enforcing this hint.

## Candidate extraction

OpenAI/OpenWebUI candidates can come from:

- `reasoning_effort`;
- `reasoning.effort`;
- `chat_template_kwargs.reasoning_effort`;
- `enable_thinking`;
- `chat_template_kwargs.enable_thinking`;
- `thinking_budget`.

Anthropic candidates can come from:

- `output_config.effort`;
- `thinking.type`;
- `thinking.budget_tokens`.

Null means omitted. A literal nested `extra_body` is not unpacked: compatible SDKs flatten `extra_body` onto the provider request before transport.

## Deterministic resolution

The resolver evaluates categories in this order:

1. malformed container;
2. leaf type;
3. enum or range;
4. named-effort conflict;
5. toggle conflict;
6. effort/toggle conflict;
7. budget conflict.

Fields within a category follow schema order, not JSON insertion order. Equivalent duplicates are accepted; public `off` and `none` are equivalent. Different public tiers conflict before template mapping, so `minimal` plus `low` is still a conflict, as is `high` plus `xhigh`.

## Canonical output

Successful serialization always writes a boolean `chat_template_kwargs.enable_thinking`. It adds `chat_template_kwargs.reasoning_effort` only when thinking is enabled and a non-null template effort was selected. Public compatibility fields are consumed rather than forwarded beside the canonical values.

## Parser ownership and response channels

### Apiserver capability and startup validation

The FastLLM apiserver owns model/parser capability, not client-field aliases. Its capability contract identifies the selected template/parser behavior, supported template controls, native effort values, defaults, and always-on or always-off constraints. It validates the selected parser and template at startup and exposes validity and error state through read-only properties. The proxy must not guess parser capability from a model name or turn an unsupported control into a silent default.

Hard reasoning-budget enforcement would also require an explicit apiserver capability. Until that capability exists, the proxy retains a finite budget as compatibility metadata and sends only the canonical template controls.

### Proxy wire adaptation

The proxy owns protocol adaptation around the engine boundary. On requests it resolves public aliases before template rendering. On responses it recognizes only valid, closed reasoning markers, promotes native tool-call output to protocol tool calls, and maps the result into the selected OpenAI or Anthropic envelope. Malformed or unclosed reasoning markers remain visible content rather than being silently reclassified.

### Independent channels and ordering

Reasoning, visible content, and tool calls are independent channels. A response may contain any valid combination, including reasoning with multiple parallel tool calls. A `tool_calls` finish reason must not erase reasoning, and a reasoning-only response still carries an empty visible-content string where the protocol requires it.

The OpenAI-compatible endpoint emits exactly one reasoning field, `reasoning_content`; it does not also emit a `reasoning` alias that clients could double-count. `content` and `tool_calls` remain beside it. Anthropic conversion preserves the logical block order: thinking first, then visible text, then tool-use blocks. Streaming preserves the same order even when SSE delimiters arrive across arbitrary byte boundaries.

See [Reasoning controls API](../../reference/api/reasoning-controls.md) for exact errors and [Client reasoning fields](../../guides/client-compatibility/reasoning-fields.md) for protocol examples.
