# OpenWebUI reasoning controls

OpenWebUI v0.9.6 sends its Advanced Params reasoning effort as a top-level OpenAI-compatible `reasoning_effort` field. The gateway normalizes that flattened provider request together with any explicit template kwargs.

## Connect the provider

Configure an OpenAI-compatible connection to the gateway and choose one of the advertised model aliases. Keep provider credentials outside repository files and browser-visible documentation.

The wire request should contain a normal top-level field:

```json
{
  "model": "<advertised-model-alias>",
  "messages": [
    {"role": "user", "content": "Explain the trade-off."}
  ],
  "reasoning_effort": "medium"
}
```

Do not put a literal `extra_body` object on the wire. Python SDK `extra_body` is a transport convenience: the SDK flattens its members into the outgoing JSON request.

## Choose an effort

The gateway accepts `off` (or input alias `none`), `minimal`, `low`, `medium`, `high`, `xhigh`, and `max`.

- Use `off` when the template must disable the reasoning channel.
- Use a named tier when the user needs a stable public intent.
- Use `max` only when unlimited metadata is intended; do not pair it with a finite budget.
- Leave the field unset to retain template-default thinking.

A finite `thinking_budget` is a tier hint, not a hard output or hidden-reasoning token cap.

## Per-chat versus model-level settings

OpenWebUI v0.9.6 applies saved model-level params after per-chat params. A saved model-level `reasoning_effort` therefore overrides the per-chat selection before the request reaches the provider boundary.

To let users adjust effort per chat:

1. remove the fixed reasoning effort from the model's saved parameters;
2. expose the per-chat reasoning control;
3. choose the value in the conversation;
4. inspect the flattened provider-boundary request when debugging precedence.

If the control is missing, check `permissions.chat.controls` and `permissions.chat.params` in the OpenWebUI permissions configuration.

## Nested template kwargs

Advanced integrations may send canonical template fields directly:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": false,
    "custom_option": "preserved"
  }
}
```

The resolver consumes recognized reasoning fields, retains unrelated template kwargs, and writes the canonical boolean. A conflicting top-level effort is rejected; it does not overwrite nested false by last-writer-wins behavior.

## What errors mean

Invalid or conflicting controls return an OpenAI-compatible HTTP 400 with a stable `code` and `param`:

```json
{
  "error": {
    "message": "reasoning_effort must be a string or null",
    "type": "invalid_request_error",
    "param": "reasoning_effort",
    "code": "invalid_type"
  }
}
```

Stable codes are `invalid_type`, `invalid_value`, and `conflicting_controls`. Fix the named field; do not retry the same body expecting a fallback to template-default `xhigh`.

## Troubleshooting sequence

1. Confirm the browser/provider boundary contains the expected top-level field.
2. Remove a saved model-level effort before testing per-chat changes.
3. Check for duplicates in `reasoning.effort` and `chat_template_kwargs.reasoning_effort`.
4. Check for contradictory `enable_thinking` values.
5. Remove a finite budget when using `off`, `none`, or `max`.
6. Use the exact error `param` to locate a malformed container or leaf.

The tracked gateway contract is documented in [Reasoning controls API](../../reference/api/reasoning-controls.md). The [current production page](../../operations/current-state/production.md) separately states whether the documentation task deployed anything; it did not.
