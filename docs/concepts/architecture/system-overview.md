# System architecture

The repository separates public protocol compatibility, canonical request preparation, engine execution, and evidence. That separation prevents a client alias or an old handoff from silently becoming model behavior.

## Request path

```text
OpenWebUI / OpenAI SDK        Anthropic / Claude-compatible client
            |                              |
            +---------- HTTP gateway ------+
                       src/gateway/thinking_proxy.py
                         |  extract typed candidates
                         v
                       src/gateway/reasoning_controls.py
                         |  one ThinkingResolution or stable error
                         v
                       fastllm_adapter.py
                         |  canonical chat_template_kwargs
                         |  normalized tools, messages, sampling
                         v
                       FastLLM apiserver
                         |  template render, scheduler, KV cache, MTP
                         v
                       Qwen3.8 weights + multimodal projector
```

Responses follow the reverse path. The gateway converts backend output into the selected OpenAI or Anthropic envelope while preserving the distinction between visible content and reasoning content.

## Ownership boundaries

### Gateway

The gateway owns public compatibility:

- OpenAI/OpenWebUI and Anthropic field extraction;
- type, enum, range, and conflict validation;
- alias normalization such as `none` to canonical public `off`;
- protocol-specific HTTP 400 envelopes;
- lifecycle, routing, and response adaptation.

The OpenAI and Anthropic adapters call one resolver. They must not maintain independent effort tables or synthesize a backend `reasoning.budget`.

### Reasoning resolver

`src/gateway/reasoning_controls.py` is pure protocol logic. It receives typed candidates in schema order and returns an immutable resolution or `ReasoningControlError`. It has no scheduler, HTTP, or template dependency.

See [Reasoning control model](../reasoning/control-model.md) for concepts and [Reasoning controls API](../../reference/api/reasoning-controls.md) for the exact table.

### Adapter

`fastllm_adapter.py` translates a canonical request into the FastLLM request shape. It chooses model-facing sampling defaults only when the client did not provide them, normalizes tool/message details, and carries canonical `chat_template_kwargs` to template rendering.

The adapter is a production safety pin in this migration: documentation and resolver work must not incidentally alter it.

### FastLLM and model assets

FastLLM owns engine capabilities: model loading, scheduling, KV-cache representation, MTP execution, token sampling, tool-call grammar counters, and read-only runtime properties. A compatibility budget does not become an engine-enforced token limit unless the engine exposes and implements such a capability; this stack currently does not claim one.

Model weights, multimodal projector, and chat template are independently identified by path and SHA-256 in the immutable [deployment snapshot](../../operations/current-state/production.md).

## Control plane and data plane

- Data-plane endpoints accept inference requests. Documentation capture must not call them.
- Read-only state comes from `/health`, `/props`, authenticated admin state, an explicit process-environment allowlist, and local checksums.
- Suspend, resume, profile switch, restart, and inference calls are mutations or workload actions. They require a separate operating decision and are outside documentation capture.

## Tracked source versus deployment

A commit can define the next gateway contract without being the running contract. The current-state page records the observed deployment; canonical reasoning docs record the tracked source contract. Operators must perform an explicit deployment and post-deploy verification before claiming that a tracked compatibility change is live.

## Evidence path

Performance and behavior claims should flow in one direction:

```text
raw capture -> evidence record -> bounded summary -> canonical guidance
```

Archive documents preserve prior conclusions and context but never flow back into current state without a new observation.
