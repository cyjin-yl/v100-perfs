# Qwen3.8 sampling and template

This reference separates the model recommendation, the tracked adapter contract, and the observed deployment. Do not use a historical profile or README to collapse those three authorities into one.

## Model-recommended sampling

Qwen3.8 27B uses different recommended sampling for thinking and non-thinking modes:

| Mode | `temperature` | `top_p` | `top_k` | `min_p` | `presence_penalty` | `repetition_penalty` |
|---|---:|---:|---:|---:|---:|---:|
| Thinking, default | 1.0 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| Instruct / non-thinking | 0.7 | 0.80 | 20 | 0.0 | 1.5 | 1.0 |

Source: [Qwen/Qwen3.8-27B README](https://huggingface.co/Qwen/Qwen3.8-27B).

`min_p=0.0` disables min-p filtering. Accepting this zero-valued field is not evidence that a non-zero min-p implementation is supported.

Qwen3.6 weights in this repository are derivatives without a separate official Qwen sampling recommendation. Their profiles or explicit client values are evidence for those runs, not a Qwen3.8 default.

## Tracked adapter behavior

`fastllm_adapter.prepare_fastllm_body()` chooses the thinking or non-thinking default set from canonical `chat_template_kwargs.enable_thinking`. Explicit client `temperature`, `top_p`, `top_k`, and `presence_penalty` values take precedence. `temperature=0` remains deterministic greedy decoding.

The gateway's public effort controls select thinking state and a template effort tier. They do not select a second sampling table for every effort tier.

Low temperature is not a universal safety setting for thinking. Over-concentrating the posterior can increase repetitive reasoning loops. Restore the model recommendation before attributing a problem to weight quantization, KV quantization, or penalties.

## Penalty semantics

FastLLM keeps three penalty concepts distinct:

- `presence_penalty` subtracts a fixed value from a token logit once the token has appeared in generated history.
- `frequency_penalty` subtracts the value multiplied by the token's generated-history count.
- `repeat_penalty` is FastLLM's internal name for multiplicative `repetition_penalty`, preserving the positive/negative logit multiply/divide rule.

The order is multiplicative repeat penalty, additive presence/frequency penalty, temperature and softmax, then top-k/top-p. Ordinary sampling, CUDA handoff, and MTP exact/typical acceptance must use the same penalized distribution; otherwise draft and verify posteriors diverge.

## Chat-template identity

The production template must be exported from the matching GGUF's embedded `tokenizer.chat_template`. Do not substitute a hand-written template or a template from another model.

The tracked Qwen3.8 Q5 and Q6 exports are each 9,993 bytes and have SHA-256:

```text
12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce
```

Matching hashes prove byte identity of those template files; they do not prove which file a running process selected. The immutable [current deployment snapshot](../../../operations/current-state/production.md) records the selected template path, checksum, startup validation, weight identity, and capture time.

## Production observation boundary

Deployment values are not maintained on this reference page. Consult the timestamped current-state page; any field that its allowlisted runtime sources did not expose remains unknown rather than being inferred from these recommendations.

For client-facing effort and budget semantics, use [Reasoning controls API](../../api/reasoning-controls.md). A finite reasoning budget is a template-tier hint, not a hard sampling or token limit.
