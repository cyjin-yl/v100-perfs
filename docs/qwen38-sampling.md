# Qwen3.5 / Qwen3.8 采样与模板配置

## 推荐采样参数

Qwen3.5 与 Qwen3.8 必须按模型代际和推理模式选择采样档，不能使用统一默认值。

Qwen3.8 27B 官方档：

| 模式 | `temperature` | `top_p` | `top_k` | `presence_penalty` |
|---|---:|---:|---:|---:|
| Thinking（默认） | 1.0 | 0.95 | 20 | 0.0 |
| Instruct / non-thinking | 0.7 | 0.80 | 20 | 1.5 |

Qwen3.5 官方通用文本档：

| 模式 | `temperature` | `top_p` | `top_k` | `presence_penalty` | `repeat_penalty` |
|---|---:|---:|---:|---:|---:|
| Thinking | 0.6 | 0.95 | 20 | 0.0 | 1.0 |
| Instruct / non-thinking | 0.7 | 0.80 | 20 | 1.5 | 1.0 |

仓库中的 Qwen3.6 是衍生权重，没有独立的 Qwen 官方采样档；其参数由对应 profile 或客户端显式给出，不伪称官方推荐。

`fastllm_adapter.prepare_fastllm_body()` 根据 `chat_template_kwargs.enable_thinking` 选择默认档。客户端显式提供的 `temperature`、`top_p`、`top_k` 或 `presence_penalty` 始终优先；`temperature=0` 仍表示确定性贪婪解码。

低温不是保守的通用修复。对 Qwen3.8 thinking，过低温度会让 posterior 过早集中在局部模式，增加长段推理循环的风险。应先恢复模型推荐采样，再判断权重量化、KV 量化或 penalty 的影响。

## Penalty 字段语义

FastLLM 分别实现三种 penalty，不再把 OpenAI `frequency_penalty` 偷换成乘法 `repeat_penalty`：

- `presence_penalty`：token 在生成历史中出现过时，logit 固定减去该值。
- `frequency_penalty`：logit 减去“该值 × 生成历史出现次数”。
- `repeat_penalty`：保留 FastLLM 的正负 logit 乘除语义；本机生产 profile 当前为 1.08。这是本地 A/B 参数，不是上表中的 Qwen 官方推荐字段。

执行顺序为乘法 repeat penalty、加法 presence/frequency penalty、temperature/softmax、top-k/top-p。普通 CPU 采样、CUDA handoff 和 MTP exact/typical acceptance 使用同一份惩罚后分布；否则 draft 与 verify posterior 不一致，会破坏 exact acceptance。

## Chat template 真源

生产模板必须来自对应 Unsloth GGUF 的内嵌 `tokenizer.chat_template`，不得以手写模板或其他模型模板替代。

UD-Q5_K_M 内嵌模板已导出为：

```text
models/unsloth/Qwen3.8-27B-UD-Q5_K_M.chat_template.jinja
```

模板长度为 9993 字符，SHA256 为：

```text
12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce
```

UD-Q5_K_M 与 UD-Q6_K_M 的内嵌模板均已分别导出并校验：二者都是 9993 字符，SHA256 相同，且与此前保存的 Qwen3.8 official reference 逐字一致。生产 profile 指向各自 GGUF 导出的文件，以明确模板来源。

## 当前生产组合

当前生产实例为 UD-Q6_K_M、turbo3 KV、MTP3、exact acceptance、repeat penalty 1.08，并按 thinking/non-thinking 选择上表 Qwen3.8 官方采样档。Q5 保留为对照 profile；两者除权重量化档位和对应 GGUF 模板导出路径外保持相同。
