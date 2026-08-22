# V100 FastLLM 全仓信息架构与思考控制兼容层设计

状态：已选择方案 B，等待书面设计复核后进入实施计划  
日期：2026-08-22  
适用仓库：`v100-perfs`，以及作为部署入口但不纳入本次目录迁移的 sibling `thinking_proxy.py`

## 1. 决策摘要

本次改造包含两个相互独立、最终在文档中汇合的交付：

1. 对 `v100-perfs` 做 manifest 驱动的全仓物理目录迁移，建立普通文档站的信息架构。所有迁移前已跟踪文件必须有唯一去向，不允许删除、静默去重或无记录改写；旧站点 URL 必须继续可访问。
2. 修复 OpenWebUI 及其他兼容客户端的思考开启、关闭和努力等级请求。OpenAI、Anthropic 与 OpenWebUI 方言在 `thinking_proxy` 中归一成一个 canonical result；FastLLM `apiserver` 暂不复制客户端别名策略。

生产推理不属于本次部署范围：不切换模型、不修改采样 profile、不重启当前 proxy/backend。OpenWebUI 修复完成后只提交 canonical source 和测试，实际生产同步与重启需要单独的显式部署步骤。

## 2. 已确认的当前状态

调查基线提交上的 Git tracked tree 为 446 个文件、13 个顶层目录。书面 spec 会新增 tracked 文件，因此实施时必须重新以实施基线的 `git ls-files -s -z` 生成 manifest；不得把 446 写死成实施计数。

当前站点存在以下结构性问题：

- 仓库阅读器只索引 410/446 个 tracked 文件；28 个已跟踪日志、3 个二进制、profile/exit/索引等 36 个文件缺失。
- `/docs` 是 9 个硬编码页面，未由规范化内容模型驱动。
- README、首页、sampling 页和多代 handoff 同时宣称不同生产状态。
- `EXPERIENCE.md` 和历史 incident/handoff 被当作当前主文档。
- Qwen3.5、Qwen3.6、Qwen3.8、DeepSeek-V4、MiniMax H3 的历史、设计、结果和运维混在同一叙事层级。
- `llms.txt` 有重复目标和部署后 404，workflow 的 path filter 也未覆盖所有已发布源文件。
- 没有 authority、status、model generation、canonical route 或 legacy alias 元数据。

OpenWebUI v0.9.6 已确认会发送顶层 `reasoning_effort`。当前失败位于 proxy：

- `reasoning_effort="off"` 未被识别为关闭，最后退化成模板默认 `xhigh`。
- `chat_template_kwargs.enable_thinking=false` 被无条件覆盖成 true。
- 未知 effort 被删除后静默变成模板默认 `xhigh`。
- Anthropic `output_config.effort=max` 与 `thinking.type=adaptive` 会被错误配上固定 4096 budget。
- named effort、显式 budget、模板 effort 和启停状态由 OpenAI/Anthropic 两套代码分别决定，已经漂移。

## 3. 约束与非目标

### 3.1 强制约束

- 所有实施基线 tracked 文件都有 manifest disposition。
- 不删除、不覆盖、不静默合并内容相同但路径不同的文件。
- move-byte-identical 文件保持 Git mode、字节长度和 SHA-256。
- approved rewrite 必须在 manifest 中逐项批准，不能因目录移动顺手重写任何历史证据或源码。
- 已跟踪的 `.log`、TSV、JSON、patch、编译二进制和 deployed snapshot 都是保留对象，不能因 `.gitignore` 或“看起来生成的”而删除。
- 当前生产进程、模型、profile、端口和 backend 均不触碰。
- `fastllm_adapter.py` 暂时固定在仓库根路径，因为 sibling 生产 proxy 在重启时从该位置导入。这个 safety pin 是全仓迁移的唯一运行时路径例外。
- 外部 sibling `/1CatVLLM/thinking_proxy.py` 不参与目录搬迁；canonical source 修复的生产同步属于单独部署。

### 3.2 非目标

- 不在本次实现 hard thinking token budget。
- 不把 `budget_tokens` 或 named effort 伪装成已强制执行的生成上限。
- 不改变 Qwen3.8 模型采样参数、MTP、KV、权重或模板内容。
- 不把所有历史绝对工作站路径改写成可运行命令；历史文件保持原文，canonical 文档不再引用它们作为当前步骤。
- 不为 GitHub Pages 假装提供真实 HTTP 301/308；使用静态 alias 页面，并明确托管限制。

## 4. 目标仓库拓扑

实施后的顶层结构：

```text
.github/                         # CI/CD，原位
.gitignore                       # 原位
README.md                        # 全仓入口，重写
llms.txt                         # 生成物，原位生成
fastllm_adapter.py               # 生产依赖，safety-pinned 原位
docs/                            # canonical 文档与 archive
site/                            # Astro 应用，原位但内部重构
src/
  gateway/
    thinking_proxy.py            # tracked canonical proxy source
config/
  chat-templates/                # 原 chat_templates/**
evidence/
  benchmarks/                    # 原 benchmarks/**
  results/                       # 原 results/**
operations/
  handoff/                       # 原 handoff/**
  network/frp/                   # 原 frp/**
  patches/                       # 原 patches/**
  snapshots/deployed/            # 原 deployed/**
tools/
  scripts/                       # 原 scripts/**
tests/                            # 标准测试目录，原位，引用随迁移更新
experiments/
  video-generation/              # 原 video-gen/**
```

### 4.1 Prefix mapping

除显式例外外，mapping 按以下顺序执行：

```text
benchmarks/**       -> evidence/benchmarks/**
results/**          -> evidence/results/**
handoff/**          -> operations/handoff/**
frp/**              -> operations/network/frp/**
patches/**          -> operations/patches/**
deployed/**         -> operations/snapshots/deployed/**
chat_templates/**   -> config/chat-templates/**
scripts/**          -> tools/scripts/**
video-gen/**        -> experiments/video-generation/**
thinking_proxy.py   -> src/gateway/thinking_proxy.py
```

以下路径 identity-pinned：

```text
.github/**
.gitignore
README.md
llms.txt
fastllm_adapter.py
site/**
tests/**
```

所有未匹配路径必须失败；不允许隐式 identity fallback。这样方案 B 的物理迁移范围是可审计的，而不是“尽量搬”。

### 4.2 Docs explicit mapping

现有叙事文件使用显式一对一 mapping：

```text
docs/EXPERIENCE.md
  -> docs/archive/chronicles/v100-fastllm-experience.md
docs/qwen38-sampling.md
  -> docs/reference/models/qwen38/sampling-and-template.md
docs/analysis/optane-l3-feasibility.md
  -> docs/research/storage/optane-l3.md
docs/analysis/prefill-exact-port-investigation.md
  -> docs/research/performance/prefill-exact-port.md
docs/analysis/shortprefill-optimization-plan.md
  -> docs/research/performance/short-prefill-plan.md
docs/analysis/silent-hang-diagnosis.md
  -> docs/operations/troubleshooting/silent-hang.md
docs/analysis/upstream-cherry-pick-assessment.md
  -> docs/archive/assessments/2026-08-22-upstream-cherry-pick.md
docs/benchmarks/fastllm-qwen36-legacy.md
  -> docs/archive/qwen36/performance/fastllm.md
docs/benchmarks/vllm-performance.md
  -> docs/archive/qwen36/performance/vllm-and-multi-engine.md
docs/legacy/qwen35-q4km-sm70-design.md
  -> docs/archive/qwen35/design/q4km-sm70.md
docs/ops/handoff-fastllm.md
  -> docs/archive/handoffs/2026-07-28-fastllm.md
docs/ops/handoff-deepseek-v4.md
  -> docs/archive/handoffs/2026-08-09-deepseek-v4.md
docs/ops/handoff-qwen38.md
  -> docs/archive/handoffs/2026-08-18-qwen38.md
docs/toolcall/constraint-architecture.md
  -> docs/concepts/tool-calling/constraint-architecture.md
docs/toolcall/copy-fidelity.md
  -> docs/operations/troubleshooting/tool-call-copy-fidelity.md
operations/handoff/HANDOFF-2026-08-20.md
  -> docs/archive/handoffs/2026-08-20-production.md
evidence/results/q38-sweep/MATRIX.md
  -> docs/performance/qwen38/sweep-matrix.md
evidence/results/q38-sweep/raw/toolcall_bench_2026-08-22/SUMMARY.md
  -> docs/performance/qwen38/toolcall-benchmark-2026-08-22.md
```

最后两项移动后，raw evidence 目录保留其余原始 JSON；摘要成为 canonical performance 文档。manifest 仍记录旧路径、经过 prefix 后的中间路径和最终路径。

本 spec 与实施计划在实施时按下列规则迁入 archive，保持字节不变：

```text
docs/superpowers/specs/**
  -> docs/archive/design-decisions/specs/**
docs/superpowers/plans/**
  -> docs/archive/design-decisions/plans/**
```

## 5. 目标文档信息架构

```text
docs/
  index.md
  getting-started/
  concepts/
    architecture/
    reasoning/
    tool-calling/
    caching/
  guides/
    deployment/
    openwebui/
    client-compatibility/
  operations/
    current-state/
    runbooks/
    troubleshooting/
  reference/
    api/
    configuration/
    models/qwen38/
    sampling/
    templates/
  performance/
    methodology/
    qwen38/
    comparisons/
  research/
    performance/
    storage/
  archive/
    chronicles/
    handoffs/
    assessments/
    design-decisions/
    qwen35/
    qwen36/
```

### 5.1 Authority model

每个 canonical 文档具有 manifest metadata：

```text
id
source_path
canonical_route
title
summary
status: current | supporting | historical | generated
authority: canonical | reference | evidence | archive
model_generation
updated_at
nav_section
nav_order
aliases[]
```

只有 `status=current` 且 `authority=canonical` 的内容可以出现在主导航和首页状态摘要。历史文件不会被改写成当前事实；它们通过 archive header 明确标记日期和代际。

当前生产页的唯一数据源是新增的不可变部署快照
`docs/_data/deployment-snapshots/<captured-at>-qwen38.json`，不得从 README、
handoff 或历史报告拼接。快照通过只读 `/health`、`/props`、admin state、
进程环境白名单和本地模型/template checksum 采集；禁止生成推理请求、修改状态
或重启。快照至少包含 `captured_at`、采集端点、model slug、权重文件与 SHA-256、
template 路径与 SHA-256、KV dtype、MTP drafts、exact acceptance、sampling
defaults、engine commit 和逐字段 provenance。页面明确显示 capture time，不声称
永久反映 live state。若无法采集，字段显示 unknown，禁止回退到旧文档猜值。

### 5.2 新 canonical narrative

新增或彻底重写以下页面：

- `docs/index.md`：文档入口和阅读路径。
- `docs/getting-started/overview.md`：仓库用途、当前支持面、非目标。
- `docs/concepts/architecture/system-overview.md`：proxy、adapter、FastLLM、模型和客户端的数据流。
- `docs/concepts/reasoning/control-model.md`：effort、toggle、budget、parser 的概念边界。
- `docs/guides/openwebui/reasoning-controls.md`：OpenWebUI 配置、权限、per-chat 与 model-level precedence。
- `docs/guides/client-compatibility/reasoning-fields.md`：OpenAI、Anthropic、vLLM、SGLang、TokenSpeed/SMG 对照。
- `docs/operations/current-state/production.md`：唯一当前状态权威页；只从最新 `status=current` 的不可变部署快照生成。
- `docs/operations/runbooks/proxy.md`：proxy 操作与部署边界。
- `docs/reference/api/reasoning-controls.md`：canonical resolver contract。
- `docs/reference/models/qwen38/sampling-and-template.md`：迁移现有 Qwen3.8 参考。
- `docs/performance/methodology/index.md`：区分 measurement、summary 和 historical benchmark。

README 只做项目入口，不再复制整套状态和编年史。

## 6. Lossless migration manifest

实施第一步生成不可变 baseline ledger
`operations/migrations/<base-commit>-repository-layout.json`。它只描述实施开始时
已 tracked 的对象，生成后不得重写或重新枚举。每一行至少包含：

```json
{
  "old_path": "...",
  "new_path": "...",
  "git_mode": "100644",
  "blob_oid": "...",
  "byte_length": 123,
  "sha256": "...",
  "disposition": "move-byte-identical"
}
```

允许 disposition：

- `move-byte-identical`
- `same-byte-identical`
- `approved-rewrite`
- `proven-generated-regenerate`

在任何 mutation 之前，以下 baseline 路径是唯一允许
`approved-rewrite` 的 allowlist；agent 无权自行扩展。其余 baseline
路径默认只能 byte-identical。若实施发现必须新增 rewrite，必须停止并取得新的
用户批准：

```text
README.md
thinking_proxy.py
tests/test_thinking_proxy_fastllm.py
docs/qwen38-sampling.md
site/README.md
.github/workflows/deploy-docs.yml
site/scripts/generate-repo-index.mjs
site/scripts/gen-llms-txt.mjs
site/src/components/ProtoDashboard.tsx
site/src/components/RepoExplorer.tsx
site/src/components/StatusCard.tsx
site/src/layouts/BaseLayout.astro
site/src/pages/index.astro
site/src/pages/docs/[...slug].astro
site/src/pages/read/[...path].astro
site/src/pages/repo.astro
site/src/styles/global.css
```

以下 baseline 路径是唯一允许 `proven-generated-regenerate` 的 allowlist：

```text
llms.txt
site/src/data/repo-index.json
```

`fastllm_adapter.py`、`site/package-lock.json`、所有 archive/evidence、patch、log、
binary 和 snapshot 均不在 rewrite/regenerate allowlist，必须 byte-identical。
历史/归档 banner 由 site metadata/layout 渲染，禁止插入历史文件正文。

生成时拒绝：

- unmapped path
- duplicate destination
- `..` 或绝对路径
- case-fold collision
- Unicode normalization collision
- destination 覆盖已存在未映射文件
- executable mode 丢失

迁移后要求：

```text
input_manifest_rows == migrated_legacy_rows
unmapped == 0
collisions == 0
lost_blobs == 0
unexpected_rewrites == 0
mode_changes == 0
```

新 canonical 文档、manifest 和 alias 页面是 additions，不可用来抵消丢失的旧文件。

baseline ledger 与站点内容索引是两个不同工件：

- baseline ledger 不可变，用于守恒、hash、mode 和 rollback。
- `site/src/data/content-manifest.json` 在全部变更完成后从 final Git index
  生成，覆盖所有最终 tracked 文件和新增 canonical 文档。

verifier 以 baseline `new_path` 连接 final content manifest，要求每条 legacy
对象恰好匹配一条最终记录；另独立要求 final tracked tree 的 coverage 为 100%。
不得用重生成 baseline ledger 的方式让新增文件“出现”，也不得让 additions
参与 legacy 守恒计数。

## 7. Astro 站点设计

### 7.1 数据源

`generate-repo-index.mjs` 改为从 final Git index 生成
`site/src/data/content-manifest.json`，而不是读取不可变 baseline ledger，也不再
递归 working tree。所有最终 tracked 文件都有索引记录：

- 可渲染文本：站内阅读页。
- 日志、环境证据和 extensionless 文件：安全文本或 metadata 页。
- 二进制：metadata、mode、size、hash 和 GitHub blob 下载链接；不尝试内联执行。
- 生成物：标记 generated，并显示生成来源。

不得发布未跟踪 `.env`、runtime、models、GGUF 或其他敏感工作态文件。

### 7.2 Routes

```text
/docs/<canonical-doc-route>
/evidence/<manifest-route>
/repository/<tracked-path>
/archive/<canonical-archive-route>
/read/<legacy-path>                 # alias only
/docs/<legacy-slug>                 # alias only
```

GitHub Pages alias 使用静态页面：canonical link、立即 meta refresh/JS 和可见目标链接。站点不得声称这是 HTTP 301/308。

在 mutation 前另冻结不可变 route ledger
`operations/migrations/<base-commit>-legacy-routes.json`。它从 baseline Astro
build 和当前 route definitions 采集全部 421 个页面：`/`、`/repo`、9 个
curated `/docs/*`、410 个 `/read/*`，同时记录 base `/v100-perfs`、原始 path、
percent-encoded path、trailing-slash form 和预期 canonical target。迁移不得靠
“重新扫描新站点”推断旧 URL。

alias verifier 对每行检查：

- build artifact 存在；
- `<link rel=\"canonical\">`、meta refresh、JS fallback 和可见链接指向同一
  expected target；
- target 是 canonical route，不是另一个 alias；
- alias graph 无 cycle；
- percent encoding 与 trailing slash 均可解析到同一目标。

本地 pre-merge gate 只验证 `dist/` 路径和 `llms.txt` URL 到本地 artifact 的映射。
部署后的绝对 HTTPS 状态码属于独立 post-deploy gate，不能伪装成 pre-merge
验收。

### 7.3 Navigation

主导航只保留：

```text
Overview
Getting Started
Concepts
Guides
Operations
Reference
Performance
Research
Archive
Repository
```

导航、面包屑、related links、sitemap 和 `llms.txt` 全部由同一 content manifest 生成，不能再次硬编码独立映射。

### 7.4 Link policy

- canonical 文档之间使用 canonical route。
- archive 内相对 Markdown 链接在构建时重写到 manifest 目标。
- 绝对工作站路径只在 archive 原文中保留，不出现在 canonical 操作步骤。
- `llms.txt` 使用部署后的绝对 HTTPS URL，每个 canonical 目标只出现一次。
- workflow path filter 覆盖所有 manifest publishable sources；最保守实现为仓库 tracked 内容变化均触发 docs build。

## 8. Canonical reasoning resolver

新增纯解析单元，不依赖 HTTP framework、scheduler 或模板。OpenAI 与 Anthropic
适配器只负责提取 typed candidates；resolver 只返回成功值或 typed error：

```text
ThinkingResolution {
  enabled: bool
  public_effort: off | minimal | low | medium | high | xhigh | max | null
  template_effort: low | medium | xhigh | null
  explicit_budget_tokens: int | null
  unlimited: bool
  sources: tuple[
    template_effort,
    top_level_effort,
    nested_effort,
    anthropic_output_effort,
    template_toggle,
    top_level_toggle,
    anthropic_thinking_type,
    openai_budget,
    anthropic_budget
  ]
}

ReasoningControlError {
  code: invalid_type | invalid_value | conflicting_controls
  field_path: string
  message: string
}
```

`sources` 只包含实际出现的来源，并严格按上面枚举顺序排列；不存在“最后写入者
胜出”或单一来源猜测。

### 8.1 Public mapping

```text
off / none       -> disabled, public_effort=off, template effort omitted
minimal          -> enabled, public_effort=minimal, template effort=low
low              -> enabled, public_effort=low, template effort=low
medium           -> enabled, public_effort=medium, template effort=medium
high             -> enabled, public_effort=high, template effort=xhigh
xhigh            -> enabled, public_effort=xhigh, template effort=xhigh
max              -> enabled, public_effort=max, template effort=xhigh,
                    unlimited=true
```

`none` canonicalize 为 public `off`。Qwen3.8 模板只有 disabled、low、medium、
xhigh 四个有效状态；public tier 的细分保留在 resolution metadata，不伪造模板
不存在的 tier。

### 8.2 Field schemas

| Field | Accepted JSON type and normalization |
|---|---|
| `reasoning_effort` | string or null；ASCII trim + lowercase；空串非法 |
| `reasoning.effort` | `reasoning` 必须是 object/null；effort 同上 |
| `chat_template_kwargs.reasoning_effort` | container 必须 object/null；effort 同上 |
| `output_config.effort` | container 必须 object/null；effort 同上；兼容接受 off/none |
| top-level / template `enable_thinking` | boolean or null；false/true 均为有效显式值 |
| `thinking.type` | object/null；type 仅 enabled/disabled/adaptive，大小写归一 |
| OpenAI `thinking_budget` | integer >= 1 or null；bool/float 非法 |
| Anthropic `thinking.budget_tokens` | enabled 时 integer >= 1024；disabled/adaptive 禁止携带 |

effort enum 严格为 `off|none|minimal|low|medium|high|xhigh|max`。null 表示字段
未提供，不进入 `sources`。未知 string、empty string、array、object、bool effort、
负数、零值、float 或 bool budget 均返回 typed error。这里的 “bool 非法”只指
effort/budget；toggle 的正确类型就是 bool。

不实现 literal nested `extra_body` unpacking；OpenAI SDK 的 `extra_body` 是客户端
传输机制，正常 wire 已经拍平。

### 8.3 Total resolution table

| Extracted controls | Exact result or error |
|---|---|
| none present | enabled=true, public=null, template=null, budget=null, unlimited=false；kwargs `{enable_thinking:true}` |
| one or more exact public `off/none` | disabled；kwargs `{enable_thinking:false}` |
| exact duplicate named effort | accept once；记录全部 ordered sources |
| `minimal` and `low` both present | 400 conflict；比较 public tier，不因同属 low template 而合并 |
| `high` and `xhigh` both present | 400 conflict；同上 |
| named enabled tier | 按 §8.1 exact resolution/kwargs |
| toggle=false only | disabled, public/template=null；kwargs `{enable_thinking:false}` |
| toggle=true only | enabled, public/template=null；kwargs `{enable_thinking:true}`，使用模板默认 xhigh |
| equal duplicate toggles | accept；记录全部 sources |
| unequal duplicate toggles | 400 conflict |
| toggle=false + off/none | accept disabled |
| toggle=true + enabled effort | accept effort resolution |
| toggle=true + off/none | 400 conflict |
| toggle=false + enabled effort | 400 conflict |
| `thinking.type=disabled` | 等价显式 toggle=false |
| `thinking.type=enabled` without effort/budget | 等价 toggle=true，template=null |
| `thinking.type=adaptive` without effort/budget | 等价 toggle=true，template=null |
| `output_config.effort` + adaptive/enabled | effort 决定 tier；thinking type 只确认 enabled |
| explicit budget without effort | enabled；按 <=4096 low、<=8192 medium、其余 xhigh 推断 template |
| non-max effort + finite budget | effort 决定 template；budget 只保留为 metadata/hint |
| max + finite budget | 400 conflict |
| disabled/off + finite budget | 400 conflict |
| adaptive/disabled + Anthropic budget | 400 invalid_value/conflict |

每个成功行还必须断言序列化后的 `chat_template_kwargs`：disabled 只含
`enable_thinking:false`；enabled 始终含 `enable_thinking:true`，仅当
`template_effort` 非 null 时再含 `reasoning_effort`。

### 8.4 Error ordering and HTTP envelopes

resolver 每次返回第一项错误，稳定顺序为：malformed container、field type、
enum/range、named-effort conflict、toggle conflict、effort/toggle conflict、
budget conflict。同一类按 schema 表的 field 顺序，不按 JSON insertion order。

OpenAI adapter 映射为：

```json
{
  "error": {
    "message": "<stable message>",
    "type": "invalid_request_error",
    "param": "<field_path>",
    "code": "<ReasoningControlError.code>"
  }
}
```

Anthropic adapter 映射为：

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "<field_path>: <stable message>"
  }
}
```

两条 HTTP path 均返回 status 400。adapter 不捕获后继续渲染，也不把 error
降级成模板默认 xhigh。

### 8.5 Budget semantics

当前 FastLLM 没有 capability-gated hard thinking budget。为兼容现有
Anthropic/OMP 客户端：

- 显式 budget 可在未指定 effort 时选择模板 tier：`<=4096 low`、
  `<=8192 medium`、其余 xhigh。
- 数值保留在 resolution 供日志和未来 capability 使用。
- 不把它映射成 `max_tokens`，不声称 hard cap，不向 apiserver 发送一个会被
  静默忽略的伪字段。
- canonical 文档明确标记为 tier hint compatibility behavior。

真正 hard budget 需要引擎跟踪 reasoning start/end token，并在边界强制 end
marker；这属于后续独立功能。

## 9. OpenWebUI contract

OpenWebUI v0.9.6 的 Advanced Params 会将 `params.reasoning_effort` 提升为 provider 顶层字段。修复后的 wire 例子：

```json
{"reasoning_effort":"off"}
```

结果：

```json
{
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

不得残留 backend budget，也不得落到 xhigh。

```json
{"reasoning_effort":"low"}
```

结果：

```json
{
  "chat_template_kwargs": {
    "enable_thinking": true,
    "reasoning_effort": "low"
  }
}
```

OpenWebUI model-level saved params 在其服务端晚于 per-chat params 应用，因此会覆盖 per-chat effort。文档要求：若允许逐会话调节，不得在 model params 保存固定 `reasoning_effort`。控件缺失时检查 `permissions.chat.controls` 与 `permissions.chat.params`。

## 10. Proxy 与 apiserver 边界

### 10.1 本次由 proxy 拥有

- OpenWebUI/OpenAI/Anthropic 输入别名。
- effort/toggle/budget 冲突处理。
- Qwen3.8 public tier 到 template tier 映射。
- response wire adaptation：OpenAI `reasoning_content` 与 Anthropic thinking block。
- client-specific compatibility logging。

### 10.2 apiserver 后续只拥有 canonical 能力

- canonical `reasoning_effort` schema。
- `chat_template_kwargs` 透传和模板错误。
- template capability metadata：toggle key、default、native effort values、always-on/off。
- parser capability metadata与启动时 parser validation。
- 实际实现后才暴露 `thinking_token_budget`。
- unsupported hard budget 返回 400。

当前生产 proxy 已渲染 `raw_prompt`；到达 apiserver 后修改 effort 已经无效。因此本次不把 aliases、OpenWebUI policy 或 Anthropic conversion 复制到 apiserver。

响应内部保持 reasoning、content、tool_calls 三个独立通道。reasoning 与 tool_calls 可以同时存在；不能因 `finish_reason=tool_calls` 删除 reasoning。不同 wire 字段只在 proxy 边界选择，不能同时输出 `reasoning` 和 `reasoning_content` 造成客户端重复拼接。

## 11. Tests and verification

### 11.1 Migration verification

- baseline ledger row 守恒，legacy join 一对一，final content manifest 覆盖 final tracked tree 100%。
- 每个 byte-identical 项比较 SHA-256、length、Git mode。
- route ledger 每个旧 URL 有目标正确、无 cycle 的 alias。
- canonical、alias、repository、evidence、archive routes 零内部 404。
- 未跟踪敏感文件零发布。
- `llms.txt` 无重复；pre-merge 映射到本地 artifact 全存在，post-deploy 再验证 HTTPS。
- `fastllm_adapter.py` 及其 production dependency closure 的 baseline/final SHA-256 完全相同。

### 11.2 Reasoning resolver TDD matrix

OpenAI/OpenWebUI：

- unset、off、none、minimal、low、medium、high、xhigh、max。
- `reasoning.effort` 同表。
- nested `enable_thinking=false`。
- top-level/nested toggle 冲突。
- effort/toggle 冲突。
- invalid effort string/bool/object、malformed container 和 invalid toggle type。
- OpenAI budget：1、4096、4097、8192、8193、0、negative、float、bool。

Anthropic：

- `thinking.type` enabled、disabled、adaptive。
- `output_config.effort` 全表。
- Anthropic `budget_tokens`：1（invalid）、1024、4096、4097、8192、8193、16384。
- adaptive + max。
- adaptive/disabled + budget 冲突。
- max + finite budget 冲突。
- public-tier duplicates：相同 tier 接受，minimal+low 与 high+xhigh 冲突。

每一行断言 resolver、最终 `chat_template_kwargs`、渲染 prompt 和无伪 backend budget。

### 11.3 Response tests

- disabled 不产生 reasoning channel。
- enabled tiers 保留 reasoning stream delta。
- reasoning + one tool call。
- reasoning + consecutive/parallel tool calls。
- delimiter 跨 chunk。
- malformed/unclosed marker。
- exactly one OpenAI reasoning field。
- Anthropic block order：thinking、text、tool_use。

### 11.4 Runtime boundary

所有 mutation 和测试在隔离 Git worktree 中进行。不得改写 active workspace 的
`fastllm_adapter.py` 或 sibling production proxy。允许对生产执行只读
`/health`、`/props`、admin state 和白名单进程环境采集部署快照；禁止发送生成
请求、修改状态或重启。resolver 行为验证使用纯函数、captured OpenWebUI fixture、
tracked canonical proxy source 和本地非生产测试进程。

## 12. Implementation sequence

1. 创建隔离 Git worktree；冻结实施基线、baseline ledger 与 baseline route ledger。
2. 写迁移 verifier；先证明它能发现 missing/collision/hash/mode/rewrite-allowlist 错误。
3. 只读采集带 provenance 的 deployment snapshot；不发送生成请求。
4. 用 `git mv` 执行 prefix 与 explicit mappings。
5. 更新 Python/import/script/config/workflow/Markdown 引用；不保留 deprecated path shim。
6. 建立 final content manifest、canonical docs routes、repository/evidence routes 和 legacy aliases。
7. 重写 allowlist 内的 canonical narrative、README、导航、sitemap 和 `llms.txt`。
8. 为 reasoning resolver 写失败测试。
9. 在 moved tracked canonical proxy source 中实现 resolver，让 OpenAI/Anthropic 路径共用；`fastllm_adapter.py` byte-identical。
10. 在隔离 worktree 运行 ledger/join/link/unit/build/browser verification，并比较 production dependency closure checksum。
11. 将验证通过的实施 commits 合并回当前 `main`；不向 sibling production proxy 同步，不执行生产重启。
12. 在合并后的主工作区重新运行 ledger/join、focused tests、Astro build 和 browser smoke，证明 merge 未改变结果。
13. 删除隔离 worktree，执行 `git worktree prune`，并断言 `git worktree list --porcelain` 不含实施 worktree 路径。

文档迁移与 resolver 实现由两个 subagent 独立负责；它们不得编辑同一文件。集成
边界：文档 agent 拥有 repository layout、全部 docs/site/workflow/README/llms；
effort agent 只拥有 moved canonical proxy source 与
`tests/test_thinking_proxy_fastllm.py`。effort agent 通过结果消息提供文档事实，
不直接编辑任何 docs/site/adapter 文件。

## 13. Rollback

- baseline ledger 保存 base commit、旧/新路径、mode 和 blob hash，可机械反向迁移，且永不重生成。
- final content manifest 可重建，但必须与 baseline destination 一对一 join。
- approved rewrite 以独立 commit 分隔于 byte-identical moves，便于审计和回退。
- alias 由不可变 route ledger + approved canonical mapping 生成，不能手工维护第二份旧 URL 清单。
- OpenWebUI resolver 以纯函数和协议矩阵测试隔离；旧 OpenAI/Anthropic 分支在 clean cutover commit 中删除，不保留双实现。
- 不部署生产，因此 resolver 回退不需要模型/backend 重启。

## 14. Acceptance criteria

完成必须同时满足：

- 实施基线全部 tracked 文件有唯一 disposition 和唯一目标；final tracked tree content-manifest coverage=100%。
- rewrite/regenerate 只发生在 §6 预批准 allowlist，其他 baseline 对象 byte-identical。
- 全域物理迁移按 prefix/explicit mapping 完成；identity-pinned 仅限本 spec 列表。
- lost blobs、unexpected rewrites、mode changes、collisions 均为零。
- 新站点是由 manifest 驱动的正常文档层级，不再以编年史和 handoff 作为主导航。
- current production 只有一个 canonical 权威页，逐字段来自带 capture time/provenance 的不可变快照；历史状态全部标记 historical。
- baseline route ledger 的全部旧 URL 均有 target 正确、无 cycle 的 alias。
- OpenWebUI `off` 真正关闭思考，所有 effort tiers 可调，nested false 不被覆盖。
- OpenAI 与 Anthropic 使用同一个 resolver；unknown/conflict 返回稳定 400 envelope。
- 不声称 budget hard-enforced，不复制客户端别名到 apiserver。
- active production `fastllm_adapter.py` 与 dependency closure checksum 不变；生产进程、模型和 profile 未改变，未执行部署重启。
- 构建、浏览器、resolver、link、ledger/join 与 post-deploy docs URL 验证全部通过。
