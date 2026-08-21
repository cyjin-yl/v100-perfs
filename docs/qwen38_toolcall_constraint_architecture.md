# Qwen3.8-27B 工具调用故障族：架构、根因与验证方法

日期：2026-08-21 → 08-22
栈：FastLLM (C++/CUDA, SM70/V100) + `thinking_proxy.py` (FastAPI 网关) + Unsloth Dynamic V3.0 UD-Q5_K_M
客户端：OMP (Anthropic 协议)、OpenCode (OpenAI 协议)、OpenWebUI

本文档记录一次跨 11 天会话的工具调用故障排查。**重点不是结论，而是哪些假设被推翻、怎么被推翻的** ——
本轮有四个看起来很有说服力的假设先后被自己的数据证伪，教训比结论更值钱。

---

## 一、生成期约束的完整架构

理解故障前必须先搞清约束链路。以下每一环都在本次排查中被误判过。

### 1.1 Jinja 模板 → 语法布局（编译，非硬编码）

`CompileToolCallGrammarLayout()`（`example/apiserver/tool_call_layout.h`）把带哨兵的模板渲染一遍，
从渲染结果反推结构：

```
functionPrefix        "\n<function="
nameTerminator        ">"
parameterPrefix       "\n<parameter="
parameterValuePrefix  "\n"
parameterClose        "\n</parameter>"
functionClose         "\n</function>"
toolCallClose         "\n</tool_call>"
```

**这套是编译出来的，不是写死的千问格式** —— 换 GLM / Claude 的模板时同一套代码仍成立。
任何修复都必须走 layout，不能引入模型专有字面量。

### 1.2 状态机（`src/models/basellm.cpp`）

```
TG_NONE   不在 <tool_call> 块内 / 块已完成 —— 不加掩码
S0  TG_FUNC_OPEN        期待 "<function="
S1  TG_FUNC_NAME        函数名生成中
S2  TG_PARAM_GAP        参数块之间
S3  TG_PARAM_NAME       参数名生成中
S4  TG_PARAM_VALUE      参数值生成中（自由文本）
S5  TG_FUNC_CLOSE_TAIL  "</function>" 之后
```

游标由 `LocateToolCallGrammarCursor()` 用 `rfind` 定位：若最后一个 `</tool_call>` 在最后一个
`<tool_call>` 之后 → `TG_NONE`，**不加任何约束**。这意味着一个调用闭合后，第二个 `<tool_call>` 完全自由
——**并行工具调用在语法层没有任何阻碍**。

### 1.3 掩码的唯一生效点

```
basellm::PrepareToolCallConstraint()          每步清空并重算
  └─ EvaluateToolCallConstraintText()          算出 allowed/blocked token id
       └─ AppendToolCallConstraintRowConfigs() MTP 逐行（draft/verify）各算一份
            └─ Qwen35ApplyCudaTokenConstraints()  ← 真正生效在这里
                 └─ FastllmCudaApplyTokenMask()   GPU 全词表 logits 打掩码，再做 top-k
```

**关键：`LLMSampling()` / `LLMSamplingOnly()` 在生产路径上根本不会被调用。**
排查时往这两个函数里加诊断，输出恒为 0 条 —— 白费两轮编译。

### 1.4 MTP 与批处理

`drafts_per_step=2` ⇒ 每个请求每步产生 3 行（当前 + 2 个 draft）。
多请求并发时 batch 会混合：trace 里见过 `batch=6`（两个请求 ×3 行）。
`generationConfigs[b]` 按行取，各行文本不同 ⇒ 各行掩码不同，`allow_n` 在同批内不一致是**正常的**。

---

## 二、被推翻的四个假设（重点）

| # | 假设 | 证伪方式 | 结论 |
|---|---|---|---|
| 1 | 模板 `NO suffix` / `NOT after` 禁止了第二个调用 | 复制模板放开那两句做 A/B | 照样只发 1 个（`out_tok` 151→425 但 `tool_calls` 恒为 1）。且**官方原版与 Unsloth 版这两句逐字节相同**，不是 Unsloth 加的 |
| 2 | 掩码屏蔽了 `bash` | 从 GGUF 取 248320 词表，在 Python 里原样复现 `IsToolCallEnumTokenAllowed` | `bash`(44675) **在** S1 的 14 个放行 token 里；引擎 trace 与 Python 复现分毫不差 |
| 3 | MTP 投机行与掩码错位 | `top_k=51` 关 MTP 重跑 | 输出**逐字节相同**（`todo`, out_tok=112） |
| 4 | 掩码"只能筛 top-k 不能提" | 定位到 `FastllmCudaApplyTokenMask` | 掩码在**全词表**上先于 top-k 生效，架构本身正确 |

**最终证据**：在 S1 那一步取掩码**前**的 argmax：

```
cudaMask row=3/6 raw_argmax=16859(logit=16.8750, allowed=1) allow_n=14 block_n=0
                          ^^^^^ = todo，且 allowed=1
```

模型的第一意愿就是 `todo`，掩码放行了它。**掩码是清白的。**

### 教训

1. **先确认代码路径真的被执行**，再分析它。两轮诊断加错位置，靠"trace 输出 0 条"才发现。
2. **A/B 必须排除混杂**。`prompt_tok` 两边都是 12808 才证明 prompt 真的一致（`raw_prompt=true` 生效）。
3. **日志是追加式的**，用全文件 grep 判定"加载失败"会命中旧代内容 —— 必须按生成分界切片。
4. **并发流量会污染 trace**。曾把用户 OMP 会话的 12 工具集 trace 误当成自己 9 工具集探针的输出，
   需用 `allow_n` / `allowed_values` 当判别器分离。

---

## 三、已确认并修复的缺陷

### C. 调度器静默死锁（thinking_proxy.py）—— 已修复并生产验证

**现象**
```
[metrics] streams=0/4 waiters=0 workers=0/4 sched_active=4 queued=4 backend=READY(active=0) reqs=0
```
队列只涨不消，服务完全静默地变砖，`/health` 仍报 READY。

**因果链**（探针抓到完整 traceback）
```
worker sched-worker-0 DIED: BackendLifecycleError: backend lease accounting underflow
  thinking_proxy.py:1498 in _worker  -> await lease.release()
  thinking_proxy.py:723  in _release -> raise BackendLifecycleError(...)
```
1. `reap_stale_leases()` 判定租约泄漏的三个条件是 `active>0`、`_INFLIGHT` 空、`pending()==0`。
   但 `_INFLIGHT` **只登记流式请求**，非流式 `_worker` 走 `httpx.post` 全程不在里面；
   `pending()` 只是队列长度，已被 worker 取走正在处理的请求算 0。
   ⇒ **两个判据叠加仍看不见"正在跑的非流式请求"**。
2. 长请求（12751 token 的 prompt 冷 prefill 轻易超 180s）被误判成泄漏 → `active` 强制归零。
3. worker 的 `finally` 再 `release()` → 下溢 → 抛异常。
4. **`finally` 里抛出的异常没有任何 try 接得住** → 协程当场死亡，且 `create_task` 的异常无人 await ⇒ 完全静默。
5. 4 个 worker 逐个死光 ⇒ 死锁。

**讽刺之处**：同一个 codebase 早就用"纪元法"修过**流式槽位**的同款问题
（`BackendScheduler._reap_epoch`，注释写着"纪元变了说明 reap 已经替我们释放过"），**租约这条路当时漏了**。

**修复（四处）**
1. `BackendLease` 加 `_epoch`；`reap_stale_leases` 归零时 `_lease_epoch++`，旧租约 release 变无害 no-op。
2. `_release()` 不再抛异常 —— 唯一调用点是 `finally`，抛出去就是杀调用者；改为记日志。
3. `reap_stale_leases` 判据补上 `scheduler.active > 0`，看得见非流式在途请求。
4. `_worker` 的 `finally` 全程异常安全 + `respawn_dead_workers()` 自愈 + `DEADLOCK` 显式报警。

**验证**：4 个不变量回归全过；生产重启后 `workers=4/4`、`reqs` 正常增长、`decode 40.4 tok/s`。

---

### A. 参数值退化与空值 —— 根因已定位，修复待验证

#### A-1 空值参数（`path=""` / `language=""` / `timeout=""`）

**现场**（dump 的原始 XML，54 次全同形态）
```xml
<tool_call>
<function=write>
<parameter=content>
#!/usr/bin/env bash
... 900 字节正常脚本 ...
</parameter>
<parameter=i>
                      ← 空
</parameter>
<parameter=path>
                      ← 空
</parameter>
</function>
</tool_call>
```
模型把长 `content` 写完后，剩余必填参数吐空壳。客户端收到 `path=""` → `EISDIR`，重复 5 次后挂死。

**根因**：S4 有个 `collectValueCloseBlocked` 守卫，本意是"屏蔽让全空白值形成完整 `</parameter>` 的 token"，
让空值不可表达。但它的判定是 `tail + tokenText` 是否构成完整闭合标签 ——
**当 `tail` 本身已含完整闭合标签时，追加任何 token 都仍然匹配，于是整个词表被塞进 blocked**：

```
=== 空值守卫实际屏蔽数量分布 ===
   1021 次  blocked=2
    286 次  blocked=3
    503 次  blocked=248320   ← 全词表
```

而 CUDA 掩码对"全禁"的兜底是：
```cpp
if (!anyAllowed) std::memset(row, 1, vocabSize);   // 全禁 => 全放开
```
⇒ **约束在最该生效的一步彻底失效**，且 `toolcall_mask_emptied` 恒为 0
（那个计数器统计的是 `LLMSampling` 里的另一条兜底，覆盖不到 CUDA 这条）——**完全静默**。

**修复**
1. 守卫在 `tail` 已含完整 `</parameter>` 时直接返回，不再全禁。
2. CUDA "全禁→全放开"兜底加计数器 `toolcall_cuda_mask_all_blocked`，杜绝静默。

#### A-2 参数值退化循环

**现场**
```
task = "Exercise core tools [in_progress] (Tool tests) [in_progress] (Tool tests) × 6"
```
种子是 OMP todo 工具**结果文本里的装饰**（`- X [in_progress] (phase)`）—— 模型照抄并陷入循环。

**消融矩阵**（确定性复现，temperature=0）

| 变体 | freq_pen | 回显 | `task` 参数 |
|---|---|---|---|
| base | 1.05 | 完整 | `[in_progress] (Tool tests)` ×6 退化 |
| no-echo | 1.05 | 极简 | **完全正常**的连贯文本 |
| no-freqpen | 0 | 完整 | **彻底崩坏** `ommaommaomma…`，参数名变成不存在的 `command` |
| freqpen-1.15 | 1.15 | 完整 | 退化消失，但 `task:""` 空值 + 思考文本灌进 `i` |

**结论**：`frequency_penalty=1.05` 是**承重结构**（关掉直接崩），但单纯调高不是解法（换成另一种失败）。
对症的是**限定在 S4 状态内的尾部循环守卫**。

**实现**（`ToolCallValueLoopPeriod`）：检测参数值尾部是否由同一子串连续重复构成，命中则屏蔽推进循环的 token。
阈值**按周期长度分档**，两道防误伤都是实测出来的真问题：

| 周期 | 阈值 | 理由 |
|---|---|---|
| ≥12 字节 | 2 次即截断 | 生产实测那次是 27 字节 `" [in_progress] (Tool tests)"`；正常文本几乎不会连抄两遍 12+ 字节片段 |
| <12 字节 | 6 次 | 短周期在正常代码里常见 |
| 纯空白循环 | 放行 | 否则**代码深缩进会被掐断** |
| 短周期单字符重复 | 放行 | 否则 `======`、`------` 这类 markdown 分隔线被误杀 |

单测 18/18：三种生产实测退化形态全命中，代码/SQL/正则/HTML 闭合/缩进零误伤。

**遗留**：守卫会掐断具体循环，但模型会**换变体继续退化**（`[1/3]`、`task=\"...\"` 属性注入）。
驱动力（上下文里的装饰文本）未消除。**此项尚未真正修好。**

---

### B. 工具名与意图不符 —— 未修复

**现象**（两条协议路径同时失败）
```
thinking: "The user is asking to run two independent bash commands in parallel."
实际发出: todo
```

**确定性**：3 轮零例外，`prompt_tok` 两边都是 12808（prompt 完全一致）。

| 配置 | 结果 | out_tok |
|---|---|---|
| grammar 开 + MTP 开 | `todo` | 112 |
| grammar 开 + MTP 关 | `todo` | 112（逐字节相同） |
| grammar 关 + MTP 开 | **`bash`** | 62 |

**已知**：掩码在 S1 放行 `bash`，模型掩码前的 argmax 就是 `todo`（logit 16.87，明显低于同批其它行的
25~30，说明该点分布平坦、模型不确定）。

**尚未解释**：既然掩码不改写，为何开关 grammar 会确定性地改变选择。
剩余可能：S0 阶段的强制介入改变了 token 化路径（`<function=`/`<parameter=` **在词表里都不是 token**，
必须拆片生成，trace 显示 `allow_n=1 且 allowed=0` 的强制步确实存在），从而改变隐状态。

**用户的架构判断**（记录以免丢失）：约束应当是**纠正性**而非**强制性**的 ——
它该做的是把 `Bash`→`bash`、`<fuction=`→`<function=` 这类拼写/标签错误纠回来，
而不是在模型明确要 `bash` 时把它推去 `todo`。关掉约束不是选项（长对话会大量出错）。

---

## 四、字符级语法推进器（B 的修复，对齐 llama.cpp 语义）

### 问题：逐状态前缀匹配拒绝跨边界 token

旧判定是「每个状态维护一个候选字符串集合，逐状态独立做前缀匹配」。
一个 token 若**跨过状态边界**——比如同时补完结构前缀并开启下一状态的内容——必然被拒：

```
实测分叉（grammar 开/关 逐 token 轨迹）：
  开: id=1628 'function' → id=28    '='   → id=16859 'todo'
  关: id=1628 'function' → id=21402 '=b'  → id=956   'ash'
                               ^^^^^^^^^^
  '=b' = functionPrefix 的结尾 '=' + 工具名 'bash' 的首字母
  旧判定: "\n<function=b" 不是 "\n<function=" 的前缀 → 拒
  模型被迫改走自己概率更低的拼法 '='，隐状态偏离，S1 选了 todo
  (cudaMask 诊断: raw_argmax=21402('=b') allowed=0 allow_n=1)
```

同形态出现在每个边界：`'bash>'`（名字+终止符）、`'i>'`（参数名+终止符）。

规模量化（一次工具调用的计数）：90 步约束中 17 步 `forced_steps`（放行集仅 1 个
token，模型完全没得选）、13 步 `mask_overrode_argmax`（掩码否决模型 argmax）。

### 语义：对齐 llama.cpp

llama.cpp `llama_grammar_accept_str`：**一个 token 合法 <=> 它的全部字符能被
语法从当前状态连续消费**。逐字符推进，跨边界天然支持，零特例。

vLLM xgrammar 同构（`matcher.accept_token` + `fill_next_token_bitmask`），
且多了投机解码必需的 `rollback(n)`。我们是「每步从 8KB 文本 rfind 重算状态」，
每次重算都是一次出错机会 —— 历史上 `c43ba036` 的 commitLen 错位、
投机行状态重算 bug 都出在这条路上。

### 实现（`ToolCallWalkChar`）

```
TG_FUNC_OPEN   精确匹配 functionPrefix；走完 → TG_FUNC_NAME
TG_FUNC_NAME   沿工具名前缀树；遇 nameTerminator 且名字完整 → TG_PARAM_GAP
TG_PARAM_GAP   双分支: parameterPrefix → TG_PARAM_NAME / functionClose → TG_FUNC_CLOSE_TAIL
TG_PARAM_NAME  沿参数名前缀树；完整+terminator → TG_PARAM_VALUE
TG_PARAM_VALUE 自由文本(接受任意字符)，跟踪 parameterClose 尾部匹配进度
TG_FUNC_CLOSE_TAIL 精确匹配 toolCallClose；走完 → TG_NONE
```

- **全部转移规则取自 `CompileToolCallGrammarLayout` 的 layout 字段**，
  无模型专有字面量，换 GLM/Claude 模板同样成立
- 起点由「从块开头逐字符重放」构造，不手工翻译各 case 的 partial；
  重放终点与 `LocateToolCallGrammarCursor` 的 rfind 语义互相校验
  （`WALKER MISMATCH` trace 暴露分歧）
- 重放卡住（某字符非法）时回退旧判定路径，不破坏既有行为
- 独立单测 16/16：`'=b'`/`'=t'`/`'=bash'`/`'bash>'`/`'i>'` 放行，
  `'=x'`/`'zzz'`/`'\n</tool_x'` 拒绝，参数值裸 `<` 不受影响

### 关键权衡

- 计算代价 O(词表 × token 平均长度) ≈ 旧实现的同量级（旧实现也是逐 token
  字符串比较）
- `rollback` 仍缺：投机行状态仍靠逐行重算。这是后续优化项，
  与 xgrammar 的对齐只差这一块

---

## 五、与 vLLM 的架构对比

`1Cat-vLLM/vllm/tool_parsers/qwen3xml_tool_parser.py` 处理的是**同一种 XML 格式**：

| | vLLM | FastLLM（本栈） |
|---|---|---|
| 工具调用提取 | **纯输出解析**，生成期不加掩码 | 生成期逐 token 掩码 |
| 多调用支持 | 显式支持（代码注释 "in multi `<tool_call>` scenarios"） | 语法层支持（`TG_NONE` 后不加约束） |
| 约束解码 | 独立路径（xgrammar），仅在用户显式请求 `guided_json` 时启用 | 对每个带 `tools` 的请求**无条件启用** |

我们选择了更强的约束，代价是"强制"语义带来的副作用。这是本次故障族的架构性背景。

---

## 五、验证方法与工具

### 5.1 确定性状态复现

`repro_state.py`：用截获的**真实 OMP system prompt**（12751 token，9 个工具）重建失败前一刻：
```
system + user("测试你有的各种工具")
      + assistant(todo init 调用)
      + toolResult(带装饰的清单回显)   ← 退化的种子
```
temperature=0，157 个 token 就能复现 session 01a02285 entry[10] 的污染调用。
变体：`base` / `no-echo` / `no-freqpen` / `freqpen-<x>` / `nomtp`。

### 5.2 双客户端验收

`verify_abc.sh` 覆盖两条协议路径（这很重要 —— 两边走的是不同代码）：

| 路径 | 端点 | 客户端 |
|---|---|---|
| OpenAI | `/v1/chat/completions` | OpenCode（`catvllm` provider → `127.0.0.1:8000`） |
| Anthropic | `/v1/messages` | OMP（`vllm-local-anthropic`） |

判定：
- **B**：thinking 里点名的工具必须就是实际发出的工具（把"模型意识到自己不对"变成可机器判定）
- **A**：参数值不得出现退化循环，并对 `toolcall_value_loop_breaks` 计数
- **C**：`workers=4/4`、`queued=0`、无 `DEADLOCK`、无 `worker DIED`

**已知不足**：A 的判定只查"尾部是否正好周期性重复"，被"换变体继续退化"绕过，曾误报 PASS。
判定标准需加强为"参数值是否包含来自 tool result 的装饰片段"。

### 5.3 诊断开关

```bash
FASTLLM_TOOLCALL_TRACE=1                  # 逐步状态 + 掩码前 argmax + 原始块 dump
FASTLLM_TOOLCALL_TRACE_DIR=<dir>          # 默认 projects/EzraVastLLM/logs
FASTLLM_TOOLCALL_VALUE_LOOP_GUARD=0       # 关闭 S4 循环守卫
```

trace 三行的含义：
```
state=S1-func-name partial_text=<已生成的名字前缀> allowed_values=<layout 推出的候选>
mask allowed_ids n=14 [65,68,76,83,...]          ← 放行的 token id
cudaMask row=3/6 raw_argmax=16859(allowed=1)     ← 模型原意 + 是否被放行
```
**`allowed=1` 就是护栏（纠正语义），`allowed=0` 才是改写（强制语义）** —— 这是区分二者的唯一硬证据。

### 5.4 可观测性计数器（`/props`）

```
toolcall_grammar_enabled           语法约束是否启用
toolcall_blocks_total              解析的完整块数
toolcall_malformed_total           结构破损被降级为裸文本
toolcall_repaired_total            缺闭合但块内结构完整，Flush 修复成功
toolcall_constraint_steps          约束激活步数
toolcall_constraint_masked_tokens  被 mask 掉的候选 token 累计
toolcall_mask_emptied              LLMSampling 兜底（生产路径不走这条，恒为 0）
toolcall_value_loop_breaks         S4 尾部循环守卫触发次数        【本次新增】
toolcall_cuda_mask_all_blocked     CUDA 全禁→全放开兜底次数        【本次新增】
```

**教训**：`toolcall_mask_emptied` 长期为 0 曾被当作"约束从未失效"的证据，
实际上生产路径的兜底在另一条分支且**没有计数器**。
新增的 `toolcall_cuda_mask_all_blocked` 就是补这个盲区。

---

## 七、模型与权重

| 项 | 值 |
|---|---|
| 权重 | `models/unsloth/Qwen3.8-27B-UD-Q5_K_M.gguf`（Unsloth Dynamic V3.0） |
| 张量 | 866 个，block 0–64，**自带完整 MTP 头**（`blk.64.*` 15 个，含 `nextn.eh_proj/enorm/hnorm/shared_head_norm`），无需 merge |
| 张量类型 | `{F32, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS}` —— FastLLM CUDA kernel 全部已实现 |
| 内嵌模板 | 9993 字节，与 `_official_qwen38_refs/.../chat_template_9993.jinja` **md5 相同**（`2a79880b328d`） |
| 词表 | 248320；`<tool_call>`=248058、`</tool_call>`=248059 是单 token，**`<function=` / `<parameter=` 不是** |

### 模板差异：官方原版（8952B）vs Unsloth（9993B）

| 改动 | 内容 |
|---|---|
| 合并 system | 支持多条前置 system + `developer` 角色；官方只认 `messages[0]` |
| **`high`→`xhigh` 别名** | **Unsloth 加的**，官方原版没有 → 官方模板遇到 `high` 会 `raise_exception` |
| 删除 | `multi_step_tool` 的 "No user query found" 异常 |
| tool_call 校验 | 缺 `name` 报错；`arguments` 必须是 mapping，字符串 JSON 直接报错 |

`NO suffix` / `NOT after` 两句**两版完全一致**，是 Qwen 官方设计，非 Unsloth 引入。

### 思考档位（Qwen3.8 只有三档）

模板硬编码：
```jinja
{%- if resolved_reasoning_effort == 'high' %}{%- set resolved_reasoning_effort = 'xhigh' %}{%- endif %}
{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
    {{- raise_exception('... Supported types are xhigh (default), medium, and low.') }}
```

| effort | 渲染效果 |
|---|---|
| `low` | "Reasoning effort is set to **low**. Keep your thinking brief..." |
| `medium` | **无 system 行**（中性档） |
| `xhigh` / 不传 | "Reasoning effort is set to **xhigh**. Please think carefully..." |

**关键：不传 = xhigh。** 任何映射失败都会静默升到最高档。

---

## 八、思考强度透传修复（已完成并实弹验证）

**问题**：Anthropic 端点不尊重思考强度，界面切档但 wire 上不生效。

**根因**：OMP 的 Anthropic adapter 主要发 `thinking.budget_tokens`，proxy 按预算反查档位，
而阈值是按 proxy 自己的 `_EFFORT_BUDGETS`(2048/4096/8192/16384) 划的，与 OMP 实际预算错位：

| OMP 档位 | budget | 旧判定 | 应为 |
|---|---|---|---|
| minimal | 1024 | low | low ✓ |
| low | 4096 | **medium** | **low** ✗ |
| medium | 8192 | **xhigh** | **medium** ✗ |
| high / xhigh | ≥16384 | xhigh | xhigh ✓ |

且显式 `output_config.effort` 分支**漏了 `minimal`**，`.get()` 返回 None 后一路落到模板默认 xhigh。

**修复**（`thinking_proxy.py`）
1. `_QWEN_TEMPLATE_EFFORT` 覆盖全部客户端档位，收敛到官方三档
2. `_budget_to_template_effort()` 阈值按 OMP 实际预算重划：`≤4096→low`、`≤8192→medium`、`else→xhigh`
3. `ANTHROPIC_EFFORT_TRACE`（默认开）每请求打一行判定来源

**验证**：函数级 12 个用例全过；生产实弹日志自证
```
[proxy] anthropic effort: medium via budget_tokens=8192
[proxy] anthropic effort: low via output_config.effort (output_config.effort='low' ...)
```
（后者证明 OMP 确实会显式发档位。）

**遗留缺口**：客户端 thinking 开启但既不发 `effort` 也不发 `budget_tokens` 时，仍落到模板默认 xhigh。
日志会打 `UNSET(template default xhigh)` 暴露，根治需 OMP 侧 `models.yml` 声明 Anthropic wire mode。

---

## 九、并行工具调用

**实测可行**，端到端验证通过：
```
block 类型: ['thinking', 'tool_use', 'tool_use']
  THINK: "these are independent calls, I can make them both in the same block"
  TOOL: get_weather {"city": "Paris"}
  TOOL: get_time   {"city": "Paris"}
stop_reason: tool_use
```

每层都支持：语法游标 `</tool_call>` 后失活 → C++ parser `while(true)` 收多块 →
流式序列化 `toolCallIndex++` 递增 → proxy Anthropic 转换按 `function_name` 递增 `block_idx`。

限制因素是模型对模板 "NO suffix" 的遵守程度（概率性，非硬墙）。
放开模板那两句**不会**增加调用数（已 A/B 验证），因此不建议改模板。

---

## 十、当前状态

| 缺陷 | 状态 |
|---|---|
| 思考强度透传 | ✅ 已修复，实弹验证 |
| **C** 调度器死锁 | ✅ 已修复，4 个不变量回归 + 生产验证 |
| **A-1** 空值参数 | 根因已定位（守卫全禁→兜底全放开），修复已实现，**待验证** |
| **A-2** 退化循环 | 守卫已上线并实测触发，但模型换变体继续退化，**未真正修好** |
| **B** 工具名与意图不符 | 四个假设全部证伪，真因未定，**未修复** |

**验收未通过前不得声称修复完成。** 上一轮 A 曾因判定标准过弱误报 PASS，已如实标记。
