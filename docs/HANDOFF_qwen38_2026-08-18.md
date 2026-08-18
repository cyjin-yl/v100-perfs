# Qwen3.8-27B on doesworkstation V100 32G — 通宵工作报告

生成时间: 2026-08-18 中午
链路: `nginx:80 /v1/` → `thinking_proxy:8000` → `fastllm(VastLLM) apiserver:8002`

---

## 一、结论先行

**生产已部署解限版 Cyber Q5**:
`runtime/fastllm-native-profiles/q38-PROD-cyber-q5-mtp2-turbo4-sm70.env`
= `Qwen3.8-27B-Uncensored-Cyber-Q5_K_M + graft MTP 头` / KV `turbo4` / `MTP2` /
`batch 2` / `262144 ctx` / SM70 算子开 / 默认缓存 / `FALLBACK_ENABLED=0` /
`FASTLLM_TOOLCALL_GRAMMAR=0`。部署后生产链路复验 **matrix 20/20**。

**解限版没有可测到的降智**: 能力题 5/5(与普通版持平), 网络安全防御题 5/5 实质作答
(普通版 4/5), 0 拒答。而且 prefill 还略快。⇒ 按你的口径, Cyber 版可以上生产。

---

## 二、修好的东西(都是"接近源头"的修法)

### 工具调用: 剥了五层洋葱
| 层 | 现象 | 根因 | 状态 |
|---|---|---|---|
| L1 | 参数块不出现/标签拼错 | 约束只管工具名 | 之前已修 |
| L2 | 修了却没生效 | Qwen3.5 独立 decode 循环从未接线约束 | 之前已修 |
| L3 | **必填参数缺失** | `tool_call_required_parameter_counts` 是 `map<string,int>` —— **只记个数不记名字**, 发个可选参数就满足"块数≥required 数" | K3 已修(a13833cc) |
| L4 | **空值 `{"path":""}`** | 没有"值非空"约束, `<parameter=path>` 后可立刻闭合 | 设计已给 K3, 待修 |
| L5 | 长上下文**结构破损** | 打点式约束覆盖不到值结束/标签闭合/块边界 | 语法状态机首版有回归, 已隔离(`GRAMMAR=0`) |

L3 的实证: omp 真实负载 14 次工具调用 3 次失败, 模型自己在 thinking 里写
"I keep forgetting to include the pattern."

### proxy 层: 四个只在 agentic 负载下才暴露的生产 bug
1. **流式 read timeout 硬编码 600s** —— httpx 的 read timeout 是"chunk 间隔上限",
   长 prefill(128K 实测冷 TTFT 543s, 262K 更久)必然断流; 非流式因为走
   `QUEUE_TIMEOUT` 从未暴露, 而 **omp/OpenCode 永远流式**。
2. **流式槽位泄漏** —— 客户端在生成器被迭代前断开(Ctrl+C), `finally` 不跑,
   槽位永远收不回。实测 `streams=2/2 inflight=[]` 后所有请求排队到超时,
   **被中断几次就能把服务卡死**。已加自愈回收(占用但无在途请求 90s 后归还)。
3. **显存水位压力直接变客户端 503** —— 同一请求非流式成功、流式 3/3 拿 503。
   改为在预算内等待压力消退。
4. **`_CONTEXT_LIMIT` 在 FastLLM 模式下退回 32768** —— 它只解析 llama-server 的 `-c`。
   于是 >16K 的请求被判"超限": `FALLBACK_ENABLED=1` 时**静默转发到云端**,
   =0 时直接 429。**这就是 proto-ui 那串 429 的来源**。改为从 `--tokens` 解析(262144)。

### Anthropic 协议
- `/v1/messages` **从不传 `reasoning_effort`** ⇒ effort 恒为模板默认 xhigh, 完全不可控。
  已补 `output_config.effort` / `thinking.type` / `budget_tokens` 的推导。
- 不再硬写 `temperature=1.0/top_p=1.0` 覆盖 Qwen 推荐采样。

---

## 三、实验矩阵(两张表见 v100-perfs/results/MATRIX.md 与 docs/EXPERIENCE.md 15.32)

关键结论:
1. **Q5 明显优于 Q4**: 工具调用空值漂移随量化误差上升 ——
   Cyber Q5 **0/40**, 普通版 Q5 1/40, Q4_K_XL 各档 2~5/40。
2. **MTP 必须开**: 关掉 decode 28.4→13.5(约 2 倍), 而且**多轮工具调用开始掉线**
   (toolloop 2/2 失败、并发 4/6 失败)。
3. **SM70 算子**: 开/关在 prefill/decode 上无可测差异(此前"提升 26%"的印象来自与
   被污染的基线对比, 不成立); 但接受率显著更高且全绿, 生产选择开启。
4. **缓存"调优"是负优化**: `CPU_TIER=1 + MIN_TOKENS=4096 + DISK_MAX=200GiB`
   把前缀命中从 0.979 打到 0.869, 且 `record{ok=0}`。
5. **长上下文的真实边界**: decode 46→29→15→**5** tok/s(200/8K/32K/128K)。
   262K 的瓶颈是 attention 不是显存。**建议 agent 工作上下文压在 ~64K,
   把 262K 当硬上限**。128K 冷 TTFT 约 510s, 命中重放 44~104s。

---

## 四、缓存: 主因不是门槛也不是逐出, 是**根本没记录**

K3 按我的要求加了 `FASTLLM_PREFIX_CACHE_STATS=1`(纯测量), 一上来就定位了:
```
periodic: reqs=64 hitReqs=8 hitTok=65024/151455 (mem=65024 cpu=0 disk=0)
  miss{no-record=55 evicted=0 below-thresh=0}
  record{ok=0 rej-min=0 rej-cap=0 rej-space=0}
  resident{mem=149MB cpu=0MB disk=0MB}
```
- `record{ok=0}` 且所有拒绝计数为 0 ⇒ **记录路径根本没走到计数点**,
  推翻了"MIN_TOKENS=65536 门槛挡住了"的先验猜测。
- 二三级从未落数据(即使 `CPU_TIER=1`), 一级只常驻 149MB(装不下一个 32K 前缀)。
- **有记录时命中极好**: 单请求 9216/9768 tok = 94%; 32K 同前缀重放 prefill
  从 503 飙到 15962 tok/s, TTFT 从 75s 降到 2.4s。
⇒ 下一步是给 `TryRecordPagedCache` 的每个 early return 打原因计数(K3 已在做),
定位卡在哪一条, 再谈三级回落与自学习策略。

---

## 五、还没做完的(按优先级)

1. **L4 空值 + L5 语法状态机**(K3 在改) —— 修好后我做 `GRAMMAR=1/0` 的 A/B,
   口径: `matrix --repeat 4`(160 格)0 破损 0 循环 + `--efforts medium --repeat 8` 0 空参数。
2. **前缀缓存记录路径**(K3 在打点) —— 这是"整轮 prefix 都缓存住"的前提。
3. **fp8 + MTP 适配** —— 我先前基于无效数据下的结论已撤回, 需干净重测 n3。
4. **剩余矩阵档位**: n2(turbo3 重测) / n3(fp8 重测) / n8(普通版 Q6) / c2(Cyber turbo3) /
   c5(Cyber AWQ) / n9(普通版 AWQ)。
5. **叫醒 z3rm / proto-ui** —— 等 omp/OpenCode 长程负载确认稳定后执行。

---

## 六、运营注意事项

- **`--thinking high` 在这台机器上代价很大**: 它经 adapter 映射成 xhigh, 单轮可能生成
  近万 token、十几分钟, 期间占着 2 个流式槽位里的 1 个。agentic 日常建议 `medium`。
- **proxy 流式槽位 = 2**(与后端 `batch=2` 对齐), 5~10 并发是**排队**而非并行。
- **同一时刻只能有一个 sweep driver**: `sweep_launch` 会清理 stale apiserver,
  两个 driver 会互相掀后端, 而探针照样拿 200 —— 测量会**静默串档**(我踩过一次)。
- **不要在脚本运行时覆盖它**: bash 按字节偏移续读, 会执行到错乱的行(我踩过两次)。
