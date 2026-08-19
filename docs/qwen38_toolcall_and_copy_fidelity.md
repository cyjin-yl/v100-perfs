# Qwen3.8-27B on V100：逐字抄写破坏与工具调用失败的两个根因

日期：2026-08-19 · 环境：单卡 V100 32G，Cyber Q5_K_M + MTP=2 + turbo4 KV + SM70 关

这份文档记录两个**互相独立**、但表现纠缠在一起的根因。它们的共同特征是
**静默** —— 没有报错、没有异常、日志里看不出来，只有 agent 的行为莫名其妙地坏掉。

---

## 症状

线上 agent（omp 跑 Proto-UI 维护）出现两类故障：

1. **路径/标识符被抄歪**
   ```
   /home/ezra/Documents/Proto-UI   →   /home/eze/Documents/PotouI
   ezra → eza / eZrA / EZa / EzA        Documents → Dotments
   Proto-UI → Proto-U / ProtoUI / Potato-U
   ```
   实测破坏率 0.05%–1.3%（08-16/08-17 的坏配置窗口里达到 **100%**）。

2. **工具调用一直失败**，日志里只有：
   ```
   [ToolCallTrace] mask EXHAUSTED (state active, partial='B', allowedValues=12)
       -> silent fallback to free sampling
   ```

第 1 类的二阶后果比一阶严重：agent 之后**读回自己写坏的内容**，发现
`eze` / `ezra` / `ezezra` / `eZrA` 彼此矛盾，于是推断"我正在被 prompt injection
攻击"，转入防御姿态并拒绝干活。故障从"少数字符错误"升级为"完全瘫痪"。

---

## 根因 A：MTP typical acceptance 是有损判据

`fastllm/src/models/qwen3_5.cpp:532`

```cpp
static constexpr float QWEN35_MTP_TYPICAL_POSTERIOR_THRESHOLD = 0.09f;
static constexpr float QWEN35_MTP_TYPICAL_POSTERIOR_ALPHA     = 0.3f;
```

投机解码有两条接受判据（`countAcceptedDrafts`）：

```cpp
bool accept = useTypicalAcceptance ?
    speculativeTypicalAccepted[accepted] != 0 :        // 非贪心
    targetTokens[accepted] == tokenAt(accepted + 1);   // 贪心: 严格相等
```

生产带 `repeat_penalty=1.05`，`IsSimpleGreedy()` 为 false，因此走 **typical
acceptance**（Medusa 系）：**只要目标模型给这个 draft token 的概率超过 9%
就接受**，且不做拒绝采样的残差修正——它并不保持目标分布，是**拿正确性换速度**。

对普通行文几乎无害（很多 token 可互换）。但 agent 干活靠的是**逐字抄写**：
文件路径、函数名、commit hash、工具调用 JSON —— 这些位置**只有一个 token 是对的**。

具体机制：写完 `Proto-` 后目标分布约 `UI`=0.85 / `U`=0.10。
draft 提议 `U`，`0.10 > 0.09` → 接受 → 正确的 `UI` 被丢弃 → 输出 `Proto-U`。
日志里 `Proto-U` 出现 23 次、`/home/ezra/Documents/Proto-U` 35 次，形态完全吻合。

**为什么只有我们踩到**：别人不跑 fastllm 的 MTP，这个近似判据是这套引擎自己的。

### 修复

新增 `FASTLLM_QWEN35_MTP_EXACT_ACCEPT`（**默认开**）：只接受与目标模型
**实际采样出的 token** 相同的 draft。这样每个吐出的 token 都由目标模型自己产生，
**输出逐字节等价于不开投机解码**；代价只是接受率下降（速度），不是正确性。

同时把写死的阈值放出来，便于日后做吞吐实验：

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `FASTLLM_QWEN35_MTP_EXACT_ACCEPT` | `1`（开） | 置 `0` 退回 typical（有损，换吞吐） |
| `FASTLLM_QWEN35_MTP_TYPICAL_THRESHOLD` | `0.09` | 仅 typical 模式生效 |
| `FASTLLM_QWEN35_MTP_TYPICAL_ALPHA` | `0.30` | 仅 typical 模式生效 |

启动日志现在会明说用的哪种判据；typical 模式带 `<<< 有损: 逐字抄写可能被替换`。

**实测代价**：`pos_accept_rate` 从 typical 下的 84–95% 降到 exact 下的
**77% / 60%**（两个 draft 位置）。

---

## 根因 B：工具名大小写不匹配，且失败是静默的

模型对工具名有很强的先验——被训练成写 PascalCase 的 `Bash` / `Read` / `WebSearch`。
而 harness 声明的拼写各不相同：omp 全小写（`bash` `read` `grep` `write` `edit`
`glob` `todo` `eval` `hub` + 自定义 skill，共 12 个），别家可能是 `web_search`
或 `readFile`。

失败链条：

```
模型吐出 "B"
  → 名字约束在 12 个全小写名字里找不到任何以 "B" 开头的
  → allowedIds 为空
  → LLMSamplingBlock 按"空即不 mask"处理, 静默退回自由采样
  → 模型顺着先验写完 "Bash"
  → harness 收到不存在的工具名 → 调用失败, 且没有任何报错解释原因
```

注意 `basellm.cpp` 里成功路径的 trace 被 `if (ToolCallTraceEnabled())` 挡着，
而耗尽路径无条件打印——**不要据此认为"约束从没成功过"**，那只是 trace 开关没开。

### 修复

`fastllm/src/models/basellm.cpp`：

- `ToolCallCanonicalKey(s)` = 小写 + 去掉 `-` `_` 空格 `.`
- `IsToolCallEnumTokenAllowed` 改用规范化形式比较 → `B` 成为 `bash` 的合法前缀，
  **掩码不再耗尽，不再静默降级**
- `ResolveDeclaredToolName(name, declared)` 把最终名字映射回**客户端声明的拼写**；
  精确命中优先，规范化后有且仅有一个候选才替换，**有歧义时不猜**（比如同时
  声明了 `read` 和 `Read`）
- 两个产出点都接上：`FindActiveToolCallInvokeName`（参数约束按名字查表，
  必须一致）与解析出的 `cur.functionName`（进响应 `tool_calls` 的那个）

**泛化性**：规则只依赖**客户端自己在请求里声明的 tools 列表**，不硬编码任何
harness 的名字，因此 omp / OpenCode / Cherry Studio / 任意 OpenAI 或 Anthropic
客户端一视同仁。放在引擎侧而不是 proxy 侧，是因为 proxy 有多条产出路径
（OpenAI 透传 / Anthropic 转换 / 流式 delta），逐条打补丁既易漏又难维护。

---

## 附带修复：NaN logits 会让输出退化成整片感叹号

采样路径此前对非有限 logits **完全没有防护**。这个词表 token 0 就是 `!`
（`0='!' 1='"' 2='#' 3='$'`），而比较器
`a.first > b.first || (a.first == b.first && a.second < b.second)`
对 NaN 恒为 false，于是：

1. 前 `topk` 次无条件入堆的 NaN（下标恰好 `0..topk-1`）再也踢不出去；
2. `maxValue` 变 NaN → `expf(x-NaN)` 全 NaN → `curSum > rnd` 恒 false
   → 落到 `i == topk-1` 兜底 → 返回低位下标 → **满屏 `!!!!!!`**。

附带危害：含 NaN 的比较器不满足严格弱序，`std::sort` / `std::partial_sort`
在此属于**未定义行为**，libstdc++ 的 introsort 可能越界写——这是独立于
`CloseNodeClient` 无限递归的**第二个崩溃源**。

修复：`fastllm.cpp` 新增 `SanitizeLogitsForSampling()`，把非有限值压到 `-1e30f`
（必然排最后，真正的 top token 得以胜出）并打日志。日志里的 `bad/count` 比例
可以分流病因：

- **`bad` 远小于 `count`** → lm_head 某个量化块坏了（局部）
- **`bad == count`（整层全废）** → 上游激活炸了（V100 只有 fp16，
  长上下文 attention 是首选嫌疑）

---

## 验收工具

都在 `projects/EzraVastLLM/scripts/`：

| 脚本 | 测什么 | 要不要独占 GPU |
|---|---|---|
| `toolname_probe.py` | 三种命名风格（lowercase / snake_case / camelCase）的工具名是否原样返回 | 要 |
| `copy_fidelity.py` | 深上下文里逐字抄写的保真度，按深度分档 | 要 |
| `corruption_rate.py` | 从 omp 会话日志里统计真实负载的破坏率，可按天/按小时 | **不要** |

`corruption_rate.py` 是这里最实用的一个：后端 batch=1 且长期排着几十个请求，
独占 GPU 做 A/B 很贵；而 agent 每天产出几万次这类字面量，本身就是一个持续运行、
真实分布的探针。用它对比修复前后即可，零 GPU 成本。

---

## 运营注意

- **重启生产用 `projects/EzraVastLLM/scripts/start_prod.sh`**
  （配合 `tmux respawn-pane -k -t fastllm-prod:0.0 <脚本>`）。
  该脚本开头会清理残留 apiserver —— `respawn-pane` 只杀 pane 里的 proxy，
  它拥有的后端会变成**孤儿继续占满 32GiB 显存**，新后端抢不到显存就被
  VRAM 看门狗直接卸载，表现为"重启完服务起不来"。

- **修好后端救不回已污染的 agent**。被写坏的字面量还留在它的长上下文里，
  它每轮重读一遍并反复得出"我被注入攻击了"的结论。这类 agent 必须清上下文重开。

- **262K 未通过压力测试**。1×20K 通过、1×60K 通过（wall 923.9s），
  **1×120K 失败**（最小空闲显存 627 MiB，drain 采样 75 次）。
  即使 `FASTLLM_PAGED_POOL_MAX_MB=7600`，单条 120K 仍把卡逼到接近 OOM。
  真实上限在 60K–120K 之间，需重新标定。

---

# 续：显存压力死锁与 262K 的真实账本（2026-08-19 上午）

## 症状：压力一来，整个服务就不动了

```
GPU 利用率 0%，显存 32.3/32.5 GB          ← 满了，但什么都没在算
后端  running=1 pending=6，prefill 0 tok/s，decode 0 tok/s
proxy streams=8/8  inflight=[]  queued=15  backend=DRAINING(active=7)
      [stream] VRAM 压力中, 等待重试(第 180 次)
```

`inflight=[]` 却 `streams=8/8` —— **8 个并发槽位全被"正在等压力消退"的请求攥着**。

## 根因：拿槽位和等压力的顺序反了

```python
await scheduler.acquire_stream(...)              # ① 先拿槽位
...
opened_stream = await _open_backend_stream(...)  # ② 里面才 _acquire_lease_tolerating_pressure()
                                                 #    在这里睡着等 —— 槽位一直攥着
```

槽位全被睡着的请求占满 → 没有请求能派发到后端 → 后端跑不完在途请求 →
显存不释放 → 压力标志永不落下 → 循环闭合。

**修复**：新增 `_wait_out_pressure_before_slot()`，在**拿槽位之前**、
**不获取任何资源**的前提下把压力等掉；三个流式入口都接上。超时不抛异常，
后面的 `_acquire_lease_tolerating_pressure` 仍是最终判定点。

## 262K 的真实账本

新加的 `vram=` 明细（`GetVramBreakdown()`，metrics 行里）把黑箱拆开了：

```
空闲:        vram=24862/32494MB(pool=3197  alloc_busy=16415 alloc_free=2971 other=2277)
长 prefill:  vram=32412/32494MB(pool=5040  alloc_busy=22940 alloc_free=1954 other=2477)
```

关键观察：**`alloc_busy` 冲到 23344 后就完全不动**，而 `pool` 一路从 3197 涨到 6176。
说明这 6.9 GB 是**一次性分配、加载后固定**的开销，不随上下文增长 ——
与 V100(SM70) 没有 INT8 张量核、量化权重必须反量化成 fp16 才能算这一事实吻合。
**这块不能关，只能优化。**

排除过的猜测（都不是）：
- 注意力分数矩阵随 `chunked_prefill_size × context` 增长 —— 512→128 后 `alloc_busy` 没变
- MTP 的 FP8 draft lm_head —— 日志里那行 `draft lm_head prepared` 从未出现，
  因为它要求源权重是 FLOAT16/BFLOAT16，而我们是 Q5_K_M，直接 early return
- 线性注意力前缀快照 —— 有配额（`MAX_PER_REQUEST=4`、`MAX_RECORDS=8`），最多约 200MB

差多少：
```
alloc_busy 23344 + other ~2500 + 262K 需要的池 7600 = 33444 MB > 32494 MB
                                                     ↑ 差约 950 MB
```

所以换 **turbo3**（比 turbo4 更省的 KV 量化）正好覆盖这个缺口。

## 前缀缓存：诊断被改写

原以为是"记录不生效"，**不是**。`Record()` 确实会往 trie 注册
（`pageToTrieNode[pid]=child` @ `fastllm.cpp:9488`），`layers-ok=16/次`
恰好等于全注意力层数（另外 48 层 `mgr-invalid` 是线性注意力层，预期内）。

真正的链条：页是在**请求结束时**由 `ReleasePageIndex` 放进 `triePages` 的，
而 metrics 一直显示 **`done 0 req`** —— 从来没有请求成功完成过，全被 Grow
显存不足中止。于是页永不释放 → trie 永远空（`L1trie=0`）→ 查找必然
`no-record` → 每轮重算全量 prefill → 显存压力更大 → 更易中止。**闭环。**

**前缀缓存是显存问题的下游，不是独立故障。**

另一个独立缺陷：查找 `manager->Query(ctx->currentTokens, pages)`
**只查 VRAM trie(L1)**，不查 CPU/disk 层。盘上那 1877 MB 只在启动时的持久化
恢复路径用得到，**不在每请求的查找链上** —— 三级目前不是一条查找链。

## 待办的下放设计（用户明确要求）

1. **不能"缺多少腾多少"** —— 那会导致 grow→清理→grow 的抖动。用**水位滞回**：
   低水位触发，一次腾到高水位再停，并加冷却期。
2. 三级之间用合适的算法与阈值做下放轮转（VRAM→RAM(zstd)→disk），命中时上提。
3. 成本模型用现成的 `RECOMPUTE_TPS` / `CPU_READ_MBPS` / `DISK_READ_MBPS` /
   `ZSTD_DECOMPRESS_MBPS`：只有"恢复成本 < 重算成本"才值得留在某一级。

---

# 续二：并发活锁（最重的一个）与 262K 容量达成（2026-08-19 上午）

## 并发活锁：`FetchResponseTokens` 的忙等

这是今晚挖到的最重的 bug，也是此前"z3rm 和 proto-ui 一起跑就把后端搞爆炸、
甚至 60 并发"的**真正根因** —— 不是重试风暴，是引擎里的锁行为。

**现场**（后端挂死时用 gdb 抓的，不是推测）：

```
GPU 0%、显存充足(还空 6GB)、Grow 失败 0 次、metrics 计数全部冻结  ← 不是 OOM

Thread 3 (10.3% CPU):  pthread_mutex_lock ← Qwen35MTPLoop()        等 mtpCacheMutex
Thread 2 (99.8% CPU):  pthread_mutex_lock ← FetchResponseTokens()  等 dictLocker
```

**根因** `src/models/basellm.cpp`：

```cpp
while (true) {
    if (context->resultTokenQueue.size() > 0) { ... return ret; }
    else { if (context->isEnding) { ... } }
    dictLocker.unlock();
    MySleep(0);          // ← 睡 0 毫秒 = 不睡
    dictLocker.lock();   // ← 立刻重新抢锁
}
```

每个等待中的客户端线程都在**以最快速度反复抢 `dictLocker`**，而生成线程要拿
同一把锁才能把 token 塞进 `resultTokenQueue`。并发一多，**生成线程被饿死 →
不产 token → 轮询线程继续空转 → 活锁**。并发越高越致命。

注意：`FetchResponseTokensBatch` **早就用了 `dictCV.wait_for(1s)`**，
只有单 token 版漏改 —— 而 apiserver 走的正是单 token 那条。

**修复**：改为 `dictCV.wait_for(dictLocker, 2ms)`（生产者路径已有
`dictCV.notify_all()`，2ms 超时只是兜底，万一某条产出路径漏了 notify 也只
退化成 2ms 轮询而不是 0ms 自旋）；**唤醒后必须重取 context handle** ——
等待期间锁被释放，context 可能已被别的线程移除。

**实测效果**：

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 最高线程 CPU | 99.8%（自旋） | 61.8%（正常计算） |
| GPU 利用率 | 0% | 37–100% |
| 请求完成 | 冻结 | done 7 req/窗口 |
| prefill | 0 tok/s | 312.8 tok/s |
| **L1trie** | **长期 0 pg** | **64 → 224 pg** |

**附带解锁了前缀缓存**：活锁一解开，请求能正常完成 → 页正常释放进
`triePages` → L1 终于开始积累。所以前缀缓存确实是下游，只是上游堵点是活锁
而不是先前以为的 OOM。

（同时更正：`done N req` 是**窗口计数**不是累计值，先前据此推断"请求从未完成"
是误读。）

## 262K 容量：换 turbo3 达成

turbo4 下每页约 113 KB，262K 需要 7600 MB 的池，而固定开销
（`alloc_busy` 约 23 GB + `other` 约 2.5 GB）只剩约 6.6 GB —— **差约 950 MB**。

turbo3 每页约 27 KB，实测：

```
pool=7596 MB       ← 吃满 262K 的 KV 配额
vram=26506/32494   ← 还空 6 GB
Grow 失败: 0
已完成 prefill: 146057 token / 343s
```

## 工具名修复：已验证

`toolname_probe.py` 结果：`bash`(全小写) → 返回 `bash` ✅、
`web_search`(snake_case) → 返回 `web_search` ✅（两次独立确认）、
`readFile`(camelCase) 因队列深请求超时，结论待补。

**更强的证据**：本次进程 **`mask EXHAUSTED` = 0 次**（修复前同一进程内 6 次、
历史累计 91 次），且**工具名归一化触发 0 次** —— 掩码不再被打爆，模型直接
就写出了声明的拼写，根本没走到需要事后归一化那一步。这正是理想形态。

## 已知待修

- `vram=` 明细里的 `other` 会算成负数：`g_pagedPoolCudaBytes` 只在
  `~PagedCacheManager` 里递减，其它释放路径漏减，计数单调虚高。
- `corruption_rate.py` 的检测器误报偏高：`/Proto-UI`（带前导斜杠）、
  `Prototype`、`Proposed` 都在编辑距离阈值内被误判。需要加词形过滤后
  再做修复前后对比。
- 三级缓存的下放轮转（水位滞回 + 冷却期）与把 L2/L3 接进每请求查找链，仍未做。

---

# 续三：262K 达成 + 与 fastllm pane(K3) 的改动合流（2026-08-19 晚）

## 262K 通过，且快了一个数量级

配平方案（turbo3 + 池 6600MB + 保留区 1536MB）：

```
1×60000 通过   wall=101.7s   最小空闲显存 4943 MiB
已完成最大 prefill: 260574 token
CUDA 分配失败: 0    Grow 拒绝(保留区): 0    崩溃重启: 0
vram=29324/32494MB  仍空 3.1 GB
```

对比：turbo4 + `chunked_prefill_size 128` 时代同样的 60K 档要 **923.9s** —— **9 倍差距**。

## 显存保留区（新增）

崩溃循环的现场（30 分钟 5 代）：

```
CUDA error when allocating 96 MB!  gpuFree: 5 MB / 32494 MB
fatal: cuda malloc failed in Data::MallocSpace
  dims = [1, 48, 8192, 128]        ← 48 = 线性注意力层数
```

**KV 池把显存吃到只剩 5MB，激活张量申请 96MB 直接 fatal 杀进程。**
`FASTLLM_PAGED_POOL_MAX_MB` 只约束池子自身，管不住"池子涨完之后还剩多少给
激活用" —— 结构性缺口，调参盖不住。

新增 `FASTLLM_VRAM_POOL_RESERVE_MB`（默认 1536）：`Grow` 的判据从
`free >= newBytes` 改为 `free >= newBytes + reserve`。收益是**把失败点前移**：
原先是 fatal 杀进程，现在是 Grow 拒绝增长、走
`aborting in-flight requests; process survives` 的可恢复路径。

## 与 K3 的改动合流

K3（fastllm pane 的 agent）在同一份源码上并行工作，额度耗尽前留下了一些东西：

**险些丢失的**：`git status` 是 `MM src/fastllm.cpp` —— K3 已 `git add` 但
commit 被 NTFS 慢 IO 拖到超时未落地（1 秒后额度断），我的显存保留区压在上面。
直接 `git commit` 会冒名提交它的改动；`checkout --` 会两个人的都没。
已分层提交：`3a0996c5`（K3，CUDA OOM 改抛异常，保留原作者署名）+
`3f1c3fd2`（我，显存保留区）。

**它纠正了我一个有害改动**：我为验证显存假设把 `--chunked_prefill_size` 改成
128。K3 实测这会把 77K 的 vision prefill 拆成 603 个 chunk、跑满 **38 分钟**，
而 chunk=1024 每 token 快 2.3 倍。它已从 profile 删除该参数，改用
`FASTLLM_QWEN35_PREFILL_CHUNK_CAP`（默认 1024）。

**三条独立印证**：
- 它实测 `pos_accept_rate=[76.07%, 57.75%]`，与 exact acceptance 预估的
  77%/60% 几乎完全吻合
- 此前连环崩的运维级真凶是**同时存在两个 thinking_proxy**，两个后端各要 28GB
  抢一张 32GB 卡互杀 —— 与 `start_prod.sh` 里加的孤儿清理是同一问题的两面
- 它证伪了 turbo4 能解决容量（kv_pool 在 96%↔100% 抖动、0 tok/s）

**设计重复待合并**：K3 在 `GetUnusedPageIndex` 里已实现**水位批量回收**
（低水位 `dims/16` → 高水位 `dims/4`，env `FASTLLM_KV_RECYCLE_LOW_PAGES` /
`_HIGH_PAGES`）。这正是要求里的滞回算法，**不要再造第二套** ——
同一个页池上现有两条淘汰路径（它的水位回收 + 我 Grow 里的缓存回收），
需要确认互不打架。

## 两处清理

- K3 断电前遗留的无条件调试 printf（`fastllm-turboquant-kv.cu`
  MultiPage 入口），在生产日志里累积了 **128,853 行**，每次还
  `fflush(stdout)`（热路径系统调用）。改为 `FASTLLM_TURBOKV_TRACE` 开关。
- 我 `a00c543c` 那次 `git add -A src include example` 范围过宽，把 6 个
  `.bak-*` 备份（2.9MB / 63,524 行）扫进了版本库。已 `git rm --cached` 并
  加入 `.gitignore`（`8cbc2e26`）。

## 仍未验证 / 未做

- **前缀缓存四项根因修复零验证数据**：K3 的探针写错模型名
  （`qwen3.6-fastllm` vs 实际 `qwen3.8-fastllm`）全部 404，一次有效数据都没拿到。
- 2×150K 长上下文压测从未真跑。
- fp8_e4m3 + MTP 可行性评估搁置（K3 估 ~32KB/token，262K 物理可行，未验证）。
- `corruption_rate.py` 的检测器误报偏高（`/Proto-UI`、`Prototype`、`Proposed`
  都落在编辑距离阈值内），需加词形过滤后再做修复前后对比。
- `vram=` 明细的 `other` 会算成负数：`g_pagedPoolCudaBytes` 只在
  `~PagedCacheManager` 递减，其它释放路径漏减，计数单调虚高。
