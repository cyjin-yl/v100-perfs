# 傲腾 NVMe 作为 L3/prefill-cache 盘：可行性测算（2026-08-22）

硬件背景：ASUS TUF GAMING B660M-PLUS WIFI D4（BIOS 3212）+ i3-12100 + V100 32G。
所有速度参数为本机在线实测，非纸面值。

## 结论

- **傲腾 DIMM (PMem)**：不支持。ACPI 无 NFIT 表、无 /dev/pmem*。B660 消费级
  平台不支持 Optane Persistent Memory。
- **傲腾 NVMe SSD**：支持且划算。690 元的 256G 傲腾 NVMe 做 L3 盘，
  把"多 subagent 轮转切换"从每次吃冷启动罚变成秒级恢复。

## 实测锚点

| 参数 | 值 | 来源 |
|---|---:|---|
| 重算(prefill) | 667 tok/s | 在线 recompute TPS |
| zstd 解压 | 5477 MiB/s | 在线 |
| 每 tok KV(zstd 后) | 744 B | L1 resident / trie tokens |
| decode 单流 | 14.8 tok/s | 生产日志 |

## 单页(128 tok)恢复成本

重算 192 ms；傲腾读+解压+H2D ≈ **0.1 ms**（快 ~1900×）。
HDD(现状) 首页 seek 12ms + 顺序读；多 agent 并发下寻道串行化进一步劣化。

## 多 subagent 轮转 TTFT

共享大前缀场景（128K system+history）：

| 命中 | 恢复耗时(傲腾) | 总 TTFT | vs 全冷 197s |
|---:|---:|---:|---:|
| 100% | ~40 ms | <0.1 s | ≈瞬 |
| 75% | ~30 ms | 49 s | 4× |
| 50% | ~20 ms | 98 s | 2× |

恢复本身在傲腾上可忽略——总 TTFT 由未命中部分的 prefill 主导。
真正的价值是**让 batch=4 轮转调度可行**：否则每次切回一个已挂起的 agent
都要付一次全量重算。

15K 小前缀场景：全冷 22.5s → 高命中下傲腾 TTFT ≈1s。

## 容量

每 token KV(zstd 后) ≈ 744 B：

- 当前 L2 (16GiB RAM) ≈ 23M tok
- **傲腾 256GB ≈ 369M tok** ≈ 24000 个 15K 会话，或几十个 100K+ 长会话
- 8 agent × (15K system + 50K history) = 0.52M tok —— 轻松覆盖

## 吞吐影响

GPU 仍是唯一瓶颈（decode 14.8 tok/s × batch4）。傲腾不提高 decode 吞吐，
它消除的是轮换时的 TTFT 罚，让"挂起/恢复"式多 agent 调度从"不可行"变
"可行"。聚合 decode 吞吐不变（~60 tok/s @batch4），但任务周转率大幅提升。

## 落地前提

1. restore 永久失败 bug 已修（8ebfe39a）：WriteTensor 紧凑化非 stride 连续张量
2. `FASTLLM_PREFIX_CACHE_DISK_DIR` 指向傲腾挂载点（独立挂载，勿塞进 91% 满的系统盘）
3. StorageWins 介质画像自动识别 SSD（sysfs 检测，傲腾 NVMe 天然命中）
4. MIN_HITS/MIN_TOKENS 维持现状

## 遗留

- 修复前写的 gen-1（padded 格式）无法被新代码读取——下次 checkpoint 写出
  gen-2 后由 prune 自动收掉；或手动删 root 重建
- `.staging-*` 残留由 RemoveStaleStaging 在下次提交时自动清理
