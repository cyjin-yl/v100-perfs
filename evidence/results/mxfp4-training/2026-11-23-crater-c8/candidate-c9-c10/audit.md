# C9/C10 原始证据独立审计

审计状态：`verified`。审计本身没有执行 CUDA、编译、远端命令或归档中的代码。

冻结源提交 `ed39b0d715271e7b359672a081d758b4e4d5070f`；源 tar SHA-256 `3fb8cc55336a187f3c863eb98ae4d5089f57aabd6b8b585db85c0d09e03c1bd0`。结果 tar SHA-256 `d6775a7149d64b175ff20af085e1f9788937531561f10a80edccd5efb09462b8`。

源文件 25 项逐文件验证；结果归档 174 个普通文件逐文件验证。已核验计时运行 24 / 24。

## 同输入、完整 packed-to-FP32 比较

每项汇总同形状、同 variant 两次独立进程的全部 10 个 round；吞吐使用 `2*M*N*K/(median_ms*1e9)`。

| 形状 | 输入 | Variant | Wall ms | Wall TFLOPS | Event ms | Event TFLOPS | Wall 吞吐相对 C8 |
|---|---|---|---:|---:|---:|---:|---:|
| 2048³ | random_block_scales | C2 | 0.313140850 | 54.863073 | 0.312283206 | 55.013747 | +24.627% |
| 2048³ | random_block_scales | C8 | 0.390258150 | 44.021808 | 0.389369595 | 44.122267 | +0.000% |
| 2048³ | random_block_scales | C9 | 0.405116950 | 42.407184 | 0.404286408 | 42.494303 | -3.668% |
| 2048³ | random_block_scales | C10 | 0.350270750 | 49.047399 | 0.349456000 | 49.161752 | +11.416% |
| 4096³ | random_block_scales | C2 | 1.823881050 | 75.355218 | 1.823054409 | 75.389387 | +38.515% |
| 4096³ | random_block_scales | C8 | 2.526348350 | 54.402218 | 2.524486446 | 54.442342 | +0.000% |
| 4096³ | random_block_scales | C9 | 2.646101350 | 51.940170 | 2.645132828 | 51.959188 | -4.526% |
| 4096³ | random_block_scales | C10 | 2.312967300 | 59.421053 | 2.312134361 | 59.442460 | +9.225% |
| 8192³ | random_unit_scale | C2 | 12.313997600 | 89.289576 | 12.313134384 | 89.295836 | +50.842% |
| 8192³ | random_unit_scale | C8 | 18.574722750 | 59.193973 | 18.573967743 | 59.196379 | +0.000% |
| 8192³ | random_unit_scale | C9 | 19.385501650 | 56.718245 | 19.384726715 | 56.720512 | -4.182% |
| 8192³ | random_unit_scale | C10 | 17.165362300 | 64.054088 | 17.164460754 | 64.057452 | +8.210% |

## 正确性与边界探针

- `check-c2-baseline`：通过内容核验；46 次调用、175429 个输出。
- `check-c2-tiled`：通过内容核验；46 次调用、175429 个输出。
- `check-certified-c2`：通过内容核验；69 次调用、405930 个输出。
- `check-certified-c8`：通过内容核验；78 次调用、586154 个输出。
- `check-certified-c9`：通过内容核验；79 次调用、602538 个输出。
- `check-certified-c10`：通过内容核验；79 次调用、602538 个输出。
- `check-half2-microcheck`：通过内容核验；851968 个 case。
- `torch-baseline-tiled`：通过内容核验；119 个 case。
- `torch-certified-c8`：通过内容核验；16 个 case。
- `torch-certified-c10`：通过内容核验；16 个 case。

8192³ `random_block_scales` 独立 correctness-only 探针：accepted=`False`，reject_flags=`4`，U=25710077、V=26753427；全量比较 67108864 个输出。未计时，不能与 unit-scale 吞吐混报。

## SASS 静态证据

| Variant | 完整函数指令槽 | HMMA | HMUL2 | 寄存器/线程 | Stack / Spill store / Spill load bytes |
|---|---:|---:|---:|---:|---|
| C8 | 1664 | 256 | 0 | 119 | 0 / 0 / 0 |
| C9 | 1664 | 256 | 0 | 121 | 0 / 0 / 0 |
| C10 | 1568 | 256 | 16 | 119 | 0 / 0 / 0 |

按实际 backward-BRA 范围识别的 K-stage 循环，C8/C9/C10 分别为 620/624/524 个静态指令槽。三者保留 256 条 HMMA、48 条 LDS、16 条 STS 和两条 BAR。

C10 的循环以 16 条 HMUL2 实现精确的 half2 二幂缩放；相对 C8，IMAD 少 36、LOP3 少 48、SEL 少 32、LEA 多 4，合计少 96 个指令槽。它与减少整数解码工作的设计意图一致，但不能据此把全部实测加速归因于某条指令或推算动态执行时间。

C9 的 cursor 确实改变了 load 调度：两条 packed LDG 后剩余的静态 HMMA 为 249/247 条，C8 为 186/18 条；但 C9 的实测速度更低。提前发出 load 并不足以保证提速，不能把这份负结果省略。C10 的两条 packed LDG 相对 HMMA 的位置仍与 C8 一致。

完整 opcode、循环范围、load 位置和原始 SHA 在 JSON 中。SASS 的节点端 binary SHA 与本次编译记录交叉一致；本地不含 ELF，未声称重算 ELF 字节。

## 实际限制

- Packed-to-FP32 pipeline timings include the certificate and selected GEMM path; exclude quantization, H2D, allocation, and correctness copies.
- 4096/2048 use random_block_scales; 8192 timings use random_unit_scale. The separate 8192 block-scale correctness probe is never timing evidence.
- Two independent processes per variant/shape, five rounds each; all ten raw rounds are pooled. Within-run rounds are correlated, not ten independent trials.
- Original benchmark verifies every initial output and actual final timed certified output against baseline, with 256 independent CPU oracle positions; it does not archive full output arrays or validate every intermediate timed invocation.
- The original benchmark shares one device output and does not poison before every backend. Dedicated native halfrow fixtures include poison and canonical-zero checks; these scopes are distinct.
- 119 Torch cases cover baseline/tiled, with separate 16-case certified suites for C8 and C10. No C9-specific Torch suite or full LLM pretraining/convergence was run.
- GPU timing is historical node execution. This audit performs no GPU execution. Boundary NVML snapshots do not establish exclusive use or continuous clocks.
- Actual source/archive/log hashes were locally recomputed. ELF bytes are absent from the native results tar; binary hashes are crosschecked among manifests, stage before/after records, and completion.
- SASS static instruction/resource comparisons explain candidate mechanisms, not measured dynamic speed or per-instruction causality.
