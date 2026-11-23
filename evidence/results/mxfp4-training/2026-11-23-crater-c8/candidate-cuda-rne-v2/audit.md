# CUDA RNE v2 原始归档独立审计

审计状态：**verified**。审计本身没有运行 GPU、编译器、远端命令或归档代码。

## 来源与正确性

实际 results.tar.gz 为 146,654 B，SHA `95f8691ff4f437d0c2c3f8e93cad326f28f98796274d1257a89cef961b93d45e`；源 commit `f338faaa5cbdcd900aa82ae95af00ed5fb770235`。
29 个冻结源文件与全部结果成员按实际字节重算，完整 20 个阶段的输出、日志、源码、编译与库绑定一致。RNE 归档没有 ELF；二进制 SHA 来自节点实测记录，已交叉绑定到各阶段前后记录。

C2 native 69 次调用、405,930 输出、2,950 canonical +0 通过；baseline/tiled 各46例。原 Torch 119 项与 C2 独立16项通过。
CUDA RNE 135项包括45个有限量化fixture、12个非有限拒绝/恢复、72个F/D/W recipe及6个SR路由对照。147个逐字节比较和306个数值比较逐项核验，exact记录零mismatch；FP32对照使用已声明的3e-5容差。
253个尺度的midpoint邻域、FP32二幂边界、次正规数、负零、padding、非连续/零stride、int64 stride和lazy view均有实际记录。native direct输出预先填oracle逐字节反码。

## 初始失败与唯一修复

原运行7个stage通过、1个失败、12个blocked，量化先通过44例，随后旧Torch control的`x.new_tensor`不支持lazy Negative dispatch key。原失败与同步状态没有改写。
v2源归档逐文件比较只有check_quantize_cuda.py不同：仅解析旧Torch control输入的lazy view并记录原因。native wrapper/direct仍接收原view；量化kernel、torch_ops、SR及比较标准未改。

## 两输入完整量化 wrapper

同步wall；每个backend预热10次、5轮×10次，轮内交替顺序。每轮最后实际timed codes/scales在两个timer结束后全量对独立NumPy oracle。

| 输入尺寸（两矩阵） | Torch ms | CUDA RNE ms | wall速度比 |
|---|---:|---:|---:|
| 1024² × 2 | 1.843861397 | 0.297238299 | 6.203310× |
| 4096² × 2 | 11.972170000 | 0.677043502 | 17.683014× |

这是两个完整量化调用，包含分配、验证与有限值标志同步，不是kernel-only或GEMM吞吐。输入真实SHA和全部round保存在audit.json。

## 完整 Torch 量化＋certified GEMM

每个形状按Torch→CUDA RNE→CUDA RNE→Torch进行4个独立进程；每job预热3次、5轮×10次。每种量化器合并两个进程的10个round取中位数。

| 形状 | 量化器 | 完整wall ms | useful TFLOPS |
|---|---|---:|---:|
| 1024³ | torch | 4.110380450 | 0.522453742 |
| 1024³ | cuda_rne | 2.779605650 | 0.772585726 |
| 4096³ | torch | 16.043664901 | 8.566555979 |
| 4096³ | cuda_rne | 5.451041700 | 25.213337383 |

计时包含每次context创建、workspace分配、prepare/证书/GEMM/epilogue和同步销毁。此处不能与native复用context的TFLOPS当作同一scope，也不是完整训练step。七个job的全部wall/event原始round和重算统计保留在audit.json。

## GPU 训练 smoke

cuda_tiled与cuda_certified均配CUDA RNE，各五组配方×40步；十组全部17个参数张量实际更新，参数与AdamW状态保持有限FP32，训练loss有限且验证loss下降。SR组继续使用原Torch stochastic路由。合成width-32 decoder不建立LLM训练吞吐或收敛结论。

## 限制

- No local ELF bytes in these RNE archives; actual binary hashes are node records cross-bound to build and each stage.
- Quantizer GPU capture rejection and missing-library rejection were not executed by the 135-case checker; certified GEMM capture rejection is a separate executed test.
- The initial failed check remains failed and reports all_attempted_work_synchronized=false; later diagnostic synchronized its own work.
- Quantization latency measures two complete wrappers, with allocations, validation and finite-flag synchronization; CUDA events include host submission gaps.
- Quantization uses one process per shape with five correlated rounds, no confidence interval or causal clock claim.
- Torch complete-wrapper comparison uses two processes per quantizer with ten correlated pooled rounds; input bytes are not fingerprinted across processes.
- Only final actual timed outputs per complete-wrapper job are checked, not every intermediate timed call; standalone quantization checks the final output of every round.
- Certified initial diagnostic records one fresh context; timed calls have the same per-call synchronous lifecycle, not native reused workspace timing.
- Boundary NVML samples exist for standalone quantization, not a continuous full-wrapper ABBA clock trace or exclusivity proof.
- Synthetic smoke proves finite updates for its five recipes only; it establishes no real-corpus LLM convergence, training throughput, or peak-memory benefit.
