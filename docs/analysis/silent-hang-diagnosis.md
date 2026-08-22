# 后端「静默僵死」的诊断手册

2026-08-20 的一次生产事故：后端进程活着、显存照占，但**一个请求都不再完成**，
持续约一小时才被发现。本文把当时用到的判据和手法固化下来，下次几分钟定位。

## 一、怎么认出是这一类故障

看后端 metrics 这一行，同时满足以下几条基本就是：

```
[metrics] running=1 pending=2 (total=21) | prefill 0 tok (0.0 tok/s) | decode 0 tok (0.0 tok/s)
          | done 0 req | kv_pool=49728/65536 pg (76%) | vram=27914/32494MB
```

- `done 0 req` —— **最强判据**。自启动以来零完成，不是"慢"，是"停"。
- `prefill` 与 `decode` 同时为 0 tok/s，但 `running` 非零。
- GPU 利用率 0%，显存却照占（`nvidia-smi` 看得到）。
- 代理侧 `queued=` 单调上涨，永不回落。

**反例**：`running=1 pending=1` 且 `decode 442 tok (29.5 tok/s)` 是正常繁忙，不要误判。

## 二、三条命令定位到具体锁

### 1. 谁在等锁

```bash
PID=$(pgrep -f 'build-rw/apiserver' | head -1)
gdb -p $PID -batch -ex "set pagination off" -ex "thread apply all bt 8"
```

关注两点：
- 有没有线程停在 `__lll_lock_wait` ← 它在等一把 mutex。
- **有没有任何线程在跑模型前向**。如果所有其它线程都在 `__syscall_cancel_arch` /
  `pthread_cond_wait` / `accept`，说明没人干活，是死锁不是慢。

### 2. 这把锁的持有者是谁（决定性一步）

glibc 的 `pthread_mutex_t` 前 5 个 int 依次是
`__lock / __count / __owner / __nusers / __kind`。
`__lll_lock_wait` 的第一个参数（`$rdi`）就是这把锁的地址：

```bash
gdb -p $PID -batch -ex "set pagination off" -ex "thread N" -ex "frame 0" \
  -ex "printf \"MUTEX=%p owner=%d lock=%d kind=%d\\n\", \$rdi, *(int*)(\$rdi+8), *(int*)\$rdi, *(int*)(\$rdi+16)"
```

- `owner` **等于该线程自己的 LWP** ⇒ **自死锁，实锤**，不需要再猜。
- `kind=0` ⇒ 普通非递归 mutex，同线程二次加锁必然永久阻塞。

对照 `ls /proc/$PID/task/` 拿到各线程 LWP。

### 3. 锁在哪一行

没有调试信息时，用「函数内只有几个 `std::mutex::lock()` 调用点」来收敛：

```bash
gdb -p $PID -batch -ex "set pagination off" \
  -ex "disassemble _ZN7fastllm12Qwen3_5Model13Qwen35MTPLoopEv" \
  | grep -E "call.*_ZNSt5mutex4lockEv"
```

再用 `info symbol <栈上的返回地址>` 得到 `函数名 + 偏移`，和上面列出的调用点偏移对上，
就知道是第几个加锁点。然后回源码数「该函数里的裸 `.lock()` 有哪几处」即可锁定。

**注意**：`std::unique_lock::lock()` 在已持有时会**抛异常**（`resource_deadlock_would_occur`），
不会死锁；所以能死锁的一定是**裸 `std::mutex` 的 `.lock()`** 或 `lock_guard/unique_lock` 的构造。
这条能一下子排除掉一半候选。

## 三、地址落在哪，能区分是哪个对象的成员

```bash
python3 -c "
addr=0x55b8a6993898
for line in open('/proc/$PID/maps'):
    rng=line.split()[0]; lo,hi=[int(x,16) for x in rng.split('-')]
    if lo<=addr<hi: print(line.rstrip())
"
```

- 落在**二进制映射之后的匿名段**（如 `0x55b8a...`，二进制在 `0x55b894...`）⇒ 主堆 ⇒
  大概率是主线程 new 出来的对象（例如模型对象）的成员锁。
- 落在 `0x7f...` 的匿名段 ⇒ 某个线程的 malloc arena ⇒ 对照日志里打印过的对象指针
  （例如 `mgr=0x7f6eb37b2400`）判断归属。
- 落在二进制自身的数据段 ⇒ 函数静态/全局锁。

## 四、本次的根因（一个值得记住的模式）

```cpp
auto &forwardLocker = model->forwardLocker;   // 裸引用, 没有 RAII
...
forwardLocker.lock();
    ... 批前向 ...        // 内部 PagedCacheManager::Grow 显存不足时会抛异常
forwardLocker.unlock();   // 异常展开会跳过这一行
```

异常一抛，锁永久不释放；外层 catch 打印「process survives」后 `while` 继续，
下一轮再 `lock()` 即自死锁，而且线程是**攥着锁**死的，于是所有客户端线程
堵在 `FetchResponseTokens` 上陪葬。

**间歇性来自状态**：只有异常恰好抛在 lock/unlock 窗口内才死。当时日志里
`Grow` 抛了 6 次，前 5 次都活了下来，第 6 次才僵死。所以它看起来像"偶发玄学"，
不要因为"上次没复现"就放过。

修法：`std::unique_lock<std::mutex> x(m, std::defer_lock);` —— 使用点
`.lock()/.unlock()` 一个字不用改，但异常展开会析构释放；真重复 lock 也会抛异常
而不是静默挂死。**有报错永远好过静默僵死。**

## 五、为什么拖了一小时才发现（配套修复）

后端的 `maxActivateQueryNumber = min(256, --batch)`，生产 `--batch 1` ⇒ 1，
而派发闸门对**所有路由**一视同仁 —— 只要有一个请求在生成，`/health`、`/version`
就一直排队到客户端超时。于是上游代理**永远分不清"后端在忙"和"后端已僵死"**，
死锁期间代理始终显示 `backend=READY`。

已修：这些元数据路由走独立队列与独立并发计数，不占推理额度（commit 06a0ed53）。
`/admin/*` 不在其列 —— 它们会 suspend/resume 模型，必须串行。

**教训**：健康检查如果会被业务负载阻塞，它就不是健康检查。

## 六、顺带记一个坑

`/v1/models` 在这个后端**根本没实现**（路由表里只有 `/health` `/version` `/props`
`/config.json` `/generate` `/v1/chat/completions` `/admin/*`），用它做健康探测
只会得到 HTTP 000，与"僵死"无法区分。探活请用 `/health`。
