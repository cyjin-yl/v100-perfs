#!/usr/bin/env python3
"""MTP batch>1 接受率可复现测法 (EXPERIENCE 15.29 配套; haru 编排 GPU 窗口时跑)

原理
----
后端日志每 64 次 validation 打印一行:
    [Qwen3.5 MTP] pos_accept_rate=[p0%%, p1%%]
p0/p1 是按 draft 位置的**进程累计**接受率(atomic 计数器, 自进程启动累计)。
同一档负载稳定跑几百个 decode step 后, 累计比率收敛到稳态值(波动 <0.5pp)。
因此: 分别在不同并发档下跑固定负载, 取日志**最后 3 行的均值**作为该档稳态
接受率, 对比 batch=1 基线。如需窗口精确值, 重启后端清零计数器再测单档。

用法
----
1. 由编排者(haru)起目标 profile(如 qwen38 batch=2/4), 确认 /v1/models 别名。
2. 跑本脚本:
     python3 mtp_batch_acceptance.py --url http://100.94.73.9:8000/v1/chat/completions \
         --model qwen3.8-27b --conc 1 2 4 8 --tokens 512 --rounds 3 \
         --backend-log v100-perfs/runtime/fastllm-native-profiles/logs/<active>-live.log
3. 每档: 并发 N 个相同 decode 负载(长输出, 让 MTP 充分验证), 期间 tail 后端日志
   收集 pos_accept_rate 行; 档位间冷却 5s。
4. 输出表格: conc | rounds | mean_tok_per_s(聚合) | pos0%% | pos1%% | 样本行数。

判读
----
- batch>1 接受率与 batch=1 相差 <3pp ⇒ MTP 在并发下无退化, 5-10 subagent 可开。
- 若 p0/p1 显著下滑(>5pp) ⇒ 并发 verify 路径有干扰(如典型接受阈值在
  大 batch 下分布漂移), 考虑并发时回退 MTP1 或关闭。
- 同时看聚合 tok/s: MTP 收益 = 接受率 × draft 数; 并发下即使接受率持平,
  GPU 吞吐上升也会摊薄 MTP 相对收益——以聚合 tok/s 为最终决策指标。
"""
import argparse, json, re, subprocess, sys, threading, time, urllib.request

RATE_RE = re.compile(r"pos_accept_rate=\[([0-9., %]+)\]")

def one_request(url, model, tokens, tag, results, idx):
    body = {
        "model": model, "stream": False, "temperature": 0.7,
        "max_tokens": tokens,
        "messages": [{"role": "user", "content":
            f"Write a long detailed story (request {tag}-{idx})."}],
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            j = json.loads(r.read())
        dt = time.time() - t0
        ct = (j.get("usage") or {}).get("completion_tokens", 0)
        results[idx] = (ct, dt, None)
    except Exception as e:
        results[idx] = (0, time.time() - t0, str(e)[:100])

def tail_rates(log_path, stop, sink):
    """持续 tail 后端日志收集 pos_accept_rate 行(从文件末尾开始, 只收新增)。"""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            while not stop[0]:
                chunk = f.read(65536)
                if chunk:
                    for line in chunk.decode("utf-8", "replace").splitlines():
                        m = RATE_RE.search(line)
                        if m:
                            vals = [float(x) for x in m.group(1).split(",")]
                            sink.append(vals)
                else:
                    time.sleep(1)
    except FileNotFoundError:
        pass

def run_conc(url, model, conc, tokens, rounds, log_path):
    all_rates, agg_tps = [], []
    for rd in range(rounds):
        rates, results = [], [None] * conc
        stop = [False]
        t = threading.Thread(target=tail_rates, args=(log_path, stop, rates))
        t.start()
        ths = [threading.Thread(target=one_request,
                                args=(url, model, tokens, f"c{conc}r{rd}", results, i))
               for i in range(conc)]
        t0 = time.time()
        for th in ths: th.start()
        for th in ths: th.join()
        wall = time.time() - t0
        stop[0] = True; t.join(timeout=3)
        errs = [r for r in results if r and r[2]]
        tot_tok = sum(r[0] for r in results if r)
        agg_tps.append(tot_tok / wall if wall else 0)
        all_rates.extend(rates)
        if errs:
            print(f"  round {rd}: {len(errs)} errors: {errs[0][2]}", flush=True)
        time.sleep(5)  # 档位间冷却
    # 稳态: 取最后 3 行
    tail = all_rates[-3:] if all_rates else []
    p0 = sum(r[0] for r in tail) / len(tail) if tail else 0
    p1 = sum(r[1] for r in tail) / len(tail) if tail and len(tail[0]) > 1 else 0
    mean_tps = sum(agg_tps) / len(agg_tps)
    return mean_tps, p0, p1, len(all_rates)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--conc", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--backend-log", required=True)
    a = ap.parse_args()
    print(f"conc | agg_tok/s | pos0_acc | pos1_acc | log_lines")
    base = None
    for conc in a.conc:
        tps, p0, p1, n = run_conc(a.url, a.model, conc, a.tokens, a.rounds, a.backend_log)
        if base is None: base = (p0, p1)
        print(f"{conc:4d} | {tps:9.1f} | {p0:7.2f}% | {p1:7.2f}% | {n}", flush=True)

if __name__ == "__main__":
    main()
