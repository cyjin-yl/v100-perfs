#!/usr/bin/env python3
"""按 agent 循环的**真实形态**验证前缀缓存: 上下文单调增长, 每轮完整包含上一轮。

为什么这个形态才是对的(之前的测法是错的):
  agent 每做一次工具调用, 都会把**之前全部 turn + tool_call + tool_result**
  重新发一遍。所以:
      第 1 轮  [system, user]
      第 2 轮  [system, user, assistant(tool_call), tool_result]
      第 3 轮  [system, user, assistant, tool_result, assistant, tool_result]
  **第 N 轮的 prompt 严格包含第 N-1 轮的全部内容作为前缀。**
  这不是"多个 agent 共享同一段 system prompt"的并行复用, 而是"同一个 agent
  把自己的历史一次次重发"的顺序复用 —— 后者才是绝对主导的流量形态。

  理论上第 N 轮只需要 prefill "新增的那一段", 前面全部命中。token 级命中率
  应当逼近 90%+。生产实测只有 1.2%, 差距大到不可能是调优问题。

判据(每轮打印, 不用事后猜):
  new_tokens   本轮相对上一轮**新增**的字符/token 估算
  prefill_tok  后端实际 prefill 的 token 数(从日志抓)
  两者接近  = 前缀被完整复用
  prefill_tok 接近整个历史 = 前缀完全没命中, 每轮都在全量重算

用法:
  python verify_prefix_cache.py --rounds 6
"""

import os

# ---------------------------------------------------------------------------
# 绕过代理访问本机。
# 这台机器的 ~/.zshrc 里 export 了 http_proxy/https_proxy=127.0.0.1:10808
# (给 HuggingFace 上传用)。Python 的 urllib **不会**自动把 localhost 排除在
# 代理之外 —— 于是打 127.0.0.1:8000 的请求会被塞进那个代理转一圈:
# 请求送得到(后端确实会处理), 但响应卡在代理回程上, 表现为"探针挂住不返回",
# 而后端日志里明明写着 prefill 6.29s 就完成了。
# 更阴的是它会污染**时延测量** —— 任何本地 HTTP 基准都可能悄悄多算一段代理开销。
for _v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_v, None)
os.environ["no_proxy"] = "127.0.0.1,localhost"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
# ---------------------------------------------------------------------------

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request

SYSTEM = (
    "你是一个严谨的软件工程助手, 在一台 V100 32GB 的机器上工作。\n"
    "工作目录 /run/media/ezra/13D010B6FDBC1A06/1CatVLLM。\n"
    "回答时先给结论再给证据, 证据要能被复现; 不确定的地方明确说不确定。\n"
    "可用工具: bash(command) / read(path) / write(path, content) / grep(pattern, path)\n"
) * 8   # 拉长, 确保跨越多个缓存页(页长 128 token)

TOOL_RESULT = (
    "total 128\n"
    "drwxr-xr-x 12 ezra ezra  4096 Aug 20 06:30 fastllm\n"
    "drwxr-xr-x  9 ezra ezra  4096 Aug 19 22:11 llama.cpp\n"
    "drwxr-xr-x  7 ezra ezra  4096 Aug 20 05:16 v100-perfs\n"
    "-rw-r--r--  1 ezra ezra 20812 Aug 18 03:02 thinking_proxy.py\n"
) * 4   # 模拟真实的工具输出体量


def post(base, token, model, messages, timeout):
    body = {"model": model, "max_tokens": 40, "temperature": 0.0,
            "stream": False, "messages": messages}
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    return d, time.time() - t0


def backend_prefill(log_path):
    """抓最近一条 '[req N] done: prefill X tok' 的 X。"""
    try:
        tail = subprocess.run(["tail", "-200", log_path], capture_output=True,
                              text=True, timeout=20).stdout
    except Exception:
        return None
    for line in reversed(tail.split("\n")):
        m = re.search(r"done: prefill (\d+) tok", line)
        if m:
            return int(m.group(1))
    return None


def stats(log_path):
    try:
        tail = subprocess.run(["tail", "-300", log_path], capture_output=True,
                              text=True, timeout=20).stdout
    except Exception:
        return {}
    out = {}
    for line in reversed(tail.split("\n")):
        if "L1trie=" not in line:
            continue
        for key, pat in (("L1trie", r"L1trie=(\d+) pg"),
                         ("hits", r"hits=(\d+)"),
                         ("pool%", r"kv_pool=\d+/\d+ pg \((\d+)%\)")):
            m = re.search(pat, line)
            if m:
                out[key] = int(m.group(1))
        break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8002")
    ap.add_argument("--token", default="x")
    ap.add_argument("--model", default="qwen3.8-fastllm")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--log", default="/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/"
                                     "v100-perfs/runtime/fastllm-native-profiles/"
                                     "logs/backend-PROD-cyber-q5.log")
    args = ap.parse_args()

    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "看一下工作目录里有什么。"}]

    print(f"起始: {stats(args.log)}")
    print(f"{'轮':>3} {'历史字符':>9} {'新增字符':>9} {'后端prefill':>11} "
          f"{'wall':>7}   缓存状态")
    prev_chars = 0
    rows = []
    for i in range(args.rounds):
        chars = sum(len(m["content"]) for m in msgs)
        new_chars = chars - prev_chars
        try:
            d, wall = post(args.base_url, args.token, args.model, msgs,
                           args.timeout)
        except Exception as e:
            print(f"  第 {i+1} 轮失败: {type(e).__name__}: {e}")
            break
        time.sleep(1.5)          # 等 metrics 行落盘
        pf = backend_prefill(args.log)
        st = stats(args.log)
        print(f"{i+1:>3} {chars:>9} {new_chars:>9} {str(pf):>11} "
              f"{wall:>6.2f}s   {st}")
        rows.append((chars, new_chars, pf))
        prev_chars = chars

        # 追加一轮 assistant 工具调用 + 工具结果, 模拟 agent 循环
        msgs.append({"role": "assistant",
                     "content": f'<tool_call>{{"name":"bash",'
                                f'"arguments":{{"command":"ls -la . # 第{i+1}次"}}}}'
                                f'</tool_call>'})
        msgs.append({"role": "user", "content": f"工具结果:\n{TOOL_RESULT}"})

    print()
    print("═══ 判读 ═══")
    if len(rows) >= 3:
        # 跳过第 1 轮(冷启动必然全量)
        tot_new = sum(r[1] for r in rows[1:])
        tot_pf = sum(r[2] for r in rows[1:] if r[2] is not None)
        tot_hist = sum(r[0] for r in rows[1:])
        print(f"  第 2 轮起: 历史合计 {tot_hist} 字符, 新增合计 {tot_new} 字符, "
              f"后端实际 prefill {tot_pf} token")
        print("  理想情况: prefill 只覆盖新增部分 —— 每轮的历史部分全部命中。")
        print("  若 prefill 随历史线性增长, 说明**每轮都在全量重算**, 前缀缓存没起作用。")
    print()
    print(f"  结束: {stats(args.log)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
