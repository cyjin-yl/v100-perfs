#!/usr/bin/env python3
"""逐字抄写保真探针(走真实 HTTP 服务栈)。

为什么这是 IQ4_XS 能否上生产的**首要**判据, 比 PPL 更重要:
  agent 每一轮都要把上下文里的字面量原样抄进命令 —— 仓库路径、设备 UUID、
  commit 哈希、配置文件名。抄歪一个字符命令就失败; 更糟的是 agent 之后读回
  自己写坏的内容, 发现前后矛盾, 会判定"我正在被 prompt injection 攻击"
  然后拒绝干活。生产上真实出现过 Proto-UI -> PotouI。
  PPL 是全词表的平均对数似然, 对"低频长串抄写"这种尾部行为几乎不敏感 ——
  PPL 只掉 1% 也可能把抄写打穿, 所以必须单独测。

为什么走 HTTP 而不是 llama-cli:
  1. llama-cli 每条用例都要重新从机械盘加载 15GB, 5 条 = 5 次加载;
     HTTP 只加载一次, 之后每条是秒级。
  2. 更重要的是: 生产跑的是 fastllm, 不是 llama.cpp。分词器、预分词正则、
     采样路径、MTP 草稿接受逻辑全都不同 —— 用 llama.cpp 测出的结论
     **不能外推到生产**。今天要回答的问题是"IQ4_XS 在我们这套栈上能不能用"。

判定:
  - 剥掉 <think>...</think> 再匹配, 免得模型只在思考里复述了一遍就算过。
  - 期望串必须**逐字**出现在正文里(区分大小写、不做归一化), 因为 shell 命令
    就是逐字执行的。
  - 同时统计"近似但不相同"的变体并打印出来, 这样失败时能直接看出是抄歪了
    哪个字符, 而不是只知道 FAIL。

用法:
  python copy_fidelity_http.py --base-url http://127.0.0.1:8000 --token "$AUTH_TOKEN"
  python copy_fidelity_http.py --repeat 3     # 每条跑 3 次, 看是否偶发
"""

import argparse
import json
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

import re
import sys
import time
import urllib.error
import urllib.request

THINK_RE = re.compile(r"<think>.*?</think>", re.S)

# 全部取自这台机器上真实出现过的字面量 —— 不是造的测试串。
CASES = [
    ("长路径",
     "/home/ezra/Documents/Proto-UI/packages/prototypes/brutalist/src/card/index.ts"),
    ("设备UUID", "13D010B6FDBC1A06"),
    ("commit哈希", "3f9a2c7e15b8d046"),
    ("中英混排", "在 /run/media/ezra/13D010B6FDBC1A06 上跑 Qwen3.8-27B"),
    ("深层配置路径",
     "/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/v100-perfs/runtime/"
     "fastllm-native-profiles/q38-PROD-cyber-iq4xs-imatrix-mtp2-sm70.env"),
    ("环境变量名", "FASTLLM_CUDA_SM70_IQ4XS_WIDE_N_TILE=1"),
    ("带下划线的包名", "nvidia_cusparselt_cu12-0.7.1.dist-info"),
]


def strip_think(text):
    return THINK_RE.sub("", text or "")


def near_miss(body, expect):
    """在正文里找"长得像但不相同"的片段, 用来指出抄歪在哪。"""
    if len(expect) < 6:
        return None
    head = re.escape(expect[:5])
    for m in re.finditer(head, body):
        seg = body[m.start():m.start() + len(expect) + 12]
        seg = seg.split("\n")[0].strip().strip("`\"'")
        if seg and seg != expect:
            return seg
    return None


def ask(base, token, model, prompt, timeout):
    body = {
        "model": model,
        "max_tokens": 512,
        "temperature": 0.0,
        "stream": False,
        # 抄写不需要思考; 开着只会让 max_tokens 被思考吃掉, 制造假阴性
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
    return (msg.get("content") or ""), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--token", default=os.environ.get("AUTH_TOKEN", ""))
    ap.add_argument("--model", default="qwen3.8-fastllm")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    print(f"== 逐字抄写保真(HTTP) base={args.base_url} repeat={args.repeat} ==")
    ok = bad = 0
    results = []
    for name, expect in CASES:
        prompt = ("把下面这行原样重复一遍, 不要加任何其它内容, "
                  "不要加引号或代码块:\n" + expect)
        for k in range(args.repeat):
            tag = name if args.repeat == 1 else f"{name}#{k+1}"
            try:
                content, wall = ask(args.base_url, args.token, args.model,
                                    prompt, args.timeout)
            except Exception as e:          # noqa: BLE001 - 网络/超时都要如实报
                print(f"  ! {tag:<16} 请求失败: {e}", flush=True)
                bad += 1
                results.append({"case": tag, "error": str(e)})
                continue
            body = strip_think(content)
            hit = expect in body
            if hit:
                ok += 1
                print(f"  ✓ {tag:<16} {wall:6.1f}s", flush=True)
            else:
                bad += 1
                nm = near_miss(body, expect)
                print(f"  ✗ {tag:<16} {wall:6.1f}s", flush=True)
                print(f"      期望 {expect}", flush=True)
                if nm:
                    print(f"      抄成 {nm}      <-- 逐字对比这两行", flush=True)
                else:
                    flat = " ".join(body.split())[:180]
                    print(f"      实得 {flat!r}", flush=True)
            results.append({"case": tag, "ok": hit, "wall_s": round(wall, 1),
                            "expect": expect, "got": body[:400]})

    total = ok + bad
    rate = (ok / total * 100) if total else 0.0
    print(f"\n  抄写保真 {ok}/{total} = {rate:.1f}%")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"ok": ok, "total": total, "rate_pct": round(rate, 1),
                       "results": results}, f, ensure_ascii=False, indent=2)
        print(f"  已写入 {args.json}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
