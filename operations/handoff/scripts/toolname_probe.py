#!/usr/bin/env python3
"""工具名保真探针: 模型返回的 tool call 名字必须**严格等于**客户端声明的拼写。

为什么需要:
  模型对工具名有很强的先验(训练成写 "Bash"/"Read"/"WebSearch"), 而各家 harness
  声明的拼写五花八门: omp 全小写(bash/read/grep), 别家可能是 snake_case
  (web_search) 或 camelCase(readFile)。名字对不上 = 工具调用直接失败。

  生产上实测到的失败链:
    模型吐出 "B" -> 名字约束在 12 个全小写名字里找不到任何以 B 开头的
    -> 掩码清空 -> **静默**退回自由采样 -> 写出 "Bash"
    -> harness 收到不存在的工具名 -> 调用失败, 且没有任何报错说明原因。

  修复在引擎侧: 掩码用规范化形式匹配(大小写/分隔符不敏感), 名字最终由
  ResolveDeclaredToolName 归一化回**声明的拼写**。规则只依赖客户端声明的
  tools 列表, 因此对任何 harness 通用。

这个探针就是那条修复的验收标准: 三种命名风格都必须原样返回。

用法:
  python toolname_probe.py --base-url http://127.0.0.1:8000 --token "$AUTH_TOKEN"
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

import sys
import time
import urllib.error
import urllib.request

# 三种常见命名风格; 模型的先验都倾向于写成 PascalCase
CASES = [
    {
        "declared": "bash",
        "style": "全小写(omp 风格)",
        "prompt": "请用工具执行 shell 命令 `echo hello`。只调用工具, 不要解释。",
        "params": {
            "type": "object",
            "properties": {"command": {"type": "string",
                                       "description": "要执行的 shell 命令"}},
            "required": ["command"],
        },
        "desc": "Run a shell command",
    },
    {
        "declared": "web_search",
        "style": "snake_case",
        "prompt": "请用工具搜索 `V100 flash attention`。只调用工具, 不要解释。",
        "params": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "搜索词"}},
            "required": ["query"],
        },
        "desc": "Search the web",
    },
    {
        "declared": "readFile",
        "style": "camelCase",
        "prompt": "请用工具读取 `/etc/hostname`。只调用工具, 不要解释。",
        "params": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "文件路径"}},
            "required": ["path"],
        },
        "desc": "Read a file from disk",
    },
]


def call(base_url, token, model, case, timeout):
    body = {
        "model": model,
        "max_tokens": 256,
        "temperature": 0.0,
        "stream": False,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": case["prompt"]}],
        "tools": [{
            "type": "function",
            "function": {
                "name": case["declared"],
                "description": case["desc"],
                "parameters": case["params"],
            },
        }],
        "tool_choice": "auto",
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--token", default=os.environ.get("AUTH_TOKEN", "x"))
    ap.add_argument("--model", default="qwen3.8-fastllm")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    results = []
    passed = 0
    for case in CASES:
        try:
            data, wall = call(args.base_url, args.token, args.model,
                              case, args.timeout)
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, OSError) as e:
            print(f"✗ {case['declared']:<12} ({case['style']}) 请求失败: {e}",
                  flush=True)
            results.append({"declared": case["declared"], "error": str(e)})
            continue

        msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
        calls = msg.get("tool_calls") or []
        got = [c.get("function", {}).get("name") for c in calls]
        ok = bool(got) and all(n == case["declared"] for n in got)
        passed += 1 if ok else 0
        mark = "✓" if ok else "✗"
        print(f"{mark} {case['declared']:<12} ({case['style']}) "
              f"返回={got or '无工具调用'} wall={wall:.1f}s", flush=True)
        if not ok and not got:
            # 没触发工具调用时把正文头部打出来, 便于判断是模型没调还是格式坏了
            body_text = (msg.get("content") or "")[:200]
            print(f"    正文: {body_text!r}", flush=True)
        results.append({"declared": case["declared"], "style": case["style"],
                        "got": got, "ok": ok, "wall_s": round(wall, 1)})

    print(f"\n工具名保真: {passed}/{len(CASES)} 通过", flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"passed": passed, "total": len(CASES),
                       "results": results}, f, ensure_ascii=False, indent=2)
        print(f"已写入 {args.json}", flush=True)
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
