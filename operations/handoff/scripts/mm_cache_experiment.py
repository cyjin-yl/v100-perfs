#!/usr/bin/env python3
"""多模态前缀缓存的受控实验: 命中之后, 输出还对不对?

## 为什么要做这个实验

`FASTLLM_PREFIX_CACHE_MULTIMODAL=0` 的代价已经精确量到: agent 每轮要把整个
75820 token 重新 prefill(实测 114 秒), 而每轮只新增约 280 个 token —— 约 270 倍
的浪费, 命中时同样的事约 1 秒。

它避开的是两个**疑似**问题(只有代码阅读, 没有运行时证据):
  1. 续 prefill 分支上 mrope_position_delta 不会被写入, 位置退化成普通顺序;
  2. 落在未缓存余段里的图像不过 EncodeVisualItems, 模型看到的是
     <|image_pad|> 的 token embedding 而不是图像特征。

关键判断: 这两条如果成立, 后果是**整段位置错位**或**图像内容丢失**, 会立刻
产生明显不连贯的输出, 而不是细微漂移。所以一个请求就能判定 —— 不需要等一个
完善的端到端探针。

## 实验设计

三发请求, 第二发是第一发的**严格前缀延长**(同样的图 + 同样的开头文本 + 追加
一句), 因此第二发必须命中缓存:

  A  冷启动          -> 预期 miss(第一次见), 记下输出
  B  A 的严格前缀延长 -> 预期 **hit**, 这一发的输出质量就是判据
  C  同 B 但去掉图像  -> 纯文本对照, 用来区分"命中坏了"与"模型本来就这样"

判据(全部要过):
  1. B 必须真的命中(日志里 hit>0), 否则实验无效 —— 没命中就什么也没测到
  2. B 的输出里必须逐字包含指定的字面量(位置错位会先打穿逐字抄写)
  3. B 的输出必须是连贯中文, 不能是重复串/乱码

用法: python mm_cache_experiment.py            # 需要生产处于 MULTIMODAL=1
"""

import os
for _v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
           "all_proxy", "ALL_PROXY"):
    os.environ.pop(_v, None)
os.environ["no_proxy"] = os.environ["NO_PROXY"] = "127.0.0.1,localhost"

import base64
import io
import json
import re
import sys
import time
import urllib.request

ROOT = "/run/media/ezra/13D010B6FDBC1A06/1CatVLLM"
LOG = (ROOT + "/v100-perfs/runtime/fastllm-native-profiles/logs/"
       "backend-PROD-cyber-iq4xs.log")
MARK = "13D010B6FDBC1A06"          # 要求逐字抄写的字面量


def token():
    for line in open(ROOT + "/.env", encoding="utf-8"):
        if line.startswith("AUTH_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"\'')
    raise SystemExit("读不到 AUTH_TOKEN")


def make_images(n, size):
    """固定内容(不随机), 保证 A/B 两发的图像逐字节相同 —— 否则 token 序列不同,
    严格前缀关系就不成立, 实验也就无从谈起。"""
    from PIL import Image
    out = []
    for k in range(n):
        img = Image.new("RGB", (size, size))
        px = img.load()
        for y in range(0, size, 4):
            for x in range(0, size, 4):
                c = ((x + k * 31) % 256, (y + k * 47) % 256, (x ^ y) % 256)
                for dy in range(4):
                    for dx in range(4):
                        if x + dx < size and y + dy < size:
                            px[x + dx, y + dy] = c
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append("data:image/png;base64," +
                   base64.b64encode(buf.getvalue()).decode())
    return out


def post(tok, parts, max_tokens=200, timeout=2400):
    body = {"model": "qwen3.8-fastllm", "max_tokens": max_tokens,
            "temperature": 0.0, "stream": False, "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": parts}]}
    req = urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {tok}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    m = (d.get("choices") or [{}])[0].get("message", {}) or {}
    text = (m.get("content") or "") + (m.get("reasoning_content") or "")
    return text, d.get("usage") or {}, time.time() - t0


def log_tail(since):
    with open(LOG, "rb") as f:
        f.seek(since)
        return f.read().decode("utf-8", "replace")


def cache_result(tail):
    """从日志切片里取本次请求的命中情况。"""
    m = re.findall(r"req#(\d+) total=(\d+) hit=(\d+) layer=(\S+) miss=(\S*)",
                   tail)
    return m[-1] if m else None


def gibberish(s):
    """粗判乱码/退化: 同一字符连续 20 次以上, 或同一 8 字符片段重复 5 次以上。"""
    if re.search(r"(.)\1{19,}", s):
        return "单字符长串"
    for i in range(0, max(0, len(s) - 8), 8):
        seg = s[i:i + 8]
        if len(seg) == 8 and s.count(seg) >= 5:
            return f"片段重复 {seg!r}"
    return None


def main():
    tok = token()
    imgs = make_images(3, 512)          # 3 张 512² —— 够触发 vision 路径, 又不至于太贵
    base_text = ("这是一组测试图像。请记住下面这个设备标识, 后面会用到:\n"
                 f"{MARK}\n"
                 "现在先简单描述一下这些图像的整体色彩倾向。")
    ext_text = base_text + ("\n\n补充要求: 在回答的最后一行, 把上面那个设备标识"
                            "原样抄写一遍, 不要加任何标点或说明。")

    def parts(text):
        return ([{"type": "image_url", "image_url": {"url": u}} for u in imgs]
                + [{"type": "text", "text": text}])

    print("== 多模态前缀缓存受控实验 ==")
    results = {}
    for tag, text, expect_hit in (("A 冷启动", base_text, False),
                                  ("B 前缀延长", ext_text, True),
                                  ("C 纯文本对照", ext_text, None)):
        p = parts(text) if tag != "C 纯文本对照" else [{"type": "text", "text": text}]
        mark0 = os.path.getsize(LOG)
        try:
            out, usage, wall = post(tok, p)
        except Exception as e:                       # noqa: BLE001
            print(f"  ! {tag:<12} 失败: {e}")
            results[tag] = {"error": str(e)}
            continue
        cache = cache_result(log_tail(mark0))
        hit = int(cache[2]) if cache else -1
        total = int(cache[1]) if cache else -1
        miss = cache[4] if cache else "?"
        flat = " ".join(out.split())
        print(f"  {tag:<12} {wall:6.1f}s  prompt={usage.get('prompt_tokens')} "
              f"completion={usage.get('completion_tokens')}")
        print(f"      缓存: total={total} hit={hit} miss={miss or '-'}")
        print(f"      输出({len(flat)} 字符): {flat[:180]!r}")
        g = gibberish(flat)
        if g:
            print(f"      [!] 疑似退化: {g}")
        if tag.startswith("B"):
            print(f"      逐字抄写 {MARK}: "
                  f"{'✓ 命中' if MARK in flat else '✗ 未出现'}")
        results[tag] = {"hit": hit, "total": total, "miss": miss,
                        "text": out, "wall": wall, "gibberish": g,
                        "usage": usage}

    print()
    print("== 判定 ==")
    b = results.get("B 前缀延长", {})
    if b.get("hit", 0) <= 0:
        print("  实验无效: B 没有命中缓存, 什么也没测到。")
        print(f"    (miss={b.get('miss')}; 若为 multimodal-disabled 说明开关没打开)")
        rc = 2
    elif b.get("gibberish"):
        print(f"  **不通过**: B 命中后输出退化 —— {b['gibberish']}")
        rc = 1
    elif MARK not in " ".join(b.get("text", "").split()):
        print("  **不通过**: B 命中后没能逐字抄出设备标识 —— "
              "与位置编码错位的预期后果一致")
        rc = 1
    else:
        print(f"  通过: B 命中 {b['hit']}/{b['total']} token, 输出连贯且逐字抄写正确")
        rc = 0
    with open(ROOT + "/imatrix/logs/mm-cache-experiment.json", "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  详细结果已写入 {ROOT}/imatrix/logs/mm-cache-experiment.json")
    return rc


if __name__ == "__main__":
    sys.exit(main())
