#!/usr/bin/env python3
"""带图像的逐字抄写探针 —— 多模态 decode 正确性的判据。

为什么需要它:
  F9 会改多模态 decode 的位置编码(标量 -> {3, seqLen} 的 mRoPE)。改错了不会
  报错, 只会让输出"稍微不对" —— 正是本仓反复出现的静默失效。而现有的
  copy_fidelity_http.py 是**纯文本**的, 图像根本不进上下文, 抓不到这类回归。

  逐字抄写之所以是好判据: 它要求模型把上下文里的字面量原样复制出来。位置
  编码一旦错位, 复制就会在某个字符上歪掉 —— 比"看起来还行"的自由生成敏感
  得多, 而且失败是可判定的, 不需要人眼看。

自检(这一条比测试本身更重要):
  1. 必须确认图像**真的进了模型**。判据是后端日志里的
       [Qwen3.5 vision] aggregate image pixels capped
     所以默认发 5 张 1024x1024(聚合 5.24M 像素, 超过 4014080 的上限), 保证这行
     必然打印。没有这一行 -> 判"探针无效"(exit 2), 不判通过。
  2. 输出为空时**不能**判通过。空串与空串比较恒等, 那是假阳性。
     (上一版探针就埋过这个坑。)

用法:
  python copy_fidelity_vision.py --token "$AUTH_TOKEN"
  python copy_fidelity_vision.py --no-image     # 同一批用例的纯文本对照
"""

import argparse
import base64
import io
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
import urllib.request

VISION_MARK = "[Qwen3.5 vision] aggregate image pixels capped"
BACKEND_LOG = ("/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/v100-perfs/runtime/"
               "fastllm-native-profiles/logs/backend-PROD-cyber-iq4xs.log")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)

# 与纯文本版同一批字面量, 便于直接对比"有图 vs 无图"
CASES = [
    ("长路径",
     "/home/ezra/Documents/Proto-UI/packages/prototypes/brutalist/src/card/index.ts"),
    ("设备UUID", "13D010B6FDBC1A06"),
    ("commit哈希", "3f9a2c7e15b8d046"),
    ("中英混排", "在 /run/media/ezra/13D010B6FDBC1A06 上跑 Qwen3.8-27B"),
    ("环境变量名", "FASTLLM_CUDA_SM70_IQ4XS_WIDE_N_TILE=1"),
]


def make_images(count, w, h):
    """造 count 张 w*h 的图。内容用渐变而非纯色, 避免被某些实现当成空图跳过。"""
    from PIL import Image
    out = []
    for k in range(count):
        img = Image.new("RGB", (w, h))
        px = img.load()
        for y in range(0, h, 8):
            for x in range(0, w, 8):
                c = ((x + k * 37) % 256, (y + k * 53) % 256, (x + y) % 256)
                for dy in range(8):
                    for dx in range(8):
                        if x + dx < w and y + dy < h:
                            px[x + dx, y + dy] = c
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append("data:image/png;base64," +
                   base64.b64encode(buf.getvalue()).decode())
    return out


def log_len(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def log_since(path, start):
    try:
        with open(path, "rb") as f:
            f.seek(start)
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def ask(base, token, model, parts, timeout, max_tokens):
    body = {
        "model": model, "max_tokens": max_tokens, "temperature": 0.0,
        "stream": False, "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": parts}],
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
    # reasoning_content 也算输出 —— 只看 content 会把"其实有输出"误判成空
    text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
    usage = data.get("usage") or {}
    return text, usage, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--token", default=os.environ.get("AUTH_TOKEN", ""))
    ap.add_argument("--model", default="qwen3.8-fastllm")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--images", type=int, default=5)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--no-image", action="store_true",
                    help="同一批用例的纯文本对照")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    imgs = [] if args.no_image else make_images(args.images, args.size, args.size)
    mode = "纯文本对照" if args.no_image else f"{args.images}x{args.size}²图像"
    print(f"== 带图逐字抄写  {mode}  base={args.base_url} ==")

    ok = bad = invalid = 0
    results = []
    for name, expect in CASES:
        prompt = ("把下面这行原样重复一遍, 不要加任何其它内容, "
                  "不要加引号或代码块:\n" + expect)
        parts = ([{"type": "image_url", "image_url": {"url": u}} for u in imgs] +
                 [{"type": "text", "text": prompt}])
        mark0 = log_len(BACKEND_LOG)
        try:
            text, usage, wall = ask(args.base_url, args.token, args.model,
                                    parts, args.timeout, args.max_tokens)
        except Exception as e:                      # noqa: BLE001
            print(f"  ! {name:<12} 请求失败: {e}", flush=True)
            invalid += 1
            results.append({"case": name, "error": str(e)})
            continue
        tail = log_since(BACKEND_LOG, mark0)

        # 自检 1: 图像真的进了模型吗
        if not args.no_image and VISION_MARK not in tail:
            print(f"  ? {name:<12} 探针无效: 日志里没有 vision 标记, "
                  f"图像可能没进模型", flush=True)
            invalid += 1
            results.append({"case": name, "invalid": "no-vision-marker"})
            continue
        # 自检 2: 输出为空不能判通过
        body = THINK_RE.sub("", text).strip()
        if not body:
            print(f"  ? {name:<12} 探针无效: 输出为空 "
                  f"(completion={usage.get('completion_tokens')})", flush=True)
            invalid += 1
            results.append({"case": name, "invalid": "empty-output",
                            "usage": usage})
            continue

        hit = expect in body
        if hit:
            ok += 1
            print(f"  ✓ {name:<12} {wall:6.1f}s  "
                  f"prompt={usage.get('prompt_tokens')} tok", flush=True)
        else:
            bad += 1
            print(f"  ✗ {name:<12} {wall:6.1f}s", flush=True)
            print(f"      期望 {expect}", flush=True)
            print(f"      实得 {' '.join(body.split())[:160]!r}", flush=True)
        results.append({"case": name, "ok": hit, "wall_s": round(wall, 1),
                        "expect": expect, "got": body[:400], "usage": usage})

    total = ok + bad
    print(f"\n  通过 {ok}/{total}   无效 {invalid}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"mode": mode, "ok": ok, "total": total,
                       "invalid": invalid, "results": results},
                      f, ensure_ascii=False, indent=2)
        print(f"  已写入 {args.json}")
    if invalid:
        print("  [!] 有无效用例 -> 本次结论不可用")
        return 2
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
