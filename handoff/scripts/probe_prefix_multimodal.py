#!/usr/bin/env python3
"""带图请求的前缀缓存探针 —— **会自证它真的测到了 vision**。

为什么重写(2026-08-20):
  旧版本的 image 轮次报出 hit=512 / hit=896, 看着像"vision 命中了", 但当时
  日志里同时出现 `Jinja Error ... -> using MakeInput fallback`, 无法判断图片
  到底有没有进到模型里 —— 它是个**假阳性源**。教训见 EXPERIENCE.md §15.40:
  **探针必须自证它测到了目标, 观察不到目标就判"无效", 而不是判"通过"。**

自证手段(硬判据):
  后端在做视觉预处理时会打
      [Qwen3.5 vision] aggregate image pixels capped: images=K original=... limit=...
  这一行只在**聚合像素超过上限**时才打。所以本探针默认发 5 张 1024x1024
  (聚合 5.24M px > 4.01M 上限), 让这行**必然**出现。每一轮如果没有观察到
  新的该标记, 直接判 PROBE INVALID 退出, 不给出任何"通过"结论。
  (这也正好复刻生产形态: 两个 agent 每轮都是 images=5 且超限。)

两种模式:
  agentic     —— 上下文单调增长, 图片留在前缀里, 第 2 轮起应当 hit>0
  determinism —— **同一段对话发两次**, temperature=0。第 1 次冷(必然 miss),
                 第 2 次应当命中。两次输出必须逐字相同; 不同 = 命中路径
                 破坏了状态(例如 mRoPE 位置没有按 cacheLen 续算, 或残段里的
                 图片没有被视觉塔编码)。这是唯一不需要重启就能验的正确性判据。

用法:
  python3 probe_prefix_multimodal.py --mode determinism
  python3 probe_prefix_multimodal.py --mode agentic --rounds 3
退出码: 0=通过 1=判据不满足 2=探针无效(没测到 vision)
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

import argparse, base64, io, json, re, subprocess, sys, time, urllib.request

LOG_DEFAULT = ("/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/v100-perfs/runtime/"
               "fastllm-native-profiles/logs/backend-PROD-cyber-iq4xs.log")
VISION_MARK = "[Qwen3.5 vision] aggregate image pixels capped"

SYS = ("你是一个严谨的软件工程助手, 在 V100 32GB 上工作。回答简短。\n") * 8
TOOL = ("total 128\ndrwxr-xr-x 12 ezra ezra 4096 Aug 20 06:30 fastllm\n") * 4


SRC_IMAGE = "/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/fastllm/test/beauty_test_input.png"


def make_images(count, w, h, src=SRC_IMAGE):
    """默认拿一张**真实照片**放大后重复 count 张。

    不要用合成的纯色/棋盘图: 实测 5 张合成图会让模型直接吐 EOS
    (finish_reason=stop, completion_tokens=0), 于是探针拿不到任何 decode token,
    既做不了逐字比对, 也满足不了 F9 那条"vision 窗口里必须出现 pos_accept_rate"
    的判据 —— 那需要真的解码出 token。
    """
    from PIL import Image
    out = []
    base = None
    try:
        base = Image.open(src).convert("RGB").resize((w, h))
    except Exception as e:
        print("  警告: 读不到真实图片 %s (%s), 退回合成图; "
              "模型可能直接 EOS 而不产出 token。" % (src, e))
    for i in range(count):
        if base is not None:
            img = base.copy()
            # 每张改一个角标, 保证 5 张不是逐字节相同(避免被任何去重逻辑折叠)
            for y in range(0, 16):
                for x in range(0, 16 + i * 4):
                    img.putpixel((x, y), (255, 0, 0))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            out.append("data:image/png;base64," +
                       base64.b64encode(buf.getvalue()).decode("ascii"))
            continue
        img = Image.new("RGB", (w, h),
                        ((i * 47) % 256, (i * 91 + 30) % 256, (i * 133 + 60) % 256))
        # 加一点结构, 避免纯色被任何一层特殊处理
        for y in range(0, h, 64):
            for x in range(0, w, 64):
                if ((x // 64) + (y // 64) + i) % 2 == 0:
                    for yy in range(y, min(y + 64, h), 8):
                        for xx in range(x, min(x + 64, w), 8):
                            img.putpixel((xx, yy), (255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append("data:image/png;base64," +
                   base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


def log_len(path):
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def log_since(path, start):
    try:
        with open(path, "r", errors="replace") as f:
            return f.read().split("\n")[start:]
    except Exception:
        return []


def correlate(new_lines):
    """在新增日志里找 vision 标记 -> [req] start -> [PrefixCache] req# 这条链。

    返回 (vision_seen, total, hit, miss)。vision_seen=False 时其余无意义。
    """
    vision_seen = any(VISION_MARK in l for l in new_lines)
    idx = next((i for i, l in enumerate(new_lines) if VISION_MARK in l), None)
    if idx is None:
        return False, None, None, None
    total = None
    for l in new_lines[idx:]:
        m = re.search(r"\] start: prefill (\d+) tok", l)
        if m:
            total = int(m.group(1))
            break
    if total is None:
        return True, None, None, None
    for l in new_lines[idx:]:
        m = re.search(r"\[PrefixCache\] req#\d+ total=(\d+) hit=(\d+) layer=\S+ miss=(\S+)", l)
        if m and int(m.group(1)) == total:
            return True, total, int(m.group(2)), m.group(3)
    return True, total, None, None


def post(base, model, messages, timeout, max_tokens):
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0.0,
            "top_p": 1.0, "stream": False, "messages": messages}
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    msg = d.get("choices", [{}])[0].get("message", {}) or {}
    # 有些档位会把内容全放进 reasoning_content, content 为空。两段都要参与比对,
    # 否则"两次都是空字符串"会假装成"输出一致" —— 探针自己的假阳性。
    text = (msg.get("content") or "") + "\x1f" + (msg.get("reasoning_content") or "")
    return text, time.time() - t0


def send(a, messages, tag):
    """发一次请求并做 vision 自证。返回 (text, total, hit, miss) 或 None(无效)。"""
    start = log_len(a.log)
    try:
        text, wall = post(a.base_url, a.model, messages, a.timeout, a.max_tokens)
    except Exception as e:
        print("  %-12s 请求失败: %s: %s" % (tag, type(e).__name__, e))
        return None
    time.sleep(1.5)
    new = log_since(a.log, start)
    vision, total, hit, miss = correlate(new)
    if not vision:
        print("  %-12s PROBE INVALID: 日志里没有观察到 vision 标记 (%s)" % (tag, VISION_MARK))
        print("               => 图片没有进到模型里, 本轮结果**不能**当作 vision 证据。")
        return None
    print("  %-12s wall=%6.2fs  total=%s hit=%s miss=%s  out=%r"
          % (tag, wall, total, hit, miss, (text or "").replace("\x1f", "|")[:40]))
    return text, total, hit, miss


def mode_determinism(a, imgs):
    print("\n===== determinism: 同一段对话发两次, temperature=0 =====")
    print("判据: 第 2 次应当命中前缀; 两次输出必须逐字相同。")
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content":
                [{"type": "text", "text": "这几张图上大致是什么内容? 用一句话概括。"}] +
                [{"type": "image_url", "image_url": {"url": u}} for u in imgs]}]
    first = send(a, msgs, "cold")
    if first is None:
        return 2
    second = send(a, msgs, "warm")
    if second is None:
        return 2
    t1, _, h1, _ = first
    t2, _, h2, m2 = second
    if h2 == 0:
        print("  => 第 2 次没有命中 (miss=%s)。若 miss=multimodal-disabled, 说明" % m2)
        print("     FASTLLM_PREFIX_CACHE_MULTIMODAL=0 生效中, 正确性判据**无法执行**")
        print("     (探针有效, 但结论是 INCONCLUSIVE, 不是 PASS)。")
        return 1
    if len(t1.replace("\x1f", "").strip()) == 0:
        print("  PROBE INVALID: 两次输出都是空的, 逐字比对没有意义(空 == 空)。")
        print("               => 调大 --max-tokens, 或换一个必定产出可见文本的提问。")
        return 2
    if t1 == t2:
        print("  PASS: 命中 %s token, 两次输出逐字相同(%d 字符) -> 命中路径没有破坏状态。"
              % (h2, len(t1.replace("\x1f", ""))))
        return 0
    print("  FAIL: 命中 %s token, 但两次输出不同 -> 命中路径破坏了状态。" % h2)
    print("        cold: %r" % (t1 or "")[:200])
    print("        warm: %r" % (t2 or "")[:200])
    return 1


def mode_agentic(a, imgs):
    print("\n===== agentic: 上下文单调增长, 图片留在前缀里 =====")
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content":
                [{"type": "text", "text": "看一下工作目录里有什么。"}] +
                [{"type": "image_url", "image_url": {"url": u}} for u in imgs]}]
    hits = []
    for i in range(a.rounds):
        r = send(a, msgs, "round %d" % (i + 1))
        if r is None:
            return 2
        hits.append(r[2])
        msgs.append({"role": "assistant", "content": "我先看看目录 (第%d次)。" % (i + 1)})
        msgs.append({"role": "user", "content": "工具结果:\n%s" % TOOL})
    later = [h for h in hits[1:] if h is not None]
    if later and all(h > 0 for h in later):
        print("  PASS: 第 2 轮起全部命中 %s" % later)
        return 0
    print("  FAIL/INCONCLUSIVE: 第 2 轮起命中 %s (0 可能是 multimodal-disabled)" % later)
    return 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8002")
    p.add_argument("--model", default="qwen3.8-fastllm")
    p.add_argument("--mode", default="determinism", choices=["determinism", "agentic"])
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--images", type=int, default=5)
    p.add_argument("--image-w", type=int, default=1024)
    p.add_argument("--image-h", type=int, default=1024)
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--log", default=LOG_DEFAULT)
    a = p.parse_args()
    px = a.images * a.image_w * a.image_h
    print("聚合像素 = %d x %dx%d = %d px (后端上限约 4014080; 必须超过它, "
          "vision 标记才会打印)" % (a.images, a.image_w, a.image_h, px))
    if px <= 4014080:
        print("警告: 聚合像素没超过上限, vision 标记不会出现, 探针会判 INVALID。")
    imgs = make_images(a.images, a.image_w, a.image_h)
    return mode_determinism(a, imgs) if a.mode == "determinism" else mode_agentic(a, imgs)


if __name__ == "__main__":
    sys.exit(main())
