#!/usr/bin/env python3
"""从真实 omp 会话日志构建 imatrix 校准语料。

为什么不用现成的校准集:
  llama.cpp 社区常用的 wiki.train.raw / groups_merged.txt 是通用英文散文。
  但我们真正被量化打坏的是**另一类 token**:
    - 工具调用的 JSON 结构与工具名(bash / read / web_search)
    - chat template 的特殊 token(<|im_start|> / <tool_call>)
    - 长字面量: 仓库路径、包名、commit hash、设备 UUID
    - 中英混排(用户说中文, 路径与代码是英文)
  生产实测的破坏样本 /home/ezra/Documents/Proto-UI -> /home/eze/Documents/PotouI
  全部落在这些 token 上, 而通用英文语料里它们几乎不出现 —— 于是 imatrix
  认为这些通道"不重要", 量化时优先牺牲它们, 正好放大了我们的故障。

  omp 的会话日志就是这个负载的**真实分布本身**, 每天几万条, 免费且无偏。

规模取舍:
  目标 ~600KB(约 130K token)。业界常用的 groups_merged.txt 同量级, imatrix
  的统计量在这个规模已基本收敛; 再加大只是线性增加 CPU 上的校准时间
  (27B 在 8 核上跑 prompt 大约几 tok/s, 语料翻倍就是几小时翻倍)。

用法:
  python build_imatrix_corpus.py --out calib.txt --target-bytes 600000
"""

import argparse
import glob
import json
import os
import random
import sys

DEFAULT_GLOB = "/home/ezra/.omp/agent/sessions/*/*.jsonl"


def text_of(content):
    """message.content 可能是 str, 也可能是 [{type:text,text:...}] 列表。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                if c.get("type") == "text" and c.get("text"):
                    parts.append(c["text"])
                elif c.get("type") == "reasoning" and c.get("text"):
                    parts.append(c["text"])
        return "\n".join(parts)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-bytes", type=int, default=600_000)
    ap.add_argument("--max-per-file", type=int, default=40_000,
                    help="每个会话文件最多贡献多少字节, 保证跨会话多样性")
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--skip-files", type=int, default=0,
                    help="洗牌后先跳过 N 个会话文件。用来造**留出集**: "
                         "校准语料用了前 N 个, 评测就必须用后面的, 否则等于拿"
                         "训练集当测试集, imatrix 的收益会被高估到毫无意义")
    ap.add_argument("--exclude-range", default=None, metavar="A:B",
                    help="排除洗牌后 [A,B) 区间的会话文件。扩充校准语料时用: "
                         "留出集占用了某段索引, 新语料必须继续避开它, "
                         "否则评测集被污染, 三方对比全部作废")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"没找到会话文件: {args.glob}", file=sys.stderr)
        return 2
    random.Random(args.seed).shuffle(files)
    # 用同一个 seed 洗牌, 再按 skip-files 切开, 保证校准集与留出集
    # 严格不相交且可复现
    if args.skip_files > 0:
        files = files[args.skip_files:]
    if args.exclude_range:
        a, b = (int(x) for x in args.exclude_range.split(":"))
        # 注意: 这里的索引是**洗牌后**的下标, 与造留出集时用的是同一个 seed,
        # 所以能精确对应上。改 seed 会让两边错位, 评测集就被污染了。
        files = [f for i, f in enumerate(files) if not (a <= i < b)]

    chunks = []
    total = 0
    stats = {"message": 0, "toolcall": 0, "files": 0}

    for path in files:
        if total >= args.target_bytes:
            break
        got = 0
        try:
            f = open(path, encoding="utf-8", errors="replace")
        except OSError:
            continue
        with f:
            for line in f:
                if got >= args.max_per_file or total >= args.target_bytes:
                    break
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                t = d.get("type")

                if t == "message":
                    msg = d.get("message") or {}
                    role = msg.get("role", "user")
                    body = text_of(msg.get("content"))
                    if not body or len(body) < 20:
                        continue
                    # 用真实 chat template 的标记, 让特殊 token 也进校准分布
                    piece = f"<|im_start|>{role}\n{body}<|im_end|>\n"

                elif t == "custom":
                    data = d.get("data") or {}
                    name = data.get("toolName")
                    if not name:
                        continue
                    call = {"name": name, "arguments": data.get("args") or {}}
                    # 工具调用按模型实际输出的格式还原 —— 这是标准语料完全没有的部分
                    piece = ("<|im_start|>assistant\n<tool_call>\n"
                             + json.dumps(call, ensure_ascii=False)
                             + "\n</tool_call><|im_end|>\n")
                    result = data.get("result") or data.get("output")
                    if isinstance(result, str) and result.strip():
                        piece += (f"<|im_start|>tool\n{result[:2000]}"
                                  f"<|im_end|>\n")
                    stats["toolcall"] += 1
                else:
                    continue

                if t == "message":
                    stats["message"] += 1
                b = len(piece.encode("utf-8"))
                chunks.append(piece)
                got += b
                total += b
        if got:
            stats["files"] += 1

    with open(args.out, "w", encoding="utf-8") as out:
        out.write("\n".join(chunks))

    size = os.path.getsize(args.out)
    print(f"已写入 {args.out}")
    print(f"  {size/1000:.0f} KB, 来自 {stats['files']} 个会话")
    print(f"  对话片段 {stats['message']}, 工具调用 {stats['toolcall']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
