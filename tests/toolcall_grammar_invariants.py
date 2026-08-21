#!/usr/bin/env python3
"""tool_call 语法状态机硬不变量测试(复刻 fastllm/src/models/basellm.cpp)。

回归背景(409412d6 事故): 多参数块场景下模型在 </parameter><parameter=
之间循环、发明非 schema 参数名(pattern/paath/maax_)、永远关不掉
</function>。根因:
  A. HasUnclosedToolCallParameterBefore / FindActiveToolCallInvokeName
     闭合标签只有 DSML 格式, Qwen 的 </parameter>/</function> 匹配不上
     -> 第二个参数块起 S3 永久失效 -> 参数名不 mask(存量 bug, 被状态机
     S2 白名单的分布压缩放大成循环)。已修: close 标签并集加 Qwen。
  B. S2 用 FindForced 的模糊 false 语义(必填齐 vs 缺必填但尾格式不符
     不分), 缺必填时放行 </function>。已修: 显式 MissingRequired 判定。
  C. S4 空值: 无值就闭合 </parameter>。已修: 允许普通 '<' 和闭合
     前缀，只屏蔽会实际完成全空白 </parameter> 的 token。

断言语义: can_spell(text, target) -- 从 text 出发, 逐 token(DFS)地问
约束, target 是否可拼出。这是用户点名的硬不变量形式。

对应 C++ 版本: 409412d6 + {DSML/Qwen close 修复, S2 显式 missing,
S4 blocked 通道}。C++ 侧改动需同步本文件。

用法: python3 v100-perfs/tests/toolcall_grammar_invariants.py
"""

PARAM_OPEN = "<parameter="
PARAM_CLOSE = "</parameter>"
FUNC_OPEN = "<function="
FUNC_CLOSE = "</function>"
TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
TERM = ">"
WS = " \t\r\n"


def overlap(tail, target):
    for n in range(min(len(tail), len(target)), 0, -1):
        if tail.endswith(target[:n]):
            return n
    return 0


def collect_closed(text, invoke_pos):
    closed = set()
    pos = text.find(PARAM_OPEN, invoke_pos)
    while pos != -1:
        ns = pos + len(PARAM_OPEN)
        tp = text.find(TERM, ns)
        if tp == -1:
            break
        cp = text.find(PARAM_CLOSE, tp)
        if cp == -1:
            break
        closed.add(text[ns:tp])
        pos = text.find(PARAM_OPEN, cp + len(PARAM_CLOSE))
    return closed


def missing_required(text, cfg, tool_name, invoke_pos):
    req = cfg["required"].get(tool_name, [])
    closed = collect_closed(text, invoke_pos)
    return [r for r in req if r not in closed]


def has_unclosed_before(text, invoke_pos, parameter_pos):
    """修复后: 闭合标签含 Qwen </parameter>。"""
    prev = text.rfind(PARAM_OPEN, 0, parameter_pos)
    if prev == -1 or prev < invoke_pos:
        return False
    close = max(
        text.rfind(t, 0, parameter_pos)
        for t in ["</｜DSML｜parameter>", "</\\DSML\\parameter>", PARAM_CLOSE])
    return not (close != -1 and close > prev)


def locate(text):
    open_ = text.rfind(TOOL_OPEN)
    if open_ == -1:
        return ("none", "", -1, -1)
    done = text.rfind(TOOL_CLOSE)
    if done != -1 and done > open_:
        return ("none", "", -1, -1)
    invoke_pos = text.rfind(FUNC_OPEN)
    if invoke_pos == -1 or invoke_pos < open_:
        return ("S0-func-open", "", -1, open_ + len(TOOL_OPEN))
    name_start = invoke_pos + len(FUNC_OPEN)
    name_term = text.find(TERM, name_start)
    if name_term == -1:
        return ("S1-func-name", "", invoke_pos, name_start)
    func_name = text[name_start:name_term]
    pos = name_term + len(TERM)
    while True:
        fc = text.find(FUNC_CLOSE, pos)
        pp = text.find(PARAM_OPEN, pos)
        if fc == -1 and pp == -1:
            return ("S2-param-gap", func_name, invoke_pos, pos)
        if fc != -1 and (pp == -1 or fc < pp):
            tail_start = fc + len(FUNC_CLOSE)
            if text[tail_start:] == TOOL_CLOSE:
                return ("none", func_name, invoke_pos, -1)
            return ("S5-close-tail", func_name, invoke_pos, tail_start)
        p_name_start = pp + len(PARAM_OPEN)
        p_term = text.find(TERM, p_name_start)
        if p_term == -1:
            return ("S3-param-name", func_name, invoke_pos, p_name_start)
        val_close = text.find(PARAM_CLOSE, p_term + len(TERM))
        if val_close == -1:
            return ("S4-param-value", func_name, invoke_pos,
                    p_term + len(TERM))
        pos = val_close + len(PARAM_CLOSE)


# ---------- 模拟 vocab(真实 Qwen 分块形态) ----------
VOCAB = [
    "<tool_call>", "</tool_call>", "<function=", "</function>",
    "<parameter=", "</parameter>", "<", "</", "<p", "<pa", "<par",
    "<parameter", "</pa", "</param", "</parame", "</f", "</fu",
    "</functio", "</function", "</tool", "</tool_", "list_dir",
    "read_file", "list", "_dir", "path", "max_depth", "max", "_depth",
    "pa", "at", "h", "pattern", "pat", "tern", "paath", "ma", "ax",
    "ax_", "aax", "m", "a", "x", "t", "e", "r", "n", "d", "p", "i",
    "s", "l", "_", "th", "ath", ">", "/etc", "/", "etc", "2", "\n",
    " ", "  ", "hello", "1", "10", "txt", ".", "readme", "><",
    "]", "+", ")", "span", "</parameter ",
]
T2I = {t: i for i, t in enumerate(VOCAB)}


def evaluate(text, cfg):
    """返回 (state, ids, blocked)。ids=None 不 mask; blocked=set()。"""
    state, fname, invoke_pos, seg = locate(text)
    if state == "none":
        return (state, None, set())
    partial = ""
    allowed = []
    term = "\x01"
    blocked = set()
    if state == "S0-func-open":
        tail = text[seg:]
        target = "\n" + FUNC_OPEN
        partial = tail[len(tail) - overlap(tail, target):]
        allowed = [target]
    elif state == "S1-func-name":
        # FindActiveToolCallNamePartial: partial = 整个已写名字
        partial = text[seg:]
        allowed = cfg["functions"]
        term = TERM
    elif state == "S2-param-gap":
        tail = text[seg:]
        missing = missing_required(text, cfg, fname, invoke_pos)
        parameter_target = "\n" + PARAM_OPEN
        close_target = "\n" + FUNC_CLOSE
        m1 = overlap(tail, parameter_target)
        if missing:
            partial = tail[len(tail) - m1:]
            allowed = [parameter_target]
        else:
            m2 = overlap(tail, close_target)
            partial = tail[len(tail) - max(m1, m2):]
            allowed = [parameter_target, close_target]
    elif state == "S3-param-name":
        pp = text.rfind(PARAM_OPEN)
        if has_unclosed_before(text, invoke_pos, pp):
            return (state, None, set())
        ns = pp + len(PARAM_OPEN)
        if text.find(TERM, ns) != -1:
            return (state, None, set())
        partial = text[ns:]
        allowed = list(cfg["params"].get(fname, []))
        missing = missing_required(text, cfg, fname, invoke_pos)
        if missing:
            req_only = [n for n in allowed if n in missing]
            if req_only:
                allowed = req_only
        term = TERM
    elif state == "S4-param-value":
        tail = text[seg:]
        marker = "</parameter"
        for tid, tok in enumerate(VOCAB):
            pos = tok.find(marker)
            while pos != -1:
                after = pos + len(marker)
                if after < len(tok) and tok[after] != ">":
                    blocked.add(tid)
                    break
                pos = tok.find(marker, pos + 1)
        close_prefix = overlap(tail, PARAM_CLOSE)
        committed = tail[:-close_prefix] if close_prefix else tail
        value_empty = not any(c not in WS for c in committed)
        if not value_empty and close_prefix == len(PARAM_CLOSE) - 1:
            partial = tail[-close_prefix:]
            allowed = [PARAM_CLOSE]
            term = "\x01"
        elif value_empty:
            # 只屏蔽会实际完成全空白 </parameter> 的 token。前缀 '<'
            # 必须自由，否则 HTML 和正则 r"[^<]" 无法表达。
            for tid, tok in enumerate(VOCAB):
                combined = tail + tok
                first = next(
                    (i for i, c in enumerate(combined) if c not in WS),
                    None)
                if first is not None and \
                        combined.startswith(PARAM_CLOSE, first):
                    blocked.add(tid)
            return (state, None, blocked)
        else:
            return (state, None, blocked)
    elif state == "S5-close-tail":
        tail = text[seg:]
        target = "\n" + TOOL_CLOSE
        partial = tail[len(tail) - overlap(tail, target):]
        allowed = [target]
    ids = []
    for tid, tok in enumerate(VOCAB):
        comb = partial + tok
        for a in allowed:
            tp = comb.find(term)
            if tp != -1:
                if comb[:tp] == a:
                    ids.append(tid)
                    break
            elif a.startswith(comb):
                ids.append(tid)
                break
    return (state, ids, blocked)


def can_spell(start_text, target, cfg):
    """硬不变量判定: target 从 start_text 出发是否可拼出(DFS)。"""
    memo = {}

    def dfs(text, pos):
        if pos >= len(target):
            return True
        key = (text[-40:], pos)
        if key in memo:
            return memo[key]
        st, ids, blocked = evaluate(text, cfg)
        if st == "none":
            return True
        if ids is None and not blocked and st != "S4-param-value":
            return True  # 结构外自由；S4 仍需逐 token 检查后续空值闭合
        rem = target[pos:]
        ok = False
        for tid, tok in enumerate(VOCAB):
            if not tok or not rem.startswith(tok):
                continue
            if ids is not None and tid not in ids:
                continue
            if tid in blocked:
                continue
            if dfs(text + tok, pos + len(tok)):
                ok = True
                break
        memo[key] = ok
        return ok

    return dfs(start_text, 0)


FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


CFG = {
    "functions": ["list_dir", "read_file"],
    "params": {"list_dir": ["path", "max_depth"], "read_file": ["path"]},
    "required": {"list_dir": ["path"], "read_file": ["path"]},
}

# ============ I0: 结构严格来自 Jinja ============
st, ids, _ = evaluate("<tool_call>", CFG)
check("I0.1 S0 状态正确", st == "S0-func-open", st)
check("I0.2 S0 先允许 Jinja 换行", T2I["\n"] in ids)
check("I0.3 S0 不跳过换行直发 function",
      T2I["<function="] not in ids)
check("I0.4 S0 不允许下划线", T2I["_"] not in ids)
check("I0.5 S0 不允许空格", T2I[" "] not in ids)
check("I0.6 S0 不允许点号", T2I["."] not in ids)
st, ids, _ = evaluate("<tool_call>\n", CFG)
check("I0.7 S0 换行后允许 function", T2I["<function="] in ids)
st, ids, _ = evaluate("<tool_call>\n<function=", CFG)
check("I0.8 S1 允许 schema 精确工具名", T2I["list"] in ids)
check("I0.9 S1 不允许前导下划线", T2I["_"] not in ids)
check("I0.10 S1 不允许前导点号", T2I["."] not in ids)

# ============ I1: 多参数块间隙(用户点名硬不变量) ============
# 前缀: 已写完 <parameter=path>/etc</parameter>
P1 = "<tool_call>\n<function=list_dir>\n<parameter=path>/etc</parameter>"
st, _, _ = evaluate(P1, CFG)
check("I1.1 状态=S2", st == "S2-param-gap", st)
check("I1.2 Jinja 换行后 </function> 可拼出",
      can_spell(P1, "\n</function>", CFG))
check("I1.3 Jinja 换行后 <parameter=max_depth> 可拼出",
      can_spell(P1, "\n<parameter=max_depth>", CFG))
check("I1.4 非 schema 名裸 pattern 不可拼出",
      not can_spell(P1, "pattern", CFG))
check("I1.5 裸 max_depth(无 <parameter= 前缀)不可拼出",
      not can_spell(P1, "max_depth", CFG))

# ============ I2: S2 -> S3 多参数块回转 ============
P2 = P1 + "\n<parameter="
st, _, _ = evaluate(P2, CFG)
check("I2.1 状态=S3", st == "S3-param-name", st)
check("I2.2 schema 名 path 可拼", can_spell(P2, "path>", CFG))
check("I2.3 schema 名 max_depth 可拼", can_spell(P2, "max_depth>", CFG))
check("I2.4 非 schema 名 pattern 不可拼(根因 A 修复)",
      not can_spell(P2, "pattern>", CFG))
check("I2.5 拼错名 paath 不可拼", not can_spell(P2, "paath>", CFG))
check("I2.6 拼错名 maax_ 不可拼", not can_spell(P2, "maax_depth>", CFG))
check("I2.7 其他工具名 read_file 的参数在此不可拼(schema 隔离)",
      not can_spell(P2, "read_file>", CFG))

# ============ I3: 缺必填时 S2 不放行 </function>(修复 B) ============
P4 = "<tool_call>\n<function=list_dir>\n<parameter=max_depth>2</parameter>"
st, _, _ = evaluate(P4, CFG)
check("I3.1 缺 path 状态=S2", st == "S2-param-gap", st)
check("I3.2 缺 path 时 Jinja 换行闭合不可拼",
      not can_spell(P4, "\n</function>", CFG))
check("I3.3 缺 path 时 Jinja 参数起点可拼",
      can_spell(P4, "\n<parameter=path>", CFG))
P5 = P1 + "\n<parameter=max_depth>2</parameter>"
check("I3.4 必填齐后 Jinja 换行闭合可拼",
      can_spell(P5, "\n</function>", CFG))

# ============ I4: 空值拒绝闭合(黑名单) ============
P6 = "<tool_call>\n<function=list_dir>\n<parameter=path>"
st, _, _ = evaluate(P6, CFG)
check("I4.1 值起点=S4", st == "S4-param-value", st)
check("I4.2 空值换行后 </parameter> 不可拼",
      not can_spell(P6, "\n</parameter>", CFG))
check("I4.3 全空白值仍不可闭合",
      not can_spell(P6, "\n  \n</parameter>", CFG))
check("I4.4 空值可写非空白值", can_spell(P6, "\n/etc", CFG))
P7 = P6 + "\n/etc"
check("I4.5 非空值后 Jinja 闭合可拼",
      can_spell(P7, "\n</parameter>", CFG))
check("I4.6 非空值末尾裸 < 保持自由",
      evaluate(P6 + "\nitems = re.findall(r'itemtitle\">([^<", CFG)[1] is None)
check("I4.7 正则字符类中的 < 可逐 token 拼出",
      can_spell(P6 + "\nitems = re.findall(r'itemtitle\">([^",
                "<]+)</span>", CFG))
check("I4.8 参数值允许以 HTML < 开头",
      can_spell(P6, "<span", CFG))
P8 = P6 + "\n/etc\n</parameter"
check("I4.9 完整闭合名后进入精确 > 白名单",
      evaluate(P8, CFG)[1] is not None)
check("I4.10 完整闭合名后 > 可拼",
      can_spell(P8, ">", CFG))
check("I4.11 完整闭合名后空格不可拼",
      not can_spell(P8, " >", CFG))
check("I4.12 分 token 畸形 </parameter 空格不可拼",
      not can_spell(P6 + "\n/etc\n", "</parameter >", CFG))
check("I4.13 普通 HTML 闭合标签仍可拼",
      can_spell(P6 + "\n<div>", "</span>", CFG))

# ============ I5: 全流程逐步可生成性 ============
SEQ = ["<tool_call>", "\n", "<function=", "list", "_dir", ">", "\n",
       "<parameter=", "path", ">", "\n", "/etc", "\n", "</parameter>", "\n",
       "<parameter=", "max", "_depth", ">", "\n", "2", "\n",
       "</parameter>", "\n", "</function>", "\n", "</tool_call>"]
acc = ""
all_ok = True
for tok in SEQ[:-1]:
    st, ids, blocked = evaluate(acc, CFG)
    if st == "none":
        acc += tok  # constraint 未激活(块外), 自由
        continue
    if ids is not None and T2I.get(tok) not in ids:
        all_ok = False
        check("I5 token 被拒", False, f"{tok!r} @ {st}")
        break
    if T2I.get(tok) in blocked:
        all_ok = False
        check("I5 token 被黑名单", False, f"{tok!r} @ {st}")
        break
    acc += tok
else:
    acc += SEQ[-1]
    st, _, _ = evaluate(acc, CFG)
    check("I5 全流程可生成且闭合后回 none", st == "none", st)

# ============ I6: has_unclosed 的 Qwen 闭合识别(根因 A) ============
ip2 = P2.find(FUNC_OPEN)
pp2 = P2.rfind(PARAM_OPEN)
check("I6.1 第二参数块 has_unclosed=False",
      not has_unclosed_before(P2, ip2, pp2))
P9 = "<tool_call>\n<function=list_dir>\n<parameter=path>/etc<parameter="
check("I6.2 真未闭合仍 True",
      has_unclosed_before(P9, P9.find(FUNC_OPEN), P9.rfind(PARAM_OPEN)))

# ============ I7: 回归场景复现(409412d6 死循环形态) ============
# 修复后: 每个 </parameter> 闭合后的 S2, 必填 path 已齐时 </function> 可达
LOOP = ("<tool_call>\n<function=list_dir>\n"
        "<parameter=path>/etc</parameter>\n" * 3)
check("I7.1 重复闭合 path 后 Jinja 换行闭合仍可达",
      can_spell(LOOP, "</function>", CFG))
# 发明 pattern 的链路从 S3 被阻断 -> 循环不可形成
BAD = "<tool_call>\n<function=list_dir>\n<parameter=path>/etc</parameter>\n<parameter="
check("I7.2 循环起点(S3)上 pattern 不可拼",
      not can_spell(BAD, "pattern>", CFG))

print()
if FAILS:
    print(f"FAILED: {len(FAILS)} 项 -> {FAILS}")
    raise SystemExit(1)
print("ALL PASS")
