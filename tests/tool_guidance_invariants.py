#!/usr/bin/env python3
"""Internal schema guidance must not mutate the rendered user prompt."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "fastllm_adapter", ROOT / "v100-perfs" / "fastllm_adapter.py")
assert SPEC is not None and SPEC.loader is not None
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


flat_todo = {
    "type": "function",
    "function": {
        "name": "todo",
        "description": """
<examples>
<example>
todo(i="intent", op="init", list=[{"phase":"P","items":["T"]}])
</example>
<example>
todo(i="intent", op="done", task="T")
</example>
<example>
todo(i="intent", op="done", phase="P")
</example>
<example>
todo(i="intent", op="drop", task="T")
</example>
<example>
todo(i="intent", op="append", phase="P", items=["T"])
</example>
</examples>
""",
        "parameters": {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": [
                    "init", "start", "done", "rm", "drop", "block",
                    "unblock", "append", "view"]},
                "i": {"type": "string"},
                "list": {"type": "array"},
                "task": {"type": "string"},
                "phase": {"type": "string"},
                "items": {"type": "array"},
            },
            "required": ["op", "i"],
        },
    },
}

enriched = adapter._enrich_tool_schemas([flat_todo])
branches = enriched[0]["function"]["parameters"]["anyOf"]
branch_shapes = {
    (
        branch["properties"]["op"].get("const"),
        tuple(sorted(name for name in branch["required"]
                     if name not in {"op", "i"})),
    )
    for branch in branches
}
assert ("init", ("list",)) in branch_shapes, branch_shapes
assert ("done", ("task",)) in branch_shapes, branch_shapes
assert ("done", ("phase",)) in branch_shapes, branch_shapes
assert ("append", ("items", "phase")) in branch_shapes, branch_shapes
for fallback_op in {"start", "rm", "block", "unblock", "view"}:
    assert (fallback_op, ()) in branch_shapes, branch_shapes

body = {
    "messages": [
        {"role": "system", "content": "BASE SYSTEM"},
        {"role": "user", "content": "run a task"},
    ],
    "tools": [flat_todo],
}
template = ROOT / "models" / "Qwen3.8-27B-Uncensored-Cyber-BF16" / \
    "chat_template.jinja"
rendered = adapter.render_fastllm_prompt(body, template)
prepared = adapter.prepare_fastllm_body(body, template)
assert prepared["prompt"] == rendered, "internal schema enrichment changed prompt"
assert "Tool signatures for this request:" not in prepared["prompt"]
assert prepared["messages"] == body["messages"]
assert "anyOf" in prepared["tools"][0]["function"]["parameters"]
assert "anyOf" not in body["tools"][0]["function"]["parameters"]

print("ALL PASS")
