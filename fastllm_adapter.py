"""Pure FastLLM protocol adapters used by thinking_proxy.

FastLLM's bundled Jinja parser cannot render the macro-heavy Qwen3.6 template,
so the proxy renders it with Jinja2 and sends the result as a raw prompt.  The
C++ apiserver returns generated tool XML as plain content; this module promotes
that XML to standard OpenAI ``tool_calls`` for agent clients.
"""

from __future__ import annotations

import copy
import json
import re
import uuid
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined


_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<fu[n]?ction=([^>\n]+)>\s*(.*?)\s*</fu[n]?ction>\s*</tool_call>",
    re.DOTALL,
)
_PARAMETER_RE = re.compile(
    r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.DOTALL
)


def _raise_exception(message: str) -> None:
    raise ValueError(message)


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(messages)
    for message in normalized:
        message.setdefault("tool_calls", [])
        for tool_call in message.get("tool_calls", []):
            function = tool_call.get("function", tool_call)
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    function["arguments"] = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
    return normalized


def render_fastllm_prompt(body: dict[str, Any], template_path: str | Path) -> str:
    path = Path(template_path)
    environment = Environment(
        loader=FileSystemLoader(str(path.parent)),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    environment.globals["raise_exception"] = _raise_exception
    template = environment.get_template(path.name)
    kwargs = body.get("chat_template_kwargs") or {}
    return template.render(
        messages=_normalize_messages(body.get("messages") or []),
        tools=copy.deepcopy(body.get("tools") or []),
        add_generation_prompt=True,
        enable_thinking=kwargs.get("enable_thinking", True),
        preserve_thinking=kwargs.get("preserve_thinking", False),
        add_vision_id=kwargs.get("add_vision_id", False),
    )


def prepare_fastllm_body(
    body: dict[str, Any], template_path: str | Path
) -> dict[str, Any]:
    prepared = copy.deepcopy(body)
    prepared["prompt"] = render_fastllm_prompt(body, template_path)
    prepared["raw_prompt"] = True
    stops = prepared.get("stop")
    if stops is None:
        prepared["stop"] = ["<|im_end|>"]
    elif isinstance(stops, str):
        prepared["stop"] = [stops, "<|im_end|>"] if stops != "<|im_end|>" else [stops]
    elif "<|im_end|>" not in stops:
        prepared["stop"] = [*stops, "<|im_end|>"]
    return prepared


def _parse_parameter_value(raw: str) -> Any:
    value = raw.strip()
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def parse_fastllm_tool_calls(content: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for match in _TOOL_CALL_RE.finditer(content):
        name = match.group(1).strip()
        arguments = {
            parameter.group(1).strip(): _parse_parameter_value(parameter.group(2))
            for parameter in _PARAMETER_RE.finditer(match.group(2))
        }
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        })
    return calls


def _split_reasoning(content: str) -> tuple[str, str]:
    if "</think>" not in content:
        return "", content.strip()
    reasoning, answer = content.split("</think>", 1)
    if "<think>" in reasoning:
        reasoning = reasoning.split("<think>", 1)[1]
    answer = answer.split("<|im_end|>", 1)[0]
    return reasoning.strip(), answer.strip()


def adapt_fastllm_response(response: dict[str, Any]) -> dict[str, Any]:
    adapted = copy.deepcopy(response)
    choices = adapted.get("choices") or []
    if not choices:
        return adapted
    choice = choices[0]
    message = choice.get("message") or {}
    raw_content = message.get("content") or ""
    reasoning, content = _split_reasoning(raw_content)
    tool_calls = parse_fastllm_tool_calls(content)
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        first_tool = content.find("<tool_call>")
        message["content"] = content[:first_tool].strip() if first_tool >= 0 else ""
        message["tool_calls"] = tool_calls
        choice["finish_reason"] = "tool_calls"
    else:
        message["content"] = content
        if choice.get("finish_reason") is None:
            choice["finish_reason"] = "stop"
    choice["message"] = message
    return adapted
