"""Pure FastLLM protocol adapters used by thinking_proxy.

FastLLM's bundled Jinja parser cannot render the macro-heavy Qwen3.6 template,
so the proxy renders it with Jinja2 and sends the result as a raw prompt.  The
C++ apiserver returns generated tool XML as plain content; this module promotes
that XML to standard OpenAI ``tool_calls`` for agent clients.
"""

from __future__ import annotations

import copy
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

# 乘法 repeat penalty 与 OpenAI additive presence/frequency penalty 是三套
# 独立语义。Qwen3.8 官方默认是 1.0;旧衍生模型仍保留 1.05,profile 可显式覆盖。
_CONFIGURED_REPEAT_PENALTY = os.environ.get("FASTLLM_DEFAULT_REPEAT_PENALTY")
_LEGACY_REPEAT_PENALTY = float(_CONFIGURED_REPEAT_PENALTY or "1.05")
_QWEN38_REPEAT_PENALTY = float(_CONFIGURED_REPEAT_PENALTY or "1.0")



_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>", re.DOTALL
)
_FUNCTION_TAG_RE = re.compile(
    r"<(?:fu[n]?ction=([^>\n]+)|([A-Za-z0-9_.\-]+))>"
)
_PARAMETER_RE = re.compile(
    r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.DOTALL
)

_QWEN_STOP_MARKERS = ("<|im_end|>", "<|endoftext|>")



def _raise_exception(message: str) -> None:
    raise ValueError(message)


def _fill_missing_tool_call_content(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for message in messages:
        if (
            message.get("role") == "assistant"
            and message.get("tool_calls")
            and message.get("content") is None
        ):
            message["content"] = ""
    return messages


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = _fill_missing_tool_call_content(copy.deepcopy(messages))
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
        content = message.get("content")
        if isinstance(content, list):
            # 模板只认 {"type": "video", "video": ...};把 OpenAI/vLLM 标准
            # video_url 块转成该形态(仅影响模板渲染副本,后端仍收原始
            # messages 抽取 URL)。
            for block in content:
                if isinstance(block, dict) and block.get("type") == "video_url":
                    video_url = block.get("video_url") or {}
                    url = video_url.get("url", "")
                    block.clear()
                    block["type"] = "video"
                    block["video"] = url
    return normalized

_SCHEMA_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _schema_name(value: Any) -> str | None:
    text = str(value or "")
    return text if _SCHEMA_NAME_RE.fullmatch(text) else None




_EXAMPLE_BLOCK_RE = re.compile(
    r"<example(?:\s[^>]*)?>(.*?)</example>", re.DOTALL | re.IGNORECASE)


def _example_schema_branches(
    function: dict[str, Any],
) -> tuple[str, list[tuple[str, tuple[str, ...]]]] | None:
    name = _schema_name(function.get("name"))
    parameters = function.get("parameters") or {}
    properties = parameters.get("properties") or {}
    description = function.get("description") or ""
    if not name or not isinstance(properties, dict) or \
            not isinstance(description, str):
        return None
    enum_fields = {
        key: schema.get("enum")
        for key, schema in properties.items()
        if isinstance(schema, dict) and
        isinstance(schema.get("enum"), list) and
        all(_schema_name(item) for item in schema["enum"])
    }
    if not enum_fields:
        return None
    calls: list[tuple[dict[str, str], tuple[str, ...]]] = []
    call_re = re.compile(
        rf"\b{re.escape(name)}\s*\((.*)\)\s*$", re.DOTALL)
    for block in _EXAMPLE_BLOCK_RE.findall(description):
        match = call_re.search(block.strip())
        if match is None:
            continue
        arguments = match.group(1)
        names = tuple(dict.fromkeys(
            item for item in re.findall(
                r"(?:^|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s*=",
                arguments)
            if item in properties))
        literals: dict[str, str] = {}
        for key in enum_fields:
            literal = re.search(
                rf"(?:^|,)\s*{re.escape(key)}\s*=\s*"
                r"([\"'])(.*?)\1",
                arguments, re.DOTALL)
            if literal is not None and _schema_name(literal.group(2)):
                literals[key] = literal.group(2)
        calls.append((literals, names))
    for key in (["op"] if "op" in enum_fields else []) + [
            item for item in enum_fields if item != "op"]:
        branches = [
            (literals[key], names)
            for literals, names in calls if key in literals]
        if branches and len({value for value, _ in branches}) >= 2:
            return key, branches
    return None


def _enrich_tool_schemas(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched = copy.deepcopy(tools)
    for tool in enriched:
        function = tool.get("function") or {}
        parameters = function.get("parameters") or {}
        if not isinstance(parameters, dict) or \
                parameters.get("oneOf") or parameters.get("anyOf"):
            continue
        inferred = _example_schema_branches(function)
        if inferred is None:
            continue
        discriminant, examples = inferred
        properties = parameters.get("properties") or {}
        enum_values = properties.get(discriminant, {}).get("enum") or []
        original_required = [
            item for item in parameters.get("required") or []
            if item in properties]
        additional = parameters.get("additionalProperties", False)
        branches: list[dict[str, Any]] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        covered: set[str] = set()
        for value, example_names in examples:
            required = tuple(dict.fromkeys(
                [*original_required, *example_names]))
            shape = (value, tuple(sorted(required)))
            if shape in seen:
                continue
            seen.add(shape)
            covered.add(value)
            branch_properties = {
                name: copy.deepcopy(properties[name])
                for name in required if name in properties}
            branch_properties[discriminant] = copy.deepcopy(
                properties[discriminant])
            branch_properties[discriminant].pop("enum", None)
            branch_properties[discriminant]["const"] = value
            branches.append({
                "type": "object",
                "properties": branch_properties,
                "required": list(required),
                "additionalProperties": additional,
            })
        for value in enum_values:
            safe_value = _schema_name(value)
            if not safe_value or safe_value in covered:
                continue
            fallback_properties = copy.deepcopy(properties)
            fallback_properties[discriminant].pop("enum", None)
            fallback_properties[discriminant]["const"] = safe_value
            branches.append({
                "type": "object",
                "properties": fallback_properties,
                "required": list(original_required),
                "additionalProperties": additional,
            })
        if branches:
            function["parameters"] = {"anyOf": branches}
    return enriched






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
    tools = copy.deepcopy(body.get("tools") or [])
    messages = _normalize_messages(body.get("messages") or [])
    template_kwargs = {
        "messages": messages,
        "tools": tools,
        "add_generation_prompt": True,
        "enable_thinking": kwargs.get("enable_thinking", True),
        "preserve_thinking": kwargs.get("preserve_thinking", False),
        "add_vision_id": kwargs.get("add_vision_id", False),
    }
    # reasoning_effort 别名归一(用户约定):
    # high/max/ultra → xhigh,medium/low 原样,off/'' → 关闭思考;
    # 未列出的值兜底 medium。
    if "reasoning_effort" in kwargs:
        effort = str(kwargs.get("reasoning_effort") or "").lower()
        aliases = {
            "xhigh": "xhigh", "high": "xhigh", "max": "xhigh",
            "ultra": "xhigh",
            "medium": "medium", "low": "low",
            "off": "off", "": "off",
        }
        resolved = aliases.get(effort, "medium")
        if resolved == "off":
            template_kwargs["enable_thinking"] = False
        else:
            template_kwargs["reasoning_effort"] = resolved
    return template.render(**template_kwargs)




def _body_has_images(body: dict[str, Any]) -> bool:
    for message in body.get("messages") or []:
        content = message.get("content", "")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") in {
                "image",
                "image_url",
            }:
                return True
    return False


def prepare_fastllm_body(
    body: dict[str, Any], template_path: str | Path
) -> dict[str, Any]:
    prepared = copy.deepcopy(body)
    _fill_missing_tool_call_content(prepared.get("messages") or [])
    prepared["tools"] = _enrich_tool_schemas(
        prepared.get("tools") or [])
    prepared["prompt"] = render_fastllm_prompt(body, template_path)
    prepared["raw_prompt"] = True
    stops = prepared.get("stop")
    if stops is None:
        merged_stops = []
    elif isinstance(stops, str):
        merged_stops = [stops]
    else:
        merged_stops = list(stops)
    for marker in _QWEN_STOP_MARKERS:
        if marker not in merged_stops:
            merged_stops.append(marker)
    prepared["stop"] = merged_stops
    # Qwen3.8 官方采样档按 thinking 模式分流。OpenAI 协议没有 top_k,
    # 因而必须由 adapter 补齐;客户端显式值始终优先。temperature=0 仍保留
    # 严格贪婪语义。FastLLM 对 presence/frequency/repeat 三种 penalty
    # 分别实现,这里不做字段偷换。
    model_name = str(prepared.get("model") or "").lower()
    enable_thinking = bool(
        (prepared.get("chat_template_kwargs") or {}).get(
            "enable_thinking", True))
    if "qwen3.8" in model_name:
        prepared.setdefault("top_k", 20)
        prepared.setdefault("top_p", 0.95 if enable_thinking else 0.8)
        prepared.setdefault("temperature", 1.0 if enable_thinking else 0.7)
        prepared.setdefault(
            "presence_penalty", 0.0 if enable_thinking else 1.5)
    else:
        prepared.setdefault("top_k", 20)
        prepared.setdefault("top_p", 0.8)
        prepared.setdefault("temperature", 0.6)
    if "repeat_penalty" not in prepared:
        prepared["repeat_penalty"] = (
            _QWEN38_REPEAT_PENALTY
            if "qwen3.8" in model_name
            else _LEGACY_REPEAT_PENALTY)
    return prepared


def _parse_parameter_value(raw: str) -> Any:
    value = raw.strip()
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def parse_fastllm_tool_calls(content: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for block_match in _TOOL_CALL_BLOCK_RE.finditer(content):
        inner = block_match.group(1)
        tag = _FUNCTION_TAG_RE.search(inner)
        if tag is None:
            # Qwen native JSON form: {"name": ..., "arguments": {...}}
            # (single object or an array of objects)
            parsed_json = None
            try:
                parsed_json = json.loads(inner.strip())
            except json.JSONDecodeError:
                parsed_json = None
            if isinstance(parsed_json, dict):
                parsed_json = [parsed_json]
            if isinstance(parsed_json, list):
                for item in parsed_json:
                    if not isinstance(item, dict) or not item.get("name"):
                        continue
                    raw_args = item.get("arguments")
                    if isinstance(raw_args, str):
                        try:
                            raw_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            pass
                    calls.append({
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": str(item["name"]),
                            "arguments": json.dumps(
                                raw_args if isinstance(raw_args, dict) else {},
                                ensure_ascii=False),
                        },
                    })
            continue
        name = (tag.group(1) or tag.group(2) or "").strip()
        if not name or name == "parameter":
            continue
        arguments = {
            parameter.group(1).strip(): _parse_parameter_value(parameter.group(2))
            for parameter in _PARAMETER_RE.finditer(inner)
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


def _truncate_qwen_tail(content: str) -> str:
    positions = [
        position for marker in _QWEN_STOP_MARKERS
        if (position := content.find(marker)) >= 0
    ]
    return content[:min(positions)] if positions else content


def _split_reasoning(content: str) -> tuple[str, str]:
    if "</think>" not in content:
        return "", _truncate_qwen_tail(content).strip()
    reasoning, answer = content.split("</think>", 1)
    if "<think>" in reasoning:
        reasoning = reasoning.split("<think>", 1)[1]
    answer = _truncate_qwen_tail(answer)
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
