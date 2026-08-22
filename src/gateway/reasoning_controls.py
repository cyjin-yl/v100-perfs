"""Pure extraction and resolution for public reasoning controls."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


class ReasoningSource(str, Enum):
    TEMPLATE_EFFORT = "template_effort"
    TOP_LEVEL_EFFORT = "top_level_effort"
    NESTED_EFFORT = "nested_effort"
    ANTHROPIC_OUTPUT_EFFORT = "anthropic_output_effort"
    TEMPLATE_TOGGLE = "template_toggle"
    TOP_LEVEL_TOGGLE = "top_level_toggle"
    ANTHROPIC_THINKING_TYPE = "anthropic_thinking_type"
    OPENAI_BUDGET = "openai_budget"
    ANTHROPIC_BUDGET = "anthropic_budget"


@dataclass(frozen=True)
class ReasoningCandidates:
    efforts: tuple[tuple[ReasoningSource, str], ...] = ()
    toggles: tuple[tuple[ReasoningSource, bool], ...] = ()
    openai_budget: int | None = None
    anthropic_thinking_type: str | None = None
    anthropic_budget: int | None = None


@dataclass(frozen=True)
class ThinkingResolution:
    enabled: bool
    public_effort: str | None
    template_effort: str | None
    explicit_budget_tokens: int | None
    unlimited: bool
    sources: tuple[ReasoningSource, ...]


class ReasoningControlError(ValueError):
    def __init__(self, code: str, field_path: str, message: str):
        self.code = code
        self.field_path = field_path
        super().__init__(message)


PUBLIC_TO_TEMPLATE = {
    "off": None,
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "xhigh",
    "xhigh": "xhigh",
    "max": "xhigh",
}

_EFFORT_VALUES = ("off", "none", "minimal", "low", "medium", "high", "xhigh", "max")
_THINKING_TYPES = ("enabled", "disabled", "adaptive")
_ASCII_WHITESPACE = " \t\n\r\v\f"

_SOURCE_PATHS = {
    ReasoningSource.TEMPLATE_EFFORT: "chat_template_kwargs.reasoning_effort",
    ReasoningSource.TOP_LEVEL_EFFORT: "reasoning_effort",
    ReasoningSource.NESTED_EFFORT: "reasoning.effort",
    ReasoningSource.ANTHROPIC_OUTPUT_EFFORT: "output_config.effort",
    ReasoningSource.TEMPLATE_TOGGLE: "chat_template_kwargs.enable_thinking",
    ReasoningSource.TOP_LEVEL_TOGGLE: "enable_thinking",
    ReasoningSource.ANTHROPIC_THINKING_TYPE: "thinking.type",
    ReasoningSource.OPENAI_BUDGET: "thinking_budget",
    ReasoningSource.ANTHROPIC_BUDGET: "thinking.budget_tokens",
}

_FIELD_ORDER = {
    "reasoning_effort": 0,
    "reasoning.effort": 1,
    "chat_template_kwargs.reasoning_effort": 2,
    "output_config.effort": 3,
    "enable_thinking": 4,
    "chat_template_kwargs.enable_thinking": 5,
    "thinking.type": 6,
    "thinking_budget": 7,
    "thinking.budget_tokens": 8,
}


def _invalid_type(field_path: str, expected: str) -> ReasoningControlError:
    return ReasoningControlError(
        "invalid_type",
        field_path,
        f"{field_path} must be {expected} or null",
    )


def _invalid_value(field_path: str, message: str) -> ReasoningControlError:
    return ReasoningControlError("invalid_value", field_path, message)


def _conflict(field_path: str, message: str) -> ReasoningControlError:
    return ReasoningControlError("conflicting_controls", field_path, message)


def _normalized_string(value: str) -> str:
    return value.strip(_ASCII_WHITESPACE).lower()


def _normalize_effort(value: str, field_path: str) -> str:
    normalized = _normalized_string(value)
    if not normalized:
        raise _invalid_value(field_path, f"{field_path} must not be empty")
    if normalized not in _EFFORT_VALUES:
        accepted = ", ".join(_EFFORT_VALUES)
        raise _invalid_value(
            field_path,
            f"{field_path} must be one of: {accepted}",
        )
    return normalized


def _normalize_thinking_type(value: str) -> str:
    field_path = "thinking.type"
    normalized = _normalized_string(value)
    if not normalized:
        raise _invalid_value(field_path, f"{field_path} must not be empty")
    if normalized not in _THINKING_TYPES:
        accepted = ", ".join(_THINKING_TYPES)
        raise _invalid_value(
            field_path,
            f"{field_path} must be one of: {accepted}",
        )
    return normalized


def _field_sort_key(source_and_value: tuple[ReasoningSource, object]) -> int:
    source, _ = source_and_value
    return _FIELD_ORDER[_SOURCE_PATHS[source]]


def _validate_candidate_types(candidates: ReasoningCandidates) -> None:
    for source, effort in sorted(candidates.efforts, key=_field_sort_key):
        if not isinstance(effort, str):
            raise _invalid_type(_SOURCE_PATHS[source], "a string")

    for source, toggle in sorted(candidates.toggles, key=_field_sort_key):
        if not isinstance(toggle, bool):
            raise _invalid_type(_SOURCE_PATHS[source], "a boolean")

    if (
        candidates.anthropic_thinking_type is not None
        and not isinstance(candidates.anthropic_thinking_type, str)
    ):
        raise _invalid_type("thinking.type", "a string")

    budgets = (
        ("thinking_budget", candidates.openai_budget),
        ("thinking.budget_tokens", candidates.anthropic_budget),
    )
    for field_path, budget in budgets:
        if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int)):
            raise _invalid_type(field_path, "an integer")


def _validated_candidate_values(
    candidates: ReasoningCandidates,
) -> tuple[
    list[tuple[ReasoningSource, str]],
    list[tuple[ReasoningSource, bool]],
    str | None,
]:
    efforts: list[tuple[ReasoningSource, str]] = []
    for source, effort in sorted(candidates.efforts, key=_field_sort_key):
        efforts.append((source, _normalize_effort(effort, _SOURCE_PATHS[source])))

    thinking_type = candidates.anthropic_thinking_type
    if thinking_type is not None:
        thinking_type = _normalize_thinking_type(thinking_type)

    if candidates.openai_budget is not None and candidates.openai_budget < 1:
        raise _invalid_value(
            "thinking_budget",
            "thinking_budget must be at least 1",
        )
    if candidates.anthropic_budget is not None and candidates.anthropic_budget < 1024:
        raise _invalid_value(
            "thinking.budget_tokens",
            "thinking.budget_tokens must be at least 1024",
        )

    toggles = sorted(candidates.toggles, key=_field_sort_key)
    return efforts, toggles, thinking_type


def _ordered_sources(candidates: ReasoningCandidates) -> tuple[ReasoningSource, ...]:
    present = {source for source, _ in candidates.efforts}
    present.update(source for source, _ in candidates.toggles)
    if candidates.anthropic_thinking_type is not None:
        present.add(ReasoningSource.ANTHROPIC_THINKING_TYPE)
    if candidates.openai_budget is not None:
        present.add(ReasoningSource.OPENAI_BUDGET)
    if candidates.anthropic_budget is not None:
        present.add(ReasoningSource.ANTHROPIC_BUDGET)
    return tuple(source for source in ReasoningSource if source in present)


def _single_effort(
    efforts: list[tuple[ReasoningSource, str]],
) -> tuple[str | None, ReasoningSource | None]:
    if not efforts:
        return None, None

    first_source, first_effort = efforts[0]
    canonical = "off" if first_effort == "none" else first_effort
    for source, effort in efforts[1:]:
        current = "off" if effort == "none" else effort
        if current != canonical:
            field_path = _SOURCE_PATHS[source]
            first_path = _SOURCE_PATHS[first_source]
            raise _conflict(
                field_path,
                f"{field_path} conflicts with {first_path}",
            )
    return canonical, first_source


def _single_toggle(
    toggles: list[tuple[ReasoningSource, bool]],
    thinking_type: str | None,
) -> tuple[bool | None, ReasoningSource | None]:
    effective = list(toggles)
    if thinking_type is not None:
        effective.append((
            ReasoningSource.ANTHROPIC_THINKING_TYPE,
            thinking_type != "disabled",
        ))
    effective.sort(key=_field_sort_key)
    if not effective:
        return None, None

    first_source, first_toggle = effective[0]
    for source, toggle in effective[1:]:
        if toggle != first_toggle:
            field_path = _SOURCE_PATHS[source]
            first_path = _SOURCE_PATHS[first_source]
            raise _conflict(
                field_path,
                f"{field_path} conflicts with {first_path}",
            )
    return first_toggle, first_source


def resolve_reasoning_candidates(
    candidates: ReasoningCandidates,
) -> ThinkingResolution:
    """Resolve typed candidates into one canonical, backend-neutral result."""
    _validate_candidate_types(candidates)
    efforts, toggles, thinking_type = _validated_candidate_values(candidates)

    public_effort, effort_source = _single_effort(efforts)
    toggle, toggle_source = _single_toggle(toggles, thinking_type)

    if public_effort is not None and toggle is not None:
        effort_enabled = public_effort != "off"
        if effort_enabled != toggle:
            toggle_path = _SOURCE_PATHS[toggle_source]
            effort_path = _SOURCE_PATHS[effort_source]
            raise _conflict(
                toggle_path,
                f"{toggle_path} conflicts with {effort_path}",
            )

    enabled = public_effort != "off" if public_effort is not None else (
        True if toggle is None else toggle
    )

    if (
        candidates.anthropic_budget is not None
        and thinking_type in ("adaptive", "disabled")
    ):
        field_path = "thinking.budget_tokens"
        raise _conflict(
            field_path,
            f"{field_path} cannot be used when thinking.type is {thinking_type}",
        )

    if (
        candidates.openai_budget is not None
        and candidates.anthropic_budget is not None
        and candidates.openai_budget != candidates.anthropic_budget
    ):
        raise _conflict(
            "thinking.budget_tokens",
            "thinking.budget_tokens conflicts with thinking_budget",
        )

    budget = (
        candidates.openai_budget
        if candidates.openai_budget is not None
        else candidates.anthropic_budget
    )
    budget_path = (
        "thinking_budget"
        if candidates.openai_budget is not None
        else "thinking.budget_tokens"
    )

    if budget is not None and not enabled:
        raise _conflict(
            budget_path,
            f"{budget_path} cannot be used when thinking is disabled",
        )
    if budget is not None and public_effort == "max":
        raise _conflict(
            budget_path,
            f"{budget_path} cannot be used with max reasoning effort",
        )

    template_effort = (
        PUBLIC_TO_TEMPLATE[public_effort]
        if public_effort is not None
        else None
    )
    if budget is not None and public_effort is None:
        if budget <= 4096:
            template_effort = "low"
        elif budget <= 8192:
            template_effort = "medium"
        else:
            template_effort = "xhigh"

    return ThinkingResolution(
        enabled=enabled,
        public_effort=public_effort,
        template_effort=template_effort,
        explicit_budget_tokens=budget,
        unlimited=public_effort == "max",
        sources=_ordered_sources(candidates),
    )


def chat_template_kwargs_for(
    resolution: ThinkingResolution,
) -> dict[str, object]:
    """Serialize only the template controls represented by a resolution."""
    kwargs: dict[str, object] = {"enable_thinking": resolution.enabled}
    if resolution.enabled and resolution.template_effort is not None:
        kwargs["reasoning_effort"] = resolution.template_effort
    return kwargs


def _optional_container(
    body: Mapping[str, object],
    field_path: str,
) -> Mapping[str, object]:
    value = body.get(field_path)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise _invalid_type(field_path, "an object")
    return value


def _validate_effort_types(
    fields: tuple[tuple[str, ReasoningSource, object], ...],
) -> None:
    for field_path, _, value in fields:
        if value is not None and not isinstance(value, str):
            raise _invalid_type(field_path, "a string")


def _validate_toggle_types(
    fields: tuple[tuple[str, ReasoningSource, object], ...],
) -> None:
    for field_path, _, value in fields:
        if value is not None and not isinstance(value, bool):
            raise _invalid_type(field_path, "a boolean")


def _ordered_effort_candidates(
    fields: tuple[tuple[str, ReasoningSource, object], ...],
) -> tuple[tuple[ReasoningSource, str], ...]:
    normalized = {
        source: _normalize_effort(value, field_path)
        for field_path, source, value in fields
        if value is not None
    }
    return tuple(
        (source, normalized[source])
        for source in ReasoningSource
        if source in normalized
    )


def _ordered_toggle_candidates(
    fields: tuple[tuple[str, ReasoningSource, object], ...],
) -> tuple[tuple[ReasoningSource, bool], ...]:
    toggles = {
        source: value
        for _, source, value in fields
        if value is not None
    }
    return tuple(
        (source, toggles[source])
        for source in ReasoningSource
        if source in toggles
    )


def extract_openai_candidates(
    body: Mapping[str, object],
) -> ReasoningCandidates:
    """Extract validated OpenAI controls without mutating the request body."""
    reasoning = _optional_container(body, "reasoning")
    template_kwargs = _optional_container(body, "chat_template_kwargs")

    effort_fields = (
        (
            "reasoning_effort",
            ReasoningSource.TOP_LEVEL_EFFORT,
            body.get("reasoning_effort"),
        ),
        (
            "reasoning.effort",
            ReasoningSource.NESTED_EFFORT,
            reasoning.get("effort"),
        ),
        (
            "chat_template_kwargs.reasoning_effort",
            ReasoningSource.TEMPLATE_EFFORT,
            template_kwargs.get("reasoning_effort"),
        ),
    )
    toggle_fields = (
        (
            "enable_thinking",
            ReasoningSource.TOP_LEVEL_TOGGLE,
            body.get("enable_thinking"),
        ),
        (
            "chat_template_kwargs.enable_thinking",
            ReasoningSource.TEMPLATE_TOGGLE,
            template_kwargs.get("enable_thinking"),
        ),
    )
    budget = body.get("thinking_budget")

    _validate_effort_types(effort_fields)
    _validate_toggle_types(toggle_fields)
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int)):
        raise _invalid_type("thinking_budget", "an integer")

    efforts = _ordered_effort_candidates(effort_fields)
    toggles = _ordered_toggle_candidates(toggle_fields)
    if budget is not None and budget < 1:
        raise _invalid_value(
            "thinking_budget",
            "thinking_budget must be at least 1",
        )

    return ReasoningCandidates(
        efforts=efforts,
        toggles=toggles,
        openai_budget=budget,
    )


def extract_anthropic_candidates(
    body: Mapping[str, object],
) -> ReasoningCandidates:
    """Extract validated Anthropic controls without mutating the request body."""
    output_config = _optional_container(body, "output_config")
    thinking = _optional_container(body, "thinking")

    effort_fields = ((
        "output_config.effort",
        ReasoningSource.ANTHROPIC_OUTPUT_EFFORT,
        output_config.get("effort"),
    ),)
    thinking_type = thinking.get("type")
    budget = thinking.get("budget_tokens")

    _validate_effort_types(effort_fields)
    if thinking_type is not None and not isinstance(thinking_type, str):
        raise _invalid_type("thinking.type", "a string")
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int)):
        raise _invalid_type("thinking.budget_tokens", "an integer")

    efforts = _ordered_effort_candidates(effort_fields)
    if thinking_type is not None:
        thinking_type = _normalize_thinking_type(thinking_type)
    if budget is not None and budget < 1024:
        raise _invalid_value(
            "thinking.budget_tokens",
            "thinking.budget_tokens must be at least 1024",
        )

    return ReasoningCandidates(
        efforts=efforts,
        anthropic_thinking_type=thinking_type,
        anthropic_budget=budget,
    )
