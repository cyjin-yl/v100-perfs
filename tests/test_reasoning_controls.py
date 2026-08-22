import copy
import unittest
from dataclasses import FrozenInstanceError

from src.gateway.reasoning_controls import (
    ReasoningCandidates,
    ReasoningControlError,
    ReasoningSource,
    ThinkingResolution,
    chat_template_kwargs_for,
    extract_anthropic_candidates,
    extract_openai_candidates,
    resolve_reasoning_candidates,
)


class ReasoningResolutionTests(unittest.TestCase):
    def test_public_effort_table_is_total(self):
        cases = (
            ("off", False, "off", None, False, {"enable_thinking": False}),
            ("none", False, "off", None, False, {"enable_thinking": False}),
            ("minimal", True, "minimal", "low", False, {
                "enable_thinking": True,
                "reasoning_effort": "low",
            }),
            ("low", True, "low", "low", False, {
                "enable_thinking": True,
                "reasoning_effort": "low",
            }),
            ("medium", True, "medium", "medium", False, {
                "enable_thinking": True,
                "reasoning_effort": "medium",
            }),
            ("high", True, "high", "xhigh", False, {
                "enable_thinking": True,
                "reasoning_effort": "xhigh",
            }),
            ("xhigh", True, "xhigh", "xhigh", False, {
                "enable_thinking": True,
                "reasoning_effort": "xhigh",
            }),
            ("max", True, "max", "xhigh", True, {
                "enable_thinking": True,
                "reasoning_effort": "xhigh",
            }),
        )
        for effort, enabled, public, template, unlimited, kwargs in cases:
            with self.subTest(effort=effort):
                result = resolve_reasoning_candidates(ReasoningCandidates(
                    efforts=((ReasoningSource.TOP_LEVEL_EFFORT, effort),),
                ))
                self.assertEqual(result.enabled, enabled)
                self.assertEqual(result.public_effort, public)
                self.assertEqual(result.template_effort, template)
                self.assertIsNone(result.explicit_budget_tokens)
                self.assertEqual(result.unlimited, unlimited)
                self.assertEqual(
                    result.sources,
                    (ReasoningSource.TOP_LEVEL_EFFORT,),
                )
                self.assertEqual(chat_template_kwargs_for(result), kwargs)

    def test_no_controls_uses_enabled_template_default(self):
        result = resolve_reasoning_candidates(ReasoningCandidates())
        self.assertEqual(result, ThinkingResolution(
            enabled=True,
            public_effort=None,
            template_effort=None,
            explicit_budget_tokens=None,
            unlimited=False,
            sources=(),
        ))
        self.assertEqual(
            chat_template_kwargs_for(result),
            {"enable_thinking": True},
        )

    def test_toggle_only_preserves_both_explicit_boolean_values(self):
        for toggle, expected in ((False, False), (True, True)):
            with self.subTest(toggle=toggle):
                result = resolve_reasoning_candidates(ReasoningCandidates(
                    toggles=((ReasoningSource.TEMPLATE_TOGGLE, toggle),),
                ))
                self.assertEqual(result.enabled, expected)
                self.assertIsNone(result.public_effort)
                self.assertIsNone(result.template_effort)
                self.assertEqual(
                    chat_template_kwargs_for(result),
                    {"enable_thinking": expected},
                )

    def test_equal_duplicate_efforts_and_toggles_are_accepted(self):
        result = resolve_reasoning_candidates(ReasoningCandidates(
            efforts=(
                (ReasoningSource.NESTED_EFFORT, " LOW "),
                (ReasoningSource.TOP_LEVEL_EFFORT, "low"),
            ),
            toggles=(
                (ReasoningSource.TOP_LEVEL_TOGGLE, True),
                (ReasoningSource.TEMPLATE_TOGGLE, True),
            ),
        ))
        self.assertEqual(result.public_effort, "low")
        self.assertEqual(result.sources, (
            ReasoningSource.TOP_LEVEL_EFFORT,
            ReasoningSource.NESTED_EFFORT,
            ReasoningSource.TEMPLATE_TOGGLE,
            ReasoningSource.TOP_LEVEL_TOGGLE,
        ))

    def test_off_and_none_are_the_same_canonical_public_effort(self):
        result = resolve_reasoning_candidates(ReasoningCandidates(efforts=(
            (ReasoningSource.TOP_LEVEL_EFFORT, "none"),
            (ReasoningSource.NESTED_EFFORT, "off"),
        )))
        self.assertFalse(result.enabled)
        self.assertEqual(result.public_effort, "off")

    def test_distinct_public_efforts_conflict_before_template_mapping(self):
        cases = (
            ("minimal", "low"),
            ("high", "xhigh"),
            ("low", "medium"),
        )
        for first, second in cases:
            with self.subTest(first=first, second=second):
                with self.assertRaises(ReasoningControlError) as raised:
                    resolve_reasoning_candidates(ReasoningCandidates(efforts=(
                        (ReasoningSource.TOP_LEVEL_EFFORT, first),
                        (ReasoningSource.NESTED_EFFORT, second),
                    )))
                self.assertEqual(raised.exception.code, "conflicting_controls")
                self.assertEqual(raised.exception.field_path, "reasoning.effort")
                self.assertEqual(
                    str(raised.exception),
                    "reasoning.effort conflicts with reasoning_effort",
                )

    def test_unequal_toggles_conflict(self):
        with self.assertRaises(ReasoningControlError) as raised:
            resolve_reasoning_candidates(ReasoningCandidates(toggles=(
                (ReasoningSource.TEMPLATE_TOGGLE, False),
                (ReasoningSource.TOP_LEVEL_TOGGLE, True),
            )))
        self.assertEqual(raised.exception.code, "conflicting_controls")
        self.assertEqual(
            raised.exception.field_path,
            "chat_template_kwargs.enable_thinking",
        )
        self.assertEqual(
            str(raised.exception),
            "chat_template_kwargs.enable_thinking conflicts with enable_thinking",
        )

    def test_effort_and_toggle_compatibility_table(self):
        accepted = (
            ("off", False, False),
            ("low", True, True),
        )
        for effort, toggle, enabled in accepted:
            with self.subTest(effort=effort, toggle=toggle):
                result = resolve_reasoning_candidates(ReasoningCandidates(
                    efforts=((ReasoningSource.TOP_LEVEL_EFFORT, effort),),
                    toggles=((ReasoningSource.TOP_LEVEL_TOGGLE, toggle),),
                ))
                self.assertEqual(result.enabled, enabled)

        rejected = (("off", True), ("low", False))
        for effort, toggle in rejected:
            with self.subTest(effort=effort, toggle=toggle):
                with self.assertRaises(ReasoningControlError) as raised:
                    resolve_reasoning_candidates(ReasoningCandidates(
                        efforts=((ReasoningSource.TOP_LEVEL_EFFORT, effort),),
                        toggles=((ReasoningSource.TOP_LEVEL_TOGGLE, toggle),),
                    ))
                self.assertEqual(raised.exception.code, "conflicting_controls")
                self.assertEqual(raised.exception.field_path, "enable_thinking")
                self.assertEqual(
                    str(raised.exception),
                    "enable_thinking conflicts with reasoning_effort",
                )

    def test_anthropic_thinking_types_have_toggle_semantics(self):
        cases = (
            ("enabled", True),
            ("adaptive", True),
            ("disabled", False),
        )
        for thinking_type, enabled in cases:
            with self.subTest(thinking_type=thinking_type):
                result = resolve_reasoning_candidates(ReasoningCandidates(
                    anthropic_thinking_type=thinking_type,
                ))
                self.assertEqual(result.enabled, enabled)
                self.assertIsNone(result.public_effort)
                self.assertIsNone(result.template_effort)
                self.assertEqual(
                    result.sources,
                    (ReasoningSource.ANTHROPIC_THINKING_TYPE,),
                )

    def test_anthropic_enabled_or_adaptive_accepts_named_effort(self):
        for thinking_type in ("enabled", "adaptive"):
            with self.subTest(thinking_type=thinking_type):
                result = resolve_reasoning_candidates(ReasoningCandidates(
                    efforts=((ReasoningSource.ANTHROPIC_OUTPUT_EFFORT, "high"),),
                    anthropic_thinking_type=thinking_type,
                ))
                self.assertEqual(result.public_effort, "high")
                self.assertEqual(result.template_effort, "xhigh")

    def test_budget_only_infers_template_effort_at_exact_boundaries(self):
        cases = (
            (1, "low"),
            (4096, "low"),
            (4097, "medium"),
            (8192, "medium"),
            (8193, "xhigh"),
        )
        for budget, template in cases:
            with self.subTest(budget=budget):
                result = resolve_reasoning_candidates(ReasoningCandidates(
                    openai_budget=budget,
                ))
                self.assertTrue(result.enabled)
                self.assertIsNone(result.public_effort)
                self.assertEqual(result.template_effort, template)
                self.assertEqual(result.explicit_budget_tokens, budget)
                self.assertFalse(result.unlimited)
                self.assertEqual(chat_template_kwargs_for(result), {
                    "enable_thinking": True,
                    "reasoning_effort": template,
                })

    def test_finite_budget_is_metadata_when_non_max_effort_is_named(self):
        result = resolve_reasoning_candidates(ReasoningCandidates(
            efforts=((ReasoningSource.TOP_LEVEL_EFFORT, "high"),),
            openai_budget=1024,
        ))
        self.assertEqual(result.public_effort, "high")
        self.assertEqual(result.template_effort, "xhigh")
        self.assertEqual(result.explicit_budget_tokens, 1024)
        self.assertFalse(result.unlimited)

    def test_equal_cross_protocol_budgets_are_accepted(self):
        result = resolve_reasoning_candidates(ReasoningCandidates(
            openai_budget=4096,
            anthropic_budget=4096,
        ))
        self.assertEqual(result.explicit_budget_tokens, 4096)
        self.assertEqual(result.sources, (
            ReasoningSource.OPENAI_BUDGET,
            ReasoningSource.ANTHROPIC_BUDGET,
        ))

    def test_unequal_cross_protocol_budgets_conflict(self):
        with self.assertRaises(ReasoningControlError) as raised:
            resolve_reasoning_candidates(ReasoningCandidates(
                openai_budget=4096,
                anthropic_budget=8192,
            ))
        self.assertEqual(raised.exception.code, "conflicting_controls")
        self.assertEqual(raised.exception.field_path, "thinking.budget_tokens")
        self.assertEqual(
            str(raised.exception),
            "thinking.budget_tokens conflicts with thinking_budget",
        )

    def test_budget_conflicts_are_stable(self):
        cases = (
            (
                ReasoningCandidates(
                    efforts=((ReasoningSource.TOP_LEVEL_EFFORT, "off"),),
                    openai_budget=4096,
                ),
                "thinking_budget",
                "thinking_budget cannot be used when thinking is disabled",
            ),
            (
                ReasoningCandidates(
                    efforts=((ReasoningSource.TOP_LEVEL_EFFORT, "max"),),
                    openai_budget=4096,
                ),
                "thinking_budget",
                "thinking_budget cannot be used with max reasoning effort",
            ),
            (
                ReasoningCandidates(
                    anthropic_thinking_type="adaptive",
                    anthropic_budget=4096,
                ),
                "thinking.budget_tokens",
                "thinking.budget_tokens cannot be used when thinking.type is adaptive",
            ),
            (
                ReasoningCandidates(
                    anthropic_thinking_type="disabled",
                    anthropic_budget=4096,
                ),
                "thinking.budget_tokens",
                "thinking.budget_tokens cannot be used when thinking.type is disabled",
            ),
        )
        for candidates, field_path, message in cases:
            with self.subTest(field_path=field_path, message=message):
                with self.assertRaises(ReasoningControlError) as raised:
                    resolve_reasoning_candidates(candidates)
                self.assertEqual(raised.exception.code, "conflicting_controls")
                self.assertEqual(raised.exception.field_path, field_path)
                self.assertEqual(str(raised.exception), message)

    def test_resolution_sources_follow_canonical_order_not_input_order(self):
        result = resolve_reasoning_candidates(ReasoningCandidates(
            efforts=(
                (ReasoningSource.NESTED_EFFORT, "medium"),
                (ReasoningSource.TEMPLATE_EFFORT, "medium"),
                (ReasoningSource.TOP_LEVEL_EFFORT, "medium"),
                (ReasoningSource.ANTHROPIC_OUTPUT_EFFORT, "medium"),
            ),
            toggles=(
                (ReasoningSource.TOP_LEVEL_TOGGLE, True),
                (ReasoningSource.TEMPLATE_TOGGLE, True),
            ),
            anthropic_thinking_type="enabled",
            openai_budget=4096,
            anthropic_budget=4096,
        ))
        self.assertEqual(result.sources, tuple(ReasoningSource))

    def test_candidate_and_resolution_records_are_frozen(self):
        candidates = ReasoningCandidates()
        resolution = resolve_reasoning_candidates(candidates)
        with self.assertRaises(FrozenInstanceError):
            candidates.openai_budget = 4096
        with self.assertRaises(FrozenInstanceError):
            resolution.enabled = False


class OpenAIExtractionTests(unittest.TestCase):
    def test_extracts_and_normalizes_every_openai_field_in_source_order(self):
        body = {
            "reasoning_effort": " LOW\t",
            "reasoning": {"effort": " low\n"},
            "chat_template_kwargs": {
                "reasoning_effort": "LoW",
                "enable_thinking": True,
            },
            "enable_thinking": True,
            "thinking_budget": 4096,
        }
        original = copy.deepcopy(body)

        candidates = extract_openai_candidates(body)

        self.assertEqual(candidates, ReasoningCandidates(
            efforts=(
                (ReasoningSource.TEMPLATE_EFFORT, "low"),
                (ReasoningSource.TOP_LEVEL_EFFORT, "low"),
                (ReasoningSource.NESTED_EFFORT, "low"),
            ),
            toggles=(
                (ReasoningSource.TEMPLATE_TOGGLE, True),
                (ReasoningSource.TOP_LEVEL_TOGGLE, True),
            ),
            openai_budget=4096,
        ))
        self.assertEqual(body, original)

    def test_null_fields_and_containers_are_omitted(self):
        candidates = extract_openai_candidates({
            "reasoning_effort": None,
            "reasoning": None,
            "chat_template_kwargs": None,
            "enable_thinking": None,
            "thinking_budget": None,
        })
        self.assertEqual(candidates, ReasoningCandidates())

    def test_literal_extra_body_is_not_unpacked(self):
        candidates = extract_openai_candidates({
            "extra_body": {
                "reasoning_effort": "off",
                "thinking_budget": 4096,
            },
        })
        self.assertEqual(candidates, ReasoningCandidates())

    def test_malformed_containers_are_rejected_before_leaf_fields(self):
        cases = (
            (
                {"reasoning_effort": 1, "reasoning": []},
                "reasoning",
                "reasoning must be an object or null",
            ),
            (
                {
                    "reasoning_effort": 1,
                    "reasoning": {},
                    "chat_template_kwargs": [],
                },
                "chat_template_kwargs",
                "chat_template_kwargs must be an object or null",
            ),
        )
        for body, field_path, message in cases:
            with self.subTest(field_path=field_path):
                with self.assertRaises(ReasoningControlError) as raised:
                    extract_openai_candidates(body)
                self.assertEqual(raised.exception.code, "invalid_type")
                self.assertEqual(raised.exception.field_path, field_path)
                self.assertEqual(str(raised.exception), message)

    def test_effort_schema_rejects_invalid_types_empty_and_unknown_values(self):
        cases = (
            (True, "invalid_type", "reasoning_effort must be a string or null"),
            ({}, "invalid_type", "reasoning_effort must be a string or null"),
            ([], "invalid_type", "reasoning_effort must be a string or null"),
            (" \t\n", "invalid_value", "reasoning_effort must not be empty"),
            (
                "ultra",
                "invalid_value",
                "reasoning_effort must be one of: off, none, minimal, low, medium, high, xhigh, max",
            ),
        )
        for value, code, message in cases:
            with self.subTest(value=value):
                with self.assertRaises(ReasoningControlError) as raised:
                    extract_openai_candidates({"reasoning_effort": value})
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(raised.exception.field_path, "reasoning_effort")
                self.assertEqual(str(raised.exception), message)

    def test_nested_and_template_effort_errors_use_exact_paths(self):
        cases = (
            (
                {"reasoning": {"effort": False}},
                "reasoning.effort",
                "reasoning.effort must be a string or null",
            ),
            (
                {"chat_template_kwargs": {"reasoning_effort": "unknown"}},
                "chat_template_kwargs.reasoning_effort",
                "chat_template_kwargs.reasoning_effort must be one of: off, none, minimal, low, medium, high, xhigh, max",
            ),
        )
        for body, field_path, message in cases:
            with self.subTest(field_path=field_path):
                with self.assertRaises(ReasoningControlError) as raised:
                    extract_openai_candidates(body)
                self.assertEqual(raised.exception.field_path, field_path)
                self.assertEqual(str(raised.exception), message)

    def test_budget_schema_rejects_bool_float_zero_and_negative(self):
        cases = (
            (True, "invalid_type", "thinking_budget must be an integer or null"),
            (1.0, "invalid_type", "thinking_budget must be an integer or null"),
            (0, "invalid_value", "thinking_budget must be at least 1"),
            (-1, "invalid_value", "thinking_budget must be at least 1"),
        )
        for value, code, message in cases:
            with self.subTest(value=value):
                with self.assertRaises(ReasoningControlError) as raised:
                    extract_openai_candidates({"thinking_budget": value})
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(raised.exception.field_path, "thinking_budget")
                self.assertEqual(str(raised.exception), message)

    def test_top_level_and_template_toggles_require_boolean_or_null(self):
        cases = (
            (
                {"enable_thinking": 1},
                "enable_thinking",
                "enable_thinking must be a boolean or null",
            ),
            (
                {"chat_template_kwargs": {"enable_thinking": "false"}},
                "chat_template_kwargs.enable_thinking",
                "chat_template_kwargs.enable_thinking must be a boolean or null",
            ),
        )
        for body, field_path, message in cases:
            with self.subTest(field_path=field_path):
                with self.assertRaises(ReasoningControlError) as raised:
                    extract_openai_candidates(body)
                self.assertEqual(raised.exception.code, "invalid_type")
                self.assertEqual(raised.exception.field_path, field_path)
                self.assertEqual(str(raised.exception), message)

    def test_field_type_errors_precede_enum_and_range_errors(self):
        body = {
            "thinking_budget": 0,
            "chat_template_kwargs": {"enable_thinking": "yes"},
            "reasoning_effort": "unknown",
        }
        with self.assertRaises(ReasoningControlError) as raised:
            extract_openai_candidates(body)
        self.assertEqual(raised.exception.code, "invalid_type")
        self.assertEqual(
            raised.exception.field_path,
            "chat_template_kwargs.enable_thinking",
        )

    def test_same_category_errors_follow_schema_not_json_order(self):
        body = {
            "chat_template_kwargs": {"reasoning_effort": []},
            "reasoning": {"effort": {}},
            "reasoning_effort": True,
        }
        with self.assertRaises(ReasoningControlError) as raised:
            extract_openai_candidates(body)
        self.assertEqual(raised.exception.field_path, "reasoning_effort")


class AnthropicExtractionTests(unittest.TestCase):
    def test_extracts_and_normalizes_anthropic_fields_without_mutation(self):
        body = {
            "output_config": {"effort": " MAX\t"},
            "thinking": {"type": " EnAbLeD\n", "budget_tokens": 16384},
        }
        original = copy.deepcopy(body)

        candidates = extract_anthropic_candidates(body)

        self.assertEqual(candidates, ReasoningCandidates(
            efforts=((ReasoningSource.ANTHROPIC_OUTPUT_EFFORT, "max"),),
            anthropic_thinking_type="enabled",
            anthropic_budget=16384,
        ))
        self.assertEqual(body, original)

    def test_null_anthropic_fields_and_containers_are_omitted(self):
        candidates = extract_anthropic_candidates({
            "output_config": None,
            "thinking": None,
        })
        self.assertEqual(candidates, ReasoningCandidates())

    def test_anthropic_containers_are_validated_in_schema_order(self):
        cases = (
            (
                {"output_config": [], "thinking": []},
                "output_config",
                "output_config must be an object or null",
            ),
            (
                {"output_config": {}, "thinking": []},
                "thinking",
                "thinking must be an object or null",
            ),
        )
        for body, field_path, message in cases:
            with self.subTest(field_path=field_path):
                with self.assertRaises(ReasoningControlError) as raised:
                    extract_anthropic_candidates(body)
                self.assertEqual(raised.exception.code, "invalid_type")
                self.assertEqual(raised.exception.field_path, field_path)
                self.assertEqual(str(raised.exception), message)

    def test_output_effort_accepts_off_and_none(self):
        for effort in ("off", "none"):
            with self.subTest(effort=effort):
                candidates = extract_anthropic_candidates({
                    "output_config": {"effort": effort},
                })
                result = resolve_reasoning_candidates(candidates)
                self.assertFalse(result.enabled)
                self.assertEqual(result.public_effort, "off")

    def test_thinking_type_requires_string_and_known_enum(self):
        cases = (
            (True, "invalid_type", "thinking.type must be a string or null"),
            ("  ", "invalid_value", "thinking.type must not be empty"),
            (
                "manual",
                "invalid_value",
                "thinking.type must be one of: enabled, disabled, adaptive",
            ),
        )
        for value, code, message in cases:
            with self.subTest(value=value):
                with self.assertRaises(ReasoningControlError) as raised:
                    extract_anthropic_candidates({"thinking": {"type": value}})
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(raised.exception.field_path, "thinking.type")
                self.assertEqual(str(raised.exception), message)

    def test_anthropic_budget_requires_integer_at_least_1024(self):
        rejected = (
            (True, "invalid_type", "thinking.budget_tokens must be an integer or null"),
            (1024.0, "invalid_type", "thinking.budget_tokens must be an integer or null"),
            (0, "invalid_value", "thinking.budget_tokens must be at least 1024"),
            (-1, "invalid_value", "thinking.budget_tokens must be at least 1024"),
            (1023, "invalid_value", "thinking.budget_tokens must be at least 1024"),
        )
        for value, code, message in rejected:
            with self.subTest(value=value):
                with self.assertRaises(ReasoningControlError) as raised:
                    extract_anthropic_candidates({
                        "thinking": {"type": "enabled", "budget_tokens": value},
                    })
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(
                    raised.exception.field_path,
                    "thinking.budget_tokens",
                )
                self.assertEqual(str(raised.exception), message)

        candidates = extract_anthropic_candidates({
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        })
        self.assertEqual(candidates.anthropic_budget, 1024)

    def test_adaptive_and_disabled_reject_anthropic_budget(self):
        for thinking_type in ("adaptive", "disabled"):
            with self.subTest(thinking_type=thinking_type):
                candidates = extract_anthropic_candidates({
                    "thinking": {
                        "type": thinking_type,
                        "budget_tokens": 4096,
                    },
                })
                with self.assertRaises(ReasoningControlError) as raised:
                    resolve_reasoning_candidates(candidates)
                self.assertEqual(raised.exception.code, "conflicting_controls")
                self.assertEqual(
                    raised.exception.field_path,
                    "thinking.budget_tokens",
                )
                self.assertEqual(
                    str(raised.exception),
                    f"thinking.budget_tokens cannot be used when thinking.type is {thinking_type}",
                )

    def test_anthropic_leaf_error_order_is_type_then_enum_then_range(self):
        body = {
            "thinking": {"type": "unknown", "budget_tokens": 1024.0},
            "output_config": {"effort": "unknown"},
        }
        with self.assertRaises(ReasoningControlError) as raised:
            extract_anthropic_candidates(body)
        self.assertEqual(raised.exception.code, "invalid_type")
        self.assertEqual(raised.exception.field_path, "thinking.budget_tokens")


if __name__ == "__main__":
    unittest.main()
