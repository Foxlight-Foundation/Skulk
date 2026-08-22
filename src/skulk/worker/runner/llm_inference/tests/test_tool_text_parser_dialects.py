"""Tool-call dialect coverage for the shared text parser.

Models do not agree on how to say "call this function". A parser that knows
only one dialect does not merely miss the call: the markup falls through to
`content`, so the caller receives template scaffolding as if it were the
model's answer. This module pins one example per dialect we claim, plus the
guards that stop prose from being read as a call.
"""

import json

from skulk.worker.runner.llm_inference.tool_text_parser import (
    parse_tool_calls_from_text,
)


def _one(text: str) -> tuple[str, dict[str, object]]:
    """Parse text expected to carry exactly one call; return name and args."""
    calls = parse_tool_calls_from_text(text)
    assert calls is not None, f"no tool call parsed from {text!r}"
    assert len(calls) == 1, f"expected one call, got {len(calls)}"
    return calls[0].name, json.loads(calls[0].arguments)


class TestLlama:
    """Llama 3.1+ marks calls with <|python_tag|> and uses `parameters`."""

    def test_python_tag_call(self) -> None:
        text = (
            '<|python_tag|>{"name": "get_weather", '
            '"parameters": {"location": "Cedar Rapids, Iowa"}}<|eom_id|>'
        )
        name, args = _one(text)
        assert name == "get_weather"
        assert args == {"location": "Cedar Rapids, Iowa"}

    def test_terminator_is_not_swallowed_into_arguments(self) -> None:
        # The observed failure leaked <|eom_id|> and the next header into
        # content; the parser must stop at the message boundary.
        text = (
            '<|python_tag|>{"name": "f", "parameters": {"a": 1}}<|eom_id|>'
            "<|start_header_id|>assistant<|end_header_id|>"
        )
        _, args = _one(text)
        assert args == {"a": 1}

    def test_chained_calls_separated_by_semicolons(self) -> None:
        text = (
            '<|python_tag|>{"name": "a", "parameters": {}};'
            '{"name": "b", "parameters": {"x": 2}}<|eom_id|>'
        )
        calls = parse_tool_calls_from_text(text)
        assert calls is not None
        assert [call.name for call in calls] == ["a", "b"]

    def test_unmarked_call_object_is_accepted(self) -> None:
        name, args = _one('{"name": "get_weather", "parameters": {"location": "X"}}')
        assert name == "get_weather"
        assert args == {"location": "X"}


class TestMistral:
    def test_tool_calls_array(self) -> None:
        text = '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "Paris"}}]'
        name, args = _one(text)
        assert name == "get_weather"
        assert args == {"location": "Paris"}

    def test_multiple_calls_in_one_array(self) -> None:
        text = '[TOOL_CALLS] [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]'
        calls = parse_tool_calls_from_text(text)
        assert calls is not None
        assert [call.name for call in calls] == ["a", "b"]

    def test_trailing_prose_after_the_array_is_ignored(self) -> None:
        text = '[TOOL_CALLS] [{"name": "a", "arguments": {}}]\nI will check that.'
        name, _ = _one(text)
        assert name == "a"


class TestGlm:
    def test_arg_key_value_pairs(self) -> None:
        text = (
            "<tool_call>get_weather\n"
            "<arg_key>location</arg_key><arg_value>Cedar Rapids</arg_value>\n"
            "<arg_key>unit</arg_key><arg_value>celsius</arg_value>\n"
            "</tool_call>"
        )
        name, args = _one(text)
        assert name == "get_weather"
        assert args == {"location": "Cedar Rapids", "unit": "celsius"}


class TestExistingDialectsStillWork:
    """The new branches must not shadow the dialects that already worked."""

    def test_hermes_json_block(self) -> None:
        name, args = _one('<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>')
        assert name == "f"
        assert args == {"a": 1}

    def test_qwen3_xml_block(self) -> None:
        text = (
            "<tool_call><function=get_weather>"
            "<parameter=location>Cedar Rapids</parameter>"
            "</function></tool_call>"
        )
        name, args = _one(text)
        assert name == "get_weather"
        assert args == {"location": "Cedar Rapids"}


class TestFalsePositiveGuards:
    """Prose must never be read as a tool call."""

    def test_plain_prose_is_not_a_call(self) -> None:
        assert parse_tool_calls_from_text("The weather in Cedar Rapids is fine.") is None

    def test_json_answer_embedded_in_prose_is_not_a_call(self) -> None:
        text = 'Here is the record you asked for: {"name": "Ada", "parameters": {}}'
        assert parse_tool_calls_from_text(text) is None

    def test_json_object_without_a_name_is_not_a_call(self) -> None:
        assert parse_tool_calls_from_text('{"location": "Paris"}') is None

    def test_json_answer_with_a_name_but_no_arguments_is_not_a_call(self) -> None:
        # A model answering with a record that happens to have a name field.
        assert parse_tool_calls_from_text('{"name": "Ada", "born": 1815}') is None

    def test_empty_text(self) -> None:
        assert parse_tool_calls_from_text("") is None


class TestOfferedToolsOnly:
    """A parsed call must name a tool the caller actually offered."""

    WEATHER = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        }
    ]

    def test_a_llama_builtin_is_not_a_tool_call(self) -> None:
        # Llama answers some plain questions with its own `print` built-in.
        # Reporting that as a call hands the caller a name they cannot run.
        assert (
            parse_tool_calls_from_text(
                '<|python_tag|>{"name": "print", "parameters": {"value": "hi"}}',
                self.WEATHER,
            )
            is None
        )

    def test_an_offered_tool_still_parses(self) -> None:
        calls = parse_tool_calls_from_text(
            '<|python_tag|>{"name": "get_weather", "parameters": {"location": "x"}}',
            self.WEATHER,
        )
        assert calls is not None
        assert [call.name for call in calls] == ["get_weather"]

    def test_the_offered_call_survives_alongside_a_builtin(self) -> None:
        calls = parse_tool_calls_from_text(
            '<|python_tag|>{"name": "print", "parameters": {}};'
            '{"name": "get_weather", "parameters": {"location": "x"}}',
            self.WEATHER,
        )
        assert calls is not None
        assert [call.name for call in calls] == ["get_weather"]

    def test_an_absent_tools_list_is_not_a_statement_that_nothing_may_be_called(
        self,
    ) -> None:
        # This is a shared helper: the steward parses its own turns through the
        # same dialects without passing a tools list. Whether a request that
        # declared no tools may return a call is decided by the caller, which
        # is the only place that knows.
        calls = parse_tool_calls_from_text(
            '<|python_tag|>{"name": "print", "parameters": {"value": "hi"}}'
        )
        assert calls is not None
        assert [call.name for call in calls] == ["print"]

