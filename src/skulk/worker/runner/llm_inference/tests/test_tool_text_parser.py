"""Tests for recovering reasoning-model tool calls from llama.cpp text (#416)."""

import json

from skulk.worker.runner.llm_inference.tool_text_parser import (
    parse_tool_calls_from_text,
)

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            },
        },
    }
]


def _one(text, tools=None):
    calls = parse_tool_calls_from_text(text, tools)
    assert calls is not None and len(calls) == 1
    return calls[0].name, json.loads(calls[0].arguments)


def test_gpt_oss_harmony_commentary_tool_call() -> None:
    # The exact shape captured live from gpt-oss-20b GGUF on the llama.cpp engine.
    raw = (
        "<|channel|>analysis<|message|>We need to call get_weather for Paris."
        "<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather "
        '<|constrain|>json<|message|>{"city":"Paris"}'
    )
    name, args = _one(raw, _TOOLS)
    assert name == "get_weather" and args == {"city": "Paris"}


def test_qwen3_xml_tool_call() -> None:
    # The exact shape captured live from Ornith-1.0-35B GGUF (Qwen3 XML form),
    # with the reasoning block ahead of it.
    raw = (
        "I need to call get_weather.\n</think>\n\n<tool_call>\n"
        "<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    name, args = _one(raw, _TOOLS)
    assert name == "get_weather" and args == {"city": "Tokyo"}


def test_hermes_json_tool_call() -> None:
    raw = (
        "reasoning...</think>\n<tool_call>\n"
        '{"name": "get_weather", "arguments": {"city": "Berlin"}}\n</tool_call>'
    )
    name, args = _one(raw, _TOOLS)
    assert name == "get_weather" and args == {"city": "Berlin"}


def test_argument_types_coerced_to_schema() -> None:
    # Qwen3 XML parameter values are raw strings; schema coercion makes `days` int.
    raw = (
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n"
        "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"
    )
    name, args = _one(raw, _TOOLS)
    assert name == "get_weather" and args == {"city": "Oslo", "days": 3}


def test_multiple_tool_calls() -> None:
    raw = (
        "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n"
        '</tool_call> then <tool_call>{"name":"b","arguments":{"y":"2"}}</tool_call>'
    )
    calls = parse_tool_calls_from_text(raw)
    assert calls is not None
    assert [c.name for c in calls] == ["a", "b"]


def test_prose_answer_returns_none() -> None:
    assert parse_tool_calls_from_text("The weather in Paris is sunny today.") is None
    assert parse_tool_calls_from_text("") is None


def test_harmony_takes_precedence_when_both_markers_present() -> None:
    # A harmony tool call should be read as harmony even if stray text contains
    # an angle bracket; the to=functions. marker is the trigger.
    raw = "<|channel|>commentary to=functions.ping <|message|>{}"
    calls = parse_tool_calls_from_text(raw)
    assert calls is not None and calls[0].name == "ping"


def test_braces_inside_string_values_do_not_break_scan() -> None:
    # The bracket-scan fallback (triggered by trailing text after the JSON) must
    # not miscount a brace inside a quoted string value.
    raw = (
        '<|channel|>commentary to=functions.search <|message|>'
        '{"pattern": "a{2}b", "note": "}"} trailing junk after the call'
    )
    calls = parse_tool_calls_from_text(raw)
    assert calls is not None and calls[0].name == "search"
    assert json.loads(calls[0].arguments) == {"pattern": "a{2}b", "note": "}"}


def test_harmony_no_arg_call_keeps_empty_object() -> None:
    # An empty body is a genuine no-argument call -> {} is correct.
    calls = parse_tool_calls_from_text("commentary to=functions.now <|message|>")
    assert calls is not None and calls[0].name == "now"
    assert json.loads(calls[0].arguments) == {}


def test_harmony_unparseable_body_is_skipped_not_fabricated() -> None:
    # A non-empty body that does not parse is a truncated/garbled call; skip it
    # rather than emit a call with fabricated empty arguments.
    raw = 'commentary to=functions.search <|message|>{"city": "Tok'  # truncated
    assert parse_tool_calls_from_text(raw) is None


def test_hermes_json_with_function_literal_in_arg_value() -> None:
    # A Hermes JSON call whose argument value merely contains the literal
    # "<function=" must NOT be misclassified as Qwen3 XML (no real tag present).
    raw = (
        '<tool_call>{"name": "search", "arguments": '
        '{"q": "how to use <function=foo>"}}</tool_call>'
    )
    name, args = _one(raw)
    assert name == "search" and args == {"q": "how to use <function=foo>"}


def test_non_object_arguments_fall_back_to_empty_object() -> None:
    # Malformed non-object arguments (a list) must not surface downstream where a
    # JSON object is required; fall back to {}.
    raw = '<tool_call>{"name": "go", "arguments": [1, 2, 3]}</tool_call>'
    name, args = _one(raw)
    assert name == "go" and args == {}


def test_qwen3_xml_object_param_with_name_field_keeps_function_name() -> None:
    # A Qwen3 XML param value that is a JSON object containing a "name" field must
    # not be misread as the Hermes JSON form; the function name comes from
    # <function=...>, not the nested object.
    raw = (
        "<tool_call>\n<function=add_person>\n<parameter=person>\n"
        '{"name": "Alice"}\n</parameter>\n</function>\n</tool_call>'
    )
    name, args = _one(raw)
    assert name == "add_person"
    assert json.loads(args["person"]) == {"name": "Alice"}
