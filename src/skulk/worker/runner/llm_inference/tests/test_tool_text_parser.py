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
