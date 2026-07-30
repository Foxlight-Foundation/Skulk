"""Fallback text tool-call parsing for the steward harness (MLX lane)."""

import json

from skulk.api.steward import parse_text_tool_calls, strip_tool_markup


def test_parses_qwen_xml_function_format() -> None:
    text = (
        "<tool_call>\n<function=get_cluster_state>\n</function>\n</tool_call>"
    )
    calls = parse_text_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].function.name == "get_cluster_state"
    assert json.loads(calls[0].function.arguments) == {}


def test_parses_xml_parameters_and_strips_markup() -> None:
    text = (
        "Investigating.\n<tool_call>\n<function=get_node_resources>\n"
        "<parameter=node_id>\nmac-den\n</parameter>\n</function>\n</tool_call>"
    )
    calls = parse_text_tool_calls(text)
    assert calls[0].function.name == "get_node_resources"
    assert json.loads(calls[0].function.arguments) == {"node_id": "mac-den"}
    assert strip_tool_markup(text) == "Investigating."


def test_parses_hermes_json_format() -> None:
    text = '<tool_call>{"name": "run_doctor", "arguments": {}}</tool_call>'
    calls = parse_text_tool_calls(text)
    assert calls[0].function.name == "run_doctor"


def test_bare_function_block_without_wrapper() -> None:
    calls = parse_text_tool_calls("<function=get_model_catalog>\n</function>")
    assert calls[0].function.name == "get_model_catalog"


def test_plain_text_yields_no_calls() -> None:
    assert parse_text_tool_calls("The cluster is healthy.") == []
