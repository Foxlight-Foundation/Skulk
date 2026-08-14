"""Fallback text tool-call parsing for the steward harness (MLX lane)."""

import json

from skulk.api.steward import parse_text_tool_calls, strip_tool_markup
from skulk.worker.runner.llm_inference.tool_text_parser import (
    parse_tool_calls_from_text,
)

# Verbatim shape of the block Qwen3.6's chat template emits and instructs the
# model to produce: `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n
# </parameter>\n</function>\n</tool_call>`, with each value on its own lines
# and free to span several of them. The steward brain card
# (unsloth/Qwen3.6-35B-A3B-GGUF and its MLX sibling) declares
# tool_call_format = "generic" on the strength of these two parsers handling
# exactly this, so the shape is pinned here rather than paraphrased.
_QWEN36_MULTILINE_CALL = (
    "I will check the node first.\n"
    "<tool_call>\n"
    "<function=get_node_resources>\n"
    "<parameter=node_id>\n"
    "den-node-1\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


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


def test_qwen36_template_shape_parses_in_the_harness_recovery_parser() -> None:
    calls = parse_text_tool_calls(_QWEN36_MULTILINE_CALL)
    assert len(calls) == 1
    assert calls[0].function.name == "get_node_resources"
    assert json.loads(calls[0].function.arguments) == {"node_id": "den-node-1"}
    assert strip_tool_markup(_QWEN36_MULTILINE_CALL) == "I will check the node first."


def test_qwen36_template_shape_parses_in_the_shared_engine_parser() -> None:
    """The engine-side generic parser must agree with the harness fallback.

    Both are in play for a Qwen3.6 steward: the served and llama_cpp engines
    reparse the model's text with this one, and the MLX vision loader wires
    the same function through its marker mechanism, while the harness parser
    is the last net if an engine passes markup through as content. A steward
    whose tool calls parse on one lane and not the other is the failure this
    guards.
    """
    items = parse_tool_calls_from_text(_QWEN36_MULTILINE_CALL)
    assert items is not None
    assert len(items) == 1
    assert items[0].name == "get_node_resources"
    assert json.loads(items[0].arguments) == {"node_id": "den-node-1"}


def test_qwen36_multiline_parameter_value_survives_both_parsers() -> None:
    """The template explicitly allows a value spanning several lines."""
    text = (
        "<tool_call>\n"
        "<function=search_docs>\n"
        "<parameter=query>\n"
        "zenoh transport\nmodel store staging\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    expected = {"query": "zenoh transport\nmodel store staging"}
    assert json.loads(parse_text_tool_calls(text)[0].function.arguments) == expected
    items = parse_tool_calls_from_text(text)
    assert items is not None
    assert json.loads(items[0].arguments) == expected
