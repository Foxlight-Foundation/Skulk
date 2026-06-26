# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""MLX-free recovery of a reasoning model's tool call from llama.cpp output.

``llama_cpp``'s ``create_chat_completion(tools=...)`` only populates structured
``tool_calls`` for models whose native tool-call format its bundled chat handlers
recognize. A reasoning model emits its tool call in its own format that
llama-cpp-python leaves unparsed, so the call falls through into the message
``content`` as raw text instead of a structured ``tool_calls`` (#416). The three
formats seen on the llama.cpp engine:

- **gpt-oss harmony**: a ``commentary`` channel whose header carries
  ``to=functions.NAME`` and whose ``<|message|>`` body is the JSON arguments,
  e.g. ``...<|channel|>commentary to=functions.get_weather <|constrain|>json``
  ``<|message|>{"city":"Paris"}``.
- **Qwen3 XML**: ``<tool_call><function=NAME><parameter=KEY>VALUE</parameter>``
  ``...</function></tool_call>``.
- **Hermes / older Qwen JSON**: ``<tool_call>{"name":..,"arguments":{..}}``
  ``</tool_call>``.

This module reparses those from the string so the runner can emit a proper
``ToolCallChunk``, mirroring what the MLX engine does at the token level. It is
pure-Python (no MLX) because it runs on non-Mac GPU nodes (e.g. AMD).
"""

from __future__ import annotations

import json
import re
from typing import Any

from skulk.api.types import ToolCallItem
from skulk.worker.runner.llm_inference.tool_parsers import coerce_tool_calls_to_schema

# gpt-oss harmony: `to=functions.NAME` in a channel header, then the `<|message|>`
# body holds the JSON arguments, up to the next control marker (or end of text).
_HARMONY_CALL_RE = re.compile(
    r"to=functions\.([A-Za-z0-9_.\-]+).*?<\|message\|>(.*?)"
    r"(?=<\|call\|>|<\|end\|>|<\|return\|>|<\|start\|>|<\|channel\|>|$)",
    re.DOTALL,
)
# A `<tool_call>...</tool_call>` block (JSON or Qwen3 XML inside), embedded in
# prose/reasoning. There may be several.
_TOOLCALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_PARAMETER_RE = re.compile(
    r"<parameter=([^>\s]+)\s*>\s*(.*?)\s*</parameter>", re.DOTALL
)


def _first_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first balanced ``{...}`` JSON object in ``text``, or None."""
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001 - fall through to a bracket scan
        pass
    start = stripped.find("{")
    if start == -1:
        return None
    # Brace scan to find the end of the first object. Track string state so a
    # brace inside a string value (e.g. {"pattern": "{a}"}) does not throw off
    # the depth count.
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(stripped[start : index + 1])
                    return obj if isinstance(obj, dict) else None
                except Exception:  # noqa: BLE001 - malformed JSON, give up
                    return None
    return None


def _harmony_tool_calls(text: str) -> list[ToolCallItem]:
    calls: list[ToolCallItem] = []
    for match in _HARMONY_CALL_RE.finditer(text):
        name = match.group(1)
        body = match.group(2)
        obj = _first_json_object(body)
        if obj is not None:
            calls.append(ToolCallItem(name=name, arguments=json.dumps(obj)))
        elif not body.strip():
            # A genuine no-argument call (empty body) is valid; only then is {}
            # correct. A non-empty body that did not parse is a truncated/garbled
            # call, so skip it rather than fabricate empty arguments.
            calls.append(ToolCallItem(name=name, arguments="{}"))
    return calls


def _toolcall_block_calls(text: str) -> list[ToolCallItem]:
    calls: list[ToolCallItem] = []
    for block in _TOOLCALL_BLOCK_RE.finditer(text):
        inner = block.group(1).strip()
        # Disambiguate by the unambiguous Qwen3 XML marker FIRST. A JSON-scan
        # first would misread an object-valued XML parameter that happens to
        # contain a "name" field (e.g. <parameter=person>{"name":"Alice"}) as the
        # Hermes JSON form and take "Alice" as the function name.
        if "<function=" in inner:
            for function in _FUNCTION_RE.finditer(inner):
                name = function.group(1)
                params = {
                    key: value.strip()
                    for key, value in _PARAMETER_RE.findall(function.group(2))
                }
                calls.append(ToolCallItem(name=name, arguments=json.dumps(params)))
            continue
        # Hermes / older Qwen JSON form: {"name": ..., "arguments": {...}}.
        obj = _first_json_object(inner)
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            args = obj.get("arguments", obj.get("parameters", {}))
            args_str = (
                json.dumps(args) if isinstance(args, (dict, list)) else str(args)
            )
            calls.append(ToolCallItem(name=obj["name"], arguments=args_str))
    return calls


def parse_tool_calls_from_text(
    text: str, tools: list[dict[str, Any]] | None = None
) -> list[ToolCallItem] | None:
    """Recover tool calls a reasoning model emitted as text (llama.cpp engine).

    Detects the format from the markers present (a harmony ``to=functions.``
    channel, or a ``<tool_call>`` block in JSON or Qwen3 XML), parses the calls,
    and coerces argument types to the tool schema. Returns ``None`` when no tool
    call is present (the model answered in prose), so the caller can fall back to
    emitting the content.
    """
    if not text:
        return None
    calls: list[ToolCallItem] = []
    if "to=functions." in text:
        calls = _harmony_tool_calls(text)
    if not calls and "<tool_call>" in text:
        calls = _toolcall_block_calls(text)
    if not calls:
        return None
    if tools is not None:
        calls = coerce_tool_calls_to_schema(calls, tools)
    return calls
