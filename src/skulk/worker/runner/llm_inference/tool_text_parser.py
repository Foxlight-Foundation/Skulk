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
from typing import Any, cast

from skulk.api.types import ToolCallItem
from skulk.worker.runner.llm_inference.tool_parsers import (
    coerce_tool_calls_to_schema,
    declared_tool_calls,
)

# gpt-oss harmony tool call: the recipient `to=functions.NAME` and a `commentary`
# channel together, then a `<|message|>` body holding the JSON arguments (up to
# the next control marker or end of text). gpt-oss emits the recipient in EITHER
# order relative to the channel marker, both documented by the repo's
# FORMAT_A/FORMAT_B fixtures, so match both:
#   B (channel first):   <|channel|>commentary ... to=functions.NAME ... <|message|>
#   A (recipient first): to=functions.NAME<|channel|>commentary ... <|message|>
# Both tie the recipient to the commentary channel header (before `<|message|>`),
# so a bare `to=functions.` written as prose in the analysis (reasoning) channel
# body is NOT matched. `_HARMONY_MESSAGE_TAIL` is the shared args + terminator.
_HARMONY_MESSAGE_TAIL = (
    r"(.*?)(?=<\|call\|>|<\|end\|>|<\|return\|>|<\|start\|>|<\|channel\|>|$)"
)
_HARMONY_CALL_RES = (
    re.compile(
        r"<\|channel\|>commentary[^<]*?to=functions\.([A-Za-z0-9_.\-]+)"
        r".*?<\|message\|>" + _HARMONY_MESSAGE_TAIL,
        re.DOTALL,
    ),
    re.compile(
        r"to=functions\.([A-Za-z0-9_.\-]+)\s*<\|channel\|>commentary"
        r".*?<\|message\|>" + _HARMONY_MESSAGE_TAIL,
        re.DOTALL,
    ),
)
# A `<tool_call>...</tool_call>` block (JSON or Qwen3 XML inside), embedded in
# prose/reasoning. There may be several.
_TOOLCALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_PARAMETER_RE = re.compile(
    r"<parameter=([^>\s]+)\s*>\s*(.*?)\s*</parameter>", re.DOTALL
)
# Llama 3.1+ marks a tool call with <|python_tag|> and ends the turn with
# <|eom_id|> (end of MESSAGE, handing off to a tool) rather than <|eot_id|>
# (end of TURN, handing back to the user). The body is one or more JSON
# objects using "parameters" rather than "arguments"; several calls are
# separated by ";".
_PYTHON_TAG_RE = re.compile(
    r"<\|python_tag\|>(.*?)(?=<\|eom_id\|>|<\|eot_id\|>|<\|start_header_id\|>|$)",
    re.DOTALL,
)
# Mistral emits a JSON array behind a [TOOL_CALLS] marker.
_MISTRAL_RE = re.compile(r"\[TOOL_CALLS\]\s*(\[.*)", re.DOTALL)
# GLM puts the function name on its own line inside <tool_call>, then names
# arguments in <arg_key>/<arg_value> pairs rather than as JSON.
_GLM_ARG_RE = re.compile(
    r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>",
    re.DOTALL,
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


def _call_from_json_object(obj: object) -> ToolCallItem | None:
    """Build a call from the ``{"name": ..., "arguments"/"parameters": ...}`` shape.

    Shared by every JSON-carrying dialect (Hermes, Llama, Mistral), which
    differ only in the markup around this object. Llama uses ``parameters``
    where Hermes uses ``arguments``; both are accepted.

    ``ToolCallItem.arguments`` must decode to a JSON object downstream (schema
    coercion, the Claude adapter's dict input). A dict is re-serialized; the
    OpenAI shape where ``arguments`` is already a JSON-encoded string is kept
    as-is when it decodes to an object; any other shape (list, scalar, or a
    string that is not a JSON object) is malformed and falls back to ``{}``
    rather than being invented.
    """

    if not isinstance(obj, dict):
        return None
    payload = cast("dict[str, Any]", obj)
    if not isinstance(payload.get("name"), str):
        return None
    args = payload.get("arguments", payload.get("parameters", {}))
    if isinstance(args, dict):
        args_str = json.dumps(args)
    elif isinstance(args, str) and _first_json_object(args) is not None:
        args_str = args
    else:
        args_str = "{}"
    return ToolCallItem(name=str(payload["name"]), arguments=args_str)


def _python_tag_calls(text: str) -> list[ToolCallItem]:
    """Parse Llama 3.1+ ``<|python_tag|>`` calls."""

    calls: list[ToolCallItem] = []
    for match in _PYTHON_TAG_RE.finditer(text):
        for chunk in match.group(1).split(";"):
            call = _call_from_json_object(_first_json_object(chunk))
            if call is not None:
                calls.append(call)
    return calls


def _mistral_calls(text: str) -> list[ToolCallItem]:
    """Parse Mistral ``[TOOL_CALLS] [...]`` arrays."""

    match = _MISTRAL_RE.search(text)
    if match is None:
        return []
    decoder = json.JSONDecoder()
    try:
        array, _ = decoder.raw_decode(match.group(1).strip())
    except ValueError:
        return []
    if not isinstance(array, list):
        return []
    calls: list[ToolCallItem] = []
    for entry in array:
        call = _call_from_json_object(entry)
        if call is not None:
            calls.append(call)
    return calls


def _bare_json_call(text: str) -> list[ToolCallItem]:
    """Parse an unmarked call that is the entire message.

    Llama omits ``<|python_tag|>`` in some templates and simply emits the call
    object. Accepted only when the whole message is that one object, so a model
    answering a question *with* JSON is never mistaken for calling a tool.
    """

    stripped = text.strip()
    if not stripped.startswith("{"):
        return []
    decoder = json.JSONDecoder()
    try:
        obj, end = decoder.raw_decode(stripped)
    except ValueError:
        return []
    if stripped[end:].strip():
        return []
    if not isinstance(obj, dict):
        return []
    payload = cast("dict[str, Any]", obj)
    if "name" not in payload:
        return []
    if not isinstance(payload.get("arguments", payload.get("parameters")), (dict, str)):
        return []
    call = _call_from_json_object(payload)
    return [call] if call is not None else []


def _harmony_tool_calls(text: str) -> list[ToolCallItem]:
    calls: list[ToolCallItem] = []
    seen: set[tuple[int, str]] = set()
    for pattern in _HARMONY_CALL_RES:
        for match in pattern.finditer(text):
            # A given call matches only one ordering, but guard against a region
            # being claimed twice by keying on its start offset + name.
            key = (match.start(), match.group(1))
            if key in seen:
                continue
            seen.add(key)
            name = match.group(1)
            body = match.group(2)
            obj = _first_json_object(body)
            if obj is not None:
                calls.append(ToolCallItem(name=name, arguments=json.dumps(obj)))
            elif not body.strip():
                # A genuine no-argument call (empty body) is valid; only then is
                # {} correct. A non-empty body that did not parse is a
                # truncated/garbled call, so skip it rather than fabricate args.
                calls.append(ToolCallItem(name=name, arguments="{}"))
    return calls


def _toolcall_block_calls(text: str) -> list[ToolCallItem]:
    calls: list[ToolCallItem] = []
    for block in _TOOLCALL_BLOCK_RE.finditer(text):
        inner = block.group(1).strip()
        # Disambiguate by a real Qwen3 XML function tag FIRST. Matching the full
        # <function=NAME>...</function> (not just the substring "<function=")
        # avoids two failure modes: a JSON-scan first would misread an
        # object-valued XML parameter containing a "name" field as the Hermes
        # form, and a substring check would misclassify a Hermes JSON call whose
        # argument value merely contains the literal "<function=" as XML.
        xml_functions = list(_FUNCTION_RE.finditer(inner))
        if xml_functions:
            for function in xml_functions:
                name = function.group(1)
                params = {
                    key: value.strip()
                    for key, value in _PARAMETER_RE.findall(function.group(2))
                }
                calls.append(ToolCallItem(name=name, arguments=json.dumps(params)))
            continue
        # Hermes / older Qwen JSON form: {"name": ..., "arguments": {...}}.
        # GLM names arguments in <arg_key>/<arg_value> pairs with the function
        # name on the first line, so there is no JSON object to find. Checked
        # before the JSON scan because a value may itself contain JSON.
        arg_pairs = _GLM_ARG_RE.findall(inner)
        if arg_pairs:
            name = inner.split("<arg_key>", 1)[0].strip().splitlines()
            if name and name[-1].strip():
                params = {key: value for key, value in arg_pairs}
                calls.append(
                    ToolCallItem(
                        name=name[-1].strip(), arguments=json.dumps(params)
                    )
                )
                continue
        call = _call_from_json_object(_first_json_object(inner))
        if call is not None:
            calls.append(call)
    return calls


def parse_tool_calls_from_text(
    text: str, tools: list[dict[str, Any]] | None = None
) -> list[ToolCallItem] | None:
    """Recover tool calls a reasoning model emitted as text (llama.cpp engine).

    Detects the dialect from the markers present and parses the calls, then
    coerces argument types to the tool schema. Recognized dialects:

    - harmony ``to=functions.`` channels (gpt-oss)
    - ``<tool_call>`` blocks carrying Hermes JSON, Qwen3 XML, or GLM
      ``<arg_key>``/``<arg_value>`` pairs
    - Llama ``<|python_tag|>`` calls, which use ``parameters`` rather than
      ``arguments`` and may chain several with ``;``
    - Mistral ``[TOOL_CALLS]`` arrays
    - an unmarked call object, accepted only when it is the entire message

    When ``tools`` is given, calls naming a tool the caller did not offer are
    dropped, because a model reaching for one of its own built-ins has not
    called anything the caller can run.

    Returns ``None`` when no tool call is present (the model answered in prose),
    so the caller can fall back to emitting the content.
    """
    if not text:
        return None
    calls: list[ToolCallItem] = []
    if "to=functions." in text:
        calls = _harmony_tool_calls(text)
    if not calls and "<tool_call>" in text:
        calls = _toolcall_block_calls(text)
    if not calls and "<|python_tag|>" in text:
        calls = _python_tag_calls(text)
    if not calls and "[TOOL_CALLS]" in text:
        calls = _mistral_calls(text)
    if not calls:
        # Last resort, and deliberately strict: only when the whole message is
        # one call object. Unmarked dialects are indistinguishable from a model
        # answering in JSON, so anything looser invents tool calls from prose.
        calls = _bare_json_call(text)
    if not calls:
        return None
    if tools is not None:
        # A model may reach for one of its own built-ins: Llama answers some
        # plain questions with a call to `print`, and gpt-oss has `python` and
        # `browser`. Those name nothing the caller can run, so a block left with
        # no offered tool reads as prose and the caller gets the text instead.
        calls = declared_tool_calls(calls, tools)
        if not calls:
            return None
        calls = coerce_tool_calls_to_schema(calls, tools)
    return calls
