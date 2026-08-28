import json
import math
from dataclasses import dataclass
from typing import Any, Callable, cast

from skulk.api.types import ToolCallItem

UNMARKED_TOOL_DIALECT = "skulk:unmarked-tool-dialect"
"""Sentinel tool parser meaning "read the whole block with the text dialects".

A tokenizer carries a callable in its tool-parser slot for marker-delimited
families, and the runner strips the markers before calling it. Llama has no
opening marker to strip, so the runner must build a different parser rather
than call anything; this sentinel is how it tells the two cases apart.
"""


@dataclass
class ToolParser:
    start_parsing: str
    end_parsing: str
    _inner_parser: Callable[[str], list[ToolCallItem] | None]
    extra_start_parsing: tuple[str, ...] = ()
    """Further markers that also open a tool-call block.

    A family can open a call more than one way. Llama writes the bare call
    object most of the time but prefixes ``<|python_tag|>`` when it reaches for
    a tool by name, and a marker that does not open the block is emitted to the
    caller as content.
    """
    anchored: bool = False
    """Whether the primary marker opens a block only at the start of a message.

    A distinctive marker opens a block wherever it appears, because models
    routinely write a sentence before calling. The unmarked dialect's marker is
    ``{``, which also appears in prose and in JSON answers, so letting it open
    a block anywhere would turn any brace mid-answer into a call. The families
    using it write the call as the whole message, so anchoring loses nothing.
    """
    unparsed_is_text: bool = False
    """Whether a block that fails to parse is content rather than a failure.

    Marker-delimited dialects open on a token no ordinary answer emits, so a
    block that will not parse is genuinely broken output. Unmarked dialects open
    on ``{``, which a model asked for JSON also emits, so there the safe reading
    of an unparsable block is that the model simply answered in JSON and the
    text should be delivered as content.
    """

    @property
    def start_markers(self) -> tuple[str, ...]:
        """Every marker whose appearance opens a tool-call block."""

        return (self.start_parsing, *self.extra_start_parsing)

    def parse(
        self, text: str, tools: list[dict[str, Any]] | None
    ) -> list[ToolCallItem] | None:
        parsed = self._inner_parser(text)
        if parsed is None:
            return None
        if tools is not None:
            parsed = coerce_tool_calls_to_schema(parsed, tools)
        return parsed


def _json_type_matches(value: Any, expected_type: str) -> bool:  # pyright: ignore[reportAny]
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(
            value, float
        )
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return False


def _coerce_tool_arg_with_schema(value: Any, schema: dict[str, Any]) -> Any:  # pyright: ignore[reportAny]
    schema_type = schema.get("type")

    if isinstance(schema_type, list):
        for candidate in schema_type:  # pyright: ignore[reportUnknownVariableType]
            if not isinstance(candidate, str):
                continue
            if candidate == "null" and value is None:
                return None
            candidate_schema = {**schema, "type": candidate}
            coerced = _coerce_tool_arg_with_schema(value, candidate_schema)  # pyright: ignore[reportAny]
            if _json_type_matches(coerced, candidate):
                return coerced  # pyright: ignore[reportAny]
        return value  # pyright: ignore[reportAny]

    if not isinstance(schema_type, str):
        return value  # pyright: ignore[reportAny]

    if schema_type == "object":
        parsed = value  # pyright: ignore[reportAny]
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)  # pyright: ignore[reportAny]
            except Exception:
                return value  # pyright: ignore[reportAny]
        if not isinstance(parsed, dict):
            return value  # pyright: ignore[reportAny]
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return parsed  # pyright: ignore[reportUnknownVariableType]
        return {
            key: (
                _coerce_tool_arg_with_schema(prop_value, prop_schema)  # pyright: ignore[reportUnknownArgumentType]
                if isinstance(prop_schema, dict)
                else prop_value
            )
            for key, prop_value in parsed.items()  # pyright: ignore[reportUnknownVariableType]
            for prop_schema in [properties.get(key)]  # type: ignore
        }

    if schema_type == "array":
        parsed = value  # pyright: ignore[reportAny]
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)  # pyright: ignore[reportAny]
            except Exception:
                return value  # pyright: ignore[reportAny]
        if not isinstance(parsed, list):
            return value  # pyright: ignore[reportAny]
        item_schema = schema.get("items")
        if not isinstance(item_schema, dict):
            return parsed  # pyright: ignore[reportUnknownVariableType]
        return [_coerce_tool_arg_with_schema(item, item_schema) for item in parsed]  # type: ignore

    if schema_type == "integer":
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                return value
        return value

    if schema_type == "number":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            try:
                num = float(value.strip())
                if math.isfinite(num):
                    return num
            except ValueError:
                return value
        return value

    if schema_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
        return value

    return value  # pyright: ignore[reportAny]


def coerce_tool_calls_to_schema(
    tool_calls: list[ToolCallItem], tools: list[dict[str, Any]]
) -> list[ToolCallItem]:
    """Coerce each tool call's argument values to the requested tool's schema.

    A model may emit arguments with loose types (e.g. a number as a string, or a
    string value where the schema wants an integer). For each call whose name
    matches a tool in ``tools``, this re-types the JSON arguments against that
    tool's ``parameters`` schema and returns the call with the re-serialized
    arguments. ``ToolCallItem.arguments`` must be a JSON object string; a call
    whose name is unknown, whose arguments do not parse, or which is not a JSON
    object is passed through unchanged. Used by both the MLX token-level tool
    parsers and the llama.cpp string parser (``tool_text_parser``).
    """
    schema_by_name: dict[str, dict[str, Any]] = {}
    for tool in tools:
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")  # type: ignore
        parameters = function.get("parameters")  # type: ignore
        if isinstance(name, str) and isinstance(parameters, dict):
            schema_by_name[name] = parameters

    if not schema_by_name:
        return tool_calls

    coerced_calls: list[ToolCallItem] = []
    for tool_call in tool_calls:
        schema = schema_by_name.get(tool_call.name)
        if schema is None:
            coerced_calls.append(tool_call)
            continue

        try:
            parsed_args = json.loads(tool_call.arguments)  # pyright: ignore[reportAny]
        except Exception:
            coerced_calls.append(tool_call)
            continue

        if not isinstance(parsed_args, dict):
            coerced_calls.append(tool_call)
            continue

        coerced_args = _coerce_tool_arg_with_schema(parsed_args, schema)  # pyright: ignore[reportAny]
        if not isinstance(coerced_args, dict):
            coerced_calls.append(tool_call)
            continue

        coerced_calls.append(
            tool_call.model_copy(update={"arguments": json.dumps(coerced_args)})
        )
    return coerced_calls


def make_mlx_parser(
    tool_call_start: str,
    tool_call_end: str,
    tool_parser: Callable[[str], dict[str, Any] | list[dict[str, Any]]],
) -> ToolParser:
    def parse_tool_calls(text: str) -> list[ToolCallItem] | None:
        try:
            text = text.removeprefix(tool_call_start)
            text = text.removesuffix(tool_call_end)
            parsed = tool_parser(text)
            if isinstance(parsed, list):
                return [ToolCallItem.model_validate(_flatten(p)) for p in parsed]
            else:
                return [ToolCallItem.model_validate(_flatten(parsed))]

        except Exception:
            return None

    return ToolParser(
        start_parsing=tool_call_start,
        end_parsing=tool_call_end,
        _inner_parser=parse_tool_calls,
    )


# TODO / example code:
def _parse_json_calls(text: str) -> list[ToolCallItem] | None:
    try:
        text = text.removeprefix("<tool_call>")
        text = text.removesuffix("</tool_call>")
        top_level = {
            k: json.dumps(v) if isinstance(v, (dict, list)) else v
            for k, v in json.loads(text).items()  # pyright: ignore[reportAny]
        }
        return [ToolCallItem.model_validate(top_level)]
    except Exception:
        return None


def _flatten(p: dict[str, Any]) -> dict[str, str]:
    return {
        k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)  # pyright: ignore[reportAny]
        for k, v in p.items()  # pyright: ignore[reportAny]
    }


def make_json_parser() -> ToolParser:
    return ToolParser(
        start_parsing="<tool_call>",
        end_parsing="</tool_call>",
        _inner_parser=_parse_json_calls,
    )


def make_text_dialect_parser(tool_call_start: str, tool_call_end: str) -> ToolParser:
    """Build a parser that reads the whole block with the cross-family dialects.

    Unlike :func:`make_mlx_parser`, the markers are not stripped before parsing:
    for several families the opening marker is part of the call itself (Llama
    writes the bare call object, so its opening marker is ``{``), and the
    dialect detection in :func:`parse_tool_calls_from_text` keys off the markers
    that are present. A block that does not parse is treated as content.
    """

    # Imported at call time: tool_text_parser imports the schema coercion from
    # this module, so a module-level import here would be circular.
    from skulk.worker.runner.llm_inference.tool_text_parser import (
        parse_tool_calls_from_text,
    )

    return ToolParser(
        start_parsing=tool_call_start,
        end_parsing=tool_call_end,
        _inner_parser=lambda text: parse_tool_calls_from_text(text),
        extra_start_parsing=("<|python_tag|>",),
        anchored=True,
        unparsed_is_text=True,
    )


def infer_tool_parser(chat_template: str) -> ToolParser | None:
    """Attempt to auto-infer a tool parser from the chat template."""
    if "<tool_call>" in chat_template and "tool_call.name" in chat_template:
        return make_json_parser()
    return None


def declared_tool_calls(
    tool_calls: list[ToolCallItem], tools: list[dict[str, Any]] | None
) -> list[ToolCallItem]:
    """Keep only calls naming a tool the caller actually offered.

    Some families reach for a built-in the caller never declared: Llama answers
    a plain question with ``<|python_tag|>print("hello")``, which parses as a
    call to ``print``. Surfacing that as a tool call hands the caller a name
    they have no implementation for, so it is dropped here and the block is
    delivered as content instead.

    ``tools`` of ``None`` means the caller had no list to check against, not
    that nothing may be called: this is a shared helper, and the steward parses
    its own turns through the same dialects without passing one. Whether a
    request that declared no tools may return a call is decided by the caller,
    which is the only place that knows.
    """

    declared: set[str] = set()
    if tools is None:
        return tool_calls
    for tool in tools:
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = cast("object", function.get("name"))  # pyright: ignore[reportUnknownMemberType]
        if isinstance(name, str):
            declared.add(name)
    if not declared:
        # Tools were offered but none is usably named, which is the caller's
        # malformed input rather than a statement that nothing may be called.
        return tool_calls
    return [call for call in tool_calls if call.name in declared]
