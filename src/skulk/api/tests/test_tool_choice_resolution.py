"""Coverage for applying ``tool_choice`` before dispatch.

Only the served engines forward ``tool_choice`` to a server that acts on it, so
without this resolution an in-process engine answers a ``"none"`` request with
a tool call. That was observed live: a Llama model on the MLX engine returned
`get_weather` on all four attempts of a `"none"` request that asked for the
tool by name.
"""

from __future__ import annotations

from typing import Any

from skulk.api.adapters.chat_completions import resolve_tool_choice

WEATHER: dict[str, Any] = {
    "type": "function",
    "function": {"name": "get_weather", "parameters": {"type": "object"}},
}
TIME: dict[str, Any] = {
    "type": "function",
    "function": {"name": "get_time", "parameters": {"type": "object"}},
}
BOTH = [WEATHER, TIME]


def names(tools: list[dict[str, Any]] | None) -> list[str]:
    return [] if tools is None else [tool["function"]["name"] for tool in tools]


class TestNone:
    def test_none_removes_the_tools_entirely(self) -> None:
        tools, choice = resolve_tool_choice(BOTH, "none")
        assert tools is None
        assert choice is None

    def test_none_without_tools_is_a_no_op(self) -> None:
        assert resolve_tool_choice(None, "none") == (None, "none")


class TestNamedFunction:
    def test_a_named_function_narrows_the_offered_tools(self) -> None:
        tools, choice = resolve_tool_choice(
            BOTH, {"type": "function", "function": {"name": "get_time"}}
        )
        assert names(tools) == ["get_time"]
        # The choice still travels, so a served engine enforces it server-side.
        assert choice == {"type": "function", "function": {"name": "get_time"}}

    def test_a_name_matching_nothing_is_left_for_the_engine_to_report(self) -> None:
        # Silently sending no tools would turn the caller's mistake into a
        # confusing prose answer instead of an error.
        tools, _ = resolve_tool_choice(
            BOTH, {"type": "function", "function": {"name": "nope"}}
        )
        assert names(tools) == ["get_weather", "get_time"]

    def test_a_malformed_choice_object_passes_through(self) -> None:
        tools, choice = resolve_tool_choice(BOTH, {"type": "function"})
        assert names(tools) == ["get_weather", "get_time"]
        assert choice == {"type": "function"}


class TestPassThrough:
    def test_auto_is_untouched(self) -> None:
        assert resolve_tool_choice(BOTH, "auto") == (BOTH, "auto")

    def test_required_is_untouched(self) -> None:
        # In-process there is no constrained decoding to force a call, so this
        # stays a best-effort instruction rather than being reinterpreted here.
        assert resolve_tool_choice(BOTH, "required") == (BOTH, "required")

    def test_an_absent_choice_is_untouched(self) -> None:
        assert resolve_tool_choice(BOTH, None) == (BOTH, None)
