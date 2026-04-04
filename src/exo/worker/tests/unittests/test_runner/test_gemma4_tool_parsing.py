"""Tests for Gemma 4 tool call parsing.

Covers the ``<|"|>`` quoting format, bare key quoting, type preservation,
internal quotes, and backslash escaping — matching ollama's test cases."""

import pytest

from exo.worker.engines.mlx.utils_mlx import _parse_gemma4_tool_calls


class TestGemma4ToolCallParsing:
    def test_simple_string_arg(self):
        text = 'call:get_weather{location:<|"|>San Francisco<|"|>}'
        result = _parse_gemma4_tool_calls(text)
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["arguments"]["location"] == "San Francisco"

    def test_numeric_arg_unquoted(self):
        """Bare numeric values should preserve their type."""
        text = "call:set_temp{value:42}"
        result = _parse_gemma4_tool_calls(text)
        assert result[0]["arguments"]["value"] == 42

    def test_numeric_arg_quoted_is_string(self):
        """Quoted numeric values should be strings, not numbers."""
        text = 'call:set_temp{value:<|"|>42<|"|>}'
        result = _parse_gemma4_tool_calls(text)
        assert result[0]["arguments"]["value"] == "42"

    def test_boolean_unquoted(self):
        text = "call:toggle{enabled:true}"
        result = _parse_gemma4_tool_calls(text)
        assert result[0]["arguments"]["enabled"] is True

    def test_boolean_quoted_is_string(self):
        text = 'call:toggle{enabled:<|"|>true<|"|>}'
        result = _parse_gemma4_tool_calls(text)
        assert result[0]["arguments"]["enabled"] == "true"

    def test_null_unquoted(self):
        text = "call:clear{field:null}"
        result = _parse_gemma4_tool_calls(text)
        assert result[0]["arguments"]["field"] is None

    def test_internal_double_quotes(self):
        """Strings containing double quotes must be properly escaped."""
        text = 'call:run{cmd:<|"|>git commit -m "fix bug"<|"|>}'
        result = _parse_gemma4_tool_calls(text)
        assert result[0]["arguments"]["cmd"] == 'git commit -m "fix bug"'

    def test_windows_path_backslashes(self):
        """Backslashes in paths must be properly escaped."""
        text = 'call:read{path:<|"|>C:\\Temp\\file.txt<|"|>}'
        result = _parse_gemma4_tool_calls(text)
        assert result[0]["arguments"]["path"] == "C:\\Temp\\file.txt"

    def test_multiple_args(self):
        text = 'call:search{query:<|"|>hello world<|"|>,limit:10,exact:true}'
        result = _parse_gemma4_tool_calls(text)
        args = result[0]["arguments"]
        assert args["query"] == "hello world"
        assert args["limit"] == 10
        assert args["exact"] is True

    def test_multiple_tool_calls(self):
        text = (
            'call:foo{a:<|"|>x<|"|>}\n'
            'call:bar{b:42}'
        )
        result = _parse_gemma4_tool_calls(text)
        assert len(result) == 2
        assert result[0]["name"] == "foo"
        assert result[0]["arguments"]["a"] == "x"
        assert result[1]["name"] == "bar"
        assert result[1]["arguments"]["b"] == 42

    def test_no_tool_calls_raises(self):
        with pytest.raises(ValueError, match="No Gemma 4 tool calls found"):
            _parse_gemma4_tool_calls("just some text")

    def test_empty_args(self):
        text = "call:ping{}"
        result = _parse_gemma4_tool_calls(text)
        assert result[0]["name"] == "ping"
        assert result[0]["arguments"] == {}

    def test_string_with_newlines(self):
        """Newlines inside quoted strings should be preserved."""
        text = 'call:write{content:<|"|>line1\nline2<|"|>}'
        result = _parse_gemma4_tool_calls(text)
        assert result[0]["arguments"]["content"] == "line1\nline2"
