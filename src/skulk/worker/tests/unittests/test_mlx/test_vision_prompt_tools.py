"""Tools offered with an image reach the vision prompt as they reach a text prompt (#938)."""

from typing import Any, cast

from mlx_lm.tokenizer_utils import TokenizerWrapper

from skulk.shared.types.mlx import Model
from skulk.worker.engines.mlx import vision as vision_module

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


class _RecordingTokenizer:
    """Renders tool names and image placeholders, recording what it was given."""

    def __init__(self, chat_template: str = "{{ messages }}") -> None:
        self.chat_template = chat_template
        self.calls: list[tuple[list[dict[str, Any]], dict[str, object]]] = []

    def apply_chat_template(
        self, messages: list[dict[str, Any]], **kwargs: object
    ) -> str:
        self.calls.append((messages, dict(kwargs)))
        tools = cast(list[dict[str, Any]] | None, kwargs.get("tools")) or []
        names = "".join(f"<tool>{tool['function']['name']}</tool>" for tool in tools)
        images = sum(
            1
            for message in messages
            if isinstance(message.get("content"), list)
            for part in cast(list[dict[str, Any]], message["content"])
            if part.get("type") == "image"
        )
        return f"<bos>{names}<user>" + "<|image|>" * images + "<model>"

    def decode(self, _token_ids: list[int]) -> str:
        return ""


def _tokenizer(recording: _RecordingTokenizer) -> TokenizerWrapper:
    return cast(TokenizerWrapper, cast(object, recording))


def _labelled_messages() -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "label": "Image 1"},
                {"type": "image", "label": "Image 2"},
                {"type": "text", "text": "Which city is in the second picture?"},
            ],
        }
    ]


def test_tools_reach_the_template_with_an_image() -> None:
    recording = _RecordingTokenizer()
    built = vision_module._build_vision_prompt_with_debug(  # pyright: ignore[reportPrivateUsage]
        _tokenizer(recording),
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "?"}]}],
        [2],
        "<|image|>",
        model_type="qwen3_vl",
        tools=TOOLS,
    )

    (_, kwargs) = recording.calls[0]
    assert kwargs["tools"] == TOOLS
    assert "<tool>lookup_weather</tool>" in built.prompt
    assert built.prompt.count("<|image|>") == 2


def test_gemma4_with_tools_renders_through_its_own_template_keeping_labels() -> None:
    """Gemma 4's reference renderer has no tool grammar; the model's template does."""
    recording = _RecordingTokenizer()
    messages = _labelled_messages()
    built = vision_module._build_vision_prompt_with_debug(  # pyright: ignore[reportPrivateUsage]
        _tokenizer(recording),
        messages,
        [1, 1],
        "<|image|>",
        model_type="gemma4",
        tools=TOOLS,
    )

    (rendered, kwargs) = recording.calls[0]
    assert kwargs["tools"] == TOOLS
    assert "<tool>lookup_weather</tool>" in built.prompt
    assert rendered[0]["content"] == [
        {"type": "text", "text": "Image 1:"},
        {"type": "image"},
        {"type": "text", "text": "Image 2:"},
        {"type": "image"},
        {"type": "text", "text": "Which city is in the second picture?"},
    ]
    # The caller's messages are not rewritten.
    assert messages == _labelled_messages()


class _ReferenceOnlyTokenizer:
    def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
        raise AssertionError("Gemma 4 without tools uses the reference renderer")

    def decode(self, _token_ids: list[int]) -> str:
        return ""


def test_gemma4_without_tools_keeps_the_reference_renderer() -> None:
    built = vision_module._build_vision_prompt_with_debug(  # pyright: ignore[reportPrivateUsage]
        cast(TokenizerWrapper, cast(object, _ReferenceOnlyTokenizer())),
        _labelled_messages(),
        [1, 1],
        "<|image|>",
        model_type="gemma4",
    )

    assert "Image 2:\n<|image|>" in built.prompt


def test_tool_call_history_is_normalized_for_the_template_only() -> None:
    """Templates such as Gemma 4's refuse string arguments in tool-call history."""
    recording = _RecordingTokenizer()
    history: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "?"}]},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "arguments": '{"city": "Lyon"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]
    vision_module._build_vision_prompt_with_debug(  # pyright: ignore[reportPrivateUsage]
        _tokenizer(recording),
        history,
        [1],
        "<|image|>",
        model_type="qwen3_vl",
        tools=TOOLS,
    )

    (rendered, _) = recording.calls[0]
    assert rendered[1]["tool_calls"][0]["function"]["arguments"] == {"city": "Lyon"}
    assert history[1]["tool_calls"][0]["function"]["arguments"] == '{"city": "Lyon"}'


def test_a_lossy_template_is_patched_when_tools_are_offered() -> None:
    lossy = (
        '{% if inner_type == "object | object" or inner_type|length > 50 %}'
        "any[]{% endif %}"
    )
    recording = _RecordingTokenizer(chat_template=lossy)
    vision_module._build_vision_prompt_with_debug(  # pyright: ignore[reportPrivateUsage]
        _tokenizer(recording),
        [{"role": "user", "content": [{"type": "image"}]}],
        [1],
        "<|image|>",
        model_type="qwen3_vl",
        tools=TOOLS,
    )

    (_, kwargs) = recording.calls[0]
    patched = kwargs["chat_template"]
    assert isinstance(patched, str) and "inner_type|length" not in patched


def test_prepare_vision_passes_the_requests_tools_to_the_processor() -> None:
    seen: list[object] = []

    class _Processor:
        def process(self, **kwargs: object) -> object:
            seen.append(kwargs["tools"])
            return object()

    vision_module.prepare_vision(
        images=["aW1hZ2U="],
        chat_template_messages=[{"role": "user", "content": [{"type": "image"}]}],
        vision_processor=cast(vision_module.VisionProcessor, cast(object, _Processor())),
        tokenizer=_tokenizer(_RecordingTokenizer()),
        model=cast(Model, object()),
        tools=TOOLS,
    )

    assert seen == [TOOLS]


def test_a_tool_carrying_the_image_placeholder_fails_the_request() -> None:
    """Expansion would bind the first image's features to the tool text."""
    import pytest

    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "describe",
                "description": "Answers questions about an <|image|> attachment",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    recording = _RecordingTokenizer()
    with pytest.raises(vision_module.VisionPreprocessingError, match="image placeholder"):
        vision_module._build_vision_prompt_with_debug(  # pyright: ignore[reportPrivateUsage]
            _tokenizer(recording),
            [{"role": "user", "content": [{"type": "image"}]}],
            [1],
            "<|image|>",
            model_type="qwen3_vl",
            tools=tools,
        )
    assert recording.calls == []
    # Without the token, the same tools render.
    vision_module._build_vision_prompt_with_debug(  # pyright: ignore[reportPrivateUsage]
        _tokenizer(recording),
        [{"role": "user", "content": [{"type": "image"}]}],
        [1],
        "<image_soft_token>",
        model_type="qwen3_vl",
        tools=tools,
    )
    assert len(recording.calls) == 1
