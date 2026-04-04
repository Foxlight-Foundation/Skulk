"""Tests for Chat Completions request conversion edge cases."""

import pytest

from exo.api.adapters.chat_completions import chat_request_to_text_generation
from exo.api.types import (
    ChatCompletionMessage,
    ChatCompletionMessageImageUrl,
    ChatCompletionMessageText,
    ChatCompletionRequest,
)
from exo.shared.types.common import ModelId


@pytest.mark.anyio
async def test_single_image_object_builds_multimodal_message() -> None:
    request = ChatCompletionRequest(
        model=ModelId("mlx-community/gemma-4-e4b-it-8bit"),
        messages=[
            ChatCompletionMessage(
                role="user",
                content=ChatCompletionMessageImageUrl(
                    image_url={"url": "data:image/png;base64,AAAA"}
                ),
            )
        ],
    )

    params = await chat_request_to_text_generation(request)

    assert params.images == ["AAAA"]
    assert params.chat_template_messages == [
        {"role": "user", "content": [{"type": "image"}]}
    ]


@pytest.mark.anyio
async def test_multimodal_list_preserves_text_and_image_order() -> None:
    request = ChatCompletionRequest(
        model=ModelId("mlx-community/gemma-4-e4b-it-8bit"),
        messages=[
            ChatCompletionMessage(
                role="user",
                content=[
                    ChatCompletionMessageText(text="before"),
                    ChatCompletionMessageImageUrl(
                        image_url={"url": "data:image/png;base64,BBBB"}
                    ),
                    ChatCompletionMessageText(text="after"),
                ],
            )
        ],
    )

    params = await chat_request_to_text_generation(request)

    assert params.images == ["BBBB"]
    assert params.chat_template_messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image"},
                {"type": "text", "text": "after"},
            ],
        }
    ]
