"""top_logprobs implies logprobs at the chat-completions boundary (#385).

A client may signal a logprobs request with `logprobs=true` or by setting
`top_logprobs` alone (OpenAI semantics). Normalizing it once at the adapter
boundary keeps every engine consistent rather than each runner re-deriving it.
"""

from skulk.api.adapters.chat_completions import chat_request_to_text_generation
from skulk.api.types.api import ChatCompletionMessage, ChatCompletionRequest
from skulk.shared.types.common import ModelId


def _request(**kwargs: object) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=ModelId("org/model"),
        messages=[ChatCompletionMessage(role="user", content="hi")],
        **kwargs,  # type: ignore[arg-type]
    )


async def test_top_logprobs_alone_sets_logprobs() -> None:
    params = await chat_request_to_text_generation(_request(top_logprobs=5))
    assert params.logprobs is True
    assert params.top_logprobs == 5


async def test_explicit_logprobs_preserved() -> None:
    params = await chat_request_to_text_generation(
        _request(logprobs=True, top_logprobs=3)
    )
    assert params.logprobs is True and params.top_logprobs == 3


async def test_no_logprobs_request_stays_off() -> None:
    params = await chat_request_to_text_generation(_request())
    assert params.logprobs is False
    assert params.top_logprobs is None
