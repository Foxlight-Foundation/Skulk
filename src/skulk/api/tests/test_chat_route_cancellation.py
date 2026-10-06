"""The non-streaming chat route must stay cancellable end to end.

Two properties keep an abandoned request from burning cluster capacity, and
each has been broken once:

- the route returns a `StreamingResponse`, because that is what makes Starlette
  cancel the body when the caller disconnects
- that cancellation reaches the chunk stream, which is what triggers the
  runner-side cancellation

A test that only proves Starlette's behaviour in a minimal app leaves both open:
a refactor of this route back to an awaited body, or a wrapper that swallows
`CancelledError` between the response and the stream, would pass it while the
leak returned. So this drives the real handler.
"""

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from fastapi.responses import StreamingResponse

from skulk.api.main import API
from skulk.api.types import ChatCompletionMessage, ChatCompletionRequest
from skulk.shared.types.common import CommandId, ModelId

MODEL_ID = ModelId("org/model")


class _Command:
    """Minimal stand-in for the dispatched generation command."""

    def __init__(self) -> None:
        self.command_id = CommandId("cmd-cancel")


def _prepare(monkeypatch: pytest.MonkeyPatch, stream: AsyncGenerator[Any, None]) -> API:
    """Build an API whose dispatch yields the given chunk stream."""

    async def _resolve(_self: API, model_id: ModelId) -> ModelId:
        return model_id

    async def _send(_self: API, _task_params: object) -> _Command:
        return _Command()

    def _tapped(
        _self: API,
        _command_id: CommandId,
        _model_id: ModelId,
        *,
        task_params: object = None,
    ) -> AsyncGenerator[Any, None]:
        return stream

    async def _to_task_params(
        payload: ChatCompletionRequest, *args: object, **kwargs: object
    ) -> object:
        class _Params:
            model = MODEL_ID
            images: list[object] = []

        return _Params()

    async def _card(_self: API, _model_id: ModelId) -> object:
        return object()

    monkeypatch.setattr(API, "_resolve_and_validate_text_model", _resolve)
    monkeypatch.setattr(API, "_get_running_model_card", _card)
    monkeypatch.setattr(API, "_send_text_generation_with_images", _send)
    monkeypatch.setattr(API, "_tapped_text_stream", _tapped)
    monkeypatch.setattr("skulk.api.main.chat_request_to_text_generation", _to_task_params)

    api = object.__new__(API)
    api._extensions = None  # pyright: ignore[reportPrivateUsage]
    return api


def _payload() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=MODEL_ID,
        messages=[ChatCompletionMessage(role="user", content="hello")],
        stream=False,
    )


@pytest.mark.anyio
async def test_non_streaming_route_returns_a_streaming_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The response type is the load-bearing part, so assert it directly.

    Returning an awaited body here would read as a harmless simplification and
    would silently remove disconnect cancellation.
    """

    async def chunks() -> AsyncGenerator[Any, None]:
        await asyncio.sleep(0)
        return
        yield  # pragma: no cover

    api = _prepare(monkeypatch, chunks())
    response = await api.chat_completions(_payload())

    assert isinstance(response, StreamingResponse)


@pytest.mark.anyio
async def test_cancelling_the_body_reaches_the_generation_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation must propagate from the response body into the stream.

    This is the link that turns a client disconnect into runner-side
    cancellation. A wrapper that swallowed CancelledError between the two
    would leave the runner generating for a caller that has gone.
    """
    reached_stream = asyncio.Event()
    started = asyncio.Event()

    async def chunks() -> AsyncGenerator[Any, None]:
        try:
            started.set()
            await asyncio.sleep(30)
            yield None  # pragma: no cover
        except asyncio.CancelledError:
            reached_stream.set()
            raise

    api = _prepare(monkeypatch, chunks())
    response = await api.chat_completions(_payload())
    assert isinstance(response, StreamingResponse)

    body = response.body_iterator

    async def drain() -> None:
        async for _ in body:
            pass

    task = asyncio.ensure_future(drain())
    await asyncio.wait_for(started.wait(), timeout=10)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    await asyncio.wait_for(reached_stream.wait(), timeout=10)
    assert reached_stream.is_set()
