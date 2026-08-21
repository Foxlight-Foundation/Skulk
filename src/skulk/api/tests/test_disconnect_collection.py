"""Disconnect handling for non-streaming chat responses.

A `StreamingResponse` gets disconnect handling for free: Starlette cancels the
body generator when the caller leaves, which reaches the chunk stream's
cancellation handler and stops the runner. Collecting the body to choose a real
status code gives that up unless it is replaced deliberately, and losing it is
expensive: an abandoned request would keep a runner generating for a caller
that is gone.
"""

import asyncio

import pytest
from starlette.requests import Request

from skulk.api.main import (
    CLIENT_DISCONNECTED_STATUS,
    collect_response_watching_disconnect,
)


def _request(*, disconnected: bool) -> Request:
    """Build a request whose receive channel reports the given liveness."""

    async def receive() -> dict[str, object]:
        if disconnected:
            return {"type": "http.disconnect"}
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(
        {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []},
        receive=receive,
    )


@pytest.mark.anyio
async def test_returns_the_collected_response_when_the_caller_stays() -> None:
    """The ordinary path must be unaffected by the disconnect watch."""

    async def collect() -> tuple[int, str]:
        return (200, '{"object":"chat.completion"}')

    status, body = await collect_response_watching_disconnect(
        _request(disconnected=False), collect()
    )

    assert status == 200
    assert body == '{"object":"chat.completion"}'


@pytest.mark.anyio
async def test_cancels_the_collection_when_the_caller_disconnects() -> None:
    """The point of the watch: work stops rather than running on unattended.

    Asserts the cancellation actually reached the collection, which is what
    propagates into the chunk stream and sends TaskCancelled to the runner.
    """
    cancelled = asyncio.Event()

    async def collect() -> tuple[int, str]:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return (200, "never reached")

    status, _ = await collect_response_watching_disconnect(
        _request(disconnected=True), collect()
    )

    assert status == CLIENT_DISCONNECTED_STATUS
    assert cancelled.is_set(), "the collection must be cancelled, not abandoned"
