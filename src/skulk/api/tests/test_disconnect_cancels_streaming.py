"""The disconnect property the non-streaming chat path depends on.

`collect_chat_response` produces one complete JSON object and yields it only
after generation finishes, yet it is returned as a `StreamingResponse`. That
looks redundant and is not: Starlette watches for `http.disconnect` while the
body task runs, so a caller that goes away cancels the generator, the
`CancelledError` propagates into the chunk stream, and the API sends
`TaskCancelled` so the runner stops working for nobody.

That property is load-bearing and framework-owned, and it has already been
broken once by replacing the streaming response with an awaited body plus a
`Request.is_disconnected()` watcher, which does not work under the
`BaseHTTPMiddleware` this app always installs. The test that missed it drove a
bare `Request` with an immediate-return receive stub.

So this exercises the real shape: a middleware-wrapped app, driven at the ASGI
level, with a body generator that yields nothing until long after the
disconnect, exactly like the production one.
"""

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Awaitable, Callable, MutableMapping
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from starlette.requests import Request


def _app_with_middleware(cancelled: asyncio.Event, started: asyncio.Event) -> FastAPI:
    """Build an app whose middleware stack matches the API's."""

    app = FastAPI()

    async def passthrough(
        request: Request,
        call_next: Callable[[Request], Awaitable[StreamingResponse]],
    ) -> StreamingResponse:
        return await call_next(request)

    # Registering any http middleware installs BaseHTTPMiddleware, which is
    # what wraps `receive` and made the previous disconnect watcher useless.
    app.middleware("http")(passthrough)

    async def body() -> AsyncGenerator[str, None]:
        # Nothing is yielded until the work finishes, like collect_chat_response.
        try:
            started.set()
            await asyncio.sleep(30)
            yield '{"object":"chat.completion"}'
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def chat() -> StreamingResponse:
        return StreamingResponse(body(), media_type="application/json")

    app.post("/v1/chat/completions")(chat)

    return app


@pytest.mark.anyio
async def test_client_disconnect_cancels_a_streaming_body_through_middleware() -> None:
    """A caller that goes away must cancel the generation, not be waited on.

    Asserts cancellation reached the generator rather than merely that the
    response ended, because the capacity leak this guards against is precisely
    work continuing after the caller is gone.
    """
    cancelled = asyncio.Event()
    started = asyncio.Event()
    app = _app_with_middleware(cancelled, started)

    disconnect = asyncio.Event()
    messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        if not disconnect.is_set():
            disconnect.set()
            return {"type": "http.request", "body": b"{}", "more_body": False}
        # Every later poll reports the caller has gone.
        return {"type": "http.disconnect"}

    async def send(message: MutableMapping[str, Any]) -> None:
        messages.append(dict(message))

    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.1"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1234),
        "server": ("127.0.0.1", 52415),
    }

    call = asyncio.ensure_future(app(scope, receive, send))
    await asyncio.wait_for(started.wait(), timeout=10)
    await asyncio.wait_for(cancelled.wait(), timeout=10)
    call.cancel()
    with contextlib.suppress(asyncio.CancelledError, RuntimeError):
        await call

    assert cancelled.is_set(), (
        "a disconnected caller must cancel the body generator; without this the "
        "runner keeps generating for a client that has gone"
    )
