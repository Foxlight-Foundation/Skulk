"""Metadata privacy, bounded lifecycle grammar, and real local socket tests."""

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import cast

import pytest
from starlette.types import Message, Receive, Scope, Send

from bench.operator_fixture_observer import (
    FixtureObservationError,
    FixtureObserver,
    ObservationEvent,
)
from bench.operator_fixture_proxy import observation_proxy


@dataclass
class _Clock:
    now: float = 0

    def __call__(self) -> float:
        return self.now


def test_fixed_categories_and_transport_close_own_request_failure() -> None:
    """Neither paths nor late success callbacks can rewrite a failed socket."""
    clock = _Clock()
    events: list[ObservationEvent] = []
    observer = FixtureObserver(events.append, clock=clock)
    observer.begin("cold-launch")
    connection = observer.accepted()
    observer.bind_peer(connection, ("127.0.0.1", 12345))
    request = observer.started(("127.0.0.1", 12345), "/private-secret-model-name")
    observer.body_bytes(request, 12, response=False)
    observer.body_bytes(request, 35, response=True)
    clock.now = 1
    observer.closed(connection)
    observer.finished(request, successful=True)
    observer.body_bytes(request, 987, response=True)
    observer.end()
    assert observer.idle()
    assert events == [
        {"type": "flow-start", "flow": "cold-launch", "at": 0},
        {"type": "connection-open", "id": 1, "at": 0},
        {
            "type": "request-start",
            "id": 1,
            "connection": 1,
            "operation": "other",
            "at": 0,
        },
        {"type": "request-bytes", "id": 1, "bytes": 12, "at": 0},
        {"type": "response-chunk", "id": 1, "bytes": 35, "at": 0},
        {"type": "request-end", "id": 1, "outcome": "failed", "at": 1000},
        {"type": "connection-close", "id": 1, "at": 1000},
        {"type": "flow-end", "at": 1000},
    ]
    assert "private-secret" not in json.dumps(events)


def test_pairing_outside_flow_is_not_a_partial_connection_sample() -> None:
    """Unobserved pairing is allowed, but an open socket prevents flow changes."""
    events: list[ObservationEvent] = []
    observer = FixtureObserver(events.append)
    connection = observer.accepted()
    observer.bind_peer(connection, ("127.0.0.1", 1))
    assert observer.started(("127.0.0.1", 1), "/state") is None
    assert events == []
    with pytest.raises(FixtureObservationError):
        observer.begin("chat")
    with pytest.raises(FixtureObservationError):
        observer.idle()


@pytest.mark.parametrize(
    "fault", ["clock", "socket-bound", "bytes", "unknown-request", "peer", "sink"]
)
def test_invalid_capture_stays_invalid(fault: str) -> None:
    """Clock, bounds, association and lossless-sink violations poison the run."""
    clock = _Clock()
    observer = FixtureObserver(lambda _: None, clock=clock)
    observer.begin("chat")
    connection = observer.accepted()
    observer.bind_peer(connection, ("127.0.0.1", 1))
    request = observer.started(("127.0.0.1", 1), "/v1/chat/completions")
    action: Callable[[], object]
    if fault == "clock":
        clock.now = -1
        action = partial(observer.body_bytes, request, 1, response=True)
    elif fault == "socket-bound":
        for _ in range(511):
            observer.accepted()
        action = observer.accepted
    elif fault == "bytes":
        action = partial(observer.body_bytes, request, 10 * 1024**3 + 1, response=True)
    elif fault == "unknown-request":
        action = partial(observer.finished, 99, successful=True)
    elif fault == "peer":
        action = partial(observer.started, ("127.0.0.1", 2), "/state")
    else:

        def failed_sink(_: ObservationEvent) -> None:
            raise RuntimeError("sensitive downstream exception")

        observer = FixtureObserver(failed_sink, clock=clock)
        action = partial(observer.begin, "chat")
    with pytest.raises(FixtureObservationError, match="^invalid fixture observation$"):
        action()
    with pytest.raises(FixtureObservationError):
        observer.idle()


async def test_asgi_preserves_messages_but_exports_only_lengths() -> None:
    """Header values, input, output and query strings never reach the sink."""
    events: list[ObservationEvent] = []
    observer = FixtureObserver(events.append)
    observer.begin("chat")
    connection = observer.accepted()
    observer.bind_peer(connection, ("127.0.0.1", 8))
    sent: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"private-input", "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        assert scope["query_string"] == b"secret=query"
        assert (await receive())["body"] == b"private-input"
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"secret", b"value")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b"private-output",
                "more_body": False,
            }
        )

    scope: Scope = {
        "type": "http",
        "path": "/v1/chat/completions",
        "client": ("127.0.0.1", 8),
        "query_string": b"secret=query",
    }
    await observer.wrap(app)(scope, receive, send)
    observer.closed(connection)
    await asyncio.sleep(0.002)
    observer.end()
    assert sent[-1]["body"] == b"private-output"
    encoded = json.dumps(events)
    assert all(
        secret not in encoded for secret in ("private", "secret", "query", "127.0.0.1")
    )
    assert any(event.get("outcome") == "completed" for event in events)
    assert sum(int(event.get("bytes", 0)) for event in events) == 27


async def test_proxy_observes_socket_lifetime_not_request_duration() -> None:
    """A real socket remains active after its request finishes until TCP closes."""
    events: list[ObservationEvent] = []
    observer = FixtureObserver(events.append)
    observer.begin("foreground-refresh")
    received = asyncio.Event()

    async def backend(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            assert await reader.readexactly(3) == b"TLS"
            writer.write(b"opaque")
            await writer.drain()
            received.set()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(backend, "127.0.0.1", 0)
    try:
        async with observation_proxy(
            observer, cast(tuple[str, int], server.sockets[0].getsockname())[1]
        ) as port:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"TLS")
            await writer.drain()
            assert await reader.readexactly(6) == b"opaque"
            await received.wait()
            assert not observer.idle()
            assert not any(event["type"] == "connection-close" for event in events)
            writer.close()
            await writer.wait_closed()
            async with asyncio.timeout(2):
                while not observer.idle():
                    await asyncio.sleep(0.01)
            observer.end()
    finally:
        server.close()
        await server.wait_closed()
    assert [event["type"] for event in events] == [
        "flow-start",
        "connection-open",
        "connection-close",
        "flow-end",
    ]
