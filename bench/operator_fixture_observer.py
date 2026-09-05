"""Metadata-only observation at the isolated gateway TCP and ASGI boundaries.

This is not an app-side packet trace. Real gateway TCP lifetime, application
body sizes offered to ASGI, and gateway completion are distinct measurements.
The sink receives only fixed vocabulary and numeric counters, never payloads.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, cast, final

from starlette.types import ASGIApp, Message, Receive, Scope, Send

type ObservationFlow = Literal[
    "cold-launch",
    "foreground-refresh",
    "settled-foreground",
    "background-resume",
    "reconnect",
    "chat",
    "speech",
]
type ObservationEvent = Mapping[str, str | int]
type ObservationSink = Callable[[ObservationEvent], None]
type _Peer = tuple[str, int]

_FLOWS: frozenset[str] = frozenset(
    (
        "cold-launch",
        "foreground-refresh",
        "settled-foreground",
        "background-resume",
        "reconnect",
        "chat",
        "speech",
    )
)
_OPERATIONS = {
    "/state": "state",
    "/v1/models": "models",
    "/store/registry": "registry",
    "/store/storage": "storage",
    "/store/downloads": "downloads",
    "/v1/chat/completions": "chat",
    "/v1/audio/speech": "speech",
    "/v1/audio/voices": "voices",
    "/v1/auth/pairing-sessions/challenge": "connection",
    "/v1/auth/pairing-sessions/exchange": "connection",
    "/v1/auth/token": "connection",
}


class FixtureObservationError(RuntimeError):
    """Fixed redacted failure; an invalid observation cannot become evidence."""

    def __init__(self) -> None:
        """Avoid including paths, payloads, peers, or exception contents."""
        super().__init__("invalid fixture observation")


@dataclass
class _Connection:
    observed: bool
    peer: _Peer | None = None
    requests: set[int] = field(default_factory=set)


@final
class FixtureObserver:
    """Translate owned socket/request lifecycles into the bounded recorder grammar.

    Call only on the fixture event loop. The injected sink must be synchronous,
    bounded and lossless; it must raise instead of dropping an event. No event
    history, completed request IDs, payload, or real identity is retained here.
    """

    def __init__(
        self,
        sink: ObservationSink,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize an in-memory adapter with a fixed-vocabulary output sink."""
        self._sink = sink
        self._clock = clock
        self._epoch = clock()
        self._invalid = False
        self._flow: ObservationFlow | None = None
        self._flow_started = 0
        self._last_time = 0
        self._events = 0
        self._bytes = 0
        self._next_connection = 0
        self._next_request = 0
        self._connections: dict[int, _Connection] = {}
        self._peers: dict[_Peer, int] = {}
        self._requests: dict[int, int] = {}

    def _check(self, condition: bool) -> None:
        if self._invalid or not condition:
            self._invalid = True
            raise FixtureObservationError

    def _emit(self, event: dict[str, str | int]) -> None:
        now = int((self._clock() - self._epoch) * 1000)
        self._check(self._last_time <= now <= 7200000 and self._events < 100000)
        event["at"] = now
        self._last_time = now
        self._events += 1
        try:
            self._sink(event)
        except Exception:
            self._invalid = True
            raise FixtureObservationError from None

    def begin(self, flow: ObservationFlow) -> None:
        """Begin a named flow only when no old observed or idle socket is open."""
        self._check(flow in _FLOWS and self._flow is None and not self._connections)
        self._emit({"type": "flow-start", "flow": flow})
        self._flow = flow
        self._flow_started = self._last_time

    def end(self) -> None:
        """End a nonzero-duration flow only after all sockets and requests close."""
        self._check(
            self._flow is not None and not self._connections and not self._requests
        )
        self._check(int((self._clock() - self._epoch) * 1000) > self._flow_started)
        self._emit({"type": "flow-end"})
        self._flow = None

    def accepted(self) -> int:
        """Assign a local integer at TCP accept, not at the first HTTP request."""
        self._check(len(self._connections) < 512 and self._next_connection < 100000)
        self._next_connection += 1
        identifier = self._next_connection
        observed = self._flow is not None
        if observed:
            self._emit({"type": "connection-open", "id": identifier})
        self._connections[identifier] = _Connection(observed)
        return identifier

    def bind_peer(self, identifier: int, peer: _Peer) -> None:
        """Associate the owned downstream socket with its ASGI peer, in memory only."""
        connection = self._connections.get(identifier)
        self._check(connection is not None and peer not in self._peers)
        assert connection is not None
        self._check(connection.peer is None)
        connection.peer = peer
        self._peers[peer] = identifier

    def closed(self, identifier: int) -> None:
        """Close a real TCP lifecycle, failing any still-active gateway requests."""
        connection = self._connections.get(identifier)
        self._check(connection is not None)
        assert connection is not None
        for request in tuple(connection.requests):
            self.finished(request, successful=False)
        if connection.observed:
            self._emit({"type": "connection-close", "id": identifier})
        if connection.peer is not None:
            del self._peers[connection.peer]
        del self._connections[identifier]

    def started(self, peer: _Peer, path: str) -> int | None:
        """Classify a path locally, exporting only a fixed operation category."""
        identifier = self._peers.get(peer)
        self._check(identifier is not None)
        assert identifier is not None
        connection = self._connections[identifier]
        if not connection.observed:
            return None
        self._check(len(self._requests) < 512 and self._next_request < 100000)
        self._next_request += 1
        request = self._next_request
        self._emit(
            {
                "type": "request-start",
                "id": request,
                "connection": identifier,
                "operation": _OPERATIONS.get(path, "other"),
            }
        )
        self._requests[request] = identifier
        connection.requests.add(request)
        return request

    def body_bytes(self, request: int | None, count: int, *, response: bool) -> None:
        """Offer only positive body lengths, never the body or its hash."""
        self._check(type(count) is int and count >= 0)
        self._check(request is None or 0 < request <= self._next_request)
        if request is None or request not in self._requests or count == 0:
            return
        self._check(self._bytes + count <= 10 * 1024**3)
        self._bytes += count
        self._emit(
            {
                "type": "response-chunk" if response else "request-bytes",
                "id": request,
                "bytes": count,
            }
        )

    def finished(self, request: int | None, *, successful: bool) -> None:
        """Finish once; late ASGI unwind cannot erase a transport-close failure."""
        self._check(request is None or 0 < request <= self._next_request)
        if request is None or request not in self._requests:
            return
        identifier = self._requests[request]
        self._emit(
            {
                "type": "request-end",
                "id": request,
                "outcome": "completed" if successful else "failed",
            }
        )
        self._connections[identifier].requests.remove(request)
        del self._requests[request]

    def idle(self) -> bool:
        """Return whether every owned TCP connection and request has closed.

        Raises if capture has been invalidated; polling this cannot turn a
        failed observer into a successful capture.
        """
        self._check(True)
        return not self._connections and not self._requests

    def invalidate(self) -> None:
        """Permanently reject capture after an external adapter or sink failure."""
        self._check(False)

    def wrap(self, app: ASGIApp) -> ASGIApp:
        """Observe body lengths and gateway completion without changing responses."""

        async def observed(scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] != "http":
                await app(scope, receive, send)
                return
            peer = cast(_Peer | None, scope.get("client"))
            self._check(peer is not None)
            assert peer is not None
            request = self.started(peer, cast(str, scope["path"]))
            status = 0
            complete = False
            disconnected = False

            async def observed_receive() -> Message:
                nonlocal disconnected
                message = await receive()
                if message["type"] == "http.request":
                    self.body_bytes(
                        request,
                        len(cast(bytes, message.get("body", b""))),
                        response=False,
                    )
                elif message["type"] == "http.disconnect":
                    disconnected = True
                return message

            async def observed_send(message: Message) -> None:
                nonlocal status, complete
                await send(message)
                if message["type"] == "http.response.start":
                    status = cast(int, message["status"])
                elif message["type"] == "http.response.body":
                    self.body_bytes(
                        request,
                        len(cast(bytes, message.get("body", b""))),
                        response=True,
                    )
                    complete = not message.get("more_body", False)

            succeeded = False
            try:
                await app(scope, observed_receive, observed_send)
                succeeded = complete and 200 <= status < 300 and not disconnected
            finally:
                self.finished(request, successful=succeeded)

        return observed
