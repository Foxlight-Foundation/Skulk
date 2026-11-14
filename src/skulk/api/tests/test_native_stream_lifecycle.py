"""Real WebSocket/auth lifecycle with deliberately stalled invocation effects.

Signed native dispatch and codecs have their separate installed gate. These
cases isolate disconnect observation during discovery/startup and input stalls.
"""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

import pytest
from starlette.types import Message, Scope

from skulk.api.native_inference import create_native_inference_router
from skulk.api.tests.test_inference_access import application
from skulk.extensions.inference_callers import AuthenticatedInferenceCaller
from skulk.extensions.native_host_budget import NativePayloadBudget
from skulk.extensions.native_inference_calls import NativeInferenceCalls
from skulk.extensions.native_inference_streams import (
    NativeInferenceOutput,
    NativeInferenceStream,
)
from skulk.shared.types.common import NodeId


@pytest.mark.parametrize(
    "phase", ["discovery", "startup", "backpressure", "overflow", "co_ready"]
)
async def test_disconnect_or_full_ingress_cancels_and_drains_one_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """One receiver watches the socket before admission and during blocked writes."""
    entered, blocked, cleaned, release = (asyncio.Event() for _ in range(4))
    messages: asyncio.Queue[Message] = asyncio.Queue()
    sent: list[Message] = []
    budgets: list[NativePayloadBudget] = []
    dispatched = False

    class Port:
        """Injected stalled application effects, not authentication or a codec."""

        write_seconds = 1
        call_id = "a" * 64
        mode = "bidirectional"

        async def send_input(self, _wire: bytes) -> None:
            """Keep owner acknowledgement pending while the receiver keeps running."""
            blocked.set()
            await asyncio.Future[None]()

        async def next_output(self) -> NativeInferenceOutput | None:
            """Keep a quiet uncapped provider pending until its caller is cancelled."""
            await asyncio.Future[None]()
            return None

    async def invoke(
        self: NativeInferenceCalls,
        caller: AuthenticatedInferenceCaller,
        _plugin: str,
        _capability: str,
        _version: str,
        payload: bytes,
        consume: Callable[[NativeInferenceStream], Awaitable[None]],
        _seconds: float | None = None,
        *,
        caller_guard: Callable[[], None] | None = None,
    ) -> None:
        nonlocal dispatched
        assert caller.principal.kind == "client" and payload == b"{}"
        budgets.append(self.hosts.budget)
        assert caller_guard is not None
        entered.set()
        try:
            if phase in {"discovery", "co_ready"}:
                await release.wait()
            caller_guard()
            dispatched = True
            if phase == "startup":
                await release.wait()
            await consume(cast(NativeInferenceStream, cast(object, Port())))
        finally:
            cleaned.set()

    monkeypatch.setattr(NativeInferenceCalls, "invoke_stream", invoke)
    app, callers = application(tmp_path)
    _, token = callers.create(callers.inventory().revision, "Socket lifecycle", None)
    app.include_router(
        create_native_inference_router(
            NodeId("access-core"), connection_path=tmp_path / "connection.json"
        )
    )
    scope: Scope = {
        "type": "websocket",
        "asgi": {"version": "3.0"},
        "scheme": "wss",
        "path": "/v1/native-inference/fixture/chat/1.0.0/stream",
        "raw_path": b"/v1/native-inference/fixture/chat/1.0.0/stream",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"authorization", ("Bearer " + token).encode())],
        "client": ("192.0.2.1", 1),
        "server": ("localhost", 443),
        "subprotocols": [],
    }

    async def receive() -> Message:
        message = await messages.get()
        if phase == "co_ready" and message["type"] == "websocket.disconnect":
            release.set()
        return message

    async def send(message: Message) -> None:
        sent.append(message)
        if message["type"] == "websocket.send" and phase in {
            "backpressure",
            "overflow",
        }:
            messages.put_nowait({"type": "websocket.receive", "bytes": b"input"})

    async with app.router.lifespan_context(app):
        messages.put_nowait({"type": "websocket.connect"})
        messages.put_nowait({"type": "websocket.receive", "text": "{}"})
        task = asyncio.create_task(app(scope, receive, send))
        try:
            async with asyncio.timeout(2):
                await entered.wait()
                if phase in {"backpressure", "overflow"}:
                    await blocked.wait()
                if phase == "overflow":
                    for _ in range(9):
                        messages.put_nowait(
                            {"type": "websocket.receive", "bytes": b"bounded"}
                        )
                else:
                    messages.put_nowait({"type": "websocket.disconnect", "code": 1000})
                await task
                assert cleaned.is_set()
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert dispatched == (phase not in {"discovery", "co_ready"})
    assert budgets and all(budget.used == 0 for budget in budgets)
    if phase == "overflow":
        assert any(
            message["type"] == "websocket.close" and message["code"] == 1011
            for message in sent
        )
