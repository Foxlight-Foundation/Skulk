"""HTTP cleanup regressions; dispatch results are injected, authentication is real.

Actual signed app dispatch has its separate full TCP installed qualification.
These cases force cancellation in the narrow HTTP result/delivery ownership gap.
"""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request
from starlette.types import Message, Receive, Scope, Send

from skulk.api.native_calls import create_native_calls_router
from skulk.extensions.caller_identity import AuthenticatedOwner
from skulk.extensions.native_calls import NativeCalls
from skulk.extensions.native_host_budget import NativePayloadBudget
from skulk.extensions.native_host_protocol import Received, ResponseFrame
from skulk.extensions.runtime_attachment import ServiceConnection
from skulk.extensions.runtime_files import write_private
from skulk.shared.types.common import NodeId


@dataclass(frozen=True)
class Delivery:
    """Actual HTTP router and its injected owned result budgets."""

    app: FastAPI
    budgets: list[NativePayloadBudget]


@pytest.fixture
async def delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Delivery]:
    """Inject only invocation output; keep production owner HTTP authentication."""
    budgets: list[NativePayloadBudget] = []
    connection = ServiceConnection(manager_root=str(tmp_path), profile_id="a" * 32)
    write_private(tmp_path / "connection.json", connection.model_dump_json().encode())

    async def invoke(
        self: NativeCalls,
        owner: AuthenticatedOwner,
        _plugin: str,
        _capability: str,
        _version: str,
        body: bytes,
        _seconds: float,
    ) -> Received:
        assert owner.identifier == "direct-owner" and owner.issuer_node_id == NodeId(
            "delivery-node"
        )
        assert body == b"{}"
        budgets.append(self.budget)
        return Received(
            ResponseFrame(
                request_id=1, outcome="complete", parameters={}, payload_size=2
            ),
            b"{}",
            self.budget.reserve(2),
        )

    monkeypatch.setattr(NativeCalls, "invoke", invoke)
    app = FastAPI()
    app.include_router(
        create_native_calls_router(
            NodeId("delivery-node"), connection_path=tmp_path / "connection.json"
        )
    )
    async with app.router.lifespan_context(app):
        yield Delivery(app, budgets)
    assert budgets and all(budget.used == 0 for budget in budgets)


def client_for(app: FastAPI) -> httpx.AsyncClient:
    """Use an actual loopback ASGI peer and the required matching dashboard origin."""
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
        base_url="http://localhost",
        headers={"origin": "http://localhost", "x-skulk-dashboard": "pairing-v1"},
    )


async def test_disconnect_poll_that_consumes_cancellation_cannot_hold_the_response(
    delivery: Delivery, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A terminal response must not wait forever on the disconnect poll's scope."""
    entered, consumed = asyncio.Event(), asyncio.Event()

    async def disconnect(_self: Request) -> bool:
        entered.set()
        try:
            await asyncio.Future[None]()
        except asyncio.CancelledError:
            consumed.set()
        return False

    monkeypatch.setattr(Request, "is_disconnected", disconnect)
    async with client_for(delivery.app) as client:
        async with asyncio.timeout(1):
            response = await client.post(
                "/v1/native-capabilities/managed.fixture/read/1.0.0/call",
                content=b"{}",
                headers={"content-type": "application/json"},
            )
        assert response.status_code == 200 and response.content == b"{}"
        assert entered.is_set() and consumed.is_set()


async def test_cancellation_during_watcher_cleanup_releases_the_undelivered_result(
    delivery: Delivery, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed native body stays owned until cleanup can transfer it safely."""
    entered, release = asyncio.Event(), asyncio.Event()

    async def drain(call: asyncio.Task[Received], watch: asyncio.Task[None]) -> None:
        await asyncio.gather(call, watch, return_exceptions=True)
        entered.set()
        await release.wait()

    monkeypatch.setattr("skulk.api.native_calls._drain", drain)
    async with client_for(delivery.app) as client:
        request = asyncio.create_task(
            client.post(
                "/v1/native-capabilities/managed.fixture/read/1.0.0/call",
                content=b"{}",
                headers={"content-type": "application/json"},
            )
        )
        async with asyncio.timeout(1):
            await entered.wait()
            request.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await request
        assert all(budget.used == 0 for budget in delivery.budgets)


async def test_cancelled_asgi_send_releases_the_delivered_response_lease(
    delivery: Delivery,
) -> None:
    """The response owner releases its native bytes when physical HTTP send fails."""

    async def interrupted(scope: Scope, receive: Receive, _send: Send) -> None:
        async def cancel_send(_message: Message) -> None:
            raise asyncio.CancelledError

        await delivery.app(scope, receive, cancel_send)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=interrupted, client=("127.0.0.1", 1)),
        base_url="http://localhost",
        headers={"origin": "http://localhost", "x-skulk-dashboard": "pairing-v1"},
    ) as client:
        with pytest.raises(asyncio.CancelledError):
            await client.post(
                "/v1/native-capabilities/managed.fixture/read/1.0.0/call",
                content=b"{}",
                headers={"content-type": "application/json"},
            )
    assert all(budget.used == 0 for budget in delivery.budgets)
