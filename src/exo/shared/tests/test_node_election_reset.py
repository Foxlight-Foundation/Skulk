"""Regression coverage for election-driven event-router replacement ordering."""

# pyright: reportAny=false, reportPrivateUsage=false

from dataclasses import dataclass
from typing import Any, cast

import pytest

from exo.api.main import API
from exo.main import Node
from exo.routing.event_router import EventRouter
from exo.routing.router import Router
from exo.shared.election import Election, ElectionResult
from exo.shared.types.common import NodeId, SessionId
from exo.utils.channels import channel
from exo.worker.main import Worker


@dataclass
class _FakeTaskGroup:
    events: list[str]

    def start_soon(self, func: Any, *args: Any, name: object = None) -> None:
        owner = getattr(func, "__self__", None)
        owner_name = type(owner).__name__ if owner is not None else ""
        label = f"{owner_name}.{func.__name__}".strip(".")
        self.events.append(f"start:{label}")


class _FakeRouter:
    def sender(self, _topic: object) -> object:
        return object()

    def receiver(self, _topic: object) -> object:
        return object()


class _OldEventRouter:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def shutdown(self) -> None:
        self._events.append("old_event_router.shutdown")


class _OldWorker:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def shutdown(self) -> None:
        self._events.append("old_worker.shutdown")


class _FakeApi:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def reset(self, _result_clock: int, _event_receiver: object) -> None:
        self._events.append("api.reset")

    def unpause(self, _result_clock: int) -> None:
        self._events.append("api.unpause")


@pytest.mark.asyncio
async def test_election_restarts_event_router_after_receivers_are_rewired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order_events: list[str] = []

    class NewEventRouter:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            order_events.append("new_event_router.created")

        def sender(self) -> object:
            order_events.append("new_event_router.sender")
            return object()

        def receiver(self) -> object:
            order_events.append("new_event_router.receiver")
            return object()

        async def run(self) -> None:
            return

    class NewWorker:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            order_events.append("new_worker.created")

        async def run(self) -> None:
            return

    monkeypatch.setattr("exo.main.EventRouter", NewEventRouter)
    monkeypatch.setattr("exo.main.Worker", NewWorker)
    async def _fake_request_cluster_config(
        *_args: Any, **_kwargs: Any
    ) -> str | None:
        return None

    monkeypatch.setattr(Node, "_request_cluster_config", _fake_request_cluster_config)

    election_sender, election_receiver = channel[ElectionResult]()
    node = Node(
        router=cast(Router, cast(object, _FakeRouter())),
        event_router=cast(EventRouter, cast(object, _OldEventRouter(order_events))),
        download_coordinator=None,
        worker=cast(Worker, cast(object, _OldWorker(order_events))),
        election=cast(Election, object()),
        election_result_receiver=election_receiver,
        master=None,
        api=cast(API, cast(object, _FakeApi(order_events))),
        node_id=NodeId("self"),
        offline=False,
        exo_config=None,
        store_client=None,
        store_server=None,
    )
    node._tg = _FakeTaskGroup(order_events)  # pyright: ignore[reportAttributeAccessIssue]

    await election_sender.send(
        ElectionResult(
            session_id=SessionId(master_node_id=NodeId("other"), election_clock=7),
            won_clock=7,
            is_new_master=True,
        )
    )
    election_sender.close()

    await node._elect_loop()

    assert order_events.index("new_event_router.receiver") < order_events.index(
        "api.reset"
    )
    assert order_events.index("api.reset") < order_events.index(
        "start:NewEventRouter.run"
    )
    assert order_events.index("new_worker.created") < order_events.index(
        "start:NewEventRouter.run"
    )


@pytest.mark.asyncio
async def test_new_master_does_not_wait_on_unavailable_state_sync_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order_events: list[str] = []

    class NewEventRouter:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            order_events.append("new_event_router.created")

        def sender(self) -> object:
            return object()

        def receiver(self) -> object:
            return object()

        async def run(self) -> None:
            return

    class NewMaster:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            order_events.append("new_master.created")

        async def run(self) -> None:
            return

    class NewWorker:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            order_events.append("new_worker.created")

        async def run(self) -> None:
            return

    monkeypatch.setattr("exo.main.EventRouter", NewEventRouter)
    monkeypatch.setattr("exo.main.Master", NewMaster)
    monkeypatch.setattr("exo.main.Worker", NewWorker)

    async def _unexpected_request_cluster_config(
        *_args: Any, **_kwargs: Any
    ) -> str | None:
        raise AssertionError("new master should not request config before startup")

    monkeypatch.setattr(
        Node, "_request_cluster_config", _unexpected_request_cluster_config
    )

    election_sender, election_receiver = channel[ElectionResult]()
    node = Node(
        router=cast(Router, cast(object, _FakeRouter())),
        event_router=cast(EventRouter, cast(object, _OldEventRouter(order_events))),
        download_coordinator=None,
        worker=cast(Worker, cast(object, _OldWorker(order_events))),
        election=cast(Election, object()),
        election_result_receiver=election_receiver,
        master=None,
        api=cast(API, cast(object, _FakeApi(order_events))),
        node_id=NodeId("self"),
        offline=False,
        exo_config=None,
        store_client=None,
        store_server=None,
    )
    node._tg = _FakeTaskGroup(order_events)  # pyright: ignore[reportAttributeAccessIssue]

    await election_sender.send(
        ElectionResult(
            session_id=SessionId(master_node_id=NodeId("self"), election_clock=8),
            won_clock=8,
            is_new_master=True,
        )
    )
    election_sender.close()

    await node._elect_loop()

    assert "new_master.created" in order_events
