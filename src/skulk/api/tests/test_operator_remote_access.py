"""Tests for the relay ingress supervisor's start, stop, and restart decisions."""

from __future__ import annotations

from collections.abc import Callable
from typing import final

import anyio

from skulk.api.operator_remote_access import OperatorRemoteAccessSupervisor
from skulk.operator.relay import (
    OperatorRelayConfiguration,
    OperatorRelayRouteRejectedError,
)


def _route(locator: str, generation: int = 0) -> OperatorRelayConfiguration:
    """Return a stored route whose identity depends only on ``locator``."""

    return OperatorRelayConfiguration.model_construct(
        routing_locator=locator * 43,
        operator_api_port=52417,
        gateway_server_name=f"skulk-{locator}.remote",
        connector_authority_key_id=f"key-{locator}",
        connector_generation=generation,
    )


@final
class _Sessions:
    """Record session starts and stops; queued outcomes end sessions early."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.outcomes: list[Exception] = []

    async def run(
        self,
        configuration: OperatorRelayConfiguration,
        stop: anyio.Event,
    ) -> None:
        """Run until stopped, or end with the next queued outcome."""

        locator = configuration.routing_locator[0]
        self.started.append(locator)
        try:
            if self.outcomes:
                raise self.outcomes.pop(0)
            await stop.wait()
        finally:
            self.stopped.append(locator)


async def _eventually(predicate: Callable[[], bool]) -> None:
    """Wait up to five seconds for ``predicate`` to hold."""

    with anyio.fail_after(5):
        while not predicate():
            await anyio.sleep(0.005)


def _supervisor(
    stored: list[OperatorRelayConfiguration | None],
    sessions: _Sessions,
    *,
    poll_seconds: float = 0.01,
    failure_retry_seconds: float = 0.05,
) -> OperatorRemoteAccessSupervisor:
    """Build a supervisor over a mutable stored route."""

    return OperatorRemoteAccessSupervisor(
        load_configuration=lambda: stored[0],
        run_session=sessions.run,
        poll_seconds=poll_seconds,
        minimum_failure_retry_seconds=failure_retry_seconds,
    )


async def test_a_new_route_starts_ingress_and_forgetting_it_stops_ingress() -> None:
    """Registration and forgetting take effect without a restart."""

    stored: list[OperatorRelayConfiguration | None] = [None]
    sessions = _Sessions()
    supervisor = _supervisor(stored, sessions)
    shutdown = anyio.Event()
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(supervisor.run, shutdown)
        await anyio.sleep(0.05)
        assert sessions.started == []
        assert supervisor.state == "not_configured"

        stored[0] = _route("a")
        await _eventually(lambda: sessions.started == ["a"])
        assert supervisor.state == "running"

        stored[0] = None
        await _eventually(lambda: sessions.stopped == ["a"])
        await _eventually(lambda: supervisor.state == "not_configured")
        shutdown.set()


async def test_generation_advances_never_restart_but_a_new_route_does() -> None:
    """The connector's own reconnect bookkeeping must not bounce its session."""

    stored: list[OperatorRelayConfiguration | None] = [_route("a", 0)]
    sessions = _Sessions()
    supervisor = _supervisor(stored, sessions)
    shutdown = anyio.Event()
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(supervisor.run, shutdown)
        await _eventually(lambda: sessions.started == ["a"])

        stored[0] = _route("a", 7)
        await anyio.sleep(0.1)
        assert (sessions.started, sessions.stopped) == (["a"], [])

        stored[0] = _route("b")
        await _eventually(lambda: sessions.started == ["a", "b"])
        assert sessions.stopped == ["a"]
        shutdown.set()
    assert sessions.stopped == ["a", "b"]


async def test_a_rejected_route_stays_stopped_until_it_is_replaced() -> None:
    """A permanent relay refusal is surfaced instead of retried forever."""

    stored: list[OperatorRelayConfiguration | None] = [_route("a")]
    sessions = _Sessions()
    sessions.outcomes.append(OperatorRelayRouteRejectedError("revoked"))
    supervisor = _supervisor(stored, sessions)
    shutdown = anyio.Event()
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(supervisor.run, shutdown)
        await _eventually(lambda: supervisor.state == "rejected")
        await anyio.sleep(0.1)
        assert sessions.started == ["a"]

        stored[0] = _route("b")
        await _eventually(lambda: sessions.started == ["a", "b"])
        assert supervisor.state == "running"
        shutdown.set()


async def test_an_unexpected_session_failure_is_retried_after_a_delay() -> None:
    """A listener or connector crash leaves the API up and retries remote access."""

    stored: list[OperatorRelayConfiguration | None] = [_route("a")]
    sessions = _Sessions()
    sessions.outcomes.append(OSError("private listener detail"))
    supervisor = _supervisor(stored, sessions, failure_retry_seconds=0.1)
    shutdown = anyio.Event()
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(supervisor.run, shutdown)
        await _eventually(lambda: supervisor.state == "failed")
        assert sessions.started == ["a"]
        await _eventually(lambda: sessions.started == ["a", "a"])
        assert supervisor.state == "running"
        shutdown.set()


async def test_unreadable_route_state_keeps_a_running_session() -> None:
    """A transient journal read failure must not tear down working ingress."""

    stored: list[OperatorRelayConfiguration | None] = [_route("a")]
    unreadable = [False]

    def load() -> OperatorRelayConfiguration | None:
        if unreadable[0]:
            raise RuntimeError("journal busy")
        return stored[0]

    sessions = _Sessions()
    supervisor = OperatorRemoteAccessSupervisor(
        load_configuration=load,
        run_session=sessions.run,
        poll_seconds=0.01,
    )
    shutdown = anyio.Event()
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(supervisor.run, shutdown)
        await _eventually(lambda: sessions.started == ["a"])
        unreadable[0] = True
        await anyio.sleep(0.1)
        assert (sessions.stopped, supervisor.state) == ([], "running")
        unreadable[0] = False
        shutdown.set()


async def test_a_change_request_is_applied_without_waiting_for_the_poll() -> None:
    """The dashboard's own registration starts ingress immediately."""

    stored: list[OperatorRelayConfiguration | None] = [None]
    sessions = _Sessions()
    supervisor = _supervisor(stored, sessions, poll_seconds=60.0)
    shutdown = anyio.Event()
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(supervisor.run, shutdown)
        await anyio.sleep(0.05)
        stored[0] = _route("a")
        supervisor.request_check()
        await _eventually(lambda: sessions.started == ["a"])
        shutdown.set()
        supervisor.request_check()
