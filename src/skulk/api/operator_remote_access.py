"""Keep the gateway's relay ingress in step with its stored relay route.

Remote access used to start once, at API startup, so registering or forgetting
a relay route took a restart. The supervisor here polls the stored route and
reacts to explicit change requests: it starts the relay-only TLS listener and
outbound connector when a route appears, stops them when the route is
forgotten, restarts them when the route is replaced, and leaves a route the
relay permanently refused stopped until it changes. The ordinary local API is
never coupled to any of this.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Awaitable, Callable
from typing import Final, Literal, final

import anyio
from loguru import logger

from skulk.operator.relay import (
    OperatorRelayConfiguration,
    OperatorRelayRouteRejectedError,
)

type OperatorRemoteAccessState = Literal[
    "not_configured",
    "running",
    "rejected",
    "failed",
]
"""Supervisor state.

``not_configured``: no relay route is stored. ``running``: the relay listener
and connector are up (the connector may still be reconnecting).
``rejected``: the relay permanently refused this route; nothing restarts it
until the route is forgotten or replaced. ``failed``: the last session ended
unexpectedly and will be retried after a bounded delay.
"""

type _RouteIdentity = tuple[str, int, str, str | None]

_DEFAULT_POLL_SECONDS: Final = 5.0
_MINIMUM_FAILURE_RETRY_SECONDS: Final = 5.0
_MAXIMUM_FAILURE_RETRY_SECONDS: Final = 60.0


@final
class _Unreadable:
    """Marker for a relay-route read that failed transiently."""


_UNREADABLE: Final = _Unreadable()


def _route_identity(configuration: OperatorRelayConfiguration) -> _RouteIdentity:
    """Identify a route by what a restart must react to.

    The stored configuration also carries the connector generation, which the
    running connector advances on every reconnect; comparing whole
    configurations would restart the session it is supervising.
    """

    return (
        configuration.routing_locator,
        configuration.operator_api_port,
        configuration.gateway_server_name,
        configuration.connector_authority_key_id,
    )


@final
class OperatorRemoteAccessSupervisor:
    """Start, stop, and restart relay ingress as the stored route changes."""

    def __init__(
        self,
        *,
        load_configuration: Callable[[], OperatorRelayConfiguration | None],
        run_session: Callable[
            [OperatorRelayConfiguration, anyio.Event], Awaitable[None]
        ],
        poll_seconds: float = _DEFAULT_POLL_SECONDS,
        minimum_failure_retry_seconds: float = _MINIMUM_FAILURE_RETRY_SECONDS,
        on_state_change: Callable[[OperatorRemoteAccessState], None] | None = None,
    ) -> None:
        """Create a supervisor with injected route loading and session effects.

        Args:
            load_configuration: Reads the stored relay route; ``None`` means no
                route is in use.
            run_session: Runs one relay listener and connector session for a
                route until the given stop event is set. It raises
                ``OperatorRelayRouteRejectedError`` when the relay permanently
                refuses the route.
            poll_seconds: Interval between checks for route changes made by
                another process, such as ``skulk operator pair``.
            minimum_failure_retry_seconds: First delay before retrying a session
                that ended unexpectedly; later delays double up to one minute.
            on_state_change: Called with each new state, for example to
                advertise that this node holds the cluster's relay route.
        """

        self._load_configuration = load_configuration
        self._run_session = run_session
        self._poll_seconds = poll_seconds
        self._minimum_failure_retry_seconds = minimum_failure_retry_seconds
        self._state: OperatorRemoteAccessState = "not_configured"
        self._on_state_change = on_state_change
        self._rejected_route: _RouteIdentity | None = None
        self._wake = anyio.Event()

    @property
    def state(self) -> OperatorRemoteAccessState:
        """Return the current supervisor state."""

        return self._state

    def _set_state(self, state: OperatorRemoteAccessState) -> None:
        """Record a new state and report it once when it changes."""

        if state == self._state:
            return
        self._state = state
        if self._on_state_change is not None:
            self._on_state_change(state)

    def request_check(self) -> None:
        """Re-read the stored route now instead of at the next poll.

        Call after registering or forgetting a route in this process.
        """

        self._wake.set()

    async def run(self, shutdown: anyio.Event) -> None:
        """Supervise relay ingress until ``shutdown`` is set.

        Args:
            shutdown: The API's shutdown signal; setting it stops any session.
        """

        failure_delay = self._minimum_failure_retry_seconds
        while not shutdown.is_set():
            configuration = self._read()
            if isinstance(configuration, _Unreadable) or configuration is None:
                if configuration is None:
                    self._set_state("not_configured")
                    self._rejected_route = None
                await self._wait(shutdown)
                continue
            route = _route_identity(configuration)
            if route == self._rejected_route:
                self._set_state("rejected")
                await self._wait(shutdown)
                continue
            outcome = await self._supervise_session(configuration, route, shutdown)
            if outcome == "rejected":
                self._rejected_route = route
                self._set_state("rejected")
                failure_delay = self._minimum_failure_retry_seconds
            elif outcome == "failed":
                self._set_state("failed")
                with anyio.move_on_after(failure_delay):
                    await shutdown.wait()
                failure_delay = min(failure_delay * 2, _MAXIMUM_FAILURE_RETRY_SECONDS)
            else:
                failure_delay = self._minimum_failure_retry_seconds

    async def _supervise_session(
        self,
        configuration: OperatorRelayConfiguration,
        route: _RouteIdentity,
        shutdown: anyio.Event,
    ) -> Literal["changed", "rejected", "failed"]:
        """Run one session until shutdown, a route change, or its own end."""

        stop = anyio.Event()
        ended: list[Literal["rejected", "failed"]] = []

        async def session() -> None:
            try:
                await self._run_session(configuration, stop)
                if not stop.is_set():
                    ended.append("failed")
            except OperatorRelayRouteRejectedError:
                ended.append("rejected")
            except Exception:
                # Never log exception text: relay paths and transport errors
                # can carry private deployment metadata.
                logger.warning(
                    "Operator remote access stopped unexpectedly; the local API "
                    "remains available and remote access will retry"
                )
                ended.append("failed")
            finally:
                stop.set()

        self._set_state("running")
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(session)
            while not stop.is_set():
                await self._wait(shutdown, stop)
                if shutdown.is_set():
                    break
                current = self._read()
                if isinstance(current, _Unreadable):
                    # A transient read failure must not tear down a working
                    # session; the next poll decides.
                    continue
                if current is None or _route_identity(current) != route:
                    break
            stop.set()
        return ended[0] if ended else "changed"

    def _read(self) -> OperatorRelayConfiguration | None | _Unreadable:
        """Read the stored route, reporting failures without their details."""

        try:
            return self._load_configuration()
        except (OSError, RuntimeError, ValueError, sqlite3.Error):
            logger.warning(
                "Operator relay configuration is unavailable; remote access "
                "keeps its current state"
            )
            return _UNREADABLE

    async def _wait(self, *events: anyio.Event) -> None:
        """Sleep until the next poll, a change request, or any given event."""

        async def wait_for(event: anyio.Event, scope: anyio.CancelScope) -> None:
            await event.wait()
            scope.cancel()

        with anyio.move_on_after(self._poll_seconds) as scope:
            async with anyio.create_task_group() as task_group:
                for event in (self._wake, *events):
                    task_group.start_soon(wait_for, event, scope)
        if self._wake.is_set():
            self._wake = anyio.Event()
