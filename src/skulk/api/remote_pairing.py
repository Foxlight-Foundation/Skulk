"""Dashboard control of phone pairing through the relay.

The dashboard's **Pair a phone** turns phone pairing on: when this node has no
relay route, it registers one and remote access starts without a restart.
**Turn off phone pairing** forgets the route. Between the two, the dashboard
polls a status that says whether the relay holds a live session from this
gateway, whether the relay permanently refused the route, and whether another
node in the cluster already manages phone pairing.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal, final
from urllib.parse import urlsplit

from pydantic import Field

from skulk.api.operator_remote_access import OperatorRemoteAccessState
from skulk.operator.pairing import OperatorPairingService
from skulk.operator.relay import (
    DEFAULT_OPERATOR_API_PORT,
    OperatorRelayAlreadyConfiguredError,
    OperatorRelayProvisioning,
)
from skulk.operator.relay_registration import (
    RelayRegistrationError,
    RelayRegistrationFailure,
    RelayRegistrationMaterial,
    resolve_registration_origin,
)
from skulk.store.config import RelayConnectivityConfig
from skulk.utils.pydantic_ext import FrozenModel

type RemotePairingState = Literal[
    "not_set_up",
    "registering",
    "connecting",
    "connected",
    "relay_unreachable",
    "revoked",
    "managed_elsewhere",
]
"""Phone-pairing state for the dashboard.

``not_set_up``: no relay route; Pair a phone registers one. ``registering``: a
registration is in flight. ``connecting``: a route is stored and its relay
session is starting. ``connected``: the relay holds a live session from this
gateway. ``relay_unreachable``: a route is stored but no session has been live
for the grace period. ``revoked``: the relay permanently refused the route;
turn phone pairing off and on again. ``managed_elsewhere``: another node holds
the cluster's relay route.
"""

type RegistrationBlockedReason = Literal["disabled", "offline", "not_configured"]

type RemotePairingRefusal = Literal["managed_elsewhere", "device_limit", "busy"]

_CONNECT_GRACE_SECONDS: Final = 20.0


@final
@dataclass(frozen=True, slots=True)
class RelayLink:
    """Liveness of the running relay connector."""

    connected: bool
    disconnected_seconds: float


class RemotePairingStatus(FrozenModel):
    """Phone-pairing status shown in the dashboard's Devices & pairing."""

    state: RemotePairingState = Field(description="Current phone-pairing state.")
    registration_available: bool = Field(
        description=(
            "Whether this node can register a relay route: registration is "
            "turned on, the node is online, and a relay is configured."
        )
    )
    registration_blocked_reason: RegistrationBlockedReason | None = Field(
        default=None,
        description="Why this node cannot register a relay route, when it cannot.",
    )
    last_failure: RelayRegistrationFailure | None = Field(
        default=None,
        description=(
            "Stable code of the last failed registration since phone pairing "
            "was last turned on or off."
        ),
    )
    retry_after_seconds: int | None = Field(
        default=None,
        description="Seconds the relay asked to wait before registering again.",
    )
    managed_on_node_id: str | None = Field(
        default=None,
        description="Node that holds the cluster's relay route, when it is another node.",
    )
    managed_on_node_name: str | None = Field(
        default=None,
        description="Friendly name of that node, when known.",
    )
    relay_host: str | None = Field(
        default=None,
        description="Host of the relay this node uses, or would register with.",
    )


class RemotePairingErrorBody(FrozenModel):
    """Error body returned when phone pairing cannot be turned on or off."""

    detail: str = Field(description="Operator-readable explanation in English.")
    code: str = Field(
        description=(
            "Stable failure code: a registration failure code, or "
            "managed_elsewhere, device_limit, or busy."
        )
    )
    retry_after_seconds: int | None = Field(
        default=None,
        description="Seconds to wait before retrying, when known.",
    )


class RemotePairingRefusedError(RuntimeError):
    """Raised when this node must not turn phone pairing on or off right now."""

    def __init__(self, refusal: RemotePairingRefusal, message: str) -> None:
        """Create a refusal with its stable code and English explanation.

        Args:
            refusal: Stable refusal code.
            message: Operator-readable explanation.
        """

        super().__init__(message)
        self.refusal: RemotePairingRefusal = refusal


@final
class RemotePairingController:
    """Compute phone-pairing status and turn relay pairing on and off.

    Methods block on journal reads and, for :meth:`enable`, on the relay; call
    them from a worker thread.
    """

    def __init__(
        self,
        *,
        service: OperatorPairingService,
        relay_settings: Callable[[], RelayConnectivityConfig | None],
        offline: Callable[[], bool],
        remote_access_state: Callable[[], OperatorRemoteAccessState],
        relay_link: Callable[[], RelayLink | None],
        request_remote_access_check: Callable[[], None],
        pairing_gateway_elsewhere: Callable[[], tuple[str, str | None] | None],
        operator_api_port: int = DEFAULT_OPERATOR_API_PORT,
        register: Callable[[str, RelayRegistrationMaterial], OperatorRelayProvisioning]
        | None = None,
        monotonic_seconds: Callable[[], float] = time.monotonic,
        connect_grace_seconds: float = _CONNECT_GRACE_SECONDS,
    ) -> None:
        """Create a controller over injected node effects.

        Args:
            service: This node's pairing service.
            relay_settings: Current ``connectivity.relay`` settings, if any.
            offline: Whether the node runs offline.
            remote_access_state: State of the relay ingress supervisor.
            relay_link: Liveness of the running connector; ``None`` when no
                relay session is running.
            request_remote_access_check: Make relay ingress follow a route
                change immediately.
            pairing_gateway_elsewhere: Another node advertising the cluster's
                relay route, as ``(node_id, friendly_name)``.
            operator_api_port: Loopback port of the relay-only TLS listener.
            register: Registration call; ``None`` uses the real relay client.
            monotonic_seconds: Monotonic clock for retry delays.
            connect_grace_seconds: How long a stored route may stay without a
                live session before it reads as unreachable.
        """

        self._service = service
        self._relay_settings = relay_settings
        self._offline = offline
        self._remote_access_state = remote_access_state
        self._relay_link = relay_link
        self._request_remote_access_check = request_remote_access_check
        self._pairing_gateway_elsewhere = pairing_gateway_elsewhere
        self._operator_api_port = operator_api_port
        self._register = register
        self._monotonic_seconds = monotonic_seconds
        self._connect_grace_seconds = connect_grace_seconds
        self._lock = threading.Lock()
        self._registering = False
        self._last_failure: RelayRegistrationFailure | None = None
        self._retry_until: float | None = None

    def status(self) -> RemotePairingStatus:
        """Return the current phone-pairing status.

        Returns:
            The state plus registration availability, the last failure, and
            where pairing is managed when it is another node.
        """

        blocked = self._registration_blocked_reason()
        configuration = self._service.relay_configuration()
        relay_host: str | None = None
        managed_on: tuple[str, str | None] | None = None
        state: RemotePairingState
        if configuration is not None:
            relay_host = urlsplit(configuration.app_websocket_url).hostname
            state = self._route_state()
        else:
            relay_host = self._registration_host()
            managed_on = self._pairing_gateway_elsewhere()
            state = "managed_elsewhere" if managed_on is not None else "not_set_up"
        if self._registering:
            state = "registering"
        return RemotePairingStatus(
            state=state,
            registration_available=blocked is None,
            registration_blocked_reason=blocked,
            last_failure=self._last_failure,
            retry_after_seconds=self._retry_after_seconds(),
            managed_on_node_id=managed_on[0] if managed_on is not None else None,
            managed_on_node_name=managed_on[1] if managed_on is not None else None,
            relay_host=relay_host,
        )

    def enable(self) -> RemotePairingStatus:
        """Turn phone pairing on: register a relay route if none is stored.

        Idempotent: with a route already stored, or a registration already in
        flight, this returns the current status.

        Returns:
            The status after registering.

        Raises:
            RemotePairingRefusedError: Another node manages phone pairing, or
                the cluster already has its maximum of paired devices.
            RelayRegistrationError: Registration is unavailable here, or the
                relay refused or could not be reached.
        """

        if not self._lock.acquire(blocking=False):
            return self.status()
        try:
            if self._service.relay_configuration() is not None:
                return self.status()
            managed_on = self._pairing_gateway_elsewhere()
            if managed_on is not None:
                raise RemotePairingRefusedError(
                    "managed_elsewhere",
                    "Phone pairing for this cluster is managed on "
                    f"{managed_on[1] or managed_on[0]}. Open that node's dashboard "
                    "to pair a phone.",
                )
            if self._service.pairing_capacity().available_slots == 0:
                raise RemotePairingRefusedError(
                    "device_limit",
                    "This cluster already has the most paired devices it allows. "
                    "Revoke a device before pairing another.",
                )
            origin = self._registration_origin()
            self._registering = True
            try:
                self._service.register_relay(
                    registration_origin=origin,
                    operator_api_port=self._operator_api_port,
                    register=self._register,
                )
            except OperatorRelayAlreadyConfiguredError:
                # Another process, such as `skulk operator pair`, registered
                # first; its route serves this node just as well.
                pass
            except RelayRegistrationError as exc:
                self._last_failure = exc.failure
                self._retry_until = (
                    self._monotonic_seconds() + exc.retry_after_seconds
                    if exc.retry_after_seconds is not None
                    else None
                )
                raise
            finally:
                self._registering = False
            self._last_failure = None
            self._retry_until = None
            self._request_remote_access_check()
            return self.status()
        finally:
            self._lock.release()

    def disable(self) -> RemotePairingStatus:
        """Turn phone pairing off: forget this node's relay route.

        Phones paired through the route lose remote access, and invitations
        bound to it are revoked because they can no longer reach this node.

        Returns:
            The status after forgetting the route.

        Raises:
            RemotePairingRefusedError: A registration is in flight.
        """

        if not self._lock.acquire(blocking=False):
            raise RemotePairingRefusedError(
                "busy",
                "Phone pairing is being turned on. Wait for it to finish, then "
                "turn it off.",
            )
        try:
            self._service.forget_relay()
            self._last_failure = None
            self._retry_until = None
            self._request_remote_access_check()
            return self.status()
        finally:
            self._lock.release()

    def _route_state(self) -> RemotePairingState:
        """Classify a stored route by its relay session's liveness."""

        remote_access_state = self._remote_access_state()
        if remote_access_state == "rejected":
            return "revoked"
        link = self._relay_link()
        if link is not None and link.connected:
            return "connected"
        if remote_access_state == "failed":
            return "relay_unreachable"
        if link is None or link.disconnected_seconds < self._connect_grace_seconds:
            # Either relay ingress has not picked the route up yet (it polls
            # every few seconds) or its session is still starting.
            return "connecting"
        return "relay_unreachable"

    def _registration_blocked_reason(self) -> RegistrationBlockedReason | None:
        """Return why this node cannot register a route, or ``None``."""

        try:
            self._registration_origin()
        except RelayRegistrationError as exc:
            if exc.failure == "disabled":
                return "disabled"
            if exc.failure == "offline":
                return "offline"
            return "not_configured"
        return None

    def _registration_origin(self) -> str:
        """Resolve the relay origin this node would register with."""

        settings = self._relay_settings()
        return resolve_registration_origin(
            enabled=settings.enabled if settings is not None else True,
            configured_origin=(
                settings.registration_url if settings is not None else None
            ),
            offline=self._offline(),
        )

    def _registration_host(self) -> str | None:
        """Return the host of the relay this node would register with."""

        try:
            return urlsplit(self._registration_origin()).hostname
        except RelayRegistrationError:
            return None

    def _retry_after_seconds(self) -> int | None:
        """Return the remaining relay-requested delay, if any."""

        if self._retry_until is None:
            return None
        remaining = self._retry_until - self._monotonic_seconds()
        if remaining <= 0:
            return None
        return int(remaining) + 1
