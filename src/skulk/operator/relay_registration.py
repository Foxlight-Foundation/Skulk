"""Self-service registration of this gateway's on-demand relay route.

The gateway generates every secret it uses with the relay: the P-256 connector
authority key, the 16-byte authority epoch, and both 32-byte carrier
credentials. Registration sends the relay only the connector key's identifier
and SHA-256 digests of the two credentials, so no relay state yields a usable
credential, and repeating a lost request with the same values returns the same
route. The relay answers with an opaque route locator, the route's region, and
its three WebSocket endpoints. Skulk stores the result in the existing
on-demand provisioning shape, so the connector, pairing QR, and exchange are
unchanged. The connector's first signed hello establishes the authority epoch,
exactly as for a hand-provisioned on-demand route.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from collections.abc import Callable
from typing import Final, Literal, cast, final

import httpx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from pydantic import Field

from skulk.operator.relay import OperatorRelayProvisioning
from skulk.utils.pydantic_ext import FrozenModel
from skulk.utils.relay_origin import validate_relay_origin

DEFAULT_RELAY_REGISTRATION_ORIGIN: Final[str | None] = None
"""Relay origin used when ``connectivity.relay.registration_url`` is unset.

PENDING: the production relay origin has not been decided (Phase 0 of the relay
self-enrollment plan). Until it is set here, a node registers only when
``skulk.yaml`` names a relay in ``connectivity.relay.registration_url``.
"""

REGISTRATION_PATH: Final = "/v1/registrations"
_MAXIMUM_RESPONSE_BYTES: Final = 16_384
_RETRY_DELAYS_SECONDS: Final = (0.5, 1.5)
_REQUEST_TIMEOUT: Final = httpx.Timeout(15.0, connect=10.0)

type RelayRegistrationFailure = Literal[
    "disabled",
    "offline",
    "not_configured",
    "invalid_request",
    "already_registered",
    "rate_limited",
    "registration_paused",
    "capacity_exhausted",
    "registration_unsupported",
    "relay_busy",
    "unreachable",
    "invalid_response",
]

# Stable refusal codes the relay may return, mapped to this module's failures.
_RELAY_FAILURE_CODES: Final[dict[str, RelayRegistrationFailure]] = {
    "invalid_request": "invalid_request",
    "already_registered": "already_registered",
    "rate_limited": "rate_limited",
    "registration_paused": "registration_paused",
    "capacity_exhausted": "capacity_exhausted",
    # A relay with registration turned off answers 404 ``not_found``.
    "not_found": "registration_unsupported",
}
# Coded answers meaning the relay is reachable but momentarily unable to
# register (admission or worker capacity, draining, a persistence failure).
# They are retried like a network failure and reported as busy if they persist.
_TRANSIENT_RELAY_CODES: Final[frozenset[str]] = frozenset({"unavailable"})
# Refusals whose ``Retry-After`` delay is passed on to the caller.
_DELAYED_FAILURES: Final[frozenset[RelayRegistrationFailure]] = frozenset(
    {"rate_limited", "registration_paused"}
)
_STATUS_FAILURES: Final[dict[int, RelayRegistrationFailure]] = {
    400: "invalid_request",
    404: "registration_unsupported",
    409: "already_registered",
    429: "rate_limited",
}
_FAILURE_MESSAGES: Final[dict[RelayRegistrationFailure, str]] = {
    "disabled": (
        "relay registration is turned off (connectivity.relay.enabled is false)"
    ),
    "offline": "relay registration needs network access, and this node runs offline",
    "not_configured": (
        "no relay is configured for registration; set "
        "connectivity.relay.registration_url in skulk.yaml"
    ),
    "invalid_request": "the relay rejected the registration request",
    "already_registered": (
        "the relay already holds a route for this gateway key with different "
        "credentials"
    ),
    "rate_limited": "the relay is limiting registrations from this network",
    "registration_paused": "the relay is not accepting new registrations right now",
    "capacity_exhausted": "the relay has no room for new routes right now",
    "registration_unsupported": (
        "this relay does not accept registrations; check "
        "connectivity.relay.registration_url in skulk.yaml"
    ),
    "relay_busy": "the relay is busy right now",
    "unreachable": "the relay could not be reached",
    "invalid_response": "the relay returned an invalid registration response",
}


class RelayRegistrationError(RuntimeError):
    """Safe, typed failure to register this gateway's relay route.

    The message never contains credential material, request bodies, or relay
    response text.
    """

    def __init__(
        self,
        failure: RelayRegistrationFailure,
        *,
        retry_after_seconds: int | None = None,
    ) -> None:
        """Create a failure from its stable code and optional retry hint.

        Args:
            failure: Stable failure code.
            retry_after_seconds: Seconds to wait before retrying, when the relay
                asked for a delay.
        """

        message = _FAILURE_MESSAGES[failure]
        if retry_after_seconds is not None:
            message = f"{message}; try again in {retry_after_seconds} seconds"
        super().__init__(message)
        self.failure: RelayRegistrationFailure = failure
        self.retry_after_seconds = retry_after_seconds


@final
class RelayRegistrationMaterial(FrozenModel):
    """Secrets this gateway generates for one relay registration.

    Only the key identifier and the two credential digests ever leave the node.
    """

    connector_authority_private_key_pkcs8: str = Field(
        description="Unpadded base64url DER PKCS8 P-256 connector authority key."
    )
    connector_authority_key_id: str = Field(
        description="Unpadded base64url SHA-256 of the key's DER SubjectPublicKeyInfo."
    )
    connector_authority_epoch: str = Field(
        description="Unpadded base64url 16-byte authority epoch."
    )
    app_carrier_credential: str = Field(
        description="Unpadded base64url 32-byte app-role carrier credential."
    )
    gateway_carrier_credential: str = Field(
        description="Unpadded base64url 32-byte gateway-role carrier credential."
    )

    @classmethod
    def generate(cls) -> "RelayRegistrationMaterial":
        """Create fresh connector authority and carrier secrets for this gateway.

        Returns:
            New material; the two carrier credentials are always distinct.
        """

        private_key = ec.generate_private_key(ec.SECP256R1())
        public_key = private_key.public_key().public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        app_credential = secrets.token_bytes(32)
        gateway_credential = secrets.token_bytes(32)
        while gateway_credential == app_credential:
            gateway_credential = secrets.token_bytes(32)
        return cls(
            connector_authority_private_key_pkcs8=_encode(
                private_key.private_bytes(
                    serialization.Encoding.DER,
                    serialization.PrivateFormat.PKCS8,
                    serialization.NoEncryption(),
                )
            ),
            connector_authority_key_id=_encode(hashlib.sha256(public_key).digest()),
            connector_authority_epoch=_encode(secrets.token_bytes(16)),
            app_carrier_credential=_encode(app_credential),
            gateway_carrier_credential=_encode(gateway_credential),
        )

    def registration_body(self) -> dict[str, str]:
        """Return the exact registration request body.

        Returns:
            The key identifier plus SHA-256 digests of the raw 32-byte carrier
            credentials, all canonical unpadded base64url. No secret is
            included.
        """

        return {
            "connectorAuthorityKeyId": self.connector_authority_key_id,
            "appCarrierCredentialDigest": _credential_digest(
                self.app_carrier_credential
            ),
            "gatewayCarrierCredentialDigest": _credential_digest(
                self.gateway_carrier_credential
            ),
        }


def resolve_registration_origin(
    *,
    enabled: bool,
    configured_origin: str | None,
    offline: bool,
) -> str:
    """Choose the relay origin this gateway registers with.

    Args:
        enabled: ``connectivity.relay.enabled`` from ``skulk.yaml``.
        configured_origin: ``connectivity.relay.registration_url``, if set.
        offline: Whether the node runs in offline mode.

    Returns:
        A canonical relay origin.

    Raises:
        RelayRegistrationError: Registration is disabled, the node is offline,
            or no relay is configured and the build has no default.
    """

    if not enabled:
        raise RelayRegistrationError("disabled")
    if offline:
        raise RelayRegistrationError("offline")
    origin = configured_origin or DEFAULT_RELAY_REGISTRATION_ORIGIN
    if origin is None:
        raise RelayRegistrationError("not_configured")
    return validate_relay_origin(origin)


def register_relay_route(
    origin: str,
    material: RelayRegistrationMaterial,
    *,
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> OperatorRelayProvisioning:
    """Register one on-demand route with a relay and return its provisioning.

    Transient failures (network errors, timeouts, uncoded 5xx responses such
    as a tunnel outage, and the relay's ``unavailable`` code) are retried with
    the same material, which the relay treats as the same registration. Other
    coded refusals are not retried.

    Args:
        origin: Relay origin, validated again here.
        material: Secrets generated for this registration.
        transport: Optional HTTP transport, for tests.
        sleep: Delay function between attempts, for tests.

    Returns:
        Version-two provisioning combining the relay's route with this
        gateway's secrets, fully validated.

    Raises:
        RelayRegistrationError: The relay refused, could not be reached, or
            answered invalidly.
    """

    url = f"{validate_relay_origin(origin)}{REGISTRATION_PATH}"
    body = material.registration_body()
    last_failure = RelayRegistrationError("unreachable")
    for attempt in range(len(_RETRY_DELAYS_SECONDS) + 1):
        if attempt:
            sleep(_RETRY_DELAYS_SECONDS[attempt - 1])
        try:
            status, retry_after, payload = _post_registration(url, body, transport)
        except httpx.HTTPError:
            last_failure = RelayRegistrationError("unreachable")
            continue
        if 200 <= status < 300:
            return _provisioning_from_response(payload, material)
        failure, transient = _refusal(status, payload)
        if transient:
            last_failure = RelayRegistrationError(failure)
            continue
        raise RelayRegistrationError(
            failure,
            retry_after_seconds=retry_after if failure in _DELAYED_FAILURES else None,
        )
    raise last_failure


def _post_registration(
    url: str,
    body: dict[str, str],
    transport: httpx.BaseTransport | None,
) -> tuple[int, int | None, bytes]:
    """Send one registration and read a bounded response.

    Returns:
        The status code, a parsed ``Retry-After`` in seconds when present, and
        the response body.

    Raises:
        httpx.HTTPError: The request failed in transit.
        RelayRegistrationError: The response exceeded its size bound or
            redirected elsewhere.
    """

    with (
        httpx.Client(
            timeout=_REQUEST_TIMEOUT,
            transport=transport,
            follow_redirects=False,
        ) as client,
        client.stream(
            "POST",
            url,
            json=body,
            headers={"Accept": "application/json"},
        ) as response,
    ):
        if 300 <= response.status_code < 400:
            raise RelayRegistrationError("invalid_response")
        payload = bytearray()
        for chunk in response.iter_bytes():
            payload.extend(chunk)
            if len(payload) > _MAXIMUM_RESPONSE_BYTES:
                raise RelayRegistrationError("invalid_response")
        retry_after = cast(str | None, response.headers.get("retry-after"))
        return (
            response.status_code,
            _retry_after_seconds(retry_after),
            bytes(payload),
        )


def _refusal(status: int, payload: bytes) -> tuple[RelayRegistrationFailure, bool]:
    """Map a non-success response to a stable failure and whether to retry it.

    A recognized code names the failure. A 5xx without a recognized code is
    treated as transient, because the relay's tunnel answers 502, 503, or 504
    while the relay process is unreachable.

    Returns:
        The failure, and ``True`` when the request should be retried.
    """

    code = _response_code(payload)
    if code in _TRANSIENT_RELAY_CODES:
        return "relay_busy", True
    failure = _RELAY_FAILURE_CODES.get(code) if code is not None else None
    if failure is not None:
        return failure, False
    if status >= 500:
        return "unreachable", True
    return _STATUS_FAILURES.get(status, "invalid_response"), False


def _response_code(payload: bytes) -> str | None:
    """Read the ``code`` string from a refusal body, ignoring anything else."""

    try:
        data = cast(object, json.loads(payload))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    code = cast(dict[str, object], data).get("code")
    return code if isinstance(code, str) else None


def _provisioning_from_response(
    payload: bytes,
    material: RelayRegistrationMaterial,
) -> OperatorRelayProvisioning:
    """Combine a successful relay response with this gateway's secrets.

    Unknown response fields are ignored so the relay can add fields without
    breaking older gateways; the five required fields are validated through the
    existing provisioning model.
    """

    try:
        data = cast(object, json.loads(payload))
    except (ValueError, UnicodeDecodeError) as exc:
        raise RelayRegistrationError("invalid_response") from exc
    if not isinstance(data, dict):
        raise RelayRegistrationError("invalid_response")
    response = cast(dict[str, object], data)
    fields: dict[str, str] = {}
    for name in (
        "routingLocator",
        "connectorRegion",
        "appWebsocketUrl",
        "gatewayControlWebsocketUrl",
        "gatewayDataWebsocketUrl",
    ):
        value = response.get(name)
        if not isinstance(value, str):
            raise RelayRegistrationError("invalid_response")
        fields[name] = value
    try:
        return OperatorRelayProvisioning(
            version=2,
            app_websocket_url=fields["appWebsocketUrl"],
            gateway_control_websocket_url=fields["gatewayControlWebsocketUrl"],
            gateway_data_websocket_url=fields["gatewayDataWebsocketUrl"],
            routing_locator=fields["routingLocator"],
            app_carrier_credential=material.app_carrier_credential,
            gateway_carrier_credential=material.gateway_carrier_credential,
            connector_authority_private_key_pkcs8=(
                material.connector_authority_private_key_pkcs8
            ),
            connector_authority_key_id=material.connector_authority_key_id,
            connector_region=fields["connectorRegion"],
            connector_authority_epoch=material.connector_authority_epoch,
        )
    except ValueError as exc:
        raise RelayRegistrationError("invalid_response") from exc


def _retry_after_seconds(value: str | None) -> int | None:
    """Parse a bounded integer ``Retry-After`` delay, ignoring HTTP dates."""

    if value is None or not value.strip().isdigit():
        return None
    return min(int(value.strip()), 86_400)


def _credential_digest(encoded_credential: str) -> str:
    """Return the base64url SHA-256 of one raw 32-byte carrier credential."""

    return _encode(hashlib.sha256(_decode(encoded_credential)).digest())


def _encode(value: bytes) -> str:
    """Encode canonical unpadded base64url."""

    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    """Decode canonical unpadded base64url."""

    return base64.urlsafe_b64decode(f"{value}{'=' * (-len(value) % 4)}")
