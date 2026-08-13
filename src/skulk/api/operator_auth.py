# pyright: reportUnusedFunction=false
"""FastAPI routes for Skulk operator pairing and credential lifecycle."""

from datetime import timedelta
from ipaddress import ip_address
from typing import Annotated, Never
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Request, Response, status

from skulk.operator.pairing import (
    OperatorCredentialExpiredError,
    OperatorCredentialInvalidError,
    OperatorDeviceNotFoundError,
    OperatorDevicesResponse,
    OperatorPairingService,
    OperatorScopeError,
    OperatorTokenRequest,
    OperatorTokenResponse,
    PairingChallengeRequest,
    PairingChallengeResponse,
    PairingExchangeRequest,
    PairingExchangeResponse,
    PairingGatewayNotInitializedError,
    PairingInvitationCapacityError,
    PairingInvitationCreateRequest,
    PairingInvitationCreateResponse,
    PairingInvitationSummary,
    PairingPackageTooLargeError,
    PairingProofError,
    PairingSessionExpiredError,
    PairingSessionNotFoundError,
    PairingSessionStateError,
)
from skulk.operator.relay import OperatorRelayUnavailableError

_BEARER_CHALLENGE = {"WWW-Authenticate": "Bearer"}
_PAIRING_INVITATION_PATH = "/pairing-invitations"
_DASHBOARD_REQUEST_HEADER = "pairing-v1"
_FORWARDED_REQUEST_HEADERS = frozenset(
    {
        b"forwarded",
        b"x-forwarded-for",
        b"x-forwarded-host",
        b"x-forwarded-proto",
        b"x-real-ip",
        b"cf-connecting-ip",
        b"true-client-ip",
    }
)


def _local_dashboard_authority_request(request: Request) -> bool:
    """Return whether a browser request came from the node-local dashboard.

    Minting a bearer invitation establishes persistent remote operator access,
    so same-origin headers alone are insufficient: a network client can forge
    them and a browser can be DNS-rebound. Both the socket peer and browser
    origin must therefore be loopback. Forwarding headers are rejected because
    a local reverse proxy would otherwise erase the real peer boundary.
    """

    if any(name.lower() in _FORWARDED_REQUEST_HEADERS for name, _ in request.headers.raw):
        return False
    if request.headers.get("x-skulk-dashboard") != _DASHBOARD_REQUEST_HEADER:
        return False
    origin = request.headers.get("origin") or request.headers.get("referer")
    client_host = request.client.host if request.client is not None else None
    if origin is None or client_host is None:
        return False

    def is_loopback(host: str | None) -> bool:
        if host == "localhost":
            return True
        if host is None:
            return False
        try:
            return ip_address(host).is_loopback
        except ValueError:
            return False

    try:
        parsed_origin = urlsplit(origin)
    except ValueError:
        return False
    return (
        is_loopback(client_host)
        and parsed_origin.scheme in {"http", "https"}
        and parsed_origin.username is None
        and parsed_origin.password is None
        and is_loopback(parsed_origin.hostname)
    )


def _require_local_dashboard_authority(request: Request) -> None:
    """Reject invitation management outside the node-local dashboard."""

    if not _local_dashboard_authority_request(request):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "pairing invitations may be managed only from a dashboard opened "
                "through localhost on the configured operator gateway"
            ),
        )


def _require_bearer(authorization: str | None) -> str:
    """Extract one opaque bearer token without accepting alternate schemes."""

    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="bearer credential is required",
            headers=_BEARER_CHALLENGE,
        )
    scheme, separator, token = authorization.partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="bearer credential is invalid",
            headers=_BEARER_CHALLENGE,
        )
    return token.strip()


def _raise_credential_http_error(exc: Exception) -> Never:
    """Map safe credential-domain failures onto stable HTTP semantics."""

    if isinstance(exc, PairingGatewayNotInitializedError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if isinstance(exc, OperatorScopeError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(
        exc,
        (OperatorCredentialInvalidError, OperatorCredentialExpiredError),
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers=_BEARER_CHALLENGE,
        ) from exc
    raise exc


def create_operator_auth_router(service: OperatorPairingService) -> APIRouter:
    """Create narrowly scoped pairing routes for one designated gateway.

    Args:
        service: Encrypted local pairing service owned by the API node.

    Returns:
        Router exposing pairing, refresh rotation, and paired-device
        management. General Skulk model and inference APIs remain canonical.
    """

    router = APIRouter(prefix="/v1/auth", tags=["Authentication"])

    @router.post(
        _PAIRING_INVITATION_PATH,
        response_model=PairingInvitationCreateResponse,
        response_model_by_alias=True,
        summary="Create a dashboard pairing invitation",
        description=(
            "Create one bounded, revocable pairing invitation from the node-local "
            "Skulk dashboard. The secret pairing code is returned "
            "once with no-store response headers and is never relay-accessible."
        ),
    )
    def create_pairing_invitation(
        payload: PairingInvitationCreateRequest,
        request: Request,
        response: Response,
    ) -> PairingInvitationCreateResponse:
        """Create one invitation and return its QR payload exactly once."""

        _require_local_dashboard_authority(request)
        try:
            package = service.create_invitation(
                lifetime=timedelta(seconds=payload.valid_for_seconds),
                max_pairings=payload.max_pairings,
            )
            invitation = next(
                item
                for item in service.invitations()
                if item.invitation_id == package.invitation_id
            )
        except PairingPackageTooLargeError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="pairing invitation is too large for a reliable QR code",
            ) from exc
        except (PairingGatewayNotInitializedError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "generate pairing invitations from the configured operator gateway; "
                    "configure relay access on this node first"
                ),
            ) from exc
        except OperatorRelayUnavailableError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="the operator gateway identity is temporarily unavailable",
            ) from exc
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return PairingInvitationCreateResponse(
            invitation=invitation,
            pairing_code=package.as_url(),
        )

    @router.get(
        _PAIRING_INVITATION_PATH,
        response_model=list[PairingInvitationSummary],
        response_model_by_alias=True,
        summary="List dashboard pairing invitations",
        description=(
            "Return safe invitation status without bearer nonces or pairing "
            "codes. This management route is available only to the direct "
            "node-local Skulk dashboard and never through the relay gateway."
        ),
    )
    def list_pairing_invitations(request: Request) -> list[PairingInvitationSummary]:
        """List safe status for invitations created on this gateway."""

        _require_local_dashboard_authority(request)
        return list(service.invitations())

    @router.delete(
        f"{_PAIRING_INVITATION_PATH}/{{invitation_id}}",
        status_code=status.HTTP_204_NO_CONTENT,
        summary="Revoke a dashboard pairing invitation",
        description=(
            "Immediately prevent new and unfinished pairing attempts for one "
            "invitation without revoking devices that already paired. This "
            "management route is never relay-accessible."
        ),
    )
    def revoke_pairing_invitation(invitation_id: UUID, request: Request) -> Response:
        """Revoke one invitation from the node-local dashboard."""

        _require_local_dashboard_authority(request)
        try:
            service.revoke_invitation(invitation_id)
        except PairingSessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PairingSessionStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @router.post(
        "/pairing-sessions/challenge",
        response_model=PairingChallengeResponse,
        response_model_exclude_none=True,
        summary="Bind a device key to a local pairing session",
        description=(
            "Accept a candidate Ed25519 public key only when the nonce names an "
            "unexpired host-created pairing session or invitation, then return "
            "one random challenge for proof of possession. Reusable invitations "
            "return an independent five-minute attempt identity."
        ),
    )
    def create_pairing_challenge(
        request: PairingChallengeRequest,
    ) -> PairingChallengeResponse:
        """Create one device proof challenge for a pending pairing session."""

        try:
            return service.create_challenge(request)
        except PairingGatewayNotInitializedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except PairingSessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PairingSessionExpiredError as exc:
            raise HTTPException(status_code=410, detail=str(exc)) from exc
        except PairingSessionStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except PairingInvitationCapacityError as exc:
            raise HTTPException(
                status_code=429,
                detail=str(exc),
                headers={"Retry-After": str(exc.retry_after_seconds)},
            ) from exc
        except PairingProofError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post(
        "/pairing-sessions/exchange",
        response_model=PairingExchangeResponse,
        summary="Exchange a device-key proof for operator credentials",
        description=(
            "Verify the candidate device's Ed25519 signature, consume its "
            "single-use session or independent invitation attempt, and return "
            "short-lived access plus rotating refresh credentials exactly once."
        ),
    )
    def exchange_pairing_proof(
        request: PairingExchangeRequest,
    ) -> PairingExchangeResponse:
        """Verify a candidate device and consume its pairing capability."""

        try:
            return service.exchange(request)
        except PairingGatewayNotInitializedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except PairingSessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PairingSessionExpiredError as exc:
            raise HTTPException(status_code=410, detail=str(exc)) from exc
        except PairingSessionStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except PairingProofError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    @router.post(
        "/token",
        response_model=OperatorTokenResponse,
        summary="Rotate an operator refresh credential",
        description=(
            "Accept the current opaque refresh credential for one paired "
            "device, invalidate its existing token pair, and return a fresh "
            "short-lived access token plus rotating refresh token exactly once."
        ),
    )
    def refresh_operator_token(request: OperatorTokenRequest) -> OperatorTokenResponse:
        """Rotate the access and refresh credentials for one paired device."""

        try:
            return service.refresh(request)
        except OperatorDeviceNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="refresh credential is invalid",
                headers=_BEARER_CHALLENGE,
            ) from exc
        except (
            OperatorCredentialInvalidError,
            OperatorCredentialExpiredError,
            PairingGatewayNotInitializedError,
        ) as exc:
            _raise_credential_http_error(exc)
        except PairingSessionStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get(
        "/devices",
        response_model=OperatorDevicesResponse,
        summary="List paired operator devices",
        description=(
            "Return safe active and revoked device projections for a bearer "
            "credential with device-management scope. Credential material is "
            "never included."
        ),
    )
    def list_operator_devices(
        authorization: Annotated[str | None, Header()] = None,
    ) -> OperatorDevicesResponse:
        """List devices visible to an authorized operator."""

        try:
            return service.devices(_require_bearer(authorization))
        except (
            OperatorCredentialInvalidError,
            OperatorCredentialExpiredError,
            OperatorScopeError,
            PairingGatewayNotInitializedError,
        ) as exc:
            _raise_credential_http_error(exc)

    @router.delete(
        "/devices/{device_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        summary="Revoke a paired operator device",
        description=(
            "Immediately invalidate the target device's access and refresh "
            "credentials. Repeating revocation for an already revoked device "
            "is idempotent."
        ),
    )
    def revoke_operator_device(
        device_id: UUID,
        authorization: Annotated[str | None, Header()] = None,
    ) -> Response:
        """Revoke one stable paired-device identity."""

        try:
            service.revoke_device(_require_bearer(authorization), device_id)
        except OperatorDeviceNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (
            OperatorCredentialInvalidError,
            OperatorCredentialExpiredError,
            OperatorScopeError,
            PairingGatewayNotInitializedError,
        ) as exc:
            _raise_credential_http_error(exc)
        except PairingSessionStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return router
