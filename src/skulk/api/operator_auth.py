# pyright: reportUnusedFunction=false
"""FastAPI routes for Skulk operator pairing and credential lifecycle."""

from typing import Annotated, Never
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Response, status

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
    PairingProofError,
    PairingSessionExpiredError,
    PairingSessionNotFoundError,
    PairingSessionStateError,
)

_BEARER_CHALLENGE = {"WWW-Authenticate": "Bearer"}


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
        "/pairing-sessions/challenge",
        response_model=PairingChallengeResponse,
        summary="Bind a device key to a local pairing session",
        description=(
            "Accept a candidate Ed25519 public key only when the nonce names an "
            "unexpired host-created pairing session, then return one random "
            "challenge for proof of possession."
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
        except PairingProofError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post(
        "/pairing-sessions/exchange",
        response_model=PairingExchangeResponse,
        summary="Exchange a device-key proof for operator credentials",
        description=(
            "Verify the candidate device's Ed25519 signature, consume the "
            "single-use session, and return short-lived access plus rotating "
            "refresh credentials exactly once."
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
