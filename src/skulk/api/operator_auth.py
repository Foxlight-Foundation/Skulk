# pyright: reportUnusedFunction=false
"""FastAPI routes for host-authorized Skulk device pairing."""

from fastapi import APIRouter, HTTPException

from skulk.operator.pairing import (
    OperatorPairingService,
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


def create_operator_auth_router(service: OperatorPairingService) -> APIRouter:
    """Create narrowly scoped pairing routes for one designated gateway.

    Args:
        service: Encrypted local pairing service owned by the API node.

    Returns:
        Router exposing only challenge and credential exchange before
        authentication. General Skulk APIs are not added to this router.
    """

    router = APIRouter(prefix="/v1/auth", tags=["Authentication"])

    @router.post(
        "/pairing-sessions/{nonce}/challenge",
        response_model=PairingChallengeResponse,
        summary="Bind a device key to a local pairing session",
        description=(
            "Accept a candidate Ed25519 public key only when the nonce names an "
            "unexpired host-created pairing session, then return one random "
            "challenge for proof of possession."
        ),
    )
    def create_pairing_challenge(
        nonce: str,
        request: PairingChallengeRequest,
    ) -> PairingChallengeResponse:
        """Create one device proof challenge for a pending pairing session."""

        try:
            return service.create_challenge(nonce, request)
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
        "/pairing-sessions/{nonce}/exchange",
        response_model=PairingExchangeResponse,
        summary="Exchange a device-key proof for operator credentials",
        description=(
            "Verify the candidate device's Ed25519 signature, consume the "
            "single-use session, and return short-lived access plus rotating "
            "refresh credentials exactly once."
        ),
    )
    def exchange_pairing_proof(
        nonce: str,
        request: PairingExchangeRequest,
    ) -> PairingExchangeResponse:
        """Verify a candidate device and consume its pairing capability."""

        try:
            return service.exchange(nonce, request)
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

    return router
