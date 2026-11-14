"""Direct-owner client enrollment and actual inference caller authentication."""

import asyncio
import ipaddress
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import cast, final

from fastapi import APIRouter, HTTPException, Request, Response, WebSocket
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field
from starlette.types import Message

from skulk.api.operator_auth import authenticate_capability_owner
from skulk.api.operator_gateway import OPERATOR_GATEWAY_AUTHORIZED_SCOPE_KEY
from skulk.connectivity.tailscale import is_tailscale_peer
from skulk.extensions.inference_callers import (
    AuthenticatedInferenceCaller,
    ClientInventory,
    ClientSummary,
    InferenceCallers,
)
from skulk.extensions.inference_enrollment import (
    InferenceEnrollment,
    InferencePolicyUpdate,
)
from skulk.extensions.native_inference_authority import InferenceAccessPolicy
from skulk.extensions.runtime_attachment import ServiceConnection
from skulk.extensions.runtime_files import read_private
from skulk.shared.constants import SKULK_CONFIG_HOME
from skulk.shared.types.common import NodeId

_FORWARDED = frozenset(
    {
        "forwarded",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-proto",
        "x-real-ip",
        "cf-connecting-ip",
        "true-client-ip",
    }
)


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


@final
class ClientCreate(_Contract):
    """Owner-reviewed new unprivileged inference client."""

    expected_revision: str = Field(
        pattern=r"^[a-f0-9]{64}$", description="Current credential-inventory revision."
    )
    label: str = Field(
        min_length=1,
        max_length=128,
        description="Label identifying the client application.",
    )
    expires_unix_ms: int | None = Field(
        default=None,
        ge=1,
        description="Optional future expiry in Unix milliseconds; null has no scheduled expiry.",
    )


@final
class ClientCreated(_Contract):
    """One-time bearer disclosure; subsequent listings contain metadata only."""

    client: ClientSummary = Field(description="New server-issued client identity.")
    token: str = Field(
        repr=False,
        description="Bearer displayed once; store it in the client securely.",
    )
    revision: str = Field(description="New inventory revision.")


@final
class ClientRevoke(_Contract):
    """Revision-fenced removal of one client credential lifetime."""

    expected_revision: str = Field(
        pattern=r"^[a-f0-9]{64}$", description="Current credential-inventory revision."
    )


def _no_store(response: Response) -> None:
    response.headers["cache-control"] = "no-store"
    response.headers["pragma"] = "no-cache"


async def authenticate_inference_request(
    request: Request | WebSocket, callers: InferenceCallers, issuer_node_id: NodeId
) -> AuthenticatedInferenceCaller:
    """Authenticate a direct owner or enrolled bearer over a protected transport.

    Invalid, duplicated or paired credentials never fall back to ambient owner
    access. Caller IDs in a prompt, header or gateway-signed body are not used.
    This establishes identity only; it does not grant inference permission.
    """
    headers = request.headers.getlist("authorization")
    if not headers:
        if isinstance(request, WebSocket):
            raise HTTPException(
                401, "Native inference WebSockets require an enrolled bearer."
            )
        return callers.owner(
            await authenticate_capability_owner(request, issuer_node_id)
        )
    if (
        len(headers) != 1
        or not headers[0].isascii()
        or request.scope.get(OPERATOR_GATEWAY_AUTHORIZED_SCOPE_KEY) is True
    ):
        raise HTTPException(
            401,
            "Inference client credential refused.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if any(name in request.headers for name in _FORWARDED):
        raise HTTPException(
            403, "Inference clients require a direct protected connection."
        )
    host = request.client.host if request.client is not None else None
    try:
        address = None if host is None else ipaddress.ip_address(host)
    except ValueError:
        address = None
    protected = request.url.scheme in {"https", "wss"} or (
        address is not None and address.is_loopback
    )
    if (
        not protected
        and address is not None
        and (
            address in ipaddress.ip_network("100.64.0.0/10")
            or address in ipaddress.ip_network("fd7a:115c:a1e0::/48")
        )
    ):
        protected = await is_tailscale_peer(str(address))
    if not protected:
        raise HTTPException(
            403,
            "Inference clients require HTTPS, loopback or a verified Tailscale peer.",
        )
    scheme, separator, token = headers[0].partition(" ")
    if scheme.lower() != "bearer" or not separator:
        raise HTTPException(
            401,
            "Inference client credential refused.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return callers.authenticate(token)
    except (OSError, ValueError):
        raise HTTPException(
            401,
            "Inference client credential refused.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None


def create_inference_access_router(
    node_id: NodeId,
    *,
    connection_path: Path = SKULK_CONFIG_HOME / "managed-service" / "connection.json",
) -> APIRouter:
    """Bind documented enrollment routes to the serving Core's actual manager."""

    class AccessRoute(APIRoute):
        """Authenticate before parsing a bounded management body."""

        def get_route_handler(
            self,
        ) -> Callable[[Request], Coroutine[object, object, Response]]:
            """Keep FastAPI's typed body/OpenAPI while bounding its actual receive."""
            original = super().get_route_handler()

            async def bounded(request: Request) -> Response:
                await authenticate_capability_owner(request, node_id)
                remaining = 131072
                deadline = asyncio.get_running_loop().time() + 10

                async def receive() -> Message:
                    nonlocal remaining
                    try:
                        async with asyncio.timeout_at(deadline):
                            message = await request.receive()
                    except TimeoutError:
                        raise HTTPException(
                            408, "Inference management body timed out."
                        ) from None
                    body = cast(object, message.get("body", b""))
                    if not isinstance(body, bytes):
                        raise HTTPException(400, "Inference management body refused.")
                    remaining -= len(body)
                    if remaining < 0:
                        raise HTTPException(
                            413, "Inference management body exceeds its byte bound."
                        )
                    return message

                return await original(Request(request.scope, receive=receive))

            return bounded

    router = APIRouter(
        prefix="/v1/inference-access",
        tags=["Inference access"],
        route_class=AccessRoute,
    )

    def host() -> InferenceCallers:
        try:
            connection = ServiceConnection.model_validate_json(
                read_private(connection_path)
            )
            callers = InferenceCallers(
                Path(connection.manager_root), connection.profile_id, str(node_id)
            )
            callers.check()
            return callers
        except (OSError, ValueError):
            raise HTTPException(
                503, "Inference authority host is unavailable."
            ) from None

    async def clients(request: Request, response: Response) -> ClientInventory:
        """Read safe client metadata after actual direct-owner authentication."""
        await authenticate_capability_owner(request, node_id)
        _no_store(response)
        try:
            return host().inventory()
        except (OSError, ValueError):
            raise HTTPException(
                503, "Inference client inventory is unavailable."
            ) from None

    async def create(
        request: Request, body: ClientCreate, response: Response
    ) -> ClientCreated:
        """Issue one private bearer without inheriting provider or owner authority."""
        await authenticate_capability_owner(request, node_id)
        _no_store(response)
        try:
            service = host()
            client, token = service.create(
                body.expected_revision, body.label, body.expires_unix_ms
            )
            return ClientCreated(
                client=client, token=token, revision=service.inventory().revision
            )
        except (OSError, ValueError):
            raise HTTPException(
                409, "Inference client enrollment changed or is unavailable."
            ) from None

    async def revoke(
        request: Request, identifier: str, body: ClientRevoke, response: Response
    ) -> ClientInventory:
        """Revoke one credential while preserving every other client and provider."""
        await authenticate_capability_owner(request, node_id)
        _no_store(response)
        try:
            return host().revoke(body.expected_revision, identifier)
        except (OSError, ValueError):
            raise HTTPException(
                409, "Inference client revocation changed or is unavailable."
            ) from None

    async def policy(request: Request, response: Response) -> InferenceAccessPolicy:
        """Read exact permissions without treating readiness as an authorization input."""
        await authenticate_capability_owner(request, node_id)
        _no_store(response)
        try:
            return host().policy()
        except (OSError, ValueError):
            raise HTTPException(503, "Inference policy is unavailable.") from None

    async def update_policy(
        request: Request, body: InferencePolicyUpdate, response: Response
    ) -> InferenceAccessPolicy:
        """Verify exact signed targets and replace authority under the dispatch fence."""
        await authenticate_capability_owner(request, node_id)
        _no_store(response)
        try:
            return await InferenceEnrollment(host()).update(body)
        except (OSError, ValueError):
            raise HTTPException(
                409, "Inference policy enrollment changed or is unavailable."
            ) from None

    router.get(
        "/clients",
        response_model=ClientInventory,
        summary="List inference client enrollments",
        description="Direct owner access. List active client metadata and the current revision. No bearer, bearer digest, provider secret or execution permission is returned.",
    )(clients)
    router.post(
        "/clients",
        response_model=ClientCreated,
        status_code=201,
        summary="Enroll an inference client",
        description="Direct owner access. Enroll a new client using expected_revision and return its bearer exactly once with no-store headers. New clients have no execution permissions until the owner separately enrolls exact signed destinations in the inference policy. Optional expires_unix_ms must be in the future. Stale revisions return 409.",
    )(create)
    router.delete(
        "/clients/{identifier}",
        response_model=ClientInventory,
        summary="Revoke an inference client",
        description="Direct owner access. Remove the exact client credential lifetime using expected_revision in the JSON body. The change shares the final owner dispatch fence. Stale revision or unknown identity returns 409. Historical application state cannot recreate this credential.",
    )(revoke)
    router.get(
        "/policy",
        response_model=InferenceAccessPolicy,
        summary="Read inference caller policy",
        description="Direct owner access. Return explicit per-caller signed destinations, effects, domains and optional duration ceilings. An absent policy is disabled. Read-runtime enrollment and provider login do not grant inference authority.",
    )(policy)
    router.put(
        "/policy",
        response_model=InferenceAccessPolicy,
        summary="Update inference caller policy",
        description="Direct owner access. Atomically replace policy using expected_revision. Enabling requires current callers and exact selected signed runtime5 declarations; discovery or runtime readiness supplies no permission. Read and inference effects and local/private_site/external domains are distinct. Owner maximum_seconds is optional, with null adding no duration ceiling. Stale or unqualified enrollment returns 409. This management endpoint does not enable a not-yet-qualified invocation path.",
    )(update_policy)

    return router
