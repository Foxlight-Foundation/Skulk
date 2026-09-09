# pyright: reportUnusedFunction=false
"""Explicitly scoped HTTP access to the independently supervised local manager."""

import asyncio
import json
from collections.abc import Awaitable, Callable

import anyio
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from skulk.api.operator_auth import TailnetPeerVerifier, authorize_plugin_request
from skulk.extensions.loader import LoadedExtensions
from skulk.extensions.managed_services import (
    ManagedInstallation,
    ManagedInventory,
    ManagedServices,
)
from skulk.extensions.runtime_attachment import (
    InstallationIdentifier,
    ProfileIdentifier,
)
from skulk.extensions.runtime_controller import LifecycleOperation, LifecycleRequest
from skulk.extensions.runtime_manager import (
    InstallationRequest,
    OperationRequest,
    SubmitRequest,
)
from skulk.extensions.runtime_selection import RuntimeSelection
from skulk.operator.pairing import OperatorPairingService
from skulk.operator.plugin_scopes import PluginScope


class RegistrationRequest(BaseModel):
    """Create only an empty local installation; no executable or service-path choice."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    plugin_id: InstallationIdentifier = Field(
        description="Stable managed plugin installation ID."
    )


class ManagedSelection(BaseModel):
    """Current local installation and desired signed runtime, separate from readiness."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    installation: ManagedInstallation = Field(
        description="Current bounded service observation."
    )
    selection: RuntimeSelection | None = Field(
        description="Desired generation, absent until selected."
    )


def create_managed_plugins_router(
    extensions: LoadedExtensions,
    pairing_service: OperatorPairingService | None,
    tailnet_peer_verifier: TailnetPeerVerifier,
) -> APIRouter:
    """Bind fixed local management verbs to explicit plugin grants and owner origin checks."""
    router = APIRouter(prefix="/managed", tags=["Plugins"])
    capacity = anyio.CapacityLimiter(8)

    async def authorized(
        request: Request, response: Response, scope: PluginScope
    ) -> ManagedServices:
        await authorize_plugin_request(
            request, pairing_service, scope, tailnet_peer_verifier
        )
        response.headers["Cache-Control"] = "no-store"
        if extensions.managed_services is None:
            raise HTTPException(
                status_code=503, detail="local plugin service setup is unavailable"
            )
        return extensions.managed_services

    async def invoke[Result](call: Callable[[], Awaitable[Result]]) -> Result:
        try:
            capacity.acquire_nowait()
        except anyio.WouldBlock:
            raise HTTPException(
                status_code=429, detail="plugin management is busy"
            ) from None
        try:
            async with asyncio.timeout(30):
                return await call()
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="local plugin service setup or operation is unavailable",
            ) from None
        except TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="plugin manager response timed out; read the original operation status",
            ) from None
        except (OSError, RuntimeError):
            raise HTTPException(
                status_code=503, detail="local plugin manager is unavailable"
            ) from None
        except ValueError:
            raise HTTPException(
                status_code=409,
                detail="local plugin operation refused; refresh selection and operation status",
            ) from None
        finally:
            capacity.release()

    @router.get(
        "",
        response_model=ManagedInventory,
        summary="Read local managed plugin installations",
        description="Read installed plugin selections and stale/process health independently of child availability. Requires plugins:read or direct owner authority. Local setup supplies the manager connection; no path is accepted.",
    )
    async def inventory(request: Request, response: Response) -> ManagedInventory:
        """Read the live local manager and reconcile cached plugin membership."""
        services = await authorized(request, response, "plugins:read")
        return await invoke(services.refresh)

    @router.post(
        "/installations",
        response_model=ManagedInstallation,
        summary="Register an empty managed plugin installation",
        description="Register one stable managed plugin ID using the locally generated host binding. Requires plugins:manage or direct owner authority. Accepts no artifact, path, command, credential or paid approval; does not enable a runtime.",
    )
    async def register(
        body: RegistrationRequest, request: Request, response: Response
    ) -> ManagedInstallation:
        """Create the manager-owned installation through the same terminal operation."""
        services = await authorized(request, response, "plugins:manage")

        async def action() -> ManagedInstallation:
            result = await services.request(
                InstallationRequest(action="register", plugin_id=body.plugin_id)
            )
            return ManagedInstallation.model_validate_json(json.dumps(result))

        return await invoke(action)

    @router.get(
        "/installations/{plugin_id}",
        response_model=ManagedSelection,
        summary="Read a managed plugin's selected runtime",
        description="Read exact desired runtime selection and current local process observation. Requires plugins:read or direct owner authority. Disabled selections retain their state and records.",
    )
    async def selection(
        plugin_id: InstallationIdentifier, request: Request, response: Response
    ) -> ManagedSelection:
        """Inspect one exact registered installation, without activating it."""
        services = await authorized(request, response, "plugins:read")

        async def action() -> ManagedSelection:
            result = await services.request(
                InstallationRequest(action="get", plugin_id=plugin_id)
            )
            return ManagedSelection.model_validate_json(json.dumps(result))

        return await invoke(action)

    @router.post(
        "/installations/{plugin_id}/operations",
        response_model=LifecycleOperation,
        summary="Submit a local plugin lifecycle operation",
        description="Submit revision-fenced activate or disable with a retained operation_id. Activation selects only an already staged verified runtime; explicit rollback and permission acceptance are separate flags. Requires plugins:manage or direct owner authority. Client disconnect does not abandon accepted work. This never approves spending or replays provider requests.",
    )
    async def submit(
        plugin_id: InstallationIdentifier,
        body: LifecycleRequest,
        request: Request,
        response: Response,
    ) -> LifecycleOperation:
        """Submit exact local lifecycle intent with the caller's durable operation ID."""
        services = await authorized(request, response, "plugins:manage")

        async def action() -> LifecycleOperation:
            result = await services.request(
                SubmitRequest(plugin_id=plugin_id, request=body)
            )
            return LifecycleOperation.model_validate_json(json.dumps(result))

        return await invoke(action)

    @router.get(
        "/installations/{plugin_id}/operations/{operation_id}",
        response_model=LifecycleOperation,
        summary="Read a retained local plugin operation",
        description="Read the original accepted local operation after reconnect or manager restart. Requires plugins:read or direct owner authority. Read status before resubmitting any request whose response was lost.",
    )
    async def operation(
        plugin_id: InstallationIdentifier,
        operation_id: ProfileIdentifier,
        request: Request,
        response: Response,
    ) -> LifecycleOperation:
        """Read the original local operation result without repeating its effect."""
        services = await authorized(request, response, "plugins:read")

        async def action() -> LifecycleOperation:
            result = await services.request(
                OperationRequest(plugin_id=plugin_id, operation_id=operation_id)
            )
            return LifecycleOperation.model_validate_json(json.dumps(result))

        return await invoke(action)

    @router.post(
        "/installations/{plugin_id}/operations/{operation_id}/recover",
        response_model=LifecycleOperation,
        summary="Recover an interrupted local plugin operation",
        description="Explicitly finish only an existing journaled local selection using the original operation ID. Completed operations are not reapplied. Requires plugins:manage or direct owner authority. Recovery never recreates a provider resource or grants spending approval.",
    )
    async def recover(
        plugin_id: InstallationIdentifier,
        operation_id: ProfileIdentifier,
        request: Request,
        response: Response,
    ) -> LifecycleOperation:
        """Resume only previously accepted local lifecycle intent."""
        services = await authorized(request, response, "plugins:manage")

        async def action() -> LifecycleOperation:
            result = await services.request(
                OperationRequest(
                    action="recover", plugin_id=plugin_id, operation_id=operation_id
                )
            )
            return LifecycleOperation.model_validate_json(json.dumps(result))

        return await invoke(action)

    return router
