# pyright: reportUnusedFunction=false
"""Owner-authorized, schema-driven configuration of installed capability nodes."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

import anyio
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import Field, JsonValue

from skulk.api.managed_plugins import create_managed_plugins_router
from skulk.api.operator_auth import TailnetPeerVerifier, authorize_plugin_request
from skulk.connectivity.tailscale import is_tailscale_peer
from skulk.extensions.configuration import (
    ConfigurableNode,
    ConfigurationMutation,
    ConfigurationResult,
    NodeConfiguration,
    NodeConfigurationProvider,
)
from skulk.extensions.loader import LoadedExtensions
from skulk.operator.pairing import OperatorPairingService
from skulk.utils.pydantic_ext import FrozenModel

_Result = TypeVar("_Result")


class PluginNodes(FrozenModel):
    """Installed nodes exposed by one plugin, including management availability."""

    plugin_id: str = Field(description="Local installed extension identifier.")
    nodes: tuple[ConfigurableNode, ...] = Field(
        description="Persistent installed nodes, including disabled nodes."
    )
    available: bool = Field(
        description="Whether the management provider answered this read."
    )


def _public_configuration(value: NodeConfiguration, node_id: str) -> NodeConfiguration:
    """Bound ordinary responses and reject credentials misdeclared as settings."""
    if value.node_id != node_id or len(value.model_dump_json().encode()) > 131072:
        raise ValueError("invalid configuration response")
    pending: list[JsonValue] = [value.configuration_schema]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            # Secret replacement gets its own write-only interface. Refuse a
            # mixed schema rather than relying on a renderer to hide its values.
            if item.get("writeOnly") is True or item.get("format") == "password":
                raise ValueError("credentials cannot be returned as ordinary settings")
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return value


def create_plugins_router(
    extensions: LoadedExtensions,
    pairing_service: OperatorPairingService | None,
    *,
    tailnet_peer_verifier: TailnetPeerVerifier = is_tailscale_peer,
) -> APIRouter:
    """Expose ordinary configuration through optional plugin management facets.

    Management is intentionally independent of capability readiness. Providers
    validate their own settings; the API owns authorization, bounded dispatch
    and payload-safe failure responses. No cloud effect or executable selection
    is accepted here.
    """
    router = APIRouter(prefix="/v1/plugins", tags=["Plugins"])
    router.include_router(
        create_managed_plugins_router(
            extensions, pairing_service, tailnet_peer_verifier
        )
    )
    capacity = anyio.CapacityLimiter(8)

    async def invoke(call: Callable[[], Awaitable[_Result]]) -> _Result:
        try:
            capacity.acquire_nowait()
        except anyio.WouldBlock:
            raise HTTPException(
                status_code=429, detail="plugin management is busy"
            ) from None
        try:
            async with asyncio.timeout(30):
                return await call()
        except LookupError:
            raise HTTPException(
                status_code=404, detail="installed node not found"
            ) from None
        except ValueError:
            raise HTTPException(
                status_code=409,
                detail="configuration refused; reload settings and validation",
            ) from None
        except TimeoutError:
            raise HTTPException(
                status_code=504, detail="plugin management timed out"
            ) from None
        except Exception:  # noqa: BLE001 - provider failures must not expose payloads
            raise HTTPException(
                status_code=503, detail="plugin management unavailable"
            ) from None
        finally:
            capacity.release()

    def provider(plugin_id: str) -> NodeConfigurationProvider:
        selected = extensions.configuration_providers.get(plugin_id)
        if selected is None:
            raise HTTPException(
                status_code=404, detail="configuration provider not found"
            )
        return selected

    @router.get(
        "",
        response_model=tuple[PluginNodes, ...],
        summary="List configurable plugin nodes",
        description=(
            "List installed capability nodes and management availability without "
            "filtering disabled or failed children. Requires direct dashboard owner "
            "authority or the explicit plugins:read paired-operator scope."
        ),
    )
    async def list_nodes(
        request: Request, response: Response
    ) -> tuple[PluginNodes, ...]:
        """Read bounded installed-node inventories without starting children."""
        await authorize_plugin_request(
            request, pairing_service, "plugins:read", tailnet_peer_verifier
        )
        response.headers["Cache-Control"] = "no-store"

        async def read(
            plugin_id: str, selected: NodeConfigurationProvider
        ) -> PluginNodes:
            try:
                async with asyncio.timeout(5):
                    nodes = await selected.configuration_nodes()
                result = PluginNodes(plugin_id=plugin_id, nodes=nodes, available=True)
                if len(nodes) > 128 or len(result.model_dump_json().encode()) > 131072:
                    raise ValueError("inventory exceeds bound")
                if len({node.node_id for node in nodes}) != len(nodes):
                    raise ValueError("ambiguous node identity")
                return result
            except Exception:  # noqa: BLE001 - inventory errors contain no provider text
                return PluginNodes(plugin_id=plugin_id, nodes=(), available=False)

        async def collect() -> tuple[PluginNodes, ...]:
            providers = extensions.configuration_providers
            if len(providers) > 32:
                raise ValueError("too many management providers")
            return tuple(
                await asyncio.gather(
                    *(read(key, value) for key, value in providers.items())
                )
            )

        return await invoke(collect)

    @router.get(
        "/{plugin_id}/nodes/{node_id}/configuration",
        response_model=NodeConfiguration,
        summary="Read a capability node's configuration",
        description=(
            "Return the plugin-provided ordinary-settings schema, values, enabled "
            "state and revision for one persistent installed node. Multiple "
            "capabilities may share this node configuration. Secret values are excluded. "
            "Requires direct owner authority or plugins:read."
        ),
    )
    async def get_configuration(
        plugin_id: str, node_id: str, request: Request, response: Response
    ) -> NodeConfiguration:
        """Return the configuration of this exact installed node."""
        await authorize_plugin_request(
            request, pairing_service, "plugins:read", tailnet_peer_verifier
        )
        response.headers["Cache-Control"] = "no-store"
        selected = provider(plugin_id)

        async def read() -> NodeConfiguration:
            return _public_configuration(
                await selected.node_configuration(node_id), node_id
            )

        return await invoke(read)

    @router.post(
        "/{plugin_id}/nodes/{node_id}/configuration",
        response_model=ConfigurationResult,
        summary="Validate or change capability-node configuration",
        description=(
            "Validate, edit, enable or disable one node using the exact observed "
            "configuration revision and schema digest. The plugin owns validation "
            "and preflight. This route cannot approve paid effects. Requires direct "
            "owner authority or plugins:manage; conflicts require a fresh read."
        ),
    )
    async def change_configuration(
        plugin_id: str,
        node_id: str,
        mutation: ConfigurationMutation,
        request: Request,
        response: Response,
    ) -> ConfigurationResult:
        """Dispatch one revision-fenced ordinary-settings action to its owner."""
        await authorize_plugin_request(
            request, pairing_service, "plugins:manage", tailnet_peer_verifier
        )
        response.headers["Cache-Control"] = "no-store"
        selected = provider(plugin_id)
        if (mutation.operation in {"validate", "edit"}) != (
            mutation.values is not None
        ):
            raise HTTPException(
                status_code=422, detail="values are required only for validate/edit"
            )

        async def change() -> ConfigurationResult:
            result = await selected.configure_node(node_id, mutation)
            _public_configuration(result.configuration, node_id)
            return result

        return await invoke(change)

    return router
