"""Node-driven setup of the plugin manager service.

A Skulk node sets up its own plugin manager the first time capabilities are
used, so no terminal, command, or administrator password is involved: the
manager registers as a per-user launchd agent or systemd user unit, with the
same lifetime as the node itself. The node runs at most one setup at a time
and reports its progress and outcome. A host that already runs the manager as
a system service keeps it; repairing one of those still takes the terminal
command, because registering a system service needs local elevation.
"""

import asyncio
from pathlib import Path
from typing import Literal, final

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from skulk.extensions.runtime_attachment import ServiceConnection
from skulk.extensions.runtime_files import read_private
from skulk.extensions.runtime_manager import InventoryRequest, manager_request
from skulk.extensions.service_registration import ServiceScope
from skulk.extensions.service_setup import (
    ServiceReadinessPendingError,
    connected_scope,
    setup_service,
)
from skulk.shared.constants import SKULK_CONFIG_HOME

PluginServiceState = Literal[
    "absent", "setting_up", "ready", "unavailable", "failed", "unsupported"
]


class PluginServiceStatus(BaseModel):
    """Whether this node's plugin manager is set up and answering."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    state: PluginServiceState = Field(
        description=(
            "absent: never set up; setting_up: setup is running; ready: the "
            "manager answers; unavailable: set up but not answering; failed: "
            "the last setup failed (see error); unsupported: this host cannot "
            "run the manager (see error)."
        )
    )
    scope: ServiceScope | None = Field(
        description="user (a per-user service) or system, once set up."
    )
    progress: str | None = Field(
        description="The latest setup step while setup is running."
    )
    error: str | None = Field(
        description="Why the last setup failed or the host is unsupported."
    )


_SYSTEM_REPAIR = (
    "This host runs the plugin service as a system service; repair it from a "
    "terminal with: skulk-plugin-service setup --system"
)


async def _manager_answers(root: Path) -> bool:
    try:
        result = await manager_request(root, InventoryRequest())
    except (OSError, ValueError, TimeoutError):
        return False
    return "result" in result and "error" not in result


@final
class ServiceSetupRunner:
    """Run at most one manager setup at a time and remember how it ended."""

    def __init__(self) -> None:
        """Start with no setup running and no recorded failure."""
        self._task: asyncio.Task[None] | None = None
        self._progress: str | None = None
        self._error: str | None = None

    def start(self) -> None:
        """Begin setup in the background unless one is already running.

        A host connected to a system service is not set up again here: that
        needs elevation, which a node never requests on its own.
        """
        if self._task is not None and not self._task.done():
            return
        try:
            scope = connected_scope()
        except ValueError as unsupported:
            self._error = str(unsupported)
            return
        if scope == "system":
            self._error = _SYSTEM_REPAIR
            return
        self._error = None
        self._progress = "Starting plugin setup..."
        self._task = asyncio.create_task(self._run())

    def _report(self, message: str) -> None:
        self._progress = message
        logger.info(f"plugin service setup: {message}")

    async def _run(self) -> None:
        try:
            await setup_service("user", self._report)
        except ServiceReadinessPendingError:
            self._error = (
                "The plugin service was registered but has not answered yet; "
                "it may still be starting."
            )
        except Exception as error:  # noqa: BLE001
            # A background setup has no caller to raise to; the failure must
            # reach the operator through the status instead of vanishing.
            logger.opt(exception=error).warning("plugin service setup failed")
            self._error = (
                str(error) if isinstance(error, ValueError) else "Plugin setup failed."
            )
        finally:
            self._progress = None

    async def status(self) -> PluginServiceStatus:
        """Report setup progress, the outcome of the last setup, or readiness."""
        if self._task is not None and not self._task.done():
            return PluginServiceStatus(
                state="setting_up", scope="user", progress=self._progress, error=None
            )
        try:
            scope = connected_scope()
        except ValueError as unsupported:
            return PluginServiceStatus(
                state="unsupported", scope=None, progress=None, error=str(unsupported)
            )
        if scope is None:
            return PluginServiceStatus(
                state="failed" if self._error else "absent",
                scope=None,
                progress=None,
                error=self._error,
            )
        connection = ServiceConnection.model_validate_json(
            read_private(SKULK_CONFIG_HOME / "managed-service" / "connection.json")
        )
        if await _manager_answers(Path(connection.manager_root)):
            return PluginServiceStatus(
                state="ready", scope=scope, progress=None, error=None
            )
        return PluginServiceStatus(
            state="failed" if self._error else "unavailable",
            scope=scope,
            progress=None,
            error=self._error,
        )
