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
import os
import sys
import time
from pathlib import Path
from typing import Final, Literal, final

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from skulk.extensions import service_bootstrap
from skulk.extensions.managed_services import RuntimeUpdate, RuntimeUpdateFailure
from skulk.extensions.runtime_attachment import ServiceConnection
from skulk.extensions.runtime_files import read_private
from skulk.extensions.runtime_manager import InventoryRequest, manager_request
from skulk.extensions.service_registration import ServiceScope
from skulk.extensions.service_setup import (
    ServiceReadinessPendingError,
    connected_scope,
    registered_unit_names_base,
    setup_service,
)
from skulk.shared.constants import SKULK_CONFIG_HOME

PluginServiceState = Literal[
    "absent", "setting_up", "ready", "unavailable", "failed", "unsupported"
]

PluginServicePurpose = Literal["setup", "update"]
"""What a setup, or its failure, is for: the first setup (or one the owner
started), or bringing the plugin service in line with a Skulk that was
updated or moved."""


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
    purpose: PluginServicePurpose | None = Field(
        default=None,
        description=(
            "While setting_up, and when failed: setup (the first setup, or one "
            "the owner started) or update (bringing the plugin service in line "
            "with a Skulk that was updated or moved; installed plugins and their "
            "settings are kept). Null in every other state."
        ),
    )


_EXTENSIONS_DISABLED = (
    "Plugins are turned off on this node (SKULK_EXTENSIONS_DISABLE=1)."
)


def _extensions_disabled() -> bool:
    """The node-wide extensions kill switch, read as the extension loader reads it."""
    return os.environ.get("SKULK_EXTENSIONS_DISABLE", "").strip() == "1"


_SYSTEM_REPAIR = (
    "This host runs the plugin service as a system service; repair it from a "
    "terminal with: skulk-plugin-service setup --system"
)

_UPDATING: Final = (
    "Skulk was updated. Updating the plugin service to match; plugins come "
    "back in a minute or two."
)
"""Progress while the plugin service is brought to a new Skulk build."""

_RELOCATED: Final = (
    "Skulk moved or was updated. Setting up the plugin service again to "
    "match; plugins come back in a minute or two."
)
"""Progress while a user service whose interpreter moved is registered again."""

_UPDATE_FAILURES: Final[dict[RuntimeUpdateFailure, str]] = {
    "staging_failed": (
        "Skulk was updated, but the plugin service could not be updated to "
        "match: copying this Skulk for it failed, often because the disk is "
        "nearly full."
    ),
    "refused": (
        "Skulk was updated, but the plugin service did not accept the update "
        "to match it."
    ),
    "other_interpreter": (
        "Skulk was updated, but the plugin service is registered to start a "
        "different Skulk installation."
    ),
    "not_restarted": (
        "Skulk was updated, but the plugin service has not come back on the "
        "new version."
    ),
}
"""What happened when bringing the plugin service to a new build failed."""

_UPDATE_RETRY: Final = (
    "Try again to set the plugin service up for this version; installed "
    "plugins and their settings are kept."
)
_UPDATE_RETRY_AFTER_SPACE: Final = (
    "Free some disk space, then try again; installed plugins and their "
    "settings are kept."
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
        self._repair_attempted = False
        # Why the running (or last) setup re-registers an existing service:
        # its interpreter moved, or an update replaced it in place. None for
        # a first setup or one the owner started.
        self._repair: Literal["relocated", "updated"] | None = None
        # Monotonic time the last setup ended, so a manager update failure
        # recorded before it is not reported over its outcome.
        self._finished_at: float | None = None

    def start(self) -> None:
        """Begin setup in the background unless one is already running.

        A host connected to a system service is not set up again here: that
        needs elevation, which a node never requests on its own.
        """
        self._begin(None)

    def _begin(self, repair: Literal["relocated", "updated"] | None) -> None:
        if self._task is not None and not self._task.done():
            return
        self._repair = repair
        if _extensions_disabled():
            self._error = _EXTENSIONS_DISABLED
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

    async def _repair_moved_interpreter(self, root: Path) -> bool:
        """Re-register a silent user service whose interpreter changed, once.

        Moving the packaged app changes the interpreter path the registered
        service starts, and updating it replaces the interpreter's bytes at the
        same path; either way the service's sealed runtime can no longer
        start. A user service needs no elevation to register again, so the
        node does it instead of asking the owner to.
        """
        if self._repair_attempted:
            return False
        base = Path(sys.executable).resolve(strict=True)
        same_path = await asyncio.to_thread(registered_unit_names_base, base)
        if same_path and await asyncio.to_thread(
            service_bootstrap.selected_base_matches, root, base
        ):
            return False
        self._repair_attempted = True
        logger.info("plugin service names another interpreter; setting it up again")
        self._begin("updated" if same_path else "relocated")
        return self._task is not None and not self._task.done()

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
            self._finished_at = time.monotonic()

    def _setting_up(self, update: RuntimeUpdate | None) -> PluginServiceStatus:
        """The running setup, worded for an update when it is one.

        A first setup keeps its step-by-step progress; one that brings the
        service in line with an updated or moved Skulk says so instead of
        naming registration steps the owner never asked for.
        """
        purpose: PluginServicePurpose = "update"
        if update is not None or self._repair == "updated":
            progress = _UPDATING
        elif self._repair == "relocated":
            progress = _RELOCATED
        else:
            progress, purpose = self._progress, "setup"
        return PluginServiceStatus(
            state="setting_up",
            scope="user",
            progress=progress,
            error=None,
            purpose=purpose,
        )

    @staticmethod
    def _updating(scope: ServiceScope) -> PluginServiceStatus:
        return PluginServiceStatus(
            state="setting_up",
            scope=scope,
            progress=_UPDATING,
            error=None,
            purpose="update",
        )

    @staticmethod
    def _update_failed(
        scope: ServiceScope, failure: RuntimeUpdateFailure
    ) -> PluginServiceStatus:
        """Say what went wrong with the update and the step that fixes it."""
        if scope == "system":
            # Registering a system service needs elevation the node never asks for.
            remedy = _SYSTEM_REPAIR
        elif failure == "staging_failed":
            remedy = _UPDATE_RETRY_AFTER_SPACE
        else:
            remedy = _UPDATE_RETRY
        return PluginServiceStatus(
            state="failed",
            scope=scope,
            progress=None,
            error=f"{_UPDATE_FAILURES[failure]} {remedy}",
            purpose="update",
        )

    async def status(self, update: RuntimeUpdate | None = None) -> PluginServiceStatus:
        """Report setup progress, the outcome of the last setup, or readiness.

        Args:
            update: Where bringing the manager to this host's Skulk build
                stands, from the node's manager observation; None when no
                update is pending. While a refresh attempt runs, the status
                is answered from it alone: the previous build's manager may
                still answer, and repairing the service then would race the
                restart the refresh asks for.
        """
        if self._task is not None and not self._task.done():
            return self._setting_up(update)
        if _extensions_disabled():
            return PluginServiceStatus(
                state="unsupported",
                scope=None,
                progress=None,
                error=_EXTENSIONS_DISABLED,
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
                purpose=self._failed_purpose(),
            )
        if (
            update is not None
            and update.phase == "failed"
            and self._finished_at is not None
            and self._finished_at >= update.since
        ):
            # A setup that ended after the update failed is the newer answer.
            if self._error is not None:
                # It failed too, so the update is still not done: the
                # previous build's manager answering must not read as ready.
                return PluginServiceStatus(
                    state="failed",
                    scope=scope,
                    progress=None,
                    error=self._error,
                    purpose="update",
                )
            update = None
        if update is not None and update.phase == "refreshing":
            return self._updating(scope)
        connection = ServiceConnection.model_validate_json(
            read_private(SKULK_CONFIG_HOME / "managed-service" / "connection.json")
        )
        if await _manager_answers(Path(connection.manager_root)):
            if update is None:
                return PluginServiceStatus(
                    state="ready", scope=scope, progress=None, error=None
                )
            # Answering is not attaching: the previous build's manager answers
            # too, until the refreshed one replaces it.
            if update.failure is None:
                return self._updating(scope)
            return self._update_failed(scope, update.failure)
        # A repair under a running refresh attempt would contend with it.
        if (
            scope == "user"
            and (update is None or not update.refresh_running)
            and await self._repair_moved_interpreter(Path(connection.manager_root))
        ):
            return self._setting_up(update)
        if update is not None:
            if update.failure is None:
                return self._updating(scope)
            return self._update_failed(scope, update.failure)
        return PluginServiceStatus(
            state="failed" if self._error else "unavailable",
            scope=scope,
            progress=None,
            error=self._error,
            purpose=self._failed_purpose(),
        )

    def _failed_purpose(self) -> PluginServicePurpose | None:
        """What the setup that failed was for; None when none failed."""
        if self._error is None:
            return None
        return "update" if self._repair is not None else "setup"
