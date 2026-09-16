"""Live local manager inventory shared by plugin configuration and Fabric lookup."""

import asyncio
import contextlib
import json
import os
import shutil
import signal
import time
from pathlib import Path
from typing import Literal, final

import psutil
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from skulk.extensions.managed import ManagedConnection, ManagedOwner
from skulk.extensions.managed_attachment import ManagedAttachment
from skulk.extensions.runtime_artifacts import Digest, ProtocolUnsupportedError
from skulk.extensions.runtime_attachment import (
    InstallationIdentifier,
    ProfileIdentifier,
    ServiceConnection,
)
from skulk.extensions.runtime_files import read_private
from skulk.extensions.runtime_manager import (
    InstallationRequest,
    InstallRecoveryRequest,
    InstallSubmission,
    InventoryRequest,
    ManagerBuildMismatchError,
    OperationRequest,
    ReleaseRequest,
    ReloadRuntimeRequest,
    SourceRegistration,
    SubmitRequest,
    manager_request,
)
from skulk.extensions.runtime_service import RuntimeServiceStatus
from skulk.extensions.service_setup import record_refreshed_snapshot
from skulk.extensions.service_snapshot import (
    ServiceSnapshot,
    activate_service_runtime,
    selected_generation,
    stage_service_runtime,
)
from skulk.extensions.types import ExtensionContext

_WINDOW = TypeAdapter(list[int])


def protocol_refusal(result: dict[str, JsonValue]) -> ProtocolUnsupportedError | None:
    """Rebuild the typed refusal from the manager's fixed vocabulary, or nothing."""
    if result.get("error") != "release_protocol_unsupported":
        return None
    kind, offered = result.get("kind"), result.get("offered")
    try:
        accepted = _WINDOW.validate_python(result.get("accepted"), strict=True)
    except ValueError:
        return None
    if (
        kind not in ("release", "runtime")
        or not isinstance(offered, int)
        or isinstance(offered, bool)
    ):
        return None
    return ProtocolUnsupportedError(str(kind), offered, tuple(accepted))


# A stopping manager allows each supervised owner thirty seconds to exit and
# closes them together; this covers that plus its client drain, with margin.
_MANAGER_STOP_SECONDS = 120.0


def _selected(root: Path, generation: str) -> bool:
    return selected_generation(root) == generation


def manager_pids(root: Path) -> list[int]:
    """Processes serving the manager at ``root``; they run as this user."""
    found: list[int] = []
    arguments = TypeAdapter(list[str])
    for process in psutil.process_iter():
        try:
            cmdline = arguments.validate_python(process.cmdline())
        except (psutil.Error, ValueError):
            continue
        if (
            "skulk.extensions.runtime_manager" in cmdline
            and "serve" in cmdline
            and str(root) in cmdline
        ):
            found.append(int(process.pid))
    return found


def reload_legacy_manager(root: Path, snapshot: ServiceSnapshot) -> None:
    """Select a staged generation for a manager that predates reload_runtime.

    The manager holds its fence while it runs, so it is stopped first (it is
    this user's process, no elevation), the generation is selected under both
    fences, and the OS service's keep-alive starts the manager on it. The
    stopped manager keeps its fence until it has closed every owner it
    supervises, which takes up to ``_MANAGER_STOP_SECONDS``; the selection
    waits for that exit rather than for a shorter deadline, or the keep-alive
    would restart the old generation and every attempt would fail the same
    way. The restart races the selection; the fences settle it, with bounded
    retries.
    """
    pids = manager_pids(root)
    for pid in pids:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)
    exited = time.monotonic() + _MANAGER_STOP_SECONDS
    while any(pid in manager_pids(root) for pid in pids):
        if time.monotonic() >= exited:
            raise OSError("the stopped plugin manager has not exited")
        time.sleep(0.5)
    deadline = time.monotonic() + 15
    while True:
        try:
            activate_service_runtime(root, snapshot)
            return
        except (OSError, ValueError):
            # The fence is still held while the stopped manager finishes, or
            # its keep-alive replacement took it first: keep asking.
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.25)
            for pid in manager_pids(root):
                if pid not in pids and _selected(root, snapshot.generation):
                    return
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGTERM)


def staged_generation_for(root: Path, build: str) -> ServiceSnapshot | None:
    """A generation already staged from ``build`` that is not the selected one."""
    try:
        selected = read_private(root / "core-runtime.json", 4096)
    except (OSError, ValueError):
        selected = b""
    generations = root / "core-runtimes"
    try:
        names = sorted(path.name for path in generations.iterdir() if path.is_dir())
    except OSError:
        return None
    for name in names:
        if name.encode() in selected:
            continue
        try:
            snapshot = ServiceSnapshot.model_validate_json(
                read_private(generations / name / "staged.json", 131072)
            )
        except (OSError, ValueError):
            continue
        if snapshot.generation == name and snapshot.skulk_build_sha256 == build:
            return snapshot
    return None


class ManagedInstallation(BaseModel):
    """Bounded local desired state and process observation, with no paths or secrets."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    plugin_id: InstallationIdentifier = Field(
        description="Stable local installed-plugin identifier."
    )
    error_code: (
        Literal[
            "initializing",
            "installation_unavailable",
            "service_unavailable",
            "attachment_recovery_required",
        ]
        | None
    ) = Field(default=None, description="Sanitized local installation failure class.")
    operation_id: ProfileIdentifier | None = Field(
        default=None,
        description="Pending or selected local operation, for reconnect without resubmission.",
    )
    operation_state: (
        Literal["accepted", "applying", "complete", "failed", "recovery_required"]
        | None
    ) = Field(
        default=None, description="Current status of the referenced local operation."
    )
    selected_digest: Digest | None = Field(
        default=None, description="Desired signed runtime generation."
    )
    selection_revision: int = Field(
        default=0,
        ge=0,
        description="Revision of the selected runtime, or zero before selection.",
    )
    enabled: bool = Field(
        default=False, description="Whether the selected runtime is enabled."
    )
    uninstalled: bool = Field(
        default=False,
        description="Published uninstall intent; retained state remains available for cleanup and explicit reinstallation.",
    )
    service: RuntimeServiceStatus | None = Field(
        default=None,
        description="Observed process health, independent of capability readiness.",
    )
    stale: bool = Field(
        description="Whether the process observation is absent or stale."
    )


class ManagedInventory(BaseModel):
    """Manager inventory available even when every plugin runtime is disabled or broken."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    installations: tuple[ManagedInstallation, ...] = Field(
        max_length=16, description="Bounded registered plugin installations."
    )
    reload_runtime: bool = Field(
        default=False,
        description="Whether this manager can select a staged runtime and restart on request.",
    )


type ManagementRequest = (
    InventoryRequest
    | InstallationRequest
    | SubmitRequest
    | OperationRequest
    | ReleaseRequest
    | InstallSubmission
    | SourceRegistration
    | InstallRecoveryRequest
)


@final
class ManagedServices:
    """Observe locally installed services without reloading existing extensions.

    Missing setup is inert. A later locally generated connection starts observation
    automatically. API shutdown withdraws its adapters and releases attachment;
    the OS continues to own plugin controllers and independent cleanup.
    """

    def __init__(
        self,
        connection_path: Path,
        *,
        disabled: bool = False,
        existing_owners: tuple[ManagedOwner, ...] = (),
    ) -> None:
        """Observe one fixed protected local connection, never an HTTP-supplied path."""
        self.path = connection_path
        self.disabled = disabled
        self.existing_owners = existing_owners
        self.context: ExtensionContext | None = None
        self.connection: ServiceConnection | None = None
        self.attachment: ManagedAttachment | None = None
        self.owners: dict[str, ManagedOwner] = {}
        self.task: asyncio.Task[None] | None = None
        self.runtime_refresh: asyncio.Task[None] | None = None
        self.runtime_refreshed = 0.0
        self.guard = asyncio.Lock()
        self.closed = False

    def on_start(self, context: ExtensionContext) -> None:
        """Start nonblocking discovery, including when local setup does not yet exist."""
        if self.task is not None or self.closed:
            return
        self.context = context
        self.task = asyncio.create_task(self._poll())

    async def _connect(self) -> None:
        if self.closed or self.context is None:
            raise ValueError("managed service observation is not running")
        connection = ServiceConnection.model_validate_json(
            read_private(self.path, 8192)
        )
        if connection != self.connection:
            await self._detach()
            self.connection = connection
            self.attachment = next(
                (
                    owner.attachment
                    for owner in self.existing_owners
                    if owner.attachment is not None
                    and owner.attachment.root == Path(connection.manager_root)
                    and owner.attachment.profile_id == connection.profile_id
                ),
                None,
            )
            if self.attachment is None:
                self.attachment = ManagedAttachment(
                    Path(connection.manager_root), connection.profile_id
                )
            self.attachment.retain(str(self.context.node_id))
        assert self.attachment is not None
        await self.attachment.ensure()

    def _schedule_runtime_refresh(self, differs: ManagerBuildMismatchError) -> None:
        if self.runtime_refresh is not None and not self.runtime_refresh.done():
            return
        if time.monotonic() - self.runtime_refreshed < 300:
            return
        self.runtime_refreshed = time.monotonic()
        assert self.connection is not None
        root = Path(self.connection.manager_root)
        logger.warning(
            "plugin manager runs Skulk build "
            f"{differs.manager[:12]} while this host runs {differs.live[:12]}; "
            "staging a matching manager runtime"
        )
        self.runtime_refresh = asyncio.create_task(
            self._refresh_manager_runtime(root, differs.live)
        )

    async def _settle_runtime_refresh(self) -> None:
        """Wait for a manager refresh to finish before this observer detaches.

        The refresh runs its selection in a worker thread that cancellation
        would not stop, so the task is drained rather than cancelled: a
        refresh must not keep selecting or restarting a manager this host has
        let go of. The refresh already handles its own failures.
        """
        task, self.runtime_refresh = self.runtime_refresh, None
        if task is not None and not task.done():
            await asyncio.gather(task, return_exceptions=True)

    async def _refresh_manager_runtime(self, root: Path, live: str) -> None:
        """Stage the manager runtime from this host's build and ask for a reload.

        A generation already staged for this build is reused, and one the
        manager refuses is removed, so repeated attempts against a broken
        manager do not accumulate copies of Skulk on the service volume.
        """
        candidate: Path | None = None
        try:
            # The inventory answers without an attachment and names whether
            # the manager knows the reload request, so a manager from before
            # it is told apart from one that refuses the staged generation.
            inventory = (await manager_request(root, InventoryRequest())).get("result")
            reloads = (
                isinstance(inventory, dict) and inventory.get("reload_runtime") is True
            )
            snapshot = staged_generation_for(root, live)
            if snapshot is None:
                snapshot = await stage_service_runtime(root)
            candidate = root / "core-runtimes" / snapshot.generation
            if reloads:
                reply = await manager_request(
                    root,
                    ReloadRuntimeRequest(
                        generation=snapshot.generation,
                        manifest_sha256=snapshot.manifest_sha256,
                    ),
                )
                if "result" not in reply:
                    raise ValueError("manager refused the staged generation")
            else:
                # A manager from before this protocol: select the generation
                # for it and restart it.
                await asyncio.to_thread(reload_legacy_manager, root, snapshot)
            # Setup state names the generation the service runs on; status
            # verifies against it and a setup rerun would otherwise stage
            # and restart yet another generation for the same build.
            record_refreshed_snapshot(root, snapshot)
            logger.info(
                "plugin manager runtime refreshed to this host's Skulk build; "
                "the service restarts on it"
            )
        except ValueError as error:
            # An explicit refusal: the candidate generation is not selected
            # and can go, whether it was staged now or reused. Transport
            # failures are ambiguous (the manager drains an activation past
            # the request deadline), so those keep it.
            if candidate is not None and not _selected(root, candidate.name):
                shutil.rmtree(candidate, ignore_errors=True)
            logger.warning(
                "plugin manager runtime refresh failed: "
                f"{type(error).__name__}; rerun skulk-plugin-service setup"
            )
        except (OSError, TimeoutError) as error:
            logger.warning(
                "plugin manager runtime refresh is indeterminate: "
                f"{type(error).__name__}; the staged generation is retained"
            )

    async def request(self, request: ManagementRequest) -> dict[str, JsonValue]:
        """Send one typed local management request; never accept an attachment override.

        Local setup determines the service root and profile. The original operation
        identifier is preserved; a lost mutation response is never replayed here.
        """
        async with self.guard:
            await self._connect()
            assert self.connection is not None
            root = Path(self.connection.manager_root)
        # Source HTTPS and durable operations must not monopolize membership
        # observation. The fixed connection is captured before this request starts.
        result = await manager_request(root, request)
        payload = result.get("result")
        if set(result) != {"result"} or not isinstance(payload, dict):
            refused = protocol_refusal(result)
            if refused is not None:
                raise refused
            raise ValueError("managed service request refused")
        return payload

    async def refresh(self) -> ManagedInventory:
        """Reconcile current manager membership without restarting Skulk's API."""
        async with self.guard:
            try:
                try:
                    await self._connect()
                except ManagerBuildMismatchError as differs:
                    # A Skulk update restarted this host on a build the manager
                    # does not run. Stage a matching manager runtime from this
                    # process and ask the manager to reload; the OS service
                    # restarts it and the next refresh attaches.
                    self._schedule_runtime_refresh(differs)
                    raise
                assert self.connection is not None and self.context is not None
                assert self.attachment is not None
                result = await manager_request(
                    Path(self.connection.manager_root), InventoryRequest()
                )
                if set(result) != {"result"}:
                    raise ValueError("managed service inventory refused")
                inventory = ManagedInventory.model_validate_json(
                    json.dumps(result["result"])
                )
                identifiers = [item.plugin_id for item in inventory.installations]
                if len(set(identifiers)) != len(identifiers):
                    raise ValueError("ambiguous managed installation inventory")
                for identifier in set(self.owners) - set(identifiers):
                    owner = self.owners.pop(identifier)
                    await owner.on_stop()
                for item in inventory.installations:
                    identifier = item.plugin_id
                    if identifier not in self.owners:
                        expected_root = (
                            Path(self.connection.manager_root)
                            / "installations"
                            / identifier
                        )
                        owner = next(
                            (
                                existing
                                for existing in self.existing_owners
                                if existing.name == identifier
                                and existing.root == expected_root
                                and existing.attachment is self.attachment
                            ),
                            None,
                        )
                        if owner is None:
                            owner = ManagedOwner(
                                ManagedConnection(
                                    plugin_id=identifier,
                                    state_root=str(expected_root),
                                    manager_root=self.connection.manager_root,
                                    profile_id=self.connection.profile_id,
                                ),
                                disabled=self.disabled,
                                attachment=self.attachment,
                            )
                        self.owners[identifier] = owner
                        owner.on_start(self.context)
                    # Error summaries omit selection fields and default enabled
                    # to false. Only an observed selection proves owner withdrawal.
                    self.owners[identifier].manager_enabled = (
                        item.enabled
                        if item.selected_digest is not None and item.error_code is None
                        else None
                    )
                    self.owners[identifier].manager_available = (
                        item.enabled
                        and not item.stale
                        and item.error_code is None
                        and item.service is not None
                        and item.service.state == "running"
                        and item.service.active_digest == item.selected_digest
                    )
                return inventory
            except (OSError, ValueError, TimeoutError):
                # Keep unavailable installations visible for configuration, but
                # synchronously withdraw admission while manager state is unknown.
                for owner in self.owners.values():
                    owner.available = False
                    owner.manager_available = False
                    owner.manager_enabled = None
                raise

    async def _poll(self) -> None:
        while True:
            with contextlib.suppress(OSError, ValueError, TimeoutError):
                await self.refresh()
            await asyncio.sleep(1)

    async def _detach(self) -> None:
        await self._settle_runtime_refresh()
        owners, self.owners = tuple(self.owners.values()), {}
        await asyncio.gather(*(owner.on_stop() for owner in owners))
        if self.attachment is not None:
            await self.attachment.release()
            self.attachment = None
        self.connection = None

    async def on_stop(self) -> None:
        """Stop observation and local API attachment without stopping managed services."""
        if self.closed:
            return
        self.closed = True
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None
        await self._settle_runtime_refresh()
        async with self.guard:
            await self._detach()
