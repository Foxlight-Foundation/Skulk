"""Live local manager inventory shared by plugin configuration and Fabric lookup."""

import asyncio
import contextlib
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Literal, final

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from skulk.extensions.managed import ManagedConnection, ManagedOwner
from skulk.extensions.managed_attachment import (
    ManagedAttachment,
    manager_pids,
    manager_processes,
    runs_generation,
)
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
from skulk.extensions.service_setup import (
    record_refreshed_snapshot,
    registered_unit_names_base,
    setup_state_names,
)
from skulk.extensions.service_snapshot import (
    ServiceSnapshot,
    activate_service_runtime,
    selected_generation,
    service_source_identity,
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
# The keep-alive throttles restarts to ten seconds apart and the bootstrap
# verifies the runtime tree before it starts the manager; several such
# rounds fit in this, after which the refresh is reported indeterminate.
_MANAGER_START_SECONDS = 90.0


# How long a manager that answered nothing gets to show the selection it made.
_SELECTION_WAIT_SECONDS = 120.0


def _selected(root: Path, generation: str) -> bool:
    return selected_generation(root) == generation


def _selected_within(root: Path, generation: str, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while not _selected(root, generation):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.5)
    return True


@final
class ManagerNotRestartedError(RuntimeError):
    """The selection landed but no manager started on it within the wait.

    Distinct from a transport failure: the pointer alone must not be read as
    a completed refresh, since no usable manager runs the selection yet.
    """

    def __init__(self, generation: str) -> None:
        super().__init__(
            f"the plugin manager has not started on generation {generation}"
        )
        self.generation = generation


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
            break
        except (OSError, ValueError):
            # The fence is still held while the stopped manager finishes, or
            # its keep-alive replacement took it first: keep asking.
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.25)
            for pid in manager_pids(root):
                if pid not in pids and _selected(root, snapshot.generation):
                    break
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGTERM)
            else:
                continue
            break
    # The keep-alive may have started a manager between the stop and the
    # selection; it read the previous pointer and runs the old generation,
    # or is still in its bootstrap and not yet visible as a manager at all.
    # Stop each such manager as it appears until one runs the selected
    # generation: only then has the service restarted on it.
    start_deadline = time.monotonic() + _MANAGER_START_SECONDS
    while True:
        processes = manager_processes(root)
        for pid, cmdline in processes:
            if not runs_generation(cmdline, root, snapshot.generation):
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGTERM)
        if any(
            runs_generation(cmdline, root, snapshot.generation)
            for _, cmdline in processes
        ):
            return
        if time.monotonic() >= start_deadline:
            raise ManagerNotRestartedError(snapshot.generation)
        time.sleep(0.5)


def selected_generation_for(root: Path, build: str) -> ServiceSnapshot | None:
    """The selected generation when it was staged from ``build``.

    A previous attempt may have selected a matching generation while the
    manager kept running the old one; nothing new is staged for that.
    """
    generation = selected_generation(root)
    if generation is None:
        return None
    try:
        snapshot = ServiceSnapshot.model_validate_json(
            read_private(root / "core-runtimes" / generation / "staged.json", 131072)
        )
    except (OSError, ValueError):
        return None
    if snapshot.generation == generation and snapshot.skulk_build_sha256 == build:
        return snapshot
    return None


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
        # None until an attempt is made: the monotonic clock starts near zero
        # at boot, so a zero origin would hold the first attempt for five
        # minutes on a host that starts Skulk right after booting.
        self.runtime_refreshed: float | None = None
        # Staging retains an interrupted copy and never activates it; an
        # environment that cannot be staged would grow one such copy every
        # attempt, so after a staging failure no further attempt is made until
        # the host restarts (which is when its environment changes).
        self.runtime_refresh_exhausted = False
        # The selected generation the setup state was last reconciled with,
        # so an attached manager costs no file reads on every poll.
        self.setup_reconciled: str | None = None
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
        if self.runtime_refresh_exhausted:
            return
        if (
            self.runtime_refreshed is not None
            and time.monotonic() - self.runtime_refreshed < 300
        ):
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
        snapshot: ServiceSnapshot | None = None
        try:
            # The inventory answers without an attachment and names whether
            # the manager knows the reload request, so a manager from before
            # it is told apart from one that refuses the staged generation.
            inventory = (await manager_request(root, InventoryRequest())).get("result")
            reloads = (
                isinstance(inventory, dict) and inventory.get("reload_runtime") is True
            )
            if not reloads and not await asyncio.to_thread(
                registered_unit_names_base, Path(sys.executable).resolve(strict=True)
            ):
                # The legacy path selects a generation sealed to this
                # interpreter for the OS service to start; a service
                # registered to invoke another one could not come back, and
                # only an elevated setup re-registers it. Nothing is staged.
                raise ValueError(
                    "the registered service invokes another interpreter; "
                    "rerun skulk-plugin-service setup"
                )
            snapshot = selected_generation_for(root, live) or staged_generation_for(
                root, live
            )
            if snapshot is None:
                try:
                    snapshot = await stage_service_runtime(root)
                except (OSError, TimeoutError, ValueError):
                    # Qualification past its deadline has populated the
                    # generation directory as surely as a failed copy has.
                    self.runtime_refresh_exhausted = True
                    raise ValueError(
                        "the manager runtime could not be staged from this "
                        "environment; no further attempt until the host restarts"
                    ) from None
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
                    # The manager's own handler deadline answers with the
                    # generic refusal while its selection may still finish;
                    # a named refusal is final.
                    if reply.get(
                        "error"
                    ) == "manager_operation_refused" and await asyncio.to_thread(
                        _selected_within,
                        root,
                        snapshot.generation,
                        _SELECTION_WAIT_SECONDS,
                    ):
                        await self._record_refresh(root, snapshot)
                        return
                    raise ValueError("manager refused the staged generation")
            else:
                # A manager from before this protocol: select the generation
                # for it and restart it.
                await asyncio.to_thread(reload_legacy_manager, root, snapshot)
            await self._record_refresh(root, snapshot)
        except ValueError as error:
            # An explicit refusal (or a service this host cannot restart): the
            # candidate generation is not selected and can go, whether it was
            # staged now or reused. Transport
            # failures are ambiguous (the manager drains an activation past
            # the request deadline), so those keep it.
            if candidate is not None and not _selected(root, candidate.name):
                shutil.rmtree(candidate, ignore_errors=True)
            logger.warning(
                f"plugin manager runtime refresh failed: {error}; "
                "rerun skulk-plugin-service setup"
            )
        except ManagerNotRestartedError as error:
            # The selection is in place and kept. A service still verifying
            # the generation attaches on it later and the setup state follows
            # then; a manager that comes up on an older generation is the
            # mismatch the next attempt restarts onto this selection.
            logger.warning(
                f"plugin manager runtime selected but not yet started: {error}; "
                "the setup state follows when the service attaches on it"
            )
        except (OSError, TimeoutError) as error:
            # The manager may still be verifying the seal past the request
            # deadline and select the generation afterwards; once it does,
            # nothing else would reconcile the setup state, so wait for it.
            if snapshot is not None and await asyncio.to_thread(
                _selected_within, root, snapshot.generation, _SELECTION_WAIT_SECONDS
            ):
                await self._record_refresh(root, snapshot)
                return
            logger.warning(
                "plugin manager runtime refresh is indeterminate: "
                f"{type(error).__name__}; the staged generation is retained"
            )

    async def _record_refresh(
        self,
        root: Path,
        snapshot: ServiceSnapshot,
        message: str = (
            "plugin manager runtime refreshed to this host's Skulk build; "
            "the service restarts on it"
        ),
    ) -> bool:
        # Setup state names the generation the service runs on, its build and
        # the source identity it was staged from; status verifies against it
        # and a setup rerun would otherwise stage and restart yet another
        # generation for the same build. A record that fails is reported,
        # not believed: the next attach reconciles it.
        try:
            source = await asyncio.to_thread(service_source_identity)
            record_refreshed_snapshot(root, snapshot, source)
        except (OSError, ValueError) as error:
            logger.warning(
                f"setup state not recorded: {type(error).__name__}; "
                "reconciled on the next attach"
            )
            return False
        logger.info(message)
        return True

    async def _reconcile_setup_state(self, root: Path, build: str) -> None:
        """Make the setup state follow a selection the manager attached on.

        A refresh that could not see the manager start (a bootstrap that
        outlived the wait) recorded nothing; once the manager attaches on the
        selected generation staged from this build, the setup state is the
        only thing left behind, and no mismatch would ever schedule it.
        """
        selected = selected_generation_for(root, build)
        if selected is None or self.setup_reconciled == selected.generation:
            return
        # Remembered only once the state is confirmed current, so a state
        # that could not be read or recorded is retried on the next attach
        # rather than skipped for the rest of the process.
        try:
            current = setup_state_names(root, selected.generation)
        except (OSError, ValueError) as error:
            logger.warning(
                f"setup state not readable: {type(error).__name__}; "
                "reconciled on the next attach"
            )
            return
        if not current and not await self._record_refresh(
            root,
            selected,
            "plugin manager attached on the generation staged for this "
            "host's Skulk build; the setup state now names it",
        ):
            return
        self.setup_reconciled = selected.generation

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
                if self.attachment.build is not None:
                    await self._reconcile_setup_state(
                        Path(self.connection.manager_root), self.attachment.build
                    )
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
                    # A stale or errored row may carry the previous service
                    # instance's state; only a current observation says an
                    # owner is absent by design.
                    self.owners[identifier].manager_state = (
                        item.service.state
                        if item.service is not None
                        and not item.stale
                        and item.error_code is None
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
                    # The last reported owner state is no longer known
                    # either; an absent owner must warn again.
                    owner.manager_state = None
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
