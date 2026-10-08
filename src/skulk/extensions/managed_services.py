"""Live local manager inventory shared by plugin configuration and Fabric lookup."""

import asyncio
import contextlib
import json
import os
import re
import shutil
import signal
import sys
import time
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, final

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from skulk.extensions.managed import ManagedConnection, ManagedOwner
from skulk.extensions.managed_attachment import (
    ManagedAttachment,
    manager_pids,
    manager_processes,
    running_generations,
    runs_generation,
)
from skulk.extensions.runtime_artifacts import Digest, ProtocolUnsupportedError
from skulk.extensions.runtime_attachment import (
    InstallationIdentifier,
    ProfileIdentifier,
    ServiceConnection,
)
from skulk.extensions.runtime_catalog import catalog_refusal
from skulk.extensions.runtime_files import RuntimeLock, read_private
from skulk.extensions.runtime_manager import (
    CatalogInstallRequest,
    CatalogRegistration,
    CatalogRequest,
    CatalogStoreRequest,
    InstallationRequest,
    InstalledRelease,
    InstallRecoveryRequest,
    InstallSubmission,
    InventoryRequest,
    ManagerBuildMismatchError,
    OperationRequest,
    ReleaseRequest,
    ReloadRuntimeRequest,
    SourceRegistration,
    StoreTrustRequest,
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
from skulk.shared.constants import offline_mode

_WINDOW = TypeAdapter(list[int])

_STORE_TRUST_SECONDS: Final = 3600.0
"""How often installations bound to the built-in store have its trust renewed.

Hourly keeps a store revocation, such as a compromised publisher key, from
waiting long, and keeps renewals far ahead of any trust expiry."""

_STORE_TRUST_RETRY_SECONDS: Final = 300.0
"""How soon a renewal the manager could not complete is tried again."""


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
        kind not in ("release", "runtime", "catalog")
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

_RUNTIME_REFRESH_RETRY_SECONDS: Final = 300.0
"""How long after one automatic manager refresh attempt began the next may begin."""

_UPDATE_REPORT_SECONDS: Final = 2 * _RUNTIME_REFRESH_RETRY_SECONDS
"""How long after a build mismatch is first seen an update reads as in progress.

Two automatic attempts begin inside it. An attempt begun inside it is followed
to its end and given ``_UPDATE_SETTLE_SECONDS`` more; later attempts keep
running in the background but no longer extend it, so an update that never
converges reads as a failure the owner can act on rather than as progress for
ever."""

_UPDATE_SETTLE_SECONDS: Final = 2 * _MANAGER_START_SECONDS
"""How long after a refresh attempt ends the manager has to attach on the new build.

A successful reload returns before the manager restarts, and the keep-alive
and bootstrap verification take up to ``_MANAGER_START_SECONDS``; twice that
covers a slow restart without calling it a failure."""

RuntimeUpdateFailure = Literal[
    "staging_failed", "refused", "other_interpreter", "not_restarted"
]
"""Why bringing the plugin manager to this host's Skulk build did not finish.

``staging_failed``: this host's environment could not be copied for the
manager (no further automatic attempt until Skulk restarts);
``refused``: the manager refused the matching runtime;
``other_interpreter``: the registered service starts another interpreter, which
only setup registers again; ``not_restarted``: the manager has not attached on
this build within ``_UPDATE_REPORT_SECONDS``."""


@final
@dataclass(frozen=True)
class RuntimeUpdate:
    """Where bringing the plugin manager to this host's Skulk build stands.

    A Skulk update leaves the manager on the previous build until a matching
    runtime is staged and the manager restarts on it. The service status
    reports that window as setup in progress rather than as a broken
    inventory. Built from memory alone, so reading it never waits on the copy.
    """

    phase: Literal["refreshing", "waiting", "failed"]
    """``refreshing``: a refresh attempt is staging or reloading;
    ``waiting``: no attempt is running and the manager has not attached on
    this build yet; ``failed``: see ``failure``."""
    since: float
    """Monotonic time this phase began; orders a failure against a later setup."""
    failure: RuntimeUpdateFailure | None = None
    """Why the update did not finish; set only when ``phase`` is ``failed``."""


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


RUNTIME_PRUNE_RETRY_SECONDS: Final = 300.0
"""Wait before retrying a prune that could not remove every superseded runtime.

The inventory refresh runs every second; retrying a persistent failure (a
permission problem, say) at that rate would re-walk gigabytes and log each
time.
"""

RETAINED_PREVIOUS_GENERATIONS: Final = 1
"""Complete manager runtimes kept beside the selected one, newest first.

Each generation is a full copy of the host's Skulk environment (about 1.7 GB
on macOS), staged again on every Skulk update, so without a bound the
service volume fills. One previous generation is kept for going back to the
previous build by hand.
"""

_GENERATION_NAME = re.compile(r"[a-f0-9]{32}")


def _staged_at(generation: Path) -> float:
    try:
        return (generation / "staged.json").stat().st_mtime
    except OSError:
        return 0.0


def superseded_generations(root: Path, running: AbstractSet[str]) -> list[Path]:
    """Manager runtime generations nothing uses any more, oldest first.

    Kept: the selected generation, every generation a manager process still
    runs from, and the newest ``RETAINED_PREVIOUS_GENERATIONS`` other complete
    generations (a ``staged.json`` marks one complete). Everything else goes,
    interrupted copies included: staging never activates them. Entries that
    are not generation directories are left alone.

    Args:
        root: The manager's service root.
        running: Generations running managers use (``running_generations``).

    Returns:
        The generation directories that may be removed.
    """
    generations = root / "core-runtimes"
    try:
        entries = [
            path
            for path in generations.iterdir()
            if _GENERATION_NAME.fullmatch(path.name)
            and path.is_dir()
            and not path.is_symlink()
        ]
    except OSError:
        return []
    keep = set(running)
    selected = selected_generation(root)
    if selected is not None:
        keep.add(selected)
    previous = sorted(
        (
            path
            for path in entries
            if path.name not in keep and (path / "staged.json").is_file()
        ),
        key=_staged_at,
        reverse=True,
    )
    keep.update(path.name for path in previous[:RETAINED_PREVIOUS_GENERATIONS])
    return sorted((path for path in entries if path.name not in keep), key=_staged_at)


def _tree_bytes(path: Path) -> int:
    total = 0
    for directory, _, files in os.walk(path):
        for name in files:
            with contextlib.suppress(OSError):
                total += os.lstat(os.path.join(directory, name)).st_size
    return total


@final
@dataclass(frozen=True)
class RuntimePruneOutcome:
    """What one pass over superseded manager runtimes did."""

    removed: int
    """Generations removed completely."""
    freed_bytes: int
    """Bytes those generations held."""
    remaining: int
    """Superseded generations still present, to be tried again later."""


def prune_superseded_generations(root: Path) -> RuntimePruneOutcome | None:
    """Remove superseded manager runtimes under the installer fence.

    Staging holds the same fence, so a copy in progress is never touched.
    Removal is best effort: a generation that cannot be removed fully stays,
    counts as remaining, and is tried again later.

    Args:
        root: The manager's service root.

    Returns:
        What the pass removed and what remains, or ``None`` when the fence is
        held elsewhere and nothing was examined.
    """
    try:
        fence = RuntimeLock(root)
    except BlockingIOError:
        return None
    try:
        removed = 0
        freed = 0
        remaining = 0
        for generation in superseded_generations(root, running_generations(root)):
            size = _tree_bytes(generation)
            shutil.rmtree(generation, ignore_errors=True)
            if generation.exists():
                remaining += 1
            else:
                removed += 1
                freed += size
        return RuntimePruneOutcome(
            removed=removed, freed_bytes=freed, remaining=remaining
        )
    finally:
        fence.close()


class ManagedInstallation(BaseModel):
    """Bounded local desired state and process observation, with no paths or secrets."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    plugin_id: InstallationIdentifier = Field(
        description="Stable local installed-plugin identifier."
    )
    release: InstalledRelease | None = Field(
        default=None,
        description="Name, version and publisher of the selected signed release, read from its staged metadata; absent before a release is staged or when that metadata is unreadable.",
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
    store_trust_retry: bool = Field(
        default=False,
        description="Whether installations following the built-in capability "
        "store were busy when its trust was last followed, so the host renews "
        "it again within minutes.",
    )


type ManagementRequest = (
    InventoryRequest
    | CatalogRequest
    | CatalogRegistration
    | CatalogStoreRequest
    | CatalogInstallRequest
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
        # Superseded runtimes are removed once per selected generation, after
        # a manager has attached on it, in the background: a copy is large,
        # and the inventory refresh must not wait for its removal.
        self.runtime_prune: asyncio.Task[None] | None = None
        self.runtimes_pruned_for: str | None = None
        # Monotonic time before which a failed or partial prune is not retried.
        self.runtime_prune_retry_at = 0.0
        # The store's trust renewal for installations bound to it runs in the
        # background on its own schedule; None until the first one is due.
        self.store_trust: asyncio.Task[None] | None = None
        self.store_trust_due: float | None = None
        # One warning per run of failed renewals, not one every five minutes.
        self.store_trust_failing = False
        # Counts deferral notices; a renewal that started before one arrived
        # must not push the retry it asked for back out to an hour.
        self.store_trust_deferrals = 0
        # The manager's retry flag as last read, so only its rise is a notice.
        self.store_trust_retry_seen = False
        # A build mismatch opens an update that the next attachment on this
        # build closes. The service status reads it, so the minutes a Skulk
        # update spends staging and restarting the manager read as setup in
        # progress instead of a broken inventory. Monotonic times throughout.
        self.update_started: float | None = None
        self.update_attempt_ended: float | None = None
        self.update_failure: RuntimeUpdateFailure | None = None
        self.update_failed_at = 0.0
        # Set once the first inventory refresh has an outcome: until then a
        # manager still on the previous build answers like a ready one.
        self.first_observation = asyncio.Event()
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
        if self.update_started is None:
            self.update_started = time.monotonic()
        if self.runtime_refresh is not None and not self.runtime_refresh.done():
            return
        if self.runtime_refresh_exhausted:
            return
        if (
            self.runtime_refreshed is not None
            and time.monotonic() - self.runtime_refreshed
            < _RUNTIME_REFRESH_RETRY_SECONDS
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

    def runtime_update(self) -> RuntimeUpdate | None:
        """Report an update of the manager to this host's build; None when none is pending.

        Reads memory only: the copy runs in its own task, so this never waits
        on it. Once the update has run past its bound without the manager
        attaching on this build, the failure is kept until it does, or until
        a later setup supersedes it, so background retries do not make the
        report alternate between progress and failure.
        """
        started = self.update_started
        if self.closed or started is None:
            return None
        failure = self.update_failure
        if failure is not None:
            return RuntimeUpdate("failed", self.update_failed_at, failure)
        bound = started + _UPDATE_REPORT_SECONDS
        attempt = self.runtime_refreshed
        # Only an attempt begun inside the bound is followed past it.
        followed = attempt is not None and attempt < bound
        task = self.runtime_refresh
        if attempt is not None and followed and task is not None and not task.done():
            return RuntimeUpdate("refreshing", attempt)
        ended = self.update_attempt_ended
        now = time.monotonic()
        if now < bound or (
            followed and ended is not None and now < ended + _UPDATE_SETTLE_SECONDS
        ):
            return RuntimeUpdate("waiting", started if ended is None else ended)
        self._update_failed("not_restarted")
        return RuntimeUpdate("failed", self.update_failed_at, "not_restarted")

    async def first_observed(self, seconds: float) -> None:
        """Wait, at most ``seconds``, for the first manager observation of this run.

        Until the first inventory refresh has an outcome, a manager still on
        the previous Skulk build answers like a ready one, so a status read
        right after Skulk starts waits for that outcome instead of reporting
        it ready. Returns at once when observation is not running or has
        already observed once. The wait involves no staging: a refresh only
        schedules its copy.
        """
        if self.closed or self.task is None or self.task.done():
            return
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(seconds):
                await self.first_observation.wait()

    def _update_failed(self, failure: RuntimeUpdateFailure) -> None:
        """Record why the open update did not finish; nothing once it has closed."""
        if self.update_started is None:
            return
        self.update_failure = failure
        self.update_failed_at = time.monotonic()

    def _update_finished(self) -> None:
        """Close the update: the manager attached on this host's build."""
        self.update_started = None
        self.update_attempt_ended = None
        self.update_failure = None

    async def _refresh_manager_runtime(self, root: Path, live: str) -> None:
        """Stage the manager runtime from this host's build and ask for a reload.

        A generation already staged for this build is reused, and one the
        manager refuses is removed, so repeated attempts against a broken
        manager do not accumulate copies of Skulk on the service volume.
        """
        candidate: Path | None = None
        snapshot: ServiceSnapshot | None = None
        # What an explicit refusal below means for the owner; the status
        # route words it, and the log keeps the detail.
        failure: RuntimeUpdateFailure = "refused"
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
                failure = "other_interpreter"
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
                    failure = "staging_failed"
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
            # The service status surfaces this to the owner with the next step.
            self._update_failed(failure)
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
        finally:
            # Starts the settle time the manager has to attach on this build.
            self.update_attempt_ended = time.monotonic()

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

    def _schedule_runtime_prune(self, root: Path, build: str) -> None:
        """Remove superseded runtimes once the manager runs this build's selection."""
        if self.closed or (
            self.runtime_prune is not None and not self.runtime_prune.done()
        ):
            return
        selected = selected_generation_for(root, build)
        if selected is None or self.runtimes_pruned_for == selected.generation:
            return
        if time.monotonic() < self.runtime_prune_retry_at:
            return
        self.runtime_prune = asyncio.create_task(
            self._prune_runtimes(root, selected.generation)
        )

    async def _prune_runtimes(self, root: Path, generation: str) -> None:
        try:
            outcome = await asyncio.to_thread(prune_superseded_generations, root)
        except (OSError, ValueError) as error:
            self.runtime_prune_retry_at = time.monotonic() + RUNTIME_PRUNE_RETRY_SECONDS
            logger.warning(
                "superseded plugin manager runtimes not removed: "
                f"{type(error).__name__}; tried again in "
                f"{RUNTIME_PRUNE_RETRY_SECONDS / 60:.0f} minutes"
            )
            return
        if outcome is None:
            # Staging or an install holds the fence; the next refresh retries.
            return
        if outcome.removed:
            logger.info(
                f"removed {outcome.removed} superseded plugin manager runtime"
                f"{'' if outcome.removed == 1 else 's'} "
                f"({outcome.freed_bytes / 2**30:.1f} GiB)"
            )
        if outcome.remaining:
            # A partial removal is not done: leaving the latch unset retries the
            # rest without a restart, after a wait so a persistent failure does
            # not re-walk gigabytes on every one-second refresh.
            self.runtime_prune_retry_at = time.monotonic() + RUNTIME_PRUNE_RETRY_SECONDS
            logger.warning(
                f"{outcome.remaining} superseded plugin manager runtime"
                f"{'' if outcome.remaining == 1 else 's'} could not be removed "
                f"fully; tried again in {RUNTIME_PRUNE_RETRY_SECONDS / 60:.0f} minutes"
            )
            return
        self.runtimes_pruned_for = generation

    async def _settle_runtime_prune(self) -> None:
        """Let a removal in its worker thread finish rather than abandon it."""
        task, self.runtime_prune = self.runtime_prune, None
        if task is not None and not task.done():
            await asyncio.gather(task, return_exceptions=True)

    def _schedule_store_trust(self, root: Path) -> None:
        """Renew the built-in store's trust for installations bound to it.

        The manager runs with a fixed environment, so the node drives the
        renewal and states its own offline mode: offline, the manager uses
        only the store's last verified trust. Runs in the background so a
        slow store never holds the inventory refresh.
        """
        if self.store_trust is not None and not self.store_trust.done():
            return
        if self.store_trust_due is not None and time.monotonic() < self.store_trust_due:
            return
        self.store_trust_due = time.monotonic() + _STORE_TRUST_RETRY_SECONDS
        self.store_trust = asyncio.create_task(self._refresh_store_trust(root))

    async def _refresh_store_trust(self, root: Path) -> None:
        """Ask the manager to renew store trust; a refusal is retried sooner."""
        deferrals = self.store_trust_deferrals
        try:
            result = await manager_request(
                root, StoreTrustRequest(offline=offline_mode())
            )
        except (OSError, ValueError, TimeoutError):
            return
        payload = result.get("result")
        if set(result) != {"result"} or not isinstance(payload, dict):
            # Includes the manager refusing because installations follow the
            # store and no verified store trust is in force: those keep
            # running on the trust they hold until it expires.
            if not self.store_trust_failing:
                logger.warning(
                    "the plugin manager could not renew the built-in capability "
                    "store's trust for its installations; retrying every five "
                    "minutes"
                )
            self.store_trust_failing = True
            return
        self.store_trust_failing = False
        # A busy installation is retried soon rather than in an hour, and so is
        # one a catalog read deferred while this renewal was in flight.
        if not payload.get("deferred") and self.store_trust_deferrals == deferrals:
            self.store_trust_due = time.monotonic() + _STORE_TRUST_SECONDS
            # Every follower was reached, which cleared the manager's flag; a
            # flag seen again after this is a new deferral, even if no
            # inventory read caught it cleared in between.
            self.store_trust_retry_seen = False

    def _retry_deferred_followers(self, payload: dict[str, JsonValue]) -> None:
        """Bring the next renewal forward when a catalog read deferred followers.

        A read renews the store's trust and follows it at once, but an
        installation busy at that moment keeps its earlier trust; the node's
        next renewal then runs within five minutes instead of up to an hour.
        """
        if payload.get("store_trust_deferred"):
            self._note_store_trust_deferral()

    def _note_store_trust_deferral(self) -> None:
        """Bring the next renewal within five minutes, and keep it there."""
        self.store_trust_deferrals += 1
        soon = time.monotonic() + _STORE_TRUST_RETRY_SECONDS
        if self.store_trust_due is None or self.store_trust_due > soon:
            self.store_trust_due = soon

    async def _settle_store_trust(self) -> None:
        """Let a renewal in flight finish rather than abandon it mid-write."""
        task, self.store_trust = self.store_trust, None
        if task is not None and not task.done():
            await asyncio.gather(task, return_exceptions=True)

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
            catalog_refused = catalog_refusal(result)
            if catalog_refused is not None:
                raise catalog_refused
            raise ValueError("managed service request refused")
        if isinstance(request, CatalogRequest) and request.action == "read_catalog":
            self._retry_deferred_followers(payload)
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
                # Attached on this host's build: any update has finished.
                self._update_finished()
                assert self.connection is not None and self.context is not None
                assert self.attachment is not None
                if self.attachment.build is not None:
                    await self._reconcile_setup_state(
                        Path(self.connection.manager_root), self.attachment.build
                    )
                    self._schedule_runtime_prune(
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
                if inventory.store_trust_retry and not self.store_trust_retry_seen:
                    # Any deferral the manager saw, a terminal read's included.
                    self._note_store_trust_deferral()
                self.store_trust_retry_seen = inventory.store_trust_retry
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
                self._schedule_store_trust(Path(self.connection.manager_root))
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
            finally:
                self.first_observation.set()

    async def _poll(self) -> None:
        while True:
            with contextlib.suppress(OSError, ValueError, TimeoutError):
                await self.refresh()
            await asyncio.sleep(1)

    async def _detach(self) -> None:
        await self._settle_runtime_refresh()
        await self._settle_runtime_prune()
        await self._settle_store_trust()
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
        await self._settle_runtime_prune()
        await self._settle_store_trust()
        async with self.guard:
            await self._detach()
