"""One finite receive-only peer lifetime, owned by Core rather than a browser."""

from __future__ import annotations

import asyncio
import subprocess
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Literal, final

from foxlight_model_preservation.peer_runtime import (
    PeerClient,
    executable,
    initialize_peer,
    read_profile,
    set_bootstrap,
)
from foxlight_model_preservation.signing import canonical_bytes, digest
from foxlight_model_preservation.storage import atomic_bytes, directory

from skulk.store.peer_contracts import PeerSetup, PeerState
from skulk.store.peer_worker import PeerWorkerRequest
from skulk.store.preservation_io import PreservationIO
from skulk.store.preservation_journal import (
    PreservationHostError,
    PreservationJournal,
    safe_json,
    secure_directory,
)
from skulk.utils.process_groups import observe_process_exit, terminate_process_group


@final
class PeerLease:
    """Actual native process group and its held Core lifetime/serialization fence."""

    def __init__(
        self,
        owner: PeerHost,
        process: subprocess.Popen[str],
        path: Path,
        repository: Path,
        binary: str,
    ):
        self.owner, self.process, self.path = owner, process, path
        self.repository, self.binary = repository, binary
        self._closing: asyncio.Task[None] | None = None

    async def close(self) -> None:
        """Drain the actual worker/group even if the requesting job is cancelled."""
        if self._closing is None:
            self._closing = asyncio.create_task(self._stop())
        try:
            await asyncio.shield(self._closing)
        except asyncio.CancelledError:
            await self._closing
            raise

    async def _stop(self) -> None:
        process = self.process

        def stop() -> None:
            # Popen has no asyncio child watcher: retain the unreaped leader PID
            # through descendant signalling, including normal worker exit.
            if process.stdin is not None:
                with suppress(BrokenPipeError, ConnectionResetError):
                    process.stdin.write("cancel\n")
                    process.stdin.flush()
                with suppress(BrokenPipeError, ConnectionResetError):
                    process.stdin.close()
            deadline = time.monotonic() + 8
            while observe_process_exit(process) is None and time.monotonic() < deadline:
                time.sleep(0.025)
            terminate_process_group(process)

        await self.owner.storage.run(stop)
        try:
            await self.owner.storage.run(lambda: self.path.unlink(missing_ok=True))
        finally:
            self.owner.release(self)


@final
class PeerHost:
    """Explicit settings and receive-only native supervision under the host owner."""

    def __init__(
        self,
        root: Path,
        journal: PreservationJournal,
        storage: PreservationIO,
        command: Callable[[Path], tuple[str, ...]],
    ):
        self.root, self.journal, self.storage, self.command = (
            root,
            journal,
            storage,
            command,
        )
        self._gate = asyncio.Lock()
        self._active: PeerLease | None = None
        self._closed = False
        # A crashed caller's worker already owns its loaded request and EOF
        # fence. This known control file is never a model or a replay trigger.
        (self.root / "peer-request.json").unlink(missing_ok=True)

    def release(self, lease: PeerLease) -> None:
        """Release serialization only after this exact owned native lease drains."""
        if self._active is lease:
            self._active = None
            self._gate.release()

    def selected(self) -> tuple[PeerSetup, str]:
        """Read the protected current routing/exposure policy without networking."""
        value = self.journal.setting("peer-setup")
        if value is None:
            raise PreservationHostError("peer_setup_required")
        setup = PeerSetup.model_validate_json(safe_json(value))
        return setup, digest(canonical_bytes(setup.model_dump(mode="json")))

    @staticmethod
    def _runtime() -> tuple[
        Literal["available", "missing", "incompatible"], str | None
    ]:
        try:
            return "available", executable()
        except (OSError, ValueError):
            import shutil

            return ("incompatible" if shutil.which("ipfs") else "missing"), None

    async def status(self) -> PeerState:
        """Observe installed software/setup locally; start no daemon or probes."""
        runtime, _ = await self.storage.probe(
            "peer-runtime", self._runtime, ("missing", None)
        )
        value = self.journal.setting("peer-setup")
        setup, identity = self.selected() if value is not None else (None, None)
        return PeerState(
            runtime=runtime,
            setup_id=identity,
            bootstrap=setup.bootstrap if setup else (),
            allow_private=setup.allow_private if setup else False,
            active=self._active is not None,
        )

    async def configure(self, setup: PeerSetup) -> PeerState:
        """Save reviewed contacts offline; refuse changes during actual peer work."""
        if self._closed or self._gate.locked():
            raise PreservationHostError("peer_busy_try_again")
        async with self._gate:
            self.journal.set_setting("peer-setup", setup.model_dump(mode="json"))
        return await self.status()

    async def open(
        self, expected_setup: str, seconds: int, cancelled: Callable[[], bool]
    ) -> PeerLease:
        """Start one known finite native session after exact local consent checks."""
        while True:
            if self._closed or cancelled():
                raise PreservationHostError("cancelled")
            try:
                await asyncio.wait_for(self._gate.acquire(), 0.2)
                break
            except TimeoutError:
                continue
        lease: PeerLease | None = None
        try:
            setup, identity = self.selected()
            if identity != expected_setup:
                raise PreservationHostError("peer_setup_changed")
            state, binary = await self.storage.run(self._runtime)
            if state != "available" or binary is None:
                raise PreservationHostError("peer_runtime_required")
            repository = (
                self.root
                / "peer-cache"
                / ("private" if setup.allow_private else "public")
            )
            request_path = self.root / "peer-request.json"

            def prepare() -> None:
                secure_directory(repository.parent)
                if not repository.exists():
                    initialize_peer(
                        repository,
                        binary,
                        allow_private_peers=setup.allow_private,
                        bootstrap=setup.bootstrap,
                    )
                else:
                    profile = read_profile(repository)
                    if (
                        profile.mode != "receive"
                        or profile.allow_private_peers != setup.allow_private
                    ):
                        raise PreservationHostError("peer_profile_changed")
                    set_bootstrap(repository, setup.bootstrap)
                request = PeerWorkerRequest(
                    repository=repository.absolute(),
                    binary=binary,
                    seconds=min(seconds + 20, 86400),
                )
                with directory(self.root) as root:
                    atomic_bytes(
                        root, request_path.name, request.model_dump_json().encode()
                    )

            await self.storage.run(prepare)
            if cancelled() or self._closed:
                raise PreservationHostError("cancelled")
            process = await self.storage.run(
                lambda: subprocess.Popen(
                    self.command(request_path),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    start_new_session=True,
                ),
                discard=terminate_process_group,
            )
            lease = PeerLease(self, process, request_path, repository, binary)
            self._active = lease
            end = time.monotonic() + 25
            while time.monotonic() < end:
                if cancelled() or self._closed:
                    raise PreservationHostError("cancelled")
                if observe_process_exit(process) is not None:
                    raise PreservationHostError("peer_runtime_unavailable")

                def ready() -> bool:
                    try:
                        PeerClient(
                            repository, binary, allow_network=True, seconds=3
                        ).wire_usage()
                        return True
                    except (OSError, ValueError):
                        return False

                if await self.storage.run(ready):
                    return lease
                await asyncio.sleep(0.1)
            raise PreservationHostError("peer_start_deadline_exceeded")
        except BaseException:
            if lease is not None:
                await lease.close()
            else:
                self._gate.release()
            raise

    async def close(self) -> None:
        """Refuse new work and drain any actual remaining native process group."""
        self._closed = True
        if self._active is not None:
            await self._active.close()
