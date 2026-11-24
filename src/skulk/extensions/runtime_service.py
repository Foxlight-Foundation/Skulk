"""Fixed nonroot launcher for a verified managed runtime, separate from inference.

The owner is started through :func:`skulk.utils.owned_process.start_supervised`
and holds one end of a lifetime pipe. Closing the launcher's end, which also
happens when the launcher dies, asks the owner to shut down. Where the owner
finds its end is the one thing that differs by platform, and the launcher says
so on the command line:

- POSIX: ``--lifetime-fd <number>``, an inherited descriptor.
- Windows: ``--lifetime-stdin``, the owner's standard input. A descriptor
  number means nothing to a Windows child. The argument is explicit so that an
  owner started by hand never mistakes a closed or redirected standard input
  for the end of a launcher. An owner has to be built to accept it; one that
  is not exits at once and the lifetime is reported as ``owner_exited``.
"""

import argparse
import asyncio
import sys
import time
import zipfile
from pathlib import Path
from typing import Literal, cast, final
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from skulk.extensions.native_host_binding import provision_native_host
from skulk.extensions.runtime_artifacts import Digest
from skulk.extensions.runtime_files import (
    RuntimeLock,
    acquire_runtime_lock,
    write_private,
)
from skulk.extensions.runtime_install import finish_runtime_work
from skulk.extensions.runtime_integrity import (
    owned_generation,
    runtime_interpreter,
)
from skulk.extensions.runtime_selection import RuntimeSelection, RuntimeSelector
from skulk.utils.owned_process import SupervisedProcess, start_supervised
from skulk.utils.stop_signals import stop_on_signal

_BOOTSTRAP = "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('__owner__',run_name='__main__')"
# A bundle with no owner shim is launched from the signed wheels installed in
# the generation runtime, through the public launcher entry-point group. The
# artifact argument is consumed so the owner sees the same argv either way.
_LAUNCHER_BOOTSTRAP = (
    "import sys;from importlib.metadata import entry_points;sys.argv.pop(1);"
    "found=[e for e in entry_points(group='skulk.capability_runtime') if e.name=='owner'];"
    "sys.exit('no owner launcher in the runtime') if not found else sys.exit(found[0].load()())"
)


def lifetime_arguments(descriptor: int | None) -> tuple[str, ...]:
    """Tell the owner where its end of the lifetime pipe is.

    Args:
        descriptor: The number of the inherited descriptor, or ``None`` when
            the owner's standard input is its end of the pipe.

    Returns:
        The arguments the owner's command line ends with.
    """
    if descriptor is None:
        return ("--lifetime-stdin",)
    return ("--lifetime-fd", str(descriptor))


def owner_bootstrap(artifact: Path) -> str:
    """Use the archive shim when present, otherwise the runtime's declared launcher."""
    with zipfile.ZipFile(artifact) as archive:
        if "__owner__.py" in archive.namelist():
            return _BOOTSTRAP
    return _LAUNCHER_BOOTSTRAP


type ServiceProcessState = Literal[
    "verifying", "running", "stopping", "stopped", "failed"
]
"""Observed local owner process states, independent of desired selection."""

_CHECK_SECONDS = 30.0
_SHUTDOWN_SECONDS = 30.0
# How long a tree that was ended by force is given to let go of its fence.
_RELEASE_SECONDS = 5.0


class RuntimeServiceStatus(BaseModel):
    """Protected process observation; running does not imply capability readiness."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    service_instance: str = Field(description="Unique launcher lifetime identifier.")
    observed_at: float = Field(
        description="Unix observation time; old records are stale."
    )
    state: ServiceProcessState = Field(
        description="Observed local owner process state, independent of desired selection."
    )
    selected_digest: Digest | None = Field(description="Desired selected generation.")
    active_digest: Digest | None = Field(
        description="Generation owned by this launcher."
    )
    error_code: (
        Literal[
            "verification_failed", "owner_exited", "ownership_busy", "service_failed"
        ]
        | None
    ) = Field(default=None, description="Sanitized actionable failure class.")
    output_bytes: int = Field(
        ge=0, description="Discarded owner output volume; no payloads."
    )


@final
class RuntimeService:
    """Own one fixed plugin process until shutdown or admission verification fails.

    This launcher has no provider SDK, spending policy or witness lifecycle. A
    service manager may restart it; each lifetime revalidates local selection and
    artifacts. An owner failure ends this lifetime without replaying API work.
    """

    def __init__(self, root: Path) -> None:
        """Use protected local installation storage without starting plugin code."""
        self.selector = RuntimeSelector(root)
        self.root = self.selector.root
        self.instance = uuid4().hex
        self.process: SupervisedProcess | None = None
        self.selection: RuntimeSelection | None = None
        self.output_bytes = 0
        self.drains: list[asyncio.Task[None]] = []

    def _status(
        self,
        state: ServiceProcessState,
        error: Literal[
            "verification_failed", "owner_exited", "ownership_busy", "service_failed"
        ]
        | None = None,
    ) -> None:
        active = self.process is not None and self.process.returncode is None
        value = RuntimeServiceStatus(
            service_instance=self.instance,
            observed_at=time.time(),
            state=state,
            selected_digest=self.selection.runtime_digest if self.selection else None,
            active_digest=self.selection.runtime_digest
            if self.selection and active
            else None,
            error_code=error,
            output_bytes=self.output_bytes,
        )
        write_private(
            self.root / "service-status.json", value.model_dump_json().encode()
        )

    async def _drain(self, stream: asyncio.StreamReader) -> None:
        while block := await stream.read(16384):
            self.output_bytes += len(block)

    async def _spawn(self, lock: RuntimeLock) -> None:
        assert self.selection is not None
        generation = owned_generation(self.root, self.selection.runtime_digest)
        provision_native_host(self.root, self.selection.runtime_digest, self.instance)
        bundle = generation / "artifacts" / "bundle.pyz"
        command = (
            str(runtime_interpreter(generation / "runtime")),
            "-I",
            "-B",
            "-c",
            owner_bootstrap(bundle),
            str(bundle),
            "--root",
            str(self.root),
        )
        # The owner holds the service fence, but only the read end of its
        # lifetime pipe. Abrupt launcher death therefore asks it to shut down
        # before another launcher can acquire this service's ownership.
        self.process = await start_supervised(
            lambda descriptor: (*command, *lifetime_arguments(descriptor)),
            cwd=self.root,
            lock=lock,
        )
        self.drains.append(asyncio.create_task(self._drain(self.process.stdout)))

    async def _close(self) -> None:
        process = self.process
        ended_by_force = False
        if process is not None:
            # Pipe EOF is the managed owner's graceful shutdown request. Sending
            # SIGTERM as well can race its final interpreter shutdown and turn a
            # successful teardown into a signal failure after handlers reset.
            process.release()
            try:
                async with asyncio.timeout(_SHUTDOWN_SECONDS):
                    await process.wait()
            except TimeoutError:
                # The bound ends the whole tree. That includes what the owner
                # left running if it went, itself, just as the bound ran out.
                process.kill()
                ended_by_force = True
                await process.wait()
        # A descendant can keep stdout open after its owner exits. Its pipe must
        # not prevent bounded shutdown or cause us to retain any output payload.
        for drain in self.drains:
            drain.cancel()
        await asyncio.gather(*self.drains, return_exceptions=True)
        if process is not None:
            # On Windows this ends what the owner left running and waits until
            # none of it is left; on POSIX the check below finds a survivor.
            await process.close()
        # Children inherit the owner's fence. Do not report stopped or allow a
        # replacement if an old child survived the bounded owner teardown.
        if ended_by_force:
            # An ended process lets go of its files a moment after it is
            # told to go, and on POSIX there is nothing to wait on for a
            # descendant. The fence itself says when the tree has gone.
            fence = await acquire_runtime_lock(
                self.root, wait_seconds=_RELEASE_SECONDS, name="supervisor.lock"
            )
            fence.close()
        else:
            RuntimeLock(self.root, "supervisor.lock").close()

    async def serve(self, stopped: asyncio.Event) -> bool:
        """Run one verified owner lifetime; return false for an actionable failure.

        Cancellation waits for owned process cleanup. Local pending selection is
        recovered before launch; provider operations are never replayed here.
        A disabled or absent selection returns a successful stopped observation.
        """
        lock = RuntimeLock(self.root, "service.lock")
        success = False
        failure: (
            Literal[
                "verification_failed",
                "owner_exited",
                "ownership_busy",
                "service_failed",
            ]
            | None
        ) = None
        waiter: asyncio.Task[bool] | None = None
        exited: asyncio.Task[int] | None = None
        try:
            try:
                if self.selector.pending.exists():
                    await self.selector.recover()
                self.selection = self.selector.current()
                self._status("verifying")
                if (
                    self.selection is None
                    or not self.selection.enabled
                    or stopped.is_set()
                ):
                    success = True
                else:
                    # Discovery may still own a metadata measurement after
                    # its observer times out. Brief installer contention
                    # before execution must not consume crash backoff.
                    async with self.selector.installer.locked_generation(
                        self.selection.runtime_digest, wait_for_ownership=True
                    ):
                        if self.selector.current() != self.selection:
                            raise ValueError("runtime selection changed")
                        RuntimeLock(self.root, "supervisor.lock").close()
                        if not stopped.is_set():
                            await finish_runtime_work(
                                asyncio.create_task(self._spawn(lock))
                            )
                    if self.process is None:
                        success = True
                    else:
                        self._status("running")
                        waiter = asyncio.create_task(stopped.wait())
                        exited = asyncio.create_task(self.process.wait())
                        while True:
                            done, _ = await asyncio.wait(
                                (waiter, exited),
                                timeout=_CHECK_SECONDS,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            if waiter in done:
                                success = True
                                break
                            if exited in done:
                                failure = "owner_exited"
                                break
                            try:
                                async with self.selector.installer.locked_generation(
                                    self.selection.runtime_digest
                                ):
                                    if self.selector.current() != self.selection:
                                        raise ValueError("runtime selection changed")
                            except BlockingIOError:
                                # Offline staging holds this same lock. This is
                                # contention, not evidence that the owner failed.
                                continue
                            self._status("running")
            except BlockingIOError:
                failure = "ownership_busy"
            except ValueError:
                failure = "verification_failed"
            except OSError:
                failure = "service_failed"
        finally:
            try:
                try:
                    self._status("stopping")
                finally:
                    await finish_runtime_work(asyncio.create_task(self._close()))
            except BlockingIOError:
                success, failure = False, "ownership_busy"
            finally:
                for task in (waiter, exited):
                    if task is not None:
                        task.cancel()
                if waiter is not None:
                    await asyncio.gather(waiter, return_exceptions=True)
                if exited is not None:
                    await asyncio.gather(exited, return_exceptions=True)
                try:
                    self._status(
                        "stopped" if success else "failed",
                        failure if success or failure else "service_failed",
                    )
                finally:
                    lock.close()
        return success


async def _run(root: Path) -> bool:
    stopped = asyncio.Event()
    with stop_on_signal(stopped):
        return await RuntimeService(root).serve(stopped)


def main() -> None:
    """Launch a fixed verified owner using only a locally provisioned root path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        success = asyncio.run(_run(cast(Path, arguments.root)))
    except (OSError, ValueError):
        print(
            "managed plugin service unavailable; inspect local service status",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
