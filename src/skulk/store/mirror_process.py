"""Bounded mirror-worker observations with unreaped process-group ownership."""

from __future__ import annotations

import asyncio
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

from skulk.store.mirror_worker import MirrorWorkerEvent
from skulk.store.preservation_io import PreservationIO
from skulk.store.preservation_journal import PreservationHostError
from skulk.utils.process_groups import observe_process_exit, terminate_process_group


class MirrorProcess:
    """Own the exact worker until its process group and observation pipe drain."""

    def __init__(self, process: subprocess.Popen[str]) -> None:
        self.process = process
        # Two unrelated mounted-archive operations may occupy every storage
        # effect slot. Withdrawal of public participation cannot wait for them.
        # Each owned worker has one deduplicated drain, with at most two workers
        # admitted by the mirror journal; no filesystem work uses this executor.
        self._control = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="preservation-mirror-stop"
        )
        self._transport: asyncio.ReadTransport | None = None
        self._reader: asyncio.StreamReader | None = None
        self._stopping: asyncio.Task[None] | None = None

    @classmethod
    async def start(
        cls,
        command: tuple[str, ...],
        descriptor: int | None,
        storage: PreservationIO,
    ) -> MirrorProcess:
        """Start only the host's known command, retaining cancellation ownership."""
        process = await storage.run(
            lambda: subprocess.Popen(  # noqa: S603 - fixed Core-authored worker command
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                start_new_session=True,
                pass_fds=(descriptor,) if descriptor is not None else (),
            ),
            discard=terminate_process_group,
        )
        owned = cls(process)
        try:
            if process.stdout is None:
                raise PreservationHostError("mirror_worker_protocol")
            reader = asyncio.StreamReader(limit=16384)
            transport, _ = await asyncio.get_running_loop().connect_read_pipe(
                lambda: asyncio.StreamReaderProtocol(reader), process.stdout
            )
            owned._reader, owned._transport = reader, transport
            return owned
        except BaseException:
            await owned.stop()
            raise

    def alive(self) -> bool:
        """Observe without reaping or substituting a historical listening port."""
        return (
            self.process.returncode is None
            and observe_process_exit(self.process) is None
        )

    async def observations(self, receive: Callable[[MirrorWorkerEvent], None]) -> None:
        """Read bounded owned JSON until exit; retain the leader PID throughout."""
        reader = self._reader
        if reader is None:
            raise PreservationHostError("mirror_worker_protocol")
        while True:
            try:
                line = await asyncio.wait_for(reader.readline(), timeout=2)
            except TimeoutError:
                if not self.alive():
                    return
                continue
            if not line:
                # EOF alone is not proof of process/group shutdown. The caller
                # always drains this owner before publishing a terminal state.
                return
            if len(line) > 16384:
                raise PreservationHostError("mirror_worker_protocol")
            receive(MirrorWorkerEvent.model_validate_json(line))

    async def stop(self) -> None:
        """Cooperatively cancel then drain the actual group, even if cancelled."""
        if self._stopping is None or (
            self._stopping.done()
            and not self._stopping.cancelled()
            and self._stopping.exception() is not None
            and self.process.returncode is None
        ):
            self._stopping = asyncio.create_task(self._drain_owned())
        interrupted = False
        while not self._stopping.done():
            try:
                await asyncio.shield(self._stopping)
            except asyncio.CancelledError:
                interrupted = True
        # Real signal/drain failures propagate and keep this owner retained.
        self._stopping.result()
        if self._transport is not None:
            self._transport.close()
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None and not stream.closed:
                stream.close()
        if interrupted:
            raise asyncio.CancelledError

    async def _drain_owned(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self._control, self._drain)
        # Failed drains retain their executor and owner so explicit Stop may
        # retry. Release it only after the actual process group has drained.
        self._control.shutdown(wait=False)

    def _drain(self) -> None:
        stream = self.process.stdin
        if stream is not None and not stream.closed:
            with suppress(BrokenPipeError, ConnectionResetError):
                stream.write("cancel\n")
                stream.flush()
            with suppress(BrokenPipeError, ConnectionResetError):
                stream.close()
        deadline = time.monotonic() + 8
        while (
            observe_process_exit(self.process) is None and time.monotonic() < deadline
        ):
            time.sleep(0.025)
        terminate_process_group(self.process)


async def remove_mirror_request(path: Path, storage: PreservationIO) -> None:
    """Remove only the drained private command file, never a retained model."""
    await storage.run(lambda: path.unlink(missing_ok=True))
