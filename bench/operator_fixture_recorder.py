"""Lossless bounded pipe to the pinned relay aggregate recorder, never a trace file."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast, final

from pydantic import BaseModel, ConfigDict, Field

from bench.operator_fixture_observer import FixtureObservationError, ObservationEvent


class RecorderSettings(BaseModel):
    """Explicit local Node executable and digest-pinned compiled recorder modules."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    node_binary: Path = Field(
        description="Existing local Node executable; never downloaded."
    )
    recorder_module: Path = Field(
        description="Compiled workload-observation.js source."
    )
    recorder_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Expected recorder digest."
    )
    cli_module: Path = Field(description="Compiled workload-observation-cli.js source.")
    cli_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Expected CLI digest."
    )


def verified_recorder_copy(source: Path, digest: str, destination: Path) -> None:
    """Write digest-verified module bytes to a new protected file.

    `source` is read once with a two-MiB bound; `digest` must match those bytes.
    `destination` must not already exist and receives exactly the hashed bytes,
    with owner-only permissions. Returns nothing; raises on mismatches or I/O.
    """
    with source.open("rb") as stream:
        contents = stream.read(2 * 1024**2 + 1)
    if len(contents) > 2 * 1024**2 or hashlib.sha256(contents).hexdigest() != digest:
        raise FixtureObservationError
    # Execute the same bounded bytes that were hashed, not a mutable build path.
    with destination.open("xb") as stream:
        destination.chmod(0o600)
        stream.write(contents)


@final
class FixtureRecorder:
    """Own a bounded event queue and aggregate subprocess for one observation.

    The synchronous `record` sink raises on overflow; it never silently drops
    events. Only `finish` returns evidence, after every required flow has been
    accepted by the recorder. Raw metadata exists only in bounded memory/pipes.
    """

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        """Attach to an owned local recorder process with stdin/stdout pipes."""
        self._process = process
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=512)
        self._invalid = False
        self._finishing = False
        self._finished = False
        self._feed_task: asyncio.Task[None] | None = None
        self._output_task: asyncio.Task[bytes] | None = None

    def _reject(self) -> None:
        self._invalid = True
        raise FixtureObservationError

    def record(self, event: ObservationEvent) -> None:
        """Queue one metadata event; reject after failure/finish or at queue limits."""
        if self._invalid or self._finishing or self._process.returncode is not None:
            self._reject()
        if len(event) > 8 or any(
            len(key) > 32
            or type(value) not in (str, int)
            or (isinstance(value, str) and len(value) > 64)
            or (type(value) is int and not 0 <= value <= 10 * 1024**3)
            for key, value in event.items()
        ):
            self._reject()
        try:
            line = json.dumps(
                dict(event), separators=(",", ":"), allow_nan=False
            ).encode()
            if len(line) > 512:
                self._reject()
            self._queue.put_nowait(line + b"\n")
        except (ValueError, TypeError, asyncio.QueueFull):
            self._invalid = True
            raise FixtureObservationError from None

    async def _feed(self) -> None:
        assert self._process.stdin is not None
        try:
            while (line := await self._queue.get()) is not None:
                self._process.stdin.write(line)
                await self._process.stdin.drain()
            self._process.stdin.close()
            await self._process.stdin.wait_closed()
        except (ConnectionError, OSError):
            self._invalid = True
            raise FixtureObservationError from None

    async def _output(self) -> bytes:
        assert self._process.stdout is not None
        result = bytearray()
        while chunk := await self._process.stdout.read(16384):
            if len(result) + len(chunk) > 262144:
                self._reject()
            result.extend(chunk)
        if not self._finishing or await self._process.wait() != 0:
            self._reject()
        try:
            decoded = cast(object, json.loads(result))
            if not isinstance(decoded, dict):
                self._reject()
            document = cast(dict[str, object], decoded)
            if (
                document.get("schema") != "relay-workload-observation.v1"
                or document.get("evidence") != "unattested-aggregate"
            ):
                self._reject()
        except (ValueError, TypeError):
            self._invalid = True
            raise FixtureObservationError from None
        return bytes(result)

    def start(self, group: asyncio.TaskGroup) -> None:
        """Start owned pipe tasks; failures cancel the enclosing capture scope."""
        if self._feed_task is not None:
            self._reject()
        self._feed_task = group.create_task(self._feed())
        self._output_task = group.create_task(self._output())

    async def finish(self) -> bytes:
        """Return bounded aggregate JSON after recorder validation; close input.

        Requires all seven flow categories, with no active sockets or requests.
        A five-second drain/exit deadline prevents a stuck sink from yielding
        partial evidence. Calling more than once is invalid.
        """
        if self._invalid or self._finishing:
            self._reject()
        self._finishing = True
        assert self._feed_task is not None and self._output_task is not None
        try:
            async with asyncio.timeout(5):
                await self._queue.put(None)
                await self._feed_task
                result = await self._output_task
        except TimeoutError:
            self._invalid = True
            raise FixtureObservationError from None
        if self._invalid:
            self._reject()
        self._finished = True
        return result

    def cancel(self) -> None:
        """Stop pending pipe tasks when the enclosing scope exits or is cancelled."""
        for task in (self._feed_task, self._output_task):
            if task is not None:
                task.cancel()

    def require_finished(self) -> None:
        """Reject a normal exit that did not validate complete aggregate evidence."""
        if not self._finished or self._invalid:
            self._reject()


@asynccontextmanager
async def aggregate_recorder(
    settings: RecorderSettings,
) -> AsyncIterator[FixtureRecorder]:
    """Run a pinned local aggregate recorder, then always reap it and remove copies.

    `settings` selects existing modules and their SHA-256 digests. Both modules
    are copied from the verified bytes to a protected ephemeral directory.
    The context yields the lossless sink and never writes request metadata or
    aggregate evidence to disk. The caller chooses where final aggregate JSON
    belongs. No package install, provider resource, or production target exists.
    """
    with TemporaryDirectory(prefix="skulk-fixture-recorder-") as temporary:
        directory = Path(temporary)
        verified_recorder_copy(
            settings.recorder_module,
            settings.recorder_sha256,
            directory / "workload-observation.js",
        )
        verified_recorder_copy(
            settings.cli_module,
            settings.cli_sha256,
            directory / "workload-observation-cli.mjs",
        )
        (directory / "package.json").write_text('{"type":"module"}')
        process = await asyncio.create_subprocess_exec(
            str(settings.node_binary.resolve(strict=True)),
            str(directory / "workload-observation-cli.mjs"),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=16384,
        )
        recorder = FixtureRecorder(process)
        try:
            async with asyncio.TaskGroup() as group:
                recorder.start(group)
                try:
                    yield recorder
                    recorder.require_finished()
                finally:
                    recorder.cancel()
        finally:
            if process.returncode is None:
                process.kill()
            await process.wait()
