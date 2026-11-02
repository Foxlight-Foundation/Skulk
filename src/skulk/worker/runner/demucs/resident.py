"""One owned native model lifetime with bounded, correlated serial job control."""

from __future__ import annotations

import math
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO, Protocol, final
from uuid import uuid4

from pydantic import TypeAdapter

from skulk.shared.types.audio_separation import (
    NativeSeparationCompleted,
    NativeSeparationFailed,
    NativeSeparationFrame,
    NativeSeparationJob,
    NativeSeparationProgress,
    NativeSeparationReady,
    NativeSeparationResidentProgress,
    NativeSeparationResult,
    NativeSeparationServeConfig,
    NativeSeparationServeRequest,
)
from skulk.utils.process_groups import observe_process_exit, terminate_process_group
from skulk.utils.runtime_files import RuntimeLock, private_directory, write_private
from skulk.worker.runner.demucs.process import (
    separation_environment,
    verify_native_result,
)

_FRAME = TypeAdapter[NativeSeparationFrame](NativeSeparationFrame)
_RECORD_BYTES = 16 * 1024


class ResidentSpawn(Protocol):
    """Inject only native process creation; production uses a verified interpreter."""

    def __call__(
        self,
        interpreter: Path,
        configuration: Path,
        descriptor: int,
        environment: dict[str, str],
    ) -> subprocess.Popen[bytes]:
        """Start a private session and inherit its lifetime filesystem fence."""
        ...


def spawn_resident(
    interpreter: Path,
    configuration: Path,
    descriptor: int,
    environment: dict[str, str],
) -> subprocess.Popen[bytes]:
    """Launch the fixed installed resident CLI, never a caller-supplied command."""
    return subprocess.Popen(
        [
            str(interpreter),
            "-I",
            "-B",
            "-m",
            "skulk_demucs",
            "--serve",
            str(configuration),
            "--parent-pid",
            str(os.getpid()),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        start_new_session=True,
        pass_fds=(descriptor,),
    )


class ResidentSeparationError(RuntimeError):
    """The exact native lifetime failed; diagnostics remain local to its owner."""


@final
class ResidentSeparator:
    """Serialize one loaded model, with explicit cancellation and retained cleanup.

    Calls are synchronous and belong off an API event loop. Cancellation stops
    the complete native lifetime, after which normal runner supervision must
    create a new object. Successful outputs remain for addressed delivery. A
    failed process drain or busy close preserves ownership and files for retry.
    """

    def __init__(
        self,
        interpreter: Path,
        root: Path,
        configuration: NativeSeparationServeConfig,
        *,
        payload_sha256: str,
        weights_sha256: str,
        spawn: ResidentSpawn = spawn_resident,
        resource_guard: Callable[[int], str | None] | None = None,
    ) -> None:
        """Select owned resources without starting a process or advertising readiness."""
        if any(
            re.fullmatch(r"[a-f0-9]{64}", value) is None
            for value in (payload_sha256, weights_sha256)
        ):
            raise ValueError(
                "native lifetime requires exact payload and weight digests"
            )
        self.interpreter = interpreter
        self.root = root
        self.jobs = root / "jobs"
        self.configuration = configuration
        self.payload_sha256 = payload_sha256
        self.weights_sha256 = weights_sha256
        self.spawn = spawn
        self.resource_guard = resource_guard
        self.ready: NativeSeparationReady | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._ownership: RuntimeLock | None = None
        self._actor = threading.Lock()
        self._observations = threading.Lock()
        self._stop_requested = threading.Event()
        self._terminals: queue.Queue[NativeSeparationFrame | None] = queue.Queue(
            maxsize=4
        )
        self._progress: NativeSeparationResidentProgress | None = None
        self._reader_error: str | None = None
        self._active: str | None = None
        self._sequence = 0
        self._pending_output: Path | None = None
        self._threads: list[threading.Thread] = []
        self._diagnostics: deque[bytes] = deque(maxlen=24)
        self._last_resource_check = 0.0

    @property
    def diagnostic_tail(self) -> str:
        """Return bounded private stderr for local diagnostics, never job/event payloads."""
        with self._observations:
            return b"".join(self._diagnostics)[-4096:].decode("utf-8", errors="replace")

    def _terminal(self, record: NativeSeparationFrame | None) -> None:
        try:
            self._terminals.put_nowait(record)
        except queue.Full:
            with self._observations:
                self._reader_error = "native terminal queue exceeded its bound"

    def _read_output(self, stream: BinaryIO) -> None:
        saw_ready = False
        try:
            with stream:
                while line := stream.readline(_RECORD_BYTES + 1):
                    if len(line) > _RECORD_BYTES or not line.endswith(b"\n"):
                        raise ValueError("native response framing exceeded its bound")
                    frame = _FRAME.validate_json(line)
                    if isinstance(frame, NativeSeparationReady):
                        if saw_ready:
                            raise ValueError("native lifetime repeated readiness")
                        saw_ready = True
                        self._terminal(frame)
                        continue
                    if not saw_ready:
                        if isinstance(frame, NativeSeparationFailed):
                            self._terminal(frame)
                            continue
                        raise ValueError("native job response preceded model readiness")
                    with self._observations:
                        if frame.request_id != self._active:
                            raise ValueError("native frame belongs to another request")
                        if isinstance(frame, NativeSeparationResidentProgress):
                            self._progress = frame
                            continue
                    self._terminal(frame)
        except (OSError, ValueError):
            with self._observations:
                self._reader_error = "native response protocol failed"
        finally:
            self._terminal(None)

    def _read_diagnostics(self, stream: BinaryIO) -> None:
        with stream:
            while block := stream.read(2048):
                with self._observations:
                    self._diagnostics.append(block)

    def _checkpoint(self, deadline: float, is_cancelled: Callable[[], bool]) -> None:
        if self._stop_requested.is_set() or is_cancelled():
            raise InterruptedError("native separation was cancelled")
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError("native separation exceeded its deadline")
        with self._observations:
            error = self._reader_error
        if error is not None:
            raise ResidentSeparationError(error)
        process = self._process
        if process is None:
            raise ResidentSeparationError("native separation is not running")
        if self.resource_guard is not None and now - self._last_resource_check >= 0.5:
            self._last_resource_check = now
            refusal = self.resource_guard(process.pid)
            if refusal is not None:
                raise ResidentSeparationError(refusal)
        # Keep the leader unreaped until the complete group has been drained.
        if observe_process_exit(process) is not None:
            raise ResidentSeparationError("native separation exited before completion")

    def _next(
        self,
        deadline: float,
        is_cancelled: Callable[[], bool],
        progress: Callable[[NativeSeparationProgress], None] | None = None,
    ) -> NativeSeparationFrame:
        while True:
            self._checkpoint(deadline, is_cancelled)
            if progress is not None:
                with self._observations:
                    latest, self._progress = self._progress, None
                if latest is not None:
                    progress(latest.progress)
            try:
                frame = self._terminals.get(timeout=0.1)
            except queue.Empty:
                continue
            if frame is None:
                raise ResidentSeparationError(
                    "native separation closed its control stream"
                )
            if isinstance(frame, NativeSeparationFailed):
                raise ResidentSeparationError(
                    "native request failed; inspect runner diagnostics"
                )
            return frame

    @staticmethod
    def _deadline(timeout_seconds: float) -> float:
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 6 * 60 * 60:
            raise ValueError("native lifetime needs a finite bounded execution budget")
        return time.monotonic() + timeout_seconds

    def _send(
        self, data: bytes, deadline: float, is_cancelled: Callable[[], bool]
    ) -> None:
        process = self._process
        assert process is not None and process.stdin is not None
        descriptor = process.stdin.fileno()
        os.set_blocking(descriptor, False)
        remaining = memoryview(data)
        while remaining:
            self._checkpoint(deadline, is_cancelled)
            try:
                written = os.write(descriptor, remaining)
            except BlockingIOError:
                time.sleep(0.02)
                continue
            if written == 0:
                raise ResidentSeparationError("native control write did not advance")
            remaining = remaining[written:]

    def start(
        self, *, is_cancelled: Callable[[], bool], timeout_seconds: float = 600.0
    ) -> NativeSeparationReady:
        """Load and verify one actual model before returning an exact ready receipt."""
        deadline = self._deadline(timeout_seconds)
        if not self._actor.acquire(blocking=False):
            raise ResidentSeparationError("native lifetime is busy")
        try:
            if self.ready is not None:
                self._checkpoint(deadline, is_cancelled)
                return self.ready
            if self._stop_requested.is_set() or self._ownership is not None:
                raise ResidentSeparationError("native lifetime must be recreated")
            if is_cancelled():
                raise InterruptedError("native load was cancelled")
            self._ownership = RuntimeLock(self.root, name="service.lock")
            private_directory(self.jobs)
            configuration = self.root / "model.json"
            write_private(configuration, self.configuration.model_dump_json().encode())
            self._process = self.spawn(
                self.interpreter,
                configuration,
                self._ownership.descriptor,
                separation_environment(self.configuration.threads),
            )
            process = self._process
            assert process.stdout is not None and process.stderr is not None
            self._threads = [
                threading.Thread(
                    target=self._read_output, args=(process.stdout,), daemon=True
                ),
                threading.Thread(
                    target=self._read_diagnostics, args=(process.stderr,), daemon=True
                ),
            ]
            for reader in self._threads:
                reader.start()
            frame = self._next(deadline, is_cancelled)
            if not isinstance(frame, NativeSeparationReady) or (
                frame.payload_sha256,
                frame.weights_sha256,
                frame.device,
                frame.threads,
            ) != (
                self.payload_sha256,
                self.weights_sha256,
                self.configuration.device,
                self.configuration.threads,
            ):
                raise ResidentSeparationError(
                    "native ready receipt differs from admitted policy"
                )
            self.ready = frame
            return frame
        except BaseException:
            self._stop()
            raise
        finally:
            self._actor.release()

    def _stop(self) -> None:
        self._stop_requested.set()
        self.ready = None
        if self._process is not None:
            terminate_process_group(
                self._process, grace_seconds=1.0, confirm_members=True
            )
            process, self._process = self._process, None
            if process.stdin is not None:
                with suppress(BrokenPipeError):
                    process.stdin.close()
        for reader in self._threads:
            if reader.ident is None:
                continue
            reader.join(timeout=2.0)
            if reader.is_alive():
                raise ResidentSeparationError("native reader cleanup remains pending")
        self._threads.clear()
        if self._pending_output is not None:
            shutil.rmtree(self._pending_output)
            self._pending_output = None
        if self._ownership is not None:
            self._ownership.close()
            self._ownership = None

    def is_ready(self) -> bool:
        """Check idle native liveness; an active control call owns its own checks.

        A dead or protocol-failed idle process is drained before returning false.
        The worker must poll this during ordinary runner supervision rather than
        advertising the last ready receipt after an independent native crash.
        """
        if not self._actor.acquire(blocking=False):
            return self.ready is not None and not self._stop_requested.is_set()
        try:
            if self.ready is None:
                return False
            try:
                self._checkpoint(time.monotonic() + 1.0, lambda: False)
                if not self._terminals.empty():
                    raise ResidentSeparationError(
                        "native idle stream has an unexpected terminal"
                    )
            except (InterruptedError, TimeoutError, ResidentSeparationError):
                self._stop()
                return False
            return True
        finally:
            self._actor.release()

    def close(self) -> None:
        """Cancel, join the active control call, then drain before releasing its fence."""
        self._stop_requested.set()
        if not self._actor.acquire(timeout=20.0):
            raise ResidentSeparationError("native owner cleanup remains pending")
        try:
            self._stop()
        finally:
            self._actor.release()

    def run(
        self,
        job: NativeSeparationJob,
        *,
        progress: Callable[[NativeSeparationProgress], None],
        is_cancelled: Callable[[], bool],
        timeout_seconds: float = 6 * 60 * 60,
    ) -> NativeSeparationResult | None:
        """Run one bounded job; failure/cancel drains the resident before file cleanup.

        Successful verified files remain for addressed transport. A concurrent
        caller is refused without cancelling the existing owner's work.
        """
        deadline = self._deadline(timeout_seconds)
        if not self._actor.acquire(blocking=False):
            raise ResidentSeparationError("native model already owns an active job")
        try:
            ready, process = self.ready, self._process
            if ready is None or process is None or process.stdin is None:
                raise ResidentSeparationError("native model is not ready")
            requested = Path(job.output_dir)
            if (
                requested.parent.resolve() != self.jobs.resolve()
                or re.fullmatch(r"[a-f0-9]{32}", requested.name) is None
            ):
                raise ValueError("native result must use a new owned job directory")
            if (job.weights_path, job.device, job.threads) != (
                self.configuration.weights_path,
                self.configuration.device,
                self.configuration.threads,
            ):
                raise ValueError("native request differs from the loaded policy")
            self._checkpoint(deadline, is_cancelled)
            requested.mkdir(mode=0o700, exist_ok=False)
            directory = requested
            self._pending_output = directory
            self._sequence += 1
            request = NativeSeparationServeRequest(
                request_id=uuid4().hex, sequence=self._sequence, job=job
            )
            encoded = request.model_dump_json().encode() + b"\n"
            if len(encoded) > _RECORD_BYTES:
                raise ValueError("native job exceeds its control bound")
            with self._observations:
                self._active = request.request_id
                self._progress = None
            self._send(encoded, deadline, is_cancelled)
            frame = self._next(deadline, is_cancelled, progress)
            if (
                not isinstance(frame, NativeSeparationCompleted)
                or frame.request_id != request.request_id
            ):
                raise ResidentSeparationError(
                    "native completion belongs to another request"
                )
            measured = verify_native_result(
                directory,
                job,
                checkpoint=lambda: self._checkpoint(deadline, is_cancelled),
            )
            if measured != frame.result or any(
                stem.sample_rate != ready.sample_rate for stem in measured.output.stems
            ):
                raise ResidentSeparationError(
                    "native completion differs from verified files"
                )
            self._checkpoint(deadline, is_cancelled)
            self._pending_output = None
            return measured
        except BaseException as error:
            self._stop()
            if isinstance(error, InterruptedError):
                return None
            raise
        finally:
            with self._observations:
                self._active = None
                self._progress = None
            self._actor.release()
