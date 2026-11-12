"""Supervise one actual native ACE lifetime without confusing completion and custody."""

from __future__ import annotations

import math
import os
import queue
import re
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO, Protocol, final

from pydantic import TypeAdapter

from skulk.shared.types.common import CommandId
from skulk.shared.types.native_music import (
    MAX_NATIVE_MUSIC_RECORD_BYTES,
    NativeMusicCompleted,
    NativeMusicFailed,
    NativeMusicFrame,
    NativeMusicMemoryFrame,
    NativeMusicProgress,
    NativeMusicReady,
    NativeMusicReadyFrame,
    NativeMusicRequest,
    NativeMusicResult,
    NativeMusicServeConfig,
    NativeMusicServeRequest,
    native_music_live_memory_error,
)
from skulk.utils.process_groups import observe_process_exit, terminate_process_group
from skulk.utils.runtime_files import write_private
from skulk.worker.runner.ace_step.process import (
    expected_music_execution,
    verify_music_result,
)
from skulk.worker.runner.ace_step.result_custody import (
    music_result_directory,
    reclaim_result_directory,
    verify_prepared_result_directory,
)
from skulk.worker.runner.ace_step.workspace import NativeMusicWorkspace
from skulk.worker.runner.demucs.process import separation_environment
from skulk.worker.runner.diagnostics import record_native_music_memory

_FRAME = TypeAdapter[NativeMusicFrame](NativeMusicFrame)


@final
class MusicResidentProcess(subprocess.Popen[bytes]):
    """Actual native leader plus an EOF receipt from its independent lock owner."""

    def __init__(
        self,
        command: list[str],
        *,
        descriptor: int,
        source_fence: int | None,
        environment: dict[str, str],
    ) -> None:
        """Create the custody pipe before any native process can create children."""
        read, write = os.pipe()
        self._custody_descriptor: int | None = read
        inherited = (
            (descriptor,) if source_fence is None else (descriptor, source_fence)
        )
        try:
            super().__init__(
                [*command, "--drain-fd", str(write)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                start_new_session=True,
                umask=0o077,
                pass_fds=(*inherited, write),
            )
        except BaseException:
            os.close(read)
            self._custody_descriptor = None
            raise
        finally:
            os.close(write)

    def wait_for_custody(self) -> None:
        """Confirm independent group drain before releasing Core's file ownership.

        A timeout preserves the pipe and owner for retry. The native leader
        closes its copy before loading; only the detached guardian can retain
        the write end after that leader exits.
        """
        descriptor = self._custody_descriptor
        if descriptor is None:
            return
        os.set_blocking(descriptor, False)
        deadline = time.monotonic() + 5
        while True:
            try:
                data = os.read(descriptor, 1)
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "native music guardian custody remains pending"
                    ) from None
                time.sleep(0.025)
                continue
            if data:
                raise ValueError(
                    "native music guardian emitted unexpected custody data"
                )
            os.close(descriptor)
            self._custody_descriptor = None
            return


class MusicResidentSpawn(Protocol):
    """Inject only native process creation, retaining the existing service fence."""

    def __call__(
        self,
        interpreter: Path,
        configuration: Path,
        descriptor: int,
        environment: dict[str, str],
    ) -> MusicResidentProcess:
        """Create a dedicated private session with bounded binary control pipes."""
        ...


def spawn_music_resident(
    interpreter: Path,
    configuration: Path,
    descriptor: int,
    environment: dict[str, str],
    *,
    source_fence: int | None = None,
) -> MusicResidentProcess:
    """Start verified native code, inheriting service and optional source-read locks.

    The parent verifies the exact installed interpreter/import bytes before this
    effect. Its private CLI watches the actual parent and retires its complete
    group after any native/protocol failure or parent loss. Owner-only umask
    applies to the child without mutating the multithreaded Core process.
    """
    return MusicResidentProcess(
        [
            str(interpreter),
            "-I",
            "-B",
            "-m",
            "skulk_ace_step",
            "--serve",
            str(configuration),
            "--parent-pid",
            str(os.getpid()),
        ],
        descriptor=descriptor,
        source_fence=source_fence,
        environment=environment,
    )


class NativeMusicProcessError(RuntimeError):
    """An owned native lifetime failed; local diagnostics stay with its supervisor."""


@final
class ResidentMusic:
    """Serialize one model, retain exact output custody and drain before cleanup.

    Synchronous methods belong off the API event loop. A successful constructor
    takes the already opened workspace from its model adapter. Cancellation or
    any execution failure retires this object; normal runner supervision creates
    a replacement. A successful result is retained independently of this model's
    loader and must be reclaimed only after addressed delivery or cancellation.
    """

    def __init__(
        self,
        interpreter: Path,
        workspace: NativeMusicWorkspace,
        configuration: NativeMusicServeConfig,
        *,
        payload_sha256: str,
        spawn: MusicResidentSpawn = spawn_music_resident,
        resource_guard: Callable[[int], str | None] | None = None,
    ) -> None:
        """Take a prepared owned workspace without spawning or claiming readiness."""
        if re.fullmatch(r"[a-f0-9]{64}", payload_sha256) is None:
            raise ValueError("native music requires its exact installed payload digest")
        workspace.verify()
        if (
            configuration.directory,
            configuration.directory_identity,
            configuration.requests_identity,
            configuration.loader_identity,
            configuration.weights,
        ) != (
            str(workspace.directory),
            workspace.directory_identity,
            workspace.requests_identity,
            workspace.loader_identity,
            workspace.weights,
        ):
            raise ValueError(
                "native music configuration differs from its prepared workspace"
            )
        self.interpreter, self.workspace, self.configuration = (
            interpreter,
            workspace,
            configuration,
        )
        self.payload_sha256 = payload_sha256
        self.spawn, self.resource_guard = spawn, resource_guard
        self.ready: NativeMusicReady | None = None
        self._process: MusicResidentProcess | None = None
        self._actor = threading.Lock()
        self._observations = threading.Lock()
        self._stop_requested = threading.Event()
        self._workspace_open = True
        self._terminals: queue.Queue[NativeMusicFrame | None] = queue.Queue(maxsize=4)
        self._progress: NativeMusicProgress | None = None
        self._reader_error: str | None = None
        self._active: str | None = None
        self._terminal_seen = False
        self._sequence = 0
        self._pending_output: Path | None = None
        self._threads: list[threading.Thread] = []
        self._diagnostics: deque[bytes] = deque(maxlen=24)
        self._last_resource_check = 0.0
        self._memory_frame: tuple[NativeMusicMemoryFrame, float] | None = None
        self._memory_sequence = 0
        self._memory_started_at = 0.0

    @property
    def diagnostic_tail(self) -> str:
        """Return bounded private diagnostics, never a public job/event payload."""
        with self._observations:
            return b"".join(self._diagnostics)[-4096:].decode("utf-8", errors="replace")

    def _terminal(self, frame: NativeMusicFrame | None) -> None:
        try:
            self._terminals.put_nowait(frame)
        except queue.Full:
            with self._observations:
                self._reader_error = "native music terminal queue exceeded its bound"

    def _read_output(self, stream: BinaryIO) -> None:
        initialized = False
        try:
            with stream:
                while line := stream.readline(MAX_NATIVE_MUSIC_RECORD_BYTES + 1):
                    if len(line) > MAX_NATIVE_MUSIC_RECORD_BYTES or not line.endswith(
                        b"\n"
                    ):
                        raise ValueError(
                            "native music response framing exceeded its bound"
                        )
                    frame = _FRAME.validate_json(line)
                    if isinstance(frame, NativeMusicMemoryFrame):
                        process = self._process
                        observed = frame.observation
                        if (
                            process is None
                            or observed.process_id != process.pid
                            or observed.payload_sha256 != self.payload_sha256
                            or observed.device != self.configuration.device
                            or frame.sequence != self._memory_sequence + 1
                        ):
                            raise ValueError(
                                "native music resource observation has foreign ownership or sequence"
                            )
                        with self._observations:
                            self._memory_sequence = frame.sequence
                            self._memory_frame = frame, time.monotonic()
                        record_native_music_memory(observed)
                        continue
                    if isinstance(frame, NativeMusicReadyFrame):
                        if initialized:
                            raise ValueError("native music repeated model readiness")
                        if (
                            self.configuration.memory_budget is not None
                            and self._memory_frame is None
                        ):
                            raise ValueError(
                                "native music readiness preceded its live memory observation"
                            )
                        initialized = True
                        self._terminal(frame)
                        continue
                    if not initialized:
                        if (
                            isinstance(frame, NativeMusicFailed)
                            and frame.request_id is None
                        ):
                            self._terminal(frame)
                            continue
                        if (
                            not isinstance(frame, NativeMusicProgress)
                            or frame.request_id is not None
                            or frame.stage not in {"verifying_weights", "loading"}
                        ):
                            raise ValueError(
                                "native music response preceded model readiness"
                            )
                        with self._observations:
                            self._progress = frame
                        continue
                    identifier = (
                        frame.result.request_id
                        if isinstance(frame, NativeMusicCompleted)
                        else frame.request_id
                    )
                    with self._observations:
                        if (
                            self._active is None
                            or identifier != self._active
                            or self._terminal_seen
                        ):
                            raise ValueError(
                                "native music response has no matching active request"
                            )
                        if isinstance(frame, NativeMusicProgress):
                            if frame.stage not in {
                                "conditioning",
                                "generating",
                                "writing",
                            }:
                                raise ValueError(
                                    "native music request emitted initialization progress"
                                )
                            self._progress = frame
                            continue
                        self._terminal_seen = True
                    self._terminal(frame)
        except (OSError, ValueError):
            with self._observations:
                self._reader_error = "native music response protocol failed"
        finally:
            self._terminal(None)

    def _read_diagnostics(self, stream: BinaryIO) -> None:
        try:
            with stream:
                while block := stream.read(2048):
                    with self._observations:
                        self._diagnostics.append(block)
        except OSError:
            with self._observations:
                self._reader_error = "native music diagnostic pipe failed"

    @staticmethod
    def _deadline(timeout_seconds: float) -> float:
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 6 * 60 * 60:
            raise ValueError("native music needs a finite bounded execution budget")
        return time.monotonic() + timeout_seconds

    def _checkpoint(self, deadline: float, is_cancelled: Callable[[], bool]) -> None:
        if self._stop_requested.is_set() or is_cancelled():
            raise InterruptedError("native music was cancelled")
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError("native music exceeded its execution deadline")
        with self._observations:
            error = self._reader_error
        if error is not None:
            raise NativeMusicProcessError(error)
        process = self._process
        if process is None:
            raise NativeMusicProcessError("native music is not running")
        budget = self.configuration.memory_budget
        if budget is not None:
            with self._observations:
                sample = self._memory_frame
            if sample is None:
                if now - self._memory_started_at > 30:
                    raise NativeMusicProcessError(
                        "native music emitted no initial live memory observation"
                    )
            else:
                frame, received = sample
                if now - received > 5:
                    raise NativeMusicProcessError(
                        "native music live memory heartbeat stopped"
                    )
                refusal = native_music_live_memory_error(
                    budget, frame.observation, now_unix_ms=time.time_ns() // 1_000_000
                )
                if refusal is not None:
                    raise NativeMusicProcessError(refusal)
        if self.resource_guard is not None and now - self._last_resource_check >= 0.5:
            self._last_resource_check = now
            refusal = self.resource_guard(process.pid)
            if refusal is not None:
                raise NativeMusicProcessError(refusal)
        if observe_process_exit(process) is not None:
            raise NativeMusicProcessError("native music exited before completion")

    def _next(
        self,
        deadline: float,
        is_cancelled: Callable[[], bool],
        progress: Callable[[NativeMusicProgress], None] | None,
    ) -> NativeMusicFrame:
        while True:
            self._checkpoint(deadline, is_cancelled)
            if progress is not None:
                with self._observations:
                    latest, self._progress = self._progress, None
                if latest is not None:
                    progress(latest)
            try:
                frame = self._terminals.get(timeout=0.1)
            except queue.Empty:
                continue
            # A terminal may wake get() before this loop observes the coalesced
            # progress slot. Flush its last actual stage before returning, so a
            # fast operation cannot lose every progress observation.
            if progress is not None:
                with self._observations:
                    latest, self._progress = self._progress, None
                if latest is not None:
                    progress(latest)
            if frame is None:
                raise NativeMusicProcessError("native music closed its control stream")
            if isinstance(frame, NativeMusicFailed):
                raise NativeMusicProcessError(
                    "native music failed; inspect bounded runner diagnostics"
                )
            # The blocking dequeue can overlap a new resource sample, caller
            # cancellation or process exit. Recheck before accepting readiness
            # or a terminal result rather than spending the pre-wait snapshot.
            self._checkpoint(deadline, is_cancelled)
            return frame

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
                raise NativeMusicProcessError(
                    "native music control write did not advance"
                )
            remaining = remaining[written:]

    def start(
        self,
        *,
        is_cancelled: Callable[[], bool],
        timeout_seconds: float = 600,
        progress: Callable[[NativeMusicProgress], None] | None = None,
    ) -> NativeMusicReady:
        """Load once and verify exact model/payload/component readiness before return."""
        deadline = self._deadline(timeout_seconds)
        if not self._actor.acquire(blocking=False):
            raise NativeMusicProcessError("native music lifetime is busy")
        try:
            if self.ready is not None:
                self._checkpoint(deadline, is_cancelled)
                return self.ready
            if self._stop_requested.is_set() or not self._workspace_open:
                raise NativeMusicProcessError("native music lifetime must be recreated")
            if is_cancelled():
                raise InterruptedError("native music load was cancelled")
            self.workspace.verify()
            configuration = self.workspace.root / "model.json"
            data = self.configuration.model_dump_json().encode()
            if len(data) > MAX_NATIVE_MUSIC_RECORD_BYTES:
                raise ValueError(
                    "native music model configuration exceeds its control bound"
                )
            write_private(configuration, data)
            self._process = self.spawn(
                self.interpreter,
                configuration,
                self.workspace.descriptor,
                separation_environment(self.configuration.threads),
            )
            self._memory_started_at = time.monotonic()
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
            frame = self._next(deadline, is_cancelled, progress)
            if not isinstance(frame, NativeMusicReadyFrame) or (
                frame.ready.model_identity,
                frame.ready.payload_sha256,
                frame.ready.execution,
            ) != (
                self.configuration.model_identity,
                self.payload_sha256,
                expected_music_execution(self.configuration),
            ):
                raise NativeMusicProcessError(
                    "native music readiness differs from the admitted model/runtime policy"
                )
            self.ready = frame.ready
            return frame.ready
        except BaseException as error:
            self._stop_after_error(error)
            raise
        finally:
            self._actor.release()

    def _stop_after_error(self, error: BaseException) -> None:
        """Drain native custody before retaining bounded private failure context."""
        observed_exit = (
            observe_process_exit(self._process)
            if self._process is not None
            else None
        )
        self._stop()
        # The stderr reader must finish before its final resource-guard message
        # is retained. Repr keeps native control characters out of log framing;
        # bootstrap publishes str(error) as shared startup state, so private
        # context belongs only in traceback notes, never in the public message.
        tail = self.diagnostic_tail
        if isinstance(error, NativeMusicProcessError) and tail:
            error.add_note(
                f"native_exit_before_cleanup={observed_exit}; "
                f"native_stderr_tail={tail!r}"
            )

    def _stop(self) -> None:
        self._stop_requested.set()
        self.ready = None
        if self._process is not None:
            if self._process.returncode is None:
                terminate_process_group(
                    self._process, grace_seconds=1.0, confirm_members=True
                )
            self._process.wait_for_custody()
            process, self._process = self._process, None
            if process.stdin is not None:
                with suppress(BrokenPipeError):
                    process.stdin.close()
        for reader in self._threads:
            if reader.ident is None:
                continue
            reader.join(timeout=2)
            if reader.is_alive():
                raise NativeMusicProcessError(
                    "native music pipe cleanup remains pending"
                )
        self._threads.clear()
        if self._pending_output is not None:
            if not reclaim_result_directory(self._pending_output):
                raise NativeMusicProcessError(
                    "native music partial result cleanup remains reserved"
                )
            self._pending_output = None
        if self._workspace_open:
            if not self.workspace.retire_loading():
                raise NativeMusicProcessError(
                    "native music loader cleanup remains reserved"
                )
            self.workspace.close()
            self._workspace_open = False

    def is_ready(self) -> bool:
        """Poll actual idle liveness; active work keeps its own hard checks."""
        if not self._actor.acquire(blocking=False):
            return self.ready is not None and not self._stop_requested.is_set()
        try:
            if self.ready is None:
                return False
            try:
                self._checkpoint(time.monotonic() + 1, lambda: False)
                if not self._terminals.empty():
                    raise NativeMusicProcessError(
                        "native music idle stream has an unexpected terminal"
                    )
                self.workspace.verify()
            except (
                OSError,
                ValueError,
                InterruptedError,
                TimeoutError,
                NativeMusicProcessError,
            ):
                self._stop()
                return False
            return True
        finally:
            self._actor.release()

    def close(self) -> None:
        """Cancel and join active control, drain the group, then release owned loading.

        A failed drain or filesystem cleanup keeps ownership for another close
        attempt. Completed output remains with its independent result receipt.
        """
        self._stop_requested.set()
        if not self._actor.acquire(timeout=20):
            raise NativeMusicProcessError("native music actor cleanup remains pending")
        try:
            self._stop()
        finally:
            self._actor.release()

    def run(
        self,
        request: NativeMusicRequest,
        *,
        progress: Callable[[NativeMusicProgress], None],
        is_cancelled: Callable[[], bool],
        timeout_seconds: float = 2 * 60 * 60,
    ) -> NativeMusicResult | None:
        """Run one prepared call; return only after independent complete WAV checks.

        Native completion is not an output-return acknowledgement. Success keeps
        files/receipts for addressed transport. Failure/cancel drains the actual
        group before reclaiming this call's partial files or any loader copies.
        A busy caller is refused without cancelling the existing operation.
        """
        deadline = self._deadline(timeout_seconds)
        if not self._actor.acquire(blocking=False):
            raise NativeMusicProcessError(
                "native music model already owns an active call"
            )
        try:
            ready = self.ready
            if ready is None or self._process is None:
                raise NativeMusicProcessError("native music model is not ready")
            self.workspace.verify()
            directory = music_result_directory(
                self.workspace.root, CommandId(request.request_id)
            )
            if directory != Path(request.output_directory):
                raise ValueError(
                    "native music output differs from its fixed instance request path"
                )
            verify_prepared_result_directory(
                directory,
                request.output_directory_identity,
                self.configuration.requests_identity,
            )
            self._pending_output = directory
            self._checkpoint(deadline, is_cancelled)
            self._sequence += 1
            envelope = NativeMusicServeRequest(sequence=self._sequence, request=request)
            data = envelope.model_dump_json().encode() + b"\n"
            if len(data) > MAX_NATIVE_MUSIC_RECORD_BYTES:
                raise ValueError("native music request exceeds its control bound")
            with self._observations:
                self._active = request.request_id
                self._terminal_seen = False
                self._progress = None
            self._send(data, deadline, is_cancelled)
            frame = self._next(deadline, is_cancelled, progress)
            if (
                not isinstance(frame, NativeMusicCompleted)
                or frame.result.request_id != request.request_id
            ):
                raise NativeMusicProcessError(
                    "native music completion belongs to another request"
                )
            result = verify_music_result(
                directory,
                request,
                ready,
                checkpoint=lambda: self._checkpoint(deadline, is_cancelled),
            )
            if result != frame.result:
                raise NativeMusicProcessError(
                    "native music completion differs from its retained files"
                )
            self.workspace.verify()
            self._checkpoint(deadline, is_cancelled)
            self._pending_output = None
            return result
        except BaseException as error:
            self._stop_after_error(error)
            if isinstance(error, InterruptedError):
                return None
            raise
        finally:
            with self._observations:
                self._active = None
                self._progress = None
            self._actor.release()
