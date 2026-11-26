"""One engine package's adapter process, owned by the runner that started it.

The session starts the adapter in its own process group with an explicit
environment (nothing is inherited), speaks the control protocol over its
standard input and output, and stops it. Every call has a deadline. Any
failure of the channel itself (timeout, early exit, a malformed or mismatched
frame) is fatal to the session: the whole process group is terminated and
the session refuses further calls. A typed refusal from the engine is not
fatal; the session stays usable and the caller decides.

Calls are serialized by the caller. The session does not queue them, and it
does not replay one after an uncertain outcome.
"""

import contextlib
import os
import select
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import BinaryIO, Final, NoReturn, TypeVar, final

from skulk.engines.protocol import (
    MAX_CONTROL_FRAME_BYTES,
    PROTOCOL_VERSIONS_SUPPORTED,
    CancelRequest,
    CancelResponse,
    ControlProtocolError,
    DescribeRequest,
    DescribeResponse,
    ErrorCode,
    ErrorResponse,
    EstimateRequest,
    EstimateResponse,
    HealthRequest,
    HealthResponse,
    HelloRequest,
    HelloResponse,
    LoadRequest,
    LoadResponse,
    LoadSettings,
    ModelIdentity,
    ProbeRequest,
    ProbeResponse,
    RequestMessage,
    ResponseMessage,
    StopRequest,
    UnloadRequest,
    UnloadResponse,
    decode_response,
    encode_frame,
)
from skulk.utils.process_groups import observe_process_exit, terminate_process_group

START_DEADLINE_SECONDS: Final = 30.0
"""Time allowed for the adapter to start and answer ``hello``."""

DEFAULT_CALL_DEADLINE_SECONDS: Final = 30.0
"""Time allowed for quick calls: describe, probe, estimate, health, cancel, unload."""

LOAD_DEADLINE_SECONDS: Final = 900.0
"""Time allowed for a load. Large checkpoints take minutes, not seconds."""

_STOP_DEADLINE_SECONDS: Final = 2.0
_TERMINATE_GRACE_SECONDS: Final = 1.0
_SELECT_SLICE_SECONDS: Final = 0.25
_CHUNK_BYTES: Final = 64 * 1024
_LENGTH_BYTES: Final = 4

_Response = TypeVar("_Response", bound=ResponseMessage)


@final
class AdapterRefusedError(Exception):
    """The engine refused a request with a typed reason. The session stays usable."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        """Keep the engine's stable refusal code and its own bounded reason."""
        super().__init__(f"{code}: {message}")
        self.code: ErrorCode = code
        self.reason: str = message


@final
class AdapterFailedError(Exception):
    """The control channel failed. The process group is gone and the session is dead.

    ``reason`` is one of ``exited``, ``timeout``, ``truncated``, ``protocol``,
    ``sequence``, ``identity`` or ``unexpected``. A caller must create a new
    session and may not replay a call whose outcome is uncertain.
    """

    def __init__(self, reason: str, message: str) -> None:
        """Record the failure class and a short description."""
        super().__init__(message)
        self.reason: str = reason


@final
class AdapterSession:
    """A single adapter process and its control channel.

    Construct with an explicit ``environment``; nothing from the host's
    environment is passed on. ``command`` is the complete argument vector,
    for example an isolated interpreter followed by the adapter file. The
    caller keeps ``stderr`` open for the session's lifetime and owns it.
    """

    def __init__(
        self,
        *,
        command: Sequence[str],
        environment: Mapping[str, str],
        working_directory: Path,
        stderr: BinaryIO,
        build_digest: str,
    ) -> None:
        """Prepare, but do not start, an adapter for one installed package build."""
        self._command: tuple[str, ...] = tuple(command)
        self._environment: dict[str, str] = dict(environment)
        self._working_directory: Path = working_directory
        self._stderr: BinaryIO = stderr
        self._build_digest: str = build_digest
        self._process: subprocess.Popen[bytes] | None = None
        self._sequence: int = 0
        self._version: int | None = None
        self._dead: bool = False
        self._closed: bool = False

    @property
    def version(self) -> int | None:
        """The protocol version negotiated at start, or ``None`` before start."""
        return self._version

    @property
    def process_id(self) -> int | None:
        """The adapter's process ID, or ``None`` before start."""
        return None if self._process is None else self._process.pid

    @property
    def alive(self) -> bool:
        """Whether the session can still accept calls."""
        return self._process is not None and not self._dead and not self._closed

    def start(
        self,
        *,
        versions: tuple[int, ...] = PROTOCOL_VERSIONS_SUPPORTED,
        deadline_seconds: float = START_DEADLINE_SECONDS,
    ) -> int:
        """Start the adapter and negotiate the protocol version.

        ``versions`` is what the host offers: the versions this build speaks
        that the package's manifest also declares. The adapter picks one of
        them. Returns the chosen version. Raises ``AdapterFailedError`` when
        the program cannot be started, on any channel fault, or when the
        adapter picks a version that was not offered; the process group is
        terminated in every case.
        """
        if self._process is not None or self._closed:
            raise RuntimeError("adapter session can be started only once")
        try:
            self._process = subprocess.Popen(
                list(self._command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                env=self._environment,
                cwd=self._working_directory,
                start_new_session=True,
                close_fds=True,
            )
        except OSError as error:
            self._dead = True
            raise AdapterFailedError(
                "exited", f"adapter could not be started: {error.strerror}"
            ) from None
        assert self._process.stdin is not None and self._process.stdout is not None
        os.set_blocking(self._process.stdin.fileno(), False)
        os.set_blocking(self._process.stdout.fileno(), False)
        response = self._expect(
            self._exchange(
                HelloRequest(
                    sequence=self._take_sequence(),
                    build_digest=self._build_digest,
                    versions=versions,
                ),
                deadline_seconds,
            ),
            HelloResponse,
        )
        if response.version not in versions:
            self._fail(
                "protocol",
                f"adapter selected version {response.version}, which was not offered",
            )
        self._version = response.version
        return response.version

    def describe(
        self, *, deadline_seconds: float = DEFAULT_CALL_DEADLINE_SECONDS
    ) -> DescribeResponse:
        """Return the package's engine version and the families it claims."""
        return self._expect(
            self._exchange(
                DescribeRequest(
                    sequence=self._take_sequence(), build_digest=self._build_digest
                ),
                deadline_seconds,
            ),
            DescribeResponse,
        )

    def probe(
        self, *, deadline_seconds: float = DEFAULT_CALL_DEADLINE_SECONDS
    ) -> ProbeResponse:
        """Check that the package runs on this node. This never makes a card ready."""
        return self._expect(
            self._exchange(
                ProbeRequest(
                    sequence=self._take_sequence(), build_digest=self._build_digest
                ),
                deadline_seconds,
            ),
            ProbeResponse,
        )

    def estimate(
        self,
        *,
        instance_id: str,
        model: ModelIdentity,
        settings: LoadSettings,
        deadline_seconds: float = DEFAULT_CALL_DEADLINE_SECONDS,
    ) -> EstimateResponse:
        """Return the memory an exact model and settings need. Unknown is a legal answer."""
        return self._expect(
            self._exchange(
                EstimateRequest(
                    sequence=self._take_sequence(),
                    build_digest=self._build_digest,
                    instance_id=instance_id,
                    model=model,
                    settings=settings,
                ),
                deadline_seconds,
            ),
            EstimateResponse,
        )

    def load(
        self,
        *,
        instance_id: str,
        model: ModelIdentity,
        settings: LoadSettings,
        deadline_seconds: float = LOAD_DEADLINE_SECONDS,
    ) -> LoadResponse:
        """Load a model. The caller must compare ``applied`` with ``settings``.

        A difference is a failed load: the engine took settings other than
        the ones admitted. The caller stops the instance.
        """
        return self._expect(
            self._exchange(
                LoadRequest(
                    sequence=self._take_sequence(),
                    build_digest=self._build_digest,
                    instance_id=instance_id,
                    model=model,
                    settings=settings,
                ),
                deadline_seconds,
            ),
            LoadResponse,
        )

    def health(
        self,
        *,
        instance_id: str,
        deadline_seconds: float = DEFAULT_CALL_DEADLINE_SECONDS,
    ) -> HealthResponse:
        """Ask whether a loaded instance is still serving."""
        return self._expect(
            self._exchange(
                HealthRequest(
                    sequence=self._take_sequence(),
                    build_digest=self._build_digest,
                    instance_id=instance_id,
                ),
                deadline_seconds,
            ),
            HealthResponse,
        )

    def cancel(
        self,
        *,
        instance_id: str,
        request_id: str,
        deadline_seconds: float = DEFAULT_CALL_DEADLINE_SECONDS,
    ) -> CancelResponse:
        """Cancel one request. The answer comes only after the engine acknowledges."""
        return self._expect(
            self._exchange(
                CancelRequest(
                    sequence=self._take_sequence(),
                    build_digest=self._build_digest,
                    instance_id=instance_id,
                    request_id=request_id,
                ),
                deadline_seconds,
            ),
            CancelResponse,
        )

    def unload(
        self,
        *,
        instance_id: str,
        deadline_seconds: float = DEFAULT_CALL_DEADLINE_SECONDS,
    ) -> UnloadResponse:
        """Release an instance's memory. Reported separately from cancellation."""
        return self._expect(
            self._exchange(
                UnloadRequest(
                    sequence=self._take_sequence(),
                    build_digest=self._build_digest,
                    instance_id=instance_id,
                ),
                deadline_seconds,
            ),
            UnloadResponse,
        )

    def close(self) -> None:
        """Stop the adapter, then terminate its whole process group.

        Idempotent. A graceful stop is attempted first on a healthy session;
        the group is terminated regardless, so engine children cannot outlive
        the session. An unconfirmed group drain raises as
        ``terminate_process_group`` does and keeps its leader unreaped.
        """
        if self._closed:
            return
        process = self._process
        if process is None:
            self._closed = True
            return
        if not self._dead:
            # A failed graceful stop is expected; the group is terminated below.
            # The session is marked closed only afterwards, or the stop itself
            # would be refused as a call on a closed session.
            with contextlib.suppress(
                AdapterFailedError, AdapterRefusedError, ControlProtocolError
            ):
                self._exchange(
                    StopRequest(
                        sequence=self._take_sequence(), build_digest=self._build_digest
                    ),
                    _STOP_DEADLINE_SECONDS,
                )
        self._closed = True
        self._dead = True
        for pipe in (process.stdin, process.stdout):
            if pipe is not None:
                pipe.close()
        # A failed session already drained its group and reaped the leader; an
        # unreaped leader still owns the group id, so only then is it signalled.
        if process.returncode is None:
            terminate_process_group(
                process,
                grace_seconds=_TERMINATE_GRACE_SECONDS,
                confirm_members=True,
            )

    def _take_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _expect(self, response: ResponseMessage, kind: type[_Response]) -> _Response:
        if isinstance(response, kind):
            return response
        self._fail(
            "unexpected",
            f"adapter answered {response.type!r} where {kind.__name__} was due",
        )

    def _exchange(
        self, request: RequestMessage, deadline_seconds: float
    ) -> ResponseMessage:
        if not self.alive:
            raise AdapterFailedError("exited", "adapter session is not running")
        deadline = time.monotonic() + deadline_seconds
        try:
            self._write_all(encode_frame(request), deadline)
            header = self._read_exact(_LENGTH_BYTES, deadline)
            if header is None:
                raise AdapterFailedError("exited", "adapter closed its channel")
            length = int.from_bytes(header, "big")
            if length > MAX_CONTROL_FRAME_BYTES:
                raise AdapterFailedError(
                    "protocol", f"adapter announced a {length}-byte frame"
                )
            body = self._read_exact(length, deadline)
            if body is None:
                raise AdapterFailedError(
                    "truncated", "adapter closed its channel inside a frame"
                )
            try:
                response = decode_response(body)
            except ControlProtocolError as error:
                raise AdapterFailedError("protocol", str(error)) from None
            if response.sequence != request.sequence:
                raise AdapterFailedError(
                    "sequence",
                    f"adapter answered sequence {response.sequence} to request {request.sequence}",
                )
            if response.build_digest != self._build_digest:
                raise AdapterFailedError(
                    "identity", "adapter answered for a different package build"
                )
        except AdapterFailedError as error:
            self._fail(error.reason, str(error))
        if isinstance(response, ErrorResponse):
            raise AdapterRefusedError(response.code, response.message)
        return response

    def _fail(self, reason: str, message: str) -> NoReturn:
        """Terminate the group and mark the session dead, then raise."""
        self._dead = True
        if self._process is not None:
            terminate_process_group(
                self._process,
                grace_seconds=_TERMINATE_GRACE_SECONDS,
                confirm_members=True,
            )
        raise AdapterFailedError(reason, message)

    def _write_all(self, data: bytes, deadline: float) -> None:
        assert self._process is not None and self._process.stdin is not None
        descriptor = self._process.stdin.fileno()
        view = memoryview(data)
        while view:
            self._wait(descriptor, writable=True, deadline=deadline)
            try:
                written = os.write(descriptor, view[:_CHUNK_BYTES])
            except BlockingIOError:
                continue
            except BrokenPipeError:
                raise AdapterFailedError(
                    "exited", "adapter stopped reading its channel"
                ) from None
            view = view[written:]

    def _read_exact(self, count: int, deadline: float) -> bytes | None:
        """Return ``count`` bytes, or ``None`` on a clean end before the first byte."""
        assert self._process is not None and self._process.stdout is not None
        descriptor = self._process.stdout.fileno()
        buffer = bytearray()
        while len(buffer) < count:
            self._wait(descriptor, writable=False, deadline=deadline)
            try:
                chunk = os.read(descriptor, count - len(buffer))
            except BlockingIOError:
                continue
            if not chunk:
                if not buffer:
                    return None
                raise AdapterFailedError(
                    "truncated", "adapter closed its channel inside a frame"
                )
            buffer.extend(chunk)
        return bytes(buffer)

    def _wait(self, descriptor: int, *, writable: bool, deadline: float) -> None:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AdapterFailedError(
                    "timeout", "adapter did not answer within its deadline"
                )
            slice_seconds = min(remaining, _SELECT_SLICE_SECONDS)
            if writable:
                _, ready, _ = select.select([], [descriptor], [], slice_seconds)
            else:
                ready, _, _ = select.select([descriptor], [], [], slice_seconds)
            if ready:
                return
            if (
                self._process is not None
                and observe_process_exit(self._process) is not None
            ):
                # The leader exited; anything still readable is drained first.
                # An exit with nothing left surfaces as EOF on the next read.
                if not writable:
                    ready, _, _ = select.select([descriptor], [], [], 0)
                    if ready:
                        return
                raise AdapterFailedError("exited", "adapter process exited")
