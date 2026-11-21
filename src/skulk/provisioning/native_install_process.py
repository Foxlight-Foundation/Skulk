"""Cancellable native bootstrap commands with bounded output and owned groups."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import BinaryIO, final
from uuid import uuid4

from skulk.utils.process_groups import observe_process_exit, terminate_process_group

_OUTPUT_LIMIT = 128 * 1024


@final
class NativeInstallCommandRunner:
    """Execute serialized bootstrap commands without leaving orphaned installers.

    Each invocation owns a new session. A trusted standard-library guardian
    retains its leader and optional build-lock descriptor until all descendants
    are drained, and kills the session if the controller disappears. Diagnostic
    streams and returned stdout are bounded; no caller payload is interpreted.
    """

    def __init__(
        self,
        diagnostics: Path,
        *,
        checkpoint: Callable[[], None],
        lock_descriptor: int | None = None,
        guardian_interpreter: Path | None = None,
    ) -> None:
        """Select private logs, the owner's cancellation hook and inherited lock.

        The owner must keep the descriptor open for the runner's lifetime and
        serialize calls. The checkpoint may raise cancellation or a total-stage
        deadline; such exceptions propagate after the complete process drain.
        A frozen Core must explicitly supply a separately verified Python for
        the guardian; its own executable is an application launcher, not Python.
        """
        if guardian_interpreter is None and getattr(sys, "frozen", False):
            raise ValueError("frozen Core requires a verified guardian interpreter")
        diagnostics.mkdir(mode=0o700, parents=True, exist_ok=True)
        if (
            diagnostics.is_symlink()
            or not diagnostics.is_dir()
            or diagnostics.stat().st_uid != os.getuid()
            or stat.S_IMODE(diagnostics.stat().st_mode) & 0o077
        ):
            raise ValueError("native installation diagnostics are redirected")
        self._diagnostics = diagnostics.resolve() / uuid4().hex
        self._diagnostics.mkdir(mode=0o700)
        self._checkpoint = checkpoint
        self._lock_descriptor = lock_descriptor
        self._guardian_interpreter = str(guardian_interpreter or sys.executable)
        self._sequence = 0
        self._pending_process: subprocess.Popen[bytes] | None = None

    @property
    def quiescent(self) -> bool:
        """Return whether every command group is confirmed stopped.

        A preparation owner must preserve installation files when this is false.
        Retain this runner and use drain() to retry cleanup with its original
        unreaped process handle; a numeric PID from a local record is inadequate.
        """
        return self._pending_process is None

    def drain(self) -> None:
        """Stop a retained owned group, clearing ownership only after confirmation.

        Cleanup failure preserves the unreaped handle and inherited lock for a
        subsequent retry. No further command may run until this method succeeds.
        """
        if self._pending_process is not None:
            terminate_process_group(
                self._pending_process, grace_seconds=0.1, confirm_members=True
            )
            self._pending_process = None

    def __call__(
        self,
        arguments: Sequence[str],
        *,
        environment: dict[str, str],
        timeout_seconds: int,
        capture: bool,
    ) -> str:
        """Run one command, retaining bounded private logs before returning.

        Raise on cancellation, timeout, excessive output or nonzero exit. Never
        clean the caller's installation files; the caller may remove them only
        after this method confirms its complete process group has drained.
        """
        self._checkpoint()
        if not self.quiescent:
            raise RuntimeError("earlier native install command still requires drain")
        if not arguments or timeout_seconds <= 0:
            raise ValueError("native install command or deadline is invalid")
        self._sequence += 1
        if self._sequence > 32:
            raise ValueError("native install command count exceeds its bound")
        output = bytearray()
        errors = bytearray()
        overflow = threading.Event()
        readers: list[threading.Thread] = []
        process: subprocess.Popen[bytes] | None = None
        completed = bytearray()
        completion_read, completion_write = os.pipe()
        os.set_blocking(completion_read, False)

        def consume(stream: BinaryIO, destination: bytearray) -> None:
            with stream:
                while block := stream.read(16 * 1024):
                    remaining = _OUTPUT_LIMIT - len(destination)
                    destination.extend(block[:remaining])
                    if len(block) > remaining:
                        overflow.set()

        started = time.monotonic()
        try:
            lock = self._lock_descriptor
            descriptors = (completion_write,) + (() if lock is None else (lock,))
            # Guardian source belongs to Core. A packaged controller supplies
            # its verified standalone Python, never reexecutes an app launcher.
            # Neither Core's import path nor dependency set is injected.
            process = subprocess.Popen(
                [
                    self._guardian_interpreter,
                    "-I",
                    "-B",
                    str(Path(__file__).with_name("_native_install_guard.py")),
                    str(os.getpid()),
                    str(completion_write),
                    str(-1 if lock is None else lock),
                    *arguments,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # Native helpers can re-exec Python through console scripts;
                # -B applies only to the explicit interpreter, not descendants.
                # Keep their imports from changing the independently sealed tree.
                env=environment | {"PYTHONDONTWRITEBYTECODE": "1"},
                start_new_session=True,
                pass_fds=descriptors,
            )
            self._pending_process = process
            os.close(completion_write)
            completion_write = -1
            assert process.stdout is not None and process.stderr is not None
            readers = [
                threading.Thread(
                    target=consume, args=(process.stdout, output), daemon=True
                ),
                threading.Thread(
                    target=consume, args=(process.stderr, errors), daemon=True
                ),
            ]
            for reader in readers:
                reader.start()
            while len(completed) < 4:
                self._checkpoint()
                if overflow.is_set():
                    raise ValueError("native install command output exceeds its bound")
                if time.monotonic() - started >= timeout_seconds:
                    raise TimeoutError("native install command exceeded its deadline")
                try:
                    block = os.read(completion_read, 5 - len(completed))
                except BlockingIOError:
                    block = None
                if block == b"" or observe_process_exit(process) is not None:
                    raise ValueError("native install guardian exited before completion")
                if block is not None:
                    completed.extend(block)
                if len(completed) > 4:
                    raise ValueError("native install completion exceeds its bound")
                time.sleep(0.025)
            # Do not reap the guardian before draining its descendants. Its
            # inherited lock remains held until the owner begins this drain.
            self.drain()
            process = None
            for reader in readers:
                reader.join(timeout=2)
            if any(reader.is_alive() for reader in readers):
                raise ValueError(
                    "native install output did not finish after process drain"
                )
            self._checkpoint()
            if overflow.is_set():
                raise ValueError("native install command output exceeds its bound")
            status = int.from_bytes(completed, byteorder="big", signed=True)
            if status != 0:
                raise ValueError(
                    f"native install command failed ({status}): "
                    + errors[-4096:].decode("utf-8", errors="replace")
                )
            return output.decode("utf-8") if capture else ""
        finally:
            if completion_write >= 0:
                os.close(completion_write)
            os.close(completion_read)
            if not self.quiescent:
                # A failed drain propagates and preserves owned installation
                # files. Cleanup must never race an installer still writing.
                self.drain()
            for reader in readers:
                reader.join(timeout=2)
            for label, contents in (("stdout", output), ("stderr", errors)):
                destination = (
                    self._diagnostics / f"command-{self._sequence:04d}-{label}.log"
                )
                descriptor = os.open(
                    destination,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                )
                with os.fdopen(descriptor, "wb") as diagnostic:
                    diagnostic.write(contents)
