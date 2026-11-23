"""Child processes this process owns outright: it can stop the whole tree.

Code that installs a plugin runtime runs other programs (the interpreter's
``venv`` module, ``pip``) inside a directory it holds a lock on. Two things
must be true of such a child, on every platform:

- **The whole tree can be stopped.** A timeout or a cancelled request ends
  the child and everything it started, not just the first process.
- **The child never outlives the lock.** If this process dies abruptly, no
  other process may start work in the same directory while the orphaned child
  is still running there.

On POSIX the child leads a new session, so one signal reaches the tree, and
it inherits the lock's descriptor, so the lock stays held for as long as the
child runs. That covers a descendant only if it kept the descriptor: a program
started with Python's ``subprocess`` closes it by default, so a descendant
that outlived both this process and the child would run without the lock.
The installer's children wait for the programs they start, so that case needs
a second failure.

On Windows the tree runs inside a Job Object that the operating system
destroys, with every process in it, when this process's handle to it closes
(:mod:`skulk.utils._owned_process_windows`). A file lock there belongs to one
process and cannot be inherited, so the child instead holds a file that the
lock checks: the lock cannot be taken again until the child has ended. The
job ends the rest of the tree in the same operation as the child, though
Windows does not promise the order within it.

A terminal command that ends by running an installed program in its own place
uses :func:`hand_over`. POSIX replaces the process. Windows cannot, so there
the program runs as an owned child on the same terminal and this process
waits for it and then exits with its exit code.
"""

import asyncio
import contextlib
import os
import signal
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn, Protocol, final

from skulk.utils.runtime_files import RuntimeLock

if sys.platform == "win32":
    from skulk.utils import _owned_process_windows as _windows


class OwnedProcess(Protocol):
    """One started child and the means to stop everything it started."""

    @property
    def pid(self) -> int:
        """The child's process identifier."""
        ...

    @property
    def stdout(self) -> asyncio.StreamReader:
        """The child's output and error output, as one stream."""
        ...

    @property
    def returncode(self) -> int | None:
        """The child's exit code, or ``None`` while it is still running."""
        ...

    async def wait(self) -> int:
        """Wait for the child to end; return its exit code."""
        ...

    def kill(self) -> None:
        """Stop every process in the tree at once; quiet when none is left.

        Call it before waiting for the child. On POSIX the tree is addressed
        by the child's process identifier, which the system may give to an
        unrelated process once the child has been waited for, so after that
        this does nothing there.
        """
        ...

    async def close(self) -> None:
        """Release what the platform holds for the tree; call after waiting.

        On Windows this ends any member still running and returns only when
        none is left, however long that takes, so nothing of theirs is still
        open in the directory. On POSIX there is nothing to release.
        """
        ...


def isolated_environment() -> dict[str, str]:
    """The fixed environment an owned child starts with.

    Nothing of the operator's environment reaches it: no interpreter or
    installer settings, no credentials. Windows additionally needs the system
    directory and a temporary directory to start a program at all.
    """
    if sys.platform == "win32":
        return _windows.isolated_environment()
    return {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


if sys.platform != "win32":

    @final
    class _SessionChild:
        """A child that leads its own session and holds the lock's descriptor."""

        def __init__(self, process: asyncio.subprocess.Process) -> None:
            self._process = process
            assert process.stdout is not None
            self.stdout = process.stdout

        @property
        def pid(self) -> int:
            return self._process.pid

        @property
        def returncode(self) -> int | None:
            return self._process.returncode

        async def wait(self) -> int:
            return await self._process.wait()

        def kill(self) -> None:
            if self._process.returncode is not None:
                # The leader has been waited for, so its identifier no longer
                # belongs to this session and may already name another
                # process group. Signalling it could end a stranger.
                return
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self._process.pid, signal.SIGKILL)

        async def close(self) -> None:
            """Nothing to release: the session ends with its last member."""


async def start_owned(
    arguments: Sequence[str], *, cwd: Path, lock: RuntimeLock
) -> OwnedProcess:
    """Start a program whose whole tree this process owns.

    The child gets no input, one combined output stream, and
    :func:`isolated_environment`.

    Args:
        arguments: The program and its arguments.
        cwd: The directory the program starts in.
        lock: The held lock that fences that directory; the child must not
            outlive it.

    Returns:
        The started child. The caller waits for it and then awaits ``close``,
        calling ``kill`` first when it gives up on the child.
    """
    if sys.platform == "win32":
        return await _windows.start(arguments, cwd=cwd, lock=lock.path)
    return _SessionChild(
        await asyncio.create_subprocess_exec(
            *arguments,
            cwd=cwd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
            pass_fds=(lock.descriptor,),
            env=isolated_environment(),
        )
    )


def hand_over(arguments: Sequence[str], *, lock: Path) -> NoReturn:
    """Give this process's terminal, and its place under a lock, to a program.

    The program gets this process's standard input, output and error as they
    are, its working directory, and :func:`isolated_environment`. Interrupts
    from the terminal are the program's to handle. This function does not
    return.

    On POSIX the program replaces this process. The caller must already have
    made the lock's descriptor inheritable, which is how the program comes to
    hold the lock until it exits.

    On Windows a process cannot be replaced and a file lock cannot be passed
    on. The program runs as an owned child holding the lock's children fence;
    this process keeps the lock, waits, ends anything the program left
    running, and then exits with the program's exit code. Unwinding on the way
    out releases the lock, so the caller holds it for exactly as long as the
    program ran.

    Args:
        arguments: The program and its arguments.
        lock: The path of the lock this process holds.

    Raises:
        SystemExit: On Windows, when the program has ended, with its exit code.
        ValueError: On Windows, when the lock's children fence is not an
            owner-only file.
        OSError: When the program could not be started.
    """
    if sys.platform == "win32":
        code = _windows.run_in_place(arguments, lock=lock)
        # Windows exit codes are unsigned 32-bit values, and a program ended
        # by the system reports a large one. Python exits with a signed
        # integer of the same bits.
        raise SystemExit(code - (1 << 32) if code >= 1 << 31 else code)
    os.execve(arguments[0], tuple(arguments), isolated_environment())
