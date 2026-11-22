"""Child processes this process owns outright: it can stop the whole tree.

Code that installs a plugin runtime runs other programs (the interpreter's
``venv`` module, ``pip``) inside a directory it holds a lock on. Two things
must be true of such a child, on every platform:

- **The whole tree can be stopped.** A timeout or a cancelled request ends
  the child and everything it started, not just the first process.
- **The tree never outlives the lock.** If this process dies abruptly, no
  other process may start work in the same directory while an orphan of the
  old tree is still writing there.

On POSIX the child leads a new session, so one signal reaches the tree, and
it inherits the lock's descriptor, so the lock stays held until the last
member exits. On Windows the tree runs inside a Job Object that the operating
system destroys, with every process in it, when this process's handle to it
closes (:mod:`skulk.utils._owned_process_windows`); a file lock there belongs
to one process and cannot be inherited.
"""

import asyncio
import contextlib
import os
import signal
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import final

from skulk.utils.runtime_files import RuntimeLock

if sys.platform == "win32":
    from skulk.utils import _owned_process_windows as _windows


def isolated_environment() -> dict[str, str]:
    """The fixed environment an owned child starts with.

    Nothing of the operator's environment reaches it: no interpreter or
    installer settings, no credentials. Windows additionally needs the system
    directory and a temporary directory to start a program at all.
    """
    if sys.platform == "win32":
        return _windows.isolated_environment()
    return {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


@final
class OwnedProcess:
    """One started child and the means to stop everything it started."""

    def __init__(self, process: asyncio.subprocess.Process, job: int | None) -> None:
        """Bind a started child to the platform's handle on its tree."""
        self.process = process
        self._job = job

    def kill(self) -> None:
        """Stop every process in the tree at once; quiet when none is left."""
        if sys.platform == "win32":
            if self._job is not None:
                _windows.terminate(self._job)
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.process.pid, signal.SIGKILL)

    def close(self) -> None:
        """Release the platform's handle on the tree; call after waiting for it.

        On Windows this also ends any member still running, which is the same
        guarantee an abrupt exit of this process gives.
        """
        if sys.platform == "win32" and self._job is not None:
            _windows.close(self._job)
            self._job = None


async def start_owned(
    arguments: Sequence[str], *, cwd: Path, lock: RuntimeLock
) -> OwnedProcess:
    """Start a program whose whole tree this process owns.

    The child gets no input, one combined output stream to read from
    ``process.stdout``, and :func:`isolated_environment`.

    Args:
        arguments: The program and its arguments.
        cwd: The directory the program starts in.
        lock: The held lock that fences that directory; the tree must not
            outlive it.

    Returns:
        The started child. The caller waits for ``process`` and then calls
        :meth:`OwnedProcess.close`, calling :meth:`OwnedProcess.kill` first
        when it gives up on the child.
    """
    if sys.platform == "win32":
        process, job = await _windows.start(arguments, cwd=cwd)
        return OwnedProcess(process, job)
    process = await asyncio.create_subprocess_exec(
        *arguments,
        cwd=cwd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
        pass_fds=(lock.descriptor,),
        env=isolated_environment(),
    )
    return OwnedProcess(process, None)
