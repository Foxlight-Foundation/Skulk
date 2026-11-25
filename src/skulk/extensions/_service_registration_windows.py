# ruff: noqa: E402
# The platform assertion has to come before the Windows-only imports: it is
# what tells the type checker to skip this module on other platforms.
"""Registering the plugin service with Windows: one Task Scheduler task.

launchd and systemd keep a service for each account. Windows has nothing of
the kind for an ordinary account, so the service is a Task Scheduler task the
account registers for itself, with no elevation: it runs as that account, in
its signed-in session, with its ordinary rights. ``service_registration``
holds the task's definition; this module holds everything that touches the
machine, because that needs the file seam and the account's identity, which
the standalone registration file may not import.

Task Scheduler is driven through ``schtasks.exe``. Three things about it shape
this module, each seen on a real machine:

- It answers in the system's language and reports every failure as exit code
  1, so nothing here reads its words. Whether the service is still running is
  asked of the lock the service holds (``service_bootstrap.LIFETIME_LOCK``).
- It prints a task in the console's code page, which cannot hold every
  character a path may have. The task is therefore read with the code page
  set to UTF-8.
- Ending a task ends only the program the task started. The service's
  startup file holds the manager in a job that ends with it, which is what
  makes a forced end complete.
"""

import sys

assert sys.platform == "win32"

import ctypes
import ctypes.wintypes as wt
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from skulk.extensions import service_bootstrap
from skulk.extensions.runtime_artifacts import current_platform
from skulk.extensions.service_registration import (
    ServiceScope,
    WindowsServiceLayout,
    task_fields,
)
from skulk.utils._windows_account import local_data_directory, this_account
from skulk.utils.runtime_files import (
    RuntimeLock,
    has_root_authority,
    is_private_directory,
    private_directory,
    remove_private,
    write_private,
)

# The time launchd and systemd give the service to stop before they end it.
_STOP_WAIT_SECONDS = 180.0
# How long a service that was ended by force gets to let go of its files.
_END_WAIT_SECONDS = 15.0
_POLL_SECONDS = 0.25
_TASK_NAME = re.compile(r"[A-Za-z0-9.-]{1,128}")
_MAXIMUM_DEFINITION = 262144

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.GetSystemDirectoryW.restype = wt.UINT
_kernel32.GetSystemDirectoryW.argtypes = (wt.LPWSTR, wt.UINT)
_system_directory_path: Callable[..., int] = _kernel32.GetSystemDirectoryW


def _system_directory() -> Path:
    """Where Windows keeps its own programs, asked of Windows."""
    buffer = ctypes.create_unicode_buffer(32768)
    if not _system_directory_path(buffer, len(buffer)):
        raise ctypes.WinError(ctypes.get_last_error(), "GetSystemDirectory")
    return Path(ctypes.wstring_at(buffer))


def _run(arguments: list[str]) -> "subprocess.CompletedProcess[bytes]":
    """Run one of Windows' own programs with a fixed environment, no window."""
    system = _system_directory()
    return subprocess.run(
        arguments,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=60,
        check=False,
        cwd=system,
        env={"SystemRoot": str(system.parent), "PATH": str(system)},
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def _schedule(*arguments: str) -> None:
    """Have Task Scheduler do one thing to the task.

    Raises:
        ValueError: Task Scheduler refused or failed.
    """
    try:
        result = _run([str(_system_directory() / "schtasks.exe"), *arguments])
    except (OSError, subprocess.TimeoutExpired):
        raise ValueError("Task Scheduler did not answer") from None
    if result.returncode != 0:
        raise ValueError("Task Scheduler refused; inspect the task in Task Scheduler")


def _definition(name: str) -> str | None:
    """The task as Task Scheduler holds it, or None when it gives none.

    None covers both "there is no such task" and "it could not be read":
    Task Scheduler does not say which. No caller may take it as proof that
    the name is free.

    Raises:
        ValueError: What Task Scheduler printed is not text or is too long.
    """
    if not _TASK_NAME.fullmatch(name):
        raise ValueError("invalid task name")
    system = _system_directory()
    try:
        # The one use of the command interpreter here: nothing else can set
        # the code page of the console that schtasks prints to. The line holds
        # no caller's text; the task name was matched against a fixed pattern.
        result = _run(
            [
                str(system / "cmd.exe"),
                "/d",
                "/c",
                f"chcp.com 65001>nul && schtasks.exe /Query /TN {name} /XML",
            ]
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    if len(result.stdout) > _MAXIMUM_DEFINITION:
        raise ValueError("service definition exceeds bound")
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("invalid service definition") from None


def owner_layout(scope: ServiceScope) -> WindowsServiceLayout:
    """The layout of the plugin service that belongs to this account.

    Raises:
        ValueError: For a machine-wide service, which Windows is not offered,
            and on a host the runtime installer cannot name: a service that
            could never be given a runtime is not registered either.
    """
    if scope != "user":
        raise ValueError(
            "on Windows the plugin service belongs to one account; "
            "a service for the whole machine is not offered"
        )
    current_platform()
    return WindowsServiceLayout(this_account(), Path(local_data_directory()))


def registered_for(layout: WindowsServiceLayout, base: Path) -> bool:
    """Whether the registered task is exactly the one that starts ``base``.

    Task Scheduler's own copy is read, never a file. A task that is switched
    off does not start ``base`` and is not registered for it.
    """
    try:
        registered = _definition(layout.label)
        if registered is None:
            return False
        return task_fields(registered) == task_fields(
            layout.definition(base).decode("utf-16")
        )
    except ValueError:
        return False


def _existing(layout: WindowsServiceLayout) -> str | None:
    """The registered task, verified to be local setup's own, if one is readable."""
    registered = _definition(layout.label)
    if registered is not None:
        layout.verify_existing(registered)
    return registered


def _free(root: Path, name: str) -> bool:
    try:
        RuntimeLock(root, name).close()
    except BlockingIOError:
        return False
    return True


def _stopped_within(root: Path, seconds: float) -> bool:
    """Whether the service and its manager let go of ``root`` in time."""
    deadline = time.monotonic() + seconds
    while not (
        _free(root, service_bootstrap.LIFETIME_LOCK) and _free(root, "manager.lock")
    ):
        if time.monotonic() >= deadline:
            return False
        time.sleep(_POLL_SECONDS)
    return True


def _stop(layout: WindowsServiceLayout) -> None:
    registered = _existing(layout)
    if registered is not None:
        # First, so that the minute does not start it again behind the stop.
        _schedule("/Change", "/TN", layout.label, "/DISABLE")
    try:
        if not is_private_directory(layout.root):
            raise ValueError("existing service state ownership differs")
    except FileNotFoundError:
        return
    request = layout.root / service_bootstrap.STOP_REQUEST
    write_private(request, b"")
    if not _stopped_within(layout.root, _STOP_WAIT_SECONDS):
        if registered is not None:
            _schedule("/End", "/TN", layout.label)
        if not _stopped_within(layout.root, _END_WAIT_SECONDS):
            # The request stays: whatever still runs stops when it can.
            raise ValueError("plugin service has not stopped")
    remove_private(request)


def _install(layout: WindowsServiceLayout) -> None:
    registered = _existing(layout)
    base = service_bootstrap.current_base()
    if not layout.launcher(base).is_file():
        raise ValueError(
            "this Python has no pythonw.exe, which the plugin service needs "
            "to run without a window"
        )
    private_directory(layout.root)
    # Setup stops the old service before installing; taking the lock here
    # also creates the file the service's startup file opens.
    if not _free(layout.root, service_bootstrap.LIFETIME_LOCK):
        raise ValueError("plugin service is still running")
    remove_private(layout.root / service_bootstrap.STOP_REQUEST)
    definition = layout.root / "service-task.xml"
    write_private(definition, layout.definition(base))
    try:
        # Replacing is asked for only over a task that was read and verified.
        # Without it Task Scheduler refuses to write over anything, which
        # protects a task under this name that could not be read.
        replace = ("/F",) if registered is not None else ()
        _schedule("/Create", "/TN", layout.label, "/XML", str(definition), *replace)
    finally:
        remove_private(definition)
    # The minute would start it too; this does not wait for the minute.
    _schedule("/Run", "/TN", layout.label)


def register_user(
    layout: WindowsServiceLayout, action: Literal["prepare", "stop", "install"]
) -> None:
    """Perform one fixed setup action as the account, with no elevation.

    Raises:
        ValueError: As an administrator or another account, for a task under
            the reserved name that local setup did not write, when Task
            Scheduler refuses, and when the service does not stop.
    """
    if has_root_authority() or layout.account != this_account():
        raise ValueError(
            "the plugin service registers as its own account, not as an administrator"
        )
    if action == "prepare":
        private_directory(layout.root)
    elif action == "stop":
        _stop(layout)
    else:
        _install(layout)
