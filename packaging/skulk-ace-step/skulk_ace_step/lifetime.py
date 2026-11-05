"""Owned process-group lifetime for native work that may outlive Python timeouts."""

from __future__ import annotations

import os
import select
import signal
import time
from contextlib import suppress

import psutil


def terminate_owned_group() -> None:
    """Kill only this dedicated session, including lingering native threads/children."""
    if os.getpid() != os.getpgrp():
        raise RuntimeError("native music cannot terminate a foreign process group")
    os.killpg(os.getpgrp(), signal.SIGKILL)


def _same_process(identifier: int, created: float) -> bool:
    try:
        process = psutil.Process(identifier)
        return (
            process.create_time() == created
            and process.status() != psutil.STATUS_ZOMBIE
        )
    except psutil.NoSuchProcess:
        return False


def _live_members(group: int, anchor: int) -> bool:
    for identifier in psutil.pids():
        try:
            if (
                identifier != anchor
                and os.getpgid(identifier) == group
                and psutil.Process(identifier).status() != psutil.STATUS_ZOMBIE
            ):
                return True
        except (ProcessLookupError, psutil.NoSuchProcess):
            continue
    return False


def _signal(group: int, anchor: int, sent: signal.Signals) -> None:
    try:
        os.killpg(group, sent)
    except ProcessLookupError:
        return
    except PermissionError:
        # Darwin refuses a group containing only zombies. A live member must
        # preserve the real failure and every inherited ownership fence.
        if _live_members(group, anchor):
            raise


def _guard(
    parent: int, parent_created: float, leader: int, leader_created: float, anchor: int
) -> None:
    while _same_process(parent, parent_created) and _same_process(
        leader, leader_created
    ):
        time.sleep(0.1)
    _signal(leader, anchor, signal.SIGTERM)
    deadline = time.monotonic() + 0.5
    while _live_members(leader, anchor) and time.monotonic() < deadline:
        time.sleep(0.025)
    _signal(leader, anchor, signal.SIGKILL)
    while _live_members(leader, anchor):
        # An uninterruptible descendant can take arbitrarily long to drain.
        # Keep the actual inherited locks; elapsed time grants no file custody.
        time.sleep(0.1)
    os.waitpid(anchor, 0)


def _anchor(armed: int, drain: int | None) -> None:
    os.close(armed)
    if drain is not None:
        os.close(drain)
    for descriptor in (0, 1, 2):
        os.close(descriptor)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        signal.pause()


def start_parent_guardian(parent_pid: int, drain_descriptor: int | None = None) -> None:
    """Arm an independent custody owner before imports can create native children.

    The detached guardian inherits the service/source locks, outlives both Core
    and the native leader, and closes the drain pipe only after actual group
    exit. Its unreaped anchor remains in the original group, reserving that
    numeric group identity even when the leader is lost. Native/model processes
    never create this guardian after loading Torch or starting inference threads.
    """
    if parent_pid <= 1 or os.getppid() != parent_pid or os.getpid() != os.getpgrp():
        raise ValueError(
            "native music requires its actual supervisor and a private process session"
        )

    # Resolve both identities before the fork so their disappearance during
    # arming becomes a drain condition, not an unhandled setup exception.
    parent_created = psutil.Process(parent_pid).create_time()
    leader, leader_created = os.getpid(), psutil.Process().create_time()
    read, write = os.pipe()
    guardian = os.fork()
    if guardian:
        os.close(write)
        try:
            if not select.select([read], [], [], 5)[0] or os.read(read, 1) != b"1":
                raise RuntimeError("native music custody guardian did not arm")
        finally:
            os.close(read)
            if drain_descriptor is not None:
                os.close(drain_descriptor)
        return
    os.close(read)
    try:
        anchor = os.fork()
        if not anchor:
            _anchor(write, drain_descriptor)
            os._exit(0)
        os.setsid()
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        null = os.open(os.devnull, os.O_RDWR)
        os.dup2(null, 0)
        os.dup2(null, 1)
        if null > 2:
            os.close(null)
        try:
            os.write(write, b"1")
        except BrokenPipeError:
            # Losing the leader during arming still requires actual group
            # drain, including the anchor, before dropping inherited locks.
            pass
        finally:
            os.close(write)
        _guard(parent_pid, parent_created, leader, leader_created, anchor)
    except BaseException:
        # Unknown process state cannot authorize recovery. The parent reports
        # a bounded drain timeout while this independent owner retains locks.
        # Core loss can close diagnostics too; that is never permission to
        # release the remaining service/source ownership.
        with suppress(OSError):
            os.write(2, b"native music custody drain remains uncertain\n")
        while True:
            time.sleep(1)
    os._exit(0)
