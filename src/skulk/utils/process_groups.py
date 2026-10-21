"""Stop owned subprocess sessions without releasing their leader PID early."""

from __future__ import annotations

import os
import signal
import subprocess
import time

import psutil


def observe_process_exit(process: subprocess.Popen[str]) -> os.waitid_result | None:
    """Observe an owned child's exit without reaping it or releasing its PID.

    Return None while running, otherwise its exit status. Callers must not
    poll or wait for the child before terminating its owned process group.
    """
    return os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)


def _signal_group(process: subprocess.Popen[str], sent: signal.Signals) -> None:
    try:
        os.killpg(process.pid, sent)
    except ProcessLookupError:
        return
    except PermissionError:
        # Darwin returns EPERM for an owned group containing only zombies.
        # A live member must still surface the real failure; never report
        # successful cancellation when an engine could not be stopped.
        for identifier in psutil.pids():
            try:
                if os.getpgid(identifier) == process.pid and psutil.Process(identifier).status() != psutil.STATUS_ZOMBIE:
                    raise
            except (ProcessLookupError, psutil.NoSuchProcess):
                continue


def terminate_process_group(process: subprocess.Popen[str], *, grace_seconds: float = 1.0) -> None:
    """Terminate an owned session and reap its leader after signalling descendants.

    The child must have been started with start_new_session=True and remain
    unreaped. Send TERM, allow the leader its grace interval, then send KILL
    before waiting. Real permission failures propagate to the supervisor.
    """
    # A caller that already reaped the leader no longer owns this numeric
    # group identity. Refuse before signalling a potentially reused PID.
    if process.returncode is not None:
        raise ValueError("cannot stop a process group after its leader was reaped")
    observe_process_exit(process)
    _signal_group(process, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while observe_process_exit(process) is None and time.monotonic() < deadline:
        time.sleep(0.025)
    _signal_group(process, signal.SIGKILL)
    process.wait(timeout=3)
