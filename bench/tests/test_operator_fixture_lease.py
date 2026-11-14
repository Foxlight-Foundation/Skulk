"""Injected local-process lifecycle tests; no cloud resources are created."""

import os
import select
import subprocess
import sys

import pytest


@pytest.mark.parametrize("close_parent", [False, True])
def test_watchdog_reaps_child_on_expiry_or_parent_eof(
    close_parent: bool,
) -> None:
    """An independent process reaps its child even without runner cleanup code."""
    child_code = "import time; time.sleep(30)"
    # Observe the actual spawned PID, rather than requiring the child interpreter
    # to start and write a file within one second on a busy test host.
    guardian_code = (
        "import subprocess, sys\n"
        "from bench.operator_fixture_lease import supervise_local_process\n"
        "original_spawn = subprocess.Popen\n"
        "def record_child(*arguments, **options):\n"
        "    child = original_spawn(*arguments, **options)\n"
        "    print(child.pid, flush=True)\n"
        "    return child\n"
        "subprocess.Popen = record_child\n"
        "raise SystemExit(supervise_local_process([sys.executable, '-c', sys.argv[1]], "
        "parent_descriptor=sys.stdin.fileno(), lifetime_seconds=float(sys.argv[2])))"
    )
    # EOF must stop a still-active lease, rather than passing because expiry won.
    lifetime_seconds = 30 if close_parent else 1.5
    guardian = subprocess.Popen(
        [sys.executable, "-c", guardian_code, child_code, str(lifetime_seconds)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert guardian.stdout is not None
        readable, _, _ = select.select([guardian.stdout], [], [], 5)
        assert readable, "The guardian did not report its spawned child"
        child_pid = int(guardian.stdout.readline())
        if close_parent:
            assert guardian.poll() is None
            os.kill(child_pid, 0)
            assert guardian.stdin is not None
            guardian.stdin.close()
        assert guardian.wait(timeout=5) == 0
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        assert guardian.stdout is not None and guardian.stdout.read() == b""
        assert guardian.stderr is not None and guardian.stderr.read() == b""
    finally:
        if guardian.poll() is None:
            guardian.terminate()
            guardian.wait(timeout=5)


def test_watchdog_reports_unexpected_child_exit() -> None:
    """A prematurely exited relay is a fixture failure, not successful expiry."""
    code = (
        "import sys; from bench.operator_fixture_lease import supervise_local_process; "
        "raise SystemExit(supervise_local_process([sys.executable, '-c', 'raise SystemExit(7)'], "
        "parent_descriptor=sys.stdin.fileno(), lifetime_seconds=2))"
    )
    process = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE)
    try:
        assert process.wait(timeout=5) == 1
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
