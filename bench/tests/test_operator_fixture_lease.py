"""Injected local-process lifecycle tests; no cloud resources are created."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.mark.parametrize("close_parent", [False, True])
def test_watchdog_reaps_child_on_expiry_or_parent_eof(
    tmp_path: Path, close_parent: bool
) -> None:
    """An independent process reaps its child even without runner cleanup code."""
    pid_file = tmp_path / "child.pid"
    child_code = "import time; time.sleep(30)"
    guardian_code = """
import sys
from pathlib import Path
import bench.operator_fixture_lease as lease

launch = lease.subprocess.Popen

def record_child(*args, **kwargs):
    process = launch(*args, **kwargs)
    path = Path(sys.argv[2])
    temporary = path.with_suffix('.tmp')
    temporary.write_text(str(process.pid))
    temporary.replace(path)
    return process

lease.subprocess.Popen = record_child
raise SystemExit(lease.supervise_local_process(
    [sys.executable, '-c', sys.argv[1]],
    parent_descriptor=sys.stdin.fileno(), lifetime_seconds=float(sys.argv[3]),
))
"""
    # Observe the actual owned PID at allocation, independently of interpreter
    # startup. EOF must end a long lease rather than pass by waiting for expiry.
    lifetime = 30 if close_parent else 1.5
    guardian = subprocess.Popen(
        [sys.executable, "-c", guardian_code, child_code, str(pid_file), str(lifetime)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 120
        while (
            not pid_file.exists()
            and guardian.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert pid_file.exists()
        child_pid = int(pid_file.read_text())
        if close_parent:
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
