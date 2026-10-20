# pyright: reportPrivateUsage=false
"""Bounded source preparation cancellation reaps the decoder and its descendants."""

import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import psutil
import pytest

from skulk.worker.runner import video_input_process
from skulk.worker.runner.video_input import MediaInputRequest


def _await_exit(process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 5
    while video_input_process._observe_exit(process) is None:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert process.returncode is None


def _is_live(process: psutil.Process) -> bool:
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def test_child_reaped_between_liveness_probes_counts_as_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful reap can race the test's is-running/status observations."""
    process = psutil.Process(os.getpid())

    def gone() -> str:
        raise psutil.NoSuchProcess(process.pid)

    monkeypatch.setattr(process, "is_running", lambda: True)
    monkeypatch.setattr(process, "status", gone)
    assert not _is_live(process)


def test_exited_decoder_is_reaped_after_its_owned_group_is_stopped() -> None:
    """Darwin's zombie-only group must not replace a successful decoder result."""
    process = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True, text=True)
    try:
        _await_exit(process)
        video_input_process._reap(process)
        assert process.returncode == 0
    finally:
        process.wait(timeout=3)


def test_exited_decoder_still_cleans_up_a_term_resistant_descendant(tmp_path: Path) -> None:
    """Reaping the parent early must not lose its child's process-group identity."""
    marker = tmp_path / "ready"
    child_script = f"import os,signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path({str(marker)!r}).write_text(str(os.getpid()));time.sleep(60)"
    parent_script = f"import subprocess,sys,time;from pathlib import Path;subprocess.Popen([sys.executable,'-c',{child_script!r}]);p=Path({str(marker)!r});\nwhile not p.exists():time.sleep(.01)"
    process = subprocess.Popen([sys.executable, "-c", parent_script], start_new_session=True, text=True)
    child: psutil.Process | None = None
    try:
        _await_exit(process)
        child = psutil.Process(int(marker.read_text()))
        video_input_process._reap(process)
        for _ in range(40):
            if not _is_live(child):
                break
            time.sleep(0.025)
        else:
            pytest.fail("an exited decoder left its TERM-resistant child alive")
    finally:
        with suppress(psutil.NoSuchProcess):
            if child is not None and _is_live(child):
                child.kill()
        if process.returncode is None:
            process.kill()
        process.wait(timeout=3)


def test_live_decoder_permission_failure_is_not_suppressed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only an already-dead group can turn Darwin's EPERM into a clean outcome."""
    process = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True, text=True)

    def denied(_group: int, _sent: int) -> None:
        raise PermissionError("controlled live-group denial")

    monkeypatch.setattr(video_input_process.os, "killpg", denied)
    try:
        with pytest.raises(PermissionError, match="live-group denial"):
            video_input_process._signal_group(process, signal.SIGTERM)
    finally:
        process.kill()
        process.wait(timeout=3)


def test_protocol_files_are_removed_even_when_cleanup_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cleanup failure cannot retain command-private preparation files."""
    interpreter = tmp_path / "interpreter"
    interpreter.write_text(f"#!{sys.executable}\n")
    interpreter.chmod(0o700)
    original = video_input_process._reap

    def failed_cleanup(process: subprocess.Popen[str]) -> None:
        original(process)
        raise RuntimeError("controlled cleanup failure")

    monkeypatch.setattr(video_input_process, "_reap", failed_cleanup)
    request = MediaInputRequest(operation="audio", source=str(tmp_path / "source"), destination=str(tmp_path / "prepared.wav"), duration_seconds=1.0)
    with pytest.raises(RuntimeError, match="cleanup failure"):
        video_input_process.prepare_media_input(interpreter, request, tmp_path, is_cancelled=lambda: False)
    assert not (tmp_path / "_media_request.json").exists()
    assert not (tmp_path / "_media_verified.json").exists()


@pytest.mark.parametrize("timeout", [False, True])
def test_decoder_cancel_and_deadline_remove_protocol_and_descendants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timeout: bool) -> None:
    """A real process tree catches orphaning that a mocked decoder would hide."""
    interpreter = tmp_path / "interpreter"
    child_pid = tmp_path / "child.pid"
    interpreter.write_text(f"#!{sys.executable}\nimport subprocess,time\nfrom pathlib import Path\nchild=subprocess.Popen([{sys.executable!r},'-c','import time;time.sleep(60)'])\nPath({str(child_pid)!r}).write_text(str(child.pid))\ntime.sleep(60)\n")
    interpreter.chmod(0o700)
    if timeout:
        monkeypatch.setattr(video_input_process, "_PREPARATION_TIMEOUT_SECONDS", 1.0)
    started = time.monotonic()
    request = MediaInputRequest(operation="audio", source=str(tmp_path / "source"), destination=str(tmp_path / "prepared.wav"), duration_seconds=1.0)
    def cancelled() -> bool:
        return not timeout and child_pid.exists()
    if timeout:
        with pytest.raises(ValueError, match="deadline"):
            video_input_process.prepare_media_input(interpreter, request, tmp_path, is_cancelled=cancelled)
    else:
        assert video_input_process.prepare_media_input(interpreter, request, tmp_path, is_cancelled=cancelled) is None
    assert time.monotonic() - started < 5
    assert not (tmp_path / "_media_request.json").exists()
    assert not (tmp_path / "_media_verified.json").exists()
    identifier = int(child_pid.read_text())
    for _ in range(40):
        if not psutil.pid_exists(identifier) or psutil.Process(identifier).status() == psutil.STATUS_ZOMBIE:
            break
        time.sleep(0.025)
    else:
        os.kill(identifier, 9)
        pytest.fail("source decoder left a live descendant")
