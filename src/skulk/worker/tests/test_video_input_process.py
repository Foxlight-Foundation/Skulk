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

from skulk.utils import process_groups
from skulk.worker.runner import video_input, video_input_process
from skulk.worker.runner.video_input import MediaInputRequest


def test_linux_decoder_memory_retains_its_own_resident_peak(monkeypatch: pytest.MonkeyPatch) -> None:
    """A parent's historical peak must not reject a small decoder after exec."""
    monkeypatch.setattr(video_input.sys, "platform", "linux")

    def memory_status(path: Path) -> str:
        assert path == Path("/proc/self/status")
        return "VmHWM:\t 262144 kB\nVmRSS:\t 20480 kB\n"

    monkeypatch.setattr(video_input.Path, "read_text", memory_status)

    def inherited_peak(_who: int) -> None:
        pytest.fail("Linux decoder consulted its inherited high-water mark")

    monkeypatch.setattr(video_input.resource, "getrusage", inherited_peak)
    assert video_input._resident_peak_bytes() == 256 * 1024**2


@pytest.mark.parametrize("status", ["VmRSS: 20480 kB\n", "VmHWM: 20 pages\n", "VmHWM: -1 kB\n"])
def test_linux_decoder_refuses_missing_or_invalid_peak_accounting(monkeypatch: pytest.MonkeyPatch, status: str) -> None:
    """An unavailable peak must refuse preparation instead of disabling its guard."""
    monkeypatch.setattr(video_input.sys, "platform", "linux")

    def memory_status(_path: Path) -> str:
        return status

    monkeypatch.setattr(video_input.Path, "read_text", memory_status)
    with pytest.raises(ValueError, match="resident peak accounting"):
        video_input._resident_peak_bytes()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux preserves pre-exec RSS accounting")
def test_linux_decoder_does_not_inherit_parent_memory_charge() -> None:
    """Exercise a real exec from a larger process image, not a mocked counter."""
    child = f"import os,resource,runpy;from pathlib import Path;media=runpy.run_path({str(Path(video_input.__file__))!r});allocation=bytearray(96*1024**2);del allocation;print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,media['_resident_peak_bytes'](),int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'))"
    parent = f"import subprocess,sys;allocation=bytearray(320*1024**2);subprocess.run([sys.executable,'-I','-c',{child!r}],check=True,start_new_session=True)"
    result = subprocess.run([sys.executable, "-I", "-c", parent], capture_output=True, text=True, check=True, timeout=20)
    inherited_peak, decoder_peak, decoder_resident = (int(value) for value in result.stdout.split())
    assert inherited_peak >= 320 * 1024**2
    assert 96 * 1024**2 <= decoder_peak < inherited_peak - 128 * 1024**2
    assert decoder_resident < decoder_peak - 64 * 1024**2


def _await_exit(process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 5
    while process_groups.observe_process_exit(process) is None:
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
        process_groups.terminate_process_group(process)
        assert process.returncode == 0
    finally:
        process.wait(timeout=3)


def test_reaped_group_identity_is_never_signalled(monkeypatch: pytest.MonkeyPatch) -> None:
    """A released numeric PID must never authorize a later process-group signal."""
    process = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True, text=True)
    process.wait(timeout=3)

    def unrelated_group(_group: int, _sent: signal.Signals) -> None:
        pytest.fail("cleanup signalled a group after releasing its owned leader PID")

    monkeypatch.setattr(process_groups.os, "killpg", unrelated_group)
    with pytest.raises(ValueError, match="leader was reaped"):
        process_groups.terminate_process_group(process)


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
        process_groups.terminate_process_group(process)
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

    monkeypatch.setattr(process_groups.os, "killpg", denied)
    try:
        with pytest.raises(PermissionError, match="live-group denial"):
            process_groups._signal_group(process, signal.SIGTERM)
    finally:
        process.kill()
        process.wait(timeout=3)


def test_protocol_files_are_removed_even_when_cleanup_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cleanup failure cannot retain command-private preparation files."""
    interpreter = tmp_path / "interpreter"
    interpreter.write_text(f"#!{sys.executable}\n")
    interpreter.chmod(0o700)
    original = video_input_process.terminate_process_group

    def failed_cleanup(process: subprocess.Popen[str]) -> None:
        original(process)
        raise RuntimeError("controlled cleanup failure")

    monkeypatch.setattr(video_input_process, "terminate_process_group", failed_cleanup)
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
