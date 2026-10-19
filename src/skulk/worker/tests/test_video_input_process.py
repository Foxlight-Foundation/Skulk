"""Bounded source preparation cancellation reaps the decoder and its descendants."""

import os
import sys
import time
from pathlib import Path

import psutil
import pytest

from skulk.worker.runner import video_input_process
from skulk.worker.runner.video_input import MediaInputRequest


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
