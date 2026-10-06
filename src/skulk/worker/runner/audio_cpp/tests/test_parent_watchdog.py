"""A macOS runner crash must not leave an audio.cpp sidecar holding memory."""

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from skulk.worker.runner.audio_cpp import server


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS parent reparenting")
def test_watchdog_terminates_sidecar_when_runner_exits(tmp_path: Path) -> None:
    """A detached server receives SIGTERM after its runner disappears."""
    child = tmp_path / "child.py"
    child.write_text(
        "import os, signal, sys, time\n"
        "from pathlib import Path\n"
        "pid_file, stopped_file = map(Path, sys.argv[1:])\n"
        "def stop(_signal, _frame):\n"
        "    stopped_file.write_text('stopped')\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "pid_file.write_text(str(os.getpid()))\n"
        "while True: time.sleep(0.1)\n"
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import os, signal, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "watchdog, child, pid_file, stopped_file, watchdog_file = sys.argv[1:]\n"
        "process = subprocess.Popen([sys.executable, watchdog, str(os.getpid()), "
        "sys.executable, child, pid_file, stopped_file], start_new_session=True)\n"
        "Path(watchdog_file).write_text(str(process.pid))\n"
        "for _ in range(200):\n"
        "    if Path(pid_file).exists(): break\n"
        "    time.sleep(0.05)\n"
        "else: raise RuntimeError('sidecar did not start')\n"
        "os.kill(os.getpid(), signal.SIGKILL)\n"
    )
    pid_file = tmp_path / "sidecar.pid"
    stopped_file = tmp_path / "stopped"
    watchdog_file = tmp_path / "watchdog.pid"
    watchdog = Path(server.__file__).with_name("parent_watchdog.py")
    try:
        result = subprocess.run(  # noqa: S603 - test launches its own Python scripts
            [
                sys.executable, str(parent), str(watchdog), str(child),
                str(pid_file), str(stopped_file), str(watchdog_file),
            ],
            check=False,
            timeout=15,
        )
        assert result.returncode == -signal.SIGKILL
        deadline = time.monotonic() + 10
        while not stopped_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert stopped_file.read_text() == "stopped"
    finally:
        for path in (watchdog_file, pid_file):
            if path.exists() and not stopped_file.exists():
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(int(path.read_text()), signal.SIGKILL)
