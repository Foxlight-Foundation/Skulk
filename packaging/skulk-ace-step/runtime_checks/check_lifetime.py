"""Real private CLI/process-group retirement; no model libraries or weights."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import skulk_ace_step

PACKAGE = Path(skulk_ace_step.__file__).resolve().parent.parent


class NativeLifetimeTests(unittest.TestCase):
    """Exercise actual process exits rather than treating a failure frame as cleanup."""

    def command(self, config: Path) -> list[str]:
        """Start the real module with isolated imports and the actual test parent."""
        return [
            sys.executable,
            "-I",
            "-B",
            "-c",
            'import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module("skulk_ace_step",run_name="__main__")',
            str(PACKAGE),
            "--serve",
            str(config),
            "--parent-pid",
            str(os.getpid()),
        ]

    def test_real_cli_reports_one_terminal_and_kills_its_private_group(self) -> None:
        """Malformed configuration is actionable and cannot leave a live runtime."""
        with tempfile.TemporaryDirectory(prefix="native-music-cli-") as root:
            config = Path(root) / "invalid.json"
            config.write_text("{}")
            result = subprocess.run(
                self.command(config),
                start_new_session=True,
                capture_output=True,
                timeout=15,
            )
        self.assertEqual(result.returncode, -signal.SIGKILL)
        frames = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([f["kind"] for f in frames], ["failed"])
        self.assertNotIn(b"acestep.gpu_config", result.stderr)

    def test_closed_control_pipe_still_terminates_the_actual_process(self) -> None:
        """A BrokenPipe during failure reporting cannot skip whole-group retirement."""
        with tempfile.TemporaryDirectory(prefix="native-music-pipe-") as root:
            config = Path(root) / "invalid.json"
            config.write_text("{}")
            read, write = os.pipe()
            os.close(read)
            try:
                child = subprocess.Popen(
                    self.command(config),
                    start_new_session=True,
                    stdout=write,
                    stderr=subprocess.PIPE,
                )
            finally:
                os.close(write)
            try:
                _, diagnostics = child.communicate(timeout=15)
            except BaseException:
                os.killpg(child.pid, signal.SIGKILL)
                child.communicate()
                raise
        self.assertEqual(child.returncode, -signal.SIGKILL)
        self.assertIn(b"BrokenPipe", diagnostics)

    def test_actual_parent_death_retires_its_native_session(self) -> None:
        """The guardian is armed before its real supervising parent exits."""
        child_code = 'import sys,time;sys.path.insert(0,sys.argv[1]);from skulk_ace_step.lifetime import start_parent_guardian;start_parent_guardian(int(sys.argv[2]));print("ARMED",flush=True);time.sleep(60)'
        owner_code = 'import os,subprocess,sys;child=subprocess.Popen([sys.executable,"-I","-B","-c",sys.argv[1],sys.argv[2],str(os.getpid())],start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL);assert child.stdout.readline()==b"ARMED\\n";print(child.pid,flush=True);os._exit(0)'
        owner = subprocess.run(
            [sys.executable, "-I", "-B", "-c", owner_code, child_code, str(PACKAGE)],
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(owner.returncode, 0, owner.stderr.decode())
        native_pid = int(owner.stdout.strip())
        ended = False
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                state = subprocess.run(
                    ["ps", "-o", "stat=", "-p", str(native_pid)],
                    capture_output=True,
                    text=True,
                )
                # An orphan may remain a reaper-owned zombie briefly, but it
                # must have no executing threads or active inference left.
                if state.returncode == 1 or state.stdout.strip().startswith("Z"):
                    ended = True
                    break
                time.sleep(0.05)
        finally:
            if not ended:
                os.killpg(native_pid, signal.SIGKILL)
        self.assertTrue(ended, "native process survived its actual supervisor")
