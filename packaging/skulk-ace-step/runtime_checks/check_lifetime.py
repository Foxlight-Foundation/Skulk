"""Real private CLI/process-group retirement; no model libraries or weights."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import suppress
from pathlib import Path

import psutil
import skulk_ace_step

PACKAGE = Path(skulk_ace_step.__file__).resolve().parent.parent


def _alive(process: psutil.Process) -> bool:
    try:
        return process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


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

    def test_full_control_pipe_cannot_block_native_retirement(self) -> None:
        """A live owner that stops reading cannot strand the guard's output thread."""
        with tempfile.TemporaryDirectory(prefix="native-full-pipe-") as root:
            config = Path(root) / "invalid.json"
            config.write_text("{}")
            read, write = os.pipe()
            os.set_blocking(write, False)
            try:
                while True:
                    try:
                        os.write(write, b"x" * 4096)
                    except BlockingIOError:
                        break
                child = subprocess.Popen(
                    self.command(config),
                    start_new_session=True,
                    stdout=write,
                    stderr=subprocess.PIPE,
                )
                try:
                    _, diagnostic = child.communicate(timeout=15)
                    self.assertEqual(child.returncode, -signal.SIGKILL)
                    self.assertIn(b"control egress stopped draining", diagnostic)
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.communicate()
            finally:
                os.close(read)
                os.close(write)

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

    def test_leader_loss_retains_fence_for_close_fds_descendant(self) -> None:
        """Packaged guardian owns the real lock even after both original owners die."""
        child_code = """
import json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from skulk_ace_step.lifetime import start_parent_guardian
start_parent_guardian(int(sys.argv[2]))
child=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print("READY",flush=True);time.sleep(60)'],stdout=subprocess.PIPE,close_fds=True)
assert child.stdout.readline()==b'READY\\n'
child.stdout.close()
# Publish only complete identities before the test begins retiring their owners.
result=Path(sys.argv[3])
temporary=result.with_suffix('.partial')
temporary.write_text(json.dumps([os.getpid(),child.pid]))
temporary.replace(result)
time.sleep(60)
"""
        parent_code = """
import fcntl,os,subprocess,sys,time
fd=os.open(sys.argv[1],os.O_CREAT|os.O_RDWR,0o600)
fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
child=subprocess.Popen([sys.executable,'-I','-B','-c',sys.argv[2],sys.argv[3],str(os.getpid()),sys.argv[4]],start_new_session=True,pass_fds=(fd,))
time.sleep(60)
"""
        with tempfile.TemporaryDirectory(prefix="native-music-custody-") as scratch:
            root = Path(scratch)
            parent = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-c",
                    parent_code,
                    str(root / "owner.lock"),
                    child_code,
                    str(PACKAGE),
                    str(root / "children.json"),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            guardian = None
            leader = descendant = None
            try:
                deadline = time.monotonic() + 5
                while not (root / "children.json").exists():
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.01)
                leader_pid, descendant_pid = json.loads(
                    (root / "children.json").read_text()
                )
                leader, descendant = (
                    psutil.Process(leader_pid),
                    psutil.Process(descendant_pid),
                )
                detached = [
                    child
                    for child in leader.children()
                    if os.getpgid(child.pid) == child.pid
                ]
                self.assertEqual(len(detached), 1)
                guardian = detached[0]
                guardian.suspend()
                for anchor in guardian.children():
                    anchor.kill()
                os.kill(parent.pid, signal.SIGSTOP)
                leader.kill()
                parent.kill()
                parent.wait(timeout=5)
                self.assertNotEqual(descendant.status(), psutil.STATUS_ZOMBIE)
                with (
                    (root / "owner.lock").open("rb") as fence,
                    self.assertRaises(BlockingIOError),
                ):
                    fcntl.flock(fence.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                guardian.resume()
                deadline = time.monotonic() + 10
                while any(_alive(process) for process in (descendant, guardian)):
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.01)
                with (root / "owner.lock").open("rb") as fence:
                    fcntl.flock(fence.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                if guardian is not None and _alive(guardian):
                    guardian.resume()
                if parent.poll() is None:
                    parent.kill()
                    parent.wait(timeout=5)
                for process in (leader, descendant):
                    if process is not None and _alive(process):
                        with suppress(psutil.NoSuchProcess):
                            process.kill()
