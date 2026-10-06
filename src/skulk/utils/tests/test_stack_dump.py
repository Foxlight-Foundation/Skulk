"""SIGUSR1 writes Python stacks to stderr and leaves the process running."""

import signal
import subprocess
import sys
import time

_CHILD = """
import sys, time
from skulk.utils.stack_dump import install_stack_dump_signal

def waiting_in_a_named_frame() -> None:
    print("ready", flush=True)
    time.sleep(30)

install_stack_dump_signal()
waiting_in_a_named_frame()
"""


def test_sigusr1_dumps_every_threads_python_stack_and_the_process_lives() -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        child.send_signal(signal.SIGUSR1)
        time.sleep(0.5)
        # Still running: the dump does not stop the process.
        assert child.poll() is None
    finally:
        child.terminate()
        _, stderr = child.communicate(timeout=10)
    assert "waiting_in_a_named_frame" in stderr
    assert "Thread" in stderr or "Current thread" in stderr
