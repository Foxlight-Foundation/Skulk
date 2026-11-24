"""A service learns of a request to stop on every platform it runs on."""

import os
import signal
import subprocess
import sys

import pytest

# Waits to be asked to stop, then says that it was and ends in order.
_SERVICE = (
    "import asyncio\n"
    "from skulk.utils.stop_signals import stop_on_signal\n"
    "async def main():\n"
    "    stopped = asyncio.Event()\n"
    "    with stop_on_signal(stopped):\n"
    "        print('ready', flush=True)\n"
    "        await stopped.wait()\n"
    "    print('stopped', flush=True)\n"
    "asyncio.run(main())\n"
)

if sys.platform == "win32":
    # Ctrl-Break is the one console signal that can be sent to another
    # process's group. Ctrl-C cannot: a process that leads a new group, which
    # the test needs so as not to stop itself, has Ctrl-C switched off.
    _REQUESTS = [signal.CTRL_BREAK_EVENT]
    _NEW_GROUP = subprocess.CREATE_NEW_PROCESS_GROUP
else:
    _REQUESTS = [signal.SIGTERM, signal.SIGINT]
    _NEW_GROUP = 0


def _has_a_console() -> bool:
    """Whether a console signal can be sent at all from this process."""
    if sys.platform != "win32":
        return True
    import ctypes

    return bool(ctypes.WinDLL("kernel32").GetConsoleCP())


@pytest.mark.parametrize("request_to_stop", _REQUESTS)
def test_a_request_to_stop_sets_the_event_and_the_service_ends_in_order(
    request_to_stop: int,
) -> None:
    if not _has_a_console():
        pytest.skip("no console to send Ctrl-Break through")
    service = subprocess.Popen(
        [sys.executable, "-c", _SERVICE],
        stdout=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        creationflags=_NEW_GROUP,
    )
    try:
        assert service.stdout is not None
        assert service.stdout.readline().strip() == b"ready"
        os.kill(service.pid, request_to_stop)
        assert service.stdout.read().strip() == b"stopped"
        assert service.wait(15) == 0
    finally:
        service.kill()
        service.wait()
