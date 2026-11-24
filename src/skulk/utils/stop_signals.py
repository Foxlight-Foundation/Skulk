"""This process's own requests to stop, delivered as an event.

A service is stopped by a signal: ``SIGTERM`` from a service manager or
``SIGINT`` from a terminal on POSIX; Ctrl-C or Ctrl-Break on Windows, where
Ctrl-Break is the one console signal a supervisor can send to another
process's group. The event loop can register a signal handler itself on POSIX
only, so the two platforms are wired differently here and nowhere else.
"""

import asyncio
import contextlib
import signal
import sys
from collections.abc import Iterator
from types import FrameType


@contextlib.contextmanager
def stop_on_signal(stopped: asyncio.Event) -> Iterator[None]:
    """Set ``stopped`` when this process is asked to stop, for the block's duration.

    Call it from a coroutine running on the main thread. On leaving the block
    the handlers are taken away again: Windows gets back what was set before,
    POSIX the default.

    Args:
        stopped: The event the running service waits on.
    """
    loop = asyncio.get_running_loop()
    if sys.platform == "win32":

        def request(_number: int, _frame: FrameType | None) -> None:
            # Python runs this on the main thread once the loop has been
            # woken. Handing over through the loop keeps the event's waiters
            # on the loop's own schedule.
            loop.call_soon_threadsafe(stopped.set)

        numbers = (signal.SIGINT, signal.SIGBREAK, signal.SIGTERM)
        previous = {number: signal.signal(number, request) for number in numbers}
        try:
            yield
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
        return
    numbers = (signal.SIGTERM, signal.SIGINT)
    for number in numbers:
        loop.add_signal_handler(number, stopped.set)
    try:
        yield
    finally:
        for number in numbers:
            loop.remove_signal_handler(number)
