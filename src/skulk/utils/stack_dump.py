"""Dump every thread's Python stack on demand, without root or a profiler."""

import faulthandler
import signal
import sys

STACK_DUMP_SIGNAL = signal.SIGUSR1
"""Signal that makes a Skulk process write its threads' Python stacks to stderr."""


def install_stack_dump_signal() -> None:
    """Write every thread's Python stack to stderr when the process gets SIGUSR1.

    A process stuck in native code shows only C frames to an OS sampler, since
    the whole Python call chain collapses into one evaluation frame, and a
    Python-level profiler needs root on macOS. ``faulthandler`` writes the
    Python stacks from a signal-safe handler without stopping the process, so
    ``kill -USR1 <pid>`` on a stuck node or runner puts its stacks in the
    node's log. The process keeps running afterwards.
    """
    faulthandler.register(STACK_DUMP_SIGNAL, file=sys.stderr, all_threads=True)
