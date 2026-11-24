"""Subprocess body for the traceback privacy tests.

The tests in ``test_logging_traceback_privacy.py`` run this module in a fresh
interpreter, because the property under test depends on process-level state the
pytest process cannot reproduce: the real ``stderr`` and ``stdout`` streams,
loguru's import-time defaults, and a spawned runner child. Each mode logs one
failed "generation" whose frames hold the request text in a local variable, the
shape a served runner's failure has, and the test then reads every sink.
"""

from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path
from typing import Literal, cast

import loguru

from skulk.shared.logging import logger_cleanup, logger_setup, set_structured_stdout

FAILURE_MESSAGE = "prompt is longer than the context window"
"""Exception message every probe failure carries; never contains request text."""

FAILURE_TYPE_NAME = "ValueError"
"""Name of the exception type every probe failure raises."""

REQUEST_FRAME_SOURCE = "_reject(request_text)"
"""Source text of the call that holds the request text.

A traceback that prints this line shows the frame loguru would annotate with the
variable's value, so its presence proves an absent value was withheld rather
than never reachable.
"""

NODE_LABEL = "probe: node generation failed"
"""Message logged through the node's configured sinks."""

RUNNER_SUPERVISOR_LOGGER_LABEL = "probe: runner generation failed (supervisor logger)"
"""Message a runner child logs through the logger its supervisor passed in."""

RUNNER_PROCESS_LOGGER_LABEL = "probe: runner generation failed (process logger)"
"""Message a runner child logs through its own process-global loguru logger."""

DEFAULT_LABEL = "probe: unconfigured process failed"
"""Message logged by a process that never configures logging."""

JsonSinkRoute = Literal["startup", "runtime"]


def _reject(request_text: str) -> None:
    """Refuse a request the way an engine refuses an oversized prompt."""
    if request_text:
        raise ValueError(FAILURE_MESSAGE)


def _generate(request_text: str) -> None:
    """Hold the request text in a frame the failure's traceback includes."""
    _reject(request_text)


def log_failed_generation(
    target: "loguru.Logger", request_text: str, label: str
) -> None:
    """Fail one generation and log it the way the served runners do.

    Args:
        target: Logger that receives the failure.
        request_text: Text held in the failing frames' local variables.
        label: Log message; it must not contain the request text.
    """
    try:
        _generate(request_text)
    except ValueError as error:
        target.opt(exception=error).warning(label)


def runner_child(supervisor_logger: "loguru.Logger", request_text: str) -> None:
    """Log one failure through each logger a runner subprocess can reach.

    A runner receives its supervisor's logger as a process argument, and the
    modules it imports also log through the child's own process-global loguru
    logger, whose handler is whatever loguru installed at import.

    Args:
        supervisor_logger: The node's configured logger, passed like a runner's.
        request_text: Text held in the failing frames' local variables.
    """
    log_failed_generation(
        supervisor_logger, request_text, RUNNER_SUPERVISOR_LOGGER_LABEL
    )
    log_failed_generation(loguru.logger, request_text, RUNNER_PROCESS_LOGGER_LABEL)


def _run_node(
    request_text: str, log_file: Path, verbosity: int, json_sink: JsonSinkRoute
) -> None:
    """Configure logging as a node does, then fail in-process and in a child."""
    # The node forces this start method before it configures logging, so its
    # queued sinks and its runner children share one context. Where the
    # platform default is fork, a queued sink built in that context cannot be
    # handed to a spawned child.
    multiprocessing.set_start_method("spawn", force=True)
    logger_setup(
        log_file,
        verbosity=verbosity,
        structured_stdout=json_sink == "startup",
    )
    if json_sink == "runtime":
        # The cluster-synced logging config adds the sink through this call.
        set_structured_stdout(True)
    log_failed_generation(loguru.logger, request_text, NODE_LABEL)
    child = multiprocessing.Process(
        target=runner_child, args=(loguru.logger, request_text)
    )
    child.start()
    child.join()
    logger_cleanup()
    if child.exitcode != 0:
        raise SystemExit(f"runner child exited with {child.exitcode}")


def main() -> None:
    """Run the mode named on the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("node", "default"), required=True)
    parser.add_argument("--request-text", required=True)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument(
        "--json-sink", choices=("startup", "runtime"), default="startup"
    )
    arguments = parser.parse_args()
    request_text = cast(str, arguments.request_text)
    if cast(str, arguments.mode) == "default":
        log_failed_generation(loguru.logger, request_text, DEFAULT_LABEL)
        return
    _run_node(
        request_text,
        cast(Path, arguments.log_file),
        cast(int, arguments.verbosity),
        cast(JsonSinkRoute, arguments.json_sink),
    )
