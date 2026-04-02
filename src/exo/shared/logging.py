from __future__ import annotations

import json
import logging
import socket
import sys
import traceback
from collections.abc import Iterator
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING

import zstandard
from hypercorn import Config
from hypercorn.logging import Logger as HypercornLogger
from loguru import logger

if TYPE_CHECKING:
    from loguru import Message

_MAX_LOG_ARCHIVES = 5

_node_name: str = socket.gethostname()


def _zstd_compress(filepath: str) -> None:
    source = Path(filepath)
    dest = source.with_suffix(source.suffix + ".zst")
    cctx = zstandard.ZstdCompressor()
    with open(source, "rb") as f_in, open(dest, "wb") as f_out:
        cctx.copy_stream(f_in, f_out)
    source.unlink()


def _once_then_never() -> Iterator[bool]:
    yield True
    while True:
        yield False


class InterceptLogger(HypercornLogger):
    def __init__(self, config: Config):
        super().__init__(config)
        assert self.error_logger
        self.error_logger.handlers = [_InterceptHandler()]


class _InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        logger.opt(depth=3, exception=record.exc_info).log(level, record.getMessage())


# ---------------------------------------------------------------------------
# Structured JSON sink — writes one JSON object per line to stdout.
# Vector (or any log shipper) reads stdout and forwards to VictoriaLogs.
# ---------------------------------------------------------------------------


def _json_sink(message: Message) -> None:
    """Loguru sink that writes newline-delimited JSON to stdout.

    Each log entry is a single JSON object with fields that map directly
    to VictoriaLogs stream fields (``node_id``, ``component``) and the
    message field (``msg``).  Vector consumes this on stdin and ships it.
    """
    record = message.record
    name = record["name"]
    entry = {
        "ts": record["time"].astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        + "Z",
        "level": record["level"].name,
        "node_id": _node_name,
        "component": (name.split(".")[1] if "." in name else name)
        if name
        else "unknown",
        "module": name or "unknown",
        "function": record["function"],
        "line": record["line"],
        "msg": str(record["message"]),
    }
    if record["exception"] is not None:
        exc_type, exc_value, exc_tb = record["exception"]
        if exc_type is not None:
            entry["exception"] = "".join(
                traceback.format_exception(exc_type, exc_value, exc_tb)
            )

    # Write to the real stdout (not loguru's stderr sink).
    # Use sys.__stdout__ to bypass any redirections.
    stdout = sys.__stdout__
    if stdout is not None:
        stdout.write(json.dumps(entry, default=str) + "\n")
        stdout.flush()


def logger_setup(
    log_file: Path | None,
    verbosity: int = 0,
    structured_stdout: bool = False,
):
    """Set up logging for this process — formatting, file handles, verbosity, and structured output.

    Args:
        log_file: Path to the local log file. ``None`` disables file logging.
        verbosity: 0 = INFO on console, >=1 = DEBUG with source locations.
        structured_stdout: When ``True``, emit structured JSON on stdout for
            consumption by Vector or another log shipper.
    """
    logging.getLogger("exo_pyo3_bindings").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logger.remove()

    # replace all stdlib loggers with _InterceptHandlers that log to loguru
    logging.basicConfig(handlers=[_InterceptHandler()], level=0)

    if verbosity == 0:
        logger.add(
            sys.__stderr__,  # type: ignore
            format="[ {time:hh:mm:ss.SSSSA} | <level>{level: <8}</level>] <level>{message}</level>",
            level="INFO",
            colorize=True,
            enqueue=True,
        )
    else:
        logger.add(
            sys.__stderr__,  # type: ignore
            format="[ {time:HH:mm:ss.SSS} | <level>{level: <8}</level> | {name}:{function}:{line} ] <level>{message}</level>",
            level="DEBUG",
            colorize=True,
            enqueue=True,
        )
    if log_file:
        rotate_once = _once_then_never()
        logger.add(
            log_file,
            format="[ {time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} ] {message}",
            level="INFO",
            colorize=False,
            enqueue=True,
            rotation=lambda _, __: next(rotate_once),
            retention=_MAX_LOG_ARCHIVES,
            compression=_zstd_compress,
        )

    if structured_stdout:
        logger.add(
            _json_sink,
            level="INFO",
            enqueue=False,
        )


def logger_cleanup():
    """Flush all queues before shutting down so any in-flight logs are written to disk"""
    logger.complete()


""" --- TODO: Capture MLX Log output:
import contextlib
import sys
from loguru import logger

class StreamToLogger:

    def __init__(self, level="INFO"):
        self._level = level

    def write(self, buffer):
        for line in buffer.rstrip().splitlines():
            logger.opt(depth=1).log(self._level, line.rstrip())

    def flush(self):
        pass

logger.remove()
logger.add(sys.__stdout__)

stream = StreamToLogger()
with contextlib.redirect_stdout(stream):
    print("Standard output is sent to added handlers.")
"""
