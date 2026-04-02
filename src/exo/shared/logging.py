from __future__ import annotations

import atexit
import json
import logging
import socket
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import zstandard
from hypercorn import Config
from hypercorn.logging import Logger as HypercornLogger
from loguru import logger

if TYPE_CHECKING:
    from loguru import Message

_MAX_LOG_ARCHIVES = 5


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
# VictoriaLogs sink — batches structured JSON and ships over HTTP
# ---------------------------------------------------------------------------


class _VictoriaLogsSink:
    """Loguru sink that pushes newline-delimited JSON to VictoriaLogs.

    The sink accumulates log entries in memory and flushes them either when
    the buffer hits ``batch_size`` or every ``flush_interval`` seconds,
    whichever comes first.  Flushes happen on the loguru background thread
    (``enqueue=True``), so they never block the event loop.

    For ERROR/CRITICAL messages the sink flushes immediately — these are
    the lines most likely to precede a hard crash where the periodic timer
    would never fire.
    """

    def __init__(
        self,
        url: str,
        node_id: str,
        flush_interval: float = 2.0,
        batch_size: int = 64,
    ) -> None:
        self._url = url
        self._node_id = node_id
        self._batch_size = batch_size
        self._buffer: list[str] = []
        self._lock = threading.Lock()

        # Periodic flush timer
        self._flush_interval = flush_interval
        self._timer: threading.Timer | None = None
        self._closed = False
        self._schedule_flush()

        atexit.register(self.close)

    # -- loguru sink interface ------------------------------------------------

    def __call__(self, message: Message) -> None:
        record = message.record
        entry = {
            "ts": record["time"].strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level": record["level"].name,
            "node_id": self._node_id,
            "component": (name.split(".")[1] if "." in name else name)
            if (name := record["name"])
            else "unknown",
            "module": name or "unknown",
            "function": record["function"],
            "line": record["line"],
            "msg": str(record["message"]),
        }
        if record["exception"] is not None:
            entry["exception"] = str(record["exception"])

        line = json.dumps(entry, default=str)

        flush_now = False
        with self._lock:
            self._buffer.append(line)
            if (
                len(self._buffer) >= self._batch_size
                or record["level"].no >= logging.ERROR
            ):
                flush_now = True

        if flush_now:
            self._flush()

    # -- flush mechanics ------------------------------------------------------

    def _schedule_flush(self) -> None:
        if self._closed:
            return
        self._timer = threading.Timer(self._flush_interval, self._timer_flush)
        self._timer.daemon = True
        self._timer.start()

    def _timer_flush(self) -> None:
        self._flush()
        self._schedule_flush()

    def _flush(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            payload = "\n".join(self._buffer) + "\n"
            self._buffer.clear()

        try:
            req = urllib.request.Request(
                self._url,
                data=payload.encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)  # noqa: S310 — URL is operator-configured
        except (urllib.error.URLError, OSError):
            # Remote logger is down — drop silently rather than cascade.
            # Local file and console sinks still capture everything.
            pass

    def close(self) -> None:
        """Flush remaining entries and stop the timer."""
        self._closed = True
        if self._timer is not None:
            self._timer.cancel()
        self._flush()


# Keep a module-level reference so logger_cleanup can flush it.
_victorialogs_sink: _VictoriaLogsSink | None = None


def logger_setup(
    log_file: Path | None,
    verbosity: int = 0,
    victorialogs_url: str | None = None,
    victorialogs_flush_interval: float = 2.0,
    victorialogs_batch_size: int = 64,
):
    """Set up logging for this process — formatting, file handles, verbosity, output, and optional remote shipping.

    Args:
        log_file: Path to the local log file. ``None`` disables file logging.
        verbosity: 0 = INFO on console, >=1 = DEBUG with source locations.
        victorialogs_url: Full VictoriaLogs ingest URL (e.g.
            ``http://host:9428/insert/jsonline?_stream_fields=...``).
            ``None`` disables remote log shipping.
        victorialogs_flush_interval: Seconds between periodic HTTP flushes.
        victorialogs_batch_size: Buffer size that triggers an immediate flush.
    """
    global _victorialogs_sink  # noqa: PLW0603

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

    if victorialogs_url:
        node_name = socket.gethostname()
        _victorialogs_sink = _VictoriaLogsSink(
            url=victorialogs_url,
            node_id=node_name,
            flush_interval=victorialogs_flush_interval,
            batch_size=victorialogs_batch_size,
        )
        logger.add(
            _victorialogs_sink,
            level="DEBUG",
            enqueue=False,
        )
        logger.info(f"Remote logging enabled: {victorialogs_url}")


def logger_cleanup():
    """Flush all queues before shutting down so any in-flight logs are written to disk"""
    logger.complete()
    if _victorialogs_sink is not None:
        _victorialogs_sink.close()


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
