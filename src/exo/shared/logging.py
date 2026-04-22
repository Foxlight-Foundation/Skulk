from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import traceback
from collections.abc import Iterator
from datetime import UTC
from io import TextIOWrapper
from pathlib import Path
from typing import TYPE_CHECKING

import zstandard
from hypercorn import Config
from hypercorn.logging import Logger as HypercornLogger
from loguru import logger

if TYPE_CHECKING:
    from loguru import Message

_MAX_LOG_ARCHIVES = 5
_json_sink_id: int | None = None
_vector_process: subprocess.Popen[bytes] | None = None
_vector_pipe: TextIOWrapper | None = None

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


# HTTP access log patterns that are too noisy at INFO — demote to DEBUG
_NOISY_PATTERNS = ("GET /downloads", "GET /health")


class _InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Demote noisy polling endpoints to DEBUG
        msg = record.getMessage()
        if any(p in msg for p in _NOISY_PATTERNS):
            level = "DEBUG"

        logger.opt(depth=3, exception=record.exc_info).log(level, record.getMessage())


# ---------------------------------------------------------------------------
# Structured JSON sink — writes to Vector subprocess pipe or stdout
# ---------------------------------------------------------------------------


def _json_sink(message: Message) -> None:
    """Loguru sink that writes newline-delimited JSON to the Vector pipe.

    Each log entry is a single JSON object with fields that map directly
    to VictoriaLogs stream fields (``node_id``, ``component``) and the
    message field (``msg``).
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

    line = json.dumps(entry, default=str) + "\n"

    # Write to Vector's stdin pipe if available, otherwise stdout
    target = _vector_pipe or sys.__stdout__
    if target is not None:
        try:
            target.write(line)
            target.flush()
        except (BrokenPipeError, OSError):
            # Vector has exited — disable the sink
            global _json_sink_id  # noqa: PLW0603
            if _json_sink_id is not None:
                with contextlib.suppress(ValueError):
                    logger.remove(_json_sink_id)
                _json_sink_id = None
            logger.warning("Log shipper disconnected — JSON sink disabled")


# ---------------------------------------------------------------------------
# Vector subprocess management
# ---------------------------------------------------------------------------

_VECTOR_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "deployment" / "logging" / "vector.yaml"
)


def _start_vector(ingest_url: str) -> bool:
    """Spawn Vector as a child process reading JSON from a pipe.

    Returns True if Vector was started successfully, False otherwise.
    """
    global _vector_process, _vector_pipe  # noqa: PLW0603

    vector_bin = shutil.which("vector")
    if vector_bin is None:
        logger.warning(
            "Vector not found on PATH — structured logging disabled. Install: brew install vectordotdev/brew/vector"
        )
        return False

    if not _VECTOR_CONFIG_PATH.exists():
        logger.warning(
            f"Vector config not found at {_VECTOR_CONFIG_PATH} — structured logging disabled"
        )
        return False

    env = {
        **dict(os.environ),
        "SKULK_LOGGING_INGEST_URL": ingest_url,
        "EXO_LOGGING_INGEST_URL": ingest_url,  # legacy compat for vector.yaml
        "SKULK_VECTOR_DATA_DIR": str(Path.home() / ".skulk" / "vector"),
        "EXO_VECTOR_DATA_DIR": str(Path.home() / ".skulk" / "vector"),  # legacy compat
    }

    # Ensure the data dir exists
    Path(env["SKULK_VECTOR_DATA_DIR"]).mkdir(parents=True, exist_ok=True)

    _vector_process = subprocess.Popen(
        [vector_bin, "--config", str(_VECTOR_CONFIG_PATH)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    if _vector_process.stdin is not None:
        _vector_pipe = TextIOWrapper(
            _vector_process.stdin, encoding="utf-8", line_buffering=True
        )

    atexit.register(_stop_vector)
    logger.info(
        f"Vector started (pid={_vector_process.pid}), shipping logs to {ingest_url}"
    )
    return True


def _stop_vector() -> None:
    """Gracefully stop the Vector subprocess."""
    global _vector_process, _vector_pipe  # noqa: PLW0603
    if _vector_pipe is not None:
        with contextlib.suppress(Exception):
            _vector_pipe.close()
        _vector_pipe = None
    if _vector_process is not None:
        with contextlib.suppress(Exception):
            _vector_process.terminate()
            _vector_process.wait(timeout=5)
        _vector_process = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def logger_setup(
    log_file: Path | None,
    verbosity: int = 0,
    structured_stdout: bool = False,
    ingest_url: str = "",
):
    """Set up logging for this process — formatting, file handles, verbosity, and structured output.

    Args:
        log_file: Path to the local log file. ``None`` disables file logging.
        verbosity: 0 = INFO on console, >=1 = DEBUG with source locations.
        structured_stdout: When ``True`` and ``ingest_url`` is set, spawn
            Vector and pipe structured JSON logs to it.
        ingest_url: VictoriaLogs ingest URL. Required for Vector to know
            where to ship logs.
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

    if structured_stdout and ingest_url and _start_vector(ingest_url):
        global _json_sink_id  # noqa: PLW0603
        _json_sink_id = logger.add(
            _json_sink,
            level="INFO",
            enqueue=False,
        )


def set_structured_stdout(enabled: bool, ingest_url: str = "") -> None:
    """Enable or disable the structured JSON stdout sink at runtime.

    Called when logging config is synced across the cluster.  Safe to call
    repeatedly — adding when already active or removing when already
    inactive is a no-op.
    """
    global _json_sink_id  # noqa: PLW0603
    if enabled and ingest_url and _json_sink_id is None:
        if _start_vector(ingest_url):
            _json_sink_id = logger.add(
                _json_sink,
                level="INFO",
                enqueue=False,
            )
            logger.info("Structured JSON log shipping enabled")
    elif not enabled and _json_sink_id is not None:
        with contextlib.suppress(ValueError):
            logger.remove(_json_sink_id)
        _json_sink_id = None
        _stop_vector()
        logger.info("Structured JSON log shipping disabled")


def logger_cleanup():
    """Flush all queues before shutting down so any in-flight logs are written to disk"""
    logger.complete()
    _stop_vector()


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
