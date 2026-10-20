"""Cancel, bound and reap media preparation in the selected engine interpreter."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Final

import psutil
from loguru import logger

from skulk.provisioning.mlx_video import runtime_environment
from skulk.worker.runner import video_input
from skulk.worker.runner.video_input import MediaInputRequest, VerifiedMediaInput

_PREPARATION_TIMEOUT_SECONDS: Final = 60.0


def _observe_exit(process: subprocess.Popen[str]) -> os.waitid_result | None:
    # Keep the session leader's PID reserved until its group is stopped. poll()
    # and wait() reap it, allowing a later group signal to address a reused PID.
    return os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)


def _signal_group(process: subprocess.Popen[str], sent: signal.Signals) -> None:
    try:
        os.killpg(process.pid, sent)
    except ProcessLookupError:
        return
    except PermissionError:
        # Darwin returns EPERM for an owned group containing only zombies.
        # A live member must still surface the real failure; never turn an
        # inability to stop a decoder into successful cancellation.
        for identifier in psutil.pids():
            try:
                if os.getpgid(identifier) == process.pid and psutil.Process(identifier).status() != psutil.STATUS_ZOMBIE:
                    raise
            except (ProcessLookupError, psutil.NoSuchProcess):
                continue


def _reap(process: subprocess.Popen[str]) -> None:
    _signal_group(process, signal.SIGTERM)
    deadline = time.monotonic() + 1
    while _observe_exit(process) is None and time.monotonic() < deadline:
        time.sleep(0.025)
    _signal_group(process, signal.SIGKILL)
    process.wait(timeout=3)


def prepare_media_input(
    interpreter: Path,
    request: MediaInputRequest,
    output_directory: Path,
    *,
    is_cancelled: Callable[[], bool],
    native: bool = False,
) -> VerifiedMediaInput | None:
    """Prepare verified input with a 60-second deadline and bounded diagnostics.

    Args:
        interpreter: Exact selected engine interpreter, never base Python.
        request: Worker-resolved media path and output bounds.
        output_directory: Command-owned directory for ephemeral protocol files.
        is_cancelled: Drain pending cancellation while preparation runs.
        native: Execute the wheel's own bundled copy instead of the worker file.

    Returns:
        Verified input facts, or None after cancellation. Protocol files are
        removed on every exit; the caller owns normalized media cleanup.
    """
    if is_cancelled():
        return None
    output_directory.mkdir(parents=True, exist_ok=True)
    request_path = output_directory / "_media_request.json"
    result_path = output_directory / "_media_verified.json"
    request_path.write_text(request.model_dump_json())
    command = [str(interpreter), "-I", "-B"]
    command.extend(["-m", "skulk_mlx_video.media"] if native else [str(Path(video_input.__file__))])
    command.extend(["--request", str(request_path), "--result", str(result_path), "--parent-pid", str(os.getpid())])
    errors: deque[str] = deque(maxlen=24)
    process: subprocess.Popen[str] | None = None
    reader: threading.Thread | None = None
    try:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, start_new_session=True, env=runtime_environment())
        stream = process.stderr
        assert stream is not None

        def consume() -> None:
            with stream:
                while line := stream.readline(8193):
                    errors.append(line[-8192:])

        reader = threading.Thread(target=consume, daemon=True)
        reader.start()
        deadline = time.monotonic() + _PREPARATION_TIMEOUT_SECONDS
        while (exit_status := _observe_exit(process)) is None:
            if is_cancelled():
                return None
            if time.monotonic() >= deadline:
                raise ValueError("source media preparation exceeded its 60-second deadline")
            time.sleep(0.1)
        reader.join(timeout=1)
        if is_cancelled():
            return None
        if exit_status.si_code != os.CLD_EXITED or exit_status.si_status != 0:
            logger.warning("Video input preparation failed: {}", "".join(errors))
            # Decoder tracebacks belong in worker diagnostics. Our own bounded
            # validation reasons are sufficient for the operator-facing refusal.
            reason = next((line.removeprefix("ValueError: ").strip() for line in reversed(errors) if line.startswith("ValueError: ")), "the selected engine could not decode this source within its media limits")
            raise ValueError(reason)
        if not result_path.is_file() or result_path.stat().st_size > 16 * 1024:
            raise ValueError("source media preparation did not return bounded verified facts")
        result = VerifiedMediaInput.model_validate_json(result_path.read_bytes())
        expected_path = request.destination if request.operation == "audio" else request.source
        if result.path != expected_path:
            raise ValueError("source media preparation returned an unexpected local path")
        return result
    finally:
        try:
            if process is not None:
                _reap(process)
        finally:
            if reader is not None:
                reader.join(timeout=1)
            request_path.unlink(missing_ok=True)
            result_path.unlink(missing_ok=True)
