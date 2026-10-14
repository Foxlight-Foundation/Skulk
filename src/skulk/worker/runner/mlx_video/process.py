"""Supervise a serial native render with bounded messages and process-group cleanup."""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Literal, TextIO, final

from pydantic import Field

from skulk.provisioning.mlx_video import runtime_environment
from skulk.shared.types.video import VideoStage
from skulk.utils.pydantic_ext import FrozenModel
from skulk.worker.runner.mlx_video.plan import NativeVideoJob


@final
class NativeProgress(FrozenModel):
    """One bounded stage/step event from the isolated engine."""

    stage: VideoStage
    step: int | None = Field(default=None, ge=0, le=512)
    total_steps: int | None = Field(default=None, ge=1, le=512)
    sampling_stage: Literal[1, 2] | None = Field(default=None, description="Base or refinement stage, preserving whole-job progress.")


@final
class NativeResult(FrozenModel):
    """Actual native sampling work and allocator high-water measurement."""

    steps: int = Field(ge=1, le=512)
    sampling_seconds: float = Field(ge=0, description="Observed native sampling duration in seconds.")
    total_generation_time: float = Field(ge=0, description="Wall time through encoding, sampling, decode and mux.")
    peak_memory_bytes: int = Field(ge=0, description="MLX allocator high-water bytes for this job; not whole-host RAM pressure.")


def _terminate(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=3)
    except ProcessLookupError:
        process.wait(timeout=3)
    finally:
        # A crashed/exited wrapper can leave its encoder child alive. Reaping
        # only the wrapper PID would miss those descendants on normal exit.
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)


def run_native_job(interpreter: Path, job: NativeVideoJob, *, progress: Callable[[NativeProgress], None], is_cancelled: Callable[[], bool]) -> NativeResult | None:
    """Run one local job, cancelling its whole group and retaining bounded diagnostics.

    The child independently reaps itself and FFmpeg when its runner parent dies.
    Messages cannot block rendering or grow an unbounded progress queue.
    """
    output = Path(job.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    request = output / "job.json"
    request.write_text(job.model_dump_json())
    messages: queue.Queue[str] = queue.Queue(maxsize=64)
    errors: deque[str] = deque(maxlen=24)
    process = subprocess.Popen([str(interpreter), "-I", "-B", "-m", "skulk_mlx_video", "--job", str(request), "--parent-pid", str(os.getpid())], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=runtime_environment(), start_new_session=True)

    def consume(stream: TextIO, *, diagnostics: bool) -> None:
        with stream:
            while line := stream.readline(8193):
                if diagnostics:
                    errors.append(line[-8192:])
                elif len(line) <= 8192 and line.endswith("\n"):
                    try:
                        messages.put_nowait(line)
                    except queue.Full:
                        # Intermediate progress is replaceable; completion is
                        # a separate bounded result file read after process exit.
                        with suppress(queue.Empty):
                            messages.get_nowait()
                        with suppress(queue.Full):
                            messages.put_nowait(line)

    assert process.stdout is not None and process.stderr is not None
    readers = [threading.Thread(target=consume, args=(process.stdout,), kwargs={"diagnostics": False}, daemon=True), threading.Thread(target=consume, args=(process.stderr,), kwargs={"diagnostics": True}, daemon=True)]
    for reader in readers:
        reader.start()
    started = time.monotonic()
    try:
        while process.poll() is None or not messages.empty():
            if is_cancelled():
                return None
            if time.monotonic() - started > 6 * 60 * 60:
                raise ValueError("native video render exceeded its bounded execution deadline")
            try:
                message = messages.get(timeout=0.1)
            except queue.Empty:
                continue
            progress(NativeProgress.model_validate_json(message))
        if process.wait() != 0:
            raise ValueError("native video render failed: " + "".join(errors)[-4096:])
        result = output / "result.json"
        if not result.is_file() or result.stat().st_size > 4096:
            raise ValueError("native video render did not return a bounded completion record")
        return NativeResult.model_validate_json(result.read_bytes())
    finally:
        _terminate(process)
        for reader in readers:
            reader.join(timeout=2)
        request.unlink(missing_ok=True)
