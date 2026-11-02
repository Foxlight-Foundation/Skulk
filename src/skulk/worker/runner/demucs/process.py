"""Bounded native separation protocol with owned process-tree and file cleanup."""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import TextIO

from skulk.shared.types.audio_separation import (
    MAX_SEPARATION_STEM_BYTES,
    NativeSeparationJob,
    NativeSeparationProgress,
    NativeSeparationResult,
    stem_from_wav,
)
from skulk.utils.process_groups import observe_process_exit, terminate_process_group


def separation_environment(threads: int) -> dict[str, str]:
    """Preserve system/device selection while removing inherited Python tuning.

    Native dependencies belong to the provisioned environment. The adapter owns
    offline loading, FP32 execution and its admitted CPU thread count.
    """
    inherited = (
        "PATH",
        "HOME",
        "USER",
        "TMPDIR",
        "TMP",
        "TEMP",
        "SYSTEMROOT",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HSA_OVERRIDE_GFX_VERSION",
    )
    environment = {name: os.environ[name] for name in inherited if name in os.environ}
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
        }
    )
    return environment


def verify_native_result(
    directory: Path, job: NativeSeparationJob
) -> NativeSeparationResult:
    """Verify bounded completion JSON and all four exact files before returning."""
    with (directory / "result.json").open("rb") as result:
        record = result.read(16 * 1024 + 1)
    if len(record) > 16 * 1024:
        raise ValueError("native separation completion record exceeds 16 KiB")
    measured = NativeSeparationResult.model_validate_json(record)
    if measured.output.source != job.source or measured.device != job.device:
        raise ValueError("native separation completed a different source or device")
    for stem in measured.output.stems:
        with (directory / f"{stem.name}.wav").open("rb") as source:
            data = source.read(MAX_SEPARATION_STEM_BYTES + 1)
        if stem_from_wav(stem.name, data) != stem:
            raise ValueError(
                "native separation stem bytes differ from its completion manifest"
            )
    return measured


def run_native_separation(
    interpreter: Path,
    job: NativeSeparationJob,
    *,
    progress: Callable[[NativeSeparationProgress], None],
    is_cancelled: Callable[[], bool],
    resource_guard: Callable[[int], str | None] | None = None,
    timeout_seconds: float = 6 * 60 * 60,
) -> NativeSeparationResult | None:
    """Execute one admitted job and retain files only after complete verification.

    The job directory must not exist. Cancellation returns None after terminating
    the complete owned process group and removing partial files. Native failures,
    resource refusals, deadlines and protocol violations raise after the same
    cleanup. Successful verified files remain for bounded addressed delivery.
    """
    if is_cancelled():
        return None
    directory = Path(job.output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    request = directory / "job.json"
    process: subprocess.Popen[str] | None = None
    readers: list[threading.Thread] = []
    messages: queue.Queue[str] = queue.Queue(maxsize=64)
    errors: deque[str] = deque(maxlen=24)
    complete = False

    def consume(stream: TextIO, *, diagnostics: bool) -> None:
        with stream:
            while line := stream.readline(2049):
                if diagnostics:
                    errors.append(line[-2048:])
                elif len(line) <= 2048 and line.endswith("\n"):
                    # Progress is replaceable. Completion is a separately bounded
                    # file, so a noisy or slow observer cannot lose the result.
                    try:
                        messages.put_nowait(line)
                    except queue.Full:
                        with suppress(queue.Empty):
                            messages.get_nowait()
                        with suppress(queue.Full):
                            messages.put_nowait(line)

    try:
        request.write_text(job.model_dump_json())
        process = subprocess.Popen(
            [
                str(interpreter),
                "-I",
                "-B",
                "-m",
                "skulk_demucs",
                "--job",
                str(request),
                "--parent-pid",
                str(os.getpid()),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=separation_environment(job.threads),
            start_new_session=True,
        )
        assert process.stdout is not None and process.stderr is not None
        readers = [
            threading.Thread(
                target=consume,
                args=(process.stdout,),
                kwargs={"diagnostics": False},
                daemon=True,
            ),
            threading.Thread(
                target=consume,
                args=(process.stderr,),
                kwargs={"diagnostics": True},
                daemon=True,
            ),
        ]
        for reader in readers:
            reader.start()
        started = time.monotonic()
        last_resource_check = started - 1
        while (
            exit_status := observe_process_exit(process)
        ) is None or not messages.empty():
            if is_cancelled():
                return None
            now = time.monotonic()
            if now - started >= timeout_seconds:
                raise ValueError("native separation exceeded its execution deadline")
            if resource_guard is not None and now - last_resource_check >= 0.5:
                last_resource_check = now
                refusal = resource_guard(process.pid)
                if refusal is not None:
                    raise ValueError(refusal)
            try:
                message = messages.get(timeout=0.1)
            except queue.Empty:
                continue
            progress(NativeSeparationProgress.model_validate_json(message))
        assert exit_status is not None
        if exit_status.si_code != os.CLD_EXITED or exit_status.si_status != 0:
            raise ValueError("native separation failed: " + "".join(errors)[-4096:])
        # Drain every descendant before trusting result files. Keep the leader
        # unreaped until group cleanup so its PID cannot be reused underneath us.
        terminate_process_group(process, grace_seconds=1.0, confirm_members=True)
        process = None
        if is_cancelled():
            return None
        result = verify_native_result(directory, job)
        complete = True
        return result
    finally:
        if process is not None:
            # Preserve owned files for recovery if the process drain fails;
            # removing them while a native writer is alive would race cleanup.
            terminate_process_group(process, grace_seconds=1.0, confirm_members=True)
        for reader in readers:
            reader.join(timeout=2)
        request.unlink(missing_ok=True)
        if not complete:
            shutil.rmtree(directory)
