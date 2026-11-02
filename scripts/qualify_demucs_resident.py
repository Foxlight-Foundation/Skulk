"""Exercise an actual sealed resident model, serial reuse, cancellation and recovery.

All inputs and outputs stay in a new private qualification directory. This
command never downloads, publishes, advertises capacity or changes a running
service. Synthetic short signals establish execution/custody, not music quality
or admission limits for a complete 120-second production workload.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import struct
import subprocess
import sys
import time
import wave
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

import psutil

from skulk.shared.types.audio_separation import (
    NativeSeparationJob,
    NativeSeparationProgress,
    NativeSeparationResult,
    NativeSeparationServeConfig,
    SeparationDevice,
    SeparationSource,
    source_from_wav,
)
from skulk.utils.runtime_files import RuntimeLock, private_directory, write_private
from skulk.worker.runner.demucs.resident import ResidentSeparator, spawn_resident

_WEIGHTS = "d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd"


class Arguments(argparse.Namespace):
    """Explicit interpreter, exact payload identity, local weights and new output."""

    interpreter: Path
    weights: Path
    payload_sha256: str
    output: Path
    device: SeparationDevice
    owner_probe: bool


@dataclass
class Processes:
    """Retain exact OS process identities and sample a bounded native RSS guard."""

    processes: list[psutil.Process] = field(default_factory=list)
    peak_rss_bytes: int = 0

    def spawn(
        self,
        interpreter: Path,
        configuration: Path,
        descriptor: int,
        environment: dict[str, str],
    ) -> subprocess.Popen[bytes]:
        """Observe the actual production spawn without substituting native code."""
        process = spawn_resident(interpreter, configuration, descriptor, environment)
        self.processes.append(psutil.Process(process.pid))
        return process

    def guard(self, pid: int) -> str | None:
        """Sample actual resident RSS and refuse more than 3 GiB for these fixtures."""
        process = self.processes[-1]
        if process.pid != pid:
            raise AssertionError("qualification observed a foreign process")
        resident = process.memory_info().rss
        self.peak_rss_bytes = max(self.peak_rss_bytes, resident)
        return "qualification RSS budget exceeded" if resident > 3 * 1024**3 else None


def fixture(path: Path, *, rate: int, frames: int, channels: int) -> SeparationSource:
    """Write a bounded non-silent PCM16 signal with exact source geometry."""
    with wave.open(str(path), "wb") as output:
        output.setparams((channels, 2, rate, frames, "NONE", "not compressed"))
        for offset in range(0, frames, 4096):
            samples = bytearray()
            for frame in range(offset, min(offset + 4096, frames)):
                value = int(3000 * math.sin(2 * math.pi * 220 * frame / rate))
                samples.extend(struct.pack("<h", value))
                if channels == 2:
                    samples.extend(struct.pack("<h", -value))
            output.writeframes(samples)
    return source_from_wav(path.read_bytes())


def control(
    arguments: Arguments, root: Path, processes: Processes
) -> ResidentSeparator:
    """Use the real supervisor and exact package/weight/device policy."""
    return ResidentSeparator(
        arguments.interpreter.absolute(),
        root,
        NativeSeparationServeConfig(
            weights_path=str(arguments.weights.resolve(strict=True)),
            device=arguments.device,
            threads=2,
        ),
        payload_sha256=arguments.payload_sha256,
        weights_sha256=_WEIGHTS,
        spawn=processes.spawn,
        resource_guard=processes.guard,
    )


def owner_death(arguments: Arguments, root: Path) -> None:
    """Prove the actual native watchdog and inherited fence survive controller death.

    Pause only our exact native child so the watchdog cannot win the race before
    lock inspection. The lock must remain held after its controller exits, then
    release only when the resumed native watchdog has stopped the process.
    """
    native: psutil.Process | None = None
    with (root / "owner-probe.log").open("wb") as log:
        owner = subprocess.Popen(
            [
                sys.executable,
                "-B",
                str(Path(__file__).absolute()),
                "--interpreter",
                str(arguments.interpreter.absolute()),
                "--weights",
                str(arguments.weights.resolve(strict=True)),
                "--payload-sha256",
                arguments.payload_sha256,
                "--device",
                arguments.device,
                "--output",
                str(root / "owner-probe"),
                "--owner-probe",
            ],
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            receipt = root / "owner-probe" / "native.pid"
            deadline = time.monotonic() + 240
            while not receipt.exists():
                if owner.poll() is not None or time.monotonic() >= deadline:
                    raise AssertionError("resident owner probe failed to load")
                time.sleep(0.1)
            native = psutil.Process(int(receipt.read_text()))
            if native.ppid() != owner.pid or os.getpgid(native.pid) != native.pid:
                raise AssertionError("owner probe returned a foreign native process")
            native.send_signal(signal.SIGSTOP)
            owner.terminate()
            owner.wait(timeout=10)
            try:
                acquired = RuntimeLock(receipt.parent / "model", name="service.lock")
            except BlockingIOError:
                pass
            else:
                acquired.close()
                raise AssertionError("native lifetime lost its inherited owner fence")
            native.send_signal(signal.SIGCONT)
            deadline = time.monotonic() + 10
            while native.is_running() and native.status() != psutil.STATUS_ZOMBIE:
                if time.monotonic() >= deadline:
                    raise AssertionError(
                        "native watchdog did not stop after owner death"
                    )
                time.sleep(0.05)
            RuntimeLock(receipt.parent / "model", name="service.lock").close()
        finally:
            if native is not None and native.is_running():
                with suppress(psutil.NoSuchProcess):
                    native.kill()
            if owner.poll() is None:
                owner.kill()
            owner.wait(timeout=10)
            if owner.stdin is not None:
                owner.stdin.close()


def main() -> None:
    """Run immutable model reuse, in-flight cancellation, reload and watchdog checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interpreter", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--payload-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda", "rocm"), default="cpu"
    )
    parser.add_argument("--owner-probe", action="store_true", help=argparse.SUPPRESS)
    arguments = Arguments()
    parser.parse_args(namespace=arguments)
    root = arguments.output.absolute()
    if root.exists() or root.is_symlink():
        raise ValueError("resident qualification requires a new private directory")
    private_directory(root)
    processes = Processes()
    resident = control(arguments, root / "model", processes)
    started = time.monotonic()
    ready = resident.start(is_cancelled=lambda: False, timeout_seconds=180)
    if arguments.owner_probe:
        try:
            write_private(
                root / "native.pid", str(processes.processes[-1].pid).encode()
            )
            _ = sys.stdin.buffer.read(1)
        finally:
            resident.close()
        return

    results: list[NativeSeparationResult] = []
    job_seconds: list[float] = []
    sources: list[tuple[Path, SeparationSource]] = []

    def job(
        number: int, *, rate: int, frames: int, channels: int
    ) -> NativeSeparationJob:
        source = root / f"source-{number}.wav"
        facts = fixture(source, rate=rate, frames=frames, channels=channels)
        sources.append((source, facts))
        return NativeSeparationJob(
            weights_path=resident.configuration.weights_path,
            source_path=str(source),
            output_dir=str(resident.jobs / f"{number:032x}"),
            source=facts,
            device=arguments.device,
            threads=2,
            segment_seconds=2.0,
        )

    def complete(request: NativeSeparationJob) -> None:
        job_started = time.monotonic()
        result = resident.run(
            request,
            progress=lambda _: None,
            is_cancelled=lambda: False,
            timeout_seconds=180,
        )
        if (
            result is None
            or result.load_seconds != 0
            or not any(stem.peak > 0 for stem in result.output.stems)
        ):
            raise AssertionError("resident warm execution failed or returned silence")
        job_seconds.append(time.monotonic() - job_started)
        results.append(result)

    try:
        complete(job(1, rate=8000, frames=8001, channels=1))
        complete(job(2, rate=44101, frames=44102, channels=2))
        if len(processes.processes) != 1 or not resident.is_ready():
            raise AssertionError("serial jobs did not retain one ready native process")
        inference_started: float | None = None

        def progress(record: NativeSeparationProgress) -> None:
            nonlocal inference_started
            if record.stage == "separating" and inference_started is None:
                inference_started = time.monotonic()

        request = job(3, rate=8000, frames=60 * 8000, channels=1)
        cancelled = resident.run(
            request,
            progress=progress,
            is_cancelled=lambda: inference_started is not None
            and time.monotonic() - inference_started >= 0.25,
            timeout_seconds=180,
        )
        if cancelled is not None or inference_started is None or resident.is_ready():
            raise AssertionError("in-flight cancellation did not withdraw the lifetime")
        if Path(request.output_dir).exists():
            raise AssertionError(
                "cancelled output remained after confirmed native drain"
            )
    finally:
        resident.close()

    resident = control(arguments, root / "model", processes)
    try:
        replacement = resident.start(is_cancelled=lambda: False, timeout_seconds=180)
        complete(job(4, rate=8000, frames=8001, channels=1))
        if len(processes.processes) != 2 or processes.processes[0].is_running():
            raise AssertionError(
                "cancellation recovery did not replace the native lifetime"
            )
        for number in (1, 2, 4):
            if not (resident.jobs / f"{number:032x}" / "vocals.wav").is_file():
                raise AssertionError("recovery lost an earlier completed result")
        for path, original in sources:
            if source_from_wav(path.read_bytes()) != original:
                raise AssertionError("resident execution changed an original")
    finally:
        resident.close()
    owner_death(arguments, root)
    write_private(
        root / "qualification.json",
        json.dumps(
            {
                "status": "pass",
                "ready": ready.model_dump(mode="json"),
                "replacement_ready": replacement.model_dump(mode="json"),
                "results": [result.model_dump(mode="json") for result in results],
                "supervised_job_seconds": job_seconds,
                "sampled_native_peak_rss_bytes": processes.peak_rss_bytes,
                "whole_qualification_seconds": time.monotonic() - started,
                "serial_reuse": True,
                "inference_cancel_and_reload": True,
                "owner_death_and_inherited_fence": True,
                "musical_quality_or_full_workload_qualification": False,
            },
            indent=2,
        ).encode(),
    )
    print("resident native qualification passed", flush=True)


if __name__ == "__main__":
    main()
