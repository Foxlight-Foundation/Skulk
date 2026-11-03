"""Bounded private resident control loop; any failure withdraws its model."""

from __future__ import annotations

import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, Protocol

from pydantic import ConfigDict, Field

from .contracts import (
    MAX_NATIVE_MUSIC_RECORD_BYTES,
    NativeMusicCompleted,
    NativeMusicContract,
    NativeMusicExecution,
    NativeMusicFailed,
    NativeMusicFrame,
    NativeMusicProgress,
    NativeMusicReady,
    NativeMusicReadyFrame,
    NativeMusicRequest,
    NativeMusicResult,
    NativeMusicServeConfig,
    NativeMusicServeRequest,
)

Progress = Callable[[NativeMusicProgress], None]


class _BuildIdentity(NativeMusicContract):
    """Identity projection from a complete manifest already verified by Core."""

    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    payload_sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$",
        description="Actual installed payload identity checked against Core's pin.",
    )


class MusicModel(Protocol):
    """The loaded-model facts and exclusive execution used by native control."""

    @property
    def configuration(self) -> NativeMusicServeConfig:
        """Return the immutable verified artifact/device policy."""
        ...

    @property
    def ready(self) -> NativeMusicReady:
        """Return measured complete model initialization, never an install hint."""
        ...

    def run(self, request: NativeMusicRequest, progress: Progress) -> NativeMusicResult:
        """Produce an exclusive complete WAV and durable private result record."""
        ...


def installed_payload_identity() -> str:
    """Read installed identity, independently authenticated by the parent before launch."""
    with (Path(__file__).parent / "build.json").open("rb") as stream:
        data = stream.read(128 * 1024 + 1)
    if len(data) > 128 * 1024:
        raise ValueError("native music build manifest exceeds its bound")
    return _BuildIdentity.model_validate_json(data).payload_sha256


def load_resident(
    configuration: NativeMusicServeConfig, payload_sha256: str, progress: Progress
) -> MusicModel:
    """Import heavy libraries only in the separately supervised runtime process."""
    from .engine import load_music

    return load_music(configuration, payload_sha256, progress)


def _retained_result(
    request: NativeMusicRequest,
    result: NativeMusicResult,
    configuration: NativeMusicServeConfig,
) -> None:
    directory = Path(configuration.directory) / "requests" / request.request_id
    if (
        Path(request.output_directory) != directory
        or directory.is_symlink()
        or directory.parent.is_symlink()
    ):
        raise ValueError("native music result escaped its private request lifetime")
    descriptor = os.open(
        directory / "result.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    )
    with os.fdopen(descriptor, "rb") as stream:
        facts = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(facts.st_mode)
            or facts.st_size > 16 * 1024
            or stat.S_IMODE(facts.st_mode) != 0o600
        ):
            raise ValueError(
                "native music retained result is not bounded private metadata"
            )
        data = stream.read(16 * 1024 + 1)
    if len(data) > 16 * 1024 or NativeMusicResult.model_validate_json(data) != result:
        raise ValueError("native music completion differs from retained result")


def serve(
    configuration: NativeMusicServeConfig,
    requests: BinaryIO,
    emit: Callable[[NativeMusicFrame], None],
    *,
    payload_sha256: str,
    load: Callable[[NativeMusicServeConfig, str, Progress], MusicModel] = load_resident,
) -> None:
    """Load once and serialize framed requests until clean EOF or the first failure.

    Contiguous sequence admission rejects replay without growing an ID history.
    Completion follows fsynced private result custody. Failure is terminal even
    when upstream reports a cooperative timeout: its inference thread may still
    be active, so the CLI kills the whole owned group rather than reuse it.
    """
    request_id: str | None = None
    try:

        def initialization_progress(record: NativeMusicProgress) -> None:
            if record.request_id is not None or record.stage not in (
                "verifying_weights",
                "loading",
            ):
                raise ValueError("native music initialization emitted foreign progress")
            emit(record)

        model = load(configuration, payload_sha256, initialization_progress)
        ready = model.ready
        expected_execution = NativeMusicExecution(
            device=configuration.device,
            precision=configuration.precision,
            text_encoder_device=configuration.device,
            text_encoder_precision=configuration.precision,
            autoencoder_device=configuration.device,
            autoencoder_precision="float16"
            if configuration.device == "mps"
            else configuration.precision,
        )
        if (
            model.configuration != configuration
            or ready.model_identity != configuration.model_identity
            or ready.payload_sha256 != payload_sha256
            or ready.execution != expected_execution
        ):
            raise ValueError("native music loaded a different model/runtime policy")
        emit(NativeMusicReadyFrame(ready=ready))
        sequence = 0
        while line := requests.readline(MAX_NATIVE_MUSIC_RECORD_BYTES + 1):
            request_id = None
            if len(line) > MAX_NATIVE_MUSIC_RECORD_BYTES or not line.endswith(b"\n"):
                raise ValueError("native music request framing exceeds its bound")
            envelope = NativeMusicServeRequest.model_validate_json(line)
            request = envelope.request
            request_id = request.request_id
            if envelope.sequence != sequence + 1:
                raise ValueError("native music request sequence is not contiguous")
            sequence = envelope.sequence

            def progress(
                record: NativeMusicProgress, *, owner: str = request.request_id
            ) -> None:
                if record.request_id != owner or record.stage not in (
                    "conditioning",
                    "generating",
                    "writing",
                ):
                    raise ValueError("native music request emitted foreign progress")
                emit(record)

            result = model.run(request, progress)
            if (
                result.request_id,
                result.model_identity,
                result.payload_sha256,
                result.operation,
                result.seed,
                result.source_sha256,
                result.execution,
            ) != (
                request.request_id,
                ready.model_identity,
                payload_sha256,
                request.operation,
                request.seed,
                request.source.sha256 if request.source is not None else None,
                ready.execution,
            ):
                raise ValueError(
                    "native music result belongs to a different request or model"
                )
            if (
                request.operation != "generate"
                and result.output.frames != request.target_frames
            ):
                raise ValueError("native music edited output changed timeline geometry")
            _retained_result(request, result, configuration)
            emit(NativeMusicCompleted(result=result))
    except Exception:
        emit(NativeMusicFailed(request_id=request_id))
        raise
