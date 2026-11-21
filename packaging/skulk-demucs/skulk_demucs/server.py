"""Bounded private resident protocol; a failed request ends the native lifetime."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, BinaryIO, Literal, Protocol

from pydantic import Field, JsonValue, TypeAdapter

from .contracts import (
    NativeSeparationCompleted,
    NativeSeparationFailed,
    NativeSeparationFrame,
    NativeSeparationJob,
    NativeSeparationProgress,
    NativeSeparationReady,
    NativeSeparationResidentProgress,
    NativeSeparationResult,
    NativeSeparationServeConfig,
    NativeSeparationServeRequest,
    SeparationStemName,
)

MAX_NATIVE_RECORD_BYTES = 16 * 1024
_PAYLOAD: TypeAdapter[str] = TypeAdapter(
    Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
)
_BUILD = TypeAdapter(dict[str, JsonValue])


class ResidentSeparator(Protocol):
    """Only the model policy and verified facts the private control loop needs."""

    @property
    def configuration(self) -> NativeSeparationServeConfig:
        """Return the fixed loaded model/device/thread policy."""
        ...

    @property
    def weights_sha256(self) -> str:
        """Return the complete verified weight identity."""
        ...

    @property
    def load_seconds(self) -> float:
        """Return observed verification, construction and placement wall time."""
        ...

    @property
    def sample_rate(self) -> Literal[44100]:
        """Return the loaded model's checked native sampling rate."""
        ...

    @property
    def audio_channels(self) -> Literal[2]:
        """Return the loaded model's checked stereo channel count."""
        ...

    @property
    def stems(self) -> tuple[SeparationStemName, ...]:
        """Return the loaded model's checked canonical output order."""
        ...

    def run(
        self,
        job: NativeSeparationJob,
        progress: Callable[[NativeSeparationProgress], None],
    ) -> NativeSeparationResult:
        """Execute one job without reconstructing the resident model."""
        ...


def installed_payload_identity() -> str:
    """Read identity from the already verified installed package, never its request.

    Core independently authenticates all installed bytes before launching this
    process and compares the returned identity with its exact selected pin.
    """
    with (Path(__file__).parent / "build.json").open("rb") as source:
        content = source.read(128 * 1024 + 1)
    if len(content) > 128 * 1024:
        raise ValueError("installed native manifest exceeds its bound")
    document = _BUILD.validate_json(content)
    return _PAYLOAD.validate_python(document.get("payload_sha256"), strict=True)


def load_resident(configuration: NativeSeparationServeConfig) -> ResidentSeparator:
    """Import native libraries only in the isolated inference process."""
    from .engine import load_separator

    return load_separator(configuration)


def serve(
    configuration: NativeSeparationServeConfig,
    requests: BinaryIO,
    emit: Callable[[NativeSeparationFrame], None],
    *,
    payload_sha256: str,
    load: Callable[[NativeSeparationServeConfig], ResidentSeparator] = load_resident,
) -> None:
    """Load once, then serialize bounded jobs until EOF or the first failure.

    Contiguous sequence admission rejects transport replay without an unbounded
    history. The supervisor owns request IDs and directories and never pipelines
    jobs. Completion carries facts only; raw WAVs remain in owned private files.
    Any request/protocol/inference failure is reported without private detail,
    then escapes so this process cannot reuse a potentially damaged model state.
    """
    request_id: str | None = None
    try:
        separator = load(configuration)
        if separator.configuration != configuration:
            raise ValueError("native model loaded a different immutable policy")
        emit(
            NativeSeparationReady(
                payload_sha256=payload_sha256,
                weights_sha256=separator.weights_sha256,
                device=configuration.device,
                threads=configuration.threads,
                load_seconds=separator.load_seconds,
                sample_rate=separator.sample_rate,
                audio_channels=separator.audio_channels,
                stems=separator.stems,
            )
        )
        sequence = 0
        while line := requests.readline(MAX_NATIVE_RECORD_BYTES + 1):
            request_id = None
            if len(line) > MAX_NATIVE_RECORD_BYTES or not line.endswith(b"\n"):
                raise ValueError(
                    "resident separation request framing exceeds its bound"
                )
            request = NativeSeparationServeRequest.model_validate_json(line)
            request_id = request.request_id
            if request.sequence != sequence + 1:
                raise ValueError("resident request sequence is not contiguous")
            sequence = request.sequence
            job = request.job
            if (job.weights_path, job.device, job.threads) != (
                configuration.weights_path,
                configuration.device,
                configuration.threads,
            ):
                raise ValueError("request cannot replace the resident execution policy")

            def progress(
                record: NativeSeparationProgress, *, native_id: str = request.request_id
            ) -> None:
                emit(
                    NativeSeparationResidentProgress(
                        request_id=native_id, progress=record
                    )
                )

            result = separator.run(job, progress)
            if result.output.source != job.source or result.device != job.device:
                raise ValueError("resident model returned a foreign source or device")
            destination = Path(job.output_dir) / "result.json"
            with destination.open("x") as completed:
                completed.write(result.model_dump_json())
            emit(
                NativeSeparationCompleted(request_id=request.request_id, result=result)
            )
    except Exception:
        # The owner gets a bounded terminal, while stderr retains the actual
        # diagnostic and the process exits. Continuing could hide a device fault.
        emit(NativeSeparationFailed(request_id=request_id))
        raise
