"""Resident control admission tests without importing model/native libraries."""

import io
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pytest
from pydantic import TypeAdapter
from skulk_demucs.contracts import (
    SEPARATION_STEMS,
    AudioSeparationOutput,
    NativeSeparationCompleted,
    NativeSeparationFailed,
    NativeSeparationFrame,
    NativeSeparationJob,
    NativeSeparationProgress,
    NativeSeparationReady,
    NativeSeparationResult,
    NativeSeparationServeConfig,
    NativeSeparationServeRequest,
    SeparatedStem,
    SeparationSource,
    SeparationStemName,
)
from skulk_demucs.server import MAX_NATIVE_RECORD_BYTES, serve


@dataclass(frozen=True)
class LoadedFixture:
    """Strict loaded-model boundary; no codec, tensor or model execution is claimed."""

    configuration: NativeSeparationServeConfig
    calls: list[NativeSeparationJob] = field(default_factory=list)
    weights_sha256: str = "a" * 64
    load_seconds: float = 0.25
    sample_rate: Literal[44100] = 44100
    audio_channels: Literal[2] = 2
    stems: tuple[SeparationStemName, ...] = SEPARATION_STEMS
    foreign: bool = False

    def run(
        self,
        job: NativeSeparationJob,
        progress: Callable[[NativeSeparationProgress], None],
    ) -> NativeSeparationResult:
        """Return bounded facts to exercise correlation, sequence and publication."""
        self.calls.append(job)
        progress(NativeSeparationProgress(stage="separating", completed_frames=44100))
        Path(job.output_dir).mkdir(parents=True, exist_ok=True)
        source = (
            job.source.model_copy(update={"sha256": "f" * 64})
            if self.foreign
            else job.source
        )
        return NativeSeparationResult(
            output=AudioSeparationOutput(
                source=source,
                stems=tuple(
                    SeparatedStem(
                        name=name,
                        size_bytes=264644,
                        sha256="b" * 64,
                        sample_rate=44100,
                        frames=44100,
                        peak=0.25,
                    )
                    for name in SEPARATION_STEMS
                ),
                common_gain=1.0,
                pre_gain_peak=0.25,
            ),
            device=job.device,
            load_seconds=0.0,
            separation_seconds=0.1,
            total_seconds=0.2,
            process_peak_rss_bytes=1024,
        )


def configuration() -> NativeSeparationServeConfig:
    """Create fixed private model policy shared by the test's requests."""
    return NativeSeparationServeConfig(
        weights_path="/owned/model.safetensors", device="cpu", threads=2
    )


def request(tmp_path: Path, sequence: int) -> NativeSeparationServeRequest:
    """Make one bounded command-private request, without model or audio fixtures."""
    return NativeSeparationServeRequest(
        request_id=f"{sequence:032x}",
        sequence=sequence,
        job=NativeSeparationJob(
            weights_path=configuration().weights_path,
            source_path="/owned/original.wav",
            output_dir=str(tmp_path / f"job-{sequence}"),
            device="cpu",
            threads=2,
            source=SeparationSource(
                sha256="c" * 64,
                size_bytes=16044,
                frames=8000,
                sample_rate=8000,
                channels=1,
                encoding="pcm",
                bit_depth=16,
            ),
        ),
    )


def wire(*requests: NativeSeparationServeRequest) -> io.BytesIO:
    """Encode actual bounded newline framing used by the private CLI."""
    return io.BytesIO(
        b"".join(item.model_dump_json().encode() + b"\n" for item in requests)
    )


def test_two_requests_load_once_and_correlate_complete_closed_files(
    tmp_path: Path,
) -> None:
    """A ready model is reused; each result exists before its terminal is emitted."""
    loaded = LoadedFixture(configuration())
    constructions: list[NativeSeparationServeConfig] = []
    frames: list[NativeSeparationFrame] = []
    requests = (request(tmp_path, 1), request(tmp_path, 2))

    def load(value: NativeSeparationServeConfig) -> LoadedFixture:
        constructions.append(value)
        return loaded

    def emitted(record: NativeSeparationFrame) -> None:
        if isinstance(record, NativeSeparationCompleted):
            expected = next(
                item for item in requests if item.request_id == record.request_id
            )
            saved = NativeSeparationResult.model_validate_json(
                (Path(expected.job.output_dir) / "result.json").read_bytes()
            )
            assert saved == record.result
        frames.append(record)

    serve(configuration(), wire(*requests), emitted, payload_sha256="d" * 64, load=load)
    assert constructions == [configuration()]
    assert loaded.calls == [item.job for item in requests]
    assert sum(isinstance(frame, NativeSeparationReady) for frame in frames) == 1
    assert [
        frame.request_id
        for frame in frames
        if isinstance(frame, NativeSeparationCompleted)
    ] == [item.request_id for item in requests]
    parsed = TypeAdapter[NativeSeparationFrame](NativeSeparationFrame)
    for frame in frames:
        assert parsed.validate_json(frame.model_dump_json()) == frame
        assert len(frame.model_dump_json().encode()) + 1 <= MAX_NATIVE_RECORD_BYTES


def test_sequence_replay_ends_lifetime_without_executing_again(tmp_path: Path) -> None:
    """Transport retry cannot silently replay an already completed model job."""
    loaded = LoadedFixture(configuration())
    first = request(tmp_path, 1)
    repeated = request(tmp_path, 2).model_copy(update={"sequence": 1})
    frames: list[NativeSeparationFrame] = []
    with pytest.raises(ValueError, match="sequence"):
        serve(
            configuration(),
            wire(first, repeated, request(tmp_path, 3)),
            frames.append,
            payload_sha256="d" * 64,
            load=lambda _: loaded,
        )
    assert loaded.calls == [first.job]
    assert isinstance(frames[-1], NativeSeparationFailed)
    assert not (tmp_path / "job-2").exists()
    assert not (tmp_path / "job-3").exists()


@pytest.mark.parametrize("change", ["weights", "device", "threads"])
def test_job_cannot_replace_loaded_policy(tmp_path: Path, change: str) -> None:
    """A different model/device/thread intent is refused before native execution."""
    loaded = LoadedFixture(configuration())
    original = request(tmp_path, 1)
    updates: dict[str, object] = {"weights_path": "/owned/another.safetensors"}
    if change == "device":
        updates = {"device": "mps"}
    elif change == "threads":
        updates = {"threads": 4}
    changed = original.model_copy(
        update={"job": original.job.model_copy(update=updates)}
    )
    frames: list[NativeSeparationFrame] = []
    with pytest.raises(ValueError, match="policy"):
        serve(
            configuration(),
            wire(changed),
            frames.append,
            payload_sha256="d" * 64,
            load=lambda _: loaded,
        )
    assert loaded.calls == []
    assert isinstance(frames[-1], NativeSeparationFailed)
    assert not (tmp_path / "job-1").exists()


@pytest.mark.parametrize(
    "data",
    [
        b"x" * (MAX_NATIVE_RECORD_BYTES + 1) + b"\n",
        b"{}",
        b"not-json\n",
        b'{"options":{"backend":"arbitrary"}}\n',
    ],
)
def test_invalid_input_fails_boundedly_without_running_a_job(data: bytes) -> None:
    """Over-limit, incomplete and malformed private requests terminate the lifetime."""
    loaded = LoadedFixture(configuration())
    frames: list[NativeSeparationFrame] = []
    with pytest.raises(ValueError):
        serve(
            configuration(),
            io.BytesIO(data),
            frames.append,
            payload_sha256="d" * 64,
            load=lambda _: loaded,
        )
    assert loaded.calls == []
    assert isinstance(frames[-1], NativeSeparationFailed)
    assert frames[-1].request_id is None


def test_foreign_native_completion_cannot_publish_result_file(tmp_path: Path) -> None:
    """A native result must still bind the admitted original before publication."""
    loaded = LoadedFixture(configuration(), foreign=True)
    frames: list[NativeSeparationFrame] = []
    with pytest.raises(ValueError, match="foreign"):
        serve(
            configuration(),
            wire(request(tmp_path, 1)),
            frames.append,
            payload_sha256="d" * 64,
            load=lambda _: loaded,
        )
    assert isinstance(frames[-1], NativeSeparationFailed)
    assert not (tmp_path / "job-1" / "result.json").exists()


def test_load_failure_never_reports_ready() -> None:
    """Failed construction/placement is an unready stopped lifetime."""
    frames: list[NativeSeparationFrame] = []

    def unavailable(_: NativeSeparationServeConfig) -> LoadedFixture:
        raise ValueError("unavailable test device")

    with pytest.raises(ValueError, match="unavailable"):
        serve(
            configuration(),
            io.BytesIO(),
            frames.append,
            payload_sha256="d" * 64,
            load=unavailable,
        )
    assert len(frames) == 1 and isinstance(frames[0], NativeSeparationFailed)


def test_end_of_input_stops_cleanly_without_unrequested_execution() -> None:
    """Closing the owned control pipe after startup performs no synthetic job."""
    loaded = LoadedFixture(configuration())
    frames: list[NativeSeparationFrame] = []
    serve(
        configuration(),
        io.BytesIO(),
        frames.append,
        payload_sha256="d" * 64,
        load=lambda _: loaded,
    )
    assert len(frames) == 1 and isinstance(frames[0], NativeSeparationReady)
    assert not loaded.calls
