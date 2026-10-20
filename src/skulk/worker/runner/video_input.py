"""Bounded media preparation, also bundled in the isolated native engine.

This module imports no Skulk modules: the Comfy interpreter can execute the
owned worker file, and the native wheel carries the same immutable source.
PyAV is imported only inside preparation, never in the base worker environment.
"""

from __future__ import annotations

import argparse
import array
import importlib
import math
import os
import resource
import signal
import sys
import threading
import time
import wave
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Final, Literal, Protocol, final, overload, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class AudioPlane(Protocol):
    """Narrow AudioPlane read interface verified against the pinned PyAV runtime."""

    buffer_size: int
    def __bytes__(self) -> bytes:
        """Return this bounded decoder plane, including its alignment padding."""
        ...

class AudioFormat(Protocol):
    """Narrow AudioFormat read interface verified against the pinned PyAV runtime."""

    name: str
    bytes: int

class AudioLayout(Protocol):
    """Narrow AudioLayout read interface verified against the pinned PyAV runtime."""

    nb_channels: int

class AudioFrame(Protocol):
    """Narrow AudioFrame read interface verified against the pinned PyAV runtime."""

    samples: int
    sample_rate: int
    format: AudioFormat
    layout: AudioLayout
    planes: tuple[AudioPlane, ...]
    is_corrupt: bool

class VideoFrame(Protocol):
    """Narrow VideoFrame read interface verified against the pinned PyAV runtime."""

    width: int
    height: int
    pts: int | None
    time_base: Fraction | None
    is_corrupt: bool

class VideoCodecContext(Protocol):
    """Narrow VideoCodecContext read interface verified against the pinned PyAV runtime."""

    width: int
    height: int

class AudioCodecContext(Protocol):
    """Narrow AudioCodecContext read interface verified against the pinned PyAV runtime."""

    sample_rate: int
    channels: int

class VideoStream(Protocol):
    """Narrow VideoStream read interface verified against the pinned PyAV runtime."""

    codec_context: VideoCodecContext
    average_rate: Fraction | None
    frames: int

class AudioStream(Protocol):
    """Narrow AudioStream read interface verified against the pinned PyAV runtime."""

    codec_context: AudioCodecContext

class StreamContainer(Protocol):
    """Narrow StreamContainer read interface verified against the pinned PyAV runtime."""

    video: tuple[VideoStream, ...]
    audio: tuple[AudioStream, ...]
    def __len__(self) -> int:
        """Return all container streams, including unsupported extra tracks."""
        ...

class InputContainer(Protocol):
    """Narrow InputContainer read interface verified against the pinned PyAV runtime."""

    streams: StreamContainer
    def __enter__(self) -> InputContainer:
        """Open the input context for streaming decode."""
        ...
    def __exit__(self, exception_type: type[BaseException] | None, exception: BaseException | None, traceback: TracebackType | None) -> None:
        """Close decoder resources when preparation completes or fails."""
        ...
    @overload
    def decode(self, *, video: int) -> Iterator[VideoFrame]:
        """Yield decoded frames from the selected video track."""
        ...
    @overload
    def decode(self, *, audio: int) -> Iterator[AudioFrame]:
        """Yield decoded frames from the selected audio track."""
        ...

class AudioResampler(Protocol):
    """Narrow AudioResampler read interface verified against the pinned PyAV runtime."""

    def resample(self, frame: AudioFrame | None) -> list[AudioFrame]:
        """Convert one frame, or flush pending samples when frame is None."""
        ...

@runtime_checkable
class AudioDecoder(Protocol):
    """Pinned decoder surface loaded only in the isolated helper interpreter."""

    def open(self, file: str) -> InputContainer:
        """Open the worker-verified local file for read-only decode."""
        ...
    def AudioResampler(self, *, format: str, layout: str, rate: int) -> AudioResampler:  # noqa: A002, N802 -- exact pinned PyAV export and keyword
        """Construct the selected decoder's bounded streaming audio resampler."""
        ...

MAX_SOURCE_VIDEO_BYTES: Final = 2 * 1024**3
"""Maximum fp32 source-pixel volume later materialized by the native encoder."""
MAX_SOURCE_FRAME_BYTES: Final = 128 * 1024**2
MAX_AUDIO_FRAME_BYTES: Final = 16 * 1024**2
_OUTPUT_SAMPLE_RATE: Final = 48000


@final
class MediaInputRequest(BaseModel):
    """Worker-owned preparation command; no caller-supplied executable or URL."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    operation: Literal["audio", "video"] = Field(description="Normalize one audio segment or verify a source video.")
    source: str = Field(max_length=4096, description="Verified worker-staged local attachment.")
    destination: str | None = Field(default=None, max_length=4096, description="Worker-owned WAV destination for audio preparation.")
    start_seconds: float = Field(default=0.0, ge=0.0, le=120.0, description="Audio segment offset measured in decoded waveform samples.")
    duration_seconds: float = Field(gt=0.0, le=121.0, description="Maximum output duration, including trained-frame-grid alignment.")
    width: int = Field(default=128, ge=128, le=65535, description="Required source-video width.")
    height: int = Field(default=128, ge=128, le=65535, description="Required source-video height.")
    fps: int = Field(default=24, ge=1, le=1000, description="Required constant source-video frame rate.")
    maximum_frames: int = Field(default=1, ge=1, le=120001, description="Maximum source frame count before any encoder allocation.")
    exact_frames: bool = Field(default=False, description="Require the complete requested frame count for a retake.")
    require_audio: bool = Field(default=False, description="Preserving audible output requires an existing source track.")


@final
class VerifiedMediaInput(BaseModel):
    """Measured input facts carried privately to the exact selected runner."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    path: str = Field(max_length=4096, description="Verified source or normalized worker-owned WAV.")
    frames: int = Field(default=0, ge=0, description="Fully decoded video frames, zero for an audio segment.")
    has_audio: bool = Field(description="Whether a finite decoded audio track is present.")
    audio_sample_rate: int | None = Field(default=None, description="Verified source sample rate; normalized audio uses 48 kHz.")
    audio_channels: int | None = Field(default=None, description="Verified source channel count; normalized audio is stereo.")


def validate_audio_frame(frame: AudioFrame) -> None:
    """Bound one decoded frame and refuse corrupt or non-finite floating audio."""
    if frame.is_corrupt or not 8000 <= frame.sample_rate <= 192000 or not 1 <= frame.layout.nb_channels <= 8:
        raise ValueError("source audio has corrupt frames or unsupported sample rate/channels")
    if frame.samples < 1 or sum(plane.buffer_size for plane in frame.planes) > MAX_AUDIO_FRAME_BYTES:
        raise ValueError("source audio exceeds the decoded-frame budget")
    if frame.format.name.startswith(("flt", "dbl")):
        code = "f" if frame.format.name.startswith("flt") else "d"
        active_samples = frame.samples * (1 if frame.format.name.endswith("p") else frame.layout.nb_channels)
        for plane in frame.planes:
            # Decoder planes may include uninitialized alignment padding.
            # Only the declared waveform participates in finiteness checks.
            samples = array.array(code, bytes(plane)[:active_samples * frame.format.bytes])
            if any(not math.isfinite(sample) for sample in samples):
                raise ValueError("source audio contains non-finite samples")


def prepare_audio_input(request: MediaInputRequest) -> VerifiedMediaInput:
    """Write an exact bounded 48 kHz stereo PCM segment, refusing a short source."""
    decoder = importlib.import_module("av")
    if not isinstance(decoder, AudioDecoder):
        raise ValueError("selected engine lacks the pinned audio/video decoder interface")

    if request.destination is None:
        raise ValueError("audio preparation requires a worker-owned destination")
    destination = Path(request.destination)
    needed = round(request.duration_seconds * _OUTPUT_SAMPLE_RATE)
    start = round(request.start_seconds * _OUTPUT_SAMPLE_RATE)
    decoded = 0
    written = 0
    resampler = decoder.AudioResampler(format="s16", layout="stereo", rate=_OUTPUT_SAMPLE_RATE)
    with decoder.open(request.source) as container, wave.open(str(destination), "wb") as output:
        if len(container.streams.audio) != 1 or len(container.streams) != 1:
            raise ValueError("audio conditioning requires exactly one audio track and no video")
        codec = container.streams.audio[0].codec_context
        if not 8000 <= codec.sample_rate <= 192000 or not 1 <= codec.channels <= 8:
            raise ValueError("source audio has unsupported sample rate/channels")
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(_OUTPUT_SAMPLE_RATE)

        def write_segment(frame: AudioFrame) -> None:
            nonlocal decoded, written
            validate_audio_frame(frame)
            beginning = max(0, start - decoded)
            end = min(frame.samples, start + needed - decoded)
            if beginning < end:
                output.writeframesraw(bytes(frame.planes[0])[beginning * 4:end * 4])
                written += end - beginning
            decoded += frame.samples

        for frame in container.decode(audio=0):
            validate_audio_frame(frame)
            for converted in resampler.resample(frame):
                write_segment(converted)
            if written >= needed:
                break
        if written < needed:
            for converted in resampler.resample(None):
                write_segment(converted)
    if written != needed:
        destination.unlink(missing_ok=True)
        raise ValueError("source audio does not cover the requested offset and complete output duration")
    return VerifiedMediaInput(path=str(destination), has_audio=True, audio_sample_rate=_OUTPUT_SAMPLE_RATE, audio_channels=2)


def verify_video_input(request: MediaInputRequest) -> VerifiedMediaInput:
    """Decode a bounded constant-rate clip and verify every source frame and track."""
    decoder = importlib.import_module("av")
    if not isinstance(decoder, AudioDecoder):
        raise ValueError("selected engine lacks the pinned audio/video decoder interface")

    frames = 0
    initial_time: Fraction | None = None
    with decoder.open(request.source) as container:
        if len(container.streams.video) != 1 or len(container.streams.audio) > 1 or len(container.streams) != 1 + len(container.streams.audio):
            raise ValueError("editing requires one video track and at most one audio track")
        stream = container.streams.video[0]
        if stream.codec_context.width != request.width or stream.codec_context.height != request.height or stream.average_rate != request.fps:
            raise ValueError("source video must match the requested size and frame rate")
        pixels = request.width * request.height * 12
        if pixels > MAX_SOURCE_FRAME_BYTES or (stream.frames > 0 and pixels * stream.frames > MAX_SOURCE_VIDEO_BYTES):
            raise ValueError("source video exceeds the bounded decoded-pixel budget")
        audio_stream = next(iter(container.streams.audio), None)
        has_audio = audio_stream is not None
        if request.require_audio and not has_audio:
            raise ValueError("preserving audible output requires a source audio track")
        sample_rate = audio_stream.codec_context.sample_rate if audio_stream is not None else None
        channels = audio_stream.codec_context.channels if audio_stream is not None else None
        for frame in container.decode(video=0):
            if frame.is_corrupt or frame.width != request.width or frame.height != request.height or frame.pts is None or frame.time_base is None:
                raise ValueError("source video contains corrupt, resized or untimed frames")
            timestamp = frame.pts * frame.time_base
            if initial_time is None:
                initial_time = timestamp
            if abs(timestamp - initial_time - Fraction(frames, request.fps)) > Fraction(1, 1000):
                raise ValueError("editing requires a constant-rate source timeline")
            frames += 1
            if frames * pixels > MAX_SOURCE_VIDEO_BYTES:
                raise ValueError("source video exceeds the bounded decoded-pixel budget")
            if frames > request.maximum_frames:
                raise ValueError("source video exceeds the requested output duration")
    if frames < 1 or frames % 8 != 1 or (request.exact_frames and frames != request.maximum_frames):
        raise ValueError("source video must fit the requested duration and trained 8n+1 frame grid")
    if has_audio:
        decoded_audio = 0
        with decoder.open(request.source) as container:
            for frame in container.decode(audio=0):
                validate_audio_frame(frame)
                if frame.sample_rate != sample_rate or frame.layout.nb_channels != channels:
                    raise ValueError("source audio changes its sample rate or channel count")
                decoded_audio += frame.samples
                if sample_rate is None or decoded_audio > math.ceil((frames / request.fps + 0.25) * sample_rate):
                    raise ValueError("source audio exceeds the bounded source duration")
        if sample_rate is None or decoded_audio / sample_rate < frames / request.fps - 0.05:
            raise ValueError("source audio does not cover the source video timeline")
    return VerifiedMediaInput(path=request.source, frames=frames, has_audio=has_audio, audio_sample_rate=sample_rate, audio_channels=channels)


class Arguments(argparse.Namespace):
    """Private preparation command paths and supervising worker identity."""

    request: Path
    result: Path
    parent_pid: int


def _resident_memory_bytes() -> int:
    """Read decoder RSS without Linux's inherited pre-exec high-water mark."""
    if sys.platform == "linux":
        # Linux carries the spawning runner's RSS peak through exec. The
        # decoder must be charged for its own pages, not model/worker memory
        # that belonged to the previous process image.
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def main() -> None:
    """Prepare one bounded local attachment with heap/file limits and parent reaping."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    arguments = Arguments()
    parser.parse_args(namespace=arguments)
    if arguments.request.stat().st_size > 16 * 1024:
        raise ValueError("media preparation request exceeds its bounded protocol")
    request = MediaInputRequest.model_validate_json(arguments.request.read_bytes())
    # Darwin refuses lowering DATA/AS/RSS for these arm64 interpreters. Linux
    # enforces DATA; on both platforms the watchdog bounds resident decoder
    # memory independently of virtual address reservations and kills only this
    # separately supervised process group. Neither path loads model weights.
    if sys.platform != "darwin":
        resource.setrlimit(resource.RLIMIT_DATA, (512 * 1024**2, 512 * 1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))

    def reap() -> None:
        while os.getppid() == arguments.parent_pid:
            try:
                resident_bytes = _resident_memory_bytes()
            except (OSError, ValueError, IndexError):
                print("ValueError: source decoder could not measure its resident memory", file=sys.stderr, flush=True)
                break
            if resident_bytes > 512 * 1024**2:
                print("ValueError: source decoder exceeded its resident-memory budget", file=sys.stderr, flush=True)
                break
            time.sleep(0.05)
        os.killpg(os.getpgrp(), signal.SIGKILL)

    threading.Thread(target=reap, daemon=True).start()
    result = prepare_audio_input(request) if request.operation == "audio" else verify_video_input(request)
    arguments.result.write_text(result.model_dump_json())


if __name__ == "__main__":
    main()
