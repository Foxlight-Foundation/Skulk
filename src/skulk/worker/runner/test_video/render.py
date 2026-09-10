"""Synthetic clip rendering and a minimal MP4 muxer for the test video engine.

Everything here is deterministic for a given plan: the same seed, canvas,
and duration produce byte-identical output on the same Pillow build. The
container is a plain ISO base media file with an MJPEG video track and,
when the card and request ask for audio, an uncompressed 16-bit PCM stereo
track. Nothing in the substrate parses the container; it is muxed properly
anyway so operators can play what the test engine delivers.
"""

from __future__ import annotations

import array
import hashlib
import io
import math
import random
import struct
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from skulk.shared.models.model_cards import VideoCardConfig
from skulk.shared.types.video import (
    VIDEO_OUTPUT_FILENAME,
    VIDEO_THUMBNAIL_FILENAME,
    VideoGenerationStats,
    VideoGenerationTaskParams,
    VideoOutputManifest,
    VideoStage,
)

DEFAULT_SHORT_EDGE = 64
"""Canvas short edge when neither the request nor the card names one."""
DEFAULT_SAMPLE_RATE = 32000
DEFAULT_CHANNELS = 2
MAX_CONTAINER_EDGE = 65535
"""MP4 sample entries carry 16-bit width and height fields."""

ProgressCallback = Callable[[VideoStage, int | None, int | None, float], None]
"""Called with ``(stage, step, total_steps, fraction)`` as the render advances."""


@dataclass(frozen=True, slots=True)
class RenderPlan:
    """Everything the renderer needs, resolved from the request and the card."""

    width: int
    height: int
    fps: int
    frame_count: int
    steps: int
    seed: int
    audio: bool
    sample_rate: int
    channels: int

    @property
    def seconds(self) -> float:
        """Clip duration as muxed."""
        return self.frame_count / self.fps


def _parse_ratio(value: str | None) -> tuple[int, int]:
    if value is None:
        return (1, 1)
    left, _, right = value.partition(":")
    return (int(left), int(right))


def plan_render(params: VideoGenerationTaskParams, video: VideoCardConfig) -> RenderPlan:
    """Resolve the request against the card the way a real engine would.

    The canvas is the explicit ``size`` or the card's short edge scaled by
    the advisory aspect ratio and snapped to the canvas grid; the frame count
    is the card's aligned count for the requested duration; steps fall back
    to the card default; the seed is the request's or a digest of the prompt
    so an unseeded request is still reproducible.
    """

    canvas = params.width_height
    if canvas is None:
        short = video.default_short_edge or DEFAULT_SHORT_EDGE
        width_ratio, height_ratio = _parse_ratio(params.aspect_ratio)
        if width_ratio >= height_ratio:
            width, height = round(short * width_ratio / height_ratio), short
        else:
            width, height = short, round(short * height_ratio / width_ratio)
        multiple = max(1, video.canvas_multiple)
        canvas = (
            max(multiple, round(width / multiple) * multiple),
            max(multiple, round(height / multiple) * multiple),
        )
    if max(canvas) > MAX_CONTAINER_EDGE:
        raise ValueError(
            f"canvas {canvas[0]}x{canvas[1]} exceeds the container's "
            f"{MAX_CONTAINER_EDGE} pixel edge limit"
        )
    seed = params.seed
    if seed is None:
        seed = int.from_bytes(hashlib.sha256(params.prompt.encode()).digest()[:4], "big")
    audio = params.audio and video.audio_output
    return RenderPlan(
        width=canvas[0],
        height=canvas[1],
        fps=video.fps,
        frame_count=video.frame_count_for_seconds(params.seconds),
        steps=params.steps or video.default_steps,
        seed=seed,
        audio=audio,
        sample_rate=video.audio_sample_rate or DEFAULT_SAMPLE_RATE,
        channels=video.audio_channels or DEFAULT_CHANNELS,
    )


def _frame_jpeg(plan: RenderPlan, index: int) -> bytes:
    """One frame: a seeded gradient with a square sweeping across it."""

    rng = random.Random(plan.seed)
    base = tuple(rng.randrange(32, 224) for _ in range(3))
    accent = tuple(255 - channel for channel in base)
    image = Image.new("RGB", (plan.width, plan.height))
    draw = ImageDraw.Draw(image)
    for row in range(plan.height):
        shade = row / max(1, plan.height - 1)
        color = tuple(int(component * (0.55 + 0.45 * shade)) for component in base)
        draw.line([(0, row), (plan.width, row)], fill=color)
    side = max(4, min(plan.width, plan.height) // 4)
    travel = max(1, plan.frame_count - 1)
    left = round((plan.width - side) * index / travel)
    top = round((plan.height - side) * (0.5 + 0.5 * math.sin(index / travel * math.pi)))
    draw.rectangle([left, top, left + side, top + side], fill=accent)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return buffer.getvalue()


def _audio_pcm(plan: RenderPlan) -> bytes:
    """Interleaved little-endian 16-bit PCM: one seeded tone, slowly panned."""

    total = round(plan.seconds * plan.sample_rate)
    frequency = 220 + (plan.seed % 660)
    samples = array.array("h")
    for index in range(total):
        phase = 2 * math.pi * frequency * index / plan.sample_rate
        pan = 0.5 + 0.5 * math.sin(2 * math.pi * index / max(1, total))
        value = math.sin(phase) * 0.25 * 32767
        for channel in range(plan.channels):
            weight = pan if channel % 2 == 0 else 1 - pan
            samples.append(int(value * weight))
    if struct.pack("=h", 1) != struct.pack("<h", 1):
        samples.byteswap()
    return samples.tobytes()


# --- ISO base media boxes -----------------------------------------------------

_MATRIX = struct.pack(">9I", 0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000)
_COMPRESSOR = b"Skulk test video"
_LANGUAGE_UNDETERMINED = 0x55C4


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def _full_box(kind: bytes, version: int, flags: int, payload: bytes) -> bytes:
    return _box(kind, struct.pack(">B", version) + flags.to_bytes(3, "big") + payload)


def _tkhd(track_id: int, duration: int, width: int, height: int, *, audio: bool) -> bytes:
    payload = (
        struct.pack(">IIII", 0, 0, track_id, 0)
        + struct.pack(">I", duration)
        + struct.pack(">II", 0, 0)
        + struct.pack(">hhHH", 0, 0, 0x0100 if audio else 0, 0)
        + _MATRIX
        + struct.pack(">II", width << 16, height << 16)
    )
    return _full_box(b"tkhd", 0, 0x000003, payload)


def _mdhd(timescale: int, duration: int) -> bytes:
    return _full_box(
        b"mdhd", 0, 0, struct.pack(">IIIIHH", 0, 0, timescale, duration, _LANGUAGE_UNDETERMINED, 0)
    )


def _hdlr(handler: bytes, name: bytes) -> bytes:
    return _full_box(
        b"hdlr", 0, 0, struct.pack(">I", 0) + handler + struct.pack(">III", 0, 0, 0) + name + b"\0"
    )


def _dinf() -> bytes:
    url = _full_box(b"url ", 0, 1, b"")
    return _box(b"dinf", _full_box(b"dref", 0, 0, struct.pack(">I", 1) + url))


def _stts(sample_count: int) -> bytes:
    return _full_box(b"stts", 0, 0, struct.pack(">III", 1, sample_count, 1))


def _stsc(samples_per_chunk: int) -> bytes:
    return _full_box(b"stsc", 0, 0, struct.pack(">IIII", 1, 1, samples_per_chunk, 1))


def _stsz(sizes: Sequence[int] | None, sample_count: int, fixed_size: int = 0) -> bytes:
    if sizes is None:
        return _full_box(b"stsz", 0, 0, struct.pack(">II", fixed_size, sample_count))
    return _full_box(
        b"stsz",
        0,
        0,
        struct.pack(">II", 0, len(sizes)) + b"".join(struct.pack(">I", size) for size in sizes),
    )


def _stco(offset: int) -> bytes:
    return _full_box(b"stco", 0, 0, struct.pack(">II", 1, offset))


def _jpeg_sample_entry(width: int, height: int) -> bytes:
    name = struct.pack(">B", len(_COMPRESSOR)) + _COMPRESSOR
    payload = (
        b"\0" * 6
        + struct.pack(">H", 1)
        + struct.pack(">HH", 0, 0)
        + struct.pack(">III", 0, 0, 0)
        + struct.pack(">HH", width, height)
        + struct.pack(">II", 0x00480000, 0x00480000)
        + struct.pack(">I", 0)
        + struct.pack(">H", 1)
        + name.ljust(32, b"\0")
        + struct.pack(">Hh", 0x0018, -1)
    )
    return _box(b"jpeg", payload)


def _pcm_sample_entry(channels: int, sample_rate: int) -> bytes:
    payload = (
        b"\0" * 6
        + struct.pack(">H", 1)
        + struct.pack(">HHI", 0, 0, 0)
        + struct.pack(">HHHHI", channels, 16, 0, 0, sample_rate << 16)
    )
    return _box(b"sowt", payload)


def _video_trak(
    frames: Sequence[bytes], width: int, height: int, fps: int, offset: int, movie_duration: int
) -> bytes:
    stbl = _box(
        b"stbl",
        _full_box(b"stsd", 0, 0, struct.pack(">I", 1) + _jpeg_sample_entry(width, height))
        + _stts(len(frames))
        + _stsc(len(frames))
        + _stsz([len(frame) for frame in frames], len(frames))
        + _stco(offset),
    )
    minf = _box(
        b"minf", _full_box(b"vmhd", 0, 1, struct.pack(">HHHH", 0, 0, 0, 0)) + _dinf() + stbl
    )
    mdia = _box(b"mdia", _mdhd(fps, len(frames)) + _hdlr(b"vide", b"VideoHandler") + minf)
    return _box(b"trak", _tkhd(1, movie_duration, width, height, audio=False) + mdia)


def _audio_trak(
    sample_count: int, channels: int, sample_rate: int, offset: int, movie_duration: int
) -> bytes:
    stbl = _box(
        b"stbl",
        _full_box(b"stsd", 0, 0, struct.pack(">I", 1) + _pcm_sample_entry(channels, sample_rate))
        + _stts(sample_count)
        + _stsc(sample_count)
        + _stsz(None, sample_count, channels * 2)
        + _stco(offset),
    )
    minf = _box(b"minf", _full_box(b"smhd", 0, 0, struct.pack(">HH", 0, 0)) + _dinf() + stbl)
    mdia = _box(
        b"mdia", _mdhd(sample_rate, sample_count) + _hdlr(b"soun", b"SoundHandler") + minf
    )
    return _box(b"trak", _tkhd(2, movie_duration, 0, 0, audio=True) + mdia)


def mux_mp4(
    frames: Sequence[bytes],
    *,
    width: int,
    height: int,
    fps: int,
    audio: bytes | None,
    sample_rate: int,
    channels: int,
) -> bytes:
    """Mux JPEG frames and optional PCM audio into a minimal MP4.

    Args:
        frames: One complete JPEG file per video frame, in display order;
            every frame is shown for exactly one tick of ``fps``.
        width: Frame width in pixels, declared in the sample entry.
        height: Frame height in pixels, declared in the sample entry.
        fps: Frames per second; also the video track's timescale.
        audio: Interleaved little-endian 16-bit PCM, or ``None`` for a
            silent file with a single video track.
        sample_rate: Audio sample rate in hertz.
        channels: Audio channel count; each PCM sample is ``channels * 2``
            bytes.

    Returns:
        The complete container bytes: ``ftyp``, one ``mdat`` holding the
        frames followed by the audio, then ``moov`` with an MJPEG track and,
        when audio is given, an uncompressed PCM (``sowt``) track. Pure
        function with no side effects; the caller writes the bytes.
    """

    ftyp = _box(b"ftyp", b"isom" + struct.pack(">I", 0x200) + b"isom" + b"iso2" + b"mp41")
    video_data = b"".join(frames)
    audio_data = audio or b""
    mdat = _box(b"mdat", video_data + audio_data)
    video_offset = len(ftyp) + 8
    audio_offset = video_offset + len(video_data)
    movie_timescale = 1000
    movie_duration = round(len(frames) / fps * movie_timescale)
    tracks = _video_trak(frames, width, height, fps, video_offset, movie_duration)
    next_track = 2
    if audio_data:
        sample_count = len(audio_data) // (channels * 2)
        tracks += _audio_trak(sample_count, channels, sample_rate, audio_offset, movie_duration)
        next_track = 3
    mvhd = _full_box(
        b"mvhd",
        0,
        0,
        struct.pack(">IIII", 0, 0, movie_timescale, movie_duration)
        + struct.pack(">IH", 0x00010000, 0x0100)
        + b"\0" * 10
        + _MATRIX
        + b"\0" * 24
        + struct.pack(">I", next_track),
    )
    return ftyp + mdat + _box(b"moov", mvhd + tracks)


def render_clip(
    plan: RenderPlan,
    output_dir: Path,
    *,
    progress: ProgressCallback,
    is_cancelled: Callable[[], bool],
    step_seconds: float,
) -> tuple[VideoOutputManifest, VideoGenerationStats] | None:
    """Render the plan into ``output_dir`` and describe the result.

    Sampling is simulated as ``plan.steps`` sleeps of ``step_seconds`` so a
    render takes observable time and can be cancelled between steps. Returns
    ``None`` when cancelled, in which case nothing is written.
    """

    started = time.monotonic()
    progress("encoding", None, None, 0.05)
    sampling_started = time.monotonic()
    for step in range(1, plan.steps + 1):
        if is_cancelled():
            return None
        time.sleep(step_seconds)
        progress("sampling", step, plan.steps, 0.05 + 0.75 * step / plan.steps)
    sampling_seconds = time.monotonic() - sampling_started
    progress("decoding", None, None, 0.85)
    frames = [_frame_jpeg(plan, index) for index in range(plan.frame_count)]
    if is_cancelled():
        return None
    progress("muxing", None, None, 0.95)
    audio = _audio_pcm(plan) if plan.audio else None
    container = mux_mp4(
        frames,
        width=plan.width,
        height=plan.height,
        fps=plan.fps,
        audio=audio,
        sample_rate=plan.sample_rate,
        channels=plan.channels,
    )
    thumbnail = frames[len(frames) // 2]
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in ((VIDEO_OUTPUT_FILENAME, container), (VIDEO_THUMBNAIL_FILENAME, thumbnail)):
        staging = output_dir / f"{name}.part"
        staging.write_bytes(payload)
        staging.replace(output_dir / name)
    manifest = VideoOutputManifest(
        sha256=hashlib.sha256(container).hexdigest(),
        size_bytes=len(container),
        width=plan.width,
        height=plan.height,
        frame_count=plan.frame_count,
        fps=plan.fps,
        seconds=plan.seconds,
        audio_sample_rate=plan.sample_rate if audio is not None else None,
        audio_channels=plan.channels if audio is not None else None,
        thumbnail_sha256=hashlib.sha256(thumbnail).hexdigest(),
        thumbnail_size_bytes=len(thumbnail),
    )
    stats = VideoGenerationStats(
        steps=plan.steps,
        seconds_per_step=sampling_seconds / max(1, plan.steps),
        total_generation_time=time.monotonic() - started,
    )
    return manifest, stats
