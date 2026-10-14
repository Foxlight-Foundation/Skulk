"""Read bounded container metadata instead of inventing dynamic render facts."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from skulk.worker.runner.video_plan import RenderPlan

_PYAV_PROBE: Final = """
import json,sys,av
with av.open(sys.argv[1]) as container:
    streams = []
    for stream in container.streams:
        codec = stream.codec_context
        record = {"codec_type": stream.type, "codec_name": codec.codec.canonical_name}
        if stream.type == "video":
            record.update(width=codec.width, height=codec.height, nb_frames=str(stream.frames),
                          r_frame_rate=str(stream.average_rate) if stream.average_rate else None)
        elif stream.type == "audio":
            record.update(sample_rate=str(codec.sample_rate), channels=codec.channels)
        streams.append(record)
    print(json.dumps({"streams": streams, "format": {"format_name": container.format.name}}))
"""


class _Stream(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    codec_type: str
    codec_name: str
    width: int | None = None
    height: int | None = None
    nb_frames: str | None = None
    r_frame_rate: str | None = None
    sample_rate: str | None = None
    channels: int | None = None


class _Format(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    format_name: str


class _Probe(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    streams: list[_Stream] = Field(max_length=8)
    format: _Format


@dataclass(frozen=True, slots=True)
class ProbedVideo:
    """Verified actual frame count and duration for an engine-owned MP4."""

    frame_count: int
    seconds: float


def probe_video_output(
    path: Path,
    plan: RenderPlan,
    *,
    codec: str,
    interpreter: Path | None = None,
    minimum_frames: int | None = None,
    frame_multiple: int = 8,
    frame_offset: int = 1,
) -> ProbedVideo:
    """Verify geometry, codecs, audio and bounded duration in an isolated engine.

    Args:
        path: Engine-owned completed container, never a caller-supplied URL.
        plan: Maximum allocation and expected output geometry/audio contract.
        codec: Selected output codec.
        interpreter: Engine interpreter with PyAV; omitted uses local ffprobe.
        minimum_frames: Lower predicted bound; omitted requires the exact plan.
        frame_multiple: Trained output frame grid.
        frame_offset: Trained grid offset.

    Returns:
        Actual output frame count and duration, suitable for a manifest.

    Raises:
        ValueError: Metadata is missing or disagrees with the selected contract.
    """
    command = [str(interpreter), "-I", "-c", _PYAV_PROBE, str(path)] if interpreter is not None else [
        "ffprobe", "-v", "error", "-show_entries",
        "format=format_name:stream=codec_type,codec_name,width,height,nb_frames,r_frame_rate,sample_rate,channels",
        "-of", "json", str(path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True, timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("Cannot verify video output; the selected engine needs a working media probe") from error
    if len(result.stdout) > 16 * 1024:
        raise ValueError("Video probe exceeded its bounded metadata response")
    metadata = _Probe.model_validate_json(result.stdout)
    if "mp4" not in metadata.format.format_name.split(","):
        raise ValueError("Video output is not an MP4 container")
    streams = metadata.streams
    videos = [stream for stream in streams if stream.codec_type == "video"]
    audios = [stream for stream in streams if stream.codec_type == "audio"]
    if len(videos) != 1 or len(audios) != int(plan.audio) or len(streams) != len(videos) + len(audios):
        raise ValueError("Video output contains unexpected tracks")
    video = videos[0]
    if video.width != plan.width or video.height != plan.height or video.codec_name != codec:
        raise ValueError("Video output disagrees with selected geometry or codec")
    if video.nb_frames is None or not video.nb_frames.isdecimal() or video.r_frame_rate is None:
        raise ValueError("Video output does not declare a verifiable frame count/rate")
    frames = int(video.nb_frames)
    try:
        rate = Fraction(video.r_frame_rate)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("Video output declares an invalid frame rate") from error
    if rate != plan.fps:
        raise ValueError("Video output disagrees with selected frame rate")
    minimum = plan.frame_count if minimum_frames is None else minimum_frames
    if not minimum <= frames <= plan.frame_count or frames % frame_multiple != frame_offset:
        raise ValueError("Video output exceeds duration bounds or trained frame grid")
    if audios and (audios[0].codec_name != "aac" or audios[0].sample_rate != str(plan.sample_rate) or audios[0].channels != plan.channels):
        raise ValueError("Video output disagrees with selected audio contract")
    return ProbedVideo(frame_count=frames, seconds=frames / plan.fps)
