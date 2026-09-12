"""Resolve one audio-video request against its card the way every engine must.

The request carries what the caller asked for; the card carries what the
model was trained on. The plan is the meeting point: an explicit canvas or
the card's trained short edge scaled by the advisory aspect ratio and snapped
to the canvas grid, the frame count aligned onto the card's grid for the
requested duration, steps from the request or the card default, and a seed
that is reproducible even when the caller gave none. Engines add their own
mechanism-specific choices on top (adapters, graph shape) but never re-derive
these facts differently from one another.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from skulk.shared.models.model_cards import VideoCardConfig
from skulk.shared.types.video import VideoGenerationTaskParams, VideoStage
from skulk.worker.runner.image_dimensions import image_dimensions

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


def keyframe_shape(params: VideoGenerationTaskParams) -> tuple[int, int] | None:
    """The pixel shape of the keyframe that anchors the clip, when one is attached.

    The first frame decides; the last frame stands in when there is no
    first. Only an image whose verified bytes the worker placed on disk
    (``local_path``) can be read; a plain ``reference`` never sets the
    canvas, since it conditions content, not framing.
    """
    for role in ("first_frame", "last_frame"):
        for reference in params.references:
            if reference.role != role or reference.kind != "image":
                continue
            if reference.local_path is None:
                return None
            return image_dimensions(Path(reference.local_path))
    return None


def _fit_pixel_budget(
    canvas: tuple[int, int], max_pixels: int, multiple: int
) -> tuple[int, int]:
    width, height = canvas
    while width * height > max_pixels:
        if width >= height:
            if width - multiple < multiple:
                break
            width -= multiple
        else:
            if height - multiple < multiple:
                break
            height -= multiple
    return (width, height)


def plan_render(
    params: VideoGenerationTaskParams, video: VideoCardConfig
) -> RenderPlan:
    """Resolve the request against the card the way a real engine would.

    The canvas is the explicit ``size``, or the card's short edge scaled by
    the advisory aspect ratio (or, with neither, by the shape of the first
    or last frame attached, so a keyframe keeps its framing) and snapped to
    the canvas grid; the frame count
    is the card's aligned count for the requested duration; steps fall back
    to the card default; the seed is the request's or a digest of the prompt
    so an unseeded request is still reproducible.
    """

    canvas = params.width_height
    if canvas is None:
        short = video.default_short_edge or DEFAULT_SHORT_EDGE
        shape = keyframe_shape(params) if params.aspect_ratio is None else None
        width_ratio, height_ratio = shape or _parse_ratio(params.aspect_ratio)
        if width_ratio >= height_ratio:
            width, height = round(short * width_ratio / height_ratio), short
        else:
            width, height = short, round(short * height_ratio / width_ratio)
        multiple = max(1, video.canvas_multiple)
        if video.max_pixels is not None and width * height > video.max_pixels:
            # The trained short edge at a wide ratio exceeds the trained pixel
            # budget; scale both edges so the ratio survives (21:9 at 768p
            # becomes 1536x672, not a 16:9 canvas with the height kept).
            scale = math.sqrt(video.max_pixels / (width * height))
            width, height = width * scale, height * scale
        canvas = (
            max(multiple, round(width / multiple) * multiple),
            max(multiple, round(height / multiple) * multiple),
        )
        # Snapping can still overshoot by one grid step (768p at 16:9 rounds
        # to 1376 wide; the model was trained at 1344). Step the long edge
        # down until the canvas fits.
        if video.max_pixels is not None:
            canvas = _fit_pixel_budget(canvas, video.max_pixels, multiple)
    if max(canvas) > MAX_CONTAINER_EDGE:
        raise ValueError(
            f"canvas {canvas[0]}x{canvas[1]} exceeds the container's "
            f"{MAX_CONTAINER_EDGE} pixel edge limit"
        )
    seed = params.seed
    if seed is None:
        seed = int.from_bytes(
            hashlib.sha256(params.prompt.encode()).digest()[:4], "big"
        )
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
