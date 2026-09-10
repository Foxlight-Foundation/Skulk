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
from collections.abc import Callable
from dataclasses import dataclass

from skulk.shared.models.model_cards import VideoCardConfig
from skulk.shared.types.video import VideoGenerationTaskParams, VideoStage

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


def _fit_pixel_budget(canvas: tuple[int, int], max_pixels: int, multiple: int) -> tuple[int, int]:
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
        # Rounding the long edge up can overshoot the trained pixel budget
        # by one grid step (768p at 16:9 rounds to 1376 wide; the model was
        # trained at 1344). Step the long edge down until the canvas fits.
        if video.max_pixels is not None:
            canvas = _fit_pixel_budget(canvas, video.max_pixels, multiple)
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
