"""Bounded implementation of Lightricks' sparse Motion Track raster format."""

import json
from collections.abc import Callable
from math import isfinite
from typing import cast

import torch
from torch import Tensor

from skulk_comfy_ltx.motion_geometry import (
    MAX_TRACK_BYTES as MAX_TRACK_BYTES,
)
from skulk_comfy_ltx.motion_geometry import (
    validate_motion_geometry,
)


def parse_motion_tracks(raw: bytes, expected_frames: int) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Validate version-one normalized paths before allocating device tensors.

    Each path contains exactly one position per output frame. Missing points,
    padded tracks, non-finite coordinates and unrecognized fields are refused.
    """
    if not 0 < len(raw) <= MAX_TRACK_BYTES or not 9 <= expected_frames <= 2881 or (expected_frames - 1) % 8:
        raise ValueError("Motion Track media or output timeline exceeds its bound")
    decoded = cast(object, json.loads(raw))
    if not isinstance(decoded, dict):
        raise ValueError("Motion Track data must be a version-one object")
    data = cast(dict[str, object], decoded)
    if set(data) != {"version", "frame_count", "tracks"} or type(data.get("version")) is not int or data["version"] != 1 or type(data.get("frame_count")) is not int or data["frame_count"] != expected_frames:
        raise ValueError("Motion Track data must match version one and the complete output timeline")
    tracks = data["tracks"]
    if not isinstance(tracks, list):
        raise ValueError("Motion Track data requires between one and sixteen paths")
    paths = cast(list[object], tracks)
    if not 1 <= len(paths) <= 16:
        raise ValueError("Motion Track data requires between one and sixteen paths")
    result: list[tuple[tuple[float, float], ...]] = []
    for candidate in paths:
        if not isinstance(candidate, list):
            raise ValueError("Every motion path must cover every output frame")
        positions = cast(list[object], candidate)
        if len(positions) != expected_frames:
            raise ValueError("Every motion path must cover every output frame")
        points: list[tuple[float, float]] = []
        for candidate_point in positions:
            if not isinstance(candidate_point, dict):
                raise ValueError("A motion position must contain normalized x and y")
            point = cast(dict[str, object], candidate_point)
            if set(point) != {"x", "y"}:
                raise ValueError("A motion position must contain only x and y")
            coordinates = (point["x"], point["y"])
            if any(type(value) not in (float, int) or not isinstance(value, (float, int)) or not isfinite(value) or not 0.0 <= value <= 1.0 for value in coordinates):
                raise ValueError("Motion positions must be finite normalized numbers")
            points.append((float(cast(float | int, point["x"])), float(cast(float | int, point["y"]))))
        result.append(tuple(points))
    return tuple(result)


def _age_colors(ratios: Tensor, device: torch.device) -> Tensor:
    colors = torch.zeros(ratios.shape[0], 3, device=device)
    first = ratios <= 1 / 3
    colors[first, 1] = ratios[first] * 3
    colors[first, 2] = 1 - ratios[first] * 3
    second = (ratios > 1 / 3) & (ratios <= 2 / 3)
    colors[second, 0] = (ratios[second] - 1 / 3) * 3
    colors[second, 1] = 1
    third = ratios > 2 / 3
    colors[third, 0] = 1
    colors[third, 1] = 1 - (ratios[third] - 2 / 3) * 3
    return colors


def _rasterise(frame: Tensor, points: Tensor, radii: Tensor, colors: Tensor, distance_squared: Tensor) -> None:
    height, width = frame.shape[:2]
    count = points.shape[0]
    device = points.device
    offsets = torch.arange(19, device=device) - 9
    vertical = (points[:, 1].round().long().view(count, 1, 1) + offsets.view(1, 19, 1)).expand(count, 19, 19)
    horizontal = (points[:, 0].round().long().view(count, 1, 1) + offsets.view(1, 1, 19)).expand(count, 19, 19)
    valid = (distance_squared.unsqueeze(0) <= radii.square().view(count, 1, 1)) & (vertical >= 0) & (vertical < height) & (horizontal >= 0) & (horizontal < width)
    positions = (vertical[valid] * width + horizontal[valid]).long()
    painter_order = torch.arange(count, device=device, dtype=torch.float32).view(count, 1, 1).expand_as(valid)[valid]
    priority = torch.full((height * width,), -1.0, device=device)
    priority.scatter_reduce_(0, positions, painter_order, reduce="amax", include_self=False)
    priority = priority.view(height, width).long()
    painted = priority >= 0
    frame[painted] = colors[priority[painted]]


def render_motion_tracks(raw: bytes, width: int, height: int, expected_frames: int, device: torch.device, *, check_cancelled: Callable[[], None]) -> Tensor:
    """Render the trained BGR trail format with one high-resolution frame live.

    The 1080-pixel short side, fifty-frame trails, age gradient, radii, overlap
    ordering and bilinear resize follow the pinned upstream sparse renderer.
    Only its allocation lifetime changes; complete high-resolution clips are
    never retained. Cancellation is checked between frames.
    """
    paths = parse_motion_tracks(raw, expected_frames)
    render_width, render_height = validate_motion_geometry(width, height, expected_frames)
    points = torch.tensor(paths, dtype=torch.float32, device=device)
    points[..., 0] *= (width - 1) * render_width / width
    points[..., 1] *= (height - 1) * render_height / height
    offsets = torch.arange(19, device=device) - 9
    vertical, horizontal = torch.meshgrid(offsets, offsets, indexing="ij")
    distance_squared = vertical.float().square() + horizontal.float().square()
    result = torch.empty(expected_frames, height, width, 3, device=device, dtype=torch.float16)
    for frame_index in range(expected_frames):
        check_cancelled()
        start = max(0, frame_index - 50)
        window = frame_index - start + 1
        ages = torch.arange(window - 1, -1, -1, device=device, dtype=torch.float32)
        ratios = 1.0 - ages / 50
        radii = (2 + 6 * ratios).unsqueeze(0).expand(len(paths), -1).reshape(-1)
        colors = _age_colors(ratios, device).unsqueeze(0).expand(len(paths), -1, -1).reshape(-1, 3)
        order = ages.unsqueeze(0).expand(len(paths), -1).reshape(-1).argsort(descending=True)
        frame = torch.zeros(render_height, render_width, 3, device=device)
        _rasterise(frame, points[:, start:frame_index + 1].reshape(-1, 2)[order], radii[order], colors[order], distance_squared)
        resized = torch.nn.functional.interpolate(frame.permute(2, 0, 1).unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0)
        result[frame_index] = resized[..., [2, 1, 0]].half()
        # Release frame views before allocating the next high-resolution raster.
        del frame, resized
    return result
