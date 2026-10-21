"""Pure Motion Track geometry policy shared with the independent engine wheel.

The engine cannot import Core. Its byte-identical policy copy is checked by
the packaging suite so request admission and device allocation cannot drift.
"""

import math

MAX_TRACK_BYTES = 4 * 1024 * 1024
MAX_TRACK_IMAGE_BYTES = 512 * 1024 * 1024
MAX_TRACK_WORKSPACE_BYTES = 256 * 1024 * 1024


def resolve_motion_canvas(requested: tuple[int, int] | None, short_edge: int | None, aspect_ratio: str | None, max_pixels: int | None) -> tuple[int, int]:
    """Resolve deterministic canvas defaults without inspecting the opening image.

    Motion coordinates describe the final canvas, so its geometry must agree
    at admission and execution, before reference bytes have worker-local paths.
    Explicit dimensions are retained. Otherwise the card's short edge and the
    requested ratio (square when omitted) are snapped to the 64-pixel grid.
    """
    if requested is not None:
        return requested
    horizontal, vertical = (int(value) for value in (aspect_ratio or "1:1").split(":"))
    short = short_edge or 64
    width, height = (round(short * horizontal / vertical), short) if horizontal >= vertical else (short, round(short * vertical / horizontal))
    if max_pixels is not None and width * height > max_pixels:
        scale = math.sqrt(max_pixels / (width * height))
        width, height = round(width * scale), round(height * scale)
    width, height = max(64, round(width / 64) * 64), max(64, round(height / 64) * 64)
    if max_pixels is not None:
        while width * height > max_pixels:
            if width >= height and width > 64:
                width -= 64
            elif height > 64:
                height -= 64
            else:
                break
    return width, height


def validate_motion_geometry(width: int, height: int, frames: int) -> tuple[int, int]:
    """Validate retained guide and transient workspace before device allocation.

    Return the trained raster resolution. Workspace conservatively accounts
    for float32 raster pixels, float/long overlap-priority maps, resized output
    temporaries and bounded circle arrays. The limit describes this workflow
    implementation, independently of model or hardware qualification.
    """
    if min(width, height) < 128 or max(width, height) > 4096 or width % 64 or height % 64:
        raise ValueError("Motion Track output requires a bounded 64-pixel canvas grid")
    if not 9 <= frames <= 2881 or (frames - 1) % 8:
        raise ValueError("Motion Track requires a bounded complete 8n+1 timeline")
    if width * height * frames * 6 > MAX_TRACK_IMAGE_BYTES:
        raise ValueError("Motion Track guide exceeds its decoded-pixel budget; reduce canvas or duration")
    raster_width = int(width * 1080 / height) if height <= width else 1080
    raster_height = 1080 if height <= width else int(height * 1080 / width)
    workspace = raster_width * raster_height * 32 + width * height * 32 + 16 * 1024 * 1024
    if workspace > MAX_TRACK_WORKSPACE_BYTES:
        raise ValueError("Motion Track raster workspace exceeds its budget; reduce canvas or aspect ratio")
    return raster_width, raster_height
