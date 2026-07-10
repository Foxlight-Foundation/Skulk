"""Shared lifecycle vocabulary for ordered fabric streams."""

from typing import Literal, TypeGuard

StreamFrameKind = Literal[
    "started",
    "chunk",
    "completed",
    "failed",
    "cancelled",
]
"""Lifecycle kind carried by every ordered stream frame."""

TerminalStreamFrameKind = Literal["completed", "failed", "cancelled"]
"""Lifecycle kinds after which a producer must emit no further frames."""


def is_terminal_stream_frame_kind(
    kind: StreamFrameKind,
) -> TypeGuard[TerminalStreamFrameKind]:
    """Return whether ``kind`` closes an ordered stream."""

    return kind in ("completed", "failed", "cancelled")
