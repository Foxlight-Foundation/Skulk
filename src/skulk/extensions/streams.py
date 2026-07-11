"""Transport-independent provider streaming contracts and receive state."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Literal, cast, final

import anyio
from pydantic import BaseModel, ConfigDict, Field, model_validator

from skulk.extensions.calls import CapabilityResult
from skulk.shared.types.streaming import StreamFrameKind

StreamDirection = Literal["caller_to_provider", "provider_to_caller"]
"""One independently sequenced direction of a provider call."""

CapabilityStreamErrorCode = Literal[
    "provider_error",
    "transport_error",
    "unreachable",
    "cancelled",
    "timeout",
    "invalid_frame",
]
"""Machine-readable terminal failure classes for provider streams."""

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_INLINE_MEDIA_BYTES = 1_048_576
"""Largest realtime media attachment accepted in one provider frame."""


class CapabilityStreamError(BaseModel):
    """Typed provider-stream failure carried by failed/cancelled frames."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    code: CapabilityStreamErrorCode
    message: str


class InlineMediaAttachment(BaseModel):
    """Realtime media bytes carried beside a structured stream header."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    kind: Literal["inline"] = "inline"
    data: bytes = Field(max_length=MAX_INLINE_MEDIA_BYTES)
    media_type: str
    codec: str | None = None
    sample_rate: int | None = Field(default=None, gt=0)
    channels: int | None = Field(default=None, gt=0)
    timestamp_seconds: float | None = Field(default=None, ge=0)
    duration_seconds: float | None = Field(default=None, ge=0)


class BlobMediaAttachment(BaseModel):
    """Reference to a staged immutable media object with integrity metadata."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    kind: Literal["blob"] = "blob"
    blob_id: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=_SHA256_PATTERN.pattern)
    media_type: str


MediaAttachment = Annotated[
    InlineMediaAttachment | BlobMediaAttachment,
    Field(discriminator="kind"),
]
"""Binary-inline or staged/blob-backed provider stream media."""


class CapabilityStreamFrame(BaseModel):
    """One ordered lifecycle frame for a provider call direction."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    call_id: str
    direction: StreamDirection
    sequence: int = Field(ge=0)
    kind: StreamFrameKind
    synthetic: bool = False
    payload: dict[str, object] | None = None
    media: MediaAttachment | None = None
    error: CapabilityStreamError | None = None

    @model_validator(mode="after")
    def _validate_lifecycle_shape(self) -> "CapabilityStreamFrame":
        if self.sequence == 0 and self.kind != "started" and not self.synthetic:
            raise ValueError("provider stream sequence 0 must be started")
        if self.kind == "started":
            if self.sequence != 0:
                raise ValueError("started provider stream frame must be sequence 0")
            if (
                self.payload is not None
                or self.media is not None
                or self.error is not None
            ):
                raise ValueError("started provider stream frame carries no payload")
        elif self.kind == "chunk":
            if self.payload is None and self.media is None:
                raise ValueError(
                    "chunk provider stream frame requires payload or media"
                )
            if self.error is not None:
                raise ValueError("chunk provider stream frame must not carry an error")
        elif self.kind == "completed":
            if self.error is not None:
                raise ValueError(
                    "completed provider stream frame must not carry an error"
                )
        elif self.kind == "failed":
            if self.error is None or self.error.code == "cancelled":
                raise ValueError(
                    "failed provider stream frame requires a non-cancel error"
                )
            if self.payload is not None or self.media is not None:
                raise ValueError("failed provider stream frame carries no payload")
        else:
            if self.error is None or self.error.code != "cancelled":
                raise ValueError(
                    "cancelled provider stream frame requires cancelled error"
                )
            if self.payload is not None or self.media is not None:
                raise ValueError(
                    "cancelled provider stream frame carries no payload"
                )

        if self.payload is not None:
            try:
                json.dumps(self.payload, allow_nan=False)
            except (TypeError, ValueError, RecursionError, OverflowError) as exc:
                raise ValueError(
                    "provider stream payload must be JSON-serializable; use media "
                    "attachments for bytes"
                ) from exc
        return self

    @property
    def is_terminal(self) -> bool:
        """Whether this frame closes its call direction."""

        return self.kind in ("completed", "failed", "cancelled")


class CapabilityStreamCancel(BaseModel):
    """Control-sized request to cancel one admitted provider stream."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    call_id: str
    caller_node: str
    target_node: str
    message: str = "caller stopped consuming the provider stream"


@final
class CapabilityStreamInput:
    """Caller-owned input direction for a client-streaming provider call.

    The sink owns lifecycle and sequence numbers so callers cannot accidentally
    create gaps or duplicate terminals. ``complete`` is an input half-close: it
    ends only ``caller_to_provider`` and leaves provider output active.
    """

    def __init__(
        self,
        *,
        call_id: str,
        deadline_at: float,
        send_frame: Callable[[CapabilityStreamFrame], Awaitable[None]],
    ) -> None:
        """Create an unopened input direction for one admitted call."""

        self.call_id = call_id
        self._deadline_at = deadline_at
        self._send_frame = send_frame
        self._next_sequence = 0
        self._closed = False
        self._terminal_sent = False
        self._lock = anyio.Lock()

    @property
    def closed(self) -> bool:
        """Whether this input direction rejects further caller sends."""

        return self._closed

    async def start(self) -> None:
        """Emit the caller direction's required sequence-zero opener."""

        async with self._lock:
            # Provider output may fail and close the sink while this task waits
            # to acquire the lock. That terminal output is authoritative; an
            # input opener is no longer needed and must not escape as a caller
            # exception instead of letting the queued terminal be consumed.
            if self._closed:
                return
            if self._next_sequence != 0:
                raise RuntimeError("provider input direction is already started")
            await self._emit("started")

    async def send_chunk(
        self,
        *,
        payload: dict[str, object] | None = None,
        media: MediaAttachment | None = None,
    ) -> None:
        """Send one structured or binary media input chunk.

        Args:
            payload: JSON metadata validated against ``input_chunk_schema`` by
                the provider node.
            media: Optional inline bytes or staged blob reference.

        Raises:
            RuntimeError: The input direction is closed or was not started.
            ValueError: Neither payload nor media was supplied.
            TimeoutError: The call's single deadline budget has expired.
        """

        if payload is None and media is None:
            raise ValueError("provider input chunk requires payload or media")
        async with self._lock:
            self._require_open()
            await self._emit("chunk", payload=payload, media=media)

    async def complete(self) -> None:
        """Half-close caller input while leaving provider output active."""

        async with self._lock:
            self._require_open()
            await self._emit("completed")
            self._terminal_sent = True
            self._closed = True

    async def cancel(
        self,
        message: str = "caller cancelled the provider input stream",
    ) -> None:
        """Cancel the logical call from the caller input direction."""

        async with self._lock:
            if self._closed:
                return
            if self._next_sequence == 0:
                raise RuntimeError("provider input direction is not started")
            await self._emit(
                "cancelled",
                error=CapabilityStreamError(code="cancelled", message=message),
            )
            self._terminal_sent = True
            self._closed = True

    def close_locally(self) -> None:
        """Prevent late sends after the provider output direction terminates."""

        # Cleanup must not wait behind a send holding ``_lock``: on a saturated
        # same-node channel that send may depend on the DATA consumer currently
        # closing this stream. One already in-flight frame may finish and will
        # be dropped as late by the removed provider state.
        self._closed = True

    async def finish_local_close(self) -> None:
        """Best-effort publish cancellation after synchronous local closure.

        The API schedules this in a separate task so waiting for an in-flight
        send cannot block the DATA consumer responsible for draining that send.
        """

        async with self._lock:
            if self._next_sequence == 0 or self._terminal_sent:
                return
            try:
                await self._emit(
                    "cancelled",
                    error=CapabilityStreamError(
                        code="cancelled",
                        message="provider output closed the caller input stream",
                    ),
                )
            except (
                TimeoutError,
                anyio.BrokenResourceError,
                anyio.ClosedResourceError,
            ):
                return
            self._terminal_sent = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("provider input direction is closed")
        if self._next_sequence == 0:
            raise RuntimeError("provider input direction is not started")

    async def _emit(
        self,
        kind: StreamFrameKind,
        *,
        payload: dict[str, object] | None = None,
        media: MediaAttachment | None = None,
        error: CapabilityStreamError | None = None,
    ) -> None:
        remaining = self._deadline_at - anyio.current_time()
        if remaining <= 0:
            self._closed = True
            raise TimeoutError("provider stream deadline expired before input send")
        frame = CapabilityStreamFrame(
            call_id=self.call_id,
            direction="caller_to_provider",
            sequence=self._next_sequence,
            kind=kind,
            payload=payload,
            media=media,
            error=error,
        )
        try:
            with anyio.fail_after(remaining):
                await self._send_frame(frame)
        except TimeoutError:
            self._closed = True
            raise
        except (anyio.BrokenResourceError, anyio.ClosedResourceError):
            self._closed = True
            raise
        self._next_sequence += 1


@final
@dataclass(frozen=True)
class CapabilityStreamSession:
    """Opening result, optional input sink, and one-shot output iterator."""

    open_result: CapabilityResult
    frames: AsyncIterator[CapabilityStreamFrame]
    input: CapabilityStreamInput | None = None


def encode_capability_stream_frame(
    frame: CapabilityStreamFrame,
) -> tuple[bytes, bytes | None]:
    """Encode one provider frame as JSON header plus optional raw media bytes."""

    header: dict[str, object] = {
        "call_id": frame.call_id,
        "direction": frame.direction,
        "sequence": frame.sequence,
        "kind": frame.kind,
        "synthetic": frame.synthetic,
        "payload": frame.payload,
        "media": None,
        "error": (
            None
            if frame.error is None
            else {"code": frame.error.code, "message": frame.error.message}
        ),
    }
    media_bytes: bytes | None = None
    if isinstance(frame.media, InlineMediaAttachment):
        media_bytes = frame.media.data
        header["media"] = {
            "kind": "inline",
            "media_type": frame.media.media_type,
            "codec": frame.media.codec,
            "sample_rate": frame.media.sample_rate,
            "channels": frame.media.channels,
            "timestamp_seconds": frame.media.timestamp_seconds,
            "duration_seconds": frame.media.duration_seconds,
        }
    elif isinstance(frame.media, BlobMediaAttachment):
        header["media"] = frame.media.model_dump(mode="python")
    return (
        json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8"),
        media_bytes,
    )


def decode_capability_stream_frame(
    header: bytes,
    media_bytes: bytes | None,
) -> CapabilityStreamFrame:
    """Decode a provider frame header and its optional raw media attachment."""

    decoded_object = cast(object, json.loads(header.decode("utf-8")))
    if not isinstance(decoded_object, dict):
        raise ValueError("provider stream header must decode to an object")
    decoded = cast(dict[str, object], decoded_object)
    media = decoded.get("media")
    typed_media = cast(dict[str, object], media) if isinstance(media, dict) else None
    if typed_media is not None and typed_media.get("kind") == "inline":
        if media_bytes is None:
            raise ValueError("inline provider stream frame is missing media bytes")
        typed_media["data"] = media_bytes
    elif media_bytes is not None:
        raise ValueError("raw media bytes require an inline media attachment header")
    return CapabilityStreamFrame.model_validate(decoded)


@dataclass(frozen=True)
class CapabilityStreamReceiveBatch:
    """Outcome of accepting or expiring one provider stream frame."""

    ready: tuple[CapabilityStreamFrame, ...] = ()
    duplicate_frames: int = 0
    out_of_order_frames: int = 0
    late_frames: int = 0
    synthesized_terminal: CapabilityStreamFrame | None = None


class CapabilityStreamReceiver:
    """Enforce ordered, bounded, single-terminal receive semantics for one call."""

    def __init__(
        self,
        *,
        call_id: str,
        direction: StreamDirection,
        reorder_limit: int = 64,
        gap_timeout_seconds: float = 5.0,
        idle_timeout_seconds: float = 120.0,
    ) -> None:
        """Create receiver state for one ``(call_id, direction)`` stream."""

        if reorder_limit < 1:
            raise ValueError("reorder_limit must be positive")
        if gap_timeout_seconds <= 0 or idle_timeout_seconds <= 0:
            raise ValueError("stream deadlines must be positive")
        self.call_id = call_id
        self.direction: StreamDirection = direction
        self.reorder_limit = reorder_limit
        self.gap_timeout_seconds = gap_timeout_seconds
        self.idle_timeout_seconds = idle_timeout_seconds
        self.next_sequence = 0
        self._pending: dict[int, CapabilityStreamFrame] = {}
        self._gap_sequence: int | None = None
        self._gap_since: float | None = None
        self._last_progress_at: float | None = None
        self._terminal: CapabilityStreamFrame | None = None

    @property
    def terminal(self) -> CapabilityStreamFrame | None:
        """The accepted or synthesized terminal frame, when closed."""

        return self._terminal

    def accept(
        self,
        frame: CapabilityStreamFrame,
        *,
        observed_at: float,
    ) -> CapabilityStreamReceiveBatch:
        """Accept one frame, releasing only a contiguous ordered prefix."""

        if frame.call_id != self.call_id or frame.direction != self.direction:
            terminal = self._synthesize_failure(
                "invalid_frame",
                "provider stream frame identity does not match receiver",
            )
            return CapabilityStreamReceiveBatch(synthesized_terminal=terminal)
        if self._terminal is not None:
            if frame.sequence < self.next_sequence:
                return CapabilityStreamReceiveBatch(duplicate_frames=1)
            return CapabilityStreamReceiveBatch(late_frames=1)
        if frame.sequence < self.next_sequence or frame.sequence in self._pending:
            return CapabilityStreamReceiveBatch(duplicate_frames=1)

        out_of_order = int(frame.sequence > self.next_sequence)
        self._pending[frame.sequence] = frame
        if len(self._pending) > self.reorder_limit:
            terminal = self._synthesize_failure(
                "transport_error",
                "provider stream reorder window exceeded",
            )
            return CapabilityStreamReceiveBatch(
                out_of_order_frames=out_of_order,
                synthesized_terminal=terminal,
            )

        ready: list[CapabilityStreamFrame] = []
        while self.next_sequence in self._pending:
            current = self._pending.pop(self.next_sequence)
            ready.append(current)
            self.next_sequence += 1
            self._last_progress_at = observed_at
            if current.is_terminal:
                self._terminal = current
                self._pending.clear()
                break
        if self._pending:
            if self._gap_sequence != self.next_sequence:
                self._gap_sequence = self.next_sequence
                self._gap_since = observed_at
        else:
            self._gap_sequence = None
            self._gap_since = None
        return CapabilityStreamReceiveBatch(
            ready=tuple(ready),
            out_of_order_frames=out_of_order,
        )

    def expire(self, *, observed_at: float) -> CapabilityStreamReceiveBatch:
        """Synthesize a typed terminal when a gap or active stream times out."""

        if self._terminal is not None:
            return CapabilityStreamReceiveBatch()
        if (
            self._pending
            and self._gap_since is not None
            and observed_at - self._gap_since >= self.gap_timeout_seconds
        ):
            terminal = self._synthesize_failure(
                "transport_error",
                "provider stream sequence gap did not resolve before deadline",
            )
            return CapabilityStreamReceiveBatch(synthesized_terminal=terminal)
        if (
            self._last_progress_at is not None
            and observed_at - self._last_progress_at >= self.idle_timeout_seconds
        ):
            terminal = self._synthesize_failure(
                "timeout",
                "provider stream exceeded its idle deadline",
            )
            return CapabilityStreamReceiveBatch(synthesized_terminal=terminal)
        return CapabilityStreamReceiveBatch()

    def cancel(
        self, message: str = "provider stream cancelled"
    ) -> CapabilityStreamFrame | None:
        """Close locally with one idempotent cancelled terminal frame."""

        if self._terminal is not None:
            return None
        terminal = CapabilityStreamFrame(
            call_id=self.call_id,
            direction=self.direction,
            sequence=self.next_sequence,
            kind="cancelled",
            synthetic=True,
            error=CapabilityStreamError(code="cancelled", message=message),
        )
        self._terminal = terminal
        self.next_sequence += 1
        self._pending.clear()
        self._gap_sequence = None
        self._gap_since = None
        return terminal

    def fail(
        self,
        code: CapabilityStreamErrorCode,
        message: str,
    ) -> CapabilityStreamFrame | None:
        """Close locally with one idempotent synthesized failed frame."""

        if self._terminal is not None:
            return None
        return self._synthesize_failure(code, message)

    def _synthesize_failure(
        self,
        code: CapabilityStreamErrorCode,
        message: str,
    ) -> CapabilityStreamFrame:
        terminal = CapabilityStreamFrame(
            call_id=self.call_id,
            direction=self.direction,
            sequence=self.next_sequence,
            kind="failed",
            synthetic=True,
            error=CapabilityStreamError(code=code, message=message),
        )
        self._terminal = terminal
        self.next_sequence += 1
        self._pending.clear()
        self._gap_sequence = None
        self._gap_since = None
        return terminal
