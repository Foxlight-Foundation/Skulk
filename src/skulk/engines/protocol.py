"""Control protocol between Skulk's host and an engine package's adapter.

Every control message is one frame: a four-byte big-endian length followed by
a UTF-8 JSON object of at most ``MAX_CONTROL_FRAME_BYTES``. Each message names
its ``type``, carries the request's ``sequence`` (echoed by the response) and
the package ``build_digest``. Model-bearing messages also carry the instance and
the exact model identity. Bulk data never travels here: a successful load
returns a loopback port that serves the chat dialect directly.

The contract is strict. Unknown fields and unknown types are refused, so a
package built against a newer protocol fails loudly instead of being guessed at.
Version negotiation picks the highest version both sides support.

This module is dependency-light and performs no I/O beyond the frame helpers,
so an adapter can carry an equivalent copy of the framing without importing
Skulk.
"""

import struct
from typing import Annotated, BinaryIO, Final, Literal, final

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

PROTOCOL_VERSIONS_SUPPORTED: Final[tuple[int, ...]] = (1,)
"""Control protocol versions this Skulk build speaks."""

MAX_CONTROL_FRAME_BYTES: Final = 64 * 1024
"""Upper bound on one frame's JSON body, enforced on read and write."""

_LENGTH_PREFIX: Final = struct.Struct(">I")
_MAX_TEXT_CHARS: Final = 1024
_SHA256_HEX: Final = r"^[0-9a-f]{64}$"

ControlErrorCode = Literal[
    "frame_too_large",
    "truncated",
    "invalid_frame",
    "version_unsupported",
]


@final
class ControlProtocolError(Exception):
    """A control frame could not be read, written or validated.

    ``code`` names the failure in a stable vocabulary. Messages never carry
    engine output or file contents, only the reason.
    """

    def __init__(self, code: ControlErrorCode, message: str) -> None:
        """Record the stable failure code and a short human-readable reason."""
        super().__init__(message)
        self.code: ControlErrorCode = code


class _Strict(BaseModel):
    """Base for every control message: strict types, no unknown fields."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class ModelIdentity(_Strict):
    """The exact model a message refers to, and where its files are on this node.

    The directory is the installed model as Skulk verified it. An adapter
    reads it and never writes to it.
    """

    model_id: str = Field(
        min_length=1,
        max_length=_MAX_TEXT_CHARS,
        description="Catalog model identifier.",
    )
    source_revision: str | None = Field(
        pattern=r"^[0-9a-f]{40}$",
        description=(
            "Full immutable source revision the weights were taken from, or "
            "null for a card that pins none."
        ),
    )
    directory: str = Field(
        min_length=1,
        max_length=4096,
        pattern=r"^/",
        description="Absolute path of the model's files on this node.",
    )


class LoadSettings(_Strict):
    """Admitted settings the adapter must launch the engine with, stated outright.

    The engine must not size itself from free memory. A load that reports
    different applied settings fails.
    """

    context_tokens: int = Field(
        ge=1, le=1 << 22, description="Context window in tokens."
    )
    parallel_streams: int = Field(
        ge=1, le=256, description="Concurrent streams the engine may serve."
    )
    cache_precision: str = Field(
        min_length=1,
        max_length=32,
        description="Prompt-cache precision label, engine-specific.",
    )
    drafting: bool = Field(description="Whether speculative drafting is enabled.")


class _Frame(_Strict):
    """Fields every control message carries."""

    sequence: int = Field(
        ge=1, le=(1 << 53) - 1, description="Request sequence; responses echo it."
    )
    build_digest: str = Field(
        pattern=_SHA256_HEX,
        description="Digest of the installed package manifest this message belongs to.",
    )


class HelloRequest(_Frame):
    """Open the session and offer the protocol versions the host speaks."""

    type: Literal["hello"] = "hello"
    versions: tuple[int, ...] = Field(
        min_length=1, max_length=16, description="Versions the host supports."
    )


class DescribeRequest(_Frame):
    """Ask what the package is and which model families it claims."""

    type: Literal["describe"] = "describe"


class ProbeRequest(_Frame):
    """Check that the package can run on this node. A probe is never readiness."""

    type: Literal["probe"] = "probe"


class EstimateRequest(_Frame):
    """Ask how many bytes one model needs at these exact settings."""

    type: Literal["estimate"] = "estimate"
    instance_id: str = Field(min_length=1, max_length=128)
    model: ModelIdentity
    settings: LoadSettings


class LoadRequest(_Frame):
    """Load a model and report the loopback endpoint that serves it."""

    type: Literal["load"] = "load"
    instance_id: str = Field(min_length=1, max_length=128)
    model: ModelIdentity
    settings: LoadSettings


class HealthRequest(_Frame):
    """Ask whether a loaded instance is still serving."""

    type: Literal["health"] = "health"
    instance_id: str = Field(min_length=1, max_length=128)


class CancelRequest(_Frame):
    """Cancel one in-flight request. Answered only after the engine acknowledges."""

    type: Literal["cancel"] = "cancel"
    instance_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)


class UnloadRequest(_Frame):
    """Release an instance's memory. Reported separately from cancellation."""

    type: Literal["unload"] = "unload"
    instance_id: str = Field(min_length=1, max_length=128)


class StopRequest(_Frame):
    """Stop the engine and exit. The adapter answers, then ends."""

    type: Literal["stop"] = "stop"


RequestMessage = Annotated[
    HelloRequest
    | DescribeRequest
    | ProbeRequest
    | EstimateRequest
    | LoadRequest
    | HealthRequest
    | CancelRequest
    | UnloadRequest
    | StopRequest,
    Field(discriminator="type"),
]


class HelloResponse(_Frame):
    """Chosen protocol version; the host refuses a version it did not offer."""

    type: Literal["hello"] = "hello"
    version: int = Field(ge=1, description="Version the adapter selected.")


class DescribeResponse(_Frame):
    """The package's engine version and the model families it claims to serve."""

    type: Literal["describe"] = "describe"
    engine_version: str = Field(min_length=1, max_length=_MAX_TEXT_CHARS)
    model_families: tuple[
        Annotated[str, Field(min_length=1, max_length=_MAX_TEXT_CHARS)], ...
    ] = Field(
        max_length=64,
        description="Families the package claims. A claim is not a load promise.",
    )


class ProbeResponse(_Frame):
    """Whether the package runs here, with the engine's own detail."""

    type: Literal["probe"] = "probe"
    usable: bool
    detail: str = Field(max_length=_MAX_TEXT_CHARS)


class EstimateResponse(_Frame):
    """Memory need for one model. ``None`` byte counts mean unknown, never zero."""

    type: Literal["estimate"] = "estimate"
    pool: Literal["gpu", "host"]
    load_bytes: int | None = Field(ge=0)
    steady_bytes: int | None = Field(ge=0)


class LoadResponse(_Frame):
    """The loopback port serving the loaded model, and the settings actually taken."""

    type: Literal["load"] = "load"
    endpoint_port: int = Field(ge=1, le=65535)
    applied: LoadSettings


class HealthResponse(_Frame):
    """Whether a loaded instance is ready to serve."""

    type: Literal["health"] = "health"
    ready: bool


class CancelResponse(_Frame):
    """The engine acknowledged the cancellation of ``request_id``."""

    type: Literal["cancel"] = "cancel"
    request_id: str = Field(min_length=1, max_length=128)
    acknowledged: bool


class UnloadResponse(_Frame):
    """Whether the instance's memory was released."""

    type: Literal["unload"] = "unload"
    released: bool


class StopResponse(_Frame):
    """The adapter accepted the stop and will exit."""

    type: Literal["stop"] = "stop"


ErrorCode = Literal[
    "load_refused",
    "unsupported_model",
    "engine_failed",
    "busy",
    "unknown_instance",
    "invalid_request",
    "internal",
]


class ErrorResponse(_Frame):
    """A typed refusal. The message is the engine's own reason, bounded."""

    type: Literal["error"] = "error"
    code: ErrorCode
    message: str = Field(max_length=_MAX_TEXT_CHARS)


ResponseMessage = Annotated[
    HelloResponse
    | DescribeResponse
    | ProbeResponse
    | EstimateResponse
    | LoadResponse
    | HealthResponse
    | CancelResponse
    | UnloadResponse
    | StopResponse
    | ErrorResponse,
    Field(discriminator="type"),
]

_REQUEST_ADAPTER: Final[TypeAdapter[RequestMessage]] = TypeAdapter(RequestMessage)
_RESPONSE_ADAPTER: Final[TypeAdapter[ResponseMessage]] = TypeAdapter(ResponseMessage)


def encode_frame(message: _Frame) -> bytes:
    """Serialize one control message as a length-prefixed frame.

    Raises ``ControlProtocolError`` with code ``frame_too_large`` when the JSON
    body exceeds ``MAX_CONTROL_FRAME_BYTES``.
    """
    body = message.model_dump_json().encode("utf-8")
    if len(body) > MAX_CONTROL_FRAME_BYTES:
        raise ControlProtocolError(
            "frame_too_large",
            f"control frame of {len(body)} bytes exceeds {MAX_CONTROL_FRAME_BYTES}",
        )
    return _LENGTH_PREFIX.pack(len(body)) + body


def read_frame_body(stream: BinaryIO) -> bytes | None:
    """Read one frame's JSON body, or ``None`` at a clean end of stream.

    An oversized length is refused before any body byte is read. End of stream
    inside a frame raises ``truncated``.
    """
    header = _read_exactly(stream, _LENGTH_PREFIX.size, allow_clean_eof=True)
    if header is None:
        return None
    length = int.from_bytes(header, "big")
    if length > MAX_CONTROL_FRAME_BYTES:
        raise ControlProtocolError(
            "frame_too_large",
            f"control frame announces {length} bytes; the limit is {MAX_CONTROL_FRAME_BYTES}",
        )
    body = _read_exactly(stream, length, allow_clean_eof=False)
    if body is None:  # pragma: no cover - allow_clean_eof=False never returns None
        raise ControlProtocolError("truncated", "control frame body ended early")
    return body


def write_frame(stream: BinaryIO, message: _Frame) -> None:
    """Write one control message to ``stream`` and flush it."""
    stream.write(encode_frame(message))
    stream.flush()


def decode_request(body: bytes) -> RequestMessage:
    """Validate a frame body as a host-to-adapter request."""
    return _decode(_REQUEST_ADAPTER, body)


def decode_response(body: bytes) -> ResponseMessage:
    """Validate a frame body as an adapter-to-host response."""
    return _decode(_RESPONSE_ADAPTER, body)


def negotiate_version(
    offered: tuple[int, ...],
    supported: tuple[int, ...] = PROTOCOL_VERSIONS_SUPPORTED,
) -> int:
    """Return the highest version both sides support.

    Raises ``ControlProtocolError`` with code ``version_unsupported`` when the
    two sets are disjoint. Core moves this window with its releases; it never
    pins one version.
    """
    common = set(offered) & set(supported)
    if not common:
        raise ControlProtocolError(
            "version_unsupported",
            f"no common control protocol: package offers {sorted(offered)}, "
            f"this build supports {sorted(supported)}",
        )
    return max(common)


def _decode[Message: BaseModel](adapter: TypeAdapter[Message], body: bytes) -> Message:
    try:
        return adapter.validate_json(body)
    except ValidationError as error:
        raise ControlProtocolError(
            "invalid_frame",
            f"control frame failed validation with {error.error_count()} error(s)",
        ) from None


def _read_exactly(
    stream: BinaryIO, count: int, *, allow_clean_eof: bool
) -> bytes | None:
    if count == 0:
        return b""
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if allow_clean_eof and remaining == count:
                return None
            raise ControlProtocolError(
                "truncated", f"control stream ended with {remaining} bytes outstanding"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
