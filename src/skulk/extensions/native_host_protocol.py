"""Private Core/owner host protocol; public native parsing never grants a role.

This module is mirrored byte-for-byte in the independently distributed owner.
It authenticates host transport, not original callers or author contracts. The
compiled public kernel remains the sole native/schema admission implementation.
"""

import asyncio
import contextlib
import hashlib
import hmac
import json
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal, Protocol, final

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    model_validator,
)

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Profile = Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
Installation = Annotated[str, Field(pattern=r"^managed\.[a-z0-9][a-z0-9._-]{0,80}$")]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9._:@-]{1,128}$")]
MAXIMUM_METADATA = 262144
MAXIMUM_PAYLOAD = 16777216
type Operation = Literal[
    "describe",
    "invoke",
    "delegate",
    "describe_inference",
    "infer",
    "infer_stream",
    "stream_ready",
    "stream_input",
    "stream_next",
    "delegate_inference",
]


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


@final
class HostBinding(_Contract):
    """Manager-provisioned host attachment, never included in author startup."""

    protocol: Literal[1] = Field(description="Explicit private host protocol.")
    profile_id: Profile = Field(description="Actual manager service profile.")
    plugin_id: Installation = Field(description="Exact registered installation.")
    runtime_digest: Digest = Field(description="Exact selected signed generation.")
    transport_node_id: Identifier = Field(description="Actual serving Core identity.")
    service_instance: Profile = Field(description="Manager-owned process lifetime.")
    core_public_key: Digest = Field(description="Core host-role verification key.")
    server_secret: Digest = Field(
        repr=False, description="Private server/session proof seed."
    )

    def transcript(self) -> bytes:
        """Export public binding bytes; omit the server proof secret."""
        return json.dumps(
            self.model_dump(exclude={"server_secret"}),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def digest(self) -> str:
        """Bind all public profile, process, runtime and transport identities."""
        return hashlib.sha256(
            b"skulk.native-host.binding/1\0" + self.transcript()
        ).hexdigest()


@final
class ReadGrant(_Contract):
    """Owner enrollment of one exact trusted local read implementation."""

    plugin_id: Installation = Field(
        description="Registered installation selected by owner."
    )
    runtime_digest: Digest = Field(
        description="Enrolled signed runtime; updates need regrant."
    )
    capability_id: Identifier = Field(description="Exact signed read capability.")
    version: str = Field(
        pattern=r"^\d+\.\d+\.\d+$", description="Exact capability version."
    )
    descriptor_revision: Digest = Field(
        description="Complete compiled descriptor identity."
    )
    processing_domain: Literal["local"] = Field(
        description="Owner-approved local processing."
    )


@final
class ReadPolicy(_Contract):
    """Atomic Core-owned permission; disabled until explicit read enrollment."""

    protocol: Literal[1] = 1
    revision: Digest = Field(
        description="Opaque revision fence changed by every update."
    )
    enabled: bool = Field(description="Whether enrolled local reads may execute.")
    grants: tuple[ReadGrant, ...] = Field(
        max_length=64, description="Exact owner-enrolled reads."
    )

    @model_validator(mode="after")
    def unique_grants(self) -> "ReadPolicy":
        """Reject ambiguous enrollment of two versions of one installed read."""
        identities = [(g.plugin_id, g.capability_id, g.version) for g in self.grants]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate native read enrollment")
        return self


class _Hello(_Contract):
    protocol: Literal[1]
    binding_digest: Digest
    nonce: Digest


class _Challenge(_Contract):
    protocol: Literal[1]
    nonce: Digest
    proof: Digest


class _Proof(_Contract):
    protocol: Literal[1]
    signature: str = Field(pattern=r"^[a-f0-9]{128}$")


@final
@dataclass(frozen=True, slots=True, repr=False)
class SessionKeys:
    """Connection/direction-bound integrity keys; never exposed to author code."""

    outbound: bytes
    inbound: bytes


async def _hello_read(reader: asyncio.StreamReader) -> bytes:
    raw = await reader.readline()
    if not raw.endswith(b"\n") or len(raw) > 4096:
        raise ValueError("native host handshake refused")
    return raw


async def _hello_send(writer: asyncio.StreamWriter, value: BaseModel) -> None:
    writer.write(value.model_dump_json().encode() + b"\n")
    await writer.drain()


def _proof_bytes(binding: HostBinding, client: str, server: str) -> bytes:
    return binding.transcript() + b"\0" + client.encode() + b"\0" + server.encode()


def _keys(
    binding: HostBinding, proof: bytes, signature: bytes, *, core: bool
) -> SessionKeys:
    seed = hmac.digest(
        bytes.fromhex(binding.server_secret),
        b"skulk.native-host.session/1\0" + proof + signature,
        "sha256",
    )
    forward = hmac.digest(seed, b"core-to-owner", "sha256")
    backward = hmac.digest(seed, b"owner-to-core", "sha256")
    return SessionKeys(forward, backward) if core else SessionKeys(backward, forward)


async def admit_core(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    binding: HostBinding,
    current: Callable[[], None],
) -> SessionKeys:
    """Owner admits one fresh Core role proof, bound to this exact live installation.

    The expected binding comes only from normal manager provisioning. A public
    node ID or a structural Context cannot substitute for this handshake.
    """
    async with asyncio.timeout(3):
        current()
        hello = _Hello.model_validate_json(await _hello_read(reader))
        if hello.binding_digest != binding.digest:
            raise ValueError("native host binding differs")
        nonce = secrets.token_hex(32)
        proof = _proof_bytes(binding, hello.nonce, nonce)
        await _hello_send(
            writer,
            _Challenge(
                protocol=1,
                nonce=nonce,
                proof=hmac.digest(
                    bytes.fromhex(binding.server_secret), b"server\0" + proof, "sha256"
                ).hex(),
            ),
        )
        answer = _Proof.model_validate_json(await _hello_read(reader))
        signature = bytes.fromhex(answer.signature)
        try:
            Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(binding.core_public_key)
            ).verify(signature, b"skulk.native-host.core/1\0" + proof)
        except InvalidSignature:
            raise ValueError("native Core role proof refused") from None
        current()
        writer.write(b'{"protocol":1,"ready":true}\n')
        await writer.drain()
        return _keys(binding, proof, signature, core=False)


async def admit_owner(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    binding: HostBinding,
    key: Ed25519PrivateKey,
    current: Callable[[], None],
) -> SessionKeys:
    """Verify the owner before disclosing caller data or signing its challenge."""
    async with asyncio.timeout(3):
        current()
        if key.public_key().public_bytes_raw().hex() != binding.core_public_key:
            raise ValueError("native host role key differs")
        nonce = secrets.token_hex(32)
        await _hello_send(
            writer, _Hello(protocol=1, binding_digest=binding.digest, nonce=nonce)
        )
        challenge = _Challenge.model_validate_json(await _hello_read(reader))
        proof = _proof_bytes(binding, nonce, challenge.nonce)
        expected = hmac.digest(
            bytes.fromhex(binding.server_secret), b"server\0" + proof, "sha256"
        )
        if not hmac.compare_digest(expected, bytes.fromhex(challenge.proof)):
            raise ValueError("native owner proof differs")
        current()
        signature = key.sign(b"skulk.native-host.core/1\0" + proof)
        await _hello_send(writer, _Proof(protocol=1, signature=signature.hex()))
        if await _hello_read(reader) != b'{"protocol":1,"ready":true}\n':
            raise ValueError("native host role was not admitted")
        current()
        return _keys(binding, proof, signature, core=True)


@final
class RequestFrame(_Contract):
    """Host-only operation; author frames never enter this channel directly."""

    protocol: Literal[1] = 1
    kind: Literal["request"] = "request"
    request_id: int = Field(
        ge=1, le=2**53 - 1, description="Increasing peer-local request sequence."
    )
    operation: Operation = Field(description="Closed host operation.")
    parameters: dict[str, JsonValue] = Field(
        description="Operation-specific admitted host metadata."
    )
    payload_size: int = Field(
        ge=0, le=MAXIMUM_PAYLOAD, description="Raw body bytes, separately budgeted."
    )


@final
class ResponseFrame(_Contract):
    """Trusted host outcome; provider strings cannot establish pre-dispatch refusal."""

    protocol: Literal[1] = 1
    kind: Literal["response"] = "response"
    request_id: int = Field(
        ge=1, le=2**53 - 1, description="Exact outstanding request."
    )
    outcome: Literal["complete", "pre_dispatch", "unknown"] = Field(
        description="Host-observed effect boundary."
    )
    parameters: dict[str, JsonValue] = Field(description="Bounded reply metadata.")
    payload_size: int = Field(
        ge=0, le=MAXIMUM_PAYLOAD, description="Raw response bytes."
    )


@final
class CancelFrame(_Contract):
    """Cancel one outstanding host request; never authorize another execution."""

    protocol: Literal[1] = 1
    kind: Literal["cancel"] = "cancel"
    request_id: int = Field(
        ge=1, le=2**53 - 1, description="Exact peer-owned request being retired."
    )
    payload_size: Literal[0] = 0


type Frame = RequestFrame | ResponseFrame | CancelFrame
_FRAME: TypeAdapter[Frame] = TypeAdapter(Annotated[Frame, Field(discriminator="kind")])


class Reservation(Protocol):
    """Capacity custody supplied by the actual owner, not a request allowance."""

    def release(self) -> None:
        """Return exactly this reservation once."""
        ...


class Budget(Protocol):
    """One event-loop-owned aggregate byte/frame budget shared across connections."""

    def reserve(self, size: int) -> Reservation:
        """Reserve before reading a body that might not fit."""
        ...


@final
class Received:
    """Owned authenticated frame/body; consumers retain its lease through delivery."""

    def __init__(self, frame: Frame, payload: bytes, reservation: Reservation) -> None:
        """Acquire only after complete session MAC and sequence admission."""
        self.frame, self.payload = frame, payload
        self._reservation = reservation
        self._released = False

    def release(self) -> None:
        """Release owned bytes once without freeing another frame's capacity."""
        if not self._released:
            self._reservation.release()
            self._released = True
            self.payload = b""


@final
class AuthenticatedStream:
    """One mutually authenticated private channel with directional replay fencing."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        keys: SessionKeys,
        budget: Budget,
    ) -> None:
        """Bind actual handshake keys and aggregate host capacity before reading."""
        self.reader, self.writer, self.keys, self.budget = reader, writer, keys, budget
        self.closed = False
        self.write_lock = asyncio.Lock()
        self._outbound = self._inbound = 0
        self._reading = False

    def poison(self) -> None:
        """Synchronously withdraw a partial, corrupt or uncertain channel."""
        self.closed = True
        self.writer.close()

    async def receive(self) -> Received:
        """Read one bounded authenticated body; a single owning pump is mandatory."""
        if self.closed or self._reading:
            self.poison()
            raise ValueError("native host channel unavailable")
        self._reading = True
        reservation: Reservation | None = None
        try:
            prefix = await self.reader.readexactly(12)
            sequence, size = (
                int.from_bytes(prefix[:8], "big"),
                int.from_bytes(prefix[8:], "big"),
            )
            if sequence != self._inbound + 1 or not 2 <= size <= MAXIMUM_METADATA:
                raise ValueError("native host sequence or metadata refused")
            metadata = await self.reader.readexactly(size)
            frame = _FRAME.validate_json(metadata)
            reservation = self.budget.reserve(frame.payload_size)
            payload = await self.reader.readexactly(frame.payload_size)
            signature = await self.reader.readexactly(32)
            authentication = hmac.new(self.keys.inbound, prefix, hashlib.sha256)
            authentication.update(metadata)
            authentication.update(payload)
            if not hmac.compare_digest(authentication.digest(), signature):
                raise ValueError("native host frame authentication refused")
            self._inbound = sequence
            return Received(frame, payload, reservation)
        except BaseException:
            if reservation is not None:
                reservation.release()
            self.poison()
            raise
        finally:
            self._reading = False

    async def send(
        self,
        frame: Frame,
        payload: bytes,
        *,
        check: Callable[[], None],
        dispatched: Callable[[], None] = lambda: None,
    ) -> None:
        """Recheck current policy after queueing, just before the physical write."""
        if len(payload) != frame.payload_size:
            raise ValueError("native host body length differs")
        metadata = frame.model_dump_json().encode()
        if len(metadata) > MAXIMUM_METADATA:
            raise ValueError("native host metadata exceeds bound")
        async with self.write_lock:
            if self.closed or self._outbound >= 2**64 - 1:
                raise ValueError("native host channel unavailable")
            check()
            sequence = self._outbound + 1
            prefix = sequence.to_bytes(8, "big") + len(metadata).to_bytes(4, "big")
            authentication = hmac.new(self.keys.outbound, prefix, hashlib.sha256)
            authentication.update(metadata)
            authentication.update(payload)
            dispatched()
            self._outbound = sequence
            try:
                self.writer.write(prefix)
                self.writer.write(metadata)
                self.writer.write(payload)
                self.writer.write(authentication.digest())
                await self.writer.drain()
            except BaseException:
                self.poison()
                raise

    async def close(self) -> None:
        """Close the actual stream after its owning peer has drained handlers."""
        self.poison()
        with contextlib.suppress(OSError):
            await self.writer.wait_closed()
