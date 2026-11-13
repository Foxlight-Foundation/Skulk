"""Exact public inference reader and signed-record projections, never authority.

Mirrored in Core and the owner. Only the published compiled implementation
admits author records. The signed runtime and live caller policy independently
select execution; package installation and structural parsing grant no rights.
"""

import hashlib
import importlib
import importlib.metadata
import json
from collections.abc import Awaitable, Callable
from functools import lru_cache
from pathlib import Path
from typing import Literal, Protocol, Self, cast, final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    model_validator,
)

INFERENCE_KERNEL_VERSION = "0.7.0rc3"
INFERENCE_KERNEL_SOURCE = "c8f869ba525d6fdcff64e90cd3c14954eaad34da"
INFERENCE_COHORT_SHA256 = (
    "da446b54ddd8996ea21cbd61f1155bc1e840ddfff2e36f435ffc973806c865ae"
)
_OBJECT = TypeAdapter(dict[str, JsonValue])
INFERENCE_PROFILE_READERS = (
    ("skulk.portable/1", "portable_profile_digest"),
    ("skulk.managed/1", "managed_profile_digest"),
    ("skulk.handshake/1", "handshake_profile_digest"),
    ("skulk.transport/1", "transport_profile_digest"),
    ("skulk.composition/1", "composition_profile_digest"),
    ("skulk.calls/1", "calls_profile_digest"),
    ("skulk.portable/2", "portable_v2_profile_digest"),
    ("skulk.composition/2", "composition_v2_profile_digest"),
    ("skulk.managed/2", "managed_v2_profile_digest"),
    ("skulk.handshake/2", "handshake_v2_profile_digest"),
    ("skulk.composition/3", "composition_v3_profile_digest"),
    ("skulk.calls/2", "calls_v2_profile_digest"),
    ("skulk.transport/2", "transport_v2_profile_digest"),
)


class ManagedRecord(Protocol):
    """Immutable native manifest/startup handle with explicit protected export."""

    @property
    def digest(self) -> str:
        """Return the complete profile/model-domain identity."""
        ...

    def canonical(self) -> bytes:
        """Export already admitted bytes only for an explicit protected use."""
        ...

    def hello(self, call_id: str | None = None) -> "HandshakeRecord":
        """Derive the exact protected startup/call-bound hello."""
        ...

    def admit_hello(self, source: str, call_id: str | None = None) -> "HandshakeRecord":
        """Match startup identity; the host separately consumes its live grant."""
        ...


class HandshakeRecord(Protocol):
    """Public native hello handle; parsing does not authenticate a caller."""

    def canonical(self) -> bytes:
        """Explicitly export protected handshake bytes."""
        ...


class ManagedFactory(Protocol):
    """The actual compiled manifest/startup constructor."""

    def from_json(self, name: str, source: str) -> ManagedRecord:
        """Admit the explicitly selected managed/2 record."""
        ...


class CompositionRecord(Protocol):
    """Native sensitive metadata; possession alone cannot activate a scope."""

    @property
    def name(self) -> str:
        """Expose only the public record kind."""
        ...

    def canonical(self) -> bytes:
        """Explicitly export canonical metadata for protected IPC."""
        ...

    def admit(
        self,
        request: "CompositionRecord",
        policy: "CompositionRecord",
        executor: str,
        request_bytes: bytes,
        now_unix_ms: int,
        scope_id: str,
    ) -> "CompositionRecord":
        """Apply pure effect/domain/deadline attenuation against host observations."""
        ...


class CompositionFactory(Protocol):
    """The compiled nullable-authority composition constructor."""

    def from_json(self, name: str, source: str) -> CompositionRecord:
        """Admit a closed composition/3 record without authorizing execution."""
        ...


class NativeStream(Protocol):
    """Compiled media lifecycle; the host separately owns active-call admission."""

    def next_input(self) -> Awaitable[tuple[str, bytes] | None]:
        """Receive one validated input frame or a completed half-close."""
        ...

    def send_output(self, projection: str, media: bytes) -> Awaitable[None]:
        """Send bounded media, holding terminal output until handler cleanup."""
        ...

    def finish(self) -> Awaitable[None]:
        """Complete cleanup and the terminal/end acknowledgment."""
        ...

    def close(self) -> Awaitable[None]:
        """Fence further work and await owned socket/task cleanup."""
        ...


class LegacyControl(Protocol):
    """Unchanged old control interface for explicitly selected legacy runtimes."""

    def receive(self) -> Awaitable[bytes]:
        """Receive an admitted legacy control body."""
        ...

    def send(self, source: str) -> Awaitable[None]:
        """Send an admitted legacy control body."""
        ...

    def close(self) -> Awaitable[None]:
        """Await legacy channel cleanup."""
        ...


class CallControl(Protocol):
    """Compiled composed-call transport owned by the author endpoint."""

    def receive(self) -> Awaitable[tuple[bytes, bytes]]:
        """Receive separate canonical metadata and exact raw body bytes."""
        ...

    def send(self, source: str, payload: bytes) -> Awaitable[None]:
        """Admit the exact metadata/body binding and write under I/O fences."""
        ...

    def close(self) -> Awaitable[None]:
        """Fence pending operations and drain compiled channel ownership."""
        ...


class Schema(Protocol):
    """Public compiled portable-schema matcher."""

    def matches(self, source: str) -> bool:
        """Validate one bounded JSON value using the compiled profile."""
        ...


class SchemaFactory(Protocol):
    """Public compiled schema constructor."""

    def __call__(self, source: str) -> Schema:
        """Compile the explicit closed portable schema language."""
        ...


class InferenceKernel(Protocol):
    """Published inference ABI, independently of owner and provider policy."""

    ManagedContract: ManagedFactory
    ManagedV2Contract: ManagedFactory
    CompositionV3Contract: CompositionFactory
    PortableSchema: SchemaFactory

    def portable_contract_json(self, name: str, source: str) -> bytes:
        """Admit an unchanged portable/1 record without changing its identity."""
        ...

    def portable_contract_digest(self, name: str, source: str) -> str:
        """Return an unchanged portable/1 record identity."""
        ...

    def portable_profile_digest(self) -> str:
        """Identify the frozen portable/1 semantics."""
        ...

    def managed_profile_digest(self) -> str:
        """Identify the frozen managed/1 semantics."""
        ...

    def handshake_profile_digest(self) -> str:
        """Identify the frozen handshake/1 semantics."""
        ...

    def transport_profile_digest(self) -> str:
        """Identify the frozen transport/1 semantics."""
        ...

    def encode_portable_control(self, source: str) -> bytes:
        """Frame only an explicitly selected old control record."""
        ...

    def decode_portable_control(self, packet: bytes) -> bytes:
        """Admit the unchanged old control framing."""
        ...

    def open_portable_control(
        self, startup: ManagedRecord, seconds: float
    ) -> Awaitable[LegacyControl]:
        """Open a frozen old startup without replacing its signed closure."""
        ...

    def portable_v2_contract_json(self, name: str, source: str) -> bytes:
        """Admit native author declarations with their explicit effect/domain."""
        ...

    def portable_v2_contract_digest(self, name: str, source: str) -> str:
        """Return a complete native author identity, never a legacy revision."""
        ...

    def composition_v3_profile_digest(self) -> str:
        """Identify nullable authority attenuation semantics."""
        ...

    def calls_v2_profile_digest(self) -> str:
        """Identify exact composed-call framing semantics."""
        ...

    def transport_v2_profile_digest(self) -> str:
        """Identify protocol-six control and binary-media semantics."""
        ...

    def call_v2_payload_size(self, source: str) -> int:
        """Return a bounded body-allocation hint; full admission remains required."""
        ...

    def encode_call_v2_frame(self, source: str, payload: bytes) -> bytes:
        """Admit complete metadata and its exact raw body binding."""
        ...

    def decode_call_v2_frame(self, packet: bytes) -> tuple[bytes, bytes]:
        """Refuse malformed/trailing frames and return owned metadata/body bytes."""
        ...

    def encode_transport_v2_packet(self, projection: str, media: bytes) -> bytes:
        """Admit and frame one bounded protocol-six binary-media record."""
        ...

    def decode_transport_v2_packet(self, packet: bytes) -> tuple[str, bytes]:
        """Return canonical projection and exact bounded media bytes."""
        ...

    def open_call_v2_control(
        self, startup: ManagedRecord, seconds: float
    ) -> Awaitable[CallControl]:
        """Derive the exact protected endpoint and hello in compiled code."""
        ...

    def open_transport_v2_stream(
        self,
        startup: ManagedRecord,
        call_id: str,
        mode: str,
        seconds: float | None = None,
    ) -> Awaitable[NativeStream]:
        """Open native media with an actual admitted budget, including explicit null."""
        ...


def inference_cohort() -> dict[str, JsonValue]:
    """Read the exact anonymously verified public manifest, with no network lookup."""
    data = Path(__file__).with_name("native-inference-cohort.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != INFERENCE_COHORT_SHA256:
        raise ValueError("native inference acquisition receipt differs")
    return _OBJECT.validate_json(data)


@lru_cache(maxsize=1)
def public_inference_kernel() -> InferenceKernel:
    """Require the published binary and all thirteen exact semantic identities.

    This permits the new reader to understand unchanged older profiles. It does
    not replace any old application's signed wheel closure or activate a caller.
    """
    try:
        if (
            importlib.metadata.version("skulk-capability-kernel")
            != INFERENCE_KERNEL_VERSION
        ):
            raise ValueError("qualified inference reader is unavailable")
        kernel = cast(
            InferenceKernel,
            cast(object, importlib.import_module("_skulk_capability_kernel")),
        )
        profiles = inference_cohort()["profiles"]
        if not isinstance(profiles, list):
            raise ValueError("native inference profile receipt refused")
        expected: dict[str, JsonValue] = {}
        for entry in profiles:
            if not isinstance(entry, dict) or not isinstance(entry.get("profile"), str):
                raise ValueError("native inference profile receipt refused")
            name = cast(str, entry["profile"])
            if name in expected:
                raise ValueError("duplicate native inference profile")
            expected[name] = entry["profile_digest"]
        if set(expected) != {name for name, _ in INFERENCE_PROFILE_READERS}:
            raise ValueError("native inference profile inventory differs")
        for name, attribute in INFERENCE_PROFILE_READERS:
            reader = cast(object, getattr(kernel, attribute))
            if (
                not callable(reader)
                or cast(Callable[[], str], reader)() != expected[name]
            ):
                raise ValueError("native inference profile differs")
    except (ImportError, AttributeError, importlib.metadata.PackageNotFoundError):
        raise ValueError("qualified inference reader is unavailable") from None
    return kernel


def inference_native_wheel(family: str) -> tuple[str, str, int]:
    """Return one exact published wheel; declaration does not authorize installation."""
    platforms = inference_cohort()["platforms"]
    if isinstance(platforms, list):
        for entry in platforms:
            if isinstance(entry, dict) and entry.get("platform") == family:
                wheel = _OBJECT.validate_python(entry["wheel"], strict=True)
                name, digest, size = wheel["name"], wheel["sha256"], wheel["size"]
                if (
                    isinstance(name, str)
                    and isinstance(digest, str)
                    and type(size) is int
                ):
                    return name, digest, size
    raise ValueError("native inference wheel is unavailable for this platform")


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


@final
class InferenceProfiles(_Contract):
    """Exact signed execution cohort, separately selected from issued runtimes."""

    version: Literal["0.7.0rc3"] = Field(
        description="Published native ABI package version."
    )
    source_commit: Literal["c8f869ba525d6fdcff64e90cd3c14954eaad34da"] = Field(
        description="Immutable public source selected by the closure."
    )
    portable: Literal[
        "7996ed84e821ddacfb48e00b99a5886948fd20e43bc2c7993c5ed98217834033"
    ] = Field(description="Complete skulk.portable/2 fingerprint.")
    managed: Literal[
        "add1362825790e109c779ae5ec5b3467710b42acebf8a2081b404fb18f73cacc"
    ] = Field(description="Complete skulk.managed/2 fingerprint.")
    handshake: Literal[
        "a621193448577ba88a492bf5a8b61166b56e72d99fda7ceaa2efbb5632ae253c"
    ] = Field(description="Complete skulk.handshake/2 fingerprint.")
    transport: Literal[
        "bfc5b4ece93e9bf0083b29ad46a69977b2d765d613494a9f0a59f4ee8ddf6769"
    ] = Field(description="Complete skulk.transport/2 fingerprint.")
    composition: Literal[
        "4097a7f03943da0bb44dc1a312976c0eb92854f016dc5a276bb8e8b85854f6f1"
    ] = Field(description="Complete skulk.composition/3 fingerprint.")
    calls: Literal[
        "81f72dc3e2a07321480af80bafede3da4a3b14ab80ea620c5c8600b51bc76101"
    ] = Field(description="Complete skulk.calls/2 fingerprint.")

    @classmethod
    @lru_cache(maxsize=1)
    def qualified(cls) -> "InferenceProfiles":
        """Supply publisher defaults; signed parsing requires every field."""
        receipt = inference_cohort()
        selected = {
            "portable": "skulk.portable/2",
            "managed": "skulk.managed/2",
            "handshake": "skulk.handshake/2",
            "transport": "skulk.transport/2",
            "composition": "skulk.composition/3",
            "calls": "skulk.calls/2",
        }
        profiles = receipt["profiles"]
        if not isinstance(profiles, list):
            raise ValueError("native inference profile receipt refused")
        values = {
            cast(str, value["profile"]): value["profile_digest"]
            for value in profiles
            if isinstance(value, dict)
        }
        return cls.model_validate(
            {
                "version": receipt["version"],
                "source_commit": receipt["source_commit"],
                **{name: values[profile] for name, profile in selected.items()},
            },
            strict=True,
        )


@final
class InferenceDescriptor(_Contract):
    """Typed view over compiled author admission; effect/domain are not consent."""

    contract_profile: Literal["skulk.portable/2"] = Field(
        description="Explicit inference author profile."
    )
    profile_digest: str = Field(description="Complete native author-profile identity.")
    id: str = Field(description="Exact portable capability identifier.")
    version: str = Field(description="Exact signed semantic version.")
    title: str = Field(description="Declared operator title.")
    description: str = Field(description="Declared capability purpose.")
    behavior: Literal["read", "inference", "proposal", "execute"] = Field(
        description="Declared effect ceiling, not caller authorization."
    )
    processing_domain: Literal["local", "private_site", "external"] = Field(
        description="Signed processing boundary requiring independent caller consent."
    )
    io_mode: Literal[
        "unary", "server_streaming", "client_streaming", "bidirectional"
    ] = Field(description="Exact declared media mode.")
    input_schema: dict[str, JsonValue] | bool = Field(
        description="Compiled portable input schema."
    )
    output_schema: dict[str, JsonValue] | bool | None = Field(
        description="Optional compiled result schema."
    )
    input_chunk_schema: dict[str, JsonValue] | bool | None = Field(
        description="Optional compiled inbound-frame schema."
    )
    output_chunk_schema: dict[str, JsonValue] | bool | None = Field(
        description="Optional compiled outbound-frame schema."
    )
    annotations: dict[str, str] | None = Field(
        description="Opaque hints, never permission or provenance."
    )

    @model_validator(mode="before")
    @classmethod
    def admitted_source(cls, source: object) -> object:
        """Use the actual compiled validator before projecting any descriptor fields."""
        value = _OBJECT.validate_python(source, strict=True)
        admitted = public_inference_kernel().portable_v2_contract_json(
            "Descriptor", json.dumps(value, allow_nan=False)
        )
        return _OBJECT.validate_json(admitted)

    @property
    def revision(self) -> str:
        """Return the full profile/model-domain identity without truncation."""
        return public_inference_kernel().portable_v2_contract_digest(
            "Descriptor", self.model_dump_json()
        )

    @property
    def qualified_id(self) -> str:
        """Return id and version without inferring caller authority."""
        return f"{self.id}@{self.version}"


@final
class InferenceManifest(_Contract):
    """Exact managed/2 manifest bytes, independent of installation authorization."""

    canonical_json: str = Field(
        repr=False,
        max_length=262144,
        description="Exact native-admitted canonical manifest bytes.",
    )

    @classmethod
    def from_json(cls, source: str) -> "InferenceManifest":
        """Normalize author input before signing, without admitting any execution."""
        record = public_inference_kernel().ManagedV2Contract.from_json(
            "Manifest", source
        )
        return cls(canonical_json=record.canonical().decode())

    @model_validator(mode="after")
    def admitted(self) -> Self:
        """Require exact canonical bytes and the selected managed/2 profile."""
        record = public_inference_kernel().ManagedV2Contract.from_json(
            "Manifest", self.canonical_json
        )
        if record.canonical() != self.canonical_json.encode():
            raise ValueError("inference manifest must carry exact canonical bytes")
        return self

    def value(self) -> dict[str, JsonValue]:
        """Return a fresh view of already-admitted public metadata."""
        return _OBJECT.validate_json(self.canonical_json)

    @property
    def digest(self) -> str:
        """Return the full managed/2 manifest identity."""
        return (
            public_inference_kernel()
            .ManagedV2Contract.from_json("Manifest", self.canonical_json)
            .digest
        )

    @property
    def bundle_id(self) -> str:
        """Return the stable signed bundle identity."""
        return _text(self.value()["bundle_id"])

    @property
    def bundle_version(self) -> str:
        """Return the signed bundle version."""
        return _text(self.value()["bundle_version"])

    @property
    def skulk_requires(self) -> str:
        """Return the signed compatible Core version range."""
        return _text(self.value()["skulk_requires"])

    @property
    def executable(self) -> str:
        """Return the signed relative executable without resolving it."""
        return _text(self.value()["executable"])

    @property
    def executable_sha256(self) -> str:
        """Return the complete signed executable content identity."""
        return _text(self.value()["executable_sha256"])

    @property
    def runtime_dependencies(self) -> tuple[str, ...]:
        """Return native-admitted dependency roots without changing their names."""
        return TypeAdapter(tuple[str, ...]).validate_json(
            json.dumps(self.value()["runtime_dependencies"])
        )

    @property
    def configuration_schema(self) -> JsonValue:
        """Return the native-admitted configuration schema, including boolean roots."""
        contracts = self.value()["contracts"]
        if not isinstance(contracts, dict):
            raise ValueError("inference manifest projection refused")
        return contracts["configuration_schema"]

    def with_artifact(self, digest: str) -> "InferenceManifest":
        """Bind normalized artifact custody before signing this new author profile."""
        value = self.value()
        value.update(executable="bundle.pyz", executable_sha256=digest)
        return self.from_json(json.dumps(value, allow_nan=False))

    @property
    def descriptors(self) -> tuple[InferenceDescriptor, ...]:
        """Project exact compiled declarations without adding authority or readiness."""
        contracts = _OBJECT.validate_python(self.value()["contracts"], strict=True)
        return tuple(
            TypeAdapter(list[InferenceDescriptor]).validate_python(
                contracts["descriptors"], strict=True
            )
        )


def _text(value: JsonValue) -> str:
    if not isinstance(value, str):
        raise ValueError("inference manifest projection refused")
    return value
