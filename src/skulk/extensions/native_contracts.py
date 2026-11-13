"""Core views over the qualified public native contract reader.

Author/schema admission belongs to the public compiled kernel. These views keep
its exact canonical bytes and expose only the claims needed for installation.
"""

import importlib
import importlib.metadata
import json
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

_OBJECT = TypeAdapter(dict[str, JsonValue])
KERNEL_VERSION = "0.4.0rc1"
KERNEL_SOURCE = "43b4442c1f22dd75ccc1b19d1252272ca3eefb4b"


class NativeRecord(Protocol):
    """Immutable public managed record with explicit byte export."""

    @property
    def digest(self) -> str:
        """Return the complete native profile/model-domain identity."""
        ...

    def canonical(self) -> bytes:
        """Export bytes already admitted by the compiled implementation."""
        ...


class NativeFactory(Protocol):
    """The compiled factory needed by Core, with no Python fallback."""

    def from_json(self, name: str, source: str) -> NativeRecord:
        """Admit one managed model using the qualified public implementation."""
        ...


class NativeKernel(Protocol):
    """Public binary semantics consumed by native runtime installation."""

    ManagedContract: NativeFactory

    def portable_contract_json(self, name: str, source: str) -> bytes:
        """Admit one public portable model through its compiled factory."""
        ...

    def portable_contract_digest(self, name: str, source: str) -> str:
        """Return the complete portable profile/model-domain identity."""
        ...

    def portable_profile_digest(self) -> str:
        """Return the portable author contract fingerprint."""
        ...

    def managed_profile_digest(self) -> str:
        """Return the managed record fingerprint."""
        ...

    def handshake_profile_digest(self) -> str:
        """Return the protected hello fingerprint."""
        ...

    def transport_profile_digest(self) -> str:
        """Return the control/media transport fingerprint."""
        ...


@final
class NativeProfiles(BaseModel):
    """Exact source and semantic identity of the publicly qualified native cohort."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    version: Literal["0.4.0rc1"] = Field(
        description="Qualified native package version."
    )
    source_commit: Literal["43b4442c1f22dd75ccc1b19d1252272ca3eefb4b"] = Field(
        description="Immutable public native source revision."
    )
    portable: Literal[
        "ada52fb17016f873140aa4d6cd7ae2f8c9a0895eed997a92201ed5f45a60e3d3"
    ] = Field(description="Complete skulk.portable/1 normative fingerprint.")
    managed: Literal[
        "a70a5fdf56aa24775bdce26dc4ccecc4fc50bd2d3acbd78b5b8b47f4599fe4a1"
    ] = Field(description="Complete skulk.managed/1 normative fingerprint.")
    handshake: Literal[
        "57413947cae8d31de25b3283dc1ffa87d9c9c28f0d98b5e8eb593c9dd64e07a0"
    ] = Field(description="Complete skulk.handshake/1 normative fingerprint.")
    transport: Literal[
        "18e4096b5802dc4ff9c251b1819a0c5062d42713edd4e8fc30fa94224044b435"
    ] = Field(description="Complete skulk.transport/1 normative fingerprint.")


def cohort() -> dict[str, JsonValue]:
    """Read the public release's exact checked-in binary acquisition receipt."""
    value = _OBJECT.validate_json(
        Path(__file__).with_name("native-cohort.json").read_bytes()
    )
    if value["version"] != KERNEL_VERSION or value["source_commit"] != KERNEL_SOURCE:
        raise ValueError("native cohort identity refused")
    return value


@lru_cache(maxsize=1)
def public_native_kernel() -> NativeKernel:
    """Load a qualified reader for frozen native semantics without swapping closures.

    The new binary repeats the unchanged four profiles. Its ability to read old
    records does not change the exact old wheel selected by a signed runtime.
    """
    from skulk.extensions.native_call_contracts import (
        CALL_KERNEL_VERSION,
        CallKernel,
        call_cohort,
    )

    try:
        version = importlib.metadata.version("skulk-capability-kernel")
        if version not in (KERNEL_VERSION, CALL_KERNEL_VERSION):
            raise ValueError("native kernel version refused")
        kernel = cast(
            NativeKernel,
            cast(object, importlib.import_module("_skulk_capability_kernel")),
        )
        actual: tuple[str, ...] = (
            kernel.portable_profile_digest(),
            kernel.managed_profile_digest(),
            kernel.handshake_profile_digest(),
            kernel.transport_profile_digest(),
        )
        if version == CALL_KERNEL_VERSION:
            selected = cast(CallKernel, kernel)
            actual += (
                selected.composition_profile_digest(),
                selected.calls_profile_digest(),
            )
            profiles = call_cohort()["profiles"]
        else:
            profiles = cohort()["profiles"]
        if (
            not isinstance(profiles, list)
            or tuple(
                item.get("profile_digest")
                for item in profiles
                if isinstance(item, dict)
            )
            != actual
        ):
            raise ValueError("native kernel profile refused")
    except (ImportError, AttributeError, importlib.metadata.PackageNotFoundError):
        raise ValueError("qualified native reader is unavailable") from None
    return kernel


def _text(value: JsonValue) -> str:
    if not isinstance(value, str):
        raise ValueError("native claim projection refused")
    return value


@final
class NativeDescriptor(BaseModel):
    """Typed projection of a compiled native descriptor, never a Python validator.

    Native admission supplies every canonical default before field projection.
    This installed-contract view does not establish live executable capacity.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    contract_profile: Literal["skulk.portable/1"] = Field(
        description="Explicit native descriptor profile."
    )
    profile_digest: str = Field(description="Complete compiled profile identity.")
    id: str = Field(description="Opaque native capability identifier.")
    version: str = Field(description="Native semantic version.")
    title: str = Field(description="Declared operator title.")
    description: str = Field(description="Declared capability purpose.")
    behavior: Literal["read", "proposal", "execute"] = Field(
        description="Declared behavior, never authorization."
    )
    io_mode: Literal[
        "unary", "server_streaming", "client_streaming", "bidirectional"
    ] = Field(description="Declared media contract.")
    input_schema: dict[str, JsonValue] | bool = Field(
        description="Compiled native input schema, including boolean roots."
    )
    output_schema: dict[str, JsonValue] | bool | None = Field(
        description="Compiled native result schema."
    )
    input_chunk_schema: dict[str, JsonValue] | bool | None = Field(
        description="Compiled inbound media-frame schema."
    )
    output_chunk_schema: dict[str, JsonValue] | bool | None = Field(
        description="Compiled outbound media-frame schema."
    )
    annotations: dict[str, str] | None = Field(description="Opaque declared hints.")

    @model_validator(mode="before")
    @classmethod
    def admitted_source(cls, source: object) -> object:
        """Use the native authority before creating the typed view."""
        value = _OBJECT.validate_python(source, strict=True)
        admitted = public_native_kernel().portable_contract_json(
            "Descriptor", json.dumps(value, allow_nan=False)
        )
        return _OBJECT.validate_json(admitted)

    @property
    def qualified_id(self) -> str:
        """Return id and version without splitting an internal @ in the id."""
        return f"{self.id}@{self.version}"

    @property
    def revision(self) -> str:
        """Return the full native descriptor identity without truncation."""
        return public_native_kernel().portable_contract_digest(
            "Descriptor", self.model_dump_json()
        )


@final
class NativeManifest(BaseModel):
    """Exact native manifest bytes carried by explicitly native runtime envelopes."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    canonical_json: str = Field(
        repr=False,
        max_length=262144,
        description="Exact canonical manifest bytes admitted by the public kernel.",
    )

    @model_validator(mode="after")
    def admitted(self) -> Self:
        """Refuse noncanonical text or any record the compiled reader rejects."""
        record = public_native_kernel().ManagedContract.from_json(
            "Manifest", self.canonical_json
        )
        if record.canonical() != self.canonical_json.encode():
            raise ValueError("native manifest must carry exact canonical bytes")
        return self

    def value(self) -> dict[str, JsonValue]:
        """Return a fresh view without mutating the signed native record."""
        return _OBJECT.validate_json(self.canonical_json)

    @property
    def digest(self) -> str:
        """Return the untruncated native manifest identity."""
        return (
            public_native_kernel()
            .ManagedContract.from_json("Manifest", self.canonical_json)
            .digest
        )

    @property
    def bundle_id(self) -> str:
        """Return the stable bundle identity from the admitted record."""
        return _text(self.value()["bundle_id"])

    @property
    def bundle_version(self) -> str:
        """Return the admitted bundle version."""
        return _text(self.value()["bundle_version"])

    @property
    def skulk_requires(self) -> str:
        """Return the signed Core compatibility range."""
        return _text(self.value()["skulk_requires"])

    @property
    def executable(self) -> str:
        """Return the signed relative executable without resolving it in place."""
        return _text(self.value()["executable"])

    @property
    def executable_sha256(self) -> str:
        """Return the signed executable content identity."""
        return _text(self.value()["executable_sha256"])

    @property
    def configuration_schema(self) -> JsonValue:
        """Return the compiled schema, including a boolean root when declared."""
        contracts = self.value()["contracts"]
        if not isinstance(contracts, dict):
            raise ValueError("native contract projection refused")
        return contracts["configuration_schema"]


def native_wheel(family: str) -> tuple[str, str, int]:
    """Select the public cohort's exact wheel bytes for this artifact family."""
    platforms = cohort()["platforms"]
    if isinstance(platforms, list):
        for item in platforms:
            if isinstance(item, dict) and item.get("platform") == family:
                wheel = item.get("wheel")
                if isinstance(wheel, dict):
                    size = wheel.get("size")
                    if type(size) is int:
                        return _text(wheel["name"]), _text(wheel["sha256"]), size
    raise ValueError("qualified native binary is unavailable for this artifact family")
