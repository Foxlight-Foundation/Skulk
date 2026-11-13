"""Explicit composed-call cohort identity; issued native envelopes stay unchanged."""

import importlib.metadata
from pathlib import Path
from typing import Literal, Protocol, cast, final

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from .native_contracts import NativeKernel, public_native_kernel

_OBJECT = TypeAdapter(dict[str, JsonValue])
CALL_KERNEL_VERSION = "0.6.0rc1"
CALL_KERNEL_SOURCE = "cca3dcea5ecce5e282e0afc2d205fbd586ba0add"


@final
class CallProfiles(BaseModel):
    """Exact six-profile identity; it is not an authorization grant."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    version: Literal["0.6.0rc1"] = Field(
        description="Qualified composed-call native package version."
    )
    source_commit: Literal["cca3dcea5ecce5e282e0afc2d205fbd586ba0add"] = Field(
        description="Immutable public source for the selected call cohort."
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
    composition: Literal[
        "6b3b7ce596b414b40d7f74d8fe7351cb3ad93a2ecdc5c0e7623ece86b2c252d3"
    ] = Field(description="Complete skulk.composition/1 normative fingerprint.")
    calls: Literal[
        "8a2cfb90e8c986bdeb8fdc7b4dcc223740ad5b8e3c710e8e334a0a7e31e7e002"
    ] = Field(description="Complete skulk.calls/1 normative fingerprint.")


class CompositionRecord(Protocol):
    """Immutable compiled composition metadata, never caller authority."""

    def canonical(self) -> bytes:
        """Export only already admitted public metadata."""
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
        """Apply the public compiled attenuation and ancestry rules."""
        ...


class CompositionFactory(Protocol):
    """Actual public native composition factory; no Python fallback."""

    def from_json(self, name: str, source: str) -> CompositionRecord:
        """Admit one closed record using the qualified public kernel."""
        ...


class Schema(Protocol):
    """Compiled portable JSON schema, owned by the public SDK."""

    def matches(self, source: str) -> bool:
        """Check complete bounded JSON through the public compiled validator."""
        ...


class SchemaFactory(Protocol):
    """Public compiled schema constructor."""

    def __call__(self, source: str) -> Schema:
        """Compile the supplied native author schema."""
        ...


class CallKernel(NativeKernel, Protocol):
    """Compatible public binary with the explicit composed-call semantics."""

    CompositionContract: CompositionFactory
    PortableSchema: SchemaFactory

    def composition_profile_digest(self) -> str:
        """Identify the pure public caller-composition profile."""
        ...

    def calls_profile_digest(self) -> str:
        """Identify the independent composed-call control profile."""
        ...


def call_cohort() -> dict[str, JsonValue]:
    """Read the reviewed public acquisition receipt without network resolution."""
    value = _OBJECT.validate_json(
        Path(__file__).with_name("native-call-cohort.json").read_bytes()
    )
    if (
        value.get("version") != CALL_KERNEL_VERSION
        or value.get("source_commit") != CALL_KERNEL_SOURCE
        or value.get("source_tree") != "d39a5a4d7e618aaef576539610aaaa9490a7af5c"
    ):
        raise ValueError("native call cohort identity refused")
    return value


def call_native_wheel(family: str) -> tuple[str, str, int]:
    """Select the exact published binary for the explicit call-runtime family."""
    platforms = call_cohort()["platforms"]
    if not isinstance(platforms, list):
        raise ValueError("native call cohort platforms refused")
    for entry in platforms:
        if not isinstance(entry, dict) or entry.get("platform") != family:
            continue
        wheel = _OBJECT.validate_python(entry["wheel"], strict=True)
        name, digest, size = wheel["name"], wheel["sha256"], wheel["size"]
        if (
            not isinstance(name, str)
            or not isinstance(digest, str)
            or type(size) is not int
        ):
            raise ValueError("native call wheel identity refused")
        return name, digest, size
    raise ValueError("native call wheel is unavailable for this platform")


def public_call_kernel() -> CallKernel:
    """Require the selected call ABI; an old loaded reader never downgrades it."""
    try:
        version = importlib.metadata.version("skulk-capability-kernel")
    except importlib.metadata.PackageNotFoundError:
        raise ValueError("qualified native call reader is unavailable") from None
    if version != CALL_KERNEL_VERSION:
        raise ValueError("qualified native call reader is unavailable")
    return cast(CallKernel, public_native_kernel())
