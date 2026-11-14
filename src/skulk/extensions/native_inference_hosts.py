"""Exact signed inference observations over the protected installed-owner bridge."""

import hashlib
import json
from pathlib import Path
from typing import Literal, final

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from skulk.extensions.native_calls import NativeCalls
from skulk.extensions.native_host_protocol import (
    Digest,
    Identifier,
    Profile,
    ResponseFrame,
)
from skulk.extensions.native_inference_authority import InferenceGrant
from skulk.extensions.native_inference_contracts import (
    InferenceDescriptor,
    InferenceManifest,
    host_stream_kernel,
)
from skulk.extensions.runtime_files import read_private


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


@final
class InferenceTarget(_Contract):
    """Complete composition3 destination identity, never a bearer or permission."""

    node_id: Identifier = Field(
        description="Persisted installed native child identity."
    )
    capability_id: Identifier = Field(description="Exact signed capability.")
    version: str = Field(description="Exact capability version.")
    descriptor_revision: Digest = Field(
        description="Complete compiled declaration identity."
    )
    behavior: Literal["read", "inference"] = Field(
        description="Exact signed effect ceiling."
    )
    processing_domain: Literal["local", "private_site", "external"] = Field(
        description="Explicit signed processing boundary."
    )

    @classmethod
    def from_grant(cls, grant: InferenceGrant) -> "InferenceTarget":
        """Project permission independently of capacity; dispatch still needs observation."""
        return cls(
            node_id=grant.node_id,
            capability_id=grant.capability_id,
            version=grant.version,
            descriptor_revision=grant.descriptor_revision,
            behavior=grant.behavior,
            processing_domain=grant.processing_domain,
        )


@final
class InferenceListing(_Contract):
    """Signed declaration plus protected owner/child readiness observation."""

    plugin_id: str = Field(description="Actual registered installation.")
    runtime_digest: Digest = Field(description="Selected signed runtime generation.")
    incarnation: Profile = Field(description="Actual admitted child channel lifetime.")
    target: InferenceTarget = Field(description="Exact compiled destination identity.")
    descriptor: InferenceDescriptor = Field(
        description="Exact signed native declaration."
    )
    stream_seconds: float = Field(
        default=300,
        gt=0,
        description="Signed partial-I/O and write resource watchdog; not generation lifetime.",
    )
    cleanup_seconds: float = Field(
        default=3,
        gt=0,
        description="Signed-derived cancellation and process-reap resource grace, never generation lifetime.",
    )
    startup_seconds: float = Field(
        default=5,
        gt=0,
        description="Signed connect resource window, independent of generation lifetime.",
    )
    inference_seconds: float | None = Field(
        gt=0,
        description="Optional signed owner duration ceiling; null adds no ceiling.",
    )

    def grant(self) -> InferenceGrant:
        """Describe an exact enrollment candidate; discovery itself grants nothing."""
        return InferenceGrant(
            plugin_id=self.plugin_id,
            runtime_digest=self.runtime_digest,
            node_id=self.target.node_id,
            capability_id=self.target.capability_id,
            version=self.target.version,
            descriptor_revision=self.target.descriptor_revision,
            behavior=self.target.behavior,
            processing_domain=self.target.processing_domain,
            io_mode=self.descriptor.io_mode,
        )


class _Child(_Contract):
    node_id: Identifier
    incarnation: Profile
    manifest_digest: Digest
    descriptors: tuple[InferenceDescriptor, ...] = Field(max_length=64)
    streaming: bool = Field(
        default=False,
        description="Actual protected owner stream admission availability.",
    )


class _Description(_Contract):
    children: tuple[_Child, ...] = Field(max_length=1)
    diagnostics: dict[str, JsonValue] | None = None


@final
class InferenceHosts:
    """Share bounded host observation while keeping new inference semantics explicit."""

    def __init__(self, hosts: NativeCalls) -> None:
        """Use the Core-owned connector; author code never receives this object."""
        self.hosts = hosts

    async def directory(
        self, *, streaming: bool = False
    ) -> tuple[InferenceListing, ...]:
        """Observe signed ready targets without rewriting current caller permission."""
        return await self.hosts.scan_signed_owners(
            lambda path: self._discover(path, streaming=streaming),
            lambda value: value.target.model_dump_json(),
        )

    async def _discover(
        self, path: Path, *, streaming: bool = False
    ) -> tuple[InferenceListing, ...]:
        try:
            binding, runtime, peer = await self.hosts.observed_owner(path.stem)
            manifest = runtime.native_manifest
            if (
                runtime.claims.protocol != 5
                or not isinstance(manifest, InferenceManifest)
                or runtime.digest != binding.runtime_digest
            ):
                return ()
            root = self.hosts.root / "installations" / binding.plugin_id
            identity = read_private(
                root
                / (
                    "bundle-"
                    + hashlib.sha256(manifest.bundle_id.encode()).hexdigest()[:24]
                )
                / "identity"
            ).decode()
            reply = await peer.request("describe_inference", {})
            try:
                if (
                    not isinstance(reply.frame, ResponseFrame)
                    or reply.frame.outcome != "complete"
                    or reply.payload
                ):
                    raise ValueError("inference owner description refused")
                description = _Description.model_validate_json(
                    json.dumps(reply.frame.parameters)
                )
                listings: list[InferenceListing] = []
                for child in description.children:
                    if (
                        child.node_id != identity
                        or child.manifest_digest != manifest.digest
                        or child.descriptors != manifest.descriptors
                    ):
                        raise ValueError(
                            "inference owner differs from signed installation"
                        )
                    for descriptor in child.descriptors:
                        if descriptor.behavior not in ("read", "inference") or (
                            descriptor.io_mode != "unary"
                            and (
                                not streaming
                                or not child.streaming
                                or host_stream_kernel() is None
                            )
                        ):
                            continue
                        listings.append(
                            InferenceListing(
                                plugin_id=binding.plugin_id,
                                runtime_digest=binding.runtime_digest,
                                incarnation=child.incarnation,
                                target=InferenceTarget(
                                    node_id=child.node_id,
                                    capability_id=descriptor.id,
                                    version=descriptor.version,
                                    descriptor_revision=descriptor.revision,
                                    behavior=descriptor.behavior,
                                    processing_domain=descriptor.processing_domain,
                                ),
                                descriptor=descriptor,
                                stream_seconds=TypeAdapter(float).validate_python(
                                    TypeAdapter(dict[str, JsonValue]).validate_python(
                                        manifest.value()["bounds"], strict=True
                                    )["stream_seconds"],
                                    strict=True,
                                ),
                                cleanup_seconds=max(
                                    3.0,
                                    3
                                    * TypeAdapter(float).validate_python(
                                        TypeAdapter(
                                            dict[str, JsonValue]
                                        ).validate_python(
                                            manifest.value()["bounds"], strict=True
                                        )["shutdown_seconds"],
                                        strict=True,
                                    )
                                    + TypeAdapter(float).validate_python(
                                        TypeAdapter(
                                            dict[str, JsonValue]
                                        ).validate_python(
                                            manifest.value()["bounds"], strict=True
                                        )["health_seconds"],
                                        strict=True,
                                    )
                                    + 1,
                                ),
                                startup_seconds=TypeAdapter(float).validate_python(
                                    TypeAdapter(dict[str, JsonValue]).validate_python(
                                        manifest.value()["bounds"], strict=True
                                    )["startup_seconds"],
                                    strict=True,
                                ),
                                inference_seconds=TypeAdapter[float | None](
                                    float | None
                                ).validate_python(
                                    TypeAdapter(dict[str, JsonValue]).validate_python(
                                        manifest.value()["bounds"], strict=True
                                    )["inference_seconds"],
                                    strict=True,
                                ),
                            )
                        )
                self.hosts.check_owner(binding.plugin_id, binding.runtime_digest, peer)
                return tuple(listings)
            finally:
                reply.release()
        except (OSError, ValueError, TimeoutError):
            return ()
