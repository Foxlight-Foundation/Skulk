"""Signed inference enrollment, separate from live capacity and native reads."""

import asyncio
import hashlib
import secrets
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from skulk.extensions.inference_callers import (
    AuthenticatedInferenceCaller,
    InferenceCallers,
)
from skulk.extensions.managed_limits import MAX_RUNTIME_METADATA_BYTES
from skulk.extensions.native_inference_authority import (
    InferenceAccessPolicy,
    InferenceGrant,
    PrincipalAccess,
)
from skulk.extensions.native_inference_contracts import (
    InferenceDescriptor,
    InferenceManifest,
)
from skulk.extensions.runtime_files import RuntimeLock, read_private, write_private
from skulk.extensions.runtime_install import finish_runtime_work
from skulk.extensions.runtime_selection import RuntimeSelector


class InferencePolicyUpdate(BaseModel):
    """Owner-reviewed replacement with exact original-caller permissions."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    expected_revision: str = Field(
        pattern=r"^[a-f0-9]{64}$",
        description="Current policy revision; stale updates fail.",
    )
    enabled: bool = Field(
        description="Enable explicitly enrolled inference/read effects."
    )
    callers: tuple[PrincipalAccess, ...] = Field(
        max_length=64,
        description="Per-principal exact signed destinations and optional owner ceilings.",
    )

    @field_validator("callers", mode="before")
    @classmethod
    def json_callers(cls, value: object) -> object:
        """Admit the JSON array container without weakening nested field types."""
        if isinstance(value, list):
            return tuple(
                TypeAdapter(list[PrincipalAccess]).validate_python(value, strict=True)
            )
        return value


class InferenceEnrollment:
    """Authenticate exact installed declarations before changing host permission."""

    def __init__(self, callers: InferenceCallers) -> None:
        """Use the actual host-configured manager custody, never a request path."""
        self.callers = callers

    async def _verified_grant(self, grant: InferenceGrant) -> Callable[[], None]:
        """Verify signed installation metadata without requiring runtime readiness."""
        root = self.callers.root / "installations" / grant.plugin_id
        selector = RuntimeSelector(root)
        selection = selector.current()
        if selection is None or selection.runtime_digest != grant.runtime_digest:
            raise ValueError("inference grant requires the selected signed runtime")
        metadata_path = root / "generations" / grant.runtime_digest / "staged.json"
        metadata = read_private(metadata_path, MAX_RUNTIME_METADATA_BYTES)
        trust = read_private(root / "publisher-trust.json")
        runtime = await finish_runtime_work(
            asyncio.create_task(selector.installer.inspect_metadata(metadata))
        )
        manifest = runtime.native_manifest
        if (
            runtime.claims.protocol != 5
            or not isinstance(manifest, InferenceManifest)
            or runtime.digest != grant.runtime_digest
        ):
            raise ValueError("inference grant requires a qualified inference closure")
        identity_path = (
            root
            / ("bundle-" + hashlib.sha256(manifest.bundle_id.encode()).hexdigest()[:24])
            / "identity"
        )
        node_identity = read_private(identity_path).decode()
        if node_identity != grant.node_id:
            raise ValueError("inference installed child identity differs")
        contracts = manifest.descriptors
        matches = [
            d
            for d in contracts
            if d.id == grant.capability_id and d.version == grant.version
        ]
        if len(matches) != 1:
            raise ValueError("inference signed capability differs")
        descriptor: InferenceDescriptor = matches[0]
        if (
            descriptor.revision,
            descriptor.behavior,
            descriptor.processing_domain,
            descriptor.io_mode,
        ) != (
            grant.descriptor_revision,
            grant.behavior,
            grant.processing_domain,
            grant.io_mode,
        ):
            raise ValueError("inference grant effect, domain or identity differs")

        def current() -> None:
            if (
                selector.current() != selection
                or read_private(identity_path).decode() != node_identity
                or read_private(root / "publisher-trust.json") != trust
                or read_private(metadata_path, MAX_RUNTIME_METADATA_BYTES) != metadata
            ):
                raise ValueError("inference signed enrollment changed")

        current()
        return current

    async def update(self, update: InferencePolicyUpdate) -> InferenceAccessPolicy:
        """Replace permission under the physical dispatch fence after verification.

        A disabled policy may retain stale entries for owner inspection. Enabling
        it requires valid, unexpired clients and exact currently selected signed
        targets. An offline target remains a permission; it supplies no capacity.
        """
        candidate = InferenceAccessPolicy(
            revision=secrets.token_hex(32),
            enabled=update.enabled,
            callers=update.callers,
        )
        observations: list[Callable[[], None]] = []
        if candidate.enabled:
            grants = tuple(
                dict.fromkeys(
                    grant for access in candidate.callers for grant in access.grants
                )
            )
            for grant in grants:
                observations.append(await self._verified_grant(grant))
        lock = RuntimeLock(self.callers.root, "native-policy.lock")
        try:
            if self.callers.policy().revision != update.expected_revision:
                raise ValueError("inference policy revision changed")
            if candidate.enabled:
                state = self.callers.state()
                for access in candidate.callers:
                    principal = access.principal
                    credential_revision: str | None = None
                    if principal.kind == "client":
                        matches = (
                            []
                            if state is None
                            else [
                                c
                                for c in state.clients
                                if f"client:{c.identifier}" == principal.identifier
                            ]
                        )
                        if len(matches) != 1:
                            raise ValueError("inference client is not enrolled")
                        credential_revision = matches[0].revision
                    self.callers.current(
                        AuthenticatedInferenceCaller(principal, credential_revision)
                    )
                for current in observations:
                    current()
            write_private(
                self.callers.root / "native-inference-policy.json",
                candidate.model_dump_json().encode(),
            )
            return candidate
        finally:
            lock.close()
