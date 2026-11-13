"""Host-owned inference authority records, independent of native author parsing.

Mirrored in the owner. Signed installation metadata establishes what an app is;
these separate records establish which original caller may use it. Neither an
owner transport proof nor discovery supplies an entry in this policy.
"""

import hashlib
import json
from typing import Annotated, Literal, Self, final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9._:@-]{1,128}$")]


class _Contract(BaseModel):
    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", allow_inf_nan=False
    )


@final
class InferencePrincipal(_Contract):
    """Original caller established by Core authentication, never author input."""

    identifier: str = Field(description="Fixed owner realm or server-issued client ID.")
    kind: Literal["operator", "client"] = Field(
        description="Authenticated caller category."
    )
    issuer_node_id: Identifier = Field(
        description="Actual authenticating Core identity."
    )

    @model_validator(mode="after")
    def supported_identity(self) -> Self:
        """Keep the direct owner and enrolled clients in disjoint namespaces."""
        import re

        if self.kind == "operator":
            if self.identifier != "direct-owner":
                raise ValueError("unqualified inference operator identity")
        elif re.fullmatch(r"client:[a-f0-9]{32}", self.identifier) is None:
            raise ValueError("invalid inference client identity")
        return self


@final
class InferenceGrant(_Contract):
    """One exact signed implementation and its declared effect/domain/mode."""

    plugin_id: str = Field(
        pattern=r"^managed\.[a-z0-9][a-z0-9._-]{0,80}$",
        description="Registered installation.",
    )
    runtime_digest: Digest = Field(
        description="Exact enrolled signed runtime; updates require regrant."
    )
    node_id: Identifier = Field(
        description="Persisted installed child identity, independent of readiness."
    )
    capability_id: Identifier = Field(description="Exact signed capability.")
    version: str = Field(
        pattern=r"^\d+\.\d+\.\d+$", description="Exact capability version."
    )
    descriptor_revision: Digest = Field(
        description="Complete compiled declaration identity."
    )
    behavior: Literal["read", "inference"] = Field(
        description="Explicit effect; no write or execution grant."
    )
    processing_domain: Literal["local", "private_site", "external"] = Field(
        description="Owner-approved data-processing boundary."
    )
    io_mode: Literal[
        "unary", "server_streaming", "client_streaming", "bidirectional"
    ] = Field(description="Signed input/output lifecycle.")


@final
class PrincipalAccess(_Contract):
    """Least authority for one authenticated caller, independent of readiness."""

    principal: InferencePrincipal = Field(
        description="Original principal, including its issuing Core."
    )
    grants: tuple[InferenceGrant, ...] = Field(
        max_length=64, description="Exact permitted implementations."
    )
    maximum_seconds: float | None = Field(
        default=None,
        gt=0,
        description="Optional owner duration ceiling; null adds no ceiling.",
    )

    @field_validator("grants", mode="before")
    @classmethod
    def json_grants(cls, value: object) -> object:
        """Admit a JSON array container while keeping each grant strictly typed."""
        if isinstance(value, list):
            return tuple(
                TypeAdapter(list[InferenceGrant]).validate_python(value, strict=True)
            )
        return value

    @model_validator(mode="after")
    def unambiguous(self) -> Self:
        """Refuse competing runtime identities for the same capability/version."""
        identities = [(g.plugin_id, g.capability_id, g.version) for g in self.grants]
        if len(identities) != len(set(identities)):
            raise ValueError("ambiguous inference grants")
        return self


@final
class InferenceAccessPolicy(_Contract):
    """Atomic host permission snapshot; absent policy admits no effects."""

    protocol: Literal[1] = Field(default=1, description="Explicit host policy version.")
    revision: Digest = Field(description="Opaque compare-and-swap revision.")
    enabled: bool = Field(
        description="Whether explicitly enrolled callers may execute."
    )
    callers: tuple[PrincipalAccess, ...] = Field(
        max_length=64,
        description="Per-caller permissions, never inferred from provider login.",
    )

    @model_validator(mode="after")
    def bounded_unique_callers(self) -> Self:
        """Keep the durable record readable and prevent ambiguous caller policy."""
        identities = [
            (c.principal.identifier, c.principal.kind, c.principal.issuer_node_id)
            for c in self.callers
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate inference caller")
        if (
            sum(len(c.grants) for c in self.callers) > 64
            or len(self.model_dump_json().encode()) > 131072
        ):
            raise ValueError("inference policy exceeds bound")
        return self


@final
class InferenceCredential(_Contract):
    """One enrolled client; only a keyed digest of its bearer is retained."""

    identifier: str = Field(
        pattern=r"^[a-f0-9]{32}$", description="Server-issued client identity."
    )
    revision: Digest = Field(
        description="Fresh credential lifetime, never reused by reenrollment."
    )
    label: str = Field(
        min_length=1, max_length=128, description="Owner-assigned client label."
    )
    token_digest: Digest = Field(
        repr=False,
        description="Keyed bearer digest; never returned by management APIs.",
    )
    expires_unix_ms: int | None = Field(
        default=None,
        ge=1,
        description="Optional credential expiry; null has no scheduled expiry.",
    )


@final
class InferenceCredentialState(_Contract):
    """Private Core custody, kept outside capability backups and migrations."""

    protocol: Literal[1] = Field(
        default=1, description="Explicit credential-store version."
    )
    profile_id: str = Field(
        pattern=r"^[a-f0-9]{32}$", description="Actual manager service profile."
    )
    issuer_node_id: Identifier = Field(description="Authenticating Core identity.")
    revision: Digest = Field(
        description="Credential-inventory revision for owner mutations."
    )
    digest_key: Digest = Field(
        repr=False,
        description="Private digest seed; not a provider or gateway credential.",
    )
    clients: tuple[InferenceCredential, ...] = Field(
        max_length=256,
        description="Active client credentials; revocation removes the exact lifetime.",
    )

    @model_validator(mode="after")
    def bounded_unique_clients(self) -> Self:
        """Refuse ambiguous identity or an unreadable protected record."""
        if len({c.identifier for c in self.clients}) != len(self.clients):
            raise ValueError("duplicate inference client")
        if len(self.model_dump_json().encode()) > 131072:
            raise ValueError("inference credential store exceeds bound")
        return self


def inference_access(
    policy: InferenceAccessPolicy, principal: InferencePrincipal, grant: InferenceGrant
) -> PrincipalAccess:
    """Require current explicit permission for this exact signed destination."""
    matches = [access for access in policy.callers if access.principal == principal]
    if not policy.enabled or len(matches) != 1 or grant not in matches[0].grants:
        raise ValueError("inference caller or destination is not enrolled")
    return matches[0]


def inference_revision(
    policy: InferenceAccessPolicy,
    principal: InferencePrincipal,
    credential_revision: str | None,
) -> str:
    """Fence a context to both current policy and its authenticated credential.

    A global policy change or a client re-enrollment yields a different context
    identity. The direct owner has no client credential generation. This digest
    is an identity fence, not a bearer proof or an execution grant.
    """
    source = json.dumps(
        {
            "policy": policy.revision,
            "principal": principal.model_dump(mode="json"),
            "credential": credential_revision,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(b"skulk.inference.authority/1\0" + source).hexdigest()
