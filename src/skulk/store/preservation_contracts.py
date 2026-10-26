"""Bounded operator controls for host-owned model preservation jobs."""

from __future__ import annotations

import json
from typing import Annotated, Literal, final

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    model_validator,
)


def bounded_model_details(value: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Keep optional read-only descriptive excerpts within the control budget."""
    if len(json.dumps(value, separators=(",", ":"))) > 1536:
        raise ValueError("model description exceeds the metadata budget")
    return value


ModelDetails = Annotated[dict[str, JsonValue], AfterValidator(bounded_model_details)]

Handle = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$")]
OperationId = Annotated[
    str, Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9._:@-]+$")
]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Fingerprint = Annotated[str, Field(pattern=r"^ed25519:[0-9a-f]{64}$")]
JobKind = Literal[
    "preserve",
    "check",
    "kit_export",
    "recover",
    "publish",
    "torrent_export",
    "http_share",
    "torrent_share",
    "peer_share",
    "mirror_import",
    "receive",
]
JobState = Literal[
    "queued", "running", "cancelling", "succeeded", "failed", "cancelled", "interrupted"
]


def omit_peer_default(value: object) -> bool:
    """Keep additive peer defaults out of HTTP v1 responses and replay digests."""
    return (
        value is None or value is False or value == "http" or value == 0 or value == ()
    )


class PreservationContract(BaseModel):
    """Immutable strict local-control contract; never a filesystem authority grant."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


@final
class PreservationLocation(PreservationContract):
    """An enrolled location without its host path or signing-key material."""

    location_id: Handle = Field(description="Opaque host-enrolled storage identity.")
    label: str = Field(
        min_length=1, max_length=80, description="Owner-chosen storage label."
    )
    purpose: Literal["private", "public"] = Field(
        description="Private captures or separately published public copies."
    )
    reachable: bool = Field(
        description="Whether a bounded observation confirmed the enrolled directory; false also covers a pending or busy filesystem check."
    )
    free_bytes: int | None = Field(
        default=None,
        ge=0,
        description="Observed available bytes, or unknown when offline.",
    )
    publisher_key_id: str = Field(
        max_length=80,
        description="Public fingerprint of this location's capture or publishing identity.",
    )


@final
class FolderReviewRequest(PreservationContract):
    """Explicit trusted-operator setup input, never a model/copy authority."""

    path: str = Field(
        min_length=1,
        max_length=4096,
        description="Absolute folder path on the Core host, kept only in private setup state.",
    )
    label: str = Field(
        min_length=1,
        max_length=80,
        description="Human storage name, without credentials.",
    )


@final
class FolderReview(PreservationContract):
    """Expiring observation of one destination; review writes no archive bytes."""

    review_id: Digest = Field(description="Exact private setup observation to confirm.")
    label: str = Field(max_length=80, description="Reviewed human destination name.")
    creates_folder: bool = Field(
        description="Confirmation will create one private leaf beneath an existing parent."
    )
    free_bytes: int = Field(
        ge=0,
        description="Observed filesystem space, not a reservation or quota guarantee.",
    )
    expires_at: int = Field(ge=0, description="Review expiry in Unix seconds.")


@final
class FolderConfirmation(PreservationContract):
    """Confirm only the exact reviewed storage setup, without a new path."""

    review_id: Digest = Field(description="Exact reviewed destination observation.")
    confirmed: bool = Field(
        description="Explicit consent to private folder and local capture identity setup."
    )


@final
class DefaultLocationRequest(PreservationContract):
    """Select enrolled private storage for subsequent new-copy reviews."""

    location_id: Handle = Field(
        description="Existing private location; no path or automatic fallback."
    )


@final
class PreservationModel(PreservationContract):
    """One exact installed base generation selectable for preservation."""

    model_id: str = Field(
        min_length=1, max_length=512, description="Installed model alias."
    )
    generation_id: Digest = Field(
        description="Digest of the observed exact installed record."
    )
    size_bytes: int = Field(
        ge=0, description="Observed base artifact bytes; plans expand companions."
    )
    verification: str = Field(
        max_length=64,
        description="Existing installed verification class, not a portable proof.",
    )
    details: ModelDetails | None = Field(
        default=None,
        description="Optional bounded installed-card claims for browsing, never byte, registry or runtime authority.",
    )


@final
class PreservationCopy(PreservationContract):
    """A completed copy and independent current availability/check observations."""

    copy_id: Handle = Field(description="Opaque retained-copy handle.")
    publisher_key_ids: tuple[str, ...] = Field(
        default=(),
        max_length=32,
        description="Exact public-copy signature fingerprints; not registry authority.",
    )
    model_id: str = Field(
        min_length=1, max_length=512, description="Exact selected model alias."
    )
    location_id: Handle = Field(description="Enrolled location holding this copy.")
    manifest_id: Digest = Field(
        description="Exact signed preservation manifest identity."
    )
    audience: Literal["private", "public"] = Field(
        description="Signed copy audience; not authentication or encryption."
    )
    size_bytes: int = Field(ge=0, description="Declared complete closure bytes.")
    created_at: int = Field(ge=0, description="Host completion time as Unix seconds.")
    checked_at: int = Field(
        ge=0, description="Last successful actual file verification as Unix seconds."
    )
    member_count: int = Field(ge=1, description="Number of required retained members.")


@final
class PublicationMember(PreservationContract):
    """One retained member's public-facing terms hints; never a rights grant."""

    member_id: str = Field(max_length=64, description="Exact signed member selection.")
    role: str = Field(max_length=100, description="Declared closure role.")
    repository: str = Field(max_length=512, description="Original repository identity.")
    revision: str = Field(max_length=40, description="Exact original revision.")
    format: str = Field(max_length=128, description="Declared file format.")
    size_bytes: int = Field(ge=0, description="Declared selected member bytes.")
    file_count: int = Field(ge=0, description="Declared selected file count.")
    license_id: str = Field(max_length=200, description="Existing terms hint only.")
    terms_reference: str = Field(
        max_length=2048, description="Existing terms reference."
    )
    notice_paths: tuple[str, ...] = Field(
        max_length=100, description="Original required public notices."
    )


@final
class PublicationSource(PreservationContract):
    """Metadata-only source for an exact private copy's publication review."""

    copy_id: Handle = Field(description="Selected kept private copy.")
    manifest_id: Digest = Field(description="Authenticated source identity.")
    name: str = Field(max_length=200, description="Existing public-name suggestion.")
    workloads: tuple[str, ...] = Field(
        max_length=100, description="Declared workloads."
    )
    runtime_requirements: tuple[str, ...] = Field(
        max_length=100, description="Declared runtime requirements."
    )
    members: tuple[PublicationMember, ...] = Field(
        max_length=16,
        description="Every supported retained member; large selections refuse.",
    )


@final
class PublicCopyReview(PreservationContract):
    """Authenticated public-copy terms; no byte check or redistribution grant."""

    schema_version: Literal["skulk.preservation-public-review.v1"] = Field(
        default="skulk.preservation-public-review.v1",
        description="Versioned all-member offline review contract.",
    )
    copy_id: Handle = Field(description="Exact accepted public-copy handle.")
    manifest_id: Digest = Field(description="Exact enrolled signed archive identity.")
    archive_format: Literal["preservation-v1", "pmf-v0.1"] = Field(
        description="Original archive format; review does not convert the archive.",
    )
    name: str = Field(max_length=200, description="Existing signed display name.")
    size_bytes: int = Field(ge=0, description="Declared complete member file bytes.")
    workloads: tuple[str, ...] = Field(
        max_length=100, description="Declared workloads, without runtime parsing."
    )
    runtime_requirements: tuple[str, ...] = Field(
        max_length=100, description="Declared runtime requirements."
    )
    members: tuple[PublicationMember, ...] = Field(
        min_length=1,
        max_length=256,
        description="Every signed member and all its notice paths; oversized reviews refuse rather than omit members.",
    )


@final
class InvitationRequest(PreservationContract):
    """Explicit mirror hints for a portable public-copy handoff; no network effect."""

    mirrors: tuple[Annotated[str, Field(max_length=2048)], ...] = Field(
        strict=False,  # FastAPI supplies decoded JSON lists; string items stay strict.
        min_length=1,
        max_length=16,
        description="Credential-free HTTP(S) bases, independently validated by the invitation writer.",
    )


@final
class InvitationPreview(PreservationContract):
    """Candidate-signed offline metadata; no independent owner or registry trust."""

    receiving: ReceivingPolicy | None = Field(
        default=None,
        exclude_if=omit_peer_default,
        description="Read-only original local policy on a retained receive preview; never imported authority or renewed consent.",
    )
    manifest_id: Digest = Field(description="Exact included signed identity.")
    name: str = Field(max_length=200, description="Signed display name.")
    size_bytes: int = Field(ge=0, description="Complete declared closure bytes.")
    member_count: int = Field(
        ge=1, le=256, description="Complete selected member count."
    )
    members: tuple[str, ...] = Field(
        max_length=256,
        description="All selected member IDs and roles, without runtime parsing.",
    )
    publisher_key_ids: tuple[str, ...] = Field(
        max_length=32,
        description="Candidate fingerprints; not independently authenticated.",
    )
    threshold: int = Field(
        ge=1,
        le=32,
        description="Sender-supplied threshold, also requiring independent authentication.",
    )
    transport: Literal["http", "peer"] = Field(
        default="http",
        exclude_if=omit_peer_default,
        description="Unsigned supplier transport, never model trust.",
    )
    mirrors: tuple[str, ...] = Field(
        max_length=16, description="Unsigned transport hints; reading contacts none."
    )
    contacts: tuple[str, ...] = Field(
        default=(),
        max_length=8,
        exclude_if=omit_peer_default,
        description="Untrusted numeric starting hints; review locally before saving or connecting.",
    )


@final
class HandoffRequest(PreservationContract):
    """Export one public copy with HTTP hints or one owned peer sharing receipt."""

    mirrors: tuple[Annotated[str, Field(max_length=2048)], ...] = Field(
        default=(),
        strict=False,
        max_length=16,
        description="Explicit HTTP(S) mirror hints, exclusive with sharing_job_id.",
    )
    sharing_job_id: Digest | None = Field(
        default=None,
        description="Exact owned peer sharing job for this copy; no caller paths or contact inventory.",
    )


@final
class I2PProxySettings(PreservationContract):
    """Explicit local I2P HTTP proxy, without a router or anonymity assertion."""

    host: Literal["127.0.0.1", "::1"] = Field(
        default="127.0.0.1",
        description="Numeric loopback proxy on the Core host; never DNS or a remote proxy.",
    )
    port: int = Field(
        default=4444,
        ge=1,
        le=65535,
        description="Explicit local HTTP proxy port; a listener alone does not authenticate its software.",
    )


@final
class ReceivingPolicy(PreservationContract):
    """Finite direct-receiving policy retained unchanged across explicit resumes."""

    transport: Literal["http", "peer", "i2p"] = Field(
        default="http",
        exclude_if=omit_peer_default,
        description="Explicit selected receiving transport.",
    )
    allow_peer_network: bool = Field(
        default=False,
        exclude_if=omit_peer_default,
        description="Explicit public peer participation for this copy.",
    )
    peer_private: bool = Field(
        default=False,
        exclude_if=omit_peer_default,
        description="Fixed private-address peer scope; independent of HTTP flags.",
    )
    allow_i2p: bool = Field(
        default=False,
        exclude_if=omit_peer_default,
        description="Explicit receive-only I2P application participation; grants no payload seeding.",
    )
    i2p_proxy: I2PProxySettings | None = Field(
        default=None,
        exclude_if=omit_peer_default,
        description="Exact immutable proxy selection for this receiving operation.",
    )
    mirrors: tuple[str, ...] = Field(
        min_length=1, max_length=16, description="Exact reviewed HTTP(S) bases."
    )
    max_bytes: int = Field(
        ge=1,
        le=2**53 - 1,
        description="Cumulative object-byte allowance, including failed attempts.",
    )
    deadline_seconds: int = Field(
        ge=1,
        le=86400,
        description="Finite maximum time for one explicitly authorized attempt.",
    )
    allow_http: bool = Field(
        default=False, description="Separate deliberate permission for plaintext HTTP."
    )
    allow_private: bool = Field(
        default=False,
        description="Separate deliberate permission for non-global addresses.",
    )

    @model_validator(mode="after")
    def check_i2p(self) -> "ReceivingPolicy":
        """Keep protected proxy routing separate from direct and peer permissions."""
        if self.transport == "i2p":
            if (
                not self.allow_i2p
                or self.i2p_proxy is None
                or self.allow_http
                or self.allow_private
                or self.allow_peer_network
                or self.peer_private
            ):
                raise ValueError(
                    "I2P receiving requires only its exact proxy and consent"
                )
            from foxlight_model_preservation.i2p_http import i2p_http_endpoint

            for mirror in self.mirrors:
                i2p_http_endpoint(mirror)
        elif self.allow_i2p or self.i2p_proxy is not None:
            raise ValueError("Other receiving transports cannot carry I2P grants")
        return self


@final
class PreservationPlanRequest(PreservationContract):
    """Select exact opaque resources; callers never provide host/key paths."""

    allow_peer_network: bool = Field(
        default=False,
        exclude_if=omit_peer_default,
        description="Permit finite peer receiving after reviewing Core network exposure.",
    )
    peer_setup_id: Digest | None = Field(
        default=None,
        exclude_if=omit_peer_default,
        description="Exact currently reviewed Core peer settings for this attempt.",
    )
    allow_i2p: bool = Field(
        default=False,
        exclude_if=omit_peer_default,
        description="Permit this exact finite I2P receive after review; starts no router or sharing.",
    )
    i2p_proxy: I2PProxySettings | None = Field(
        default=None,
        exclude_if=omit_peer_default,
        description="Selected Core-local proxy; independent from model, registry and code authority.",
    )
    kind: JobKind = Field(description="Requested bounded effect.")
    resume_job_id: Digest | None = Field(
        default=None,
        description="Stopped receive job whose original exact inputs and remaining budget are reviewed again.",
    )
    accepted_key_ids: tuple[Fingerprint, ...] = Field(
        strict=False,  # Accept the JSON array container without coercing fingerprints.
        default=(),
        max_length=32,
        description="Independently authenticated invitation fingerprints selected for this copy only.",
    )
    download_limit_bytes: int = Field(
        default=20 * 1024**3,
        ge=1,
        le=2**53 - 1,
        description="Finite receiving object-byte allowance.",
    )
    download_deadline_seconds: int = Field(
        default=3600,
        ge=1,
        le=86400,
        description="Finite deadline for one deliberate download attempt.",
    )
    allow_http: bool = Field(
        default=False,
        description="Explicit direct plaintext HTTP receiving permission.",
    )
    allow_private: bool = Field(
        default=False, description="Explicit private-address receiving permission."
    )
    public_network: bool = Field(
        default=False,
        description="Explicit network binding and public BitTorrent discovery; false is a loopback preview.",
    )
    listen_port: int = Field(
        default=0,
        ge=0,
        le=65535,
        description="Reviewed listener port; zero asks the OS for an available port.",
    )
    advertised_url: str | None = Field(
        default=None,
        max_length=2048,
        description="Operator-supplied public HTTP mirror base; never fetched or asserted reachable.",
    )
    upload_id: Handle | None = Field(
        default=None, description="Completed local kit upload; no host path."
    )
    location_id: Handle = Field(description="Enrolled source or destination archive.")
    model_id: str | None = Field(
        default=None, max_length=512, description="Installed base alias for Preserve."
    )
    generation_id: Digest | None = Field(
        default=None, description="Exact installed observation selected by the caller."
    )
    copy_id: Handle | None = Field(
        default=None,
        description="Exact retained copy for check/recovery/publication/sharing.",
    )
    destination_id: Handle | None = Field(
        default=None,
        description="Separate enrolled public destination when publishing.",
    )
    historical: bool = Field(
        default=False, description="Explicit historical registry-proof evaluation."
    )
    public_request: dict[str, JsonValue] | None = Field(
        default=None,
        description="Reviewed public metadata and per-member terms; never private paths or keys.",
    )
    duration_seconds: int = Field(
        default=3600,
        ge=1,
        le=86400,
        description="Maximum explicit sharing-session lifetime.",
    )
    upload_rate_bytes: int = Field(
        default=1048576,
        ge=1024,
        le=104857600,
        description="Native BitTorrent upload rate; bytes per second.",
    )
    upload_limit_bytes: int = Field(
        default=1073741824,
        ge=1,
        le=2**53 - 1,
        description="HTTP response allowance or observed torrent payload/peer wire stop threshold, with native overshoot.",
    )

    @model_validator(mode="after")
    def check_i2p(self) -> "PreservationPlanRequest":
        """Allow I2P only for explicit receiving, with no direct/peer permission."""
        if self.allow_i2p:
            if (
                self.kind != "receive"
                or self.i2p_proxy is None
                or self.allow_http
                or self.allow_private
                or self.allow_peer_network
                or self.peer_setup_id is not None
            ):
                raise ValueError(
                    "I2P grant requires an exact receive proxy without other transport grants"
                )
        elif self.i2p_proxy is not None:
            raise ValueError("I2P proxy requires explicit I2P participation")
        return self


@final
class SharingPolicy(PreservationContract):
    """Exact operator-reviewed listener, discovery and session allowances."""

    public_network: bool = Field(
        description="Network binding/public peer discovery; false is loopback only."
    )
    listen_port: int = Field(
        ge=0, le=65535, description="Requested port, zero for an OS-selected port."
    )
    duration_seconds: int = Field(
        ge=1, le=86400, description="Maximum owned serving-session lifetime."
    )
    upload_limit_bytes: int = Field(
        ge=1,
        le=2**53 - 1,
        description="HTTP response allowance or observed torrent payload/peer wire stop threshold.",
    )
    upload_rate_bytes: int = Field(
        ge=1024,
        le=104857600,
        description="Native BitTorrent payload-rate limit; not HTTP rate shaping.",
    )
    advertised_url: str | None = Field(
        default=None,
        max_length=2048,
        description="User-supplied mirror hint, not verified reachability.",
    )
    peer_private: bool = Field(
        default=False,
        exclude_if=omit_peer_default,
        description="Exact reviewed Core peer address scope, separate from HTTP permissions.",
    )


@final
class SharingStatus(PreservationContract):
    """Owned aggregate observations and an export receipt, without remote peer history."""

    handoff_sha256: Digest | None = Field(
        default=None,
        exclude_if=omit_peer_default,
        description="Hash of this job's retained one-file peer handoff, not reachability evidence.",
    )

    metainfo_sha256: Digest | None = Field(
        default=None,
        description="Retained standard torrent metadata hash, available before seeding.",
    )

    ready: bool = Field(
        default=False, description="Observed owned listener/seed readiness."
    )
    listen_port: int = Field(
        default=0,
        ge=0,
        le=65535,
        description="Actual owned listener port; zero after stop.",
    )
    charged_bytes: int = Field(
        default=0,
        ge=0,
        description="HTTP reservations, torrent payload or native peer wire bytes, according to accounting.",
    )
    accounting: Literal[
        "reserved-response", "observed-torrent-payload", "observed-peer-wire"
    ] = Field(
        description="Exact meaning of charged_bytes; peer wire observations include native routing overhead and may overshoot."
    )
    connected_peers: int = Field(
        default=0,
        ge=0,
        description="Aggregate BitTorrent peer count; no identities retained.",
    )


@final
class PreservationPlan(PreservationContract):
    """Exact reviewed plan summary; large manifests and private paths stay on host."""

    archive_format: Literal["preservation-v1", "pmf-v0.1"] = Field(
        default="preservation-v1",
        description="Authenticated archive format, or the native format planned for Preserve.",
    )

    existing_companions: tuple[str, ...] = Field(
        default=(),
        max_length=16,
        description="Reviewed matching owned companions; old files remain intact while the complete generation is prepared.",
    )
    receiving: ReceivingPolicy | None = Field(
        default=None,
        description="Exact receiving exposure/allowance; never a serving grant.",
    )
    resume_job_id: Digest | None = Field(
        default=None,
        description="Original receive job selected for an explicit resume action.",
    )
    sharing: SharingPolicy | None = Field(
        default=None,
        description="Exact reviewed sharing controls, never inherited from a kit.",
    )
    upload_id: Handle | None = Field(
        default=None, description="Exact selected kit upload, if any."
    )
    publisher_key_ids: tuple[str, ...] = Field(
        default=(),
        max_length=32,
        description="Candidate capture keys; never registry authority.",
    )
    historical: bool = Field(
        default=False,
        description="Allow historical proof inspection; never current revocation assurance.",
    )
    plan_id: Digest = Field(description="Immutable host plan identity.")
    kind: JobKind = Field(description="Reviewed effect kind.")
    model_id: str = Field(
        min_length=1, max_length=512, description="Bound selected model alias."
    )
    location_id: Handle = Field(description="Bound enrolled archive.")
    destination_id: Handle | None = Field(
        default=None, description="Bound separate destination, if applicable."
    )
    copy_id: Handle | None = Field(
        default=None, description="Bound retained source copy."
    )
    manifest_id: Digest | None = Field(
        default=None,
        description="Selected or planned signed identity, when already known.",
    )
    generation_id: Digest | None = Field(
        default=None, description="Exact installed generation reviewed for Preserve."
    )
    size_bytes: int = Field(ge=0, description="Complete logical closure size.")
    members: tuple[str, ...] = Field(
        default=(),
        max_length=16,
        description="First required aliases; details remain paged and the complete plan stays bound.",
    )
    member_count: int = Field(ge=1, description="Full number of selected members.")
    requires_owner_acceptance: bool = Field(
        default=False, description="Recovery requires deliberate exact-copy acceptance."
    )
    requires_code_permission: bool = Field(
        default=False, description="Separate repository-code permission is required."
    )
    deleted_aliases: tuple[str, ...] = Field(
        default=(),
        max_length=16,
        description="Removed aliases requiring explicit restore acknowledgement.",
    )
    expires_at: int = Field(
        ge=0, description="Last time this reviewed plan may start a new job."
    )
    publication: str | None = Field(
        default=None,
        max_length=64,
        description="Independently inspected original-publication class, if available.",
    )
    file_coverage: str | None = Field(
        default=None,
        max_length=64,
        description="Original signed file binding, separate from captured hashes.",
    )


@final
class PreservationStart(PreservationContract):
    """A deliberate start bound to an exact plan and retained idempotency identity."""

    plan_id: Digest = Field(description="Exact plan the operator reviewed.")
    operation_id: OperationId = Field(
        description="Caller-retained idempotency key; never reused for changed input."
    )
    owner_id: Handle = Field(
        default="operator",
        description="Stable workflow owner; used for scoped management cancellation.",
    )
    accept_owner_capture: bool = Field(
        default=False,
        description="Accept this exact non-registry/historical copy, not other models.",
    )
    allow_repository_code: bool = Field(
        default=False,
        description="Separate explicit permission for this selected potentially executable card.",
    )
    restore_deleted: bool = Field(
        default=False,
        description="Explicitly restore the removed aliases displayed by this plan.",
    )
    public_participation: bool = Field(
        default=False,
        description="Explicit public-audience publication, hosting or swarm participation; publication starts no listener.",
    )


@final
class RecoveryCleanup(PreservationContract):
    """Result of explicit temporary-file reclamation; never a restore receipt."""

    job_id: Digest = Field(description="Exact stopped recovery job inspected.")
    generation_removed: bool = Field(description="An unindexed generation was removed.")
    temporary_removed: bool = Field(
        description="Known worker temporary files were removed."
    )
    indexed_generation_retained: bool = Field(
        description="An indexed generation was retained."
    )


@final
class TemporaryWork(PreservationContract):
    """Bounded observations before optional cleanup of one stopped job's work."""

    job_id: Digest = Field(description="Exact stopped public-sharing job inspected.")
    found: bool = Field(
        default=False, description="Temporary work was found before this action."
    )
    logical_bytes: int = Field(
        default=0,
        ge=0,
        description="Observed regular-file lengths, not allocated or guaranteed freed storage.",
    )
    files: int = Field(default=0, ge=0, description="Observed regular-file count.")
    complete: bool = Field(
        default=True,
        description="False means the bounded measurement is a lower bound.",
    )
    removed: bool = Field(
        default=False,
        description="Explicit cleanup finished; no kept copy or torrent artifact was removed.",
    )


@final
class PreservationJob(PreservationContract):
    """Canonical durable host status; cancellation intent is not proof of stop."""

    bytes_received: int = Field(
        default=0,
        ge=0,
        description="Largest observed attempt's received bytes; not a remaining-quota or unique-coverage claim.",
    )
    sharing_policy: SharingPolicy | None = Field(
        default=None,
        description="Immutable reviewed session exposure and allowances, retained after plan expiry.",
    )

    restore_deleted: bool = Field(
        default=False,
        description="This receiver's confirmation of the exact reviewed removals.",
    )
    sharing: SharingStatus | None = Field(
        default=None,
        description="Latest safe owned serving observation; not a public availability guarantee.",
    )
    accept_owner_capture: bool = Field(
        default=False,
        description="This receiver's exact job consent, never an imported grant.",
    )
    allow_repository_code: bool = Field(
        default=False,
        description="This receiver's separate exact repository-code consent.",
    )
    location_id: Handle | None = Field(
        default=None, description="Exact source location retained by this job's plan."
    )
    job_id: Handle = Field(description="Durable host-owned job handle.")
    operation_id: OperationId = Field(
        description="Original caller idempotency identity."
    )
    owner_id: Handle = Field(description="Stable workflow owner.")
    plan_id: Digest = Field(description="Exact retained plan.")
    kind: JobKind = Field(description="Actual owned effect.")
    model_id: str = Field(
        min_length=1,
        max_length=512,
        description="Target model; completion actions must use this identity.",
    )
    state: JobState = Field(description="Actual host lifecycle state.")
    stage: str = Field(
        default="queued",
        max_length=64,
        description="Safe current stage code, not a success assertion.",
    )
    created_at: int = Field(ge=0, description="Initial accepted time as Unix seconds.")
    updated_at: int = Field(
        ge=0, description="Most recent persisted observation as Unix seconds."
    )
    finished_at: int | None = Field(
        default=None, ge=0, description="Confirmed terminal time, when known."
    )
    attempts: int = Field(
        default=0,
        ge=0,
        description="Retained execution attempts; restart never resets this count.",
    )
    bytes_copied: int = Field(
        default=0,
        ge=0,
        description="Observed copy work, not unique verified coverage or a percentage.",
    )
    bytes_hashed: int = Field(
        default=0,
        ge=0,
        description="Observed hashing work, which may read bytes more than once.",
    )
    expected_manifest_id: Digest | None = Field(
        default=None,
        description="Durably recorded expected outcome before publication is allowed.",
    )
    copy_id: Handle | None = Field(
        default=None, description="Actual completed retained copy, if any."
    )
    error_code: str | None = Field(
        default=None,
        max_length=64,
        description="Safe stable error code; no paths, keys or raw exception text.",
    )
    result: dict[str, JsonValue] | None = Field(
        default=None,
        description="Bounded safe terminal receipt, independently checked by the host.",
    )


@final
class KitUploadRequest(PreservationContract):
    """Declare a bounded kit byte transfer before any ingress."""

    size_bytes: int = Field(
        ge=1, le=8 * 1024**4, description="Exact whole kit file length."
    )
    purpose: Literal["kit", "invitation", "publication", "source"] = Field(
        default="kit",
        description="Declared ingress use; invitations/publications up to 8 MiB, source bootstraps up to 256 KiB.",
    )


@final
class KitUploadStatus(PreservationContract):
    """Durable ingress observation, not a signature or admission result."""

    upload_id: Handle = Field(description="Opaque host-created ingress handle.")
    purpose: Literal["kit", "invitation", "publication", "source"] = Field(
        default="kit",
        description="Declared ingress purpose; legacy uploads remain recovery kits.",
    )
    size_bytes: int = Field(ge=1, description="Declared exact kit length.")
    offset: int = Field(ge=0, description="Flushed bytes; resume at this offset.")
    complete: bool = Field(
        description="Transfer length reached; actual hashing occurs during restore."
    )


@final
class PreservationPage(PreservationContract):
    """A bounded page of private host observations."""

    default_location_id: Handle | None = Field(
        default=None,
        description="Durable private destination for new copies, retained even when disconnected.",
    )
    copy_details: dict[Handle, ModelDetails] = Field(
        default_factory=dict,
        max_length=4,
        description="Optional authenticated archive metadata for copies on this page. Missing or unavailable excerpts never prevent recovery; original cards and receipts remain unchanged.",
    )

    publishing: PreservationLocation | None = Field(
        default=None,
        description="Separately configured public publishing identity; never a private Preserve destination.",
    )
    uploads: tuple[KitUploadStatus, ...] = Field(
        default=(),
        max_length=8,
        description="Retained resumable kit transfers; reads do not start work.",
    )
    temporary_work_jobs: tuple[Digest, ...] = Field(
        default=(),
        max_length=16,
        description="Stopped jobs on this page with possible temporary work; no content scan or cleanup occurs during listing.",
    )
    bittorrent_available: bool = Field(
        default=False,
        description="Native transport module is installed; execution still verifies its binding before participation.",
    )
    public_copies: tuple[PreservationCopy, ...] = Field(
        default=(),
        max_length=16,
        description="Explicitly imported signed public bundles, separate from private captures.",
    )
    locations: tuple[PreservationLocation, ...] = Field(
        default=(), max_length=32, description="Enrolled roots on this page."
    )
    models: tuple[PreservationModel, ...] = Field(
        default=(), max_length=16, description="Installed bases on this page."
    )
    copies: tuple[PreservationCopy, ...] = Field(
        default=(), max_length=16, description="Retained copies on this page."
    )
    jobs: tuple[PreservationJob, ...] = Field(
        default=(), max_length=16, description="Durable operations on this page."
    )
    next_offset: int | None = Field(
        default=None, ge=0, description="Next bounded listing offset, if present."
    )
