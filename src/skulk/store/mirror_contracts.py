"""Operator controls for the Core-owned persistent public mirror, without paths."""

from __future__ import annotations

from typing import Annotated, Literal, final

from foxlight_model_preservation.mirror_cards import base_url
from foxlight_model_preservation.mirror_models import IntakePolicy
from pydantic import Field, model_validator

from skulk.store.preservation_contracts import (
    Digest,
    Fingerprint,
    Handle,
    JobState,
    OperationId,
    PreservationContract,
    PublicationMember,
)

MirrorKind = Literal[
    "configure",
    "retain",
    "run",
    "grant_resume",
    "revoke_resume",
    "model",
    "prune",
    "prepare_kit",
    "recover",
    "intake",
]
MirrorAction = Literal[
    "pin",
    "unpin",
    "block",
    "unblock",
    "retry",
    "withhold",
    "release",
]
# The announcement disclosure version this Core recognizes: the text a client
# shows before consent (public address, DHT and Nostr exposure, no anonymity).
ANNOUNCE_DISCLOSURE = 1
AddressOutcome = Literal[
    "this_mirror",
    "other_mirror",
    "no_card",
    "not_a_mirror",
    "http_error",
    "private_address",
    "too_slow",
    "unreachable",
]


def _omit_absent(value: object) -> bool:
    """Preserve existing job wire records when no prepared artifact exists."""
    return value is None


@final
class MirrorConfiguration(PreservationContract):
    """Offline serving/resource choices; saving never starts a listener or router."""

    name: str = Field(
        min_length=1, max_length=200, description="Operator's mirror name."
    )
    disk_bytes: int = Field(
        ge=256 * 1024**2,
        le=2**53 - 1,
        description="Retained/partial/cache disk allowance.",
    )
    ingress_daily: int = Field(
        ge=1,
        le=2**53 - 1,
        description="Daily network intake allowance; local retention spends none.",
    )
    egress_daily: int | None = Field(
        ge=1,
        le=2**53 - 1,
        description="Conservatively reserved daily upload allowance; null is no cap (uploads are still counted).",
    )
    egress_monthly: int | None = Field(
        ge=1,
        le=2**53 - 1,
        description="Conservatively reserved monthly upload allowance; null is no cap (uploads are still counted).",
    )
    http: bool = Field(
        default=False,
        description="Separately select HTTP serving; no listener starts on save.",
    )
    exposure: Literal["loopback", "network"] = Field(
        default="loopback",
        description="Numeric HTTP bind scope, not proof of public routing.",
    )
    port: int = Field(
        default=8080,
        ge=0,
        le=65535,
        description="HTTP port; zero requests an ephemeral port.",
    )
    ipfs: bool = Field(
        default=False, description="Select separately configured native IPFS sharing."
    )
    torrent: bool = Field(
        default=False,
        description="Select public BitTorrent sharing; receiving never implies this.",
    )
    peer_setup_id: Digest | None = Field(
        default=None,
        description="Exact reviewed Core peer settings when selecting IPFS.",
    )
    max_active: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Maximum owned native workers, including preparation.",
    )
    session_seconds: int = Field(
        default=3600,
        ge=1,
        le=86400,
        description="Finite native session duration inside the controller.",
    )
    session_bytes: int = Field(
        default=64 * 1024**2,
        ge=1,
        le=2**53 - 1,
        description="Upload reservation per native session, spent before startup.",
    )
    upload_rate: int = Field(
        default=1024**2,
        ge=1024,
        le=100 * 1024**2,
        description="Native upload rate in bytes per second.",
    )
    public_url: str | None = Field(
        default=None,
        min_length=8,
        max_length=2048,
        description="Credential-free HTTP(S) base URL announced in this mirror's signed card. Saving it contacts nothing; an address is public, not anonymous.",
    )
    announce_dht: bool = Field(
        default=False,
        description="Also announce the public address on the public BitTorrent DHT (plain-HTTP root URLs only).",
    )
    publish_nostr: bool = Field(
        default=False,
        description="Also copy the signed card to third-party Nostr relays.",
    )
    announce_disclosure: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="Announcement disclosure version shown before consent. Nothing is announced until a recognized version is recorded with this configuration.",
    )

    @model_validator(mode="after")
    def check_peer_selection(self) -> MirrorConfiguration:
        """Bind IPFS to current reviewed setup, without granting network effects."""
        if self.ipfs != (self.peer_setup_id is not None):
            raise ValueError("IPFS requires its exact peer setup identity")
        if self.public_url is not None and base_url(self.public_url) != self.public_url:
            raise ValueError("public URL must be a plain HTTP(S) base URL")
        if (self.announce_dht or self.publish_nostr) and self.public_url is None:
            raise ValueError("announcing needs a public URL")
        return self

    def announces(self) -> bool:
        """Whether this configuration may announce: an address and its disclosure."""
        return (
            self.public_url is not None
            and self.announce_disclosure is not None
            and self.announce_disclosure >= ANNOUNCE_DISCLOSURE
        )


@final
class MirrorIntake(PreservationContract):
    """The owner's standing public intake policy, held by Core under the grant.

    ``eligible`` admits a newly kept public copy whose every member licence is
    listed, without new consent: the restart grant already covers it.
    ``chosen`` admits nothing on its own. ``withheld`` copies stay back from
    serving and from standing intake; withholding only narrows.
    """

    share: Literal["eligible", "chosen"] = Field(
        description="Share every eligible public copy, or only individually chosen ones."
    )
    licenses: tuple[Annotated[str, Field(min_length=1, max_length=200)], ...] = Field(
        strict=False,  # FastAPI supplies decoded JSON lists; items stay strict.
        min_length=1,
        max_length=128,
        description="Normalized, unique licence identifiers the owner reviewed.",
    )
    definition: int = Field(
        ge=1,
        le=1000,
        description="Licence-definition version the list was reviewed under.",
    )
    withheld: tuple[Digest, ...] = Field(
        default=(),
        strict=False,
        max_length=1000,
        description="Exact copies kept back from serving and standing intake.",
    )

    @model_validator(mode="after")
    def check_policy(self) -> MirrorIntake:
        """Refuse anything the core intake policy would refuse, before any effect."""
        self.core()
        return self

    def core(self) -> IntakePolicy:
        """The exact core ``IntakePolicy`` this decision records."""
        try:
            return IntakePolicy(
                share=self.share,
                licenses=self.licenses,
                definition=self.definition,
                withheld=self.withheld,
            )
        except ValueError as error:
            raise ValueError("invalid standing intake policy") from error

    @classmethod
    def observed(cls, policy: IntakePolicy) -> MirrorIntake:
        """Project a saved core policy for the mirror page."""
        return cls(
            share=policy.share,
            licenses=policy.licenses,
            definition=policy.definition,
            withheld=policy.withheld,
        )


@final
class MirrorRequest(PreservationContract):
    """One permanently identified decision; replay observes the original job."""

    owner_id: Handle = Field(
        description="Client namespace, retained across page reload."
    )
    operation_id: OperationId = Field(
        description="Permanent client operation identity."
    )
    kind: MirrorKind = Field(
        description="Exactly one mirror control or byte operation."
    )
    policy_id: Digest | None = Field(
        default=None,
        description="Exact current reviewed policy; absent only at first configuration.",
    )
    configuration: MirrorConfiguration | None = Field(
        default=None, description="Offline policy, required only for configure."
    )
    copy_id: Handle | None = Field(
        default=None,
        description="Already accepted public Library copy, never a host path.",
    )
    manifest_id: Digest | None = Field(
        default=None, description="Exact selected signed model identity."
    )
    action: MirrorAction | None = Field(
        default=None,
        description="Explicit pin/block/retry action; deletion is separate.",
    )
    confirm_redistribution: bool = Field(
        default=False,
        description="Review redistribution terms for every selected member.",
    )
    confirm_reintake: bool = Field(
        default=False,
        description="Deliberately re-admit a removed selection without resetting accounting.",
    )
    public_participation: bool = Field(
        default=False,
        description="Separate explicit public serving permission for this run.",
    )
    mode: Literal["finite", "resume"] = Field(
        default="finite",
        description="Finite manual run or separately approved standing grant.",
    )
    seconds: int = Field(
        default=3600,
        ge=1,
        le=86400,
        description="Finite manual duration; never a restart grant.",
    )
    resume_expires: int | None = Field(
        default=None,
        ge=1,
        description="Exact separately confirmed standing-grant expiry, Unix seconds.",
    )
    confirm_resume: bool = Field(
        default=False,
        description="Explicit standing permission, independent of model and transport trust.",
    )
    confirm_delete_manifest: Digest | None = Field(
        default=None,
        description="Exact operator deletion confirmation; no automatic eviction.",
    )
    intake: MirrorIntake | None = Field(
        default=None,
        exclude_if=_omit_absent,
        description="Standing intake policy for an intake decision; null clears it. Any effective change clears restart permission, except the withheld list, which only narrows.",
    )

    @model_validator(mode="after")
    def check_decision(self) -> MirrorRequest:
        """Refuse missing authority and inputs belonging to a different operation."""
        if (self.configuration is not None) != (self.kind == "configure"):
            raise ValueError("configuration belongs only to configure")
        if self.kind != "intake" and self.intake is not None:
            raise ValueError("an intake policy belongs only to intake")
        if self.kind != "configure" and self.policy_id is None:
            raise ValueError("review the current mirror policy first")
        if self.kind == "retain":
            if (
                not self.copy_id
                or not self.manifest_id
                or not self.confirm_redistribution
            ):
                raise ValueError("retain needs an exact public copy and reviewed terms")
        elif (
            self.copy_id is not None
            or self.confirm_redistribution
            or self.confirm_reintake
        ):
            raise ValueError("copy intake grants belong only to retain")
        if (
            self.kind in {"model", "prune", "prepare_kit", "recover"}
            and self.manifest_id is None
        ):
            raise ValueError("select an exact retained model")
        if (
            self.kind not in {"model", "prune", "retain", "prepare_kit", "recover"}
            and self.manifest_id is not None
        ):
            raise ValueError("model identity belongs only to model operations")
        if (self.action is not None) != (self.kind == "model"):
            raise ValueError("model action required only for model controls")
        if self.kind == "run":
            if not self.public_participation:
                raise ValueError("public mirror participation must be explicit")
        elif self.public_participation or self.mode != "finite" or self.seconds != 3600:
            raise ValueError("run permission belongs only to run")
        if self.kind == "grant_resume":
            if not self.confirm_resume or self.resume_expires is None:
                raise ValueError("standing permission requires exact confirmed expiry")
        elif self.confirm_resume or self.resume_expires is not None:
            raise ValueError("standing permission belongs only to grant_resume")
        if self.kind == "prune":
            if self.confirm_delete_manifest != self.manifest_id:
                raise ValueError("deletion requires the exact selected manifest")
        elif self.confirm_delete_manifest is not None:
            raise ValueError("deletion confirmation belongs only to prune")
        return self


@final
class MirrorActivity(PreservationContract):
    """Live owned activity, never a count of publicly reachable replicas."""

    running: bool = Field(description="False only after owned work/listeners drain.")
    http_port: int | None = Field(
        default=None,
        ge=1,
        le=65535,
        description="Actual owned local listener, if enabled.",
    )
    native_workers: int = Field(
        ge=0, le=4, description="Live native workers, including preparation and drain."
    )


@final
class MirrorKit(PreservationContract):
    """Prepared content identity; recipient model trust remains independent."""

    state: Literal["prepared", "removed"] = Field(
        description="Historical preparation or explicit removal, not a fresh payload scan."
    )
    manifest_id: Digest = Field(
        description="Original signed model identity inside this kit."
    )
    size_bytes: int = Field(
        ge=1, le=2**53 - 1, description="Prepared complete kit length."
    )
    sha256: Digest = Field(
        description="Prepared kit bytes digest, separate from its signed model identity."
    )


@final
class MirrorKitRemoval(PreservationContract):
    """Separate exact-artifact deletion, including stopped incomplete preparation."""

    confirm_job_id: Digest = Field(
        description="Explicitly confirm this preparation job's private kit and temporary output only."
    )


@final
class MirrorJob(PreservationContract):
    """Durable control/worker receipt separate from existing finite copy jobs."""

    job_id: Digest = Field(description="Host-owned permanent job identity.")
    owner_id: Handle = Field(description="Original submitting client.")
    operation_id: OperationId = Field(
        description="Original permanent operation identity."
    )
    kind: MirrorKind = Field(
        description="Selected operation, immutable after admission."
    )
    state: JobState = Field(description="Durable owned job lifecycle.")
    created_at: int = Field(ge=0, description="Admission Unix time.")
    updated_at: int = Field(ge=0, description="Last durable observation Unix time.")
    manifest_id: Digest | None = Field(
        default=None, description="Exact selected model, when applicable."
    )
    activity: MirrorActivity | None = Field(
        default=None,
        description="Live observations only while the exact controller is owned.",
    )
    error_code: str | None = Field(
        default=None,
        max_length=80,
        description="Safe bounded refusal; no path/exception text.",
    )
    bytes_hashed: int = Field(
        default=0, ge=0, description="Observed verification bytes for this attempt."
    )
    bytes_copied: int = Field(
        default=0, ge=0, description="Observed local copy bytes for this attempt."
    )
    bytes_received: int = Field(
        default=0, ge=0, description="Observed receive bytes, not a new budget grant."
    )
    kit: MirrorKit | None = Field(
        default=None,
        exclude_if=_omit_absent,
        description="Prepared content or explicit removal receipt; absent before complete sealing.",
    )
    kit_removed: bool = Field(
        default=False,
        description="Operator removed this job's prepared or incomplete kit generation; retained models are unaffected.",
    )
    admitted_by: Literal["intake"] | None = Field(
        default=None,
        exclude_if=_omit_absent,
        description="Present on a retain that Core started itself under the standing intake policy and the existing grant, without a new consent.",
    )


@final
class MirrorModel(PreservationContract):
    """Bounded retained-selection status with no private authority or paths."""

    manifest_id: Digest = Field(description="Original exact signed manifest identity.")
    state: Literal["pending", "retained", "removed"] = Field(
        description="Custody receipt, not public availability."
    )
    pinned: bool = Field(description="Operator pin; prune must refuse while pinned.")
    blocked: bool = Field(
        description="Operator serving block; source loss never deletes bytes."
    )
    error_code: str | None = Field(
        default=None,
        max_length=80,
        description="Safe intake/verification refusal, if any.",
    )
    name: str | None = Field(
        default=None,
        max_length=200,
        description="Authenticated original name, when local metadata is available.",
    )
    member_count: int | None = Field(
        default=None,
        ge=1,
        le=256,
        description="Declared full member count; no fresh payload completeness claim.",
    )
    size_bytes: int | None = Field(
        default=None,
        ge=0,
        description="Declared logical closure bytes, without scanning weights.",
    )
    withheld: bool = Field(
        default=False,
        description="Kept back from serving and standing intake by the owner.",
    )


@final
class MirrorUsage(PreservationContract):
    """Counted transfer volume in the current UTC day and month."""

    uploaded_today_bytes: int = Field(ge=0, description="Bytes served today.")
    uploaded_month_bytes: int = Field(ge=0, description="Bytes served this month.")
    received_today_bytes: int = Field(ge=0, description="Bytes received today.")


@final
class MirrorAddressCheck(PreservationContract):
    """Best-effort reachability of the configured public address, from this host only."""

    address: str = Field(max_length=2048, description="The configured public URL.")
    outcome: AddressOutcome = Field(
        description="What this host saw. this_mirror shows the address answers here; it cannot prove others can reach it."
    )
    http_status: int | None = Field(
        default=None, ge=100, le=599, description="Status of an HTTP error, if any."
    )
    best_effort: Literal[True] = Field(
        default=True,
        description="Always true: a check from this host, not proof of public reachability.",
    )


@final
class MirrorSelection(PreservationContract):
    """Exact identity lookup independent of application receipts and list pages."""

    schema_version: Literal["skulk.preservation-mirror-selection.v1"] = Field(
        default="skulk.preservation-mirror-selection.v1",
        description="Exact local selection observation contract.",
    )
    manifest_id: Digest = Field(
        description="Exact requested original manifest identity."
    )
    configured: bool = Field(description="Core has an enrolled mirror directory.")
    available: bool = Field(
        description="Local state was available; false never means an unknown selection."
    )
    policy_id: Digest | None = Field(
        default=None, description="Observed policy to review before a new effect."
    )
    model: MirrorModel | None = Field(
        default=None, description="None means unknown only when available is true."
    )


@final
class MirrorReview(PreservationContract):
    """Complete authenticated member terms, independent of original import records."""

    schema_version: Literal["skulk.preservation-mirror-review.v1"] = Field(
        default="skulk.preservation-mirror-review.v1",
        description="Bounded retained-copy metadata review contract.",
    )
    manifest_id: Digest = Field(description="Exact retained signed identity.")
    archive_format: Literal["preservation-v1"] = Field(
        default="preservation-v1",
        description="Current public mirror transport profile, without conversion.",
    )
    name: str = Field(max_length=200, description="Original signed display name.")
    size_bytes: int = Field(
        ge=0, description="Declared logical closure size, not a fresh weight check."
    )
    workloads: tuple[str, ...] = Field(
        max_length=100, description="Original declared workloads."
    )
    runtime_requirements: tuple[str, ...] = Field(
        max_length=100,
        description="Original runtime declarations; no code or model admission.",
    )
    members: tuple[PublicationMember, ...] = Field(
        min_length=1,
        max_length=256,
        description="Every signed member and required notice; oversized reviews refuse entirely.",
    )
    capture_key_ids: tuple[Fingerprint, ...] = Field(
        min_length=1,
        max_length=32,
        description="Locally accepted non-revoked capture keys, not registry authority or a recipient trust grant.",
    )
    capture_threshold: int = Field(
        ge=1, le=32, description="Previously accepted capture verification threshold."
    )


@final
class MirrorPage(PreservationContract):
    """Read-only page; constructing or observing it never starts sharing."""

    schema_version: Literal["skulk.preservation-mirror.v1"] = Field(
        default="skulk.preservation-mirror.v1",
        description="Additive dedicated control contract.",
    )
    configured: bool = Field(
        description="Whether Core enrolled its managed mirror directory."
    )
    available: bool = Field(
        description="Bounded local observation succeeded; false also covers a busy filesystem."
    )
    policy_id: Digest | None = Field(
        default=None, description="Exact current policy to review before a new action."
    )
    configuration: MirrorConfiguration | None = Field(
        default=None, description="Actual offline settings without executable paths."
    )
    paused: bool = Field(
        default=True, description="Durable Stop fence, independent of retained bytes."
    )
    restart_expires: int | None = Field(
        default=None,
        description="Saved standing permission expiry, not an active session.",
    )
    restart_allowed: bool = Field(
        default=False,
        description="Exact unexpired policy permission; does not start work.",
    )
    controller_job_id: Digest | None = Field(
        default=None, description="Currently owned controller job, if any."
    )
    activity: MirrorActivity | None = Field(
        default=None,
        description="Actual owned controller activity; no historical port substitution.",
    )
    models: tuple[MirrorModel, ...] = Field(
        default=(), max_length=20, description="One bounded retained-selection page."
    )
    model_count: int = Field(
        default=0,
        ge=0,
        le=1000,
        description="All selected identities, including deletion tombstones.",
    )
    next_offset: int | None = Field(
        default=None, ge=0, le=1000, description="Next model-page offset."
    )
    intake: MirrorIntake | None = Field(
        default=None,
        description="Saved standing intake policy (share mode and licences), or null.",
    )
    usage: MirrorUsage | None = Field(
        default=None,
        description="Counted uploads and intake, including when uploads are uncapped.",
    )
    dht_announcing: bool = Field(
        default=False,
        description="The owned controller is announcing the public address on the DHT now.",
    )
    nostr_publishing: bool = Field(
        default=False,
        description="The owned controller is publishing the signed card to Nostr relays now.",
    )
    announce_reasons: tuple[str, ...] = Field(
        default=(),
        max_length=16,
        description="Why announcing is off or limited, as stable codes; empty while announcing as configured.",
    )


@final
class MirrorJobsPage(PreservationContract):
    """Bounded permanent mirror history, independent of finite sharing jobs."""

    jobs: tuple[MirrorJob, ...] = Field(
        default=(), max_length=20, description="One mirror receipt page."
    )
    next_offset: int | None = Field(
        default=None, ge=0, le=4096, description="Next history offset, if any."
    )
