"""Operator controls for the Core-owned persistent public mirror, without paths."""

from __future__ import annotations

from typing import Literal, final

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
]
MirrorAction = Literal["pin", "unpin", "block", "unblock", "retry"]


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
    egress_daily: int = Field(
        ge=1,
        le=2**53 - 1,
        description="Conservatively reserved daily upload allowance.",
    )
    egress_monthly: int = Field(
        ge=1,
        le=2**53 - 1,
        description="Conservatively reserved monthly upload allowance.",
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

    @model_validator(mode="after")
    def check_peer_selection(self) -> MirrorConfiguration:
        """Bind IPFS to current reviewed setup, without granting network effects."""
        if self.ipfs != (self.peer_setup_id is not None):
            raise ValueError("IPFS requires its exact peer setup identity")
        return self


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

    @model_validator(mode="after")
    def check_decision(self) -> MirrorRequest:
        """Refuse missing authority and inputs belonging to a different operation."""
        if (self.configuration is not None) != (self.kind == "configure"):
            raise ValueError("configuration belongs only to configure")
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
        if self.kind in {"model", "prune", "prepare_kit"} and self.manifest_id is None:
            raise ValueError("select an exact retained model")
        if (
            self.kind not in {"model", "prune", "retain", "prepare_kit"}
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


@final
class MirrorJobsPage(PreservationContract):
    """Bounded permanent mirror history, independent of finite sharing jobs."""

    jobs: tuple[MirrorJob, ...] = Field(
        default=(), max_length=20, description="One mirror receipt page."
    )
    next_offset: int | None = Field(
        default=None, ge=0, le=4096, description="Next history offset, if any."
    )
