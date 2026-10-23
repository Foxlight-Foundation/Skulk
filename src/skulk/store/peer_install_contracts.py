"""Separate native-software consent and durable receipts, without peer authority."""

from typing import final

from foxlight_model_preservation.peer_install import RuntimePlan
from pydantic import Field, model_validator

from skulk.store.preservation_contracts import (
    Digest,
    Handle,
    JobState,
    OperationId,
    PreservationContract,
)


@final
class RuntimeSetupRequest(PreservationContract):
    """Authorize only the exact packaged native build reviewed on this Core."""

    owner_id: Handle = Field(description="Stable local client namespace.")
    operation_id: OperationId = Field(
        description="Permanent exact-request replay identity."
    )
    plan_id: Digest = Field(
        description="Exact offline native acquisition plan reviewed."
    )
    confirmed: bool = Field(
        description="Explicit software download and private installation consent, not peer network consent."
    )

    @model_validator(mode="after")
    def require_consent(self) -> "RuntimeSetupRequest":
        """Reject a request without the separate acquisition grant."""
        if not self.confirmed:
            raise ValueError("native acquisition consent required")
        return self


@final
class RuntimeSetupJob(PreservationContract):
    """Observed Core-owned acquisition; browser closure cannot stop or replay it."""

    job_id: Digest = Field(description="Exact owner/operation-bound receipt identity.")
    owner_id: Handle = Field(description="Client that submitted this setup.")
    state: JobState = Field(
        description="Actual owned worker state; restart never replays."
    )
    bytes_copied: int = Field(
        default=0,
        ge=0,
        description="Archive bytes staged by the worker, including abandoned attempts.",
    )
    created_at: int = Field(description="Unix creation time.")
    updated_at: int = Field(description="Unix last state transition time.")
    error_code: str | None = Field(
        default=None,
        max_length=80,
        description="Safe local outcome without filesystem paths or upstream error text.",
    )


@final
class RuntimeSetupView(PreservationContract):
    """Offline software review and this client's latest durable setup observation."""

    supported: bool = Field(description="A pinned managed build exists for this host.")
    plan_id: Digest | None = Field(
        default=None,
        description="Exact reviewed plan required to authorize acquisition.",
    )
    plan: RuntimePlan | None = Field(
        default=None,
        description="Fixed official origins, version, sizes and distribution identities; no host path.",
    )
    job: RuntimeSetupJob | None = Field(
        default=None,
        description="Most recent setup for this client, including restart interruption.",
    )


@final
class RuntimeSetupRecord(PreservationContract):
    """Private journal binding of the immutable original request and public receipt."""

    request: RuntimeSetupRequest = Field(
        description="Original exact acquisition grant."
    )
    job: RuntimeSetupJob = Field(description="Latest host-observed outcome.")


@final
class RuntimeSetupHistory(PreservationContract):
    """Bound retained replay identities instead of silently forgetting authorization."""

    records: tuple[RuntimeSetupRecord, ...] = Field(
        default=(),
        max_length=64,
        description="At most 64 permanent native acquisition receipts; no automatic retry.",
    )
