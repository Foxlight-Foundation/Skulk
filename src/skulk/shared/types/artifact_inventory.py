"""Ephemeral node-local artifact availability carried on telemetry."""

from typing import Literal

from pydantic import ConfigDict, Field

from skulk.utils.pydantic_ext import FrozenModel, TaggedModel

ARTIFACT_INVENTORY_ENTRY_LIMIT = 2_048
"""Maximum compact artifact locations carried in one node reading."""

ARTIFACT_INVENTORY_REFRESH_SECONDS = 60.0
"""Periodic repair cadence for node artifact availability."""

ARTIFACT_INVENTORY_DEBOUNCE_SECONDS = 2.0
"""Quiet period used to coalesce clustered filesystem mutations."""

ARTIFACT_INVENTORY_STALE_SECONDS = ARTIFACT_INVENTORY_REFRESH_SECONDS * 2
"""Age after which an inventory reading is partial rather than current."""


class NodeArtifactAvailability(FrozenModel):
    """One installed generation available from a node-local cache.

    The reading deliberately excludes card bodies and file manifests. Those
    remain canonical store truth; telemetry carries only the compact identity
    and status needed by operator views.
    """

    model_id: str = Field(description="Artifact model alias in owner/repository form.")
    installed_identity: str = Field(
        description="Durable installed generation identity retained by this copy."
    )
    size_bytes: int = Field(ge=0, description="Bytes retained by this local copy.")
    last_used_epoch_seconds: float = Field(
        ge=0,
        description="Unix time of the best-known most recent local use.",
    )
    in_use: bool = Field(
        description="Whether a live runner on this node currently depends on the copy."
    )
    manifest_complete: bool = Field(
        description="Whether every retained manifest path and size is present."
    )
    location_kind: Literal["node_cache"] = Field(
        default="node_cache",
        description="Node-local cache classification; canonical store locality is synthesized.",
    )


class NodeArtifactInventory(TaggedModel):
    """One node's latest compact artifact-availability snapshot.

    This is lossy operator telemetry, not permission to import, delete, or
    execute bytes. Reconciliation continues to verify storage directly.
    """

    model_config = ConfigDict(frozen=True)

    artifacts: list[NodeArtifactAvailability] = Field(
        max_length=ARTIFACT_INVENTORY_ENTRY_LIMIT,
        description="Bounded node-cache locations ordered by operational relevance.",
    )
    store_host: bool = Field(
        description="Whether this node owns the authoritative canonical model store."
    )
    truncated: bool = Field(
        description="Whether additional node-cache entries were omitted to preserve bounds."
    )
