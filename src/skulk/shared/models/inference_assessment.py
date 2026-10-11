"""Read-only, estimate-based inference planning contracts."""

from typing import Literal, final

from pydantic import Field

from skulk.utils.pydantic_ext import FrozenModel


@final
class InferenceWorkload(FrozenModel):
    """Requested text workload; context includes input and generated tokens."""

    context_tokens: int = Field(
        default=8192,
        ge=256,
        le=1_048_576,
        description="Input plus output context tokens per simultaneous request.",
    )
    concurrent_requests: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Requested simultaneous serving width, without a performance SLO.",
    )


@final
class AssessedShard(FrozenModel):
    """Estimated allocation on one node, without creating a runner."""

    node_id: str
    backend: str | None
    estimated_memory_bytes: int | None
    """Shared estimated footprint; null for runtime-selected RPC partitions."""
    usable_memory_bytes: int
    """Memory available after current commitments and working-set caps."""
    memory_pool: Literal["ram", "vram"]


@final
class AssessedReplica(FrozenModel):
    """One independent model replica in the conservative concurrency plan."""

    sharding: str
    shards: tuple[AssessedShard, ...]
    concurrent_requests: int
    context_pool_tokens: int
    """Aggregate reserved context for requests assigned to this replica."""
    existing_instance_id: str | None = None


@final
class ExternalConfiguration(FrozenModel):
    """Hypothetical complete remote allocation, not a provider offer."""

    backend: str
    engine_build: str | None
    hardware_classes: tuple[str, ...]
    replica_count: int
    requests_per_replica: int
    context_pool_tokens: int
    nodes_per_replica: Literal[1] = 1
    gpus_per_node: Literal[0, 1]
    """Discrete GPU requirement; zero includes CPU and Apple unified-memory hosts."""
    memory_pool: Literal["ram", "vram"]
    estimated_memory_bytes_per_replica: int
    minimum_memory_bytes_per_node: int
    """Physical RAM or VRAM needed to expose the estimated usable working set."""
    minimum_weight_disk_bytes_per_node: int
    """Selected weights and projector only; no installation or staging allowance."""
    host_ram_bytes: int | None = None
    """Unknown for discrete GPU configurations; staging/runtime need extra RAM."""
    offer_available: Literal[False] = False
    """No live offer, stock or price was read for this hypothetical specification."""


@final
class InferenceAssessment(FrozenModel):
    """Estimated feasibility, independently of observed serving readiness."""

    model_id: str
    card_digest: str
    """Canonical authorized-card content identity, excluding catalog snapshot."""
    source_repository: str
    source_revision: str | None
    selected_file: str | None
    quantization: str
    workload: InferenceWorkload
    status: Literal["local", "external_required", "blocked", "insufficient_evidence"]
    readiness: Literal["not_observed", "loaded_not_workload_verified"] = "not_observed"
    read_at: str
    """API-side UTC assessment time, not payload measurement freshness."""
    reasons: tuple[str, ...]
    assumptions: tuple[str, ...]
    uncertainty: tuple[str, ...]
    weight_bytes: int
    weight_evidence: Literal["card_declared"] = "card_declared"
    """Weight provenance is the bound card, not a measured resident-memory claim."""
    local_replicas: tuple[AssessedReplica, ...] = ()
    external_configurations: tuple[ExternalConfiguration, ...] = ()
    provisioning: Literal["recommendation_only"] = "recommendation_only"
    """This interface performs no acquisition and grants no execution authority."""
