from typing import Literal, final

from pydantic import Field

from exo.shared.types.memory import Memory
from exo.shared.types.worker.runners import RunnerId
from exo.shared.types.worker.shards import LarqlExpertRange, LarqlPreset
from exo.utils.pydantic_ext import CamelCaseModel

LarqlReadinessStatus = Literal["ready", "not_ready", "failed"]


@final
class LarqlRunnerReadiness(CamelCaseModel):
    """Event-sourced readiness metadata for one supervised LARQL server."""

    runner_id: RunnerId = Field(description="Runner that owns this LARQL server.")
    vindex_uri: str = Field(description="Immutable source URI for the vindex artifact.")
    preset: LarqlPreset = Field(description="LARQL serving preset.")
    start_layer: int = Field(ge=0, description="Inclusive first served layer.")
    end_layer: int = Field(ge=0, description="Exclusive final served layer.")
    expert_range: LarqlExpertRange | None = Field(
        default=None,
        description="Optional half-open expert range for expert-server slices.",
    )
    units_manifest_path: str | None = Field(
        default=None,
        description="Optional local LARQL units manifest path.",
    )
    host: str = Field(description="Host where the local LARQL server listens.")
    port: int = Field(ge=1, le=65535, description="TCP port for the LARQL server.")
    status: LarqlReadinessStatus = Field(description="Current readiness state.")
    ram_footprint: Memory | None = Field(
        default=None,
        description="Measured resident memory for the LARQL process, when known.",
    )
    error_message: str | None = Field(
        default=None,
        description="Failure or readiness error detail, when unavailable.",
    )
