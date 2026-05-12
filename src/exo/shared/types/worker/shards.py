from enum import Enum
from typing import Literal, TypeAlias, final

from pydantic import Field, model_validator

from exo.shared.models.model_cards import ModelCard
from exo.utils.pydantic_ext import CamelCaseModel, TaggedModel

LarqlPreset = Literal["full", "expert-server"]


@final
class LarqlExpertRange(CamelCaseModel):
    """Half-open expert range served by one LARQL expert-server runner."""

    start_expert: int = Field(ge=0, description="Inclusive first expert index.")
    end_expert: int = Field(ge=0, description="Exclusive final expert index.")

    @model_validator(mode="after")
    def validate_non_empty(self) -> "LarqlExpertRange":
        """Require a non-empty expert interval."""

        if self.end_expert <= self.start_expert:
            raise ValueError("end_expert must be greater than start_expert")
        return self


class Sharding(str, Enum):
    Tensor = "Tensor"
    Pipeline = "Pipeline"


class BaseShardMetadata(TaggedModel):
    """
    Defines a specific shard of the model that is ready to be run on a device.
    Replaces previous `Shard` object.
    """

    model_card: ModelCard
    device_rank: int
    world_size: int

    # Error handling; equivalent to monkey-patch, but we can't monkey-patch runner.py
    # This is kinda annoying because it allocates memory in the ShardMetadata object. Can be rethought after Shanghai.
    immediate_exception: bool = False
    should_timeout: float | None = None

    start_layer: int = Field(ge=0)
    end_layer: int = Field(ge=0)
    n_layers: int = Field(ge=0)

    @property
    def is_first_layer(self) -> bool:
        return self.start_layer == 0

    @property
    def is_last_layer(self) -> bool:
        return self.end_layer == self.n_layers

    def __hash__(self) -> int:
        return hash(
            (
                self.model_card.model_id,
                self.start_layer,
                self.end_layer,
                self.n_layers,
                self.device_rank,
                self.world_size,
            )
        )


@final
class PipelineShardMetadata(BaseShardMetadata):
    """
    Pipeline parallelism shard meta.

    Layers are represented as a half-open interval [start_layer, end_layer),
    where start_layer is inclusive and end_layer is exclusive.
    """


@final
class CfgShardMetadata(BaseShardMetadata):
    """Shard metadata for CFG-parallel image generation models."""

    cfg_rank: int  # 0 = positive branch, 1 = negative branch
    cfg_world_size: int = 2

    # Pipeline-relative coordinates (computed at placement time)
    pipeline_rank: int  # rank within the pipeline group (0, 1, 2, ...)
    pipeline_world_size: int  # number of nodes per pipeline group


@final
class TensorShardMetadata(BaseShardMetadata):
    pass


@final
class LarqlShardMetadata(BaseShardMetadata):
    """Shard metadata for a future worker-managed LARQL cold-tier runner."""

    vindex_uri: str = Field(
        description="Immutable URI for the vindex directory artifact."
    )
    preset: LarqlPreset = Field(description="LARQL serving preset for this shard.")
    local_vindex_path: str | None = Field(
        default=None,
        description="Resolved local vindex directory after staging, if known.",
    )
    server_host: str = Field(
        default="127.0.0.1",
        description="Local bind host for the supervised LARQL HTTP server.",
    )
    server_port: int | None = Field(
        default=None,
        ge=1,
        le=65535,
        description="Requested LARQL port; omitted means allocate a free local port.",
    )
    expert_range: LarqlExpertRange | None = Field(
        default=None,
        description="Optional expert range for expert-server slices.",
    )
    units_manifest_path: str | None = Field(
        default=None,
        description="Optional LARQL units manifest path; mutually exclusive with expert_range.",
    )
    max_crash_restarts: int = Field(
        default=3,
        ge=0,
        description="Maximum ordinary crash restarts before terminal failure.",
    )
    readiness_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Maximum time to wait for LARQL readiness after process start.",
    )

    @model_validator(mode="after")
    def validate_slice_arguments(self) -> "LarqlShardMetadata":
        """Reject ambiguous expert selection for LARQL serve commands."""

        if self.expert_range is not None and self.units_manifest_path is not None:
            raise ValueError("expert_range and units_manifest_path are mutually exclusive")
        return self


ShardMetadata: TypeAlias = (
    PipelineShardMetadata | CfgShardMetadata | TensorShardMetadata | LarqlShardMetadata
)
