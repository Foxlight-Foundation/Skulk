from enum import Enum

from pydantic import model_validator

from skulk.shared.models.model_cards import ModelTask
from skulk.shared.types.common import Host, Id, NodeId
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments, ShardMetadata
from skulk.utils.pydantic_ext import CamelCaseModel, TaggedModel


class InstanceId(Id):
    pass


class InstanceMeta(str, Enum):
    MlxRing = "MlxRing"
    MlxJaccl = "MlxJaccl"


class BaseInstance(TaggedModel):
    instance_id: InstanceId
    shard_assignments: ShardAssignments
    # Context-admission ceiling (#145/#279 slice 2): the master computes this
    # ONCE at placement time and stamps it here, in the event-sourced placement
    # decision. Every rank then reads the identical value off replicated state
    # instead of recomputing from per-node memory — required now that node
    # memory lives in the (last-write-wins, unordered) telemetry plane rather
    # than the ordered event log, where divergent per-rank ceilings would
    # deadlock the collectives. ``None`` means no enforceable ceiling.
    context_token_limit: int | None = None

    def shard(self, runner_id: RunnerId) -> ShardMetadata | None:
        return self.shard_assignments.runner_to_shard.get(runner_id, None)

    @model_validator(mode="after")
    def _backfill_context_token_limit(self) -> "BaseInstance":
        """Give legacy/hydrated instances a context-admission ceiling.

        Instances created before #279 slice 2 (replayed from an old event log
        or snapshot, or accepted via an older ``CreateInstance`` payload) carry
        ``context_token_limit=None``, which both the API pre-flight and the
        runner treat as "no ceiling" — so an upgraded cluster would serve those
        existing placements without the #145 admission guard, letting oversized
        requests reach MLX and OOM instead of getting a clean
        ``context_length_exceeded``. Fall back to the card's advertised
        ``context_length`` (static, deterministic across ranks). Freshly placed
        instances are already stamped by the master (placement's
        ``instance_context_token_limit`` itself returns the card limit when no
        memory-derived ceiling applies), so this only fills the legacy gap.
        """
        if self.context_token_limit is None:
            for shard in self.shard_assignments.runner_to_shard.values():
                if shard.model_card.context_length > 0:
                    self.context_token_limit = shard.model_card.context_length
                break
        return self


class MlxRingInstance(BaseInstance):
    hosts_by_node: dict[NodeId, list[Host]]
    ephemeral_port: int


class MlxJacclInstance(BaseInstance):
    jaccl_devices: list[list[str | None]]
    jaccl_coordinators: dict[NodeId, str]


# TODO: Single node instance
Instance = MlxRingInstance | MlxJacclInstance


class BoundInstance(CamelCaseModel):
    instance: Instance
    bound_runner_id: RunnerId
    bound_node_id: NodeId

    @property
    def bound_shard(self) -> ShardMetadata:
        shard = self.instance.shard(self.bound_runner_id)
        assert shard is not None
        return shard

    @property
    def is_image_model(self) -> bool:
        return (
            ModelTask.TextToImage in self.bound_shard.model_card.tasks
            or ModelTask.ImageToImage in self.bound_shard.model_card.tasks
        )

    @property
    def is_embedding_model(self) -> bool:
        return ModelTask.TextEmbedding in self.bound_shard.model_card.tasks

    @model_validator(mode="after")
    def validate_shard_exists(self) -> "BoundInstance":
        assert (
            self.bound_runner_id in self.instance.shard_assignments.runner_to_shard
        ), (
            "Bound Instance must be constructed with a runner_id that is in the instances assigned shards"
        )
        return self
