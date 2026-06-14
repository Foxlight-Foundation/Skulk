from pydantic import Field

from skulk.api.types import (
    ImageEditsTaskParams,
    ImageGenerationTaskParams,
)
from skulk.shared.models.model_cards import ModelCard, ModelId
from skulk.shared.types.chunks import InputImageChunk
from skulk.shared.types.common import CommandId, NodeId, SystemId
from skulk.shared.types.embedding import TextEmbeddingTaskParams
from skulk.shared.types.text_generation import TextGenerationTaskParams
from skulk.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from skulk.shared.types.worker.shards import Sharding, ShardMetadata
from skulk.utils.pydantic_ext import CamelCaseModel, TaggedModel


class BaseCommand(TaggedModel):
    command_id: CommandId = Field(default_factory=CommandId)


class TestCommand(BaseCommand):
    __test__ = False


class TextGeneration(BaseCommand):
    task_params: TextGenerationTaskParams


class ImageGeneration(BaseCommand):
    task_params: ImageGenerationTaskParams


class ImageEdits(BaseCommand):
    task_params: ImageEditsTaskParams


class TextEmbedding(BaseCommand):
    task_params: TextEmbeddingTaskParams


class SetTracingEnabled(BaseCommand):
    """Command to toggle runtime tracing for new requests cluster-wide."""

    enabled: bool


class PlaceInstance(BaseCommand):
    model_card: ModelCard
    sharding: Sharding
    instance_meta: InstanceMeta
    min_nodes: int
    # Per-placement node exclusions — the planner treats these nodes as if
    # they were absent from the topology when scoring this placement only.
    # Empty list (default) preserves the unfiltered behavior. Already-running
    # instances on these nodes are not affected — exclusion is purely a hint
    # to the candidate-cycle search for *this* placement.
    excluded_nodes: list[NodeId] = Field(default_factory=list)


class CreateInstance(BaseCommand):
    instance: Instance


class DeleteInstance(BaseCommand):
    instance_id: InstanceId


class RefuseInstancePlacement(BaseCommand):
    """Worker → master: a node cannot fit its shard for this instance at load
    time, so the master should re-place the model on a *wider* split rather
    than silently tearing it down (#290).

    The master's placement admission reads the gossiped (telemetry-plane,
    last-write-wins) ``ramAvailable``, while the worker's pre-spawn guard reads
    a fresh live ``vm_stat`` GPU-wireable figure at load time. On a borderline
    multi-node split the live reading can sit just under the admitted estimate,
    so the master admits a cycle the worker then refuses. Treating that refusal
    as a re-placement signal (one node wider, which shrinks every node's share)
    lets the cluster self-correct to a split that fits instead of leaving the
    operator with a placement that vanished without explanation.
    """

    instance_id: InstanceId
    # The node whose worker guard refused its shard, for diagnostics/logging.
    node_id: NodeId
    # The worker's refusal message (footprint vs. usable memory).
    reason: str


class TaskCancelled(BaseCommand):
    cancelled_command_id: CommandId


class TaskFinished(BaseCommand):
    finished_command_id: CommandId


class SendInputChunk(BaseCommand):
    """Command to send an input image chunk (converted to event by master)."""

    chunk: InputImageChunk


class RequestEventLog(BaseCommand):
    since_idx: int


class StartDownload(BaseCommand):
    target_node_id: NodeId
    shard_metadata: ShardMetadata


class DeleteDownload(BaseCommand):
    target_node_id: NodeId
    model_id: ModelId


class CancelDownload(BaseCommand):
    target_node_id: NodeId
    model_id: ModelId


class SyncConfig(BaseCommand):
    """Broadcast updated exo.yaml content to all nodes in the cluster."""

    config_yaml: str


class PurgeStagingCache(BaseCommand):
    """Broadcast command to purge staged model caches on all nodes."""

    model_id: ModelId | None = None


class RestartNode(BaseCommand):
    """Command to restart a specific node in the cluster."""

    target_node_id: NodeId


class AddCustomModelCard(BaseCommand):
    model_card: ModelCard


class DeleteCustomModelCard(BaseCommand):
    model_id: ModelId


DownloadCommand = (
    StartDownload
    | DeleteDownload
    | CancelDownload
    | SyncConfig
    | PurgeStagingCache
    | RestartNode
)

CustomModelCardCommand = AddCustomModelCard | DeleteCustomModelCard


Command = (
    TestCommand
    | RequestEventLog
    | TextGeneration
    | ImageGeneration
    | ImageEdits
    | TextEmbedding
    | SetTracingEnabled
    | PlaceInstance
    | CreateInstance
    | DeleteInstance
    | RefuseInstancePlacement
    | TaskCancelled
    | TaskFinished
    | SendInputChunk
    | AddCustomModelCard
    | DeleteCustomModelCard
)


class ForwarderCommand(CamelCaseModel):
    origin: SystemId
    command: Command


class ForwarderDownloadCommand(CamelCaseModel):
    origin: SystemId
    command: DownloadCommand
