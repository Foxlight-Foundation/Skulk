from pydantic import Field

from exo.api.types import (
    ImageEditsTaskParams,
    ImageGenerationTaskParams,
)
from exo.shared.models.model_cards import ModelCard, ModelId
from exo.shared.types.chunks import InputImageChunk
from exo.shared.types.common import CommandId, NodeId, SystemId
from exo.shared.types.embedding import TextEmbeddingTaskParams
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from exo.shared.types.worker.shards import Sharding, ShardMetadata
from exo.utils.pydantic_ext import CamelCaseModel, TaggedModel


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


class CreateInstance(BaseCommand):
    instance: Instance


class DeleteInstance(BaseCommand):
    instance_id: InstanceId


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
    StartDownload | DeleteDownload | CancelDownload | SyncConfig | PurgeStagingCache | RestartNode
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
