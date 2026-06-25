from enum import Enum

from pydantic import Field

from skulk.api.types import (
    ImageEditsTaskParams,
    ImageGenerationTaskParams,
)
from skulk.shared.types.common import CommandId, Id, NodeId
from skulk.shared.types.embedding import TextEmbeddingTaskParams
from skulk.shared.types.text_generation import TextGenerationTaskParams
from skulk.shared.types.worker.instances import BoundInstance, InstanceId
from skulk.shared.types.worker.runners import RunnerId
from skulk.shared.types.worker.shards import ShardMetadata
from skulk.utils.pydantic_ext import TaggedModel


class TaskId(Id):
    pass


CANCEL_ALL_TASKS = TaskId("CANCEL_ALL_TASKS")


class TaskStatus(str, Enum):
    Pending = "Pending"
    Running = "Running"
    Complete = "Complete"
    TimedOut = "TimedOut"
    Failed = "Failed"
    Cancelled = "Cancelled"


class BaseTask(TaggedModel):
    task_id: TaskId = Field(default_factory=TaskId)
    task_status: TaskStatus = Field(default=TaskStatus.Pending)
    instance_id: InstanceId


class CreateRunner(BaseTask):  # emitted by Worker
    bound_instance: BoundInstance


class DownloadModel(BaseTask):  # emitted by Worker
    shard_metadata: ShardMetadata


class LoadModel(BaseTask):  # emitted by Worker
    pass


class ConnectToGroup(BaseTask):  # emitted by Worker
    pass


class StartWarmup(BaseTask):  # emitted by Worker
    pass


class TextGeneration(BaseTask):  # emitted by Master
    command_id: CommandId
    # The API node that owns this command; the rank-0 supervisor stamps it onto
    # each DataChunk so the Zenoh data plane can address output per-owner (#279
    # Phase 2). Optional so the gossipsub path is unaffected.
    owner_node: NodeId | None = None
    task_params: TextGenerationTaskParams
    trace_enabled: bool = False

    error_type: str | None = Field(default=None)
    error_message: str | None = Field(default=None)


class CancelTask(BaseTask):
    cancelled_task_id: TaskId
    runner_id: RunnerId


class ImageGeneration(BaseTask):  # emitted by Master
    command_id: CommandId
    owner_node: NodeId | None = None  # owning API node (#279 Phase 2; see TextGeneration)
    task_params: ImageGenerationTaskParams
    trace_enabled: bool = False

    error_type: str | None = Field(default=None)
    error_message: str | None = Field(default=None)


class ImageEdits(BaseTask):  # emitted by Master
    command_id: CommandId
    owner_node: NodeId | None = None  # owning API node (#279 Phase 2; see TextGeneration)
    task_params: ImageEditsTaskParams
    trace_enabled: bool = False

    error_type: str | None = Field(default=None)
    error_message: str | None = Field(default=None)


class TextEmbedding(BaseTask):  # emitted by Master
    command_id: CommandId
    owner_node: NodeId | None = None  # owning API node (#279 Phase 2; see TextGeneration)
    task_params: TextEmbeddingTaskParams
    trace_enabled: bool = False

    error_type: str | None = Field(default=None)
    error_message: str | None = Field(default=None)


class Shutdown(BaseTask):  # emitted by Worker
    runner_id: RunnerId


Task = (
    CreateRunner
    | DownloadModel
    | ConnectToGroup
    | LoadModel
    | StartWarmup
    | TextGeneration
    | CancelTask
    | ImageGeneration
    | ImageEdits
    | TextEmbedding
    | Shutdown
)
