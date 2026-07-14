from datetime import timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from skulk.shared.types.common import Id, NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.shards import ShardMetadata
from skulk.utils.pydantic_ext import CamelCaseModel, TaggedModel


class DownloadProgressData(CamelCaseModel):
    total: Memory
    downloaded: Memory
    downloaded_this_session: Memory

    completed_files: int
    total_files: int

    speed: float
    eta_ms: int

    files: dict[str, "DownloadProgressData"]


class DownloadAttemptId(Id):
    """Opaque identity shared by transient and terminal status for one attempt."""


class BaseDownloadProgress(TaggedModel):
    node_id: NodeId
    shard_metadata: ShardMetadata
    model_directory: str = ""
    attempt_id: DownloadAttemptId | None = Field(
        default=None,
        description=(
            "Download attempt identity used to reject late transient telemetry "
            "after its terminal control event. Null is accepted for legacy logs."
        ),
    )


class DownloadPending(BaseDownloadProgress):
    downloaded: Memory = Memory()
    total: Memory = Memory()


class DownloadCompleted(BaseDownloadProgress):
    total: Memory
    read_only: bool = False


class DownloadFailed(BaseDownloadProgress):
    error_message: str


class DownloadOngoing(BaseDownloadProgress):
    download_progress: DownloadProgressData


DownloadProgress = (
    DownloadPending | DownloadCompleted | DownloadFailed | DownloadOngoing
)
LiveDownloadProgress = DownloadPending | DownloadOngoing
TerminalDownloadProgress = DownloadCompleted | DownloadFailed


class ModelSafetensorsIndexMetadata(BaseModel):
    total_size: PositiveInt


class ModelSafetensorsIndex(BaseModel):
    metadata: ModelSafetensorsIndexMetadata | None
    weight_map: dict[str, str]


class FileListEntry(BaseModel):
    type: Literal["file", "directory"]
    path: str
    size: int | None = None


class RepoFileDownloadProgress(BaseModel):
    repo_id: str
    repo_revision: str
    file_path: str
    downloaded: Memory
    downloaded_this_session: Memory
    total: Memory
    speed: float
    eta: timedelta
    status: Literal["not_started", "in_progress", "complete"]
    start_time: float

    model_config = ConfigDict(frozen=True)


class RepoDownloadProgress(BaseModel):
    repo_id: str
    repo_revision: str
    shard: ShardMetadata
    completed_files: int
    total_files: int
    downloaded: Memory
    downloaded_this_session: Memory
    total: Memory
    overall_speed: float
    overall_eta: timedelta
    status: Literal["not_started", "in_progress", "complete"]
    file_progress: dict[str, RepoFileDownloadProgress] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)
