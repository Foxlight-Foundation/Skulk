"""Node-addressed trace payloads carried outside the ordered event log."""

from pydantic import ConfigDict, Field

from skulk.shared.types.common import NodeId
from skulk.shared.types.events import TraceEventData
from skulk.shared.types.tasks import TaskId
from skulk.utils.pydantic_ext import CamelCaseModel


class TraceDataPacket(CamelCaseModel):
    """One runner rank's complete task trace addressed to the owning API node."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    owner_node: NodeId
    source_node: NodeId
    task_id: TaskId
    rank: int = Field(ge=0)
    expected_ranks: tuple[int, ...] = Field(min_length=1)
    traces: tuple[TraceEventData, ...] = ()
