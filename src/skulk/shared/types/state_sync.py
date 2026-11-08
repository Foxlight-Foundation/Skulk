from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from skulk.shared.types.common import NodeId, SessionId, SystemId
from skulk.shared.types.state import State
from skulk.utils.pydantic_ext import CamelCaseModel


class StateSnapshot(CamelCaseModel):
    """Serializable state snapshot used for follower bootstrap and persistence."""

    session_id: SessionId
    last_event_applied_idx: int = Field(ge=-1)
    state: State

    @model_validator(mode="after")
    def _validate_snapshot_index(self) -> "StateSnapshot":
        if self.state.last_event_applied_idx != self.last_event_applied_idx:
            raise ValueError(
                "State snapshot last_event_applied_idx must match embedded state"
            )
        return self


class StateSyncMessage(CamelCaseModel):
    """Cluster transport message for requesting or returning a state snapshot."""

    kind: Literal["request", "response"]
    requester: SystemId
    session_id: SessionId
    snapshot: StateSnapshot | None = None
    config_yaml: str | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "StateSyncMessage":
        if self.kind == "request" and (
            self.snapshot is not None or self.config_yaml is not None
        ):
            raise ValueError(
                "State sync requests cannot carry snapshots or config payloads"
            )
        if self.kind == "response" and self.snapshot is None:
            raise ValueError("State sync responses must carry a snapshot")
        return self


class LocalEventReceipt(CamelCaseModel):
    """Ephemeral request or master receipt for a consumed local source prefix.

    Receipts release ordered delivery storage, including rejected no-op task
    updates. They never enter replicated State or the durable event log.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["request", "response"]
    """Requests ask for progress; only the session master sends responses."""
    requester: NodeId
    """Node whose EventRouter owns the retained source."""
    source: SystemId
    """Producer ordering lineage within the elected session."""
    session_id: SessionId
    """Session that consumed the source, preventing cross-election release."""
    through_idx: int = Field(default=-1, ge=-1)
    """Last fully processed source index; requests leave this at minus one."""
    next_global_idx: int = Field(default=0, ge=0)
    """Master log length, allowing the recipient to request missing replay."""
