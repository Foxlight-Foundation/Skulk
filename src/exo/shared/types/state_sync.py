from typing import Literal

from pydantic import Field, model_validator

from exo.shared.types.common import SessionId, SystemId
from exo.shared.types.state import State
from exo.utils.pydantic_ext import CamelCaseModel


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
