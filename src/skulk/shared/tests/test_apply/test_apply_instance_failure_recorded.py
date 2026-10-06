from datetime import datetime, timezone
from typing import cast

import pytest
from pydantic import TypeAdapter

from skulk.shared.apply import apply_instance_failure_recorded
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.events import (
    Event,
    InstanceFailureRecorded,
    StateSnapshotHydrated,
)
from skulk.shared.types.state import State
from skulk.shared.types.worker.instances import (
    INSTANCE_FAILURE_HISTORY_LIMIT,
    InstanceFailure,
    InstanceId,
)


def _failure(index: int) -> InstanceFailure:
    return InstanceFailure(
        instance_id=InstanceId(f"instance-{index}"),
        model_id=ModelId(f"org/model-{index}"),
        error_code="runner_crashed",
        error_message=f"runner failed {index}",
        affected_node_ids=(NodeId("node-b"), NodeId("node-a")),
        recorded_at=datetime(2026, 8, 15, 12, index % 60, tzinfo=timezone.utc),
    )


def test_failure_history_is_newest_first_deduplicated_and_bounded() -> None:
    """A retrying rank cannot grow replicated failure state without bound."""
    state = State()
    for index in range(INSTANCE_FAILURE_HISTORY_LIMIT + 5):
        state = apply_instance_failure_recorded(
            InstanceFailureRecorded(failure=_failure(index)), state
        )

    assert len(state.instance_failures) == INSTANCE_FAILURE_HISTORY_LIMIT
    assert state.instance_failures[0].instance_id == InstanceId(
        f"instance-{INSTANCE_FAILURE_HISTORY_LIMIT + 4}"
    )
    assert state.instance_failures[-1].instance_id == InstanceId("instance-5")

    replacement = _failure(INSTANCE_FAILURE_HISTORY_LIMIT + 4).model_copy(
        update={"error_message": "newer detail"}
    )
    state = apply_instance_failure_recorded(
        InstanceFailureRecorded(failure=replacement), state
    )
    assert len(state.instance_failures) == INSTANCE_FAILURE_HISTORY_LIMIT
    assert state.instance_failures[0].error_message == "newer detail"
    assert state.instance_failures[0].affected_node_ids == (
        NodeId("node-a"),
        NodeId("node-b"),
    )
    with pytest.raises(AttributeError):
        cast("list[InstanceFailure]", cast(object, state.instance_failures)).append(
            _failure(99)
        )


def test_failure_event_round_trips_through_persisted_wire_shape() -> None:
    """Strict event-log replay restores nested timestamps from serialized JSON."""
    event = InstanceFailureRecorded(failure=_failure(1))
    adapter: TypeAdapter[Event] = TypeAdapter(Event)

    restored = adapter.validate_json(adapter.dump_json(event))

    assert isinstance(restored, InstanceFailureRecorded)
    assert restored.failure == event.failure


def test_failure_history_round_trips_through_snapshot_wire_shape() -> None:
    """JSON array snapshots restore failure history as an immutable tuple."""
    event = StateSnapshotHydrated(state=State(instance_failures=(_failure(1),)))
    adapter: TypeAdapter[Event] = TypeAdapter(Event)

    restored = adapter.validate_json(adapter.dump_json(event))

    assert isinstance(restored, StateSnapshotHydrated)
    assert restored.state.instance_failures == (_failure(1),)
    assert isinstance(restored.state.instance_failures, tuple)
