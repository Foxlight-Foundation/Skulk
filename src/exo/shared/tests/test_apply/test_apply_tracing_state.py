"""Tests for applying tracing state events."""

from exo.shared.apply import apply
from exo.shared.types.events import IndexedEvent, TracingStateChanged
from exo.shared.types.state import State


def test_apply_tracing_state_changed_updates_cluster_state() -> None:
    """TracingStateChanged should update the immutable cluster state toggle."""

    enabled_state = apply(
        State(),
        IndexedEvent(idx=0, event=TracingStateChanged(enabled=True)),
    )

    assert enabled_state.tracing_enabled is True
    assert enabled_state.last_event_applied_idx == 0

    disabled_state = apply(
        enabled_state,
        IndexedEvent(idx=1, event=TracingStateChanged(enabled=False)),
    )

    assert disabled_state.tracing_enabled is False
    assert disabled_state.last_event_applied_idx == 1
