# pyright: reportPrivateUsage=false
"""Recovery credits late operator placements once while library information arrives."""

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

import pytest

from skulk.master.placement import PlacementInfoPendingError
from skulk.master.tests.test_restart_recovery import (
    LOCAL_NODE,
    _advance,
    _cold_master_with,
    _Planner,
)
from skulk.recovery.restore import INFO_WAIT_AFTER_SETTLE_SECONDS
from skulk.recovery.tests.fixtures import sample_instance
from skulk.shared.types.commands import PlaceInstance
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import InstanceCreated, RestartRecoveryCompleted
from skulk.shared.types.worker.instances import Instance, InstanceId


@pytest.mark.parametrize("reserved", [False, True])
async def test_operator_mount_during_information_wait_satisfies_restore(
    monkeypatch: pytest.MonkeyPatch, *, reserved: bool
) -> None:
    """An accepted or indexed mount must not be duplicated by the next restore pass."""
    planner = _Planner(pending=True)
    master = await _cold_master_with(monkeypatch, "org/music", planner=planner)
    _advance(master)
    mounted = sample_instance("operator-mount", "org/music", (str(LOCAL_NODE),))
    if reserved:
        master._pending_instance_reservations[mounted.instance_id] = mounted
    else:
        master.state = master.state.model_copy(
            update={"instances": {mounted.instance_id: mounted}}
        )
    monkeypatch.setattr(planner, "_pending", False)

    events = _advance(master)

    assert not any(isinstance(event, InstanceCreated) for event in events)
    assert isinstance(events[-1], RestartRecoveryCompleted)
    assert events[-1].restored_placements == 0


async def test_existing_placement_is_not_credited_again_after_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two recorded copies and one existing copy still require one restoration."""
    planner = _Planner(pending=True)
    master = await _cold_master_with(
        monkeypatch, "org/music", "org/music", planner=planner
    )
    mounted = sample_instance("existing-copy", "org/music", (str(LOCAL_NODE),))
    master.state = master.state.model_copy(
        update={"instances": {mounted.instance_id: mounted}}
    )
    _advance(master)
    monkeypatch.setattr(planner, "_pending", False)

    events = _advance(master)

    assert sum(isinstance(event, InstanceCreated) for event in events) == 1
    assert isinstance(events[-1], RestartRecoveryCompleted)
    assert events[-1].restored_placements == 1


async def test_partial_restoration_and_late_mount_keep_recorded_multiplicity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A previously restored copy cannot consume a second remaining intent."""
    planner = _Planner()
    master = await _cold_master_with(
        monkeypatch, "org/music", "org/music", "org/music", planner=planner
    )
    blocked = True
    attempts = 0

    def place(
        command: PlaceInstance,
        current: Mapping[InstanceId, Instance],
        *,
        stamped_exclusions: set[NodeId] | None = None,
    ) -> dict[InstanceId, Instance]:
        nonlocal attempts
        attempts += 1
        if attempts > 1 and blocked:
            raise PlacementInfoPendingError("waiting for the next library observation")
        return planner(command, current)

    monkeypatch.setattr(master, "_place_for_steward_action", place)
    first = _advance(master)
    restored = next(event.instance for event in first if isinstance(event, InstanceCreated))
    mounted = sample_instance("late-copy", "org/music", (str(LOCAL_NODE),))
    master.state = master.state.model_copy(
        update={"instances": {restored.instance_id: restored, mounted.instance_id: mounted}}
    )
    blocked = False

    events = _advance(master)

    assert sum(isinstance(event, InstanceCreated) for event in events) == 1
    assert isinstance(events[-1], RestartRecoveryCompleted)
    assert events[-1].restored_placements == 2


async def test_late_mount_gives_the_next_intent_its_own_information_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Crediting a pending model cannot expire the next model's fresh wait budget."""
    planner = _Planner(pending=True)
    master = await _cold_master_with(
        monkeypatch, "org/music", "org/second", planner=planner
    )
    start = datetime.now(tz=timezone.utc)
    _advance(master, now=start)
    mounted = sample_instance("late-copy", "org/music", (str(LOCAL_NODE),))
    master.state = master.state.model_copy(
        update={"instances": {mounted.instance_id: mounted}}
    )

    events = _advance(
        master, now=start + timedelta(seconds=INFO_WAIT_AFTER_SETTLE_SECONDS - 1)
    )

    assert events == []
    assert _advance(
        master, now=start + timedelta(seconds=INFO_WAIT_AFTER_SETTLE_SECONDS + 1)
    ) == []


async def test_late_dying_mount_cannot_satisfy_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mount already selected for teardown cannot consume the missing model."""
    planner = _Planner(pending=True)
    master = await _cold_master_with(monkeypatch, "org/music", planner=planner)
    _advance(master)
    mounted = sample_instance("dying-copy", "org/music", (str(LOCAL_NODE),))
    master.state = master.state.model_copy(
        update={"instances": {mounted.instance_id: mounted}}
    )
    monkeypatch.setattr(planner, "_pending", False)

    events = _advance(master, dying={mounted.instance_id})

    assert sum(isinstance(event, InstanceCreated) for event in events) == 1
