# pyright: reportArgumentType=false
"""Tests for the worker's first-status-report deadline (#272).

A runner frozen between spawn and its first status report stalls pre-init
coordination forever (the group-init gate waits for every rank, and the crash
breaker never trips because the process is alive). ``runners_never_reported``
finds those so the worker can give the instance up.
"""

from dataclasses import dataclass

from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.worker.instances import InstanceId, RunnerId
from skulk.worker.main import runners_never_reported

_DEADLINE = 120.0


@dataclass
class _StubCard:
    model_id: ModelId


@dataclass
class _StubShard:
    model_card: _StubCard


@dataclass
class _StubInstance:
    instance_id: InstanceId


@dataclass
class _StubBound:
    instance: _StubInstance


@dataclass
class _StubSupervisor:
    """Minimal stand-in exposing only what ``runners_never_reported`` reads."""

    instance_id: InstanceId
    model_id: ModelId
    has_reported_status: bool
    _created_monotonic: float

    @property
    def bound_instance(self) -> _StubBound:
        return _StubBound(instance=_StubInstance(instance_id=self.instance_id))

    @property
    def shard_metadata(self) -> _StubShard:
        return _StubShard(model_card=_StubCard(model_id=self.model_id))

    def seconds_since_created(self, now_monotonic: float) -> float:
        return now_monotonic - self._created_monotonic


def _runners(*supervisors: _StubSupervisor) -> dict[RunnerId, _StubSupervisor]:
    return {RunnerId(): s for s in supervisors}


def test_never_reported_past_deadline_is_flagged() -> None:
    iid = InstanceId()
    runners = _runners(
        _StubSupervisor(iid, ModelId("m"), has_reported_status=False, _created_monotonic=0.0)
    )
    out = runners_never_reported(runners, {iid}, now_monotonic=_DEADLINE + 1, deadline_seconds=_DEADLINE)
    assert out == [(iid, ModelId("m"))]


def test_reported_runner_is_never_flagged() -> None:
    iid = InstanceId()
    runners = _runners(
        _StubSupervisor(iid, ModelId("m"), has_reported_status=True, _created_monotonic=0.0)
    )
    out = runners_never_reported(runners, {iid}, now_monotonic=_DEADLINE + 100, deadline_seconds=_DEADLINE)
    assert out == []


def test_within_deadline_not_flagged() -> None:
    iid = InstanceId()
    runners = _runners(
        _StubSupervisor(iid, ModelId("m"), has_reported_status=False, _created_monotonic=0.0)
    )
    out = runners_never_reported(runners, {iid}, now_monotonic=_DEADLINE - 1, deadline_seconds=_DEADLINE)
    assert out == []


def test_instance_no_longer_live_is_skipped() -> None:
    # A supervisor whose instance was already removed must not be given up again.
    iid = InstanceId()
    runners = _runners(
        _StubSupervisor(iid, ModelId("m"), has_reported_status=False, _created_monotonic=0.0)
    )
    out = runners_never_reported(runners, set(), now_monotonic=_DEADLINE + 1, deadline_seconds=_DEADLINE)
    assert out == []
