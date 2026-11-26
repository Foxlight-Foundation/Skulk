# pyright: reportPrivateUsage=false, reportAny=false
"""The worker's side of an engine package change: make it, re-read its facts, answer once."""

from __future__ import annotations

import time
from collections.abc import Container
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

import skulk.engines.actions as actions
import skulk.facts as facts
from skulk.engines.load_results import EngineLoadResults, LoadObservation
from skulk.provisioning.engine_packages import EnginePackageRefusedError
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.engine_packages import NodeEnginePackage
from skulk.shared.types.events import (
    EnginePackageChangeCompleted,
    EnginePackageChangeRequested,
)
from skulk.shared.types.profiling import NodeResources
from skulk.shared.types.worker.runners import RunnerId
from skulk.worker.main import Worker, engine_package_builds_in_use
from skulk.worker.runner.runner_supervisor import RunnerSupervisor

BUILD = "c" * 64
OLDER = "b" * 64
RESOURCES = NodeResources(
    engine_packages=(
        NodeEnginePackage(
            engine_id="foxlight.tensorfold",
            lane="rocm",
            backend_tag="tensorfold-rocm",
            status="active",
            build=BUILD,
            installed_builds=(OLDER, BUILD),
            rollback_build=OLDER,
        ),
    )
)


def _runner(tag: str | None, build: str | None) -> object:
    return SimpleNamespace(
        shard_metadata=SimpleNamespace(resolved_backend=tag, resolved_engine_build=build),
        load_observation=None,
    )


def _runners(*supervisors: object) -> dict[RunnerId, RunnerSupervisor]:
    return cast(
        "dict[RunnerId, RunnerSupervisor]",
        cast(object, {f"runner-{index}": each for index, each in enumerate(supervisors)}),
    )


def _worker(*supervisors: object) -> Any:
    worker: Any = object.__new__(Worker)
    worker.node_id = NodeId("worker-node")
    worker.runners = _runners(*supervisors)
    worker._on_demand_backends = frozenset({"stale"})
    worker._telemetry_sender = SimpleNamespace(send=AsyncMock())
    worker.event_sender = SimpleNamespace(send=AsyncMock())
    worker._gather_node_resources = AsyncMock(return_value=RESOURCES)
    return worker


def _request(*, expires_in: float = 60.0) -> EnginePackageChangeRequested:
    return EnginePackageChangeRequested(
        request_id=CommandId("change"),
        target_node=NodeId("worker-node"),
        owner_node=NodeId("api-node"),
        engine_id="foxlight.tensorfold",
        lane="rocm",
        action="activate",
        build=BUILD,
        expires_at=time.time() + expires_in,
    )


def _answer(worker: Any) -> EnginePackageChangeCompleted:
    worker.event_sender.send.assert_awaited_once()
    answer = worker.event_sender.send.await_args.args[0]
    assert isinstance(answer, EnginePackageChangeCompleted)
    assert (answer.request_id, answer.target_node, answer.owner_node) == (
        CommandId("change"),
        NodeId("worker-node"),
        NodeId("api-node"),
    )
    return answer


def _fresh_facts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stand in for the facts re-read and record that it happened after the change."""
    steps: list[str] = []
    monkeypatch.setattr(facts, "refresh_node_facts", lambda: steps.append("facts"))
    monkeypatch.setattr(
        facts,
        "current_backend_derivation",
        lambda: SimpleNamespace(on_demand_backends=frozenset({"fresh"})),
    )
    return steps


def test_only_runners_on_the_lane_hold_its_builds() -> None:
    runners = _runners(
        _runner("tensorfold-rocm", BUILD),
        _runner("tensorfold-rocm", BUILD),
        _runner("tensorfold-cuda", OLDER),
        _runner("llama_server-vulkan", None),
        _runner(None, None),
    )

    assert engine_package_builds_in_use(runners, "tensorfold-rocm") == frozenset({BUILD})
    assert engine_package_builds_in_use(runners, "tensorfold-metal") == frozenset()


def test_a_runner_with_no_stamped_build_holds_every_build_of_its_lane() -> None:
    """It runs on whichever build was active when it started, which is not recorded."""
    in_use = engine_package_builds_in_use(
        _runners(_runner("tensorfold-rocm", None)), "tensorfold-rocm"
    )

    assert BUILD in in_use and OLDER in in_use


async def test_a_change_is_made_then_facts_are_read_again_then_it_is_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    steps = _fresh_facts(monkeypatch)
    seen: dict[str, object] = {}

    def change(
        engine_id: str,
        lane: str,
        action: str,
        build: str | None,
        *,
        builds_in_use: Container[str],
    ) -> None:
        steps.append("change")
        seen.update(asked=(engine_id, lane, action, build), in_use=builds_in_use)

    monkeypatch.setattr(actions, "change_engine_package", change)
    worker = _worker(_runner("tensorfold-rocm", OLDER))

    await worker._change_engine_package(_request())

    assert steps == ["change", "facts"]
    assert seen == {
        "asked": ("foxlight.tensorfold", "rocm", "activate", BUILD),
        "in_use": frozenset({OLDER}),
    }
    assert worker._on_demand_backends == frozenset({"fresh"})
    worker._telemetry_sender.send.assert_awaited_once()
    answer = _answer(worker)
    assert (answer.success, answer.code, answer.resources) == (True, None, RESOURCES)


async def test_a_refusal_is_answered_with_its_code_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reason can name a local path, so only the code leaves the node."""
    steps = _fresh_facts(monkeypatch)

    def refuse(*_arguments: object, **_keywords: object) -> None:
        raise EnginePackageRefusedError(
            "installation_damaged", "/private/engines/x differs from its inventory"
        )

    monkeypatch.setattr(actions, "change_engine_package", refuse)
    worker = _worker()

    await worker._change_engine_package(_request())

    assert steps == []
    worker._telemetry_sender.send.assert_not_awaited()
    answer = _answer(worker)
    assert (answer.success, answer.code, answer.resources) == (
        False,
        "installation_damaged",
        None,
    )
    assert "/private" not in answer.model_dump_json()


async def test_an_unexpected_failure_is_still_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fresh_facts(monkeypatch)

    def fail(*_arguments: object, **_keywords: object) -> None:
        raise OSError("disk went away at /private/engines")

    monkeypatch.setattr(actions, "change_engine_package", fail)
    worker = _worker()

    await worker._change_engine_package(_request())

    answer = _answer(worker)
    assert (answer.success, answer.code, answer.resources) == (False, "change_failed", None)
    assert "/private" not in answer.model_dump_json()


async def test_a_request_past_its_deadline_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request replayed from the log long after it was made must not act."""
    called: list[str] = []

    def change(*_arguments: object, **_keywords: object) -> None:
        called.append("change")

    monkeypatch.setattr(actions, "change_engine_package", change)
    worker = _worker()

    await worker._change_engine_package(_request(expires_in=-1.0))

    assert called == []
    worker.event_sender.send.assert_not_awaited()


def _loaded_runner(at_unix_ms: int) -> object:
    return SimpleNamespace(
        shard_metadata=SimpleNamespace(
            model_card=SimpleNamespace(model_id="org/model"),
            resolved_backend="tensorfold-rocm",
            resolved_engine_build=BUILD,
        ),
        load_observation=LoadObservation("loaded", None, at_unix_ms),
    )


def test_a_load_that_ended_is_new_once_and_then_already_reported() -> None:
    """The planning loop asks every tick; only a new result is worth a report."""
    worker = _worker(_loaded_runner(1000))
    worker._engine_load_results = EngineLoadResults()

    assert worker._note_engine_package_loads() is True
    assert worker._note_engine_package_loads() is False

    worker.runners = _runners(_loaded_runner(2000), _runner("llama_server-vulkan", None))
    assert worker._note_engine_package_loads() is True
    assert [
        result.at_unix_ms
        for result in worker._engine_load_results.for_tag("tensorfold-rocm")
    ] == [2000]


async def test_a_load_report_that_cannot_be_sent_never_reaches_the_planning_loop() -> None:
    worker = _worker()
    worker._publish_fresh_node_resources = AsyncMock(side_effect=OSError("no route"))

    await worker._report_engine_package_loads()

    worker._publish_fresh_node_resources.assert_awaited_once()
