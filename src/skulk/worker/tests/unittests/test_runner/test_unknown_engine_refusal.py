"""A placement for an engine this build does not have is refused, never served by a default runner."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import loguru
import pytest

from skulk.shared.backends import ENGINE_UNKNOWN_FAILURE_MARKER
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.audio import RealtimeAudioInputFrame
from skulk.shared.types.common import NodeId
from skulk.shared.types.diagnostics import RunnerDiagnosticUpdate
from skulk.shared.types.events import Event, RunnerStatusUpdated
from skulk.shared.types.memory import Memory
from skulk.shared.types.tasks import Task, TaskId
from skulk.shared.types.worker.instances import BoundInstance, InstanceId
from skulk.shared.types.worker.runners import RunnerFailed, RunnerId, RunnerReady
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.channels import mp_channel
from skulk.worker.main import engine_unknown_live_instances
from skulk.worker.plan import _retried_on_failure  # pyright: ignore[reportPrivateUsage]
from skulk.worker.runner import bootstrap
from skulk.worker.runner.runner_supervisor import RunnerSupervisor
from skulk.worker.tests.unittests.conftest import get_mlx_ring_instance

NODE = NodeId("node-a")
RUNNER = RunnerId("runner-a")
MODEL = ModelId("org/model")


def _bound(resolved_backend: str | None, task: ModelTask) -> BoundInstance:
    """One single-node placement stamped with the given backend."""
    shard = PipelineShardMetadata(
        model_card=ModelCard(
            model_id=MODEL,
            storage_size=Memory.from_mb(1000),
            n_layers=4,
            hidden_size=64,
            supports_tensor=False,
            tasks=[task],
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=4,
        n_layers=4,
        resolved_backend=resolved_backend,
    )
    instance = get_mlx_ring_instance(
        InstanceId("instance-a"), MODEL, {NODE: RUNNER}, {RUNNER: shard}
    )
    return BoundInstance(instance=instance, bound_runner_id=RUNNER, bound_node_id=NODE)


def _skip_process_setup(*_arguments: object, **_keywords: object) -> None:
    """Keep signal, limit, watchdog and diagnostic effects out of an in-process run."""


@pytest.mark.parametrize("task", [ModelTask.TextGeneration, ModelTask.TextEmbedding])
def test_the_runner_refuses_an_unknown_engine_before_choosing_a_runner(
    task: ModelTask, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Text and embedding placements both used to reach an MLX runner."""
    events: list[Event] = []
    reached: list[str] = []

    def default_runner_reached(*_arguments: object, **_keywords: object) -> None:
        reached.append("default runner")
        raise AssertionError("an unknown engine reached a default runner")

    event_sender, event_receiver = mp_channel[Event]()
    diagnostic_sender, diagnostic_receiver = mp_channel[RunnerDiagnosticUpdate]()
    task_sender, task_receiver = mp_channel[Task]()
    cancel_sender, cancel_receiver = mp_channel[TaskId]()
    audio_sender, audio_receiver = mp_channel[RealtimeAudioInputFrame]()
    monkeypatch.setattr(event_sender, "send", events.append)
    monkeypatch.setattr(bootstrap.signal, "signal", _skip_process_setup)
    monkeypatch.setattr(bootstrap.resource, "setrlimit", _skip_process_setup)
    monkeypatch.setenv("MLX_METAL_FAST_SYNCH", "0")
    monkeypatch.setattr(bootstrap, "logger", bootstrap.logger)
    for name in (
        "install_stack_dump_signal",
        "_install_parent_death_watchdog",
        "configure_runner_diagnostics",
        "record_runner_phase",
        "_release_metal_resources",
        # Card trust is a separate, earlier check with its own tests.
        "require_remote_code_approval",
    ):
        monkeypatch.setattr(bootstrap, name, _skip_process_setup)
    # The text path resolves its engine before it picks a runner; that must
    # not happen for a placement this build cannot run. The embedding path
    # has no such hook, so there the single marked failure below is the proof.
    monkeypatch.setattr(bootstrap, "_resolve_text_engine", default_runner_reached)
    try:
        bootstrap.entrypoint(
            _bound("tensorfold-rocm", task),
            event_sender,
            diagnostic_sender,
            task_receiver,
            cancel_receiver,
            audio_receiver,
            loguru.logger,
        )
    finally:
        for channel_end in (
            event_sender, diagnostic_sender, task_sender, cancel_sender, audio_sender,
            event_receiver, diagnostic_receiver, task_receiver, cancel_receiver,
            audio_receiver,
        ):
            channel_end.close()

    failures = [
        event.runner_status
        for event in events
        if isinstance(event, RunnerStatusUpdated)
        and isinstance(event.runner_status, RunnerFailed)
    ]
    assert reached == []
    assert len(failures) == 1
    message = failures[0].error_message or ""
    assert message.startswith(f"{ENGINE_UNKNOWN_FAILURE_MARKER}: ")
    assert "'tensorfold-rocm'" in message


def test_an_unknown_engine_failure_is_not_relaunched() -> None:
    refused = RunnerFailed(error_message=f"{ENGINE_UNKNOWN_FAILURE_MARKER}: no engine")

    assert not _retried_on_failure(refused)
    assert _retried_on_failure(RunnerFailed(error_message="ordinary crash"))
    assert not _retried_on_failure(RunnerReady())


def test_the_worker_gives_up_only_live_instances_refused_for_an_unknown_engine() -> None:
    def supervisor(instance_id: str, message: str) -> object:
        return SimpleNamespace(
            bound_instance=SimpleNamespace(
                instance=SimpleNamespace(instance_id=instance_id)
            ),
            shard_metadata=SimpleNamespace(model_card=SimpleNamespace(model_id=MODEL)),
            status=RunnerFailed(error_message=message),
        )

    refused = f"{ENGINE_UNKNOWN_FAILURE_MARKER}: this node has no engine"
    runners = cast(
        "dict[RunnerId, RunnerSupervisor]",
        cast(
            object,
            {
                "runner-a": supervisor("instance-a", refused),
                "runner-b": supervisor("instance-b", "ordinary crash"),
                "runner-c": supervisor("instance-gone", refused),
            },
        ),
    )

    assert engine_unknown_live_instances(
        runners, {InstanceId("instance-a"), InstanceId("instance-b")}
    ) == [("instance-a", MODEL, refused)]
