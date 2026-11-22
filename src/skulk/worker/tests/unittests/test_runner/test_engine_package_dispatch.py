"""A placement on a held engine package reaches the generic runner; a refused load is given up."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import loguru
import pytest

from skulk.shared.backends import engine_load_refusal
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.audio import RealtimeAudioInputFrame
from skulk.shared.types.common import NodeId
from skulk.shared.types.diagnostics import RunnerDiagnosticUpdate
from skulk.shared.types.events import Event, RunnerStatusUpdated
from skulk.shared.types.memory import Memory
from skulk.shared.types.tasks import Task, TaskId
from skulk.shared.types.worker.instances import BoundInstance, InstanceId
from skulk.shared.types.worker.runners import RunnerFailed, RunnerId
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.channels import mp_channel
from skulk.worker.main import (
    _engine_load_refused_instance_failure_message,  # pyright: ignore[reportPrivateUsage]
    engine_load_refused_live_instances,
)
from skulk.worker.runner import bootstrap
from skulk.worker.runner.engine_package import runner as runner_module
from skulk.worker.runner.runner_supervisor import RunnerSupervisor
from skulk.worker.tests.unittests.conftest import get_mlx_ring_instance

NODE = NodeId("node-a")
RUNNER = RunnerId("runner-a")
MODEL = ModelId("org/model")
BUILD = "a" * 64


def _bound() -> BoundInstance:
    shard = PipelineShardMetadata(
        model_card=ModelCard(
            model_id=MODEL,
            storage_size=Memory.from_mb(1000),
            n_layers=4,
            hidden_size=64,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=4,
        n_layers=4,
        resolved_backend="tensorfold-rocm",
        resolved_engine_build=BUILD,
    )
    instance = get_mlx_ring_instance(
        InstanceId("instance-a"), MODEL, {NODE: RUNNER}, {RUNNER: shard}
    )
    return BoundInstance(instance=instance, bound_runner_id=RUNNER, bound_node_id=NODE)


def _skip_process_setup(*_arguments: object, **_keywords: object) -> None:
    """Keep signal, limit, watchdog and diagnostic effects out of an in-process run."""


def test_a_placement_on_a_held_package_reaches_the_generic_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stamped backend and build select the package; no built-in runner is consulted."""
    package = object()
    looked_up: list[tuple[str | None, str | None]] = []
    started: list[tuple[object, int | None]] = []
    events: list[Event] = []

    def lookup(backend: str | None, build: str | None) -> object:
        looked_up.append((backend, build))
        return package

    class RecordingRunner:
        """Stand in for the generic runner; record what it was built with."""

        def __init__(
            self,
            _bound_instance: BoundInstance,
            _events: object,
            _tasks: object,
            _cancels: object,
            installed: object,
            context_token_limit: int | None = None,
        ) -> None:
            self.installed = installed
            self.context_token_limit = context_token_limit

        def main(self) -> None:
            started.append((self.installed, self.context_token_limit))

    def built_in_reached(*_arguments: object, **_keywords: object) -> None:
        raise AssertionError("a package placement reached built-in engine resolution")

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
        "require_remote_code_approval",
    ):
        monkeypatch.setattr(bootstrap, name, _skip_process_setup)
    monkeypatch.setattr(bootstrap, "installed_engine_package", lookup)
    monkeypatch.setattr(bootstrap, "_resolve_text_engine", built_in_reached)
    monkeypatch.setattr(runner_module, "Runner", RecordingRunner)
    try:
        bootstrap.entrypoint(
            _bound(),
            event_sender,
            diagnostic_sender,
            task_receiver,
            cancel_receiver,
            audio_receiver,
            loguru.logger,
            context_token_limit=4096,
        )
    finally:
        for channel_end in (
            event_sender,
            diagnostic_sender,
            task_sender,
            cancel_sender,
            audio_sender,
            event_receiver,
            diagnostic_receiver,
            task_receiver,
            cancel_receiver,
            audio_receiver,
        ):
            channel_end.close()

    assert looked_up == [("tensorfold-rocm", BUILD)]
    assert started == [(package, 4096)]
    assert not [
        event
        for event in events
        if isinstance(event, RunnerStatusUpdated)
        and isinstance(event.runner_status, RunnerFailed)
    ]


def test_the_worker_gives_up_a_refused_load_with_the_code_only() -> None:
    """The engine's own words may name local paths, so only its code is kept."""

    def supervisor(instance_id: str, message: str) -> object:
        return SimpleNamespace(
            bound_instance=SimpleNamespace(
                instance=SimpleNamespace(instance_id=instance_id)
            ),
            shard_metadata=SimpleNamespace(model_card=SimpleNamespace(model_id=MODEL)),
            status=RunnerFailed(error_message=message),
        )

    refused = engine_load_refusal("load_refused", "cannot map /private/weights.bin")
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

    found = engine_load_refused_live_instances(
        runners, {InstanceId("instance-a"), InstanceId("instance-b")}
    )

    assert found == [("instance-a", MODEL, "load_refused", refused)]
    durable = _engine_load_refused_instance_failure_message(MODEL, found[0][2])
    assert "load_refused" in durable and "/private" not in durable
