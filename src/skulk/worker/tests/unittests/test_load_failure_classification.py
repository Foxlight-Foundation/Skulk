# pyright: reportPrivateUsage=false
"""LoadModel dispatch classifies missing model files apart from trust refusals.

The formatter and selector tests cover each message on its own; these drive
the worker's real ``_start_runner_task`` so a regression that routes a
``FileNotFoundError`` back to the trust message fails here.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from skulk.download.download_utils import MODEL_FILES_INCOMPLETE_MARKER
from skulk.routing.router import get_node_id_keypair
from skulk.shared.models.remote_code_approval import MODEL_TRUST_FAILURE_MARKER
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.events import (
    Event,
    IndexedEvent,
    RunnerStatusUpdated,
    TaskStatusUpdated,
)
from skulk.shared.types.state import State
from skulk.shared.types.tasks import LoadModel, TaskStatus
from skulk.shared.types.worker.instances import InstanceId
from skulk.shared.types.worker.runners import RunnerFailed, RunnerId
from skulk.store.config import StagingNodeConfig
from skulk.store.model_store_client import ModelStoreClient
from skulk.utils.channels import Receiver, channel
from skulk.worker.main import Worker
from skulk.worker.runner.runner_supervisor import RunnerSupervisor
from skulk.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

_INSTANCE = InstanceId("instance-load")
_RUNNER = RunnerId("runner-load")
_MODEL = ModelId("org/incomplete")


def _worker_with_assigned_runner(tmp_path: Path) -> tuple[Worker, Receiver[Event]]:
    _, indexed_receiver = channel[IndexedEvent]()
    event_sender, event_receiver = channel[Event]()
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    worker = Worker(
        node_id=NodeId(get_node_id_keypair().to_node_id()),
        event_receiver=indexed_receiver,
        event_sender=event_sender,
        command_sender=command_sender,
        download_command_sender=download_sender,
        store_client=ModelStoreClient(store_host="store.local"),
        staging_config=StagingNodeConfig(node_cache_path=str(tmp_path)),
    )
    shard = get_pipeline_shard_metadata(model_id=_MODEL, device_rank=0, world_size=1)
    instance = get_mlx_ring_instance(
        instance_id=_INSTANCE,
        model_id=_MODEL,
        node_to_runner={worker.node_id: _RUNNER},
        runner_to_shard={_RUNNER: shard},
    )
    worker.state = State(instances={_INSTANCE: instance})
    worker.runners[_RUNNER] = cast(
        RunnerSupervisor, cast(object, SimpleNamespace(status=None))
    )
    return worker, event_receiver


def _identity_accepted(*_args: object, **_kwargs: object) -> None:
    return None


@pytest.fixture(autouse=True)
def authorized_card(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass the identity check so each test reaches the model path lookup."""
    monkeypatch.setattr(
        "skulk.worker.main._require_worker_model_execution_identity",
        _identity_accepted,
    )


async def _failed_status(
    worker: Worker, events: Receiver[Event]
) -> tuple[RunnerFailed, TaskStatusUpdated]:
    await worker._start_runner_task(LoadModel(instance_id=_INSTANCE))
    emitted = events.collect()
    runner_updates = [event for event in emitted if isinstance(event, RunnerStatusUpdated)]
    task_updates = [event for event in emitted if isinstance(event, TaskStatusUpdated)]
    assert len(runner_updates) == 1 and len(task_updates) == 1
    status = runner_updates[0].runner_status
    assert isinstance(status, RunnerFailed)
    return status, task_updates[0]


@pytest.mark.asyncio
async def test_missing_files_at_load_fail_as_incomplete_not_trust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, events = _worker_with_assigned_runner(tmp_path)

    def _missing(*_args: object, **_kwargs: object) -> Path:
        raise FileNotFoundError(f"Model {_MODEL} not found on disk.")

    monkeypatch.setattr("skulk.worker.main.build_model_path", _missing)

    status, task = await _failed_status(worker, events)

    assert status.error_message is not None
    assert status.error_message.startswith(f"{MODEL_FILES_INCOMPLETE_MARKER}:")
    assert MODEL_TRUST_FAILURE_MARKER not in status.error_message
    assert task.task_status == TaskStatus.Failed


@pytest.mark.asyncio
async def test_identity_refusal_at_load_keeps_the_trust_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, events = _worker_with_assigned_runner(tmp_path)

    def _refused(*_args: object, **_kwargs: object) -> Path:
        raise PermissionError("installed artifact identity mismatch")

    monkeypatch.setattr("skulk.worker.main.build_model_path", _refused)

    status, task = await _failed_status(worker, events)

    assert status.error_message is not None
    assert status.error_message.startswith(f"{MODEL_TRUST_FAILURE_MARKER}:")
    assert MODEL_FILES_INCOMPLETE_MARKER not in status.error_message
    assert task.task_status == TaskStatus.Failed
