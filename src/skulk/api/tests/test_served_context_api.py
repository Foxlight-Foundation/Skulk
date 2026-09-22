# pyright: reportPrivateUsage=false
"""The place route's context request and the previews' window facts."""

from collections.abc import Mapping

import pytest
from fastapi.testclient import TestClient

import skulk.api.main as api_main
from skulk.api.main import API
from skulk.shared.election import ElectionMessage
from skulk.shared.models.memory_estimate import per_token_kv_bytes
from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.types.commands import (
    Command,
    ForwarderCommand,
    ForwarderDownloadCommand,
    PlaceInstance,
)
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.instances import Instance, InstanceId, MlxRingInstance
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.store.config import InferenceConfig, SkulkConfig
from skulk.utils.channels import channel

_MODEL_ID = ModelId("unsloth/served-context-test-GGUF")


def _build_api() -> API:
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    return API(
        NodeId("local-node"),
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_sender,
        election_receiver=election_receiver,
        enable_event_log=False,
        mount_dashboard=False,
    )


def _card(*, gguf: bool = True) -> ModelCard:
    return ModelCard(
        model_id=_MODEL_ID,
        storage_size=Memory.from_gb(4),
        n_layers=32,
        hidden_size=4096,
        supports_tensor=True,
        num_key_value_heads=8,
        tasks=[ModelTask.TextGeneration],
        gguf_file="model-Q4_K_M.gguf" if gguf else None,
        context_length=262144,
    )


def _instance(card: ModelCard, backend: str, window: int) -> MlxRingInstance:
    runner = RunnerId("runner-0")
    return MlxRingInstance(
        instance_id=InstanceId("instance-0"),
        shard_assignments=ShardAssignments(
            model_id=_MODEL_ID,
            node_to_runner={NodeId("node-0"): runner},
            runner_to_shard={
                runner: PipelineShardMetadata(
                    model_card=card,
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=32,
                    n_layers=32,
                    resolved_backend=backend,
                )
            },
        ),
        hosts_by_node={},
        ephemeral_port=0,
        context_token_limit=window,
    )


def test_preview_shows_the_maximum_the_default_and_the_reservation() -> None:
    api = _build_api()
    api._skulk_config = SkulkConfig(
        inference=InferenceConfig(served_context_tokens=16384)
    )
    card = _card()
    fields = api._preview_context_fields(
        _instance(card, "llama_server-vulkan", 262144)
    )
    assert fields["max_context_tokens"] == 262144
    assert fields["default_context_tokens"] == 16384
    assert fields["reserves_context_at_load"] is True
    assert fields["kv_bytes_per_token"] == per_token_kv_bytes(
        card, resolved_backend="llama_server-vulkan"
    )


def test_preview_leaves_a_lazy_engine_at_its_maximum() -> None:
    api = _build_api()
    fields = api._preview_context_fields(_instance(_card(gguf=False), "mlx", 230093))
    assert fields["default_context_tokens"] == 230093
    assert fields["reserves_context_at_load"] is False


def test_place_route_forwards_the_requested_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _build_api()
    client = TestClient(api.app)
    sent: list[Command] = []
    dry_runs: list[PlaceInstance] = []

    async def _load(_model_id: object) -> ModelCard:
        return _card()

    def _dry_run(command: PlaceInstance, **_: object) -> Mapping[InstanceId, Instance]:
        dry_runs.append(command)
        return {}

    async def _send(command: Command) -> None:
        sent.append(command)

    monkeypatch.setattr(ModelCard, "load", staticmethod(_load))
    monkeypatch.setattr(api_main, "get_instance_placements", _dry_run)
    monkeypatch.setattr(api, "_send", _send)

    response = client.post(
        "/place_instance", json={"model_id": str(_MODEL_ID), "context_tokens": 65536}
    )

    assert response.status_code == 200, response.text
    assert dry_runs and dry_runs[0].requested_context_tokens == 65536
    assert isinstance(sent[0], PlaceInstance)
    assert sent[0].requested_context_tokens == 65536


@pytest.mark.parametrize("value", [0, 100, 2_000_000])
def test_place_route_rejects_an_out_of_range_window(value: int) -> None:
    client = TestClient(_build_api().app)
    response = client.post(
        "/place_instance", json={"model_id": str(_MODEL_ID), "context_tokens": value}
    )
    assert response.status_code == 422
