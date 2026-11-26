"""The engine-order routes: list, set and clear a person's order for a model."""

from collections.abc import Mapping
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import skulk.api.main as api_main
from skulk.api.main import API
from skulk.shared.election import ElectionMessage
from skulk.shared.models.model_cards import ModelCard, ModelTask, PlacementCardConfig
from skulk.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    PlaceInstance,
    SetModelEngineOrder,
)
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.engine_orders import MAX_MODEL_ENGINE_ORDERS
from skulk.shared.types.events import IndexedEvent
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.instances import Instance, InstanceId, MlxRingInstance
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.channels import channel

MODEL = ModelId("test/text")
CARD = ModelCard(
    model_id=MODEL,
    source_revision="a" * 40,
    storage_size=Memory.from_gb(1),
    n_layers=1,
    hidden_size=32,
    supports_tensor=False,
    tasks=[ModelTask.TextGeneration],
    placement=PlacementCardConfig(
        compatible_backends=frozenset({"llama_cpp", "tensorfold"}),
        backend_preference=("llama_cpp-vulkan",),
    ),
)


def _build_api() -> API:
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    return API(
        NodeId("test-node"),
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_sender,
        election_receiver=election_receiver,
        enable_event_log=False,
        mount_dashboard=False,
    )


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> API:
    """An API whose catalog holds exactly one model."""

    async def load(model_id: ModelId) -> ModelCard:
        if model_id != MODEL:
            raise HTTPException(status_code=404, detail="unknown model")
        return CARD

    def known_card(model_id: ModelId) -> ModelCard | None:
        return CARD if model_id == MODEL else None

    monkeypatch.setattr(API, "_load_authorized_model_card", staticmethod(load))
    monkeypatch.setattr(api_main, "get_card", known_card)
    return _build_api()


def _applying(api: API) -> AsyncMock:
    """Stand in for the master: apply each change to the API's view at once."""

    async def send(command: object) -> None:
        assert isinstance(command, SetModelEngineOrder)
        orders = dict(api.state.model_engine_orders)
        if command.engine_order:
            orders[command.model_id] = tuple(command.engine_order)
        else:
            orders.pop(command.model_id, None)
        api.state = api.state.model_copy(update={"model_engine_orders": orders})

    return AsyncMock(side_effect=send)


def test_a_cluster_with_no_orders_lists_none(api: API) -> None:
    response = TestClient(api.app).get("/models/engine-orders")

    assert response.status_code == 200
    assert response.json() == {"orders": []}


def test_setting_an_order_sends_the_change_and_returns_what_is_in_force(
    api: API,
) -> None:
    with patch.object(api, "_send", new=_applying(api)) as send:
        response = TestClient(api.app).put(
            "/models/engine-order",
            json={"modelId": str(MODEL), "engineOrder": ["tensorfold-rocm", "llama_cpp"]},
        )

    assert response.status_code == 200
    assert response.json() == {
        "modelId": str(MODEL),
        "engineOrder": ["tensorfold-rocm", "llama_cpp"],
        "cardEngineOrder": ["llama_cpp-vulkan"],
    }
    assert send.await_args is not None
    command = cast(object, send.await_args.args[0])
    assert isinstance(command, SetModelEngineOrder)
    assert command.model_id == MODEL
    assert command.engine_order == ["tensorfold-rocm", "llama_cpp"]

    listed = TestClient(api.app).get("/models/engine-orders")
    assert listed.json() == {"orders": [response.json()]}


def test_a_model_the_catalog_does_not_hold_is_refused_before_anything_is_sent(
    api: API,
) -> None:
    with patch.object(api, "_send", new=AsyncMock()) as send:
        response = TestClient(api.app).put(
            "/models/engine-order",
            json={"modelId": "test/unknown", "engineOrder": ["mlx"]},
        )

    assert response.status_code == 404
    send.assert_not_awaited()


@pytest.mark.parametrize("order", [[], ["Not A Tag"], ["mlx", "mlx"]])
def test_a_malformed_order_is_refused_before_anything_is_sent(
    api: API, order: list[str]
) -> None:
    with patch.object(api, "_send", new=AsyncMock()) as send:
        response = TestClient(api.app).put(
            "/models/engine-order", json={"modelId": str(MODEL), "engineOrder": order}
        )

    assert response.status_code == 422
    send.assert_not_awaited()


def test_a_new_model_is_refused_once_the_cluster_holds_its_limit(api: API) -> None:
    api.state = api.state.model_copy(
        update={
            "model_engine_orders": {
                ModelId(f"test/model-{index}"): ("mlx",)
                for index in range(MAX_MODEL_ENGINE_ORDERS)
            }
        }
    )
    with patch.object(api, "_send", new=AsyncMock()) as send:
        response = TestClient(api.app).put(
            "/models/engine-order", json={"modelId": str(MODEL), "engineOrder": ["mlx"]}
        )

    assert response.status_code == 409
    send.assert_not_awaited()


def test_a_change_the_cluster_does_not_confirm_is_reported_as_unconfirmed(
    api: API, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_main, "_ENGINE_ORDER_CONFIRM_SECONDS", 0.1)
    with patch.object(api, "_send", new=AsyncMock()):
        response = TestClient(api.app).put(
            "/models/engine-order", json={"modelId": str(MODEL), "engineOrder": ["mlx"]}
        )

    assert response.status_code == 503


def test_clearing_returns_the_model_to_its_cards_order(api: API) -> None:
    api.state = api.state.model_copy(
        update={"model_engine_orders": {MODEL: ("tensorfold-rocm",)}}
    )
    with patch.object(api, "_send", new=_applying(api)) as send:
        response = TestClient(api.app).delete(
            "/models/engine-order", params={"model_id": str(MODEL)}
        )

    assert response.status_code == 200
    assert response.json() == {
        "modelId": str(MODEL),
        "engineOrder": None,
        "cardEngineOrder": ["llama_cpp-vulkan"],
    }
    command = cast(object, send.await_args_list[0].args[0])
    assert isinstance(command, SetModelEngineOrder)
    assert command.engine_order is None


def test_an_order_left_by_a_removed_model_can_still_be_cleared(api: API) -> None:
    gone = ModelId("test/removed")
    api.state = api.state.model_copy(update={"model_engine_orders": {gone: ("mlx",)}})
    with patch.object(api, "_send", new=_applying(api)):
        response = TestClient(api.app).delete(
            "/models/engine-order", params={"model_id": str(gone)}
        )

    assert response.status_code == 200
    assert response.json() == {
        "modelId": str(gone),
        "engineOrder": None,
        "cardEngineOrder": [],
    }
    assert api.state.model_engine_orders == {}


def _planned_on(backend: str | None):
    """Stand in for the dry run: the launch would start on ``backend``."""

    def dry_run(command: PlaceInstance, **_: object) -> Mapping[InstanceId, Instance]:
        runner = RunnerId("runner-0")
        instance_id = InstanceId(str(command.command_id))
        return {
            instance_id: MlxRingInstance(
                instance_id=instance_id,
                shard_assignments=ShardAssignments(
                    model_id=MODEL,
                    node_to_runner={NodeId("node-0"): runner},
                    runner_to_shard={
                        runner: PipelineShardMetadata(
                            model_card=CARD,
                            device_rank=0,
                            world_size=1,
                            start_layer=0,
                            end_layer=1,
                            n_layers=1,
                            resolved_backend=backend,
                        )
                    },
                ),
                hosts_by_node={},
                ephemeral_port=0,
            )
        }

    return dry_run


@pytest.mark.parametrize(
    ("asked", "planned"),
    [
        (["tensorfold-rocm"], "tensorfold-rocm"),
        (["tensorfold"], "tensorfold-rocm"),
        (["vllm-cuda", "llama_cpp"], "llama_cpp-vulkan"),
    ],
)
def test_a_launch_may_name_its_engines_for_that_launch_only(
    api: API, monkeypatch: pytest.MonkeyPatch, asked: list[str], planned: str
) -> None:
    dry_runs: list[PlaceInstance] = []

    def dry_run(command: PlaceInstance, **extra: object) -> Mapping[InstanceId, Instance]:
        dry_runs.append(command)
        return _planned_on(planned)(command, **extra)

    monkeypatch.setattr(api_main, "get_instance_placements", dry_run)
    with patch.object(api, "_send", new=AsyncMock()) as send:
        response = TestClient(api.app).post(
            "/place_instance", json={"model_id": str(MODEL), "engine_order": asked}
        )

    assert response.status_code == 200, response.text
    assert dry_runs[0].engine_order == asked
    command = cast(object, send.await_args_list[0].args[0])
    assert isinstance(command, PlaceInstance)
    assert command.engine_order == asked
    assert response.json()["engine_pick"] == {
        "asked": asked,
        "engines": [planned],
        "used": True,
    }
    # Trying an engine sets no standing order.
    assert api.state.model_engine_orders == {}


def test_a_trial_whose_engines_cannot_run_starts_on_the_next_and_says_so(
    api: API, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An order only ranks; the answer names the engine used instead."""
    monkeypatch.setattr(
        api_main, "get_instance_placements", _planned_on("llama_cpp-vulkan")
    )
    with patch.object(api, "_send", new=AsyncMock()) as send:
        response = TestClient(api.app).post(
            "/place_instance",
            json={"model_id": str(MODEL), "engine_order": ["tensorfold-rocm"]},
        )

    assert response.status_code == 200, response.text
    assert response.json()["engine_pick"] == {
        "asked": ["tensorfold-rocm"],
        "engines": ["llama_cpp-vulkan"],
        "used": False,
    }
    command = cast(object, send.await_args_list[0].args[0])
    assert isinstance(command, PlaceInstance)
    assert command.engine_order == ["tensorfold-rocm"]


def test_a_trial_planned_without_a_named_engine_does_not_claim_either_way(
    api: API, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_main, "get_instance_placements", _planned_on(None))
    with patch.object(api, "_send", new=AsyncMock()):
        response = TestClient(api.app).post(
            "/place_instance",
            json={"model_id": str(MODEL), "engine_order": ["tensorfold-rocm"]},
        )

    assert response.status_code == 200, response.text
    assert response.json()["engine_pick"] == {
        "asked": ["tensorfold-rocm"],
        "engines": [],
        "used": None,
    }


def test_a_launch_that_names_no_engines_is_sent_as_before(
    api: API, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        api_main, "get_instance_placements", _planned_on("llama_cpp-vulkan")
    )
    with patch.object(api, "_send", new=AsyncMock()) as send:
        response = TestClient(api.app).post(
            "/place_instance", json={"model_id": str(MODEL)}
        )

    assert response.status_code == 200, response.text
    assert response.json()["engine_pick"] is None
    command = cast(object, send.await_args_list[0].args[0])
    assert isinstance(command, PlaceInstance)
    assert command.engine_order is None


def test_a_malformed_launch_order_is_refused(api: API) -> None:
    with patch.object(api, "_send", new=AsyncMock()) as send:
        response = TestClient(api.app).post(
            "/place_instance",
            json={"model_id": str(MODEL), "engine_order": ["Not A Tag"]},
        )

    assert response.status_code == 422
    send.assert_not_awaited()
