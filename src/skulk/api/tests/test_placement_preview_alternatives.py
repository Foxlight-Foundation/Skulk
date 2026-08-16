"""Tests for per-host placement-preview alternatives (#557).

On a heterogeneous fleet the previews endpoint returns one ranked pick per
placement shape, so the winning host (typically the largest free GPU) hides
every other host that passes admission. The alternatives pass re-runs the
planner per remaining host and surfaces the viable single-node options.
"""

from datetime import datetime, timezone
from typing import cast

import pytest
from fastapi.testclient import TestClient

import skulk.api.main as api_main
import skulk.master.placement as placement_module
from skulk.api.main import API
from skulk.master.placement import PlacementModelCodeApprovalError
from skulk.shared.election import ElectionMessage
from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.models.registry import RegistryEngineSupportClaim
from skulk.shared.models.remote_code_approval import remote_code_trust_identity
from skulk.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    PlaceInstance,
)
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import NodeResources
from skulk.shared.types.telemetry import NodeTelemetry
from skulk.shared.types.worker.instances import (
    InstanceId,
    MlxRingInstance,
)
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.channels import channel

_MODEL_ID = ModelId("unsloth/preview-alt-test-GGUF")


def _build_api(node_id: str = "local-node") -> API:
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    return API(
        NodeId(node_id),
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_sender,
        election_receiver=election_receiver,
        enable_event_log=False,
        mount_dashboard=False,
    )


def _card() -> ModelCard:
    return ModelCard(
        model_id=_MODEL_ID,
        storage_size=Memory.from_mb(100),
        n_layers=8,
        hidden_size=64,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )


def _single_node_instance(
    node: str,
    *,
    model_card: ModelCard | None = None,
    resolved_backend: str | None = None,
) -> MlxRingInstance:
    runner = RunnerId(f"runner-{node}")
    card = _card() if model_card is None else model_card
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=8,
        n_layers=8,
        resolved_backend=resolved_backend,
    )
    return MlxRingInstance(
        instance_id=InstanceId(f"instance-{node}"),
        shard_assignments=ShardAssignments(
            model_id=_MODEL_ID,
            node_to_runner={NodeId(node): runner},
            runner_to_shard={runner: shard},
        ),
        hosts_by_node={},
        ephemeral_port=0,
    )


async def test_exact_instance_creation_reports_model_trust_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /instance rejects unapproved shard cards before acknowledgement."""

    api = _build_api()
    client = TestClient(api.app)
    card = _card().model_copy(
        update={
            "source_revision": "a" * 40,
            "registry_card_id": "card_" + "d" * 52,
            "registry_snapshot_id": "snapshot-test",
            "registry_provenance": "agent",
            "trust_remote_code": True,
        }
    )
    instance = _single_node_instance("unapproved-node", model_card=card)

    async def _load(_model_id: object) -> ModelCard:
        return card

    monkeypatch.setattr(ModelCard, "load", staticmethod(_load))
    response = client.post(
        "/instance",
        json={"instance": instance.model_dump(mode="json")},
    )

    assert response.status_code == 400
    assert (
        response.headers["X-Skulk-Placement-Failure"]
        == "model_code_approval_required"
    )
    assert remote_code_trust_identity(card) in response.json()["error"]["message"]


async def test_exact_instance_creation_rejects_mismatched_shard_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /instance cannot mix assignment and embedded model identities."""

    api = _build_api()
    client = TestClient(api.app)
    canonical_card = _card()
    mismatched_card = canonical_card.model_copy(
        update={"model_id": ModelId("other-org/other-model")}
    )
    instance = _single_node_instance(
        "mismatched-node",
        model_card=mismatched_card,
    )

    async def _load(_model_id: object) -> ModelCard:
        return canonical_card

    monkeypatch.setattr(ModelCard, "load", staticmethod(_load))
    response = client.post(
        "/instance",
        json={"instance": instance.model_dump(mode="json")},
    )

    assert response.status_code == 400
    assert (
        response.headers["X-Skulk-Placement-Failure"]
        == "model_card_identity_mismatch"
    )
    assert "other-org/other-model" in response.json()["error"]["message"]


async def test_preview_exposes_exact_signed_engine_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operators can distinguish matrix-admitted placements from legacy cards."""
    api = _build_api()
    client = TestClient(api.app)
    node_id = NodeId("future-engine-node")
    api.state.topology.add_node(node_id)
    api._telemetry_view.apply(  # pyright: ignore[reportPrivateUsage]
        NodeTelemetry(
            node_id=node_id,
            info=NodeResources(
                backends=frozenset({"vllm", "vllm-cuda"}),
                engine_builds={
                    "vllm": "vllm@9.9.9",
                    "vllm-cuda": "vllm@9.9.9",
                },
                hardware_classes=frozenset({"nvidia"}),
            ),
        ),
        received_at=datetime.now(tz=timezone.utc),
    )
    card = _card().model_copy(
        update={
            "source_revision": "a" * 40,
            "registry_card_id": "card_" + "a" * 52,
            "registry_snapshot_id": "snapshot-test",
            "registry_provenance": "agent",
            "registry_architecture": "future_architecture_v1",
            "registry_artifact_format": "safetensors",
        }
    )
    monkeypatch.setattr(
        api,
        "_cluster_remote_code_approvals",
        lambda: frozenset({card.registry_card_id or ""}),
    )

    def approval_not_required(_card: ModelCard) -> bool:
        return False

    monkeypatch.setattr(
        placement_module,
        "remote_code_approval_required",
        approval_not_required,
    )
    support = RegistryEngineSupportClaim.model_validate(
        {
            "claim_id": "support_" + "b" * 52,
            "engine": "vllm",
            "engine_build": "vllm@9.9.9",
            "architecture": "future_architecture_v1",
            "artifact_format": "safetensors",
            "artifact_card_id": "card_" + "a" * 52,
            "capability_id": "text.generate",
            "status": "supported",
            "evidence_kind": "load_qualification",
            "evidence_trust": "foxlight_observed",
            "source_url": "https://evidence.example/load",
            "source_sha256": "c" * 64,
            "rationale": "Qualified the exact build and artifact.",
            "hardware_classes": ["nvidia"],
            "recorded_by": "operator@example.com",
            "created_at": "2026-08-16T12:00:00Z",
        },
        strict=False,
    )

    async def _load(_model_id: object) -> ModelCard:
        return card

    def _fake_placements(
        _command: PlaceInstance, **_kwargs: object
    ) -> dict[InstanceId, MlxRingInstance]:
        instance = _single_node_instance(
            str(node_id), resolved_backend="vllm-cuda"
        )
        return {instance.instance_id: instance}

    def _support_for_card(_card: ModelCard) -> tuple[RegistryEngineSupportClaim, ...]:
        return (support,)

    monkeypatch.setattr(ModelCard, "load", staticmethod(_load))
    monkeypatch.setattr(api_main, "get_model_engine_support", _support_for_card)
    monkeypatch.setattr(api_main, "get_instance_placements", _fake_placements)

    response = client.get("/instance/previews", params={"model_id": str(_MODEL_ID)})

    assert response.status_code == 200
    previews = cast("list[dict[str, object]]", response.json()["previews"])
    successful = [preview for preview in previews if preview["instance"] is not None]
    assert successful
    assert all(
        preview["compatibility_source"] == "signed_engine_support"
        for preview in successful
    )
    assert all(
        preview["support_claim_ids"] == [support.claim_id]
        for preview in successful
    )
    assert all(preview["trust_requirement"] is None for preview in successful)


async def test_preview_reports_stable_trust_failure_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fleet-wide trust block is actionable without freezing a preview."""
    api = _build_api()
    client = TestClient(api.app)
    api.state.topology.add_node(NodeId("unapproved-node"))

    async def _load(_model_id: object) -> ModelCard:
        return _card()

    def _blocked(
        _command: PlaceInstance, **_kwargs: object
    ) -> dict[InstanceId, MlxRingInstance]:
        raise PlacementModelCodeApprovalError(
            "No candidate serving cycle approves the exact model card."
        )

    monkeypatch.setattr(ModelCard, "load", staticmethod(_load))
    monkeypatch.setattr(api_main, "get_instance_placements", _blocked)

    response = client.get("/instance/previews", params={"model_id": str(_MODEL_ID)})

    assert response.status_code == 200
    previews = cast("list[dict[str, object]]", response.json()["previews"])
    assert previews
    assert all(preview["instance"] is None for preview in previews)
    assert all(
        preview["error_code"] == "model_code_approval_required"
        for preview in previews
    )
    assert all(preview["trust_requirement"] for preview in previews)


async def test_launch_reports_stable_trust_failure_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adaptive launch keeps readable detail plus a machine-stable category."""
    api = _build_api()
    client = TestClient(api.app)

    async def _load(_model_id: object) -> ModelCard:
        return _card()

    def _blocked(
        _command: PlaceInstance, **_kwargs: object
    ) -> dict[InstanceId, MlxRingInstance]:
        raise PlacementModelCodeApprovalError(
            "Approve the exact model card in cluster Settings."
        )

    monkeypatch.setattr(ModelCard, "load", staticmethod(_load))
    monkeypatch.setattr(api_main, "get_instance_placements", _blocked)

    response = client.post(
        "/place_instance",
        json={"model_id": str(_MODEL_ID)},
    )

    assert response.status_code == 400
    assert response.headers["X-Skulk-Placement-Failure"] == (
        "model_code_approval_required"
    )
    assert "Approve the exact model card" in response.json()["error"]["message"]


async def test_previews_surface_per_host_alternatives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The losing-but-valid host appears as an alternative preview."""

    api = _build_api()
    client = TestClient(api.app)
    api.state.topology.add_node(NodeId("gpu-winner"))
    api.state.topology.add_node(NodeId("ryzen-loser"))

    async def _load(_model_id: object) -> ModelCard:
        return _card()

    monkeypatch.setattr(ModelCard, "load", staticmethod(_load))

    def _fake_placements(
        command: PlaceInstance,
        **kwargs: object,
    ) -> dict[InstanceId, MlxRingInstance]:
        required = cast("set[NodeId] | None", kwargs.get("required_nodes"))
        if required is None:
            # The ranked pass: the planner always picks the big GPU.
            instance = _single_node_instance("gpu-winner")
            return {instance.instance_id: instance}
        if required == {NodeId("ryzen-loser")}:
            instance = _single_node_instance("ryzen-loser")
            return {instance.instance_id: instance}
        raise ValueError(f"no viable placement for {required}")

    monkeypatch.setattr(api_main, "get_instance_placements", _fake_placements)

    response = client.get("/instance/previews", params={"model_id": str(_MODEL_ID)})

    assert response.status_code == 200
    payload = cast("dict[str, object]", cast(object, response.json()))
    previews = cast("list[dict[str, object]]", payload["previews"])
    ranked = [p for p in previews if not p.get("alternative")]
    alternatives = [p for p in previews if p.get("alternative")]
    assert any(p.get("instance") is not None for p in ranked)
    assert all(
        "cluster operator approval" in str(preview.get("trust_requirement"))
        for preview in previews
    )
    assert len(alternatives) == 1
    alt_instance = cast("dict[str, object]", alternatives[0]["instance"])
    inner = cast("dict[str, object]", next(iter(alt_instance.values())))
    assignments = cast("dict[str, object]", inner["shardAssignments"])
    assert list(cast("dict[str, object]", assignments["nodeToRunner"]).keys()) == [
        "ryzen-loser"
    ]


async def test_alternatives_skipped_when_caller_pins_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """node_ids callers already constrained hosts; no alternative fan-out."""

    api = _build_api()
    client = TestClient(api.app)
    api.state.topology.add_node(NodeId("gpu-winner"))
    api.state.topology.add_node(NodeId("ryzen-loser"))

    async def _load(_model_id: object) -> ModelCard:
        return _card()

    monkeypatch.setattr(ModelCard, "load", staticmethod(_load))

    calls: list[object] = []

    def _fake_placements(
        command: PlaceInstance,
        **kwargs: object,
    ) -> dict[InstanceId, MlxRingInstance]:
        calls.append(kwargs.get("required_nodes"))
        instance = _single_node_instance("gpu-winner")
        return {instance.instance_id: instance}

    monkeypatch.setattr(api_main, "get_instance_placements", _fake_placements)

    response = client.get(
        "/instance/previews",
        params={"model_id": str(_MODEL_ID), "node_ids": "gpu-winner"},
    )

    assert response.status_code == 200
    # Every planner call carried the caller's constraint; no per-host fan-out.
    assert all(req == {NodeId("gpu-winner")} for req in calls)


async def test_alternatives_use_ranked_single_node_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model whose single-node placement is Tensor (not Pipeline/MlxRing)
    must still surface losing hosts as alternatives, using the shape the
    ranked pass proved viable (#557)."""

    from skulk.shared.types.worker.instances import InstanceMeta
    from skulk.shared.types.worker.shards import Sharding

    api = _build_api()
    client = TestClient(api.app)
    api.state.topology.add_node(NodeId("gpu-winner"))
    api.state.topology.add_node(NodeId("second-host"))

    async def _load(_model_id: object) -> ModelCard:
        return _card()

    monkeypatch.setattr(ModelCard, "load", staticmethod(_load))

    seen_shardings: list[str] = []

    def _fake_placements(
        command: PlaceInstance,
        **kwargs: object,
    ) -> dict[InstanceId, MlxRingInstance]:
        required = cast("set[NodeId] | None", kwargs.get("required_nodes"))
        # Only Tensor sharding places this model single-node.
        if command.sharding != Sharding.Tensor:
            raise ValueError("this model only single-node places under Tensor")
        if required is None:
            seen_shardings.append("ranked")
            instance = _single_node_instance("gpu-winner")
            return {instance.instance_id: instance}
        if required == {NodeId("second-host")}:
            seen_shardings.append("alt")
            instance = _single_node_instance("second-host")
            return {instance.instance_id: instance}
        raise ValueError(f"no viable placement for {required}")

    monkeypatch.setattr(api_main, "get_instance_placements", _fake_placements)
    _ = InstanceMeta  # imported for clarity; shape comes from ranked previews

    response = client.get("/instance/previews", params={"model_id": str(_MODEL_ID)})

    assert response.status_code == 200
    payload = cast("dict[str, object]", cast(object, response.json()))
    previews = cast("list[dict[str, object]]", payload["previews"])
    alternatives = [p for p in previews if p.get("alternative")]
    # The alternative surfaced despite the model never placing under
    # Pipeline/MlxRing, proving the shape came from the ranked pass.
    assert len(alternatives) == 1
    assert alternatives[0]["sharding"] == "Tensor"
    assert "alt" in seen_shardings


async def test_placement_apis_forward_unified_memory_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Placement responses must stamp the same UMA-safe context as launch."""

    api = _build_api()
    node_id = NodeId("uma-host")
    api.state.topology.add_node(node_id)
    expected_unified_nodes = frozenset({node_id})

    async def _load(_model_id: object) -> ModelCard:
        return _card()

    def _classify_uma(
        *_args: object, **_kwargs: object
    ) -> frozenset[NodeId]:
        return expected_unified_nodes

    monkeypatch.setattr(ModelCard, "load", staticmethod(_load))
    monkeypatch.setattr(
        api_main,
        "unified_memory_gpu_node_ids",
        _classify_uma,
    )

    seen_unified_nodes: list[object] = []

    def _fake_placements(
        command: PlaceInstance,
        **kwargs: object,
    ) -> dict[InstanceId, MlxRingInstance]:
        del command
        seen_unified_nodes.append(kwargs.get("unified_memory_gpu_nodes"))
        instance = _single_node_instance(str(node_id))
        return {instance.instance_id: instance}

    monkeypatch.setattr(api_main, "get_instance_placements", _fake_placements)

    placement = await api.get_placement(_MODEL_ID)
    previews = await api.get_placement_previews(_MODEL_ID)

    assert placement.shard_assignments.node_to_runner.keys() == {node_id}
    assert previews.previews
    assert seen_unified_nodes
    assert all(value == expected_unified_nodes for value in seen_unified_nodes)
