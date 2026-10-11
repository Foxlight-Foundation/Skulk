"""Workload assessment witnesses use independent memory boundary calculations."""

from datetime import datetime, timedelta, timezone

import pytest

import skulk.shared.models.model_cards as model_cards
from skulk.master.inference_assessment import assess_inference
from skulk.master.placement import place_instance
from skulk.master.placement_utils import usable_vram_by_node
from skulk.shared.models.inference_assessment import InferenceWorkload
from skulk.shared.models.llama_server_settings import LlamaServerSettings
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelTask,
    PlacementCardConfig,
    RuntimeCapabilityCardConfig,
    VisionCardConfig,
)
from skulk.shared.models.registry import (
    RegistryCapabilityClaim,
    RegistryEngineSupportClaim,
)
from skulk.shared.topology import Topology
from skulk.shared.types.commands import PlaceInstance
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import (
    AcceleratorMetrics,
    MemoryUsage,
    NodeNetworkInfo,
    NodeResources,
    SystemPerformanceProfile,
)
from skulk.shared.types.telemetry import NodeTelemetry, TelemetryView
from skulk.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from skulk.shared.types.worker.runners import RunnerReady, RunnerRunning
from skulk.shared.types.worker.shards import Sharding
from skulk.utils.info_gatherer.linux_gpu import LinuxGpuMetrics

_NOW = datetime(2026, 10, 11, tzinfo=timezone.utc)
_NODE = NodeId("local")


def _card(backend: str = "mlx") -> ModelCard:
    return ModelCard(
        model_id=ModelId("test/model"),
        source_revision="a" * 40,
        storage_size=Memory.from_bytes(1024**3),
        n_layers=32,
        hidden_size=4096,
        supports_tensor=False,
        num_key_value_heads=8,
        context_length=131072,
        tasks=[ModelTask.TextGeneration],
        trust_remote_code=False,
        gguf_file="model-Q4_K_M.gguf" if backend.startswith("llama") else None,
        quantization="Q4_K_M" if backend.startswith("llama") else "4bit",
        placement=PlacementCardConfig(compatible_backends=frozenset({backend})),
    )


def _fleet(
    available_bytes: int, *, backend: str = "mlx"
) -> tuple[Topology, TelemetryView]:
    topology = Topology()
    topology.add_node(_NODE)
    view = TelemetryView()
    view.apply(
        NodeTelemetry(
            node_id=_NODE,
            info=MemoryUsage.from_bytes(
                ram_total=available_bytes * 4 // 3,
                ram_available=available_bytes,
                swap_total=0,
                swap_available=0,
            ),
        ),
        received_at=_NOW,
    )
    view.apply(
        NodeTelemetry(
            node_id=_NODE,
            info=NodeResources(
                backends=frozenset({backend}),
                llama_server_settings=LlamaServerSettings(),
            ),
        ),
        received_at=_NOW,
    )
    return topology, view


def _assess(
    card: ModelCard,
    topology: Topology,
    view: TelemetryView,
    *,
    context: int = 8192,
    concurrent: int = 1,
    instances: dict[InstanceId, Instance] | None = None,
):
    return assess_inference(
        card,
        InferenceWorkload(context_tokens=context, concurrent_requests=concurrent),
        topology=topology,
        instances=instances or {},
        network={_NODE: NodeNetworkInfo()},
        telemetry=view,
        now=_NOW,
    )


@pytest.mark.parametrize("context", [512, 8192, 32768])
def test_boundary_uses_weight_overhead_context_and_usable_pool(context: int) -> None:
    # Independent witness: 32 layers * 8 KV heads * 128 fp16 elements * K+V.
    cache = 32 * 8 * 128 * 2 * 2 * context
    required = int(1024**3 * 1.3) + 256 * 1024**2 + cache
    card = _card()
    topology, view = _fleet(required + 1024**2)
    before = view.node_memory.copy()
    assert _assess(card, topology, view, context=context).status == "local"
    assert view.node_memory == before
    assert len(list(topology.list_nodes())) == 1
    topology, view = _fleet(required - 1024**2)
    result = _assess(card, topology, view, context=context)
    assert result.status == "external_required"
    candidate = result.external_configurations[0]
    assert candidate.estimated_memory_bytes_per_replica >= required - 1
    assert candidate.minimum_memory_bytes_per_node > required
    assert not candidate.offer_available
    assert result.readiness == "not_observed"


def test_larger_context_changes_admission_and_remote_allocation_is_whole_model() -> (
    None
):
    topology, view = _fleet(4 * 1024**3)
    card = _card()
    assert _assess(card, topology, view, context=8192).status == "local"
    result = _assess(card, topology, view, context=65536)
    assert result.status == "external_required"
    assert (
        result.external_configurations[0].estimated_memory_bytes_per_replica
        > 8 * 1024**3
    )
    assert (
        result.external_configurations[0].minimum_weight_disk_bytes_per_node == 1024**3
    )


@pytest.mark.parametrize("field", ["source_revision", "num_key_value_heads"])
def test_missing_identity_or_cache_evidence_abstains(field: str) -> None:
    topology, view = _fleet(64 * 1024**3)
    card = _card().model_copy(update={field: None})
    result = _assess(card, topology, view)
    assert result.status == "insufficient_evidence"
    assert not result.external_configurations


def test_context_above_artifact_limit_is_blocked() -> None:
    topology, view = _fleet(64 * 1024**3)
    assert _assess(_card(), topology, view, context=262144).status == "blocked"


def test_heartbeat_does_not_refresh_stale_memory() -> None:
    topology, view = _fleet(64 * 1024**3)
    view.node_memory_received_at[_NODE] = _NOW - timedelta(minutes=2)
    view.node_last_heartbeat[_NODE] = _NOW
    assert _assess(_card(), topology, view).status == "insufficient_evidence"
    view.prune(_NODE)
    assert _NODE not in view.node_memory_received_at
    assert _NODE not in view.node_resources_received_at


def test_external_compatible_engine_can_resolve_local_engine_absence() -> None:
    topology, view = _fleet(64 * 1024**3, backend="mlx")
    result = _assess(_card("llama_server"), topology, view)
    assert result.status == "external_required"
    assert result.external_configurations[0].backend == "llama_server"


def test_unsupported_engine_is_blocked_not_a_memory_recommendation() -> None:
    topology, view = _fleet(64 * 1024**3)
    result = _assess(_card("unknown_backend"), topology, view)
    assert result.status == "blocked"
    assert not result.external_configurations
    advertised_topology, advertised_view = _fleet(
        64 * 1024**3, backend="unknown_backend"
    )
    assert (
        _assess(_card("unknown_backend"), advertised_topology, advertised_view).status
        == "blocked"
    )


def test_concurrent_server_requests_share_weights_and_aggregate_cache() -> None:
    topology, view = _fleet(8 * 1024**3, backend="llama_server")
    result = _assess(_card("llama_server"), topology, view, concurrent=4)
    assert result.status == "local"
    assert len(result.local_replicas) == 1
    assert result.local_replicas[0].context_pool_tokens == 32768
    assert result.local_replicas[0].concurrent_requests == 4


def test_serial_replica_reservations_prevent_overcommit() -> None:
    topology, view = _fleet(4 * 1024**3)
    assert _assess(_card(), topology, view).status == "local"
    result = _assess(_card(), topology, view, concurrent=4)
    assert result.status == "insufficient_evidence"
    assert "concurrent capacity" in result.reasons[0]


def test_committed_placement_reduces_new_capacity_but_idle_ready_can_be_reused() -> (
    None
):
    topology, view = _fleet(4 * 1024**3, backend="llama_server")
    card = _card("llama_server")
    instances = place_instance(
        PlaceInstance(
            model_card=card,
            min_nodes=1,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            requested_context_tokens=8192,
        ),
        topology=topology,
        current_instances={},
        node_memory=view.node_memory,
        node_network={_NODE: NodeNetworkInfo()},
        node_resources=view.node_resources,
    )
    assert (
        _assess(card, topology, view, instances=instances).status == "external_required"
    )
    runners = {
        runner: RunnerReady()
        for instance in instances.values()
        for runner in instance.shard_assignments.runner_to_shard
    }
    result = assess_inference(
        card,
        InferenceWorkload(),
        topology=topology,
        instances=instances,
        network={_NODE: NodeNetworkInfo()},
        telemetry=view,
        now=_NOW,
        runners=runners,
    )
    assert result.status == "local"
    assert result.readiness == "loaded_not_workload_verified"
    assert result.local_replicas[0].existing_instance_id is not None
    busy = assess_inference(
        card,
        InferenceWorkload(),
        topology=topology,
        instances=instances,
        network={_NODE: NodeNetworkInfo()},
        telemetry=view,
        now=_NOW,
        runners={runner: RunnerRunning() for runner in runners},
    )
    assert busy.status == "insufficient_evidence"
    assert "already serving locally" in busy.reasons[0]


def test_no_local_nodes_still_produces_external_estimate() -> None:
    result = _assess(_card("llama_server"), Topology(), TelemetryView())
    assert result.status == "external_required"
    assert result.external_configurations


def test_missing_gpu_memory_cannot_be_replaced_by_large_host_ram() -> None:
    topology, view = _fleet(64 * 1024**3, backend="llama_server-cuda")
    assert (
        _assess(_card("llama_server-cuda"), topology, view).status
        == "insufficient_evidence"
    )


def test_discrete_memory_occupancy_and_staleness_are_distinct() -> None:
    topology, view = _fleet(64 * 1024**3, backend="llama_server-cuda")
    profile = SystemPerformanceProfile(
        accelerator=AcceleratorMetrics(
            vendor="nvidia",
            vram_total_bytes=24 * 1024**3,
            vram_used_bytes=23 * 1024**3,
        )
    )
    view.apply(
        NodeTelemetry(node_id=_NODE, info=LinuxGpuMetrics(system_profile=profile)),
        received_at=_NOW,
    )
    card = _card("llama_server-cuda")
    result = _assess(card, topology, view)
    assert result.status == "external_required"
    assert result.external_configurations[0].memory_pool == "vram"
    assert result.external_configurations[0].gpus_per_node == 1
    view.node_system_received_at[_NODE] = _NOW - timedelta(minutes=2)
    assert _assess(card, topology, view).status == "insufficient_evidence"


def test_published_head_width_changes_cache_estimate_without_a_benchmark() -> None:
    topology, view = _fleet(4 * 1024**3)
    card = _card()
    assert _assess(card, topology, view).status == "local"
    wide = card.model_copy(update={"attention_head_dim": 512})
    result = _assess(wide, topology, view)
    assert result.status == "external_required"
    assert not any("128-element" in text for text in result.uncertainty)


def _matrix_card(backend: str, *, legacy: bool) -> ModelCard:
    card = _card(backend)
    return card.model_copy(
        update={
            "placement": PlacementCardConfig(
                compatible_backends=frozenset({backend}) if legacy else frozenset()
            ),
            "registry_card_id": "card_" + "a" * 52,
            "registry_provenance": "agent",
            "registry_architecture": "test-model",
            "registry_artifact_format": "gguf" if card.gguf_file else "safetensors",
            "registry_capability_claims": (
                RegistryCapabilityClaim(
                    capability_id="text_generation",
                    scope="model",
                    status="claimed",
                    source="upstream_structured",
                    confidence=0.9,
                ),
            ),
        }
    )


def _support(card: ModelCard, engine: str) -> RegistryEngineSupportClaim:
    return RegistryEngineSupportClaim(
        engine=engine,
        engine_build="test-build",
        architecture="test-model",
        artifact_format="gguf" if card.gguf_file else "safetensors",
        capability_id="text_generation",
        status="supported",
        evidence_kind="upstream_compatibility",
        evidence_trust="third_party_reported",
        source_url="https://example.org/engine-support",
        source_sha256="b" * 64,
        rationale="Published architecture compatibility, not a measured artifact load.",
        hardware_classes=("nvidia-sm80",),
        claim_id="support_" + "b" * 52,
        recorded_by="test-publisher",
        created_at=_NOW,
    )


@pytest.mark.parametrize("engine", ["vllm", "llama_server"])
def test_family_only_matrix_claim_never_invents_a_cpu_or_vram_lane(
    monkeypatch: pytest.MonkeyPatch,
    engine: str,
) -> None:
    card = _matrix_card(engine, legacy=False)
    monkeypatch.setattr(
        model_cards, "_registry_engine_support", (_support(card, engine),)
    )
    result = _assess(card, Topology(), TelemetryView())
    assert result.status == "insufficient_evidence"
    assert not result.external_configurations
    assert "concrete backend" in result.reasons[0]


def test_concrete_matrix_lane_preserves_gpu_pool_and_exact_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = _matrix_card("vllm-cuda", legacy=False)
    monkeypatch.setattr(
        model_cards, "_registry_engine_support", (_support(card, "vllm-cuda"),)
    )
    result = _assess(card, Topology(), TelemetryView())
    assert result.status == "external_required"
    assert len(result.external_configurations) == 1
    candidate = result.external_configurations[0]
    assert candidate.memory_pool == "vram"
    assert candidate.gpus_per_node == 1
    assert candidate.engine_build == "test-build"


def test_legacy_and_deduplicated_matrix_alternatives_are_both_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = _matrix_card("llama_server-cuda", legacy=True)
    claim = _support(card, "llama_server-cuda")
    duplicate = claim.model_copy(update={"claim_id": "support_" + "c" * 52})
    monkeypatch.setattr(model_cards, "_registry_engine_support", (claim, duplicate))
    result = _assess(card, Topology(), TelemetryView())
    assert result.status == "external_required"
    assert len(result.external_configurations) == 2
    assert {candidate.engine_build for candidate in result.external_configurations} == {
        None,
        "test-build",
    }
    assert all(
        candidate.memory_pool == "vram" for candidate in result.external_configurations
    )
    legacy = next(
        candidate
        for candidate in result.external_configurations
        if candidate.engine_build is None
    )
    assert legacy.hardware_classes == ()


def test_management_host_never_requires_inference_memory_telemetry() -> None:
    topology, view = _fleet(1 * 1024**3)
    management = NodeId("management")
    topology.add_node(management)
    view.apply(
        NodeTelemetry(
            node_id=management,
            info=NodeResources(
                participation="management",
                backends=frozenset(),
            ),
        ),
        received_at=_NOW,
    )
    assert management not in view.node_memory
    assert _assess(_card(), topology, view).status == "external_required"


def test_incomplete_matrix_evidence_does_not_override_legacy_compatibility() -> None:
    topology, view = _fleet(64 * 1024**3)
    claim = RegistryCapabilityClaim(
        capability_id="vision",
        scope="artifact",
        status="incomplete",
        source="artifact_manifest",
        confidence=0.9,
    )
    card = _card().model_copy(update={"registry_capability_claims": (claim,)})
    result = _assess(card, topology, view)
    assert result.status == "local"
    assert any(
        "legacy compatibility remains independent" in text
        for text in result.uncertainty
    )
    matrix_only = card.model_copy(
        update={"placement": PlacementCardConfig(compatible_backends=frozenset())}
    )
    assert _assess(matrix_only, topology, view).status == "blocked"


def test_speculative_vision_remote_grouping_uses_effective_slots() -> None:
    card = _card("llama_server-cpu").model_copy(
        update={
            "vision": VisionCardConfig(
                projector_file="mmproj.gguf", projector_size=1024**2
            ),
            "runtime": RuntimeCapabilityCardConfig(served_spec_type="ngram"),
        }
    )
    result = _assess(card, Topology(), TelemetryView(), concurrent=2)
    assert result.status == "external_required"
    candidate = result.external_configurations[0]
    assert candidate.requests_per_replica == 1
    assert candidate.replica_count == 2


def test_external_groups_obey_the_command_context_limit() -> None:
    card = _card("llama_server-cpu").model_copy(update={"context_length": 2_097_152})
    result = _assess(card, Topology(), TelemetryView(), context=1_048_576, concurrent=2)
    assert result.status == "external_required"
    candidate = result.external_configurations[0]
    assert candidate.context_pool_tokens == 1_048_576
    assert candidate.replica_count == 2


def test_incompatible_worker_does_not_gate_artifact_memory_freshness() -> None:
    topology, view = _fleet(1 * 1024**3)
    unrelated = NodeId("unrelated-cpu")
    topology.add_node(unrelated)
    view.apply(
        NodeTelemetry(
            node_id=unrelated,
            info=NodeResources(
                backends=frozenset({"llama_server-cpu"}),
                participation="full",
            ),
        ),
        received_at=_NOW,
    )
    assert unrelated not in view.node_memory
    result = _assess(_card("mlx"), topology, view)
    assert result.status == "external_required"


def test_shared_gpu_pool_is_rebuilt_after_pending_ram_reservations() -> None:
    topology, view = _fleet(80 * 1024**3, backend="llama_server-cuda")
    view.node_memory[_NODE] = MemoryUsage.from_bytes(
        ram_total=128 * 1024**3,
        ram_available=80 * 1024**3,
        swap_total=0,
        swap_available=0,
    )
    profile = SystemPerformanceProfile(
        accelerator=AcceleratorMetrics(
            vendor="nvidia",
            name="NVIDIA GB10",
            compute_capability="12.1",
            vram_total_bytes=128 * 1024**3,
            vram_used_bytes=0,
        )
    )
    view.apply(
        NodeTelemetry(node_id=_NODE, info=LinuxGpuMetrics(system_profile=profile)),
        received_at=_NOW,
    )
    existing = _card("llama_server-cuda").model_copy(
        update={
            "model_id": ModelId("existing/large"),
            "storage_size": Memory.from_gb(40),
        }
    )
    pending = place_instance(
        PlaceInstance(
            model_card=existing,
            min_nodes=1,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            requested_context_tokens=8192,
        ),
        topology=topology,
        current_instances={},
        node_memory=view.node_memory,
        node_network={_NODE: NodeNetworkInfo()},
        node_resources=view.node_resources,
        node_vram=usable_vram_by_node(
            view.node_system, view.node_resources, view.node_memory
        ),
        unified_memory_gpu_nodes=frozenset({_NODE}),
    )
    target = _card("llama_server-cuda").model_copy(
        update={"storage_size": Memory.from_gb(30)}
    )
    assert _assess(target, topology, view).status == "local"
    assert (
        _assess(target, topology, view, instances=pending).status == "external_required"
    )
