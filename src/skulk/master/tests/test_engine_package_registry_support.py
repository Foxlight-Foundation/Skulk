"""A signed registry entry makes an engine package eligible for a card that does not name it.

A signed card usually names no add-on engine: a card without a placement
section is an MLX card. The registry's engine-support list is the other way an
engine becomes eligible for a model, and these tests hold that it works for an
engine this build has no name for, without widening anything else.
"""

from datetime import UTC, datetime

import pytest

import skulk.shared.models.model_cards as cards_module
from skulk.master.placement import PlacementError, place_instance
from skulk.master.tests.conftest import create_node_memory, create_node_network
from skulk.master.tests.test_placement import place_instance_command
from skulk.shared.models.model_cards import ModelCard, ModelTask, VisionCardConfig
from skulk.shared.models.registry import (
    RegistryCapabilityClaim,
    RegistryEngineSupportClaim,
)
from skulk.shared.topology import Topology
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.engine_packages import (
    NodeEngineLoadResult,
    NodeEnginePackage,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import NodeResources
from skulk.shared.types.worker.shards import ShardMetadata

NODE = NodeId("node-a")
CARD_ID = "card_" + "a" * 52
OTHER_CARD_ID = "card_" + "b" * 52
ARCHITECTURE = "qwen3_5"
ROCM_TAG = "tensorfold-rocm"
METAL_TAG = "tensorfold-metal"
BUILD = "c" * 64
OFFERED_BUILD = "d" * 64


def _signed_card(
    *,
    card_id: str = CARD_ID,
    architecture: str = ARCHITECTURE,
    model_id: str = "test/signed-text",
) -> ModelCard:
    """A signed text card with no placement section, so MLX is its only engine."""
    return ModelCard(
        model_id=ModelId(model_id),
        source_revision="a" * 40,
        storage_size=Memory.from_bytes(1024),
        n_layers=1,
        hidden_size=32,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        quantization="4bit",
        registry_card_id=card_id,
        registry_snapshot_id="snapshot-test",
        registry_provenance="foxlight",
        registry_architecture=architecture,
        registry_artifact_format="safetensors",
        registry_capability_claims=(
            RegistryCapabilityClaim(
                capability_id="text.generate",
                scope="model",
                status="claimed",
                source="upstream_structured",
                confidence=1.0,
            ),
        ),
    )


def _entry(
    *,
    engine: str = "tensorfold",
    architecture: str = ARCHITECTURE,
    status: str = "supported",
    evidence_kind: str = "upstream_compatibility",
    artifact_card_id: str | None = None,
    quantization: str | None = None,
    capability_id: str = "text.generate",
) -> RegistryEngineSupportClaim:
    return RegistryEngineSupportClaim.model_validate(
        {
            "engine": engine,
            "engine_build": "foxlight.tensorfold@1.3.2",
            "architecture": architecture,
            "artifact_format": "safetensors",
            "artifact_card_id": artifact_card_id,
            "quantization": quantization,
            "capability_id": capability_id,
            "status": status,
            "evidence_kind": evidence_kind,
            "evidence_trust": "foxlight_observed",
            "source_url": "https://example.org/evidence",
            "source_sha256": "e" * 64,
            "rationale": "The engine implements this architecture.",
            "hardware_classes": ("amd",),
            "claim_id": "support_" + "a" * 52,
            "recorded_by": "test",
            "created_at": datetime.now(UTC),
        }
    )


def _holds(tag: str = ROCM_TAG, *extra: str) -> NodeResources:
    """A node that holds the engine package for ``tag`` beside ``extra`` engines."""
    return NodeResources(
        backends=frozenset({tag, *extra}),
        engine_builds={tag: BUILD},
    )


def _place(
    card: ModelCard, resources: NodeResources, *, steward: bool = False
) -> ShardMetadata:
    topology = Topology()
    topology.add_node(NODE)
    command = place_instance_command(card)
    if steward:
        command = command.model_copy(update={"system_role": "steward"})
    placements = place_instance(
        command,
        topology,
        {},
        {NODE: create_node_memory(Memory.from_gb(16).in_bytes)},
        {NODE: create_node_network()},
        node_resources={NODE: resources},
    )
    instance = next(iter(placements.values()))
    return next(iter(instance.shard_assignments.runner_to_shard.values()))


def _publish(
    monkeypatch: pytest.MonkeyPatch, *entries: RegistryEngineSupportClaim
) -> None:
    monkeypatch.setattr(cards_module, "_registry_engine_support", entries)


def test_without_an_entry_a_package_only_node_cannot_run_the_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The card names MLX only, so holding the package changes nothing."""
    _publish(monkeypatch)

    with pytest.raises(PlacementError):
        _place(_signed_card(), _holds())


def test_an_entry_places_the_card_on_the_package_with_its_exact_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _publish(monkeypatch, _entry())

    shard = _place(_signed_card(), _holds())

    assert shard.resolved_backend == ROCM_TAG
    assert shard.resolved_engine_build == BUILD


def _signed_vision_card() -> ModelCard:
    """The shape most signed chat cards have: text generation plus a vision section."""
    card = _signed_card()
    return card.model_copy(
        update={
            "vision": VisionCardConfig(image_token_id=7, model_type="qwen3_5"),
            "registry_capability_claims": (
                *card.registry_capability_claims,
                RegistryCapabilityClaim(
                    capability_id="vision.understand",
                    scope="model",
                    status="claimed",
                    source="upstream_structured",
                    confidence=1.0,
                ),
            ),
        }
    )


def test_an_entry_places_a_card_that_also_has_a_vision_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A text entry is enough for a chat card that also understands images.

    Every card entered in the registry for an add-on engine had a vision
    section, and a guard that treated vision as a non-text operation made all
    of those entries do nothing on a real node.
    """
    _publish(monkeypatch, _entry())

    shard = _place(_signed_vision_card(), _holds())

    assert shard.resolved_backend == ROCM_TAG
    assert shard.resolved_engine_build == BUILD


def test_a_vision_card_without_an_entry_is_still_refused_on_a_package_only_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vision alone admits nothing: the card names MLX only until an entry exists."""
    _publish(monkeypatch)

    with pytest.raises(PlacementError):
        _place(_signed_vision_card(), _holds())


def test_a_vision_card_keeps_its_built_in_engine_when_the_node_has_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An entry does not move a vision card off MLX on a node that has both."""
    _publish(monkeypatch, _entry())

    shard = _place(_signed_vision_card(), _holds(METAL_TAG, "mlx", "mlx-metal"))

    assert shard.resolved_backend == "mlx-metal"


def test_an_entry_covers_a_package_the_node_will_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _publish(monkeypatch, _entry())

    shard = _place(
        _signed_card(),
        NodeResources(
            backends=frozenset({ROCM_TAG}),
            on_demand_backends=frozenset({ROCM_TAG}),
            installable_engine_builds={ROCM_TAG: OFFERED_BUILD},
        ),
    )

    assert shard.resolved_backend == ROCM_TAG
    assert shard.resolved_engine_build == OFFERED_BUILD


def test_an_entry_does_not_draw_the_card_from_its_built_in_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Mac that holds both keeps serving the card on MLX."""
    _publish(monkeypatch, _entry())

    shard = _place(_signed_card(), _holds(METAL_TAG, "mlx", "mlx-metal"))

    assert shard.resolved_backend == "mlx-metal"


def test_an_entry_for_one_lane_admits_that_lane_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _publish(monkeypatch, _entry(engine=ROCM_TAG))

    assert _place(_signed_card(), _holds(ROCM_TAG)).resolved_backend == ROCM_TAG
    with pytest.raises(PlacementError):
        _place(_signed_card(), _holds("tensorfold-cuda"))


@pytest.mark.parametrize(
    "changed",
    [
        {"status": "experimental"},
        {"status": "unsupported"},
        {"evidence_kind": "load_qualification", "artifact_card_id": CARD_ID},
        {"architecture": "another_architecture"},
        {"quantization": "8bit"},
        {"capability_id": "image.generate"},
        {"artifact_card_id": OTHER_CARD_ID},
    ],
)
def test_an_entry_that_does_not_cover_the_card_admits_nothing(
    monkeypatch: pytest.MonkeyPatch, changed: dict[str, str]
) -> None:
    """Only a supported upstream-compatibility entry for this card's artifact counts."""
    _publish(monkeypatch, _entry(**changed))

    with pytest.raises(PlacementError):
        _place(_signed_card(), _holds())


def test_an_entry_for_a_card_without_a_curated_architecture_covers_that_card_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Such a card's architecture is its own identity, so the entry cannot spread."""
    sentinel = f"unknown:{CARD_ID}"
    _publish(monkeypatch, _entry(architecture=sentinel))

    covered = _signed_card(architecture=sentinel)
    other = _signed_card(
        card_id=OTHER_CARD_ID,
        architecture=f"unknown:{OTHER_CARD_ID}",
        model_id="test/other-signed-text",
    )

    assert _place(covered, _holds()).resolved_backend == ROCM_TAG
    with pytest.raises(PlacementError):
        _place(other, _holds())


def test_an_entry_does_not_bring_back_a_build_that_refused_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What a node learned from a load outranks what the registry says in general."""
    _publish(monkeypatch, _entry())
    card = _signed_card()
    refused = NodeEngineLoadResult(
        build=BUILD,
        outcome="refused",
        code="unsupported_model",
        model_id=str(card.model_id),
        at_unix_ms=1000,
    )
    resources = _holds().model_copy(
        update={
            "engine_packages": (
                NodeEnginePackage(
                    engine_id="foxlight.tensorfold",
                    lane="rocm",
                    backend_tag=ROCM_TAG,
                    status="active",
                    build=BUILD,
                    installed_builds=(BUILD,),
                    unsupported_models=(refused,),
                ),
            )
        }
    )

    with pytest.raises(PlacementError):
        _place(card, resources)


def test_a_system_placement_is_never_put_on_an_engine_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resident assistant needs tools, which a package's chat dialect lacks."""
    _publish(monkeypatch, _entry())
    card = _signed_card()

    assert _place(card, _holds()).resolved_backend == ROCM_TAG
    with pytest.raises(PlacementError):
        _place(card, _holds(), steward=True)
    assert (
        _place(card, _holds(METAL_TAG, "mlx", "mlx-metal"), steward=True).resolved_backend
        == "mlx-metal"
    )


def test_a_system_placement_on_a_cluster_without_packages_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _publish(monkeypatch)
    resources = NodeResources(backends=frozenset({"mlx", "mlx-metal"}))
    card = _signed_card()

    ordinary = _place(card, resources)
    steward = _place(card, resources, steward=True)

    assert steward.resolved_backend == ordinary.resolved_backend == "mlx-metal"
    assert steward.resolved_engine_build == ordinary.resolved_engine_build
