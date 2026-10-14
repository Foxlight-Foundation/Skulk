# pyright: reportPrivateUsage=false
"""Representative qualification must never become a hardware permission list."""

from datetime import UTC, datetime

import pytest

from skulk.master.placement import place_instance
from skulk.master.tests.conftest import create_node_memory, create_node_network
from skulk.master.tests.test_placement import place_instance_command
from skulk.provisioning.stable_diffusion_cpp import PUBLISHED_WHEELS
from skulk.shared.models import model_cards as cards_module
from skulk.shared.models.image import ImageCardConfig
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelId,
    ModelTask,
    PlacementCardConfig,
)
from skulk.shared.models.registry import RegistryEngineSupportClaim
from skulk.shared.models.tests.test_image_contract import _document
from skulk.shared.topology import Topology
from skulk.shared.types.common import NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import NodeResources


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("status", ["supported", "unsupported", None])
@pytest.mark.parametrize(
    "compute,hardware",
    [
        ("vulkan", frozenset({"intel"})),
        ("cuda", frozenset({"nvidia", "nvidia:sm-8.6"})),
        ("rocm", frozenset({"amd", "amd:gfx942", "amd:pci-1002-74a1"})),
    ],
)
def test_native_placement_on_untested_device_does_not_require_qualification(
    monkeypatch: pytest.MonkeyPatch,
    status: str | None,
    compute: str,
    hardware: frozenset[str],
    legacy: bool,
) -> None:
    """Normal placement uses the engine contract despite absent/differently scoped evidence."""
    card = ModelCard(
        model_id=ModelId("test/native-image"),
        source_revision="a" * 40,
        source_repository=ModelId("owner/model"),
        storage_size=Memory.from_bytes(28),
        n_layers=1,
        hidden_size=32,
        supports_tensor=False,
        gguf_file="model.gguf",
        tasks=[ModelTask.TextToImage, ModelTask.ImageToImage],
        image=ImageCardConfig.model_validate(_document()),
        placement=PlacementCardConfig(
            compatible_backends=frozenset({"stable_diffusion_cpp"})
        ),
    )
    if legacy:
        card = card.model_copy(
            update={"placement": PlacementCardConfig(compatible_backends=frozenset())}
        )
    if status is not None:
        evidence = RegistryEngineSupportClaim.model_validate(
            {
                "engine": "stable_diffusion_cpp",
                "engine_build": "a-different-tested-build",
                "architecture": "qwen-image-21-turbo",
                "artifact_format": "gguf",
                "artifact_card_id": "card_" + "a" * 52,
                "capability_id": "image.edit",
                "status": status,
                "evidence_kind": "feature_qualification",
                "evidence_trust": "foxlight_observed",
                "source_url": "https://example.org/test",
                "source_sha256": "b" * 64,
                "rationale": "Scoped editing result on another device",
                "hardware_classes": ("nvidia:nvidia-a100",),
                "claim_id": "support_" + "a" * 52,
                "recorded_by": "test",
                "created_at": datetime.now(UTC),
            }
        )
        monkeypatch.setattr(cards_module, "_registry_engine_support", (evidence,))
    else:
        monkeypatch.setattr(cards_module, "_registry_engine_support", ())
    node = NodeId("untested-node")
    topology = Topology()
    topology.add_node(node)
    wheel = next(
        item
        for item in PUBLISHED_WHEELS
        if item.architecture == "x86_64" and item.compute == compute
    )
    backend = f"stable_diffusion_cpp-{compute}"
    resources = NodeResources(
        backends=frozenset({backend}),
        hardware_classes=hardware,
        on_demand_backends=frozenset({backend}),
        installable_engine_builds={backend: wheel.build_identity},
    )
    placements = place_instance(
        place_instance_command(card),
        topology,
        {},
        {node: create_node_memory(Memory.from_gb(16).in_bytes)},
        {node: create_node_network()},
        node_resources={node: resources},
        node_vram={node: Memory.from_gb(8)},
    )
    shard = next(
        iter(next(iter(placements.values())).shard_assignments.runner_to_shard.values())
    )
    assert shard.resolved_backend == backend
    assert shard.resolved_engine_build == wheel.build_identity
