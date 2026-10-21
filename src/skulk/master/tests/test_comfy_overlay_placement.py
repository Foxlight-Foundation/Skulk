"""New Comfy placements bind issued updates without restricting untested GPUs."""

import asyncio
from typing import Literal

import pytest

from skulk.master.placement import add_instance_to_placements, place_instance
from skulk.master.tests.conftest import create_node_memory, create_node_network
from skulk.master.tests.test_placement import place_instance_command
from skulk.provisioning import comfy_extension
from skulk.provisioning.manifest import COMFY_PIN
from skulk.shared.tests.model_card_fixtures import load_fixture_card
from skulk.shared.topology import Topology
from skulk.shared.types.commands import CreateInstance
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import NodeResources


@pytest.mark.parametrize("exact_backend", [None, True, False])
@pytest.mark.parametrize("installed_build", ["missing", "base", "overlay"])
def test_overlay_binding_is_shared_by_automatic_and_explicit_placement(monkeypatch: pytest.MonkeyPatch, exact_backend: bool | None, installed_build: Literal["missing", "base", "overlay"]) -> None:
    """New placements prefer the issued update; retained instances keep their build."""
    wheel = comfy_extension.ComfyExtensionWheel(filename="skulk_comfy_ltx-0.1.1-py3-none-any.whl", sha256="a" * 64, payload_sha256="b" * 64, size_bytes=40115, comfy_revision=COMFY_PIN, ltx_revision=comfy_extension.LTX_REFERENCE_REVISION, features=("res2s",))
    older = wheel.model_copy(update={"filename": "skulk_comfy_ltx-0.1.0-py3-none-any.whl", "sha256": "c" * 64, "payload_sha256": "d" * 64})
    monkeypatch.setattr(comfy_extension, "PUBLISHED_EXTENSIONS", (wheel, older))
    base = f"comfy@{COMFY_PIN}/torch@2.14.0+cu130"
    build = wheel.build_identity(base)
    card = asyncio.run(load_fixture_card(ModelId("Lightricks/LTX-2.5-distilled-comfy-int8")))
    assert card is not None
    node = NodeId("compatible-untested-gpu")
    topology = Topology()
    topology.add_node(node)
    installed = {"missing": None, "base": base, "overlay": older.build_identity(base)}[installed_build]
    resources = NodeResources(backends=frozenset({"comfy-cuda"}), on_demand_backends=frozenset() if installed is not None else frozenset({"comfy-cuda"}), engine_builds={"comfy-cuda": installed} if installed is not None else {}, installable_engine_builds={"comfy-cuda": build}, hardware_classes=frozenset({"nvidia", "nvidia:unqualified-device"}))
    memory = {node: create_node_memory(Memory.from_gb(256).in_bytes)}
    placements = place_instance(place_instance_command(card), topology, {}, memory, {node: create_node_network()}, node_resources={node: resources}, node_vram={node: Memory.from_gb(256)})
    if exact_backend is not None:
        instance = next(iter(placements.values()))
        assignments = instance.shard_assignments
        unbound = {runner: shard.model_copy(update={"resolved_backend": "comfy-cuda" if exact_backend else None, "resolved_engine_build": None}) for runner, shard in assignments.runner_to_shard.items()}
        instance = instance.model_copy(update={"shard_assignments": assignments.model_copy(update={"runner_to_shard": unbound})})
        placements = add_instance_to_placements(CreateInstance(instance=instance), topology, {}, memory, node_vram={node: Memory.from_gb(256)}, node_resources={node: resources})
    selected = next(iter(placements.values()))
    assert {shard.resolved_engine_build for shard in selected.shard_assignments.runner_to_shard.values()} == {build}
    assignments = selected.shard_assignments
    retained_build = installed or base
    retained = {runner: shard.model_copy(update={"resolved_engine_build": retained_build}) for runner, shard in assignments.runner_to_shard.items()}
    restored = selected.model_copy(update={"shard_assignments": assignments.model_copy(update={"runner_to_shard": retained})})
    placements = add_instance_to_placements(CreateInstance(instance=restored), topology, {}, memory, node_vram={node: Memory.from_gb(256)}, node_resources={node: resources})
    assert {shard.resolved_engine_build for shard in next(iter(placements.values())).shard_assignments.runner_to_shard.values()} == {retained_build}


def test_h3_placement_keeps_the_installed_base_without_acquiring_ltx_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    wheel = comfy_extension.ComfyExtensionWheel(filename="skulk_comfy_ltx-0.1.0-py3-none-any.whl", sha256="a" * 64, payload_sha256="b" * 64, size_bytes=40115, comfy_revision=COMFY_PIN, ltx_revision=comfy_extension.LTX_REFERENCE_REVISION, features=("res2s",))
    monkeypatch.setattr(comfy_extension, "PUBLISHED_EXTENSIONS", (wheel,))
    base = f"comfy@{COMFY_PIN}/torch@2.14.0+cu130"
    build = wheel.build_identity(base)
    card = asyncio.run(load_fixture_card(ModelId("Comfy-Org/MiniMax-H3-FL2VA-comfy-int8")))
    assert card is not None
    node = NodeId("compatible-untested-gpu")
    topology = Topology()
    topology.add_node(node)
    resources = NodeResources(backends=frozenset({"comfy-cuda"}), engine_builds={"comfy-cuda": build}, installable_engine_builds={"comfy-cuda": build})
    placements = place_instance(place_instance_command(card), topology, {}, {node: create_node_memory(Memory.from_gb(256).in_bytes)}, {node: create_node_network()}, node_resources={node: resources}, node_vram={node: Memory.from_gb(256)})
    assert {shard.resolved_engine_build for shard in next(iter(placements.values())).shard_assignments.runner_to_shard.values()} == {base}
