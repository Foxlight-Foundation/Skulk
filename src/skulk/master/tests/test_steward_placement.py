"""System-role (steward) placement stamping and repair-survival tests."""

from skulk.master.placement import (
    fallback_command_for_refused_instance,
    place_instance,
    replacement_command_for_download_failed_instance,
    replacement_command_for_refused_instance,
)
from skulk.master.tests.test_placement import fully_connected_three_nodes
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.commands import PlaceInstance
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.instances import InstanceMeta, MlxRingInstance
from skulk.shared.types.worker.shards import Sharding


def _steward_card() -> ModelCard:
    return ModelCard(
        model_id=ModelId("steward-brain-model"),
        storage_size=Memory.from_gb(3),
        n_layers=12,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )


def _place_steward() -> MlxRingInstance:
    topology, node_memory, node_network, _node_ids = fully_connected_three_nodes(
        (10.0, 10.0, 10.0)
    )
    command = PlaceInstance(
        model_card=_steward_card(),
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
        system_role="steward",
    )
    placed = place_instance(command, topology, {}, node_memory, node_network)
    instance = next(iter(placed.values()))
    assert isinstance(instance, MlxRingInstance)
    return instance


def test_place_instance_stamps_system_role() -> None:
    """The minted instance carries the command's system-role marker."""
    instance = _place_steward()
    assert instance.system_role == "steward"


def test_default_placement_has_no_system_role() -> None:
    """Ordinary placements stay unmarked (and old event logs replay as None)."""
    topology, node_memory, node_network, _node_ids = fully_connected_three_nodes(
        (10.0, 10.0, 10.0)
    )
    command = PlaceInstance(
        model_card=_steward_card(),
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
    )
    placed = place_instance(command, topology, {}, node_memory, node_network)
    instance = next(iter(placed.values()))
    assert instance.system_role is None


def test_repair_commands_preserve_system_role() -> None:
    """Every repair builder re-stamps the steward flag (failover survival).

    Repair reconstructs placement intent from the instance's shards, which
    have no channel for the flag; without explicit re-stamping a repaired
    steward would silently demote to an ordinary instance and the invariant
    pass would then place a second steward.
    """
    instance = _place_steward()

    wider = replacement_command_for_refused_instance(instance)
    assert wider.system_role == "steward"

    refusing = next(iter(instance.shard_assignments.node_to_runner))
    fallback = fallback_command_for_refused_instance(instance, refusing)
    assert fallback.system_role == "steward"

    failed = replacement_command_for_download_failed_instance(instance, {refusing})
    assert failed.system_role == "steward"


def test_repair_commands_keep_none_for_ordinary_instances() -> None:
    """Ordinary instances never acquire a system role through repair."""
    topology, node_memory, node_network, _node_ids = fully_connected_three_nodes(
        (10.0, 10.0, 10.0)
    )
    command = PlaceInstance(
        model_card=_steward_card(),
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
    )
    placed = place_instance(command, topology, {}, node_memory, node_network)
    instance = next(iter(placed.values()))
    assert replacement_command_for_refused_instance(instance).system_role is None
