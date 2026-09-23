"""The served context window: fleet default, caller requests, and repairs."""

import pytest

from skulk.master.placement import (
    PlacementError,
    fallback_command_for_refused_instance,
    place_instance,
    replacement_command_for_download_failed_instance,
    replacement_command_for_refused_instance,
    served_context_window,
)
from skulk.master.tests.conftest import create_node_memory, create_node_network
from skulk.shared.models.memory_estimate import SERVED_CONTEXT_DEFAULT_TOKENS
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.topology import Topology
from skulk.shared.types.commands import PlaceInstance
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.instances import Instance, InstanceMeta
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata, Sharding


def _card(*, gguf: bool) -> ModelCard:
    """A card whose KV cost is known and whose advertised maximum is large."""
    return ModelCard(
        model_id=ModelId("org/served" if gguf else "org/lazy"),
        storage_size=Memory.from_gb(4),
        n_layers=32,
        hidden_size=4096,
        supports_tensor=True,
        num_key_value_heads=8,
        tasks=[ModelTask.TextGeneration],
        gguf_file="model-Q4_K_M.gguf" if gguf else None,
        context_length=262144,
    )


def _assignments(card: ModelCard, backend: str | None) -> ShardAssignments:
    runner = RunnerId("r0")
    return ShardAssignments(
        model_id=card.model_id,
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
        node_to_runner={NodeId("n0"): runner},
    )


def test_default_caps_an_engine_that_reserves_at_load() -> None:
    assignments = _assignments(_card(gguf=True), "llama_server-vulkan")
    assert (
        served_context_window(assignments, 262144, requested=None, served_default=32768)
        == 32768
    )
    # A smaller fit stays the fit; the default never raises a window.
    assert (
        served_context_window(assignments, 16384, requested=None, served_default=32768)
        == 16384
    )


def test_default_leaves_a_lazy_engine_at_its_fit() -> None:
    assignments = _assignments(_card(gguf=False), "mlx")
    assert (
        served_context_window(assignments, 230093, requested=None, served_default=32768)
        == 230093
    )


def test_a_request_is_honored_exactly_up_to_the_ceiling() -> None:
    assignments = _assignments(_card(gguf=True), "llama_server-vulkan")
    assert (
        served_context_window(assignments, 262144, requested=131072, served_default=32768)
        == 131072
    )
    # A request below the default is honored too: the caller chose it.
    assert (
        served_context_window(assignments, 262144, requested=4096, served_default=32768)
        == 4096
    )
    # A request on an MLX placement caps it the same way.
    lazy = _assignments(_card(gguf=False), "mlx")
    assert (
        served_context_window(lazy, 230093, requested=65536, served_default=32768)
        == 65536
    )


def test_a_request_above_the_ceiling_is_refused_by_name() -> None:
    assignments = _assignments(_card(gguf=True), "llama_server-vulkan")
    with pytest.raises(PlacementError, match="exceeds the 100000 tokens"):
        served_context_window(
            assignments, 100000, requested=131072, served_default=32768
        )


def _place(command: PlaceInstance, *, served_default: int | None) -> Instance:
    topology = Topology()
    node_id = NodeId()
    topology.add_node(node_id)
    placements = place_instance(
        command,
        topology,
        {},
        {
            node_id: create_node_memory(
                Memory.from_gb(100).in_bytes, ram_total=Memory.from_gb(128).in_bytes
            )
        },
        {node_id: create_node_network()},
        served_context_default=served_default,
    )
    return next(iter(placements.values()))


def _command(card: ModelCard, requested: int | None = None) -> PlaceInstance:
    return PlaceInstance(
        command_id=CommandId(),
        model_card=card,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
        requested_context_tokens=requested,
    )


def test_placement_stamps_the_fleet_default_on_a_served_card() -> None:
    instance = _place(
        _command(_card(gguf=True)), served_default=SERVED_CONTEXT_DEFAULT_TOKENS
    )
    assert instance.context_token_limit == SERVED_CONTEXT_DEFAULT_TOKENS
    assert instance.requested_context_tokens is None


def test_placement_stamps_and_records_a_caller_request() -> None:
    instance = _place(
        _command(_card(gguf=True), requested=65536),
        served_default=SERVED_CONTEXT_DEFAULT_TOKENS,
    )
    assert instance.context_token_limit == 65536
    assert instance.requested_context_tokens == 65536


def test_placement_refuses_a_request_the_nodes_cannot_hold() -> None:
    card = _card(gguf=True).model_copy(update={"context_length": 1048576})
    with pytest.raises(PlacementError, match="exceeds"):
        _place(_command(card, requested=1048576), served_default=32768)


def test_repairs_carry_the_callers_request_forward() -> None:
    instance = _place(
        _command(_card(gguf=True), requested=65536),
        served_default=SERVED_CONTEXT_DEFAULT_TOKENS,
    )
    node = next(iter(instance.shard_assignments.node_to_runner))
    for command in (
        fallback_command_for_refused_instance(instance, node),
        replacement_command_for_refused_instance(instance),
        replacement_command_for_download_failed_instance(instance, frozenset()),
    ):
        assert command.requested_context_tokens == 65536


def test_master_reads_the_default_from_the_converged_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import skulk.master.main as master_main
    from skulk.store.config import InferenceConfig, SkulkConfig

    master = object.__new__(master_main.Master)
    monkeypatch.setattr(
        master_main,
        "load_skulk_config",
        lambda: SkulkConfig(inference=InferenceConfig(served_context_tokens=65536)),
    )
    assert master._served_context_default() == 65536  # pyright: ignore[reportPrivateUsage]

    def unreadable() -> SkulkConfig:
        raise OSError("config unreadable")

    monkeypatch.setattr(master_main, "load_skulk_config", unreadable)
    assert master._served_context_default() == SERVED_CONTEXT_DEFAULT_TOKENS  # pyright: ignore[reportPrivateUsage]
