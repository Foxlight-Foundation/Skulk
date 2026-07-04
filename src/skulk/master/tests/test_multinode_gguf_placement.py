# pyright: reportPrivateUsage=false
"""Multi-node llama.cpp (RPC driver + donors) placement (#328) and the
common-engine cycle rule (#414).

Live-measured grounding (Phase 0 spike, kite4+kite5): llama.cpp's RPC split
allocates from the VRAM carve only, the driver should be the biggest-VRAM node,
and donor endpoints must be routable addresses chosen at placement time.
"""

import pytest

from skulk.master.placement import (
    PlacementError,
    PlacementInfoPendingError,
    _cycle_common_multi_node_engines,
    place_instance,
)
from skulk.master.tests.conftest import (
    create_node_memory,
    create_node_network,
    create_socket_connection,
)
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelId,
    ModelTask,
    PlacementCardConfig,
)
from skulk.shared.topology import Topology
from skulk.shared.types.commands import PlaceInstance
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.multiaddr import Multiaddr
from skulk.shared.types.profiling import NodeResources
from skulk.shared.types.topology import Connection, Cycle, SocketConnection
from skulk.shared.types.worker.instances import InstanceMeta, LlamaRpcInstance
from skulk.shared.types.worker.shards import (
    PipelineShardMetadata,
    RpcDonorShardMetadata,
    Sharding,
)


def _gguf_card(storage_gb: float = 30.0) -> ModelCard:
    return ModelCard(
        model_id=ModelId("test/pooled-gguf"),
        storage_size=Memory.from_gb(storage_gb),
        n_layers=36,
        hidden_size=2880,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        placement=PlacementCardConfig(
            compatible_backends=frozenset(
                {"llama_server-vulkan", "llama_cpp-vulkan"}
            ),
            backend_preference=("llama_server-vulkan", "llama_cpp-vulkan"),
        ),
    )


def _amd_resources() -> NodeResources:
    return NodeResources(
        backends=frozenset(
            {"llama_server", "llama_server-vulkan", "llama_cpp", "llama_cpp-vulkan"}
        ),
        participation="full",
    )


def _mac_resources() -> NodeResources:
    return NodeResources(backends=frozenset({"mlx", "mlx-metal"}), participation="full")


def _command(card: ModelCard, min_nodes: int = 1) -> PlaceInstance:
    return PlaceInstance(
        command_id=CommandId(),
        model_card=card,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=min_nodes,
    )


def _routable_connection(ip_octet: int) -> SocketConnection:
    """An observed connection on a ROUTABLE (static-subnet) address.

    The shared conftest helper uses 169.254/16 link-local addresses, which
    donor-endpoint selection rejects by contract; RPC tests need the
    static-subnet shape (the kite pair's 10.99.0.0/30 TB link).
    """
    return SocketConnection(
        sink_multiaddr=Multiaddr(address=f"/ip4/10.99.0.{ip_octet}/tcp/1234"),
    )


def _amd_pair() -> tuple[Topology, NodeId, NodeId]:
    """Two bidirectionally connected AMD nodes (the kite4+kite5 shape),
    observed over a routable static-subnet link."""
    topology = Topology()
    big = NodeId()
    small = NodeId()
    topology.add_node(big)
    topology.add_node(small)
    topology.add_connection(
        Connection(source=big, sink=small, edge=_routable_connection(2))
    )
    topology.add_connection(
        Connection(source=small, sink=big, edge=_routable_connection(1))
    )
    return topology, big, small


def test_common_engine_intersection_across_cycle() -> None:
    # Mac advertises mlx only, AMD advertises llama_* only: no common engine.
    cycle = Cycle(node_ids=[NodeId("mac"), NodeId("amd")])
    resources = {NodeId("mac"): _mac_resources(), NodeId("amd"): _amd_resources()}
    hybrid_backends = frozenset({"mlx", "llama_cpp-vulkan", "llama_server-vulkan"})
    assert (
        _cycle_common_multi_node_engines(cycle, hybrid_backends, resources) == set()
    )
    # Two AMD nodes share llama_server (multi-node) and llama_cpp (not).
    amd_cycle = Cycle(node_ids=[NodeId("a"), NodeId("b")])
    amd_resources = {NodeId("a"): _amd_resources(), NodeId("b"): _amd_resources()}
    assert _cycle_common_multi_node_engines(
        amd_cycle, hybrid_backends, amd_resources
    ) == {"llama_server"}


def test_hybrid_card_rejects_mixed_engine_cycle() -> None:
    """#414: a hybrid card must not admit a cycle mixing an MLX-only node with
    a llama.cpp-only node; the old any() over the card's backends allowed it."""
    topology = Topology()
    mac = NodeId()
    amd = NodeId()
    topology.add_node(mac)
    topology.add_node(amd)
    topology.add_connection(
        Connection(source=mac, sink=amd, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=amd, sink=mac, edge=create_socket_connection(2))
    )
    hybrid_card = ModelCard(
        model_id=ModelId("test/hybrid"),
        storage_size=Memory.from_gb(1),
        n_layers=10,
        hidden_size=1000,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        placement=PlacementCardConfig(
            compatible_backends=frozenset({"mlx", "llama_cpp-vulkan"})
        ),
    )
    node_memory = {
        mac: create_node_memory(Memory.from_gb(8).in_bytes),
        amd: create_node_memory(Memory.from_gb(8).in_bytes),
    }
    node_network = {mac: create_node_network(), amd: create_node_network()}
    resources = {mac: _mac_resources(), amd: _amd_resources()}
    command = _command(hybrid_card, min_nodes=2)
    with pytest.raises(PlacementError, match="requires one engine common"):
        place_instance(
            command,
            topology,
            {},
            node_memory,
            node_network,
            node_resources=resources,
        )


def test_pooled_gguf_places_rpc_shape_on_amd_pair() -> None:
    """A GGUF that fits no single node but fits the pooled pair mints the
    driver-plus-donor shape: driver = biggest-VRAM node, donor carries an
    endpoint and a degenerate shard."""
    topology, big, small = _amd_pair()
    node_memory = {
        big: create_node_memory(Memory.from_gb(64).in_bytes),
        small: create_node_memory(Memory.from_gb(32).in_bytes),
    }
    node_network = {big: create_node_network(), small: create_node_network()}
    resources = {big: _amd_resources(), small: _amd_resources()}
    node_vram_strict = {
        big: Memory.from_gb(57.6),
        small: Memory.from_gb(28.8),
    }
    # 55GB of weights: does not fit either node alone under overhead, fits the
    # ~86GB pool proportionally.
    command = _command(_gguf_card(storage_gb=55.0), min_nodes=1)
    placements = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        node_resources=resources,
        node_vram=node_vram_strict,
        node_vram_strict=node_vram_strict,
    )
    instance = next(iter(placements.values()))
    assert isinstance(instance, LlamaRpcInstance)
    assert instance.driver_node == big
    assert set(instance.donor_endpoints) == {small}
    endpoint = instance.donor_endpoints[small]
    ip, _, port = endpoint.rpartition(":")
    # The donor-side address of the observed driver->donor connection, and by
    # contract a routable one (never link-local/loopback).
    assert ip == "10.99.0.2"
    assert int(port) > 0
    driver_runner = instance.shard_assignments.node_to_runner[big]
    donor_runner = instance.shard_assignments.node_to_runner[small]
    driver_shard = instance.shard_assignments.runner_to_shard[driver_runner]
    donor_shard = instance.shard_assignments.runner_to_shard[donor_runner]
    assert isinstance(driver_shard, PipelineShardMetadata)
    assert driver_shard.device_rank == 0
    assert driver_shard.world_size == 2
    assert isinstance(donor_shard, RpcDonorShardMetadata)
    assert donor_shard.start_layer == donor_shard.end_layer == 0
    # #330 stamping applies to both roles.
    assert driver_shard.resolved_backend == "llama_server-vulkan"
    assert donor_shard.resolved_backend == "llama_server-vulkan"


def test_link_local_only_path_fails_placement() -> None:
    """A pair whose only observed path is link-local (e.g. an unconfigured
    Thunderbolt link) must fail cleanly rather than stamp an endpoint the
    driver cannot reliably dial (the two-TB-port asymmetric-routing trap)."""
    topology = Topology()
    big = NodeId()
    small = NodeId()
    topology.add_node(big)
    topology.add_node(small)
    # conftest connections are 169.254/16 link-local.
    topology.add_connection(
        Connection(source=big, sink=small, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=small, sink=big, edge=create_socket_connection(2))
    )
    node_memory = {
        big: create_node_memory(Memory.from_gb(64).in_bytes),
        small: create_node_memory(Memory.from_gb(32).in_bytes),
    }
    node_network = {big: create_node_network(), small: create_node_network()}
    resources = {big: _amd_resources(), small: _amd_resources()}
    node_vram = {big: Memory.from_gb(57.6), small: Memory.from_gb(28.8)}
    command = _command(_gguf_card(storage_gb=55.0))
    with pytest.raises(ValueError, match="ROUTABLE"):
        place_instance(
            command,
            topology,
            {},
            node_memory,
            node_network,
            node_resources=resources,
            node_vram=node_vram,
            node_vram_strict=node_vram,
        )


def test_vision_card_never_pools_on_served_backends() -> None:
    """MODEL truth vs PLATFORM truth: a vision card may declare llama_server
    compatibility (the model runs there), but the served runner cannot load
    its projector yet, so the platform gate strips llama_server before the
    common-engine rule and the card never lands on a served or pooled
    placement where its advertised vision capability would silently break."""
    from skulk.shared.models.model_cards import VisionCardConfig

    topology, big, small = _amd_pair()
    card = _gguf_card(storage_gb=70.0).model_copy(
        update={"vision": VisionCardConfig(model_type="gemma4")}
    )
    node_memory = {
        big: create_node_memory(Memory.from_gb(64).in_bytes),
        small: create_node_memory(Memory.from_gb(32).in_bytes),
    }
    node_network = {big: create_node_network(), small: create_node_network()}
    resources = {big: _amd_resources(), small: _amd_resources()}
    node_vram = {big: Memory.from_gb(57.6), small: Memory.from_gb(28.8)}
    with pytest.raises(PlacementError, match="requires one engine common"):
        place_instance(
            _command(card, min_nodes=2),
            topology,
            {},
            node_memory,
            node_network,
            node_resources=resources,
            node_vram=node_vram,
            node_vram_strict=node_vram,
        )


def test_missing_strict_vram_is_info_pending_not_ram_fallback() -> None:
    """A node in an RPC cycle with no strict-VRAM entry (telemetry warm-up:
    NodeResources arrived, node_system's accelerator reading not yet) must
    surface as info-pending, not fall back to system RAM inside the memory
    filter — RAM fallback would over-admit a pooled model that llama-server
    then fails to load, and the RPC path bypasses the worker fit guard."""
    topology, big, small = _amd_pair()
    node_memory = {
        big: create_node_memory(Memory.from_gb(64).in_bytes),
        # Plenty of system RAM: the RAM fallback WOULD admit this cycle.
        small: create_node_memory(Memory.from_gb(64).in_bytes),
    }
    node_network = {big: create_node_network(), small: create_node_network()}
    resources = {big: _amd_resources(), small: _amd_resources()}
    # The small node is missing from the strict map entirely.
    node_vram_strict = {big: Memory.from_gb(57.6)}
    command = _command(_gguf_card(storage_gb=70.0), min_nodes=2)
    with pytest.raises(PlacementInfoPendingError, match=str(small)):
        place_instance(
            command,
            topology,
            {},
            node_memory,
            node_network,
            node_resources=resources,
            node_vram={big: Memory.from_gb(57.6), small: Memory.from_gb(28.8)},
            node_vram_strict=node_vram_strict,
        )


def test_ipv6_only_path_fails_placement() -> None:
    """A pair whose only observed path is IPv6 must fail cleanly: donor
    endpoints are stamped as bare host:port, which llama-server's --rpc
    parsing does not accept for IPv6, so stamping one would mint an instance
    whose driver can never load."""
    topology = Topology()
    big = NodeId()
    small = NodeId()
    topology.add_node(big)
    topology.add_node(small)
    def ipv6_edge(host: int) -> SocketConnection:
        return SocketConnection(
            sink_multiaddr=Multiaddr(address=f"/ip6/fd00::{host}/tcp/1234")
        )

    topology.add_connection(Connection(source=big, sink=small, edge=ipv6_edge(2)))
    topology.add_connection(Connection(source=small, sink=big, edge=ipv6_edge(1)))
    node_memory = {
        big: create_node_memory(Memory.from_gb(64).in_bytes),
        small: create_node_memory(Memory.from_gb(32).in_bytes),
    }
    node_network = {big: create_node_network(), small: create_node_network()}
    resources = {big: _amd_resources(), small: _amd_resources()}
    node_vram = {big: Memory.from_gb(57.6), small: Memory.from_gb(28.8)}
    command = _command(_gguf_card(storage_gb=55.0))
    with pytest.raises(ValueError, match="ROUTABLE"):
        place_instance(
            command,
            topology,
            {},
            node_memory,
            node_network,
            node_resources=resources,
            node_vram=node_vram,
            node_vram_strict=node_vram,
        )


def test_rpc_shards_stamp_llama_server_even_when_llama_cpp_sorts_first() -> None:
    """A card allowing both engines with NO preference must still stamp RPC
    shards with the llama_server tag: alphabetical resolution would pick
    llama_cpp-vulkan and dispatch the driver to the single-node in-process
    runner."""
    topology, big, small = _amd_pair()
    node_memory = {
        big: create_node_memory(Memory.from_gb(64).in_bytes),
        small: create_node_memory(Memory.from_gb(32).in_bytes),
    }
    node_network = {big: create_node_network(), small: create_node_network()}
    resources = {big: _amd_resources(), small: _amd_resources()}
    node_vram = {big: Memory.from_gb(57.6), small: Memory.from_gb(28.8)}
    card = ModelCard(
        model_id=ModelId("test/no-preference"),
        storage_size=Memory.from_gb(55.0),
        n_layers=36,
        hidden_size=2880,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        placement=PlacementCardConfig(
            compatible_backends=frozenset(
                {"llama_cpp-vulkan", "llama_server-vulkan"}
            ),
            # No backend_preference: sorted() puts llama_cpp-vulkan first.
        ),
    )
    placements = place_instance(
        _command(card),
        topology,
        {},
        node_memory,
        node_network,
        node_resources=resources,
        node_vram=node_vram,
        node_vram_strict=node_vram,
    )
    instance = next(iter(placements.values()))
    assert isinstance(instance, LlamaRpcInstance)
    for shard in instance.shard_assignments.runner_to_shard.values():
        assert shard.resolved_backend == "llama_server-vulkan"


def test_small_gguf_still_prefers_single_node() -> None:
    """Smallest-cycle ranking is untouched: a model that fits one node places
    single-node exactly as before the RPC shape existed."""
    topology, big, small = _amd_pair()
    node_memory = {
        big: create_node_memory(Memory.from_gb(64).in_bytes),
        small: create_node_memory(Memory.from_gb(32).in_bytes),
    }
    node_network = {big: create_node_network(), small: create_node_network()}
    resources = {big: _amd_resources(), small: _amd_resources()}
    node_vram = {big: Memory.from_gb(57.6), small: Memory.from_gb(28.8)}
    command = _command(_gguf_card(storage_gb=5.0))
    placements = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        node_resources=resources,
        node_vram=node_vram,
        node_vram_strict=node_vram,
    )
    instance = next(iter(placements.values()))
    assert not isinstance(instance, LlamaRpcInstance)
    assert len(instance.shard_assignments.node_to_runner) == 1


def test_excluded_node_is_neither_driver_nor_donor() -> None:
    """Operator exclusion keeps working: excluding a node a pooled model needs
    yields a clean error, never a placement touching the excluded node."""
    topology, big, small = _amd_pair()
    node_memory = {
        big: create_node_memory(Memory.from_gb(64).in_bytes),
        small: create_node_memory(Memory.from_gb(32).in_bytes),
    }
    node_network = {big: create_node_network(), small: create_node_network()}
    resources = {big: _amd_resources(), small: _amd_resources()}
    node_vram = {big: Memory.from_gb(57.6), small: Memory.from_gb(28.8)}
    command = _command(_gguf_card(storage_gb=55.0))
    with pytest.raises(PlacementError):
        place_instance(
            command,
            topology,
            {},
            node_memory,
            node_network,
            excluded_nodes={big},
            node_resources=resources,
            node_vram=node_vram,
            node_vram_strict=node_vram,
        )


def test_rpc_admission_uses_strict_vram() -> None:
    """Pooled admission must use the VRAM-carve-only figure: a pool that only
    fits with the UMA/GTT spill is rejected (spike-measured: RPC allocations
    never touch GTT)."""
    topology, big, small = _amd_pair()
    node_memory = {
        big: create_node_memory(Memory.from_gb(64).in_bytes),
        small: create_node_memory(Memory.from_gb(32).in_bytes),
    }
    node_network = {big: create_node_network(), small: create_node_network()}
    resources = {big: _amd_resources(), small: _amd_resources()}
    # UMA-inflated view says the pool is huge; the strict carve says it is not.
    node_vram_uma = {big: Memory.from_gb(150.0), small: Memory.from_gb(40.0)}
    node_vram_strict = {big: Memory.from_gb(57.6), small: Memory.from_gb(28.8)}
    command = _command(_gguf_card(storage_gb=120.0))
    with pytest.raises(PlacementError, match="fits|No candidate cycle"):
        place_instance(
            command,
            topology,
            {},
            node_memory,
            node_network,
            node_resources=resources,
            node_vram=node_vram_uma,
            node_vram_strict=node_vram_strict,
        )


def test_tensor_request_never_lands_on_rpc_cycle() -> None:
    """The RPC shape is pipeline-only; a Tensor placement on an AMD-only
    topology fails cleanly instead of minting an unservable instance."""
    topology, big, small = _amd_pair()
    node_memory = {
        big: create_node_memory(Memory.from_gb(64).in_bytes),
        small: create_node_memory(Memory.from_gb(32).in_bytes),
    }
    node_network = {big: create_node_network(), small: create_node_network()}
    resources = {big: _amd_resources(), small: _amd_resources()}
    node_vram = {big: Memory.from_gb(57.6), small: Memory.from_gb(28.8)}
    card = _gguf_card(storage_gb=55.0)
    command = PlaceInstance(
        command_id=CommandId(),
        model_card=card,
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=2,
    )
    with pytest.raises(PlacementError, match="Tensor"):
        place_instance(
            command,
            topology,
            {},
            node_memory,
            node_network,
            node_resources=resources,
            node_vram=node_vram,
            node_vram_strict=node_vram,
        )


def test_llama_cpp_only_card_stays_single_node() -> None:
    """The in-process llama.cpp engine remains single-node: a card without the
    llama_server tags never places multi-node even on the AMD pair."""
    topology, big, small = _amd_pair()
    node_memory = {
        big: create_node_memory(Memory.from_gb(64).in_bytes),
        small: create_node_memory(Memory.from_gb(32).in_bytes),
    }
    node_network = {big: create_node_network(), small: create_node_network()}
    resources = {big: _amd_resources(), small: _amd_resources()}
    card = ModelCard(
        model_id=ModelId("test/llama-cpp-only"),
        storage_size=Memory.from_gb(55.0),
        n_layers=36,
        hidden_size=2880,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        placement=PlacementCardConfig(
            compatible_backends=frozenset({"llama_cpp-vulkan"})
        ),
    )
    node_vram = {big: Memory.from_gb(57.6), small: Memory.from_gb(28.8)}
    command = _command(card, min_nodes=2)
    with pytest.raises(PlacementError, match="requires one engine common"):
        place_instance(
            command,
            topology,
            {},
            node_memory,
            node_network,
            node_resources=resources,
            node_vram=node_vram,
            node_vram_strict=node_vram,
        )
