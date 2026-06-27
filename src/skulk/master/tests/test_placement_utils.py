import pytest

from skulk.master.placement_utils import (
    allocate_layers_proportionally,
    filter_cycles_by_memory,
    get_mlx_jaccl_coordinators,
    get_shard_assignments,
    get_shard_assignments_for_pipeline_parallel,
    get_smallest_cycles,
    usable_vram_by_node,
)
from skulk.master.tests.conftest import (
    create_node_memory,
    create_socket_connection,
)
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.topology import Topology
from skulk.shared.types.common import NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import (
    AcceleratorMetrics,
    AcceleratorVendor,
    NetworkInterfaceInfo,
    NodeNetworkInfo,
    NodeResources,
    SystemPerformanceProfile,
)
from skulk.shared.types.topology import Connection, SocketConnection
from skulk.shared.types.worker.shards import (
    CfgShardMetadata,
    PipelineShardMetadata,
    Sharding,
)


def _card(
    storage_gb: float,
    *,
    kv_heads: int | None = None,
    n_layers: int = 1,
    context_length: int = 0,
    gguf_file: str | None = None,
) -> ModelCard:
    """Minimal ModelCard for memory-filter tests.

    Defaults to no KV heads so the KV reservation is zero and the
    weight-overhead path is exercised in isolation; pass ``kv_heads`` /
    ``n_layers`` / ``context_length`` to drive the KV term. Pass ``gguf_file`` to
    mark the card as a GGUF/llama.cpp model (lighter overhead factor).
    """
    return ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_gb(storage_gb),
        n_layers=n_layers,
        hidden_size=1000,
        supports_tensor=True,
        num_key_value_heads=kv_heads,
        context_length=context_length,
        gguf_file=gguf_file,
        tasks=[ModelTask.TextGeneration],
    )


def test_filter_cycles_by_memory():
    # arrange
    node1_id = NodeId()
    node2_id = NodeId()
    connection1 = Connection(
        source=node1_id, sink=node2_id, edge=create_socket_connection(1)
    )
    connection2 = Connection(
        source=node2_id, sink=node1_id, edge=create_socket_connection(2)
    )

    node1_mem = create_node_memory(Memory.from_gb(8).in_bytes)
    node2_mem = create_node_memory(Memory.from_gb(8).in_bytes)
    node_memory = {node1_id: node1_mem, node2_id: node2_mem}

    topology = Topology()
    topology.add_node(node1_id)
    topology.add_node(node2_id)
    topology.add_connection(connection1)
    topology.add_connection(connection2)

    cycles = [c for c in topology.get_cycles() if len(c) != 1]
    assert len(cycles) == 1
    assert len(cycles[0]) == 2

    # act
    filtered_cycles, diagnostics = filter_cycles_by_memory(
        cycles, node_memory, _card(4)
    )

    # assert
    assert len(filtered_cycles) == 1
    assert len(filtered_cycles[0]) == 2
    assert set(n for n in filtered_cycles[0]) == {node1_id, node2_id}
    assert diagnostics.pending_info_node_ids == []
    assert diagnostics.rejection_reasons == []


def test_filter_cycles_by_insufficient_memory():
    # arrange
    node1_id = NodeId()
    node2_id = NodeId()
    connection1 = Connection(
        source=node1_id, sink=node2_id, edge=create_socket_connection(1)
    )
    connection2 = Connection(
        source=node2_id, sink=node1_id, edge=create_socket_connection(2)
    )

    node1_mem = create_node_memory(Memory.from_gb(8).in_bytes)
    node2_mem = create_node_memory(Memory.from_gb(8).in_bytes)
    node_memory = {node1_id: node1_mem, node2_id: node2_mem}

    topology = Topology()
    topology.add_node(node1_id)
    topology.add_node(node2_id)
    topology.add_connection(connection1)
    topology.add_connection(connection2)

    # act
    filtered_cycles, diagnostics = filter_cycles_by_memory(
        topology.get_cycles(), node_memory, _card(17)
    )

    # assert
    assert len(filtered_cycles) == 0
    assert diagnostics.rejection_reasons  # every cycle rejected with a reason
    assert diagnostics.pending_info_node_ids == []


def test_heterogeneous_pipeline_split_weighs_by_usable_not_raw_available():
    """A heterogeneous cycle that fits must not be rejected by over-weighting.

    The pipeline split and the per-node admission must both weigh by
    ``_node_usable_memory`` (capped at GPU_WORKING_SET_FRACTION * ram_total),
    not raw ``ram_available``. Otherwise a node whose free RAM exceeds its GPU
    ceiling gets a share larger than it can wire, and a placement that would
    fit is rejected.

    Concrete pair (model = 15 GB weights, no KV):
      - small: 16 GB RAM, 15 GB free  -> usable cap 12 GB
      - large: 64 GB RAM,  9 GB free  -> usable cap  9 GB
    Splitting by raw available (15 / 24) assigns the small node ~9.4 GB, which
    with the 1.30x overhead needs ~12.4 GB > its 12 GB cap -> wrongly rejected.
    Splitting by usable (12 / 21) assigns it ~8.6 GB (~11.4 GB with overhead),
    and both nodes fit -> admitted. This is exactly the mixed-memory cluster
    the memory-aware placement is meant to support.
    """
    node_small = NodeId()
    node_large = NodeId()
    topology = Topology()
    topology.add_node(node_small)
    topology.add_node(node_large)
    topology.add_connection(
        Connection(source=node_small, sink=node_large, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_large, sink=node_small, edge=create_socket_connection(2))
    )

    node_memory = {
        node_small: create_node_memory(
            Memory.from_gb(15).in_bytes, ram_total=Memory.from_gb(16).in_bytes
        ),
        node_large: create_node_memory(
            Memory.from_gb(9).in_bytes, ram_total=Memory.from_gb(64).in_bytes
        ),
    }

    cycles = [c for c in topology.get_cycles() if len(c) == 2]
    assert len(cycles) == 1

    filtered_cycles, diagnostics = filter_cycles_by_memory(
        cycles, node_memory, _card(15), sharding=Sharding.Pipeline
    )

    assert len(filtered_cycles) == 1, diagnostics.rejection_reasons
    assert diagnostics.rejection_reasons == []


def test_filter_multiple_cycles_by_memory():
    # arrange
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()
    connection1 = Connection(
        source=node_a_id, sink=node_b_id, edge=create_socket_connection(1)
    )
    connection2 = Connection(
        source=node_b_id, sink=node_a_id, edge=create_socket_connection(2)
    )
    connection3 = Connection(
        source=node_a_id, sink=node_c_id, edge=create_socket_connection(3)
    )
    connection4 = Connection(
        source=node_c_id, sink=node_b_id, edge=create_socket_connection(4)
    )

    node_a_mem = create_node_memory(Memory.from_gb(5).in_bytes)
    node_b_mem = create_node_memory(Memory.from_gb(5).in_bytes)
    node_c_mem = create_node_memory(Memory.from_gb(10).in_bytes)
    node_memory = {
        node_a_id: node_a_mem,
        node_b_id: node_b_mem,
        node_c_id: node_c_mem,
    }

    topology = Topology()
    topology.add_node(node_a_id)
    topology.add_node(node_b_id)
    topology.add_node(node_c_id)
    topology.add_connection(connection1)
    topology.add_connection(connection2)
    topology.add_connection(connection3)
    topology.add_connection(connection4)

    cycles = topology.get_cycles()

    # act
    filtered_cycles, _diagnostics = filter_cycles_by_memory(
        cycles, node_memory, _card(12)
    )

    # assert
    assert len(filtered_cycles) == 1
    assert len(filtered_cycles[0]) == 3
    assert set(n for n in filtered_cycles[0]) == {
        node_a_id,
        node_b_id,
        node_c_id,
    }


def test_filter_cycles_tensor_rejects_uneven_pair_that_sums():
    """Tensor sharding splits weights evenly, so a 16+24 GB pair whose SUM
    covers the model must still be rejected when the even split overloads
    the smaller node. The old sum-across-cycle check admitted exactly this
    and produced the silent-thrash placements from the 2026-06-05 smoke."""
    node_small = NodeId()
    node_large = NodeId()
    topology = Topology()
    topology.add_node(node_small)
    topology.add_node(node_large)
    topology.add_connection(
        Connection(source=node_small, sink=node_large, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_large, sink=node_small, edge=create_socket_connection(2))
    )
    node_memory = {
        node_small: create_node_memory(Memory.from_gb(7).in_bytes),
        node_large: create_node_memory(Memory.from_gb(14).in_bytes),
    }
    cycles = [c for c in topology.get_cycles() if len(c) == 2]

    # 18 GB of weights: sums under 21 GB, but the 9 GB even split plus
    # headroom does not fit the 7 GB node.
    fitting, diagnostics = filter_cycles_by_memory(
        cycles, node_memory, _card(18), Sharding.Tensor
    )

    assert fitting == []
    assert len(diagnostics.rejection_reasons) == 1
    assert str(node_small) in diagnostics.rejection_reasons[0]

    # The same pair under Pipeline sharding allocates proportionally
    # (6 GB / 12 GB shares) and fits.
    fitting_pipeline, _ = filter_cycles_by_memory(
        cycles, node_memory, _card(14), Sharding.Pipeline
    )
    assert len(fitting_pipeline) == 1


def test_filter_cycles_reports_pending_node_info():
    """A cycle touching a node with no memory info yet must be reported as
    info-pending, not silently dropped: placing right after cluster
    formation (identities gossiped, NodeGatheredInfo not yet) is a
    retry-shortly condition, not a memory shortfall."""
    node_known = NodeId()
    node_pending = NodeId()
    topology = Topology()
    topology.add_node(node_known)
    topology.add_node(node_pending)
    topology.add_connection(
        Connection(
            source=node_known, sink=node_pending, edge=create_socket_connection(1)
        )
    )
    topology.add_connection(
        Connection(
            source=node_pending, sink=node_known, edge=create_socket_connection(2)
        )
    )
    node_memory = {node_known: create_node_memory(Memory.from_gb(16).in_bytes)}
    cycles = [c for c in topology.get_cycles() if len(c) == 2]

    fitting, diagnostics = filter_cycles_by_memory(cycles, node_memory, _card(4))

    assert fitting == []
    assert diagnostics.pending_info_node_ids == [node_pending]
    assert diagnostics.rejection_reasons == []


def test_filter_cycles_rejects_exact_fit_without_headroom():
    """Available memory exactly equal to the weight bytes is a guaranteed
    thrash (no room for KV cache, activations, or the runner); the filter
    must demand runtime headroom on top of the weights."""
    node_id = NodeId()
    topology = Topology()
    topology.add_node(node_id)
    node_memory = {node_id: create_node_memory(Memory.from_gb(8).in_bytes)}
    cycles = topology.get_cycles()

    fitting, diagnostics = filter_cycles_by_memory(cycles, node_memory, _card(8))
    assert fitting == []
    assert diagnostics.rejection_reasons

    # A 5 GB model on the 8 GB node clears the 1.30x weight-overhead wall
    # (5 * 1.30 + 0.25 floor = ~6.8 GB <= 8 GB), so it is admitted.
    fitting_with_slack, _ = filter_cycles_by_memory(cycles, node_memory, _card(5))
    assert len(fitting_with_slack) == 1


def test_filter_cycles_admits_against_discrete_vram():
    """A discrete-VRAM GPU node (e.g. Strix Halo: ~64 GB system + 64 GB VRAM)
    must admit a model that fits its VRAM even though it exceeds 0.75 x system
    RAM, because the GPU-offload engine (llama.cpp/vLLM) allocates from VRAM, not
    system RAM. This is the kite4 big-model placement bug."""
    node_id = NodeId()
    topology = Topology()
    topology.add_node(node_id)
    # 50 GB system RAM -> system ceiling = 0.75 * 50 = 37.5 GB; 80 GB VRAM ->
    # usable = 0.90 * 80 = 72 GB.
    node_memory = {
        node_id: create_node_memory(
            Memory.from_gb(48).in_bytes, ram_total=Memory.from_gb(50).in_bytes
        )
    }
    node_vram = {node_id: Memory.from_gb(80) * 0.90}
    cycles = topology.get_cycles()

    # 40 GB weights: 40 * 1.30 + floor ~= 52.25 GB. Exceeds the 37.5 GB system
    # ceiling, so VRAM-blind admission rejects it.
    rejected, diagnostics = filter_cycles_by_memory(cycles, node_memory, _card(40))
    assert rejected == []
    assert diagnostics.rejection_reasons

    # With the node's discrete VRAM, 52.25 GB fits the 72 GB VRAM budget.
    fitting, _ = filter_cycles_by_memory(
        cycles, node_memory, _card(40), node_vram=node_vram
    )
    assert len(fitting) == 1


def test_usable_vram_by_node_selects_discrete_gpus_only():
    """The VRAM map covers only AMD/NVIDIA nodes with a nonzero VRAM reading,
    nets out VRAM already in use, and caps at the working-set fraction. Apple
    unified-memory nodes (no discrete VRAM) are absent, keeping the RAM path."""
    amd = NodeId()
    apple = NodeId()
    nvidia_busy = NodeId()

    def _profile(
        vendor: AcceleratorVendor, total_gb: float | None, used_gb: float = 0.0
    ) -> SystemPerformanceProfile:
        acc = AcceleratorMetrics(
            vendor=vendor,
            vram_total_bytes=(
                Memory.from_gb(total_gb).in_bytes if total_gb is not None else None
            ),
            vram_used_bytes=Memory.from_gb(used_gb).in_bytes,
        )
        return SystemPerformanceProfile(accelerator=acc)

    cpu_only = NodeId()
    node_system = {
        amd: _profile("amd", 64.0),
        apple: _profile("apple", None),
        nvidia_busy: _profile("nvidia", 24.0, used_gb=20.0),
        cpu_only: _profile("amd", 64.0),
    }
    usable = usable_vram_by_node(node_system)

    assert apple not in usable  # unified memory -> no discrete VRAM entry
    # AMD idle: min(64 - 0, 0.90 * 64) = 57.6 GB.
    assert usable[amd].in_bytes == int(Memory.from_gb(64).in_bytes * 0.90)
    # NVIDIA with 20/24 GB in use: available (4 GB) is below the 0.90 cap, so it
    # nets out the in-use VRAM rather than assuming the whole card is free.
    assert usable[nvidia_busy].in_bytes == Memory.from_gb(4).in_bytes

    # Backend gate: with node_resources, a discrete-GPU node that advertises only
    # llama_cpp-cpu (runs GGUF on CPU out of system RAM) is excluded, while a
    # vulkan node keeps its VRAM entry.
    resources = {
        amd: NodeResources(backends=frozenset({"llama_cpp", "llama_cpp-vulkan"})),
        cpu_only: NodeResources(backends=frozenset({"llama_cpp", "llama_cpp-cpu"})),
    }
    gated = usable_vram_by_node(node_system, resources)
    assert amd in gated
    assert cpu_only not in gated  # CPU-only backend -> no VRAM admission
    assert nvidia_busy not in gated  # no resources entry -> excluded under gating


def test_usable_vram_by_node_admits_served_engine_gpu_nodes():
    """A served-backend (llama_server) GPU node offloads weights+KV to the GPU
    (llama-server -ngl 99) exactly like the in-process llama.cpp runner, so it must
    be admitted against VRAM. A served node advertising only llama_server-cpu is
    not, like its llama_cpp-cpu counterpart."""
    served_gpu = NodeId()
    served_cpu = NodeId()
    acc = AcceleratorMetrics(vendor="amd", vram_total_bytes=Memory.from_gb(64).in_bytes)
    node_system = {
        served_gpu: SystemPerformanceProfile(accelerator=acc),
        served_cpu: SystemPerformanceProfile(accelerator=acc),
    }
    resources = {
        served_gpu: NodeResources(
            backends=frozenset({"llama_server", "llama_server-vulkan"})
        ),
        served_cpu: NodeResources(
            backends=frozenset({"llama_server", "llama_server-cpu"})
        ),
    }
    gated = usable_vram_by_node(node_system, resources)
    assert served_gpu in gated  # GPU-offload served engine -> VRAM-admitted
    assert served_cpu not in gated  # CPU-only -> no VRAM admission


def test_usable_vram_by_node_uma_counts_gtt():
    """A unified-memory APU (Strix Halo: GTT spans system RAM) must count the
    GPU's GTT-mapped system RAM, not just the BIOS VRAM carve-out. With 64 GiB
    VRAM, 124 GiB GTT, and 59 GiB free of 61 GiB system RAM, usable GPU memory is
    the working-set-capped VRAM (0.9*64 = 57.6) + (59 - 16 headroom) = 100.6 GiB,
    far above the bare 0.9*64, so a 58.5 GiB GGUF (e.g. gpt-oss-120B) places."""
    node_id = NodeId()
    accelerator = AcceleratorMetrics(
        vendor="amd",
        vram_total_bytes=Memory.from_gb(64).in_bytes,
        vram_used_bytes=0,
        gtt_total_bytes=Memory.from_gb(124).in_bytes,
    )
    node_system = {node_id: SystemPerformanceProfile(accelerator=accelerator)}
    node_memory = {
        node_id: create_node_memory(
            Memory.from_gb(59).in_bytes, ram_total=Memory.from_gb(61).in_bytes
        )
    }
    resources = {
        node_id: NodeResources(backends=frozenset({"llama_cpp", "llama_cpp-vulkan"}))
    }

    usable = usable_vram_by_node(node_system, resources, node_memory=node_memory)

    expected = int(Memory.from_gb(64).in_bytes * 0.90) + (
        Memory.from_gb(59).in_bytes - Memory.from_gb(16).in_bytes
    )
    assert usable[node_id].in_bytes == expected
    # Clearly above the VRAM-only ceiling and large enough for a 58.5 GiB model.
    assert usable[node_id].in_bytes > int(Memory.from_gb(64).in_bytes * 0.90)
    assert usable[node_id].in_bytes > Memory.from_gb(58.5).in_bytes

    # The same node admits the 58.5 GiB GGUF through filter_cycles_by_memory,
    # which it would not against the bare 0.9*64 = 57.6 GiB VRAM ceiling.
    topology = Topology()
    topology.add_node(node_id)
    fitting, diagnostics = filter_cycles_by_memory(
        topology.get_cycles(),
        node_memory,
        _card(58.5, gguf_file="gpt-oss-120b.gguf"),
        node_vram=usable,
    )
    assert len(fitting) == 1, diagnostics.rejection_reasons


def test_usable_vram_by_node_discrete_without_gtt_uses_vram_only():
    """A discrete GPU (no GTT, or GTT smaller than VRAM) must keep the VRAM-only
    0.9*vram path even when node_memory is supplied: the UMA branch is for APUs
    whose GTT spans system memory, not for dedicated cards."""
    no_gtt = NodeId()
    small_gtt = NodeId()
    gtt_equals_vram = NodeId()
    node_system = {
        no_gtt: SystemPerformanceProfile(
            accelerator=AcceleratorMetrics(
                vendor="nvidia",
                vram_total_bytes=Memory.from_gb(48).in_bytes,
                vram_used_bytes=0,
                gtt_total_bytes=None,
            )
        ),
        small_gtt: SystemPerformanceProfile(
            accelerator=AcceleratorMetrics(
                vendor="amd",
                vram_total_bytes=Memory.from_gb(48).in_bytes,
                vram_used_bytes=0,
                # GTT present but smaller than VRAM -> not a system-spanning APU.
                gtt_total_bytes=Memory.from_gb(8).in_bytes,
            )
        ),
        gtt_equals_vram: SystemPerformanceProfile(
            accelerator=AcceleratorMetrics(
                vendor="amd",
                vram_total_bytes=Memory.from_gb(48).in_bytes,
                vram_used_bytes=0,
                # A discrete amdgpu card's GTT default can EQUAL its VRAM. That is
                # not a system-spanning aperture (gtt < system RAM), so it must
                # stay on the VRAM-only path instead of over-admitting 128 GiB.
                gtt_total_bytes=Memory.from_gb(48).in_bytes,
            )
        ),
    }
    node_memory = {
        no_gtt: create_node_memory(
            Memory.from_gb(120).in_bytes, ram_total=Memory.from_gb(128).in_bytes
        ),
        small_gtt: create_node_memory(
            Memory.from_gb(120).in_bytes, ram_total=Memory.from_gb(128).in_bytes
        ),
        gtt_equals_vram: create_node_memory(
            Memory.from_gb(120).in_bytes, ram_total=Memory.from_gb(128).in_bytes
        ),
    }

    usable = usable_vram_by_node(node_system, node_memory=node_memory)

    vram_only = int(Memory.from_gb(48).in_bytes * 0.90)
    assert usable[no_gtt].in_bytes == vram_only
    assert usable[small_gtt].in_bytes == vram_only
    assert usable[gtt_equals_vram].in_bytes == vram_only


def test_filter_cycles_respects_gpu_working_set_ceiling():
    """A node can report generous ram_available yet still abort if the shard
    exceeds Metal's GPU working set (~0.75 * ram_total). The filter caps usable
    memory at that ceiling, so a shard that fits 'available' RAM but not the GPU
    budget is rejected — the GLM-4.7-Flash failure class (2026-06-08)."""
    node_id = NodeId()
    topology = Topology()
    topology.add_node(node_id)
    # 14 GB free out of 16 GB total -> GPU ceiling = 0.75 * 16 = 12 GB.
    node_memory = {
        node_id: create_node_memory(
            Memory.from_gb(14).in_bytes, ram_total=Memory.from_gb(16).in_bytes
        )
    }
    cycles = topology.get_cycles()

    # 10 GB weights: 10 * 1.30 + floor ~= 13.3 GB. Fits the 14 GB available but
    # exceeds the 12 GB GPU ceiling -> rejected.
    rejected, diagnostics = filter_cycles_by_memory(cycles, node_memory, _card(10))
    assert rejected == []
    assert diagnostics.rejection_reasons

    # 8 GB weights: 8 * 1.30 + floor ~= 10.7 GB, under the 12 GB ceiling -> fits.
    fitting, _ = filter_cycles_by_memory(cycles, node_memory, _card(8))
    assert len(fitting) == 1


def test_filter_cycles_reserves_kv_cache():
    """The KV-cache reservation must count against capacity: a model whose
    weights alone fit can be rejected once KV for the planning context budget is
    added on top."""
    node_id = NodeId()
    topology = Topology()
    topology.add_node(node_id)
    # 12 GB available, ram_total high enough the GPU ceiling does not bind.
    node_memory = {
        node_id: create_node_memory(
            Memory.from_gb(12).in_bytes, ram_total=Memory.from_gb(48).in_bytes
        )
    }
    cycles = topology.get_cycles()

    # 8 GB weights alone: 8 * 1.30 + floor ~= 10.7 GB <= 12 GB -> fits.
    fitting, _ = filter_cycles_by_memory(cycles, node_memory, _card(8))
    assert len(fitting) == 1

    # Same weights, but reserving KV for 32k tokens (32 layers, 8 KV heads)
    # adds several GB and pushes it over 12 GB -> rejected.
    kv_card = _card(8, kv_heads=8, n_layers=32)
    rejected, diagnostics = filter_cycles_by_memory(
        cycles, node_memory, kv_card, context_budget=32768
    )
    assert rejected == []
    assert "KV@32768tok" in diagnostics.rejection_reasons[0]


def test_get_smallest_cycles():
    # arrange
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()

    topology = Topology()
    topology.add_node(node_a_id)
    topology.add_node(node_b_id)
    topology.add_node(node_c_id)

    connection1 = Connection(
        source=node_a_id, sink=node_b_id, edge=create_socket_connection(1)
    )
    connection2 = Connection(
        source=node_b_id, sink=node_a_id, edge=create_socket_connection(2)
    )
    connection3 = Connection(
        source=node_a_id, sink=node_c_id, edge=create_socket_connection(3)
    )
    connection4 = Connection(
        source=node_c_id, sink=node_b_id, edge=create_socket_connection(4)
    )

    topology.add_connection(connection1)
    topology.add_connection(connection2)
    topology.add_connection(connection3)
    topology.add_connection(connection4)

    cycles = [c for c in topology.get_cycles() if len(c) != 1]  # ignore singletons

    # act
    smallest_cycles = get_smallest_cycles(cycles)

    # assert
    assert len(smallest_cycles) == 1
    assert len(smallest_cycles[0]) == 2
    assert set(n for n in smallest_cycles[0]) == {node_a_id, node_b_id}


@pytest.mark.parametrize(
    "available_memory,total_layers,expected_layers",
    [
        ((500, 500, 1000), 12, (3, 3, 6)),
        ((500, 500, 500), 12, (4, 4, 4)),
        ((312, 518, 1024), 12, (2, 3, 7)),
        # Edge case: one node has ~90% of memory - should not over-allocate.
        # Each node must have enough memory for at least 1 layer (50 KB = 1000/20).
        ((900, 50, 50), 20, (18, 1, 1)),
    ],
)
def test_get_shard_assignments(
    available_memory: tuple[int, int, int],
    total_layers: int,
    expected_layers: tuple[int, int, int],
):
    # arrange
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()

    # create connections (A -> B -> C -> A forms a 3-cycle, plus B -> A also exists)
    connection1 = Connection(
        source=node_a_id, sink=node_b_id, edge=create_socket_connection(1)
    )
    connection2 = Connection(
        source=node_b_id, sink=node_c_id, edge=create_socket_connection(2)
    )
    connection3 = Connection(
        source=node_c_id, sink=node_a_id, edge=create_socket_connection(3)
    )
    connection4 = Connection(
        source=node_b_id, sink=node_a_id, edge=create_socket_connection(4)
    )

    topology = Topology()
    topology.add_node(node_a_id)
    topology.add_node(node_b_id)
    topology.add_node(node_c_id)
    topology.add_connection(connection1)
    topology.add_connection(connection2)
    topology.add_connection(connection3)
    topology.add_connection(connection4)

    node_a_mem = create_node_memory(available_memory[0] * 1024)
    node_b_mem = create_node_memory(available_memory[1] * 1024)
    node_c_mem = create_node_memory(available_memory[2] * 1024)
    node_memory = {
        node_a_id: node_a_mem,
        node_b_id: node_b_mem,
        node_c_id: node_c_mem,
    }

    model_card = ModelCard(
        model_id=ModelId("test-model"),
        n_layers=total_layers,
        storage_size=Memory.from_kb(1000),
        hidden_size=1000,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )

    cycles = topology.get_cycles()

    # pick the 3-node cycle deterministically (cycle ordering can vary)
    selected_cycle = next(cycle for cycle in cycles if len(cycle) == 3)

    # act
    shard_assignments = get_shard_assignments(
        model_card, selected_cycle, Sharding.Pipeline, node_memory=node_memory
    )

    # assert
    runner_id_a = shard_assignments.node_to_runner[node_a_id]
    runner_id_b = shard_assignments.node_to_runner[node_b_id]
    runner_id_c = shard_assignments.node_to_runner[node_c_id]

    assert (
        shard_assignments.runner_to_shard[runner_id_a].end_layer
        - shard_assignments.runner_to_shard[runner_id_a].start_layer
        == expected_layers[0]
    )
    assert (
        shard_assignments.runner_to_shard[runner_id_b].end_layer
        - shard_assignments.runner_to_shard[runner_id_b].start_layer
        == expected_layers[1]
    )
    assert (
        shard_assignments.runner_to_shard[runner_id_c].end_layer
        - shard_assignments.runner_to_shard[runner_id_c].start_layer
        == expected_layers[2]
    )


def test_get_mlx_jaccl_coordinators():
    # arrange
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()

    # fully connected (directed) between the 3 nodes
    conn_a_b = Connection(
        source=node_a_id, sink=node_b_id, edge=create_socket_connection(1)
    )
    conn_b_a = Connection(
        source=node_b_id, sink=node_a_id, edge=create_socket_connection(2)
    )
    conn_b_c = Connection(
        source=node_b_id, sink=node_c_id, edge=create_socket_connection(3)
    )
    conn_c_b = Connection(
        source=node_c_id, sink=node_b_id, edge=create_socket_connection(4)
    )
    conn_c_a = Connection(
        source=node_c_id, sink=node_a_id, edge=create_socket_connection(5)
    )
    conn_a_c = Connection(
        source=node_a_id, sink=node_c_id, edge=create_socket_connection(6)
    )

    network_a = NodeNetworkInfo(
        interfaces=[
            NetworkInterfaceInfo(name="en0", ip_address="169.254.0.5"),
            NetworkInterfaceInfo(name="en0", ip_address="169.254.0.2"),
        ]
    )
    network_b = NodeNetworkInfo(
        interfaces=[
            NetworkInterfaceInfo(name="en0", ip_address="169.254.0.1"),
            NetworkInterfaceInfo(name="en0", ip_address="169.254.0.4"),
        ]
    )
    network_c = NodeNetworkInfo(
        interfaces=[
            NetworkInterfaceInfo(name="en0", ip_address="169.254.0.3"),
            NetworkInterfaceInfo(name="en0", ip_address="169.254.0.6"),
        ]
    )
    node_network = {
        node_a_id: network_a,
        node_b_id: network_b,
        node_c_id: network_c,
    }

    topology = Topology()
    topology.add_node(node_a_id)
    topology.add_node(node_b_id)
    topology.add_node(node_c_id)

    topology.add_connection(conn_a_b)
    topology.add_connection(conn_b_a)
    topology.add_connection(conn_b_c)
    topology.add_connection(conn_c_b)
    topology.add_connection(conn_c_a)
    topology.add_connection(conn_a_c)

    # act
    coordinators = get_mlx_jaccl_coordinators(
        node_a_id,
        coordinator_port=5000,
        cycle_digraph=topology,
        node_network=node_network,
    )

    # assert
    assert len(coordinators) == 3
    assert node_a_id in coordinators
    assert node_b_id in coordinators
    assert node_c_id in coordinators

    # All coordinators should have IP:PORT format
    for node_id, coordinator in coordinators.items():
        assert ":" in coordinator, (
            f"Coordinator for {node_id} should have ':' separator"
        )

    # Verify port is correct
    for node_id, coordinator in coordinators.items():
        assert coordinator.endswith(":5000"), (
            f"Coordinator for {node_id} should use port 5000"
        )

    # Rank 0 (node_a) treats this as the listen socket so should listen on all IPs
    assert coordinators[node_a_id].startswith("0.0.0.0:"), (
        "Rank 0 node should use 0.0.0.0 as coordinator listen address"
    )

    # Non-rank-0 nodes should use the specific IP from their connection to rank 0
    # node_b uses the IP from conn_b_a (node_b -> node_a)
    assert isinstance(conn_b_a.edge, SocketConnection)
    assert (
        coordinators[node_b_id] == f"{conn_b_a.edge.sink_multiaddr.ip_address}:5000"
    ), "node_b should use the IP from conn_b_a"

    # node_c uses the IP from conn_c_a (node_c -> node_a)
    assert isinstance(conn_c_a.edge, SocketConnection)
    assert coordinators[node_c_id] == (
        f"{conn_c_a.edge.sink_multiaddr.ip_address}:5000"
    ), "node_c should use the IP from conn_c_a"


class TestAllocateLayersProportionally:
    def test_empty_node_list_raises(self):
        with pytest.raises(ValueError, match="empty node list"):
            allocate_layers_proportionally(total_layers=10, memory_fractions=[])

    def test_zero_layers_raises(self):
        with pytest.raises(ValueError, match="need at least 1 layer per node"):
            allocate_layers_proportionally(total_layers=0, memory_fractions=[0.5, 0.5])

    def test_negative_layers_raises(self):
        with pytest.raises(ValueError, match="need at least 1 layer per node"):
            allocate_layers_proportionally(total_layers=-1, memory_fractions=[0.5, 0.5])

    def test_fewer_layers_than_nodes_raises(self):
        with pytest.raises(ValueError, match="need at least 1 layer per node"):
            allocate_layers_proportionally(
                total_layers=2, memory_fractions=[0.33, 0.33, 0.34]
            )

    def test_equal_distribution(self):
        result = allocate_layers_proportionally(
            total_layers=12, memory_fractions=[0.25, 0.25, 0.25, 0.25]
        )
        assert result == [3, 3, 3, 3]
        assert sum(result) == 12

    def test_proportional_distribution(self):
        result = allocate_layers_proportionally(
            total_layers=12, memory_fractions=[0.25, 0.25, 0.50]
        )
        assert result == [3, 3, 6]
        assert sum(result) == 12

    def test_extreme_imbalance_ensures_minimum(self):
        result = allocate_layers_proportionally(
            total_layers=20, memory_fractions=[0.975, 0.0125, 0.0125]
        )
        assert all(layers >= 1 for layers in result)
        assert sum(result) == 20
        # Small nodes get minimum 1 layer
        assert result == [18, 1, 1]

    def test_single_node_gets_all_layers(self):
        result = allocate_layers_proportionally(total_layers=10, memory_fractions=[1.0])
        assert result == [10]

    def test_minimum_viable_allocation(self):
        result = allocate_layers_proportionally(
            total_layers=3, memory_fractions=[0.33, 0.33, 0.34]
        )
        assert result == [1, 1, 1]
        assert sum(result) == 3


def test_get_shard_assignments_insufficient_memory_raises():
    """Test that ValueError is raised when a node has insufficient memory for its layers."""
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()
    topology = Topology()

    # Node C has only 10 KB but would need 50 KB for 1 layer (1000 KB / 20 layers)
    node_a_mem = create_node_memory(900 * 1024)
    node_b_mem = create_node_memory(50 * 1024)
    node_c_mem = create_node_memory(10 * 1024)  # Insufficient memory

    topology.add_node(node_a_id)
    topology.add_node(node_b_id)
    topology.add_node(node_c_id)

    conn_a_b = Connection(
        source=node_a_id, sink=node_b_id, edge=create_socket_connection(1)
    )
    conn_b_c = Connection(
        source=node_b_id, sink=node_c_id, edge=create_socket_connection(2)
    )
    conn_c_a = Connection(
        source=node_c_id, sink=node_a_id, edge=create_socket_connection(3)
    )
    conn_b_a = Connection(
        source=node_b_id, sink=node_a_id, edge=create_socket_connection(3)
    )
    topology.add_connection(conn_a_b)
    topology.add_connection(conn_b_c)
    topology.add_connection(conn_c_a)
    topology.add_connection(conn_b_a)

    node_memory = {
        node_a_id: node_a_mem,
        node_b_id: node_b_mem,
        node_c_id: node_c_mem,
    }

    model_card = ModelCard(
        model_id=ModelId("test-model"),
        n_layers=20,
        storage_size=Memory.from_kb(1000),
        hidden_size=1000,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )
    cycles = topology.get_cycles()
    selected_cycle = cycles[0]

    with pytest.raises(ValueError, match="insufficient memory"):
        get_shard_assignments(
            model_card, selected_cycle, Sharding.Pipeline, node_memory
        )


class TestCfgParallelPlacement:
    def _create_ring_topology(self, node_ids: list[NodeId]) -> Topology:
        topology = Topology()
        for node_id in node_ids:
            topology.add_node(node_id)

        for i, node_id in enumerate(node_ids):
            next_node = node_ids[(i + 1) % len(node_ids)]
            conn = Connection(
                source=node_id,
                sink=next_node,
                edge=create_socket_connection(i + 1),
            )
            topology.add_connection(conn)

        return topology

    def test_two_nodes_cfg_model_uses_cfg_parallel(self):
        """Two nodes with CFG model should use CFG parallel (no pipeline)."""
        node_a = NodeId()
        node_b = NodeId()

        topology = self._create_ring_topology([node_a, node_b])
        cycles = [c for c in topology.get_cycles() if len(c) == 2]
        cycle = cycles[0]

        node_memory = {
            node_a: create_node_memory(1000 * 1024),
            node_b: create_node_memory(1000 * 1024),
        }

        model_card = ModelCard(
            model_id=ModelId("qwen-image-test"),
            n_layers=60,
            storage_size=Memory.from_kb(1000),
            hidden_size=1,
            supports_tensor=False,
            uses_cfg=True,
            tasks=[ModelTask.TextToImage],
        )

        assignments = get_shard_assignments_for_pipeline_parallel(
            model_card, cycle, node_memory
        )

        shards = list(assignments.runner_to_shard.values())
        assert len(shards) == 2

        # CFG models should get CfgShardMetadata
        for shard in shards:
            assert isinstance(shard, CfgShardMetadata)
            # Both nodes should have all layers (no pipeline split)
            assert shard.start_layer == 0
            assert shard.end_layer == 60
            assert shard.cfg_world_size == 2
            # Each node is the only stage in its pipeline group
            assert shard.pipeline_world_size == 1
            assert shard.pipeline_rank == 0

        cfg_ranks = sorted(
            s.cfg_rank for s in shards if isinstance(s, CfgShardMetadata)
        )
        assert cfg_ranks == [0, 1]

    def test_four_nodes_cfg_model_uses_hybrid(self):
        """Four nodes with CFG model should use 2 CFG groups x 2 pipeline stages."""
        nodes = [NodeId() for _ in range(4)]

        topology = self._create_ring_topology(nodes)
        cycles = [c for c in topology.get_cycles() if len(c) == 4]
        cycle = cycles[0]

        node_memory = {n: create_node_memory(1000 * 1024) for n in nodes}

        model_card = ModelCard(
            model_id=ModelId("qwen-image-test"),
            n_layers=60,
            storage_size=Memory.from_kb(1000),
            hidden_size=1,
            supports_tensor=False,
            uses_cfg=True,
            tasks=[ModelTask.TextToImage],
        )

        assignments = get_shard_assignments_for_pipeline_parallel(
            model_card, cycle, node_memory
        )

        shards = list(assignments.runner_to_shard.values())
        assert len(shards) == 4

        # CFG models should get CfgShardMetadata
        for shard in shards:
            assert isinstance(shard, CfgShardMetadata)
            assert shard.cfg_world_size == 2
            assert shard.pipeline_world_size == 2
            assert shard.pipeline_rank in [0, 1]

        # Check we have 2 nodes in each CFG group
        cfg_0_shards = [
            s for s in shards if isinstance(s, CfgShardMetadata) and s.cfg_rank == 0
        ]
        cfg_1_shards = [
            s for s in shards if isinstance(s, CfgShardMetadata) and s.cfg_rank == 1
        ]
        assert len(cfg_0_shards) == 2
        assert len(cfg_1_shards) == 2

        # Both CFG groups should have the same layer assignments
        cfg_0_layers = [(s.start_layer, s.end_layer) for s in cfg_0_shards]
        cfg_1_layers = [(s.start_layer, s.end_layer) for s in cfg_1_shards]
        assert sorted(cfg_0_layers) == sorted(cfg_1_layers)

    def test_three_nodes_cfg_model_uses_sequential_cfg(self):
        """Three nodes (odd) with CFG model should use sequential CFG (PipelineShardMetadata)."""
        nodes = [NodeId() for _ in range(3)]

        topology = self._create_ring_topology(nodes)
        cycles = [c for c in topology.get_cycles() if len(c) == 3]
        cycle = cycles[0]

        node_memory = {n: create_node_memory(1000 * 1024) for n in nodes}

        model_card = ModelCard(
            model_id=ModelId("qwen-image-test"),
            n_layers=60,
            storage_size=Memory.from_kb(1000),
            hidden_size=1,
            supports_tensor=False,
            uses_cfg=True,
            tasks=[ModelTask.TextToImage],
        )

        assignments = get_shard_assignments_for_pipeline_parallel(
            model_card, cycle, node_memory
        )

        shards = list(assignments.runner_to_shard.values())
        assert len(shards) == 3

        # Odd node count with CFG model falls back to PipelineShardMetadata (sequential CFG)
        for shard in shards:
            assert isinstance(shard, PipelineShardMetadata)

    def test_two_nodes_non_cfg_model_uses_pipeline(self):
        """Two nodes with non-CFG model should use pure pipeline (PipelineShardMetadata)."""
        node_a = NodeId()
        node_b = NodeId()

        topology = self._create_ring_topology([node_a, node_b])
        cycles = [c for c in topology.get_cycles() if len(c) == 2]
        cycle = cycles[0]

        node_memory = {
            node_a: create_node_memory(1000 * 1024),
            node_b: create_node_memory(1000 * 1024),
        }

        model_card = ModelCard(
            model_id=ModelId("flux-test"),
            n_layers=57,
            storage_size=Memory.from_kb(1000),
            hidden_size=1,
            supports_tensor=False,
            uses_cfg=False,  # Non-CFG model
            tasks=[ModelTask.TextToImage],
        )

        assignments = get_shard_assignments_for_pipeline_parallel(
            model_card, cycle, node_memory
        )

        shards = list(assignments.runner_to_shard.values())
        assert len(shards) == 2

        # Non-CFG models should get PipelineShardMetadata
        for shard in shards:
            assert isinstance(shard, PipelineShardMetadata)

        # Should have actual layer sharding (pipeline)
        layer_ranges = sorted(
            (s.start_layer, s.end_layer)
            for s in shards
            if isinstance(s, PipelineShardMetadata)
        )
        # First shard starts at 0, last shard ends at 57
        assert layer_ranges[0][0] == 0
        assert layer_ranges[-1][1] == 57
