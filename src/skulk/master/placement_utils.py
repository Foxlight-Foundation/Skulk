import ipaddress
from collections.abc import Generator, Mapping
from typing import final

from loguru import logger
from pydantic import Field

from skulk.shared.models.memory_estimate import (
    GPU_VRAM_WORKING_SET_FRACTION,
    GPU_WORKING_SET_FRACTION,
)
from skulk.shared.models.memory_estimate import (
    KV_CONTEXT_BUDGET_TOKENS as PLACEMENT_KV_CONTEXT_BUDGET_TOKENS,
)
from skulk.shared.models.memory_estimate import (
    MEMORY_OVERHEAD_FACTOR as PLACEMENT_MEMORY_OVERHEAD_FACTOR,
)
from skulk.shared.models.memory_estimate import (
    MEMORY_OVERHEAD_FLOOR as PLACEMENT_MEMORY_OVERHEAD_FLOOR,
)
from skulk.shared.models.memory_estimate import (
    estimate_kv_cache_bytes as _estimate_kv_cache_bytes,
)
from skulk.shared.models.model_cards import ModelCard
from skulk.shared.topology import Topology
from skulk.shared.types.common import Host, NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import (
    MemoryUsage,
    NodeNetworkInfo,
    NodeResources,
    SystemPerformanceProfile,
)
from skulk.shared.types.topology import Cycle, RDMAConnection, SocketConnection
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import (
    CfgShardMetadata,
    PipelineShardMetadata,
    Sharding,
    ShardMetadata,
    TensorShardMetadata,
)
from skulk.utils.pydantic_ext import CamelCaseModel

# Memory-fit constants (overhead factor/floor, GPU working-set fraction, KV
# context budget) and the KV-cache estimator are imported above under the
# historical ``PLACEMENT_*`` / ``_estimate_kv_cache_bytes`` names. They live in
# skulk.shared.models.memory_estimate so this placement fit-check and the
# worker's local pre-spawn OOM guard share one source of truth.


@final
class CycleMemoryDiagnostics(CamelCaseModel):
    """Why candidate cycles were rejected by the memory filter.

    Carried alongside the surviving cycles so the caller can raise an error
    that names the actual cause instead of a generic "insufficient memory".
    """

    pending_info_node_ids: list[NodeId] = Field(default_factory=list)
    """Nodes that appeared in candidate cycles but have no memory info yet.

    This happens in the window between a node joining the topology (identity
    gossiped) and its first NodeGatheredInfo event arriving — placing during
    cluster startup lands here. It is a retry-shortly condition, not a
    memory shortfall.
    """

    rejection_reasons: list[str] = Field(default_factory=list)
    """One human-readable line per cycle rejected on real memory grounds."""


def _has_gpu_offload_backend(backends: frozenset[str]) -> bool:
    """Whether a node advertises a discrete-GPU-offload llama.cpp backend.

    True only for a non-CPU llama.cpp compute tag (``llama_cpp-vulkan/rocm/cuda``).
    A node that exposes a GPU but advertises only ``llama_cpp-cpu`` (the default
    when ``SKULK_LLAMA_CPP_BACKENDS`` is unset, or after a GPU wheel is replaced)
    runs GGUF on the CPU out of system RAM, so it must NOT be admitted against
    VRAM it will not use.
    """
    return any(
        tag.startswith("llama_cpp-") and not tag.endswith("-cpu") for tag in backends
    )


def usable_vram_by_node(
    node_system: Mapping[NodeId, SystemPerformanceProfile],
    node_resources: Mapping[NodeId, NodeResources] | None = None,
) -> dict[NodeId, Memory]:
    """Per-node usable discrete-GPU VRAM for placement, keyed by node.

    A node appears here only when it both (a) reports a discrete-VRAM accelerator
    (AMD/NVIDIA with a nonzero ``vram_total_bytes`` in ``node_system``) and
    (b) advertises a GPU-offload llama.cpp backend (``_has_gpu_offload_backend``),
    so its engine actually allocates weights + KV from that VRAM pool. That pool
    is reported separately from system RAM (e.g. a Strix Halo box's BIOS carves
    128 GB into ~64 GB system + 64 GB VRAM), so placement admits against VRAM, not
    ``0.75 * system_ram``. Apple unified-memory nodes report no discrete VRAM and
    are absent, keeping the system-RAM path. When ``node_resources`` is omitted
    the backend gate is skipped (test/simple use).

    The usable figure mirrors the RAM path: ``min(vram_total - vram_used,
    vram_total * GPU_VRAM_WORKING_SET_FRACTION)`` so it accounts for VRAM already
    in use by a resident instance and never assumes the whole card is free.
    """
    usable: dict[NodeId, Memory] = {}
    for node_id, profile in node_system.items():
        accelerator = profile.accelerator
        if accelerator is None or accelerator.vendor not in ("amd", "nvidia"):
            continue
        total = accelerator.vram_total_bytes
        if not total or total <= 0:
            continue
        if node_resources is not None:
            resources = node_resources.get(node_id)
            if resources is None or not _has_gpu_offload_backend(resources.backends):
                continue
        used = accelerator.vram_used_bytes or 0
        available = max(0, total - used)
        ceiling = int(total * GPU_VRAM_WORKING_SET_FRACTION)
        usable[node_id] = Memory.from_bytes(min(available, ceiling))
    return usable


def _per_node_required_memory(
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
    required_memory: Memory,
    sharding: Sharding,
    node_vram: Mapping[NodeId, Memory] | None = None,
) -> dict[NodeId, Memory]:
    """Estimate the weight bytes each node in the cycle must hold.

    Tensor parallelism splits the weights evenly across ranks, so every node
    carries ``required_memory / len(cycle)`` regardless of its capacity.
    Pipeline parallelism allocates layers proportionally to each node's
    usable memory (see ``allocate_layers_proportionally``), so a node's
    share scales with its fraction of the cycle's total usable memory.
    The continuous fraction is an estimate of the integer layer allocation:
    the two can differ by up to one layer per node (a few percent of the
    weights on realistic layer counts), which sits comfortably inside the
    ``PLACEMENT_MEMORY_OVERHEAD_FACTOR`` margin applied on top.

    Both the split here and the per-node admission below weigh by
    ``_node_usable_memory`` (capped at the Metal working-set ceiling), not raw
    ``ram_available``. Splitting by raw available while admitting against the
    cap over-weights a node whose free RAM exceeds its GPU ceiling (e.g. a
    16 GB node with 15 GB free but a ~12 GB cap), pushing its share past the
    cap and rejecting cycles that would fit if sized by usable memory — exactly
    the heterogeneous clusters this check is meant to support.
    """
    node_vram = node_vram or {}
    if sharding == Sharding.Tensor:
        even_share = required_memory / len(cycle.node_ids)
        return {node_id: even_share for node_id in cycle.node_ids}
    total_usable = sum(
        (
            _node_usable_memory(node_memory[node_id], node_vram.get(node_id))
            for node_id in cycle.node_ids
        ),
        start=Memory(),
    )
    if total_usable.in_bytes == 0:
        # Degenerate: no node reports any memory; assign everything everywhere
        # so the fit check below rejects the cycle with a concrete reason.
        return {node_id: required_memory for node_id in cycle.node_ids}
    return {
        node_id: required_memory
        * (
            _node_usable_memory(node_memory[node_id], node_vram.get(node_id))
            / total_usable
        )
        for node_id in cycle.node_ids
    }


def _node_usable_memory(
    memory: MemoryUsage, vram_usable: Memory | None = None
) -> Memory:
    """Memory a node can actually dedicate to a shard.

    On a discrete-VRAM GPU node (``vram_usable`` provided, see
    ``usable_vram_by_node``), this is the usable VRAM pool the GPU-offload engine
    allocates from, which is distinct from system RAM. Otherwise (Apple unified
    memory) it is capped at the Metal GPU working-set ceiling
    (``GPU_WORKING_SET_FRACTION`` of total RAM): gossiped ``ram_available`` can
    exceed what the GPU is allowed to wire, so a shard that fits "available" RAM
    can still abort on the GPU working-set wall.
    """
    if vram_usable is not None:
        return vram_usable
    ceiling = memory.ram_total * GPU_WORKING_SET_FRACTION
    return min(memory.ram_available, ceiling)


def filter_cycles_by_memory(
    cycles: list[Cycle],
    node_memory: Mapping[NodeId, MemoryUsage],
    model_card: ModelCard,
    sharding: Sharding = Sharding.Pipeline,
    context_budget: int = PLACEMENT_KV_CONTEXT_BUDGET_TOKENS,
    node_vram: Mapping[NodeId, Memory] | None = None,
) -> tuple[list[Cycle], CycleMemoryDiagnostics]:
    """Keep cycles whose every node can hold its shard with runtime headroom.

    The fit test is per node, not summed across the cycle: a 16 GB + 24 GB
    pair summing 21 GB of available memory cannot run a Tensor-sharded model
    whose even split overloads the 16 GB node, and a sum check would happily
    admit it. Each node must satisfy::

        weights_share * PLACEMENT_MEMORY_OVERHEAD_FACTOR
            + kv_share + PLACEMENT_MEMORY_OVERHEAD_FLOOR
            <= node usable memory

    where usable memory is the node's discrete VRAM pool when it has one
    (``node_vram``, see ``usable_vram_by_node``) or
    ``min(ram_available, GPU_WORKING_SET_FRACTION * ram_total)`` on Apple unified
    memory.

    ``kv_share`` reserves KV cache for ``context_budget`` tokens. It scales with
    a node's shard exactly as the weights do (per-layer for pipeline, per-rank
    for tensor), so it is derived from the node's weight share rather than
    recomputing the split.

    Cycles touching nodes with no memory info at all (cluster still starting
    up) are recorded in ``diagnostics.pending_info_node_ids`` instead of being
    silently dropped — the caller distinguishes "retry shortly" from "does
    not fit".

    Returns the surviving cycles plus diagnostics describing every rejection.
    """
    node_vram = node_vram or {}
    diagnostics = CycleMemoryDiagnostics()
    filtered_cycles: list[Cycle] = []
    required_memory = model_card.storage_size
    total_kv = _estimate_kv_cache_bytes(model_card, model_card.n_layers, context_budget)
    kv_ratio = (
        total_kv.in_bytes / required_memory.in_bytes
        if required_memory.in_bytes
        else 0.0
    )
    for cycle in cycles:
        missing = [node for node in cycle.node_ids if node not in node_memory]
        if missing:
            for node_id in missing:
                if node_id not in diagnostics.pending_info_node_ids:
                    diagnostics.pending_info_node_ids.append(node_id)
            continue

        node_shares = _per_node_required_memory(
            cycle, node_memory, required_memory, sharding, node_vram
        )
        overloaded: list[str] = []
        for node_id, share in node_shares.items():
            kv_share = share * kv_ratio
            required_with_overhead = (
                share * PLACEMENT_MEMORY_OVERHEAD_FACTOR
                + kv_share
                + PLACEMENT_MEMORY_OVERHEAD_FLOOR
            )
            node_vram_usable = node_vram.get(node_id)
            available = _node_usable_memory(node_memory[node_id], node_vram_usable)
            if required_with_overhead > available:
                pool = (
                    "usable GPU VRAM"
                    if node_vram_usable is not None
                    else f"min of available and {GPU_WORKING_SET_FRACTION:.0%} of RAM"
                )
                overloaded.append(
                    f"node {node_id} needs ~{required_with_overhead.in_gb:.1f}GB "
                    f"({share.in_gb:.1f}GB weights + {kv_share.in_gb:.1f}GB "
                    f"KV@{context_budget}tok + runtime headroom) but can use "
                    f"{available.in_gb:.1f}GB ({pool})"
                )
        if overloaded:
            diagnostics.rejection_reasons.append(
                f"cycle [{', '.join(str(n) for n in cycle.node_ids)}] "
                f"({sharding.value} sharding): " + "; ".join(overloaded)
            )
            continue
        filtered_cycles.append(cycle)
    return filtered_cycles, diagnostics


def get_smallest_cycles(
    cycles: list[Cycle],
) -> list[Cycle]:
    min_nodes = min(len(cycle) for cycle in cycles)
    return [cycle for cycle in cycles if len(cycle) == min_nodes]


def allocate_layers_proportionally(
    total_layers: int,
    memory_fractions: list[float],
) -> list[int]:
    n = len(memory_fractions)
    if n == 0:
        raise ValueError("Cannot allocate layers to an empty node list")
    if total_layers < n:
        raise ValueError(
            f"Cannot distribute {total_layers} layers across {n} nodes "
            "(need at least 1 layer per node)"
        )

    # Largest remainder: floor each, then distribute remainder by fractional part
    raw = [f * total_layers for f in memory_fractions]
    result = [int(r) for r in raw]
    by_remainder = sorted(range(n), key=lambda i: raw[i] - result[i], reverse=True)
    for i in range(total_layers - sum(result)):
        result[by_remainder[i]] += 1

    # Ensure minimum 1 per node by taking from the largest
    for i in range(n):
        if result[i] == 0:
            max_idx = max(range(n), key=lambda j: result[j])
            assert result[max_idx] > 1
            result[max_idx] -= 1
            result[i] = 1

    return result


def _validate_cycle(cycle: Cycle) -> None:
    if not cycle.node_ids:
        raise ValueError("Cannot create shard assignments for empty node cycle")


def _compute_total_memory(
    node_ids: list[NodeId],
    node_memory: Mapping[NodeId, MemoryUsage],
    node_vram: Mapping[NodeId, Memory] | None = None,
) -> Memory:
    # Weigh by usable (working-set-capped) memory, matching the admission check
    # in filter_cycles_by_memory: splitting by raw ram_available while admitting
    # against the cap over-weights nodes whose free RAM exceeds their GPU
    # ceiling and produces shards that don't fit.
    node_vram = node_vram or {}
    total_memory = sum(
        (
            _node_usable_memory(node_memory[node_id], node_vram.get(node_id))
            for node_id in node_ids
        ),
        start=Memory(),
    )
    if total_memory.in_bytes == 0:
        raise ValueError("Cannot create shard assignments: total usable memory is 0")
    return total_memory


def _allocate_and_validate_layers(
    node_ids: list[NodeId],
    node_memory: Mapping[NodeId, MemoryUsage],
    total_memory: Memory,
    model_card: ModelCard,
    node_vram: Mapping[NodeId, Memory] | None = None,
) -> list[int]:
    node_vram = node_vram or {}
    layer_allocations = allocate_layers_proportionally(
        total_layers=model_card.n_layers,
        memory_fractions=[
            _node_usable_memory(node_memory[node_id], node_vram.get(node_id))
            / total_memory
            for node_id in node_ids
        ],
    )

    total_storage = model_card.storage_size
    total_layers = model_card.n_layers
    for i, node_id in enumerate(node_ids):
        node_layers = layer_allocations[i]
        required_memory = (total_storage * node_layers) // total_layers
        usable_memory = _node_usable_memory(node_memory[node_id], node_vram.get(node_id))
        if required_memory > usable_memory:
            pool = (
                "GPU VRAM"
                if node_vram.get(node_id) is not None
                else f"{GPU_WORKING_SET_FRACTION:.0%} of RAM cap"
            )
            raise ValueError(
                f"Node {i} ({node_id}) has insufficient memory: "
                f"requires {required_memory.in_gb:.2f} GB for {node_layers} layers, "
                f"but can only use {usable_memory.in_gb:.2f} GB ({pool})"
            )

    return layer_allocations


def get_shard_assignments_for_pipeline_parallel(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
    node_vram: Mapping[NodeId, Memory] | None = None,
) -> ShardAssignments:
    """Create shard assignments for pipeline parallel execution."""
    world_size = len(cycle)
    use_cfg_parallel = model_card.uses_cfg and world_size >= 2 and world_size % 2 == 0

    if use_cfg_parallel:
        return _get_shard_assignments_for_cfg_parallel(
            model_card, cycle, node_memory, node_vram
        )
    else:
        return _get_shard_assignments_for_pure_pipeline(
            model_card, cycle, node_memory, node_vram
        )


def _get_shard_assignments_for_cfg_parallel(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
    node_vram: Mapping[NodeId, Memory] | None = None,
) -> ShardAssignments:
    """Create shard assignments for CFG parallel execution.

    CFG parallel runs two independent pipelines. Group 0 processes the positive
    prompt, group 1 processes the negative prompt. The ring topology places
    group 1's ranks in reverse order so both "last stages" are neighbors for
    efficient CFG exchange.
    """
    _validate_cycle(cycle)

    world_size = len(cycle)
    cfg_world_size = 2
    pipeline_world_size = world_size // cfg_world_size

    # Allocate layers for one pipeline group (both groups run the same layers)
    pipeline_node_ids = cycle.node_ids[:pipeline_world_size]
    pipeline_memory = _compute_total_memory(pipeline_node_ids, node_memory, node_vram)
    layer_allocations = _allocate_and_validate_layers(
        pipeline_node_ids, node_memory, pipeline_memory, model_card, node_vram
    )

    # Ring topology: group 0 ascending [0,1,2,...], group 1 descending [...,2,1,0]
    # This places both last stages as neighbors for CFG exchange.
    position_to_cfg_pipeline = [(0, r) for r in range(pipeline_world_size)] + [
        (1, r) for r in reversed(range(pipeline_world_size))
    ]

    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for device_rank, node_id in enumerate(cycle.node_ids):
        cfg_rank, pipeline_rank = position_to_cfg_pipeline[device_rank]
        layers_before = sum(layer_allocations[:pipeline_rank])
        node_layers = layer_allocations[pipeline_rank]

        shard = CfgShardMetadata(
            model_card=model_card,
            device_rank=device_rank,
            world_size=world_size,
            start_layer=layers_before,
            end_layer=layers_before + node_layers,
            n_layers=model_card.n_layers,
            cfg_rank=cfg_rank,
            cfg_world_size=cfg_world_size,
            pipeline_rank=pipeline_rank,
            pipeline_world_size=pipeline_world_size,
        )

        runner_id = RunnerId()
        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    return ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )


def _get_shard_assignments_for_pure_pipeline(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
    node_vram: Mapping[NodeId, Memory] | None = None,
) -> ShardAssignments:
    """Create shard assignments for pure pipeline execution."""
    _validate_cycle(cycle)
    total_memory = _compute_total_memory(cycle.node_ids, node_memory, node_vram)

    layer_allocations = _allocate_and_validate_layers(
        cycle.node_ids, node_memory, total_memory, model_card, node_vram
    )

    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for pipeline_rank, node_id in enumerate(cycle.node_ids):
        layers_before = sum(layer_allocations[:pipeline_rank])
        node_layers = layer_allocations[pipeline_rank]

        shard = PipelineShardMetadata(
            model_card=model_card,
            device_rank=pipeline_rank,
            world_size=len(cycle),
            start_layer=layers_before,
            end_layer=layers_before + node_layers,
            n_layers=model_card.n_layers,
        )

        runner_id = RunnerId()
        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    return ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )


def get_shard_assignments_for_tensor_parallel(
    model_card: ModelCard,
    cycle: Cycle,
):
    total_layers = model_card.n_layers
    world_size = len(cycle)
    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for i, node_id in enumerate(cycle):
        shard = TensorShardMetadata(
            model_card=model_card,
            device_rank=i,
            world_size=world_size,
            start_layer=0,
            end_layer=total_layers,
            n_layers=total_layers,
        )

        runner_id = RunnerId()

        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    shard_assignments = ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )

    return shard_assignments


def get_shard_assignments(
    model_card: ModelCard,
    cycle: Cycle,
    sharding: Sharding,
    node_memory: Mapping[NodeId, MemoryUsage],
    node_vram: Mapping[NodeId, Memory] | None = None,
) -> ShardAssignments:
    match sharding:
        case Sharding.Pipeline:
            return get_shard_assignments_for_pipeline_parallel(
                model_card=model_card,
                cycle=cycle,
                node_memory=node_memory,
                node_vram=node_vram,
            )
        case Sharding.Tensor:
            return get_shard_assignments_for_tensor_parallel(
                model_card=model_card,
                cycle=cycle,
            )


def get_mlx_jaccl_devices_matrix(
    selected_cycle: list[NodeId],
    cycle_digraph: Topology,
) -> list[list[str | None]]:
    """Build connectivity matrix mapping device i to device j via RDMA interface names.

    The matrix element [i][j] contains the interface name on device i that connects
    to device j, or None if no connection exists or no interface name is found.
    Diagonal elements are always None.
    """
    num_nodes = len(selected_cycle)
    matrix: list[list[str | None]] = [
        [None for _ in range(num_nodes)] for _ in range(num_nodes)
    ]

    for i, node_i in enumerate(selected_cycle):
        for j, node_j in enumerate(selected_cycle):
            if i == j:
                continue

            for conn in cycle_digraph.get_all_connections_between(node_i, node_j):
                if isinstance(conn, RDMAConnection):
                    matrix[i][j] = conn.source_rdma_iface
                    break
            else:
                raise ValueError(
                    "Current jaccl backend requires all-to-all RDMA connections"
                )

    return matrix


def _find_connection_ip(
    node_i: NodeId,
    node_j: NodeId,
    cycle_digraph: Topology,
) -> Generator[str, None, None]:
    """Find all IP addresses that connect node i to node j."""
    for connection in cycle_digraph.get_all_connections_between(node_i, node_j):
        if isinstance(connection, SocketConnection):
            yield connection.sink_multiaddr.ip_address


def _is_vpn_address(ip: str) -> bool:
    """Whether an address belongs to a VPN/overlay network (Tailscale).

    Detected by ADDRESS, not by the gossiped interface label: utun
    interfaces never appear in ``networksetup`` output so their gossiped
    type is "unknown", and adding a new ``InterfaceType`` literal would
    break old nodes decoding gossiped ``NetworkInterfaceInfo`` in a
    mixed-version rollout. Tailscale assigns from CGNAT ``100.64.0.0/10``
    (IPv4) and ``fd7a:115c:a1e0::/48`` (IPv6).
    """
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv4Address):
        return address in ipaddress.IPv4Network("100.64.0.0/10")
    return address in ipaddress.IPv6Network("fd7a:115c:a1e0::/48")


# Transport ranking for ring placements: VPN/overlay addresses are the LAST
# resort, below even "unknown" — Tailscale exists for reaching a node from
# OUTSIDE its local network and may be DERP-relayed (observed live: a
# kite-pair ring link riding the Dallas relay between two machines on the
# same switch). A pair whose only observed path is the overlay (genuinely
# cross-network placement) still works; a pair with any TB/LAN candidate
# never selects it (#265).
_RING_TRANSPORT_PRIORITY: dict[str, int] = {
    "thunderbolt": 0,
    "maybe_ethernet": 1,
    "ethernet": 2,
    "wifi": 3,
    "unknown": 4,
    "vpn": 5,
}

# RDMA/jaccl coordinator preference (control plane, not the data path).
_JACCL_TRANSPORT_PRIORITY: dict[str, int] = {
    "ethernet": 0,
    "wifi": 1,
    "unknown": 2,
    "maybe_ethernet": 3,
    "thunderbolt": 4,
    "vpn": 5,
}


def _find_ip_prioritised(
    node_id: NodeId,
    other_node_id: NodeId,
    cycle_digraph: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
    ring: bool,
) -> str | None:
    """Find an IP address between nodes with transport prioritization.

    Candidates are the OBSERVED libp2p connections between the pair, ranked
    by the sink node's gossiped interface type — ring placements prefer the
    fastest local interconnect (TB first), and VPN/overlay addresses rank
    strictly last regardless of gossiped label (see ``_is_vpn_address``).

    TODO: Profile and get actual connection speeds.
    """
    ips = list(_find_connection_ip(node_id, other_node_id, cycle_digraph))
    if not ips:
        return None
    other_network = node_network.get(other_node_id, NodeNetworkInfo())
    ip_to_type = {
        iface.ip_address: iface.interface_type for iface in other_network.interfaces
    }
    priority = _RING_TRANSPORT_PRIORITY if ring else _JACCL_TRANSPORT_PRIORITY

    def transport_rank(ip: str) -> int:
        if _is_vpn_address(ip):
            return priority["vpn"]
        return priority.get(ip_to_type.get(ip, "unknown"), priority["unknown"])

    return min(ips, key=transport_rank)


def get_mlx_ring_hosts_by_node(
    selected_cycle: Cycle,
    cycle_digraph: Topology,
    ephemeral_port: int,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> dict[NodeId, list[Host]]:
    """Generate per-node host lists for MLX ring backend.

    Each node gets a list where:
    - Self position: Host(ip="0.0.0.0", port=ephemeral_port)
    - Left/right neighbors: actual connection IPs
    - Non-neighbors: Host(ip="198.51.100.1", port=0) placeholder (RFC 5737 TEST-NET-2)
    """
    world_size = len(selected_cycle)
    if world_size == 0:
        return {}

    hosts_by_node: dict[NodeId, list[Host]] = {}

    for rank, node_id in enumerate(selected_cycle):
        left_rank = (rank - 1) % world_size
        right_rank = (rank + 1) % world_size

        hosts_for_node: list[Host] = []

        for idx, other_node_id in enumerate(selected_cycle):
            if idx == rank:
                hosts_for_node.append(Host(ip="0.0.0.0", port=ephemeral_port))
                continue

            if idx not in {left_rank, right_rank}:
                # Placeholder IP from RFC 5737 TEST-NET-2
                hosts_for_node.append(Host(ip="198.51.100.1", port=0))
                continue

            connection_ip = _find_ip_prioritised(
                node_id, other_node_id, cycle_digraph, node_network, ring=True
            )
            if connection_ip is None:
                raise ValueError(
                    "MLX ring backend requires connectivity between neighbouring nodes"
                )

            hosts_for_node.append(Host(ip=connection_ip, port=ephemeral_port))

        hosts_by_node[node_id] = hosts_for_node

    return hosts_by_node


def get_mlx_jaccl_coordinators(
    coordinator: NodeId,
    coordinator_port: int,
    cycle_digraph: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> dict[NodeId, str]:
    """Get the coordinator addresses for MLX JACCL (rank 0 device).

    Select an IP address that each node can reach for the rank 0 node. Returns
    address in format "X.X.X.X:PORT" per node.
    """
    logger.debug(f"Selecting coordinator: {coordinator}")

    def get_ip_for_node(n: NodeId) -> str:
        if n == coordinator:
            return "0.0.0.0"

        ip = _find_ip_prioritised(
            n, coordinator, cycle_digraph, node_network, ring=False
        )
        if ip is not None:
            return ip

        raise ValueError(
            "Current jaccl backend requires all participating devices to be able to communicate"
        )

    return {
        n: f"{get_ip_for_node(n)}:{coordinator_port}"
        for n in cycle_digraph.list_nodes()
    }
