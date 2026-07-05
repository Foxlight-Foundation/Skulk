import random
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from copy import deepcopy
from typing import Sequence

from skulk.master.placement_utils import (
    Cycle,
    filter_cycles_by_memory,
    get_llama_rpc_donor_endpoints,
    get_mlx_jaccl_coordinators,
    get_mlx_jaccl_devices_matrix,
    get_mlx_ring_hosts_by_node,
    get_shard_assignments,
    get_shard_assignments_for_llama_rpc,
    get_smallest_cycles,
)
from skulk.shared.backends import (
    EngineType,
    engine_of,
    engine_supports_multi_node,
    platform_compatible_backends,
    resolve_node_backend,
)
from skulk.shared.models.memory_estimate import instance_context_token_limit
from skulk.shared.models.model_cards import ModelCard, ModelId
from skulk.shared.topology import Topology
from skulk.shared.types.commands import (
    CancelDownload,
    CreateInstance,
    DeleteInstance,
    DownloadCommand,
    PlaceInstance,
)
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import (
    Event,
    InstanceCreated,
    InstanceDeleted,
    TaskStatusUpdated,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import (
    MemoryUsage,
    NodeNetworkInfo,
    NodeResources,
)
from skulk.shared.types.tasks import Task, TaskId, TaskStatus
from skulk.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
)
from skulk.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceMeta,
    LlamaRpcInstance,
    MlxJacclInstance,
    MlxRingInstance,
)
from skulk.shared.types.worker.runners import ShardAssignments
from skulk.shared.types.worker.shards import Sharding, TensorShardMetadata


def random_ephemeral_port() -> int:
    port = random.randint(49153, 65535)
    return port - 1 if port <= 52415 else port


def add_instance_to_placements(
    command: CreateInstance,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
    node_memory: Mapping[NodeId, MemoryUsage],
    node_vram: Mapping[NodeId, Memory] | None = None,
) -> Mapping[InstanceId, Instance]:
    # TODO: validate against topology

    # Stamp the memory-derived context-admission ceiling, same as the
    # place_instance path (#279 slice 2). A client-supplied exact placement
    # usually omits contextTokenLimit; without this it would get only the
    # card-limit backfill (BaseInstance validator), and since runners now trust
    # the stamped value instead of recomputing from per-node RAM, a prompt that
    # fits the card but not the hosting node could reach MLX instead of a clean
    # context_length_exceeded. instance_context_token_limit falls back to the
    # card's advertised context_length whenever ANY hosting node's ram_total is
    # still unknown, so a transient missing reading yields the card guard rather
    # than None (review catch on #292).
    assignments = command.instance.shard_assignments
    ceiling = instance_context_token_limit(
        assignments,
        {
            node_id: node_memory[node_id].ram_total
            for node_id in assignments.node_to_runner
            if node_id in node_memory
        },
        node_vram=node_vram,
    )
    instance = command.instance.model_copy(update={"context_token_limit": ceiling})
    return {**current_instances, instance.instance_id: instance}


def _get_node_download_fraction(
    node_id: NodeId,
    model_id: ModelId,
    download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> float:
    """Return the download fraction (0.0–1.0) for a model on a given node."""
    for progress in download_status.get(node_id, []):
        if progress.shard_metadata.model_card.model_id != model_id:
            continue
        match progress:
            case DownloadCompleted():
                return 1.0
            case DownloadOngoing():
                total = progress.download_progress.total.in_bytes
                return (
                    progress.download_progress.downloaded.in_bytes / total
                    if total > 0
                    else 0.0
                )
            case DownloadPending():
                total = progress.total.in_bytes
                return progress.downloaded.in_bytes / total if total > 0 else 0.0
            case DownloadFailed():
                return 0.0
    return 0.0


def _cycle_download_score(
    cycle: Cycle,
    model_id: ModelId,
    download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> float:
    """Sum of download fractions across all nodes in a cycle."""
    return sum(
        _get_node_download_fraction(node_id, model_id, download_status)
        for node_id in cycle
    )


def _cycle_backend_preference_score(
    cycle: Cycle,
    node_resources: Mapping[NodeId, NodeResources],
    backend_preference: Sequence[str],
) -> int:
    """Rank a cycle by how well its nodes satisfy the card's backend preference.

    ``backend_preference`` is an ordered list of backend tags (most preferred
    first). The score is ``len(backend_preference) - i`` for the earliest index ``i``
    whose tag is advertised by every node in the cycle that has a resources
    entry, or ``0`` when no preferred tag is satisfiable (and when there is no
    preference at all). Higher is better, so a cycle that can serve the top
    preference outranks one that can only serve a fallback, which in turn
    outranks one that serves none (graceful degradation by construction).

    This is a SOFT signal: ``compatible_backends`` has already hard-filtered the
    candidates, so every cycle here is runnable; this only orders them. Nodes
    without a resources entry yet (gossip warming up) are not counted against a
    preference, matching the optimistic eligibility default elsewhere.
    """
    for index, tag in enumerate(backend_preference):
        if all(
            tag in node_resources[node_id].backends
            for node_id in cycle
            if node_id in node_resources
        ):
            return len(backend_preference) - index
    return 0


def _card_platform_backends(card: ModelCard) -> frozenset[str]:
    """The card's compatible backends minus current platform limitations.

    Cards declare MODEL truth (what the model and its artifacts can do); which
    of those capabilities our runner implementations can currently exploit is
    PLATFORM truth and lives in code (``platform_compatible_backends``), so a
    platform gap is never encoded on a card. Every placement-side read of
    ``compatible_backends`` goes through this helper so eligibility, the
    common-engine cycle rule, and backend stamping all agree.
    """
    return platform_compatible_backends(
        card.placement.compatible_backends,
        card_serves_vision=card.vision is not None,
    )


def _cycle_common_multi_node_engines(
    cycle: Cycle,
    card_backends: AbstractSet[str],
    node_resources: Mapping[NodeId, NodeResources],
) -> set[EngineType]:
    """Multi-node-capable engines advertised by EVERY node in the cycle.

    A multi-node placement runs ONE engine across the whole cycle, so a cycle
    is only viable for a card when some single engine is both (a) present in
    the intersection of the card's ``compatible_backends`` with every node's
    advertised backends and (b) multi-node capable. Checking per node
    independently instead (the pre-#414 behavior) admits a mixed cycle for a
    hybrid card — e.g. an MLX-only Mac plus a llama.cpp-only AMD node, two
    engines that cannot form one ring.

    A node with no resources entry yet (gossip warming up) is treated as
    advertising the ``NodeResources`` default (``{"mlx"}``), matching the
    optimistic eligibility default elsewhere: MLX cards keep placing during
    warmup, while engine-specific multi-node shapes (llama_server RPC) wait
    until the node's real advertisement arrives.
    """
    common: set[EngineType] | None = None
    for node_id in cycle.node_ids:
        resources = node_resources.get(node_id)
        backends = (
            resources.backends if resources is not None else frozenset({"mlx"})
        )
        node_engines: set[EngineType] = {
            engine
            for tag in backends & card_backends
            if (engine := engine_of(tag)) is not None
        }
        common = node_engines if common is None else (common & node_engines)
        if not common:
            return set()
    return {
        engine for engine in (common or set()) if engine_supports_multi_node(engine)
    }


def _is_llama_rpc_cycle(
    cycle: Cycle,
    card_backends: AbstractSet[str],
    node_resources: Mapping[NodeId, NodeResources],
) -> bool:
    """Whether a multi-node cycle would serve this card via the RPC shape (#328).

    True when ``llama_server`` is a common multi-node engine of the cycle and
    ``mlx`` is not: the proven MLX shapes keep priority whenever they are an
    option (in practice the sets never overlap, since no node advertises both).
    Single-node cycles are never RPC (the served engine runs standalone there).
    """
    if len(cycle) <= 1:
        return False
    engines = _cycle_common_multi_node_engines(cycle, card_backends, node_resources)
    return "llama_server" in engines and "mlx" not in engines


class PlacementError(ValueError):
    """Placement is impossible for the requested command and current state.

    Subclasses ``ValueError`` so existing callers that catch ``ValueError``
    (the API preview endpoint, the master command processor) keep working.
    """


class PlacementInfoPendingError(PlacementError):
    """Placement cannot be judged yet because cluster info is still arriving.

    Covers both phases of the cluster-startup window: connection edges lag
    node identities by a few gossip rounds (enough nodes exist but no
    connected cycle is known yet), and per-node memory info lags the edges
    (a cycle exists but a node's first NodeGatheredInfo event has not
    arrived). Both are retry-shortly conditions — distinct from a real
    topology gap or memory shortfall so callers can wait instead of
    reporting a false error.
    """


def place_instance(
    command: PlaceInstance,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
    node_memory: Mapping[NodeId, MemoryUsage],
    node_network: Mapping[NodeId, NodeNetworkInfo],
    required_nodes: set[NodeId] | None = None,
    download_status: Mapping[NodeId, Sequence[DownloadProgress]] | None = None,
    excluded_nodes: set[NodeId] | None = None,
    node_resources: Mapping[NodeId, NodeResources] | None = None,
    node_vram: Mapping[NodeId, Memory] | None = None,
    node_vram_strict: Mapping[NodeId, Memory] | None = None,
) -> dict[InstanceId, Instance]:
    cycles = topology.get_cycles()
    candidate_cycles = list(filter(lambda it: len(it) >= command.min_nodes, cycles))
    if not candidate_cycles:
        known_nodes = sum(1 for _ in topology.list_nodes())
        if known_nodes >= command.min_nodes:
            # Enough nodes exist for this placement — they just aren't
            # bidirectionally connected (yet). Right after cluster formation
            # the connection edges lag the node identities by a few gossip
            # rounds, so this is usually a retry-shortly condition rather
            # than a network problem.
            raise PlacementInfoPendingError(
                f"The topology knows {known_nodes} node(s) but none form a "
                f"connected cycle of at least {command.min_nodes} node(s) yet. "
                "Connection info may still be gossiping (typical right after "
                "cluster formation) — retry shortly. If this persists, check "
                "network connectivity between the nodes."
            )
        raise PlacementError(
            f"The topology has only {known_nodes} node(s); a placement with "
            f"min_nodes={command.min_nodes} is impossible. Multi-node "
            "placement requires bidirectional connectivity between the nodes."
        )

    # Drop any cycle that touches an operator-excluded node. Exclusion is the
    # operator's "don't pick this node for new placements" signal — already
    # placed instances on the excluded node remain unaffected (they live in
    # current_instances, which this function never mutates), but the planner
    # treats the excluded node as if it were absent from the topology when
    # scoring fresh placements.
    if excluded_nodes:
        candidate_cycles = [
            cycle
            for cycle in candidate_cycles
            if not (set(cycle.node_ids) & excluded_nodes)
        ]
        if not candidate_cycles:
            raise PlacementError(
                f"All cycles of at least {command.min_nodes} node(s) touch an "
                f"excluded node ({len(excluded_nodes)} node(s) excluded). "
                "Check the excluded_nodes list — node IDs change when a "
                "cluster session restarts."
            )

    # Hard-filter on node participation and backend compatibility (#149).
    # A node is ineligible for an inference shard of THIS model when it
    # declares a non-``full`` participation role (e.g. a ``management`` /
    # edge node that serves the API but never joins a ring), or when its
    # advertised backends do not intersect the card's compatible_backends.
    # Nodes with no resources entry yet (gossip still warming up) are treated
    # as eligible so behavior matches the pre-#149 default of full/mlx.
    if node_resources:
        compatible_backends = _card_platform_backends(command.model_card)
        ineligible_nodes = {
            node_id
            for node_id, resources in node_resources.items()
            if resources.participation != "full"
            or not (resources.backends & compatible_backends)
        }
        if ineligible_nodes:
            candidate_cycles = [
                cycle
                for cycle in candidate_cycles
                if not (set(cycle.node_ids) & ineligible_nodes)
            ]
            if not candidate_cycles:
                raise PlacementError(
                    f"All cycles of at least {command.min_nodes} node(s) touch a "
                    f"node ineligible for this model: either a non-participating "
                    f"(management/edge) node or one whose backends do not match "
                    f"the model's compatible_backends "
                    f"({sorted(compatible_backends)})."
                )

    # Common-engine cycle rule. A multi-node placement runs ONE engine across
    # the whole cycle, so a multi-node cycle is admissible only when some
    # single engine is advertised by every node (within the card's
    # compatible_backends) AND that engine can shard across nodes
    # (`engine_supports_multi_node`: mlx via ring/jaccl, llama_server via the
    # RPC driver+donors shape, #328). This subsumes two prior checks in one
    # place: the single-node engine guard (a card whose only engine is the
    # in-process llama.cpp gets no multi-node cycles, since llama_cpp is never
    # multi-node capable) and the #414 mixed-engine hole (a hybrid card no
    # longer admits a cycle mixing an MLX-only node with a llama.cpp-only
    # node, which the per-node any() check used to allow). Single-node cycles
    # are untouched -- the per-node participation/backend filter above already
    # covers them. Cards with no declared backends (legacy) skip the rule.
    resolved_node_resources = node_resources or {}
    card_backends = _card_platform_backends(command.model_card)
    if card_backends:
        multi_node_candidate_cycles = [
            cycle for cycle in candidate_cycles if len(cycle) > 1
        ]
        candidate_cycles = [
            cycle
            for cycle in candidate_cycles
            if len(cycle) == 1
            or _cycle_common_multi_node_engines(
                cycle, card_backends, resolved_node_resources
            )
        ]
        if not candidate_cycles:
            # A node with no NodeResources entry yet defaults to mlx in the
            # common-engine rule, so a served-only cycle evaluated during the
            # telemetry warm-up window is dropped for the WRONG reason. When
            # backends telemetry was provided but is missing for multi-node
            # candidates, surface the retry-shortly signal instead of a hard
            # error the caller would treat as terminal (observed live: a
            # pooled placement right after fleet bring-up 400s once, then
            # succeeds on retry).
            if node_resources is not None:
                resources_pending_nodes = {
                    node_id
                    for cycle in multi_node_candidate_cycles
                    for node_id in cycle.node_ids
                    if node_id not in resolved_node_resources
                }
                if resources_pending_nodes:
                    raise PlacementInfoPendingError(
                        "Backend info has not been gossiped yet for node(s) "
                        f"[{', '.join(str(n) for n in sorted(resources_pending_nodes))}] "
                        "— the cluster may still be starting up. Retry shortly."
                    )
            raise PlacementError(
                "No candidate cycle can serve this model: multi-node placement "
                "requires one engine common to every node in the cycle that "
                "supports sharding (MLX, or llama_server via RPC), and no "
                "single-node cycle is available/eligible. Place it on one "
                "node, or free a node that advertises a compatible backend."
            )

    # The RPC shape is Pipeline-shaped by construction (llama.cpp layer-splits
    # across the pooled devices); a Tensor request cannot be honored on an RPC
    # cycle, so those cycles drop out of a Tensor placement's candidates.
    if command.sharding == Sharding.Tensor:
        candidate_cycles = [
            cycle
            for cycle in candidate_cycles
            if not _is_llama_rpc_cycle(
                cycle, card_backends, resolved_node_resources
            )
        ]
        if not candidate_cycles:
            raise PlacementError(
                "Requested Tensor sharding, but every candidate cycle would "
                "serve this model via multi-node llama.cpp (RPC), which is "
                "pipeline-only."
            )

    # Filter to cycles containing all required nodes (subset matching)
    if required_nodes:
        candidate_cycles = [
            cycle
            for cycle in candidate_cycles
            if required_nodes.issubset(cycle.node_ids)
        ]
        if not candidate_cycles:
            raise PlacementError(
                "No candidate cycle contains all required nodes "
                f"[{', '.join(str(n) for n in required_nodes)}]."
            )
    # Memory admission. RPC cycles (multi-node llama.cpp) admit against the
    # strict per-node VRAM carve when the caller supplies it: measured on the
    # Strix pair, llama.cpp's RPC split allocates from the VRAM carve only
    # (never the UMA/GTT spill), so the UMA-inflated figure would over-admit a
    # pooled model that llama-server then fails to load. Everything else keeps
    # the standard figure (single-node UMA spill stays proven and admitted).
    standard_candidates = [
        cycle
        for cycle in candidate_cycles
        if not _is_llama_rpc_cycle(cycle, card_backends, resolved_node_resources)
    ]
    rpc_candidates = [
        cycle
        for cycle in candidate_cycles
        if _is_llama_rpc_cycle(cycle, card_backends, resolved_node_resources)
    ]
    cycles_with_sufficient_memory, memory_diagnostics = filter_cycles_by_memory(
        standard_candidates,
        node_memory,
        command.model_card,
        command.sharding,
        node_vram=node_vram,
    )
    if rpc_candidates:
        rpc_vram_map: Mapping[NodeId, Memory] = (
            node_vram_strict if node_vram_strict is not None else node_vram
        ) or {}
        # RPC admission is VRAM-carve-only by contract (llama.cpp's RPC split
        # never touches the UMA/GTT spill). A cycle node missing from the
        # strict map (telemetry warm-up: NodeResources arrived, node_system's
        # accelerator reading not yet) must surface as info-pending, NOT fall
        # back to system RAM inside the memory filter — that would over-admit
        # a pooled model that llama-server then fails to load, and the RPC
        # path deliberately bypasses the worker's local fit guard.
        vram_pending_nodes = {
            node_id
            for cycle in rpc_candidates
            for node_id in cycle.node_ids
            if node_id not in rpc_vram_map
        }
        vram_known_rpc_candidates = [
            cycle
            for cycle in rpc_candidates
            if all(node_id in rpc_vram_map for node_id in cycle.node_ids)
        ]
        rpc_fit, rpc_diagnostics = filter_cycles_by_memory(
            vram_known_rpc_candidates,
            node_memory,
            command.model_card,
            # llama.cpp splits the pooled model proportional to device memory,
            # which is exactly the Pipeline-proportional fit estimate.
            Sharding.Pipeline,
            node_vram=rpc_vram_map,
        )
        cycles_with_sufficient_memory = cycles_with_sufficient_memory + rpc_fit
        memory_diagnostics.pending_info_node_ids.extend(
            node_id
            for node_id in (*rpc_diagnostics.pending_info_node_ids, *sorted(vram_pending_nodes))
            if node_id not in memory_diagnostics.pending_info_node_ids
        )
        memory_diagnostics.rejection_reasons.extend(
            rpc_diagnostics.rejection_reasons
        )
    if len(cycles_with_sufficient_memory) == 0:
        if memory_diagnostics.pending_info_node_ids:
            raise PlacementInfoPendingError(
                "Memory info has not been gathered yet for node(s) "
                f"[{', '.join(str(n) for n in memory_diagnostics.pending_info_node_ids)}] "
                "— the cluster may still be starting up. Retry shortly."
            )
        detail = (
            "; ".join(memory_diagnostics.rejection_reasons)
            if memory_diagnostics.rejection_reasons
            else "no candidate cycles to evaluate"
        )
        raise PlacementError(
            f"No candidate cycle fits {command.model_card.model_id} "
            f"({command.model_card.storage_size.in_gb:.1f}GB of weights): {detail}"
        )

    if command.sharding == Sharding.Tensor:
        if not command.model_card.supports_tensor:
            raise PlacementError(
                f"Requested Tensor sharding but this model does not support tensor parallelism: {command.model_card.model_id}"
            )
        # TODO: the condition here for tensor parallel is not correct, but it works good enough for now.
        kv_heads = command.model_card.num_key_value_heads
        cycles_with_sufficient_memory = [
            cycle
            for cycle in cycles_with_sufficient_memory
            if command.model_card.hidden_size % len(cycle) == 0
            and (kv_heads is None or kv_heads % len(cycle) == 0)
        ]
        if not cycles_with_sufficient_memory:
            raise PlacementError(
                f"No tensor sharding found for model with "
                f"hidden_size={command.model_card.hidden_size}"
                f"{f', num_key_value_heads={kv_heads}' if kv_heads is not None else ''}"
                f" across candidate cycles"
            )
    if command.sharding == Sharding.Pipeline and command.model_card.model_id == ModelId(
        "mlx-community/DeepSeek-V3.1-8bit"
    ):
        raise PlacementError(
            "Pipeline parallelism is not supported for DeepSeek V3.1 (8-bit)"
        )

    smallest_cycles = get_smallest_cycles(cycles_with_sufficient_memory)

    smallest_rdma_cycles = [
        cycle for cycle in smallest_cycles if topology.is_rdma_cycle(cycle)
    ]

    if command.instance_meta == InstanceMeta.MlxJaccl:
        if not smallest_rdma_cycles:
            raise PlacementError(
                "Requested RDMA (MlxJaccl) but no RDMA-connected cycles available"
            )
        smallest_cycles = smallest_rdma_cycles

    cycles_with_leaf_nodes: list[Cycle] = [
        cycle
        for cycle in smallest_cycles
        if any(topology.node_is_leaf(node_id) for node_id in cycle)
    ]

    resolved_download_status = download_status or {}
    candidate_cycles = (
        cycles_with_leaf_nodes if cycles_with_leaf_nodes != [] else smallest_cycles
    )

    # Prefer a cycle that can serve the model's preferred backend (soft, ordered;
    # compatible_backends has already hard-filtered, so this only ranks). This
    # dominates the download/memory tie-breakers because serving a model on its
    # faster backend is the whole point of the preference; among cycles with the
    # same preference rank, download locality then free memory still decide.
    backend_preference = command.model_card.placement.backend_preference
    selected_cycle = max(
        candidate_cycles,
        key=lambda cycle: (
            _cycle_backend_preference_score(
                cycle, resolved_node_resources, backend_preference
            ),
            _cycle_download_score(
                cycle, command.model_card.model_id, resolved_download_status
            ),
            sum(
                (node_memory[node_id].ram_available for node_id in cycle),
                start=Memory(),
            ),
        ),
    )

    # Single-node: force Pipeline/Ring (Tensor and Jaccl require multi-node)
    if len(selected_cycle) == 1:
        command.instance_meta = InstanceMeta.MlxRing
        command.sharding = Sharding.Pipeline

    # The selected cycle's engine decides the instance shape. A multi-node
    # GGUF cycle (llama_server common, mlx not) resolves to the RPC
    # driver-plus-donors shape regardless of the requested instance_meta,
    # mirroring the single-node coercion above: instance_meta is an MLX-shape
    # vocabulary and the served engine has exactly one multi-node form.
    selected_is_rpc = _is_llama_rpc_cycle(
        selected_cycle, card_backends, resolved_node_resources
    )
    if command.instance_meta == InstanceMeta.LlamaRpc and not selected_is_rpc:
        raise PlacementError(
            "Requested a multi-node llama.cpp (RPC) placement, but the "
            "selected cycle does not serve this model via llama_server on "
            "every node."
        )

    driver_node: NodeId | None = None
    if selected_is_rpc:
        # Driver = the node with the largest usable VRAM (most layers land
        # locally, minimizing cross-network boundaries), tie-broken by which
        # node already has the model on disk (only the driver reads the GGUF).
        rpc_vram = node_vram_strict if node_vram_strict is not None else (
            node_vram or {}
        )
        driver_node = max(
            selected_cycle.node_ids,
            key=lambda node_id: (
                rpc_vram.get(node_id, Memory()).in_bytes,
                _get_node_download_fraction(
                    node_id, command.model_card.model_id, resolved_download_status
                ),
            ),
        )
        shard_assignments = get_shard_assignments_for_llama_rpc(
            command.model_card, selected_cycle, driver_node
        )
    else:
        shard_assignments = get_shard_assignments(
            command.model_card, selected_cycle, command.sharding, node_memory, node_vram
        )

    # Persist the per-node resolved backend tag on each shard (#330). The worker
    # then dispatches its runner from this replicated value instead of
    # re-probing its own backends at spawn time, which also lets a card resolve
    # to different engines per node on a heterogeneous cycle. Resolution uses the
    # same resolve_node_backend the worker would, against the master's telemetry
    # view of each node's advertised backends. A node absent from node_resources
    # leaves resolved_backend=None, and the worker falls back to its local probe.
    # An RPC placement restricts resolution to the llama_server tags: the whole
    # cycle runs the served RPC machinery by construction, and a card that also
    # allows in-process llama_cpp (with it earlier in preference or alphabetical
    # order) must not stamp the driver with a tag that would dispatch the
    # single-node in-process runner.
    compatible_backends = _card_platform_backends(command.model_card)
    if selected_is_rpc:
        compatible_backends = frozenset(
            tag
            for tag in compatible_backends
            if engine_of(tag) == "llama_server"
        )
    stamped_runner_to_shard = dict(shard_assignments.runner_to_shard)
    for node_id, runner_id in shard_assignments.node_to_runner.items():
        resources = resolved_node_resources.get(node_id)
        resolved_backend = (
            resolve_node_backend(
                compatible_backends, backend_preference, resources.backends
            )
            if resources is not None
            else None
        )
        if resolved_backend is not None:
            shard = stamped_runner_to_shard[runner_id]
            stamped_runner_to_shard[runner_id] = shard.model_copy(
                update={"resolved_backend": resolved_backend}
            )
    shard_assignments = ShardAssignments(
        model_id=shard_assignments.model_id,
        runner_to_shard=stamped_runner_to_shard,
        node_to_runner=shard_assignments.node_to_runner,
    )

    # Stamp the context-admission ceiling into the placement decision (#279
    # slice 2). Computed once here from the hosting nodes' static ram_total, so
    # every rank reads the identical value off replicated state rather than
    # recomputing from the (now telemetry-plane, last-write-wins) node memory.
    context_token_limit = instance_context_token_limit(
        shard_assignments,
        {
            node_id: node_memory[node_id].ram_total
            for node_id in selected_cycle.node_ids
        },
        node_vram=node_vram,
    )

    cycle_digraph: Topology = topology.get_subgraph_from_nodes(selected_cycle.node_ids)

    instance_id = InstanceId()
    target_instances = dict(deepcopy(current_instances))

    if selected_is_rpc:
        assert driver_node is not None
        # Donor endpoints are chosen once here and stamped on the instance:
        # the donor binds exactly this address and the driver dials it, so the
        # two sides can never disagree. Address selection reuses the ring's
        # observed-connection prioritiser (Thunderbolt first, VPN last, #265).
        donor_endpoints = get_llama_rpc_donor_endpoints(
            selected_cycle=selected_cycle,
            driver_node=driver_node,
            cycle_digraph=cycle_digraph,
            node_network=node_network,
            donor_ports={
                node_id: random_ephemeral_port()
                for node_id in selected_cycle.node_ids
                if node_id != driver_node
            },
        )
        target_instances[instance_id] = LlamaRpcInstance(
            instance_id=instance_id,
            shard_assignments=shard_assignments,
            context_token_limit=context_token_limit,
            driver_node=driver_node,
            donor_endpoints=donor_endpoints,
        )
        return target_instances

    match command.instance_meta:
        case InstanceMeta.LlamaRpc:
            # Unreachable: an explicit LlamaRpc request either resolved to the
            # RPC shape (returned above) or raised when the cycle cannot serve
            # it. Kept for match exhaustiveness.
            raise PlacementError(
                "Requested a multi-node llama.cpp (RPC) placement, but the "
                "selected cycle does not serve this model via llama_server."
            )
        case InstanceMeta.MlxJaccl:
            # TODO(evan): shard assignments should contain information about ranks, this is ugly
            def get_device_rank(node_id: NodeId) -> int:
                runner_id = shard_assignments.node_to_runner[node_id]
                shard_metadata = shard_assignments.runner_to_shard.get(runner_id)
                assert shard_metadata is not None
                return shard_metadata.device_rank

            zero_node_ids = [
                node_id
                for node_id in selected_cycle.node_ids
                if get_device_rank(node_id) == 0
            ]
            assert len(zero_node_ids) == 1
            coordinator_node_id = zero_node_ids[0]

            mlx_jaccl_devices = get_mlx_jaccl_devices_matrix(
                [node_id for node_id in selected_cycle],
                cycle_digraph,
            )
            mlx_jaccl_coordinators = get_mlx_jaccl_coordinators(
                coordinator=coordinator_node_id,
                coordinator_port=random_ephemeral_port(),
                cycle_digraph=cycle_digraph,
                node_network=node_network,
            )
            target_instances[instance_id] = MlxJacclInstance(
                instance_id=instance_id,
                shard_assignments=shard_assignments,
                context_token_limit=context_token_limit,
                jaccl_devices=mlx_jaccl_devices,
                jaccl_coordinators=mlx_jaccl_coordinators,
            )
        case InstanceMeta.MlxRing:
            ephemeral_port = random_ephemeral_port()
            hosts_by_node = get_mlx_ring_hosts_by_node(
                selected_cycle=selected_cycle,
                cycle_digraph=cycle_digraph,
                ephemeral_port=ephemeral_port,
                node_network=node_network,
            )
            target_instances[instance_id] = MlxRingInstance(
                instance_id=instance_id,
                shard_assignments=shard_assignments,
                context_token_limit=context_token_limit,
                hosts_by_node=hosts_by_node,
                ephemeral_port=ephemeral_port,
            )

    return target_instances


def _placement_intent_from_instance(
    instance: Instance,
) -> tuple[ModelCard, Sharding, InstanceMeta, int]:
    """Recover the placement intent (model, sharding, backend, width) of an instance.

    The model card, sharding family, and instance backend are read back off the
    instance's shards so a recovery re-placement reproduces the original intent.
    Returns the width (node count) too. Raises :class:`PlacementError` if the
    instance carries no shards (a structurally-allowed empty ``ShardAssignments``
    has no model to re-place), letting callers tear the husk down on the same
    path they use for a genuine no-fit rather than crashing.
    """
    shards = list(instance.shard_assignments.runner_to_shard.values())
    if not shards:
        raise PlacementError(
            f"Cannot re-place instance {instance.instance_id}: it has no shards"
        )
    # All shards of an instance share one model card.
    model_card = shards[0].model_card
    sharding = (
        Sharding.Tensor
        if any(isinstance(shard, TensorShardMetadata) for shard in shards)
        else Sharding.Pipeline
    )
    if isinstance(instance, MlxJacclInstance):
        instance_meta = InstanceMeta.MlxJaccl
    elif isinstance(instance, LlamaRpcInstance):
        instance_meta = InstanceMeta.LlamaRpc
    else:
        instance_meta = InstanceMeta.MlxRing
    width = len(instance.shard_assignments.node_to_runner)
    return model_card, sharding, instance_meta, width


def replacement_command_for_refused_instance(instance: Instance) -> PlaceInstance:
    """Build a *wider* placement command to recover a memory-refused instance.

    When a worker refuses its shard for lack of GPU-wireable memory at load
    time (#290), the master re-places the same model one node wider so every
    node holds a smaller share. The placement intent (model card, sharding
    family, instance backend) is recovered from the instance itself; the
    operator's original per-placement node exclusions are not retained on the
    instance, so the wider re-placement searches the full topology.

    ``min_nodes`` is the refused width plus one. Raising it past the cluster
    size makes :func:`place_instance` raise :class:`PlacementError`, which the
    caller treats as the terminal "cannot fit anywhere" outcome — this bounds
    the refuse→re-place loop to at most ``cluster_size`` iterations.

    Raises :class:`PlacementError` if the instance carries no shards.
    """
    model_card, sharding, instance_meta, width = _placement_intent_from_instance(
        instance
    )
    return PlaceInstance(
        model_card=model_card,
        sharding=sharding,
        instance_meta=instance_meta,
        min_nodes=width + 1,
    )


def replacement_command_for_download_failed_instance(
    instance: Instance, excluded_nodes: AbstractSet[NodeId]
) -> PlaceInstance:
    """Build a *same-width* re-placement that avoids the download-failed nodes (#381).

    When a rank's model download terminally fails (disk full, transient HF or
    network error), the instance can never become ready. Recovery keeps the
    same world size but excludes the node(s) whose download failed, forcing the
    planner onto healthy nodes. If too few nodes remain to host the width,
    :func:`place_instance` raises :class:`PlacementError`, which the caller
    treats as the terminal "cannot recover" outcome and stops at the teardown —
    bounding recovery to the available healthy nodes rather than looping.

    The excluded set must be passed to :func:`place_instance` as well; it is
    carried on the command only as a record of intent.

    Raises :class:`PlacementError` if the instance carries no shards.
    """
    model_card, sharding, instance_meta, width = _placement_intent_from_instance(
        instance
    )
    return PlaceInstance(
        model_card=model_card,
        sharding=sharding,
        instance_meta=instance_meta,
        min_nodes=width,
        excluded_nodes=list(excluded_nodes),
    )


def delete_instance(
    command: DeleteInstance,
    current_instances: Mapping[InstanceId, Instance],
) -> dict[InstanceId, Instance]:
    target_instances = dict(deepcopy(current_instances))
    if command.instance_id in target_instances:
        del target_instances[command.instance_id]
        return target_instances
    raise ValueError(f"Instance {command.instance_id} not found")


def get_transition_events(
    current_instances: Mapping[InstanceId, Instance],
    target_instances: Mapping[InstanceId, Instance],
    tasks: Mapping[TaskId, Task],
) -> Sequence[Event]:
    events: list[Event] = []

    # find instances to create
    for instance_id, instance in target_instances.items():
        if instance_id not in current_instances:
            events.append(
                InstanceCreated(
                    instance=instance,
                )
            )

    # find instances to delete
    for instance_id in current_instances:
        if instance_id not in target_instances:
            for task in tasks.values():
                if task.instance_id == instance_id and task.task_status in [
                    TaskStatus.Pending,
                    TaskStatus.Running,
                ]:
                    events.append(
                        TaskStatusUpdated(
                            task_status=TaskStatus.Cancelled,
                            task_id=task.task_id,
                        )
                    )

            events.append(
                InstanceDeleted(
                    instance_id=instance_id,
                )
            )

    return events


def cancel_unnecessary_downloads(
    instances: Mapping[InstanceId, Instance],
    download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> Sequence[DownloadCommand]:
    commands: list[DownloadCommand] = []
    currently_downloading = [
        (k, v.shard_metadata.model_card.model_id)
        for k, vs in download_status.items()
        for v in vs
        if isinstance(v, (DownloadOngoing))
    ]
    active_models = set(
        (
            node_id,
            instance.shard_assignments.runner_to_shard[runner_id].model_card.model_id,
        )
        for instance in instances.values()
        for node_id, runner_id in instance.shard_assignments.node_to_runner.items()
    )
    for pair in currently_downloading:
        if pair not in active_models:
            commands.append(CancelDownload(target_node_id=pair[0], model_id=pair[1]))

    return commands
