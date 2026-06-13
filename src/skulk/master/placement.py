import random
from collections.abc import Mapping
from copy import deepcopy
from typing import Sequence

from skulk.master.placement_utils import (
    Cycle,
    filter_cycles_by_memory,
    get_mlx_jaccl_coordinators,
    get_mlx_jaccl_devices_matrix,
    get_mlx_ring_hosts_by_node,
    get_shard_assignments,
    get_smallest_cycles,
)
from skulk.shared.models.model_cards import ModelId
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
    MlxJacclInstance,
    MlxRingInstance,
)
from skulk.shared.types.worker.shards import Sharding


def random_ephemeral_port() -> int:
    port = random.randint(49153, 65535)
    return port - 1 if port <= 52415 else port


def add_instance_to_placements(
    command: CreateInstance,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
) -> Mapping[InstanceId, Instance]:
    # TODO: validate against topology

    return {**current_instances, command.instance.instance_id: command.instance}


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
        compatible_backends = command.model_card.placement.compatible_backends
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
    cycles_with_sufficient_memory, memory_diagnostics = filter_cycles_by_memory(
        candidate_cycles,
        node_memory,
        command.model_card,
        command.sharding,
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

    selected_cycle = max(
        candidate_cycles,
        key=lambda cycle: (
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

    shard_assignments = get_shard_assignments(
        command.model_card, selected_cycle, command.sharding, node_memory
    )

    cycle_digraph: Topology = topology.get_subgraph_from_nodes(selected_cycle.node_ids)

    instance_id = InstanceId()
    target_instances = dict(deepcopy(current_instances))

    match command.instance_meta:
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
                hosts_by_node=hosts_by_node,
                ephemeral_port=ephemeral_port,
            )

    return target_instances


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
