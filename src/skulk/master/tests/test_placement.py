import pytest

from skulk.master.placement import (
    PlacementError,
    PlacementInfoPendingError,
    add_instance_to_placements,
    get_transition_events,
    place_instance,
    replacement_command_for_refused_instance,
)
from skulk.master.tests.conftest import (
    create_node_memory,
    create_node_network,
    create_rdma_connection,
    create_socket_connection,
)
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.topology import Topology
from skulk.shared.types.commands import CreateInstance, PlaceInstance
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.events import (
    InstanceCreated,
    InstanceDeleted,
    TaskStatusUpdated,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.multiaddr import Multiaddr
from skulk.shared.types.profiling import (
    MemoryUsage,
    NetworkInterfaceInfo,
    NodeNetworkInfo,
    NodeResources,
)
from skulk.shared.types.tasks import TaskId, TaskStatus, TextGeneration
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.shared.types.topology import Connection, SocketConnection
from skulk.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadProgressData,
)
from skulk.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceMeta,
    MlxJacclInstance,
    MlxRingInstance,
)
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata, Sharding


@pytest.fixture
def instance() -> Instance:
    return MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=ModelId("test-model"), runner_to_shard={}, node_to_runner={}
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )


@pytest.fixture
def model_card() -> ModelCard:
    return ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_gb(1),
        n_layers=10,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )


def place_instance_command(model_card: ModelCard) -> PlaceInstance:
    return PlaceInstance(
        command_id=CommandId(),
        model_card=model_card,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
    )


@pytest.mark.parametrize(
    "available_memory,total_layers,expected_layers",
    [
        ((5.0, 5.0, 10.0), 12, (3, 3, 6)),
        ((5.0, 5.0, 5.0), 12, (4, 4, 4)),
        ((3.12, 4.68, 10.92), 12, (2, 3, 7)),
    ],
)
def test_get_instance_placements_create_instance(
    available_memory: tuple[float, float, float],
    total_layers: int,
    expected_layers: tuple[int, int, int],
    model_card: ModelCard,
):
    # arrange
    model_card.n_layers = total_layers
    # 65% of the cycle's total memory: large enough that no 2-node cycle or
    # singleton passes the per-node headroom check (forcing the 3-node
    # placement this test asserts on), small enough that the 3-node cycle
    # fits under the 1.30x weight-overhead factor. The layer split itself only
    # depends on the memory fractions, not this size.
    model_card.storage_size = Memory.from_gb(sum(available_memory) * 0.65)
    topology = Topology()

    cic = place_instance_command(model_card)
    node_id_a = NodeId()
    node_id_b = NodeId()
    node_id_c = NodeId()

    # fully connected (directed) between the 3 nodes
    conn_a_b = Connection(
        source=node_id_a, sink=node_id_b, edge=create_socket_connection(1)
    )
    conn_b_c = Connection(
        source=node_id_b, sink=node_id_c, edge=create_socket_connection(2)
    )
    conn_c_a = Connection(
        source=node_id_c, sink=node_id_a, edge=create_socket_connection(3)
    )
    conn_c_b = Connection(
        source=node_id_c, sink=node_id_b, edge=create_socket_connection(4)
    )
    conn_a_c = Connection(
        source=node_id_a, sink=node_id_c, edge=create_socket_connection(5)
    )
    conn_b_a = Connection(
        source=node_id_b, sink=node_id_a, edge=create_socket_connection(6)
    )

    node_memory = {
        node_id_a: create_node_memory(Memory.from_gb(available_memory[0]).in_bytes),
        node_id_b: create_node_memory(Memory.from_gb(available_memory[1]).in_bytes),
        node_id_c: create_node_memory(Memory.from_gb(available_memory[2]).in_bytes),
    }
    node_network = {
        node_id_a: create_node_network(),
        node_id_b: create_node_network(),
        node_id_c: create_node_network(),
    }
    topology.add_node(node_id_a)
    topology.add_node(node_id_b)
    topology.add_node(node_id_c)
    topology.add_connection(conn_a_b)
    topology.add_connection(conn_b_c)
    topology.add_connection(conn_c_a)
    topology.add_connection(conn_c_b)
    topology.add_connection(conn_a_c)
    topology.add_connection(conn_b_a)

    # act
    placements = place_instance(cic, topology, {}, node_memory, node_network)

    # assert
    assert len(placements) == 1
    instance_id = list(placements.keys())[0]
    instance = placements[instance_id]
    assert instance.shard_assignments.model_id == model_card.model_id

    runner_id_a = instance.shard_assignments.node_to_runner[node_id_a]
    runner_id_b = instance.shard_assignments.node_to_runner[node_id_b]
    runner_id_c = instance.shard_assignments.node_to_runner[node_id_c]

    shard_a = instance.shard_assignments.runner_to_shard[runner_id_a]
    shard_b = instance.shard_assignments.runner_to_shard[runner_id_b]
    shard_c = instance.shard_assignments.runner_to_shard[runner_id_c]

    assert shard_a.end_layer - shard_a.start_layer == expected_layers[0]
    assert shard_b.end_layer - shard_b.start_layer == expected_layers[1]
    assert shard_c.end_layer - shard_c.start_layer == expected_layers[2]

    shards = [shard_a, shard_b, shard_c]
    shards_sorted = sorted(shards, key=lambda s: s.start_layer)
    assert shards_sorted[0].start_layer == 0
    assert shards_sorted[-1].end_layer == total_layers


def test_get_instance_placements_one_node_exact_fit_is_rejected() -> None:
    """Weights exactly equal to available memory must be refused.

    An exact fit leaves nothing for KV cache, activations, or the runner
    itself — admitting it produces the silent-thrash failure observed in
    the 2026-06-05 launch smoke (12-token prefill in 1230s). The per-node
    headroom check turns that into an explicit placement error.
    """
    topology = Topology()
    node_id = NodeId()
    topology.add_node(node_id)
    node_memory = {node_id: create_node_memory(Memory.from_gb(8).in_bytes)}
    node_network = {node_id: create_node_network()}
    cic = place_instance_command(
        ModelCard(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_gb(8),
            n_layers=10,
            hidden_size=1000,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
        ),
    )
    with pytest.raises(ValueError, match="No candidate cycle fits"):
        place_instance(cic, topology, {}, node_memory, node_network)


def test_get_instance_placements_one_node_fits_with_extra_memory() -> None:
    topology = Topology()
    node_id = NodeId()
    topology.add_node(node_id)
    node_memory = {node_id: create_node_memory(Memory.from_gb(8).in_bytes)}
    node_network = {node_id: create_node_network()}
    cic = place_instance_command(
        ModelCard(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_gb(5),
            n_layers=10,
            hidden_size=1000,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
        ),
    )
    placements = place_instance(cic, topology, {}, node_memory, node_network)

    assert len(placements) == 1
    instance_id = list(placements.keys())[0]
    instance = placements[instance_id]
    assert instance.shard_assignments.model_id == "test-model"
    assert len(instance.shard_assignments.node_to_runner) == 1
    assert len(instance.shard_assignments.runner_to_shard) == 1
    assert len(instance.shard_assignments.runner_to_shard) == 1


def test_placement_stamps_context_token_limit() -> None:
    # #279 slice 2: the master computes the context-admission ceiling once at
    # placement time and stamps it onto the instance, so every rank reads the
    # identical value instead of recomputing from (now telemetry-plane) memory.
    topology = Topology()
    node_id = NodeId()
    topology.add_node(node_id)
    node_memory = {node_id: create_node_memory(Memory.from_gb(8).in_bytes)}
    node_network = {node_id: create_node_network()}
    cic = place_instance_command(
        ModelCard(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_gb(5),
            n_layers=10,
            hidden_size=1000,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
            context_length=4096,
        ),
    )
    placements = place_instance(cic, topology, {}, node_memory, node_network)
    instance = next(iter(placements.values()))
    # A card-advertised context_length makes the ceiling enforceable; the stamp
    # is present and never exceeds the advertised limit.
    assert instance.context_token_limit is not None
    assert instance.context_token_limit <= 4096


def test_get_instance_placements_one_node_not_fit() -> None:
    topology = Topology()
    node_id = NodeId()
    topology.add_node(node_id)
    node_memory = {node_id: create_node_memory(Memory.from_gb(8).in_bytes)}
    node_network = {node_id: create_node_network()}
    cic = place_instance_command(
        model_card=ModelCard(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_gb(9),
            n_layers=10,
            hidden_size=1000,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
        ),
    )

    with pytest.raises(ValueError, match="No candidate cycle fits"):
        place_instance(cic, topology, {}, node_memory, node_network)


def _two_node_topology():
    """Two-node topology where either node can host a placement on its own.

    Used by the excluded-nodes tests to verify the planner picks the alternate
    node when one is excluded, and refuses to place when *all* candidates are
    excluded. Return type is left loose intentionally — the values match the
    `place_instance` signature without us recomputing the helper return types.
    """
    topology = Topology()
    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)
    # Each node has enough memory to host the small model on its own.
    node_memory = {
        node_a: create_node_memory(Memory.from_gb(8).in_bytes),
        node_b: create_node_memory(Memory.from_gb(8).in_bytes),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    return topology, node_a, node_b, node_memory, node_network


def _small_model_card() -> ModelCard:
    return ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_mb(500),
        n_layers=10,
        hidden_size=1000,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )


def test_excluding_one_node_routes_placement_to_the_other() -> None:
    """The planner must pick a candidate cycle that doesn't touch any excluded
    node when an alternative exists."""
    topology, node_a, node_b, node_memory, node_network = _two_node_topology()
    command = place_instance_command(_small_model_card())

    placements = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        excluded_nodes={node_a},
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assigned_nodes = set(instance.shard_assignments.node_to_runner.keys())
    assert assigned_nodes == {node_b}


def test_excluding_all_candidate_nodes_fails_to_place() -> None:
    """If every cycle that would otherwise satisfy the placement is excluded,
    the planner raises rather than picking an excluded node anyway."""
    topology, node_a, node_b, node_memory, node_network = _two_node_topology()
    command = place_instance_command(_small_model_card())

    with pytest.raises(ValueError, match="touch an excluded node"):
        place_instance(
            command,
            topology,
            {},
            node_memory,
            node_network,
            excluded_nodes={node_a, node_b},
        )


def test_empty_exclusion_set_preserves_default_behavior() -> None:
    """Defensive: passing an empty set must not change which placements are
    legal — the filter should be a no-op."""
    topology, _node_a, _node_b, node_memory, node_network = _two_node_topology()
    command = place_instance_command(_small_model_card())

    without_filter = place_instance(command, topology, {}, node_memory, node_network)
    with_empty_filter = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        excluded_nodes=set(),
    )

    # Both calls must produce equivalent placement sets (same node count, same
    # model). We don't assert exact node identity because the planner may pick
    # either node when both qualify.
    assert len(without_filter) == 1
    assert len(with_empty_filter) == 1
    assert (
        next(iter(without_filter.values())).shard_assignments.model_id
        == next(iter(with_empty_filter.values())).shard_assignments.model_id
    )


def test_min_nodes_above_cluster_size_is_a_hard_error() -> None:
    """min_nodes greater than the number of known nodes can never succeed —
    hard PlacementError, no retry semantics."""
    topology, _node_a, _node_b, node_memory, node_network = _two_node_topology()
    command = place_instance_command(_small_model_card())
    command.min_nodes = 3

    with pytest.raises(PlacementError, match="min_nodes=3 is impossible"):
        place_instance(command, topology, {}, node_memory, node_network)


def test_unconnected_nodes_at_min_nodes_reports_info_pending() -> None:
    """Enough nodes exist but no connecting edges yet: right after cluster
    formation the connection edges lag node identities by a few gossip
    rounds, so this must surface as info-pending (retry shortly), not as a
    hard topology error — and never as the old 'insufficient memory' lie."""
    topology = Topology()
    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)  # no connections gossiped yet
    node_memory = {
        node_a: create_node_memory(Memory.from_gb(8).in_bytes),
        node_b: create_node_memory(Memory.from_gb(8).in_bytes),
    }
    node_network = {node_a: create_node_network(), node_b: create_node_network()}
    command = place_instance_command(_small_model_card())
    command.min_nodes = 2

    with pytest.raises(PlacementInfoPendingError, match="retry shortly"):
        place_instance(command, topology, {}, node_memory, node_network)


def test_missing_node_memory_reports_info_pending() -> None:
    """A connected pair where one node's memory info has not arrived yet is
    the startup race observed on 2026-06-06: it must surface as
    info-pending, not 'insufficient memory'."""
    topology, node_a, node_b, node_memory, node_network = _two_node_topology()
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_socket_connection(2))
    )
    command = place_instance_command(_small_model_card())
    command.min_nodes = 2
    del node_memory[node_b]

    with pytest.raises(PlacementInfoPendingError, match="Memory info"):
        place_instance(command, topology, {}, node_memory, node_network)


def test_get_transition_events_no_change(instance: Instance):
    # arrange
    instance_id = InstanceId()
    current_instances = {instance_id: instance}
    target_instances = {instance_id: instance}

    # act
    events = get_transition_events(current_instances, target_instances, {})

    # assert
    assert len(events) == 0


def test_get_transition_events_create_instance(instance: Instance):
    # arrange
    instance_id = InstanceId()
    current_instances: dict[InstanceId, Instance] = {}
    target_instances: dict[InstanceId, Instance] = {instance_id: instance}

    # act
    events = get_transition_events(current_instances, target_instances, {})

    # assert
    assert len(events) == 1
    assert isinstance(events[0], InstanceCreated)


def test_get_transition_events_delete_instance(instance: Instance):
    # arrange
    instance_id = InstanceId()
    current_instances: dict[InstanceId, Instance] = {instance_id: instance}
    target_instances: dict[InstanceId, Instance] = {}

    # act
    events = get_transition_events(current_instances, target_instances, {})

    # assert
    assert len(events) == 1
    assert isinstance(events[0], InstanceDeleted)
    assert events[0].instance_id == instance_id


def test_placement_selects_leaf_nodes(
    model_card: ModelCard,
):
    # arrange
    topology = Topology()

    # 3 GB: too big for any singleton under the headroom check (largest node
    # has 3 GB available), so the planner must use a 2-node cycle — which is
    # what lets this test observe the leaf-node preference.
    model_card.storage_size = Memory.from_gb(3)

    node_id_a = NodeId()
    node_id_b = NodeId()
    node_id_c = NodeId()
    node_id_d = NodeId()

    node_memory = {
        node_id_a: create_node_memory(Memory.from_gb(2).in_bytes),
        node_id_b: create_node_memory(Memory.from_gb(3).in_bytes),
        node_id_c: create_node_memory(Memory.from_gb(3).in_bytes),
        node_id_d: create_node_memory(Memory.from_gb(2).in_bytes),
    }
    node_network = {
        node_id_a: create_node_network(),
        node_id_b: create_node_network(),
        node_id_c: create_node_network(),
        node_id_d: create_node_network(),
    }

    topology.add_node(node_id_a)
    topology.add_node(node_id_b)
    topology.add_node(node_id_c)
    topology.add_node(node_id_d)

    # Daisy chain topology (directed)
    topology.add_connection(
        Connection(source=node_id_a, sink=node_id_b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_id_b, sink=node_id_a, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_id_b, sink=node_id_c, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_id_c, sink=node_id_b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_id_c, sink=node_id_d, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_id_d, sink=node_id_c, edge=create_socket_connection(1))
    )

    cic = place_instance_command(model_card=model_card)

    # act
    placements = place_instance(cic, topology, {}, node_memory, node_network)

    # assert
    assert len(placements) == 1
    instance = list(placements.values())[0]

    assigned_nodes = set(instance.shard_assignments.node_to_runner.keys())
    assert assigned_nodes == set((node_id_a, node_id_b)) or assigned_nodes == set(
        (
            node_id_c,
            node_id_d,
        )
    )


def test_tensor_rdma_backend_connectivity_matrix(
    model_card: ModelCard,
):
    # arrange
    topology = Topology()
    model_card.n_layers = 12
    model_card.storage_size = Memory.from_gb(1.5)

    node_a = NodeId()
    node_b = NodeId()
    node_c = NodeId()

    node_memory = {
        node_a: create_node_memory(Memory.from_gb(1).in_bytes),
        node_b: create_node_memory(Memory.from_gb(1).in_bytes),
        node_c: create_node_memory(Memory.from_gb(1).in_bytes),
    }

    ethernet_interface = NetworkInterfaceInfo(
        name="en0",
        ip_address="10.0.0.1",
    )
    ethernet_conn = SocketConnection(
        sink_multiaddr=Multiaddr(address="/ip4/10.0.0.1/tcp/8000")
    )

    node_network = {
        node_a: NodeNetworkInfo(interfaces=[ethernet_interface]),
        node_b: NodeNetworkInfo(interfaces=[ethernet_interface]),
        node_c: NodeNetworkInfo(interfaces=[ethernet_interface]),
    }

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_node(node_c)

    # RDMA connections (directed)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_rdma_connection(3))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_rdma_connection(3))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_c, edge=create_rdma_connection(4))
    )
    topology.add_connection(
        Connection(source=node_c, sink=node_b, edge=create_rdma_connection(4))
    )
    topology.add_connection(
        Connection(source=node_a, sink=node_c, edge=create_rdma_connection(5))
    )
    topology.add_connection(
        Connection(source=node_c, sink=node_a, edge=create_rdma_connection(5))
    )

    # Ethernet connections (directed)
    topology.add_connection(Connection(source=node_a, sink=node_b, edge=ethernet_conn))
    topology.add_connection(Connection(source=node_b, sink=node_c, edge=ethernet_conn))
    topology.add_connection(Connection(source=node_c, sink=node_a, edge=ethernet_conn))
    topology.add_connection(Connection(source=node_a, sink=node_c, edge=ethernet_conn))
    topology.add_connection(Connection(source=node_b, sink=node_a, edge=ethernet_conn))
    topology.add_connection(Connection(source=node_c, sink=node_b, edge=ethernet_conn))

    cic = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=1,
    )

    # act
    placements = place_instance(cic, topology, {}, node_memory, node_network)

    # assert
    assert len(placements) == 1
    instance_id = list(placements.keys())[0]
    instance = placements[instance_id]

    assert isinstance(instance, MlxJacclInstance)

    assert instance.jaccl_devices is not None
    assert instance.jaccl_coordinators is not None

    matrix = instance.jaccl_devices
    assert len(matrix) == 3
    for i in range(3):
        assert matrix[i][i] is None

    assigned_nodes = list(instance.shard_assignments.node_to_runner.keys())
    node_to_idx = {node_id: idx for idx, node_id in enumerate(assigned_nodes)}

    idx_a = node_to_idx[node_a]
    idx_b = node_to_idx[node_b]
    idx_c = node_to_idx[node_c]

    assert matrix[idx_a][idx_b] == "rdma_en3"
    assert matrix[idx_b][idx_c] == "rdma_en4"
    assert matrix[idx_c][idx_a] == "rdma_en5"

    # Verify coordinators are set for all nodes
    assert len(instance.jaccl_coordinators) == 3
    for node_id in assigned_nodes:
        assert node_id in instance.jaccl_coordinators
        coordinator = instance.jaccl_coordinators[node_id]
        assert ":" in coordinator
        # Rank 0 node should use 0.0.0.0, others should use connection-specific IPs
        if node_id == assigned_nodes[0]:
            assert coordinator.startswith("0.0.0.0:")
        else:
            ip_part = coordinator.split(":")[0]
            assert len(ip_part.split(".")) == 4


def _make_task(
    instance_id: InstanceId,
    status: TaskStatus = TaskStatus.Running,
) -> TextGeneration:
    return TextGeneration(
        task_id=TaskId(),
        task_status=status,
        instance_id=instance_id,
        command_id=CommandId(),
        task_params=TextGenerationTaskParams(
            model=ModelId("test-model"),
            input=[InputMessage(role="user", content="hello")],
        ),
    )


def test_get_transition_events_delete_instance_cancels_running_tasks(
    instance: Instance,
):
    # arrange
    instance_id = InstanceId()
    current_instances: dict[InstanceId, Instance] = {instance_id: instance}
    target_instances: dict[InstanceId, Instance] = {}
    task = _make_task(instance_id, TaskStatus.Running)
    tasks = {task.task_id: task}

    # act
    events = get_transition_events(current_instances, target_instances, tasks)

    # assert – cancellation event should come before the deletion event
    assert len(events) == 2
    assert isinstance(events[0], TaskStatusUpdated)
    assert events[0].task_id == task.task_id
    assert events[0].task_status == TaskStatus.Cancelled
    assert isinstance(events[1], InstanceDeleted)
    assert events[1].instance_id == instance_id


def test_get_transition_events_delete_instance_cancels_pending_tasks(
    instance: Instance,
):
    # arrange
    instance_id = InstanceId()
    current_instances: dict[InstanceId, Instance] = {instance_id: instance}
    target_instances: dict[InstanceId, Instance] = {}
    task = _make_task(instance_id, TaskStatus.Pending)
    tasks = {task.task_id: task}

    # act
    events = get_transition_events(current_instances, target_instances, tasks)

    # assert
    assert len(events) == 2
    assert isinstance(events[0], TaskStatusUpdated)
    assert events[0].task_id == task.task_id
    assert events[0].task_status == TaskStatus.Cancelled
    assert isinstance(events[1], InstanceDeleted)


def test_get_transition_events_delete_instance_ignores_completed_tasks(
    instance: Instance,
):
    # arrange
    instance_id = InstanceId()
    current_instances: dict[InstanceId, Instance] = {instance_id: instance}
    target_instances: dict[InstanceId, Instance] = {}
    tasks = {
        t.task_id: t
        for t in [
            _make_task(instance_id, TaskStatus.Complete),
            _make_task(instance_id, TaskStatus.Failed),
            _make_task(instance_id, TaskStatus.TimedOut),
            _make_task(instance_id, TaskStatus.Cancelled),
        ]
    }

    # act
    events = get_transition_events(current_instances, target_instances, tasks)

    # assert – only the InstanceDeleted event, no cancellations
    assert len(events) == 1
    assert isinstance(events[0], InstanceDeleted)


def test_get_transition_events_delete_instance_cancels_only_matching_tasks(
    instance: Instance,
):
    # arrange
    instance_id_a = InstanceId()
    instance_id_b = InstanceId()
    current_instances: dict[InstanceId, Instance] = {
        instance_id_a: instance,
        instance_id_b: instance,
    }
    # only delete instance A, keep instance B
    target_instances: dict[InstanceId, Instance] = {instance_id_b: instance}

    task_a = _make_task(instance_id_a, TaskStatus.Running)
    task_b = _make_task(instance_id_b, TaskStatus.Running)
    tasks = {task_a.task_id: task_a, task_b.task_id: task_b}

    # act
    events = get_transition_events(current_instances, target_instances, tasks)

    # assert – only task_a should be cancelled
    cancel_events = [e for e in events if isinstance(e, TaskStatusUpdated)]
    delete_events = [e for e in events if isinstance(e, InstanceDeleted)]
    assert len(cancel_events) == 1
    assert cancel_events[0].task_id == task_a.task_id
    assert cancel_events[0].task_status == TaskStatus.Cancelled
    assert len(delete_events) == 1
    assert delete_events[0].instance_id == instance_id_a


def _make_shard_metadata(model_card: ModelCard) -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=model_card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=model_card.n_layers,
        n_layers=model_card.n_layers,
    )


def test_legacy_instance_backfills_context_token_limit_from_card() -> None:
    # An instance hydrated without a stamped ceiling (pre-#279 slice 2 snapshot
    # / event / older CreateInstance) must fall back to the card's context
    # length so the #145 admission guard still applies (review catch on #292).
    node_id = NodeId()
    runner_id = RunnerId()
    card = ModelCard(
        model_id=ModelId("legacy-model"),
        storage_size=Memory.from_gb(1),
        n_layers=10,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        context_length=8192,
    )
    instance = MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=ModelId("legacy-model"),
            runner_to_shard={runner_id: _make_shard_metadata(card)},
            node_to_runner={node_id: runner_id},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
        # context_token_limit deliberately omitted (the legacy/None case)
    )
    assert instance.context_token_limit == 8192


def test_create_instance_restamps_context_token_limit() -> None:
    # The exact-control POST /instance path must stamp the master's
    # memory-derived ceiling too (#292 review) — runners trust the stamped
    # field now, so a client-supplied placement can't smuggle in an inflated
    # ceiling and reach MLX past the safe limit.
    node_id = NodeId()
    runner_id = RunnerId()
    card = ModelCard(
        model_id=ModelId("exact-model"),
        storage_size=Memory.from_gb(1),
        n_layers=10,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        context_length=4096,
    )
    client_instance = MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=ModelId("exact-model"),
            runner_to_shard={runner_id: _make_shard_metadata(card)},
            node_to_runner={node_id: runner_id},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
        context_token_limit=999_999,  # client-inflated; must be overridden
    )
    command = CreateInstance(command_id=CommandId(), instance=client_instance)
    node_memory = {node_id: create_node_memory(Memory.from_gb(8).in_bytes)}
    result = add_instance_to_placements(command, Topology(), {}, node_memory)
    stamped = next(iter(result.values())).context_token_limit
    assert stamped is not None and stamped <= 4096


def test_placement_prefers_cycle_with_downloaded_model(
    model_card: ModelCard,
) -> None:
    """When two cycles are otherwise equal, prefer the one with the model already downloaded."""
    topology = Topology()

    model_card.storage_size = Memory.from_mb(500)

    node_a = NodeId()
    node_b = NodeId()

    node_memory = {
        node_a: create_node_memory(Memory.from_gb(2).in_bytes),
        node_b: create_node_memory(Memory.from_gb(2).in_bytes),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }

    topology.add_node(node_a)
    topology.add_node(node_b)
    # No connections between them — two single-node cycles

    shard_meta = _make_shard_metadata(model_card)

    # node_b has the model fully downloaded, node_a does not
    download_status = {
        node_b: [
            DownloadCompleted(
                node_id=node_b,
                shard_metadata=shard_meta,
                total=model_card.storage_size,
            ),
        ],
    }

    cic = place_instance_command(model_card)
    placements = place_instance(
        cic, topology, {}, node_memory, node_network, download_status=download_status
    )

    assert len(placements) == 1
    instance = list(placements.values())[0]
    assigned_nodes = set(instance.shard_assignments.node_to_runner.keys())
    assert assigned_nodes == {node_b}


def test_placement_prefers_cycle_with_higher_download_progress(
    model_card: ModelCard,
) -> None:
    """When two cycles are otherwise equal, prefer the one with more download progress."""
    topology = Topology()

    model_card.storage_size = Memory.from_gb(1)

    node_a = NodeId()
    node_b = NodeId()

    node_memory = {
        node_a: create_node_memory(Memory.from_gb(2).in_bytes),
        node_b: create_node_memory(Memory.from_gb(2).in_bytes),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }

    topology.add_node(node_a)
    topology.add_node(node_b)

    shard_meta = _make_shard_metadata(model_card)

    # node_a: 30% downloaded, node_b: 80% downloaded
    download_status = {
        node_a: [
            DownloadOngoing(
                node_id=node_a,
                shard_metadata=shard_meta,
                download_progress=DownloadProgressData(
                    total=Memory.from_bytes(1000),
                    downloaded=Memory.from_bytes(300),
                    downloaded_this_session=Memory.from_bytes(300),
                    completed_files=0,
                    total_files=1,
                    speed=0.0,
                    eta_ms=0,
                    files={},
                ),
            ),
        ],
        node_b: [
            DownloadOngoing(
                node_id=node_b,
                shard_metadata=shard_meta,
                download_progress=DownloadProgressData(
                    total=Memory.from_bytes(1000),
                    downloaded=Memory.from_bytes(800),
                    downloaded_this_session=Memory.from_bytes(800),
                    completed_files=0,
                    total_files=1,
                    speed=0.0,
                    eta_ms=0,
                    files={},
                ),
            ),
        ],
    }

    cic = place_instance_command(model_card)
    placements = place_instance(
        cic, topology, {}, node_memory, node_network, download_status=download_status
    )

    assert len(placements) == 1
    instance = list(placements.values())[0]
    assigned_nodes = set(instance.shard_assignments.node_to_runner.keys())
    assert assigned_nodes == {node_b}


def test_placement_does_not_prefer_cycle_with_failed_download(
    model_card: ModelCard,
) -> None:
    """A failed download should count as 0% — not preferred over a node with no download history."""
    topology = Topology()

    model_card.storage_size = Memory.from_mb(500)

    node_a = NodeId()
    node_b = NodeId()

    # node_a has slightly more RAM so it would win on the RAM tiebreaker
    node_memory = {
        node_a: create_node_memory(Memory.from_gb(2.001).in_bytes),
        node_b: create_node_memory(Memory.from_gb(2).in_bytes),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }

    topology.add_node(node_a)
    topology.add_node(node_b)

    shard_meta = _make_shard_metadata(model_card)

    # node_b has a failed download — should not be preferred
    download_status = {
        node_b: [
            DownloadFailed(
                node_id=node_b,
                shard_metadata=shard_meta,
                error_message="connection reset",
            ),
        ],
    }

    cic = place_instance_command(model_card)
    placements = place_instance(
        cic, topology, {}, node_memory, node_network, download_status=download_status
    )

    assert len(placements) == 1
    instance = list(placements.values())[0]
    assigned_nodes = set(instance.shard_assignments.node_to_runner.keys())
    # node_a should win on RAM tiebreaker since failed download scores 0.0
    assert assigned_nodes == {node_a}


def test_management_node_is_excluded_from_placement() -> None:
    """A node declaring participation="management" must never receive an
    inference shard, even though it is topologically present and has memory.
    The planner routes to the full-participation node instead (#149)."""
    topology, node_a, node_b, node_memory, node_network = _two_node_topology()
    command = place_instance_command(_small_model_card())

    placements = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        node_resources={
            node_a: NodeResources(participation="management"),
            node_b: NodeResources(participation="full"),
        },
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert set(instance.shard_assignments.node_to_runner.keys()) == {node_b}


def test_all_management_nodes_fail_to_place() -> None:
    """If every candidate node is management-only, placement raises rather
    than assigning a shard to a non-participating node."""
    topology, node_a, node_b, node_memory, node_network = _two_node_topology()
    command = place_instance_command(_small_model_card())

    with pytest.raises(ValueError, match="ineligible for this model"):
        place_instance(
            command,
            topology,
            {},
            node_memory,
            node_network,
            node_resources={
                node_a: NodeResources(participation="management"),
                node_b: NodeResources(participation="management"),
            },
        )


def test_backend_incompatible_node_is_excluded() -> None:
    """A node whose advertised backends do not intersect the card's
    compatible_backends (default {"mlx"}) is filtered out (#149)."""
    topology, node_a, node_b, node_memory, node_network = _two_node_topology()
    command = place_instance_command(_small_model_card())

    placements = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        node_resources={
            node_a: NodeResources(backends=frozenset({"llama_cpp"})),
            node_b: NodeResources(backends=frozenset({"mlx"})),
        },
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert set(instance.shard_assignments.node_to_runner.keys()) == {node_b}


def test_missing_node_resources_is_treated_as_eligible() -> None:
    """Nodes with no resources entry yet (gossip still warming up) must remain
    placeable, matching the pre-#149 full/mlx default — no regression."""
    topology, _node_a, _node_b, node_memory, node_network = _two_node_topology()
    command = place_instance_command(_small_model_card())

    placements = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        node_resources={},  # nothing gossiped yet
    )

    assert len(placements) == 1


def _fully_connected_three_nodes(
    available_memory: tuple[float, float, float],
) -> tuple[
    Topology,
    dict[NodeId, MemoryUsage],
    dict[NodeId, NodeNetworkInfo],
    list[NodeId],
]:
    """Build a fully-connected 3-node topology with the given available RAM.

    Returns ``(topology, node_memory, node_network, node_ids)``. Used by the
    refused-placement re-placement tests (#290).
    """
    topology = Topology()
    node_ids = [NodeId(), NodeId(), NodeId()]
    for node_id in node_ids:
        topology.add_node(node_id)
    # directed full mesh
    port = 1
    for source in node_ids:
        for sink in node_ids:
            if source != sink:
                topology.add_connection(
                    Connection(
                        source=source,
                        sink=sink,
                        edge=create_socket_connection(port),
                    )
                )
                port += 1
    node_memory = {
        node_ids[i]: create_node_memory(Memory.from_gb(available_memory[i]).in_bytes)
        for i in range(3)
    }
    node_network = {node_id: create_node_network() for node_id in node_ids}
    return topology, node_memory, node_network, node_ids


def test_replacement_command_widens_refused_instance_by_one_node() -> None:
    """A refused instance re-places one node wider, preserving model + backend (#290)."""
    topology, node_memory, node_network, _ = _fully_connected_three_nodes(
        (10.0, 10.0, 10.0)
    )
    card = ModelCard(
        model_id=ModelId("widen-model"),
        storage_size=Memory.from_gb(6),
        n_layers=12,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )
    two_node = PlaceInstance(
        model_card=card,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=2,
    )
    placed = place_instance(two_node, topology, {}, node_memory, node_network)
    instance = next(iter(placed.values()))
    assert len(instance.shard_assignments.node_to_runner) == 2

    replacement = replacement_command_for_refused_instance(instance)
    assert replacement.min_nodes == 3
    assert replacement.model_card.model_id == card.model_id
    assert replacement.sharding == Sharding.Pipeline
    assert replacement.instance_meta == InstanceMeta.MlxRing


def test_refused_instance_replaces_onto_a_wider_split() -> None:
    """A refused 2-node placement re-places across all three nodes (#290).

    The master picks the smallest fitting cycle, so on a memory view where the
    2-node split looks fine it keeps choosing 2 nodes — exactly the view that
    over-admits the split a worker then refuses on its tighter live reading.
    Bumping ``min_nodes`` to 3 forces the wider floor, so the re-placement lands
    a 3-node split (smaller per-node share) instead of the doomed 2-node one.
    """
    topology, node_memory, node_network, _ = _fully_connected_three_nodes(
        (10.0, 10.0, 10.0)
    )
    card = ModelCard(
        model_id=ModelId("widen-fit-model"),
        storage_size=Memory.from_gb(6),
        n_layers=12,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )
    two_node = PlaceInstance(
        model_card=card,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=2,
    )
    placed = place_instance(two_node, topology, {}, node_memory, node_network)
    refused_instance = next(iter(placed.values()))
    # The same memory view keeps selecting the 2-node split.
    assert len(refused_instance.shard_assignments.node_to_runner) == 2

    replacement = replacement_command_for_refused_instance(refused_instance)
    assert replacement.min_nodes == 3
    widened = place_instance(replacement, topology, {}, node_memory, node_network)
    widened_instance = next(iter(widened.values()))
    assert len(widened_instance.shard_assignments.node_to_runner) == 3


def test_replacement_at_full_cluster_width_is_terminal() -> None:
    """Re-placing a full-width instance raises PlacementError (loop bound) (#290).

    A 3-node instance on a 3-node cluster widens to min_nodes=4, which cannot be
    satisfied — place_instance raises, and the master treats that as the terminal
    "cannot fit anywhere" outcome that stops the refuse→re-place loop.
    """
    topology, node_memory, node_network, _ = _fully_connected_three_nodes(
        (10.0, 10.0, 10.0)
    )
    card = ModelCard(
        model_id=ModelId("full-width-model"),
        storage_size=Memory.from_gb(9),
        n_layers=12,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )
    three_node = PlaceInstance(
        model_card=card,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=3,
    )
    placed = place_instance(three_node, topology, {}, node_memory, node_network)
    instance = next(iter(placed.values()))
    assert len(instance.shard_assignments.node_to_runner) == 3

    replacement = replacement_command_for_refused_instance(instance)
    assert replacement.min_nodes == 4
    with pytest.raises(PlacementError):
        place_instance(replacement, topology, {}, node_memory, node_network)
