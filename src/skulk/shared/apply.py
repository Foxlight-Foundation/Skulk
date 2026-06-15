import copy
from collections.abc import Mapping, Sequence
from datetime import datetime

from loguru import logger

from skulk.shared.types.common import NodeId
from skulk.shared.types.events import (
    ChunkGenerated,
    CustomModelCardAdded,
    CustomModelCardDeleted,
    Event,
    IndexedEvent,
    InputChunkReceived,
    InstanceCreated,
    InstanceDeleted,
    NodeDownloadProgress,
    NodeGatheredInfo,
    NodeTimedOut,
    RunnerStatusUpdated,
    StateSnapshotHydrated,
    TaskAcknowledged,
    TaskCreated,
    TaskDeleted,
    TaskFailed,
    TaskStatusUpdated,
    TestEvent,
    TopologyEdgeCreated,
    TopologyEdgeDeleted,
    TracesCollected,
    TracesMerged,
    TracingStateChanged,
)
from skulk.shared.types.profiling import (
    NodeNetworkInfo,
    NodeResources,
    NodeThunderboltInfo,
    ThunderboltBridgeStatus,
)
from skulk.shared.types.state import State
from skulk.shared.types.tasks import Task, TaskId, TaskStatus
from skulk.shared.types.topology import Connection, RDMAConnection
from skulk.shared.types.worker.downloads import DownloadProgress
from skulk.shared.types.worker.instances import Instance, InstanceId
from skulk.shared.types.worker.runners import RunnerId, RunnerShutdown, RunnerStatus
from skulk.utils.info_gatherer.info_gatherer import (
    MacmonMetrics,
    MacThunderboltConnections,
    MacThunderboltIdentifiers,
    MactopMetrics,
    MemoryUsage,
    MiscData,
    NodeConfig,
    NodeDiskUsage,
    NodeNetworkInterfaces,
    RdmaCtlStatus,
    StaticNodeInformation,
    ThunderboltBridgeInfo,
)


def event_apply(event: Event, state: State) -> State:
    """Apply an event to state."""
    match event:
        case (
            TestEvent()
            | ChunkGenerated()
            | TaskAcknowledged()
            | InputChunkReceived()
            | TracesCollected()
            | TracesMerged()
            | CustomModelCardAdded()
            | CustomModelCardDeleted()
        ):  # Pass-through events that don't modify state
            return state
        case InstanceCreated():
            return apply_instance_created(event, state)
        case InstanceDeleted():
            return apply_instance_deleted(event, state)
        case NodeTimedOut():
            return apply_node_timed_out(event, state)
        case NodeDownloadProgress():
            return apply_node_download_progress(event, state)
        case NodeGatheredInfo():
            return apply_node_gathered_info(event, state)
        case RunnerStatusUpdated():
            return apply_runner_status_updated(event, state)
        case TaskCreated():
            return apply_task_created(event, state)
        case TaskDeleted():
            return apply_task_deleted(event, state)
        case TaskFailed():
            return apply_task_failed(event, state)
        case TaskStatusUpdated():
            return apply_task_status_updated(event, state)
        case TopologyEdgeCreated():
            return apply_topology_edge_created(event, state)
        case TopologyEdgeDeleted():
            return apply_topology_edge_deleted(event, state)
        case TracingStateChanged():
            return state.model_copy(update={"tracing_enabled": event.enabled})
        case StateSnapshotHydrated():
            return event.state


def apply(state: State, event: IndexedEvent) -> State:
    if isinstance(event.event, StateSnapshotHydrated):
        assert event.event.state.last_event_applied_idx == event.idx
        return event.event.state

    # Just to test that events are only applied in correct order
    if state.last_event_applied_idx != event.idx - 1:
        logger.warning(
            f"Expected event {state.last_event_applied_idx + 1} but received {event.idx}"
        )
    assert state.last_event_applied_idx == event.idx - 1
    new_state: State = event_apply(event.event, state)
    return new_state.model_copy(update={"last_event_applied_idx": event.idx})


def apply_node_download_progress(event: NodeDownloadProgress, state: State) -> State:
    """
    Update or add a node download progress to state.
    """
    dp = event.download_progress
    node_id = dp.node_id

    current = list(state.downloads.get(node_id, ()))

    replaced = False
    for i, existing_dp in enumerate(current):
        # TODO(ciaran): deduplicate by model_id for now. Will need to use
        # shard_metadata again when pipeline and tensor downloads differ.
        # For now this is fine
        if (
            existing_dp.shard_metadata.model_card.model_id
            == dp.shard_metadata.model_card.model_id
        ):
            current[i] = dp
            replaced = True
            break

    if not replaced:
        current.append(dp)

    new_downloads: Mapping[NodeId, Sequence[DownloadProgress]] = {
        **state.downloads,
        node_id: current,
    }
    return state.model_copy(update={"downloads": new_downloads})


def apply_task_created(event: TaskCreated, state: State) -> State:
    new_tasks: Mapping[TaskId, Task] = {**state.tasks, event.task_id: event.task}
    return state.model_copy(update={"tasks": new_tasks})


def apply_task_deleted(event: TaskDeleted, state: State) -> State:
    new_tasks: Mapping[TaskId, Task] = {
        tid: task for tid, task in state.tasks.items() if tid != event.task_id
    }
    return state.model_copy(update={"tasks": new_tasks})


def apply_task_status_updated(event: TaskStatusUpdated, state: State) -> State:
    if event.task_id not in state.tasks:
        # maybe should raise
        return state

    update: dict[str, TaskStatus | None] = {
        "task_status": event.task_status,
    }
    if event.task_status != TaskStatus.Failed:
        update["error_type"] = None
        update["error_message"] = None

    updated_task = state.tasks[event.task_id].model_copy(update=update)
    new_tasks: Mapping[TaskId, Task] = {**state.tasks, event.task_id: updated_task}
    return state.model_copy(update={"tasks": new_tasks})


def apply_task_failed(event: TaskFailed, state: State) -> State:
    if event.task_id not in state.tasks:
        # maybe should raise
        return state

    # Failed is terminal: the runner supervisor's terminal-status set and the
    # master's orphaned-task sweep both rely on a failed task not being
    # re-processed — without the status flip the sweep would re-emit
    # TaskFailed for a lingering task every plan pass.
    updated_task = state.tasks[event.task_id].model_copy(
        update={
            "task_status": TaskStatus.Failed,
            "error_type": event.error_type,
            "error_message": event.error_message,
        }
    )
    new_tasks: Mapping[TaskId, Task] = {**state.tasks, event.task_id: updated_task}
    return state.model_copy(update={"tasks": new_tasks})


def apply_instance_created(event: InstanceCreated, state: State) -> State:
    instance = event.instance
    new_instances: Mapping[InstanceId, Instance] = {
        **state.instances,
        instance.instance_id: instance,
    }
    return state.model_copy(update={"instances": new_instances})


def apply_instance_deleted(event: InstanceDeleted, state: State) -> State:
    new_instances: Mapping[InstanceId, Instance] = {
        iid: inst for iid, inst in state.instances.items() if iid != event.instance_id
    }
    # Drop the deleted instance's runner records too. Runner records are
    # otherwise only removed by a terminal RunnerStatusUpdated(RunnerShutdown),
    # but that final status is unreliably delivered: the worker's Shutdown
    # handler cancels the supervisor's event forwarder (runner.shutdown()) as
    # soon as the Shutdown task completes/times out, often before the runner
    # process's RunnerShutdown is forwarded — and on a master-failover teardown
    # the forwarder is torn down outright. Every instance delete therefore
    # leaked one RunnerShuttingDown record per rank, growing State.runners
    # without bound. Cleaning them here makes deletion atomic and independent of
    # that handshake (mirrors apply_node_timed_out, which already prunes an
    # affected instance's runners). The actual process teardown is driven
    # separately by the Shutdown task, so dropping the status record early is
    # safe.
    deleted = state.instances.get(event.instance_id)
    update: dict[str, object] = {"instances": new_instances}
    if deleted is not None:
        doomed_runner_ids = set(deleted.shard_assignments.runner_to_shard.keys())
        if doomed_runner_ids:
            update["runners"] = {
                rid: rs
                for rid, rs in state.runners.items()
                if rid not in doomed_runner_ids
            }
    return state.model_copy(update=update)


def apply_runner_status_updated(event: RunnerStatusUpdated, state: State) -> State:
    if isinstance(event.runner_status, RunnerShutdown):
        new_runners: Mapping[RunnerId, RunnerStatus] = {
            rid: rs for rid, rs in state.runners.items() if rid != event.runner_id
        }
        return state.model_copy(update={"runners": new_runners})
    # Don't resurrect a record for a runner that no longer belongs to any
    # instance. During teardown the worker emits a RunnerShuttingDown status that
    # races behind InstanceDeleted; without this guard it re-adds a runner record
    # that then never receives its terminal RunnerShutdown (that final status is
    # routinely lost when the supervisor's event forwarder is cancelled on
    # shutdown), so State.runners grows without bound. A status update for a live
    # runner always has its instance present (CreateRunner is planned only after
    # the instance exists, and events apply in order), so this only drops the
    # post-deletion stragglers.
    runner_belongs_to_an_instance = any(
        event.runner_id in instance.shard_assignments.runner_to_shard
        for instance in state.instances.values()
    )
    if not runner_belongs_to_an_instance:
        return state
    new_runners = {
        **state.runners,
        event.runner_id: event.runner_status,
    }
    return state.model_copy(update={"runners": new_runners})


def apply_node_timed_out(event: NodeTimedOut, state: State) -> State:
    topology = copy.deepcopy(state.topology)
    topology.remove_node(event.node_id)
    affected_instance_ids = {
        instance_id
        for instance_id, instance in state.instances.items()
        if event.node_id in instance.shard_assignments.node_to_runner
    }
    affected_runner_ids = {
        runner_id
        for instance_id in affected_instance_ids
        for runner_id in state.instances[instance_id].shard_assignments.runner_to_shard
    }
    instances = {
        instance_id: instance
        for instance_id, instance in state.instances.items()
        if instance_id not in affected_instance_ids
    }
    runners = {
        runner_id: runner_status
        for runner_id, runner_status in state.runners.items()
        if runner_id not in affected_runner_ids
    }
    tasks = {
        task_id: task
        for task_id, task in state.tasks.items()
        if task.instance_id not in affected_instance_ids
    }
    last_seen = {
        key: value for key, value in state.last_seen.items() if key != event.node_id
    }
    downloads = {
        key: value for key, value in state.downloads.items() if key != event.node_id
    }
    # Clean up the connectivity mappings still held in State. The telemetry-plane
    # readings (node_memory/node_system since slice 2; node_identities/node_disk/
    # node_rdma_ctl since slice 3) are pruned from TelemetryView via
    # record_membership_from_event, not here.
    node_network = {
        key: value for key, value in state.node_network.items() if key != event.node_id
    }
    node_thunderbolt = {
        key: value
        for key, value in state.node_thunderbolt.items()
        if key != event.node_id
    }
    node_thunderbolt_bridge = {
        key: value
        for key, value in state.node_thunderbolt_bridge.items()
        if key != event.node_id
    }
    # Only recompute cycles if the leaving node had TB bridge enabled
    leaving_node_status = state.node_thunderbolt_bridge.get(event.node_id)
    leaving_node_had_tb_enabled = (
        leaving_node_status is not None and leaving_node_status.enabled
    )
    thunderbolt_bridge_cycles = (
        topology.get_thunderbolt_bridge_cycles(node_thunderbolt_bridge, node_network)
        if leaving_node_had_tb_enabled
        else [list(cycle) for cycle in state.thunderbolt_bridge_cycles]
    )
    return state.model_copy(
        update={
            "instances": instances,
            "runners": runners,
            "tasks": tasks,
            "downloads": downloads,
            "topology": topology,
            "last_seen": last_seen,
            "node_network": node_network,
            "node_thunderbolt": node_thunderbolt,
            "node_thunderbolt_bridge": node_thunderbolt_bridge,
            "thunderbolt_bridge_cycles": thunderbolt_bridge_cycles,
        }
    )


def apply_node_gathered_info(event: NodeGatheredInfo, state: State) -> State:
    topology = copy.deepcopy(state.topology)
    topology.add_node(event.node_id)
    info = event.info

    # Build update dict with only the mappings that change
    update: dict[str, object] = {
        "last_seen": {
            **state.last_seen,
            event.node_id: datetime.fromisoformat(event.when),
        },
        "topology": topology,
    }

    match info:
        # Memory and the system profile moved to the telemetry plane (#279
        # slice 2): workers now gossip them on the TELEMETRY topic into the
        # TelemetryView, off the event log. These cases stay only to keep the
        # match exhaustive and to no-op a legacy event from an un-upgraded
        # worker during a rolling upgrade (its readings ride telemetry once it
        # restarts on the new binary). last_seen is still bumped above.
        case MactopMetrics() | MacmonMetrics():
            pass
        case MemoryUsage():
            pass
        case NodeDiskUsage():
            # Telemetry plane since #279 slice 3 — applied to TelemetryView, not
            # State. A NodeGatheredInfo still bumps last_seen above, but the
            # reading no longer rides the event log (workers fork it to the
            # TELEMETRY topic; a legacy event from an un-upgraded worker no-ops).
            pass
        case NodeConfig():
            pass
        case MiscData():
            # Telemetry plane since #279 slice 3 (identity friendly-name).
            pass
        case StaticNodeInformation():
            # Telemetry plane since #279 slice 3 (identity static info).
            pass
        case NodeNetworkInterfaces():
            update["node_network"] = {
                **state.node_network,
                event.node_id: NodeNetworkInfo(interfaces=info.ifaces),
            }
        case MacThunderboltIdentifiers():
            update["node_thunderbolt"] = {
                **state.node_thunderbolt,
                event.node_id: NodeThunderboltInfo(interfaces=info.idents),
            }
        case MacThunderboltConnections():
            conn_map = {
                tb_ident.domain_uuid: (nid, tb_ident.rdma_interface)
                for nid in state.node_thunderbolt
                for tb_ident in state.node_thunderbolt[nid].interfaces
            }
            as_rdma_conns = [
                Connection(
                    source=event.node_id,
                    sink=conn_map[tb_conn.sink_uuid][0],
                    edge=RDMAConnection(
                        source_rdma_iface=conn_map[tb_conn.source_uuid][1],
                        sink_rdma_iface=conn_map[tb_conn.sink_uuid][1],
                    ),
                )
                for tb_conn in info.conns
                if tb_conn.source_uuid in conn_map
                if tb_conn.sink_uuid in conn_map
            ]
            topology.replace_all_out_rdma_connections(event.node_id, as_rdma_conns)
        case ThunderboltBridgeInfo():
            new_tb_bridge: dict[NodeId, ThunderboltBridgeStatus] = {
                **state.node_thunderbolt_bridge,
                event.node_id: info.status,
            }
            update["node_thunderbolt_bridge"] = new_tb_bridge
            # Only recompute cycles if the enabled status changed
            old_status = state.node_thunderbolt_bridge.get(event.node_id)
            old_enabled = old_status.enabled if old_status else False
            new_enabled = info.status.enabled
            if old_enabled != new_enabled:
                update["thunderbolt_bridge_cycles"] = (
                    topology.get_thunderbolt_bridge_cycles(
                        new_tb_bridge, state.node_network
                    )
                )
        case RdmaCtlStatus():
            # Telemetry plane since #279 slice 3 (rdma-ctl status).
            pass
        case NodeResources():
            # NodeResources travels the telemetry plane (#279), not the event
            # log — the worker routes it to the TELEMETRY topic and it lands in
            # TelemetryView, never here. Kept as an explicit no-op so the match
            # over GatheredInfo stays exhaustive and a stray log-path delivery
            # is harmless rather than a crash.
            pass

    return state.model_copy(update=update)


def apply_topology_edge_created(event: TopologyEdgeCreated, state: State) -> State:
    topology = copy.deepcopy(state.topology)
    topology.add_connection(event.conn)
    return state.model_copy(update={"topology": topology})


def apply_topology_edge_deleted(event: TopologyEdgeDeleted, state: State) -> State:
    topology = copy.deepcopy(state.topology)
    topology.remove_connection(event.conn)
    # TODO: Clean up removing the reverse connection
    return state.model_copy(update={"topology": topology})
