# pyright: reportPrivateUsage=false
"""Memory accounting reserves what an engine package said its load takes."""

from skulk.master.main import Master
from skulk.master.placement_utils import (
    reserve_system_ram_usage,
    usable_vram_by_node,
)
from skulk.master.tests.conftest import create_node_memory
from skulk.shared.models.engine_memory_settings import engine_package_memory_settings
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId, SessionId
from skulk.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    LocalForwarderEvent,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import (
    AcceleratorMetrics,
    NodeResources,
    SystemPerformanceProfile,
)
from skulk.shared.types.state import State
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.shared.types.worker.instances import InstanceId, MlxRingInstance
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.channels import channel

GPU_LANE = "tensorfold-cuda"
CPU_LANE = "tensorfold-cpu"
OVERHEAD = 2 * 1024**3
EXTRA = 4 * 1024**3
VRAM_TOTAL = Memory.from_gb(48).in_bytes


def _instance(node: NodeId, runner: RunnerId, backend: str) -> MlxRingInstance:
    card = ModelCard(
        model_id=ModelId("test/text"),
        storage_size=Memory.from_gb(8),
        n_layers=32,
        hidden_size=4096,
        supports_tensor=False,
        num_key_value_heads=8,
        tasks=[ModelTask.TextGeneration],
    )
    packaged = backend in (GPU_LANE, CPU_LANE)
    return MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            node_to_runner={node: runner},
            runner_to_shard={
                runner: PipelineShardMetadata(
                    model_card=card,
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=32,
                    n_layers=32,
                    resolved_backend=backend,
                    memory_settings=engine_package_memory_settings(OVERHEAD)
                    if packaged
                    else None,
                )
            },
        ),
        hosts_by_node={},
        ephemeral_port=50000,
        context_token_limit=8192,
    )


def _gpu() -> SystemPerformanceProfile:
    return SystemPerformanceProfile(
        accelerator=AcceleratorMetrics(
            vendor="nvidia", vram_total_bytes=VRAM_TOTAL, vram_used_bytes=0
        )
    )


def _free_vram(
    node: NodeId,
    instance: MlxRingInstance,
    package_needs: dict[RunnerId, int] | None,
) -> int:
    return usable_vram_by_node(
        {node: _gpu()},
        current_instances={instance.instance_id: instance},
        package_needs=package_needs,
    )[node].in_bytes


def test_a_figure_above_the_estimate_is_what_the_gpu_pool_reserves() -> None:
    """The engine takes more than admission assumed; the next launch sees that."""
    node, runner = NodeId(), RunnerId()
    instance = _instance(node, runner, GPU_LANE)
    estimated_free = _free_vram(node, instance, None)
    estimate = int(VRAM_TOTAL * 0.9) - estimated_free
    assert estimate > 0

    free = _free_vram(node, instance, {runner: estimate + EXTRA})

    assert free == estimated_free - EXTRA


def test_a_figure_at_or_below_the_estimate_never_lowers_the_reservation() -> None:
    node, runner = NodeId(), RunnerId()
    instance = _instance(node, runner, GPU_LANE)
    estimated_free = _free_vram(node, instance, None)

    assert _free_vram(node, instance, {runner: 1}) == estimated_free
    assert _free_vram(node, instance, {runner: 0}) == estimated_free


def test_with_no_figure_the_accounting_is_what_it_was() -> None:
    """No engine package, no difference: built-in runners have no entry."""
    node, runner = NodeId(), RunnerId()
    for backend in (GPU_LANE, "llama_cpp-cuda"):
        instance = _instance(node, runner, backend)
        unchanged = _free_vram(node, instance, None)

        assert _free_vram(node, instance, {}) == unchanged
        # A figure for some other runner is not this shard's.
        assert _free_vram(node, instance, {RunnerId(): VRAM_TOTAL}) == unchanged


def test_a_cpu_lanes_figure_is_reserved_from_system_ram() -> None:
    """A package's CPU lane loads into host memory, so its figure is charged there."""
    node, runner = NodeId(), RunnerId()
    instance = _instance(node, runner, CPU_LANE)
    instances = {instance.instance_id: instance}
    memory = {
        node: create_node_memory(
            Memory.from_gb(60).in_bytes, ram_total=Memory.from_gb(64).in_bytes
        )
    }
    estimated = reserve_system_ram_usage(memory, instances)[node].ram_available
    ceiling_free = reserve_system_ram_usage(memory, {})[node].ram_available
    assert estimated < ceiling_free

    above = reserve_system_ram_usage(
        memory,
        instances,
        package_needs={runner: Memory.from_gb(40).in_bytes},
    )[node].ram_available
    below = reserve_system_ram_usage(
        memory, instances, package_needs={runner: 1}
    )[node].ram_available

    assert above < estimated
    assert below == estimated


def test_a_node_whose_only_gpu_engine_is_a_package_is_a_gpu_memory_node() -> None:
    """Its loads are admitted against VRAM, so the VRAM map must list the node."""
    node = NodeId()
    systems = {node: _gpu()}

    with_package = usable_vram_by_node(
        systems,
        {node: NodeResources(backends=frozenset({"llama_cpp-cpu", GPU_LANE}))},
    )
    cpu_only = usable_vram_by_node(
        systems,
        {node: NodeResources(backends=frozenset({"llama_cpp-cpu", CPU_LANE}))},
    )

    assert node in with_package
    assert node not in cpu_only


def _master(state: State) -> Master:
    return Master(
        NodeId(),
        SessionId(master_node_id=NodeId(), election_clock=1),
        command_receiver=channel[ForwarderCommand]()[1],
        event_sender=channel[Event]()[0],
        local_event_receiver=channel[LocalForwarderEvent]()[1],
        global_event_sender=channel[GlobalForwarderEvent]()[0],
        state_sync_receiver=channel[StateSyncMessage]()[1],
        state_sync_sender=channel[StateSyncMessage]()[0],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
        initial_state=state,
    )


def _master_free_vram(state: State, node: NodeId) -> int:
    master = _master(state)
    master.state = state
    master._telemetry_view.node_system[node] = _gpu()
    master._telemetry_view.node_resources[node] = NodeResources(
        backends=frozenset({GPU_LANE})
    )
    memory = {node: create_node_memory(Memory.from_gb(64).in_bytes)}
    _, vram = master._reserved_placement_inputs(memory, state.instances)
    return vram[node].in_bytes


def test_the_masters_admission_reads_the_figures_from_its_state() -> None:
    """The reservation the master admits against is the one State records."""
    node, runner = NodeId(), RunnerId()
    instance = _instance(node, runner, GPU_LANE)
    placed = State(instances={instance.instance_id: instance})
    estimated_free = _master_free_vram(placed, node)
    estimate = int(VRAM_TOTAL * 0.9) - estimated_free

    stated_free = _master_free_vram(
        placed.model_copy(
            update={"engine_package_needs": {runner: estimate + EXTRA}}
        ),
        node,
    )

    assert stated_free == estimated_free - EXTRA
