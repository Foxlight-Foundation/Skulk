"""Tests for detecting instances wedged by a rank's failed download (#381).

A multi-node instance whose ring forms but where one rank's model download
terminally fails sits at ``RunnerConnected`` forever with nothing to recover it.
``instances_wedged_by_download_failure`` is the master-side detector that finds
exactly that wedge from replicated state so the plan loop can fail and re-place
the instance.
"""

from skulk.master.main import instances_wedged_by_download_failure
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.common import NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadProgress,
)
from skulk.shared.types.worker.instances import (
    Instance,
    InstanceId,
    MlxRingInstance,
    RunnerId,
    ShardAssignments,
)
from skulk.shared.types.worker.runners import (
    RunnerConnected,
    RunnerReady,
    RunnerStatus,
)
from skulk.shared.types.worker.shards import PipelineShardMetadata

_MODEL = ModelId("test-model")


def _model_card(model_id: ModelId = _MODEL) -> ModelCard:
    return ModelCard(
        model_id=model_id,
        n_layers=16,
        storage_size=Memory.from_bytes(1024),
        hidden_size=64,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )


def _shard(card: ModelCard, rank: int, world: int) -> PipelineShardMetadata:
    return PipelineShardMetadata(
        start_layer=0,
        end_layer=16,
        n_layers=16,
        model_card=card,
        device_rank=rank,
        world_size=world,
    )


def _two_node_instance(
    instance_id: InstanceId, node_a: NodeId, node_b: NodeId
) -> tuple[Instance, RunnerId, RunnerId]:
    card = _model_card()
    runner_a, runner_b = RunnerId(), RunnerId()
    instance = MlxRingInstance(
        instance_id=instance_id,
        shard_assignments=ShardAssignments(
            model_id=_MODEL,
            runner_to_shard={
                runner_a: _shard(card, 0, 2),
                runner_b: _shard(card, 1, 2),
            },
            node_to_runner={node_a: runner_a, node_b: runner_b},
        ),
        hosts_by_node={node_a: [], node_b: []},
        ephemeral_port=12345,
    )
    return instance, runner_a, runner_b


def _failed(node_id: NodeId, model_id: ModelId = _MODEL) -> DownloadFailed:
    return DownloadFailed(
        node_id=node_id,
        shard_metadata=_shard(_model_card(model_id), 0, 2),
        error_message="[Errno 28] No space left on device",
    )


def _completed(node_id: NodeId) -> DownloadCompleted:
    return DownloadCompleted(
        node_id=node_id,
        shard_metadata=_shard(_model_card(), 0, 2),
        total=Memory.from_bytes(1024),
    )


def _state(
    instance: Instance,
    runners: dict[RunnerId, RunnerStatus],
    downloads: dict[NodeId, list[DownloadProgress]],
) -> State:
    return State(
        instances={instance.instance_id: instance},
        runners=runners,
        downloads=downloads,
    )


def test_wedge_detected_when_a_rank_download_failed_and_not_ready() -> None:
    iid = InstanceId()
    node_a, node_b = NodeId("a"), NodeId("b")
    instance, runner_a, runner_b = _two_node_instance(iid, node_a, node_b)
    state = _state(
        instance,
        {runner_a: RunnerConnected(), runner_b: RunnerConnected()},
        {node_a: [_failed(node_a)], node_b: [_completed(node_b)]},
    )

    wedged = instances_wedged_by_download_failure(state)

    assert set(wedged) == {iid}
    failed_nodes, cause = wedged[iid]
    assert failed_nodes == frozenset({node_a})
    assert "No space left on device" in cause


def test_ready_instance_never_reported_even_with_stale_failure() -> None:
    # A serving instance must never be torn down by this path, even if a stale
    # DownloadFailed lingers in state.
    iid = InstanceId()
    node_a, node_b = NodeId("a"), NodeId("b")
    instance, runner_a, runner_b = _two_node_instance(iid, node_a, node_b)
    state = _state(
        instance,
        {runner_a: RunnerReady(), runner_b: RunnerReady()},
        {node_a: [_failed(node_a)], node_b: [_completed(node_b)]},
    )

    assert instances_wedged_by_download_failure(state) == {}


def test_not_ready_without_any_failure_is_not_wedged() -> None:
    # Still legitimately loading: connected, downloads completed, no failure.
    iid = InstanceId()
    node_a, node_b = NodeId("a"), NodeId("b")
    instance, runner_a, runner_b = _two_node_instance(iid, node_a, node_b)
    state = _state(
        instance,
        {runner_a: RunnerConnected(), runner_b: RunnerConnected()},
        {node_a: [_completed(node_a)], node_b: [_completed(node_b)]},
    )

    assert instances_wedged_by_download_failure(state) == {}


def test_failure_for_a_different_model_is_ignored() -> None:
    iid = InstanceId()
    node_a, node_b = NodeId("a"), NodeId("b")
    instance, runner_a, runner_b = _two_node_instance(iid, node_a, node_b)
    state = _state(
        instance,
        {runner_a: RunnerConnected(), runner_b: RunnerConnected()},
        {node_a: [_failed(node_a, ModelId("other-model"))], node_b: []},
    )

    assert instances_wedged_by_download_failure(state) == {}


def test_multiple_failed_ranks_all_reported() -> None:
    iid = InstanceId()
    node_a, node_b = NodeId("a"), NodeId("b")
    instance, runner_a, runner_b = _two_node_instance(iid, node_a, node_b)
    state = _state(
        instance,
        {runner_a: RunnerConnected(), runner_b: RunnerConnected()},
        {node_a: [_failed(node_a)], node_b: [_failed(node_b)]},
    )

    failed_nodes, _cause = instances_wedged_by_download_failure(state)[iid]
    assert failed_nodes == frozenset({node_a, node_b})
