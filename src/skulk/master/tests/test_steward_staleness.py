"""Operator-readable reasons a steward proposal no longer fits the cluster."""

from datetime import datetime, timedelta, timezone

from skulk.master.steward_staleness import (
    describe_instance_drift,
    download_target_staleness,
    lifecycle_target_staleness,
    placement_card_staleness,
    steward_proposal_staleness,
)
from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.types.common import CommandId, ModelId, NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.steward_actions import (
    StewardActionProposal,
    StewardBasicAction,
    StewardCancelDownloadAction,
    StewardPlaceModelAction,
    StewardRestartInstanceAction,
    StewardStopInstanceAction,
)
from skulk.shared.types.worker.downloads import (
    DownloadAttemptId,
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
    DownloadProgressData,
)
from skulk.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceMeta,
    MlxRingInstance,
    ShardAssignments,
)
from skulk.shared.types.worker.runners import RunnerId
from skulk.shared.types.worker.shards import PipelineShardMetadata, Sharding

MODEL = ModelId("org/model")
NODE_NAMES = {NodeId("node-a"): "kite1", NodeId("node-b"): "kite2"}


def _card(storage_gb: int = 8, model_id: ModelId = MODEL) -> ModelCard:
    return ModelCard(
        model_id=model_id,
        storage_size=Memory.from_gb(storage_gb),
        n_layers=4,
        hidden_size=8,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )


def _instance(
    *,
    node: str = "node-a",
    runner: str = "runner-a",
    card: ModelCard | None = None,
    end_layer: int = 4,
    port: int = 52415,
    context_token_limit: int | None = None,
    instance_id: str = "instance",
) -> MlxRingInstance:
    shard_card = card or _card()
    return MlxRingInstance(
        instance_id=InstanceId(instance_id),
        shard_assignments=ShardAssignments(
            model_id=shard_card.model_id,
            runner_to_shard={
                RunnerId(runner): PipelineShardMetadata(
                    model_card=shard_card,
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=end_layer,
                    n_layers=4,
                )
            },
            node_to_runner={NodeId(node): RunnerId(runner)},
        ),
        hosts_by_node={NodeId(node): []},
        ephemeral_port=port,
        context_token_limit=context_token_limit,
    )


def _proposal(action: StewardBasicAction, **update: object) -> StewardActionProposal:
    now = datetime.now(tz=timezone.utc)
    proposal = StewardActionProposal(
        action=action,
        rationale="The instance needs attention.",
        evidence=("Observed by the steward.",),
        expected_effect="Apply the reviewed action.",
        created_at=now,
        expires_at=now + timedelta(minutes=10),
    )
    return proposal.model_copy(update=update) if update else proposal


def _lifecycle(
    action: StewardStopInstanceAction | StewardRestartInstanceAction,
    instances: dict[InstanceId, Instance],
    *,
    others: tuple[StewardActionProposal, ...] = (),
    catalog_card: ModelCard | None = None,
) -> str | None:
    proposal = _proposal(action)
    return lifecycle_target_staleness(
        proposal,
        action,
        instances=instances,
        node_names=NODE_NAMES,
        other_proposals=(proposal, *others),
        ordered_card=lambda _model_id: catalog_card or _card(),
    )


def test_unchanged_instance_has_no_drift() -> None:
    reviewed = _instance()
    assert describe_instance_drift(reviewed, reviewed, node_names=NODE_NAMES) is None


def test_drift_names_the_nodes_an_instance_moved_between() -> None:
    reason = describe_instance_drift(
        _instance(), _instance(node="node-b"), node_names=NODE_NAMES
    )
    assert reason == "The instance changed after review: it now runs on kite2 instead of kite1."


def test_drift_lists_every_operator_visible_difference() -> None:
    reason = describe_instance_drift(
        _instance(),
        _instance(
            runner="runner-new",
            card=_card(storage_gb=9),
            end_layer=3,
            context_token_limit=4096,
        ),
        node_names=NODE_NAMES,
    )
    assert reason == (
        "The instance changed after review: its runners were replaced; its model "
        "card changed; its layer split changed; its context window changed."
    )


def test_internal_wiring_change_reads_as_a_placement_detail() -> None:
    reason = describe_instance_drift(
        _instance(), _instance(port=52416), node_names=NODE_NAMES
    )
    assert reason == "The instance changed after review: its placement details changed."


def test_stop_target_replaced_by_a_new_instance_names_where_it_runs() -> None:
    successor = _instance(instance_id="successor", node="node-b")
    reason = _lifecycle(
        StewardStopInstanceAction(instance=_instance()),
        {successor.instance_id: successor},
    )
    assert reason == (
        "The instance was replaced after review: org/model now runs as a new "
        "instance on kite2."
    )


def test_stop_target_removed_without_successor() -> None:
    reason = _lifecycle(StewardStopInstanceAction(instance=_instance()), {})
    assert reason == (
        "The instance no longer exists: org/model was stopped or removed after review."
    )


def test_target_that_now_serves_another_model() -> None:
    reviewed = _instance()
    current = _instance(card=_card(model_id=ModelId("org/other")))
    reason = _lifecycle(
        StewardStopInstanceAction(instance=reviewed), {current.instance_id: current}
    )
    assert reason == "The instance now serves org/other instead of org/model."


def test_another_approved_action_owning_the_instance_is_named() -> None:
    instance = _instance()
    owner = _proposal(
        StewardRestartInstanceAction(instance=instance),
        status="approved",
        decided_at=datetime.now(tz=timezone.utc),
        decided_by="trusted_fabric_operator",
        command_id=CommandId("owner"),
    )
    reason = _lifecycle(
        StewardStopInstanceAction(instance=instance),
        {instance.instance_id: instance},
        others=(owner,),
    )
    assert reason == "Another approved steward action already owns this instance."


def test_unchanged_stop_target_still_fits() -> None:
    instance = _instance()
    assert (
        _lifecycle(
            StewardStopInstanceAction(instance=instance),
            {instance.instance_id: instance},
            catalog_card=_card(storage_gb=9),
        )
        is None
    )


def test_restart_needs_the_reviewed_card_to_still_be_the_catalog_card() -> None:
    instance = _instance()
    reason = _lifecycle(
        StewardRestartInstanceAction(instance=instance),
        {instance.instance_id: instance},
        catalog_card=_card(storage_gb=9),
    )
    assert reason == (
        "The model card for org/model changed after review, so a restart would "
        "not bring back the instance reviewed."
    )


def test_restart_of_a_model_removed_from_the_catalog() -> None:
    instance = _instance()
    action = StewardRestartInstanceAction(instance=instance)
    proposal = _proposal(action)
    reason = lifecycle_target_staleness(
        proposal,
        action,
        instances={instance.instance_id: instance},
        node_names=NODE_NAMES,
        other_proposals=(proposal,),
        ordered_card=lambda _model_id: None,
    )
    assert reason == (
        "org/model was removed from the model catalog after review, so a restart "
        "could not place it again."
    )


def _cancel(attempt: str = "attempt") -> StewardCancelDownloadAction:
    return StewardCancelDownloadAction(
        node_id=NodeId("node-a"),
        node_name="kite1",
        model_id=MODEL,
        attempt_id=DownloadAttemptId(attempt),
    )


def _shard() -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=_card(),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=4,
        n_layers=4,
    )


def _downloads(*progress: DownloadProgress) -> dict[NodeId, list[DownloadProgress]]:
    return {NodeId("node-a"): list(progress)}


def test_reviewed_download_attempt_still_active() -> None:
    downloads = _downloads(
        DownloadOngoing(
            node_id=NodeId("node-a"),
            shard_metadata=_shard(),
            attempt_id=DownloadAttemptId("attempt"),
            download_progress=DownloadProgressData(
                total=Memory.from_gb(8),
                downloaded=Memory.from_gb(2),
                downloaded_this_session=Memory.from_gb(2),
                completed_files=0,
                total_files=1,
                speed=0.0,
                eta_ms=0,
                files={},
            ),
        )
    )
    assert download_target_staleness(_cancel(), downloads=downloads) is None


def test_newer_download_attempt_is_not_the_one_reviewed() -> None:
    downloads = _downloads(
        DownloadPending(
            node_id=NodeId("node-a"),
            shard_metadata=_shard(),
            attempt_id=DownloadAttemptId("retry"),
        )
    )
    assert download_target_staleness(_cancel(), downloads=downloads) == (
        "A newer download attempt replaced the one reviewed, and this proposal "
        "would not cancel it."
    )


def test_finished_and_failed_downloads_say_so() -> None:
    finished = _downloads(
        DownloadCompleted(
            node_id=NodeId("node-a"),
            shard_metadata=_shard(),
            attempt_id=DownloadAttemptId("attempt"),
            total=Memory.from_gb(8),
        )
    )
    failed = _downloads(
        DownloadFailed(
            node_id=NodeId("node-a"),
            shard_metadata=_shard(),
            attempt_id=DownloadAttemptId("attempt"),
            error_message="disk full",
        )
    )
    assert download_target_staleness(_cancel(), downloads=finished) == (
        "The download of org/model finished after review."
    )
    assert download_target_staleness(_cancel(), downloads=failed) == (
        "The download of org/model already failed after review."
    )
    assert download_target_staleness(_cancel(), downloads={}) == (
        "The proposed download is no longer active."
    )


def test_an_unseen_attempt_is_stale_only_when_absence_counts() -> None:
    """Live downloads are telemetry: a master change clears them for a while."""
    assert download_target_staleness(_cancel(), downloads={}, absence_is_evidence=False) is None


def test_an_earlier_attempts_completion_is_not_this_attempts() -> None:
    earlier = _downloads(
        DownloadCompleted(
            node_id=NodeId("node-a"),
            shard_metadata=_shard(),
            attempt_id=DownloadAttemptId("earlier"),
            total=Memory.from_gb(8),
        )
    )
    assert download_target_staleness(_cancel(), downloads=earlier) == (
        "The proposed download is no longer active."
    )
    assert (
        download_target_staleness(_cancel(), downloads=earlier, absence_is_evidence=False)
        is None
    )


def test_proven_outcomes_are_stale_even_when_absence_does_not_count() -> None:
    newer = _downloads(
        DownloadPending(
            node_id=NodeId("node-a"),
            shard_metadata=_shard(),
            attempt_id=DownloadAttemptId("retry"),
        )
    )
    failed = _downloads(
        DownloadFailed(
            node_id=NodeId("node-a"),
            shard_metadata=_shard(),
            attempt_id=DownloadAttemptId("attempt"),
            error_message="disk full",
        )
    )
    assert download_target_staleness(_cancel(), downloads=newer, absence_is_evidence=False)
    assert download_target_staleness(_cancel(), downloads=failed, absence_is_evidence=False) == (
        "The download of org/model already failed after review."
    )


def _place(card: ModelCard) -> StewardPlaceModelAction:
    return StewardPlaceModelAction(
        model_card=card,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
    )


def test_placement_card_staleness() -> None:
    assert placement_card_staleness(_place(_card()), ordered_card=_card()) is None
    assert placement_card_staleness(
        _place(_card()), ordered_card=_card(storage_gb=9)
    ) == "The model card for org/model changed after review."
    assert placement_card_staleness(_place(_card()), ordered_card=None) == (
        "org/model was removed from the model catalog after review."
    )


def test_dispatcher_routes_each_action_to_its_check() -> None:
    def check(action: StewardBasicAction) -> str | None:
        return steward_proposal_staleness(
            _proposal(action),
            instances={},
            downloads={},
            node_names=NODE_NAMES,
            ordered_card=lambda _model_id: _card(storage_gb=9),
            other_proposals=(),
        )

    assert check(_place(_card())) == "The model card for org/model changed after review."
    assert check(_cancel()) == "The proposed download is no longer active."
    assert check(StewardStopInstanceAction(instance=_instance())) == (
        "The instance no longer exists: org/model was stopped or removed after review."
    )
