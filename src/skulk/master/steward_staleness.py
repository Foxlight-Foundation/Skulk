"""Plain-language reasons a steward proposal no longer fits the cluster.

A steward proposal records the exact target the operator reviews: an instance
for stop and restart, a download attempt for cancellation, and a model card for
placement. The cluster keeps moving while a proposal waits, so the master
checks the reviewed target against live truth twice: continuously while the
proposal is pending, to mark it out of date before anyone acts on it, and again
at approval. Both checks share these functions so the operator reads the same
reason either way.

Every function here is pure: callers pass the current instances, downloads,
node names, and an injected card lookup, and receive either ``None`` (the
proposal still fits) or one operator-readable sentence.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence

from skulk.shared.models.model_cards import ModelCard, same_authorized_model_card
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.steward_actions import (
    StewardActionProposal,
    StewardCancelDownloadAction,
    StewardPlaceModelAction,
    StewardRestartInstanceAction,
    StewardStopInstanceAction,
)
from skulk.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
)
from skulk.shared.types.worker.instances import Instance, InstanceId


def _node_list(node_ids: Iterable[NodeId], node_names: Mapping[NodeId, str]) -> str:
    """Name nodes by their friendly names, in a stable order."""
    names = sorted(node_names.get(node_id, str(node_id)) for node_id in node_ids)
    return ", ".join(names) if names else "no nodes"


def _instance_card(instance: Instance) -> ModelCard | None:
    """Return the model card the instance's shards were placed with."""
    shard = next(iter(instance.shard_assignments.runner_to_shard.values()), None)
    return None if shard is None else shard.model_card


def _layer_layout(instance: Instance) -> dict[NodeId, tuple[int, int]]:
    """Map each node to the layer range its runner serves."""
    layout: dict[NodeId, tuple[int, int]] = {}
    for node_id, runner_id in instance.shard_assignments.node_to_runner.items():
        shard = instance.shard_assignments.runner_to_shard.get(runner_id)
        if shard is not None:
            layout[node_id] = (shard.start_layer, shard.end_layer)
    return layout


def describe_instance_drift(
    reviewed: Instance,
    current: Instance,
    *,
    node_names: Mapping[NodeId, str],
) -> str | None:
    """Explain how an instance differs from the one an operator reviewed.

    Args:
        reviewed: Instance recorded in the proposal.
        current: Live instance with the same identity.
        node_names: Friendly node names for readable placement changes.

    Returns:
        ``None`` when the instance is unchanged, otherwise one sentence listing
        every difference the operator would care about.
    """
    if current == reviewed:
        return None
    differences: list[str] = []
    if type(current) is not type(reviewed):
        differences.append("its serving topology changed")
    reviewed_nodes = set(reviewed.shard_assignments.node_to_runner)
    current_nodes = set(current.shard_assignments.node_to_runner)
    if current_nodes != reviewed_nodes:
        differences.append(
            f"it now runs on {_node_list(current_nodes, node_names)} instead of "
            f"{_node_list(reviewed_nodes, node_names)}"
        )
    elif dict(current.shard_assignments.node_to_runner) != dict(
        reviewed.shard_assignments.node_to_runner
    ):
        differences.append("its runners were replaced")
    reviewed_card = _instance_card(reviewed)
    current_card = _instance_card(current)
    if (
        reviewed_card is not None
        and current_card is not None
        and not same_authorized_model_card(reviewed_card, current_card)
    ):
        differences.append("its model card changed")
    if current_nodes == reviewed_nodes and _layer_layout(current) != _layer_layout(
        reviewed
    ):
        differences.append("its layer split changed")
    if (
        current.context_token_limit != reviewed.context_token_limit
        or current.requested_context_tokens != reviewed.requested_context_tokens
    ):
        differences.append("its context window changed")
    if sorted(current.excluded_nodes) != sorted(reviewed.excluded_nodes):
        differences.append("its excluded nodes changed")
    if not differences:
        # Hosts and ports are internal wiring; the operator only needs to know
        # the instance is no longer the one they reviewed.
        differences.append("its placement details changed")
    return f"The instance changed after review: {'; '.join(differences)}."


def lifecycle_target_staleness(
    proposal: StewardActionProposal,
    action: StewardStopInstanceAction | StewardRestartInstanceAction,
    *,
    instances: Mapping[InstanceId, Instance],
    node_names: Mapping[NodeId, str],
    other_proposals: Iterable[StewardActionProposal],
    ordered_card: Callable[[ModelId], ModelCard | None],
) -> str | None:
    """Return why a stop or restart no longer applies to its reviewed instance.

    Args:
        proposal: The proposal being checked, excluded from ownership checks.
        action: The stop or restart action it carries.
        instances: Live instances by identity.
        node_names: Friendly node names for readable placement changes.
        other_proposals: Every retained proposal, used to detect an approved
            stop or restart that already owns the same instance.
        ordered_card: Injected lookup of the master's command-ordered card. A
            restart places the reviewed card again, so it must still be the
            catalog's card; a stop does not consult it.

    Returns:
        ``None`` while the action still fits, otherwise the reason it does not.
    """
    reviewed = action.instance
    model_id = reviewed.shard_assignments.model_id
    current = instances.get(reviewed.instance_id)
    if current is None:
        successor = next(
            (
                instance
                for instance in instances.values()
                if instance.shard_assignments.model_id == model_id
                and instance.system_role is None
            ),
            None,
        )
        if successor is not None:
            nodes = _node_list(successor.shard_assignments.node_to_runner, node_names)
            return (
                f"The instance was replaced after review: {model_id} now runs "
                f"as a new instance on {nodes}."
            )
        return (
            f"The instance no longer exists: {model_id} was stopped or removed "
            "after review."
        )
    if current.system_role is not None:
        return "System placements cannot be changed by steward actions."
    if current.shard_assignments.model_id != model_id:
        return (
            f"The instance now serves {current.shard_assignments.model_id} "
            f"instead of {model_id}."
        )
    drift = describe_instance_drift(reviewed, current, node_names=node_names)
    if drift is not None:
        return drift
    if any(
        other.proposal_id != proposal.proposal_id
        and other.status in {"approved", "dispatched"}
        and isinstance(
            other.action, (StewardStopInstanceAction, StewardRestartInstanceAction)
        )
        and other.action.instance.instance_id == reviewed.instance_id
        for other in other_proposals
    ):
        return "Another approved steward action already owns this instance."
    if isinstance(action, StewardRestartInstanceAction):
        catalog_card = ordered_card(model_id)
        if catalog_card is None:
            return (
                f"{model_id} was removed from the model catalog after review, "
                "so a restart could not place it again."
            )
        reviewed_card = _instance_card(reviewed)
        if reviewed_card is not None and not same_authorized_model_card(
            reviewed_card, catalog_card
        ):
            return (
                f"The model card for {model_id} changed after review, so a "
                "restart would not bring back the instance reviewed."
            )
    return None


def download_target_staleness(
    action: StewardCancelDownloadAction,
    *,
    downloads: Mapping[NodeId, Sequence[DownloadProgress]],
) -> str | None:
    """Return why a download cancellation no longer applies to its attempt.

    Args:
        action: Cancellation recording the exact attempt the operator reviewed.
        downloads: Effective download truth by node, live and terminal.

    Returns:
        ``None`` while the reviewed attempt is still active, otherwise the
        reason cancelling it would no longer do what the operator reviewed.
    """
    matching = [
        progress
        for progress in downloads.get(action.node_id, ())
        if progress.shard_metadata.model_card.model_id == action.model_id
    ]
    active = [
        progress
        for progress in matching
        if isinstance(progress, (DownloadPending, DownloadOngoing))
    ]
    if any(progress.attempt_id == action.attempt_id for progress in active):
        return None
    if active:
        return (
            "A newer download attempt replaced the one reviewed, and this "
            "proposal would not cancel it."
        )
    if any(isinstance(progress, DownloadCompleted) for progress in matching):
        return f"The download of {action.model_id} finished after review."
    if any(
        isinstance(progress, DownloadFailed) and progress.attempt_id == action.attempt_id
        for progress in matching
    ):
        return f"The download of {action.model_id} already failed after review."
    return "The proposed download is no longer active."


def placement_card_staleness(
    action: StewardPlaceModelAction,
    *,
    ordered_card: ModelCard | None,
) -> str | None:
    """Return why a placement no longer matches the reviewed model card.

    Args:
        action: Placement recording the exact card the operator reviewed.
        ordered_card: The master's command-ordered card for that model, or
            ``None`` after the model left the catalog.

    Returns:
        ``None`` while the card is unchanged, otherwise the reason. Capacity
        changes are deliberately not staleness: approval recomputes placement.
    """
    model_id = action.model_card.model_id
    if ordered_card is None:
        return f"{model_id} was removed from the model catalog after review."
    if not same_authorized_model_card(action.model_card, ordered_card):
        return f"The model card for {model_id} changed after review."
    return None


def steward_proposal_staleness(
    proposal: StewardActionProposal,
    *,
    instances: Mapping[InstanceId, Instance],
    downloads: Mapping[NodeId, Sequence[DownloadProgress]],
    node_names: Mapping[NodeId, str],
    ordered_card: Callable[[ModelId], ModelCard | None],
    other_proposals: Iterable[StewardActionProposal],
) -> str | None:
    """Return why a proposal no longer fits the cluster, or ``None``.

    Args:
        proposal: Proposal whose reviewed target is checked.
        instances: Live instances by identity.
        downloads: Effective download truth by node.
        node_names: Friendly node names for readable reasons.
        ordered_card: Injected lookup of the master's command-ordered card.
        other_proposals: Every retained proposal, for ownership conflicts.

    Returns:
        ``None`` while the reviewed target still holds, otherwise one
        operator-readable sentence explaining what changed.
    """
    action = proposal.action
    if isinstance(action, StewardPlaceModelAction):
        return placement_card_staleness(
            action, ordered_card=ordered_card(action.model_card.model_id)
        )
    if isinstance(action, StewardCancelDownloadAction):
        return download_target_staleness(action, downloads=downloads)
    return lifecycle_target_staleness(
        proposal,
        action,
        instances=instances,
        node_names=node_names,
        other_proposals=other_proposals,
        ordered_card=ordered_card,
    )
