"""Shared restart recovery test fixtures."""

from datetime import datetime, timedelta, timezone
from uuid import UUID

from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.topology import Topology
from skulk.shared.types.common import CommandId, ModelId, NodeId, SessionId
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.steward_actions import (
    StewardActionProposal,
    StewardActionProposalStatus,
    StewardStopInstanceAction,
)
from skulk.shared.types.worker.instances import (
    InstanceId,
    MlxRingInstance,
    ShardAssignments,
)
from skulk.shared.types.worker.runners import RunnerId
from skulk.shared.types.worker.shards import PipelineShardMetadata

NOW = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)
SESSION = SessionId(master_node_id=NodeId("node-a"), election_clock=3)
KITE1 = UUID("11111111-1111-4111-8111-111111111111")
KITE2 = UUID("22222222-2222-4222-8222-222222222222")
INSTALL_IDS = {NodeId("node-a"): KITE1, NodeId("node-b"): KITE2}


def sample_instance(
    instance_id: str,
    model: str,
    nodes: tuple[str, ...],
    *,
    system_role: str | None = None,
    excluded: tuple[str, ...] = (),
    requested_context_tokens: int | None = None,
) -> MlxRingInstance:
    """Return an ordinary (or steward) MLX instance spanning ``nodes``."""
    card = ModelCard(
        model_id=ModelId(model),
        storage_size=Memory.from_gb(8),
        n_layers=4,
        hidden_size=8,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    runner_to_shard = {
        RunnerId(f"{instance_id}-runner-{rank}"): PipelineShardMetadata(
            model_card=card,
            device_rank=rank,
            world_size=len(nodes),
            start_layer=rank * (4 // len(nodes)),
            end_layer=(rank + 1) * (4 // len(nodes)),
            n_layers=4,
        )
        for rank in range(len(nodes))
    }
    return MlxRingInstance(
        instance_id=InstanceId(instance_id),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            runner_to_shard=runner_to_shard,
            node_to_runner={
                NodeId(node): RunnerId(f"{instance_id}-runner-{rank}")
                for rank, node in enumerate(nodes)
            },
        ),
        hosts_by_node={NodeId(node): [] for node in nodes},
        ephemeral_port=52415,
        excluded_nodes=[NodeId(node) for node in excluded],
        system_role="steward" if system_role == "steward" else None,
        requested_context_tokens=requested_context_tokens,
    )


def connected(*nodes: str) -> Topology:
    """Return a topology holding ``nodes``, the machines currently connected."""
    topology = Topology()
    for node in nodes:
        topology.add_node(NodeId(node))
    return topology


def sample_proposal(status: StewardActionProposalStatus, minutes_ago: int) -> StewardActionProposal:
    """Return a stop proposal created ``minutes_ago`` before ``NOW`` in ``status``."""
    created = NOW - timedelta(minutes=minutes_ago)
    proposal = StewardActionProposal(
        action=StewardStopInstanceAction(instance=sample_instance("i", "org/model", ("node-a",))),
        rationale="The instance is idle.",
        evidence=("No requests for an hour.",),
        expected_effect="Stop the instance.",
        created_at=created,
        expires_at=created + timedelta(minutes=10),
    )
    if status == "pending":
        return proposal
    return proposal.model_copy(
        update={
            "status": status,
            "decided_at": created + timedelta(minutes=1),
            "decided_by": "trusted_fabric_operator",
            "command_id": CommandId("command") if status in {"approved", "dispatched"} else None,
        }
    )


def sample_state() -> State:
    """Return a view with two ordinary placements, the steward, and two proposals."""
    first = sample_instance("first", "org/big", ("node-a", "node-b"), requested_context_tokens=32768)
    steward = sample_instance("steward", "org/steward", ("node-a",), system_role="steward")
    second = sample_instance("second", "org/small", ("node-b",), excluded=("node-a",))
    return State(
        instances={first.instance_id: first, steward.instance_id: steward, second.instance_id: second},
        steward_action_proposals={
            proposal.proposal_id: proposal
            for proposal in (sample_proposal("rejected", 30), sample_proposal("pending", 2))
        },
        topology=connected("node-a", "node-b"),
        last_event_applied_idx=42,
    )
