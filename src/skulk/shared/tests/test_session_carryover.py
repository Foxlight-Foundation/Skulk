"""Tests for failover state carry-over (#273).

Pins what a promoted master inherits from the node's prior replicated state
(instances, downloads, node info, tracing) and what it must NOT inherit
(in-flight tasks, dead runner statuses, stale topology/liveness, the old
session's event index).
"""

from datetime import datetime, timezone

from skulk.shared.session_carryover import seed_state_for_new_session
from skulk.shared.topology import Topology
from skulk.shared.types.common import NodeId
from skulk.shared.types.profiling import MemoryUsage, NodeIdentity
from skulk.shared.types.state import State
from skulk.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadOngoing,
)


def _prior_state() -> tuple[State, NodeId]:
    node = NodeId()
    topology = Topology()
    topology.add_node(node)
    return State(
        # A stand-in mapping is enough: the seed must carry the field
        # through untouched, not interpret it.
        instances={},
        runners={},
        tasks={},
        downloads={node: []},
        # Deterministically stale so the "re-stamped to now, not copied from
        # prior" assertion can never collide with the test's stamp.
        last_seen={node: datetime(2020, 1, 1, tzinfo=timezone.utc)},
        topology=topology,
        tracing_enabled=True,
        last_event_applied_idx=4242,
        node_identities={node: NodeIdentity(friendly_name="kite-test")},
        node_memory={
            node: MemoryUsage.from_bytes(
                ram_total=16 * 2**30,
                ram_available=8 * 2**30,
                swap_total=0,
                swap_available=0,
            )
        },
    ), node


def test_carries_durable_facts():
    prior, node = _prior_state()
    seed = seed_state_for_new_session(prior)
    assert seed.instances == prior.instances
    assert seed.downloads == prior.downloads
    assert seed.tracing_enabled is True
    assert seed.node_identities[node].friendly_name == "kite-test"
    assert seed.node_memory[node].ram_available.in_bytes == 8 * 2**30


def test_carries_only_completed_downloads():
    # Ongoing/pending/failed downloads belong to the old session's restarted
    # coordinator — carrying DownloadOngoing would make the new planner
    # treat the download as in-hand and never re-issue it, stranding a
    # mid-download placement forever.
    from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
    from skulk.shared.types.memory import Memory
    from skulk.shared.types.worker.downloads import DownloadProgressData
    from skulk.shared.types.worker.shards import PipelineShardMetadata

    node = NodeId()
    shard = PipelineShardMetadata(
        model_card=ModelCard(
            model_id=ModelId("test-org/test-model"),
            storage_size=Memory.from_bytes(1_000_000),
            n_layers=2,
            hidden_size=64,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=2,
        n_layers=2,
    )
    completed = DownloadCompleted(node_id=node, shard_metadata=shard, total=Memory())
    ongoing = DownloadOngoing(
        node_id=node,
        shard_metadata=shard,
        download_progress=DownloadProgressData(
            total=Memory(),
            downloaded=Memory(),
            downloaded_this_session=Memory(),
            completed_files=0,
            total_files=1,
            speed=0.0,
            eta_ms=0,
            files={},
        ),
    )
    prior = State(downloads={node: [completed, ongoing]})
    seed = seed_state_for_new_session(prior)
    assert list(seed.downloads[node]) == [completed]


def test_drops_session_scoped_state():
    prior, _node = _prior_state()
    seed = seed_state_for_new_session(prior)
    # In-flight tasks died with the old session's plumbing.
    assert seed.tasks == {}
    # Runner statuses describe processes the transition tears down.
    assert seed.runners == {}
    # Liveness must come from live gossip — a carried topology would keep a
    # dead master's out-edges forever (only their source node deletes them).
    assert seed.topology.list_nodes() == []
    # The new session's event log starts at the beginning.
    assert seed.last_event_applied_idx == -1


def test_seeds_last_seen_for_carried_identities():
    # last_seen is re-stamped (not dropped) for every carried node so the
    # master's 30s prune clock is armed: a carried identity that never
    # re-gossips is reaped instead of leaking as a phantom node (#218 family).
    # Pre-fix this was dropped entirely, leaving carried identities with no
    # liveness anchor — and the prune loop only reaps via last_seen.
    prior, node = _prior_state()
    stamp = datetime(2026, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
    seed = seed_state_for_new_session(prior, now=stamp)
    # Every carried node identity gets a fresh last_seen anchor...
    assert seed.last_seen == {node: stamp}
    # ...re-stamped to `now`, NOT copied from the prior (possibly stale) view,
    # so a long-dead node is not admitted to the live clock as already-fresh.
    assert seed.last_seen[node] != prior.last_seen[node]


def test_seeds_last_seen_for_node_in_only_thunderbolt_map():
    # A node carried ONLY in a TB/RDMA map (not identities/memory) must still
    # get a last_seen anchor — otherwise it evades the prune loop and leaks as
    # a phantom after a session transition (review catch on #291). The carried
    # set must cover every node_* map copied into the seed, not just the common
    # identity/memory ones.
    from skulk.shared.types.profiling import NodeThunderboltInfo

    tb_only = NodeId()
    prior = State(node_thunderbolt={tb_only: NodeThunderboltInfo(interfaces=[])})
    stamp = datetime(2026, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
    seed = seed_state_for_new_session(prior, now=stamp)
    assert seed.node_thunderbolt[tb_only].interfaces == []
    # The TB-only node is armed for the prune loop.
    assert seed.last_seen.get(tb_only) == stamp
