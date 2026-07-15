"""Telemetry plane coverage (#279): wire round-trip + view coalescing.

The NodeTelemetry message must survive the gossip path (model_dump_json ->
bytes -> model_validate_json under the topic codec) or node telemetry never
reaches the view and the planner reads nothing. Mirrors the #287 lesson where
an in-process-only test missed a strict round-trip failure.
"""

from datetime import datetime, timedelta, timezone

from skulk.routing.topics import TELEMETRY
from skulk.shared.tests.conftest import get_pipeline_shard_metadata
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import NodeDownloadProgress, StateSnapshotHydrated
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import (
    AcceleratorMetrics,
    DiskUsage,
    MemoryUsage,
    NodeResources,
    SystemPerformanceProfile,
)
from skulk.shared.types.state import State
from skulk.shared.types.telemetry import (
    NodeTelemetry,
    TelemetryView,
    record_membership_from_event,
)
from skulk.shared.types.worker.downloads import (
    DownloadAttemptId,
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgressData,
)
from skulk.utils.info_gatherer.info_gatherer import (
    LinuxGpuMetrics,
    MactopMetrics,
    MiscData,
    NodeCapabilities,
    NodeDiskUsage,
    NodeHeartbeat,
    RdmaCtlStatus,
    StaticNodeInformation,
)
from skulk.worker.tests.constants import MODEL_A_ID


def test_node_telemetry_survives_topic_codec_round_trip() -> None:
    msg = NodeTelemetry(
        node_id=NodeId("node-a"),
        info=NodeResources(backends=frozenset({"mlx"}), participation="management"),
    )
    restored = TELEMETRY.deserialize(TELEMETRY.serialize(msg))
    assert restored == msg
    assert isinstance(restored.info, NodeResources)
    assert restored.info.participation == "management"
    assert restored.info.backends == frozenset({"mlx"})


def test_node_heartbeat_survives_topic_codec_round_trip() -> None:
    """The named liveness reading must survive the real gossip codec."""
    message = NodeTelemetry(node_id=NodeId("node-a"), info=NodeHeartbeat())

    restored = TELEMETRY.deserialize(TELEMETRY.serialize(message))

    assert restored == message
    assert isinstance(restored.info, NodeHeartbeat)


def test_live_download_progress_round_trips_and_terminal_wins() -> None:
    """Late progress cannot override its durable terminal outcome."""

    node = NodeId("node-a")
    shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=1)
    attempt = DownloadAttemptId("attempt-a")
    ongoing = DownloadOngoing(
        node_id=node,
        shard_metadata=shard,
        attempt_id=attempt,
        download_progress=DownloadProgressData(
            total=Memory.from_mb(100),
            downloaded=Memory.from_mb(50),
            downloaded_this_session=Memory.from_mb(50),
            completed_files=1,
            total_files=2,
            speed=1.0,
            eta_ms=1_000,
            files={},
        ),
    )
    message = NodeTelemetry(node_id=node, info=ongoing)
    restored = TELEMETRY.deserialize(TELEMETRY.serialize(message))
    assert restored == message
    assert isinstance(restored.info, DownloadOngoing)

    view = TelemetryView()
    view.apply(restored)
    assert view.effective_downloads({})[node] == [ongoing]

    completed = DownloadCompleted(
        node_id=node,
        shard_metadata=shard,
        attempt_id=attempt,
        total=Memory.from_mb(100),
    )
    view.record_download_event(NodeDownloadProgress(download_progress=completed))
    assert view.effective_downloads({node: [completed]})[node] == [completed]

    # Independent telemetry/control protocols do not promise cross-plane order.
    # A queued sample from the completed attempt is ignored after the terminal.
    view.apply(restored)
    assert view.effective_downloads({node: [completed]})[node] == [completed]

    view.remove_download_model(str(MODEL_A_ID))
    view.apply(restored)
    assert view.effective_downloads({}) == {}

    next_attempt = DownloadAttemptId("attempt-b")
    pending = DownloadPending(
        node_id=node,
        shard_metadata=shard,
        attempt_id=next_attempt,
    )
    view.apply(NodeTelemetry(node_id=node, info=pending))
    assert view.effective_downloads({}) == {}
    view.record_download_event(NodeDownloadProgress(download_progress=pending))
    view.apply(NodeTelemetry(node_id=node, info=pending))
    assert view.effective_downloads({})[node] == [pending]

    # The reset's fresh attempt is authoritative even if the old producer's
    # telemetry tail arrives after it on the independent protocol.
    view.apply(restored)
    assert view.effective_downloads({})[node] == [pending]


def test_live_download_statuses_share_one_coalescing_key() -> None:
    """Pending and Ongoing replace each other before telemetry egress."""

    node = NodeId("node-a")
    shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=1)
    attempt = DownloadAttemptId("attempt-a")
    pending = DownloadPending(
        node_id=node,
        shard_metadata=shard,
        attempt_id=attempt,
    )
    ongoing = DownloadOngoing(
        node_id=node,
        shard_metadata=shard,
        attempt_id=attempt,
        download_progress=DownloadProgressData(
            total=Memory.from_mb(100),
            downloaded=Memory.from_mb(50),
            downloaded_this_session=Memory.from_mb(50),
            completed_files=1,
            total_files=2,
            speed=1.0,
            eta_ms=1_000,
            files={},
        ),
    )

    assert NodeTelemetry(node_id=node, info=pending).coalescing_key() == NodeTelemetry(
        node_id=node, info=ongoing
    ).coalescing_key()


def test_delayed_pending_cannot_replace_newer_attempt_progress() -> None:
    """Only ordered Pending may advance or replace a download attempt."""

    node = NodeId("node-a")
    shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=1)
    old_pending = DownloadPending(
        node_id=node,
        shard_metadata=shard,
        attempt_id=DownloadAttemptId("attempt-old"),
    )
    new_attempt = DownloadAttemptId("attempt-new")
    new_pending = old_pending.model_copy(update={"attempt_id": new_attempt})
    new_ongoing = DownloadOngoing(
        node_id=node,
        shard_metadata=shard,
        attempt_id=new_attempt,
        download_progress=DownloadProgressData(
            total=Memory.from_mb(100),
            downloaded=Memory.from_mb(50),
            downloaded_this_session=Memory.from_mb(50),
            completed_files=1,
            total_files=2,
            speed=1.0,
            eta_ms=1_000,
            files={},
        ),
    )
    view = TelemetryView()
    view.record_download_event(NodeDownloadProgress(download_progress=new_pending))
    view.apply(NodeTelemetry(node_id=node, info=new_ongoing))

    view.apply(NodeTelemetry(node_id=node, info=old_pending))

    assert view.effective_downloads({})[node] == [new_ongoing]


def test_eviction_tombstones_model_before_any_download_is_observed() -> None:
    """Fleet-wide eviction rejects delayed telemetry for an unknown model key."""

    node = NodeId("node-a")
    shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=1)
    attempt = DownloadAttemptId("attempt-old")
    pending = DownloadPending(
        node_id=node,
        shard_metadata=shard,
        attempt_id=attempt,
    )
    ongoing = DownloadOngoing(
        node_id=node,
        shard_metadata=shard,
        attempt_id=attempt,
        download_progress=DownloadProgressData(
            total=Memory.from_mb(100),
            downloaded=Memory.from_mb(50),
            downloaded_this_session=Memory.from_mb(50),
            completed_files=1,
            total_files=2,
            speed=1.0,
            eta_ms=1_000,
            files={},
        ),
    )
    view = TelemetryView()

    view.remove_download_model(str(MODEL_A_ID))
    view.apply(NodeTelemetry(node_id=node, info=pending))
    view.apply(NodeTelemetry(node_id=node, info=ongoing))

    assert view.effective_downloads({}) == {}

    next_attempt = DownloadAttemptId("attempt-next")
    restart = pending.model_copy(update={"attempt_id": next_attempt})
    view.record_download_event(NodeDownloadProgress(download_progress=restart))
    view.apply(NodeTelemetry(node_id=node, info=pending))
    assert view.effective_downloads({}) == {}
    view.apply(NodeTelemetry(node_id=node, info=restart))
    assert view.effective_downloads({})[node] == [restart]


def test_hydrated_terminal_download_rejects_delayed_progress() -> None:
    """Snapshot state seeds ordering before independent telemetry can overlay it."""

    node = NodeId("node-a")
    shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=1)
    attempt = DownloadAttemptId("attempt-a")
    ongoing = DownloadOngoing(
        node_id=node,
        shard_metadata=shard,
        attempt_id=attempt,
        download_progress=DownloadProgressData(
            total=Memory.from_mb(100),
            downloaded=Memory.from_mb(50),
            downloaded_this_session=Memory.from_mb(50),
            completed_files=1,
            total_files=2,
            speed=1.0,
            eta_ms=1_000,
            files={},
        ),
    )
    completed = DownloadCompleted(
        node_id=node,
        shard_metadata=shard,
        attempt_id=attempt,
        total=Memory.from_mb(100),
    )
    view = TelemetryView()
    view.apply(NodeTelemetry(node_id=node, info=ongoing))

    snapshot = State(downloads={node: [completed]})
    record_membership_from_event(view, StateSnapshotHydrated(state=snapshot))
    view.apply(NodeTelemetry(node_id=node, info=ongoing))

    assert view.effective_downloads(snapshot.downloads)[node] == [completed]


def test_hydration_clears_live_downloads_absent_from_snapshot() -> None:
    """Old-session live progress cannot survive an authoritative hydration."""

    node = NodeId("node-a")
    shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=1)
    old_attempt = DownloadAttemptId("attempt-old")
    ongoing = DownloadOngoing(
        node_id=node,
        shard_metadata=shard,
        attempt_id=old_attempt,
        download_progress=DownloadProgressData(
            total=Memory.from_mb(100),
            downloaded=Memory.from_mb(50),
            downloaded_this_session=Memory.from_mb(50),
            completed_files=1,
            total_files=2,
            speed=1.0,
            eta_ms=1_000,
            files={},
        ),
    )
    view = TelemetryView()
    view.apply(NodeTelemetry(node_id=node, info=ongoing))

    record_membership_from_event(
        view,
        StateSnapshotHydrated(state=State(last_event_applied_idx=0)),
    )

    assert view.effective_downloads({}) == {}
    next_pending = DownloadPending(
        node_id=node,
        shard_metadata=shard,
        attempt_id=DownloadAttemptId("attempt-next"),
    )
    view.apply(NodeTelemetry(node_id=node, info=next_pending))
    assert view.effective_downloads({})[node] == [next_pending]


def test_pending_telemetry_can_establish_new_attempt_before_control_event() -> None:
    """Cross-plane reordering must not hide the next attempt's live status."""

    node = NodeId("node-a")
    shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=1)
    completed_attempt = DownloadAttemptId("attempt-a")
    next_attempt = DownloadAttemptId("attempt-b")
    view = TelemetryView()
    view.record_download_event(
        NodeDownloadProgress(
            download_progress=DownloadCompleted(
                node_id=node,
                shard_metadata=shard,
                attempt_id=completed_attempt,
                total=Memory.from_mb(100),
            )
        )
    )

    pending = DownloadPending(
        node_id=node,
        shard_metadata=shard,
        attempt_id=next_attempt,
    )
    view.apply(NodeTelemetry(node_id=node, info=pending))
    view.record_download_event(NodeDownloadProgress(download_progress=pending))

    assert view.effective_downloads({})[node] == [pending]


def test_legacy_terminal_replay_preserves_identified_live_attempt() -> None:
    """An ambiguous legacy terminal cannot erase newer identified progress."""

    node = NodeId("node-a")
    shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=1)
    attempt = DownloadAttemptId("attempt-current")
    pending = DownloadPending(
        node_id=node,
        shard_metadata=shard,
        attempt_id=attempt,
    )
    view = TelemetryView()
    view.apply(NodeTelemetry(node_id=node, info=pending))

    legacy_terminals = (
        DownloadCompleted(
            node_id=node,
            shard_metadata=shard,
            total=Memory.from_mb(100),
        ),
        DownloadFailed(
            node_id=node,
            shard_metadata=shard,
            error_message="legacy failure",
        ),
    )
    for terminal in legacy_terminals:
        view.record_download_event(NodeDownloadProgress(download_progress=terminal))
        assert view.effective_downloads({node: [terminal]})[node] == [pending]


def test_heartbeat_and_fallback_receipts_remain_independent() -> None:
    """Heartbeat traffic must not overwrite ordinary telemetry fallback age."""
    view = TelemetryView()
    node = NodeId("node-a")
    fallback_at = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)
    heartbeat_at = fallback_at + timedelta(seconds=1)

    view.apply(
        NodeTelemetry(node_id=node, info=NodeResources(participation="full")),
        received_at=fallback_at,
    )
    view.apply(
        NodeTelemetry(node_id=node, info=NodeHeartbeat()),
        received_at=heartbeat_at,
    )

    assert view.node_last_telemetry[node] == fallback_at
    assert view.node_last_heartbeat[node] == heartbeat_at
    assert view.last_liveness_receipt(node) == heartbeat_at


def test_view_keeps_latest_per_node() -> None:
    view = TelemetryView()
    node = NodeId("node-a")
    view.apply(NodeTelemetry(node_id=node, info=NodeResources(participation="full")))
    assert view.node_resources[node].participation == "full"
    # last write wins — a later reading replaces the earlier one
    view.apply(
        NodeTelemetry(node_id=node, info=NodeResources(participation="management"))
    )
    assert view.node_resources[node].participation == "management"
    assert len(view.node_resources) == 1


def test_view_coalesces_memory_telemetry() -> None:
    # node_memory moved onto the telemetry plane in slice 2: a MemoryUsage
    # reading must land in view.node_memory, last-write-wins per node.
    view = TelemetryView()
    node = NodeId("node-a")
    view.apply(
        NodeTelemetry(
            node_id=node,
            info=MemoryUsage.from_bytes(
                ram_total=16 * 2**30,
                ram_available=8 * 2**30,
                swap_total=0,
                swap_available=0,
            ),
        )
    )
    assert view.node_memory[node].ram_available.in_bytes == 8 * 2**30
    view.apply(
        NodeTelemetry(
            node_id=node,
            info=MemoryUsage.from_bytes(
                ram_total=16 * 2**30,
                ram_available=4 * 2**30,
                swap_total=0,
                swap_available=0,
            ),
        )
    )
    assert view.node_memory[node].ram_available.in_bytes == 4 * 2**30
    assert len(view.node_memory) == 1


def test_view_coalesces_mactop_memory_and_system() -> None:
    # MactopMetrics carries BOTH memory and the system profile; one reading
    # must populate node_memory AND node_system (the combined dispatch path).
    view = TelemetryView()
    node = NodeId("node-a")
    view.apply(
        NodeTelemetry(
            node_id=node,
            info=MactopMetrics(
                system_profile=SystemPerformanceProfile(sys_power=42.0),
                memory=MemoryUsage.from_bytes(
                    ram_total=32 * 2**30,
                    ram_available=20 * 2**30,
                    swap_total=0,
                    swap_available=0,
                ),
            ),
        )
    )
    assert view.node_memory[node].ram_available.in_bytes == 20 * 2**30
    assert view.node_system[node].sys_power == 42.0


def test_view_coalesces_disk_and_rdma_telemetry() -> None:
    # Slice 3: disk + rdma-ctl readings land in their own view maps.
    view = TelemetryView()
    node = NodeId("node-a")
    view.apply(
        NodeTelemetry(
            node_id=node,
            info=NodeDiskUsage(
                disk_usage=DiskUsage(
                    total=Memory.from_bytes(500 * 2**30),
                    available=Memory.from_bytes(200 * 2**30),
                )
            ),
        )
    )
    assert view.node_disk[node].available.in_bytes == 200 * 2**30
    view.apply(
        NodeTelemetry(
            node_id=node,
            info=RdmaCtlStatus(enabled=True, interfaces_present=True),
        )
    )
    assert view.node_rdma_ctl[node].enabled is True


def test_view_merges_identity_from_two_readings() -> None:
    # Slice 3: identity is assembled from MiscData (friendly name) and
    # StaticNodeInformation (static fields); the two readings must MERGE into one
    # NodeIdentity rather than overwrite each other (mirrors the event applier's
    # accumulation), regardless of arrival order.
    view = TelemetryView()
    node = NodeId("node-a")
    view.apply(NodeTelemetry(node_id=node, info=MiscData(friendly_name="Kite 2")))
    view.apply(
        NodeTelemetry(
            node_id=node,
            info=StaticNodeInformation(
                model="Mac mini",
                chip="M4",
                os_version="26.4",
                os_build_version="X",
                skulk_version="1.2.0",
                skulk_commit="abc123",
            ),
        )
    )
    identity = view.node_identities[node]
    assert identity.friendly_name == "Kite 2"  # preserved across the static merge
    assert identity.chip_id == "M4"
    assert identity.skulk_version == "1.2.0"
    # a later friendly-name update keeps the static fields
    view.apply(
        NodeTelemetry(node_id=node, info=MiscData(friendly_name="Kite 2 (renamed)"))
    )
    identity = view.node_identities[node]
    assert identity.friendly_name == "Kite 2 (renamed)"
    assert identity.chip_id == "M4"


def test_node_capabilities_survive_topic_codec_round_trip() -> None:
    # Extension-advertised capabilities gossip on the telemetry plane; the tag
    # set (a frozenset) must round-trip through the strict codec or advertise
    # is inert. The wire form is a JSON array; coercion restores the frozenset.
    msg = NodeTelemetry(
        node_id=NodeId("node-a"),
        info=NodeCapabilities(capabilities=frozenset({"memory", "embeddings"})),
    )
    restored = TELEMETRY.deserialize(TELEMETRY.serialize(msg))
    assert restored == msg
    assert isinstance(restored.info, NodeCapabilities)
    assert restored.info.capabilities == frozenset({"memory", "embeddings"})


def test_node_capabilities_serialize_sorted() -> None:
    # Serialization emits a sorted list so JSON encoding is deterministic and a
    # frozenset (which JSON cannot encode) round-trips.
    dumped = NodeCapabilities(
        capabilities=frozenset({"c", "a", "b"})
    ).model_dump(mode="json")
    assert dumped == {"NodeCapabilities": {"capabilities": ["a", "b", "c"]}}


def test_view_coalesces_capabilities_telemetry() -> None:
    # A NodeCapabilities reading lands in view.node_capabilities, last-write-wins:
    # the latest advertised set replaces the earlier one per node.
    view = TelemetryView()
    node = NodeId("node-a")
    view.apply(
        NodeTelemetry(node_id=node, info=NodeCapabilities(capabilities=frozenset({"memory"})))
    )
    assert view.node_capabilities[node] == frozenset({"memory"})
    view.apply(
        NodeTelemetry(
            node_id=node,
            info=NodeCapabilities(capabilities=frozenset({"memory", "search"})),
        )
    )
    assert view.node_capabilities[node] == frozenset({"memory", "search"})
    assert len(view.node_capabilities) == 1


def test_prune_keeps_local_advertised_capabilities() -> None:
    # prune() drops a peer's readings, but the node's OWN outbound advertisement
    # set is not a peer reading — pruning a timed-out peer must never clear what
    # this node offers.
    view = TelemetryView()
    peer = NodeId("node-b")
    view.local_advertised_capabilities.add("memory")
    view.apply(
        NodeTelemetry(node_id=peer, info=NodeCapabilities(capabilities=frozenset({"search"})))
    )
    view.prune(peer)
    assert peer not in view.node_capabilities
    assert view.local_advertised_capabilities == {"memory"}


def test_prune_drops_all_telemetry_for_a_node() -> None:
    # On NodeTimedOut the view must drop the node entirely (it has no natural
    # expiry); otherwise a dead node lingers as a ghost in /state and skews
    # capacity/energy aggregates (#279 slice 2).
    gib = 1024 * 1024 * 1024
    reading = MemoryUsage.from_bytes(
        ram_total=16 * gib, ram_available=8 * gib, swap_total=0, swap_available=0
    )
    profile = SystemPerformanceProfile(sys_power=10.0)
    view = TelemetryView()
    a, b = NodeId("node-a"), NodeId("node-b")
    for node in (a, b):
        view.apply(NodeTelemetry(node_id=node, info=NodeHeartbeat()))
        view.apply(
            NodeTelemetry(node_id=node, info=NodeResources(participation="full"))
        )
        view.apply(
            NodeTelemetry(
                node_id=node,
                info=MactopMetrics(system_profile=profile, memory=reading),
            )
        )
        # slice-3 maps too
        view.apply(NodeTelemetry(node_id=node, info=MiscData(friendly_name="n")))
        view.apply(
            NodeTelemetry(
                node_id=node,
                info=NodeDiskUsage(
                    disk_usage=DiskUsage(
                        total=Memory.from_bytes(500 * gib),
                        available=Memory.from_bytes(200 * gib),
                    )
                ),
            )
        )
        view.apply(
            NodeTelemetry(
                node_id=node,
                info=RdmaCtlStatus(enabled=False, interfaces_present=False),
            )
        )
        view.apply(
            NodeTelemetry(
                node_id=node,
                info=NodeCapabilities(capabilities=frozenset({"memory"})),
            )
        )
    view.prune(a)
    for m in (
        view.node_resources,
        view.node_memory,
        view.node_system,
        view.node_identities,
        view.node_disk,
        view.node_rdma_ctl,
        view.node_capabilities,
        view.node_last_heartbeat,
        view.node_last_telemetry,
    ):
        assert a not in m
        assert b in m  # only the pruned node is dropped


def test_linux_gpu_metrics_round_trip_and_apply() -> None:
    # The AMD/Linux collector must survive the gossip codec and land its
    # accelerator block in node_system (memory arrives separately).
    msg = NodeTelemetry(
        node_id=NodeId("kite4"),
        info=LinuxGpuMetrics(
            system_profile=SystemPerformanceProfile(
                accelerator=AcceleratorMetrics(
                    vendor="amd",
                    name="AMD GPU",
                    utilization_ratio=0.42,
                    vram_total_bytes=68719476736,
                    vram_used_bytes=163233792,
                    power_watts=9.03,
                    temperature_celsius=39.0,
                    clock_mhz=1100,
                )
            )
        ),
    )
    restored = TELEMETRY.deserialize(TELEMETRY.serialize(msg))
    assert restored == msg
    assert isinstance(restored.info, LinuxGpuMetrics)

    view = TelemetryView()
    view.apply(restored)
    acc = view.node_system[NodeId("kite4")].accelerator
    assert acc is not None
    assert acc.vendor == "amd"
    assert acc.vram_total_bytes == 68719476736
    assert acc.utilization_ratio == 0.42
