# pyright: reportPrivateUsage=false
"""Download-progress throttle bounds the in-progress telemetry stream (#364).

A large download fires a progress callback per 8MB chunk across parallel files,
so even lossy latest-value admission should avoid pointless serialization work.
The fraction-delta gate caps a download to roughly ``1 / _PROGRESS_STEP`` useful
readings regardless of size or duration; completion and failure still use the
ordered event plane.
"""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from anyio import WouldBlock

from skulk.download import coordinator as coordinator_mod
from skulk.download.coordinator import DownloadCoordinator
from skulk.shared.models.model_cards import ModelId
from skulk.shared.tests.conftest import get_pipeline_shard_metadata
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import Event, NodeDownloadProgress
from skulk.shared.types.memory import Memory
from skulk.shared.types.telemetry import NodeTelemetry
from skulk.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadOngoing,
    DownloadProgressData,
)
from skulk.store.config import persist_model_trust_config, update_skulk_config_atomic
from skulk.utils.channels import channel
from skulk.worker.tests.constants import MODEL_A_ID


class _FakeDownloader:
    def on_progress(self, callback: object) -> None:
        pass


def _make_coordinator() -> DownloadCoordinator:
    _, cmd_recv = channel[object]()
    event_send, _ = channel[object]()
    telemetry_send, _ = channel[NodeTelemetry]()
    return DownloadCoordinator(
        node_id=NodeId("n1"),
        shard_downloader=cast("object", _FakeDownloader()),  # pyright: ignore[reportArgumentType]
        download_command_receiver=cast("object", cmd_recv),  # pyright: ignore[reportArgumentType]
        event_sender=cast("object", event_send),  # pyright: ignore[reportArgumentType]
        telemetry_sender=telemetry_send,
    )


@pytest.mark.asyncio
async def test_synced_config_notifies_config_dependent_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Runtime config sync must refresh API capability advertisements."""

    config_path = tmp_path / "skulk.yaml"
    callbacks = 0

    def applied() -> None:
        nonlocal callbacks
        callbacks += 1

    coordinator = _make_coordinator()
    coordinator.config_applied_callback = applied
    monkeypatch.setattr(coordinator_mod, "resolve_config_path", lambda: config_path)

    await coordinator._sync_config("experiments:\n  stt_realtime: true\n")

    assert config_path.read_text() == "experiments:\n  stt_realtime: true\n"
    assert callbacks == 1


@pytest.mark.asyncio
async def test_synced_config_preserves_node_local_hugging_face_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Secret-stripped cluster sync cannot erase a node's local credential."""

    config_path = tmp_path / "skulk.yaml"
    card_id = f"card_{'a' * 52}"
    config_path.write_text(
        "hf_token: local-secret\n"
        "model_trust:\n"
        "  approved_remote_code_identities:\n"
        f"    - {card_id}\n"
    )
    coordinator = _make_coordinator()
    monkeypatch.setattr(coordinator_mod, "resolve_config_path", lambda: config_path)

    await coordinator._sync_config("logging:\n  enabled: false\n")

    synchronized = config_path.read_text()
    assert "hf_token: local-secret" in synchronized
    assert "model_trust:" in synchronized
    assert card_id in synchronized
    assert config_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_synced_config_adopts_an_incoming_hugging_face_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A token entered in one node's Settings must reach this node.

    This is the receive half of token propagation: the broadcast carries the
    token, and every node persists it (0600) and promotes it into HF_TOKEN so
    downloads authenticate without a restart.
    """

    config_path = tmp_path / "skulk.yaml"
    coordinator = _make_coordinator()
    monkeypatch.setattr(coordinator_mod, "resolve_config_path", lambda: config_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    await coordinator._sync_config("hf_token: fleet-token\n")

    assert "hf_token: fleet-token" in config_path.read_text()
    assert config_path.stat().st_mode & 0o777 == 0o600
    import os

    assert os.environ.get("HF_TOKEN") == "fleet-token"


@pytest.mark.asyncio
async def test_synced_config_blank_token_does_not_clobber_local(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Present-but-blank is not a credential and must not erase a real one."""

    config_path = tmp_path / "skulk.yaml"
    config_path.write_text("hf_token: local-secret\n")
    coordinator = _make_coordinator()
    monkeypatch.setattr(coordinator_mod, "resolve_config_path", lambda: config_path)

    await coordinator._sync_config("hf_token: ''\nlogging:\n  enabled: false\n")

    assert "hf_token: local-secret" in config_path.read_text()


@pytest.mark.asyncio
async def test_synced_config_whitespace_token_does_not_clobber_local(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Whitespace is truthy but not a credential; it must merge as absent.

    Downstream resolution strips tokens before use, so a whitespace-only value
    arriving over the wire would replace a real local token with something
    that resolves as blank.
    """

    config_path = tmp_path / "skulk.yaml"
    config_path.write_text("hf_token: local-secret\n")
    coordinator = _make_coordinator()
    monkeypatch.setattr(coordinator_mod, "resolve_config_path", lambda: config_path)

    await coordinator._sync_config("hf_token: '   '\n")

    assert "hf_token: local-secret" in config_path.read_text()


@pytest.mark.asyncio
async def test_synced_config_merges_latest_trust_inside_transaction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A peer config sync cannot overwrite a concurrent trust decision."""

    config_path = tmp_path / "skulk.yaml"
    initial_identity = f"card_{'a' * 52}"
    latest_identity = f"card_{'b' * 52}"
    persist_model_trust_config(config_path, [initial_identity])
    coordinator = _make_coordinator()
    monkeypatch.setattr(coordinator_mod, "resolve_config_path", lambda: config_path)

    def inject_latest_trust(
        path: Path,
        update: object,
    ) -> dict[str, object]:
        persist_model_trust_config(path, [latest_identity])
        return update_skulk_config_atomic(
            path,
            cast("Callable[[dict[str, object]], dict[str, object]]", update),
        )

    monkeypatch.setattr(
        coordinator_mod,
        "update_skulk_config_atomic",
        inject_latest_trust,
    )

    await coordinator._sync_config("logging:\n  enabled: false\n")

    synchronized = config_path.read_text()
    assert latest_identity in synchronized
    assert initial_identity not in synchronized


def test_in_progress_throttle_gates_by_fraction_rate_and_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    co = _make_coordinator()
    mid = ModelId("org/model")
    clock = {"now": 1000.0}

    def fake_now() -> float:
        return clock["now"]

    monkeypatch.setattr(coordinator_mod, "current_time", fake_now)

    # First update always emits (establishes the baseline).
    assert co._should_emit_in_progress(mid, 0.0) is True

    # A >=step advance but within the rate floor (<1s) is suppressed.
    clock["now"] = 1000.5
    assert co._should_emit_in_progress(mid, 0.20) is False

    # A >=step advance after the rate floor emits.
    clock["now"] = 1002.0
    assert co._should_emit_in_progress(mid, 0.20) is True

    # A sub-step advance (<5%) is suppressed even after the rate floor.
    clock["now"] = 1004.0
    assert co._should_emit_in_progress(mid, 0.23) is False

    # No meaningful advance, but the heartbeat interval elapsed -> emit.
    clock["now"] = 1002.0 + co._HEARTBEAT_SECS + 1
    assert co._should_emit_in_progress(mid, 0.24) is True


def test_in_progress_throttle_ignores_fraction_regression_within_one_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Async file callbacks can arrive out of order. A stale lower fraction must
    # not be mistaken for a fresh attempt because doing so defeats the bounded
    # event-count guarantee and can overwrite a terminal status.
    co = _make_coordinator()
    mid = ModelId("org/model")
    clock = {"now": 1000.0}

    def fake_now() -> float:
        return clock["now"]

    monkeypatch.setattr(coordinator_mod, "current_time", fake_now)

    # Climb to a high baseline.
    assert co._should_emit_in_progress(mid, 0.0) is True
    clock["now"] = 1002.0
    assert co._should_emit_in_progress(mid, 0.80) is True

    # A regression in the same attempt is stale and stays suppressed.
    clock["now"] = 1002.2
    assert co._should_emit_in_progress(mid, 0.01) is False

    # The high-water mark is retained, so another stale update cannot reopen it.
    clock["now"] = 1004.0
    assert co._should_emit_in_progress(mid, 0.10) is False

    # A lifecycle reset starts a new attempt and allows its first observation.
    co._reset_progress_throttle(mid)
    clock["now"] = 1004.1
    assert co._should_emit_in_progress(mid, 0.01) is True


def test_in_progress_throttle_bounds_event_count_for_a_long_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate the raw downloader's per-8MB-chunk firing for a large model:
    # thousands of callbacks with tiny fraction deltas and sub-second spacing.
    # The gate must admit O(1/_PROGRESS_STEP) of them, not O(callbacks).
    co = _make_coordinator()
    mid = ModelId("org/big")
    clock = {"now": 0.0}

    def fake_now() -> float:
        return clock["now"]

    monkeypatch.setattr(coordinator_mod, "current_time", fake_now)

    emitted = 0
    chunks = 4000  # ~32 GB at 8MB chunks
    for i in range(1, chunks + 1):
        clock["now"] = i * 0.1  # 100ms per chunk -> 400s total, well past heartbeats
        if co._should_emit_in_progress(mid, i / chunks):
            emitted += 1

    # Without the fix this is ~ (400s / 1s) = 400 events; with the fraction gate
    # plus the heartbeat it is sharply bounded. Assert it is a small fraction of
    # the callbacks and near the step/heartbeat budget.
    assert emitted <= 30, f"expected a small bounded count, got {emitted}"
    assert emitted >= 1


async def test_transient_progress_bypasses_event_log_but_terminal_is_durable() -> None:
    """Coordinator routing keeps progress lossy and completion ordered."""

    _, command_receiver = channel[object]()
    event_sender, event_receiver = channel[Event]()
    telemetry_sender, telemetry_receiver = channel[NodeTelemetry]()
    coordinator = DownloadCoordinator(
        node_id=NodeId("node-a"),
        shard_downloader=cast("object", _FakeDownloader()),  # pyright: ignore[reportArgumentType]
        download_command_receiver=cast("object", command_receiver),  # pyright: ignore[reportArgumentType]
        event_sender=event_sender,
        telemetry_sender=telemetry_sender,
    )
    shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=1)
    ongoing = DownloadOngoing(
        node_id=NodeId("node-a"),
        shard_metadata=shard,
        download_progress=DownloadProgressData(
            total=Memory.from_mb(10),
            downloaded=Memory.from_mb(1),
            downloaded_this_session=Memory.from_mb(1),
            completed_files=0,
            total_files=1,
            speed=1.0,
            eta_ms=1,
            files={},
        ),
    )

    emitted_ongoing = await coordinator._emit_status(ongoing)
    telemetry = telemetry_receiver.receive_nowait()
    assert telemetry.info == emitted_ongoing
    with pytest.raises(WouldBlock):
        event_receiver.receive_nowait()

    completed = DownloadCompleted(
        node_id=NodeId("node-a"),
        shard_metadata=shard,
        total=Memory.from_mb(10),
    )
    emitted_completed = await coordinator._emit_status(completed)
    event = event_receiver.receive_nowait()
    assert isinstance(event, NodeDownloadProgress)
    assert event.download_progress == emitted_completed
    assert emitted_completed.attempt_id == emitted_ongoing.attempt_id
