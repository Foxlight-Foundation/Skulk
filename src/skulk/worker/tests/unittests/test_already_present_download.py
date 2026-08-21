"""Coverage for the already-staged download announcement (worker fast path).

A model found on disk skips the download coordinator, so the worker itself is
responsible for the attempt identity that makes its terminal outcome
attributable. Without it the completion is ignored as an unorderable legacy
replay, the live pending keeps overlaying it, and the instance hangs forever in
`RunnerIdle` while the planner re-issues `DownloadModel` every tick.
"""

from skulk.shared.tests.conftest import get_pipeline_shard_metadata
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.events import NodeDownloadProgress
from skulk.shared.types.telemetry import NodeTelemetry, TelemetryView
from skulk.shared.types.worker.downloads import DownloadCompleted, DownloadPending
from skulk.worker.main import already_present_download_events

MODEL_ID = ModelId("org/already-staged")
NODE_ID = NodeId("node-a")


def _events() -> tuple[NodeDownloadProgress, NodeDownloadProgress]:
    shard = get_pipeline_shard_metadata(MODEL_ID, device_rank=0, world_size=1)
    return already_present_download_events(
        node_id=NODE_ID, shard=shard, model_directory="/models/already-staged"
    )


def test_emits_a_pending_then_a_completed() -> None:
    """The pair must be ordered: the pending opens the attempt it closes."""

    pending_event, completed_event = _events()

    assert isinstance(pending_event.download_progress, DownloadPending)
    assert isinstance(completed_event.download_progress, DownloadCompleted)


def test_both_events_share_one_non_null_attempt_id() -> None:
    """This is the property whose absence caused the hang."""

    pending_event, completed_event = _events()
    pending = pending_event.download_progress
    completed = completed_event.download_progress

    assert pending.attempt_id is not None
    assert completed.attempt_id is not None
    assert completed.attempt_id == pending.attempt_id


def test_completion_reports_the_staged_directory_as_read_only() -> None:
    """A model found in place is not owned by us and must not be deleted."""

    _, completed_event = _events()
    completed = completed_event.download_progress

    assert isinstance(completed, DownloadCompleted)
    assert completed.read_only is True
    assert completed.model_directory == "/models/already-staged"


def test_the_pair_leaves_no_live_overlay_hiding_the_completion() -> None:
    """End to end through the view that decides what a planner sees.

    Drives the real `TelemetryView` the way the plane does: the pending is
    recorded as a durable event and also arrives as live telemetry, then the
    completion closes the attempt. If the completion were unattributed, the
    pending would survive here and the planner would keep believing the model
    is absent.
    """

    pending_event, completed_event = _events()
    pending = pending_event.download_progress
    assert isinstance(pending, DownloadPending)

    view = TelemetryView()
    view.record_download_event(pending_event)
    view.apply(NodeTelemetry(node_id=NODE_ID, info=pending))
    assert view.effective_downloads({})[NODE_ID] == [pending]

    view.record_download_event(completed_event)

    assert view.effective_downloads({}).get(NODE_ID, []) == []
