"""Tests for the gemma-4 hang repro harness.

Hang detection is a pure function over snapshot data, so unit tests can
exercise every classification path without an actual cluster. The kernel-
panic, recovered-hang, and active-hang paths each have direct fixtures
because each one represents a distinct on-call signal that we cannot
afford to silently misclassify.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# bench/ is not on the standard import path; vend it in for tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from repro_gemma4_hang import (  # noqa: E402
    RunnerSnapshot,
    TimelineSnapshot,
    TimeoutEvent,
    count_images,
    evaluate_timeline_for_hang,
    parse_timeline_response,
    session_iterator,
    session_to_messages,
)

# ---------------------------------------------------------------------------
# Hang detection
# ---------------------------------------------------------------------------


def _runner(
    rank: int,
    phase: str = "decode_stream",
    seconds_in_phase: float = 0.5,
    *,
    node_id: str | None = None,
    last_progress_at: str | None = "2026-04-25T12:00:00Z",
) -> RunnerSnapshot:
    return RunnerSnapshot(
        node_id=node_id or f"node-{rank}",
        device_rank=rank,
        phase=phase,
        seconds_in_phase=seconds_in_phase,
        last_progress_at=last_progress_at,
    )


def test_healthy_cluster_returns_ok() -> None:
    snap = TimelineSnapshot(runners=tuple(_runner(r) for r in range(3)))
    verdict = evaluate_timeline_for_hang(snap, hang_after_seconds=90.0)
    assert verdict.status == "ok"


def test_long_running_phase_under_threshold_is_ok() -> None:
    snap = TimelineSnapshot(runners=(_runner(0, "decode_stream", 12.0),))
    verdict = evaluate_timeline_for_hang(snap, hang_after_seconds=90.0)
    assert verdict.status == "ok"


def test_runner_stuck_past_threshold_is_hang() -> None:
    stuck = _runner(0, "decode_stream", 142.0)
    snap = TimelineSnapshot(runners=(stuck, _runner(1), _runner(2)))
    verdict = evaluate_timeline_for_hang(snap, hang_after_seconds=90.0)
    assert verdict.status == "hang"
    assert verdict.suspect_runner == stuck
    assert "decode_stream" in verdict.reason
    assert "142" in verdict.reason


@pytest.mark.parametrize(
    "exempt_phase",
    ["warmup", "loading", "downloading", "created", "shutdown_cleanup"],
)
def test_long_running_phases_are_exempt_from_hang_detection(
    exempt_phase: str,
) -> None:
    """Cold-start operations can legitimately exceed the hang threshold."""
    snap = TimelineSnapshot(runners=(_runner(0, exempt_phase, 600.0),))
    verdict = evaluate_timeline_for_hang(snap, hang_after_seconds=90.0)
    assert verdict.status == "ok"


def test_pipeline_eval_timeout_classifies_as_recovered_hang() -> None:
    """Eval timeout firing means the supervised-recovery layer worked."""
    snap = TimelineSnapshot(
        runners=(_runner(0), _runner(1), _runner(2)),
        eval_timeout_events=(
            TimeoutEvent(
                at="2026-04-25T12:00:01Z",
                node_id="node-0",
                device_rank=0,
                detail="pipeline_last_eval_output",
            ),
        ),
    )
    verdict = evaluate_timeline_for_hang(snap, hang_after_seconds=90.0)
    assert verdict.status == "recovered_hang"
    assert "pipeline_last_eval_output" in verdict.reason


def test_recovered_hang_takes_precedence_over_active_hang() -> None:
    """If both signals are present we want the more informative one.

    A timeout event names the exact eval site that wedged. An ongoing
    secondsInPhase alarm just says "something is stuck right now." For
    the upstream bug report the site is what matters; surface it first.
    """
    snap = TimelineSnapshot(
        runners=(_runner(0, "decode_stream", 200.0),),
        eval_timeout_events=(
            TimeoutEvent(
                at="2026-04-25T12:00:01Z",
                node_id="node-0",
                device_rank=0,
                detail="pipeline_last_eval_output",
            ),
        ),
    )
    verdict = evaluate_timeline_for_hang(snap, hang_after_seconds=90.0)
    assert verdict.status == "recovered_hang"


def test_most_recent_timeout_event_is_surfaced() -> None:
    snap = TimelineSnapshot(
        runners=(_runner(0),),
        eval_timeout_events=(
            TimeoutEvent(
                at="2026-04-25T12:00:01Z",
                node_id="node-0",
                device_rank=0,
                detail="pipeline_first_eval_recv",
            ),
            TimeoutEvent(
                at="2026-04-25T12:05:00Z",
                node_id="node-1",
                device_rank=1,
                detail="mx_barrier",
            ),
        ),
    )
    verdict = evaluate_timeline_for_hang(snap, hang_after_seconds=90.0)
    assert verdict.status == "recovered_hang"
    assert "mx_barrier" in verdict.reason
    assert "node-1" in verdict.reason


def test_missing_node_classifies_as_kernel_panic_inferred() -> None:
    """A node disappearing from the timeline (without unregister) means reboot."""
    snap = TimelineSnapshot(
        runners=(_runner(1), _runner(2)),  # rank 0 missing
    )
    verdict = evaluate_timeline_for_hang(
        snap,
        hang_after_seconds=90.0,
        expected_node_ids=frozenset({"node-0", "node-1", "node-2"}),
    )
    assert verdict.status == "kernel_panic_inferred"
    assert "node-0" in verdict.reason


def test_unreachable_peer_does_not_trigger_kernel_panic_signal() -> None:
    """An unreachable-but-known peer is a soft failure, not a panic.

    The cluster timeline endpoint reports unreachable peers explicitly via
    ``unreachableNodes``. That covers temporary network blips. Only a peer
    that has *vanished entirely* (not in runners and not in unreachable)
    should flag as a kernel-panic inference.
    """
    snap = TimelineSnapshot(
        runners=(_runner(1), _runner(2)),
        unreachable_node_ids=("node-0",),
    )
    verdict = evaluate_timeline_for_hang(
        snap,
        hang_after_seconds=90.0,
        expected_node_ids=frozenset({"node-0", "node-1", "node-2"}),
    )
    assert verdict.status == "ok"


def test_kernel_panic_takes_precedence_over_recovered_hang() -> None:
    """If the box is gone we want to know that, even if a timeout had fired."""
    snap = TimelineSnapshot(
        runners=(_runner(1),),
        eval_timeout_events=(
            TimeoutEvent(
                at="2026-04-25T12:00:01Z",
                node_id="node-0",
                device_rank=0,
                detail="pipeline_last_eval_output",
            ),
        ),
    )
    verdict = evaluate_timeline_for_hang(
        snap,
        hang_after_seconds=90.0,
        expected_node_ids=frozenset({"node-0", "node-1"}),
    )
    assert verdict.status == "kernel_panic_inferred"


def test_no_expected_nodes_disables_kernel_panic_signal() -> None:
    """If the harness couldn't snapshot the node set, don't false-positive."""
    snap = TimelineSnapshot(runners=(_runner(0),))
    verdict = evaluate_timeline_for_hang(
        snap, hang_after_seconds=90.0, expected_node_ids=None
    )
    assert verdict.status == "ok"


# ---------------------------------------------------------------------------
# Timeline payload parsing
# ---------------------------------------------------------------------------


def test_parse_timeline_handles_well_formed_payload() -> None:
    payload = {
        "runners": [
            {
                "nodeId": "node-A",
                "deviceRank": 0,
                "phase": "decode_stream",
                "secondsInPhase": 1.5,
                "lastProgressAt": "2026-04-25T12:00:00Z",
            }
        ],
        "timeline": [
            {
                "at": "2026-04-25T12:00:01Z",
                "nodeId": "node-A",
                "deviceRank": 0,
                "phase": "decode_stream",
                "event": "pipeline_eval_timeout",
                "detail": "pipeline_last_eval_output",
            },
            {
                "at": "2026-04-25T12:00:00Z",
                "nodeId": "node-A",
                "deviceRank": 0,
                "phase": "decode_stream",
                "event": "enter",
                "detail": "pipeline_last_eval_output",
            },
        ],
        "unreachableNodes": [{"nodeId": "node-Z", "error": "connection refused"}],
    }
    snap = parse_timeline_response(payload)
    assert len(snap.runners) == 1
    assert snap.runners[0].node_id == "node-A"
    assert len(snap.eval_timeout_events) == 1  # only the timeout one
    assert snap.eval_timeout_events[0].detail == "pipeline_last_eval_output"
    assert snap.unreachable_node_ids == ("node-Z",)


def test_parse_timeline_tolerates_missing_top_level_fields() -> None:
    snap = parse_timeline_response({})
    assert snap.runners == ()
    assert snap.eval_timeout_events == ()
    assert snap.unreachable_node_ids == ()


def test_parse_timeline_tolerates_null_arrays() -> None:
    snap = parse_timeline_response(
        {"runners": None, "timeline": None, "unreachableNodes": None}
    )
    assert snap.runners == ()


# ---------------------------------------------------------------------------
# Session generation
# ---------------------------------------------------------------------------


def test_session_iterator_is_deterministic_for_seed() -> None:
    """Same seed → same first-N-session order, across runs and processes."""
    seq_a = list(zip(range(20), session_iterator(123), strict=False))
    seq_b = list(zip(range(20), session_iterator(123), strict=False))
    assert [s.label for _, s in seq_a] == [s.label for _, s in seq_b]


def test_session_iterator_eventually_yields_every_template() -> None:
    """Every template should appear within the first cycle through the deck."""
    seen: set[str] = set()
    it = session_iterator(7)
    for _ in range(8):
        seen.add(next(it).label)
    assert "multimodal-history-text-followup" in seen
    assert "text-only-multi-turn" in seen
    assert "three-image-sequence" in seen
    assert "large-image-single-turn" in seen


def test_session_to_messages_emits_progressive_history() -> None:
    """Turn N's history must include all prior user+assistant pairs."""
    it = session_iterator(0)
    # Walk until we land on the multi-turn multimodal template.
    while True:
        sess = next(it)
        if sess.label == "multimodal-history-text-followup":
            break
    progressions = session_to_messages(sess)
    assert len(progressions) == 3
    # Each progression is strictly larger than the prior one (history grows).
    assert len(progressions[0]) < len(progressions[1]) < len(progressions[2])
    # Final history mentions both images and ends with a user turn.
    final = progressions[-1]
    assert final[-1]["role"] == "user"
    image_count = count_images(final)
    assert image_count == 2  # two images, one in each user turn


def testcount_images_recognizes_image_url_parts() -> None:
    history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "x"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,yyy"}},
            ],
        },
        {"role": "assistant", "content": "ok"},
    ]
    assert count_images(history) == 2


def testcount_images_handles_string_content() -> None:
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    assert count_images(history) == 0
