"""Permanent mirror decisions, cancellation fences and restart observations."""

from pathlib import Path

import pytest

from skulk.store.mirror_contracts import MirrorActivity, MirrorRequest
from skulk.store.preservation_journal import PreservationHostError, PreservationJournal


def request(operation: str = "run") -> MirrorRequest:
    """Describe a fresh finite public run, without granting its actual execution."""
    return MirrorRequest(
        owner_id="library",
        operation_id=operation,
        kind="run",
        policy_id="a" * 64,
        public_participation=True,
    )


def test_mirror_restart_preserves_request_and_drops_historical_live_activity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "owner"
    journal = PreservationJournal(root)
    original = request()
    job, created = journal.reserve_mirror(original)
    assert created
    journal.update_mirror_job(
        job.model_copy(
            update={
                "state": "running",
                "bytes_hashed": 100,
                "activity": MirrorActivity(
                    running=True, http_port=8123, native_workers=1
                ),
            }
        )
    )
    journal.close()
    journal = PreservationJournal(root)
    try:
        journal.interrupt_mirror_jobs()
        retained, created = journal.reserve_mirror(original)
        assert (
            not created
            and retained.state == "interrupted"
            and retained.activity is None
        )
        assert retained.bytes_hashed == 100 and retained.error_code == "core_restarted"
        assert journal.mirror_job(retained.job_id)[1] == original
        assert journal.mirror_jobs("library") == (retained,)
        assert (
            not journal.jobs()
        )  # Existing finite copy/serving namespace is untouched.
        with pytest.raises(PreservationHostError, match="request_changed"):
            journal.reserve_mirror(original.model_copy(update={"seconds": 1800}))
    finally:
        journal.close()


def test_mirror_capacity_never_replays_or_revives_cancelled_decisions(
    tmp_path: Path,
) -> None:
    journal = PreservationJournal(tmp_path / "owner")
    try:
        first, _ = journal.reserve_mirror(request("first"))
        second, _ = journal.reserve_mirror(request("second"))
        with pytest.raises(PreservationHostError, match="capacity"):
            journal.reserve_mirror(request("third"))
        assert journal.reserve_mirror(request("first")) == (first, False)
        cancelling = first.model_copy(update={"state": "cancelling"})
        journal.update_mirror_job(cancelling)
        with pytest.raises(
            PreservationHostError, match="cancellation_already_requested"
        ):
            journal.update_mirror_job(first.model_copy(update={"state": "running"}))
        with pytest.raises(PreservationHostError, match="terminal_mirror"):
            journal.update_mirror_job(
                first.model_copy(
                    update={
                        "state": "cancelled",
                        "activity": MirrorActivity(
                            running=True, http_port=8123, native_workers=1
                        ),
                    }
                )
            )
        cancelled = first.model_copy(update={"state": "cancelled"})
        journal.update_mirror_job(cancelled)
        with pytest.raises(PreservationHostError, match="mirror_job_terminal"):
            journal.update_mirror_job(first.model_copy(update={"state": "running"}))
        with pytest.raises(PreservationHostError, match="request_changed"):
            journal.update_mirror_job(second.model_copy(update={"kind": "prune"}))
        with pytest.raises(PreservationHostError, match="invalid_offset"):
            journal.mirror_jobs(offset=4097)
        assert journal.reserve_mirror(request("first")) == (cancelled, False)
    finally:
        journal.close()


def test_mirror_decisions_reject_implicit_serving_terms_and_deletion(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="public mirror participation"):
        MirrorRequest(
            owner_id="library", operation_id="bad", kind="run", policy_id="a" * 64
        )
    with pytest.raises(ValueError, match="reviewed terms"):
        MirrorRequest(
            owner_id="library",
            operation_id="bad",
            kind="retain",
            policy_id="a" * 64,
            copy_id="copy",
            manifest_id="b" * 64,
        )
    with pytest.raises(ValueError, match="deletion requires"):
        MirrorRequest(
            owner_id="library",
            operation_id="bad",
            kind="prune",
            policy_id="a" * 64,
            manifest_id="b" * 64,
            confirm_delete_manifest="c" * 64,
        )
    with pytest.raises(ValueError, match="confirmed expiry"):
        MirrorRequest(
            owner_id="library",
            operation_id="bad",
            kind="grant_resume",
            policy_id="a" * 64,
            resume_expires=2000000000,
        )
    journal = PreservationJournal(tmp_path / "owner")
    try:
        with pytest.raises(ValueError, match="public mirror participation"):
            journal.reserve_mirror(
                request().model_copy(update={"public_participation": False})
            )
    finally:
        journal.close()
