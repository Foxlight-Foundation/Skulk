"""Durable request ownership, cancellation intent and restart reconciliation."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from skulk.store.preservation_contracts import PreservationPlan, PreservationStart
from skulk.store.preservation_journal import PreservationHostError, PreservationJournal


def _plan(journal: PreservationJournal, *, sharing: bool = False) -> PreservationStart:
    plan = PreservationPlan(
        plan_id="a" * 64,
        kind="torrent_share" if sharing else "preserve",
        model_id="example/base",
        location_id="device",
        size_bytes=100,
        members=("example/base", "example/companion"),
        member_count=2,
        expires_at=int(time.time()) + 600,
    )
    journal.add_plan(plan, {"generation": "b" * 64})
    return PreservationStart(plan_id=plan.plan_id, operation_id="original")


def test_restart_retains_idempotency_and_never_replays(tmp_path: Path) -> None:
    root = tmp_path / "private"
    journal = PreservationJournal(root)
    request = _plan(journal)
    job, created = journal.reserve(request)
    assert created
    running = job.model_copy(
        update={"state": "running", "attempts": 1, "expected_manifest_id": "b" * 64}
    )
    journal.update_job(running)
    journal.close()
    restarted = PreservationJournal(root)
    restarted.interrupt_active()
    retained, created = restarted.reserve(request)
    assert not created
    assert retained.state == "interrupted"
    assert retained.attempts == 1
    assert retained.expected_manifest_id == "b" * 64
    assert restarted.jobs() == (retained,)
    restarted.close()


def test_changed_input_and_retired_operation_never_reexecute(tmp_path: Path) -> None:
    journal = PreservationJournal(tmp_path / "private")
    request = _plan(journal)
    job, _ = journal.reserve(request)
    with pytest.raises(PreservationHostError, match="request_changed"):
        journal.reserve(request.model_copy(update={"allow_repository_code": True}))
    with sqlite3.connect(journal.root / "journal.sqlite3") as db:
        _ = db.execute("DELETE FROM jobs WHERE id=?", (job.job_id,))
    with pytest.raises(PreservationHostError, match="operation_retired"):
        journal.reserve(request)
    journal.close()


def test_public_swarm_needs_exact_explicit_consent(tmp_path: Path) -> None:
    journal = PreservationJournal(tmp_path / "private")
    request = _plan(journal, sharing=True)
    with pytest.raises(PreservationHostError, match="public_participation_required"):
        journal.reserve(request)
    accepted, created = journal.reserve(
        request.model_copy(update={"public_participation": True})
    )
    assert created and accepted.state == "queued"
    journal.close()


def test_exclusive_owner_terminal_state_and_cancel_intent(tmp_path: Path) -> None:
    root = tmp_path / "private"
    journal = PreservationJournal(root)
    with pytest.raises(PreservationHostError, match="already_running"):
        PreservationJournal(root)
    job, _ = journal.reserve(_plan(journal))
    cancelling = job.model_copy(update={"state": "cancelling"})
    journal.update_job(cancelling)
    with pytest.raises(PreservationHostError, match="cancellation_already_requested"):
        journal.update_job(job.model_copy(update={"state": "running"}))
    cancelled = cancelling.model_copy(update={"state": "cancelled"})
    journal.update_job(cancelled)
    with pytest.raises(PreservationHostError, match="already_terminal"):
        journal.update_job(cancelling)
    journal.close()
    with pytest.raises(PreservationHostError, match="host_closed"):
        journal.job(job.job_id)


def test_failed_initialization_releases_owner(tmp_path: Path) -> None:
    root = tmp_path / "private"
    first = PreservationJournal(root)
    first.close()
    with sqlite3.connect(root / "journal.sqlite3") as db:
        _ = db.execute("PRAGMA user_version=999")
    with pytest.raises(PreservationHostError, match="unsupported_journal_version"):
        PreservationJournal(root)
    with sqlite3.connect(root / "journal.sqlite3") as db:
        _ = db.execute("PRAGMA user_version=1")
    second = PreservationJournal(root)
    second.close()
