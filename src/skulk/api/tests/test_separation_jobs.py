"""Actual durable metadata, partial delivery, cancellation and storage uncertainty."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from skulk.api import separation_jobs
from skulk.api.separation_jobs import (
    MAX_RETAINED_SEPARATION_JOBS,
    SeparationJob,
    SeparationJobCapacityError,
    SeparationJobRegistry,
    SeparationJobStorageError,
)
from skulk.api.types import SeparationResource, separation_resource
from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.audio_separation import (
    AudioSeparationOutput,
    SeparatedStem,
    SeparationSource,
)
from skulk.shared.types.common import CommandId, NodeId
from skulk.utils.runtime_files import write_private


def _job(identifier: str = "job", **changes: object) -> SeparationJob:
    source = SeparationSource(
        sha256="a" * 64,
        size_bytes=16044,
        frames=8000,
        sample_rate=8000,
        channels=1,
        bit_depth=16,
        encoding="pcm",
    )
    return SeparationJob.model_validate(
        {"id": identifier, "model": "demucs", "source": source, "created_at": 100}
        | changes
    )


def _output(job: SeparationJob) -> AudioSeparationOutput:
    return AudioSeparationOutput(
        source=job.source,
        common_gain=1.0,
        pre_gain_peak=0.2,
        stems=tuple(
            SeparatedStem(
                name=name,
                size_bytes=264644,
                sha256=str(index) * 64,
                sample_rate=44100,
                frames=44100,
                peak=0.2,
            )
            for index, name in enumerate(("drums", "bass", "other", "vocals"), start=1)
        ),
    )


def _ready(
    registry: SeparationJobRegistry, job: SeparationJob
) -> AudioSeparationOutput:
    registry.create(job)
    registry.bind_worker(job.id, NodeId("worker"))
    output = _output(job)
    registry.record_manifest(job.id, NodeId("worker"), output)
    return output


def _complete(
    registry: SeparationJobRegistry, job: SeparationJob, *, expires_at: int = 1000
) -> SeparationJob:
    output = _ready(registry, job)
    registry.finish_render(job.id, NodeId("worker"))
    for stem in output.stems:
        registry.record_verified_stem(
            job.id, NodeId("worker"), stem, expires_at=expires_at
        )
    result = registry.settle(job.id)
    assert result is not None and result.status == "completed"
    return result


@pytest.mark.parametrize("render_first", [True, False])
def test_only_the_complete_verified_set_becomes_visible(
    tmp_path: Path, render_first: bool
) -> None:
    """The terminal DATA frame may race all four media streams in either direction."""
    registry = SeparationJobRegistry(
        tmp_path / "state" / "jobs.json", clock=lambda: 200
    )
    job = _job()
    output = _ready(registry, job)
    try:
        assert registry.active_ids() == frozenset({job.id})
        if render_first:
            registry.finish_render(job.id, NodeId("worker"))
        for index, stem in enumerate(reversed(output.stems)):
            registry.record_verified_stem(
                job.id, NodeId("worker"), stem, expires_at=1000 - index
            )
            current = registry.settle(job.id)
            assert current is not None
            if index < 3 or not render_first:
                assert current.status == "in_progress" and current.output is None
        registry.finish_render(job.id, NodeId("worker"))
        completed = registry.settle(job.id)
        assert (
            completed is not None
            and completed.output == output
            and completed.expires_at == 997
        )
        assert completed.completed_at == 200
        assert not registry.active_ids()
        assert registry.fail(job.id, "late cancellation", cancelled=True) == completed
        registry.create(job)  # Replayed original intent cannot reset an outcome.
        assert registry.get(job.id) == completed
    finally:
        registry.close()
    reopened = SeparationJobRegistry(
        tmp_path / "state" / "jobs.json", clock=lambda: 250
    )
    try:
        assert reopened.get(job.id) == completed
    finally:
        reopened.close()


@pytest.mark.parametrize("cancelled", [False, True])
def test_failure_discards_partial_visibility_and_late_frames_cannot_revive_it(
    tmp_path: Path, cancelled: bool
) -> None:
    """A late fourth stem/render completion cannot undo terminal cancellation."""
    registry = SeparationJobRegistry(
        tmp_path / "state" / "jobs.json", clock=lambda: 200
    )
    job = _job()
    output = _ready(registry, job)
    try:
        registry.record_verified_stem(
            job.id, NodeId("worker"), output.stems[0], expires_at=1000
        )
        failed = registry.fail(job.id, "owner stopped", cancelled=cancelled)
        assert failed is not None and failed.status == (
            "cancelled" if cancelled else "failed"
        )
        assert (
            failed.output is None
            and failed.candidate_output is None
            and not failed.verified_stems
        )
        for stem in output.stems:
            assert (
                registry.record_verified_stem(
                    job.id, NodeId("worker"), stem, expires_at=1000
                )
                == failed
            )
        assert registry.finish_render(job.id, NodeId("worker")) == failed
        assert registry.record_manifest(job.id, NodeId("worker"), output) == failed
        assert registry.settle(job.id) == failed
    finally:
        registry.close()


def test_interrupted_partial_work_fails_durably_after_restart(tmp_path: Path) -> None:
    """Restart never promotes files or metadata left by an interrupted transfer."""
    path = tmp_path / "state" / "jobs.json"
    registry = SeparationJobRegistry(path, clock=lambda: 200)
    job = _job()
    output = _ready(registry, job)
    registry.finish_render(job.id, NodeId("worker"))
    registry.record_verified_stem(
        job.id, NodeId("worker"), output.stems[0], expires_at=1000
    )
    registry.close()
    recovered = SeparationJobRegistry(path, clock=lambda: 300)
    try:
        failed = recovered.get(job.id)
        assert (
            failed is not None
            and failed.status == "failed"
            and failed.completed_at == 300
        )
        assert (
            failed.output is None
            and failed.candidate_output is None
            and not failed.verified_stems
        )
        assert "restarted" in (failed.error or "")
        assert (
            SeparationJob.model_validate_json(
                json.dumps(json.loads(path.read_text())["jobs"][0])
            )
            == failed
        )
    finally:
        recovered.close()


def test_foreign_worker_source_and_changed_stem_are_refused_without_mutation(
    tmp_path: Path,
) -> None:
    """A valid manifest cannot substitute a producer, original, or artifact."""
    registry = SeparationJobRegistry(
        tmp_path / "state" / "jobs.json", clock=lambda: 200
    )
    job = _job()
    output = _ready(registry, job)
    try:
        for effect in (
            lambda: registry.bind_worker(job.id, NodeId("other")),
            lambda: registry.record_manifest(job.id, NodeId("other"), output),
            lambda: registry.record_manifest(
                job.id,
                NodeId("worker"),
                _output(
                    _job(source=job.source.model_copy(update={"sha256": "f" * 64}))
                ),
            ),
            lambda: registry.record_verified_stem(
                job.id, NodeId("other"), output.stems[0], expires_at=1000
            ),
            lambda: registry.record_verified_stem(
                job.id,
                NodeId("worker"),
                output.stems[0].model_copy(update={"sha256": "f" * 64}),
                expires_at=1000,
            ),
            lambda: registry.record_verified_stem(
                job.id, NodeId("worker"), output.stems[0], expires_at=200
            ),
        ):
            before = registry.get(job.id)
            with pytest.raises(ValueError):
                effect()
            assert registry.get(job.id) == before
        assert registry.get(CommandId("missing")) is None
    finally:
        registry.close()


def test_exclusive_owner_and_closed_instance_fences(tmp_path: Path) -> None:
    """A second owner cannot overwrite a live index or keep using a closed one."""
    path = tmp_path / "state" / "jobs.json"
    registry = SeparationJobRegistry(path)
    with pytest.raises(BlockingIOError):
        SeparationJobRegistry(path)
    registry.close()
    with pytest.raises(SeparationJobStorageError):
        registry.get(CommandId("missing"))
    replacement = SeparationJobRegistry(path)
    replacement.close()


@pytest.mark.parametrize("after_replace", [False, True])
@pytest.mark.parametrize("at_completion", [False, True])
def test_uncertain_write_fences_memory_and_reopen_observes_actual_durable_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    after_replace: bool,
    at_completion: bool,
) -> None:
    """Neither disk-full nor a failed post-replace sync can acknowledge success."""
    path = tmp_path / "state" / "jobs.json"
    registry = SeparationJobRegistry(path, clock=lambda: 200)
    job = _job()
    if at_completion:
        output = _ready(registry, job)
        registry.finish_render(job.id, NodeId("worker"))
        for stem in output.stems:
            registry.record_verified_stem(
                job.id, NodeId("worker"), stem, expires_at=1000
            )
    original_write = separation_jobs.write_private

    def fail_write(target: Path, content: bytes) -> None:
        if after_replace:
            original_write(target, content)
        raise OSError("injected storage failure")

    monkeypatch.setattr(separation_jobs, "write_private", fail_write)
    with pytest.raises(SeparationJobStorageError):
        if at_completion:
            registry.settle(job.id)
        else:
            registry.create(job)
    with pytest.raises(SeparationJobStorageError):
        registry.list()
    registry.close()
    monkeypatch.setattr(separation_jobs, "write_private", original_write)
    reopened = SeparationJobRegistry(path, clock=lambda: 300)
    try:
        retained = reopened.get(CommandId("job"))
        if after_replace and at_completion:
            assert (
                retained is not None
                and retained.status == "completed"
                and retained.output == _output(job)
            )
        elif after_replace or at_completion:
            assert (
                retained is not None
                and retained.status == "failed"
                and "restarted" in (retained.error or "")
            )
        else:
            assert retained is None
    finally:
        reopened.close()


def test_active_capacity_is_bounded_and_completed_capacity_is_not_silently_evicted(
    tmp_path: Path,
) -> None:
    """Refusal preserves live work and all unexpired complete stem receipts."""
    path = tmp_path / "state" / "jobs.json"
    registry = SeparationJobRegistry(path, clock=lambda: 200)
    try:
        for index in range(4):
            registry.create(_job(str(index)))
        before = registry.list()
        with pytest.raises(SeparationJobCapacityError, match="four"):
            registry.create(_job("fifth"))
        assert registry.list() == before
        for index in range(4):
            registry.fail(CommandId(str(index)), "finished")
            registry.delete(CommandId(str(index)))
        for index in range(MAX_RETAINED_SEPARATION_JOBS):
            _complete(registry, _job(str(index)))
        with pytest.raises(SeparationJobCapacityError, match="expiry"):
            registry.create(_job("extra"))
        assert len(registry.list(limit=64)) == 64
    finally:
        registry.close()
    expired = SeparationJobRegistry(path, clock=lambda: 1001)
    try:
        removed = expired.create(_job("new", created_at=1001))
        assert len(removed) == 1 and expired.get(removed[0]) is None
        assert len(expired.list(limit=64)) == 64
    finally:
        expired.close()


def test_cursor_deletion_and_failed_content_adoption(tmp_path: Path) -> None:
    """Metadata supports bounded navigation and explicit corrupt-set invalidation."""
    registry = SeparationJobRegistry(
        tmp_path / "state" / "jobs.json", clock=lambda: 200
    )
    try:
        first, second = _job("first"), _job("second", created_at=101)
        _complete(registry, first)
        registry.create(second)
        assert tuple(job.id for job in registry.list(limit=1)) == (second.id,)
        assert tuple(job.id for job in registry.list(after=second.id)) == (first.id,)
        for limit in (0, 65):
            with pytest.raises(ValueError):
                registry.list(limit=limit)
        with pytest.raises(ValueError):
            registry.list(after=CommandId("missing"))
        with pytest.raises(ValueError, match="cancel"):
            registry.delete(second.id)
        failed = registry.invalidate(first.id, "one stem missing after restart")
        assert (
            failed is not None and failed.output is None and failed.status == "failed"
        )
        assert registry.delete(first.id) == failed
    finally:
        registry.close()


@pytest.mark.parametrize("mutation", ["partial", "duplicate", "version", "unknown"])
def test_invalid_index_fails_closed_and_releases_ownership(
    tmp_path: Path, mutation: str
) -> None:
    """Corrupt records are not silently skipped or converted into valid completion."""
    path = tmp_path / "state" / "jobs.json"
    path.parent.mkdir(mode=0o700)
    job = _job().model_dump(mode="json")
    index: dict[str, object] = {"version": 1, "jobs": [job]}
    if mutation == "partial":
        job.update(status="completed", completed_at=200)
    elif mutation == "duplicate":
        index["jobs"] = [job, job]
    elif mutation == "version":
        index["version"] = 2
    else:
        index["audio"] = "not metadata"
    write_private(path, json.dumps(index).encode())
    with pytest.raises(ValidationError):
        SeparationJobRegistry(path)
    path.unlink()
    replacement = SeparationJobRegistry(path)
    replacement.close()


def test_runtime_models_refuse_a_partial_completed_state() -> None:
    """Strict lifecycle validation also protects direct internal construction."""
    with pytest.raises(ValidationError):
        _job(status="completed", completed_at=200)
    for identifier in ("../other", "a/b", "a\\b", "a" * 129):
        with pytest.raises(ValidationError):
            _job(identifier)
    with pytest.raises(ValidationError):
        SeparationJob(
            id=CommandId("job"),
            model=ModelId("demucs"),
            source=_job().source,
            created_at=100,
            candidate_output=_output(_job()),
            verified_stems=frozenset({"vocals"}),
            expires_at=1000,
            status="completed",
            completed_at=200,
        )


def test_public_projection_hides_pending_content_and_internal_worker_facts(
    tmp_path: Path,
) -> None:
    """A direct JSON response never exposes partially retained output metadata."""
    registry = SeparationJobRegistry(
        tmp_path / "state" / "jobs.json", clock=lambda: 200
    )
    job = _job()
    output = _ready(registry, job)
    try:
        registry.record_verified_stem(
            job.id, NodeId("worker"), output.stems[0], expires_at=1000
        )
        current = registry.get(job.id)
        assert current is not None
        resource = separation_resource(current)
        wire = resource.model_dump(mode="json")
        assert resource.output is None and resource.expires_at is None
        assert not {
            "candidate_output",
            "producer_node",
            "verified_stems",
            "render_finished",
        }.intersection(wire)
        with pytest.raises(ValidationError):
            SeparationResource.model_validate(
                resource.model_dump() | {"output": output}
            )
        for stem in output.stems:
            registry.record_verified_stem(
                job.id, NodeId("worker"), stem, expires_at=1000
            )
        registry.finish_render(job.id, NodeId("worker"))
        completed = registry.settle(job.id)
        assert completed is not None
        public = separation_resource(completed)
        assert public.output == output and public.expires_at == 1000
        assert (
            SeparationResource.model_validate_json(public.model_dump_json()) == public
        )
    finally:
        registry.close()
