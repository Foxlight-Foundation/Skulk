"""Durable music editing never confuses native completion with delivered audio."""

from pathlib import Path

import pytest

from skulk.api.music_edit_jobs import (
    MusicEditCapacityError,
    MusicEditJob,
    MusicEditJobRegistry,
    MusicEditStorageError,
)
from skulk.shared.tests.native_music_fixtures import native_music_fixture
from skulk.shared.types.audio_separation import SeparationSource
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.music_edits import (
    MusicEditingOutput,
    MusicRepaintRequest,
    resolve_music_edit,
)
from skulk.shared.types.native_music import (
    NativeMusicExecution,
    NativeMusicResult,
    NativeMusicWav,
)
from skulk.utils.runtime_files import write_private


def _job(identifier: str = "edit") -> MusicEditJob:
    card, _ = native_music_fixture()
    source = SeparationSource(
        sha256="a" * 64, size_bytes=16044, sample_rate=8000, channels=1,
        frames=8000, encoding="pcm", bit_depth=16,
    )
    params = resolve_music_edit(
        MusicRepaintRequest(
            task="repaint", model=str(card.model_id), prompt="A warm cello response",
            source_sample_rate=8000, region_end_frame=8000,
        ), source, card, resolved_seed=0, total_input_chunks=1,
    )
    return MusicEditJob(id=CommandId(identifier), params=params, created_at=100)


def _output(job: MusicEditJob) -> MusicEditingOutput:
    return MusicEditingOutput(
        source=job.params.source,
        result=NativeMusicResult(
            request_id=str(job.id), model_identity="b" * 64, payload_sha256="c" * 64,
            operation="repaint", seed=0, source_sha256=job.params.source.sha256,
            conditioning_sha256="d" * 64, conditioning_gain=1.0,
            execution=NativeMusicExecution(
                device="cpu", precision="float32", text_encoder_device="cpu",
                text_encoder_precision="float32", autoencoder_device="cpu", autoencoder_precision="float32",
            ),
            output=NativeMusicWav(sha256="e" * 64, size_bytes=288044, frames=48000, peak=0.2, gain=1.0),
            generation_seconds=1.0,
        ),
    )


def _ready(registry: MusicEditJobRegistry, job: MusicEditJob) -> MusicEditingOutput:
    registry.create(job)
    registry.bind_worker(job.id, NodeId("producer"))
    output = _output(job)
    registry.record_manifest(job.id, NodeId("producer"), output)
    return output


@pytest.mark.parametrize("render_first", [False, True])
def test_completion_requires_render_synced_wav_and_return_acknowledgement(
    tmp_path: Path, render_first: bool,
) -> None:
    """Neither DATA/media ordering can bypass the final acknowledged custody gate."""
    path = tmp_path / "jobs.json"
    registry = MusicEditJobRegistry(path, clock=lambda: 200)
    job = _job()
    output = _ready(registry, job)
    try:
        if render_first:
            registry.finish_render(job.id, NodeId("producer"))
        registry.record_verified_output(job.id, NodeId("producer"), output, expires_at=1000)
        pending = registry.settle(job.id)
        assert pending is not None and pending.output is None
        if not render_first:
            registry.finish_render(job.id, NodeId("producer"))
        pending = registry.settle(job.id)
        assert pending is not None and pending.status == "in_progress"
        registry.acknowledge_return(job.id, NodeId("producer"))
        complete = registry.settle(job.id)
        assert complete is not None and complete.output == output and complete.completed_at == 200
        assert not registry.active_ids()
        assert registry.fail(job.id, "late cancel", cancelled=True) == complete
        assert registry.create(job) == () and registry.get(job.id) == complete
    finally:
        registry.close()
    reopened = MusicEditJobRegistry(path, clock=lambda: 300)
    try:
        assert reopened.get(job.id) == complete
    finally:
        reopened.close()


@pytest.mark.parametrize("cancelled", [False, True])
def test_terminal_failure_clears_partial_visibility_and_refuses_late_frames(
    tmp_path: Path, cancelled: bool,
) -> None:
    """A cancellation racing the final WAV or render cannot revive a failed job."""
    registry = MusicEditJobRegistry(tmp_path / "jobs.json", clock=lambda: 200)
    job = _job()
    output = _ready(registry, job)
    try:
        registry.record_verified_output(job.id, NodeId("producer"), output, expires_at=1000)
        failed = registry.fail(job.id, "Stopped", cancelled=cancelled)
        assert failed is not None and failed.status == ("cancelled" if cancelled else "failed")
        assert failed.output is None and failed.candidate_output is None
        assert not failed.verified_output and failed.expires_at is None
        assert registry.record_manifest(job.id, NodeId("producer"), output) == failed
        assert registry.finish_render(job.id, NodeId("producer")) == failed
        assert registry.record_verified_output(job.id, NodeId("producer"), output, expires_at=1000) == failed
        assert registry.acknowledge_return(job.id, NodeId("producer")) == failed
        assert registry.settle(job.id) == failed
    finally:
        registry.close()


def test_restart_retains_seed_and_source_but_interrupts_unsettled_edit(tmp_path: Path) -> None:
    """Restart cannot silently recreate, resubmit or rerandomize accepted work."""
    path = tmp_path / "jobs.json"
    registry = MusicEditJobRegistry(path, clock=lambda: 200)
    job = _job()
    _ready(registry, job)
    registry.finish_render(job.id, NodeId("producer"))
    registry.close()
    reopened = MusicEditJobRegistry(path, clock=lambda: 300)
    try:
        failed = reopened.get(job.id)
        assert failed is not None and failed.status == "failed"
        assert failed.params == job.params and failed.output is None
        assert failed.completed_at == 300
        assert reopened.create(job) == () and reopened.get(job.id) == failed
    finally:
        reopened.close()


@pytest.mark.parametrize("change", ["worker", "request", "seed", "source"])
def test_other_request_worker_seed_or_source_cannot_attach_a_result(
    tmp_path: Path, change: str,
) -> None:
    """A plausible WAV receipt must still belong to this exact accepted edit."""
    registry = MusicEditJobRegistry(tmp_path / "jobs.json", clock=lambda: 200)
    job = _job()
    registry.create(job)
    registry.bind_worker(job.id, NodeId("producer"))
    output = _output(job)
    producer = NodeId("producer")
    if change == "worker":
        producer = NodeId("other")
    elif change == "request":
        output = output.model_copy(update={"result": output.result.model_copy(update={"request_id": "other"})})
    elif change == "seed":
        output = output.model_copy(update={"result": output.result.model_copy(update={"seed": 1})})
    else:
        source = output.source.model_copy(update={"sha256": "f" * 64})
        output = output.model_copy(update={"source": source, "result": output.result.model_copy(update={"source_sha256": source.sha256})})
    try:
        with pytest.raises(ValueError):
            registry.record_manifest(job.id, producer, output)
        retained = registry.get(job.id)
        assert retained is not None and retained.candidate_output is None
    finally:
        registry.close()


@pytest.mark.parametrize("after_replace", [False, True])
def test_uncertain_write_fences_all_further_reads_until_reopen(
    tmp_path: Path, after_replace: bool,
) -> None:
    """Before/after-rename storage failures cannot leave stale in-memory authority."""
    path = tmp_path / "jobs.json"
    failed = False

    def write(target: Path, data: bytes) -> None:
        if not failed or after_replace:
            write_private(target, data)
        if failed:
            raise OSError("Injected durable job write failure")

    registry = MusicEditJobRegistry(path, clock=lambda: 200, write=write)
    job = _job()
    registry.create(job)
    failed = True
    with pytest.raises(MusicEditStorageError):
        registry.bind_worker(job.id, NodeId("producer"))
    with pytest.raises(MusicEditStorageError):
        registry.get(job.id)
    with pytest.raises(MusicEditStorageError):
        registry.list()
    registry.close()
    reopened = MusicEditJobRegistry(path, clock=lambda: 300)
    try:
        actual = reopened.get(job.id)
        assert actual is not None and actual.status == "failed" and actual.params == job.params
        assert actual.producer_node == (NodeId("producer") if after_replace else None)
    finally:
        reopened.close()


def test_active_admission_and_pagination_do_not_discard_existing_intents(tmp_path: Path) -> None:
    """A fifth edit is refused until a prior job is terminal; IDs remain paginatable."""
    registry = MusicEditJobRegistry(tmp_path / "jobs.json", clock=lambda: 200)
    jobs = [_job(str(index)) for index in range(5)]
    try:
        for job in jobs[:4]:
            registry.create(job)
        with pytest.raises(MusicEditCapacityError):
            registry.create(jobs[4])
        assert registry.get(jobs[4].id) is None and len(registry.active_ids()) == 4
        with pytest.raises(ValueError, match="Cancel"):
            registry.delete(jobs[0].id)
        registry.fail(jobs[0].id, "Stopped", cancelled=True)
        registry.create(jobs[4])
        page = registry.list(limit=2)
        assert [job.id for job in page] == [CommandId("4"), CommandId("3")]
        assert [job.id for job in registry.list(after=page[-1].id)] == [CommandId("2"), CommandId("1"), CommandId("0")]
        with pytest.raises(ValueError, match="cursor"):
            registry.list(after=CommandId("missing"))
    finally:
        registry.close()
