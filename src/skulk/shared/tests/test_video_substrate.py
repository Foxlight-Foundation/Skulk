"""Audio-video generation substrate: request, progress, output, and transport types.

These pin the contracts every later engine and the job API build on: the
request parameters and how the mode is implied from attachments, the terminal
chunk that carries a manifest, the ``OUTPUT_MEDIA`` packet lifecycle and its
binary framing, and the reference-media payload marker on the vision plane.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import anyio
import pytest
from pydantic import ValidationError

from skulk.api import main as api_main
from skulk.api.main import API
from skulk.api.video_jobs import (
    MAX_ACTIVE_JOBS,
    MAX_RETAINED_JOBS,
    VideoJob,
    VideoJobRegistry,
)
from skulk.api.video_store import VideoStore
from skulk.routing import topics
from skulk.routing.output_media import (
    OutputMediaPacket,
    decode_output_media_packet,
    encode_output_media_packet,
)
from skulk.routing.vision_media import (
    VisionMediaPacket,
    decode_vision_media_packet,
    encode_vision_media_packet,
)
from skulk.shared.models.model_cards import ModelId, VideoMode
from skulk.shared.types.chunks import DataChunk, ErrorChunk, VideoChunk
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.state import State
from skulk.shared.types.video import (
    VideoGenerationTaskParams,
    VideoOutputManifest,
    VideoReferenceSpec,
)
from skulk.utils.channels import channel

DIGEST = "a" * 64
MODEL = ModelId("org/video")


def _ample_free_bytes(_store: VideoStore) -> int:
    return 1 << 40


@pytest.fixture(autouse=True)
def ample_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the store's free-space check off the host's real disk.

    The store leaves a reserve on its filesystem; a test must not depend on
    how full the machine running it happens to be.
    """

    monkeypatch.setattr(VideoStore, "_free_disk_bytes", _ample_free_bytes)


def _reference_payload(
    slot: int, kind: str = "image", role: str = "reference"
) -> dict[str, object]:
    media_type = {"image": "image/png", "video": "video/mp4", "audio": "audio/wav"}[kind]
    return {
        "slot": slot,
        "kind": kind,
        "role": role,
        "media_type": media_type,
        "size_bytes": 10,
        "sha256": DIGEST,
    }


def _reference(slot: int, kind: str = "image", role: str = "reference") -> VideoReferenceSpec:
    return VideoReferenceSpec.model_validate(_reference_payload(slot, kind, role))


def test_params_imply_mode_from_attachments() -> None:
    text = VideoGenerationTaskParams(prompt="a fox", model="org/video", seconds=5)
    assert text.implied_mode() is VideoMode.TextToAudioVideo
    frames = VideoGenerationTaskParams(
        prompt="a fox",
        model="org/video",
        seconds=5,
        references=(_reference(0, role="first_frame"), _reference(1, role="last_frame")),
        reference_bytes=20,
        total_input_chunks=2,
    )
    assert frames.implied_mode() is VideoMode.FramesToAudioVideo
    reference = VideoGenerationTaskParams(
        prompt="a fox",
        model="org/video",
        seconds=5,
        references=(_reference(0, kind="video"),),
        reference_bytes=10,
        total_input_chunks=1,
    )
    assert reference.implied_mode() is VideoMode.ReferenceToAudioVideo
    explicit = text.model_copy(update={"mode": VideoMode.FramesToAudioVideo})
    assert explicit.implied_mode() is VideoMode.FramesToAudioVideo


def test_params_reject_inconsistent_attachments() -> None:
    with pytest.raises(ValidationError, match="contiguous"):
        VideoGenerationTaskParams(
            prompt="x",
            model="m",
            seconds=5,
            references=(_reference(1),),
            reference_bytes=10,
            total_input_chunks=1,
        )
    with pytest.raises(ValidationError, match="at most one first_frame"):
        VideoGenerationTaskParams(
            prompt="x",
            model="m",
            seconds=5,
            references=(_reference(0, role="first_frame"), _reference(1, role="first_frame")),
            reference_bytes=20,
            total_input_chunks=2,
        )
    with pytest.raises(ValidationError, match="reference_bytes"):
        VideoGenerationTaskParams(
            prompt="x",
            model="m",
            seconds=5,
            references=(_reference(0),),
            reference_bytes=99,
            total_input_chunks=1,
        )
    with pytest.raises(ValidationError, match="cover every attachment"):
        VideoGenerationTaskParams(
            prompt="x", model="m", seconds=5, references=(_reference(0),), reference_bytes=10
        )
    with pytest.raises(ValidationError, match="WIDTHxHEIGHT"):
        VideoGenerationTaskParams(prompt="x", model="m", seconds=5, size="wide")
    with pytest.raises(ValidationError, match="keyframe roles"):
        VideoReferenceSpec.model_validate(_reference_payload(0, kind="audio", role="first_frame"))
    with pytest.raises(ValidationError, match="cannot carry"):
        VideoReferenceSpec.model_validate({**_reference_payload(0), "media_type": "video/mp4"})
    params = VideoGenerationTaskParams(prompt="x", model="m", seconds=5, size="1344x768")
    assert params.width_height == (1344, 768)


def test_output_manifest_pairs_optional_fields() -> None:
    base = {
        "sha256": DIGEST,
        "size_bytes": 10,
        "width": 1344,
        "height": 768,
        "frame_count": 124,
        "fps": 24,
        "seconds": 5.17,
    }
    VideoOutputManifest.model_validate(base)
    with pytest.raises(ValidationError, match="thumbnail"):
        VideoOutputManifest.model_validate({**base, "thumbnail_sha256": DIGEST})
    with pytest.raises(ValidationError, match="audio"):
        VideoOutputManifest.model_validate({**base, "audio_channels": 2})


def test_video_chunk_terminal_requires_finish_reason() -> None:
    manifest = VideoOutputManifest(
        sha256=DIGEST, size_bytes=10, width=64, height=64, frame_count=5, fps=24, seconds=0.2
    )
    with pytest.raises(ValidationError, match="finish_reason"):
        VideoChunk(model=MODEL, stage="muxing", output=manifest)
    chunk = VideoChunk(model=MODEL, stage="muxing", output=manifest, finish_reason="stop")
    frame = DataChunk(command_id=CommandId(), kind="completed", chunk=chunk, sequence=3)
    assert frame.is_terminal
    progress = VideoChunk(model=MODEL, stage="sampling", step=3, total_steps=8, progress=0.4)
    assert "preview" in repr(progress)


def _output(kind: str, sequence: int, **fields: object) -> OutputMediaPacket:
    return OutputMediaPacket.model_validate(
        {
            "source_node": NodeId("worker"),
            "target_node": NodeId("api"),
            "command_id": CommandId("cmd-1"),
            "model": MODEL,
            "purpose": "video",
            "sequence": sequence,
            "kind": kind,
            **fields,
        }
    )


def test_output_media_lifecycle_validation() -> None:
    opened = _output("opened", 0, total_chunks=2, total_bytes=6, content_type="video/mp4")
    chunk = _output("chunk", 1, data=b"abc", total_chunks=2)
    completed = _output("completed", 3, total_chunks=2, total_bytes=6, sha256=DIGEST)
    assert not opened.is_terminal and not chunk.is_terminal and completed.is_terminal
    with pytest.raises(ValidationError, match="open requires"):
        _output("opened", 0, total_chunks=2)
    with pytest.raises(ValidationError, match="must carry bytes"):
        _output("chunk", 1, total_chunks=2)
    with pytest.raises(ValidationError, match="exceeds total_chunks"):
        _output("chunk", 3, data=b"x", total_chunks=2)
    with pytest.raises(ValidationError, match="completion sequence"):
        _output("completed", 2, total_chunks=2, total_bytes=6, sha256=DIGEST)
    with pytest.raises(ValidationError, match="requires an error message"):
        _output("transport_failed", 0)
    accepted = completed.accepted()
    assert accepted.source_node == NodeId("api") and accepted.target_node == NodeId("worker")
    failure = completed.transport_failure("bad digest")
    assert failure.target_node == NodeId("worker") and failure.error_message == "bad digest"


def test_output_media_wire_roundtrip_preserves_bytes() -> None:
    chunk = _output("chunk", 1, data=bytes(range(256)) * 4, total_chunks=1)
    decoded = decode_output_media_packet(encode_output_media_packet(chunk))
    assert decoded == chunk
    assert topics.OUTPUT_MEDIA.topic in topics.TOPIC_PLANE_CENSUS
    assert topics.TOPIC_PLANE_CENSUS[topics.OUTPUT_MEDIA.topic] == topics.MessagePlane.Data
    with pytest.raises(ValueError, match="truncated"):
        decode_output_media_packet(b"\x00\x00\x00\x10abc")


def test_vision_media_reference_payload_survives_the_wire() -> None:
    packet = VisionMediaPacket(
        source_node=NodeId("api"),
        target_node=NodeId("worker"),
        command_id=CommandId("cmd-2"),
        model=MODEL,
        sequence=1,
        kind="chunk",
        data=b"\x00\xff raw bytes",
        image_index=2,
        total_chunks=3,
        payload="reference_media",
    )
    decoded = decode_vision_media_packet(encode_vision_media_packet(packet))
    assert decoded.payload == "reference_media" and decoded.data == packet.data
    legacy = VisionMediaPacket(
        source_node=NodeId("api"),
        target_node=NodeId("worker"),
        command_id=CommandId("cmd-3"),
        model=MODEL,
        sequence=0,
        kind="opened",
        total_chunks=1,
        image_count=1,
    )
    assert legacy.payload == "base64_image"
    assert legacy.accepted().payload == "base64_image"


def test_video_store_assembles_and_verifies(tmp_path: Path) -> None:
    store = VideoStore(tmp_path, default_expiry_seconds=3600)
    command_id = CommandId("job-1")
    payload = b"0123456789" * 100
    digest = hashlib.sha256(payload).hexdigest()
    store.open_assembly(
        command_id, "video", content_type="video/mp4", total_bytes=len(payload), total_chunks=2
    )
    store.append(command_id, "video", 1, payload[:500])
    with pytest.raises(ValueError, match="exceeds the declared count"):
        store.append(command_id, "video", 3, payload[500:])
    store.append(command_id, "video", 2, payload[500:])
    stored = store.commit(command_id, "video", sha256=digest, total_chunks=2)
    assert stored.file_path.read_bytes() == payload
    assert store.get(command_id) is not None
    assert not store.has_open_assembly(command_id, "video")
    # A wrong digest discards the staging file and leaves nothing behind.
    other = CommandId("job-2")
    store.open_assembly(other, "video", content_type="video/mp4", total_bytes=3, total_chunks=1)
    store.append(other, "video", 1, b"abc")
    with pytest.raises(ValueError, match="SHA-256"):
        store.commit(other, "video", sha256=DIGEST, total_chunks=1)
    assert store.get(other) is None and not list((tmp_path / str(other)).glob("*"))
    store.delete(command_id)
    assert store.get(command_id) is None and not (tmp_path / str(command_id)).exists()


def test_video_store_expires_artifacts(tmp_path: Path) -> None:
    store = VideoStore(tmp_path, default_expiry_seconds=0)
    command_id = CommandId("job-3")
    store.open_assembly(command_id, "video", content_type="video/mp4", total_bytes=1, total_chunks=1)
    store.append(command_id, "video", 1, b"x")
    store.commit(command_id, "video", sha256=hashlib.sha256(b"x").hexdigest(), total_chunks=1)
    assert store.cleanup_expired() == 1
    assert store.get(command_id) is None


def _job(identifier: str, created_at: int = 0) -> VideoJob:
    return VideoJob(
        id=CommandId(identifier),
        model="org/video",
        prompt="a fox",
        mode="t2va",
        seconds=5,
        created_at=created_at,
    )


def test_job_registry_settles_only_with_both_halves(tmp_path: Path) -> None:
    registry = VideoJobRegistry(tmp_path / "jobs.json")
    job = registry.create(_job("j1"))
    assert job.status == "queued"
    registry.update(job.id, status="in_progress", progress=40, stage="sampling")
    assert registry.settle(job.id) is not None and registry.get(job.id) is not None
    assert registry.get(job.id).status == "in_progress"  # pyright: ignore[reportOptionalMemberAccess]
    registry.update(job.id, render_finished=True)
    assert registry.settle(job.id).status == "in_progress"  # pyright: ignore[reportOptionalMemberAccess]
    registry.update(job.id, media_delivered=True)
    settled = registry.settle(job.id)
    assert settled is not None and settled.status == "completed" and settled.progress == 100
    # Terminal jobs are immutable.
    assert registry.fail(job.id, "late").status == "completed"  # pyright: ignore[reportOptionalMemberAccess]


def test_job_registry_marks_in_flight_jobs_failed_after_restart(tmp_path: Path) -> None:
    index = tmp_path / "jobs.json"
    registry = VideoJobRegistry(index)
    registry.create(_job("live"))
    registry.update(CommandId("live"), status="in_progress")
    done = registry.create(_job("done"))
    registry.update(done.id, render_finished=True, media_delivered=True)
    registry.settle(done.id)
    reloaded = VideoJobRegistry(index)
    live = reloaded.get(CommandId("live"))
    assert live is not None and live.status == "failed" and "restarted" in (live.error or "")
    assert reloaded.get(CommandId("done")).status == "completed"  # pyright: ignore[reportOptionalMemberAccess]


def test_job_registry_bounds_retained_terminal_jobs(tmp_path: Path) -> None:
    registry = VideoJobRegistry(None)
    for index in range(MAX_RETAINED_JOBS + 5):
        job = registry.create(_job(f"j{index}", created_at=index))
        registry.fail(job.id, "done")
    assert len(registry.list(limit=1000)) == MAX_RETAINED_JOBS
    assert registry.get(CommandId("j0")) is None
    newest = registry.list(limit=1)[0]
    assert newest.id == CommandId(f"j{MAX_RETAINED_JOBS + 4}")
    page = registry.list(limit=2, after=newest.id)
    assert [job.id for job in page] == [
        CommandId(f"j{MAX_RETAINED_JOBS + 3}"),
        CommandId(f"j{MAX_RETAINED_JOBS + 2}"),
    ]


def test_video_store_adopts_committed_artifacts_after_restart(tmp_path: Path) -> None:
    store = VideoStore(tmp_path, default_expiry_seconds=3600)
    command_id = CommandId("job-restart")
    payload = b"mp4" * 50
    digest = hashlib.sha256(payload).hexdigest()
    store.open_assembly(
        command_id, "video", content_type="video/mp4", total_bytes=len(payload), total_chunks=1
    )
    store.append(command_id, "video", 1, payload)
    stored = store.commit(command_id, "video", sha256=digest, total_chunks=1)
    (tmp_path / "orphan").mkdir()
    (tmp_path / "orphan" / "output.mp4").write_bytes(b"x")
    restarted = VideoStore(tmp_path, default_expiry_seconds=3600)
    assert restarted.get(command_id) is None
    assert restarted.adopt(
        command_id,
        "video",
        content_type="video/mp4",
        size_bytes=len(payload),
        sha256=digest,
        expires_at=stored.expires_at,
    )
    adopted = restarted.get(command_id)
    assert adopted is not None and adopted.file_path == stored.file_path
    assert not restarted.adopt(
        command_id,
        "video",
        content_type="video/mp4",
        size_bytes=len(payload) + 1,
        sha256=digest,
        expires_at=stored.expires_at,
    )
    assert restarted.purge_unknown({command_id}) == 1
    assert not (tmp_path / "orphan").exists() and (tmp_path / str(command_id)).exists()


def _bare_api(tmp_path: Path) -> API:
    api = object.__new__(API)
    api._video_jobs = VideoJobRegistry(None)  # pyright: ignore[reportPrivateUsage]
    api._video_store = VideoStore(tmp_path)  # pyright: ignore[reportPrivateUsage]
    api._video_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._text_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._image_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._embedding_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._audio_speech_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._audio_transcription_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._chunk_reorder = {}  # pyright: ignore[reportPrivateUsage]
    api._cancelled_command_ids = set()  # pyright: ignore[reportPrivateUsage]
    api._video_job_media_deadlines = {}  # pyright: ignore[reportPrivateUsage]
    api._video_output_sources = {}  # pyright: ignore[reportPrivateUsage]
    api._early_output_packets = {}  # pyright: ignore[reportPrivateUsage]
    api._early_output_packet_bytes = 0  # pyright: ignore[reportPrivateUsage]
    api._pending_output_completions = {}  # pyright: ignore[reportPrivateUsage]
    api._video_upload_inflight_bytes = 0  # pyright: ignore[reportPrivateUsage]
    return api


async def test_cancelled_video_job_becomes_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api = _bare_api(tmp_path)
    finalized: list[CommandId] = []

    async def finalize(command_id: CommandId, _queue_map: object) -> None:
        finalized.append(command_id)

    monkeypatch.setattr(api, "_finalize_command_stream", finalize)
    job = api._video_jobs.create(_job("cancel-me"))  # pyright: ignore[reportPrivateUsage]
    sender, receiver = channel[VideoChunk | ErrorChunk]()
    api._video_generation_queues[job.id] = sender  # pyright: ignore[reportPrivateUsage]
    api._cancelled_command_ids.add(job.id)  # pyright: ignore[reportPrivateUsage]
    async with anyio.create_task_group() as group:
        group.start_soon(api._drain_video_job, job.id, receiver)  # pyright: ignore[reportPrivateUsage]
        await sender.send(VideoChunk(model=MODEL, stage="sampling", progress=0.25))
        await anyio.sleep(0.05)
        sender.close()
    drained = api._video_jobs.get(job.id)  # pyright: ignore[reportPrivateUsage]
    assert drained is not None and drained.status == "cancelled"
    assert drained.progress == 25
    assert finalized == [job.id]


async def test_render_error_fails_the_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api = _bare_api(tmp_path)

    async def finalize(_command_id: CommandId, _queue_map: object) -> None:
        return None

    monkeypatch.setattr(api, "_finalize_command_stream", finalize)
    job = api._video_jobs.create(_job("boom"))  # pyright: ignore[reportPrivateUsage]
    sender, receiver = channel[VideoChunk | ErrorChunk]()
    api._video_generation_queues[job.id] = sender  # pyright: ignore[reportPrivateUsage]
    async with anyio.create_task_group() as group:
        group.start_soon(api._drain_video_job, job.id, receiver)  # pyright: ignore[reportPrivateUsage]
        await sender.send(ErrorChunk(model=MODEL, error_message="runner died"))
        await anyio.sleep(0.05)
        sender.close()
    failed = api._video_jobs.get(job.id)  # pyright: ignore[reportPrivateUsage]
    assert failed is not None and failed.status == "failed" and failed.error == "runner died"


def test_job_registry_counts_live_jobs() -> None:
    registry = VideoJobRegistry(None)
    for index in range(MAX_ACTIVE_JOBS):
        registry.create(_job(f"live{index}", created_at=index))
    assert registry.active_count() == MAX_ACTIVE_JOBS
    registry.fail(CommandId("live0"), "done")
    assert registry.active_count() == MAX_ACTIVE_JOBS - 1


def test_completed_job_still_accepts_declared_thumbnail_packets(tmp_path: Path) -> None:
    """The guard in _apply_output_media must let a declared thumbnail through."""
    registry = VideoJobRegistry(None)
    job = registry.create(_job("thumb"))
    manifest = VideoOutputManifest(
        sha256=DIGEST,
        size_bytes=1,
        width=64,
        height=64,
        frame_count=5,
        fps=24,
        seconds=0.2,
        thumbnail_sha256=DIGEST,
        thumbnail_size_bytes=1,
    )
    registry.update(job.id, render_finished=True, media_delivered=True, output=manifest)
    settled = registry.settle(job.id)
    assert settled is not None and settled.status == "completed"
    assert settled.output is not None and settled.output.thumbnail_sha256 is not None


async def test_terminal_frame_keeps_job_waiting_for_media(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The DATA queue closes on the terminal frame; the job must not fail then."""
    api = _bare_api(tmp_path)

    async def finalize(_command_id: CommandId, _queue_map: object) -> None:
        return None

    monkeypatch.setattr(api, "_finalize_command_stream", finalize)
    job = api._video_jobs.create(_job("wait-media"))  # pyright: ignore[reportPrivateUsage]
    manifest = VideoOutputManifest(
        sha256=DIGEST, size_bytes=10, width=64, height=64, frame_count=5, fps=24, seconds=0.2
    )
    api._close_command_queue = lambda _command_id: None  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    api.state = State()
    sender, receiver = channel[VideoChunk | ErrorChunk]()
    api._video_generation_queues[job.id] = sender  # pyright: ignore[reportPrivateUsage]
    async with anyio.create_task_group() as group:
        group.start_soon(api._drain_video_job, job.id, receiver)  # pyright: ignore[reportPrivateUsage]
        await sender.send(
            VideoChunk(model=MODEL, stage="muxing", output=manifest, finish_reason="stop")
        )
        await anyio.sleep(0.05)
        sender.close()
    waiting = api._video_jobs.get(job.id)  # pyright: ignore[reportPrivateUsage]
    assert waiting is not None and waiting.status == "in_progress"
    assert waiting.render_finished and waiting.stage == "uploading"
    assert job.id in api._video_job_media_deadlines  # pyright: ignore[reportPrivateUsage]
    store = api._video_store  # pyright: ignore[reportPrivateUsage]
    payload = b"0123456789"
    store.open_assembly(job.id, "video", content_type="video/mp4", total_bytes=10, total_chunks=1)
    store.append(job.id, "video", 1, payload)
    store.commit(job.id, "video", sha256=hashlib.sha256(payload).hexdigest(), total_chunks=1)
    api._settle_video_job(job.id)  # pyright: ignore[reportPrivateUsage]
    mismatched = api._video_jobs.get(job.id)  # pyright: ignore[reportPrivateUsage]
    assert mismatched is not None and mismatched.status == "failed"
    assert "does not match its manifest" in (mismatched.error or "")


def test_adopted_artifact_is_hashed_before_it_is_served(tmp_path: Path) -> None:
    store = VideoStore(tmp_path, default_expiry_seconds=3600)
    command_id = CommandId("tampered")
    directory = tmp_path / str(command_id)
    directory.mkdir()
    (directory / "output.mp4").write_bytes(b"replaced-bytes")
    assert store.adopt(
        command_id,
        "video",
        content_type="video/mp4",
        size_bytes=len(b"replaced-bytes"),
        sha256=DIGEST,
        expires_at=9e12,
    )
    assert store.get(command_id) is None
    assert not directory.exists()


async def test_settle_requires_every_declared_artifact_to_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api = _bare_api(tmp_path)
    api._close_command_queue = lambda _command_id: None  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    api.state = State()
    registry = api._video_jobs  # pyright: ignore[reportPrivateUsage]
    store = api._video_store  # pyright: ignore[reportPrivateUsage]
    job = registry.create(_job("declared"))
    video = b"container"
    thumb = b"jpeg"
    manifest = VideoOutputManifest(
        sha256=hashlib.sha256(video).hexdigest(),
        size_bytes=len(video),
        width=64,
        height=64,
        frame_count=5,
        fps=24,
        seconds=0.2,
        thumbnail_sha256=hashlib.sha256(thumb).hexdigest(),
        thumbnail_size_bytes=len(thumb),
        audio_sample_rate=32000,
        audio_channels=2,
    )
    registry.update(job.id, render_finished=True, output=manifest)
    store.open_assembly(job.id, "video", content_type="video/mp4", total_bytes=len(video), total_chunks=1)
    store.append(job.id, "video", 1, video)
    store.commit(job.id, "video", sha256=manifest.sha256, total_chunks=1)
    api._settle_video_job(job.id)  # pyright: ignore[reportPrivateUsage]
    pending = registry.get(job.id)
    assert pending is not None and not pending.is_terminal and not pending.media_delivered
    store.open_assembly(job.id, "thumbnail", content_type="image/jpeg", total_bytes=len(thumb), total_chunks=1)
    store.append(job.id, "thumbnail", 1, thumb)
    store.commit(job.id, "thumbnail", sha256=hashlib.sha256(thumb).hexdigest(), total_chunks=1)
    api._settle_video_job(job.id)  # pyright: ignore[reportPrivateUsage]
    done = registry.get(job.id)
    assert done is not None and done.status == "completed" and done.media_delivered


def test_video_store_reorders_early_chunks_within_a_bounded_window(tmp_path: Path) -> None:
    store = VideoStore(tmp_path, default_expiry_seconds=3600)
    command_id = CommandId("reorder")
    parts = [b"aa", b"bb", b"cc", b"dd"]
    payload = b"".join(parts)
    store.open_assembly(
        command_id, "video", content_type="video/mp4", total_bytes=len(payload), total_chunks=4
    )
    store.append(command_id, "video", 3, parts[2])
    store.append(command_id, "video", 1, parts[0])
    store.append(command_id, "video", 4, parts[3])
    store.append(command_id, "video", 4, parts[3])  # duplicate re-delivery is ignored
    store.append(command_id, "video", 2, parts[1])
    stored = store.commit(command_id, "video", sha256=hashlib.sha256(payload).hexdigest(), total_chunks=4)
    assert stored.file_path.read_bytes() == payload
    other = CommandId("gap")
    store.open_assembly(other, "video", content_type="video/mp4", total_bytes=4, total_chunks=2)
    store.append(other, "video", 2, b"zz")
    with pytest.raises(ValueError, match="still missing"):
        store.commit(other, "video", sha256=DIGEST, total_chunks=2)


def test_nested_video_contracts_are_frozen() -> None:
    spec = _reference(0)
    with pytest.raises(ValidationError):
        spec.slot = 3
    manifest = VideoOutputManifest(
        sha256=DIGEST, size_bytes=10, width=64, height=64, frame_count=5, fps=24, seconds=0.2
    )
    with pytest.raises(ValidationError):
        manifest.size_bytes = 11


def test_output_source_survives_task_deletion(tmp_path: Path) -> None:
    from skulk.shared.types.tasks import TaskId, TaskStatus
    from skulk.shared.types.tasks import VideoGeneration as VideoGenerationTask
    from skulk.shared.types.worker.instances import InstanceId

    api = _bare_api(tmp_path)
    job = api._video_jobs.create(_job("placed"))  # pyright: ignore[reportPrivateUsage]
    instance_id = InstanceId("video")
    task = VideoGenerationTask(
        task_id=TaskId(),
        command_id=job.id,
        instance_id=instance_id,
        task_status=TaskStatus.Pending,
        owner_node=NodeId("api"),
        task_params=VideoGenerationTaskParams(prompt="x", model="org/video", seconds=5),
    )
    from skulk.shared.models.model_cards import ModelCard, ModelTask
    from skulk.shared.types.memory import Memory
    from skulk.shared.types.worker.instances import MlxRingInstance
    from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
    from skulk.shared.types.worker.shards import PipelineShardMetadata

    card = ModelCard(model_id=MODEL, storage_size=Memory.from_mb(1), n_layers=1, hidden_size=1, supports_tensor=False, tasks=[ModelTask.TextGeneration])
    shard = PipelineShardMetadata(model_card=card, device_rank=0, world_size=1, start_layer=0, end_layer=1, n_layers=1)
    instance = MlxRingInstance(
        instance_id=instance_id,
        shard_assignments=ShardAssignments(model_id=MODEL, node_to_runner={NodeId("worker-1"): RunnerId("r")}, runner_to_shard={RunnerId("r"): shard}),
        hosts_by_node={},
        ephemeral_port=1,
    )
    api.state = State(instances={instance_id: instance})
    api._record_video_output_source(task)  # pyright: ignore[reportPrivateUsage]
    api.state = State()  # the task and instance are gone once the stream finalizes
    assert api._video_output_source(job.id) == NodeId("worker-1")  # pyright: ignore[reportPrivateUsage]


def test_job_registry_invalidates_a_completed_job_after_restart(tmp_path: Path) -> None:
    registry = VideoJobRegistry(None)
    job = registry.create(_job("lost"))
    registry.update(job.id, render_finished=True, media_delivered=True)
    assert registry.settle(job.id).status == "completed"  # pyright: ignore[reportOptionalMemberAccess]
    demoted = registry.invalidate(job.id, "artifacts gone")
    assert demoted is not None and demoted.status == "failed" and demoted.expires_at is None


def test_video_store_stash_is_bounded_by_bytes(tmp_path: Path) -> None:
    from skulk.api import video_store as store_module

    store = VideoStore(tmp_path, default_expiry_seconds=3600)
    command_id = CommandId("stash-bytes")
    limit = store_module._MAX_STASHED_BYTES  # pyright: ignore[reportPrivateUsage]
    chunk = b"x" * (limit // 2 + 1)
    store.open_assembly(
        command_id, "video", content_type="video/mp4", total_bytes=len(chunk) * 4, total_chunks=4
    )
    store.append(command_id, "video", 3, chunk)
    with pytest.raises(ValueError, match="reorder window"):
        store.append(command_id, "video", 4, chunk)


async def test_cancel_during_upload_phase_cancels_job_and_notifies_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _bare_api(tmp_path)
    api.node_id = NodeId("api")
    for name in (
        "_text_generation_queues",
        "_image_generation_queues",
        "_embedding_queues",
        "_audio_speech_queues",
        "_audio_transcription_queues",
    ):
        setattr(api, name, {})
    sent: list[OutputMediaPacket] = []

    async def capture(packet: OutputMediaPacket) -> None:
        sent.append(packet)

    monkeypatch.setattr(api, "_send_output_media_terminal", capture)
    job = api._video_jobs.create(_job("uploading"))  # pyright: ignore[reportPrivateUsage]
    api._video_jobs.update(job.id, render_finished=True)  # pyright: ignore[reportPrivateUsage]
    api._video_output_sources[job.id] = NodeId("worker-1")  # pyright: ignore[reportPrivateUsage]
    response = await api.cancel_command(job.id)
    assert response.command_id == job.id
    cancelled = api._video_jobs.get(job.id)  # pyright: ignore[reportPrivateUsage]
    assert cancelled is not None and cancelled.status == "cancelled"
    assert len(sent) == 1 and sent[0].kind == "cancelled" and sent[0].target_node == NodeId("worker-1")


async def test_output_frames_overtaking_their_open_are_held_and_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _bare_api(tmp_path)
    api.node_id = NodeId("api")
    api.state = State()
    api._close_command_queue = lambda _command_id: None  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    acknowledged: list[OutputMediaPacket] = []

    async def capture(packet: OutputMediaPacket) -> None:
        acknowledged.append(packet)

    monkeypatch.setattr(api, "_send_output_media_terminal", capture)
    job = api._video_jobs.create(_job("early"))  # pyright: ignore[reportPrivateUsage]
    payload = b"abcdef"
    digest = hashlib.sha256(payload).hexdigest()
    sender, receiver = channel[OutputMediaPacket]()
    api._output_media_packet_receiver = receiver  # pyright: ignore[reportPrivateUsage]

    def packet(kind: str, sequence: int, **fields: object) -> OutputMediaPacket:
        return OutputMediaPacket.model_validate(
            {
                "source_node": NodeId("worker-1"),
                "target_node": NodeId("api"),
                "command_id": job.id,
                "model": MODEL,
                "purpose": "video",
                "sequence": sequence,
                "kind": kind,
                **fields,
            }
        )

    async with anyio.create_task_group() as group:
        group.start_soon(api._apply_output_media)  # pyright: ignore[reportPrivateUsage]
        await sender.send(packet("chunk", 2, data=b"def", total_chunks=2))
        await sender.send(packet("chunk", 1, data=b"abc", total_chunks=2))
        await sender.send(packet("opened", 0, total_chunks=2, total_bytes=6, content_type="video/mp4"))
        await sender.send(packet("completed", 3, total_chunks=2, total_bytes=6, sha256=digest))
        await anyio.sleep(0.1)
        sender.close()
    stored = api._video_store.get(job.id)  # pyright: ignore[reportPrivateUsage]
    assert stored is not None and stored.file_path.read_bytes() == payload
    assert [item.kind for item in acknowledged] == ["accepted"]
    assert not api._early_output_packets  # pyright: ignore[reportPrivateUsage]


def test_settle_requires_the_requested_audio_track(tmp_path: Path) -> None:
    api = _bare_api(tmp_path)
    api._close_command_queue = lambda _command_id: None  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    registry = api._video_jobs  # pyright: ignore[reportPrivateUsage]
    store = api._video_store  # pyright: ignore[reportPrivateUsage]
    job = registry.create(_job("silent"))
    video = b"container"
    manifest = VideoOutputManifest(
        sha256=hashlib.sha256(video).hexdigest(), size_bytes=len(video), width=64, height=64, frame_count=5, fps=24, seconds=0.2
    )
    registry.update(job.id, render_finished=True, output=manifest)
    store.open_assembly(job.id, "video", content_type="video/mp4", total_bytes=len(video), total_chunks=1)
    store.append(job.id, "video", 1, video)
    store.commit(job.id, "video", sha256=manifest.sha256, total_chunks=1)
    api._settle_video_job(job.id)  # pyright: ignore[reportPrivateUsage]
    silent = registry.get(job.id)
    assert silent is not None and silent.status == "failed" and "audio track" in (silent.error or "")


def _packet_for(job_id: CommandId, kind: str, sequence: int, **fields: object) -> OutputMediaPacket:
    return OutputMediaPacket.model_validate(
        {
            "source_node": NodeId("worker-1"),
            "target_node": NodeId("api"),
            "command_id": job_id,
            "model": MODEL,
            "purpose": "video",
            "sequence": sequence,
            "kind": kind,
            **fields,
        }
    )


async def test_completion_frame_overtaking_a_chunk_is_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On the reordering fallback the completion may land before the last chunk."""

    api = _bare_api(tmp_path)
    api.node_id = NodeId("api")
    api._close_command_queue = lambda _command_id: None  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    acknowledged: list[OutputMediaPacket] = []

    async def capture(packet: OutputMediaPacket) -> None:
        acknowledged.append(packet)

    monkeypatch.setattr(api, "_send_output_media_terminal", capture)
    job = api._video_jobs.create(_job("held"))  # pyright: ignore[reportPrivateUsage]
    payload = b"abcdef"
    digest = hashlib.sha256(payload).hexdigest()
    apply = api._apply_output_media_packet  # pyright: ignore[reportPrivateUsage]
    await apply(_packet_for(job.id, "opened", 0, total_chunks=2, total_bytes=6, content_type="video/mp4"))
    await apply(_packet_for(job.id, "chunk", 2, data=b"def", total_chunks=2))
    await apply(_packet_for(job.id, "completed", 3, total_chunks=2, total_bytes=6, sha256=digest))
    assert acknowledged == []
    assert (job.id, "video") in api._pending_output_completions  # pyright: ignore[reportPrivateUsage]
    assert api._video_store.get(job.id) is None  # pyright: ignore[reportPrivateUsage]
    await apply(_packet_for(job.id, "chunk", 1, data=b"abc", total_chunks=2))
    stored = api._video_store.get(job.id)  # pyright: ignore[reportPrivateUsage]
    assert stored is not None and stored.file_path.read_bytes() == payload
    assert [item.kind for item in acknowledged] == ["accepted"]
    assert not api._pending_output_completions  # pyright: ignore[reportPrivateUsage]


async def test_early_frames_share_one_byte_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _bare_api(tmp_path)
    api.node_id = NodeId("api")
    api.state = State()
    api._close_command_queue = lambda _command_id: None  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    failures: list[OutputMediaPacket] = []

    async def capture(packet: OutputMediaPacket) -> None:
        failures.append(packet)

    monkeypatch.setattr(api, "_send_output_media_terminal", capture)
    monkeypatch.setattr(api_main, "_OUTPUT_MEDIA_EARLY_BYTES_TOTAL", 4)
    first = api._video_jobs.create(_job("first"))  # pyright: ignore[reportPrivateUsage]
    second = api._video_jobs.create(_job("second"))  # pyright: ignore[reportPrivateUsage]
    sender, receiver = channel[OutputMediaPacket]()
    api._output_media_packet_receiver = receiver  # pyright: ignore[reportPrivateUsage]
    async with anyio.create_task_group() as group:
        group.start_soon(api._apply_output_media)  # pyright: ignore[reportPrivateUsage]
        await sender.send(_packet_for(first.id, "chunk", 2, data=b"abc", total_chunks=2))
        await sender.send(_packet_for(second.id, "chunk", 2, data=b"xy", total_chunks=2))
        await anyio.sleep(0.1)
        sender.close()
    assert api._early_output_packet_bytes == 3  # pyright: ignore[reportPrivateUsage]
    assert list(api._early_output_packets) == [(first.id, "video")]  # pyright: ignore[reportPrivateUsage]
    failed = api._video_jobs.get(second.id)  # pyright: ignore[reportPrivateUsage]
    assert failed is not None and failed.status == "failed"
    assert [item.kind for item in failures] == ["transport_failed"]
    assert api._video_jobs.get(first.id).status == "queued"  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]


def test_finish_video_job_releases_everything_held(tmp_path: Path) -> None:
    api = _bare_api(tmp_path)
    closed: list[CommandId] = []
    api._close_command_queue = closed.append  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    job = api._video_jobs.create(_job("messy"))  # pyright: ignore[reportPrivateUsage]
    api._video_store.open_assembly(  # pyright: ignore[reportPrivateUsage]
        job.id, "video", content_type="video/mp4", total_bytes=6, total_chunks=2
    )
    api._video_job_media_deadlines[job.id] = 1.0  # pyright: ignore[reportPrivateUsage]
    api._video_output_sources[job.id] = NodeId("worker-1")  # pyright: ignore[reportPrivateUsage]
    api._early_output_packets[(job.id, "thumbnail")] = [  # pyright: ignore[reportPrivateUsage]
        _packet_for(job.id, "chunk", 2, data=b"abcd", total_chunks=2)
    ]
    api._early_output_packet_bytes = 4  # pyright: ignore[reportPrivateUsage]
    api._pending_output_completions[(job.id, "video")] = _packet_for(  # pyright: ignore[reportPrivateUsage]
        job.id, "completed", 3, total_chunks=2, total_bytes=6, sha256="0" * 64
    )
    finished = api._finish_video_job(job.id, "boom")  # pyright: ignore[reportPrivateUsage]
    assert finished is not None and finished.status == "failed" and finished.error == "boom"
    assert not api._video_store.has_open_assembly(job.id, "video")  # pyright: ignore[reportPrivateUsage]
    assert not (tmp_path / str(job.id)).exists()
    assert not api._video_job_media_deadlines  # pyright: ignore[reportPrivateUsage]
    assert not api._video_output_sources  # pyright: ignore[reportPrivateUsage]
    assert not api._early_output_packets  # pyright: ignore[reportPrivateUsage]
    assert api._early_output_packet_bytes == 0  # pyright: ignore[reportPrivateUsage]
    assert not api._pending_output_completions  # pyright: ignore[reportPrivateUsage]
    assert closed == [job.id]


def _commit(store: VideoStore, identifier: str, payload: bytes) -> None:
    command_id = CommandId(identifier)
    store.open_assembly(
        command_id, "video", content_type="video/mp4", total_bytes=len(payload), total_chunks=1
    )
    store.append(command_id, "video", 1, payload)
    store.commit(command_id, "video", sha256=hashlib.sha256(payload).hexdigest(), total_chunks=1)


def test_video_store_evicts_oldest_completed_jobs_to_fit(tmp_path: Path) -> None:
    store = VideoStore(tmp_path, max_total_bytes=10)
    _commit(store, "oldest", b"123456")
    _commit(store, "newer", b"abc")
    assert store.reserved_bytes() == 9
    evicted = store.open_assembly(
        CommandId("incoming"), "video", content_type="video/mp4", total_bytes=6, total_chunks=1
    )
    assert evicted == (CommandId("oldest"),)
    assert store.get(CommandId("oldest")) is None and store.get(CommandId("newer")) is not None
    assert not (tmp_path / "oldest").exists()
    # An assembly in flight is never a victim, and an artifact that cannot
    # fit even with every finished job gone is refused before anything is
    # deleted, so the finished job survives the refusal.
    with pytest.raises(ValueError, match="no room"):
        store.open_assembly(
            CommandId("huge"), "video", content_type="video/mp4", total_bytes=20, total_chunks=1
        )
    assert store.has_open_assembly(CommandId("incoming"), "video")
    assert store.get(CommandId("newer")) is not None


def test_video_store_evicts_as_many_jobs_as_the_artifact_needs(tmp_path: Path) -> None:
    store = VideoStore(tmp_path, max_total_bytes=12)
    _commit(store, "oldest", b"12")
    _commit(store, "middle", b"345")
    _commit(store, "newest", b"6789")
    # Nine bytes held under a twelve-byte ceiling; eight more need both older
    # jobs gone, and each deletion counts exactly once against the ceiling.
    evicted = store.open_assembly(
        CommandId("incoming"), "video", content_type="video/mp4", total_bytes=8, total_chunks=1
    )
    assert evicted == (CommandId("oldest"), CommandId("middle"))
    assert store.get(CommandId("newest")) is not None
    assert store.reserved_bytes() == 4 + 8


def test_video_store_never_evicts_a_kept_job(tmp_path: Path) -> None:
    store = VideoStore(tmp_path, max_total_bytes=10)
    _commit(store, "between-halves", b"123456")
    _commit(store, "finished", b"abc")
    evicted = store.open_assembly(
        CommandId("incoming"),
        "video",
        content_type="video/mp4",
        total_bytes=4,
        total_chunks=1,
        keep={CommandId("between-halves")},
    )
    assert evicted == (CommandId("finished"),)
    assert store.get(CommandId("between-halves")) is not None
    with pytest.raises(ValueError, match="no room"):
        store.open_assembly(
            CommandId("more"),
            "video",
            content_type="video/mp4",
            total_bytes=4,
            total_chunks=1,
            keep={CommandId("between-halves")},
        )
    assert store.get(CommandId("between-halves")) is not None


def test_video_store_counts_open_assemblies_against_free_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = VideoStore(tmp_path, max_total_bytes=10**9)
    reserve = 1024 * 1024 * 1024
    monkeypatch.setattr(store, "_free_disk_bytes", lambda: reserve + 100)
    store.open_assembly(
        CommandId("first"), "video", content_type="video/mp4", total_bytes=60, total_chunks=1
    )
    # The first assembly's unwritten 60 bytes are promised; only 40 remain
    # above the reserve, so a second 60-byte artifact does not fit.
    with pytest.raises(ValueError, match="no room"):
        store.open_assembly(
            CommandId("second"), "video", content_type="video/mp4", total_bytes=60, total_chunks=1
        )
    store.append(CommandId("first"), "video", 1, b"x" * 60)
    monkeypatch.setattr(store, "_free_disk_bytes", lambda: reserve + 100)
    assert store.open_assembly(
        CommandId("second"), "video", content_type="video/mp4", total_bytes=60, total_chunks=1
    ) == ()


def test_job_registry_lists_active_ids() -> None:
    registry = VideoJobRegistry(None)
    live = registry.create(_job("live"))
    done = registry.create(_job("done", created_at=1))
    registry.fail(done.id, "x")
    assert registry.active_ids() == frozenset({live.id})


def test_job_registry_marks_completed_jobs_expired() -> None:
    registry = VideoJobRegistry(None)
    live = registry.create(_job("live"))
    assert registry.mark_expired(live.id) is live
    done = registry.create(_job("done", created_at=1))
    registry.update(done.id, render_finished=True, media_delivered=True, expires_at=10**10)
    registry.settle(done.id)
    expired = registry.mark_expired(done.id)
    assert expired is not None and expired.status == "completed"
    assert expired.expires_at is not None and expired.expires_at <= int(time.time())
