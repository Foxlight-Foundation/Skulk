"""Audio-video generation substrate: request, progress, output, and transport types.

These pin the contracts every later engine and the job API build on: the
request parameters and how the mode is implied from attachments, the terminal
chunk that carries a manifest, the ``OUTPUT_MEDIA`` packet lifecycle and its
binary framing, and the reference-media payload marker on the vision plane.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import anyio
import pytest
from pydantic import ValidationError

from skulk.api.main import API
from skulk.api.video_jobs import MAX_RETAINED_JOBS, VideoJob, VideoJobRegistry
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
from skulk.shared.types.video import (
    VideoGenerationTaskParams,
    VideoOutputManifest,
    VideoReferenceSpec,
)
from skulk.utils.channels import channel

DIGEST = "a" * 64
MODEL = ModelId("org/video")


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
            prompt="x", model="m", seconds=5, references=(_reference(1),), reference_bytes=10
        )
    with pytest.raises(ValidationError, match="at most one first_frame"):
        VideoGenerationTaskParams(
            prompt="x",
            model="m",
            seconds=5,
            references=(_reference(0, role="first_frame"), _reference(1, role="first_frame")),
            reference_bytes=20,
        )
    with pytest.raises(ValidationError, match="reference_bytes"):
        VideoGenerationTaskParams(
            prompt="x", model="m", seconds=5, references=(_reference(0),), reference_bytes=99
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
    with pytest.raises(ValueError, match="out-of-order"):
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
    api._chunk_reorder = {}  # pyright: ignore[reportPrivateUsage]
    api._cancelled_command_ids = set()  # pyright: ignore[reportPrivateUsage]
    api._video_job_media_deadlines = {}  # pyright: ignore[reportPrivateUsage]
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
