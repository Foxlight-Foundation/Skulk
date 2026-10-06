"""Worker-side handling of video reference media and finished output files."""

from __future__ import annotations

import hashlib
from pathlib import Path

from skulk.routing.vision_media import VisionMediaPacket
from skulk.shared.types.common import CommandId, ModelId, NodeId
from skulk.shared.types.events import TaskFailed
from skulk.shared.types.tasks import TaskId, TaskStatus
from skulk.shared.types.tasks import VideoGeneration as VideoGenerationTask
from skulk.shared.types.video import VideoGenerationTaskParams, VideoReferenceSpec
from skulk.shared.types.worker.instances import InstanceId
from skulk.worker.main import (
    Worker,
    _inject_reference_paths,  # pyright: ignore[reportPrivateUsage]
    _purge_stale_video_directories,  # pyright: ignore[reportPrivateUsage]
    _reference_media_extension,  # pyright: ignore[reportPrivateUsage]
    _verify_file_digest,  # pyright: ignore[reportPrivateUsage]
    _vision_media_cleanup_command_id,  # pyright: ignore[reportPrivateUsage]
    _write_reference_media,  # pyright: ignore[reportPrivateUsage]
)


def _spec(slot: int, media_type: str, kind: str) -> VideoReferenceSpec:
    return VideoReferenceSpec(
        slot=slot,
        kind=kind,  # pyright: ignore[reportArgumentType]
        media_type=media_type,
        size_bytes=3,
        sha256=hashlib.sha256(b"abc").hexdigest(),
    )


def test_reference_extension_follows_media_type() -> None:
    assert _reference_media_extension(_spec(0, "image/png", "image")) == ".png"
    assert _reference_media_extension(_spec(0, "video/mp4", "video")) == ".mp4"
    assert _reference_media_extension(_spec(0, "audio/x-unknown-type", "audio")) == ".audio"


def test_reference_media_is_written_atomically_per_slot(tmp_path: Path) -> None:
    expected = {0: _spec(0, "image/png", "image"), 1: _spec(1, "audio/wav", "audio")}
    _write_reference_media(tmp_path / "cmd", expected, {0: b"abc", 1: b"abc"})
    assert (tmp_path / "cmd" / "0.png").read_bytes() == b"abc"
    assert (tmp_path / "cmd" / "1.wav").read_bytes() == b"abc"
    assert not list((tmp_path / "cmd").glob("*.part"))


def test_inject_reference_paths_preserves_owner_and_order() -> None:
    params = VideoGenerationTaskParams(
        prompt="x",
        model="org/video",
        seconds=5,
        references=(_spec(0, "image/png", "image"), _spec(1, "image/png", "image")),
        reference_bytes=6,
        total_input_chunks=2,
    )
    task = VideoGenerationTask(
        task_id=TaskId(),
        command_id=CommandId("c"),
        instance_id=InstanceId("i"),
        task_status=TaskStatus.Pending,
        owner_node=NodeId("api"),
        task_params=params,
    )
    injected = _inject_reference_paths(task, {0: Path("/tmp/c/0.png"), 1: Path("/tmp/c/1.png")})
    assert injected.owner_node == NodeId("api")
    assert [spec.local_path for spec in injected.task_params.references] == [
        "/tmp/c/0.png",
        "/tmp/c/1.png",
    ]
    assert all(spec.local_path is None for spec in task.task_params.references)


def test_verify_file_digest_returns_size_only_on_match(tmp_path: Path) -> None:
    path = tmp_path / "output.mp4"
    payload = b"binary" * 1000
    path.write_bytes(payload)
    assert _verify_file_digest(path, hashlib.sha256(payload).hexdigest()) == len(payload)
    assert _verify_file_digest(path, "0" * 64) == -1


def test_task_failed_releases_video_reference_media() -> None:
    params = VideoGenerationTaskParams(prompt="x", model="org/video", seconds=5)
    task = VideoGenerationTask(
        task_id=TaskId("t"),
        command_id=CommandId("c"),
        instance_id=InstanceId("i"),
        task_status=TaskStatus.Failed,
        owner_node=NodeId("api"),
        task_params=params,
    )
    event = TaskFailed(task_id=TaskId("t"), error_type="video_mode_unavailable", error_message="no")
    assert _vision_media_cleanup_command_id(event, {}, {task.task_id: task}) == CommandId("c")
    assert _vision_media_cleanup_command_id(event, {}, {}) is None


def test_startup_purges_stale_video_directories(tmp_path: Path, monkeypatch: object) -> None:
    import pytest

    assert isinstance(monkeypatch, pytest.MonkeyPatch)
    from skulk.worker import main as worker_main

    inputs = tmp_path / "in"
    outputs = tmp_path / "out"
    (inputs / "cmd-a").mkdir(parents=True)
    (inputs / "cmd-a" / "0.png").write_bytes(b"x")
    (outputs / "cmd-b").mkdir(parents=True)
    (outputs / "cmd-b" / "output.mp4").write_bytes(b"y")
    (outputs / "stray.txt").write_bytes(b"z")
    monkeypatch.setattr(worker_main, "SKULK_VIDEO_INPUT_DIR", inputs)
    monkeypatch.setattr(worker_main, "SKULK_VIDEO_OUTPUT_DIR", outputs)
    _purge_stale_video_directories()
    assert not (inputs / "cmd-a").exists() and not (outputs / "cmd-b").exists()
    assert (outputs / "stray.txt").exists()


def _packet(
    command_id: CommandId, *, sequence: int, kind: str, data: bytes = b"", slot: int | None = None
) -> VisionMediaPacket:
    return VisionMediaPacket(
        source_node=NodeId("api"),
        target_node=NodeId("worker"),
        command_id=command_id,
        model=ModelId("org/video"),
        sequence=sequence,
        kind=kind,  # pyright: ignore[reportArgumentType]
        data=data,
        image_index=slot,
        total_chunks=3,
        image_count=2 if kind != "chunk" else None,
        sha256=hashlib.sha256(b"abcdefghij").hexdigest() if kind == "completed" else None,
        payload="reference_media",
    )


def _bare_worker() -> Worker:
    """A Worker with only the vision-media bookkeeping the finalize path reads."""

    worker = object.__new__(Worker)
    worker._vision_media_opened = {}  # pyright: ignore[reportPrivateUsage]
    worker._vision_media_chunks = {}  # pyright: ignore[reportPrivateUsage]
    worker._vision_media_completed = {}  # pyright: ignore[reportPrivateUsage]
    worker._vision_media_verified = {}  # pyright: ignore[reportPrivateUsage]
    worker._vision_media_verified_chunks = {}  # pyright: ignore[reportPrivateUsage]
    worker._reference_media_verified = {}  # pyright: ignore[reportPrivateUsage]
    worker._vision_media_pending_bytes = {}  # pyright: ignore[reportPrivateUsage]
    worker._vision_media_pending_since = {}  # pyright: ignore[reportPrivateUsage]
    worker._vision_media_pending_total_bytes = 0  # pyright: ignore[reportPrivateUsage]
    worker._vision_media_completed_streams = 0  # pyright: ignore[reportPrivateUsage]
    return worker


async def test_reference_stream_assembles_slots_from_frames(monkeypatch: object) -> None:
    """Frames verify against the stream digest and land in per-slot buffers."""

    import pytest

    assert isinstance(monkeypatch, pytest.MonkeyPatch)
    worker = _bare_worker()
    command_id = CommandId("cmd")
    admitted: list[CommandId] = []
    rejected: list[str] = []

    async def _admit(command: CommandId) -> None:
        admitted.append(command)

    async def _reject(packet: VisionMediaPacket, message: str) -> None:
        rejected.append(message)

    monkeypatch.setattr(worker, "_acknowledge_vision_media_if_admitted", _admit)
    monkeypatch.setattr(worker, "_reject_vision_media", _reject)
    worker._vision_media_opened[command_id] = _packet(  # pyright: ignore[reportPrivateUsage]
        command_id, sequence=0, kind="opened"
    )
    worker._vision_media_chunks[command_id] = {  # pyright: ignore[reportPrivateUsage]
        1: _packet(command_id, sequence=1, kind="chunk", data=b"abcd", slot=0),
        2: _packet(command_id, sequence=2, kind="chunk", data=b"efg", slot=1),
        3: _packet(command_id, sequence=3, kind="chunk", data=b"hij", slot=1),
    }
    worker._vision_media_completed[command_id] = _packet(  # pyright: ignore[reportPrivateUsage]
        command_id, sequence=4, kind="completed"
    )
    await worker._finalize_vision_media(command_id)  # pyright: ignore[reportPrivateUsage]
    assert rejected == []
    assert admitted == [command_id]
    assert worker._reference_media_verified[command_id] == {  # pyright: ignore[reportPrivateUsage]
        0: b"abcd",
        1: b"efghij",
    }
    assert command_id not in worker._vision_media_chunks  # pyright: ignore[reportPrivateUsage]


async def test_reference_stream_digest_mismatch_is_rejected(monkeypatch: object) -> None:
    import pytest

    assert isinstance(monkeypatch, pytest.MonkeyPatch)
    worker = _bare_worker()
    command_id = CommandId("cmd")
    rejected: list[str] = []

    async def _reject(packet: VisionMediaPacket, message: str) -> None:
        rejected.append(message)

    monkeypatch.setattr(worker, "_reject_vision_media", _reject)
    worker._vision_media_opened[command_id] = _packet(  # pyright: ignore[reportPrivateUsage]
        command_id, sequence=0, kind="opened"
    )
    worker._vision_media_chunks[command_id] = {  # pyright: ignore[reportPrivateUsage]
        1: _packet(command_id, sequence=1, kind="chunk", data=b"abcd", slot=0),
        2: _packet(command_id, sequence=2, kind="chunk", data=b"efg", slot=1),
        3: _packet(command_id, sequence=3, kind="chunk", data=b"xxx", slot=1),
    }
    worker._vision_media_completed[command_id] = _packet(  # pyright: ignore[reportPrivateUsage]
        command_id, sequence=4, kind="completed"
    )
    await worker._finalize_vision_media(command_id)  # pyright: ignore[reportPrivateUsage]
    assert rejected == ["Vision media failed SHA-256 integrity verification"]
    assert command_id not in worker._reference_media_verified  # pyright: ignore[reportPrivateUsage]
