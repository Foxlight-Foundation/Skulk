"""Worker-side handling of video reference media and finished output files."""

from __future__ import annotations

import hashlib
from pathlib import Path

from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.tasks import TaskId, TaskStatus
from skulk.shared.types.tasks import VideoGeneration as VideoGenerationTask
from skulk.shared.types.video import VideoGenerationTaskParams, VideoReferenceSpec
from skulk.shared.types.worker.instances import InstanceId
from skulk.worker.main import (
    _inject_reference_paths,  # pyright: ignore[reportPrivateUsage]
    _reference_media_extension,  # pyright: ignore[reportPrivateUsage]
    _verify_file_digest,  # pyright: ignore[reportPrivateUsage]
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
