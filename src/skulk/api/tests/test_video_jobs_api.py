# pyright: reportPrivateUsage=false, reportAny=false
"""HTTP contract of the ``/v1/videos`` job family and its terminal cleanup.

The routes are exercised through a FastAPI test client on a bare API whose
master send is a mock, so a create request ends at the command that would be
sent; the render, transfer, and settlement paths are covered by the
substrate tests. What matters here is the request parsing, the OpenAI-shaped
responses, and that every terminal path releases what it held.
"""

from __future__ import annotations

import hashlib
import tempfile
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from skulk.api import main as api_main
from skulk.api.main import API
from skulk.api.video_jobs import VideoAttachment, VideoJob, VideoJobRegistry
from skulk.api.video_store import VideoStore
from skulk.routing.output_media import OutputMediaPacket
from skulk.shared.models.model_cards import ModelCard, ModelTask, VideoCardConfig
from skulk.shared.types.chunks import ErrorChunk
from skulk.shared.types.commands import TaskCancelled, VideoGeneration
from skulk.shared.types.common import CommandId, ModelId, NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.tasks import TaskId, TaskStatus
from skulk.shared.types.tasks import VideoGeneration as VideoGenerationTask
from skulk.shared.types.video import VideoGenerationTaskParams, VideoOutputManifest
from skulk.shared.types.worker.instances import InstanceId
from skulk.utils.channels import channel

MODEL = ModelId("org/video")


def _card() -> ModelCard:
    return ModelCard(
        model_id=MODEL,
        storage_size=Memory.from_mb(128),
        n_layers=4,
        hidden_size=256,
        supports_tensor=False,
        tasks=[ModelTask.TextToVideo, ModelTask.ImageToVideo, ModelTask.ReferenceToVideo],
        video=VideoCardConfig.model_validate(
            {
                "modes": ["t2va", "fl2va", "ref2va"],
                "min_seconds": 4,
                "max_seconds": 15,
                "reference_limits": {"max_images": 4, "max_videos": 1, "max_audio_clips": 1},
            }
        ),
    )


class _StubTaskGroup:
    """Records what the API would have spawned instead of running it."""

    def __init__(self) -> None:
        self.spawned: list[tuple[object, tuple[object, ...]]] = []

    def start_soon(self, function: Callable[..., Coroutine[Any, Any, object]], *args: object) -> None:
        self.spawned.append((function, args))


def _make_api(monkeypatch: pytest.MonkeyPatch) -> Any:
    app = FastAPI()
    api: Any = object.__new__(API)
    api.app = app
    api.node_id = NodeId("api-node")
    api.state = State()
    api._text_generation_queues = {}
    api._image_generation_queues = {}
    api._video_generation_queues = {}
    api._embedding_queues = {}
    api._audio_speech_queues = {}
    api._audio_transcription_queues = {}
    api._video_jobs = VideoJobRegistry(None)
    api._video_store = VideoStore(Path(tempfile.mkdtemp()))
    api._video_job_media_deadlines = {}
    api._video_output_sources = {}
    api._early_output_packets = {}
    api._early_output_packet_bytes = 0
    api._pending_output_completions = {}
    api._video_upload_inflight_bytes = 0
    api._pending_stream_failures = {}
    api._pending_vision_media = {}
    api._pending_vision_media_bytes = 0
    api._active_vision_media_bytes = {}
    api._active_vision_media_total_bytes = 0
    api._vision_media_commands = set()
    api._vision_media_targets = {}
    api._vision_media_pending_acks = {}
    api._vision_media_ack_deadlines = {}
    api._vision_media_models = {}
    api._vision_media_failures = {}
    api._vision_media_packet_sender = object()
    api._output_media_packet_sender = None
    api._cancelled_command_ids = set()
    api._chunk_reorder = {}
    api._data_dedup_cursor = {}
    api._tg = _StubTaskGroup()
    api._send = AsyncMock()
    api._setup_exception_handlers()

    async def load(model_id: ModelId) -> ModelCard:
        if model_id != MODEL:
            raise ValueError(f"unknown model {model_id}")
        return _card()

    monkeypatch.setattr(ModelCard, "load", staticmethod(load))
    app.post("/v1/videos")(api.create_video)
    app.get("/v1/videos")(api.list_videos)
    app.get("/v1/videos/{video_id}")(api.retrieve_video)
    app.delete("/v1/videos/{video_id}")(api.delete_video)
    app.get("/v1/videos/{video_id}/content")(api.video_content)
    app.post("/v1/videos/{video_id}/cancel")(api.cancel_video)
    app.post("/v1/cancel/{command_id}")(api.cancel_command)
    return api


def _sent_command(api: Any) -> VideoGeneration:
    command = cast("VideoGeneration", api._send.call_args.args[0])
    assert isinstance(command, VideoGeneration)
    return command


def test_create_json_text_to_video(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _make_api(monkeypatch)
    client = TestClient(api.app)
    response = client.post(
        "/v1/videos",
        json={"model": str(MODEL), "prompt": "a fox in autumn leaves", "seconds": "6"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "video" and body["status"] == "queued"
    assert body["seconds"] == "6" and body["mode"] == "t2va" and body["progress"] == 0
    command = _sent_command(api)
    assert command.owner_node == NodeId("api-node")
    assert command.task_params.seconds == 6 and command.task_params.mode is not None
    assert str(command.command_id) == body["id"]
    assert body["id"] in {str(key) for key in api._video_generation_queues}


def test_create_defaults_seconds_to_the_card_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _make_api(monkeypatch)
    client = TestClient(api.app)
    response = client.post("/v1/videos", json={"model": str(MODEL), "prompt": "x"})
    assert response.status_code == 200, response.text
    assert response.json()["seconds"] == "4"


@pytest.mark.parametrize(
    ("payload", "status", "fragment"),
    [
        ({"model": str(MODEL)}, 400, "prompt"),
        ({"model": str(MODEL), "prompt": "x", "size": "wide"}, 400, "size"),
        ({"model": str(MODEL), "prompt": "x", "seconds": 40}, 400, "seconds"),
        ({"model": "org/missing", "prompt": "x"}, 404, "unknown model"),
        ({"model": str(MODEL), "prompt": "x", "bogus": 1}, 400, "bogus"),
    ],
)
def test_create_rejects_bad_requests(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, object], status: int, fragment: str
) -> None:
    api = _make_api(monkeypatch)
    client = TestClient(api.app)
    response = client.post("/v1/videos", json=payload)
    assert response.status_code == status, response.text
    assert fragment in response.json()["error"]["message"]
    api._send.assert_not_called()


def test_create_rejects_other_content_types(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _make_api(monkeypatch)
    client = TestClient(api.app)
    response = client.post("/v1/videos", content=b"prompt=x", headers={"content-type": "text/plain"})
    assert response.status_code == 415


def test_create_multipart_stages_attachments(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _make_api(monkeypatch)
    client = TestClient(api.app)
    first = b"\x89PNG first"
    clip = b"\x00\x00clip"
    response = client.post(
        "/v1/videos",
        data={"model": str(MODEL), "prompt": "keyframes", "seconds": "5", "audio": "false"},
        files=[
            ("input_reference", ("first.png", first, "image/png")),
            ("reference", ("clip.mp4", clip, "video/mp4")),
        ],
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "ref2va" and body["audio"] is False
    command = _sent_command(api)
    references = command.task_params.references
    assert [spec.role for spec in references] == ["first_frame", "reference"]
    assert [spec.kind for spec in references] == ["image", "video"]
    assert references[0].sha256 == hashlib.sha256(first).hexdigest()
    assert references[1].filename == "clip.mp4"
    assert command.task_params.reference_bytes == len(first) + len(clip)
    assert command.task_params.total_input_chunks == 2
    assert all(spec.local_path is None for spec in references)
    assert command.command_id in api._pending_vision_media
    assert api._pending_vision_media[command.command_id].payload == "reference_media"


def test_create_multipart_rejects_unlabelled_and_oversized_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _make_api(monkeypatch)
    client = TestClient(api.app)
    response = client.post(
        "/v1/videos",
        data={"model": str(MODEL), "prompt": "x"},
        files=[("first_frame", ("blob", b"data", "application/octet-stream"))],
    )
    assert response.status_code == 400 and "content type" in response.json()["error"]["message"]
    monkeypatch.setattr(api_main, "_REFERENCE_MEDIA_PENDING_COMMAND_BYTES", 8)
    response = client.post(
        "/v1/videos",
        data={"model": str(MODEL), "prompt": "x"},
        files=[("first_frame", ("big.png", b"0123456789", "image/png"))],
    )
    assert response.status_code == 413
    api._send.assert_not_called()


def test_create_multipart_rejects_two_first_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _make_api(monkeypatch)
    client = TestClient(api.app)
    response = client.post(
        "/v1/videos",
        data={"model": str(MODEL), "prompt": "x"},
        files=[
            ("input_reference", ("a.png", b"a", "image/png")),
            ("first_frame", ("b.png", b"b", "image/png")),
        ],
    )
    assert response.status_code == 400
    assert "first_frame" in response.json()["error"]["message"]


def _job(identifier: str, created_at: int, **changes: object) -> VideoJob:
    job = VideoJob(
        id=CommandId(identifier),
        model=str(MODEL),
        prompt="p",
        mode="t2va",
        seconds=5,
        created_at=created_at,
    )
    return job.model_copy(update=changes)


def test_list_pages_newest_first(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _make_api(monkeypatch)
    for index in range(3):
        api._video_jobs.create(_job(f"job-{index}", created_at=index))
    client = TestClient(api.app)
    first = client.get("/v1/videos", params={"limit": 2}).json()
    assert [item["id"] for item in first["data"]] == ["job-2", "job-1"]
    assert first["has_more"] is True and first["last_id"] == "job-1"
    second = client.get("/v1/videos", params={"limit": 2, "after": first["last_id"]}).json()
    assert [item["id"] for item in second["data"]] == ["job-0"]
    assert second["has_more"] is False
    ascending = client.get("/v1/videos", params={"order": "asc"}).json()
    assert [item["id"] for item in ascending["data"]] == ["job-0", "job-1", "job-2"]


def test_retrieve_reports_failure_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _make_api(monkeypatch)
    api._video_jobs.create(_job("gone", created_at=1))
    api._video_jobs.fail(CommandId("gone"), "the render crashed")
    client = TestClient(api.app)
    assert client.get("/v1/videos/missing").status_code == 404
    body = client.get("/v1/videos/gone").json()
    assert body["status"] == "failed"
    assert body["error"] == {"code": "job_failed", "message": "the render crashed"}


def _complete_job(api: Any, identifier: str, payload: bytes) -> None:
    command_id = CommandId(identifier)
    digest = hashlib.sha256(payload).hexdigest()
    api._video_jobs.create(_job(identifier, created_at=1))
    api._video_store.open_assembly(
        command_id, "video", content_type="video/mp4", total_bytes=len(payload), total_chunks=1
    )
    api._video_store.append(command_id, "video", 1, payload)
    stored = api._video_store.commit(command_id, "video", sha256=digest, total_chunks=1)
    manifest = VideoOutputManifest(
        sha256=digest,
        size_bytes=len(payload),
        width=16,
        height=16,
        frame_count=5,
        fps=1,
        seconds=5.0,
        audio_sample_rate=32000,
        audio_channels=2,
    )
    api._video_jobs.update(command_id, render_finished=True, output=manifest, media_delivered=True)
    api._video_jobs.settle(command_id)
    api._video_jobs.update(command_id, expires_at=int(stored.expires_at))


def test_content_serves_only_completed_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _make_api(monkeypatch)
    api._video_jobs.create(_job("queued", created_at=1))
    payload = b"mp4 bytes " * 100
    _complete_job(api, "done", payload)
    client = TestClient(api.app)
    pending = client.get("/v1/videos/queued/content")
    assert pending.status_code == 409 and "queued" in pending.json()["error"]["message"]
    response = client.get("/v1/videos/done/content")
    assert response.status_code == 200
    assert response.content == payload and response.headers["content-type"] == "video/mp4"
    assert client.get("/v1/videos/done/content", params={"variant": "thumbnail"}).status_code == 404
    body = client.get("/v1/videos/done").json()
    assert body["status"] == "completed" and body["output"]["has_thumbnail"] is False
    assert body["output"]["size_bytes"] == len(payload)


def test_cancel_streaming_job_sends_task_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _make_api(monkeypatch)
    job = api._video_jobs.create(_job("live", created_at=1))
    sender, _receiver = channel[object]()
    api._video_generation_queues[job.id] = sender
    api._video_job_media_deadlines[job.id] = time.monotonic() + 60
    client = TestClient(api.app)
    body = client.post("/v1/videos/live/cancel").json()
    assert body["status"] == "cancelled"
    assert body["error"]["code"] == "job_cancelled"
    sent = api._send.call_args.args[0]
    assert isinstance(sent, TaskCancelled) and sent.cancelled_command_id == job.id
    assert job.id in api._cancelled_command_ids
    assert job.id not in api._video_job_media_deadlines
    # Cancelling again is a no-op that returns the job unchanged.
    api._send.reset_mock()
    assert client.post("/v1/videos/live/cancel").json()["status"] == "cancelled"
    api._send.assert_not_called()


def test_cancel_uploading_job_notifies_the_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _make_api(monkeypatch)
    terminals: list[OutputMediaPacket] = []

    async def capture(packet: OutputMediaPacket) -> None:
        terminals.append(packet)

    monkeypatch.setattr(api, "_send_output_media_terminal", capture)
    job = api._video_jobs.create(_job("uploading", created_at=1, render_finished=True))
    api._video_output_sources[job.id] = NodeId("worker-1")
    api._early_output_packets[(job.id, "video")] = [
        OutputMediaPacket(
            source_node=NodeId("worker-1"),
            target_node=NodeId("api-node"),
            command_id=job.id,
            model=MODEL,
            purpose="video",
            sequence=2,
            kind="chunk",
            data=b"xx",
            total_chunks=2,
        )
    ]
    api._early_output_packet_bytes = 2
    client = TestClient(api.app)
    assert client.post("/v1/videos/uploading/cancel").json()["status"] == "cancelled"
    api._send.assert_not_called()
    assert [packet.kind for packet in terminals] == ["cancelled"]
    assert terminals[0].target_node == NodeId("worker-1")
    assert api._early_output_packets == {} and api._early_output_packet_bytes == 0
    assert job.id not in api._video_output_sources


def test_delete_cancels_then_forgets(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _make_api(monkeypatch)
    _complete_job(api, "done", b"bytes")
    job = api._video_jobs.create(_job("live", created_at=2))
    sender, _receiver = channel[object]()
    api._video_generation_queues[job.id] = sender
    client = TestClient(api.app)
    assert client.delete("/v1/videos/done").json() == {
        "id": "done",
        "object": "video.deleted",
        "deleted": True,
    }
    assert client.get("/v1/videos/done").status_code == 404
    assert not (api._video_store.storage_dir / "done").exists()
    assert client.delete("/v1/videos/live").status_code == 200
    assert isinstance(api._send.call_args.args[0], TaskCancelled)
    assert api._video_jobs.get(job.id) is None
    assert client.delete("/v1/videos/live").status_code == 404


def test_legacy_cancel_route_handles_uploading_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _make_api(monkeypatch)
    monkeypatch.setattr(api, "_send_output_media_terminal", AsyncMock())
    job = api._video_jobs.create(_job("uploading", created_at=1, render_finished=True))
    client = TestClient(api.app)
    assert client.post(f"/v1/cancel/{job.id}").status_code == 200
    cancelled = api._video_jobs.get(job.id)
    assert cancelled is not None and cancelled.status == "cancelled"


async def test_post_render_task_failure_ends_the_job_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _make_api(monkeypatch)
    job = api._video_jobs.create(_job("post-render", created_at=1, render_finished=True))
    api._video_job_media_deadlines[job.id] = time.monotonic() + 60
    task = VideoGenerationTask(
        task_id=TaskId("t"),
        command_id=job.id,
        instance_id=InstanceId("i"),
        task_status=TaskStatus.Failed,
        owner_node=NodeId("api-node"),
        task_params=VideoGenerationTaskParams(prompt="p", model=str(MODEL), seconds=5),
    )
    api.state = State(tasks={task.task_id: task})
    await api._terminate_command_stream(task.task_id, "instance lost")
    failed = api._video_jobs.get(job.id)
    assert failed is not None and failed.status == "failed" and failed.error == "instance lost"
    assert api._pending_stream_failures == {}
    assert job.id not in api._video_job_media_deadlines


async def test_late_task_failure_leaves_a_completed_job_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _make_api(monkeypatch)
    payload = b"done bytes"
    _complete_job(api, "finished", payload)
    task = VideoGenerationTask(
        task_id=TaskId("t"),
        command_id=CommandId("finished"),
        instance_id=InstanceId("i"),
        task_status=TaskStatus.Failed,
        owner_node=NodeId("api-node"),
        task_params=VideoGenerationTaskParams(prompt="p", model=str(MODEL), seconds=5),
    )
    api.state = State(tasks={task.task_id: task})
    await api._terminate_command_stream(task.task_id, "instance lost after completion")
    job = api._video_jobs.get(CommandId("finished"))
    assert job is not None and job.status == "completed" and job.error is None
    stored = api._video_store.get(CommandId("finished"))
    assert stored is not None and stored.file_path.read_bytes() == payload


async def test_task_failure_with_a_live_queue_still_streams_the_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _make_api(monkeypatch)
    job = api._video_jobs.create(_job("streaming", created_at=1))
    sender, receiver = channel[object]()
    api._video_generation_queues[job.id] = sender
    task = VideoGenerationTask(
        task_id=TaskId("t"),
        command_id=job.id,
        instance_id=InstanceId("i"),
        task_status=TaskStatus.Failed,
        owner_node=NodeId("api-node"),
        task_params=VideoGenerationTaskParams(prompt="p", model=str(MODEL), seconds=5),
    )
    api.state = State(tasks={task.task_id: task})
    await api._terminate_command_stream(task.task_id, "instance lost")
    with receiver:
        chunk = await receiver.receive()
    assert isinstance(chunk, ErrorChunk) and chunk.error_message == "instance lost"
    assert api._video_jobs.get(job.id) is not None
    assert api._video_jobs.get(job.id).status == "queued"


def test_task_command_id_covers_video_tasks() -> None:
    task = VideoGenerationTask(
        task_id=TaskId("t"),
        command_id=CommandId("c"),
        instance_id=InstanceId("i"),
        task_status=TaskStatus.Pending,
        owner_node=NodeId("api"),
        task_params=VideoGenerationTaskParams(prompt="p", model=str(MODEL), seconds=5),
    )
    assert API._task_command_id(task) == "c"


def test_reference_staging_hashes_without_a_joined_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    from skulk.api import video_jobs

    api = _make_api(monkeypatch)
    monkeypatch.setattr(video_jobs, "SKULK_MAX_CHUNK_SIZE", 4)
    raw = [b"abcdefgh", b"ij"]
    attachments = [VideoAttachment.from_bytes(data) for data in raw]
    assert [len(item.chunks) for item in attachments] == [2, 1]
    api._stage_reference_media(CommandId("c"), MODEL, attachments)
    pending = api._pending_vision_media[CommandId("c")]
    assert pending.sha256 == hashlib.sha256(b"".join(raw)).hexdigest()
    assert [slot for slot, _ in pending.chunks] == [0, 0, 1]
    # The staged frames are the attachment's own frames, not copies.
    assert pending.chunks[0][1] is attachments[0].chunks[0]


def test_create_multipart_refuses_uploads_the_node_cannot_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _make_api(monkeypatch)
    api._pending_vision_media_bytes = api_main._VISION_MEDIA_PENDING_TOTAL_BYTES - 4
    client = TestClient(api.app)
    response = client.post(
        "/v1/videos",
        data={"model": str(MODEL), "prompt": "x"},
        files=[("first_frame", ("big.png", b"0123456789", "image/png"))],
    )
    assert response.status_code == 503
    api._send.assert_not_called()
    # A refused read releases its reservation; a concurrent request that is
    # still reading counts against the budget exactly like staged media.
    assert api._video_upload_inflight_bytes == 0
    api._pending_vision_media_bytes = 0
    api._video_upload_inflight_bytes = api_main._VISION_MEDIA_PENDING_TOTAL_BYTES - 4
    response = client.post(
        "/v1/videos",
        data={"model": str(MODEL), "prompt": "x"},
        files=[("first_frame", ("big.png", b"0123456789", "image/png"))],
    )
    assert response.status_code == 503
    assert api._video_upload_inflight_bytes == api_main._VISION_MEDIA_PENDING_TOTAL_BYTES - 4


def test_create_multipart_releases_its_reservation_after_staging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _make_api(monkeypatch)
    client = TestClient(api.app)
    response = client.post(
        "/v1/videos",
        data={"model": str(MODEL), "prompt": "x"},
        files=[("first_frame", ("a.png", b"0123456789", "image/png"))],
    )
    assert response.status_code == 200, response.text
    assert api._video_upload_inflight_bytes == 0
    assert api._pending_vision_media_bytes == 10


def test_create_multipart_rejects_unknown_file_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _make_api(monkeypatch)
    client = TestClient(api.app)
    response = client.post(
        "/v1/videos",
        data={"model": str(MODEL), "prompt": "x"},
        files=[("frist_frame", ("a.png", b"0123456789", "image/png"))],
    )
    assert response.status_code == 400
    assert "frist_frame" in response.json()["error"]["message"]
    api._send.assert_not_called()


def test_multipart_openapi_schema_is_one_flat_object() -> None:
    schema = api_main._video_create_multipart_schema()
    assert "allOf" not in schema
    properties = cast("dict[str, Any]", schema["properties"])
    assert {"prompt", "model", "input_reference", "first_frame", "last_frame", "reference"} <= set(
        properties
    )
    assert properties["reference"]["items"] == {"type": "string", "format": "binary"}
    assert schema.get("additionalProperties") is False
