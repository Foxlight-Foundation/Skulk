# pyright: reportPrivateUsage=false, reportAny=false
"""Music HTTP and OUTPUT_MEDIA lifecycle contracts, including arrival races."""

from __future__ import annotations

import hashlib
import io
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import skulk.api.main as api_module
from skulk.api.main import API
from skulk.api.music_jobs import MusicJob, MusicJobRegistry
from skulk.api.types.api import CreateInstanceParams
from skulk.api.video_store import VideoStore
from skulk.routing.output_media import OutputMediaPacket
from skulk.shared.models.model_cards import ModelCard, ModelId
from skulk.shared.topology import Topology
from skulk.shared.types.chunks import MusicChunk
from skulk.shared.types.commands import MusicGeneration, PrepareAudioCpp, TaskCancelled
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.events import AudioCppPreparationCompleted
from skulk.shared.types.memory import Memory
from skulk.shared.types.music import MusicGenerationTaskParams, MusicOutputManifest
from skulk.shared.types.profiling import MemoryUsage, NodeResources
from skulk.shared.types.tasks import MusicGeneration as MusicGenerationTask
from skulk.shared.types.worker.instances import (
    InstanceId,
    LlamaRpcInstance,
    MlxRingInstance,
)
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.channels import channel

MODEL = ModelId("audio-cpp/test-music")


class _TaskGroupStub:
    def start_soon(self, *_args: object) -> None:
        pass


def _ample_free_bytes(_store: VideoStore) -> int:
    return 1 << 40


@pytest.mark.parametrize("lane", ["audio_cpp-cpu", "audio_cpp-metal"])
async def test_mount_preflight_skips_ready_build_without_signed_music_support(
    monkeypatch: pytest.MonkeyPatch, lane: str,
) -> None:
    """A supported ready lane wins even when a CPU lane is not advertised."""
    topology = Topology()
    unsupported = NodeId("z-unsupported")
    supported = NodeId("a-supported")
    for node in (unsupported, supported):
        topology.add_node(node)
    resources = {
        node: NodeResources(
            backends=frozenset({"audio_cpp", lane}),
            architecture="arm64",
            engine_builds={lane: build},
            hardware_classes=frozenset({"platform:darwin"}),
        )
        for node, build in ((unsupported, "unsupported"), (supported, "supported"))
    }
    memory = MemoryUsage.from_bytes(
        ram_total=2**30, ram_available=2**30,
        swap_total=0, swap_available=0,
    )
    api: Any = object.__new__(API)
    api.node_id = NodeId("api-node")
    api.state = SimpleNamespace(topology=topology)
    api._telemetry_view = SimpleNamespace(
        node_resources=resources,
        node_memory={node: memory for node in resources},
    )
    api._audio_cpp_prepare_events = {}
    api._audio_cpp_prepare_results = {}
    async def complete_preparation(command: PrepareAudioCpp) -> None:
        assert command.target_node == supported
        api._audio_cpp_prepare_results[command.command_id] = AudioCppPreparationCompleted(
            request_id=command.command_id,
            target_node=supported,
            owner_node=command.owner_node,
            success=True,
            resources=resources[supported],
        )
        api._audio_cpp_prepare_events[command.command_id].set()

    api._send = AsyncMock(side_effect=complete_preparation)

    def supported_backends(
        _card: ModelCard, *, node_backends: frozenset[str],
        engine_builds: dict[str, str], hardware_classes: frozenset[str],
    ) -> frozenset[str]:
        assert node_backends and hardware_classes
        return (
            frozenset({lane})
            if engine_builds.get(lane) == "supported"
            else frozenset()
        )

    monkeypatch.setattr(api_module, "registry_supported_backends_for_node", supported_backends)
    await api._prepare_music_engine_for_mount(_card(), set())
    api._send.assert_awaited_once()
    api._send.reset_mock()

    # An exact placement must not silently prepare the other eligible node.
    with pytest.raises(HTTPException, match="audio.cpp could not be prepared"):
        await api._prepare_music_engine_for_mount(
            _card(), set(), required_nodes={unsupported},
        )
    api._send.assert_not_called()


async def test_exact_music_instance_prepares_its_specified_node_before_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The direct create route cannot bypass on-demand package preparation."""
    card = _card()
    node = NodeId("music-node")
    runner = RunnerId("music-runner")
    instance = MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            runner_to_shard={runner: PipelineShardMetadata(
                model_card=card, device_rank=0, world_size=1,
                start_layer=0, end_layer=1, n_layers=1,
            )},
            node_to_runner={node: runner},
        ),
        hosts_by_node={node: []},
        ephemeral_port=52415,
    )
    api: Any = object.__new__(API)
    api._send = AsyncMock()
    api._prepare_music_engine_for_mount = AsyncMock()
    monkeypatch.setattr(API, "_load_authorized_model_card", AsyncMock(return_value=card))
    def no_remote_code_approvals(_api: API) -> frozenset[str]:
        return frozenset()

    def sufficient_memory(_api: API) -> Memory:
        return Memory.from_gb(10)

    monkeypatch.setattr(API, "_cluster_remote_code_approvals", no_remote_code_approvals)
    monkeypatch.setattr(API, "_calculate_total_available_memory", sufficient_memory)

    await api.create_instance(CreateInstanceParams(instance=instance))
    api._prepare_music_engine_for_mount.assert_awaited_once_with(
        card, set(), required_nodes={node},
    )
    api._send.assert_awaited_once()

    api._send.reset_mock()
    api._prepare_music_engine_for_mount.side_effect = HTTPException(
        status_code=503, detail="package unavailable",
    )
    with pytest.raises(HTTPException, match="package unavailable"):
        await api.create_instance(CreateInstanceParams(instance=instance))
    api._send.assert_not_awaited()


async def test_exact_music_rpc_instance_is_rejected_before_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The API rejects an RPC shape before any package or mount side effect."""
    card = _card()
    node = NodeId("music-node")
    runner = RunnerId("music-runner")
    instance = LlamaRpcInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            runner_to_shard={runner: PipelineShardMetadata(
                model_card=card, device_rank=0, world_size=1,
                start_layer=0, end_layer=1, n_layers=1,
            )},
            node_to_runner={node: runner},
        ),
        driver_node=node,
        donor_endpoints={},
    )
    api: Any = object.__new__(API)
    api._send = AsyncMock()
    api._prepare_music_engine_for_mount = AsyncMock()
    monkeypatch.setattr(API, "_load_authorized_model_card", AsyncMock(return_value=card))
    def no_remote_code_approvals(_api: API) -> frozenset[str]:
        return frozenset()

    monkeypatch.setattr(API, "_cluster_remote_code_approvals", no_remote_code_approvals)

    with pytest.raises(HTTPException, match="cannot use llama.cpp RPC"):
        await api.create_instance(CreateInstanceParams(instance=instance))

    api._prepare_music_engine_for_mount.assert_not_awaited()
    api._send.assert_not_awaited()


def _wav() -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(b"\x01\x00" * 240)
    return stream.getvalue()


def _manifest(data: bytes) -> MusicOutputManifest:
    return MusicOutputManifest(
        size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
        duration_seconds=0.01, sample_rate=24000, channels=1,
    )


def _card() -> ModelCard:
    return ModelCard.model_validate({
        "model_id": MODEL, "storage_size": Memory.from_mb(128),
        "source_revision": "a" * 40,
        "gguf_file": "language_model_q4_0.gguf",
        "n_layers": 1, "hidden_size": 1, "supports_tensor": False,
        "tasks": ["TextToMusic"],
        "music": {
            "family": "minimax_music3", "lyrics": "required",
            "min_seconds": 10, "max_seconds": 60,
            "language_model_gguf": "language_model_q4_0.gguf",
            "rvq_depth_decoder_gguf": "rvq_depth_decoder_q8_0.gguf",
            "flow_transformer_gguf": "transformer_q4_0.gguf",
        },
        "artifact_bundle": {
            "bundle_id": "bundle_" + "a" * 52,
            "files": [
                {"path": name, "size_bytes": 1}
                for name in (
                    "language_model_q4_0.gguf",
                    "rvq_depth_decoder_q8_0.gguf",
                    "transformer_q4_0.gguf",
                    "condition_encoder.gguf",
                    "vocoder.gguf",
                    "config.json",
                    "config/condition_encoder.json",
                    "config/language_model.json",
                    "config/rvq_depth_decoder.json",
                    "config/transformer.json",
                    "config/vocoder.json",
                    "tokenizer/tokenizer.json",
                    "tokenizer/tokenizer_config.json",
                )
            ],
            "download_size": 13,
        },
    })


def _api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    api: Any = object.__new__(API)
    api.app = FastAPI()
    api.node_id = NodeId("api-node")
    api._music_generation_queues = {}
    api._text_generation_queues = {}
    api._image_generation_queues = {}
    api._video_generation_queues = {}
    api._embedding_queues = {}
    api._audio_speech_queues = {}
    api._audio_transcription_queues = {}
    api._music_jobs = MusicJobRegistry(tmp_path / "music" / "jobs.json")
    api._music_store = VideoStore(tmp_path / "music")
    api._music_job_media_deadlines = {}
    api._music_output_sources = {}
    api._early_output_packets = {}
    api._early_output_packet_bytes = 0
    api._pending_output_completions = {}
    api._cancelled_command_ids = set()
    api._pending_stream_failures = {}
    api._output_media_packet_sender = None
    api._send = AsyncMock()
    api._finalize_command_stream = AsyncMock()
    api._tg = _TaskGroupStub()
    monkeypatch.setattr(VideoStore, "_free_disk_bytes", _ample_free_bytes)

    api._load_authorized_model_card = AsyncMock(return_value=_card())
    api.app.post("/v1/music")(api.create_music)
    api.app.get("/v1/music")(api.list_music)
    api.app.get("/v1/music/{music_id}")(api.retrieve_music)
    api.app.get("/v1/music/{music_id}/content")(api.music_content)
    api.app.post("/v1/music/{music_id}/cancel")(api.cancel_music)
    api.app.delete("/v1/music/{music_id}")(api.delete_music)
    return api


def test_music_create_enforces_lyric_duration_and_option_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _api(tmp_path, monkeypatch)
    client = TestClient(api.app)
    base = {"model": str(MODEL), "prompt": "Orchestral fox", "seconds": 30}
    assert client.post("/v1/music", json=base).status_code == 400
    assert client.post("/v1/music", json={**base, "lyrics": "Sing", "seconds": 61}).status_code == 400
    assert client.post("/v1/music", json={**base, "lyrics": "Sing", "server_path": "/tmp/x"}).status_code == 422
    response = client.post("/v1/music", json={**base, "lyrics": "Sing", "seed": 7})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "music" and body["status"] == "queued"
    sent = cast("MusicGeneration", api._send.call_args.args[0])
    assert isinstance(sent, MusicGeneration)
    assert sent.owner_node == api.node_id
    assert sent.task_params.lyrics == "Sing" and sent.task_params.seed == 7
    assert body["id"] == str(sent.command_id)


def _job(api: Any, command_id: CommandId, data: bytes) -> None:
    api._music_jobs.create(MusicJob(
        id=command_id, model=str(MODEL), prompt="test", seconds=20,
        status="in_progress", created_at=1, output=_manifest(data),
        render_finished=False,
    ))


def _packet(command_id: CommandId, kind: str, data: bytes, sequence: int) -> OutputMediaPacket:
    return OutputMediaPacket.model_validate({
        "source_node": NodeId("worker-node"), "target_node": NodeId("api-node"),
        "command_id": command_id, "model": MODEL, "purpose": "music",
        "kind": kind, "sequence": sequence,
        **({"total_chunks": 1, "total_bytes": len(data), "content_type": "audio/wav"} if kind == "opened" else {}),
        **({"total_chunks": 1, "data": data} if kind == "chunk" else {}),
        **({"total_chunks": 1, "total_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} if kind == "completed" else {}),
    })


def test_music_output_source_tracks_only_live_local_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replicated remote TaskCreated events must not retain output-source state."""
    api = _api(tmp_path, monkeypatch)
    instance_id = InstanceId()
    worker_node = NodeId("worker-node")
    api.state = SimpleNamespace(instances={
        instance_id: SimpleNamespace(
            shard_assignments=SimpleNamespace(node_to_runner={worker_node: object()})
        )
    })
    command_id = CommandId("music-source")
    task = MusicGenerationTask(
        instance_id=instance_id,
        command_id=command_id,
        owner_node=NodeId("other-api"),
        task_params=MusicGenerationTaskParams(
            model=str(MODEL), prompt="piano", seconds=20
        ),
    )
    api._record_music_output_source(task)
    assert api._music_output_sources == {}

    local_task = task.model_copy(update={"owner_node": api.node_id})
    api._record_music_output_source(local_task)
    assert api._music_output_sources == {}  # deleted before placement

    _job(api, command_id, _wav())
    api._record_music_output_source(local_task)
    assert api._music_output_sources == {command_id: worker_node}


@pytest.mark.anyio
async def test_media_before_terminal_settles_and_serves_wav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _api(tmp_path, monkeypatch)
    data = _wav()
    command_id = CommandId("music-early")
    _job(api, command_id, data)
    terminals: list[OutputMediaPacket] = []

    async def acknowledge(packet: OutputMediaPacket) -> None:
        terminals.append(packet)

    monkeypatch.setattr(api, "_send_output_media_terminal", acknowledge)
    await api._receive_music_output_media(_packet(command_id, "chunk", data, 1))
    await api._receive_music_output_media(_packet(command_id, "completed", data, 2))
    await api._receive_music_output_media(_packet(command_id, "opened", data, 0))
    assert api._music_jobs.get(command_id).status == "in_progress"
    assert terminals[-1].kind == "accepted"
    sender, receiver = channel[MusicChunk]()
    sender.send_nowait(MusicChunk(model=MODEL, finish_reason="stop", output=_manifest(data)))
    sender.close()
    await api._drain_music_job(command_id, receiver)
    assert api._music_jobs.get(command_id).status == "completed"
    api._finalize_command_stream.assert_awaited_once()
    response = TestClient(api.app).get(f"/v1/music/{command_id}/content")
    assert response.status_code == 200 and response.content == data
    assert response.headers["content-type"] == "audio/wav"


@pytest.mark.anyio
async def test_over_limit_music_transfer_fails_before_file_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _api(tmp_path, monkeypatch)
    data = _wav()
    command_id = CommandId("music-oversize")
    _job(api, command_id, data)
    terminals: list[OutputMediaPacket] = []

    async def acknowledge(packet: OutputMediaPacket) -> None:
        terminals.append(packet)

    monkeypatch.setattr(api, "_send_output_media_terminal", acknowledge)
    opened = _packet(command_id, "opened", data, 0).model_copy(
        update={"total_bytes": 64 * 1024 * 1024 + 1}
    )
    await api._receive_music_output_media(opened)
    assert api._music_jobs.get(command_id).status == "failed"
    assert terminals[-1].kind == "transport_failed"
    assert api._music_store.get(command_id, "music") is None


def test_restart_marks_inflight_music_failed(tmp_path: Path) -> None:
    path = tmp_path / "jobs.json"
    jobs = MusicJobRegistry(path)
    jobs.create(MusicJob(
        id=CommandId("interrupted"), model=str(MODEL), prompt="test",
        seconds=20, status="in_progress", created_at=1,
    ))
    recovered = MusicJobRegistry(path).get(CommandId("interrupted"))
    assert recovered is not None and recovered.status == "failed"
    assert "restarted" in (recovered.error or "")


def test_cancel_discards_partial_music_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _api(tmp_path, monkeypatch)
    command_id = CommandId("music-cancel")
    data = _wav()
    _job(api, command_id, data)
    sender, _receiver = channel[MusicChunk]()
    api._music_generation_queues[command_id] = sender
    client = TestClient(api.app)
    response = client.post(f"/v1/music/{command_id}/cancel")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert isinstance(api._send.call_args.args[0], TaskCancelled)
    assert api._music_store.get(command_id, "music") is None
