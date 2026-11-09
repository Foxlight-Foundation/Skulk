# pyright: reportPrivateUsage=false
"""Actual edit HTTP/source/result composition without native inference claims."""

import json
import os
import shutil
import stat
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import anyio
import httpx
import pytest
from fastapi import FastAPI

import skulk.api.music_edit_api as music_edit_api
from skulk.api.music_edit_api import MusicEditAPI
from skulk.api.music_edit_media import MusicEditMediaStore
from skulk.api.operator_gateway import OperatorGatewayAuthorization, _required_scopes
from skulk.api.tests.test_operator_gateway import paired_service
from skulk.api.types.music_edits import MusicEditListResponse, MusicEditResource
from skulk.master.main import music_editing_events
from skulk.master.tests.test_music_edit_admission import state
from skulk.routing.output_media import (
    OutputMediaPacket,
    output_media_for_api,
    output_media_for_worker,
)
from skulk.routing.router import OutboundPacket, TopicRouter
from skulk.routing.topics import OUTPUT_MEDIA
from skulk.routing.vision_media import VisionMediaPacket
from skulk.shared.constants import SKULK_MAX_CHUNK_SIZE
from skulk.shared.models.model_cards import ModelCard, ModelId
from skulk.shared.tests.music_edit_fixtures import music_edit_fixture
from skulk.shared.types.chunks import ErrorChunk, MusicEditingChunk
from skulk.shared.types.commands import Command, TaskCancelled
from skulk.shared.types.commands import MusicEditing as MusicEditCommand
from skulk.shared.types.common import CommandId, NodeId, SystemId
from skulk.shared.types.events import TaskCreated
from skulk.shared.types.profiling import NodeResources
from skulk.shared.types.state import State
from skulk.shared.types.tasks import MusicEditing, TaskStatus
from skulk.utils.channels import Receiver, Sender, channel
from skulk.utils.task_group import TaskGroup
from skulk.worker.separation_input import MusicEditInputReceiver
from skulk.worker.separation_sources import SeparationSourceStore


@dataclass
class Harness:
    root: Path
    snapshot: State
    resources: dict[NodeId, NodeResources]
    session: SystemId = field(default_factory=SystemId)
    service: MusicEditAPI | None = None
    worker: MusicEditInputReceiver | None = None
    commands: list[Command] = field(default_factory=list)
    queues: dict[CommandId, Sender[MusicEditingChunk | ErrorChunk]] = field(
        default_factory=dict
    )
    source_frames: list[VisionMediaPacket] = field(default_factory=list)
    output_controls: list[OutputMediaPacket] = field(default_factory=list)
    finalized: list[CommandId] = field(default_factory=list)
    seed_draws: int = 0
    place: bool = True

    @property
    def api(self) -> MusicEditAPI:
        """Return the real exclusive API-local owner."""
        assert self.service is not None
        return self.service

    def seed(self) -> int:
        """Expose whether request acceptance ever draws another random seed."""
        self.seed_draws += 1
        return 77

    async def card(self, model: ModelId) -> ModelCard:
        instance = next(iter(self.snapshot.instances.values()))
        card = next(
            iter(instance.shard_assignments.runner_to_shard.values())
        ).model_card
        assert model == card.model_id
        return card

    async def send_command(self, command: Command, session: SystemId) -> None:
        if session != self.session:
            raise ValueError("old cluster session")
        self.commands.append(command)
        if isinstance(command, MusicEditCommand) and self.place:
            events, _ = music_editing_events(
                self.snapshot, command, node_resources=self.resources
            )
            assert isinstance(events[0], TaskCreated) and isinstance(
                events[0].task, MusicEditing
            )
            task = events[0].task
            assert task.task_status == TaskStatus.Pending
            self.snapshot = self.snapshot.model_copy(
                update={"tasks": dict(self.snapshot.tasks) | {task.task_id: task}}
            )
            await self.api.observe_task(task)
            assert self.worker is not None
            await self.worker.begin(task.command_id)

    def open_queue(
        self, identifier: CommandId
    ) -> Receiver[MusicEditingChunk | ErrorChunk]:
        send, receive = channel[MusicEditingChunk | ErrorChunk]()
        self.queues[identifier] = send
        return receive

    def close_queue(self, identifier: CommandId) -> None:
        if sender := self.queues.get(identifier):
            sender.close()

    async def finalize(self, identifier: CommandId, _session: SystemId) -> None:
        self.finalized.append(identifier)
        if sender := self.queues.pop(identifier, None):
            sender.close()

    async def send_source(self, packet: VisionMediaPacket) -> None:
        self.source_frames.append(packet)
        assert self.worker is not None
        await self.worker.receive(packet)

    async def send_output(self, packet: OutputMediaPacket) -> None:
        self.output_controls.append(packet)

    async def task(self, identifier: CommandId) -> MusicEditing:
        with anyio.fail_after(5):
            while True:
                matches = [
                    task
                    for task in self.snapshot.tasks.values()
                    if isinstance(task, MusicEditing) and task.command_id == identifier
                ]
                if (
                    matches
                    and self.worker is not None
                    and self.worker.ready(identifier)
                ):
                    return matches[0]
                await anyio.sleep(0.01)

    async def render(
        self,
        identifier: CommandId,
        *,
        media_first: bool,
        damaged: bool = False,
        send: Callable[[OutputMediaPacket], Awaitable[None]] | None = None,
    ) -> bytes:
        """Deliver real encoded fixture bytes plus exact synthetic native provenance."""
        task = await self.task(identifier)
        _, output, _, data = music_edit_fixture(str(identifier))
        output = output.model_copy(
            update={
                "result": output.result.model_copy(
                    update={"seed": task.task_params.seed}
                )
            }
        )
        if damaged:
            output = output.model_copy(
                update={
                    "result": output.result.model_copy(
                        update={
                            "output": output.result.output.model_copy(
                                update={"peak": 0.5}
                            ),
                        }
                    )
                }
            )
        terminal = MusicEditingChunk(
            model=ModelId(task.task_params.model), output=output, finish_reason="stop"
        )
        chunks = (len(data) + SKULK_MAX_CHUNK_SIZE - 1) // SKULK_MAX_CHUNK_SIZE
        emit = send if send is not None else self.api.receive_output

        async def metadata() -> None:
            await self.queues[identifier].send(terminal)
            with anyio.fail_after(5):
                while True:
                    async with self.api.lock:
                        job = await self.api.job(identifier)
                        if job.candidate_output is not None or job.is_terminal:
                            break
                    await anyio.sleep(0.01)

        if not media_first:
            await metadata()
        await emit(
            OutputMediaPacket(
                source_node=NodeId("node-0"),
                target_node=self.api.node_id,
                command_id=identifier,
                model=ModelId(task.task_params.model),
                purpose="music_edit",
                kind="opened",
                sequence=0,
                total_bytes=len(data),
                total_chunks=chunks,
                content_type="audio/wav",
            )
        )
        # A terminal can overtake chunks on the fallback transport.
        await emit(
            OutputMediaPacket(
                source_node=NodeId("node-0"),
                target_node=self.api.node_id,
                command_id=identifier,
                model=ModelId(task.task_params.model),
                purpose="music_edit",
                kind="completed",
                sequence=chunks + 1,
                total_bytes=len(data),
                total_chunks=chunks,
                sha256=output.result.output.sha256,
            )
        )
        for number in range(chunks):
            await emit(
                OutputMediaPacket(
                    source_node=NodeId("node-0"),
                    target_node=self.api.node_id,
                    command_id=identifier,
                    model=ModelId(task.task_params.model),
                    purpose="music_edit",
                    kind="chunk",
                    sequence=number + 1,
                    total_chunks=chunks,
                    data=data[
                        number * SKULK_MAX_CHUNK_SIZE : (number + 1)
                        * SKULK_MAX_CHUNK_SIZE
                    ],
                )
            )
        if media_first:
            await metadata()
        return data


@asynccontextmanager
async def fixture(root: Path) -> AsyncIterator[tuple[Harness, httpx.AsyncClient]]:
    snapshot, resources = state()
    harness = Harness(root, snapshot, resources)
    sources = SeparationSourceStore[MusicEditing](root / "worker-inputs")
    group = TaskGroup()
    service: MusicEditAPI | None = None
    try:
        async with group as active_group:
            service = MusicEditAPI(
                NodeId("owner"),
                root / "api",
                lambda: harness.snapshot,
                group,
                harness.card,
                harness.send_command,
                lambda: harness.session,
                harness.send_source,
                harness.send_output,
                harness.open_queue,
                harness.finalize,
                harness.close_queue,
                seed=harness.seed,
            )
            harness.service = service

            async def fail(task: MusicEditing, message: str) -> None:
                await service.fail(task.command_id, message)

            harness.worker = MusicEditInputReceiver(
                NodeId("node-0"),
                lambda: harness.snapshot,
                sources,
                service.receive_source,
                fail,
            )
            app = FastAPI()
            service.register(app)

            # Register the legacy dynamic route later, as the host does. Its
            # presence must never consume /v1/music/edits or its collection GET.
            async def _legacy(music_id: str) -> str:
                return music_id

            app.get("/v1/music/{music_id}")(_legacy)

            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://api.test"
                ) as client:
                    yield harness, client
            finally:
                active_group.cancel_scope.cancel()
    finally:
        with anyio.CancelScope(shield=True):
            if service is not None:
                await service.close()
            if harness.worker is not None:
                await harness.worker.close_after_drain()
            sources.close()


def request(*, seed: int | None = 0) -> dict[str, str | int]:
    """Use the source clock, including an explicit zero seed when provided."""
    body: dict[str, str | int] = {
        "task": "repaint",
        "model": "test/native-music",
        "prompt": "Warm acoustic cello",
        "source_sample_rate": 8000,
        "region_end_frame": 8000,
    }
    if seed is not None:
        body["seed"] = seed
    return body


async def create(
    client: httpx.AsyncClient, body: dict[str, str | int] | None = None
) -> MusicEditResource:
    """Submit one real bounded multipart source and parse the public receipt."""
    _, _, original, _ = music_edit_fixture()
    response = await client.post(
        "/v1/music/edits",
        data={"request": json.dumps(body if body is not None else request())},
        files={"file": ("original.wav", original, "audio/wav")},
    )
    assert response.status_code == 202, response.text
    return MusicEditResource.model_validate_json(response.content)


async def terminal(client: httpx.AsyncClient, identifier: str) -> MusicEditResource:
    """Observe committed terminal status through the actual HTTP route."""
    with anyio.fail_after(5):
        while True:
            response = await client.get(f"/v1/music/edits/{identifier}")
            assert response.status_code == 200, response.text
            actual = MusicEditResource.model_validate_json(response.content)
            if actual.status in {"completed", "failed", "cancelled"}:
                return actual
            await anyio.sleep(0.01)


@pytest.mark.parametrize("elapsed_seconds", [3600, 7201])
async def test_render_budget_allows_slow_cpu_edits_but_expires_without_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, elapsed_seconds: int
) -> None:
    """Keep a placed edit past forty minutes, or cancel and drain at two hours."""
    now = 100.0
    real_time = music_edit_api.time.time
    # Replace only this API's clock; AnyIO and source/media deadlines must keep
    # their real clocks while actual HTTP, custody and cleanup effects run.
    monkeypatch.setattr(
        music_edit_api,
        "time",
        SimpleNamespace(monotonic=lambda: now, time=real_time),
    )
    async with fixture(tmp_path) as (harness, client):
        accepted = await create(client)
        identifier = CommandId(accepted.id)
        await harness.task(identifier)
        swept = anyio.Event()
        expire_content = harness.api.expire_content

        async def observe_sweep(timestamp: int) -> None:
            await expire_content(timestamp)
            swept.set()

        monkeypatch.setattr(harness.api, "expire_content", observe_sweep)
        now += elapsed_seconds
        harness.api.tasks.start_soon(harness.api.sweep)
        with anyio.fail_after(5):
            await swept.wait()

        if elapsed_seconds < 7200:
            current = await client.get(f"/v1/music/edits/{accepted.id}")
            assert MusicEditResource.model_validate_json(current.content).status == (
                "in_progress"
            )
            assert not any(
                isinstance(command, TaskCancelled) for command in harness.commands
            )
            data = await harness.render(identifier, media_first=True)
            completed = await terminal(client, accepted.id)
            assert completed.status == "completed" and completed.seed == 0
            assert completed.output is not None
            content = await client.get(f"/v1/music/edits/{accepted.id}/content")
            assert content.status_code == 200 and content.content == data
        else:
            ended = await terminal(client, accepted.id)
            assert ended.status == "failed" and ended.output is None
            assert ended.error == (
                "Music edit placement, render or media deadline expired."
            )
            assert (
                sum(isinstance(command, TaskCancelled) for command in harness.commands)
                == 1
            )
            assert identifier not in harness.api.active
            assert harness.api.sources.ready_path(identifier) is None
            assert not (harness.api.media.root / "wav" / accepted.id).exists()
            assert harness.api.media.reserved_bytes() == 0
            # Repeated cancellation is idempotent, and the timed-out job never
            # creates another command or resumes a partial result.
            response = await client.post(f"/v1/music/edits/{accepted.id}/cancel")
            assert response.status_code == 200
            assert (
                sum(isinstance(command, TaskCancelled) for command in harness.commands)
                == 1
            )
        assert (
            sum(isinstance(command, MusicEditCommand) for command in harness.commands)
            == 1
        )


@pytest.mark.parametrize("media_first", [False, True])
async def test_native_task_finalization_waits_for_return_acknowledgement(
    tmp_path: Path, media_first: bool
) -> None:
    """TaskFinished cannot cancel a worker still returning its independently owned WAV."""
    async with fixture(tmp_path) as (harness, client):
        accepted = await create(client)
        identifier = CommandId(accepted.id)
        acknowledgement_started, release_acknowledgement = anyio.Event(), anyio.Event()

        async def delayed_acknowledgement(packet: OutputMediaPacket) -> None:
            if packet.kind == "accepted":
                acknowledgement_started.set()
                await release_acknowledgement.wait()
            await harness.send_output(packet)

        async def worker_return(packet: OutputMediaPacket) -> None:
            # A real worker cancels its output owner when TaskFinished deletes
            # the task, so finalization must not precede any of these frames.
            assert identifier not in harness.finalized
            await harness.api.receive_output(packet)

        harness.api.send_output = delayed_acknowledgement
        try:
            await harness.render(
                identifier, media_first=media_first, send=worker_return
            )
            with anyio.fail_after(3):
                await acknowledgement_started.wait()
            assert identifier not in harness.finalized
            assert (await harness.api.job(identifier)).status != "completed"
        finally:
            release_acknowledgement.set()
        assert (await terminal(client, accepted.id)).status == "completed"
        with anyio.fail_after(3):
            while identifier not in harness.finalized:
                await anyio.sleep(0.01)
        assert harness.finalized.count(identifier) == 1


@pytest.mark.parametrize("media_first", [False, True])
async def test_http_native_handshake_render_and_verified_download(
    tmp_path: Path, media_first: bool
) -> None:
    """Both DATA/media orderings settle only after independent bytes and ACK drain."""
    async with fixture(tmp_path) as (harness, client):
        accepted = await create(client)
        assert (
            accepted.output is None and accepted.seed == 0 and harness.seed_draws == 0
        )
        assert (
            await client.get(f"/v1/music/edits/{accepted.id}/content")
        ).status_code == 409
        identifier = CommandId(accepted.id)
        data = await harness.render(identifier, media_first=media_first)
        ready = await terminal(client, accepted.id)
        assert ready.status == "completed" and ready.output is not None
        assert ready.source == accepted.source and ready.seed == accepted.seed
        assert any(frame.kind == "accepted" for frame in harness.output_controls)
        assert {frame.payload for frame in harness.source_frames} == {
            "music_edit_source"
        }
        download = await client.get(f"/v1/music/edits/{accepted.id}/content")
        assert download.status_code == 200 and download.content == data
        assert download.headers["etag"] == '"' + ready.output.result.output.sha256 + '"'
        assert int(download.headers["content-length"]) == len(data)
        listing = MusicEditListResponse.model_validate_json(
            (await client.get("/v1/music/edits")).content
        )
        assert [item.id for item in listing.data] == [accepted.id]
        assert (await client.get("/v1/music/legacy-song")).text == '"legacy-song"'
        assert (
            await client.delete(f"/v1/music/edits/{accepted.id}")
        ).status_code == 200
        assert (await client.get(f"/v1/music/edits/{accepted.id}")).status_code == 404
        assert harness.api.media.reserved_bytes() == 0


async def test_missing_seed_is_drawn_once_and_restart_never_replays(
    tmp_path: Path,
) -> None:
    """A queued immutable intent keeps its source/seed when the API owner reopens."""
    async with fixture(tmp_path) as (harness, client):
        harness.place = False
        accepted = await create(client, request(seed=None))
        assert accepted.seed == 77 and harness.seed_draws == 1
        assert (
            len(
                [
                    item
                    for item in harness.commands
                    if isinstance(item, MusicEditCommand)
                ]
            )
            == 1
        )
    async with fixture(tmp_path) as (reopened, client):
        failed = MusicEditResource.model_validate_json(
            (await client.get(f"/v1/music/edits/{accepted.id}")).content
        )
        assert (
            failed.status == "failed"
            and failed.seed == 77
            and failed.source == accepted.source
        )
        assert not reopened.commands and reopened.seed_draws == 0
        assert (
            await client.get(f"/v1/music/edits/{accepted.id}/content")
        ).status_code == 409


@pytest.mark.parametrize("mode", ["add_part", "repaint", "continue"])
async def test_all_closed_edit_operations_accept_source_clock_controls(
    tmp_path: Path, mode: str
) -> None:
    """Actual multipart parsing retains each operation's exact allowed controls."""
    body = request()
    body["task"] = mode
    if mode == "add_part":
        body["part_class"] = "strings"
    elif mode == "continue":
        body.pop("region_end_frame")
        body["target_frames"] = 16000
    async with fixture(tmp_path) as (harness, client):
        harness.place = False
        accepted = await create(client, body)
        assert accepted.task == mode and accepted.seed == 0
        assert accepted.frame_sample_rate == 48000
        assert accepted.target_frames == (96000 if mode == "continue" else 48000)
        assert accepted.region_start_frame == (48000 if mode == "continue" else 0)
        assert accepted.region_end_frame == (96000 if mode == "continue" else 48000)
        assert accepted.part_class == ("strings" if mode == "add_part" else None)
        assert (
            len(
                [
                    item
                    for item in harness.commands
                    if isinstance(item, MusicEditCommand)
                ]
            )
            == 1
        )


@pytest.mark.parametrize(
    "change", ["native_options", "path", "clock", "range", "task_controls", "blank"]
)
async def test_invalid_metadata_never_dispatches_or_allocates_an_intent(
    tmp_path: Path, change: str
) -> None:
    """Strict per-task fields and measured source truth precede persistent acceptance."""
    body = request()
    if change == "native_options":
        body["inference_steps"] = 4
    elif change == "path":
        body["source_path"] = "/not-an-input.wav"
    elif change == "clock":
        body["source_sample_rate"] = 44100
    elif change == "range":
        body["region_end_frame"] = 8001
    elif change == "task_controls":
        body["target_frames"] = 16000
    else:
        body["prompt"] = "   "
    _, _, original, _ = music_edit_fixture()
    async with fixture(tmp_path) as (harness, client):
        response = await client.post(
            "/v1/music/edits",
            data={"request": json.dumps(body)},
            files={"file": ("original.wav", original, "audio/wav")},
        )
        assert response.status_code == 422, response.text
        assert not harness.commands and not harness.api.active
        assert harness.api.uploads.borrowed_tokens == 0
        assert not (tmp_path / "api/index/jobs.json").exists()


async def test_native_metadata_cannot_bless_a_different_peak(tmp_path: Path) -> None:
    """A hash-valid container still fails if native result facts disagree with bytes."""
    async with fixture(tmp_path) as (harness, client):
        accepted = await create(client)
        await harness.render(CommandId(accepted.id), media_first=True, damaged=True)
        failed = await terminal(client, accepted.id)
        assert failed.status == "failed" and failed.output is None
        assert (
            await client.get(f"/v1/music/edits/{accepted.id}/content")
        ).status_code == 409
        assert harness.api.media.reserved_bytes() == 0
        assert not any(frame.kind == "accepted" for frame in harness.output_controls)


async def test_uncertain_index_write_returns_503_without_dispatch_or_stale_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real file-fsync failure fences admission/read authority until reopening."""
    original_sync = os.fsync

    def refuse_regular_file(descriptor: int) -> None:
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("Injected private index sync failure")
        original_sync(descriptor)

    async with fixture(tmp_path) as (harness, client):
        monkeypatch.setattr(os, "fsync", refuse_regular_file)
        _, _, original, _ = music_edit_fixture()
        response = await client.post(
            "/v1/music/edits",
            data={"request": json.dumps(request())},
            files={"file": ("source.wav", original, "audio/wav")},
        )
        assert (
            response.status_code == 503
            and not harness.commands
            and not harness.api.active
        )
        read = await client.get("/v1/music/edits")
        assert read.status_code == 503 and "Restart the API" in read.text
        monkeypatch.undo()


async def test_cancel_is_terminal_and_ignores_late_native_data(tmp_path: Path) -> None:
    """Cancellation drains one accepted operation without replacement or rerandomization."""
    async with fixture(tmp_path) as (harness, client):
        accepted = await create(client)
        identifier = CommandId(accepted.id)
        task = await harness.task(identifier)
        response = await client.post(f"/v1/music/edits/{accepted.id}/cancel")
        cancelled = MusicEditResource.model_validate_json(response.content)
        assert cancelled.status == "cancelled" and cancelled.seed == 0
        _, output, _, _ = music_edit_fixture(accepted.id)
        assert output.source == task.task_params.source
        await harness.api.receive_output(
            OutputMediaPacket(
                source_node=NodeId("node-0"),
                target_node=NodeId("owner"),
                command_id=identifier,
                model=ModelId(task.task_params.model),
                purpose="music_edit",
                kind="opened",
                sequence=0,
                total_chunks=1,
                total_bytes=100,
                content_type="audio/wav",
            )
        )
        assert (await terminal(client, accepted.id)).status == "cancelled"
        with anyio.fail_after(5):
            while not any(isinstance(item, TaskCancelled) for item in harness.commands):
                await anyio.sleep(0.01)
        assert not harness.api.active and harness.api.media.reserved_bytes() == 0
        assert (
            len(
                [
                    item
                    for item in harness.commands
                    if isinstance(item, MusicEditCommand)
                ]
            )
            == 1
        )


async def test_native_cancellation_submission_precedes_stream_finalization(
    tmp_path: Path,
) -> None:
    """A delayed cancel cannot let TaskFinished remove its still-running task first."""
    async with fixture(tmp_path) as (harness, client):
        accepted = await create(client)
        identifier = CommandId(accepted.id)
        await harness.task(identifier)
        started, release = anyio.Event(), anyio.Event()

        async def delayed(command: Command, session: SystemId) -> None:
            if isinstance(command, TaskCancelled):
                started.set()
                await release.wait()
            await harness.send_command(command, session)

        harness.api.send_command = delayed
        await harness.queues[identifier].send(
            ErrorChunk(
                model=ModelId("test/native-music"),
                error_message="fixture native failure",
            )
        )
        try:
            with anyio.fail_after(3):
                await started.wait()
            assert identifier not in harness.finalized
            assert (await terminal(client, accepted.id)).status == "failed"
            assert identifier not in harness.finalized
        finally:
            release.set()
        with anyio.fail_after(3):
            while identifier not in harness.finalized:
                await anyio.sleep(0.01)
        assert any(isinstance(command, TaskCancelled) for command in harness.commands)


async def test_background_storage_failure_cancels_native_work_and_keeps_host_alive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fenced result metadata retires the edit service without crashing host tasks."""
    sync = os.fsync

    def refuse_regular_file(descriptor: int) -> None:
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("Injected result index sync failure")
        sync(descriptor)

    async with fixture(tmp_path) as (harness, client):
        accepted = await create(client)
        identifier = CommandId(accepted.id)
        await harness.task(identifier)
        with anyio.fail_after(3):
            while harness.api._source_senders:
                await anyio.sleep(0.01)
        swept = anyio.Event()

        async def sweep() -> None:
            await harness.api.sweep()
            swept.set()

        harness.api.tasks.start_soon(sweep)
        monkeypatch.setattr(os, "fsync", refuse_regular_file)
        _, output, _, _ = music_edit_fixture(accepted.id)
        await harness.queues[identifier].send(
            MusicEditingChunk(
                model=ModelId("test/native-music"), finish_reason="stop", output=output
            )
        )
        with anyio.fail_after(3):
            await swept.wait()
            while identifier not in harness.finalized:
                await anyio.sleep(0.01)
        assert not harness.api.active
        assert any(isinstance(command, TaskCancelled) for command in harness.commands)
        assert (await client.get(f"/v1/music/edits/{accepted.id}")).status_code == 503
        assert (await client.get("/v1/music/unrelated-song")).status_code == 200
        monkeypatch.undo()
        with anyio.fail_after(5):
            while harness.api._cleanup:
                await anyio.sleep(0.01)
        assert harness.api.media.reserved_bytes() == 0
        assert (await client.get(f"/v1/music/edits/{accepted.id}")).status_code == 503


@pytest.mark.parametrize("transport", ["source", "output", "invalid_output"])
async def test_transport_failure_with_uncertain_index_write_fences_only_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transport: str
) -> None:
    """A nested failure handler must not let journal uncertainty kill the host."""
    sync = os.fsync

    def refuse_metadata(descriptor: int) -> None:
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("injected failed-transfer index sync failure")
        sync(descriptor)

    async with fixture(tmp_path) as (harness, client):
        accepted = [await create(client)]
        await harness.task(CommandId(accepted[0].id))
        # A second accepted job is still waiting for authoritative placement;
        # the single loaded instance cannot run two native edits concurrently.
        harness.place = False
        accepted.append(await create(client))
        identifiers = [CommandId(job.id) for job in accepted]
        with anyio.fail_after(5):
            while harness.api._source_senders:
                await anyio.sleep(0.01)
        async with harness.api.lock:
            for identifier in identifiers:
                harness.api.media.open_assembly(
                    identifier, total_bytes=100, total_chunks=1
                )
        finished = anyio.Event()

        async def receive() -> None:
            if transport == "source":
                await harness.api.receive_source(
                    VisionMediaPacket(
                        source_node=NodeId("node-0"),
                        target_node=NodeId("owner"),
                        command_id=identifiers[0],
                        model=ModelId("test/native-music"),
                        payload="music_edit_source",
                        kind="transport_failed",
                        sequence=0,
                        error_message="fixture source transfer failed",
                    )
                )
            elif transport == "output":
                await harness.api.receive_output(
                    OutputMediaPacket(
                        source_node=NodeId("node-0"),
                        target_node=NodeId("owner"),
                        command_id=identifiers[0],
                        model=ModelId("test/native-music"),
                        purpose="music_edit",
                        kind="transport_failed",
                        sequence=0,
                        error_message="fixture output transfer failed",
                    )
                )
            else:
                await harness.api.receive_output(
                    OutputMediaPacket(
                        source_node=NodeId("node-0"),
                        target_node=NodeId("owner"),
                        command_id=identifiers[0],
                        model=ModelId("test/native-music"),
                        purpose="music_edit",
                        kind="opened",
                        sequence=0,
                        total_bytes=100,
                        total_chunks=1,
                        content_type="audio/mpeg",
                    )
                )
            finished.set()

        monkeypatch.setattr(os, "fsync", refuse_metadata)
        try:
            harness.api.tasks.start_soon(receive)
            with anyio.fail_after(5):
                await finished.wait()
                while len(harness.finalized) != 2:
                    await anyio.sleep(0.01)
            assert not harness.api.active
            assert (
                sum(isinstance(command, TaskCancelled) for command in harness.commands)
                == 2
            )
            assert harness.api.media.reserved_bytes() == 0
            assert (await client.get("/v1/music/unrelated-song")).status_code == 200
            for job in accepted:
                assert (
                    await client.get(f"/v1/music/edits/{job.id}")
                ).status_code == 503
        finally:
            monkeypatch.undo()
        with anyio.fail_after(5):
            while harness.api._cleanup:
                await anyio.sleep(0.01)
        assert all(
            harness.api.sources.ready_path(identifier) is None
            for identifier in identifiers
        )


async def test_expiry_cleanup_failure_keeps_host_alive_and_retries_retained_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actual shared-group sweeper retains a failed unlink and retries it."""
    async with fixture(tmp_path) as (harness, client):
        accepted = await create(client)
        identifier = CommandId(accepted.id)
        data = await harness.render(identifier, media_first=True)
        completed = await terminal(client, accepted.id)
        assert completed.expires_at is not None
        expiry = completed.expires_at
        directory = harness.api.media.root / "wav" / accepted.id
        original = shutil.rmtree
        refused, allow_cleanup = threading.Event(), threading.Event()

        def remove(path: Path, *, ignore_errors: bool = False) -> None:
            if path == directory and not allow_cleanup.is_set():
                refused.set()
                return
            original(path, ignore_errors=ignore_errors)

        monkeypatch.setattr(shutil, "rmtree", remove)
        monkeypatch.setattr("skulk.api.music_edit_api.time.time", lambda: expiry)
        try:
            harness.api.tasks.start_soon(harness.api.sweep)
            with anyio.fail_after(5):
                while not refused.is_set():
                    await anyio.sleep(0.01)
            # Observe the event-loop task after its joined filesystem effect.
            # The reservation cannot disappear when deletion left files behind.
            async with harness.api.lock:
                assert identifier not in harness.api._expired
                assert harness.api.media.reserved_bytes() == len(data)
                assert directory.is_dir()
            assert (await client.get("/v1/music/unrelated-song")).status_code == 200
            assert (
                await client.get(f"/v1/music/edits/{accepted.id}")
            ).status_code == 200
            allow_cleanup.set()
            with anyio.fail_after(5):
                while identifier not in harness.api._expired:
                    await anyio.sleep(0.01)
            assert harness.api.media.reserved_bytes() == 0
            assert not directory.exists()
            assert (await client.get("/v1/music/unrelated-song")).status_code == 200
        finally:
            monkeypatch.undo()


@pytest.mark.parametrize("failed_store", ["media", "source", "both"])
@pytest.mark.parametrize("trigger", ["cancel", "timeout"])
async def test_terminal_partial_cleanup_retains_admission_until_both_stores_drain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_store: str,
    trigger: str,
) -> None:
    """Real partial copies survive cleanup errors without ending the API host."""
    async with fixture(tmp_path) as (harness, client):
        harness.place = False
        accepted = [await create(client) for _ in range(4)]
        identifiers = tuple(CommandId(job.id) for job in accepted)
        media_paths = frozenset(
            harness.api.media.root / "wav" / job.id for job in accepted
        )
        source_paths = frozenset(harness.api.sources.root / job.id for job in accepted)
        blocked = media_paths if failed_store == "media" else source_paths
        if failed_store == "both":
            blocked = media_paths | source_paths
        async with harness.api.lock:
            for identifier in identifiers:
                harness.api.media.open_assembly(
                    identifier, total_bytes=100, total_chunks=1
                )
                if trigger == "timeout":
                    harness.api.active[identifier].deadline = 0
        original = shutil.rmtree
        allow_cleanup = threading.Event()

        def remove(path: Path, *, ignore_errors: bool = False) -> None:
            if path in blocked and not allow_cleanup.is_set():
                return
            original(path, ignore_errors=ignore_errors)

        monkeypatch.setattr(shutil, "rmtree", remove)
        try:
            if trigger == "timeout":
                harness.api.tasks.start_soon(harness.api.sweep)
            else:
                for job in accepted:
                    response = await client.post(f"/v1/music/edits/{job.id}/cancel")
                    assert response.status_code == 200, response.text
            for job in accepted:
                ended = await terminal(client, job.id)
                assert ended.status == (
                    "failed" if trigger == "timeout" else "cancelled"
                )
            with anyio.fail_after(5):
                while len(harness.finalized) < 4:
                    await anyio.sleep(0.01)
            assert not harness.api.active
            assert len(harness.api._cleanup) == 4
            assert harness.api.media.reserved_bytes() == (
                0 if failed_store == "source" else 400
            )
            assert sum(
                harness.api.sources.ready_path(identifier) is not None
                for identifier in identifiers
            ) == (0 if failed_store == "media" else 4)
            assert (await client.get("/v1/music/unrelated-song")).status_code == 200
            rejected = await client.post(
                "/v1/music/edits",
                data={"request": json.dumps(request())},
                files={"file": ("original.wav", music_edit_fixture()[2], "audio/wav")},
            )
            assert rejected.status_code == 503
            assert (
                sum(
                    isinstance(command, MusicEditCommand)
                    for command in harness.commands
                )
                == 4
            )
            assert (
                await client.post(f"/v1/music/edits/{accepted[0].id}/cancel")
            ).status_code == 200
            assert (
                sum(isinstance(command, TaskCancelled) for command in harness.commands)
                == 4
            )
            allow_cleanup.set()
            with anyio.fail_after(5):
                while harness.api._cleanup:
                    await anyio.sleep(0.01)
            assert harness.api.media.reserved_bytes() == 0
            assert all(
                harness.api.sources.ready_path(identifier) is None
                for identifier in identifiers
            )
            assert all(not path.exists() for path in media_paths | source_paths)
            assert (await client.get("/v1/music/unrelated-song")).status_code == 200
        finally:
            monkeypatch.undo()


async def test_completed_result_survives_deferred_original_cleanup_and_delete_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Input cleanup failure cannot discard completed audio or stop its API host."""
    original = shutil.rmtree
    allow_cleanup = threading.Event()
    async with fixture(tmp_path) as (harness, client):

        def remove(path: Path, *, ignore_errors: bool = False) -> None:
            if (
                path.parent == harness.api.root / "inputs"
                and not allow_cleanup.is_set()
            ):
                return
            original(path, ignore_errors=ignore_errors)

        monkeypatch.setattr(shutil, "rmtree", remove)
        try:
            accepted = await create(client)
            identifier = CommandId(accepted.id)
            data = await harness.render(identifier, media_first=False)
            assert (await terminal(client, accepted.id)).status == "completed"
            assert identifier in harness.api._cleanup
            assert harness.api.sources.ready_path(identifier) is not None
            content = await client.get(f"/v1/music/edits/{accepted.id}/content")
            assert content.status_code == 200 and content.content == data
            refused = await client.delete(f"/v1/music/edits/{accepted.id}")
            assert refused.status_code == 503
            assert (
                await client.get(f"/v1/music/edits/{accepted.id}")
            ).status_code == 200
            assert (await client.get("/v1/music/unrelated-song")).status_code == 200
            allow_cleanup.set()
            with anyio.fail_after(5):
                while identifier in harness.api._cleanup:
                    await anyio.sleep(0.01)
            assert harness.api.sources.ready_path(identifier) is None
            content = await client.get(f"/v1/music/edits/{accepted.id}/content")
            assert content.status_code == 200 and content.content == data
            assert (
                await client.delete(f"/v1/music/edits/{accepted.id}")
            ).status_code == 200
            assert (
                await client.get(f"/v1/music/edits/{accepted.id}")
            ).status_code == 404
        finally:
            monkeypatch.undo()


async def test_failed_restart_orphan_cleanup_refuses_edit_readiness_not_the_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh reservation tables cannot omit bytes that orphan cleanup left behind."""
    old = MusicEditMediaStore(tmp_path / "api/media")
    _, output, _, data = music_edit_fixture("orphan")
    identifier = CommandId("orphan")
    old.open_assembly(identifier, total_bytes=len(data), total_chunks=1)
    old.append(identifier, 1, data)
    stored = old.commit(identifier, output.result.output, total_chunks=1)
    old.close()

    def retain(_path: Path, *, ignore_errors: bool = False) -> None:
        """Represent a refused physical deletion in the shared assembler."""

    monkeypatch.setattr(shutil, "rmtree", retain)
    async with fixture(tmp_path) as (harness, client):
        try:
            response = await client.get("/v1/music/edits")
            assert response.status_code == 503
            assert harness.api._jobs is None and harness.api._media is None
            assert stored.file_path.read_bytes() == data
            assert (await client.get("/v1/music/unrelated-song")).status_code == 200
            monkeypatch.undo()
            response = await client.get("/v1/music/edits")
            assert response.status_code == 200
            assert (
                MusicEditListResponse.model_validate_json(response.content).data == ()
            )
            assert not stored.file_path.exists()
        finally:
            monkeypatch.undo()


@pytest.mark.parametrize("media_first", [False, True])
async def test_same_node_router_drains_result_and_reverse_ack(
    tmp_path: Path, media_first: bool
) -> None:
    """Actual zero-buffer routing excludes local ACK echoes in either arrival order."""
    async with fixture(tmp_path) as (harness, client):
        harness.api.node_id = NodeId("node-0")
        network_send, network_receive = channel[OutboundPacket](1)
        router = TopicRouter(OUTPUT_MEDIA, network_send, local_routing_key="node-0")
        api_receive = router.new_receiver(accept=output_media_for_api)
        worker_receive = router.new_receiver(accept=output_media_for_worker)
        sender = router.new_sender()
        harness.api.send_output = sender.send
        api_frames: list[OutputMediaPacket] = []
        worker_frames: list[OutputMediaPacket] = []

        async def api_ingress() -> None:
            with api_receive:
                async for packet in api_receive:
                    api_frames.append(packet)
                    await harness.api.receive_output(packet)

        async def worker_ingress() -> None:
            with worker_receive:
                async for packet in worker_receive:
                    worker_frames.append(packet)

        async with anyio.create_task_group() as group:
            group.start_soon(router.run)
            group.start_soon(api_ingress)
            group.start_soon(worker_ingress)
            try:
                accepted = await create(client)
                with anyio.fail_after(5):
                    await harness.render(
                        CommandId(accepted.id),
                        media_first=media_first,
                        send=sender.send,
                    )
                    ready = await terminal(client, accepted.id)
                assert ready.status == "completed"
                assert any(packet.kind == "completed" for packet in api_frames)
                assert [packet.kind for packet in worker_frames] == ["accepted"]
                assert all(
                    packet.kind not in {"accepted", "cancelled"}
                    for packet in api_frames
                )
                assert (
                    await client.get(f"/v1/music/edits/{accepted.id}/content")
                ).status_code == 200
                with pytest.raises(anyio.WouldBlock):
                    network_receive.receive_nowait()
            finally:
                await router.shutdown()
                group.cancel_scope.cancel()


async def test_operator_gateway_requires_authentication_before_edit_owner(
    tmp_path: Path,
) -> None:
    """Real bearer validation protects these six routes before storage/body parsing."""
    service, credential = paired_service(tmp_path / "authority")
    async with fixture(tmp_path / "studio") as (harness, _client):
        app = FastAPI()
        harness.api.register(app)
        protected = OperatorGatewayAuthorization(app, service)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=protected), base_url="http://api.test"
        ) as client:
            assert (await client.get("/v1/music/edits")).status_code == 401
            assert (
                await client.post("/v1/music/edits", content=b"unparsed")
            ).status_code == 401
            assert (
                not harness.api.active
                and not (harness.api.root / "index/jobs.json").exists()
            )
            authenticated = await client.get(
                "/v1/music/edits",
                headers={"Authorization": "Bearer " + credential.access_token},
            )
            assert authenticated.status_code == 200
        for method, path in (
            ("POST", "/v1/music/edits"),
            ("GET", "/v1/music/edits"),
            ("GET", "/v1/music/edits/job"),
            ("GET", "/v1/music/edits/job/content"),
            ("POST", "/v1/music/edits/job/cancel"),
            ("DELETE", "/v1/music/edits/job"),
        ):
            assert _required_scopes(
                {"type": "http", "method": method, "path": path}
            ) == ("chat:write",)
        assert _required_scopes(
            {"type": "http", "method": "GET", "path": "/v1/music/edits-as-legacy-id"}
        ) == ("cluster:read",)


async def test_all_edit_routes_are_present_in_openapi(tmp_path: Path) -> None:
    """New collection routes precede the legacy dynamic music ID route."""
    async with fixture(tmp_path) as (_harness, client):
        response = await client.get("/openapi.json")
        assert response.status_code == 200
        spec = json_object(cast(object, json.loads(response.content)))
        paths = json_object(spec["paths"])
        for path, methods in {
            "/v1/music/edits": {"get", "post"},
            "/v1/music/edits/{edit_id}": {"get", "delete"},
            "/v1/music/edits/{edit_id}/content": {"get"},
            "/v1/music/edits/{edit_id}/cancel": {"post"},
        }.items():
            for method in methods:
                operation = json_object(json_object(paths[path])[method])
                assert (
                    operation["summary"]
                    and operation["description"]
                    and operation["tags"] == ["Music Editing"]
                )
        operation = json_object(json_object(paths["/v1/music/edits"])["post"])
        body = json_object(operation["requestBody"])
        content = json_object(body["content"])
        schema = json_object(json_object(content["multipart/form-data"])["schema"])
        assert (
            schema["required"] == ["request", "file"]
            and schema["additionalProperties"] is False
        )
        properties = json_object(schema["properties"])
        metadata = json_object(properties["request"])
        assert metadata["contentMediaType"] == "application/json"
        contract = json_object(metadata["contentSchema"])
        alternatives = contract["oneOf"]
        assert isinstance(alternatives, list)
        variants = [json_object(item) for item in cast(list[object], alternatives)]
        assert len(variants) == 3
        assert all(item["additionalProperties"] is False for item in variants)
        assert {
            json_object(json_object(item["properties"])["task"])["const"]
            for item in variants
        } == {"add_part", "repaint", "continue"}
        assert "#/$defs/" not in json.dumps(contract)


def json_object(value: object) -> dict[str, object]:
    """Validate JSON dictionaries before traversing the actual OpenAPI document."""
    assert isinstance(value, dict)
    candidate = cast(dict[object, object], value)
    assert all(isinstance(key, str) for key in candidate)
    return cast(dict[str, object], candidate)
