# pyright: reportPrivateUsage=false
"""Actual multipart HTTP, master placement and private addressed media composition.

Inference output is controlled PCM24 fixture audio. These tests exercise the
real job, source, worker handshake, four-role producer and receiving stores;
they do not claim a native model/platform or listening qualification.
"""

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import cast

import anyio
import httpx
import pytest
from fastapi import FastAPI
from starlette import formparsers
from starlette.types import Message

from skulk.api import separation_api as separation_module
from skulk.api.separation_api import SeparationAPI
from skulk.api.types.separation import SeparationListResponse, SeparationResource
from skulk.master.main import separation_generation_events
from skulk.master.tests.test_separation_admission import state
from skulk.routing.output_media import OutputMediaPacket
from skulk.routing.vision_media import VisionMediaPacket
from skulk.shared.models.model_cards import ModelCard, ModelId
from skulk.shared.types.audio_separation import MAX_SEPARATION_SOURCE_BYTES
from skulk.shared.types.chunks import ErrorChunk, SeparationChunk
from skulk.shared.types.commands import AudioSeparation as SeparationCommand
from skulk.shared.types.commands import Command, TaskCancelled
from skulk.shared.types.common import CommandId, NodeId, SystemId
from skulk.shared.types.events import TaskCreated
from skulk.shared.types.state import State
from skulk.shared.types.tasks import AudioSeparation
from skulk.shared.types.worker.runners import RunnerReady
from skulk.utils.channels import Receiver, Sender, channel
from skulk.utils.task_group import TaskGroup
from skulk.worker.separation_input import SeparationInputReceiver
from skulk.worker.separation_output import SeparationOutputProducer
from skulk.worker.separation_sources import SeparationSourceStore
from skulk.worker.tests.test_separation_input import original
from skulk.worker.tests.test_separation_output import result


def json_object(value: object) -> dict[str, object]:
    """Narrow an arbitrary JSON object while retaining runtime key checks."""
    assert isinstance(value, dict)
    candidate = cast(dict[object, object], value)
    assert all(isinstance(key, str) for key in candidate)
    return cast(dict[str, object], candidate)


@dataclass
class Harness:
    root: Path
    snapshot: State
    session: SystemId = field(default_factory=SystemId)
    queues: dict[CommandId, Sender[SeparationChunk | ErrorChunk]] = field(
        default_factory=dict
    )
    commands: list[Command] = field(default_factory=list)
    finalizations: list[CommandId] = field(default_factory=list)
    source_frames: list[VisionMediaPacket] = field(default_factory=list)
    output_frames: list[OutputMediaPacket] = field(default_factory=list)
    worker: SeparationInputReceiver | None = None
    service: SeparationAPI | None = None
    producer: SeparationOutputProducer | None = None

    @property
    def api(self) -> SeparationAPI:
        """Return the actual owner after fixture construction."""
        assert self.service is not None
        return self.service

    async def send_command(self, command: Command, session: SystemId) -> None:
        """Run actual admission; make no network or native inference claim."""
        if session != self.session:
            raise ValueError("old cluster session")
        self.commands.append(command)
        if isinstance(command, SeparationCommand):
            events, _ = separation_generation_events(self.snapshot, command)
            assert isinstance(events[0], TaskCreated) and isinstance(
                events[0].task, AudioSeparation
            )
            task = events[0].task
            self.snapshot = self.snapshot.model_copy(
                update={"tasks": dict(self.snapshot.tasks) | {task.task_id: task}}
            )
            await self.api.observe_task(task)
            assert self.worker is not None
            await self.worker.begin(command.command_id)

    def open_queue(
        self, identifier: CommandId
    ) -> Receiver[SeparationChunk | ErrorChunk]:
        send, receive = channel[SeparationChunk | ErrorChunk]()
        self.queues[identifier] = send
        return receive

    def close_queue(self, identifier: CommandId) -> None:
        """Wake the real metadata consumer when cancellation has no DATA reply."""
        if sender := self.queues.get(identifier):
            sender.close()

    async def finalize(self, identifier: CommandId, _session: SystemId) -> None:
        self.finalizations.append(identifier)
        sender = self.queues.pop(identifier, None)
        if sender is not None:
            sender.close()

    async def send_source(self, packet: VisionMediaPacket) -> None:
        self.source_frames.append(packet)
        assert self.worker is not None
        await self.worker.receive(packet)

    async def send_output(self, packet: OutputMediaPacket) -> None:
        self.output_frames.append(packet)
        if self.producer is not None:
            self.producer.receive(packet)

    async def card(self, model: ModelId) -> ModelCard:
        instance = next(iter(self.snapshot.instances.values()))
        card = next(
            iter(instance.shard_assignments.runner_to_shard.values())
        ).model_card
        assert model == card.model_id
        return card

    async def task(self, identifier: CommandId) -> AudioSeparation:
        with anyio.fail_after(5):
            while True:
                matches = [
                    task
                    for task in self.snapshot.tasks.values()
                    if isinstance(task, AudioSeparation)
                    and task.command_id == identifier
                ]
                if (
                    matches
                    and self.worker is not None
                    and self.worker.ready(identifier)
                ):
                    return matches[0]
                await anyio.sleep(0.01)

    async def render(self, identifier: CommandId) -> AudioSeparation:
        """Generate fixture files, publish actual metadata and deliver real bytes."""
        task = await self.task(identifier)
        output = result(self.root / "native", task)

        async def fail(_task: AudioSeparation, message: str) -> None:
            raise AssertionError(message)

        self.producer = SeparationOutputProducer(
            NodeId("node-0"), self.api.receive_output, fail
        )
        self.producer.register(task, self.root / "native", output)
        await self.queues[identifier].send(
            SeparationChunk(
                model=ModelId(task.task_params.model),
                finish_reason="stop",
                output=output,
            )
        )
        await self.producer.deliver(identifier)
        return task


@asynccontextmanager
async def fixture(root: Path) -> AsyncIterator[tuple[Harness, httpx.AsyncClient]]:
    harness = Harness(root, state(RunnerReady()))
    sources = SeparationSourceStore(root / "worker-inputs")
    group = TaskGroup()
    service: SeparationAPI | None = None
    try:
        async with group as active_group:
            service = SeparationAPI(
                NodeId("owning-api"),
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
            )
            harness.service = service

            async def fail(task: AudioSeparation, message: str) -> None:
                await service.fail(task.command_id, message, cancel=True)

            harness.worker = SeparationInputReceiver(
                NodeId("node-0"),
                lambda: harness.snapshot,
                sources,
                service.receive_source,
                fail,
            )
            app = FastAPI()
            service.register(app)
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


async def create(client: httpx.AsyncClient) -> CommandId:
    """Use the exact two multipart field names shipped by Studio's SDK adapter."""
    response = await client.post(
        "/v1/audio/separations",
        data={"request": json.dumps({"model": "adefossez/HTDemucs"})},
        files={"file": ("original.wav", original(), "audio/wav")},
    )
    assert response.status_code == 202, response.text
    return CommandId(SeparationResource.model_validate_json(response.content).id)


async def test_multipart_to_placed_worker_four_stems_and_download(
    tmp_path: Path,
) -> None:
    """A completed public receipt requires the real private four-file round trip."""
    async with fixture(tmp_path) as (harness, client):
        identifier = await create(client)
        before = await client.get(f"/v1/audio/separations/{identifier}")
        assert SeparationResource.model_validate_json(before.content).output is None
        await harness.render(identifier)
        complete = await client.get(f"/v1/audio/separations/{identifier}")
        receipt = SeparationResource.model_validate_json(complete.content)
        assert (
            complete.status_code == 200
            and receipt.status == "completed"
            and receipt.output is not None
        )
        assert [stem.name for stem in receipt.output.stems] == [
            "drums",
            "bass",
            "other",
            "vocals",
        ]
        for stem in receipt.output.stems:
            response = await client.get(
                f"/v1/audio/separations/{identifier}/content/{stem.name}"
            )
            assert response.status_code == 200
            assert len(response.content) == stem.size_bytes
            assert hashlib.sha256(response.content).hexdigest() == stem.sha256
            assert response.headers["etag"] == '"' + stem.sha256 + '"'
        listed = SeparationListResponse.model_validate_json(
            (await client.get("/v1/audio/separations")).content
        )
        assert listed.data[0].id == identifier and not listed.has_more
        assert harness.api.sources.ready_path(identifier) is None
        assert (
            sum(isinstance(command, SeparationCommand) for command in harness.commands)
            == 1
        )


async def test_reordered_media_before_data_stays_private_until_manifest(
    tmp_path: Path,
) -> None:
    """OUTPUT_MEDIA may outrun DATA, without exposing partially matched files."""
    async with fixture(tmp_path) as (harness, client):
        identifier = await create(client)
        task = await harness.task(identifier)
        output = result(tmp_path / "native", task)
        held: list[OutputMediaPacket] = []

        async def collect(packet: OutputMediaPacket) -> None:
            held.append(packet)

        async def fail(_task: AudioSeparation, message: str) -> None:
            raise AssertionError(message)

        producer = SeparationOutputProducer(NodeId("node-0"), collect, fail)
        producer.register(task, tmp_path / "native", output)
        harness.producer = producer
        sending = asyncio.create_task(producer.deliver(identifier))
        try:
            with anyio.fail_after(5):
                while sum(packet.kind == "completed" for packet in held) != 4:
                    await anyio.sleep(0.01)
            # Complete, chunk and open deliberately arrive in reverse order.
            for packet in reversed(held):
                await harness.api.receive_output(packet)
            private = SeparationResource.model_validate_json(
                (await client.get(f"/v1/audio/separations/{identifier}")).content
            )
            assert private.output is None and private.status == "in_progress"
            assert (
                await client.get(f"/v1/audio/separations/{identifier}/content/drums")
            ).status_code == 409
            await harness.queues[identifier].send(
                SeparationChunk(
                    model=ModelId(task.task_params.model),
                    finish_reason="stop",
                    output=output,
                )
            )
            await asyncio.wait_for(sending, 5)
            assert (
                SeparationResource.model_validate_json(
                    (await client.get(f"/v1/audio/separations/{identifier}")).content
                ).status
                == "completed"
            )
        finally:
            if not sending.done():
                sending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await sending


@pytest.mark.parametrize(
    "metadata",
    [
        {"model": "adefossez/HTDemucs", "path": "/server-local.wav"},
        {"model": "adefossez/HTDemucs", "options": {"device": "cuda"}},
        {"model": "adefossez/HTDemucs", "lyrics": "no"},
    ],
)
async def test_closed_metadata_cannot_supply_paths_or_native_options(
    tmp_path: Path, metadata: dict[str, object]
) -> None:
    """Unknown API controls are refused before any command or source admission."""
    async with fixture(tmp_path) as (harness, client):
        response = await client.post(
            "/v1/audio/separations",
            data={"request": json.dumps(metadata)},
            files={"file": ("source.wav", original(), "audio/wav")},
        )
        assert response.status_code == 422 and not harness.commands


async def test_original_and_body_limits_fail_before_submission(tmp_path: Path) -> None:
    """A bogus MIME label cannot turn invalid bytes into an admitted WAV."""
    async with fixture(tmp_path) as (harness, client):
        response = await client.post(
            "/v1/audio/separations",
            data={"request": '{"model":"adefossez/HTDemucs"}'},
            files={"file": ("source.wav", b"not a wave", "audio/wav")},
        )
        assert response.status_code == 422 and not harness.commands
        response = await client.post(
            "/v1/audio/separations",
            content=b"bounded",
            headers={
                "Content-Type": "multipart/form-data; boundary=boundary",
                "Content-Length": str(MAX_SEPARATION_SOURCE_BYTES + 32769),
            },
        )
        assert response.status_code == 413 and not harness.commands


async def test_cancel_delete_and_late_result_cannot_revive_job(tmp_path: Path) -> None:
    """Cancellation is terminal, removes private bytes and never resubmits."""
    async with fixture(tmp_path) as (harness, client):
        identifier = await create(client)
        task = await harness.task(identifier)
        cancelled = await client.post(f"/v1/audio/separations/{identifier}/cancel")
        assert (
            cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
        )
        assert any(isinstance(command, TaskCancelled) for command in harness.commands)
        spoof = OutputMediaPacket(
            source_node=NodeId("node-0"),
            target_node=NodeId("owning-api"),
            command_id=identifier,
            model=ModelId(task.task_params.model),
            purpose="separation_drums",
            kind="opened",
            sequence=0,
            total_chunks=1,
            total_bytes=44,
            content_type="audio/wav",
        )
        await harness.api.receive_output(spoof)
        assert (await client.get(f"/v1/audio/separations/{identifier}")).json()[
            "output"
        ] is None
        assert (
            await client.delete(f"/v1/audio/separations/{identifier}")
        ).status_code == 200
        assert (
            await client.get(f"/v1/audio/separations/{identifier}")
        ).status_code == 404
        assert (
            sum(isinstance(command, SeparationCommand) for command in harness.commands)
            == 1
        )


async def test_openapi_covers_every_new_route_and_closed_multipart(
    tmp_path: Path,
) -> None:
    """The actual route registration exports every operation and its response model."""
    async with fixture(tmp_path) as (_, client):
        spec = json_object(
            cast(object, json.loads((await client.get("/openapi.json")).content))
        )
        for path, methods in {
            "/v1/audio/separations": {"get", "post"},
            "/v1/audio/separations/{separation_id}": {"get", "delete"},
            "/v1/audio/separations/{separation_id}/cancel": {"post"},
            "/v1/audio/separations/{separation_id}/content/{stem}": {"get"},
        }.items():
            for method in methods:
                operation = json_object(
                    json_object(json_object(spec["paths"])[path])[method]
                )
                assert (
                    operation["summary"]
                    and operation["description"]
                    and operation["tags"]
                )
        operation = json_object(
            json_object(json_object(spec["paths"])["/v1/audio/separations"])["post"]
        )
        body = json_object(
            json_object(
                json_object(json_object(operation["requestBody"])["content"])[
                    "multipart/form-data"
                ]
            )["schema"]
        )
        assert (
            body["required"] == ["request", "file"]
            and body["additionalProperties"] is False
        )


async def test_completed_content_survives_api_reopen_without_reexecution(
    tmp_path: Path,
) -> None:
    """Restart adopts the complete set and sends no new edit."""
    async with fixture(tmp_path) as (harness, client):
        identifier = await create(client)
        await harness.render(identifier)
        before = (
            await client.get(f"/v1/audio/separations/{identifier}/content/drums")
        ).content
    async with fixture(tmp_path) as (replacement, client):
        metadata = SeparationResource.model_validate_json(
            (await client.get(f"/v1/audio/separations/{identifier}")).content
        )
        assert metadata.status == "completed" and metadata.output is not None
        assert (
            await client.get(f"/v1/audio/separations/{identifier}/content/drums")
        ).content == before
        assert not replacement.commands


async def test_interrupted_job_reopens_failed_without_replay(tmp_path: Path) -> None:
    """A lost API incarnation cannot silently repeat a native edit."""
    async with fixture(tmp_path) as (harness, client):
        identifier = await create(client)
        await harness.task(identifier)
    async with fixture(tmp_path) as (replacement, client):
        metadata = SeparationResource.model_validate_json(
            (await client.get(f"/v1/audio/separations/{identifier}")).content
        )
        assert metadata.status == "failed" and metadata.output is None
        assert (
            await client.get(f"/v1/audio/separations/{identifier}/content/drums")
        ).status_code == 409
        assert not replacement.commands


async def test_deletion_after_headers_cannot_truncate_open_download(
    tmp_path: Path,
) -> None:
    """The actual ASGI response leases its inode before another request unlinks it."""
    async with fixture(tmp_path) as (harness, client):
        identifier = await create(client)
        await harness.render(identifier)
        expected = (
            await client.get(f"/v1/audio/separations/{identifier}/content/drums")
        ).content
        response = await harness.api.content(str(identifier), "drums")
        collected: list[bytes] = []

        async def send(message: Message) -> None:
            if message["type"] == "http.response.start":
                assert (
                    await client.delete(f"/v1/audio/separations/{identifier}")
                ).status_code == 200
            elif message["type"] == "http.response.body":
                body = cast(object, message.get("body", b""))
                assert isinstance(body, bytes)
                collected.append(body)

        async def receive() -> Message:
            return {"type": "http.disconnect"}

        await response(
            {"type": "http", "method": "GET", "asgi": {"spec_version": "2.4"}},
            receive,
            send,
        )
        assert b"".join(collected) == expected
        assert (
            await client.get(f"/v1/audio/separations/{identifier}")
        ).status_code == 404


async def test_expiry_releases_media_budget_and_preserves_receipt(
    tmp_path: Path,
) -> None:
    """Expiration frees retained bytes as well as removing download visibility."""
    async with fixture(tmp_path) as (harness, client):
        identifier = await create(client)
        await harness.render(identifier)
        metadata = SeparationResource.model_validate_json(
            (await client.get(f"/v1/audio/separations/{identifier}")).content
        )
        assert (
            metadata.expires_at is not None and harness.api.media.reserved_bytes() > 0
        )
        await harness.api.expire_content(metadata.expires_at + 1)
        assert harness.api.media.reserved_bytes() == 0
        assert (
            await client.get(f"/v1/audio/separations/{identifier}/content/drums")
        ).status_code == 404
        assert (
            await client.get(f"/v1/audio/separations/{identifier}")
        ).status_code == 200


@pytest.mark.parametrize("cancel", [False, True])
async def test_stalled_upload_timeout_or_disconnect_closes_spools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel: bool,
) -> None:
    """An incomplete upload cannot retain temp bytes or one of four upload slots."""
    spools: list[SpooledTemporaryFile[bytes]] = []

    def spool(*, max_size: int) -> SpooledTemporaryFile[bytes]:
        opened = SpooledTemporaryFile[bytes](max_size=max_size)
        spools.append(opened)
        return opened

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", spool)
    monkeypatch.setattr(separation_module, "_UPLOAD_SECONDS", 30 if cancel else 0.05)
    started = asyncio.Event()

    async def body() -> AsyncIterator[bytes]:
        yield b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="original.wav"\r\nContent-Type: audio/wav\r\n\r\npartial'
        started.set()
        await asyncio.sleep(60)

    async with fixture(tmp_path) as (harness, client):
        uploading = asyncio.create_task(
            client.post(
                "/v1/audio/separations",
                content=body(),
                headers={"Content-Type": "multipart/form-data; boundary=boundary"},
            )
        )
        await asyncio.wait_for(started.wait(), 3)
        assert spools
        if cancel:
            uploading.cancel()
            with pytest.raises(asyncio.CancelledError):
                await uploading
        else:
            assert (await asyncio.wait_for(uploading, 3)).status_code == 408
        assert all(item.closed for item in spools)
        assert harness.api.uploads.borrowed_tokens == 0 and not harness.commands
