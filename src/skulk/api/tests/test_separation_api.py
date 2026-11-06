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
from typing import Literal, cast

import anyio
import httpx
import pytest
from fastapi import FastAPI
from starlette import formparsers
from starlette.types import Message

from skulk.api import separation_api as separation_module
from skulk.api.main import API
from skulk.api.separation_api import SeparationAPI
from skulk.api.types.separation import SeparationListResponse, SeparationResource
from skulk.master.main import separation_generation_events
from skulk.master.tests.test_separation_admission import state
from skulk.routing.output_media import (
    OutputMediaPacket,
    output_media_for_api,
    output_media_for_worker,
)
from skulk.routing.router import OutboundPacket, TopicRouter
from skulk.routing.topics import OUTPUT_MEDIA
from skulk.routing.vision_media import VisionMediaPacket
from skulk.shared.models.model_cards import ModelCard, ModelId
from skulk.shared.types.audio_separation import MAX_SEPARATION_SOURCE_BYTES
from skulk.shared.types.chunks import ErrorChunk, SeparationChunk
from skulk.shared.types.commands import AudioSeparation as SeparationCommand
from skulk.shared.types.commands import Command, TaskCancelled
from skulk.shared.types.common import CommandId, NodeId, SystemId
from skulk.shared.types.events import TaskCreated
from skulk.shared.types.state import State
from skulk.shared.types.tasks import AudioSeparation, TaskStatus
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


@pytest.mark.parametrize("purpose", ["separation_drums", "separation_vocals"])
async def test_task_finalization_waits_for_every_stem_acknowledgement(
    tmp_path: Path, purpose: str
) -> None:
    """Natural task completion cannot delete a producer awaiting a stem receipt."""
    async with fixture(tmp_path) as (harness, client):
        identifier = await create(client)
        acknowledgement_started, release_acknowledgement = anyio.Event(), anyio.Event()

        async def delayed_acknowledgement(packet: OutputMediaPacket) -> None:
            if packet.kind == "accepted" and packet.purpose == purpose:
                acknowledgement_started.set()
                await release_acknowledgement.wait()
            await harness.send_output(packet)

        harness.api.send_output = delayed_acknowledgement
        async with anyio.create_task_group() as group:
            group.start_soon(harness.render, identifier)
            try:
                with anyio.fail_after(3):
                    await acknowledgement_started.wait()
                assert identifier not in harness.finalizations
                assert (await harness.api.job(identifier)).status != "completed"
            finally:
                release_acknowledgement.set()
        with anyio.fail_after(3):
            while identifier not in harness.finalizations:
                await anyio.sleep(0.01)
        assert (await harness.api.job(identifier)).status == "completed"
        assert harness.finalizations.count(identifier) == 1


async def test_cancellation_submission_precedes_stream_finalization(
    tmp_path: Path,
) -> None:
    """A blocked cancellation keeps its task alive until submission is attempted."""
    async with fixture(tmp_path) as (harness, client):
        identifier = await create(client)
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
                model=ModelId("adefossez/HTDemucs"),
                error_message="fixture separation failure",
            )
        )
        try:
            with anyio.fail_after(3):
                await started.wait()
            assert identifier not in harness.finalizations
        finally:
            release.set()
        with anyio.fail_after(3):
            while identifier not in harness.finalizations:
                await anyio.sleep(0.01)
        assert harness.finalizations.count(identifier) == 1
        assert any(isinstance(command, TaskCancelled) for command in harness.commands)
        assert (await harness.api.job(identifier)).status == "failed"


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
    """The first role may outrun DATA; its ACK admits the remaining roles."""
    async with fixture(tmp_path) as (harness, client):
        identifier = await create(client)
        task = await harness.task(identifier)
        output = result(tmp_path / "native", task)
        held: list[OutputMediaPacket] = []
        holding = True

        async def collect(packet: OutputMediaPacket) -> None:
            if holding:
                held.append(packet)
            else:
                await harness.api.receive_output(packet)

        async def fail(_task: AudioSeparation, message: str) -> None:
            raise AssertionError(message)

        producer = SeparationOutputProducer(NodeId("node-0"), collect, fail)
        producer.register(task, tmp_path / "native", output)
        harness.producer = producer
        sending = asyncio.create_task(producer.deliver(identifier))
        try:
            with anyio.fail_after(5):
                while sum(packet.kind == "completed" for packet in held) != 1:
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
            holding = False
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


@pytest.mark.parametrize("metadata_first", [False, True])
async def test_replayed_completion_requires_verified_manifest_and_settles(
    tmp_path: Path, metadata_first: bool
) -> None:
    """Duplicate role terminals cannot bypass DATA verification or strand a job."""
    async with fixture(tmp_path) as (harness, client):
        identifier = await create(client)
        task = await harness.task(identifier)
        output = result(tmp_path / "native", task)
        first_completed = anyio.Event()

        async def duplicate_terminal(packet: OutputMediaPacket) -> None:
            await harness.api.receive_output(packet)
            if packet.kind == "completed":
                await harness.api.receive_output(packet)
                first_completed.set()

        async def fail(_task: AudioSeparation, message: str) -> None:
            raise AssertionError(message)

        producer = SeparationOutputProducer(
            NodeId("node-0"), duplicate_terminal, fail
        )
        producer.register(task, tmp_path / "native", output)
        harness.producer = producer
        manifest = SeparationChunk(
            model=ModelId(task.task_params.model), finish_reason="stop", output=output
        )
        if metadata_first:
            await harness.queues[identifier].send(manifest)
        sending = asyncio.create_task(producer.deliver(identifier))
        try:
            with anyio.fail_after(5):
                await first_completed.wait()
                # Let independently scheduled reverse controls drain. Neither
                # receipt bytes nor a repeated terminal establish manifest truth.
                await anyio.sleep(0)
                if not metadata_first:
                    assert not any(
                        packet.kind == "accepted" for packet in harness.output_frames
                    )
                    private = SeparationResource.model_validate_json(
                        (await client.get(f"/v1/audio/separations/{identifier}")).content
                    )
                    assert private.status == "in_progress" and private.output is None
                    await harness.queues[identifier].send(manifest)
                await sending
                while True:
                    completed = SeparationResource.model_validate_json(
                        (await client.get(f"/v1/audio/separations/{identifier}")).content
                    )
                    if completed.status == "completed":
                        break
                    await anyio.sleep(0.01)
            assert completed.output is not None
            for stem in completed.output.stems:
                downloaded = await client.get(
                    f"/v1/audio/separations/{identifier}/content/{stem.name}"
                )
                assert downloaded.status_code == 200
                assert hashlib.sha256(downloaded.content).hexdigest() == stem.sha256
        finally:
            if not sending.done():
                sending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await sending


@pytest.mark.parametrize("metadata_first", [False, True])
async def test_same_node_four_stem_rendezvous_routes_reverse_acknowledgments(
    tmp_path: Path, metadata_first: bool
) -> None:
    """Real zero-buffer routing must carry all four roles and reverse ACKs."""
    async with fixture(tmp_path) as (harness, client):
        harness.api.node_id = NodeId("node-0")
        network_send, network_receive = channel[OutboundPacket](1)
        router = TopicRouter(OUTPUT_MEDIA, network_send, local_routing_key="node-0")
        api_receive = router.new_receiver(accept=output_media_for_api)
        worker_receive = router.new_receiver(accept=output_media_for_worker)
        harness.api.send_output = router.new_sender().send
        api_frames: list[OutputMediaPacket] = []
        worker_frames: list[OutputMediaPacket] = []
        failures: list[str] = []

        async def api_ingress() -> None:
            with api_receive:
                async for packet in api_receive:
                    api_frames.append(packet)
                    await harness.api.receive_output(packet)

        async def worker_ingress() -> None:
            with worker_receive:
                async for packet in worker_receive:
                    worker_frames.append(packet)
                    assert harness.producer is not None
                    harness.producer.receive(packet)

        async def fail(_task: AudioSeparation, message: str) -> None:
            failures.append(message)

        async with anyio.create_task_group() as group:
            group.start_soon(router.run)
            group.start_soon(api_ingress)
            group.start_soon(worker_ingress)
            try:
                identifier = await create(client)
                task = await harness.task(identifier)
                output = result(tmp_path / "native", task)
                harness.producer = SeparationOutputProducer(
                    NodeId("node-0"), router.new_sender().send, fail, timeout_seconds=5
                )
                harness.producer.register(task, tmp_path / "native", output)
                chunk = SeparationChunk(model=ModelId(task.task_params.model), finish_reason="stop", output=output)
                if metadata_first:
                    await harness.queues[identifier].send(chunk)
                sending = asyncio.create_task(harness.producer.deliver(identifier))
                try:
                    if not metadata_first:
                        with anyio.fail_after(2):
                            while not any(packet.kind == "completed" for packet in api_frames):
                                await anyio.sleep(0.01)
                        before = SeparationResource.model_validate_json((await client.get(f"/v1/audio/separations/{identifier}")).content)
                        assert before.status == "in_progress" and before.output is None
                        await harness.queues[identifier].send(chunk)
                    await asyncio.wait_for(sending, 5)
                    assert not failures
                    complete = SeparationResource.model_validate_json((await client.get(f"/v1/audio/separations/{identifier}")).content)
                    assert complete.status == "completed" and complete.output is not None
                    for stem in complete.output.stems:
                        content = await client.get(f"/v1/audio/separations/{identifier}/content/{stem.name}")
                        assert content.status_code == 200
                        assert hashlib.sha256(content.content).hexdigest() == stem.sha256
                    assert all(packet.kind not in ("accepted", "cancelled") for packet in api_frames)
                    assert all(packet.kind in ("accepted", "cancelled", "transport_failed") for packet in worker_frames)
                    assert {packet.purpose for packet in worker_frames if packet.kind == "accepted"} == {packet.purpose for packet in api_frames if packet.kind == "completed"}
                    with pytest.raises(anyio.WouldBlock):
                        network_receive.receive_nowait()
                finally:
                    if not sending.done():
                        sending.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await sending
            finally:
                await router.shutdown()
                group.cancel_scope.cancel()


@pytest.mark.parametrize("commands", [2, 4])
async def test_concurrent_same_node_delivery_cannot_block_reverse_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, commands: int
) -> None:
    """A second command enters the real router while the API verifies the first."""
    async with fixture(tmp_path) as (harness, client):
        harness.api.node_id = NodeId("node-0")
        network_send, network_receive = channel[OutboundPacket](1)
        router = TopicRouter(OUTPUT_MEDIA, network_send, local_routing_key="node-0")
        api_receive = router.new_receiver(accept=output_media_for_api)
        worker_receive = router.new_receiver(accept=output_media_for_worker)
        harness.api.send_output = router.new_sender().send
        identifiers: list[CommandId] = []
        failures: list[str] = []
        first_completed, second_publishing = anyio.Event(), anyio.Event()
        original_publish = router.publish

        async def publish(packet: OutputMediaPacket, origin: str | None = None) -> None:
            if len(identifiers) > 1 and packet.command_id != identifiers[0] and packet.kind == "opened":
                second_publishing.set()
            await original_publish(packet, origin)

        monkeypatch.setattr(router, "publish", publish)

        async def api_ingress() -> None:
            with api_receive:
                async for packet in api_receive:
                    if packet.command_id == identifiers[0] and packet.kind == "completed" and not first_completed.is_set():
                        first_completed.set()
                        await second_publishing.wait()
                    await harness.api.receive_output(packet)

        async def worker_ingress() -> None:
            with worker_receive:
                async for packet in worker_receive:
                    assert harness.producer is not None
                    harness.producer.receive(packet)

        async def fail(_task: AudioSeparation, message: str) -> None:
            failures.append(message)

        producer = SeparationOutputProducer(NodeId("node-0"), router.new_sender().send, fail, timeout_seconds=5)
        harness.producer = producer
        for _ in range(commands):
            identifier = await create(client)
            identifiers.append(identifier)
            task = await harness.task(identifier)
            output = result(tmp_path / "native", task)
            producer.register(task, tmp_path / "native", output)
            await harness.queues[identifier].send(SeparationChunk(model=ModelId(task.task_params.model), finish_reason="stop", output=output))
            # Native execution may finish before its result delivery. Ordinary
            # placement can reuse that ready instance while earlier bytes drain.
            harness.snapshot = harness.snapshot.model_copy(update={"tasks": dict(harness.snapshot.tasks) | {task.task_id: task.model_copy(update={"task_status": TaskStatus.Complete})}})

        async with anyio.create_task_group() as group:
            group.start_soon(router.run)
            group.start_soon(api_ingress)
            group.start_soon(worker_ingress)
            try:
                with anyio.fail_after(4):
                    group.start_soon(producer.deliver, identifiers[0])
                    await first_completed.wait()
                    for identifier in identifiers[1:]:
                        group.start_soon(producer.deliver, identifier)
                    while any(producer.pending(identifier) for identifier in identifiers):
                        await anyio.sleep(0.01)
                    assert not failures
                    for identifier in identifiers:
                        complete = SeparationResource.model_validate_json((await client.get(f"/v1/audio/separations/{identifier}")).content)
                        assert complete.status == "completed" and complete.output is not None
                        for stem in complete.output.stems:
                            content = await client.get(f"/v1/audio/separations/{identifier}/content/{stem.name}")
                            assert content.status_code == 200 and hashlib.sha256(content.content).hexdigest() == stem.sha256
                    with pytest.raises(anyio.WouldBlock):
                        network_receive.receive_nowait()
            finally:
                group.cancel_scope.cancel()
                await router.shutdown()


async def test_cancelled_jobs_hold_admission_until_reverse_controls_drain(
    tmp_path: Path,
) -> None:
    """Rapid cancellation cannot detach more control owners than four job slots."""
    async with fixture(tmp_path) as (harness, client):
        identifiers: list[CommandId] = []
        for _ in range(4):
            identifier = await create(client)
            identifiers.append(identifier)
            task = await harness.task(identifier)
            harness.snapshot = harness.snapshot.model_copy(update={"tasks": dict(harness.snapshot.tasks) | {task.task_id: task.model_copy(update={"task_status": TaskStatus.Complete})}})
        released = anyio.Event()
        started: set[CommandId] = set()

        async def hold_control(packet: OutputMediaPacket) -> None:
            started.add(packet.command_id)
            await released.wait()

        harness.api.send_output = hold_control
        with anyio.fail_after(2):
            for identifier in identifiers:
                response = await client.post(f"/v1/audio/separations/{identifier}/cancel")
                assert response.status_code == 200
                assert SeparationResource.model_validate_json(response.content).status == "cancelled"
            while len(started) != 4:
                await anyio.sleep(0.01)
        assert not harness.api.active
        assert len(harness.api._output_control_counts) == 4
        response = await client.post("/v1/audio/separations", data={"request": json.dumps({"model": "adefossez/HTDemucs"})}, files={"file": ("original.wav", original(), "audio/wav")})
        assert response.status_code == 503
        released.set()
        with anyio.fail_after(2):
            while harness.api._output_control_counts:
                await anyio.sleep(0.01)
        assert harness.worker is not None
        for identifier in identifiers:
            await harness.worker.release_after_drain(identifier)
        assert await create(client) not in identifiers


async def test_failed_acknowledgement_fails_job_and_releases_control_ownership(
    tmp_path: Path,
) -> None:
    """A failed reverse send cannot expose a completed set or retain its slot."""
    async with fixture(tmp_path) as (harness, client):
        identifier = await create(client)

        async def reject_acknowledgement(packet: OutputMediaPacket) -> None:
            if packet.kind == "accepted":
                raise anyio.BrokenResourceError
            await harness.send_output(packet)

        harness.api.send_output = reject_acknowledgement
        with pytest.raises(AssertionError, match="Separation result delivery failed"):
            await harness.render(identifier)
        with anyio.fail_after(2):
            while harness.api._output_control_counts:
                await anyio.sleep(0.01)
        job = SeparationResource.model_validate_json((await client.get(f"/v1/audio/separations/{identifier}")).content)
        assert job.status == "failed" and job.output is None
        assert not harness.api.active and harness.api.media.reserved_bytes() == 0


async def test_production_control_sender_cannot_complete_after_dropping_final_ack(
    tmp_path: Path,
) -> None:
    """Use the API callback and a real channel broken after three admitted ACKs."""
    async with fixture(tmp_path) as (harness, client):
        control_send, control_receive = channel[OutputMediaPacket](0)
        host = object.__new__(API)
        host._output_media_packet_sender = control_send
        harness.api.send_output = host._send_separation_output_terminal
        identifier = await create(client)
        task = await harness.task(identifier)
        output = result(tmp_path / "native", task)
        completed: set[str] = set()
        failures: list[str] = []

        async def forward(packet: OutputMediaPacket) -> None:
            if packet.kind == "completed":
                completed.add(packet.purpose)
            await harness.api.receive_output(packet)

        async def fail(_task: AudioSeparation, message: str) -> None:
            failures.append(message)

        producer = SeparationOutputProducer(NodeId("node-0"), forward, fail, timeout_seconds=10)
        harness.producer = producer
        producer.register(task, tmp_path / "native", output)
        await harness.queues[identifier].send(SeparationChunk(model=ModelId(task.task_params.model), finish_reason="stop", output=output))

        async def admit_three_acks() -> None:
            accepted: set[str] = set()
            with control_receive:
                async for packet in control_receive:
                    producer.receive(packet)
                    if packet.kind == "accepted":
                        accepted.add(packet.purpose)
                        if len(accepted) == 3:
                            return

        async with anyio.create_task_group() as group:
            group.start_soon(admit_three_acks)
            sending = asyncio.create_task(producer.deliver(identifier))
            try:
                with anyio.fail_after(4):
                    while True:
                        job = SeparationResource.model_validate_json((await client.get(f"/v1/audio/separations/{identifier}")).content)
                        if job.status in ("failed", "completed", "cancelled"):
                            break
                        await anyio.sleep(0.01)
                    assert len(completed) == 4
                    assert job.status == "failed" and job.output is None
                    while harness.api._output_control_counts:
                        await anyio.sleep(0.01)
                    assert not harness.api.active
                    assert harness.api.media.reserved_bytes() == 0
            finally:
                if not sending.done():
                    sending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await sending
                control_send.close()
                group.cancel_scope.cancel()


@pytest.mark.parametrize("transport", ["missing", "broken", "blocked"])
async def test_strict_production_control_sender_propagates_admission_failure(
    transport: Literal["missing", "broken", "blocked"],
) -> None:
    """Real transport absence, closure and caller deadlines remain failures."""
    send, receive = channel[OutputMediaPacket](0)
    host = object.__new__(API)
    host._output_media_packet_sender = None if transport == "missing" else send
    packet = OutputMediaPacket(source_node=NodeId("api"), target_node=NodeId("worker"), command_id=CommandId("strict-control"), model=ModelId("model"), purpose="separation_drums", kind="accepted", sequence=2)
    try:
        if transport == "blocked":
            with pytest.raises(TimeoutError), anyio.fail_after(0.05):
                await host._send_separation_output_terminal(packet)
        else:
            if transport == "broken":
                receive.close()
            with pytest.raises((anyio.ClosedResourceError, anyio.BrokenResourceError)):
                await host._send_separation_output_terminal(packet)
    finally:
        send.close()
        receive.close()


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
