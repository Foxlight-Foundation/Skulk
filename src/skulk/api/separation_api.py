"""Authenticated API-local separation jobs over the existing Fabric media planes.

Only metadata enters commands and durable job records. Uploads and results use
bounded private custody; a complete render and four verified stems must agree
before any content is exposed. A restart fails interrupted work without replay.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import final

import anyio
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import ValidationError
from starlette.datastructures import FormData, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.types import Receive, Scope, Send

from skulk.api.separation_jobs import (
    MAX_RETAINED_SEPARATION_JOBS,
    SeparationJob,
    SeparationJobCapacityError,
    SeparationJobRegistry,
    SeparationJobStorageError,
)
from skulk.api.separation_media import SeparationMediaStore, StoredSeparationStem
from skulk.api.types.separation import (
    SeparationDeletedResponse,
    SeparationListResponse,
    SeparationResource,
    separation_resource,
)
from skulk.routing.output_media import OutputMediaPacket
from skulk.routing.vision_media import VisionMediaPacket
from skulk.shared.constants import SKULK_MAX_CHUNK_SIZE
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.audio_separation import (
    MAX_SEPARATION_SOURCE_BYTES,
    SEPARATION_STEMS,
    AudioSeparationRequest,
    AudioSeparationTaskParams,
    SeparationStemName,
    source_from_wav,
)
from skulk.shared.types.chunks import ErrorChunk, SeparationChunk
from skulk.shared.types.commands import (
    AudioSeparation as SeparationCommand,
)
from skulk.shared.types.commands import (
    Command,
    TaskCancelled,
)
from skulk.shared.types.common import CommandId, NodeId, SystemId
from skulk.shared.types.state import State
from skulk.shared.types.tasks import AudioSeparation
from skulk.shared.types.worker.instances import InstanceId
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.channels import Receiver
from skulk.utils.task_group import TaskGroup
from skulk.worker.separation_custody import SeparationSourceCustody
from skulk.worker.separation_input import separation_file_effect
from skulk.worker.separation_output import separation_purpose
from skulk.worker.separation_sources import SeparationSourceStore

_UPLOAD_SECONDS = 180.0


@final
class _ClosingMultipartParser(MultiPartParser):
    """Close spooled files on disconnect/cancellation as well as parser errors.

    Starlette closes its error-file collection for MultiPartException only.
    An interrupted request must not retain partially spooled originals after it
    releases upload admission. Keep this cleanup inside the parser subclass.
    """

    async def parse(self) -> FormData:
        """Return parsed form data, or close every file this parser created."""
        try:
            return await super().parse()
        except BaseException:
            for uploaded in self._files_to_close_on_error:
                uploaded.close()
            raise


def _stem_name(purpose: str) -> SeparationStemName | None:
    match purpose:
        case "separation_drums":
            return "drums"
        case "separation_bass":
            return "bass"
        case "separation_other":
            return "other"
        case "separation_vocals":
            return "vocals"
        case _:
            return None


@dataclass
class _Active:
    upload: AudioSeparation
    deadline: float
    session: SystemId
    task: AudioSeparation | None = None
    producer: NodeId | None = None
    sending: bool = False
    accepted: bool = False
    source_released: bool = False
    opened: dict[SeparationStemName, tuple[int, int]] = field(default_factory=dict)
    early: dict[SeparationStemName, list[OutputMediaPacket]] = field(
        default_factory=dict
    )
    completed: dict[SeparationStemName, OutputMediaPacket] = field(default_factory=dict)
    receipts: dict[SeparationStemName, StoredSeparationStem] = field(
        default_factory=dict
    )


class _ContentResponse(StreamingResponse):
    """Open under the owner's lock before headers, then stream an owned descriptor.

    Unlinking a job cannot interrupt an already opened download. No descriptor is
    allocated merely by returning a response; cancellation always closes it.
    """

    def __init__(
        self, service: SeparationAPI, identifier: CommandId, stem: SeparationStemName
    ) -> None:
        self.service, self.identifier = service, identifier
        self.stem: SeparationStemName = stem
        super().__init__(iter(()), media_type="audio/wav")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Lease one bounded download and close its file even on disconnect."""
        try:
            self.service.downloads.acquire_nowait()
        except anyio.WouldBlock:
            raise HTTPException(
                503, "too many separation downloads are active"
            ) from None
        descriptor: int | None = None
        try:
            async with self.service.lock:
                await self.service.ensure()
                job = await self.service.job(self.identifier)
                if job.status != "completed":
                    raise HTTPException(409, f"separation job is {job.status}")
                assert self.service.media is not None

                def open_content() -> tuple[int, StoredSeparationStem]:
                    stored = self.service.media.content(job, self.stem)
                    if stored is None:
                        raise HTTPException(
                            404, "separation content is absent or expired"
                        )
                    return os.open(
                        stored.file_path, os.O_RDONLY | os.O_NOFOLLOW
                    ), stored

                # Store the descriptor in the owner inside the joined effect so
                # cancellation at return cannot orphan a newly opened handle.
                opened: list[tuple[int, StoredSeparationStem]] = []
                try:
                    await separation_file_effect(lambda: opened.append(open_content()))
                finally:
                    if opened:
                        descriptor, stored = opened[0]
                assert descriptor is not None
                stored = opened[0][1]
            owned = descriptor

            async def frames() -> AsyncIterator[bytes]:
                remaining = stored.facts.size_bytes
                while remaining:

                    def read_frame(remaining: int = remaining) -> bytes:
                        return os.read(owned, min(remaining, SKULK_MAX_CHUNK_SIZE))

                    data = await separation_file_effect(read_frame)
                    if not data:
                        raise OSError("retained separation content became incomplete")
                    remaining -= len(data)
                    yield data

            self.body_iterator = frames()
            self.init_headers(
                {
                    "content-length": str(stored.facts.size_bytes),
                    "content-type": "audio/wav",
                    "etag": '"' + stored.facts.sha256 + '"',
                    "content-disposition": f'attachment; filename="{job.id}-{self.stem}.wav"',
                }
            )
            await super().__call__(scope, receive, send)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            self.service.downloads.release()


@final
class SeparationAPI:
    """One lazy private owner shared by REST, addressed input, DATA and results."""

    def __init__(
        self,
        node_id: NodeId,
        root: Path,
        state: Callable[[], State],
        tasks: TaskGroup,
        load_card: Callable[[ModelId], Awaitable[ModelCard]],
        send_command: Callable[[Command, SystemId], Awaitable[None]],
        session: Callable[[], SystemId],
        send_source: Callable[[VisionMediaPacket], Awaitable[None]],
        send_output: Callable[[OutputMediaPacket], Awaitable[None]],
        open_queue: Callable[[CommandId], Receiver[SeparationChunk | ErrorChunk]],
        finalize: Callable[[CommandId, SystemId], Awaitable[None]],
        close_queue: Callable[[CommandId], None],
    ) -> None:
        """Bind owners/effects without opening files or advertising a backend."""
        self.node_id, self.root, self.state, self.tasks = node_id, root, state, tasks
        self.load_card, self.send_command = load_card, send_command
        self.session = session
        self.send_source, self.send_output = send_source, send_output
        self.open_queue, self.finalize = open_queue, finalize
        self.close_queue = close_queue
        self.lock = anyio.Lock()
        self.uploads, self.downloads = anyio.CapacityLimiter(4), anyio.Semaphore(8)
        self._jobs: SeparationJobRegistry | None = None
        self._media: SeparationMediaStore | None = None
        self._sources: SeparationSourceStore | None = None
        self.active: dict[CommandId, _Active] = {}
        self._expired_content: set[CommandId] = set()

    @property
    def jobs(self) -> SeparationJobRegistry:
        """Return the initialized, lifetime-owned metadata registry."""
        assert self._jobs is not None
        return self._jobs

    @property
    def media(self) -> SeparationMediaStore:
        """Return initialized private stem custody under this service's fence."""
        assert self._media is not None
        return self._media

    @property
    def sources(self) -> SeparationSourceStore:
        """Return initialized bounded original custody, never a Studio asset path."""
        assert self._sources is not None
        return self._sources

    def register(self, app: FastAPI) -> None:
        """Register all six authenticated-by-host REST routes and OpenAPI contracts."""
        tags: list[str | Enum] = ["Audio Separation"]
        app.post(
            "/v1/audio/separations",
            response_model=SeparationResource,
            status_code=202,
            tags=tags,
            summary="Separate an uploaded WAV into four stems",
            description="Accept multipart request JSON {model} and one file WAV. Requires a mounted separator. One to 120 seconds; 64 MiB original. Returns an API-local asynchronous job; no paths or arbitrary engine options.",
            openapi_extra={
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "required": ["request", "file"],
                                "additionalProperties": False,
                                "properties": {
                                    "request": {
                                        "type": "string",
                                        "description": 'Closed JSON metadata: {"model":"mounted-model"}.',
                                    },
                                    "file": {
                                        "type": "string",
                                        "format": "binary",
                                        "description": "Original WAV, at most 64 MiB.",
                                    },
                                },
                            }
                        }
                    },
                }
            },
        )(self.create)
        app.get(
            "/v1/audio/separations",
            response_model=SeparationListResponse,
            tags=tags,
            summary="List separation jobs",
            description="List node-local jobs newest first, with a retained last-id cursor. Four active and 64 retained jobs maximum.",
        )(self.list_jobs)
        app.get(
            "/v1/audio/separations/{separation_id}",
            response_model=SeparationResource,
            tags=tags,
            summary="Retrieve a separation job",
            description="Read committed status and source facts. Output appears only after successful rendering and all four independently verified stems.",
        )(self.retrieve)
        app.get(
            "/v1/audio/separations/{separation_id}/content/{stem}",
            response_class=StreamingResponse,
            responses={
                200: {
                    "description": "Complete verified PCM24 stereo WAV.",
                    "content": {
                        "audio/wav": {"schema": {"type": "string", "format": "binary"}}
                    },
                    "headers": {
                        "ETag": {
                            "schema": {"type": "string"},
                            "description": "Quoted SHA-256 of the complete WAV.",
                        }
                    },
                }
            },
            tags=tags,
            summary="Download a completed stem WAV",
            description="Download drums, bass, other or vocals from the accepting node. PCM24 stereo at 44.1 kHz, 64 MiB per stem. Content is retained for 24 hours; incomplete sets return 409 and absent/expired content 404.",
        )(self.content)
        app.post(
            "/v1/audio/separations/{separation_id}/cancel",
            response_model=SeparationResource,
            tags=tags,
            summary="Cancel a separation job",
            description="Cancel inference and addressed media delivery, retain terminal metadata and remove partial content. Terminal jobs are unchanged.",
        )(self.cancel)
        app.delete(
            "/v1/audio/separations/{separation_id}",
            response_model=SeparationDeletedResponse,
            tags=tags,
            summary="Delete a separation job",
            description="Cancel active work, remove owned content and forget node-local metadata. A missing job returns 404.",
        )(self.delete)

    def _initialize(self) -> None:
        jobs: SeparationJobRegistry | None = None
        media: SeparationMediaStore | None = None
        sources: SeparationSourceStore | None = None
        try:
            jobs = SeparationJobRegistry(self.root / "index" / "jobs.json")
            media = SeparationMediaStore(
                self.root / "media", max_total_bytes=2 * 1024**3
            )
            custody = SeparationSourceCustody(
                self.root / "inputs",
                self.root / "upload-readers",
                lambda _task: hashlib.sha256(b"api-upload-reader").hexdigest(),
            )
            sources = SeparationSourceStore(self.root / "inputs", custody=custody)
            adopted: set[CommandId] = set()
            for job in jobs.list(limit=MAX_RETAINED_SEPARATION_JOBS):
                if job.status == "completed":
                    if media.adopt_complete(job):
                        adopted.add(job.id)
                    else:
                        jobs.invalidate(
                            job.id,
                            "Separation content did not survive restart or has expired.",
                        )
            media.purge_unknown(frozenset(adopted))
            self._jobs, self._media, self._sources = jobs, media, sources
        except BaseException:
            if sources is not None:
                sources.close()
            if media is not None:
                media.close()
            if jobs is not None:
                jobs.close()
            raise

    async def ensure(self) -> None:
        """Initialize under the actor lock; cancellation still retains any file owners."""
        if self._jobs is None:
            await separation_file_effect(self._initialize)

    async def job(self, identifier: CommandId) -> SeparationJob:
        """Read one committed node-local job, or return 404."""
        assert self.jobs is not None
        job = await separation_file_effect(lambda: self.jobs.get(identifier))
        if job is None:
            raise HTTPException(404, "separation job not found on this API node")
        return job

    async def _upload(self, request: Request) -> tuple[AudioSeparationRequest, bytes]:
        maximum = MAX_SEPARATION_SOURCE_BYTES + 32 * 1024
        length = request.headers.get("content-length")
        if length is not None:
            try:
                declared = int(length)
            except ValueError:
                raise HTTPException(400, "invalid content-length") from None
            if not 0 <= declared <= maximum:
                raise HTTPException(
                    413,
                    "separation upload exceeds 64 MiB plus bounded multipart metadata",
                )
        if (
            not request.headers.get("content-type", "")
            .lower()
            .startswith("multipart/form-data;")
        ):
            raise HTTPException(415, "use multipart/form-data with request and file")
        overflow, timed_out = False, False

        async def bounded() -> AsyncGenerator[bytes, None]:
            nonlocal overflow, timed_out
            total = 0
            stream = request.stream()
            deadline = time.monotonic() + _UPLOAD_SECONDS
            while True:
                try:
                    with anyio.fail_after(max(0.0, deadline - time.monotonic())):
                        data = await anext(stream)
                except StopAsyncIteration:
                    return
                except TimeoutError:
                    timed_out = True
                    raise MultiPartException(
                        "separation upload deadline expired"
                    ) from None
                total += len(data)
                if total > maximum:
                    overflow = True
                    raise MultiPartException("separation upload exceeds its byte limit")
                yield data

        try:
            form = await _ClosingMultipartParser(
                request.headers,
                bounded(),
                max_files=1,
                max_fields=1,
                max_part_size=4096,
            ).parse()
        except MultiPartException:
            raise HTTPException(
                413 if overflow else 408 if timed_out else 400,
                "invalid or over-limit separation multipart upload",
            ) from None
        try:
            if (
                sorted(form.keys()) != ["file", "request"]
                or len(form.multi_items()) != 2
            ):
                raise HTTPException(400, "require exactly request JSON and one file")
            metadata, upload = form.get("request"), form.get("file")
            if not isinstance(metadata, str) or not isinstance(upload, UploadFile):
                raise HTTPException(400, "require request JSON and one file")
            try:
                params = AudioSeparationRequest.model_validate_json(metadata)
            except ValidationError:
                raise HTTPException(
                    422, "invalid separation request metadata"
                ) from None
            data = await upload.read(MAX_SEPARATION_SOURCE_BYTES + 1)
            if len(data) > MAX_SEPARATION_SOURCE_BYTES:
                raise HTTPException(413, "original WAV exceeds 64 MiB")
            return params, data
        finally:
            await form.close()

    async def create(self, request: Request) -> SeparationResource:
        """Bound/verify an original, persist intent, then submit one command once."""
        try:
            self.uploads.acquire_nowait()
        except anyio.WouldBlock:
            raise HTTPException(503, "four uploads are already active") from None
        identifier: CommandId | None = None
        try:
            params, data = await self._upload(request)
            card = await self.load_card(ModelId(params.model))
            if ModelTask.AudioSeparation not in card.tasks or card.separation is None:
                raise HTTPException(400, "select a mounted source-separation model")
            try:
                source = await separation_file_effect(lambda: source_from_wav(data))
            except ValueError:
                raise HTTPException(
                    422,
                    "source must be a bounded finite one-to-120-second mono/stereo WAV",
                ) from None
            identifier = CommandId()
            chunks = (len(data) + SKULK_MAX_CHUNK_SIZE - 1) // SKULK_MAX_CHUNK_SIZE
            task_params = AudioSeparationTaskParams(
                model=params.model, source=source, total_input_chunks=chunks
            )
            upload = AudioSeparation(
                command_id=identifier,
                owner_node=self.node_id,
                instance_id=InstanceId("upload-" + str(identifier)),
                task_params=task_params,
            )
            async with self.lock:
                await self.ensure()
                assert self.jobs is not None and self.sources is not None
                job = SeparationJob(
                    id=identifier,
                    model=ModelId(params.model),
                    source=source,
                    created_at=int(time.time()),
                )
                evicted = await separation_file_effect(lambda: self.jobs.create(job))
                assert self.media is not None
                for old in evicted:
                    await separation_file_effect(lambda old=old: self.media.delete(old))
                self.active[identifier] = _Active(
                    upload, time.monotonic() + 45, self.session()
                )

                def retain() -> None:
                    assert self.sources is not None
                    self.sources.admit(upload)
                    for number in range(chunks):
                        self.sources.append(
                            upload.command_id,
                            number + 1,
                            data[
                                number * SKULK_MAX_CHUNK_SIZE : (number + 1)
                                * SKULK_MAX_CHUNK_SIZE
                            ],
                        )
                    if self.sources.complete(upload.command_id) is None:
                        raise ValueError("uploaded original did not verify")

                await separation_file_effect(retain)
                receiver = self.open_queue(identifier)
                self.tasks.start_soon(
                    self._drain, identifier, receiver, self.active[identifier].session
                )
            # Metadata only. A lost submission reply is terminal/uncertain;
            # neither this API nor Studio automatically resubmits the edit.
            with anyio.fail_after(10):
                await self.send_command(
                    SeparationCommand(
                        command_id=identifier,
                        owner_node=self.node_id,
                        task_params=task_params,
                    ),
                    self.active[identifier].session,
                )
            return separation_resource(job)
        except (
            SeparationJobCapacityError,
            OSError,
            SeparationJobStorageError,
        ) as error:
            logger.opt(exception=error).warning(
                "Separation job admission/storage failed"
            )
            if identifier is not None:
                await self.fail(
                    identifier,
                    "Separation storage admission failed; inspect API diagnostics.",
                )
            raise HTTPException(
                503, "separation capacity or storage is unavailable"
            ) from None
        except BaseException:
            if identifier is not None:
                with anyio.CancelScope(shield=True):
                    await self.fail(
                        identifier,
                        "Separation submission was interrupted; inspect the job before retrying.",
                        cancel=False,
                    )
            raise
        finally:
            self.uploads.release()

    async def observe_task(self, task: AudioSeparation) -> None:
        """Bind only the master's actual complete single-host task to a local job."""
        entry = self.active.get(task.command_id)
        if (
            entry is None
            or task.owner_node != self.node_id
            or task.task_params != entry.upload.task_params
        ):
            return
        instance = self.state().instances.get(task.instance_id)
        if instance is None or len(instance.shard_assignments.node_to_runner) != 1:
            return
        producer, runner = next(iter(instance.shard_assignments.node_to_runner.items()))
        shard = instance.shard_assignments.runner_to_shard.get(runner)
        if (
            not isinstance(shard, PipelineShardMetadata)
            or shard.device_rank != 0
            or shard.world_size != 1
            or shard.start_layer != 0
            or shard.end_layer != shard.model_card.n_layers
            or shard.resolved_backend
            not in {"demucs-cpu", "demucs-metal", "demucs-cuda", "demucs-rocm"}
        ):
            return
        async with self.lock:
            assert self.jobs is not None
            if entry.task is not None and entry.task != task:
                return
            await separation_file_effect(
                lambda: self.jobs.bind_worker(task.command_id, producer)
            )
            entry.task, entry.producer = task, producer
            entry.deadline = time.monotonic() + 40 * 60

    async def receive_source(self, packet: VisionMediaPacket) -> None:
        """Answer only a placed worker's directed source request; no packet admits work."""
        entry = self.active.get(packet.command_id)
        if (
            entry is None
            or packet.target_node != self.node_id
            or packet.payload != "separation_source"
            or packet.model != entry.upload.task_params.model
        ):
            return
        if entry.producer is None:
            for task in self.state().tasks.values():
                if (
                    isinstance(task, AudioSeparation)
                    and task.command_id == packet.command_id
                ):
                    await self.observe_task(task)
        if packet.source_node != entry.producer or entry.task is None:
            return
        if packet.kind == "requested":
            if (
                not entry.sending
                and not entry.accepted
                and packet.total_chunks == entry.upload.task_params.total_input_chunks
                and packet.image_count == 1
            ):
                entry.sending = True
                self.tasks.start_soon(self._send_original, packet.command_id, entry)
        elif packet.kind == "accepted":
            if packet.sequence != entry.upload.task_params.total_input_chunks + 1:
                return
            entry.accepted = True
            await self._release_original(packet.command_id, entry)
        elif packet.kind == "transport_failed":
            await self.fail(
                packet.command_id, "Separation original transfer failed.", cancel=False
            )

    async def _send_original(self, identifier: CommandId, entry: _Active) -> None:
        assert self.sources is not None and entry.producer is not None
        try:
            path = self.sources.ready_path(identifier)
            if path is None:
                raise ValueError("original source is no longer retained")
            source = entry.upload.task_params.source
            chunks = entry.upload.task_params.total_input_chunks
            with anyio.fail_after(180):
                with anyio.fail_after(5):
                    await self.send_source(
                        VisionMediaPacket(
                            source_node=self.node_id,
                            target_node=entry.producer,
                            command_id=identifier,
                            model=ModelId(entry.upload.task_params.model),
                            payload="separation_source",
                            kind="opened",
                            sequence=0,
                            image_count=1,
                            total_chunks=chunks,
                        )
                    )
                digest = hashlib.sha256()
                for sequence in range(1, chunks + 1):
                    if self.active.get(identifier) is not entry or entry.accepted:
                        return
                    offset = (sequence - 1) * SKULK_MAX_CHUNK_SIZE

                    def read_frame(offset: int = offset) -> bytes:
                        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                        try:
                            return os.pread(
                                descriptor,
                                min(SKULK_MAX_CHUNK_SIZE, source.size_bytes - offset),
                                offset,
                            )
                        finally:
                            os.close(descriptor)

                    data = await separation_file_effect(read_frame)
                    if not data:
                        raise ValueError("retained original is incomplete")
                    digest.update(data)
                    with anyio.fail_after(5):
                        await self.send_source(
                            VisionMediaPacket(
                                source_node=self.node_id,
                                target_node=entry.producer,
                                command_id=identifier,
                                model=ModelId(entry.upload.task_params.model),
                                payload="separation_source",
                                kind="chunk",
                                sequence=sequence,
                                total_chunks=chunks,
                                image_index=0,
                                data=data,
                            )
                        )
                if digest.hexdigest() != source.sha256:
                    raise ValueError("retained original changed")
                with anyio.fail_after(5):
                    await self.send_source(
                        VisionMediaPacket(
                            source_node=self.node_id,
                            target_node=entry.producer,
                            command_id=identifier,
                            model=ModelId(entry.upload.task_params.model),
                            payload="separation_source",
                            kind="completed",
                            sequence=chunks + 1,
                            total_chunks=chunks,
                            image_count=1,
                            sha256=source.sha256,
                        )
                    )
        except Exception as error:  # noqa: BLE001 - settle this owned source sender
            logger.opt(exception=error).warning("Separation original delivery failed")
            await self.fail(
                identifier, "Separation original delivery failed.", cancel=False
            )
        finally:
            entry.sending = False
            with anyio.CancelScope(shield=True):
                await self._release_original(identifier, entry)

    async def _release_original(self, identifier: CommandId, entry: _Active) -> None:
        async with self.lock:
            if (
                not entry.source_released
                and not entry.sending
                and (entry.accepted or self.active.get(identifier) is not entry)
            ):
                assert self.sources is not None
                await separation_file_effect(lambda: self.sources.delete(identifier))
                entry.source_released = True

    async def _drain(
        self,
        identifier: CommandId,
        receiver: Receiver[SeparationChunk | ErrorChunk],
        session: SystemId,
    ) -> None:
        try:
            with receiver:
                async for chunk in receiver:
                    if (
                        isinstance(chunk, ErrorChunk)
                        or chunk.finish_reason == "error"
                        or chunk.output is None
                    ):
                        await self.fail(
                            identifier,
                            "Native separation ended without a complete result.",
                            cancel=False,
                        )
                        return
                    async with self.lock:
                        entry = self.active.get(identifier)
                        if (
                            entry is None
                            or entry.producer is None
                            or self._jobs is None
                            or chunk.model != entry.upload.task_params.model
                        ):
                            return
                        producer, output = entry.producer, chunk.output
                        await separation_file_effect(
                            lambda producer=producer,
                            output=output: self.jobs.record_manifest(
                                identifier, producer, output
                            )
                        )
                        await separation_file_effect(
                            lambda producer=producer: self.jobs.finish_render(
                                identifier, producer
                            )
                        )
                        entry.deadline = time.monotonic() + 300
                        await self._record_receipts(identifier, entry)
                    return
            await self.fail(
                identifier,
                "Separation stream ended without a complete result.",
                cancel=False,
            )
        except (OSError, ValueError, SeparationJobStorageError) as error:
            logger.opt(exception=error).warning(
                "Separation result metadata failed verification"
            )
            await self.fail(
                identifier,
                "Separation result metadata failed verification.",
                cancel=False,
            )
        finally:
            await self.finalize(identifier, session)

    async def _record_receipts(self, identifier: CommandId, entry: _Active) -> None:
        assert self.jobs is not None
        job = await self.job(identifier)
        if job.candidate_output is None or entry.producer is None:
            return
        producer = entry.producer
        for name, receipt in tuple(entry.receipts.items()):
            await separation_file_effect(
                lambda receipt=receipt: self.jobs.record_verified_stem(
                    identifier,
                    producer,
                    receipt.facts,
                    expires_at=int(receipt.expires_at),
                )
            )
            completion = entry.completed.get(name)
            if completion is not None:
                with anyio.fail_after(5):
                    await self.send_output(completion.accepted())
        settled = await separation_file_effect(lambda: self.jobs.settle(identifier))
        if settled is not None and settled.status == "completed":
            self.active.pop(identifier, None)
            if not entry.sending and not entry.source_released:
                await separation_file_effect(lambda: self.sources.delete(identifier))
                entry.source_released = True

    async def receive_output(self, packet: OutputMediaPacket) -> None:
        """Verify bounded, reordered WAV delivery from the authoritative worker."""
        name = _stem_name(packet.purpose)
        async with self.lock:
            entry = self.active.get(packet.command_id)
            if (
                entry is None
                or name is None
                or self._media is None
                or packet.target_node != self.node_id
                or packet.source_node != entry.producer
                or packet.model != entry.upload.task_params.model
            ):
                return
            try:
                await self._output_packet(entry, name, packet)
            except (
                OSError,
                ValueError,
                SeparationJobStorageError,
                TimeoutError,
            ) as error:
                logger.opt(exception=error).warning(
                    "Separation result bytes failed verification"
                )
                await self._fail_locked(
                    packet.command_id,
                    "Separation result delivery failed verification.",
                    cancel=False,
                )

    async def _output_packet(
        self, entry: _Active, name: SeparationStemName, packet: OutputMediaPacket
    ) -> None:
        assert self.media is not None
        identifier = packet.command_id
        if packet.kind == "transport_failed":
            await self._fail_locked(
                identifier, "Separation output transport failed.", cancel=False
            )
            return
        if packet.kind not in ("opened", "chunk", "completed"):
            return
        if name in entry.receipts:
            if packet.kind == "completed" and packet == entry.completed.get(name):
                await self._record_receipts(identifier, entry)
            return
        if packet.kind != "opened" and name not in entry.opened:
            early = entry.early.setdefault(name, [])
            retained = sum(
                len(held.data)
                for active in self.active.values()
                for values in active.early.values()
                for held in values
            )
            if len(early) >= 8 or retained + len(packet.data) > 8 * 1024**2:
                raise ValueError("separation result exceeded bounded reordering")
            early.append(packet)
            return
        if packet.kind == "opened":
            if (
                packet.total_chunks is None
                or packet.total_bytes is None
                or packet.content_type != "audio/wav"
            ):
                raise ValueError("invalid separation result declaration")
            size, chunks = packet.total_bytes, packet.total_chunks
            await separation_file_effect(
                lambda: self.media.open_assembly(
                    identifier, name, total_bytes=size, total_chunks=chunks
                )
            )
            entry.opened[name] = (size, chunks)
            for held in entry.early.pop(name, []):
                await self._output_packet(entry, name, held)
            return
        size, chunks = entry.opened[name]
        if (
            packet.total_chunks != chunks
            or packet.total_bytes is not None
            and packet.total_bytes != size
        ):
            raise ValueError("separation result changed its transfer identity")
        if packet.kind == "chunk":
            await separation_file_effect(
                lambda: self.media.append(
                    identifier, name, packet.sequence, packet.data
                )
            )
        else:
            previous = entry.completed.get(name)
            if previous is not None and previous != packet:
                raise ValueError("separation result changed its terminal")
            entry.completed[name] = packet
        completion = entry.completed.get(name)
        if completion is not None and await separation_file_effect(
            lambda: self.media.assembly_complete(identifier, name)
        ):
            assert completion.sha256 is not None and completion.total_chunks is not None
            digest, count = completion.sha256, completion.total_chunks
            receipt = await separation_file_effect(
                lambda: self.media.commit(
                    identifier, name, sha256=digest, total_chunks=count
                )
            )
            entry.receipts[name] = receipt
            await self._record_receipts(identifier, entry)

    async def _fail_locked(
        self, identifier: CommandId, message: str, *, cancel: bool = False
    ) -> None:
        if self._jobs is None:
            return
        job = await separation_file_effect(lambda: self.jobs.get(identifier))
        if job is None or job.is_terminal:
            return
        await separation_file_effect(
            lambda: self.jobs.fail(identifier, message, cancelled=cancel)
        )
        entry = self.active.pop(identifier, None)
        assert self.media is not None
        await separation_file_effect(lambda: self.media.delete(identifier))
        if entry is not None:
            if not entry.sending:
                assert self.sources is not None
                await separation_file_effect(lambda: self.sources.delete(identifier))
                entry.source_released = True
            with contextlib.suppress(
                anyio.BrokenResourceError,
                anyio.ClosedResourceError,
                TimeoutError,
                HTTPException,
            ):
                with anyio.fail_after(5):
                    await self.send_command(
                        TaskCancelled(cancelled_command_id=identifier),
                        entry.session,
                    )
            self.close_queue(identifier)
            if entry.producer is not None:
                for name in SEPARATION_STEMS:
                    with contextlib.suppress(
                        anyio.BrokenResourceError,
                        anyio.ClosedResourceError,
                        TimeoutError,
                    ):
                        with anyio.fail_after(5):
                            await self.send_output(
                                OutputMediaPacket(
                                    source_node=self.node_id,
                                    target_node=entry.producer,
                                    command_id=identifier,
                                    model=job.model,
                                    purpose=separation_purpose(name),
                                    kind="cancelled",
                                    sequence=0,
                                )
                            )

    async def fail(
        self, identifier: CommandId, message: str, *, cancel: bool = False
    ) -> None:
        """Settle a known live job; later frames cannot revive its content."""
        async with self.lock:
            await self._fail_locked(identifier, message, cancel=cancel)

    async def retrieve(self, separation_id: str) -> SeparationResource:
        """Read committed metadata from the accepting node."""
        async with self.lock:
            await self.ensure()
            return separation_resource(await self.job(CommandId(separation_id)))

    async def list_jobs(
        self, limit: int = Query(20, ge=1, le=63), after: str | None = Query(None)
    ) -> SeparationListResponse:
        """List a stable newest-first page without exposing partial manifests."""
        async with self.lock:
            await self.ensure()
            assert self.jobs is not None
            try:
                jobs = await separation_file_effect(
                    lambda: self.jobs.list(
                        limit=limit + 1,
                        after=CommandId(after) if after is not None else None,
                    )
                )
            except ValueError:
                raise HTTPException(400, "separation cursor is not retained") from None
            page = tuple(separation_resource(job) for job in jobs[:limit])
            return SeparationListResponse(
                data=page,
                first_id=page[0].id if page else None,
                last_id=page[-1].id if page else None,
                has_more=len(jobs) > limit,
            )

    async def content(
        self, separation_id: str, stem: SeparationStemName
    ) -> StreamingResponse:
        """Return a descriptor-leased completed stem; deletion cannot truncate it."""
        return _ContentResponse(self, CommandId(separation_id), stem)

    async def cancel(self, separation_id: str) -> SeparationResource:
        """Cancel live work while preserving committed terminal metadata."""
        async with self.lock:
            await self.ensure()
            job = await self.job(CommandId(separation_id))
            await self._fail_locked(
                job.id, "The separation job was cancelled.", cancel=True
            )
            return separation_resource(await self.job(job.id))

    async def delete(self, separation_id: str) -> SeparationDeletedResponse:
        """Cancel, remove owned media and finally forget the local job."""
        async with self.lock:
            await self.ensure()
            job = await self.job(CommandId(separation_id))
            await self._fail_locked(
                job.id, "The separation job was deleted.", cancel=True
            )
            assert self.jobs is not None and self.media is not None
            await separation_file_effect(lambda: self.media.delete(job.id))
            await separation_file_effect(lambda: self.jobs.delete(job.id))
            return SeparationDeletedResponse(id=separation_id)

    async def sweep(self) -> None:
        """Bound admission/render/transfer waits without downloading or replaying work."""
        while True:
            await anyio.sleep(1)
            for identifier, entry in tuple(self.active.items()):
                if time.monotonic() >= entry.deadline:
                    await self.fail(
                        identifier,
                        "Separation placement, render or media deadline expired.",
                        cancel=False,
                    )
            await self.expire_content(int(time.time()))

    async def expire_content(self, now: int) -> None:
        """Release expired media capacity while retaining bounded job receipts."""
        async with self.lock:
            if self._jobs is None:
                return
            jobs = await separation_file_effect(
                lambda: self.jobs.list(limit=MAX_RETAINED_SEPARATION_JOBS)
            )
            self._expired_content.intersection_update(job.id for job in jobs)
            for job in jobs:
                if (
                    job.status == "completed"
                    and job.expires_at is not None
                    and job.expires_at <= now
                    and job.id not in self._expired_content
                ):
                    await separation_file_effect(
                        lambda job=job: self.media.delete(job.id)
                    )
                    self._expired_content.add(job.id)

    async def interrupt(self) -> None:
        """Fail old-session jobs without replaying their edits into a new master."""
        for identifier in tuple(self.active):
            await self.fail(
                identifier,
                "The cluster session changed during separation.",
                cancel=False,
            )

    async def close(self) -> None:
        """Release file owners only after API task/source effects have joined."""
        async with self.lock:
            if self._sources is not None:
                await separation_file_effect(self.sources.close)
            if self._media is not None:
                await separation_file_effect(self.media.close)
            if self._jobs is not None:
                await separation_file_effect(self.jobs.close)
