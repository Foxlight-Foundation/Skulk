"""API-local source-conditioned music jobs over existing directed media planes."""

from __future__ import annotations

import contextlib
import hashlib
import os
import secrets
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import final

import anyio
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from pydantic import JsonValue, TypeAdapter
from starlette.types import Receive, Scope, Send

from skulk.api.audio_source_delivery import send_original_wav
from skulk.api.audio_uploads import read_wav_upload
from skulk.api.music_edit_jobs import (
    MAX_ACTIVE_MUSIC_EDITS,
    MAX_RETAINED_MUSIC_EDITS,
    MusicEditCapacityError,
    MusicEditJob,
    MusicEditJobRegistry,
    MusicEditStorageError,
)
from skulk.api.music_edit_media import MusicEditMediaStore, StoredMusicEdit
from skulk.api.types.music_edits import (
    MusicEditDeletedResponse,
    MusicEditListResponse,
    MusicEditResource,
    music_edit_resource,
)
from skulk.routing.output_media import OutputMediaPacket
from skulk.routing.vision_media import VisionMediaPacket
from skulk.shared.constants import SKULK_MAX_CHUNK_SIZE
from skulk.shared.models.model_cards import ModelCard, ModelId
from skulk.shared.types.audio_separation import source_from_wav
from skulk.shared.types.chunks import ErrorChunk, MusicEditingChunk
from skulk.shared.types.commands import Command, TaskCancelled
from skulk.shared.types.commands import MusicEditing as MusicEditCommand
from skulk.shared.types.common import CommandId, NodeId, SystemId
from skulk.shared.types.music_edits import MusicEditRequest, resolve_music_edit
from skulk.shared.types.state import State
from skulk.shared.types.tasks import MusicEditing, TaskStatus
from skulk.shared.types.worker.instances import InstanceId
from skulk.utils.channels import Receiver
from skulk.utils.task_group import TaskGroup
from skulk.worker.separation_custody import SeparationSourceCustody
from skulk.worker.separation_input import separation_file_effect
from skulk.worker.separation_sources import SeparationSourceStore

_REQUEST = TypeAdapter[MusicEditRequest](MusicEditRequest)
_UPLOAD_SECONDS = 180.0


async def _storage_error_response(_request: Request, _error: Exception) -> JSONResponse:
    """Keep uncertain edit storage fenced and return an actionable safe failure."""
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Music edit state is unavailable after a storage failure. Restart the API to recheck durable state, then inspect retained jobs before retrying.",
        },
    )


def _request_content_schema() -> dict[str, JsonValue]:
    """Inline the three closed request schemas inside the multipart JSON string.

    A nested contentSchema cannot reference the OpenAPI document's root as
    though it were a standalone Pydantic schema. These fixed acyclic inputs are
    expanded so published references describe the actual accepted metadata.
    """
    schema = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue]).validate_python(
        _REQUEST.json_schema()
    )
    definitions = schema.get("$defs")
    assert isinstance(definitions, dict)

    def expand(value: JsonValue) -> JsonValue:
        if isinstance(value, dict):
            reference = value.get("$ref")
            if isinstance(reference, str):
                assert reference.startswith("#/$defs/")
                return expand(definitions[reference.removeprefix("#/$defs/")])
            return {
                key: expand(child) for key, child in value.items() if key != "$defs"
            }
        if isinstance(value, list):
            return [expand(child) for child in value]
        return value

    expanded = expand(schema)
    assert isinstance(expanded, dict)
    # OpenAPI's discriminator mapping values also point to component paths.
    # Inline oneOf alternatives retain their task const; the discriminator
    # property is sufficient here without publishing dangling mapping paths.
    discriminator = expanded.get("discriminator")
    if isinstance(discriminator, dict):
        discriminator.pop("mapping", None)
    return expanded


@dataclass
class _Active:
    upload: MusicEditing
    session: SystemId
    deadline: float
    task: MusicEditing | None = None
    producer: NodeId | None = None
    sending: bool = False
    source_accepted: bool = False
    source_released: bool = False
    opened: tuple[int, int] | None = None
    early: list[OutputMediaPacket] = field(default_factory=list)
    completion: OutputMediaPacket | None = None
    receipt: StoredMusicEdit | None = None
    acknowledging: bool = False
    cancel_started: bool = False
    cancel_submitted: anyio.Event = field(default_factory=anyio.Event)
    return_settled: anyio.Event = field(default_factory=anyio.Event)


@dataclass
class _Cleanup:
    entry: _Active | None
    delete_result: bool
    result_released: bool = False
    source_released: bool = False


class _ContentResponse(StreamingResponse):
    """Lease a verified descriptor before headers and retain it across deletion."""

    def __init__(self, service: MusicEditAPI, identifier: CommandId) -> None:
        self.service, self.identifier = service, identifier
        super().__init__(iter(()), media_type="audio/wav")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Bound concurrent downloads and join each read before closing its handle."""
        try:
            self.service.downloads.acquire_nowait()
        except anyio.WouldBlock:
            raise HTTPException(
                503, "too many music edit downloads are active"
            ) from None
        descriptor: int | None = None
        try:
            async with self.service.lock:
                await self.service.ensure()
                job = await self.service.job(self.identifier)
                if job.status != "completed":
                    raise HTTPException(409, f"music edit is {job.status}")
                opened: list[tuple[int, StoredMusicEdit]] = []

                def claim() -> None:
                    stored = self.service.media.content(job)
                    if stored is None:
                        raise HTTPException(
                            404, "music edit content is absent or expired"
                        )
                    opened.append(
                        (os.open(stored.file_path, os.O_RDONLY | os.O_NOFOLLOW), stored)
                    )

                try:
                    await separation_file_effect(claim)
                finally:
                    if opened:
                        descriptor = opened[0][0]
                assert descriptor is not None
                owned, stored = opened[0]

            async def frames() -> AsyncIterator[bytes]:
                remaining = stored.facts.size_bytes
                while remaining:

                    def read_frame(remaining: int = remaining) -> bytes:
                        return os.read(owned, min(remaining, SKULK_MAX_CHUNK_SIZE))

                    data = await separation_file_effect(read_frame)
                    if not data:
                        raise OSError("Retained music edit became incomplete")
                    remaining -= len(data)
                    yield data

            self.body_iterator = frames()
            self.init_headers(
                {
                    "content-length": str(stored.facts.size_bytes),
                    "content-type": "audio/wav",
                    "etag": '"' + stored.facts.sha256 + '"',
                    "content-disposition": f'attachment; filename="{job.id}.wav"',
                }
            )
            await super().__call__(scope, receive, send)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            self.service.downloads.release()


@final
class MusicEditAPI:
    """One lazy owner of edit intent, originals, output receipts and async drains."""

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
        open_queue: Callable[[CommandId], Receiver[MusicEditingChunk | ErrorChunk]],
        finalize: Callable[[CommandId, SystemId], Awaitable[None]],
        close_queue: Callable[[CommandId], None],
        *,
        seed: Callable[[], int] = lambda: secrets.randbits(32),
    ) -> None:
        """Bind existing authenticated host effects; perform no startup I/O or download."""
        self.node_id, self.root, self.state, self.tasks = node_id, root, state, tasks
        self.load_card, self.send_command, self.session = (
            load_card,
            send_command,
            session,
        )
        self.send_source, self.send_output = send_source, send_output
        self.open_queue, self.finalize, self.close_queue = (
            open_queue,
            finalize,
            close_queue,
        )
        self.seed = seed
        self.lock = anyio.Lock()
        self.uploads, self.downloads = anyio.CapacityLimiter(4), anyio.Semaphore(8)
        self._jobs: MusicEditJobRegistry | None = None
        self._media: MusicEditMediaStore | None = None
        self._sources: SeparationSourceStore[MusicEditing] | None = None
        self.active: dict[CommandId, _Active] = {}
        self._controls: dict[CommandId, int] = {}
        self._source_senders: set[CommandId] = set()
        self._expired: set[CommandId] = set()
        self._cleanup: dict[CommandId, _Cleanup] = {}
        self._cleanup_running = False

    @property
    def jobs(self) -> MusicEditJobRegistry:
        """Return the initialized exclusive metadata owner."""
        assert self._jobs is not None
        return self._jobs

    @property
    def media(self) -> MusicEditMediaStore:
        """Return the initialized independently verified result owner."""
        assert self._media is not None
        return self._media

    @property
    def sources(self) -> SeparationSourceStore[MusicEditing]:
        """Return private copies of encoded originals accepted by this API."""
        assert self._sources is not None
        return self._sources

    def register(self, app: FastAPI) -> None:
        """Register six authenticated-by-host routes before dynamic music IDs."""
        app.add_exception_handler(MusicEditStorageError, _storage_error_response)
        app.post(
            "/v1/music/edits",
            response_model=MusicEditResource,
            status_code=202,
            tags=["Music Editing"],
            summary="Create a source-conditioned music edit",
            description="Accept exactly request JSON and one original file WAV as multipart/form-data. task is add_part, repaint or continue; model, prompt and source_sample_rate are required. Optional lyrics and seed obey the card. Add a part requires part_class and permits a source-frame region. Repaint requires region_end_frame and permits region_start_frame. Continue requires total target_frames in the source clock. All operations are bounded to 120 seconds; no paths or native options. The accepting node persists one seed before dispatch.",
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
                                        "contentMediaType": "application/json",
                                        "contentSchema": _request_content_schema(),
                                        "description": "Closed task-discriminated JSON metadata; at most 256 KiB. Frame inputs use source_sample_rate, which must match the original WAV.",
                                    },
                                    "file": {
                                        "type": "string",
                                        "format": "binary",
                                        "description": "Finite mono/stereo PCM or float WAV, 1–120 seconds, at most 64 MiB.",
                                    },
                                },
                            }
                        }
                    },
                }
            },
        )(self.create)
        app.get(
            "/v1/music/edits",
            response_model=MusicEditListResponse,
            tags=["Music Editing"],
            summary="List music edits",
            description="Read newest-first committed jobs from the accepting node. Four active and 64 retained; use the retained last-id cursor.",
        )(self.list_jobs)
        app.get(
            "/v1/music/edits/{edit_id}",
            response_model=MusicEditResource,
            tags=["Music Editing"],
            summary="Retrieve a music edit",
            description="Read accepted source, fixed seed, resolved 48 kHz extent and committed status. Output is exposed after render, verified WAV custody and drained return acknowledgement.",
        )(self.retrieve)
        app.get(
            "/v1/music/edits/{edit_id}/content",
            response_class=StreamingResponse,
            tags=["Music Editing"],
            summary="Download a completed music edit WAV",
            description="Download PCM24 stereo 48 kHz WAV from the accepting node. Content is at most 64 MiB and retained for 24 hours. Unsettled jobs return 409; absent/expired content returns 404. ETag is its quoted SHA-256.",
            responses={
                200: {
                    "description": "Complete verified edit WAV.",
                    "content": {
                        "audio/wav": {"schema": {"type": "string", "format": "binary"}}
                    },
                    "headers": {
                        "ETag": {
                            "schema": {"type": "string"},
                            "description": "Quoted complete WAV SHA-256.",
                        }
                    },
                }
            },
        )(self.content)
        app.post(
            "/v1/music/edits/{edit_id}/cancel",
            response_model=MusicEditResource,
            tags=["Music Editing"],
            summary="Cancel a music edit",
            description="Cancel inference/source/result delivery, remove partial copies and retain terminal metadata. Completed and other terminal jobs remain unchanged; no edit is replayed.",
        )(self.cancel)
        app.delete(
            "/v1/music/edits/{edit_id}",
            response_model=MusicEditDeletedResponse,
            tags=["Music Editing"],
            summary="Delete a music edit",
            description="Cancel active work, remove API-owned content and forget local metadata. Incomplete physical cleanup returns 503, retains metadata/admission and retries in the background; retry delete after storage recovers. Creator originals remain unchanged. A missing job returns 404.",
        )(self.delete)

    def _initialize(self) -> None:
        jobs: MusicEditJobRegistry | None = None
        media: MusicEditMediaStore | None = None
        sources: SeparationSourceStore[MusicEditing] | None = None
        try:
            jobs = MusicEditJobRegistry(self.root / "index" / "jobs.json")
            media = MusicEditMediaStore(self.root / "media")
            custody = SeparationSourceCustody[MusicEditing](
                self.root / "inputs",
                self.root / "upload-readers",
                lambda _task: hashlib.sha256(b"api-upload-reader").hexdigest(),
                # API upload copies have no native result directory. Only the
                # worker owner may apply ACE's separate result-custody rules.
                reclaim_result=lambda _root, _command: True,
            )
            sources = SeparationSourceStore[MusicEditing](
                self.root / "inputs", custody=custody
            )
            adopted: set[CommandId] = set()
            for job in jobs.list(limit=MAX_RETAINED_MUSIC_EDITS):
                if job.status == "completed":
                    if media.adopt_complete(job):
                        adopted.add(job.id)
                    else:
                        jobs.invalidate(
                            job.id,
                            "Music edit content did not survive restart or has expired.",
                        )
            media.purge_unknown(frozenset(adopted))
            self._jobs, self._media, self._sources = jobs, media, sources
        except BaseException as error:
            if sources is not None:
                sources.close()
            if media is not None:
                media.close()
            if jobs is not None:
                jobs.close()
            if isinstance(error, OSError):
                raise MusicEditStorageError(
                    "Music edit storage recovery did not finish; repair storage before retrying."
                ) from error
            raise

    async def ensure(self) -> None:
        """Lazily initialize under the actor lock and retain effects through cancel."""
        if self._jobs is None:
            await separation_file_effect(self._initialize)

    async def job(self, identifier: CommandId) -> MusicEditJob:
        """Read a committed node-local intent or return a stable 404."""
        job = await separation_file_effect(lambda: self.jobs.get(identifier))
        if job is None:
            raise HTTPException(404, "music edit not found on this API node")
        return job

    async def create(self, request: Request) -> MusicEditResource:
        """Freeze exact validated intent/source and submit its command once."""
        try:
            self.uploads.acquire_nowait()
        except anyio.WouldBlock:
            raise HTTPException(503, "four audio uploads are already active") from None
        identifier: CommandId | None = None
        try:
            request_params, data = await read_wav_upload(
                request,
                _REQUEST.validate_json,
                metadata_bytes=256 * 1024,
                deadline_seconds=_UPLOAD_SECONDS,
            )
            card = await self.load_card(ModelId(request_params.model))
            try:
                source = await separation_file_effect(lambda: source_from_wav(data))
                chunks = (len(data) + SKULK_MAX_CHUNK_SIZE - 1) // SKULK_MAX_CHUNK_SIZE
                params = resolve_music_edit(
                    request_params,
                    source,
                    card,
                    resolved_seed=request_params.seed
                    if request_params.seed is not None
                    else self.seed(),
                    total_input_chunks=chunks,
                )
            except ValueError:
                raise HTTPException(
                    422,
                    "music edit source, clock, controls or card requirements are invalid",
                ) from None
            identifier = CommandId()
            upload = MusicEditing(
                command_id=identifier,
                owner_node=self.node_id,
                instance_id=InstanceId("upload-" + str(identifier)),
                task_params=params,
            )
            async with self.lock:
                await self.ensure()
                if (
                    len(
                        self.active.keys()
                        | self._controls.keys()
                        | self._source_senders
                        | self._cleanup.keys()
                    )
                    >= MAX_ACTIVE_MUSIC_EDITS
                ):
                    raise MusicEditCapacityError(
                        "Four edits or transfer drains are active"
                    )
                job = MusicEditJob(
                    id=identifier, params=params, created_at=int(time.time())
                )
                evicted = await separation_file_effect(lambda: self.jobs.create(job))
                for old in evicted:
                    await separation_file_effect(lambda old=old: self.media.delete(old))
                entry = _Active(upload, self.session(), time.monotonic() + 45)
                self.active[identifier] = entry

                def retain() -> None:
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
                        raise ValueError("Music edit original did not verify")

                await separation_file_effect(retain)
                self.tasks.start_soon(
                    self._drain,
                    identifier,
                    self.open_queue(identifier),
                    entry.session,
                    entry,
                )
            with anyio.fail_after(10):
                await self.send_command(
                    MusicEditCommand(
                        command_id=identifier,
                        owner_node=self.node_id,
                        task_params=params,
                    ),
                    entry.session,
                )
            return music_edit_resource(job)
        except (
            MusicEditCapacityError,
            MusicEditStorageError,
            OSError,
            ValueError,
        ) as error:
            logger.opt(exception=error).warning("Music edit admission/storage failed")
            if identifier is not None:
                try:
                    await self.fail(
                        identifier,
                        "Music edit storage admission failed; inspect API diagnostics.",
                    )
                except (MusicEditStorageError, OSError) as cleanup_error:
                    # The file owner retains its uncertainty fence. A failed
                    # admission must still return a useful 503, and no command
                    # is replayed to guess which durable write took effect.
                    logger.opt(exception=cleanup_error).warning(
                        "Music edit admission remains storage-fenced"
                    )
            raise HTTPException(
                503, "music edit capacity or storage is unavailable"
            ) from None
        except BaseException:
            if identifier is not None:
                with anyio.CancelScope(shield=True):
                    await self.fail(
                        identifier,
                        "Music edit submission was interrupted; inspect the job before retrying.",
                    )
            raise
        finally:
            self.uploads.release()

    async def observe_task(self, task: MusicEditing) -> None:
        """Bind only one authoritative complete native placement to an accepted edit."""
        entry = self.active.get(task.command_id)
        if (
            entry is None
            or task.owner_node != self.node_id
            or task.task_params != entry.upload.task_params
        ):
            return
        snapshot = self.state()
        instance = snapshot.instances.get(task.instance_id)
        if instance is None or snapshot.tasks.get(task.task_id) != task:
            return
        assignments = instance.shard_assignments
        if (
            len(assignments.node_to_runner) != 1
            or len(assignments.runner_to_shard) != 1
        ):
            return
        producer, runner = next(iter(assignments.node_to_runner.items()))
        shard = assignments.runner_to_shard.get(runner)
        if (
            shard is None
            or shard.model_card.model_id != task.task_params.model
            or shard.model_card.music is None
            or shard.model_card.music.native is None
            or shard.world_size != 1
            or shard.device_rank != 0
            or shard.start_layer != 0
            or shard.end_layer != shard.model_card.n_layers
            or shard.n_layers != shard.model_card.n_layers
            or shard.resolved_backend
            not in {"ace_step-cpu", "ace_step-metal", "ace_step-cuda", "ace_step-rocm"}
            or shard.resolved_engine_build is None
            or task.task_status not in {TaskStatus.Pending, TaskStatus.Running}
        ):
            return
        async with self.lock:
            if (
                self.active.get(task.command_id) is not entry
                or entry.task is not None
                and (
                    entry.task.task_id != task.task_id
                    or entry.task.instance_id != task.instance_id
                    or entry.producer != producer
                )
            ):
                return
            try:
                await separation_file_effect(
                    lambda: self.jobs.bind_worker(task.command_id, producer)
                )
            except MusicEditStorageError as error:
                logger.opt(exception=error).warning(
                    "Music edit placement owner is storage-fenced"
                )
                await self._fence_storage_locked()
                return
            entry.task, entry.producer = task, producer
            entry.deadline = time.monotonic() + 40 * 60

    async def receive_source(self, packet: VisionMediaPacket) -> None:
        """Answer a placed worker's request; packets never manufacture task authority."""
        entry = self.active.get(packet.command_id)
        if (
            entry is None
            or packet.payload != "music_edit_source"
            or packet.target_node != self.node_id
            or packet.model != entry.upload.task_params.model
        ):
            return
        if entry.producer is None:
            for task in self.state().tasks.values():
                if (
                    isinstance(task, MusicEditing)
                    and task.command_id == packet.command_id
                ):
                    await self.observe_task(task)
        async with self.lock:
            if (
                self.active.get(packet.command_id) is not entry
                or entry.task is None
                or packet.source_node != entry.producer
            ):
                return
            if (
                packet.kind == "requested"
                and not entry.sending
                and not entry.source_accepted
            ):
                if (
                    packet.total_chunks == entry.upload.task_params.total_input_chunks
                    and packet.image_count == 1
                ):
                    entry.sending = True
                    self._source_senders.add(packet.command_id)
                    try:
                        self.tasks.start_soon(
                            self._send_original, packet.command_id, entry
                        )
                    except BaseException:
                        entry.sending = False
                        self._source_senders.discard(packet.command_id)
                        raise
            elif (
                packet.kind == "accepted"
                and packet.sequence == entry.upload.task_params.total_input_chunks + 1
            ):
                entry.source_accepted = True
                await self._release_source_locked(packet.command_id, entry)
            elif packet.kind == "transport_failed":
                await self._fail_locked(
                    packet.command_id,
                    "Music edit original transfer failed.",
                    cancelled=False,
                )

    async def _send_original(self, identifier: CommandId, entry: _Active) -> None:
        try:
            path = self.sources.ready_path(identifier)
            if path is None or entry.producer is None:
                raise ValueError("Music edit original is not retained for its producer")
            await send_original_wav(
                path,
                entry.upload,
                entry.producer,
                "music_edit_source",
                self.send_source,
                lambda: self.active.get(identifier) is entry
                and not entry.source_accepted,
            )
        except Exception as error:  # noqa: BLE001 - settle this owned sender
            logger.opt(exception=error).warning("Music edit original delivery failed")
            await self.fail(identifier, "Music edit original delivery failed.")
        finally:
            with anyio.CancelScope(shield=True):
                async with self.lock:
                    entry.sending = False
                    self._source_senders.discard(identifier)
                    await self._release_source_locked(identifier, entry)

    async def _release_source_locked(
        self, identifier: CommandId, entry: _Active
    ) -> bool:
        if (
            not entry.source_released
            and not entry.sending
            and (entry.source_accepted or self.active.get(identifier) is not entry)
        ):
            try:
                await separation_file_effect(lambda: self.sources.delete(identifier))
            except OSError as error:
                logger.opt(exception=error).warning(
                    "Music edit original cleanup remains reserved for retry"
                )
                self._queue_cleanup_locked(identifier, entry, delete_result=False)
                return False
            entry.source_released = True
        return entry.source_released

    def _queue_cleanup_locked(
        self, identifier: CommandId, entry: _Active | None, *, delete_result: bool
    ) -> None:
        # Terminal metadata cannot carry physical cleanup ownership: it may be
        # deleted while a sender or failed unlink still retains bytes. Keep the
        # accepted identity independently and charge it against admission.
        pending = self._cleanup.get(identifier)
        if pending is None:
            self._cleanup[identifier] = _Cleanup(
                entry, delete_result, result_released=not delete_result
            )
        else:
            if pending.entry is None:
                pending.entry = entry
            if delete_result and not pending.delete_result:
                pending.delete_result = True
                pending.result_released = False
        if not self._cleanup_running:
            self._cleanup_running = True
            try:
                self.tasks.start_soon(self._cleanup_loop)
            except BaseException:
                self._cleanup_running = False
                raise

    async def _retry_cleanup_locked(self, identifier: CommandId) -> None:
        pending = self._cleanup.get(identifier)
        if pending is None:
            return
        if not pending.result_released:
            try:
                await separation_file_effect(lambda: self.media.delete(identifier))
            except OSError as error:
                logger.opt(exception=error).warning(
                    "Music edit partial result cleanup remains reserved for retry"
                )
            else:
                pending.result_released = True
        if not pending.source_released:
            if pending.entry is not None:
                pending.source_released = await self._release_source_locked(
                    identifier, pending.entry
                )
            else:
                try:
                    await separation_file_effect(
                        lambda: self.sources.delete(identifier)
                    )
                except OSError as error:
                    logger.opt(exception=error).warning(
                        "Music edit undispatched original cleanup remains reserved for retry"
                    )
                else:
                    pending.source_released = True
        if pending.result_released and pending.source_released:
            self._cleanup.pop(identifier)

    async def _cleanup_loop(self) -> None:
        # Cleanup has no journal reads and outlives a storage-fenced job sweeper.
        # One actor retries at most the bounded admitted/retained identities.
        try:
            while True:
                await anyio.sleep(1)
                async with self.lock:
                    for identifier in tuple(self._cleanup):
                        await self._retry_cleanup_locked(identifier)
                    if not self._cleanup:
                        return
        finally:
            self._cleanup_running = False

    async def _drain(
        self,
        identifier: CommandId,
        receiver: Receiver[MusicEditingChunk | ErrorChunk],
        session: SystemId,
        accepted: _Active,
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
                            "Native music edit ended without a complete result.",
                        )
                        return
                    async with self.lock:
                        entry = self.active.get(identifier)
                        if (
                            entry is None
                            or entry.producer is None
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
                        await self._reconcile_locked(identifier, entry)
                    # Render metadata precedes independent WAV custody. A
                    # TaskFinished here would delete the worker's task and
                    # cancel its still-owned return stream before the ACK.
                    receiver.close()
                    await accepted.return_settled.wait()
                    return
            await self.fail(
                identifier, "Music edit stream ended without a complete result."
            )
        except MusicEditStorageError as error:
            logger.opt(exception=error).warning(
                "Music edit result owner is storage-fenced"
            )
            async with self.lock:
                await self._fence_storage_locked()
        except (OSError, ValueError) as error:
            logger.opt(exception=error).warning(
                "Music edit metadata failed verification"
            )
            await self.fail(
                identifier, "Music edit result metadata failed verification."
            )
        finally:
            if accepted.cancel_started:
                # TaskFinished must not remove the authoritative task before
                # TaskCancelled has had its bounded submission attempt. This
                # wait holds no actor lock or media receiver rendezvous.
                await accepted.cancel_submitted.wait()
            await self.finalize(identifier, session)

    async def receive_output(self, packet: OutputMediaPacket) -> None:
        """Accept bounded WAV frames only from the exact authoritative producer."""
        async with self.lock:
            entry = self.active.get(packet.command_id)
            if (
                entry is None
                or packet.purpose != "music_edit"
                or packet.target_node != self.node_id
                or packet.source_node != entry.producer
                or packet.model != entry.upload.task_params.model
            ):
                return
            try:
                await self._output_locked(entry, packet)
            except MusicEditStorageError as error:
                logger.opt(exception=error).warning(
                    "Music edit return owner is storage-fenced"
                )
                await self._fence_storage_locked()
            except (OSError, ValueError) as error:
                logger.opt(exception=error).warning(
                    "Music edit bytes failed verification"
                )
                await self._fail_locked(
                    packet.command_id,
                    "Music edit result delivery failed verification.",
                    cancelled=False,
                )

    async def _output_locked(self, entry: _Active, packet: OutputMediaPacket) -> None:
        identifier = packet.command_id
        if packet.kind == "transport_failed":
            await self._fail_locked(
                identifier, "Music edit output transport failed.", cancelled=False
            )
            return
        if packet.kind not in {"opened", "chunk", "completed"}:
            return
        if entry.receipt is not None:
            if packet.kind == "completed" and packet == entry.completion:
                await self._reconcile_locked(identifier, entry)
            return
        if packet.kind != "opened" and entry.opened is None:
            retained = sum(
                len(frame.data)
                for active in self.active.values()
                for frame in active.early
            )
            if len(entry.early) >= 8 or retained + len(packet.data) > 8 * 1024**2:
                raise ValueError("Music edit result exceeded bounded reordering")
            entry.early.append(packet)
            return
        if packet.kind == "opened":
            if (
                packet.total_bytes is None
                or packet.total_chunks is None
                or packet.content_type != "audio/wav"
            ):
                raise ValueError("Invalid music edit output declaration")
            await separation_file_effect(
                lambda: self.media.open_assembly(
                    identifier,
                    total_bytes=packet.total_bytes or 0,
                    total_chunks=packet.total_chunks or 0,
                )
            )
            entry.opened = packet.total_bytes, packet.total_chunks
            held, entry.early = entry.early, []
            for earlier in held:
                await self._output_locked(entry, earlier)
            return
        assert entry.opened is not None
        size, chunks = entry.opened
        if (
            packet.total_chunks != chunks
            or packet.total_bytes is not None
            and packet.total_bytes != size
        ):
            raise ValueError("Music edit transfer changed its identity")
        if packet.kind == "chunk":
            await separation_file_effect(
                lambda: self.media.append(identifier, packet.sequence, packet.data)
            )
        else:
            if entry.completion is not None and entry.completion != packet:
                raise ValueError("Music edit transfer changed its completion")
            entry.completion = packet
        await self._reconcile_locked(identifier, entry)

    async def _reconcile_locked(self, identifier: CommandId, entry: _Active) -> None:
        job = await self.job(identifier)
        output, completion, producer = (
            job.candidate_output,
            entry.completion,
            entry.producer,
        )
        if (
            output is None
            or completion is None
            or producer is None
            or entry.opened is None
        ):
            return
        facts = output.result.output
        if (completion.sha256, completion.total_bytes, completion.total_chunks) != (
            facts.sha256,
            facts.size_bytes,
            entry.opened[1],
        ):
            raise ValueError("Music edit return differs from native result metadata")
        if entry.receipt is None:
            if not await separation_file_effect(
                lambda: self.media.assembly_complete(identifier)
            ):
                return
            count = entry.opened[1]
            entry.receipt = await separation_file_effect(
                lambda: self.media.commit(identifier, facts, total_chunks=count)
            )
            receipt = entry.receipt
            await separation_file_effect(
                lambda: self.jobs.record_verified_output(
                    identifier, producer, output, expires_at=int(receipt.expires_at)
                )
            )
        if not entry.acknowledging:
            entry.acknowledging = True
            self._start_control(
                identifier, entry, acknowledgement=completion.accepted()
            )

    def _start_control(
        self,
        identifier: CommandId,
        entry: _Active,
        *,
        acknowledgement: OutputMediaPacket | None = None,
    ) -> None:
        # One ACK and one cancellation are bounded separately. Keep admission
        # until both drain, including when terminal job metadata is deleted.
        pending = self._controls.get(identifier, 0)
        if pending >= 2:
            raise ValueError("Music edit reverse-control capacity is full")
        self._controls[identifier] = pending + 1
        if acknowledgement is None:
            entry.cancel_started = True
        try:
            self.tasks.start_soon(self._control, identifier, entry, acknowledgement)
        except BaseException:
            if acknowledgement is None:
                entry.cancel_submitted.set()
                self.close_queue(identifier)
            if pending:
                self._controls[identifier] = pending
            else:
                self._controls.pop(identifier)
            raise

    async def _control(
        self,
        identifier: CommandId,
        entry: _Active,
        acknowledgement: OutputMediaPacket | None,
    ) -> None:
        try:
            if acknowledgement is None:
                try:
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
                finally:
                    entry.cancel_submitted.set()
                    self.close_queue(identifier)
                if entry.producer is not None:
                    with anyio.fail_after(5):
                        await self.send_output(
                            OutputMediaPacket(
                                source_node=self.node_id,
                                target_node=entry.producer,
                                command_id=identifier,
                                model=ModelId(entry.upload.task_params.model),
                                purpose="music_edit",
                                kind="cancelled",
                                sequence=0,
                            )
                        )
            else:
                with anyio.fail_after(5):
                    await self.send_output(acknowledgement)
                async with self.lock:
                    if (
                        self.active.get(identifier) is entry
                        and entry.producer is not None
                    ):
                        producer = entry.producer
                        await separation_file_effect(
                            lambda: self.jobs.acknowledge_return(identifier, producer)
                        )
                        settled = await separation_file_effect(
                            lambda: self.jobs.settle(identifier)
                        )
                        if settled is not None and settled.status == "completed":
                            self.active.pop(identifier)
                            entry.return_settled.set()
                            await self._release_source_locked(identifier, entry)
        except MusicEditStorageError as error:
            logger.opt(exception=error).warning(
                "Music edit acknowledgement owner is storage-fenced"
            )
            async with self.lock:
                await self._fence_storage_locked()
        except (
            anyio.BrokenResourceError,
            anyio.ClosedResourceError,
            OSError,
            ValueError,
            TimeoutError,
        ) as error:
            logger.opt(exception=error).warning("Music edit reverse control failed")
            if acknowledgement is not None:
                await self.fail(identifier, "Music edit result acknowledgement failed.")
        finally:
            with anyio.CancelScope(shield=True):
                if acknowledgement is None:
                    entry.cancel_submitted.set()
                    self.close_queue(identifier)
                async with self.lock:
                    pending = self._controls[identifier]
                    if pending == 1:
                        self._controls.pop(identifier)
                    else:
                        self._controls[identifier] = pending - 1

    async def _fail_locked(
        self, identifier: CommandId, message: str, *, cancelled: bool
    ) -> None:
        if self._jobs is None:
            return
        try:
            job = await separation_file_effect(lambda: self.jobs.get(identifier))
            if job is None or job.is_terminal:
                await self._retry_cleanup_locked(identifier)
                return
            await separation_file_effect(
                lambda: self.jobs.fail(identifier, message, cancelled=cancelled)
            )
        except MusicEditStorageError as error:
            # Failure can itself be entered from a transport validation handler.
            # Such nested exceptions bypass that handler's sibling except blocks;
            # fence centrally before uncertainty can escape the host task group.
            logger.opt(exception=error).warning(
                "Music edit failure transition is storage-fenced"
            )
            await self._fence_storage_locked()
            return
        await self._abort_entry_locked(identifier)

    async def _abort_entry_locked(self, identifier: CommandId) -> None:
        entry = self.active.get(identifier)
        self._queue_cleanup_locked(identifier, entry, delete_result=True)
        self.active.pop(identifier, None)
        if entry is not None:
            self._start_control(identifier, entry)
            entry.return_settled.set()
        await self._retry_cleanup_locked(identifier)

    async def _fence_storage_locked(self) -> None:
        # The durable owner already refuses reads/writes. Cancel accepted native
        # work using retained in-memory identities without inventing a terminal
        # durable outcome or taking down unrelated host API task groups.
        for identifier in tuple(self.active):
            try:
                await self._abort_entry_locked(identifier)
            except OSError as error:
                logger.opt(exception=error).warning(
                    "Music edit cleanup remains retained after storage fencing"
                )

    async def fail(
        self, identifier: CommandId, message: str, *, cancel: bool = False
    ) -> None:
        """Fail/cancel accepted work once; late media cannot revive its receipt."""
        async with self.lock:
            try:
                await self._fail_locked(identifier, message, cancelled=cancel)
            except MusicEditStorageError as error:
                logger.opt(exception=error).warning(
                    "Music edit failure owner is storage-fenced"
                )
                await self._fence_storage_locked()

    async def retrieve(self, edit_id: str) -> MusicEditResource:
        """Read committed metadata from the accepting API node."""
        async with self.lock:
            await self.ensure()
            return music_edit_resource(await self.job(CommandId(edit_id)))

    async def list_jobs(
        self, limit: int = Query(20, ge=1, le=63), after: str | None = Query(None)
    ) -> MusicEditListResponse:
        """Return bounded stable pagination without partial results or worker IDs."""
        async with self.lock:
            await self.ensure()
            try:
                jobs = await separation_file_effect(
                    lambda: self.jobs.list(
                        limit=limit + 1,
                        after=CommandId(after) if after is not None else None,
                    )
                )
            except ValueError:
                raise HTTPException(400, "music edit cursor is not retained") from None
            page = tuple(music_edit_resource(job) for job in jobs[:limit])
            return MusicEditListResponse(
                data=page,
                first_id=page[0].id if page else None,
                last_id=page[-1].id if page else None,
                has_more=len(jobs) > limit,
            )

    async def content(self, edit_id: str) -> StreamingResponse:
        """Return a bounded descriptor-leased completed edit WAV."""
        return _ContentResponse(self, CommandId(edit_id))

    async def cancel(self, edit_id: str) -> MusicEditResource:
        """Record terminal cancellation and schedule native/media drain."""
        async with self.lock:
            await self.ensure()
            job = await self.job(CommandId(edit_id))
            await self._fail_locked(
                job.id, "The music edit was cancelled.", cancelled=True
            )
            return music_edit_resource(await self.job(job.id))

    async def delete(self, edit_id: str) -> MusicEditDeletedResponse:
        """Cancel active work then remove only API-owned content and metadata."""
        async with self.lock:
            await self.ensure()
            job = await self.job(CommandId(edit_id))
            await self._fail_locked(
                job.id, "The music edit was deleted.", cancelled=True
            )
            if job.id in self._cleanup:
                raise HTTPException(
                    503, "music edit cleanup is still draining; retry delete"
                )
            try:
                await separation_file_effect(lambda: self.media.delete(job.id))
            except OSError as error:
                logger.opt(exception=error).warning(
                    "Music edit deletion remains reserved for retry"
                )
                self._queue_cleanup_locked(job.id, None, delete_result=True)
                raise HTTPException(
                    503, "music edit cleanup is still draining; retry delete"
                ) from None
            await separation_file_effect(lambda: self.jobs.delete(job.id))
            return MusicEditDeletedResponse(id=edit_id)

    async def expire_content(self, now: int) -> None:
        """Release expired WAV capacity while retaining bounded terminal records."""
        async with self.lock:
            if self._jobs is None:
                return
            jobs = await separation_file_effect(
                lambda: self.jobs.list(limit=MAX_RETAINED_MUSIC_EDITS)
            )
            self._expired.intersection_update(job.id for job in jobs)
            for job in jobs:
                if (
                    job.status == "completed"
                    and job.expires_at is not None
                    and job.expires_at <= now
                    and job.id not in self._expired
                ):
                    try:
                        await separation_file_effect(
                            lambda job=job: self.media.delete(job.id)
                        )
                    except OSError as error:
                        # The media owner retains its byte reservation when
                        # removal is incomplete. Retry this job next sweep;
                        # one filesystem failure must not cancel the host.
                        logger.opt(exception=error).warning(
                            "Music edit expiry cleanup remains reserved for retry"
                        )
                        continue
                    self._expired.add(job.id)
            # Registry retention may evict old terminal metadata before an
            # unsuccessful unlink finishes. The media owner still remembers
            # those byte promises; retry them without inventing job metadata.
            owned = await separation_file_effect(self.media.owned_identifiers)
            retained = frozenset(job.id for job in jobs)
            for identifier in owned - retained:
                try:
                    await separation_file_effect(
                        lambda identifier=identifier: self.media.delete(identifier)
                    )
                except OSError as error:
                    logger.opt(exception=error).warning(
                        "Music edit orphan cleanup remains reserved for retry"
                    )

    async def sweep(self) -> None:
        """Bound placement/render/return waits and expire content without replay."""
        while True:
            await anyio.sleep(1)
            try:
                for identifier, entry in tuple(self.active.items()):
                    if time.monotonic() >= entry.deadline:
                        await self.fail(
                            identifier,
                            "Music edit placement, render or media deadline expired.",
                        )
                await self.expire_content(int(time.time()))
            except MusicEditStorageError as error:
                logger.opt(exception=error).warning(
                    "Music edit sweeper stopped for fenced storage"
                )
                async with self.lock:
                    await self._fence_storage_locked()
                return

    async def interrupt(self) -> None:
        """Fail old-session work without resubmitting into a replacement master."""
        for identifier in tuple(self.active):
            await self.fail(
                identifier, "The cluster session changed during music editing."
            )

    async def close(self) -> None:
        """Release storage only after the host's task and transfer effects joined."""
        async with self.lock:
            if self._sources is not None:
                await separation_file_effect(self.sources.close)
            if self._media is not None:
                await separation_file_effect(self.media.close)
            if self._jobs is not None:
                await separation_file_effect(self.jobs.close)
