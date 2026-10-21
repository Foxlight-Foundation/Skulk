"""Durable preservation orchestration belonging to the actual model-store owner."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Literal, cast, final

from foxlight_model_preservation.kits import inspect_kit
from foxlight_model_preservation.models import TrustProfile
from foxlight_model_preservation.signing import create_key, verify_manifest
from foxlight_model_preservation.storage import manifest_path, read_metadata
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from skulk.shared.constants import SKULK_CUSTOM_MODEL_CARDS_DIR
from skulk.store.installed_cards import read_installed_card
from skulk.store.model_store import ModelStore
from skulk.store.preservation import inspect_verified_manifest
from skulk.store.preservation_contracts import (
    KitUploadStatus,
    PreservationCopy,
    PreservationJob,
    PreservationLocation,
    PreservationModel,
    PreservationPage,
    PreservationPlan,
    PreservationPlanRequest,
    PreservationStart,
    RecoveryCleanup,
    SharingPolicy,
    SharingStatus,
)
from skulk.store.preservation_export import plan_preservation, preservation_generation
from skulk.store.preservation_journal import (
    PreservationHostError,
    PreservationJournal,
    safe_json,
    secure_directory,
)
from skulk.store.preservation_live import publish_recovered_home
from skulk.store.preservation_uploads import KitUploads
from skulk.store.preservation_worker import WorkerRequest


class _Location(BaseModel):
    """Private enrollment; path and custody never cross the operator API."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    location_id: str
    purpose: Literal["private", "public"] = "private"
    label: str
    path: Path
    device: int
    inode: int
    trust: TrustProfile
    key_path: Path


class _Observation(BaseModel):
    """Bounded safe protocol from an exclusively owned known byte worker."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    event: Literal["progress", "expected", "result", "error", "sharing"]
    sharing: SharingStatus | None = None
    stage: str | None = None
    bytes_hashed: int = 0
    bytes_copied: int = 0
    manifest_id: str | None = None
    size_bytes: int | None = None
    member_count: int | None = None
    audience: Literal["private", "public"] | None = None
    error_code: str | None = None


def preservation_worker_command(request: Path) -> tuple[str, ...]:
    """Launch the known source or frozen worker without choosing arbitrary code."""
    if cast(bool, getattr(sys, "frozen", False)):
        return (sys.executable, "_preservation-worker", str(request.absolute()))
    return (
        sys.executable,
        "-m",
        "skulk.store.preservation_worker",
        str(request.absolute()),
    )


@final
class PreservationHost:
    """One journal and queue, independent of browser and capability generations.

    Construct only around the authoritative running ModelStore. Copy work runs
    in owned subprocesses; the same store publication fence excludes eviction,
    downloads and peer imports for a Preserve source lease.
    """

    def __init__(
        self, store: ModelStore, root: Path, *, custom_cards: Path | None = None
    ) -> None:
        self.store = store
        self.root = root
        self.journal = PreservationJournal(root)
        self.journal.interrupt_active()
        self._queue = asyncio.Lock()
        self._serving = asyncio.Semaphore(2)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._closed = False
        self._setup_lock = asyncio.Lock()
        self.uploads = KitUploads(root / "uploads", self.journal)
        self.upload_lock = asyncio.Lock()
        self.custom_cards = custom_cards or Path(str(SKULK_CUSTOM_MODEL_CARDS_DIR))

    def kit_path(self, job_id: str) -> Path:
        """Resolve only a completed owned kit artifact, never arbitrary host files."""
        job = self.journal.job(job_id)
        if job.kind != "kit_export" or job.state != "succeeded":
            raise PreservationHostError("kit_not_ready")
        path = self.root / "kits" / (job.job_id + ".foxkit")
        if inspect_kit(path).manifest_id != job.expected_manifest_id:
            raise PreservationHostError("kit_identity_changed")
        return path

    def torrent_path(self, job_id: str) -> Path:
        """Return the hash-checked standard metainfo for one owned torrent job.

        Args:
            job_id: Exact retained job handle; never a host filename.

        Returns:
            Its regular, bounded metadata file after comparing the worker receipt.

        This reads at most 8 MiB, starts no network, and exposes no signing key.
        """
        job = self.journal.job(job_id)
        if (
            job.kind != "torrent_share"
            or job.sharing is None
            or job.sharing.metainfo_sha256 is None
        ):
            raise PreservationHostError("torrent_not_ready")
        path = self.root / "torrents" / (job.job_id + ".torrent")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or not 0 < info.st_size <= 8 * 1024 * 1024
            ):
                raise PreservationHostError("torrent_metadata_changed")
            encoded = stream.read(8 * 1024 * 1024 + 1)
        if hashlib.sha256(encoded).hexdigest() != job.sharing.metainfo_sha256:
            raise PreservationHostError("torrent_metadata_changed")
        return path

    def public_trust(self, copy_id: str) -> TrustProfile:
        """Return accepted public verification keys for one retained public copy.

        Args:
            copy_id: Exact public-copy handle selected by the operator.

        Returns:
            Its immutable public trust profile, without private key material.

        This grants no registry authority, performs no network IO and does not
        admit or execute any model. Recipients must authenticate keys independently.
        """
        copy = self.journal.copy(copy_id)
        location = self._location(copy.location_id)
        if copy.audience != "public" or location.purpose != "public":
            raise PreservationHostError("public_copy_required")
        return location.trust

    def _location(self, location_id: str) -> _Location:
        value = self.journal.setting("location:" + location_id)
        if value is None:
            raise PreservationHostError("archive_not_configured")
        return _Location.model_validate_json(safe_json(value))

    @asynccontextmanager
    async def _archive(self, location: _Location) -> AsyncIterator[int]:
        try:
            descriptor = os.open(
                location.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            )
        except OSError as error:
            raise PreservationHostError("archive_unavailable") from error
        try:
            info = os.fstat(descriptor)
            if (info.st_dev, info.st_ino) != (location.device, location.inode):
                raise PreservationHostError("archive_identity_changed")
            yield descriptor
        finally:
            os.close(descriptor)

    async def setup_device(self) -> PreservationLocation:
        """Explicitly enroll the built-in private archive with protected capture custody."""
        async with self._setup_lock:
            if self._closed:
                raise PreservationHostError("preservation_host_closed")
            if self.journal.setting("location:device") is None:
                archive = self.root / "device-archive"
                secure_directory(archive)
                custody = self.root / "custody"
                secure_directory(custody)
                key = custody / "capture.pem"
                # Existing unjournaled keys are not overwritten or regenerated.
                if key.exists():
                    raise PreservationHostError("capture_identity_needs_reconciliation")
                trust = create_key(key, None)
                info = archive.stat()
                location = _Location(
                    location_id="device",
                    label="This device",
                    path=archive.absolute(),
                    device=info.st_dev,
                    inode=info.st_ino,
                    trust=trust,
                    key_path=key.absolute(),
                )
                self.journal.set_setting(
                    "location:device",
                    cast(dict[str, JsonValue], location.model_dump(mode="json")),
                )
            return await self.location_view("device")

    async def location_view(self, location_id: str) -> PreservationLocation:
        """Observe actual enrolled storage without creating any archive pathname."""
        location = self._location(location_id)
        available = False
        free: int | None = None
        try:
            async with self._archive(location) as descriptor:
                free = shutil.disk_usage(descriptor).free
                available = True
        except (OSError, PreservationHostError):
            pass
        return PreservationLocation(
            location_id=location.location_id,
            label=location.label,
            purpose=location.purpose,
            reachable=available,
            free_bytes=free,
            publisher_key_id=location.trust.keys[0].key_id,
        )

    def _selected(self, model_id: str) -> Path:
        path = self.store.get_store_path(model_id)
        if path is None:
            raise PreservationHostError("installed_model_unavailable")
        return path.absolute()

    async def page(self, offset: int = 0) -> PreservationPage:
        """Read one bounded installed/library/activity page; never submit byte work."""
        if not 0 <= offset <= 2**31:
            raise PreservationHostError("invalid_offset")
        locations = (
            ()
            if self.journal.setting("location:device") is None
            else (await self.location_view("device"),)
        )
        models: list[PreservationModel] = []
        entries = self.store.list_models()[offset : offset + 5]
        for entry in entries[:4]:
            path = self.store.get_store_path(entry.model_id)
            record = read_installed_card(path) if path is not None else None
            if record is None or record.artifact_role != "base":
                continue
            try:
                generation = (
                    preservation_generation(
                        path, artifact_roots=(self.store.store_path,)
                    )
                    if path is not None
                    else ""
                )
            except (ValueError, OSError):
                continue
            models.append(
                PreservationModel(
                    model_id=entry.model_id,
                    generation_id=generation,
                    size_bytes=entry.total_bytes,
                    verification=record.verification,
                )
            )
        copies = self.journal.copies(offset)[:4]
        jobs = self.journal.jobs(offset)[:4]
        more = (
            len(entries) > 4
            or len(self.journal.copies(offset + 4)) > 0
            or len(self.journal.jobs(offset + 4)) > 0
        )
        uploads: list[KitUploadStatus] = []
        for path in tuple(self.uploads.root.iterdir())[:8]:
            try:
                uploads.append(
                    KitUploadStatus.model_validate(self.uploads.status(path.name))
                )
            except (ValueError, OSError):
                continue
        return PreservationPage(
            bittorrent_available=importlib.util.find_spec("libtorrent") is not None,
            uploads=tuple(uploads),
            locations=locations,
            models=tuple(models),
            copies=tuple(copy for copy in copies if copy.audience == "private"),
            public_copies=tuple(copy for copy in copies if copy.audience == "public"),
            jobs=jobs,
            next_offset=offset + 4 if more else None,
        )

    async def plan(self, request: PreservationPlanRequest) -> PreservationPlan:
        """Bind a no-byte-write plan to the exact source generation and enrolled root."""
        if self._closed:
            raise PreservationHostError("preservation_host_closed")
        if request.kind in {"mirror_import", "http_share", "torrent_share"}:
            return await self._sharing_plan(request)
        if request.kind not in {"preserve", "check", "kit_export", "recover"}:
            raise PreservationHostError("operation_not_available_in_this_host_version")
        kit = None
        if request.upload_id is not None:
            if request.kind != "recover" or request.location_id != "kit":
                raise PreservationHostError("kit_recovery_selection_required")
            if not self.uploads.status(request.upload_id)["complete"]:
                raise PreservationHostError("kit_upload_incomplete")
            kit = inspect_kit(self.uploads.path(request.upload_id))
            info = self.uploads.root.stat()
            location = _Location(
                location_id="kit",
                label="Selected recovery kit",
                path=self.uploads.root.absolute(),
                device=info.st_dev,
                inode=info.st_ino,
                trust=kit.trust,
                key_path=self.root / "unused",
            )
        else:
            location = self._location(request.location_id)
            if request.kind == "preserve" and location.purpose != "private":
                raise PreservationHostError("private_archive_required")
            async with self._archive(location):
                pass
        payload: dict[str, JsonValue] = {
            "location": cast(dict[str, JsonValue], location.model_dump(mode="json"))
        }
        manifest: str | None = None
        copy_id = request.copy_id
        if request.kind == "preserve":
            if request.model_id is None or request.generation_id is None:
                raise PreservationHostError("exact_installed_selection_required")
            async with self.store.preservation_lease():
                path = self._selected(request.model_id)
                generation = preservation_generation(
                    path, artifact_roots=(self.store.store_path,)
                )
                if generation != request.generation_id:
                    raise PreservationHostError("installed_selection_changed")
                selection = plan_preservation(
                    path, artifact_roots=(self.store.store_path,)
                )
                payload.update(
                    {"model_directory": str(path), "generation_id": generation}
                )
                model_id, size, members = (
                    request.model_id,
                    selection.source_bytes,
                    selection.artifacts,
                )
        elif kit is not None:
            manifest, model_id, size = kit.manifest_id, "", kit.manifest.total_bytes
            members = ()
            payload["kit_path"] = str(
                self.uploads.path(cast(str, request.upload_id)).absolute()
            )
            payload["manifest_id"] = manifest
        else:
            if copy_id is None:
                raise PreservationHostError("selected_copy_required")
            try:
                copy = self.journal.copy(copy_id)
                if copy.location_id != request.location_id:
                    raise PreservationHostError("copy_location_changed")
                manifest, model_id, size = (
                    copy.manifest_id,
                    copy.model_id,
                    copy.size_bytes,
                )
                async with self._archive(location) as descriptor:
                    metadata = verify_manifest(
                        read_metadata(descriptor, manifest_path(manifest)),
                        location.trust,
                        manifest,
                    )
                members = tuple(
                    model_id
                    if member.member_id == metadata.root_member
                    else f"{member.role}: {member.origin_repository}"
                    for member in metadata.members
                    if member.role != "notices"
                )
            except PreservationHostError as error:
                if error.code != "copy_not_found":
                    raise
                previous = self.journal.job(copy_id)
                prior_plan, _ = self.journal.plan(previous.plan_id)
                if previous.kind != "preserve":
                    raise PreservationHostError(
                        "explicit_recovery_retry_required"
                    ) from error
                if (
                    previous.state != "interrupted"
                    or previous.expected_manifest_id is None
                    or prior_plan.location_id != request.location_id
                ):
                    raise PreservationHostError("no_reconcilable_outcome") from error
                manifest, model_id, size = (
                    previous.expected_manifest_id,
                    previous.model_id,
                    prior_plan.size_bytes,
                )
                members = prior_plan.members
            payload["manifest_id"] = manifest
        inspection = None
        if request.kind == "recover":
            if manifest is None:
                raise PreservationHostError("selected_copy_required")
            if kit is not None:
                signed = kit.manifest
            else:
                async with self._archive(location) as descriptor:
                    signed = verify_manifest(
                        read_metadata(descriptor, manifest_path(manifest)),
                        location.trust,
                        manifest,
                    )
            inspection, projected_card = inspect_verified_manifest(
                signed, manifest, "historical" if request.historical else "fresh"
            )
            model_id, size, members = (
                inspection.model_id,
                inspection.total_bytes,
                inspection.members,
            )
            conflicts = set(members) & {
                entry.model_id for entry in self.store.list_models()
            }
            from skulk.store.preservation_closure import map_recovery_members
            from skulk.store.preservation_compatibility import compatible_file_tables
            from skulk.store.preservation_export import public_card_body

            selections = {
                item.model_id: item
                for item in map_recovery_members(signed, projected_card)
            }
            matching: dict[str, JsonValue] = {}
            for alias in conflicts:
                entry = self.store.get_entry(alias)
                existing = entry.installed_card if entry is not None else None
                if (
                    existing is not None
                    and existing.preservation is not None
                    and existing.preservation.manifest_id == manifest
                ):
                    continue
                item = selections[alias]
                if (
                    existing is None
                    or entry is None
                    or item.role == "base"
                    or existing.artifact_role != item.role
                    or existing.owner_model_id != model_id
                    or existing.artifact_repository != item.member.origin_repository
                    or existing.artifact_revision != item.member.origin_revision
                ):
                    raise PreservationHostError("existing_model_conflict")
                old = existing.model_card.model_copy(
                    update={
                        "source_repository": existing.model_card.artifact_repository
                    }
                )
                new = projected_card.model_copy(
                    update={"source_repository": projected_card.artifact_repository}
                )
                expected = {
                    (file.path, file.size, file.sha256) for file in item.member.files
                }
                actual = {
                    (file.path, file.size_bytes, file.sha256) for file in existing.files
                }
                if not compatible_file_tables(actual, expected) or public_card_body(
                    old
                ) != public_card_body(new):
                    raise PreservationHostError("existing_model_conflict")
                matching[alias] = hashlib.sha256(
                    entry.model_dump_json().encode()
                ).hexdigest()
            payload["existing_companions"] = matching
            from skulk.store.model_store import StoreReconciliationTombstones

            tombstone_path = (
                self.store.store_path / ".skulk" / "reconciliation-tombstones.json"
            )
            tombstones = (
                StoreReconciliationTombstones.model_validate_json(
                    tombstone_path.read_bytes()
                ).deleted_aliases
                if tombstone_path.exists()
                else {}
            )
            payload["deletions"] = {
                alias: stamp for alias, stamp in tombstones.items() if alias in members
            }
            if (
                shutil.disk_usage(self.store.store_path).free
                < size * (2 if kit is not None else 1) + 32 * 1024 * 1024
            ):
                raise PreservationHostError("insufficient_space")
            payload["historical"] = request.historical
        if len(members) > 16:
            raise PreservationHostError("selection_needs_paged_review")
        expires = int(time.time()) + 600
        identity = hashlib.sha256(
            safe_json(
                {
                    "request": request.model_dump(mode="json"),
                    "payload": payload,
                    "expires": expires,
                }
            ).encode()
        ).hexdigest()
        plan = PreservationPlan(
            plan_id=identity,
            kind=request.kind,
            model_id=model_id,
            location_id=request.location_id,
            copy_id=copy_id,
            manifest_id=manifest,
            generation_id=request.generation_id,
            size_bytes=size,
            members=members[:16],
            member_count=len(members),
            expires_at=expires,
            existing_companions=tuple(
                sorted(
                    cast(dict[str, JsonValue], payload.get("existing_companions", {}))
                )
            ),
            deleted_aliases=tuple(
                sorted(cast(dict[str, JsonValue], payload.get("deletions", {})))
            ),
            upload_id=request.upload_id,
            historical=request.historical,
            publisher_key_ids=tuple(key.key_id for key in location.trust.keys),
            requires_owner_acceptance=bool(kit)
            or (inspection.requires_owner_acceptance if inspection else False),
            requires_code_permission=inspection.requires_code_permission
            if inspection
            else False,
            publication=inspection.publication if inspection else None,
            file_coverage=inspection.file_coverage if inspection else None,
        )
        if (
            len(json.dumps(plan.model_dump(mode="json"), ensure_ascii=True).encode())
            > 48000
        ):
            raise PreservationHostError("plan_summary_too_large")
        self.journal.add_plan(plan, payload)
        return plan

    async def _sharing_plan(self, request: PreservationPlanRequest) -> PreservationPlan:
        """Bind public-copy import or serving without parsing a runtime card."""
        from urllib.parse import urlsplit

        from foxlight_model_preservation.models import Manifest

        payload: dict[str, JsonValue] = {}
        policy = None
        if request.kind == "mirror_import":
            if request.upload_id is None or request.location_id != "public-kit":
                raise PreservationHostError("public_kit_selection_required")
            if not self.uploads.status(request.upload_id)["complete"]:
                raise PreservationHostError("kit_upload_incomplete")
            kit = inspect_kit(self.uploads.path(request.upload_id))
            manifest: Manifest = kit.manifest
            identity = kit.manifest_id
            info = self.uploads.root.stat()
            location = _Location(
                location_id="public-kit",
                purpose="public",
                label="Selected public bundle",
                path=self.uploads.root.absolute(),
                device=info.st_dev,
                inode=info.st_ino,
                trust=kit.trust,
                key_path=self.root / "unused-public-key",
            )
            token = hashlib.sha256(
                (identity + safe_json(kit.trust.model_dump(mode="json"))).encode()
            ).hexdigest()
            destination_id = "public-" + token
            destination = self.root / "public-copies" / token
            payload.update(
                {
                    "kit_path": str(self.uploads.path(request.upload_id).absolute()),
                    "destination": str(destination.absolute()),
                    "destination_id": destination_id,
                }
            )
            if (
                not destination.exists()
                and shutil.disk_usage(self.root).free
                < manifest.total_bytes + 32 * 1024 * 1024
            ):
                raise PreservationHostError("insufficient_space")
        else:
            if (
                request.kind == "torrent_share"
                and importlib.util.find_spec("libtorrent") is None
            ):
                raise PreservationHostError("bittorrent_unavailable")
            if request.copy_id is None:
                raise PreservationHostError("public_copy_required")
            copy = self.journal.copy(request.copy_id)
            if copy.audience != "public" or copy.location_id != request.location_id:
                raise PreservationHostError("public_copy_required")
            location = self._location(copy.location_id)
            if location.purpose != "public":
                raise PreservationHostError("public_copy_required")
            identity = copy.manifest_id
            async with self._archive(location) as descriptor:
                manifest = verify_manifest(
                    read_metadata(descriptor, manifest_path(identity)),
                    location.trust,
                    identity,
                )
            destination_id = None
            if (
                request.kind == "torrent_share"
                and shutil.disk_usage(self.root).free
                < manifest.total_bytes + 32 * 1024 * 1024
            ):
                raise PreservationHostError("insufficient_space")
            if request.advertised_url is not None:
                parsed = urlsplit(request.advertised_url)
                if (
                    parsed.scheme not in {"http", "https"}
                    or not parsed.hostname
                    or parsed.username
                    or parsed.password
                    or parsed.query
                    or parsed.fragment
                ):
                    raise PreservationHostError("invalid_mirror_hint")
            policy = SharingPolicy(
                public_network=request.public_network,
                listen_port=request.listen_port,
                duration_seconds=request.duration_seconds,
                upload_limit_bytes=request.upload_limit_bytes,
                upload_rate_bytes=request.upload_rate_bytes,
                advertised_url=request.advertised_url,
            )
        if any(member.intake.audience != "public" for member in manifest.members):
            raise PreservationHostError("public_bundle_required")
        expires = int(time.time()) + 600
        payload["location"] = cast(
            dict[str, JsonValue], location.model_dump(mode="json")
        )
        payload["manifest_id"] = identity
        plan_id = hashlib.sha256(
            safe_json(
                {
                    "request": request.model_dump(mode="json"),
                    "payload": payload,
                    "expires": expires,
                }
            ).encode()
        ).hexdigest()
        plan = PreservationPlan(
            plan_id=plan_id,
            kind=request.kind,
            model_id=manifest.name,
            location_id=request.location_id,
            destination_id=destination_id,
            copy_id=request.copy_id,
            manifest_id=identity,
            size_bytes=manifest.total_bytes,
            members=tuple(member.origin_repository for member in manifest.members[:16]),
            member_count=len(manifest.members),
            expires_at=expires,
            upload_id=request.upload_id,
            sharing=policy,
            publisher_key_ids=tuple(key.key_id for key in location.trust.keys),
            requires_owner_acceptance=request.kind == "mirror_import",
            publication="public-publisher-declaration",
        )
        self.journal.add_plan(plan, payload)
        return plan

    def start(self, request: PreservationStart) -> PreservationJob:
        """Reserve exact idempotency and return promptly; the host owns completion."""
        if self._closed:
            raise PreservationHostError("preservation_host_closed")
        job, created = self.journal.reserve(request)
        if created:
            self._tasks[job.job_id] = asyncio.create_task(self._execute(job.job_id))
        return job

    async def cancel(self, job_id: str) -> PreservationJob:
        """Persist stop intent; only an observed stopped process confirms cancellation."""
        job = self.journal.job(job_id)
        if job.state not in {"queued", "running", "cancelling"}:
            return job
        job = job.model_copy(
            update={"state": "cancelling", "updated_at": int(time.time())}
        )
        self.journal.update_job(job)
        process = self._processes.get(job_id)
        if process is not None and process.stdin is not None:
            with suppress(BrokenPipeError, ConnectionResetError):
                process.stdin.write(b"cancel\n")
                await process.stdin.drain()
            try:
                await asyncio.wait_for(asyncio.shield(process.wait()), 5)
            except TimeoutError:
                process.terminate()
                try:
                    await asyncio.wait_for(asyncio.shield(process.wait()), 2)
                except TimeoutError:
                    process.kill()
                    await process.wait()
        return self.journal.job(job_id)

    async def reclaim_recovery(self, job_id: str) -> RecoveryCleanup:
        """Explicitly free stopped recovery work, retaining all indexed models."""
        from skulk.store.preservation_reclaim import reclaim_recovery_files

        if self._closed:
            raise PreservationHostError("preservation_host_closed")
        async with self._queue:
            if self._closed:
                raise PreservationHostError("preservation_host_closed")
            job = self.journal.job(job_id)
            if (
                job.kind != "recover"
                or job.state not in {"interrupted", "failed", "cancelled"}
                or job_id in self._tasks
                or job_id in self._processes
            ):
                raise PreservationHostError("recovery_job_not_stopped")
            plan, payload = self.journal.plan(job.plan_id)
            if plan.manifest_id is None or (
                job.expected_manifest_id is not None
                and job.expected_manifest_id != plan.manifest_id
            ):
                raise PreservationHostError("recovery_manifest_changed")
            location = _Location.model_validate_json(safe_json(payload["location"]))
            async with self.store.preservation_lease():
                maintenance = asyncio.create_task(
                    asyncio.to_thread(
                        reclaim_recovery_files,
                        self.store,
                        self.journal,
                        job_id,
                        plan.manifest_id,
                        location.trust,
                    )
                )
                try:
                    return await asyncio.shield(maintenance)
                except asyncio.CancelledError:
                    # Disconnecting an HTTP caller cannot release the mutation
                    # fence while its filesystem maintenance thread still runs.
                    while not maintenance.done():
                        try:
                            await asyncio.shield(maintenance)
                        except asyncio.CancelledError:
                            continue
                    maintenance.result()
                    raise

    async def close(self) -> None:
        """Drain owned workers before releasing the exclusive host journal owner."""
        self._closed = True
        for job_id in tuple(self._tasks):
            await self.cancel(job_id)
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks.values()), return_exceptions=True)
        async with self._queue:
            self.journal.close()

    async def _execute(self, job_id: str) -> None:
        try:
            serving = self.journal.job(job_id).kind in {"http_share", "torrent_share"}
            gate = self._serving if serving else self._queue
            async with gate:
                job = self.journal.job(job_id)
                if job.state == "cancelling":
                    self._finish(job_id, "cancelled", "cancelled")
                    return
                self.journal.update_job(
                    job.model_copy(
                        update={
                            "state": "running",
                            "stage": "starting",
                            "attempts": job.attempts + 1,
                        }
                    )
                )
                if job.kind in {"preserve", "recover"}:
                    async with self.store.preservation_lease(
                        cancelled=lambda: self.journal.job(job_id).state == "cancelling"
                    ):
                        async with asyncio.timeout(86400):
                            await self._run(job_id)
                else:
                    async with asyncio.timeout(86400):
                        await self._run(job_id)
        except Exception as error:
            job = self.journal.job(job_id)
            if job.state in {"queued", "running", "cancelling"}:
                self._finish(
                    job_id,
                    "cancelled"
                    if job.kind in {"http_share", "torrent_share"}
                    and job.state == "cancelling"
                    else "failed"
                    if job.kind in {"http_share", "torrent_share"}
                    else "interrupted"
                    if job.expected_manifest_id
                    else "cancelled"
                    if job.state == "cancelling"
                    else "failed",
                    "cancelled"
                    if job.state == "cancelling"
                    else "deadline_exceeded"
                    if isinstance(error, TimeoutError)
                    else "worker_unavailable",
                )
        finally:
            self._tasks.pop(job_id, None)

    def _finish(
        self,
        job_id: str,
        state: Literal["failed", "cancelled", "interrupted"],
        code: str,
    ) -> None:
        job = self.journal.job(job_id)
        sharing = job.sharing
        if sharing is not None:
            sharing = sharing.model_copy(
                update={"ready": False, "listen_port": 0, "connected_peers": 0}
            )
        self.journal.update_job(
            job.model_copy(
                update={
                    "state": state,
                    "stage": "check_outcome" if state == "interrupted" else state,
                    "error_code": code,
                    "updated_at": int(time.time()),
                    "finished_at": int(time.time()),
                    "sharing": sharing,
                }
            )
        )

    async def _run(self, job_id: str) -> None:
        job = self.journal.job(job_id)
        if job.state == "cancelling":
            self._finish(job_id, "cancelled", "cancelled")
            return
        plan, payload = self.journal.plan(job.plan_id)
        location = _Location.model_validate_json(safe_json(payload["location"]))
        work = (
            self.store.store_path / ".recovery-work"
            if plan.kind == "recover"
            else self.root / "work"
        ) / job_id
        secure_directory(work)
        if plan.kind == "mirror_import":
            secure_directory(self.root / "public-copies")
        if plan.kind == "kit_export":
            secure_directory(self.root / "kits")
        if plan.kind == "torrent_share":
            secure_directory(self.root / "torrents")
        async with self._archive(location) as descriptor:
            model_path = (
                self._selected(plan.model_id) if plan.kind == "preserve" else None
            )
            if model_path is not None and preservation_generation(
                model_path, artifact_roots=(self.store.store_path,)
            ) != payload.get("generation_id"):
                raise PreservationHostError("installed_selection_changed")
            submitted_value = self.journal.setting("request:" + job_id)
            submitted = PreservationStart.model_validate_json(
                safe_json(submitted_value)
            )
            request = WorkerRequest(
                kind=cast(
                    Literal[
                        "preserve",
                        "check",
                        "kit_export",
                        "recover",
                        "mirror_import",
                        "http_share",
                        "torrent_share",
                    ],
                    plan.kind,
                ),
                archive_descriptor=descriptor,
                archive_device=location.device,
                archive_inode=location.inode,
                model_directory=model_path,
                source_root=self.store.store_path.absolute(),
                generation_id=cast(str | None, payload.get("generation_id")),
                snapshot_parent=work.absolute(),
                key_path=location.key_path,
                trust=location.trust,
                manifest_id=plan.manifest_id,
                output_path=(self.root / "kits" / (job_id + ".foxkit")).absolute()
                if plan.kind == "kit_export"
                else (work / "recovered").absolute()
                if plan.kind == "recover"
                else Path(cast(str, payload["destination"]))
                if plan.kind == "mirror_import"
                else None,
                kit_path=Path(cast(str, payload["kit_path"]))
                if "kit_path" in payload
                else None,
                historical=plan.historical,
                accept_owner_capture=submitted.accept_owner_capture,
                allow_repository_code=submitted.allow_repository_code,
                existing_companions=plan.existing_companions,
                sharing=plan.sharing,
                public_participation=submitted.public_participation,
                metainfo_path=(
                    self.root / "torrents" / (job_id + ".torrent")
                ).absolute()
                if plan.kind == "torrent_share"
                else None,
            )
            request_path = work / "request.json"
            fd = os.open(
                request_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                0o600,
            )
            with os.fdopen(fd, "w") as stream:
                _ = stream.write(request.model_dump_json())
                stream.flush()
                os.fsync(stream.fileno())
            process = await asyncio.create_subprocess_exec(
                *preservation_worker_command(request_path),
                pass_fds=(descriptor,),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                limit=32768,
            )
            self._processes[job_id] = process
            if (
                self.journal.job(job_id).state == "cancelling"
                and process.stdin is not None
            ):
                await self.cancel(job_id)
            result: _Observation | None = None
            error_code = "worker_interrupted"
            try:
                if process.stdout is None or process.stdin is None:
                    raise PreservationHostError("worker_control_unavailable")
                while data := await process.stdout.readline():
                    if len(data) > 16384:
                        raise PreservationHostError("worker_observation_too_large")
                    observation = _Observation.model_validate_json(data)
                    current = self.journal.job(job_id)
                    if observation.event == "sharing":
                        if (
                            plan.kind not in {"http_share", "torrent_share"}
                            or observation.sharing is None
                        ):
                            raise PreservationHostError(
                                "unexpected_serving_observation"
                            )
                        if (
                            current.sharing is not None
                            and observation.sharing.charged_bytes
                            < current.sharing.charged_bytes
                        ):
                            raise PreservationHostError("serving_observation_regressed")
                        self.journal.update_job(
                            current.model_copy(
                                update={
                                    "sharing": observation.sharing,
                                    "stage": "sharing"
                                    if observation.sharing.ready
                                    else "checking_public_files",
                                    "updated_at": int(time.time()),
                                }
                            )
                        )
                    elif observation.event == "expected":
                        if observation.manifest_id is None:
                            raise PreservationHostError("invalid_expected_outcome")
                        self.journal.update_job(
                            current.model_copy(
                                update={
                                    "expected_manifest_id": observation.manifest_id,
                                    "updated_at": int(time.time()),
                                }
                            )
                        )
                        process.stdin.write(
                            ("publish:" + observation.manifest_id + "\n").encode()
                        )
                        await process.stdin.drain()
                    elif observation.event == "progress":
                        self.journal.update_job(
                            current.model_copy(
                                update={
                                    "stage": observation.stage or "working",
                                    "updated_at": int(time.time()),
                                    "bytes_hashed": max(
                                        current.bytes_hashed, observation.bytes_hashed
                                    ),
                                    "bytes_copied": max(
                                        current.bytes_copied, observation.bytes_copied
                                    ),
                                }
                            )
                        )
                    elif observation.event == "result":
                        result = observation
                    else:
                        error_code = observation.error_code or "operation_failed"
                exit_code = await process.wait()
                current = self.journal.job(job_id)
                if (
                    exit_code == 0
                    and result is not None
                    and result.manifest_id is not None
                ):
                    expected = (
                        current.expected_manifest_id
                        if plan.kind in {"preserve", "kit_export", "recover"}
                        else plan.manifest_id
                    )
                    if (
                        result.manifest_id != expected
                        or result.size_bytes is None
                        or result.member_count is None
                        or (
                            result.audience
                            != (
                                "public"
                                if location.purpose == "public"
                                or plan.kind == "mirror_import"
                                else "private"
                            )
                            and plan.kind != "recover"
                        )
                    ):
                        raise PreservationHostError("worker_outcome_mismatch")
                    if plan.kind in {"mirror_import", "http_share", "torrent_share"}:
                        now = int(time.time())
                        copy_id = plan.copy_id
                        if plan.kind == "mirror_import":
                            destination = Path(cast(str, payload["destination"]))
                            info = destination.stat()
                            enrolled = _Location(
                                location_id=cast(str, payload["destination_id"]),
                                purpose="public",
                                label="Public bundle copy",
                                path=destination,
                                device=info.st_dev,
                                inode=info.st_ino,
                                trust=location.trust,
                                key_path=self.root / "unused-public-key",
                            )
                            self.journal.set_setting(
                                "location:" + enrolled.location_id,
                                cast(
                                    dict[str, JsonValue],
                                    enrolled.model_dump(mode="json"),
                                ),
                            )
                            copy_id = hashlib.sha256(
                                (enrolled.location_id + result.manifest_id).encode()
                            ).hexdigest()
                            self.journal.add_copy(
                                PreservationCopy(
                                    copy_id=copy_id,
                                    model_id=plan.model_id,
                                    location_id=enrolled.location_id,
                                    manifest_id=result.manifest_id,
                                    audience="public",
                                    size_bytes=result.size_bytes,
                                    created_at=now,
                                    checked_at=now,
                                    member_count=result.member_count,
                                    publisher_key_ids=tuple(
                                        key.key_id for key in location.trust.keys
                                    ),
                                )
                            )
                        self.journal.update_job(
                            current.model_copy(
                                update={
                                    "state": "cancelled"
                                    if current.state == "cancelling"
                                    else "succeeded",
                                    "stage": "stopped"
                                    if plan.kind != "mirror_import"
                                    else "complete",
                                    "copy_id": copy_id,
                                    "updated_at": now,
                                    "finished_at": now,
                                    "result": {
                                        "manifest_id": result.manifest_id,
                                        "copy_id": copy_id,
                                        "public_copy": True,
                                    },
                                }
                            )
                        )
                        return
                    if plan.kind in {"kit_export", "recover"}:
                        if current.state == "cancelling" and plan.kind == "recover":
                            self._finish(
                                job_id,
                                "interrupted",
                                "cancelled_before_store_publication",
                            )
                            return
                        if plan.kind == "recover":
                            await publish_recovered_home(
                                self.store,
                                work / "recovered",
                                job_id,
                                self.custom_cards,
                                manifest_id=result.manifest_id,
                                trust=location.trust,
                                mode="historical" if plan.historical else "fresh",
                                deletions=TypeAdapter(dict[str, str]).validate_python(
                                    payload.get("deletions", {})
                                ),
                                existing_companions=TypeAdapter(
                                    dict[str, str]
                                ).validate_python(
                                    payload.get("existing_companions", {})
                                ),
                            )
                        now = int(time.time())
                        self.journal.update_job(
                            current.model_copy(
                                update={
                                    "state": "succeeded",
                                    "stage": "complete",
                                    "updated_at": now,
                                    "finished_at": now,
                                    "result": {
                                        "manifest_id": result.manifest_id,
                                        "restored": plan.kind == "recover",
                                        "kit_ready": plan.kind == "kit_export",
                                        "member_count": result.member_count,
                                    },
                                }
                            )
                        )
                        return
                    now = int(time.time())
                    copy_id = hashlib.sha256(
                        (plan.location_id + result.manifest_id).encode()
                    ).hexdigest()
                    self.journal.add_copy(
                        PreservationCopy(
                            copy_id=copy_id,
                            model_id=plan.model_id,
                            location_id=plan.location_id,
                            manifest_id=result.manifest_id,
                            audience=location.purpose,
                            publisher_key_ids=tuple(
                                key.key_id for key in location.trust.keys
                            )
                            if location.purpose == "public"
                            else (),
                            size_bytes=result.size_bytes,
                            created_at=now,
                            checked_at=now,
                            member_count=result.member_count,
                        )
                    )
                    if plan.kind == "check" and plan.copy_id is not None:
                        self.journal.confirm_outcome(plan.copy_id, copy_id)
                    self.journal.update_job(
                        current.model_copy(
                            update={
                                "state": "succeeded",
                                "stage": "complete",
                                "copy_id": copy_id,
                                "updated_at": now,
                                "finished_at": now,
                                "result": {
                                    "manifest_id": result.manifest_id,
                                    "copy_id": copy_id,
                                },
                            }
                        )
                    )
                else:
                    state = (
                        "cancelled"
                        if plan.kind in {"http_share", "torrent_share"}
                        and (current.state == "cancelling" or error_code == "cancelled")
                        else "failed"
                        if plan.kind in {"http_share", "torrent_share"}
                        else "interrupted"
                        if current.expected_manifest_id is not None
                        else "cancelled"
                        if current.state == "cancelling" or error_code == "cancelled"
                        else "failed"
                    )
                    self._finish(job_id, state, error_code)
            finally:
                if process.stdin is not None:
                    process.stdin.close()
                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), 2)
                    except TimeoutError:
                        process.kill()
                        await process.wait()
                self._processes.pop(job_id, None)
                request_path.unlink(missing_ok=True)
                shutil.rmtree(work)
