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
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from pathlib import Path
from typing import Literal, cast, final

from foxlight_model_preservation.archives import verify_archive_manifest
from foxlight_model_preservation.handoffs import (
    Handoff,
    read_handoff,
    reference_contacts,
)
from foxlight_model_preservation.invitations import (
    invitation_from_envelope,
)
from foxlight_model_preservation.kits import inspect_kit
from foxlight_model_preservation.mirror_models import Terms
from foxlight_model_preservation.models import Manifest, PreservationError, TrustProfile
from foxlight_model_preservation.peer_invitations import PeerInvitation
from foxlight_model_preservation.pmf_models import PMFManifest
from foxlight_model_preservation.publication import (
    PublicationRequest,
    review_publication,
)
from foxlight_model_preservation.signing import (
    canonical_bytes,
    create_key,
    digest,
    encode_record,
    strict_json,
)
from foxlight_model_preservation.storage import (
    atomic_bytes,
    directory,
    manifest_path,
    read_metadata,
    unique_objects,
)
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from skulk.shared.constants import SKULK_CUSTOM_MODEL_CARDS_DIR
from skulk.store.discovery_host import DiscoveryHost
from skulk.store.installed_cards import read_installed_card
from skulk.store.mirror_host import MirrorHost
from skulk.store.mirror_worker import MirrorRetainSource
from skulk.store.model_store import ModelStore
from skulk.store.peer_host import PeerHost, PeerLease
from skulk.store.preservation import inspect_verified_manifest
from skulk.store.preservation_contracts import (
    FolderConfirmation,
    FolderReview,
    FolderReviewRequest,
    HandoffRequest,
    InvitationPreview,
    InvitationRequest,
    KitUploadStatus,
    PreservationCopy,
    PreservationJob,
    PreservationLocation,
    PreservationModel,
    PreservationPage,
    PreservationPlan,
    PreservationPlanRequest,
    PreservationStart,
    PublicationSource,
    ReceivingPolicy,
    RecoveryCleanup,
    SharingPolicy,
    SharingStatus,
    TemporaryWork,
)
from skulk.store.preservation_delivery import invitation_preview, publication_members
from skulk.store.preservation_export import plan_preservation, preservation_generation
from skulk.store.preservation_folders import (
    FolderObservation,
    confirmed_folder,
    observe_folder,
    private_folder,
)
from skulk.store.preservation_io import PreservationIO
from skulk.store.preservation_journal import (
    PreservationHostError,
    PreservationJournal,
    safe_json,
    secure_directory,
)
from skulk.store.preservation_live import publish_recovered_home
from skulk.store.preservation_pmf import read_capture, read_kit_capture
from skulk.store.preservation_references import (
    accept_model_reference_keys as accept_invitation_keys,
)
from skulk.store.preservation_references import (
    authenticate_model_reference,
    i2p_model_mirrors,
)
from skulk.store.preservation_references import read_model_reference as read_invitation
from skulk.store.preservation_sharing import check_peer_space, require_public_bundle
from skulk.store.preservation_uploads import KitUploads
from skulk.store.preservation_work import (
    WORK_KINDS,
    manage_temporary_work,
    register_temporary_work,
    temporary_work_pending,
)
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
    requires_trust_snapshot: bool = False


class _Observation(BaseModel):
    """Bounded safe protocol from an exclusively owned known byte worker."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    event: Literal["progress", "expected", "result", "error", "sharing"]
    sharing: SharingStatus | None = None
    stage: str | None = None
    bytes_hashed: int = 0
    bytes_copied: int = 0
    bytes_received: int = 0
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
        self._closing: asyncio.Task[None] | None = None
        self._setup_lock = asyncio.Lock()
        self._storage = PreservationIO()
        self.uploads = KitUploads(root / "uploads", self.journal)
        self.upload_lock = asyncio.Lock()
        self.peers = PeerHost(
            root, self.journal, self._storage, preservation_worker_command
        )
        self.mirrors = MirrorHost(
            root,
            self.journal,
            self._storage,
            preservation_worker_command,
            self._mirror_source,
            self.peers.mirror_runtime,
        )
        self.discovery = DiscoveryHost(
            root,
            self.journal,
            self._storage,
            self.uploads,
            self.upload_lock,
            self._metadata_upload,
            preservation_worker_command,
            self.peers,
        )
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

    @asynccontextmanager
    async def _mirror_source(
        self, copy_id: str, identity: str
    ) -> AsyncIterator[MirrorRetainSource]:
        copy = self.journal.copy(copy_id)
        location = self._location(copy.location_id)
        if (
            copy.audience != "public"
            or location.purpose != "public"
            or copy.manifest_id != identity
        ):
            raise PreservationHostError("public_copy_required")
        async with self._archive(location) as descriptor:

            def inspect() -> MirrorRetainSource:
                manifest = verify_archive_manifest(
                    read_metadata(descriptor, manifest_path(identity)),
                    location.trust,
                    identity,
                )
                if not isinstance(manifest, Manifest):
                    raise PreservationHostError(
                        "public_mirror_pmf_transport_unavailable"
                    )
                require_public_bundle(manifest)
                pairs = sorted(
                    {
                        (member.intake.license_id, member.intake.terms_reference)
                        for member in manifest.members
                    }
                )
                terms = tuple(
                    Terms(license_id=license_id, terms_reference=reference)
                    for license_id, reference in pairs
                )
                info = os.fstat(descriptor)
                return MirrorRetainSource(
                    descriptor=descriptor,
                    device=info.st_dev,
                    inode=info.st_ino,
                    trust=location.trust,
                    terms=terms,
                )

            yield await self._storage.run(inspect)

    def _location(self, location_id: str) -> _Location:
        value = self.journal.setting("location:" + location_id)
        if value is None:
            raise PreservationHostError("archive_not_configured")
        return _Location.model_validate_json(safe_json(value))

    def _open_archive(self, location: _Location) -> int:
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
            if location.requires_trust_snapshot:
                private_folder(descriptor)
                try:
                    snapshot = read_metadata(descriptor, "archive-trust.json")
                except (OSError, ValueError) as error:
                    raise PreservationHostError("storage_setup_incomplete") from error
                if snapshot != encode_record(location.trust):
                    raise PreservationHostError("storage_setup_incomplete")
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    @asynccontextmanager
    async def _archive(self, location: _Location) -> AsyncIterator[int]:
        async with self._storage.directory(
            lambda: self._open_archive(location)
        ) as descriptor:
            yield descriptor

    async def _manifest(self, location: _Location, identity: str) -> Manifest:
        async with self._archive(location) as descriptor:
            return await self._storage.run(
                lambda: read_capture(descriptor, identity, location.trust)
            )

    async def _format(
        self, location: _Location, identity: str
    ) -> Literal["preservation-v1", "pmf-v0.1"]:
        async with self._archive(location) as descriptor:
            manifest = await self._storage.run(
                lambda: verify_archive_manifest(
                    read_metadata(descriptor, manifest_path(identity)),
                    location.trust,
                    identity,
                )
            )
        return "pmf-v0.1" if isinstance(manifest, PMFManifest) else "preservation-v1"

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

    async def review_folder(self, request: FolderReviewRequest) -> FolderReview:
        """Review a host folder off-loop, retaining setup ownership until it finishes."""
        async with self._setup_lock:
            return await self._storage.run(lambda: self._review_folder(request))

    def _review_folder(self, request: FolderReviewRequest) -> FolderReview:
        if self._closed:
            raise PreservationHostError("preservation_host_closed")
        selected = Path(request.path)
        path = selected.parent.resolve(strict=True) / selected.name
        enrolled = tuple(
            self._location(identity) for identity in self.journal.private_location_ids()
        )
        known = next((location for location in enrolled if location.path == path), None)
        observation = observe_folder(
            request,
            forbidden=(self.root, self.store.store_path, self.custom_cards),
            enrolled_id=None if known is None else known.location_id,
        )
        if known is not None and (observation.device, observation.inode) != (
            known.device,
            known.inode,
        ):
            raise PreservationHostError("archive_identity_changed")
        self.journal.set_setting("folder-review", observation.model_dump(mode="json"))
        return observation.view

    async def setup_folder(self, request: FolderConfirmation) -> PreservationLocation:
        """Confirm exact private storage; keys stay in protected Core custody.

        Enrollment precedes the public-trust write, so an interrupted metadata
        write can be explicitly repaired only on the recorded root. Such roots
        remain unavailable to byte work until that snapshot is complete. Default
        selection occurs last; incomplete setup never changes the old default.
        """
        if not request.confirmed:
            raise PreservationHostError("storage_confirmation_required")
        if self.journal.setting("folder-review") is None:
            raise PreservationHostError("storage_review_expired")
        observation = FolderObservation.model_validate_json(
            safe_json(self.journal.setting("folder-review"))
        )
        if (
            observation.view.review_id != request.review_id
            or observation.view.expires_at <= int(time.time())
        ):
            raise PreservationHostError("storage_review_expired")
        await self.setup_device()
        async with self._setup_lock:
            return await self._storage.run(
                lambda: self._setup_folder(request, observation)
            )

    def _setup_folder(
        self, request: FolderConfirmation, observation: FolderObservation
    ) -> PreservationLocation:
        if self._closed:
            raise PreservationHostError("preservation_host_closed")
        current = FolderObservation.model_validate_json(
            safe_json(self.journal.setting("folder-review"))
        )
        if current != observation or observation.view.expires_at <= int(time.time()):
            raise PreservationHostError("storage_review_expired")
        identity = observation.enrolled_id or "folder_" + request.review_id[:32]
        recorded = self.journal.setting("location:" + identity)
        if recorded is not None:
            enrolled = self._location(identity)
            observation = observation.model_copy(
                update={
                    "device": enrolled.device,
                    "inode": enrolled.inode,
                    "enrolled_id": identity,
                }
            )
        elif len(self.journal.private_location_ids()) >= 16:
            raise PreservationHostError("storage_location_limit")
        with confirmed_folder(observation) as descriptor:
            info = os.fstat(descriptor)
            capture = self._location("device")
            location = _Location(
                location_id=identity,
                label=observation.view.label,
                path=observation.path,
                device=info.st_dev,
                inode=info.st_ino,
                trust=capture.trust,
                key_path=capture.key_path,
                requires_trust_snapshot=True,
            )
            self.journal.set_setting(
                "location:" + identity, location.model_dump(mode="json")
            )
            atomic_bytes(
                descriptor, "archive-trust.json", encode_record(location.trust)
            )
        descriptor = self._open_archive(location)
        try:
            self.journal.set_setting("default-private-location", identity)
        finally:
            os.close(descriptor)
        return self._observe_location(location)

    async def default_location(self, location_id: str) -> PreservationLocation:
        """Persist available private storage while retaining setup ownership."""
        async with self._setup_lock:
            return await self._storage.run(lambda: self._set_default(location_id))

    def _set_default(self, location_id: str) -> PreservationLocation:
        if self._closed:
            raise PreservationHostError("preservation_host_closed")
        location = self._location(location_id)
        if location.purpose != "private":
            raise PreservationHostError("private_archive_required")
        descriptor = self._open_archive(location)
        try:
            self.journal.set_setting("default-private-location", location_id)
        finally:
            os.close(descriptor)
        return self._observe_location(location)

    def _public_copy_parent(self) -> os.stat_result:
        parent = self.root / "public-copies"
        recorded = self.journal.setting("public-copy-parent")
        if recorded is None:
            secure_directory(parent)
        info = parent.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
        ):
            raise PreservationHostError("public_destination_changed")
        identity: dict[str, JsonValue] = {"device": info.st_dev, "inode": info.st_ino}
        if recorded is not None and safe_json(recorded) != safe_json(identity):
            raise PreservationHostError("public_destination_changed")
        if self.journal.setting("location:publisher") is not None:
            publisher = self._location("publisher")
            if (info.st_dev, info.st_ino) != (publisher.device, publisher.inode):
                raise PreservationHostError("public_destination_changed")
        if recorded is None:
            self.journal.set_setting("public-copy-parent", identity)
        return info

    async def setup_publisher(self) -> PreservationLocation:
        """Explicitly create distinct protected publishing custody, without serving.

        Repeats observe the enrolled directory/key without replacing either.
        This identity is separate from private Preserve destinations and never
        grants redistribution rights, registry authority or public reachability.
        """
        async with self._setup_lock:
            if self._closed:
                raise PreservationHostError("preservation_host_closed")
            if self.journal.setting("location:publisher") is None:
                archive = self.root / "public-copies"
                custody = self.root / "custody"
                self._public_copy_parent()
                secure_directory(custody)
                key = custody / "publisher.pem"
                if key.exists() or key.is_symlink():
                    raise PreservationHostError(
                        "publisher_identity_needs_reconciliation"
                    )
                trust = create_key(key, None)
                info = self._public_copy_parent()
                location = _Location(
                    location_id="publisher",
                    purpose="public",
                    label="Public publishing",
                    path=archive.absolute(),
                    device=info.st_dev,
                    inode=info.st_ino,
                    trust=trust,
                    key_path=key.absolute(),
                )
                self.journal.set_setting(
                    "location:publisher", location.model_dump(mode="json")
                )
            return await self.location_view("publisher")

    async def _copy_metadata(
        self, copy_id: str
    ) -> tuple[PreservationCopy, _Location, Manifest]:
        copy = self.journal.copy(copy_id)
        location = self._location(copy.location_id)
        manifest = await self._manifest(location, copy.manifest_id)
        return copy, location, manifest

    async def publication_source(self, copy_id: str) -> PublicationSource:
        """Read signed member/terms hints for a kept private copy, without hashing.

        No installed model or online registry is needed. Known private capture
        bookkeeping is omitted; no notice text is invented and no rights granted.
        """
        copy, location, manifest = await self._copy_metadata(copy_id)
        if copy.audience != "private" or location.purpose != "private":
            raise PreservationHostError("private_copy_required")
        source = PublicationSource(
            copy_id=copy_id,
            manifest_id=copy.manifest_id,
            name=manifest.name,
            workloads=manifest.workloads,
            runtime_requirements=manifest.runtime_requirements,
            members=publication_members(manifest),
        )
        self._bounded_view(source)
        return source

    def _metadata_upload(
        self, upload_id: str, purpose: Literal["invitation", "publication", "source"]
    ) -> bytes:
        status = self.uploads.status(upload_id)
        if not status["complete"] or status["purpose"] != purpose:
            raise PreservationHostError("complete_metadata_upload_required")
        if cast(int, status["size_bytes"]) > 8 * 1024 * 1024:
            raise PreservationHostError("metadata_size_out_of_range")
        with directory(self.uploads.root) as root:
            encoded = read_metadata(root, upload_id)
        if len(encoded) != status["size_bytes"]:
            raise PreservationHostError("metadata_upload_changed")
        return encoded

    @staticmethod
    def _bounded_view(value: BaseModel) -> None:
        if (
            len(json.dumps(value.model_dump(mode="json"), ensure_ascii=True).encode())
            > 48000
        ):
            raise PreservationHostError("metadata_preview_too_large")

    def inspect_invitation(self, upload_id: str) -> InvitationPreview:
        """Inspect a completed invitation offline, without trusting its publisher.

        Reads at most 8 MiB from host-created binary ingress. The original signed
        closure and candidate fingerprints are shown without DNS, model parsing,
        private path reads, key enrollment, admission or sharing.
        """
        data = self._metadata_upload(upload_id, "invitation")
        preview = invitation_preview(read_invitation(data)).model_copy(
            update={"contacts": reference_contacts(data)}
        )
        self._bounded_view(preview)
        return preview

    def receive_invitation(self, job_id: str) -> InvitationPreview:
        """Inspect a receive job's retained original metadata after upload loss."""
        job = self.journal.job(job_id)
        if job.kind != "receive":
            raise PreservationHostError("receive_job_required")
        plan, payload = self.journal.plan(job.plan_id)
        invitation = read_invitation(safe_json(payload["invitation"]).encode())
        preview = invitation_preview(invitation).model_copy(
            update={"receiving": plan.receiving}
        )
        self._bounded_view(preview)
        return preview

    async def public_invitation(
        self, copy_id: str, request: InvitationRequest
    ) -> bytes:
        """Export original signed public metadata with explicitly selected hints.

        The returned portable bytes include public candidate keys only. This
        performs no network/reachability test and starts no listener or swarm.
        """
        copy, location, _ = await self._copy_metadata(copy_id)
        if copy.audience != "public" or location.purpose != "public":
            raise PreservationHostError("public_copy_required")
        async with self._archive(location) as descriptor:
            envelope = await self._storage.run(
                lambda: read_metadata(descriptor, manifest_path(copy.manifest_id))
            )
        return encode_record(
            invitation_from_envelope(
                envelope, copy.manifest_id, location.trust, request.mirrors
            )
        )

    async def public_handoff(self, copy_id: str, request: HandoffRequest) -> bytes:
        """Export one unchanged signed copy with reviewed HTTP or owned peer hints.

        Peer files are the exact bounded, hash-checked output of this copy's
        known sharing job. Stopped shares remain exportable; reading starts no
        network, asserts no reachability and never enrolls trust or deletes data.
        """
        if bool(request.mirrors) == (request.sharing_job_id is not None):
            raise PreservationHostError("handoff_route_selection_required")
        if request.sharing_job_id is None:
            data = await self.public_invitation(
                copy_id, InvitationRequest(mirrors=request.mirrors)
            )
            return encode_record(Handoff(reference=read_invitation(data)))
        copy, location, _ = await self._copy_metadata(copy_id)
        job = self.journal.job(request.sharing_job_id)
        plan, _ = self.journal.plan(job.plan_id)
        if (
            copy.audience != "public"
            or location.purpose != "public"
            or job.kind != "peer_share"
            or plan.copy_id != copy_id
            or job.sharing is None
            or job.sharing.handoff_sha256 is None
        ):
            raise PreservationHostError("handoff_not_ready")
        expected = job.sharing.handoff_sha256
        path = self.root / "handoffs" / (job.job_id + ".foxhandoff")

        def read() -> bytes:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as stream:
                info = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(info.st_mode)
                    or not 0 < info.st_size <= 8 * 1024**2
                ):
                    raise PreservationHostError("handoff_metadata_changed")
                data = stream.read(8 * 1024**2 + 1)
            if digest(data) != expected:
                raise PreservationHostError("handoff_metadata_changed")
            reference = read_handoff(data).reference
            if (
                not isinstance(reference, PeerInvitation)
                or reference.manifest_id != copy.manifest_id
            ):
                raise PreservationHostError("handoff_metadata_changed")
            authenticate_model_reference(reference, location.trust)
            return data

        return await self._storage.run(read)

    async def public_manifest(self, plan_id: str) -> bytes:
        """Download exact unsigned publication review metadata, without weight IO.

        Source metadata is authenticated again; no weights, private operational
        evidence, signing key or host location is included in this public preview.
        """
        plan, payload = self.journal.plan(plan_id)
        if plan.kind != "publish" or plan.copy_id is None:
            raise PreservationHostError("publication_plan_required")
        _, _, source = await self._copy_metadata(plan.copy_id)
        request = PublicationRequest.model_validate_json(
            safe_json(payload["publication_request"])
        )
        manifest = review_publication(source, request)
        encoded = encode_record(manifest)
        if digest(encoded) != plan.manifest_id:
            raise PreservationHostError("public_review_changed")
        return encoded

    @staticmethod
    def _location_result(
        location: _Location, available: bool, free: int | None
    ) -> PreservationLocation:
        return PreservationLocation(
            location_id=location.location_id,
            label=location.label,
            purpose=location.purpose,
            reachable=available,
            free_bytes=free,
            publisher_key_id=location.trust.keys[0].key_id,
        )

    def _observe_location(self, location: _Location) -> PreservationLocation:
        try:
            descriptor = self._open_archive(location)
            try:
                free = shutil.disk_usage(descriptor).free
            finally:
                os.close(descriptor)
        except (OSError, PreservationHostError):
            return self._location_result(location, False, None)
        return self._location_result(location, True, free)

    async def location_view(self, location_id: str) -> PreservationLocation:
        """Bound availability reads without recreating paths or queueing stalled probes."""
        location = self._location(location_id)
        return await self._storage.probe(
            location_id,
            lambda: self._observe_location(location),
            self._location_result(location, False, None),
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
        locations = tuple(
            [
                await self.location_view(identity)
                for identity in self.journal.private_location_ids()
            ]
        )
        selected = self.journal.setting("default-private-location")
        default_id = (
            selected
            if isinstance(selected, str)
            else "device"
            if any(location.location_id == "device" for location in locations)
            else None
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
            default_location_id=default_id,
            publishing=await self.location_view("publisher")
            if self.journal.setting("location:publisher") is not None
            else None,
            bittorrent_available=importlib.util.find_spec("libtorrent") is not None,
            uploads=tuple(uploads),
            temporary_work_jobs=tuple(
                job.job_id for job in jobs if temporary_work_pending(self.root, job)
            ),
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
        if request.resume_job_id is not None and request.kind != "receive":
            raise PreservationHostError("receive_resume_required")
        if request.kind == "publish":
            return await self._publication_plan(request)
        if request.kind == "receive":
            return await self._receiving_plan(request)
        if request.kind in {
            "mirror_import",
            "http_share",
            "torrent_share",
            "peer_share",
        }:
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
        archive_format: Literal["preservation-v1", "pmf-v0.1"] = (
            "pmf-v0.1" if request.kind == "preserve" else "preservation-v1"
        )
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
                metadata = await self._manifest(location, manifest)
                archive_format = await self._format(location, manifest)
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
                # A durable intent can precede the envelope. Outcome probing
                # needs the recorded format even when no completion exists.
                archive_format = prior_plan.archive_format
            payload["manifest_id"] = manifest
        if kit is not None:
            archive_format = (
                "pmf-v0.1"
                if isinstance(kit.manifest, PMFManifest)
                else "preservation-v1"
            )
        payload["archive_format"] = archive_format
        inspection = None
        if request.kind == "recover":
            if manifest is None:
                raise PreservationHostError("selected_copy_required")
            if kit is not None:
                signed = await self._storage.run(
                    lambda: read_kit_capture(
                        self.uploads.path(cast(str, request.upload_id)),
                        kit,
                        location.trust,
                    )
                )
            else:
                signed = await self._manifest(location, manifest)
            inspection, projected_card = inspect_verified_manifest(
                signed, manifest, "historical" if request.historical else "fresh"
            )
            model_id, size, members = (
                inspection.model_id,
                max(size, inspection.total_bytes),
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
            archive_format=archive_format,
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

    def _delivery_review(
        self,
        request: PreservationPlanRequest,
        payload: dict[str, JsonValue],
        manifest: Manifest,
        trust: TrustProfile,
        model_id: str,
    ) -> PreservationPlan:
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
            copy_id=request.copy_id,
            destination_id=cast(str, payload["destination_id"]),
            manifest_id=cast(str, payload["manifest_id"]),
            size_bytes=manifest.total_bytes,
            members=tuple(
                f"{m.role}: {m.origin_repository}" for m in manifest.members[:16]
            ),
            member_count=len(manifest.members),
            expires_at=expires,
            publisher_key_ids=tuple(key.key_id for key in trust.keys),
            requires_owner_acceptance=request.kind == "receive",
            publication="public-publisher-declaration",
            upload_id=request.upload_id,
            receiving=ReceivingPolicy.model_validate_json(
                safe_json(payload["receiving"])
            )
            if request.kind == "receive"
            else None,
            resume_job_id=request.resume_job_id,
        )
        self._bounded_view(plan)
        self.journal.add_plan(plan, payload)
        return plan

    @staticmethod
    def _destination_token(manifest_id: str, trust: TrustProfile) -> str:
        return hashlib.sha256(
            (manifest_id + safe_json(trust.model_dump(mode="json"))).encode()
        ).hexdigest()

    async def _publication_plan(
        self, request: PreservationPlanRequest
    ) -> PreservationPlan:
        if request.copy_id is None or request.destination_id not in {None, "publisher"}:
            raise PreservationHostError("private_publication_selection_required")
        copy, location, source = await self._copy_metadata(request.copy_id)
        if (
            copy.location_id != request.location_id
            or copy.audience != "private"
            or location.purpose != "private"
        ):
            raise PreservationHostError("private_copy_required")
        publisher = self._location("publisher")
        async with self._archive(publisher):
            pass
        if request.upload_id is not None:
            if request.public_request is not None:
                raise PreservationHostError("one_publication_request_required")
            encoded = self._metadata_upload(request.upload_id, "publication")
        elif request.public_request is not None:
            encoded = safe_json(request.public_request).encode()
        else:
            raise PreservationHostError("public_terms_required")
        selected = PublicationRequest.model_validate_json(
            canonical_bytes(strict_json(encoded))
        )
        public = review_publication(source, selected)
        publication_members(public)
        identity = digest(encode_record(public))
        token = self._destination_token(identity, publisher.trust)
        destination = publisher.path / token
        if not destination.exists():
            try:
                key_info = publisher.key_path.lstat()
            except FileNotFoundError as error:
                raise PreservationHostError(
                    "publisher_signing_key_unavailable"
                ) from error
            if (
                not stat.S_ISREG(key_info.st_mode)
                or key_info.st_uid != os.getuid()
                or key_info.st_mode & 0o077
            ):
                raise PreservationHostError("publisher_signing_key_unavailable")
        if (
            not destination.exists()
            and shutil.disk_usage(publisher.path).free
            < public.total_bytes + 32 * 1024 * 1024
        ):
            raise PreservationHostError("insufficient_space")
        if not destination.exists():
            # The PMF bridge and public output coexist in private job work.
            # Logical totals conservatively cover deduplicated object stores.
            conversion_bytes = (
                source.total_bytes
                if await self._format(location, copy.manifest_id) == "pmf-v0.1"
                else 0
            )
            if (
                shutil.disk_usage(self.root).free
                < conversion_bytes + public.total_bytes + 32 * 1024 * 1024
            ):
                raise PreservationHostError("insufficient_space")
        payload: dict[str, JsonValue] = {
            "location": location.model_dump(mode="json"),
            "publisher": publisher.model_dump(mode="json"),
            "source_manifest_id": copy.manifest_id,
            "manifest_id": identity,
            "publication_request": selected.model_dump(mode="json"),
            "destination": str(destination.absolute()),
            "destination_id": "public-" + token,
            "output_trust": publisher.trust.model_dump(mode="json"),
        }
        return self._delivery_review(
            request, payload, public, publisher.trust, copy.model_id
        )

    def _peer_receiving_setup(
        self, request: PreservationPlanRequest, scope: bool | None = None
    ) -> tuple[str, bool]:
        if (
            not request.allow_peer_network
            or request.peer_setup_id is None
            or request.allow_http
            or request.allow_private
        ):
            raise PreservationHostError("peer_network_confirmation_required")
        setup, identity = self.peers.selected()
        if identity != request.peer_setup_id:
            raise PreservationHostError("peer_setup_changed")
        if scope is not None and setup.allow_private != scope:
            raise PreservationHostError("peer_scope_changed_new_receive_required")
        return identity, setup.allow_private

    async def _receiving_plan(
        self, request: PreservationPlanRequest
    ) -> PreservationPlan:
        if request.location_id != "invitation":
            raise PreservationHostError("invitation_selection_required")
        if request.resume_job_id is not None:
            previous = self.journal.job(request.resume_job_id)
            if previous.kind != "receive" or previous.state not in {
                "failed",
                "cancelled",
                "interrupted",
            }:
                raise PreservationHostError("stopped_receive_job_required")
            original, payload = self.journal.plan(previous.plan_id)
            payload = {**payload, "resume_job_id": previous.job_id}
            invitation = read_invitation(safe_json(payload["invitation"]).encode())
            location = _Location.model_validate_json(safe_json(payload["location"]))
            # A resume is another review of unchanged execution inputs. Changed
            # sources/limits/trust require a new deliberate download, never a
            # quiet refill of the original operation's remaining allowance.
            if (
                request.upload_id is not None
                or request.accepted_key_ids
                or request.public_request is not None
            ):
                raise PreservationHostError("receive_resume_inputs_fixed")
            if original.receiving is None:
                raise PreservationHostError("receive_policy_missing")
            policy = original.receiving
            if policy.transport == "i2p":
                if not request.allow_i2p or request.i2p_proxy != policy.i2p_proxy:
                    raise PreservationHostError("receive_resume_i2p_policy_required")
            elif request.allow_i2p or request.i2p_proxy is not None:
                raise PreservationHostError("receive_resume_inputs_fixed")
            if isinstance(invitation, PeerInvitation):
                peer_setup, _ = self._peer_receiving_setup(request, policy.peer_private)
                payload = {**payload, "peer_setup_id": peer_setup}
            elif request.allow_peer_network or request.peer_setup_id is not None:
                raise PreservationHostError("http_receive_cannot_grant_peering")
            for field, expected in (
                ("download_limit_bytes", policy.max_bytes),
                ("download_deadline_seconds", policy.deadline_seconds),
                ("allow_http", policy.allow_http),
                ("allow_private", policy.allow_private),
                ("allow_i2p", policy.allow_i2p),
                ("i2p_proxy", policy.i2p_proxy),
            ):
                if (
                    field in request.model_fields_set
                    and getattr(request, field) != expected
                ):
                    raise PreservationHostError("receive_resume_inputs_fixed")
            return self._delivery_review(
                request, payload, invitation.manifest, location.trust, original.model_id
            )
        if request.upload_id is None:
            raise PreservationHostError("complete_invitation_required")
        invitation = read_invitation(
            self._metadata_upload(request.upload_id, "invitation")
        )
        trust = accept_invitation_keys(invitation, request.accepted_key_ids)
        manifest = invitation.manifest
        peer_setup: str | None = None
        peer_private = False
        if request.allow_i2p:
            if isinstance(invitation, PeerInvitation):
                raise PreservationHostError("peer_receive_cannot_grant_i2p")
            mirrors = i2p_model_mirrors(invitation.mirrors)
            if not mirrors:
                raise PreservationHostError("i2p_model_mirror_required")
            # Only unsigned routing hints change. The original upload, exact
            # signed envelope, selected model keys and registry proof are kept.
            invitation = read_invitation(
                encode_record(invitation.model_copy(update={"mirrors": mirrors}))
            )
        if isinstance(invitation, PeerInvitation):
            peer_setup, peer_private = self._peer_receiving_setup(request)
        elif request.allow_peer_network or request.peer_setup_id is not None:
            raise PreservationHostError("http_receive_cannot_grant_peering")
        policy = ReceivingPolicy(
            transport="peer"
            if isinstance(invitation, PeerInvitation)
            else "i2p"
            if request.allow_i2p
            else "http",
            allow_i2p=request.allow_i2p,
            i2p_proxy=request.i2p_proxy,
            allow_peer_network=isinstance(invitation, PeerInvitation),
            peer_private=peer_private,
            mirrors=invitation.mirrors,
            max_bytes=request.download_limit_bytes,
            deadline_seconds=request.download_deadline_seconds,
            allow_http=request.allow_http,
            allow_private=request.allow_private,
        )
        if sum(file.size for file in unique_objects(manifest)) > policy.max_bytes:
            raise PreservationHostError("receiving_allowance_too_small")
        token = self._destination_token(invitation.manifest_id, trust)
        destination = self.root / "public-copies" / token
        if (
            not destination.exists()
            and shutil.disk_usage(self.root).free
            < manifest.total_bytes + 32 * 1024 * 1024
        ):
            raise PreservationHostError("insufficient_space")
        info = self.root.stat()
        location = _Location(
            location_id="invitation",
            purpose="public",
            label="Selected public invitation",
            path=self.root.absolute(),
            device=info.st_dev,
            inode=info.st_ino,
            trust=trust,
            key_path=self.root / "unused-public-key",
        )
        receive_payload: dict[str, JsonValue] = {
            "peer_setup_id": peer_setup,
            "location": location.model_dump(mode="json"),
            "invitation": invitation.model_dump(mode="json"),
            "manifest_id": invitation.manifest_id,
            "receiving": policy.model_dump(mode="json"),
            "destination": str(destination.absolute()),
            "destination_id": "public-" + token,
            "output_trust": trust.model_dump(mode="json"),
        }
        return self._delivery_review(
            request, receive_payload, manifest, trust, manifest.name
        )

    async def _sharing_plan(self, request: PreservationPlanRequest) -> PreservationPlan:
        """Bind public-copy import or serving without parsing a runtime card."""
        from urllib.parse import urlsplit

        payload: dict[str, JsonValue] = {}
        policy = None
        if request.kind == "mirror_import":
            if request.upload_id is None or request.location_id != "public-kit":
                raise PreservationHostError("public_kit_selection_required")
            if not self.uploads.status(request.upload_id)["complete"]:
                raise PreservationHostError("kit_upload_incomplete")
            kit = inspect_kit(self.uploads.path(request.upload_id))
            if not isinstance(kit.manifest, Manifest):
                raise PreservationHostError("public_pmf_adapter_required")
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
            manifest = await self._manifest(location, identity)
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
            peer_private = False
            if request.kind == "peer_share":
                try:
                    await self._storage.run(
                        lambda: check_peer_space(
                            location.path, self.root, manifest.total_bytes
                        )
                    )
                except PreservationError as exc:
                    raise PreservationHostError("insufficient_space") from exc
                if (
                    not request.public_network
                    or request.listen_port != 0
                    or request.advertised_url is not None
                ):
                    raise PreservationHostError(
                        "peer_sharing_network_confirmation_required"
                    )
                setup_id, peer_private = self._peer_receiving_setup(request)
                payload["peer_setup_id"] = setup_id
            elif request.allow_peer_network or request.peer_setup_id is not None:
                raise PreservationHostError("http_torrent_cannot_grant_peer_access")
            policy = SharingPolicy(
                public_network=request.public_network,
                listen_port=request.listen_port,
                duration_seconds=request.duration_seconds,
                upload_limit_bytes=request.upload_limit_bytes,
                upload_rate_bytes=request.upload_rate_bytes,
                advertised_url=request.advertised_url,
                peer_private=peer_private,
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

    async def resume_receive(
        self, job_id: str, request: PreservationStart
    ) -> PreservationJob:
        """Explicitly resume one stopped receive after fresh unchanged-input review.

        This fences cleanup and validates retained work custody before requeueing.
        It preserves the original stage, plan and cumulative accounting. Replay
        of this action only observes state; restart never invokes it implicitly.
        """
        if self._closed:
            raise PreservationHostError("preservation_host_closed")
        reservation = self.journal.receive_resume_reserved(job_id, request)
        if reservation is not None:
            return reservation
        if self._queue.locked():
            raise PreservationHostError("store_busy_try_again")
        async with self._queue:
            if self._closed:
                raise PreservationHostError("preservation_host_closed")
            # Duplicate explicit actions are observational even if this job has
            # since completed or its temporary files have already been cleared.
            reservation = self.journal.receive_resume_reserved(job_id, request)
            if reservation is not None:
                return reservation
            if job_id in self._tasks or job_id in self._processes:
                raise PreservationHostError("temporary_job_not_stopped")
            previous = self.journal.job(job_id)
            if (
                not (self.root / "work" / job_id).exists()
                or (self.root / "work-trash" / job_id).exists()
            ):
                raise PreservationHostError("receive_work_unavailable")
            await asyncio.to_thread(
                manage_temporary_work, self.root, self.journal, previous, remove=False
            )
            job, created = self.journal.resume_receive(job_id, request)
            if created:
                self._tasks[job_id] = asyncio.create_task(self._execute(job_id))
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

    async def temporary_work(
        self, job_id: str, *, remove: bool = False
    ) -> TemporaryWork:
        """Inspect or explicitly clear stopped sharing/import staging.

        Selects one retained job without accepting paths. The exclusive queue
        and task/process checks prevent cleanup of active work. Disconnecting
        the caller cannot release ownership while deletion is in progress.
        Public copies, model-store files, keys and retained artifacts stay intact.
        """
        if self._closed:
            raise PreservationHostError("preservation_host_closed")
        job = self.journal.job(job_id)
        if job.state in {"queued", "running", "cancelling"}:
            raise PreservationHostError("temporary_job_not_stopped")
        async with self._queue:
            if self._closed:
                raise PreservationHostError("preservation_host_closed")
            job = self.journal.job(job_id)
            if job_id in self._tasks or job_id in self._processes:
                raise PreservationHostError("temporary_job_not_stopped")
            maintenance = asyncio.create_task(
                asyncio.to_thread(
                    manage_temporary_work, self.root, self.journal, job, remove=remove
                )
            )
            try:
                return await asyncio.shield(maintenance)
            except asyncio.CancelledError:
                while not maintenance.done():
                    try:
                        await asyncio.shield(maintenance)
                    except asyncio.CancelledError:
                        continue
                maintenance.result()
                raise

    async def close(self) -> None:
        """Own complete shutdown despite caller cancellation; allow failed cleanup retry."""
        self._closed = True
        if self._closing is None or (
            self._closing.done()
            and (self._closing.cancelled() or self._closing.exception() is not None)
        ):
            self._closing = asyncio.create_task(self._close())
        while not self._closing.done():
            try:
                await asyncio.shield(self._closing)
            except asyncio.CancelledError:
                continue
        self._closing.result()

    async def _close(self) -> None:
        async def withdraw(job_id: str) -> None:
            await self.cancel(job_id)

        # Start every withdrawal before awaiting any owner's filesystem cleanup.
        # A stalled mirror intake cannot leave an existing finite share serving,
        # nor may another failed close cancel its siblings' actual drain.
        results = await asyncio.gather(
            *(withdraw(job_id) for job_id in tuple(self._tasks)),
            self.mirrors.close(),
            self.discovery.close(),
            self.peers.close(),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks.values()), return_exceptions=True)
        async with self._queue, self._setup_lock:
            await self._storage.close()
            self.journal.close()

    async def _execute(self, job_id: str) -> None:
        try:
            serving = self.journal.job(job_id).kind in {
                "http_share",
                "torrent_share",
                "peer_share",
            }
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
                attempt_seconds = 86400
                if job.kind == "receive":
                    receiving = self.journal.plan(job.plan_id)[0].receiving
                    if receiving is None:
                        raise PreservationHostError("receive_policy_missing")
                    attempt_seconds = receiving.deadline_seconds
                if job.kind in {"preserve", "recover"}:
                    async with self.store.preservation_lease(
                        cancelled=lambda: self.journal.job(job_id).state == "cancelling"
                    ):
                        async with asyncio.timeout(86400):
                            await self._run(job_id)
                else:
                    async with asyncio.timeout(attempt_seconds):
                        await self._run(job_id)
        except Exception as error:
            job = self.journal.job(job_id)
            if job.state in {"queued", "running", "cancelling"}:
                self._finish(
                    job_id,
                    "cancelled"
                    if job.kind in {"http_share", "torrent_share", "peer_share"}
                    and job.state == "cancelling"
                    else "failed"
                    if job.kind in {"http_share", "torrent_share", "peer_share"}
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
        resuming_receive = plan.kind == "receive" and job.attempts > 1
        work_exists = work.exists() or work.is_symlink()
        if plan.kind in WORK_KINDS and work_exists and not resuming_receive:
            raise PreservationHostError("unknown_temporary_work")
        if resuming_receive and (
            not work_exists or self.journal.setting("temporary-work:" + job_id) is None
        ):
            raise PreservationHostError("receive_work_unavailable")
        secure_directory(work)
        register_temporary_work(self.root, self.journal, self.journal.job(job_id))
        if plan.kind == "mirror_import":
            self._public_copy_parent()
        publisher = (
            _Location.model_validate_json(safe_json(payload["publisher"]))
            if plan.kind == "publish"
            else None
        )
        if publisher is not None:
            async with self._archive(publisher):
                pass
        output_info = (
            self._public_copy_parent()
            if plan.kind in {"publish", "receive", "mirror_import"}
            else None
        )
        if plan.kind == "kit_export":
            secure_directory(self.root / "kits")
        if plan.kind == "torrent_share":
            secure_directory(self.root / "torrents")
        if plan.kind == "peer_share":
            secure_directory(self.root / "handoffs")
        async with (
            self._archive(location) as descriptor,
            AsyncExitStack() as peer_lifetime,
        ):
            peer: PeerLease | None = None
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
            if plan.receiving is not None and plan.receiving.transport == "peer":
                setup_id = self.journal.setting("peer-attempt:" + job_id)
                if setup_id is None:
                    setup_id = payload.get("peer_setup_id")
                if not plan.receiving.allow_peer_network or not isinstance(
                    setup_id, str
                ):
                    raise PreservationHostError("peer_network_confirmation_required")
                peer = await self.peers.open(
                    setup_id,
                    plan.receiving.deadline_seconds,
                    lambda: self._closed
                    or self.journal.job(job_id).state == "cancelling",
                )
                peer_lifetime.push_async_callback(peer.close)
            if plan.kind == "peer_share":
                setup_id = payload.get("peer_setup_id")
                if (
                    not isinstance(setup_id, str)
                    or plan.sharing is None
                    or not submitted.public_participation
                ):
                    raise PreservationHostError("peer_network_confirmation_required")
                peer = await self.peers.open_sharing(
                    setup_id,
                    plan.sharing.duration_seconds,
                    plan.sharing.upload_limit_bytes,
                    work / "peer",
                    lambda: self._closed
                    or self.journal.job(job_id).state == "cancelling",
                )
                peer_lifetime.push_async_callback(peer.close)
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
                        "peer_share",
                        "publish",
                        "receive",
                    ],
                    plan.kind,
                ),
                archive_format=plan.archive_format,
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
                if plan.kind in {"mirror_import", "publish", "receive"}
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
                source_manifest_id=cast(str | None, payload.get("source_manifest_id")),
                publication_request=PublicationRequest.model_validate_json(
                    safe_json(payload["publication_request"])
                )
                if plan.kind == "publish"
                else None,
                publisher_key_path=publisher.key_path
                if publisher is not None
                else None,
                publisher_trust=publisher.trust if publisher is not None else None,
                invitation=read_invitation(safe_json(payload["invitation"]).encode())
                if plan.kind == "receive"
                else None,
                receiving=plan.receiving,
                peer_repository=peer.repository if peer else None,
                peer_binary=peer.binary if peer else None,
                peer_contacts=self.peers.selected()[0].bootstrap
                if plan.kind == "peer_share"
                else (),
                handoff_path=(
                    self.root / "handoffs" / (job_id + ".foxhandoff")
                ).absolute()
                if plan.kind == "peer_share"
                else None,
                output_parent_device=output_info.st_dev
                if output_info is not None
                else None,
                output_parent_inode=output_info.st_ino
                if output_info is not None
                else None,
            )
            request_path = work / "request.json"
            if resuming_receive:
                with directory(work) as target:
                    atomic_bytes(
                        target, "request.json", request.model_dump_json().encode()
                    )
            else:
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
                            plan.kind
                            not in {"http_share", "torrent_share", "peer_share"}
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
                        if peer is not None and plan.kind != "peer_share":
                            await peer.close()
                            current = self.journal.job(job_id)
                        if observation.manifest_id is None or (
                            plan.manifest_id is not None
                            and observation.manifest_id != plan.manifest_id
                        ):
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
                                    "bytes_received": max(
                                        current.bytes_received,
                                        observation.bytes_received,
                                    ),
                                }
                            )
                        )
                    elif observation.event == "result":
                        result = observation
                    else:
                        error_code = observation.error_code or "operation_failed"
                exit_code = await process.wait()
                if peer is not None:
                    if plan.kind == "peer_share" and peer.failed() and exit_code == 0:
                        exit_code = 1
                        error_code = "peer_runtime_interrupted"
                    await peer.close()
                current = self.journal.job(job_id)
                if plan.kind == "peer_share" and current.sharing is not None:
                    # Native PeerLease has drained above; only now report stopped.
                    current = current.model_copy(
                        update={
                            "sharing": current.sharing.model_copy(
                                update={"ready": False}
                            )
                        }
                    )
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
                                or plan.kind in {"mirror_import", "publish", "receive"}
                                else "private"
                            )
                            and plan.kind != "recover"
                        )
                    ):
                        raise PreservationHostError("worker_outcome_mismatch")
                    if plan.kind in {
                        "mirror_import",
                        "publish",
                        "receive",
                        "http_share",
                        "torrent_share",
                        "peer_share",
                    }:
                        now = int(time.time())
                        copy_id = plan.copy_id
                        if plan.kind in {"mirror_import", "publish", "receive"}:
                            destination = Path(cast(str, payload["destination"]))
                            if output_info is not None:
                                current_parent = destination.parent.stat()
                                if (current_parent.st_dev, current_parent.st_ino) != (
                                    output_info.st_dev,
                                    output_info.st_ino,
                                ):
                                    raise PreservationHostError(
                                        "public_destination_changed"
                                    )
                            output_trust = (
                                TrustProfile.model_validate_json(
                                    safe_json(payload["output_trust"])
                                )
                                if "output_trust" in payload
                                else location.trust
                            )
                            info = destination.stat()
                            enrolled = _Location(
                                location_id=cast(str, payload["destination_id"]),
                                purpose="public",
                                label="Public bundle copy",
                                path=destination,
                                device=info.st_dev,
                                inode=info.st_ino,
                                trust=output_trust,
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
                                        key.key_id for key in output_trust.keys
                                    ),
                                )
                            )
                        self.journal.update_job(
                            current.model_copy(
                                update={
                                    "state": "cancelled"
                                    if current.state == "cancelling"
                                    and plan.kind
                                    in {"http_share", "torrent_share", "peer_share"}
                                    else "succeeded",
                                    "stage": "stopped"
                                    if plan.kind
                                    in {"http_share", "torrent_share", "peer_share"}
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
                        if plan.kind in {"http_share", "torrent_share", "peer_share"}
                        and (current.state == "cancelling" or error_code == "cancelled")
                        else "failed"
                        if plan.kind in {"http_share", "torrent_share", "peer_share"}
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
                if peer is not None:
                    await peer.close()
                self._processes.pop(job_id, None)
                if (
                    plan.kind != "receive"
                    or self.journal.job(job_id).state == "succeeded"
                ):
                    request_path.unlink(missing_ok=True)
                    if plan.kind != "peer_share":
                        shutil.rmtree(work)
