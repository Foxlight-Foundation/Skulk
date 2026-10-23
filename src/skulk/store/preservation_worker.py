"""Exclusively owned preservation byte process; parent loss stops all byte work."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Literal

from foxlight_model_preservation.archives import verify_archive
from foxlight_model_preservation.kits import export_kit, import_kit, inspect_kit
from foxlight_model_preservation.models import TrustProfile
from foxlight_model_preservation.operations import (
    OperationControl,
    Progress,
    failure_code,
)
from foxlight_model_preservation.publication import PublicationRequest
from foxlight_model_preservation.signing import load_key
from pydantic import BaseModel, ConfigDict, JsonValue

from skulk.store.preservation import RecoveryCompatibilityError, recover_home
from skulk.store.preservation_contracts import (
    ReceivingPolicy,
    SharingPolicy,
    SharingStatus,
)
from skulk.store.preservation_delivery import publish_copy, receive_copy
from skulk.store.preservation_export import (
    PreservationCapacityError,
    preservation_generation,
    preserve_installed,
)
from skulk.store.preservation_live import (
    validate_existing_recovery,
    validate_prepared_home,
)
from skulk.store.preservation_pmf import PMFProfileUnavailableError, verify_capture
from skulk.store.preservation_references import ModelInvitation as Invitation
from skulk.store.preservation_sharing import (
    import_public_bundle,
    require_public_bundle,
    serve_public_http,
    serve_public_peer,
    serve_public_torrent,
)


class WorkerRequest(BaseModel):
    """Private host-authored input; never accepted as a browser filesystem grant."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    kind: Literal[
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
    ]
    archive_format: Literal["preservation-v1", "pmf-v0.1"] = "preservation-v1"
    archive_descriptor: int
    archive_device: int
    archive_inode: int
    model_directory: Path | None = None
    source_root: Path
    generation_id: str | None = None
    snapshot_parent: Path
    key_path: Path
    trust: TrustProfile
    manifest_id: str | None = None
    output_path: Path | None = None
    kit_path: Path | None = None
    historical: bool = False
    accept_owner_capture: bool = False
    allow_repository_code: bool = False
    existing_companions: tuple[str, ...] = ()
    sharing: SharingPolicy | None = None
    public_participation: bool = False
    metainfo_path: Path | None = None
    source_manifest_id: str | None = None
    publication_request: PublicationRequest | None = None
    publisher_key_path: Path | None = None
    publisher_trust: TrustProfile | None = None
    invitation: Invitation | None = None
    peer_repository: Path | None = None
    peer_binary: str | None = None
    peer_contacts: tuple[str, ...] = ()
    handoff_path: Path | None = None
    receiving: ReceivingPolicy | None = None
    output_parent_device: int | None = None
    output_parent_inode: int | None = None


def _emit(value: dict[str, JsonValue]) -> None:
    encoded = json.dumps(value, separators=(",", ":"))
    if len(encoded.encode()) > 16384:
        raise ValueError("worker observation too large")
    _ = sys.stdout.write(encoded + "\n")
    sys.stdout.flush()


def run_worker(request: WorkerRequest) -> int:
    """Execute against the already pinned archive; EOF is loss of host ownership."""
    info = os.fstat(request.archive_descriptor)
    if (info.st_dev, info.st_ino) != (request.archive_device, request.archive_inode):
        raise ValueError("archive identity changed")
    # Process-local cwd stays on this opened filesystem even if its mount/path is
    # replaced. Public-core writes remain descriptor-relative beneath this cwd.
    os.fchdir(request.archive_descriptor)
    acknowledgement = threading.Event()
    expected: list[str] = []

    def observe(progress: Progress) -> None:
        _emit(
            {
                "event": "progress",
                "stage": "checking"
                if request.kind in {"publish", "receive"}
                and progress.stage == "complete"
                else progress.stage,
                "bytes_hashed": progress.bytes_hashed,
                "bytes_copied": progress.bytes_copied,
                "bytes_received": progress.bytes_received,
            }
        )

    control = OperationControl(observe, interval_seconds=0.25)
    deadline = time.monotonic() + (
        request.receiving.deadline_seconds
        if request.kind == "receive" and request.receiving is not None
        else 86400
    )
    control.limit_deadline(deadline)
    controls = [control]
    controls_lock = threading.Lock()
    cancelled = threading.Event()

    def byte_control() -> OperationControl:
        active = OperationControl(observe, interval_seconds=0.25)
        active.limit_deadline(deadline)
        with controls_lock:
            controls.append(active)
            if cancelled.is_set():
                active.cancel()
        return active

    def sharing_observation(status: SharingStatus) -> None:
        _emit({"event": "sharing", "sharing": status.model_dump(mode="json")})

    def commands() -> None:
        for line in sys.stdin:
            if len(line) > 1024:
                os._exit(75)
            if line.strip() == "cancel":
                cancelled.set()
                with controls_lock:
                    for active in controls:
                        active.cancel()
            elif expected and line.strip() == "publish:" + expected[0]:
                acknowledgement.set()
        # Core is gone. Do not leave an orphan copying, serving, or publishing.
        os._exit(75)

    threading.Thread(target=commands, daemon=True).start()

    def commit_intent(manifest_id: str) -> None:
        expected.append(manifest_id)
        _emit({"event": "expected", "manifest_id": manifest_id})
        while not acknowledgement.wait(0.1):
            control.checkpoint()
        control.checkpoint()

    try:
        if request.kind == "preserve":
            if request.model_directory is None:
                raise ValueError("source selection required")
            if (
                preservation_generation(
                    request.model_directory, artifact_roots=(request.source_root,)
                )
                != request.generation_id
            ):
                raise ValueError("source selection changed")
            manifest_id = preserve_installed(
                request.model_directory,
                Path("."),
                (load_key(request.key_path, None),),
                artifact_roots=(request.source_root,),
                control=control,
                snapshot_parent=request.snapshot_parent,
                before_publish=commit_intent,
                portable=request.archive_format == "pmf-v0.1",
            )
            # A separate final check does not reuse the consumed capture control.
            manifest = verify_archive(
                Path("."), manifest_id, request.trust, control=byte_control()
            )
        elif request.kind == "publish":
            if (
                request.manifest_id is None
                or request.source_manifest_id is None
                or request.output_path is None
                or request.publication_request is None
                or request.publisher_key_path is None
                or request.publisher_trust is None
                or request.output_parent_device is None
                or request.output_parent_inode is None
                or not request.public_participation
            ):
                raise ValueError("reviewed separate public publication required")
            manifest_id = request.manifest_id
            publishing_key_path = request.publisher_key_path
            manifest = publish_copy(
                request.source_manifest_id,
                request.trust,
                request.publication_request,
                request.output_path,
                manifest_id,
                lambda: load_key(publishing_key_path, None),
                request.publisher_trust,
                request.snapshot_parent,
                control,
                byte_control(),
                commit_intent,
                parent_identity=(
                    request.output_parent_device,
                    request.output_parent_inode,
                ),
                conversion_control=byte_control(),
            )
        elif request.kind == "receive":
            if (
                request.manifest_id is None
                or request.output_path is None
                or request.invitation is None
                or request.receiving is None
                or request.output_parent_device is None
                or request.output_parent_inode is None
                or not request.accept_owner_capture
                or request.invitation.manifest_id != request.manifest_id
            ):
                raise ValueError("accepted exact public invitation required")
            manifest_id = request.manifest_id
            manifest = receive_copy(
                request.invitation,
                request.trust,
                request.receiving,
                request.output_path,
                request.snapshot_parent,
                control,
                byte_control,
                commit_intent,
                peer_repository=request.peer_repository,
                peer_binary=request.peer_binary,
                parent_identity=(
                    request.output_parent_device,
                    request.output_parent_inode,
                ),
            )
        elif request.kind == "mirror_import":
            if (
                request.manifest_id is None
                or request.kit_path is None
                or request.output_path is None
                or not request.accept_owner_capture
            ):
                raise ValueError("accepted exact public kit required")
            info = inspect_kit(request.kit_path)
            if info.manifest_id != request.manifest_id or info.trust != request.trust:
                raise ValueError("public kit changed")
            from foxlight_model_preservation.models import Manifest

            if not isinstance(info.manifest, Manifest):
                raise ValueError("public PMF sharing requires a supported adapter")
            require_public_bundle(info.manifest)
            manifest_id = request.manifest_id
            manifest = import_public_bundle(
                request.kit_path,
                request.output_path,
                manifest_id,
                request.trust,
                request.snapshot_parent,
                control,
                byte_control(),
                commit_intent,
                parent_identity=(
                    request.output_parent_device,
                    request.output_parent_inode,
                )
                if request.output_parent_device is not None
                and request.output_parent_inode is not None
                else None,
            )
        elif request.kind in {"http_share", "torrent_share", "peer_share"}:
            if (
                request.manifest_id is None
                or request.sharing is None
                or not request.public_participation
            ):
                raise ValueError("reviewed public participation required")
            manifest_id = request.manifest_id
            # This check has no runtime model-card interpretation or admission.
            # Serving constructors independently rehash complete data before binding.
            from foxlight_model_preservation.signing import verify_manifest
            from foxlight_model_preservation.storage import manifest_path, read_metadata

            manifest = verify_manifest(
                read_metadata(request.archive_descriptor, manifest_path(manifest_id)),
                request.trust,
                manifest_id,
            )
            require_public_bundle(manifest)
            commit_intent(manifest_id)
            if request.kind == "http_share":
                serve_public_http(
                    manifest_id,
                    request.trust,
                    request.sharing,
                    control,
                    sharing_observation,
                )
            elif request.kind == "peer_share":
                if (
                    request.peer_repository is None
                    or request.peer_binary is None
                    or request.handoff_path is None
                ):
                    raise ValueError("owned peer mirror and handoff output required")
                serve_public_peer(
                    manifest_id,
                    request.trust,
                    request.sharing,
                    request.peer_repository,
                    request.peer_binary,
                    request.peer_contacts,
                    request.handoff_path,
                    control,
                    sharing_observation,
                )
            else:
                if request.metainfo_path is None:
                    raise ValueError("host torrent artifact required")
                serve_public_torrent(
                    manifest_id,
                    request.trust,
                    request.sharing,
                    request.snapshot_parent,
                    request.metainfo_path,
                    control,
                    byte_control(),
                    sharing_observation,
                )
        elif request.kind == "check":
            if request.manifest_id is None:
                raise ValueError("selected copy required")
            manifest_id = request.manifest_id
            manifest = verify_archive(
                Path("."), manifest_id, request.trust, control=control
            )
        else:
            if request.manifest_id is None or request.output_path is None:
                raise ValueError("exact selected copy and output required")
            manifest_id = request.manifest_id
            if request.kind == "kit_export":
                commit_intent(manifest_id)
                info = export_kit(
                    Path("."),
                    manifest_id,
                    request.trust,
                    request.output_path,
                    control=control,
                )
                manifest = info.manifest
            else:
                archive = Path(".")
                if request.kit_path is not None:
                    archive = request.snapshot_parent / "kit-store"
                    import_kit(
                        request.kit_path,
                        archive,
                        expected_manifest_id=manifest_id,
                        trust=request.trust,
                        control=byte_control(),
                    )
                manifest = verify_capture(archive, manifest_id, request.trust)
                control.checkpoint()
                recover_home(
                    archive,
                    manifest_id,
                    request.trust,
                    request.output_path,
                    mode="historical" if request.historical else "fresh",
                    accept_owner_capture=request.accept_owner_capture,
                    allow_repository_code=request.allow_repository_code,
                )
                validate_prepared_home(request.output_path)
                validate_existing_recovery(
                    request.output_path,
                    request.source_root,
                    request.existing_companions,
                )
                control.checkpoint()
                commit_intent(manifest_id)
        _emit(
            {
                "event": "result",
                "manifest_id": manifest_id,
                "size_bytes": manifest.total_bytes,
                "member_count": len(manifest.members),
                "audience": manifest.members[0].intake.audience,
            }
        )
        return 0
    except Exception as error:
        _emit(
            {
                "event": "error",
                "error_code": "insufficient_space"
                if isinstance(error, PreservationCapacityError)
                else "unsupported_pmf_profile"
                if isinstance(error, PMFProfileUnavailableError)
                else "unsupported_model_card"
                if isinstance(error, RecoveryCompatibilityError)
                else failure_code(error),
            }
        )
        return 1


def main() -> None:
    """Load one owner-private bounded request and run only the known worker."""
    if len(sys.argv) != 2:
        raise SystemExit(2)
    data = Path(sys.argv[1]).read_bytes()
    if len(data) > 8 * 1024 * 1024:
        raise SystemExit(2)
    from pydantic import TypeAdapter

    from skulk.store.discovery_worker import (
        DiscoveryWorkerRequest,
        run_discovery_worker,
    )
    from skulk.store.peer_install_worker import (
        RuntimeSetupWorkerRequest,
        run_runtime_setup_worker,
    )
    from skulk.store.peer_worker import PeerWorkerRequest, run_peer_worker

    adapter: TypeAdapter[
        WorkerRequest
        | DiscoveryWorkerRequest
        | PeerWorkerRequest
        | RuntimeSetupWorkerRequest
    ] = TypeAdapter(
        WorkerRequest
        | DiscoveryWorkerRequest
        | PeerWorkerRequest
        | RuntimeSetupWorkerRequest
    )
    request = adapter.validate_json(data)
    if isinstance(request, RuntimeSetupWorkerRequest):
        raise SystemExit(run_runtime_setup_worker(request))
    if isinstance(request, PeerWorkerRequest):
        raise SystemExit(run_peer_worker(request))
    raise SystemExit(
        run_discovery_worker(request)
        if isinstance(request, DiscoveryWorkerRequest)
        else run_worker(request)
    )


if __name__ == "__main__":
    main()
