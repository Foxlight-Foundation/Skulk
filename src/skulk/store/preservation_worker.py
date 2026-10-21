"""Exclusively owned preservation byte process; parent loss stops all byte work."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Literal

from foxlight_model_preservation.kits import export_kit, import_kit, inspect_kit
from foxlight_model_preservation.models import TrustProfile
from foxlight_model_preservation.operations import (
    OperationControl,
    Progress,
    failure_code,
)
from foxlight_model_preservation.signing import load_key
from foxlight_model_preservation.storage import verify_store
from pydantic import BaseModel, ConfigDict, JsonValue

from skulk.store.preservation import RecoveryCompatibilityError, recover_home
from skulk.store.preservation_contracts import SharingPolicy, SharingStatus
from skulk.store.preservation_export import (
    PreservationCapacityError,
    preservation_generation,
    preserve_installed,
)
from skulk.store.preservation_live import (
    validate_existing_recovery,
    validate_prepared_home,
)
from skulk.store.preservation_sharing import (
    import_public_bundle,
    require_public_bundle,
    serve_public_http,
    serve_public_torrent,
)


class WorkerRequest(BaseModel):
    """Private host-authored input; never accepted as a browser filesystem grant."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Literal[
        "preserve",
        "check",
        "kit_export",
        "recover",
        "mirror_import",
        "http_share",
        "torrent_share",
    ]
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
                "stage": progress.stage,
                "bytes_hashed": progress.bytes_hashed,
                "bytes_copied": progress.bytes_copied,
            }
        )

    control = OperationControl(observe, interval_seconds=0.25)
    control.limit_deadline(time.monotonic() + 86400)
    controls = [control]
    controls_lock = threading.Lock()
    cancelled = threading.Event()

    def export_control() -> OperationControl:
        active = OperationControl(observe, interval_seconds=0.25)
        active.limit_deadline(time.monotonic() + 86400)
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
            )
            # A separate final check does not reuse the consumed capture control.
            manifest = verify_store(Path("."), manifest_id, request.trust)
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
            require_public_bundle(info.manifest)
            manifest_id = request.manifest_id
            commit_intent(manifest_id)
            manifest = import_public_bundle(
                request.kit_path,
                request.output_path,
                manifest_id,
                request.trust,
                control,
            )
        elif request.kind in {"http_share", "torrent_share"}:
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
                    export_control(),
                    sharing_observation,
                )
        elif request.kind == "check":
            if request.manifest_id is None:
                raise ValueError("selected copy required")
            manifest_id = request.manifest_id
            manifest = verify_store(
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
                        control=control,
                    )
                manifest = verify_store(archive, manifest_id, request.trust)
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
    request = WorkerRequest.model_validate_json(data)
    raise SystemExit(run_worker(request))


if __name__ == "__main__":
    main()
