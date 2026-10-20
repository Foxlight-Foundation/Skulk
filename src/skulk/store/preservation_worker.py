"""Exclusively owned preservation byte process; parent loss stops all byte work."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Literal

from foxlight_model_preservation.models import TrustProfile
from foxlight_model_preservation.operations import (
    OperationControl,
    Progress,
    failure_code,
)
from foxlight_model_preservation.signing import load_key
from foxlight_model_preservation.storage import verify_store
from pydantic import BaseModel, ConfigDict, JsonValue

from skulk.store.preservation_export import (
    PreservationCapacityError,
    preservation_generation,
    preserve_installed,
)


class WorkerRequest(BaseModel):
    """Private host-authored input; never accepted as a browser filesystem grant."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Literal["preserve", "check"]
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

    def commands() -> None:
        for line in sys.stdin:
            if len(line) > 1024:
                os._exit(75)
            if line.strip() == "cancel":
                control.cancel()
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
        else:
            if request.manifest_id is None:
                raise ValueError("selected copy required")
            manifest_id = request.manifest_id
            manifest = verify_store(
                Path("."), manifest_id, request.trust, control=control
            )
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
