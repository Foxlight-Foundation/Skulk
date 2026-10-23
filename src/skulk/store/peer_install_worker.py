"""Known software-acquisition child with EOF cancellation and no peer operations."""

import json
import os
import sys
import threading
from pathlib import Path
from typing import Literal, final

from foxlight_model_preservation.operations import (
    OperationControl,
    Progress,
    failure_code,
)
from foxlight_model_preservation.peer_install import install_runtime, runtime_plan
from foxlight_model_preservation.signing import digest, encode_record
from pydantic import BaseModel, ConfigDict, Field, JsonValue


@final
class RuntimeSetupWorkerRequest(BaseModel):
    """Protected parent-authored paths; the HTTP API accepts neither paths nor URLs."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    worker_family: Literal["preservation-peer-install"] = "preservation-peer-install"
    runtime_directory: Path = Field(
        description="Parent-selected private software storage."
    )
    plan_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Acquisition identity authorized by the parent.",
    )


def run_runtime_setup_worker(request: RuntimeSetupWorkerRequest) -> int:
    """Acquire fixed packaged bytes; EOF immediately ends even blocked network IO."""

    def emit(value: dict[str, JsonValue]) -> None:
        _ = sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
        sys.stdout.flush()

    def progress(value: Progress) -> None:
        emit({"event": "progress", "bytes_copied": value.bytes_copied})

    control = OperationControl(progress, interval_seconds=0.25)

    def commands() -> None:
        for line in sys.stdin:
            if line == "cancel\n":
                control.cancel()
        # Atomic generation publication tolerates abrupt owner loss. Do not
        # leave acquisition alive in an uninterruptible resolver/storage call.
        os._exit(75)

    threading.Thread(target=commands, daemon=True).start()
    try:
        if digest(encode_record(runtime_plan())) != request.plan_id:
            raise ValueError("native plan changed")
        _ = install_runtime(
            request.runtime_directory, allow_download=True, control=control
        )
        emit({"event": "succeeded"})
        return 0
    except (ValueError, OSError) as error:
        emit({"event": "error", "error_code": failure_code(error)})
        return 1
