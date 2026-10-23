"""Known native peer worker: parent EOF cooperatively drains the actual daemon."""

import sys
import threading
from pathlib import Path
from typing import Literal, final

from foxlight_model_preservation.operations import OperationControl
from foxlight_model_preservation.peer_runtime import serve_peer
from pydantic import BaseModel, ConfigDict, Field


@final
class PeerWorkerRequest(BaseModel):
    """Protected Core-authored request; no browser supplies these paths or code."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    worker_family: Literal["preservation-peer"] = "preservation-peer"
    repository: Path = Field(description="Dedicated receive cache or fresh job-owned mirror repo.")
    binary: str = Field(
        description="Host-selected qualified installed Kubo executable."
    )
    seconds: int = Field(ge=1, le=86400, description="Finite owned network lifetime.")
    share_public: bool = Field(
        default=False, description="Separate host-reviewed public mirror participation."
    )
    max_upload_bytes: int = Field(
        default=1024**3,
        ge=1,
        le=2**53 - 1,
        description="Observed native-wire upload stop threshold, with in-flight overshoot.",
    )


def run_peer_worker(request: PeerWorkerRequest) -> int:
    """Own one reviewed receive/mirror profile; stop its child on cancel or EOF."""
    control = OperationControl()

    def commands() -> None:
        for line in sys.stdin:
            if line == "cancel\n":
                control.cancel()
        control.cancel()

    threading.Thread(target=commands, daemon=True).start()
    try:
        serve_peer(
            request.repository,
            request.binary,
            allow_network=True,
            seconds=request.seconds,
            share_public=request.share_public,
            max_upload_bytes=request.max_upload_bytes,
            control=control,
        )
        return 0
    except (ValueError, OSError):
        return 1
