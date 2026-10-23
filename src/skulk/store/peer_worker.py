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

    model_config = ConfigDict(frozen=True, extra="forbid")
    worker_family: Literal["preservation-peer"] = "preservation-peer"
    repository: Path = Field(description="Dedicated receive-only Core cache.")
    binary: str = Field(
        description="Host-selected qualified installed Kubo executable."
    )
    seconds: int = Field(ge=1, le=86400, description="Finite owned network lifetime.")


def run_peer_worker(request: PeerWorkerRequest) -> int:
    """Serve only a receiving profile and stop its child on cancellation or EOF."""
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
            control=control,
        )
        return 0
    except (ValueError, OSError):
        return 1
