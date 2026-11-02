"""Private native CLI with bounded job/result JSON and parent-death cleanup."""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import TextIO

from .contracts import NativeSeparationJob, NativeSeparationProgress


class Arguments(argparse.Namespace):
    """Typed private local CLI arguments, never an exposed HTTP surface."""

    job: Path
    parent_pid: int


def main() -> None:
    """Execute one strict local job and stop the process group if its owner dies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    arguments = Arguments()
    parser.parse_args(namespace=arguments)
    if arguments.parent_pid <= 1 or os.getpid() != os.getpgrp():
        raise ValueError(
            "native separation requires a supervised private process session"
        )

    def reap() -> None:
        while os.getppid() == arguments.parent_pid:
            time.sleep(0.25)
        os.killpg(os.getpgrp(), signal.SIGKILL)

    threading.Thread(target=reap, daemon=True).start()
    # Engine model management is never used. Force offline behavior as defense
    # in depth against an accidental future upstream download path.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    protocol: TextIO = sys.stdout

    def progress(record: NativeSeparationProgress) -> None:
        print(record.model_dump_json(), file=protocol, flush=True)

    with arguments.job.open("rb") as source:
        data = source.read(16 * 1024 + 1)
    if len(data) > 16 * 1024:
        raise ValueError("native separation job exceeds its private protocol bound")
    job = NativeSeparationJob.model_validate_json(data)
    # Upstream diagnostics cannot become protocol messages or block an unbounded
    # stdout parser. Only owned typed progress records use the original stdout.
    with contextlib.redirect_stdout(sys.stderr):
        from .engine import separate

        result = separate(job, progress)
    destination = Path(job.output_dir) / "result.json"
    with destination.open("x") as completed:
        completed.write(result.model_dump_json())


if __name__ == "__main__":
    main()
