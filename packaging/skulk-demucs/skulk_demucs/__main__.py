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

from .contracts import (
    NativeSeparationFrame,
    NativeSeparationJob,
    NativeSeparationProgress,
    NativeSeparationServeConfig,
)


class Arguments(argparse.Namespace):
    """Typed private local CLI arguments, never an exposed HTTP surface."""

    job: Path | None
    serve: Path | None
    parent_pid: int


def main() -> None:
    """Run a strict job or resident model; stop its group when the owner dies."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--job", type=Path)
    mode.add_argument("--serve", type=Path)
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

    selected = arguments.serve or arguments.job
    assert selected is not None
    with selected.open("rb") as source:
        data = source.read(16 * 1024 + 1)
    if len(data) > 16 * 1024:
        raise ValueError("native separation configuration exceeds its private bound")
    # Upstream diagnostics cannot become protocol messages. Owned output keeps
    # the original stdout, even while native imports print to redirected stderr.
    with contextlib.redirect_stdout(sys.stderr):
        if arguments.serve is not None:
            from .server import installed_payload_identity, serve

            def frame(record: NativeSeparationFrame) -> None:
                encoded = record.model_dump_json()
                if len(encoded.encode("utf-8")) + 1 > 16 * 1024:
                    raise ValueError("native resident response exceeds its bound")
                print(encoded, file=protocol, flush=True)

            serve(
                NativeSeparationServeConfig.model_validate_json(data),
                sys.stdin.buffer,
                frame,
                payload_sha256=installed_payload_identity(),
            )
        else:
            from .engine import separate

            job = NativeSeparationJob.model_validate_json(data)
            result = separate(job, progress)
            destination = Path(job.output_dir) / "result.json"
            with destination.open("x") as completed:
                completed.write(result.model_dump_json())


if __name__ == "__main__":
    main()
