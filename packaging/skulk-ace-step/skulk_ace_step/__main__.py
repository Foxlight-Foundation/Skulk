"""Private native CLI; upstream output is isolated from bounded control frames."""

from __future__ import annotations

import argparse
import contextlib
import os
import stat
import sys
import traceback
from pathlib import Path

from .contracts import (
    MAX_NATIVE_MUSIC_RECORD_BYTES,
    NativeMusicDevice,
    NativeMusicFailed,
    NativeMusicFrame,
    NativeMusicServeConfig,
)
from .lifetime import start_parent_guardian, terminate_owned_group


class Arguments(argparse.Namespace):
    """One explicit probe or private model intent and its actual supervisor."""

    serve: Path | None = None
    probe: NativeMusicDevice | None = None
    parent_pid: int
    drain_fd: int | None = None


def main() -> None:
    """Probe libraries or serve one model; retire the group after any failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--serve", type=Path)
    modes.add_argument("--probe", choices=("cpu", "mps", "cuda", "rocm"))
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--drain-fd", type=int)
    arguments = Arguments()
    parser.parse_args(namespace=arguments)
    start_parent_guardian(arguments.parent_pid, arguments.drain_fd)
    os.umask(0o077)
    # Set before the first torch/ACE import. The native environment has no Core
    # imports, ambient user site, planner or server-managed model downloads.
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
        }
    )
    protocol = sys.stdout
    failed = False

    def emit(frame: NativeMusicFrame) -> None:
        nonlocal failed
        encoded = frame.model_dump_json()
        if len(encoded.encode("utf-8")) + 1 > MAX_NATIVE_MUSIC_RECORD_BYTES:
            raise ValueError("native music response exceeds its control bound")
        if isinstance(frame, NativeMusicFailed):
            failed = True
        print(encoded, file=protocol, flush=True)

    try:
        if arguments.probe is not None:
            with contextlib.redirect_stdout(sys.stderr):
                from .policy import probe_libraries
                from .server import installed_payload_identity

                receipt = probe_libraries(arguments.probe, installed_payload_identity())
            encoded = receipt.model_dump_json()
            if len(encoded.encode("utf-8")) + 1 > MAX_NATIVE_MUSIC_RECORD_BYTES:
                raise ValueError("native library probe exceeds its control bound")
            print(encoded, file=protocol, flush=True)
            return
        assert arguments.serve is not None
        descriptor = os.open(
            arguments.serve, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        )
        with os.fdopen(descriptor, "rb") as stream:
            facts = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(facts.st_mode)
                or facts.st_size > MAX_NATIVE_MUSIC_RECORD_BYTES
            ):
                raise ValueError(
                    "native music configuration is not a bounded regular file"
                )
            data = stream.read(MAX_NATIVE_MUSIC_RECORD_BYTES + 1)
        if len(data) > MAX_NATIVE_MUSIC_RECORD_BYTES:
            raise ValueError("native music configuration exceeds its bound")
        with contextlib.redirect_stdout(sys.stderr):
            from .server import installed_payload_identity, serve

            serve(
                NativeMusicServeConfig.model_validate_json(data),
                sys.stdin.buffer,
                emit,
                payload_sha256=installed_payload_identity(),
            )
    except BaseException:
        try:
            try:
                if not failed:
                    emit(NativeMusicFailed())
            finally:
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
        finally:
            # Even a closed control pipe must reach this termination. Native's
            # timeout leaves a daemon inference thread alive; Python/torch
            # atexit could hang or free its active model state.
            terminate_owned_group()


if __name__ == "__main__":
    main()
