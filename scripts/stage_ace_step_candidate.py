"""Freshly install exact ACE candidate bytes through the owned native supervisor.

This validation entry point never downloads models, selects a live engine cache,
publishes packages or advertises a ready backend.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from skulk.provisioning.demucs_package import inspect_candidate, install_fresh_runtime
from skulk.provisioning.native_install_process import NativeInstallCommandRunner


class Arguments(argparse.Namespace):
    """Exact candidate object identity and a fresh private output location."""

    wheel: Path
    sha256: str
    output: Path


def main() -> None:
    """Verify ACE ownership and install offline with actual parent-loss supervision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = Arguments()
    parser.parse_args(namespace=arguments)
    output, wheel = arguments.output.resolve(), arguments.wheel.resolve()
    deadline = time.monotonic() + 20 * 60

    def checkpoint() -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError("native music bootstrap exceeded its total deadline")

    manifest = inspect_candidate(wheel, arguments.sha256, checkpoint=checkpoint)
    if manifest.source_repository != "https://github.com/ace-step/ACE-Step-1.5.git":
        raise ValueError("native music validation requires an ACE candidate")
    runner = NativeInstallCommandRunner(
        output.parent / "installer-diagnostics",
        checkpoint=checkpoint,
        guardian_interpreter=output / "interpreter/python/bin/python3.12",
    )
    try:
        print(
            install_fresh_runtime(
                wheel, output, arguments.sha256, run=runner, checkpoint=checkpoint
            )
        )
    finally:
        runner.drain()


if __name__ == "__main__":
    main()
