"""Exercise sealed offline bootstrap through Core's actual native supervisor.

This fresh-only validation entry point does not publish packages, select cached
generations, download weights or advertise a ready separation engine.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from skulk.provisioning.demucs_package import install_fresh_runtime
from skulk.provisioning.native_install_process import NativeInstallCommandRunner


class Arguments(argparse.Namespace):
    """Exact package identity and a new, explicitly owned validation directory."""

    wheel: Path
    sha256: str
    output: Path


def main() -> None:
    """Stage one candidate with bounded command lifetime and complete verification.

    Leave failed validation files for inspection. The shared bootstrap verifies
    and extracts the pinned interpreter before its first command, so even the
    guardian can use the independent bundled Python rather than Core's launcher.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = Arguments()
    parser.parse_args(namespace=arguments)
    output = arguments.output.resolve()
    deadline = time.monotonic() + 20 * 60

    def checkpoint() -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError("native candidate bootstrap exceeded its total deadline")

    runner = NativeInstallCommandRunner(
        output.parent / "installer-diagnostics",
        checkpoint=checkpoint,
        # PythonArchivePin's closed executable path is shared by all three
        # targets. It is established by extraction before any command runs.
        guardian_interpreter=output / "interpreter/python/bin/python3.12",
    )
    try:
        native = install_fresh_runtime(
            arguments.wheel.resolve(),
            output,
            arguments.sha256,
            run=runner,
            checkpoint=checkpoint,
        )
        print(native)
    finally:
        runner.drain()


if __name__ == "__main__":
    main()
