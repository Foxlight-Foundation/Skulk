"""Run npm from Skulk's cross-platform Node.js wheel."""

from __future__ import annotations

import os
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import nodejs_wheel


def _write_node_launcher(
    launcher_path: Path,
    node_binary: Path,
    javascript_entrypoint: Path,
) -> None:
    """Write an executable npm-compatible launcher around bundled Node."""

    launcher_path.write_text(
        "#!/bin/sh\n"
        f"exec {shlex.quote(str(node_binary))} "
        f"{shlex.quote(str(javascript_entrypoint))} \"$@\"\n"
    )
    launcher_path.chmod(0o700)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run bundled npm with its Node binary visible to lifecycle scripts.

    Args:
        arguments: npm arguments. Defaults to the process command line.

    Returns:
        The npm process exit status.

    Side effects:
        Prepends the bundled binary directory to ``PATH`` for npm child
        processes such as esbuild and Vite.
    """

    package_file = nodejs_wheel.__file__
    if package_file is None:
        raise RuntimeError("nodejs_wheel package has no filesystem location")
    package_directory = Path(package_file).resolve().parent
    binary_directory = package_directory / "bin"
    node_binary = binary_directory / "node"
    npm_directory = package_directory / "lib" / "node_modules" / "npm" / "bin"
    resolved_arguments = list(arguments) if arguments is not None else sys.argv[1:]
    current_path = os.environ.get("PATH", "")

    # The wheel's Python API correctly targets npm-cli.js, but its raw
    # ``bin/npm`` launcher has an invalid relative path. npm lifecycle scripts
    # can invoke npm again, so expose launchers that preserve the working Python
    # API's behavior instead of placing that broken raw launcher on PATH.
    with TemporaryDirectory(prefix="skulk-node-") as shim_directory_name:
        shim_directory = Path(shim_directory_name)
        (shim_directory / "node").symlink_to(node_binary)
        _write_node_launcher(
            shim_directory / "npm",
            node_binary,
            npm_directory / "npm-cli.js",
        )
        _write_node_launcher(
            shim_directory / "npx",
            node_binary,
            npm_directory / "npx-cli.js",
        )
        os.environ["PATH"] = (
            f"{shim_directory}{os.pathsep}{current_path}"
            if current_path
            else str(shim_directory)
        )
        return int(nodejs_wheel.npm(resolved_arguments))


if __name__ == "__main__":
    raise SystemExit(main())
