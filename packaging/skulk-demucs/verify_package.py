"""Install a candidate in a fresh private directory using only its sealed bytes.

This validation tool is not Core's engine provisioner. It cannot publish a package,
reuse an installation or advertise ready capacity. Model weights remain separate.
"""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence
from pathlib import Path

from package_inputs import (
    extract_python as extract_python,
)
from package_inputs import (
    install_fresh_runtime,
)
from package_inputs import (
    safe_member as safe_member,
)
from package_inputs import (
    unpack_candidate as unpack_candidate,
)
from package_inputs import (
    verify_site_packages as verify_site_packages,
)


def _run(
    arguments: Sequence[str],
    *,
    environment: dict[str, str],
    timeout_seconds: int,
    capture: bool,
) -> str:
    result = subprocess.run(
        arguments,
        env=environment,
        timeout=timeout_seconds,
        check=True,
        capture_output=capture,
        text=True,
    )
    return result.stdout if capture else ""


def install_candidate(wheel: Path, output: Path, expected_sha256: str) -> Path:
    """Validate one fresh candidate using the shared sealed offline bootstrap."""
    return install_fresh_runtime(wheel, output, expected_sha256, run=_run)


class Arguments(argparse.Namespace):
    """Expected package identity and a fresh explicitly owned test destination."""

    wheel: Path
    sha256: str
    output: Path


def main() -> None:
    """Install sealed candidate bytes for native tests without network resolution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = Arguments()
    parser.parse_args(namespace=arguments)
    print(
        install_candidate(
            arguments.wheel.resolve(), arguments.output.resolve(), arguments.sha256
        )
    )


if __name__ == "__main__":
    main()
