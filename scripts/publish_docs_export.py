"""Export exact documentation build inputs for independently branded mirrors."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def git_bytes(repository: Path, *arguments: str) -> bytes:
    """Read immutable Git objects from the documentation build's checkout."""
    return subprocess.check_output(["git", "-C", str(repository), *arguments])


def export(repository: Path, channel: str, run_id: int) -> None:
    """Write a revision manifest and generated OpenAPI into a completed site.

    The public export contains documentation identities and the HTTP contract,
    never runtime configuration or product implementation. Consumers retrieve
    guide content from the exact public Git revision and verify its hashes.
    """
    if channel not in ("stable", "next") or run_id < 1:
        raise ValueError("Expected a product channel and positive workflow run ID")
    if subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "website/docs",
            "website/sidebars.ts",
        ],
        check=False,
    ).returncode:
        raise ValueError("Documentation source differs from the checked-out revision")
    site = repository / "website/build"
    if not (site / "index.html").is_file():
        raise ValueError("Documentation must build successfully before export")
    contract = (repository / "docs/generated/openapi.json").read_bytes()
    schema = json.loads(contract)
    if not schema.get("openapi", "").startswith("3.") or not schema.get("paths"):
        raise ValueError("Expected a complete generated OpenAPI 3 contract")
    revision = git_bytes(repository, "rev-parse", "HEAD").decode().strip()
    names = (
        git_bytes(repository, "ls-tree", "-rz", "--name-only", revision, "website/docs")
        .decode()
        .split("\0")
    )
    files = {
        name.removeprefix("website/"): hashlib.sha256(
            git_bytes(repository, "show", f"{revision}:{name}")
        ).hexdigest()
        for name in names
        if name
    }
    manifest = {
        "schema_version": 1,
        "product": "Skulk",
        "channel": channel,
        "revision": revision,
        "run_id": run_id,
        "openapi_sha256": hashlib.sha256(contract).hexdigest(),
        "sidebar_sha256": hashlib.sha256(
            git_bytes(repository, "show", f"{revision}:website/sidebars.ts")
        ).hexdigest(),
        "source_files": files,
    }
    (site / "openapi.json").write_bytes(contract)
    (site / "docs-export.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    """Export one checked-out channel after its successful documentation build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--channel", choices=("stable", "next"), required=True)
    parser.add_argument("--run-id", type=int, required=True)
    arguments = parser.parse_args()
    export(arguments.repository, arguments.channel, arguments.run_id)


if __name__ == "__main__":
    main()
