"""Verify a bindings source archive carries its networking backport and typing."""

import argparse
import hashlib
import json
import subprocess
import tarfile
import tempfile
import tomllib
from pathlib import Path


def check_archive(archive: Path, source_root: Path) -> dict[str, str | int]:
    """Extract and resolve an archive in temporary storage, then verify its source.

    Args:
        archive: Maturin-generated source distribution to validate.
        source_root: Checkout whose locked dependency and source bytes must ship.

    Returns:
        Archive digest, size and number of verified source/provenance members.

    Raises:
        RuntimeError: The archive omits, changes or externally resolves the backport.
        subprocess.CalledProcessError: Cargo cannot resolve the extracted archive.

    Cargo may populate its normal dependency cache. Temporary extraction is removed
    on return; this function never builds, installs or starts the bindings.
    """
    expected_vendor = source_root / "rust/vendor/libp2p-gossipsub"
    expected_files = sorted(expected_vendor.rglob("*.rs"))
    expected_files.extend(
        expected_vendor / name
        for name in ("FOXLight-PATCH.md", "upstream-provenance.json")
    )
    if not expected_files or not (expected_vendor / "src/behaviour.rs").is_file():
        raise RuntimeError(
            "The source checkout does not contain the networking backport"
        )
    if archive.stat().st_size > 64 * 1024**2:
        raise RuntimeError("The bindings source archive exceeds the validation bound")

    with tempfile.TemporaryDirectory(prefix="skulk-bindings-sdist-") as temporary:
        extraction = Path(temporary).resolve()
        with tarfile.open(archive) as package:
            if sum(member.size for member in package.getmembers()) > 128 * 1024**2:
                raise RuntimeError(
                    "The expanded source archive exceeds the validation bound"
                )
            package.extractall(extraction, filter="data")
        projects = [
            path
            for path in extraction.glob("*/pyproject.toml")
            if tomllib.loads(path.read_text()).get("project", {}).get("name")
            == "skulk_pyo3_bindings"
        ]
        if len(projects) != 1:
            raise RuntimeError("Expected one bindings project in the source archive")
        project = projects[0]
        settings = tomllib.loads(project.read_text())
        expected_stub = (
            source_root / "rust/skulk_pyo3_bindings/skulk_pyo3_bindings.pyi"
        ).read_bytes()
        archived_stub = project.parent / "skulk_pyo3_bindings.pyi"
        if not archived_stub.is_file() or archived_stub.read_bytes() != expected_stub:
            raise RuntimeError(
                "The source archive omitted or changed its root typing stub"
            )
        manifest = (
            project.parent
            / settings.get("tool", {})
            .get("maturin", {})
            .get("manifest-path", "Cargo.toml")
        ).resolve()
        if not manifest.is_file() or not manifest.is_relative_to(extraction):
            raise RuntimeError(
                "The archived bindings manifest must resolve inside the archive"
            )
        expected_lock = (source_root / "Cargo.lock").read_bytes()
        archived_lock = project.parent / "Cargo.lock"
        if archived_lock.read_bytes() != expected_lock:
            raise RuntimeError(
                "The source archive changed the production dependency lock"
            )
        metadata = json.loads(
            subprocess.check_output(
                [
                    "cargo",
                    "metadata",
                    "--locked",
                    "--format-version",
                    "1",
                    "--manifest-path",
                    str(manifest),
                ],
                cwd=extraction,
                timeout=180,
            )
        )
        libraries = [
            library
            for library in metadata["packages"]
            if library["name"] == "libp2p-gossipsub"
        ]
        if len(libraries) != 1 or libraries[0]["source"] is not None:
            raise RuntimeError(
                "The archive must resolve exactly one local gossipsub backport"
            )
        actual_vendor = Path(libraries[0]["manifest_path"]).resolve().parent
        if not actual_vendor.is_relative_to(extraction):
            raise RuntimeError(
                "The backport resolved outside the extracted source archive"
            )
        for original in expected_files:
            relative = original.relative_to(expected_vendor)
            delivered = actual_vendor / relative
            if (
                not delivered.is_file()
                or delivered.read_bytes() != original.read_bytes()
            ):
                raise RuntimeError(
                    f"The archive omitted or changed backport member {relative}"
                )
        if archived_lock.read_bytes() != expected_lock:
            raise RuntimeError(
                "Archive resolution changed the production dependency lock"
            )

    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "archive": archive.name,
        "bytes": archive.stat().st_size,
        "sha256": digest,
        "verified_source_members": len(expected_files),
        "typing_stub_bytes": len(expected_stub),
    }


def main() -> None:
    """Validate the only archive in a directory and print its bounded JSON receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    arguments = parser.parse_args()
    archives = sorted(arguments.directory.glob("*.tar.gz"))
    if len(archives) != 1:
        parser.error("the directory must contain exactly one source archive")
    print(json.dumps(check_archive(archives[0], arguments.source_root.resolve())))


if __name__ == "__main__":
    main()
