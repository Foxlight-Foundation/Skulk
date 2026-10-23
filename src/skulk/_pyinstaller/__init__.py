"""Build-time resource closure for Skulk's public preservation dependency."""

from importlib.metadata import distribution
from pathlib import Path


def get_hook_dirs() -> list[str]:
    """Expose Skulk's dependency hooks to ordinary PyInstaller builds."""
    return [str(Path(__file__).parent)]


def preservation_data_files() -> list[tuple[str, str]]:
    """Collect schemas, converter source evidence, version metadata and licenses.

    Return PyInstaller source/destination pairs without importing PyInstaller
    into a runtime. Copy only needed public metadata, excluding build-machine
    direct URLs and installer state. Fail the build if the required closure is
    missing instead of creating a binary with a broken verifier.
    """
    package = distribution("foxlight-model-preservation")
    resources: list[tuple[str, str]] = []
    metadata: list[tuple[str, str]] = []
    licenses: list[tuple[str, str]] = []
    included_licenses: set[str] = set()
    adapter_sources: list[tuple[str, str]] = []
    for entry in package.files or ():
        source = str(package.locate_file(entry))
        target = str(entry.parent)
        if entry.parts[0] == "foxlight_model_preservation" and (
            entry.suffix == ".json"
            or entry.as_posix()
            == "foxlight_model_preservation/resources/kubo-NOTICE.txt"
        ):
            resources.append((source, target))
        if entry.as_posix() == "foxlight_model_preservation/pmf_preservation.py":
            # Converter receipts hash this exact source as an opaque resource.
            # PYZ bytecode alone cannot satisfy importlib.resources.read_bytes.
            adapter_sources.append((source, target))
        if entry.name == "METADATA" and entry.parent.name.endswith(".dist-info"):
            metadata.append((source, target))
        if (
            len(entry.parts) > 2
            and entry.parts[0].endswith(".dist-info")
            and entry.parts[1] == "licenses"
        ):
            licenses.append((source, target))
            included_licenses.add(Path(*entry.parts[2:]).as_posix())
    if not resources or len(metadata) != 1 or len(adapter_sources) != 1:
        raise FileNotFoundError(
            "installed preservation core lacks schemas, converter source or metadata"
        )
    if not {"kubo_v0_43_1.json", "kubo-NOTICE.txt"}.issubset(
        {Path(source).name for source, _ in resources}
    ):
        raise FileNotFoundError(
            "installed preservation core lacks native acquisition pins or notices"
        )
    declared_licenses = set(package.metadata.get_all("License-File") or ())
    if not declared_licenses.issubset(included_licenses):
        raise FileNotFoundError(
            "installed preservation core lacks a declared license file"
        )
    return resources + adapter_sources + metadata + licenses
