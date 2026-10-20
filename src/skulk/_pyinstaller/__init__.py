"""Build-time resource closure for Skulk's public preservation dependency."""

from importlib.metadata import distribution
from pathlib import Path


def get_hook_dirs() -> list[str]:
    """Expose Skulk's dependency hooks to ordinary PyInstaller builds."""
    return [str(Path(__file__).parent)]


def preservation_data_files() -> list[tuple[str, str]]:
    """Collect the installed core's JSON resources, version metadata and licenses.

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
    for entry in package.files or ():
        source = str(package.locate_file(entry))
        target = str(entry.parent)
        if entry.parts[0] == "foxlight_model_preservation" and entry.suffix == ".json":
            resources.append((source, target))
        if entry.name == "METADATA" and entry.parent.name.endswith(".dist-info"):
            metadata.append((source, target))
        if (
            len(entry.parts) > 2
            and entry.parts[0].endswith(".dist-info")
            and entry.parts[1] == "licenses"
        ):
            licenses.append((source, target))
            included_licenses.add(Path(*entry.parts[2:]).as_posix())
    if not resources or len(metadata) != 1:
        raise FileNotFoundError(
            "installed preservation core lacks schemas or distribution metadata"
        )
    declared_licenses = set(package.metadata.get_all("License-File") or ())
    if not declared_licenses.issubset(included_licenses):
        raise FileNotFoundError(
            "installed preservation core lacks a declared license file"
        )
    return resources + metadata + licenses
