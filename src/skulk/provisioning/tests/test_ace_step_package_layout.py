"""Native ACE uses the sealed audio substrate without weakening Demucs identity."""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from skulk.provisioning.demucs_package import (
    inspect_candidate,
    read_python_catalog,
    unpack_candidate,
)


def _package(
    root: Path,
    *,
    manifest_module: str,
    source_repository: str,
    duplicate_manifest: bool = False,
) -> tuple[Path, str]:
    catalog = read_python_catalog(
        Path(__file__).parents[4] / "packaging/skulk-ace-step"
    )
    pin = catalog.for_target("macosx_14_0_arm64")
    module = (
        "skulk_ace_step"
        if source_repository.endswith("ACE-Step-1.5.git")
        else "skulk_demucs"
    )
    members = {f"{module}/__init__.py": b'"""Independent fixture payload."""\n'}
    body = dict(
        protocol_version=1,
        python="3.12.13",
        python_archive=pin.model_dump(mode="json"),
        adapter_revision="1" * 40,
        source_revision="2" * 40,
        source_repository=source_repository,
        platform_tag=pin.platform_tag,
        variant="cpu",
        members={
            name: hashlib.sha256(data).hexdigest() for name, data in members.items()
        },
        features=["offline_loader"],
    )
    identity = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = json.dumps(body | {"payload_sha256": identity}).encode()
    members[f"{manifest_module}/build.json"] = manifest
    if duplicate_manifest:
        other = (
            "skulk_demucs" if manifest_module == "skulk_ace_step" else "skulk_ace_step"
        )
        members[f"{other}/build.json"] = manifest
    for name in ("METADATA", "WHEEL", "RECORD"):
        members[f"{module}_cpu-0.1.0.dist-info/{name}"] = b"fixture\n"
    package = root / "fixture.whl"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in sorted(members.items()):
            item = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            item.external_attr = 0o100644 << 16
            archive.writestr(item, data)
    return package, hashlib.sha256(package.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("module", "repository"),
    [
        ("skulk_ace_step", "https://github.com/ace-step/ACE-Step-1.5.git"),
        ("skulk_demucs", "https://github.com/adefossez/demucs.git"),
    ],
)
def test_exact_audio_package_layout_is_verified_before_extraction(
    tmp_path: Path, module: str, repository: str
) -> None:
    """Both known layouts retain the complete package and payload hash checks."""
    package, digest = _package(
        tmp_path, manifest_module=module, source_repository=repository
    )
    manifest = inspect_candidate(package, digest)
    assert manifest.source_repository == repository
    target = tmp_path / "unpacked"
    assert unpack_candidate(package, target, digest) == manifest
    assert (
        target / module / "__init__.py"
    ).read_bytes() == b'"""Independent fixture payload."""\n'


@pytest.mark.parametrize(
    ("module", "repository"),
    [
        ("skulk_demucs", "https://github.com/ace-step/ACE-Step-1.5.git"),
        ("skulk_ace_step", "https://github.com/adefossez/demucs.git"),
    ],
)
def test_native_manifest_cannot_select_the_other_package_layout(
    tmp_path: Path, module: str, repository: str
) -> None:
    """A supported repository does not authorize an unrelated package namespace."""
    package, digest = _package(
        tmp_path, manifest_module=module, source_repository=repository
    )
    with pytest.raises(ValueError, match="manifest identity or member inventory"):
        inspect_candidate(package, digest)


def test_mixed_audio_manifests_are_rejected(tmp_path: Path) -> None:
    """Refuse ambiguous build ownership before considering either manifest."""
    package, digest = _package(
        tmp_path,
        manifest_module="skulk_ace_step",
        source_repository="https://github.com/ace-step/ACE-Step-1.5.git",
        duplicate_manifest=True,
    )
    with pytest.raises(ValueError, match="exactly one supported audio build manifest"):
        inspect_candidate(package, digest)
