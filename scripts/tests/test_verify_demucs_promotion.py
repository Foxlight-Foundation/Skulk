"""Separator publication must include all fixed architectures and unchanged bytes."""

import hashlib
from pathlib import Path

import pytest

import scripts.verify_demucs_promotion as promotion
from scripts.verify_ace_step_promotion import verify_artifacts
from skulk.provisioning.demucs_inventory import DEMUCS_PACKAGES
from skulk.provisioning.demucs_runtime import DemucsWheel


def candidates(directory: Path) -> tuple[DemucsWheel, ...]:
    """Use actual platform names with independent tiny exact-content pins."""
    selected: list[DemucsWheel] = []
    for index, package in enumerate(DEMUCS_PACKAGES):
        parent = directory / f"artifact-{index}"
        parent.mkdir(parents=True)
        content = f"immutable separator fixture {index}".encode()
        (parent / package.filename).write_bytes(content)
        selected.append(
            DemucsWheel(
                platform_tag=package.platform_tag,
                filename=package.filename,
                size_bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
                payload_sha256=package.payload_sha256,
            )
        )
    return tuple(selected)


def test_separator_cli_uses_its_complete_fixed_family_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A wrapper cannot accidentally select the default ACE family or one platform."""
    packages = candidates(tmp_path)
    monkeypatch.setattr(promotion, "DEMUCS_PACKAGES", packages)
    monkeypatch.setattr("sys.argv", ["verify_demucs_promotion", str(tmp_path)])
    promotion.main()
    selected = tuple(Path(line).name for line in capsys.readouterr().out.splitlines())
    assert selected == tuple(sorted(package.filename for package in packages))
    assert len(selected) == 3


@pytest.mark.parametrize("index", range(3))
def test_each_missing_separator_architecture_refuses_publication(
    tmp_path: Path, index: int
) -> None:
    """An available Mac or Linux archive cannot hide an omitted target."""
    packages = candidates(tmp_path)
    (tmp_path / f"artifact-{index}" / packages[index].filename).unlink()
    with pytest.raises(ValueError, match="every fixed"):
        verify_artifacts(tmp_path, packages=packages)


@pytest.mark.parametrize("mutation", ["size", "digest", "extra"])
def test_separator_substitution_is_refused_before_any_paths_are_returned(
    tmp_path: Path, mutation: str
) -> None:
    """Exact filenames alone never authorize substituted or unreviewed payloads."""
    packages = candidates(tmp_path)
    original = tmp_path / "artifact-0" / packages[0].filename
    if mutation == "size":
        original.write_bytes(original.read_bytes() + b"extra")
    elif mutation == "digest":
        original.write_bytes(b"x" * original.stat().st_size)
    else:
        (tmp_path / "unreviewed.whl").write_bytes(b"unreviewed")
    with pytest.raises(ValueError):
        verify_artifacts(tmp_path, packages=packages)


@pytest.mark.parametrize("option", ["--sha256", "--platform", "--variant", "--package"])
def test_separator_cli_exposes_no_pin_or_profile_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, option: str
) -> None:
    """Publication is tied to reviewed inventory rather than dispatch-supplied pins."""
    monkeypatch.setattr(
        "sys.argv", ["verify_demucs_promotion", str(tmp_path), option, "unreviewed"]
    )
    with pytest.raises(SystemExit) as refusal:
        promotion.main()
    assert refusal.value.code == 2
