"""Windows installs the published bindings wheel of the current version."""

import tomllib
from pathlib import Path
from typing import cast

import pytest

import skulk

_BINDINGS = "skulk_pyo3_bindings"


def _repo_root() -> Path:
    root = Path(skulk.__file__).resolve().parents[2]
    if not (root / "rust" / _BINDINGS / "pyproject.toml").is_file():
        pytest.skip("not running from a source checkout")
    return root


def _table(path: Path) -> dict[str, object]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _windows_requirement(root: Path) -> str:
    project = cast(dict[str, object], _table(root / "pyproject.toml")["project"])
    dependencies = cast(list[str], project["dependencies"])
    windows = [
        dependency
        for dependency in dependencies
        if dependency.startswith(_BINDINGS) and "== 'win32'" in dependency
    ]
    assert len(windows) == 1, windows
    return windows[0]


def test_windows_pin_matches_the_bindings_version() -> None:
    """A bindings bump must not leave Windows on the previous wheel.

    The bindings version is the rollout forcing function for wire changes;
    Windows nodes get it only through the pin, because they never build.
    """
    root = _repo_root()
    bindings = cast(
        dict[str, object],
        _table(root / "rust" / _BINDINGS / "pyproject.toml")["project"],
    )
    version = cast(str, bindings["version"])

    requirement = _windows_requirement(root).split(";", 1)[0].strip()

    assert requirement == f"{_BINDINGS}=={version}"


def test_windows_bindings_come_from_the_foxlight_index() -> None:
    """Windows never builds the bindings; every other platform builds from rust/."""
    root = _repo_root()
    tool = cast(dict[str, object], _table(root / "pyproject.toml")["tool"])
    uv = cast(dict[str, object], tool["uv"])
    sources = cast(dict[str, object], uv["sources"])
    entries = cast(list[dict[str, object]], sources[_BINDINGS])

    by_marker = {cast(str, entry["marker"]): entry for entry in entries}

    assert by_marker["sys_platform == 'win32'"].get("index") == "foxlight"
    assert by_marker["sys_platform != 'win32'"].get("path") == f"rust/{_BINDINGS}"
