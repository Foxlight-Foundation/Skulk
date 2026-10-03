"""Verify publication identities against Git and reject incomplete exports."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.publish_docs_export import export


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Create an immutable product document with a complete built contract."""
    (tmp_path / "website/docs").mkdir(parents=True)
    (tmp_path / "website/docs/intro.md").write_text("# Product guide\n")
    (tmp_path / "website/sidebars.ts").write_text("export default {};\n")
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "website"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "Product docs",
        ],
        check=True,
        capture_output=True,
    )
    (tmp_path / "website/build").mkdir()
    (tmp_path / "website/build/index.html").write_text("Product guide")
    (tmp_path / "docs/generated").mkdir(parents=True)
    (tmp_path / "docs/generated/openapi.json").write_text(
        '{"openapi":"3.1.0","paths":{"/health":{"get":{}}}}\n'
    )
    return tmp_path


def test_export_binds_exact_source_and_contract(repository: Path) -> None:
    """A channel binds its own checkout, not the workflow trigger's revision."""
    export(repository, "stable", 42)
    manifest = json.loads((repository / "website/build/docs-export.json").read_text())
    assert (
        manifest["revision"]
        == subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"])
        .decode()
        .strip()
    )
    assert manifest["channel"] == "stable" and manifest["run_id"] == 42
    original = (repository / "docs/generated/openapi.json").read_bytes()
    assert (repository / "website/build/openapi.json").read_bytes() == original
    assert manifest["openapi_sha256"] == hashlib.sha256(original).hexdigest()
    assert (
        manifest["source_files"]["docs/intro.md"]
        == hashlib.sha256(
            (repository / "website/docs/intro.md").read_bytes()
        ).hexdigest()
    )


def test_dirty_documentation_cannot_claim_a_git_revision(repository: Path) -> None:
    """Local source edits cannot be presented as an immutable publication."""
    (repository / "website/docs/intro.md").write_text("Changed guide")
    with pytest.raises(ValueError, match="differs"):
        export(repository, "next", 43)
    assert not (repository / "website/build/docs-export.json").exists()


def test_failed_build_cannot_publish_an_export(repository: Path) -> None:
    """A contract alone does not qualify as a successfully built site."""
    (repository / "website/build/index.html").unlink()
    with pytest.raises(ValueError, match="build successfully"):
        export(repository, "next", 44)


def test_invalid_contract_cannot_publish_an_export(repository: Path) -> None:
    """An empty or corrupt API export leaves no publication manifest."""
    (repository / "docs/generated/openapi.json").write_text("{}")
    with pytest.raises(ValueError, match="complete generated"):
        export(repository, "next", 45)
