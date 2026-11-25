"""Ensure the raw HTML check catches duplicate and wrong-channel metadata."""

from pathlib import Path

import pytest

from scripts.check_docs_canonicals import check


def test_checks_matching_next_page_and_excludes_typedoc(tmp_path: Path) -> None:
    """A guide must match its channel; the non-mirrored reference is separate."""
    (tmp_path / "install").mkdir()
    canonical = "https://docs.foxlight.ai/skulk/next/install/"
    (tmp_path / "install/index.html").write_text(
        f'<link rel="canonical" href="{canonical}"><meta property="og:url" content="{canonical}">'
    )
    (tmp_path / "typedoc").mkdir()
    (tmp_path / "typedoc/index.html").write_text("Unmirrored reference")
    assert check(tmp_path, "/Skulk/next/") == 1


def test_checks_flat_html_routes_ending_in_index(tmp_path: Path) -> None:
    """A document name ending in index is different from a directory index."""
    canonical = "https://docs.foxlight.ai/skulk/model-index/"
    (tmp_path / "model-index.html").write_text(
        f'<link rel="canonical" href="{canonical}"><meta property="og:url" content="{canonical}">'
    )
    assert check(tmp_path, "/Skulk/") == 1


@pytest.mark.parametrize("extra", ["", '<link rel="canonical" href="https://other.example.com/">'])
def test_rejects_legacy_or_duplicate_metadata(tmp_path: Path, extra: str) -> None:
    """A superficially successful page cannot keep the old canonical host."""
    expected = "https://docs.foxlight.ai/skulk/"
    canonical = expected if extra else "https://foxlight-foundation.github.io/Skulk/"
    (tmp_path / "index.html").write_text(
        f'<link rel="canonical" href="{canonical}">{extra}'
        f'<meta property="og:url" content="{expected}">'
    )
    with pytest.raises(ValueError, match="metadata"):
        check(tmp_path, "/Skulk/")
