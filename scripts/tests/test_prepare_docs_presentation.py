"""Ensure a presentation repair does not alter pinned product documentation."""

from pathlib import Path

import pytest

from scripts.prepare_docs_presentation import FILES, prepare


def test_prepares_older_channel_without_rewriting_content(tmp_path: Path) -> None:
    """A stable checkout can receive metadata while its docs remain byte-identical."""
    website = tmp_path / "website"
    (website / "docs").mkdir(parents=True)
    (website / "docusaurus.config.ts").write_text("export default {};\n")
    guide = website / "docs/intro.md"
    guide.write_text("# Stable product\n")
    prepare(tmp_path)
    assert guide.read_text() == "# Stable product\n"
    assert all((website / name).is_file() for name in FILES)


def test_rejects_a_missing_product_checkout(tmp_path: Path) -> None:
    """An incorrect output directory cannot become a synthetic product source."""
    with pytest.raises(ValueError, match="checkout"):
        prepare(tmp_path)
