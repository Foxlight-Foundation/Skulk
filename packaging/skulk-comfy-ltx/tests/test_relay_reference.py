"""Pinned-source attention conformance, also mandatory in wheel publication CI."""

from pathlib import Path

import pytest

from scripts.check_comfy_ltx_relay import check_relay_reference


def test_pinned_comfy_attention_conditioning_grid_and_patch_lifecycle() -> None:
    """Keep source contracts distinct from model-weight or GPU qualification."""
    source = Path(__file__).resolve().parents[3] / ".runtime/comfy"
    if not source.is_dir():
        pytest.skip("The wheel workflow supplies the immutable Comfy reference checkout")
    report = check_relay_reference(source)
    assert report["pass_all"] is True
    assert report["actual_clone_install_restore"] is True
    assert report["attention_max_absolute_error"] == 0.0
