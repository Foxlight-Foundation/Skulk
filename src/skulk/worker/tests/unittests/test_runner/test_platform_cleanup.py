# pyright: reportPrivateUsage=false
"""Linux runner diagnostics and shutdown must not load Metal's native engine."""

from unittest.mock import Mock, patch

import pytest

from skulk.worker.runner import bootstrap, diagnostics


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_non_apple_cleanup_does_not_import_mlx(
    monkeypatch: pytest.MonkeyPatch, platform: str,
) -> None:
    """Unused Metal cleanup must never initialize a native MLX extension."""
    monkeypatch.setattr(bootstrap.sys, "platform", platform)
    collect = Mock(return_value=0)
    monkeypatch.setattr(bootstrap.gc, "collect", collect)
    with patch("builtins.__import__", side_effect=AssertionError("unexpected engine import")) as import_native:
        bootstrap._release_metal_resources()
    import_native.assert_not_called()
    collect.assert_called_once_with()


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_non_apple_memory_diagnostics_do_not_import_mlx(
    monkeypatch: pytest.MonkeyPatch, platform: str,
) -> None:
    """A diagnostic snapshot must remain inert outside Metal's platform."""
    monkeypatch.setattr(diagnostics.sys, "platform", platform)
    import_native = Mock(side_effect=AssertionError("unexpected engine import"))
    monkeypatch.setattr(diagnostics, "import_module", import_native)
    assert diagnostics.capture_mlx_memory_snapshot() is None
    import_native.assert_not_called()
