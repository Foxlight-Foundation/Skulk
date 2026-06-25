import pytest

from skulk.worker.engines.mlx import auto_parallel


def test_mlx_hang_debug_blank_env_value_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKULK_MLX_HANG_DEBUG", "")

    assert auto_parallel._mlx_hang_debug_enabled() is False  # pyright: ignore[reportPrivateUsage]


def test_legacy_exo_mlx_hang_debug_env_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The EXO_ deprecation runway is gone (#324): a legacy EXO_ env must not be
    # honored even when the SKULK_ name is unset.
    monkeypatch.delenv("SKULK_MLX_HANG_DEBUG", raising=False)
    monkeypatch.setenv("EXO_MLX_HANG_DEBUG", "1")

    assert auto_parallel._mlx_hang_debug_enabled() is False  # pyright: ignore[reportPrivateUsage]
