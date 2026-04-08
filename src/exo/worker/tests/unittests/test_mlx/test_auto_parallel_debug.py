import pytest

from exo.worker.engines.mlx import auto_parallel


def test_mlx_hang_debug_blank_env_value_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKULK_MLX_HANG_DEBUG", "")

    assert auto_parallel._mlx_hang_debug_enabled() is False  # pyright: ignore[reportPrivateUsage]
