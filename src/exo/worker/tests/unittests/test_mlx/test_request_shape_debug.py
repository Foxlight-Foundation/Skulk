import pytest

from exo.worker.engines.mlx import utils_mlx


@pytest.mark.parametrize("env_name", ["SKULK_TRACE_REQUEST_SHAPES", "EXO_TRACE_REQUEST_SHAPES"])
def test_request_shape_debug_blank_env_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
) -> None:
    monkeypatch.delenv("SKULK_TRACE_REQUEST_SHAPES", raising=False)
    monkeypatch.delenv("EXO_TRACE_REQUEST_SHAPES", raising=False)
    monkeypatch.setenv(env_name, "")

    assert utils_mlx._request_shape_debug_enabled() is False
