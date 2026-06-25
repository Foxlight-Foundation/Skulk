from collections.abc import Callable
from typing import cast

import pytest

from skulk.worker.engines.mlx import utils_mlx as utils_mlx_module


def _request_shape_debug_enabled() -> bool:
    module_dict = cast(dict[str, object], utils_mlx_module.__dict__)
    return cast(Callable[[], bool], module_dict["_request_shape_debug_enabled"])()


def test_request_shape_debug_blank_env_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKULK_TRACE_REQUEST_SHAPES", "")

    assert _request_shape_debug_enabled() is False


def test_legacy_exo_request_shapes_env_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The EXO_ deprecation runway is gone (#324): a legacy EXO_ env must not be
    # honored even when the SKULK_ name is unset.
    monkeypatch.delenv("SKULK_TRACE_REQUEST_SHAPES", raising=False)
    monkeypatch.setenv("EXO_TRACE_REQUEST_SHAPES", "1")

    assert _request_shape_debug_enabled() is False
