# Copyright 2026 Foxlight Foundation

"""Tests for selecting the native MLX-VLM model loader."""

from pathlib import Path
from types import SimpleNamespace
from typing import Callable, cast

import mlx.core as mx
import mlx.nn as nn
import pytest

from skulk.worker.engines.mlx import utils_mlx


class _FakeVlmModel(nn.Module):
    """Minimal native VLM returned by the mocked upstream loader."""


def test_prefer_vlm_bypasses_text_only_mlx_lm_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A vision placement must retain the checkpoint's native vision tower."""
    inner_model = _FakeVlmModel()
    seen: dict[str, object] = {}

    def _fake_vlm_load_model(model_path: Path, **kwargs: object) -> nn.Module:
        seen["model_path"] = model_path
        seen["kwargs"] = kwargs
        return inner_model

    def _fake_import_module(module_name: str) -> object:
        assert module_name == "mlx_vlm.utils"
        return SimpleNamespace(load_model=_fake_vlm_load_model)

    def _fail_mlx_lm_loader(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("prefer_vlm must bypass the text-only MLX-LM loader")

    def _identity_model(model: object) -> object:
        return model

    monkeypatch.setattr(utils_mlx, "import_module", _fake_import_module)
    monkeypatch.setattr(utils_mlx, "_mlx_lm_load_model", _fail_mlx_lm_loader)
    monkeypatch.setattr(utils_mlx, "_patch_gemma4_native_vision", _identity_model)

    model_path = Path("/models/qwen-vlm")
    model, tokenizer = utils_mlx.load_model(
        model_path,
        prefer_vlm=True,
        lazy=True,
        strict=False,
    )

    assert object.__getattribute__(model, "_inner") is inner_model
    assert tokenizer is None
    assert seen == {
        "model_path": model_path,
        "kwargs": {"lazy": True, "strict": False},
    }


def test_converted_qwen_norm_guard_preserves_mlx_norm_weights() -> None:
    """The 0.6.4 backport must undo only the duplicate converted norm shift."""

    def _sanitize_key(key: str) -> str:
        return key.replace("model.language_model.", "language_model.model.")

    def _upstream_sanitizer(
        _model: object,
        weights: dict[str, mx.array],
    ) -> dict[str, mx.array]:
        sanitized: dict[str, mx.array] = {}
        for key, value in weights.items():
            if key.endswith("input_layernorm.weight"):
                value += 1
            sanitized[_sanitize_key(key)] = value
        return sanitized

    guard_factory = cast(
        Callable[
            [
                Callable[[object, dict[str, mx.array]], dict[str, mx.array]],
                Callable[[str], str],
            ],
            Callable[[object, dict[str, mx.array]], dict[str, mx.array]],
        ],
        vars(utils_mlx)["_guard_converted_qwen_norm_sanitizer"],
    )
    guarded = guard_factory(
        _upstream_sanitizer,
        _sanitize_key,
    )
    converted_key = "language_model.model.layers.0.input_layernorm.weight"
    source_key = "model.language_model.layers.0.input_layernorm.weight"
    converted = guarded(
        object(),
        {
            converted_key: mx.ones((2,)),
            "language_model.model.layers.0.mlp.weight": mx.ones((2, 2)),
        },
    )
    source = guarded(object(), {source_key: mx.zeros((2,))})

    assert converted[converted_key].tolist() == [1.0, 1.0]
    assert source[converted_key].tolist() == [1.0, 1.0]
    assert converted["language_model.model.layers.0.mlp.weight"].tolist() == [
        [1.0, 1.0],
        [1.0, 1.0],
    ]


def test_vlm_loader_restores_qwen_sanitizer_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed model load must not leave the process-wide class patched."""

    class _GuardedModel:
        @staticmethod
        def sanitize(_model: object, weights: object) -> object:
            return weights

    original = cast(object, vars(_GuardedModel)["sanitize"])

    def _replacement(_model: object, weights: object) -> object:
        return weights

    def _fake_install(_model_path: Path) -> tuple[type[object], object]:
        type.__setattr__(_GuardedModel, "sanitize", _replacement)
        return _GuardedModel, original

    def _fail_load(_model_path: Path, **_kwargs: object) -> nn.Module:
        raise RuntimeError("simulated load failure")

    def _fake_import_module(_name: str) -> object:
        return SimpleNamespace(load_model=_fail_load)

    monkeypatch.setattr(
        utils_mlx,
        "import_module",
        _fake_import_module,
    )
    monkeypatch.setattr(
        utils_mlx,
        "_install_converted_qwen_norm_guard",
        _fake_install,
    )

    load_vlm_model = cast(
        Callable[[Path], tuple[nn.Module, object]],
        vars(utils_mlx)["_load_vlm_model"],
    )
    with pytest.raises(RuntimeError, match="simulated load failure"):
        load_vlm_model(Path("/models/qwen-vlm"))

    assert vars(_GuardedModel)["sanitize"] is original
