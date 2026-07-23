# Copyright 2026 Foxlight Foundation

"""Tests for MLX-native image processor discovery."""

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from PIL import Image

from skulk.shared.models.model_cards import VisionCardConfig
from skulk.shared.types.common import ModelId
from skulk.worker.engines.mlx import vision as vision_module
from skulk.worker.engines.mlx.vision import VisionEncoder


class _FakeImageProcessor:
    """Sentinel image processor returned by a fake family processor."""


class _FakeCombinedProcessor:
    """Minimal MLX-VLM combined processor with an image processor member."""

    image_processor = _FakeImageProcessor()


class _FakeFamilyProcessor:
    """Minimal family processor exposing the Hugging Face loader contract."""

    seen_repo: str | None = None
    seen_trust_remote_code: bool | None = None

    @classmethod
    def from_pretrained(
        cls,
        repo: str,
        **kwargs: object,
    ) -> _FakeCombinedProcessor:
        cls.seen_repo = repo
        cls.seen_trust_remote_code = cast(bool, kwargs["trust_remote_code"])
        return _FakeCombinedProcessor()


@pytest.mark.parametrize(
    ("model_type", "family_module_name"),
    [
        ("qwen3_5", "mlx_vlm.models.qwen3_5"),
        ("qwen3_vl", "mlx_vlm.models.qwen3_vl.processing_qwen3_vl"),
    ],
)
def test_processor_loader_uses_mlx_vlm_family_without_torchvision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_type: str,
    family_module_name: str,
) -> None:
    """Qwen processors should load from MLX-VLM before Transformers."""
    (tmp_path / "config.json").write_text(
        '{"vision_config": {"model_type": "qwen"}}',
        encoding="utf-8",
    )

    def _model_path(*_args: object) -> Path:
        return tmp_path

    def _no_upstream_processor(_repo: str) -> None:
        return None

    monkeypatch.setattr(vision_module, "build_model_path", _model_path)
    monkeypatch.setattr(vision_module, "load_image_processor", _no_upstream_processor)

    def _fake_import_module(module_name: str) -> object:
        if module_name == family_module_name:
            return SimpleNamespace(Qwen3VLProcessor=_FakeFamilyProcessor)
        raise ModuleNotFoundError(name=module_name)

    monkeypatch.setattr(vision_module.importlib, "import_module", _fake_import_module)

    def _fail_transformers_loader(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("MLX-VLM family processor should load first")

    monkeypatch.setattr(
        vision_module,
        "AutoImageProcessor",
        SimpleNamespace(from_pretrained=_fail_transformers_loader),
    )

    encoder = VisionEncoder(
        VisionCardConfig(
            image_token_id=1,
            model_type=model_type,
            weights_repo="mlx-community/example",
        ),
        ModelId("mlx-community/example"),
    )
    encoder.ensure_processor_loaded()

    assert isinstance(encoder.processor, _FakeImageProcessor)
    assert _FakeFamilyProcessor.seen_repo == str(tmp_path)
    assert _FakeFamilyProcessor.seen_trust_remote_code is True


def test_processor_loader_reports_transformers_torchvision_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A missing torchvision dependency should become a clear load failure."""
    (tmp_path / "config.json").write_text(
        '{"vision_config": {"model_type": "unknown"}}',
        encoding="utf-8",
    )

    def _model_path(*_args: object) -> Path:
        return tmp_path

    def _no_upstream_processor(_repo: str) -> None:
        return None

    def _missing_module(module_name: str) -> object:
        raise ModuleNotFoundError(name=module_name)

    monkeypatch.setattr(vision_module, "build_model_path", _model_path)
    monkeypatch.setattr(vision_module, "load_image_processor", _no_upstream_processor)
    monkeypatch.setattr(
        vision_module.importlib,
        "import_module",
        _missing_module,
    )

    def _raise_torchvision_import(*_args: object, **_kwargs: object) -> object:
        raise ImportError("AutoImageProcessor requires the Torchvision library")

    monkeypatch.setattr(
        vision_module,
        "AutoImageProcessor",
        SimpleNamespace(from_pretrained=_raise_torchvision_import),
    )

    encoder = VisionEncoder(
        VisionCardConfig(
            image_token_id=1,
            model_type="unknown",
            weights_repo="mlx-community/example",
        ),
        ModelId("mlx-community/example"),
    )

    with pytest.raises(RuntimeError, match="requires the Torchvision library"):
        encoder.ensure_processor_loaded()


def test_gemma3n_processor_uses_configured_pil_backend_without_torchvision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Gemma 3n should produce its configured image tensor on the PIL backend."""
    (tmp_path / "config.json").write_text(
        '{"vision_config": {"model_type": "gemma3n"}}',
        encoding="utf-8",
    )
    (tmp_path / "preprocessor_config.json").write_text(
        """{
          "do_convert_rgb": true,
          "do_normalize": false,
          "do_rescale": true,
          "do_resize": true,
          "image_mean": [0.5, 0.5, 0.5],
          "image_processor_type": "SiglipImageProcessorFast",
          "image_std": [0.5, 0.5, 0.5],
          "resample": 2,
          "rescale_factor": 0.00392156862745098,
          "size": {"height": 768, "width": 768}
        }""",
        encoding="utf-8",
    )

    def _model_path(*_args: object) -> Path:
        return tmp_path

    def _no_upstream_processor(_repo: str) -> None:
        return None

    def _fail_transformers_loader(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Gemma 3n should use the concrete PIL backend")

    monkeypatch.setattr(vision_module, "build_model_path", _model_path)
    monkeypatch.setattr(vision_module, "load_image_processor", _no_upstream_processor)
    monkeypatch.setattr(
        vision_module,
        "AutoImageProcessor",
        SimpleNamespace(from_pretrained=_fail_transformers_loader),
    )

    encoder = VisionEncoder(
        VisionCardConfig(
            image_token_id=262145,
            model_type="gemma3n",
            weights_repo="mlx-community/example",
        ),
        ModelId("mlx-community/example"),
    )
    encoder.ensure_processor_loaded()

    processor = encoder.processor
    assert processor is not None
    assert type(processor).__name__ == "SiglipImageProcessorPil"
    raw = cast(
        Mapping[str, object],
        processor(images=[Image.new("RGB", (32, 24))], return_tensors="np"),
    )
    pixel_values = cast(np.ndarray[tuple[int, ...], np.dtype[np.float32]], raw["pixel_values"])
    assert pixel_values.shape == (1, 3, 768, 768)
    assert pixel_values.dtype == np.float32
