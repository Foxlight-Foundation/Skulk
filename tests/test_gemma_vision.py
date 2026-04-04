"""Tests for Gemma 3n/4 vision model support.

Covers BOI/EOI media region expansion, projector detection patterns,
model_type auto-detection priority, and VisionCardConfig BOI/EOI fields."""

import base64
import io

import mlx.core as mx
import mlx.nn as nn
from PIL import Image

from exo.shared.models.model_cards import VisionCardConfig
from exo.worker.engines.mlx.vision import _find_media_regions


def _fake_b64_image() -> str:
    """Create a minimal valid base64-encoded PNG for testing."""
    img = Image.new("RGB", (4, 4), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class TestFindMediaRegionsBoiEoi:
    """_find_media_regions should expand region boundaries to include BOI/EOI."""

    def test_no_boi_eoi(self):
        """Without BOI/EOI config, regions cover only image tokens."""
        tokens = mx.array([10, 20, 99, 99, 99, 30, 40])
        b64 = _fake_b64_image()
        regions = _find_media_regions(tokens, [b64], image_token_id=99)
        assert len(regions) == 1
        assert regions[0].start_pos == 2
        assert regions[0].end_pos == 5

    def test_boi_eoi_expansion(self):
        """With BOI/EOI, regions expand to include surrounding markers."""
        boi, img, eoi = 255999, 262145, 262144
        tokens = mx.array([10, boi, img, img, img, eoi, 20])
        b64 = _fake_b64_image()
        regions = _find_media_regions(
            tokens, [b64], image_token_id=img,
            boi_token_id=boi, eoi_token_id=eoi,
        )
        assert len(regions) == 1
        assert regions[0].start_pos == 1
        assert regions[0].end_pos == 6

    def test_boi_only(self):
        """Only BOI present (no EOI) — expand start only."""
        boi, img = 255999, 262145
        tokens = mx.array([boi, img, img, 10])
        b64 = _fake_b64_image()
        regions = _find_media_regions(
            tokens, [b64], image_token_id=img,
            boi_token_id=boi, eoi_token_id=None,
        )
        assert len(regions) == 1
        assert regions[0].start_pos == 0
        assert regions[0].end_pos == 3

    def test_no_boi_at_boundary(self):
        """BOI token not adjacent — no expansion."""
        boi, img = 255999, 262145
        tokens = mx.array([10, 20, img, img, 30])
        b64 = _fake_b64_image()
        regions = _find_media_regions(
            tokens, [b64], image_token_id=img,
            boi_token_id=boi, eoi_token_id=None,
        )
        assert len(regions) == 1
        assert regions[0].start_pos == 2
        assert regions[0].end_pos == 4

    def test_multiple_images_boi_eoi(self):
        """Multiple images each wrapped in BOI/EOI."""
        boi, img, eoi = 255999, 262145, 262144
        tokens = mx.array([boi, img, img, eoi, 10, boi, img, eoi, 20])
        b64_a = _fake_b64_image()
        b64_b = _fake_b64_image()
        regions = _find_media_regions(
            tokens, [b64_a, b64_b], image_token_id=img,
            boi_token_id=boi, eoi_token_id=eoi,
        )
        assert len(regions) == 2
        assert regions[0].start_pos == 0
        assert regions[0].end_pos == 4
        assert regions[1].start_pos == 5
        assert regions[1].end_pos == 8


class TestVisionCardConfigBoiEoi:
    """VisionCardConfig should accept optional BOI/EOI token IDs."""

    def test_defaults_none(self):
        config = VisionCardConfig(
            image_token_id=151655, model_type="qwen3_vl"
        )
        assert config.boi_token_id is None
        assert config.eoi_token_id is None

    def test_gemma_config(self):
        config = VisionCardConfig(
            image_token_id=262145,
            model_type="gemma3n",
            boi_token_id=255999,
            eoi_token_id=262144,
        )
        assert config.boi_token_id == 255999
        assert config.eoi_token_id == 262144


class TestModelTypePriority:
    """Auto-detection should prefer top-level model_type over vision_config.model_type."""

    def test_top_level_preferred(self):
        from exo.shared.models.model_cards import ConfigData

        data = {
            "model_type": "gemma3n",
            "num_hidden_layers": 30,
            "hidden_size": 2048,
            "vision_config": {"model_type": "gemma3n_vision", "hidden_size": 2048},
            "image_token_id": 262145,
        }
        result = ConfigData.model_validate(data, context={"model_id": "test/model"})
        assert result.vision is not None
        assert result.vision.model_type == "gemma3n"

    def test_falls_back_to_vision_config(self):
        from exo.shared.models.model_cards import ConfigData

        data = {
            "num_hidden_layers": 30,
            "hidden_size": 2048,
            "vision_config": {"model_type": "qwen3_vl", "hidden_size": 2560},
            "image_token_id": 151655,
        }
        result = ConfigData.model_validate(data, context={"model_id": "test/model"})
        assert result.vision is not None
        assert result.vision.model_type == "qwen3_vl"


class TestLoadProjectorWeights:
    """_load_projector_weights should handle both quantized and unquantized."""

    def test_unquantized_loads_normally(self):
        from exo.worker.engines.mlx.vision import _load_projector_weights

        proj = nn.Linear(8, 4, bias=False)
        weight = mx.ones((4, 8))
        weights = {"weight": weight}
        config: dict[str, object] = {}
        _load_projector_weights(proj, weights, config)
        assert proj.weight.shape == (4, 8)

    def test_quantized_applies_nn_quantize(self):
        from exo.worker.engines.mlx.vision import _load_projector_weights

        # Create a quantized state dict by quantizing a reference module.
        ref = nn.Linear(64, 32, bias=False)
        nn.quantize(ref, bits=4, group_size=64)
        mx.eval(ref.parameters())
        # Flatten nested dict from ref.parameters()
        flat: dict[str, mx.array] = {}
        for k, v in ref.parameters().items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    flat[f"{k}.{sub_k}"] = sub_v
            else:
                flat[k] = v

        # Create a fresh (unquantized) module and load quantized weights.
        target = nn.Linear(64, 32, bias=False)
        config = {"quantization": {"bits": 4, "group_size": 64}}
        _load_projector_weights(target, flat, config)
        # After loading, the module should have quantized weight shape.
        assert target.weight.shape == ref.weight.shape


class TestHasNativeVision:
    """has_native_vision should detect models with built-in vision support."""

    def test_model_with_vision_tower_and_embed_vision(self):
        from exo.worker.engines.mlx.vision import has_native_vision

        class _FakeInner(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_tower = nn.Linear(4, 4)
                self.embed_vision = nn.Linear(4, 4)

        class _FakeWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self._inner = _FakeInner()

        assert has_native_vision(_FakeWrapper()) is True

    def test_model_without_vision_tower(self):
        from exo.worker.engines.mlx.vision import has_native_vision

        class _Plain(nn.Module):
            def __init__(self):
                super().__init__()
                self.some_layer = nn.Linear(4, 4)

        assert has_native_vision(_Plain()) is False

    def test_model_with_only_vision_tower(self):
        from exo.worker.engines.mlx.vision import has_native_vision

        class _Partial(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_tower = nn.Linear(4, 4)

        assert has_native_vision(_Partial()) is False


class TestPatchNativeVision:
    """patch_native_vision should inject pixel_values during prefill and restore after."""

    def test_injects_pixel_values_and_restores(self):
        from exo.worker.engines.mlx.generator.generate import patch_native_vision

        class _Inner(nn.Module):
            def __call__(self, input_ids, **kwargs):
                return kwargs

        class _Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self._inner = _Inner()

        wrapper = _Wrapper()
        inner = wrapper._inner
        pv = mx.ones((1, 3, 4))

        # Before patch: no pixel_values injected
        result_before = inner(mx.array([1, 2, 3]))
        assert "pixel_values" not in result_before

        # During patch: pixel_values injected
        with patch_native_vision(wrapper, pv):
            result_during = inner(mx.array([1, 2, 3]))
            assert "pixel_values" in result_during
            assert result_during["pixel_values"] is pv

        # After patch: restored, no pixel_values
        result_after = inner(mx.array([1, 2, 3]))
        assert "pixel_values" not in result_after
