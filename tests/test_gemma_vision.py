"""Tests for Gemma 3n/4 vision model support.

Covers BOI/EOI media region expansion, projector detection patterns,
model_type auto-detection priority, and VisionCardConfig BOI/EOI fields."""

import base64
import io
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from PIL import Image

from exo.shared.models.model_cards import VisionCardConfig
from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.worker.engines.mlx.gemma4_prompt import render_gemma4_prompt
from exo.worker.engines.mlx.utils_mlx import apply_chat_template
from exo.worker.engines.mlx.vision import _find_media_regions, _format_vlm_messages


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


class TestGemma4ReferencePromptRenderer:
    """Gemma 4 prompts should match the dedicated reference structure."""

    def test_multimodal_prompt_matches_reference_shape(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "what do you see?"},
                ],
            }
        ]

        prompt = render_gemma4_prompt(messages, add_generation_prompt=True)

        assert (
            prompt
            == "<bos><|turn>user\n\n\n<|image|>\n\nwhat do you see?<turn|>\n<|turn>model\n"
        )

    def test_thinking_requires_explicit_enable(self):
        messages = [{"role": "user", "content": "Hi"}]

        prompt = render_gemma4_prompt(
            messages,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        assert (
            prompt
            == "<bos><|turn>system\n<|think|><turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"
        )

    def test_apply_chat_template_uses_reference_renderer_for_plain_gemma4(self):
        class _Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                raise AssertionError("Gemma 4 should not use generic chat templating here")

        task = TextGenerationTaskParams(
            model=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
            input=[InputMessage(role="user", content="what do you see?")],
            chat_template_messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "what do you see?"},
                    ],
                }
            ],
            images=[_fake_b64_image()],
        )

        prompt = apply_chat_template(_Tokenizer(), task)  # type: ignore[arg-type]

        assert "<|channel>thought" not in prompt
        assert prompt.endswith("<|turn>model\n")

    def test_build_vision_prompt_uses_reference_renderer_for_gemma4(self):
        from exo.worker.engines.mlx.vision import build_vision_prompt

        class _Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                raise AssertionError("Gemma 4 vision should not use generic chat templating here")

            def decode(self, token_ids):
                mapping = {10: "<|image>", 11: "<image|>"}
                return "".join(mapping[token_id] for token_id in token_ids)

        prompt = build_vision_prompt(
            tokenizer=_Tokenizer(),  # type: ignore[arg-type]
            chat_template_messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "what do you see?"},
                    ],
                }
            ],
            n_tokens_per_image=[3],
            image_token="<|image|>",
            model_type="gemma4",
            boi_token_id=10,
            eoi_token_id=11,
        )

        assert (
            prompt
            == "<bos><|turn>user\n\n\n<|image><|image|><|image|><|image|><image|>\n\nwhat do you see?<turn|>\n<|turn>model\n"
        )


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


class TestMlxVlmProcessorFallback:
    """Gemma 4 fallback processor loading should preserve MLX-VLM config."""

    def test_prefers_processor_from_pretrained(self):
        from exo.worker.engines.mlx.vision import (
            _load_mlx_vlm_image_processor_from_pretrained,
        )

        configured_image_processor = object()

        class _FakeProcessor:
            @classmethod
            def from_pretrained(cls, repo: str) -> SimpleNamespace:
                assert repo == "/tmp/repo"
                return SimpleNamespace(image_processor=configured_image_processor)

        proc_mod = SimpleNamespace(
            FakeProcessor=_FakeProcessor,
        )

        result = _load_mlx_vlm_image_processor_from_pretrained(proc_mod, "/tmp/repo")

        assert result is configured_image_processor


class TestGemma4NativeProcessorBudget:
    """Gemma 4 processor budget should stay at reference unless overridden."""

    def test_image_only_prompt_keeps_reference_budget_by_default(self):
        from exo.worker.engines.mlx.vision import _gemma4_native_processor_kwargs

        messages = [{"role": "user", "content": [{"type": "image"}]}]
        processor = SimpleNamespace(max_soft_tokens=280)

        result = _gemma4_native_processor_kwargs(messages, processor)

        assert result == {}

    def test_image_only_prompt_respects_env_override(self, monkeypatch):
        from exo.worker.engines.mlx.vision import _gemma4_native_processor_kwargs

        monkeypatch.setenv("EXO_GEMMA4_IMAGE_ONLY_MAX_SOFT_TOKENS", "320")
        messages = [{"role": "user", "content": [{"type": "image"}]}]
        processor = SimpleNamespace(max_soft_tokens=280)

        result = _gemma4_native_processor_kwargs(messages, processor)

        assert result == {"max_soft_tokens": 320}

    def test_text_and_image_prompt_keeps_reference_budget_by_default(self):
        from exo.worker.engines.mlx.vision import _gemma4_native_processor_kwargs

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image"},
                ],
            }
        ]
        processor = SimpleNamespace(max_soft_tokens=280)

        result = _gemma4_native_processor_kwargs(messages, processor)

        assert result == {}

    def test_text_only_prompt_keeps_existing_budget(self):
        from exo.worker.engines.mlx.vision import _gemma4_native_processor_kwargs

        messages = [{"role": "user", "content": "Describe this text prompt"}]
        processor = SimpleNamespace(max_soft_tokens=280)

        result = _gemma4_native_processor_kwargs(messages, processor)

        assert result == {}


class TestGemma4DynamicVisionPooling:
    """Gemma 4 native vision should pool using the actual processed image size."""

    def test_patch_positions_expand_past_default_max_patches(self):
        from exo.worker.engines.mlx.utils_mlx import (
            _gemma4_patch_positions_and_padding,
        )

        pixel_values = mx.zeros((1, 3, 576, 1152))

        patch_positions, padding_positions = _gemma4_patch_positions_and_padding(
            pixel_values,
            patch_size=16,
            max_patches=280 * 9,
        )

        assert patch_positions.shape == (1, 2592, 2)
        assert padding_positions.shape == (1, 2592)
        assert bool(mx.any(padding_positions).item()) is False

    def test_output_length_matches_processed_wide_image(self):
        from exo.worker.engines.mlx.utils_mlx import (
            _gemma4_output_length_for_pixel_values,
        )

        pixel_values = mx.zeros((1, 3, 576, 1104))

        result = _gemma4_output_length_for_pixel_values(
            pixel_values,
            patch_size=16,
            pooling_kernel_size=3,
        )

        assert result == 276

    def test_dynamic_wrapper_passes_output_length_to_pooler(self):
        from exo.worker.engines.mlx.utils_mlx import _Gemma4DynamicVisionTower

        class _FakePatchEmbedder(nn.Module):
            def __call__(
                self,
                pixel_values: mx.array,
                patch_positions: mx.array,
                padding_positions: mx.array,
            ) -> mx.array:
                batch_size = pixel_values.shape[0]
                seq_len = patch_positions.shape[1]
                _ = padding_positions
                return mx.zeros((batch_size, seq_len, 8))

        class _FakeEncoder(nn.Module):
            def __call__(
                self,
                hidden_states: mx.array,
                patch_positions: mx.array,
                attn_mask: mx.array,
            ) -> mx.array:
                assert patch_positions.shape[1] == hidden_states.shape[1]
                assert attn_mask.shape[-1] == hidden_states.shape[1]
                return hidden_states

        class _FakePooler(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.output_length: int | None = None

            def __call__(
                self,
                hidden_states: mx.array,
                patch_positions: mx.array,
                padding_positions: mx.array,
                output_length: int | None = None,
            ) -> tuple[mx.array, mx.array]:
                _ = hidden_states
                _ = patch_positions
                _ = padding_positions
                assert output_length is not None
                self.output_length = output_length
                pooled = mx.zeros((1, output_length, 8))
                mask = mx.ones((1, output_length), dtype=mx.bool_)
                return pooled, mask

        class _FakeInner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.patch_size = 16
                self.pooling_kernel_size = 3
                self.max_patches = 280 * 9
                self.patch_embedder = _FakePatchEmbedder()
                self.encoder = _FakeEncoder()
                self.pooler = _FakePooler()
                self.config = SimpleNamespace(standardize=False)

            def _patch_positions(
                self, pixel_values: mx.array
            ) -> tuple[mx.array, mx.array]:
                import numpy as _np

                batch_size, _, height, width = pixel_values.shape
                patch_height = height // self.patch_size
                patch_width = width // self.patch_size
                num_real = patch_height * patch_width
                num_padding = self.max_patches - num_real

                positions = []
                for y in range(patch_height):
                    for x in range(patch_width):
                        positions.append([x, y])

                real_positions = _np.array(positions, dtype=_np.int32)
                real_positions = _np.tile(real_positions[None], (batch_size, 1, 1))
                if num_padding > 0:
                    pad_positions = _np.full(
                        (batch_size, num_padding, 2), -1, dtype=_np.int32
                    )
                    patch_positions = _np.concatenate(
                        [real_positions, pad_positions], axis=1
                    )
                else:
                    patch_positions = real_positions

                padding_positions = _np.zeros(
                    (batch_size, self.max_patches), dtype=bool
                )
                if num_padding > 0:
                    padding_positions[:, num_real:] = True

                return mx.array(patch_positions), mx.array(padding_positions)

        inner = _FakeInner()
        wrapped = _Gemma4DynamicVisionTower(inner)

        _ = wrapped(mx.zeros((1, 3, 576, 1104)))

        assert inner.pooler.output_length == 276


class TestFormatVlmMessages:
    """Gemma prompts should preserve multimodal ordering instead of flattening."""

    def test_gemma4_preserves_interleaving(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "image"},
                    {"type": "text", "text": "second"},
                    {"type": "image_url"},
                ],
            }
        ]

        formatted = _format_vlm_messages(messages, "gemma4")

        assert formatted == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "image"},
                    {"type": "text", "text": "second"},
                    {"type": "image"},
                ],
            }
        ]
