"""Tests for native-vision generation routing."""

from importlib import import_module

import mlx.core as mx

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.mlx.generator.generate import (
    _mlx_generate_native_vision,
    _should_use_native_vision_reference_path,
    _slice_native_pixel_values_for_uncached_suffix,
    mlx_generate,
)
from exo.worker.engines.mlx.vision import MediaRegion, VisionResult


class _FakeDetokenizer:
    """Minimal streaming detokenizer for native vision generation tests."""

    def __init__(self) -> None:
        self.last_segment = ""

    def reset(self) -> None:
        self.last_segment = ""

    def add_token(self, token: int) -> None:
        mapping = {
            101: "Hello",
            102: " world",
        }
        self.last_segment = mapping.get(token, "")

    def finalize(self) -> None:
        self.last_segment = ""


class _FakeTokenizer:
    """Tokenizer stub with detokenizer and EOS metadata."""

    def __init__(self) -> None:
        self.detokenizer = _FakeDetokenizer()
        self.eos_token_ids = [999]
        self.has_thinking = False
        self.think_start = None
        self.think_end = None

    def decode(self, token_ids: list[int]) -> str:
        return "".join(str(token_id) for token_id in token_ids)

    def encode(self, _text: str, add_special_tokens: bool = False) -> list[int]:
        return [1, 2, 3]


def test_native_vision_generation_uses_mlx_vlm_generate_step(monkeypatch) -> None:
    """Native vision should stream through MLX-VLM's multimodal generate path."""

    def _fake_generate_step(
        input_ids: mx.array,
        model: object,
        pixel_values: mx.array,
        mask: object,
        **_kwargs: object,
    ):
        assert input_ids.shape == (1, 3)
        assert pixel_values.shape == (1,)
        yield mx.array(101), mx.zeros((8,))
        yield mx.array(102), mx.zeros((8,))
        yield mx.array(999), mx.zeros((8,))

    monkeypatch.setattr(
        import_module("mlx_vlm.generate"),
        "generate_step",
        _fake_generate_step,
    )

    task = TextGenerationTaskParams(
        model=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
        input=[InputMessage(role="user", content="what is this?")],
        max_output_tokens=8,
        temperature=0.0,
    )
    vision = VisionResult(
        prompt="ignored",
        prompt_tokens=mx.array([1, 2, 3]),
        embeddings=mx.zeros((1, 0, 1)),
        media_regions=[],
        pixel_values=mx.array([1.0]),
    )

    responses = list(
        _mlx_generate_native_vision(
            model=object(),  # type: ignore[arg-type]
            tokenizer=_FakeTokenizer(),  # type: ignore[arg-type]
            task=task,
            all_prompt_tokens=vision.prompt_tokens,
            vision=vision,
            sampler=lambda logits: logits,  # type: ignore[arg-type]
            logits_processors=[],
            on_prefill_progress=None,
            on_generation_token=None,
            group=None,
        )
    )

    assert [response.text for response in responses[:-1]] == ["Hello", " world"]
    assert responses[-1].finish_reason == "stop"
    assert responses[-1].usage is not None
    assert responses[-1].usage.prompt_tokens == 3
    assert responses[-1].usage.completion_tokens == 2


def test_mlx_generate_routes_native_vision_through_reference_path(monkeypatch) -> None:
    """``mlx_generate`` should bypass generic text generation for native vision."""

    vision = VisionResult(
        prompt="ignored",
        prompt_tokens=mx.array([1, 2, 3]),
        embeddings=mx.zeros((1, 0, 1)),
        media_regions=[],
        pixel_values=mx.array([1.0]),
    )

    def _fake_prepare_vision(**_kwargs: object) -> VisionResult:
        return vision

    def _fake_native_generate(**_kwargs: object):
        yield GenerationResponse(text="native", token=101, usage=None)

    def _fail_stream_generate(*_args: object, **_kwargs: object):
        raise AssertionError("native vision should not use generic stream_generate")

    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.generate.prepare_vision",
        _fake_prepare_vision,
    )
    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.generate._should_use_native_vision_reference_path",
        lambda: True,
    )
    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.generate._mlx_generate_native_vision",
        _fake_native_generate,
    )
    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.generate.stream_generate",
        _fail_stream_generate,
    )
    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.generate.make_kv_cache",
        lambda **_kwargs: [],
    )

    task = TextGenerationTaskParams(
        model=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
        input=[InputMessage(role="user", content="what is this?")],
        chat_template_messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "what is this?"},
                ],
            }
        ],
        images=["ignored"],
        max_output_tokens=8,
        temperature=0.0,
    )

    responses = list(
        mlx_generate(
            model=object(),  # type: ignore[arg-type]
            tokenizer=_FakeTokenizer(),  # type: ignore[arg-type]
            task=task,
            prompt="<bos>",
            kv_prefix_cache=None,
            group=None,
            vision_processor=object(),  # type: ignore[arg-type]
        )
    )

    assert [response.text for response in responses] == ["native"]


def test_mlx_generate_uses_pipeline_aware_path_on_fixed_stack(monkeypatch) -> None:
    """Fixed upstream stacks should use the faster legacy generation path."""

    vision = VisionResult(
        prompt="ignored",
        prompt_tokens=mx.array([1, 2, 3]),
        embeddings=mx.zeros((1, 0, 1)),
        media_regions=[],
        pixel_values=mx.array([1.0]),
    )

    class _FakeModel:
        def __init__(self) -> None:
            self.pixel_values: mx.array | None = None

        def set_pixel_values(self, pixel_values: mx.array | None) -> None:
            self.pixel_values = pixel_values

    def _fake_prepare_vision(**_kwargs: object) -> VisionResult:
        return vision

    def _fake_prefill(*_args: object, **_kwargs: object):
        return 0.0, 2, []

    def _fake_stream_generate(*_args: object, **_kwargs: object):
        yield GenerationResponse(text="legacy", token=101, usage=None)

    def _fail_native_generate(**_kwargs: object):
        raise AssertionError("fixed stacks should not force reference native vision")

    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.generate.prepare_vision",
        _fake_prepare_vision,
    )
    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.generate._should_use_native_vision_reference_path",
        lambda: False,
    )
    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.generate._mlx_generate_native_vision",
        _fail_native_generate,
    )
    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.generate.prefill",
        _fake_prefill,
    )
    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.generate.stream_generate",
        _fake_stream_generate,
    )
    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.generate.make_kv_cache",
        lambda **_kwargs: [],
    )

    task = TextGenerationTaskParams(
        model=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
        input=[InputMessage(role="user", content="what is this?")],
        chat_template_messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "what is this?"},
                ],
            }
        ],
        images=["ignored"],
        max_output_tokens=8,
        temperature=0.0,
    )

    model = _FakeModel()
    responses = list(
        mlx_generate(
            model=model,  # type: ignore[arg-type]
            tokenizer=_FakeTokenizer(),  # type: ignore[arg-type]
            task=task,
            prompt="<bos>",
            kv_prefix_cache=None,
            group=None,
            vision_processor=object(),  # type: ignore[arg-type]
        )
    )

    assert [response.text for response in responses] == ["legacy"]
    assert model.pixel_values is None


def test_native_vision_reference_path_version_gate(monkeypatch) -> None:
    """Recent upstream MLX versions should disable the slower reference path."""

    monkeypatch.delenv("EXO_NATIVE_VISION_REFERENCE_PATH", raising=False)
    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.generate.metadata.version",
        lambda package: {"mlx": "0.31.1", "mlx-vlm": "0.4.4"}[package],
    )

    assert _should_use_native_vision_reference_path() is False


def test_native_vision_reference_path_keeps_prereleases_on_safe_path(
    monkeypatch,
) -> None:
    """Prerelease builds should keep the safer reference path enabled."""

    monkeypatch.delenv("EXO_NATIVE_VISION_REFERENCE_PATH", raising=False)
    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.generate.metadata.version",
        lambda package: {"mlx": "0.31.1rc1", "mlx-vlm": "0.4.4.dev1"}[package],
    )

    assert _should_use_native_vision_reference_path() is True
def test_slice_native_pixel_values_for_uncached_suffix_drops_cached_images() -> None:
    """Prefix hits should remove already-cached native images from pixel_values."""

    pixel_values = [mx.array([10.0]), mx.array([20.0])]
    media_regions = [
        MediaRegion("first", 1, 4),
        MediaRegion("second", 5, 8),
    ]

    result = _slice_native_pixel_values_for_uncached_suffix(
        pixel_values,
        media_regions,
        prefix_hit_length=5,
    )

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].tolist() == [20.0]


def test_slice_native_pixel_values_for_uncached_suffix_returns_none_when_fully_cached() -> (
    None
):
    """Fully cached follow-up turns should not inject any native pixel values."""

    pixel_values = [mx.array([10.0]), mx.array([20.0])]
    media_regions = [
        MediaRegion("first", 1, 4),
        MediaRegion("second", 5, 8),
    ]

    result = _slice_native_pixel_values_for_uncached_suffix(
        pixel_values,
        media_regions,
        prefix_hit_length=8,
    )

    assert result is None


def test_mlx_generate_slices_native_pixel_values_after_prefix_hit(
    monkeypatch,
) -> None:
    """Follow-up turns should not reuse stale pixel values from cached images."""

    class _FakePrefixCache:
        def get_kv_cache(self, _model, _prompt_tokens, media_regions=None):
            assert media_regions is not None
            return [], mx.array([6, 7, 8]), 0

        def add_kv_cache(self, *args, **kwargs) -> None:
            return None

        def update_kv_cache(self, *args, **kwargs) -> None:
            return None

    class _FakeModel:
        def __init__(self) -> None:
            self.pixel_values = None
            self.seen_pixel_values = None

        def set_pixel_values(self, pixel_values) -> None:
            self.pixel_values = pixel_values

    vision = VisionResult(
        prompt="ignored",
        prompt_tokens=mx.array([1, 2, 3, 4, 5, 6, 7, 8]),
        embeddings=mx.zeros((1, 0, 1)),
        media_regions=[
            MediaRegion("first", 1, 4),
            MediaRegion("second", 5, 8),
        ],
        pixel_values=[mx.array([10.0]), mx.array([20.0])],
    )

    def _fake_prepare_vision(**_kwargs: object) -> VisionResult:
        return vision

    def _fake_prefill(model, *_args, **_kwargs):
        assert isinstance(model.pixel_values, list)
        assert len(model.pixel_values) == 1
        assert model.pixel_values[0].tolist() == [20.0]
        model.seen_pixel_values = model.pixel_values
        return 0.0, 2, []

    def _fake_stream_generate(*_args: object, **_kwargs: object):
        yield GenerationResponse(text="ok", token=101, usage=None)

    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.generate.prepare_vision",
        _fake_prepare_vision,
    )
    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.generate._should_use_native_vision_reference_path",
        lambda: False,
    )
    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.generate.prefill",
        _fake_prefill,
    )
    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.generate.stream_generate",
        _fake_stream_generate,
    )

    task = TextGenerationTaskParams(
        model=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
        input=[InputMessage(role="user", content="what is this?")],
        chat_template_messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "first"},
                ],
            },
            {"role": "assistant", "content": "ok"},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "second"},
                ],
            },
        ],
        images=["ignored-a", "ignored-b"],
        max_output_tokens=8,
        temperature=0.0,
    )

    model = _FakeModel()
    responses = list(
        mlx_generate(
            model=model,  # type: ignore[arg-type]
            tokenizer=_FakeTokenizer(),  # type: ignore[arg-type]
            task=task,
            prompt="<bos>",
            kv_prefix_cache=_FakePrefixCache(),  # type: ignore[arg-type]
            group=None,
            vision_processor=object(),  # type: ignore[arg-type]
        )
    )

    assert [response.text for response in responses] == ["ok"]
    assert model.seen_pixel_values is not None
    assert model.pixel_values is None


def test_mlx_generate_skips_embedding_patch_when_native_images_are_fully_cached(
    monkeypatch,
) -> None:
    """Fully cached native images should fall back to plain text prefill only."""

    class _FakePrefixCache:
        def get_kv_cache(self, _model, _prompt_tokens, media_regions=None):
            assert media_regions is not None
            return [], mx.array([7, 8]), 0

        def add_kv_cache(self, *args, **kwargs) -> None:
            return None

        def update_kv_cache(self, *args, **kwargs) -> None:
            return None

    class _FakeModel:
        def __init__(self) -> None:
            self.pixel_values = None
            self.seen_pixel_values = "unset"

        def set_pixel_values(self, pixel_values) -> None:
            self.pixel_values = pixel_values
            self.seen_pixel_values = pixel_values

    vision = VisionResult(
        prompt="ignored",
        prompt_tokens=mx.array([1, 2, 3, 4, 5, 6, 7, 8]),
        embeddings=mx.zeros((1, 0, 1)),
        media_regions=[
            MediaRegion("first", 1, 4),
            MediaRegion("second", 5, 6),
        ],
        pixel_values=[mx.array([10.0]), mx.array([20.0])],
    )

    def _fake_prepare_vision(**_kwargs: object) -> VisionResult:
        return vision

    def _fake_prefill(model, *_args, **_kwargs):
        assert model.pixel_values is None
        return 0.0, 2, []

    def _fake_stream_generate(*_args: object, **_kwargs: object):
        yield GenerationResponse(text="ok", token=101, usage=None)

    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.generate.prepare_vision",
        _fake_prepare_vision,
    )
    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.generate._should_use_native_vision_reference_path",
        lambda: False,
    )
    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.generate.prefill",
        _fake_prefill,
    )
    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.generate.stream_generate",
        _fake_stream_generate,
    )
    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.generate.patch_embed_tokens",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("native fully-cached requests should not patch embeddings")
        ),
    )

    task = TextGenerationTaskParams(
        model=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
        input=[InputMessage(role="user", content="what is this?")],
        chat_template_messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "first"},
                ],
            },
            {"role": "assistant", "content": "ok"},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "second"},
                ],
            },
        ],
        images=["ignored-a", "ignored-b"],
        max_output_tokens=8,
        temperature=0.0,
    )

    model = _FakeModel()
    responses = list(
        mlx_generate(
            model=model,  # type: ignore[arg-type]
            tokenizer=_FakeTokenizer(),  # type: ignore[arg-type]
            task=task,
            prompt="<bos>",
            kv_prefix_cache=_FakePrefixCache(),  # type: ignore[arg-type]
            group=None,
            vision_processor=object(),  # type: ignore[arg-type]
        )
    )

    assert [response.text for response in responses] == ["ok"]
    assert model.seen_pixel_values is None
