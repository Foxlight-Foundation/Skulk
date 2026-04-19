"""Tests for native-vision generation routing."""

from collections.abc import Callable, Generator
from importlib import import_module
from typing import cast

import mlx.core as mx
import pytest
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.common import ModelId
from exo.shared.types.mlx import KVCacheType, Model
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.mlx.cache import CacheSnapshot, KVPrefixCache
from exo.worker.engines.mlx.generator import generate as generate_module
from exo.worker.engines.mlx.vision import MediaRegion, VisionProcessor, VisionResult

mlx_generate = generate_module.mlx_generate


def _mlx_generate_native_vision_fn() -> Callable[
    ..., Generator[GenerationResponse, None, None]
]:
    module_dict = cast(dict[str, object], generate_module.__dict__)
    return cast(
        Callable[..., Generator[GenerationResponse, None, None]],
        module_dict["_mlx_generate_native_vision"],
    )


def _should_use_native_vision_reference_path_fn() -> Callable[[], bool]:
    module_dict = cast(dict[str, object], generate_module.__dict__)
    return cast(
        Callable[[], bool],
        module_dict["_should_use_native_vision_reference_path"],
    )


def _slice_native_pixel_values_fn() -> Callable[
    [mx.array | list[mx.array], list[MediaRegion], int],
    mx.array | list[mx.array] | None,
]:
    module_dict = cast(dict[str, object], generate_module.__dict__)
    return cast(
        Callable[
            [mx.array | list[mx.array], list[MediaRegion], int],
            mx.array | list[mx.array] | None,
        ],
        module_dict["_slice_native_pixel_values_for_uncached_suffix"],
    )


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


def _fake_tokenizer() -> TokenizerWrapper:
    return cast(TokenizerWrapper, cast(object, _FakeTokenizer()))


def _fake_model() -> Model:
    return cast(Model, object())


def _fake_vision_processor() -> VisionProcessor:
    return cast(VisionProcessor, object())


def _identity_sampler(logits: mx.array) -> mx.array:
    return logits


def _empty_cache_list(**_kwargs: object) -> list[object]:
    return []


def _version_map(package: str, *, mlx_version: str, mlx_vlm_version: str) -> str:
    return {"mlx": mlx_version, "mlx-vlm": mlx_vlm_version}[package]


def _raise_patch_embed_tokens_error(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("native fully-cached requests should not patch embeddings")


def test_native_vision_generation_uses_mlx_vlm_generate_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        _mlx_generate_native_vision_fn()(
            model=_fake_model(),
            tokenizer=_fake_tokenizer(),
            task=task,
            all_prompt_tokens=vision.prompt_tokens,
            vision=vision,
            sampler=_identity_sampler,
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


def test_mlx_generate_routes_native_vision_through_reference_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        cast(Callable[[], bool], lambda: True),
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
        _empty_cache_list,
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
            model=_fake_model(),
            tokenizer=_fake_tokenizer(),
            task=task,
            prompt="<bos>",
            kv_prefix_cache=None,
            group=None,
            vision_processor=_fake_vision_processor(),
        )
    )

    assert [response.text for response in responses] == ["native"]


def test_mlx_generate_uses_pipeline_aware_path_on_fixed_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    def _fake_prefill(*_args: object, **_kwargs: object) -> tuple[float, int, list[CacheSnapshot]]:
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
        cast(Callable[[], bool], lambda: False),
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
        _empty_cache_list,
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
            model=cast(Model, cast(object, model)),
            tokenizer=_fake_tokenizer(),
            task=task,
            prompt="<bos>",
            kv_prefix_cache=None,
            group=None,
            vision_processor=_fake_vision_processor(),
        )
    )

    assert [response.text for response in responses] == ["legacy"]
    assert model.pixel_values is None


def test_native_vision_reference_path_version_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recent upstream MLX versions should disable the slower reference path."""

    monkeypatch.delenv("EXO_NATIVE_VISION_REFERENCE_PATH", raising=False)
    def _fake_version(package: str) -> str:
        return _version_map(
            package, mlx_version="0.31.1", mlx_vlm_version="0.4.4"
        )

    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.generate.metadata.version",
        _fake_version,
    )

    assert _should_use_native_vision_reference_path_fn()() is False


def test_native_vision_reference_path_keeps_prereleases_on_safe_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prerelease builds should keep the safer reference path enabled."""

    monkeypatch.delenv("EXO_NATIVE_VISION_REFERENCE_PATH", raising=False)
    def _fake_version(package: str) -> str:
        return _version_map(
            package, mlx_version="0.31.1rc1", mlx_vlm_version="0.4.4.dev1"
        )

    monkeypatch.setattr(
        "exo.worker.engines.mlx.generator.generate.metadata.version",
        _fake_version,
    )

    assert _should_use_native_vision_reference_path_fn()() is True


def test_slice_native_pixel_values_for_uncached_suffix_drops_cached_images() -> None:
    """Prefix hits should remove already-cached native images from pixel_values."""

    pixel_values = [mx.array([10.0]), mx.array([20.0])]
    media_regions = [
        MediaRegion("first", 1, 4),
        MediaRegion("second", 5, 8),
    ]

    result = _slice_native_pixel_values_fn()(
        pixel_values,
        media_regions,
        5,
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

    result = _slice_native_pixel_values_fn()(
        pixel_values,
        media_regions,
        8,
    )

    assert result is None


def test_mlx_generate_slices_native_pixel_values_after_prefix_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Follow-up turns should not reuse stale pixel values from cached images."""

    class _FakePrefixCache:
        def get_kv_cache(
            self,
            _model: object,
            _prompt_tokens: object,
            media_regions: list[MediaRegion] | None = None,
        ) -> tuple[KVCacheType, mx.array, int]:
            assert media_regions is not None
            return cast(KVCacheType, []), mx.array([6, 7, 8]), 0

        def add_kv_cache(self, *args: object, **kwargs: object) -> None:
            return None

        def update_kv_cache(self, *args: object, **kwargs: object) -> None:
            return None

    class _FakeModel:
        def __init__(self) -> None:
            self.pixel_values: list[mx.array] | mx.array | None = None
            self.seen_pixel_values: list[mx.array] | mx.array | None = None

        def set_pixel_values(self, pixel_values: list[mx.array] | mx.array | None) -> None:
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

    def _fake_prefill(
        model: _FakeModel, *_args: object, **_kwargs: object
    ) -> tuple[float, int, list[CacheSnapshot]]:
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
        cast(Callable[[], bool], lambda: False),
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
            model=cast(Model, cast(object, model)),
            tokenizer=_fake_tokenizer(),
            task=task,
            prompt="<bos>",
            kv_prefix_cache=cast(KVPrefixCache, cast(object, _FakePrefixCache())),
            group=None,
            vision_processor=_fake_vision_processor(),
        )
    )

    assert [response.text for response in responses] == ["ok"]
    assert model.seen_pixel_values is not None
    assert model.pixel_values is None


def test_mlx_generate_skips_embedding_patch_when_native_images_are_fully_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fully cached native images should fall back to plain text prefill only."""

    class _FakePrefixCache:
        def get_kv_cache(
            self,
            _model: object,
            _prompt_tokens: object,
            media_regions: list[MediaRegion] | None = None,
        ) -> tuple[KVCacheType, mx.array, int]:
            assert media_regions is not None
            return cast(KVCacheType, []), mx.array([7, 8]), 0

        def add_kv_cache(self, *args: object, **kwargs: object) -> None:
            return None

        def update_kv_cache(self, *args: object, **kwargs: object) -> None:
            return None

    class _FakeModel:
        def __init__(self) -> None:
            self.pixel_values: list[mx.array] | mx.array | None = None
            self.seen_pixel_values: list[mx.array] | mx.array | None = None

        def set_pixel_values(self, pixel_values: list[mx.array] | mx.array | None) -> None:
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

    def _fake_prefill(
        model: _FakeModel, *_args: object, **_kwargs: object
    ) -> tuple[float, int, list[CacheSnapshot]]:
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
        cast(Callable[[], bool], lambda: False),
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
        _raise_patch_embed_tokens_error,
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
            model=cast(Model, cast(object, model)),
            tokenizer=_fake_tokenizer(),
            task=task,
            prompt="<bos>",
            kv_prefix_cache=cast(KVPrefixCache, cast(object, _FakePrefixCache())),
            group=None,
            vision_processor=_fake_vision_processor(),
        )
    )

    assert [response.text for response in responses] == ["ok"]
    assert model.seen_pixel_values is None
