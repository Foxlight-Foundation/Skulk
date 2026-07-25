"""Tests for native-vision generation routing."""

import contextlib
from collections.abc import Callable, Generator
from importlib import import_module
from typing import cast

import mlx.core as mx
import pytest
from mlx_lm.tokenizer_utils import TokenizerWrapper

from skulk.shared.types.common import ModelId
from skulk.shared.types.mlx import KVCacheType, Model
from skulk.shared.types.tasks import TaskId
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.shared.types.worker.runner_response import GenerationResponse
from skulk.worker.engines.mlx.cache import CacheSnapshot, KVPrefixCache
from skulk.worker.engines.mlx.generator import batch_generate as batch_generate_module
from skulk.worker.engines.mlx.generator import generate as generate_module
from skulk.worker.engines.mlx.vision import (
    MediaRegion,
    VisionPreprocessingError,
    VisionProcessor,
    VisionResult,
    prepare_vision,
)

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


class _FakeGroup:
    def __init__(self, rank: int = 0, size: int = 3) -> None:
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def _fake_group() -> mx.distributed.Group:
    return cast(mx.distributed.Group, cast(object, _FakeGroup()))


def _noop_barrier(_group: object) -> None:
    return None


def _empty_cache_list(**_kwargs: object) -> list[object]:
    return []


def _always_false(**_kwargs: object) -> bool:
    return False


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
        assert cast(mx.array, _kwargs["image_grid_thw"]).tolist() == [[1, 2, 3]]
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
        image_grid_thw=mx.array([[1, 2, 3]]),
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
            max_tokens=8,
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
        "skulk.worker.engines.mlx.generator.generate.prepare_vision",
        _fake_prepare_vision,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate._should_use_native_vision_reference_path",
        cast(Callable[[], bool], lambda: True),
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate._mlx_generate_native_vision",
        _fake_native_generate,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.stream_generate",
        _fail_stream_generate,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.make_kv_cache",
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


def test_mlx_generate_rejects_image_input_without_vision_processor() -> None:
    """Image requests must never silently enter the text-only generation path."""
    task = TextGenerationTaskParams(
        model=ModelId("mlx-community/Qwen3.6-27B-4bit"),
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

    with pytest.raises(
        VisionPreprocessingError,
        match="without a vision processor",
    ):
        list(
            mlx_generate(
                model=_fake_model(),
                tokenizer=_fake_tokenizer(),
                task=task,
                prompt="<bos>",
                kv_prefix_cache=None,
                group=None,
                vision_processor=None,
            )
        )


def test_mlx_generate_rejects_failed_vision_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A processor error must fail the request instead of returning a hallucination."""

    def _fail_prepare_vision(**_kwargs: object) -> VisionResult:
        raise ImportError("processor dependency unavailable")

    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.prepare_vision",
        _fail_prepare_vision,
    )
    task = TextGenerationTaskParams(
        model=ModelId("mlx-community/Qwen3.6-27B-4bit"),
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

    with pytest.raises(
        VisionPreprocessingError,
        match="Vision preprocessing failed",
    ) as exc_info:
        list(
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

    assert isinstance(exc_info.value.__cause__, ImportError)


def test_mlx_generate_preserves_request_scoped_vision_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed vision request must not be promoted to a runner crash."""
    failure = VisionPreprocessingError("malformed image request")

    def _fail_prepare_vision(**_kwargs: object) -> VisionResult:
        raise failure

    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.prepare_vision",
        _fail_prepare_vision,
    )
    task = TextGenerationTaskParams(
        model=ModelId("mlx-community/Qwen3-VL-4B-Instruct-4bit"),
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

    with pytest.raises(VisionPreprocessingError) as error:
        list(
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

    assert error.value is failure


def test_batch_generate_rejects_failed_vision_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production batch path must not turn processor errors into guesses."""

    def _fail_prepare_vision(**_kwargs: object) -> VisionResult:
        raise ImportError("processor dependency unavailable")

    monkeypatch.setattr(batch_generate_module, "prepare_vision", _fail_prepare_vision)
    generator = object.__new__(batch_generate_module.SkulkBatchGenerator)
    object.__setattr__(generator, "model", _fake_model())
    object.__setattr__(generator, "tokenizer", _fake_tokenizer())
    object.__setattr__(generator, "vision_processor", _fake_vision_processor())
    task = TextGenerationTaskParams(
        model=ModelId("mlx-community/Qwen3-VL-4B-Instruct-4bit"),
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

    with pytest.raises(
        VisionPreprocessingError,
        match="failed to process 1 image",
    ) as exc_info:
        generator.submit(
            TaskId("vision-task"),
            task,
            "<bos>",
        )

    assert isinstance(exc_info.value.__cause__, ImportError)


@pytest.mark.parametrize("native_pixels", [False, True])
def test_batch_generate_vision_bypasses_prefix_cache(
    monkeypatch: pytest.MonkeyPatch,
    native_pixels: bool,
) -> None:
    """Batch requests must isolate native pixels and precomputed embeddings."""

    class _FailingPrefixCache:
        def get_kv_cache(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("vision must not read the prefix cache")

        def add_kv_cache(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("vision must not write the prefix cache")

    class _FakeNativeModel:
        def __init__(self) -> None:
            self.pixel_values: mx.array | list[mx.array] | None = None
            self.saw_pixel_values = False

        def set_pixel_values(
            self, pixel_values: mx.array | list[mx.array] | None
        ) -> None:
            self.pixel_values = pixel_values
            self.saw_pixel_values = self.saw_pixel_values or pixel_values is not None

    class _FakeUpstreamBatchGenerator:
        _prompt_time_counter = 0.0

        def insert(self, **_kwargs: object) -> list[int]:
            return [17]

    vision = VisionResult(
        prompt="ignored",
        prompt_tokens=mx.array([1, 2, 3, 4]),
        embeddings=mx.zeros((1, 0, 1)),
        media_regions=[],
        pixel_values=mx.array([[10.0]]) if native_pixels else None,
    )
    model = _FakeNativeModel()

    def _fake_encode_prompt(
        _tokenizer: TokenizerWrapper, _prompt: str
    ) -> mx.array:
        return mx.array([1, 2, 3, 4])

    def _fake_fix_tokens(
        tokens: mx.array, _tokenizer: TokenizerWrapper
    ) -> mx.array:
        return tokens

    def _fake_prepare_vision(**_kwargs: object) -> VisionResult:
        return vision

    def _fake_make_kv_cache(_model: Model) -> KVCacheType:
        return cast(KVCacheType, [])

    def _fake_make_sampler(**_kwargs: object) -> Callable[[mx.array], mx.array]:
        return _identity_sampler

    def _fake_make_logits_processors(
        **_kwargs: object,
    ) -> list[Callable[[mx.array, mx.array], mx.array]]:
        return []

    def _fake_patch_embed_tokens(
        *_args: object, **_kwargs: object
    ) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    monkeypatch.setattr(
        batch_generate_module,
        "encode_prompt",
        _fake_encode_prompt,
    )
    monkeypatch.setattr(
        batch_generate_module,
        "fix_unmatched_think_end_tokens",
        _fake_fix_tokens,
    )
    monkeypatch.setattr(
        batch_generate_module,
        "prepare_vision",
        _fake_prepare_vision,
    )
    monkeypatch.setattr(
        batch_generate_module,
        "make_kv_cache",
        _fake_make_kv_cache,
    )
    monkeypatch.setattr(
        batch_generate_module,
        "make_sampler",
        _fake_make_sampler,
    )
    monkeypatch.setattr(
        batch_generate_module,
        "make_logits_processors",
        _fake_make_logits_processors,
    )
    monkeypatch.setattr(
        batch_generate_module,
        "patch_embed_tokens",
        _fake_patch_embed_tokens,
    )

    def _fake_prefill(
        prefill_model: object,
        _tokenizer: object,
        _sampler: object,
        prompt_tokens: mx.array,
        _cache: object,
        *_args: object,
    ) -> tuple[float, int, list[CacheSnapshot]]:
        assert prefill_model is model
        assert (model.pixel_values is not None) is native_pixels
        assert prompt_tokens.tolist() == [1, 2, 3]
        return 1.0, len(prompt_tokens), []

    monkeypatch.setattr(batch_generate_module, "prefill", _fake_prefill)

    generator = object.__new__(batch_generate_module.SkulkBatchGenerator)
    object.__setattr__(generator, "model", cast(Model, cast(object, model)))
    object.__setattr__(generator, "tokenizer", _fake_tokenizer())
    object.__setattr__(generator, "group", None)
    object.__setattr__(generator, "kv_prefix_cache", _FailingPrefixCache())
    object.__setattr__(generator, "vision_processor", _fake_vision_processor())
    object.__setattr__(generator, "context_token_limit", None)
    object.__setattr__(generator, "_mlx_gen", _FakeUpstreamBatchGenerator())
    object.__setattr__(generator, "_active_tasks", {})
    object.__setattr__(generator, "_next_time_total", 0.0)

    task = TextGenerationTaskParams(
        model=ModelId("mlx-community/Qwen3-VL-4B-Instruct-4bit"),
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

    uid = generator.submit(TaskId("vision-task"), task, "<bos>")

    assert uid == 17
    assert model.saw_pixel_values is native_pixels
    assert model.pixel_values is None


def test_prepare_vision_rejects_images_without_structured_messages() -> None:
    """Missing multimodal prompt structure must not cause images to be ignored."""
    with pytest.raises(
        VisionPreprocessingError,
        match="without chat_template_messages",
    ):
        prepare_vision(
            images=["ignored"],
            chat_template_messages=None,
            vision_processor=_fake_vision_processor(),
            tokenizer=_fake_tokenizer(),
            model=_fake_model(),
        )


def test_resolve_request_vision_rejects_images_without_processor() -> None:
    """An image sent to a text-only card must fail instead of disappearing."""
    task = TextGenerationTaskParams(
        model=ModelId("mlx-community/text-only"),
        input=[InputMessage(role="user", content="what is this?")],
        images=["ignored"],
        max_output_tokens=8,
        temperature=0.0,
    )

    with pytest.raises(
        VisionPreprocessingError,
        match="has no vision configuration",
    ):
        batch_generate_module.resolve_request_vision(
            vision_processor=None,
            task_params=task,
            tokenizer=_fake_tokenizer(),
            model=_fake_model(),
            task_id=TaskId("vision-task"),
        )


def test_resolve_request_vision_preserves_text_only_requests() -> None:
    """A request without images should stay on the ordinary text path."""
    task = TextGenerationTaskParams(
        model=ModelId("mlx-community/text-only"),
        input=[InputMessage(role="user", content="hello")],
        max_output_tokens=8,
        temperature=0.0,
    )

    assert (
        batch_generate_module.resolve_request_vision(
            vision_processor=None,
            task_params=task,
            tokenizer=_fake_tokenizer(),
            model=_fake_model(),
            task_id=TaskId("text-task"),
        )
        is None
    )


def test_resolve_request_vision_wraps_processor_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected processor errors must become request-scoped vision errors."""
    task = TextGenerationTaskParams(
        model=ModelId("mlx-community/vision-model"),
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

    def _fail_prepare_vision(**_kwargs: object) -> None:
        raise KeyError("pixel_values")

    monkeypatch.setattr(
        batch_generate_module,
        "prepare_vision",
        _fail_prepare_vision,
    )

    with pytest.raises(
        VisionPreprocessingError,
        match="failed to process 1 image",
    ) as error:
        batch_generate_module.resolve_request_vision(
            vision_processor=_fake_vision_processor(),
            task_params=task,
            tokenizer=_fake_tokenizer(),
            model=_fake_model(),
            task_id=TaskId("vision-task"),
        )

    assert isinstance(error.value.__cause__, KeyError)


def test_mlx_generate_uses_native_reference_path_for_local_vision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local VLMs preserve family-specific multimodal position handling."""

    vision = VisionResult(
        prompt="ignored",
        prompt_tokens=mx.array([1, 2, 3]),
        embeddings=mx.zeros((1, 0, 1)),
        media_regions=[],
        pixel_values=mx.array([1.0]),
        image_grid_thw=mx.array([[1, 2, 3]]),
    )

    def _fake_prepare_vision(**_kwargs: object) -> VisionResult:
        return vision

    def _fake_native_generate(**kwargs: object):
        native_vision = cast(VisionResult, kwargs["vision"])
        assert native_vision.image_grid_thw is not None
        assert kwargs["max_tokens"] == 5
        yield GenerationResponse(text="native", token=101, usage=None)

    def _fail_stream_generate(*_args: object, **_kwargs: object):
        raise AssertionError("local native vision must use mlx-vlm generation")

    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.prepare_vision",
        _fake_prepare_vision,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate._should_use_native_vision_reference_path",
        cast(Callable[[], bool], lambda: False),
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate._mlx_generate_native_vision",
        _fake_native_generate,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.stream_generate",
        _fail_stream_generate,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.make_kv_cache",
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
            context_token_limit=8,
        )
    )

    assert [response.text for response in responses] == ["native"]


def test_native_vision_reference_path_version_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recent upstream MLX versions should disable the slower reference path."""

    monkeypatch.delenv("SKULK_NATIVE_VISION_REFERENCE_PATH", raising=False)

    def _fake_version(package: str) -> str:
        return _version_map(package, mlx_version="0.31.1", mlx_vlm_version="0.4.4")

    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.metadata.version",
        _fake_version,
    )

    assert _should_use_native_vision_reference_path_fn()() is False


def test_native_vision_reference_path_keeps_prereleases_on_safe_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prerelease builds should keep the safer reference path enabled."""

    monkeypatch.delenv("SKULK_NATIVE_VISION_REFERENCE_PATH", raising=False)

    def _fake_version(package: str) -> str:
        return _version_map(
            package, mlx_version="0.31.1rc1", mlx_vlm_version="0.4.4.dev1"
        )

    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.metadata.version",
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


def test_mlx_generate_full_prefills_native_pixels_instead_of_reusing_prefix_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native requests must re-inject every image instead of restoring text offsets."""

    class _FakePrefixCache:
        def get_kv_cache(
            self,
            _model: object,
            _prompt_tokens: object,
            media_regions: list[MediaRegion] | None = None,
        ) -> tuple[KVCacheType, mx.array, int]:
            raise AssertionError("native vision must not read the prefix cache")

        def add_kv_cache(self, *args: object, **kwargs: object) -> None:
            return None

        def update_kv_cache(self, *args: object, **kwargs: object) -> None:
            return None

    class _FakeModel:
        def __init__(self) -> None:
            self.pixel_values: list[mx.array] | mx.array | None = None
            self.seen_pixel_values: list[mx.array] | mx.array | None = None

        def set_pixel_values(
            self, pixel_values: list[mx.array] | mx.array | None
        ) -> None:
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
        assert [value.tolist() for value in model.pixel_values] == [
            [10.0],
            [20.0],
        ]
        model.seen_pixel_values = model.pixel_values
        return 0.0, 7, []

    def _fake_make_kv_cache(**_kwargs: object) -> KVCacheType:
        return cast(KVCacheType, [])

    def _fake_stream_generate(*_args: object, **_kwargs: object):
        yield GenerationResponse(text="ok", token=101, usage=None)

    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.prepare_vision",
        _fake_prepare_vision,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate._should_use_native_vision_reference_path",
        cast(Callable[[], bool], lambda: False),
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.prefill",
        _fake_prefill,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.make_kv_cache",
        _fake_make_kv_cache,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.stream_generate",
        _fake_stream_generate,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.mx_barrier",
        _noop_barrier,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate._should_force_native_vision_full_prefill_for_request",
        _always_false,
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
            group=_fake_group(),
            vision_processor=_fake_vision_processor(),
        )
    )

    assert [response.text for response in responses] == ["ok"]
    assert model.seen_pixel_values is not None
    assert model.pixel_values is None


def test_mlx_generate_reinjects_native_images_when_prefix_cache_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cache candidate must not suppress pixel injection on a later request."""

    class _FakePrefixCache:
        def get_kv_cache(
            self,
            _model: object,
            _prompt_tokens: object,
            media_regions: list[MediaRegion] | None = None,
        ) -> tuple[KVCacheType, mx.array, int]:
            raise AssertionError("native vision must not read the prefix cache")

        def add_kv_cache(self, *args: object, **kwargs: object) -> None:
            return None

        def update_kv_cache(self, *args: object, **kwargs: object) -> None:
            return None

    class _FakeModel:
        def __init__(self) -> None:
            self.pixel_values: list[mx.array] | mx.array | None = None
            self.seen_pixel_values: list[mx.array] | mx.array | None = None

        def set_pixel_values(
            self, pixel_values: list[mx.array] | mx.array | None
            ) -> None:
                self.pixel_values = pixel_values
                if pixel_values is not None:
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
        assert isinstance(model.pixel_values, list)
        assert [value.tolist() for value in model.pixel_values] == [
            [10.0],
            [20.0],
        ]
        return 0.0, 7, []

    def _fake_make_kv_cache(**_kwargs: object) -> KVCacheType:
        return cast(KVCacheType, [])

    def _fake_stream_generate(*_args: object, **_kwargs: object):
        yield GenerationResponse(text="ok", token=101, usage=None)

    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.prepare_vision",
        _fake_prepare_vision,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate._should_use_native_vision_reference_path",
        cast(Callable[[], bool], lambda: False),
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.prefill",
        _fake_prefill,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.make_kv_cache",
        _fake_make_kv_cache,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.stream_generate",
        _fake_stream_generate,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.patch_embed_tokens",
        _raise_patch_embed_tokens_error,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.mx_barrier",
        _noop_barrier,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate._should_force_native_vision_full_prefill_for_request",
        _always_false,
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
            group=_fake_group(),
            vision_processor=_fake_vision_processor(),
        )
    )

    assert [response.text for response in responses] == ["ok"]
    assert model.seen_pixel_values is not None


def test_mlx_generate_full_prefills_distributed_fully_cached_native_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distributed fully cached native-image follow-ups should skip prefix cache."""

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

    class _FakeModel:
        def __init__(self) -> None:
            self.pixel_values: list[mx.array] | mx.array | None = None
            self.seen_pixel_values: list[mx.array] | mx.array | None = None

        def set_pixel_values(
            self, pixel_values: list[mx.array] | mx.array | None
        ) -> None:
            self.pixel_values = pixel_values
            if pixel_values is not None:
                self.seen_pixel_values = pixel_values

    def _fake_make_kv_cache(**_kwargs: object) -> KVCacheType:
        return cast(KVCacheType, [])

    def _fake_prefill(
        model: _FakeModel,
        _tokenizer: object,
        _sampler: object,
        prompt_tokens: mx.array,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[float, int, list[CacheSnapshot]]:
        assert isinstance(model.pixel_values, list)
        assert [pixel_value.tolist() for pixel_value in model.pixel_values] == [
            [10.0],
            [20.0],
        ]
        assert prompt_tokens.tolist() == [1, 2, 3, 4, 5, 6, 7]
        return 0.0, len(prompt_tokens), []

    def _fake_stream_generate(*_args: object, **_kwargs: object):
        yield GenerationResponse(text="ok", token=101, usage=None)

    def _fail_native_generate(**_kwargs: object):
        raise AssertionError(
            "distributed fully-cached follow-ups should stay on the pipeline path"
        )

    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.prepare_vision",
        _fake_prepare_vision,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate._should_use_native_vision_reference_path",
        cast(Callable[[], bool], lambda: False),
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate._mlx_generate_native_vision",
        _fail_native_generate,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.make_kv_cache",
        _fake_make_kv_cache,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.prefill",
        _fake_prefill,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.stream_generate",
        _fake_stream_generate,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.mx_barrier",
        _noop_barrier,
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
            group=_fake_group(),
            vision_processor=_fake_vision_processor(),
        )
    )

    assert [response.text for response in responses] == ["ok"]
    assert model.seen_pixel_values is not None
    assert model.pixel_values is None


def test_mlx_generate_full_prefills_distributed_single_native_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distributed single-image requests also avoid unsafe prefix restoration."""

    class _FakePrefixCache:
        def get_kv_cache(
            self,
            _model: object,
            _prompt_tokens: object,
            media_regions: list[MediaRegion] | None = None,
        ) -> tuple[KVCacheType, mx.array, int]:
            raise AssertionError("native vision must not read the prefix cache")

        def add_kv_cache(self, *args: object, **kwargs: object) -> None:
            return None

        def update_kv_cache(self, *args: object, **kwargs: object) -> None:
            return None

    vision = VisionResult(
        prompt="ignored",
        prompt_tokens=mx.array([1, 2, 3, 4, 5, 6, 7, 8]),
        embeddings=mx.zeros((1, 0, 1)),
        media_regions=[MediaRegion("first", 1, 4)],
        pixel_values=mx.array([[10.0]]),
    )

    def _fake_prepare_vision(**_kwargs: object) -> VisionResult:
        return vision

    class _FakeModel:
        def __init__(self) -> None:
            self.pixel_values: mx.array | None = None
            self.saw_non_none_pixel_values = False

        def set_pixel_values(self, pixel_values: mx.array | None) -> None:
            self.pixel_values = pixel_values
            self.saw_non_none_pixel_values = (
                self.saw_non_none_pixel_values or pixel_values is not None
            )

    def _fake_make_kv_cache(**_kwargs: object) -> KVCacheType:
        return cast(KVCacheType, [])

    def _fake_prefill(
        model: _FakeModel,
        _tokenizer: object,
        _sampler: object,
        prompt_tokens: mx.array,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[float, int, list[CacheSnapshot]]:
        assert model.pixel_values is not None
        assert model.pixel_values.tolist() == [[10.0]]
        assert prompt_tokens.tolist() == [1, 2, 3, 4, 5, 6, 7]
        return 0.0, len(prompt_tokens), []

    def _fake_stream_generate(*_args: object, **_kwargs: object):
        yield GenerationResponse(text="ok", token=101, usage=None)

    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.prepare_vision",
        _fake_prepare_vision,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate._should_use_native_vision_reference_path",
        cast(Callable[[], bool], lambda: False),
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.make_kv_cache",
        _fake_make_kv_cache,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.prefill",
        _fake_prefill,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.stream_generate",
        _fake_stream_generate,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.mx_barrier",
        _noop_barrier,
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
            {"role": "user", "content": "and now?"},
        ],
        images=["ignored-a"],
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
            group=_fake_group(),
            vision_processor=_fake_vision_processor(),
        )
    )

    assert [response.text for response in responses] == ["ok"]
    assert model.saw_non_none_pixel_values is True
    assert model.pixel_values is None


def test_mlx_generate_pipeline_decode_starts_from_single_prompt_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pipeline decode should not run an extra decode-mode prompt bridge."""

    class _FakePrefixCache:
        def get_kv_cache(
            self,
            _model: object,
            _prompt_tokens: object,
            media_regions: list[MediaRegion] | None = None,
        ) -> tuple[KVCacheType, mx.array, int]:
            raise AssertionError("native vision must not read the prefix cache")

        def add_kv_cache(self, *args: object, **kwargs: object) -> None:
            return None

    vision = VisionResult(
        prompt="ignored",
        prompt_tokens=mx.array([1, 2, 3, 4, 5, 6, 7, 8]),
        embeddings=mx.zeros((1, 0, 1)),
        media_regions=[MediaRegion("first", 1, 4)],
        pixel_values=mx.array([[10.0]]),
    )

    def _fake_prepare_vision(**_kwargs: object) -> VisionResult:
        return vision

    class _FakeModel:
        def __init__(self) -> None:
            self.pixel_values: mx.array | None = None

        def set_pixel_values(self, pixel_values: mx.array | None) -> None:
            self.pixel_values = pixel_values

    def _fake_prefill(
        model: _FakeModel,
        _tokenizer: object,
        _sampler: object,
        prompt_tokens: mx.array,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[float, int, list[CacheSnapshot]]:
        assert model.pixel_values is not None
        assert model.pixel_values.tolist() == [[10.0]]
        assert prompt_tokens.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
        return 0.0, len(prompt_tokens), []

    def _fake_make_kv_cache(**_kwargs: object) -> KVCacheType:
        return cast(KVCacheType, [])

    def _fake_pipeline_decode(*_args: object, **kwargs: object):
        prompt = cast(mx.array, kwargs["prompt"])
        assert prompt.tolist() == [8]
        yield GenerationResponse(text="ok", token=101, usage=None)

    def _fail_stream_generate(*_args: object, **_kwargs: object):
        raise AssertionError("pipeline decode should use the no-lookahead generator")

    def _fake_has_pipeline_layer(_model: Model) -> bool:
        return True

    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.prepare_vision",
        _fake_prepare_vision,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate._should_use_native_vision_reference_path",
        cast(Callable[[], bool], lambda: False),
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate._has_pipeline_communication_layer",
        _fake_has_pipeline_layer,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.prefill",
        _fake_prefill,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.make_kv_cache",
        _fake_make_kv_cache,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate._stream_generate_without_lookahead",
        _fake_pipeline_decode,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.stream_generate",
        _fail_stream_generate,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.mx_barrier",
        _noop_barrier,
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
            {"role": "user", "content": "and now?"},
        ],
        images=["ignored-a"],
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
            group=_fake_group(),
            vision_processor=_fake_vision_processor(),
        )
    )

    assert [response.text for response in responses] == ["ok"]
    assert model.pixel_values is None


def test_mlx_generate_full_prefills_distributed_cached_native_image_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distributed native-image follow-ups with cached prefixes should re-prefill."""

    class _FakePrefixCache:
        def get_kv_cache(
            self,
            _model: object,
            _prompt_tokens: object,
            media_regions: list[MediaRegion] | None = None,
        ) -> tuple[KVCacheType, mx.array, int]:
            assert media_regions is not None
            return cast(KVCacheType, []), mx.array([5, 6, 7, 8]), 0

        def add_kv_cache(self, *args: object, **kwargs: object) -> None:
            return None

        def update_kv_cache(self, *args: object, **kwargs: object) -> None:
            return None

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

    class _FakeModel:
        def __init__(self) -> None:
            self.pixel_values: list[mx.array] | mx.array | None = None
            self.seen_pixel_values: list[mx.array] | mx.array | None = None

        def set_pixel_values(
            self, pixel_values: list[mx.array] | mx.array | None
        ) -> None:
            self.pixel_values = pixel_values
            if pixel_values is not None:
                self.seen_pixel_values = pixel_values

    def _fake_make_kv_cache(**_kwargs: object) -> KVCacheType:
        return cast(KVCacheType, [])

    def _fake_prefill(
        model: _FakeModel,
        _tokenizer: object,
        _sampler: object,
        prompt_tokens: mx.array,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[float, int, list[CacheSnapshot]]:
        assert isinstance(model.pixel_values, list)
        assert [pixel_value.tolist() for pixel_value in model.pixel_values] == [
            [10.0],
            [20.0],
        ]
        assert prompt_tokens.tolist() == [1, 2, 3, 4, 5, 6, 7]
        return 0.0, len(prompt_tokens), []

    def _fake_stream_generate(*_args: object, **_kwargs: object):
        yield GenerationResponse(text="ok", token=101, usage=None)

    def _fail_native_generate(**_kwargs: object):
        raise AssertionError(
            "distributed cached-image follow-ups should stay on the pipeline path"
        )

    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.prepare_vision",
        _fake_prepare_vision,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate._should_use_native_vision_reference_path",
        cast(Callable[[], bool], lambda: False),
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate._mlx_generate_native_vision",
        _fail_native_generate,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.make_kv_cache",
        _fake_make_kv_cache,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.prefill",
        _fake_prefill,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.stream_generate",
        _fake_stream_generate,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.generator.generate.mx_barrier",
        _noop_barrier,
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
            group=_fake_group(),
            vision_processor=_fake_vision_processor(),
        )
    )

    assert [response.text for response in responses] == ["ok"]
    assert model.seen_pixel_values is not None
    assert model.pixel_values is None
