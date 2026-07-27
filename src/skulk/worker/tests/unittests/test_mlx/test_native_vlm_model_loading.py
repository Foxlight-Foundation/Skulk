# Copyright 2026 Foxlight Foundation

"""Tests for selecting the native MLX-VLM model loader."""

from pathlib import Path
from types import SimpleNamespace
from typing import Callable, cast

import mlx.core as mx
import mlx.nn as nn
import pytest

from skulk.shared.models.model_cards import (
    ModelCard,
    ModelId,
    ModelTask,
    VisionCardConfig,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.shards import (
    PipelineShardMetadata,
    TensorShardMetadata,
)
from skulk.worker.engines.mlx import auto_parallel, utils_mlx


class _FakeVlmModel(nn.Module):
    """Minimal native VLM returned by the mocked upstream loader."""


class _RecordingVlmModel(nn.Module):
    """Native VLM stub that records multimodal kwargs received by prefill."""

    def __init__(self) -> None:
        super().__init__()
        self.kwargs: dict[str, object] = {}

    def __call__(self, _tokens: mx.array, **kwargs: object) -> mx.array:
        self.kwargs = kwargs
        return mx.zeros((1, 1, 4))


class _NativeLanguageModel(nn.Module):
    """Language-model stub with a family-specific hybrid cache factory."""

    def make_cache(self) -> list[object]:
        return ["native-ssm-cache", "native-kv-cache"]


class _VlmWithNativeLanguageCache(nn.Module):
    """Vision wrapper stub whose cache contract lives on its language model."""

    def __init__(self) -> None:
        super().__init__()
        self.language_model = _NativeLanguageModel()
        self.layers = [object(), object()]


class _NativeQwenLanguageModel(nn.Module):
    """Minimal model carrying MLX-VLM's runtime identity and model type."""

    __module__ = "mlx_vlm.models.qwen3_5.language"
    model_type = "qwen3_5"


class _VlmWithTensorLanguage(nn.Module):
    """Native VLM stub whose tensor-shardable trunk is nested."""

    def __init__(self) -> None:
        super().__init__()
        self.language_model = _NativeQwenLanguageModel()


class _VariadicGenerationModel(nn.Module):
    """Generation wrapper whose cache is carried through ``**kwargs``."""

    def __call__(self, *_args: object, **_kwargs: object) -> mx.array:
        return mx.zeros((1, 1, 4))


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


def test_pipeline_shards_use_native_loader_for_primary_vision_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distributed vision must preserve the same native model as one-rank loads."""
    model_id = ModelId("mlx-community/Qwen3-VL-4B-Instruct-4bit")
    shard_metadata = PipelineShardMetadata(
        model_card=ModelCard(
            model_id=model_id,
            storage_size=Memory.from_bytes(1),
            n_layers=2,
            hidden_size=4,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
            vision=VisionCardConfig(
                image_token_id=151655,
                model_type="qwen3_vl",
                weights_repo=str(model_id),
            ),
        ),
        device_rank=0,
        world_size=2,
        start_layer=0,
        end_layer=1,
        n_layers=2,
    )
    loaded_model = _FakeVlmModel()
    seen: dict[str, object] = {}

    def _record_load_model(model_path: Path, **kwargs: object) -> tuple[nn.Module, None]:
        seen["model_path"] = model_path
        seen["kwargs"] = kwargs
        return loaded_model, None

    def _model_path(*_args: object, **_kwargs: object) -> Path:
        return Path("/model")

    def _identity_pipeline(
        model: nn.Module, *_args: object, **_kwargs: object
    ) -> nn.Module:
        return model

    def _ignore(*_args: object, **_kwargs: object) -> None:
        return None

    def _tokenizer(*_args: object, **_kwargs: object) -> str:
        return "tokenizer"

    monkeypatch.setattr(utils_mlx, "build_model_path", _model_path)
    monkeypatch.setattr(utils_mlx, "load_model", _record_load_model)
    monkeypatch.setattr(utils_mlx, "pipeline_auto_parallel", _identity_pipeline)
    monkeypatch.setattr(utils_mlx, "eval_with_timeout", _ignore)
    monkeypatch.setattr(utils_mlx, "mx_barrier", _ignore)
    monkeypatch.setattr(utils_mlx.mx, "eval", _ignore)
    monkeypatch.setattr(utils_mlx, "get_tokenizer", _tokenizer)
    group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)

    model, tokenizer = utils_mlx.shard_and_load(
        shard_metadata,
        cast(utils_mlx.Group, cast(object, group)),
        on_timeout=None,
        on_layer_loaded=None,
    )

    assert model is loaded_model
    assert tokenizer == "tokenizer"
    assert seen == {
        "model_path": Path("/model"),
        "kwargs": {
            "lazy": True,
            "strict": False,
            "prefer_vlm": True,
        },
    }


def test_tensor_shards_delegate_to_native_language_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tensor sharding must retain the native VLM wrapper and vision tower."""
    model_id = ModelId("mlx-community/Qwen3.5-2B-4bit")
    shard_metadata = TensorShardMetadata(
        model_card=ModelCard(
            model_id=model_id,
            storage_size=Memory.from_bytes(1),
            n_layers=2,
            hidden_size=4,
            num_key_value_heads=2,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
            vision=VisionCardConfig(
                image_token_id=248056,
                model_type="qwen3_5",
                weights_repo=str(model_id),
            ),
        ),
        device_rank=0,
        world_size=2,
        start_layer=0,
        end_layer=2,
        n_layers=2,
    )
    inner_model = _VlmWithTensorLanguage()
    wrapper_class = cast(
        Callable[[nn.Module], nn.Module],
        vars(utils_mlx)["_VlmModelWrapper"],
    )
    wrapper = wrapper_class(inner_model)
    seen: dict[str, object] = {}

    def _model_path(*_args: object, **_kwargs: object) -> Path:
        return Path("/model")

    def _load_model(*_args: object, **_kwargs: object) -> tuple[nn.Module, None]:
        return wrapper, None

    def _tensor_parallel(
        model: nn.Module,
        _group: object,
        _timeout_seconds: float,
        _on_timeout: object,
        _on_layer_loaded: object,
        *,
        generation_model: nn.Module | None = None,
    ) -> nn.Module:
        seen["sharding_model"] = model
        seen["generation_model"] = generation_model
        assert generation_model is not None
        return generation_model

    def _ignore(*_args: object, **_kwargs: object) -> None:
        return None

    def _tokenizer(*_args: object, **_kwargs: object) -> str:
        return "tokenizer"

    def _weight_size(_metadata: object) -> Memory:
        return Memory.from_bytes(1)

    monkeypatch.setattr(utils_mlx, "build_model_path", _model_path)
    monkeypatch.setattr(utils_mlx, "load_model", _load_model)
    monkeypatch.setattr(utils_mlx, "tensor_auto_parallel", _tensor_parallel)
    monkeypatch.setattr(utils_mlx, "mx_barrier", _ignore)
    monkeypatch.setattr(utils_mlx.mx, "eval", _ignore)
    monkeypatch.setattr(utils_mlx, "get_tokenizer", _tokenizer)
    monkeypatch.setattr(
        utils_mlx,
        "get_weights_size",
        _weight_size,
    )
    group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)

    model, tokenizer = utils_mlx.shard_and_load(
        shard_metadata,
        cast(utils_mlx.Group, cast(object, group)),
        on_timeout=None,
        on_layer_loaded=None,
    )

    assert model is wrapper
    assert tokenizer == "tokenizer"
    assert seen == {
        "sharding_model": inner_model.language_model,
        "generation_model": wrapper,
    }


def test_tensor_parallel_selects_native_qwen_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nested MLX-VLM Qwen trunk must reach the Qwen sharding strategy."""
    native_model = _NativeQwenLanguageModel()
    generation_model = _FakeVlmModel()
    seen: dict[str, object] = {}

    class _RecordingStrategy:
        def __init__(self, *_args: object) -> None:
            return None

        def shard_model(
            self,
            model: nn.Module,
            _timeout_seconds: float,
            _on_timeout: object,
            _on_layer_loaded: object,
        ) -> nn.Module:
            seen["sharding_model"] = model
            return model

    def _record_patch(model: nn.Module) -> nn.Module:
        seen["generation_model"] = model
        return model

    monkeypatch.setattr(auto_parallel, "QwenShardingStrategy", _RecordingStrategy)
    monkeypatch.setattr(auto_parallel, "patch_tensor_model", _record_patch)
    group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)

    result = auto_parallel.tensor_auto_parallel(
        native_model,
        cast(mx.distributed.Group, cast(object, group)),
        timeout_seconds=1,
        on_timeout=None,
        on_layer_loaded=None,
        generation_model=generation_model,
    )

    assert result is generation_model
    assert seen == {
        "sharding_model": native_model,
        "generation_model": generation_model,
    }


def test_tensor_patch_finds_cache_in_variadic_generation_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper patch must preserve the tensor collective dependency."""
    cache_keys = mx.ones((1,))
    seen: dict[str, object] = {}

    class _CacheEntry:
        def __init__(self) -> None:
            self.keys = cache_keys

    def _record_depends(keys: mx.array, logits: mx.array) -> mx.array:
        seen["keys"] = keys
        seen["logits"] = logits
        return keys

    monkeypatch.setattr(auto_parallel.mx, "depends", _record_depends)
    model = _VariadicGenerationModel()
    patched = auto_parallel.patch_tensor_model(model)

    logits = patched(mx.array([[1]]), cache=[_CacheEntry()])

    assert seen["keys"] is cache_keys
    assert seen["logits"] is logits


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


def test_vlm_wrapper_forwards_and_clears_qwen_image_grid() -> None:
    """Batch prefill must pass Qwen's grid metadata beside its image pixels."""
    inner = _RecordingVlmModel()
    wrapper_class = cast(
        Callable[[nn.Module], nn.Module],
        vars(utils_mlx)["_VlmModelWrapper"],
    )
    wrapper = wrapper_class(inner)
    set_vision_inputs = cast(
        Callable[[mx.array | None, mx.array | None], None],
        object.__getattribute__(wrapper, "set_vision_inputs"),
    )
    pixels = mx.ones((4, 8))
    grid = mx.array([[1, 2, 2]])

    set_vision_inputs(pixels, grid)
    wrapper(mx.array([[1, 2]]))

    assert inner.kwargs["pixel_values"] is pixels
    assert inner.kwargs["image_grid_thw"] is grid

    wrapper(mx.array([[3]]))

    assert "pixel_values" not in inner.kwargs
    assert "image_grid_thw" not in inner.kwargs


def test_vlm_wrapper_uses_native_language_model_cache_factory() -> None:
    """Hybrid VLMs must receive their ArraysCache and KVCache layout intact."""
    wrapper_class = cast(
        Callable[[nn.Module], nn.Module],
        vars(utils_mlx)["_VlmModelWrapper"],
    )
    wrapper = wrapper_class(_VlmWithNativeLanguageCache())
    make_cache = cast(
        Callable[[], list[object]],
        object.__getattribute__(wrapper, "make_cache"),
    )

    assert make_cache() == ["native-ssm-cache", "native-kv-cache"]


def test_gemma4_native_vision_batches_single_image_pixels() -> None:
    """A single CHW image must reach Gemma 4 as a BCHW tensor."""

    batch_pixel_values = cast(
        Callable[[mx.array], mx.array],
        vars(utils_mlx)["_batch_gemma4_pixel_values"],
    )
    single_image = mx.zeros((3, 768, 768))
    already_batched = mx.zeros((1, 3, 768, 768))

    assert batch_pixel_values(single_image).shape == (1, 3, 768, 768)
    assert batch_pixel_values(already_batched) is already_batched


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
