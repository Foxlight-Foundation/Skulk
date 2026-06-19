# pyright: reportPrivateUsage=false
"""Tests for GGUF-repo detection and llama.cpp card creation (slice 3a)."""

from types import SimpleNamespace

import pytest

from skulk.shared.models import model_cards
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelId,
    _gguf_shard_base,
    gguf_weight_siblings,
)


def _fake_model_info(filenames: list[str]):
    """A stand-in for huggingface_hub.model_info with files_metadata=True."""

    def _factory(_model_id: object, files_metadata: bool = False) -> object:
        siblings = [SimpleNamespace(rfilename=name, size=100) for name in filenames]
        return SimpleNamespace(siblings=siblings, safetensors=None)

    return _factory


def test_gguf_weight_siblings_filters_gguf_and_mmproj(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_cards,
        "model_info",
        _fake_model_info(
            ["model.gguf", "mmproj-model.gguf", "config.json", "README.md"]
        ),
    )
    siblings = gguf_weight_siblings(ModelId("some/gguf-repo"))
    names = {name for name, _ in siblings}
    assert names == {"model.gguf"}  # .gguf only, mmproj excluded


def test_non_gguf_repo_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        model_cards,
        "model_info",
        _fake_model_info(["model.safetensors", "config.json"]),
    )
    assert gguf_weight_siblings(ModelId("some/mlx-repo")) == []


def test_shard_base_detection() -> None:
    assert _gguf_shard_base("model.gguf") is None
    assert _gguf_shard_base("model-00001-of-00003.gguf") == "model"
    assert (
        _gguf_shard_base("Qwen2.5-7B-Instruct-Q4_K_M-00001-of-00003.gguf")
        == "Qwen2.5-7B-Instruct-Q4_K_M"
    )


async def test_fetch_gguf_card_stamps_llama_cpp_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_cards, "model_info", _fake_model_info(["model-q4.gguf"]))

    async def _fake_config(_model_id: object) -> object:
        return SimpleNamespace(
            layer_count=32,
            hidden_size=4096,
            num_key_value_heads=8,
            max_position_embeddings=8192,
        )

    monkeypatch.setattr(model_cards, "fetch_config_data", _fake_config)

    card = await ModelCard.fetch_from_hf(ModelId("some/gguf-repo"))
    assert card.placement.compatible_backends == frozenset(
        {"llama_cpp-vulkan", "llama_cpp-rocm", "llama_cpp-cuda", "llama_cpp-cpu"}
    )
    assert card.placement.backend_preference[0] == "llama_cpp-vulkan"
    assert card.supports_tensor is False  # single-node engine
    assert card.n_layers == 32 and card.hidden_size == 4096
    assert card.storage_size.in_bytes == 100  # the single selected gguf


async def test_fetch_gguf_card_without_config_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A bare GGUF repo (no config.json) needs the GGUF-header parse (#327); until
    # then we fail with a clear, actionable error rather than fabricate metadata.
    monkeypatch.setattr(model_cards, "model_info", _fake_model_info(["model-q4.gguf"]))

    async def _raises(_model_id: object) -> object:
        raise FileNotFoundError("no config.json in this bare GGUF repo")

    monkeypatch.setattr(model_cards, "fetch_config_data", _raises)

    with pytest.raises(ValueError, match="#327"):
        await ModelCard.fetch_from_hf(ModelId("bare/gguf-repo"))


def test_select_preferred_gguf_prefers_quant_over_bf16() -> None:
    from skulk.shared.models.model_cards import (
        gguf_allow_patterns,
        gguf_shard_group_size,
        select_preferred_gguf,
    )

    files = [
        ("M-BF16.gguf", 2_000),
        ("M-Q4_K_M.gguf", 800),
        ("M-Q8_0.gguf", 1_300),
    ]
    sel = select_preferred_gguf(files)
    assert sel == "M-Q4_K_M.gguf"  # quant beats BF16; Q4_K_M is top preference
    assert gguf_shard_group_size(sel, files).in_bytes == 800
    assert gguf_allow_patterns(sel) == ["M-Q4_K_M.gguf"]


def test_select_preferred_gguf_sharded_group() -> None:
    from skulk.shared.models.model_cards import (
        gguf_allow_patterns,
        gguf_shard_group_size,
        select_preferred_gguf,
    )

    files = [
        ("big-Q4_K_M-00001-of-00002.gguf", 500),
        ("big-Q4_K_M-00002-of-00002.gguf", 600),
        ("big-BF16.gguf", 4_000),
    ]
    sel = select_preferred_gguf(files)
    assert sel == "big-Q4_K_M-00001-of-00002.gguf"
    assert gguf_shard_group_size(sel, files).in_bytes == 1_100  # both shards
    assert gguf_allow_patterns(sel) == ["big-Q4_K_M-*-of-*.gguf"]


async def test_gguf_card_pins_selected_quant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        model_cards,
        "model_info",
        _fake_model_info(["model-BF16.gguf", "model-Q4_K_M.gguf"]),
    )

    async def _cfg(_m: object) -> object:
        return SimpleNamespace(
            layer_count=16,
            hidden_size=2048,
            num_key_value_heads=8,
            max_position_embeddings=8192,
        )

    monkeypatch.setattr(model_cards, "fetch_config_data", _cfg)
    card = await ModelCard.fetch_from_hf(ModelId("some/gguf-repo"))
    assert card.gguf_file == "model-Q4_K_M.gguf"
    assert card.quantization == "Q4_K_M"
