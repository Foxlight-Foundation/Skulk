# pyright: reportPrivateUsage=false
"""Tests for GGUF-repo detection and llama.cpp card creation (slice 3a)."""

import struct
from collections.abc import Awaitable, Callable
from types import SimpleNamespace

import pytest

from skulk.shared.models import model_cards
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelId,
    _gguf_shard_base,
    gguf_weight_siblings,
)

# --- GGUF binary header builders (for #327 header-parse tests) --------------
#
# Minimal encoders for the GGUF metadata block: magic, version, tensor count,
# kv count, then typed key/value pairs. Mirrors the spec well enough to exercise
# read_gguf_structural_fields without a real multi-GB weights file.

_GGUF_T_FLOAT32 = 6
_GGUF_T_UINT32 = 4
_GGUF_T_STRING = 8
_GGUF_T_ARRAY = 9


def _g_str(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _kv_string(key: str, value: str) -> bytes:
    return _g_str(key) + struct.pack("<I", _GGUF_T_STRING) + _g_str(value)


def _kv_u32(key: str, value: int) -> bytes:
    return _g_str(key) + struct.pack("<I", _GGUF_T_UINT32) + struct.pack("<I", value)


def _kv_string_array(key: str, values: list[str]) -> bytes:
    body = struct.pack("<I", _GGUF_T_STRING) + struct.pack("<Q", len(values))
    for value in values:
        body += _g_str(value)
    return _g_str(key) + struct.pack("<I", _GGUF_T_ARRAY) + body


def _kv_f32_array(key: str, values: list[float]) -> bytes:
    body = struct.pack("<I", _GGUF_T_FLOAT32) + struct.pack("<Q", len(values))
    for value in values:
        body += struct.pack("<f", value)
    return _g_str(key) + struct.pack("<I", _GGUF_T_ARRAY) + body


def _build_gguf(kvs: list[bytes], *, version: int = 3, tensor_count: int = 0) -> bytes:
    return (
        model_cards._GGUF_MAGIC
        + struct.pack("<I", version)
        + struct.pack("<Q", tensor_count)
        + struct.pack("<Q", len(kvs))
        + b"".join(kvs)
    )


def _mem_fetch(blob: bytes) -> "Callable[[int, int], Awaitable[bytes]]":
    async def _fetch(offset: int, length: int) -> bytes:
        return blob[offset : offset + length]

    return _fetch


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


async def test_fetch_gguf_card_reads_header_when_no_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A bare GGUF repo (no config.json) has its structural fields read from the
    # GGUF binary header instead (#327), rather than failing or fabricating them.
    monkeypatch.setattr(model_cards, "model_info", _fake_model_info(["model-q4.gguf"]))

    async def _raises(_model_id: object) -> object:
        raise FileNotFoundError("no config.json in this bare GGUF repo")

    monkeypatch.setattr(model_cards, "fetch_config_data", _raises)

    blob = _build_gguf(
        [
            _kv_string("general.architecture", "llama"),
            _kv_u32("llama.block_count", 24),
            _kv_u32("llama.embedding_length", 3072),
            _kv_u32("llama.attention.head_count_kv", 6),
            _kv_u32("llama.context_length", 16384),
        ]
    )

    from skulk.download import download_utils

    async def _range(
        _model_id: object, _revision: str, _path: str, start: int, length: int
    ) -> bytes:
        return blob[start : start + length]

    monkeypatch.setattr(download_utils, "range_read", _range)

    card = await ModelCard.fetch_from_hf(ModelId("bare/gguf-repo"))
    assert card.n_layers == 24 and card.hidden_size == 3072
    assert card.num_key_value_heads == 6 and card.context_length == 16384
    assert card.gguf_file == "model-q4.gguf"


async def test_read_gguf_structural_fields_basic() -> None:
    blob = _build_gguf(
        [
            _kv_string("general.architecture", "llama"),
            _kv_u32("llama.block_count", 32),
            _kv_u32("llama.embedding_length", 4096),
            _kv_u32("llama.attention.head_count_kv", 8),
            _kv_u32("llama.context_length", 8192),
        ]
    )
    fields = await model_cards.read_gguf_structural_fields(_mem_fetch(blob))
    assert fields == model_cards.GgufStructuralFields(32, 4096, 8, 8192)


async def test_read_gguf_skips_arrays_before_structural_keys() -> None:
    # A multi-thousand-entry tokenizer array sitting before the structural keys
    # must be skipped, not parsed into the metadata, and not block the read.
    blob = _build_gguf(
        [
            _kv_string("general.architecture", "qwen2"),
            _kv_string_array("tokenizer.ggml.tokens", [f"t{i}" for i in range(5000)]),
            _kv_f32_array("tokenizer.ggml.scores", [0.1] * 5000),
            _kv_u32("qwen2.block_count", 28),
            _kv_u32("qwen2.embedding_length", 3584),
            _kv_u32("qwen2.attention.head_count_kv", 4),
            _kv_u32("qwen2.context_length", 32768),
        ]
    )
    fields = await model_cards.read_gguf_structural_fields(_mem_fetch(blob))
    assert fields.n_layers == 28 and fields.hidden_size == 3584
    assert fields.num_key_value_heads == 4 and fields.context_length == 32768


async def test_read_gguf_missing_kv_heads_is_none() -> None:
    blob = _build_gguf(
        [
            _kv_string("general.architecture", "llama"),
            _kv_u32("llama.block_count", 16),
            _kv_u32("llama.embedding_length", 2048),
            _kv_u32("llama.context_length", 4096),
        ]
    )
    fields = await model_cards.read_gguf_structural_fields(_mem_fetch(blob))
    assert fields.num_key_value_heads is None
    assert fields.n_layers == 16 and fields.context_length == 4096


async def test_read_gguf_bad_magic_raises() -> None:
    with pytest.raises(ValueError, match="not a GGUF"):
        await model_cards.read_gguf_structural_fields(_mem_fetch(b"XXXX" + b"\x00" * 64))


async def test_read_gguf_unsupported_version_raises() -> None:
    blob = (
        model_cards._GGUF_MAGIC
        + struct.pack("<I", 1)  # v1: 32-bit lengths, obsolete
        + struct.pack("<Q", 0)
        + struct.pack("<Q", 0)
    )
    with pytest.raises(ValueError, match="version"):
        await model_cards.read_gguf_structural_fields(_mem_fetch(blob))


async def test_read_gguf_missing_architecture_raises() -> None:
    blob = _build_gguf([_kv_u32("llama.block_count", 8)])
    with pytest.raises(ValueError, match="architecture"):
        await model_cards.read_gguf_structural_fields(_mem_fetch(blob))


async def test_read_gguf_windowed_fetch() -> None:
    # A transport that returns only a few bytes per call must be reassembled,
    # not mistaken for EOF after the first short read.
    blob = _build_gguf(
        [
            _kv_string("general.architecture", "llama"),
            _kv_u32("llama.block_count", 10),
            _kv_u32("llama.embedding_length", 100),
            _kv_u32("llama.attention.head_count_kv", 2),
            _kv_u32("llama.context_length", 2048),
        ]
    )
    calls = 0

    async def _chunked(offset: int, length: int) -> bytes:
        nonlocal calls
        calls += 1
        return blob[offset : offset + min(length, 7)]

    fields = await model_cards.read_gguf_structural_fields(_chunked)
    assert fields.n_layers == 10 and fields.hidden_size == 100
    assert calls > 1  # required multiple fetches to assemble the header


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
