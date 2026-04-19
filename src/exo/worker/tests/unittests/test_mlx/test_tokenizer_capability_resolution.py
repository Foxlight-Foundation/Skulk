from pathlib import Path
from typing import Callable, cast

import pytest

from exo.shared.models.model_cards import (
    ModelCard,
    ModelId,
    ModelTask,
    ToolCallFormat,
    ToolingCardConfig,
)
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.mlx import utils_mlx as utils_mlx_module
from exo.worker.engines.mlx.utils_mlx import get_tokenizer, load_tokenizer_for_model_id


class _FakeTokenizer:
    def __init__(self) -> None:
        self.eos_token_ids: list[int] | None = None
        self.tool_call_start: str | None = None
        self.tool_call_end: str | None = None
        self.tool_parser: Callable[[str], list[dict[str, object]]] | None = None


def _gemma4_tool_parser() -> Callable[[str], list[dict[str, object]]]:
    module_dict = cast(dict[str, object], utils_mlx_module.__dict__)
    return cast(
        Callable[[str], list[dict[str, object]]],
        module_dict["_parse_gemma4_tool_calls"],
    )


def test_load_tokenizer_for_model_id_uses_explicit_model_card_when_cache_is_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    card = ModelCard(
        model_id=ModelId("custom/tool-model"),
        storage_size=Memory.from_mb(100),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        capabilities=["text"],
        tooling=ToolingCardConfig(tool_call_format=ToolCallFormat.Gemma4),
    )

    def _missing_card(_model_id: ModelId) -> None:
        return None

    def _fake_load_tokenizer(*_args: object, **_kwargs: object) -> _FakeTokenizer:
        return _FakeTokenizer()

    monkeypatch.setattr(
        "exo.worker.engines.mlx.utils_mlx.get_card",
        _missing_card,
    )
    monkeypatch.setattr(
        "exo.worker.engines.mlx.utils_mlx.load_tokenizer",
        _fake_load_tokenizer,
    )

    tokenizer = cast(
        _FakeTokenizer,
        cast(
            object,
            load_tokenizer_for_model_id(
                card.model_id,
                tmp_path,
                model_card=card,
            ),
        ),
    )

    assert tokenizer.tool_call_start == "<|tool_call>"
    assert tokenizer.tool_call_end == "<tool_call|>"
    assert tokenizer.tool_parser is _gemma4_tool_parser()


def test_get_tokenizer_passes_shard_model_card_to_tokenizer_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    card = ModelCard(
        model_id=ModelId("custom/tool-model"),
        storage_size=Memory.from_mb(100),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        capabilities=["text"],
        tooling=ToolingCardConfig(tool_call_format=ToolCallFormat.Gemma4),
    )
    shard_metadata = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )

    captured: dict[str, object] = {}

    def _fake_loader(
        model_id: ModelId,
        model_path: Path,
        *,
        model_card: ModelCard | None = None,
        trust_remote_code: bool = False,
    ) -> _FakeTokenizer:
        captured["model_id"] = model_id
        captured["model_path"] = model_path
        captured["model_card"] = model_card
        captured["trust_remote_code"] = trust_remote_code
        return _FakeTokenizer()

    monkeypatch.setattr(
        "exo.worker.engines.mlx.utils_mlx.load_tokenizer_for_model_id",
        _fake_loader,
    )

    get_tokenizer(tmp_path, shard_metadata)

    assert captured["model_id"] == card.model_id
    assert captured["model_path"] == tmp_path
    assert captured["model_card"] == card
    assert captured["trust_remote_code"] == card.trust_remote_code
