# pyright: reportPrivateUsage=false
"""Tests for the pure helpers of the llama.cpp runner (no llama_cpp needed)."""

from pathlib import Path

import pytest

from skulk.shared.types.common import ModelId
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.worker.runner.llama_cpp.runner import (
    _generation_kwargs,
    _map_finish_reason,
    messages_for_llama,
    select_gguf_file,
)


def test_select_gguf_picks_first_shard_skips_mmproj(tmp_path: Path) -> None:
    (tmp_path / "mmproj-model.gguf").touch()  # vision projector, must be skipped
    (tmp_path / "model-00002-of-00002.gguf").touch()
    (tmp_path / "model-00001-of-00002.gguf").touch()
    assert select_gguf_file(tmp_path).name == "model-00001-of-00002.gguf"


def test_select_gguf_raises_when_absent(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        select_gguf_file(tmp_path)


def test_map_finish_reason() -> None:
    assert _map_finish_reason(None) is None
    assert _map_finish_reason("length") == "length"
    assert _map_finish_reason("stop") == "stop"
    assert _map_finish_reason("eos_token") == "stop"  # anything non-length -> stop


def _params(**kw: object) -> TextGenerationTaskParams:
    base: dict[str, object] = {"model": ModelId("m"), "input": []}
    base.update(kw)
    return TextGenerationTaskParams.model_validate(base)


def test_generation_kwargs_only_includes_set_fields() -> None:
    assert _generation_kwargs(_params()) == {}
    kwargs = _generation_kwargs(
        _params(max_output_tokens=128, temperature=0.7, top_p=0.9, stop=["X"])
    )
    assert kwargs == {
        "max_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9,
        "stop": ["X"],
    }


def test_generation_kwargs_maps_repetition_penalty() -> None:
    assert _generation_kwargs(_params(repetition_penalty=1.1))["repeat_penalty"] == 1.1


def test_messages_prefers_chat_template_messages() -> None:
    msgs = [{"role": "user", "content": "hi"}]
    assert messages_for_llama(_params(chat_template_messages=msgs)) == msgs


def test_messages_fallback_from_input_and_instructions() -> None:
    params = _params(
        instructions="be brief",
        input=[InputMessage(role="user", content="hello")],
    )
    result = messages_for_llama(params)
    assert result[0] == {"role": "system", "content": "be brief"}
    assert result[1]["role"] == "user"
