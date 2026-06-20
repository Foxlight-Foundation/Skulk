# pyright: reportPrivateUsage=false
"""Tests for the pure helpers of the llama.cpp runner (no llama_cpp needed)."""

from pathlib import Path

import pytest

from skulk.shared.types.common import ModelId
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.worker.runner.llama_cpp.runner import (
    _generation_kwargs,
    _logits_all_enabled,
    _logprob_fields,
    _map_finish_reason,
    _tool_calls_from_message,
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


def test_generation_kwargs_passes_logprobs() -> None:
    assert "logprobs" not in _generation_kwargs(_params())
    kw = _generation_kwargs(_params(logprobs=True, top_logprobs=3))
    assert kw["logprobs"] is True and kw["top_logprobs"] == 3
    # logprobs requested without a top-N: flag on, no top_logprobs key
    kw2 = _generation_kwargs(_params(logprobs=True))
    assert kw2["logprobs"] is True and "top_logprobs" not in kw2


def test_logits_all_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SKULK_LLAMA_CPP_LOGITS_ALL", raising=False)
    assert _logits_all_enabled() is True  # default on (logprobs parity)
    monkeypatch.setenv("SKULK_LLAMA_CPP_LOGITS_ALL", "0")
    assert _logits_all_enabled() is False  # explicit opt-out
    monkeypatch.setenv("SKULK_LLAMA_CPP_LOGITS_ALL", "1")
    assert _logits_all_enabled() is True


def test_tool_calls_from_message() -> None:
    msg = {
        "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city":"SF"}'}},
            {"function": {"name": "noid", "arguments": "{}"}},  # no id -> still parsed
            {"function": {"arguments": "{}"}},  # no name -> skipped
        ]
    }
    items = _tool_calls_from_message(msg)
    assert [i.name for i in items] == ["get_weather", "noid"]
    assert items[0].id == "call_1"
    assert items[0].arguments == '{"city":"SF"}'


def test_tool_calls_from_message_none() -> None:
    assert _tool_calls_from_message({"content": "hi"}) == []


def test_logprob_fields_parses_openai_shape() -> None:
    choice = {
        "logprobs": {
            "content": [
                {"token": "Hi", "logprob": -0.2,
                 "top_logprobs": [{"token": "Hi", "logprob": -0.2},
                                  {"token": "Hey", "logprob": -1.5}]}
            ]
        }
    }
    lp, top = _logprob_fields(choice)
    assert lp == -0.2
    assert top is not None and [t.token for t in top] == ["Hi", "Hey"]


def test_logprob_fields_absent_or_malformed() -> None:
    assert _logprob_fields({}) == (None, None)
    assert _logprob_fields({"logprobs": None}) == (None, None)
    assert _logprob_fields({"logprobs": {"content": []}}) == (None, None)
