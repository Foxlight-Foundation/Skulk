# pyright: reportPrivateUsage=false, reportAny=false, reportUnknownMemberType=false
"""Tests for the pure helpers of the llama.cpp runner (no llama_cpp needed)."""

from pathlib import Path

import pytest

from skulk.shared.models.memory_estimate import KV_CONTEXT_BUDGET_TOKENS
from skulk.shared.types.common import ModelId
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.worker.runner.llama_cpp.runner import (
    _DEFAULT_VISION_HANDLER,
    _VISION_HANDLER_BY_MODEL_TYPE,
    _flash_attn_enabled,
    _generation_kwargs,
    _image_data_uri,
    _logits_all_enabled,
    _logits_all_n_ctx,
    _logprob_fields,
    _map_finish_reason,
    _serving_n_ctx,
    _splice_images_into_messages,
    _tool_calls_from_message,
    find_mmproj_file,
    logprobs_unavailable_error,
    messages_for_llama,
    select_gguf_file,
)


def test_logprobs_request_without_logits_all_returns_clear_error() -> None:
    msg = logprobs_unavailable_error(
        logprobs=True, top_logprobs=None, logits_all_on=False
    )
    assert msg is not None
    assert "SKULK_LLAMA_CPP_LOGITS_ALL=1" in msg


def test_top_logprobs_alone_is_treated_as_a_logprobs_request() -> None:
    # OpenAI treats top_logprobs (with logprobs unset) as a logprobs request, so
    # it must also trip the guard rather than silently returning none.
    msg = logprobs_unavailable_error(
        logprobs=False, top_logprobs=5, logits_all_on=False
    )
    assert msg is not None
    assert "SKULK_LLAMA_CPP_LOGITS_ALL=1" in msg


def test_logprobs_request_with_logits_all_proceeds() -> None:
    assert (
        logprobs_unavailable_error(logprobs=True, top_logprobs=5, logits_all_on=True)
        is None
    )


def test_no_logprobs_request_never_errors() -> None:
    assert (
        logprobs_unavailable_error(logprobs=False, top_logprobs=None, logits_all_on=False)
        is None
    )
    assert (
        logprobs_unavailable_error(logprobs=False, top_logprobs=None, logits_all_on=True)
        is None
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
    # top_logprobs set alone implies logprobs (OpenAI semantics): flag on too,
    # so the model actually returns logprobs instead of silently none.
    kw3 = _generation_kwargs(_params(top_logprobs=5))
    assert kw3["logprobs"] is True and kw3["top_logprobs"] == 5


def test_logits_all_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    # Default OFF: logits_all at full context pre-allocates an n_ctx*vocab*4
    # buffer that OOMs the node on load, so logprobs is opt-in.
    monkeypatch.delenv("SKULK_LLAMA_CPP_LOGITS_ALL", raising=False)
    assert _logits_all_enabled() is False  # default off (avoids the OOM)
    monkeypatch.setenv("SKULK_LLAMA_CPP_LOGITS_ALL", "1")
    assert _logits_all_enabled() is True  # explicit opt-in
    monkeypatch.setenv("SKULK_LLAMA_CPP_LOGITS_ALL", "0")
    assert _logits_all_enabled() is False
    # case-insensitive truthy strings also opt in (matches repo env convention)
    for truthy in ("true", "TRUE", "Yes", " on "):
        monkeypatch.setenv("SKULK_LLAMA_CPP_LOGITS_ALL", truthy)
        assert _logits_all_enabled() is True
    monkeypatch.setenv("SKULK_LLAMA_CPP_LOGITS_ALL", "off")
    assert _logits_all_enabled() is False


def test_flash_attn_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    # Default ON: Flash Attention is the modern llama.cpp default and fixes the
    # gemma full-size-SWA-cache + V-cache-padding slow path.
    monkeypatch.delenv("SKULK_LLAMA_CPP_FLASH_ATTN", raising=False)
    assert _flash_attn_enabled() is True  # default on
    # Explicit opt-out for backends whose build lacks Flash Attention kernels.
    monkeypatch.setenv("SKULK_LLAMA_CPP_FLASH_ATTN", "0")
    assert _flash_attn_enabled() is False
    for falsy in ("0", "false", "FALSE", "no", " off "):
        monkeypatch.setenv("SKULK_LLAMA_CPP_FLASH_ATTN", falsy)
        assert _flash_attn_enabled() is False
    for truthy in ("1", "true", "Yes", " on "):
        monkeypatch.setenv("SKULK_LLAMA_CPP_FLASH_ATTN", truthy)
        assert _flash_attn_enabled() is True


def test_logits_all_n_ctx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SKULK_LLAMA_CPP_LOGITS_ALL_N_CTX", raising=False)
    assert _logits_all_n_ctx() == 8192  # bounded default, not the full context
    monkeypatch.setenv("SKULK_LLAMA_CPP_LOGITS_ALL_N_CTX", "16384")
    assert _logits_all_n_ctx() == 16384
    # garbage / non-positive falls back to the safe default, never 0 (full ctx)
    monkeypatch.setenv("SKULK_LLAMA_CPP_LOGITS_ALL_N_CTX", "0")
    assert _logits_all_n_ctx() == 8192
    monkeypatch.setenv("SKULK_LLAMA_CPP_LOGITS_ALL_N_CTX", "abc")
    assert _logits_all_n_ctx() == 8192


def test_serving_n_ctx_capped_to_placement_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SKULK_LLAMA_CPP_LOGITS_ALL_N_CTX", raising=False)
    # llama.cpp allocates the KV cache up front, so n_ctx is capped to the budget
    # placement reserved (KV_CONTEXT_BUDGET_TOKENS), NEVER the full trained context
    # (n_ctx=0) and NEVER the larger request-admission ceiling -- either would
    # exceed reserved memory and OOM-kill the node (the bug this fixes).
    assert _serving_n_ctx(32768, logits_all=False) == KV_CONTEXT_BUDGET_TOKENS
    assert _serving_n_ctx(None, logits_all=False) == KV_CONTEXT_BUDGET_TOKENS
    assert _serving_n_ctx(0, logits_all=False) == KV_CONTEXT_BUDGET_TOKENS
    # On a degenerate tiny node whose admission ceiling is even smaller than the
    # budget, clamp down to it (never allocate more than admitted).
    assert _serving_n_ctx(4096, logits_all=False) == 4096
    # With logits_all on, the logits-buffer window further bounds it (a smaller
    # window wins), never raising it above the budget.
    monkeypatch.setenv("SKULK_LLAMA_CPP_LOGITS_ALL_N_CTX", "2048")
    assert _serving_n_ctx(32768, logits_all=True) == 2048
    monkeypatch.setenv("SKULK_LLAMA_CPP_LOGITS_ALL_N_CTX", "16384")
    assert _serving_n_ctx(32768, logits_all=True) == KV_CONTEXT_BUDGET_TOKENS


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


# --- vision (#128) ---------------------------------------------------------


def test_image_data_uri_sniffs_png_and_jpeg() -> None:
    import base64

    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16).decode()
    jpeg = base64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 16).decode()
    assert _image_data_uri(png).startswith("data:image/png;base64,")
    assert _image_data_uri(jpeg).startswith("data:image/jpeg;base64,")
    # Unrecognized bytes default to png rather than raising.
    assert _image_data_uri(base64.b64encode(b"zzzz").decode()).startswith(
        "data:image/png;base64,"
    )


def test_splice_images_replaces_placeholders_in_order() -> None:
    messages = [
        {"role": "system", "content": "be brief"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "compare"},
                {"type": "image"},
                {"type": "image"},
            ],
        },
    ]
    import base64

    a = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
    b = base64.b64encode(b"\xff\xd8\xff\xe0").decode()
    out = _splice_images_into_messages(messages, [a, b])
    # system message untouched (string content)
    assert out[0] == {"role": "system", "content": "be brief"}
    parts = out[1]["content"]
    assert parts[0] == {"type": "text", "text": "compare"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parts[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_splice_images_noop_without_images() -> None:
    messages = [{"role": "user", "content": "hi"}]
    assert _splice_images_into_messages(messages, []) is messages


def test_splice_images_drops_extra_placeholder() -> None:
    # More placeholders than images: stray placeholder dropped, not malformed.
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "image"}]}
    ]
    import base64

    only = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
    out = _splice_images_into_messages(messages, [only])
    parts = out[0]["content"]
    assert len(parts) == 1
    assert parts[0]["type"] == "image_url"


def test_messages_for_llama_splices_images() -> None:
    import base64

    img = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
    params = _params(
        chat_template_messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image"},
                ],
            }
        ],
        images=[img],
    )
    out = messages_for_llama(params)
    assert out[0]["content"][1]["type"] == "image_url"


def test_find_mmproj_file(tmp_path: Path) -> None:
    (tmp_path / "model-Q4_K_M.gguf").touch()
    assert find_mmproj_file(tmp_path) is None
    (tmp_path / "mmproj-model-f16.gguf").touch()
    found = find_mmproj_file(tmp_path)
    assert found is not None and "mmproj" in found.name.lower()


def test_vision_handler_map_defaults_to_mtmd() -> None:
    # Known families map to a bespoke handler; unknown falls back to MTMD.
    assert _VISION_HANDLER_BY_MODEL_TYPE["qwen2.5-vl"] == "Qwen25VLChatHandler"
    assert _VISION_HANDLER_BY_MODEL_TYPE.get("some-new-vlm") is None
    assert _DEFAULT_VISION_HANDLER == "MTMDChatHandler"
