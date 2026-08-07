"""Regression tests that serving logs retain shape, not user/model payloads."""

from __future__ import annotations

import io
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from typing import cast

import pytest
from loguru import logger
from mlx_lm.tokenizer_utils import TokenizerWrapper

from skulk.shared.models.model_cards import (
    ModelCard,
    ModelTask,
    PromptRendererType,
    RuntimeCapabilityCardConfig,
)
from skulk.shared.types.common import ModelId
from skulk.shared.types.memory import Memory
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.shared.types.worker.runner_response import GenerationResponse
from skulk.worker.engines.mlx import utils_mlx as utils_mlx_module
from skulk.worker.engines.mlx import vision as vision_module
from skulk.worker.engines.mlx.dsml_encoding import (
    TOOL_CALLS_END,
    TOOL_CALLS_START,
)
from skulk.worker.engines.mlx.utils_mlx import apply_chat_template
from skulk.worker.runner.llm_inference.model_output_parsers import (
    parse_deepseek_v32,
    parse_gpt_oss,
    parse_tool_calls,
)
from skulk.worker.runner.llm_inference.tool_parsers import make_mlx_parser

_PAYLOAD_SENTINEL = "PRIVATE-PAYLOAD-SENTINEL-8f41"


@contextmanager
def _captured_logs() -> Iterator[io.StringIO]:
    """Capture Loguru messages emitted by the exercised serving path."""
    stream = io.StringIO()
    handler_id = logger.add(stream, format="{message}")
    try:
        yield stream
    finally:
        logger.remove(handler_id)


class _PromptEchoTokenizer:
    """Minimal tokenizer that makes prompt-content leakage observable."""

    chat_template = "{{ messages }}"

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        tools: list[dict[str, object]] | None = None,
        **_kwargs: object,
    ) -> str:
        del tokenize, add_generation_prompt, tools
        return repr(messages)


def _text_card(renderer: PromptRendererType) -> ModelCard:
    """Build a minimal card that selects one prompt renderer."""
    return ModelCard(
        model_id=ModelId(f"test/payload-free-{renderer.value}"),
        storage_size=Memory.from_mb(100),
        n_layers=2,
        hidden_size=128,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        runtime=RuntimeCapabilityCardConfig(prompt_renderer=renderer),
    )


@pytest.mark.parametrize("renderer", list(PromptRendererType))
def test_rendered_prompt_logs_only_shape(renderer: PromptRendererType) -> None:
    """Every renderer must keep the prompt payload out of ordinary logs."""
    task = TextGenerationTaskParams(
        model=ModelId(f"test/payload-free-{renderer.value}"),
        input=[InputMessage(role="user", content=_PAYLOAD_SENTINEL)],
    )
    tokenizer = cast(
        TokenizerWrapper,
        cast(object, _PromptEchoTokenizer()),
    )

    with _captured_logs() as captured:
        prompt = apply_chat_template(
            tokenizer,
            task,
            model_card=_text_card(renderer),
        )

    logs = captured.getvalue()
    assert _PAYLOAD_SENTINEL in prompt
    assert _PAYLOAD_SENTINEL not in logs
    assert f"renderer={renderer.value}" in logs
    assert "messages=1" in logs
    assert "chars=" in logs


def test_vision_prompt_logs_only_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vision prompt construction must not log even truncated user messages."""
    monkeypatch.delenv("SKULK_TRACE_REQUEST_SHAPES", raising=False)
    tokenizer = cast(
        TokenizerWrapper,
        cast(object, _PromptEchoTokenizer()),
    )

    with _captured_logs() as captured:
        built = vision_module._build_vision_prompt_with_debug(  # pyright: ignore[reportPrivateUsage]
            tokenizer,
            [{"role": "user", "content": f"<image>{_PAYLOAD_SENTINEL}"}],
            [2],
            "<image>",
        )

    logs = captured.getvalue()
    assert _PAYLOAD_SENTINEL in built.prompt
    assert _PAYLOAD_SENTINEL not in logs
    assert "messages=1" in logs
    assert "images=1" in logs


def _responses(text: str) -> Generator[GenerationResponse]:
    """Yield one completed generation response."""
    yield GenerationResponse(
        text=text,
        token=0,
        finish_reason="stop",
        usage=None,
    )


def test_generic_tool_parser_logs_only_shape() -> None:
    """Generic tool parsing failures must not retain generated arguments."""

    def _fail(_text: str) -> dict[str, object]:
        raise ValueError("expected parser failure")

    generated = f"<tool_call>{_PAYLOAD_SENTINEL}</tool_call>"
    parser = make_mlx_parser("<tool_call>", "</tool_call>", _fail)

    with _captured_logs() as captured:
        results = list(parse_tool_calls(_responses(generated), parser, tools=None))

    logs = captured.getvalue()
    assert _PAYLOAD_SENTINEL in cast(GenerationResponse, results[0]).text
    assert _PAYLOAD_SENTINEL not in logs
    assert "generated_chars=" in logs


def test_dsml_parser_logs_only_shape() -> None:
    """DSML parsing failures must not retain generated content."""
    generated = f"{TOOL_CALLS_START}{_PAYLOAD_SENTINEL}{TOOL_CALLS_END}"

    with _captured_logs() as captured:
        results = list(parse_deepseek_v32(_responses(generated)))

    logs = captured.getvalue()
    assert _PAYLOAD_SENTINEL in cast(GenerationResponse, results[0]).text
    assert _PAYLOAD_SENTINEL not in logs
    assert "generated_chars=" in logs


def test_gpt_oss_parser_logs_only_shape() -> None:
    """GPT-OSS token diagnostics must not retain generated content."""
    generated = f"ordinary generated text {_PAYLOAD_SENTINEL}"

    with _captured_logs() as captured:
        list(parse_gpt_oss(_responses(generated)))

    logs = captured.getvalue()
    assert _PAYLOAD_SENTINEL not in logs
    assert "text_chars=" in logs


def test_gemma4_parser_logs_and_errors_only_shape() -> None:
    """Gemma 4 parse diagnostics must not retain tool arguments or output."""
    invalid_args = f"call:lookup{{query:{_PAYLOAD_SENTINEL}}}"
    no_call = f"ordinary generated text {_PAYLOAD_SENTINEL}"

    with _captured_logs() as captured:
        parsed = utils_mlx_module._parse_gemma4_tool_calls(invalid_args)  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(ValueError) as error:
            utils_mlx_module._parse_gemma4_tool_calls(no_call)  # pyright: ignore[reportPrivateUsage]

    logs = captured.getvalue()
    assert parsed[0]["arguments"] == {}
    assert _PAYLOAD_SENTINEL not in logs
    assert _PAYLOAD_SENTINEL not in str(error.value)
    assert "argument_chars=" in logs
