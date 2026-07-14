# pyright: reportPrivateUsage=false
"""Unit tests for the pure helpers of the vLLM served-backend runner.

The live subprocess + streaming path is validated on GPU hardware; these cover
the pure, engine-specific logic: the ``vllm serve`` argument builder, the OpenAI
SSE parser, and the GPU-memory-utilization knob.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from skulk.worker.runner.vllm.runner import (
    _DEFAULT_GPU_MEMORY_UTILIZATION,
    _GPU_MEMORY_UTILIZATION_ENV,
    _gpu_memory_utilization,
    build_vllm_serve_args,
    parse_openai_sse_line,
    vllm_generation_kwargs,
    vllm_reasoning_overrides,
)


def _params(**overrides: object) -> SimpleNamespace:
    base: dict[str, object] = dict(
        max_output_tokens=None,
        temperature=None,
        top_p=None,
        top_k=None,
        min_p=None,
        repetition_penalty=None,
        stop=None,
        seed=None,
        enable_thinking=None,
        reasoning_effort=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_vllm_generation_kwargs_uses_vllm_parameter_names() -> None:
    kwargs = vllm_generation_kwargs(
        _params(
            max_output_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            min_p=0.05,
            repetition_penalty=1.1,
            stop=["</s>"],
            seed=7,
        )
    )
    assert kwargs["max_tokens"] == 256
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.9
    assert kwargs["top_k"] == 40
    assert kwargs["min_p"] == 0.05
    # vLLM's name, not llama.cpp's repeat_penalty (which vLLM would ignore).
    assert kwargs["repetition_penalty"] == 1.1
    assert "repeat_penalty" not in kwargs
    assert kwargs["stop"] == ["</s>"]
    assert kwargs["seed"] == 7


def test_vllm_generation_kwargs_omits_unset() -> None:
    assert vllm_generation_kwargs(_params()) == {}


def test_vllm_reasoning_overrides_maps_thinking_controls() -> None:
    assert vllm_reasoning_overrides(_params(enable_thinking=False)) == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert vllm_reasoning_overrides(_params(reasoning_effort="high")) == {
        "reasoning_effort": "high"
    }
    # "none" effort is not a valid server value; disabling goes via enable_thinking.
    assert vllm_reasoning_overrides(_params(reasoning_effort="none")) == {}
    assert vllm_reasoning_overrides(_params()) == {}


def _serve_args(**overrides: object) -> list[str]:
    kwargs: dict[str, object] = dict(
        binary="/opt/vllm/bin/vllm",
        model_dir=Path("/models/org--repo"),
        served_model_name="org/repo",
        host="127.0.0.1",
        port=51234,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
    )
    kwargs.update(overrides)
    return build_vllm_serve_args(**kwargs)  # type: ignore[arg-type]


def test_build_vllm_serve_args_shape() -> None:
    args = _serve_args()
    assert args[0] == "/opt/vllm/bin/vllm"
    assert args[1] == "serve"
    assert args[2] == "/models/org--repo"
    # served-model-name decouples the addressed id from the on-disk path.
    assert args[args.index("--served-model-name") + 1] == "org/repo"
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("--port") + 1] == "51234"
    assert args[args.index("--max-model-len") + 1] == "8192"
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.90"
    # single-node in this slice.
    assert args[args.index("--tensor-parallel-size") + 1] == "1"


def test_build_vllm_serve_args_trust_remote_code() -> None:
    assert "--trust-remote-code" not in _serve_args(trust_remote_code=False)
    assert "--trust-remote-code" in _serve_args(trust_remote_code=True)


def test_parse_sse_content_delta() -> None:
    line = 'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}'
    delta = parse_openai_sse_line(line)
    assert delta is not None
    assert delta.content == "hello"
    assert delta.reasoning == ""
    assert delta.finish is None
    assert delta.done is False


def test_parse_sse_reasoning_delta() -> None:
    line = 'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}'
    delta = parse_openai_sse_line(line)
    assert delta is not None
    assert delta.reasoning == "think"
    assert delta.content == ""


def test_parse_sse_finish_reason_mapped() -> None:
    line = 'data: {"choices":[{"delta":{"content":""},"finish_reason":"length"}]}'
    delta = parse_openai_sse_line(line)
    assert delta is not None
    assert delta.finish == "length"


def test_parse_sse_preserves_content_filter() -> None:
    # vLLM can emit content_filter; it must not be collapsed to a normal stop.
    line = 'data: {"choices":[{"delta":{"content":""},"finish_reason":"content_filter"}]}'
    delta = parse_openai_sse_line(line)
    assert delta is not None
    assert delta.finish == "content_filter"


def test_parse_sse_done_sentinel() -> None:
    delta = parse_openai_sse_line("data: [DONE]")
    assert delta is not None
    assert delta.done is True


@pytest.mark.parametrize(
    "line",
    [
        "event: ping",  # non-data line
        "data: {not json}",  # malformed json
        'data: {"choices":[]}',  # choice-less payload
        "",  # blank
    ],
)
def test_parse_sse_skips_non_deltas(line: str) -> None:
    assert parse_openai_sse_line(line) is None


def test_gpu_memory_utilization_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_GPU_MEMORY_UTILIZATION_ENV, raising=False)
    assert _gpu_memory_utilization() == _DEFAULT_GPU_MEMORY_UTILIZATION


def test_gpu_memory_utilization_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_GPU_MEMORY_UTILIZATION_ENV, "0.75")
    assert _gpu_memory_utilization() == 0.75


@pytest.mark.parametrize("bad", ["nonsense", "0", "1.5", "-0.2"])
def test_gpu_memory_utilization_rejects_bad_values(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    # Unparseable or out-of-(0,1] values fall back to the default rather than
    # passing vLLM a fraction that would fail the server at spawn.
    monkeypatch.setenv(_GPU_MEMORY_UTILIZATION_ENV, bad)
    assert _gpu_memory_utilization() == _DEFAULT_GPU_MEMORY_UTILIZATION
