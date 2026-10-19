"""Engine refusals retain context classification through both proxy paths."""

import httpx
import pytest

from skulk.shared.constants import CONTEXT_LENGTH_EXCEEDED_PREFIX
from skulk.worker.runner.vllm.server_errors import (
    VllmServerRefusalError,
    raise_for_vllm_status,
)

_REQUEST = httpx.Request("POST", "http://127.0.0.1:1/v1/chat/completions")


@pytest.mark.parametrize("streamed", [False, True])
def test_context_refusal_preserves_engine_reason(streamed: bool) -> None:
    """vLLM's token overflow maps to the API's context error envelope."""
    body = (
        b'{"error":{"message":"This model\'s maximum context length is 4096 '
        b'tokens. However, your prompt contains 5001 input tokens.",'
        b'"type":"BadRequestError","param":"input_tokens","code":400}}'
    )
    response = (
        httpx.Response(400, request=_REQUEST, stream=httpx.ByteStream(body))
        if streamed
        else httpx.Response(400, request=_REQUEST, content=body)
    )
    with pytest.raises(VllmServerRefusalError) as raised:
        raise_for_vllm_status(response)
    assert str(raised.value).startswith(CONTEXT_LENGTH_EXCEEDED_PREFIX)
    assert "maximum context length is 4096" in str(raised.value)
    assert isinstance(raised.value.__cause__, httpx.HTTPStatusError)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(400, request=_REQUEST, json={"error": {"message": "bad tool"}}),
        httpx.Response(503, request=_REQUEST, text="engine unavailable"),
        httpx.Response(500, request=_REQUEST, json={"error": "x" * 1000}),
        httpx.Response(400, request=_REQUEST),
    ],
)
def test_other_refusals_remain_bounded_engine_errors(response: httpx.Response) -> None:
    """Unrelated errors must not be classified as context overflow."""
    with pytest.raises(VllmServerRefusalError) as raised:
        raise_for_vllm_status(response)
    assert str(raised.value).startswith("vLLM answered ")
    assert len(str(raised.value)) < 350


def test_successful_stream_is_not_consumed() -> None:
    """Status checking must not buffer a successful generation stream."""
    response = httpx.Response(200, request=_REQUEST, stream=httpx.ByteStream(b"data"))
    raise_for_vllm_status(response)
    assert not response.is_stream_consumed
