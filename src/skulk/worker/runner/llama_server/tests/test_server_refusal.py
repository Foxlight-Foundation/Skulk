"""A llama-server refusal reaches the caller in the server's own words.

httpx's status error names the status and the URL only; the reason (a
prompt past the context window, say) sits in the body. The runner reads it
into the error it raises, and maps a context-size refusal onto the API's
context sentinel so the API answers 400 rather than an internal error.
"""

from __future__ import annotations

import httpx
import pytest

from skulk.shared.constants import CONTEXT_LENGTH_EXCEEDED_PREFIX
from skulk.worker.runner.llama_server.runner import (
    ServerRefusalError,
    raise_for_server_status,
    server_refusal_message,
)

_REQUEST = httpx.Request("POST", "http://127.0.0.1:1/v1/chat/completions")


def test_a_context_refusal_carries_the_server_message_and_the_sentinel() -> None:
    response = httpx.Response(
        400,
        request=_REQUEST,
        json={
            "error": {
                "code": 400,
                "message": "the request exceeds the available context size,\n try increasing it",
                "type": "exceed_context_size_error",
            }
        },
    )
    with pytest.raises(ServerRefusalError) as raised:
        raise_for_server_status(response)
    message = str(raised.value)
    assert message.startswith(f"{CONTEXT_LENGTH_EXCEEDED_PREFIX} ")
    assert (
        "llama-server answered 400: the request exceeds the available context size, "
        "try increasing it"
    ) in message
    assert isinstance(raised.value.__cause__, httpx.HTTPStatusError)


def test_other_refusals_carry_the_body_or_the_reason_phrase() -> None:
    assert server_refusal_message(
        httpx.Response(500, request=_REQUEST, text="loading model")
    ) == ("llama-server answered 500: loading model")
    assert server_refusal_message(
        httpx.Response(503, request=_REQUEST, json={"error": "no slot"})
    ) == ("llama-server answered 503: no slot")
    assert (
        server_refusal_message(httpx.Response(400, request=_REQUEST))
        == "llama-server answered 400 Bad Request"
    )
    long = "x" * 1000
    bounded = server_refusal_message(
        httpx.Response(400, request=_REQUEST, json={"error": {"message": long}})
    )
    assert len(bounded) < 400


def test_a_streamed_refusal_is_read_to_its_end_first() -> None:
    response = httpx.Response(
        400,
        request=_REQUEST,
        stream=httpx.ByteStream(
            b'{"error": {"message": "the request exceeds the available context size"}}'
        ),
    )
    assert server_refusal_message(response).startswith(CONTEXT_LENGTH_EXCEEDED_PREFIX)


def test_a_success_raises_nothing() -> None:
    raise_for_server_status(httpx.Response(200, request=_REQUEST, json={"ok": True}))
