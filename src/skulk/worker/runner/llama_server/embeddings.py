"""llama-server's embeddings mode: launch flags, truncation, and the proxy call.

llama.cpp serves GGUF embedding models through ``--embeddings`` and an
OpenAI-shaped ``/v1/embeddings``. The model's pooling comes from its own GGUF
metadata (``<arch>.pooling_type``) and vectors are L2-normalized by default,
so the server reproduces the artifact's recipe without Skulk passing either.
Two things differ from the chat lane:

* An encoder takes its whole input in one micro-batch, so the batch sizes must
  cover the window (``-b``/``-ub`` = ``n_ctx``; the server forces them equal in
  this mode, and its 512 default rejects longer inputs). One slot keeps
  concurrent inputs from contending for a cache sized for one window.
* llama.cpp never truncates: an input past the window is refused (500 past the
  micro-batch for mean or CLS pooling, 400 past the slot). Skulk's
  ``/v1/embeddings`` contract truncates at the model's length, as the
  transformers runner does, so the runner truncates first through the server's
  own ``/tokenize`` and ``/detokenize``.
"""

from collections.abc import Callable, Sequence
from typing import Final, cast

import httpx

_RETOKENIZE_MARGIN: Final = 2
"""Tokens held back when truncating: decoding a token prefix and re-encoding it
can merge or split differently at the cut, and a server refusal is worse than
two tokens of context."""


def embedding_server_args(n_ctx: int) -> list[str]:
    """The llama-server flags for a dedicated embedding server.

    Args:
        n_ctx: The stamped serving window, the longest input in tokens.

    Returns:
        ``--embeddings``, one slot, and batch sizes covering the window. No chat
        template, reasoning, speculative, or projector flags: they mean nothing
        to an embedding model.
    """
    window = str(n_ctx)
    return ["--embeddings", "--parallel", "1", "-b", window, "-ub", window]


def text_token_budget(n_ctx: int, special_tokens: int) -> int:
    """Tokens of text one input may carry once the model's special tokens are added.

    Args:
        n_ctx: The serving window.
        special_tokens: Tokens the model adds around every input (for example
            ``[CLS]`` and ``[SEP]``), measured by tokenizing an empty string.

    Returns:
        The per-input text budget, at least 1.
    """
    return max(1, n_ctx - special_tokens - _RETOKENIZE_MARGIN)


def truncate_inputs(
    texts: Sequence[str],
    *,
    max_text_tokens: int,
    tokenize: Callable[[str], list[int]],
    detokenize: Callable[[list[int]], str],
) -> list[str]:
    """Cut each text to at most ``max_text_tokens`` tokens of text.

    A text whose UTF-8 length is below the budget cannot exceed it (every token
    covers at least one byte, plus one for a leading word marker), so only
    longer texts are tokenized.

    Args:
        texts: The request's inputs, in order.
        max_text_tokens: The per-input budget from :func:`text_token_budget`.
        tokenize: Text to tokens without special tokens.
        detokenize: Tokens back to text.

    Returns:
        The inputs, truncated where needed, in order.
    """
    truncated: list[str] = []
    for text in texts:
        if len(text.encode("utf-8")) + 1 <= max_text_tokens:
            truncated.append(text)
            continue
        tokens = tokenize(text)
        truncated.append(
            text if len(tokens) <= max_text_tokens else detokenize(tokens[:max_text_tokens])
        )
    return truncated


def parse_embeddings_response(
    payload: object, *, expected: int
) -> tuple[list[list[float]], int]:
    """Read the server's ``/v1/embeddings`` reply into ordered vectors.

    Args:
        payload: The decoded JSON body.
        expected: How many inputs the request carried.

    Returns:
        One vector per input in input order, and the prompt token count.

    Raises:
        ValueError: The reply is not the OpenAI list shape or does not cover
            every input exactly once.
    """
    if not isinstance(payload, dict):
        raise ValueError("embedding reply is not an object")
    body = cast(dict[str, object], payload)
    data = body.get("data")
    if not isinstance(data, list):
        raise ValueError("embedding reply has no data list")
    by_index: dict[int, list[float]] = {}
    for raw_item in cast(list[object], data):
        if not isinstance(raw_item, dict):
            raise ValueError("embedding reply has a malformed entry")
        item = cast(dict[str, object], raw_item)
        index = item.get("index")
        vector = item.get("embedding")
        if not isinstance(index, int) or not isinstance(vector, list):
            raise ValueError("embedding entry lacks an index or a vector")
        by_index[index] = [float(cast(float, value)) for value in cast(list[object], vector)]
    if sorted(by_index) != list(range(expected)):
        raise ValueError(
            f"embedding reply covers inputs {sorted(by_index)}, expected 0..{expected - 1}"
        )
    usage = body.get("usage")
    prompt_tokens = 0
    if isinstance(usage, dict):
        raw_tokens = cast(dict[str, object], usage).get("prompt_tokens")
        if isinstance(raw_tokens, int):
            prompt_tokens = raw_tokens
    return [by_index[index] for index in range(expected)], prompt_tokens


def server_tokenize(
    client: httpx.Client, base_url: str, text: str, *, add_special: bool
) -> list[int]:
    """Tokenize through the server's ``/tokenize``, which uses the model's own vocabulary."""
    response = client.post(
        f"{base_url}/tokenize", json={"content": text, "add_special": add_special}
    )
    response.raise_for_status()
    body = cast(dict[str, object], response.json())
    tokens = body.get("tokens")
    if not isinstance(tokens, list):
        raise ValueError("tokenize reply has no token list")
    return [int(cast(int, token)) for token in cast(list[object], tokens)]


def server_detokenize(client: httpx.Client, base_url: str, tokens: list[int]) -> str:
    """Decode tokens through the server's ``/detokenize``."""
    response = client.post(f"{base_url}/detokenize", json={"tokens": tokens})
    response.raise_for_status()
    content = cast(dict[str, object], response.json()).get("content")
    if not isinstance(content, str):
        raise ValueError("detokenize reply has no content")
    return content
