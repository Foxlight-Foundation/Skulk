# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
"""Pooling and the chunked forward pass reproduce the model's own recipe.

The fake encoder writes each token's position (1-based) into its hidden state,
and padding positions carry a large sentinel, so a pooled value shows exactly
which tokens it read and that padding never leaks in.
"""

import math
from types import SimpleNamespace
from typing import cast

import torch

from skulk.worker.runner.embeddings import runner as embeddings_runner
from skulk.worker.runner.embeddings.recipe import EmbeddingRecipe

_PAD_SENTINEL = 1000.0


def _states(lengths: list[int], width: int, *, left_padded: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Hidden states ``[batch, width, 1]`` and the matching attention mask."""
    hidden = torch.full((len(lengths), width, 1), _PAD_SENTINEL)
    mask = torch.zeros((len(lengths), width), dtype=torch.long)
    for row, length in enumerate(lengths):
        offset = width - length if left_padded else 0
        for position in range(length):
            hidden[row, offset + position, 0] = float(position + 1)
            mask[row, offset + position] = 1
    return hidden, mask


def _pooled(pooling: str, lengths: list[int], *, left_padded: bool = False) -> list[float]:
    hidden, mask = _states(lengths, 4, left_padded=left_padded)
    pooled = embeddings_runner._pool(hidden, mask, pooling, left_padded=left_padded)  # pyright: ignore[reportArgumentType]
    return cast(list[float], pooled[:, 0].tolist())


def test_cls_reads_the_first_token() -> None:
    assert _pooled("cls", [2, 4]) == [1.0, 1.0]


def test_mean_ignores_padding() -> None:
    assert _pooled("mean", [2, 4]) == [1.5, 2.5]


def test_max_ignores_padding() -> None:
    assert _pooled("max", [2, 4]) == [2.0, 4.0]


def test_last_reads_the_last_real_token_on_either_padding_side() -> None:
    assert _pooled("last", [2, 4]) == [2.0, 4.0]
    assert _pooled("last", [2, 4], left_padded=True) == [2.0, 4.0]


class _Tokenizer:
    """Tokenizes ``"n"`` into n tokens, recording each call's max_length."""

    padding_side = "right"

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def __call__(self, texts: list[str], *, max_length: int, **_: object) -> dict[str, torch.Tensor]:
        lengths = [min(int(text), max_length) for text in texts]
        width = max(lengths)
        self.calls.append((len(texts), max_length))
        mask = torch.zeros((len(texts), width), dtype=torch.long)
        for row, length in enumerate(lengths):
            mask[row, :length] = 1
        return {"input_ids": mask.clone(), "attention_mask": mask}


def _encoder(**inputs: torch.Tensor) -> SimpleNamespace:
    """Hidden state per token: [position + 1, 1.0], padding carries the sentinel."""
    mask = inputs["attention_mask"]
    positions = torch.arange(1, mask.shape[1] + 1, dtype=torch.float32).expand(mask.shape)
    first = torch.where(mask == 1, positions, torch.tensor(_PAD_SENTINEL))
    second = torch.ones_like(first)
    return SimpleNamespace(last_hidden_state=torch.stack([first, second], dim=-1))


def test_forward_chunks_preserve_order_counts_and_truncation() -> None:
    tokenizer = _Tokenizer()
    texts = [str(1 + index % 5) for index in range(70)]
    recipe = EmbeddingRecipe(pooling="last", normalize=False, max_tokens=3)

    vectors, token_count = embeddings_runner._forward(
        _encoder, tokenizer, texts, torch.device("cpu"), recipe, rank=0
    )

    # Three bounded chunks, each tokenized at the recipe's truncation length.
    assert tokenizer.calls == [(32, 3), (32, 3), (6, 3)]
    expected_lengths = [min(int(text), 3) for text in texts]
    assert [vector[0] for vector in vectors] == [float(length) for length in expected_lengths]
    assert token_count == sum(expected_lengths)


def test_forward_normalizes_only_when_the_recipe_says_so() -> None:
    texts = ["3"]
    raw, _ = embeddings_runner._forward(
        _encoder, _Tokenizer(), texts, torch.device("cpu"),
        EmbeddingRecipe(pooling="last", normalize=False, max_tokens=8), rank=0,
    )
    normalized, _ = embeddings_runner._forward(
        _encoder, _Tokenizer(), texts, torch.device("cpu"),
        EmbeddingRecipe(pooling="last", normalize=True, max_tokens=8), rank=0,
    )
    assert raw == [[3.0, 1.0]]
    norm = math.sqrt(3.0 * 3.0 + 1.0)
    assert [round(value, 6) for value in normalized[0]] == [round(3.0 / norm, 6), round(1.0 / norm, 6)]


def test_pass_size_shrinks_with_the_truncation_length() -> None:
    """A pass never pads past the token budget, however long the recipe's inputs."""
    assert embeddings_runner._texts_per_pass(256) == 32
    assert embeddings_runner._texts_per_pass(512) == 32
    assert embeddings_runner._texts_per_pass(8192) == 2
    assert embeddings_runner._texts_per_pass(65536) == 1

    tokenizer = _Tokenizer()
    texts = ["5"] * 5
    embeddings_runner._forward(
        _encoder, tokenizer, texts, torch.device("cpu"),
        EmbeddingRecipe(pooling="mean", normalize=False, max_tokens=8192), rank=0,
    )
    assert [count for count, _ in tokenizer.calls] == [2, 2, 1]
