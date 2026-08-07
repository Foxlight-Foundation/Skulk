"""Tests for precomputed vision embedding injection."""

from typing import cast

import mlx.core as mx
import mlx.nn as nn

from skulk.shared.types.mlx import Model
from skulk.worker.engines.mlx.vision import create_vision_embeddings


class _TextTrunk(nn.Module):
    """Minimal text trunk exposing the embedding contract under test."""

    def __init__(self, embed_scale: float | None) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 2)
        if embed_scale is not None:
            self.embed_scale = embed_scale


class _Model(nn.Module):
    """Minimal outer model accepted by ``get_inner_model``."""

    def __init__(self, embed_scale: float | None) -> None:
        super().__init__()
        self.model = _TextTrunk(embed_scale)


def test_image_features_compensate_for_text_only_embedding_scale() -> None:
    """Gemma's later text scale must restore, not amplify, image features."""
    model = _Model(embed_scale=2.0)
    image_features = mx.array([[10.0, 20.0], [30.0, 40.0]])

    mixed = create_vision_embeddings(
        cast(Model, cast(object, model)),
        mx.array([1, 7, 7, 2]),
        image_features,
        image_token_id=7,
    )

    assert mx.allclose(mixed[0, 1:3] * 2.0, image_features).item()


def test_image_features_remain_unchanged_without_embedding_scale() -> None:
    """Families whose text trunk does not scale embeddings keep prior behavior."""
    model = _Model(embed_scale=None)
    image_features = mx.array([[10.0, 20.0], [30.0, 40.0]])

    mixed = create_vision_embeddings(
        cast(Model, cast(object, model)),
        mx.array([1, 7, 7, 2]),
        image_features,
        image_token_id=7,
    )

    assert mx.allclose(mixed[0, 1:3], image_features).item()
