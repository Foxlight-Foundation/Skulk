"""Soft-routing invariants and bounded attention against independent dense math."""

import math

import pytest
import torch
from skulk_comfy_ltx.relay import (
    relay_attention,
    relay_bias,
    relay_lengths,
    relay_segments,
)
from torch import Tensor


def test_lengths_preserve_relative_grid_and_earlier_remainder_ties() -> None:
    """Keep every scene visible without silently losing a final timeline frame."""
    assert relay_lengths((1, 1), 7) == (4, 3)
    assert relay_lengths((2, 1), 7) == (4, 3)
    assert relay_lengths((1, 1, 1), 5) == (2, 2, 1)
    assert relay_lengths((120, 1, 2), 3) == (1, 1, 1)


@pytest.mark.parametrize("weights,frames", [((1,), 4), ((1, 0), 4), ((1, 121), 4), ((1, 1, 1), 2), ((1, 1), 15002)])
def test_invalid_scene_lengths_fail_before_attention(weights: tuple[int, ...], frames: int) -> None:
    """Reject impossible timelines instead of globally ungating zero-length scenes."""
    with pytest.raises(ValueError):
        relay_lengths(weights, frames)


def test_gaussian_plateau_keeps_global_padding_and_surviving_guides_free() -> None:
    """Compare explicit penalties, including fractional positions and guide tails."""
    segments = relay_segments(((2, 4), (5, 7)), (2, 1), 7)
    frames = torch.tensor([[0.0, 2.0, 3.5, 5.0, 99.0]])
    actual = relay_bias(frames, segments, 9, torch.tensor([[True, True, True, True, False]]))[0, 0]
    scale = math.log(1000.0) ** 2 / 2
    first = torch.tensor([-4.0, 0.0, -2.25, -9.0, 0.0]) * scale
    second = torch.tensor([-25.0, -9.0, -2.25, 0.0, 0.0]) * scale
    torch.testing.assert_close(actual[:, 2], first)
    torch.testing.assert_close(actual[:, 3], first)
    torch.testing.assert_close(actual[:, 5], second)
    torch.testing.assert_close(actual[:, 6], second)
    assert not bool(actual[:, [0, 1, 4, 7, 8]].any())
    assert not bool(actual[-1].any())


@pytest.mark.parametrize("ranges", [((2, 5), (4, 7)), ((2, 2), (3, 5)), ((-1, 2), (3, 5))])
def test_invalid_text_ranges_fail_closed(ranges: tuple[tuple[int, int], ...]) -> None:
    """Never let an invalid slice silently ungate or overwrite another scene."""
    with pytest.raises(ValueError):
        relay_segments(ranges, (1, 1), 7)


@pytest.mark.parametrize("mask_heads,mask_queries", [(1, 1), (3, 53)])
def test_chunked_attention_matches_independent_dense_mask(mask_heads: int, mask_queries: int) -> None:
    """Preserve existing masks and irregular query ordering while bounding bias."""
    generator = torch.Generator().manual_seed(141)
    query = torch.randn(2, 53, 12, generator=generator)
    key = torch.randn(2, 24, 12, generator=generator)
    value = torch.randn(2, 24, 12, generator=generator)
    positions = torch.arange(53, dtype=torch.float32).remainder(7).repeat(2, 1)
    real = torch.ones_like(positions, dtype=torch.bool)
    real[:, -3:] = False
    segments = relay_segments(((3, 8), (8, 13)), (2, 1), 7)
    mask = torch.randn(2, mask_heads, mask_queries, 24, generator=generator) * 0.01
    budget = 32 * 2 * 3 * 24 * 7
    observed_chunks: list[int] = []

    def attention(queries: Tensor, keys: Tensor, values: Tensor, bias: Tensor) -> Tensor:
        observed_chunks.append(queries.shape[1])
        assert bias.numel() * bias.element_size() <= budget
        return torch.nn.functional.scaled_dot_product_attention(queries.reshape(2, -1, 3, 4).transpose(1, 2), keys.reshape(2, -1, 3, 4).transpose(1, 2), values.reshape(2, -1, 3, 4).transpose(1, 2), attn_mask=bias).transpose(1, 2).reshape(2, -1, 12)

    actual = relay_attention(query, key, value, query_frames=positions, real_queries=real, segments=segments, attention=attention, mask=mask, workspace_bytes=budget)
    expected_mask = torch.zeros(2, 1, 53, 24)
    for row in range(50):
        position = row % 7
        for start, end, midpoint in ((3, 8, 2), (8, 13, 5)):
            expected_mask[:, 0, row, start:end] = -(position - midpoint) ** 2 * math.log(1000.0) ** 2 / 2
    expected = attention(query, key, value, expected_mask + mask)
    torch.testing.assert_close(actual, expected, atol=5e-7, rtol=1e-5)
    assert len(observed_chunks[:-1]) > 1
    assert max(observed_chunks[:-1]) <= budget // (32 * 2 * mask_heads * 24)


def test_too_small_workspace_refuses_without_calling_attention() -> None:
    """Keep one large text row from bypassing the declared bias workspace."""
    query = torch.zeros(1, 3, 4)
    key = torch.zeros(1, 8, 4)
    positions = torch.zeros(1, 3)

    def forbidden_attention(_query: Tensor, _key: Tensor, _value: Tensor, _mask: Tensor) -> Tensor:
        raise AssertionError("invalid workspace must fail before attention")

    with pytest.raises(ValueError, match="bounded attention workspace"):
        relay_attention(query, key, key, query_frames=positions, real_queries=torch.ones_like(positions, dtype=torch.bool), segments=relay_segments(((1, 3), (4, 6)), (1, 1), 3), attention=forbidden_attention, workspace_bytes=1)
