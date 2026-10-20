"""Actual timeline coordinates survive filtering and appended image guides."""

import pytest
import torch
from skulk_comfy_ltx.relay_encoding import RelayLayout
from skulk_comfy_ltx.relay_geometry import prepare_relay_geometry


def test_filtered_causal_queries_keep_their_original_scene_positions() -> None:
    """Removing one real frame must not renumber later queries or gate guides."""
    video = torch.zeros(2, 128, 5, 1, 2)
    coordinates = torch.zeros(1, 3, 8, 2)
    coordinates[:, 0, :, 0] = torch.tensor([0, 0, 9, 9, 17, 17, 12, 12])
    guides = torch.zeros(1, 3, 2, 2)
    geometry = prepare_relay_geometry(video, coordinates, guides, 2, scale_factors=(8, 32, 32), patch_size=(1, 1, 1), causal=True, layout=RelayLayout(((2, 4), (4, 7)), (1, 1)))
    torch.testing.assert_close(geometry.query_frames, torch.tensor([[0, 0, 2, 2, 3, 3, 2, 2]], dtype=torch.float32).expand(2, -1))
    assert bool(geometry.real_queries[:, :6].all())
    assert not bool(geometry.real_queries[:, 6:].any())
    assert [segment.midpoint for segment in geometry.segments] == [1, 3]


def test_noncausal_grid_uses_actual_scale_without_a_shift() -> None:
    """Preserve the pinned noncausal coordinate convention when explicitly used."""
    video = torch.zeros(1, 128, 4, 1, 1)
    coordinates = torch.zeros(1, 3, 4, 2)
    coordinates[:, 0, :, 0] = torch.tensor([0, 8, 16, 24])
    geometry = prepare_relay_geometry(video, coordinates, None, 0, scale_factors=(8, 32, 32), patch_size=(1, 1, 1), causal=False, layout=RelayLayout(((1, 3), (3, 5)), (1, 1)))
    torch.testing.assert_close(geometry.query_frames, torch.arange(4, dtype=torch.float32).reshape(1, -1))


@pytest.mark.parametrize("bad_position", [float("nan"), float("inf"), -1.0, 40.0])
def test_invalid_real_coordinates_fail_before_attention(bad_position: float) -> None:
    """An invalid grid cannot silently attach a scene to guessed query indices."""
    video = torch.zeros(1, 128, 4, 1, 1)
    coordinates = torch.zeros(1, 3, 4, 2)
    coordinates[:, 0, :, 0] = torch.tensor([0, 1, 9, bad_position])
    with pytest.raises(ValueError):
        prepare_relay_geometry(video, coordinates, None, 0, scale_factors=(8, 32, 32), patch_size=(1, 1, 1), causal=True, layout=RelayLayout(((1, 3), (3, 5)), (1, 1)))


def test_unknown_guide_rows_are_refused() -> None:
    """Do not infer guide counts from token order without the actual metadata."""
    video = torch.zeros(1, 128, 4, 1, 1)
    with pytest.raises(ValueError, match="surviving guide metadata"):
        prepare_relay_geometry(video, torch.zeros(1, 3, 4, 2), None, 1, scale_factors=(8, 32, 32), patch_size=(1, 1, 1), causal=True, layout=RelayLayout(((1, 3), (3, 5)), (1, 1)))
