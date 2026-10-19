"""Staged native weights remain conservative without hardware qualification gates."""

import pytest

from skulk.shared.models.memory_estimate import (
    MEMORY_OVERHEAD_FACTOR,
    MEMORY_OVERHEAD_FLOOR,
    estimate_shard_footprint,
)
from skulk.shared.models.model_cards import RuntimeCapabilityCardConfig
from skulk.shared.models.video_memory import staged_video_weights
from skulk.shared.tests.model_card_fixtures import load_fixture_card
from skulk.shared.types.common import ModelId


async def test_native_stage_reservation_fits_24_gib_without_removing_headroom() -> None:
    """The complete pack fits by staging while retaining overhead above measured use."""
    card = await load_fixture_card(ModelId("dgrauet/LTX-2.5-distilled-mlx-q4"))
    assert card is not None
    staged = staged_video_weights(card, resolved_backend="mlx_video-metal")
    assert staged is not None and staged.in_bytes == 14_595_453_751
    native = estimate_shard_footprint(card, 1.0, resolved_backend="mlx_video-metal")
    whole = estimate_shard_footprint(card, 1.0)
    assert native == staged * MEMORY_OVERHEAD_FACTOR + MEMORY_OVERHEAD_FLOOR
    assert 16_223_772_796 < native.in_bytes < 18 * 1024**3 < whole.in_bytes


@pytest.mark.parametrize("backend", [None, "comfy-rocm", "mlx-metal", "mlx_video-cpu"])
async def test_other_engines_keep_their_full_weight_reservation(backend: str | None) -> None:
    """A shared card never transfers a native engine's staging assumption to another engine."""
    card = await load_fixture_card(ModelId("dgrauet/LTX-2.5-distilled-mlx-q4"))
    assert card is not None
    assert staged_video_weights(card, resolved_backend=backend) is None
    assert estimate_shard_footprint(card, 1.0, resolved_backend=backend) == (
        card.storage_size * MEMORY_OVERHEAD_FACTOR + MEMORY_OVERHEAD_FLOOR
    )


async def test_missing_pack_layout_keeps_conservative_accounting() -> None:
    """A partial or legacy card cannot opt into an unsupported residency assumption."""
    card = await load_fixture_card(ModelId("dgrauet/LTX-2.5-distilled-mlx-q4"))
    assert card is not None and card.artifact_bundle is not None
    bundle = card.artifact_bundle.model_copy(
        update={
            "files": tuple(
                entry for entry in card.artifact_bundle.files
                if entry.path != "connector.safetensors"
            )
        }
    )
    partial = card.model_copy(update={"artifact_bundle": bundle})
    assert staged_video_weights(partial, resolved_backend="mlx_video-metal") is None
    assert staged_video_weights(card.model_copy(update={"artifact_bundle": None}), resolved_backend="mlx_video-metal") is None
    assert estimate_shard_footprint(card, 0.5, resolved_backend="mlx_video-metal") == (
        card.storage_size * 0.5 * MEMORY_OVERHEAD_FACTOR + MEMORY_OVERHEAD_FLOOR
    )


async def test_explicit_budget_can_raise_but_never_reduce_memory_admission() -> None:
    """A larger workload can reserve more memory without hardware qualification rows."""
    card = await load_fixture_card(ModelId("dgrauet/LTX-2.5-distilled-mlx-q4"))
    assert card is not None
    original = estimate_shard_footprint(card, 1.0, resolved_backend="mlx_video-metal")
    for budget in (1, 64 * 1024**3):
        configured = card.model_copy(update={"runtime": RuntimeCapabilityCardConfig(video_memory_budget_bytes=budget)})
        actual = estimate_shard_footprint(configured, 1.0, resolved_backend="mlx_video-metal")
        assert actual.in_bytes == max(original.in_bytes, budget)


def test_absent_budget_preserves_existing_runtime_authorization_projection() -> None:
    """An unused additive setting must not change already-authorized card bytes."""
    runtime = RuntimeCapabilityCardConfig(metal_fast_synch=False)
    assert "video_memory_budget_bytes" not in runtime.model_dump()
    assert "videoMemoryBudgetBytes" not in runtime.model_dump(by_alias=True)
    configured = runtime.model_copy(update={"video_memory_budget_bytes": 64 * 1024**3})
    assert configured.model_dump()["video_memory_budget_bytes"] == 64 * 1024**3
