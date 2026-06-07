"""Tests for the multi-node speculation card gate.

Cards can forbid speculation on multi-node placements where it is
measured slower than plain distributed decode (2026-06-06 matrix:
gemma-4-26B-A4B 2-node pipeline, 30.2 plain vs 28.2 with MTP). The
predicate is shared by the drafter-load gate and the distributed
agreement gate so every rank decides identically from the card.
"""

from exo.shared.models.model_cards import (
    RuntimeCapabilityCardConfig,
    multi_node_speculation_disabled,
)


def test_unset_knob_places_no_restriction() -> None:
    runtime = RuntimeCapabilityCardConfig(
        assistant_model_repo="mlx-community/gemma-4-26B-A4B-it-assistant-bf16"
    )
    assert not multi_node_speculation_disabled(runtime, 1)
    assert not multi_node_speculation_disabled(runtime, 2)


def test_false_knob_blocks_multi_node_only() -> None:
    runtime = RuntimeCapabilityCardConfig(
        speculative_multi_node=False,
        assistant_model_repo="mlx-community/gemma-4-26B-A4B-it-assistant-bf16",
    )
    assert not multi_node_speculation_disabled(runtime, 1)
    assert multi_node_speculation_disabled(runtime, 2)
    assert multi_node_speculation_disabled(runtime, 3)


def test_true_knob_and_missing_runtime_allow_everything() -> None:
    assert not multi_node_speculation_disabled(None, 2)
    runtime = RuntimeCapabilityCardConfig(speculative_multi_node=True)
    assert not multi_node_speculation_disabled(runtime, 2)


def test_26b_card_pins_the_measured_negative() -> None:
    import asyncio

    from exo.shared.models.model_cards import ModelCard, ModelId

    card = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        ModelCard.load(ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"))
    )
    assert card.runtime is not None
    assert card.runtime.speculative_multi_node is False
    assert multi_node_speculation_disabled(card.runtime, 2)
    assert not multi_node_speculation_disabled(card.runtime, 1)
