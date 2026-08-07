# pyright: reportPrivateUsage=false
"""Tests for model-card metadata used by the public API."""

import pathlib

import pytest
from anyio import Path

from skulk.shared.constants import RESOURCES_DIR
from skulk.shared.models import model_cards as model_cards_module
from skulk.shared.models.model_cards import (
    ModelCard,
    PlacementCardConfig,
    RuntimeCapabilityCardConfig,
    get_bundled_card,
    preserve_generated_card_constraints,
)
from skulk.shared.types.common import ModelId

_MINIMAL_CARD = """\
model_id = "testorg/override-model"
n_layers = 4
hidden_size = 64
supports_tensor = false
tasks = ["TextGeneration"]
quantization = "{quantization}"

[storage_size]
in_bytes = 1024
"""


def test_pipeline_split_limit_must_be_inside_model() -> None:
    """A split at or beyond the final layer cannot constrain a pipeline."""
    with pytest.raises(
        ValueError,
        match="max_pipeline_split_layer must be smaller than n_layers",
    ):
        ModelCard.model_validate(
            {
                "model_id": "testorg/invalid-split",
                "storage_size": {"in_bytes": 1024},
                "n_layers": 4,
                "hidden_size": 64,
                "supports_tensor": False,
                "tasks": ["TextGeneration"],
                "placement": {"max_pipeline_split_layer": 4},
            }
        )


@pytest.mark.anyio
async def test_custom_card_overrides_bundled(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A custom card with a bundled card's model id must win (#652).

    The custom directory exists for operator override; before the fix the
    first-wins cache silently kept the bundled card, so the operator's card
    appeared installed while the bundled card actually served.
    """
    builtin_dir = tmp_path / "builtin"
    custom_dir = tmp_path / "custom"
    builtin_dir.mkdir()
    custom_dir.mkdir()
    (builtin_dir / "override-model.toml").write_text(
        _MINIMAL_CARD.format(quantization="fp16")
    )
    # Named with the normalized model id so delete_custom_card resolves it.
    (custom_dir / "testorg--override-model.toml").write_text(
        _MINIMAL_CARD.format(quantization="int4")
    )
    # An invalid custom card must be skipped (with a warning), not abort the
    # load of the valid ones.
    (custom_dir / "broken.toml").write_text("model_id = [not, valid, toml")

    monkeypatch.setattr(model_cards_module, "_card_cache", {})
    await model_cards_module._load_cards_from_dir(
        Path(str(builtin_dir)), is_custom=False
    )
    await model_cards_module._load_cards_from_dir(
        Path(str(custom_dir)), is_custom=True
    )

    card = model_cards_module._card_cache[ModelId("testorg/override-model")]
    assert card.is_custom, "the custom card must replace the bundled card"
    assert card.quantization == "int4"

    # Within the builtin pass, first-wins is preserved: reloading the builtin
    # dir must NOT displace the custom card.
    await model_cards_module._load_cards_from_dir(
        Path(str(builtin_dir)), is_custom=False
    )
    assert model_cards_module._card_cache[
        ModelId("testorg/override-model")
    ].is_custom

    # Deleting the override restores the BUNDLED card immediately (PR #655
    # review): the cache only self-refreshes when empty, so without the
    # delete-path reload the bundled model would vanish from the catalog
    # until process restart.
    monkeypatch.setattr(
        model_cards_module, "_custom_cards_dir", Path(str(custom_dir))
    )
    monkeypatch.setattr(
        model_cards_module, "_BUILTIN_CARD_DIRS", [Path(str(builtin_dir))]
    )
    assert await model_cards_module.delete_custom_card(
        ModelId("testorg/override-model")
    )
    restored = model_cards_module._card_cache[ModelId("testorg/override-model")]
    assert not restored.is_custom, "bundled card must return after delete"
    assert restored.quantization == "fp16"


_STAMPED_CARD = """\
model_id = "testorg/override-model"
n_layers = 4
hidden_size = 64
supports_tensor = false
tasks = ["TextGeneration"]
quantization = "{quantization}"
generator_revision = {generator_revision}

[storage_size]
in_bytes = 1024
"""


async def _load_both(
    builtin_dir: pathlib.Path,
    custom_dir: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> ModelCard:
    """Load a builtin then a custom dir into a fresh cache; return the winner."""
    monkeypatch.setattr(model_cards_module, "_card_cache", {})
    await model_cards_module._load_cards_from_dir(
        Path(str(builtin_dir)), is_custom=False
    )
    await model_cards_module._load_cards_from_dir(Path(str(custom_dir)), is_custom=True)
    return model_cards_module._card_cache[ModelId("testorg/override-model")]


@pytest.mark.anyio
async def test_stale_generated_card_is_superseded_by_bundled(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machine-generated card older than the generator loses its override.

    A generated card is a cache of Hugging Face metadata plus generator
    logic, not operator intent: left in place it silently pins the model to
    whatever the old generator got wrong (the fresh-fleet audit found
    pre-#652-fix cards forcing the in-process engine over the served one).
    """
    builtin_dir = tmp_path / "builtin"
    custom_dir = tmp_path / "custom"
    builtin_dir.mkdir()
    custom_dir.mkdir()
    (builtin_dir / "override-model.toml").write_text(
        _MINIMAL_CARD.format(quantization="fp16")
    )
    stale_revision = model_cards_module.CARD_GENERATOR_REVISION - 1
    (custom_dir / "testorg--override-model.toml").write_text(
        _STAMPED_CARD.format(quantization="int4", generator_revision=stale_revision)
    )

    card = await _load_both(builtin_dir, custom_dir, monkeypatch)

    assert not card.is_custom, "the bundled card must supersede a stale one"
    assert card.quantization == "fp16"


@pytest.mark.anyio
async def test_current_generated_card_keeps_override(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A card stamped with the current generator revision overrides normally."""
    builtin_dir = tmp_path / "builtin"
    custom_dir = tmp_path / "custom"
    builtin_dir.mkdir()
    custom_dir.mkdir()
    (builtin_dir / "override-model.toml").write_text(
        _MINIMAL_CARD.format(quantization="fp16")
    )
    (custom_dir / "testorg--override-model.toml").write_text(
        _STAMPED_CARD.format(
            quantization="int4",
            generator_revision=model_cards_module.CARD_GENERATOR_REVISION,
        )
    )

    card = await _load_both(builtin_dir, custom_dir, monkeypatch)

    assert card.is_custom
    assert card.quantization == "int4"


@pytest.mark.anyio
async def test_current_generated_card_preserves_bundled_split_constraint(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generated overrides must retain curated architecture invariants."""
    builtin_dir = tmp_path / "builtin"
    custom_dir = tmp_path / "custom"
    builtin_dir.mkdir()
    custom_dir.mkdir()
    bundled = ModelCard(
        model_id=ModelId("testorg/override-model"),
        storage_size=model_cards_module.Memory(in_bytes=1024),
        n_layers=4,
        hidden_size=64,
        supports_tensor=False,
        tasks=[model_cards_module.ModelTask.TextGeneration],
        placement=PlacementCardConfig(max_pipeline_split_layer=2),
    )
    await bundled.save(Path(str(builtin_dir / "override-model.toml")))
    (custom_dir / "testorg--override-model.toml").write_text(
        _STAMPED_CARD.format(
            quantization="int4",
            generator_revision=model_cards_module.CARD_GENERATOR_REVISION,
        )
    )

    card = await _load_both(builtin_dir, custom_dir, monkeypatch)

    assert card.is_custom
    assert card.placement.max_pipeline_split_layer == 2


def test_generated_card_preserves_constraint_before_catalog_event() -> None:
    """The API event payload must be safe before workers persist the card."""
    bundled = ModelCard(
        model_id=ModelId("testorg/override-model"),
        storage_size=model_cards_module.Memory(in_bytes=1024),
        n_layers=4,
        hidden_size=64,
        supports_tensor=False,
        tasks=[model_cards_module.ModelTask.TextGeneration],
        placement=PlacementCardConfig(max_pipeline_split_layer=2),
    )
    generated = bundled.model_copy(
        update={
            "placement": PlacementCardConfig(),
            "is_custom": True,
            "generator_revision": model_cards_module.CARD_GENERATOR_REVISION,
        }
    )

    preserved = preserve_generated_card_constraints(generated, bundled)

    assert preserved.placement.max_pipeline_split_layer == 2


def test_hand_authored_card_keeps_explicit_placement_override() -> None:
    """Unstamped operator policy must continue to override bundled truth."""
    bundled = ModelCard(
        model_id=ModelId("testorg/override-model"),
        storage_size=model_cards_module.Memory(in_bytes=1024),
        n_layers=4,
        hidden_size=64,
        supports_tensor=False,
        tasks=[model_cards_module.ModelTask.TextGeneration],
        placement=PlacementCardConfig(max_pipeline_split_layer=2),
    )
    authored = bundled.model_copy(
        update={"placement": PlacementCardConfig(), "is_custom": True}
    )

    preserved = preserve_generated_card_constraints(authored, bundled)

    assert preserved.placement.max_pipeline_split_layer is None


@pytest.mark.anyio
async def test_bundled_lookup_ignores_cached_custom_winner(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repeated add must still find curated constraints beneath the cache."""
    builtin_dir = tmp_path / "builtin"
    builtin_dir.mkdir()
    bundled = ModelCard(
        model_id=ModelId("testorg/override-model"),
        storage_size=model_cards_module.Memory(in_bytes=1024),
        n_layers=4,
        hidden_size=64,
        supports_tensor=False,
        tasks=[model_cards_module.ModelTask.TextGeneration],
        placement=PlacementCardConfig(max_pipeline_split_layer=2),
    )
    await bundled.save(Path(str(builtin_dir / "testorg--override-model.toml")))
    cached_custom = bundled.model_copy(
        update={
            "placement": PlacementCardConfig(),
            "is_custom": True,
            "generator_revision": model_cards_module.CARD_GENERATOR_REVISION,
        }
    )
    monkeypatch.setattr(model_cards_module, "_BUILTIN_CARD_DIRS", [Path(str(builtin_dir))])
    monkeypatch.setattr(
        model_cards_module,
        "_card_cache",
        {ModelId("testorg/override-model"): cached_custom},
    )

    resolved = await get_bundled_card(ModelId("testorg/override-model"))

    assert resolved is not None
    assert not resolved.is_custom
    assert resolved.placement.max_pipeline_split_layer == 2


@pytest.mark.anyio
async def test_stale_generated_card_without_bundled_still_serves(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no bundled counterpart, a stale generated card still serves."""
    builtin_dir = tmp_path / "builtin"
    custom_dir = tmp_path / "custom"
    builtin_dir.mkdir()
    custom_dir.mkdir()
    stale_revision = model_cards_module.CARD_GENERATOR_REVISION - 1
    (custom_dir / "testorg--override-model.toml").write_text(
        _STAMPED_CARD.format(quantization="int4", generator_revision=stale_revision)
    )

    card = await _load_both(builtin_dir, custom_dir, monkeypatch)

    assert card.is_custom
    assert card.quantization == "int4"


@pytest.mark.anyio
async def test_generated_cards_are_stamped_and_stamp_round_trips(
    tmp_path: pathlib.Path,
) -> None:
    """The stamp persists through save/load so staleness survives restarts."""
    card = ModelCard(
        model_id=ModelId("testorg/stamped-model"),
        storage_size=model_cards_module.Memory(in_bytes=1024),
        n_layers=4,
        hidden_size=64,
        supports_tensor=False,
        tasks=[model_cards_module.ModelTask.TextGeneration],
        generator_revision=model_cards_module.CARD_GENERATOR_REVISION,
    )
    path = tmp_path / "stamped.toml"
    await card.save(Path(str(path)))

    loaded = await ModelCard.load_from_path(Path(str(path)))

    assert loaded.generator_revision == model_cards_module.CARD_GENERATOR_REVISION


def test_vllm_spec_pairing_validator() -> None:
    """The vllm speculative fields must be internally consistent at card load.

    Operator-authored custom cards get the same loud failure the bundled
    invariant suite gives shipped cards: depth without a method, dflash
    without its drafter repo, and a drafter repo under mtp (whose drafter
    lives inside the target checkpoint) are all card-authoring errors that
    would otherwise surface as opaque serve-time failures.
    """
    with pytest.raises(ValueError, match="requires vllm_spec_method"):
        RuntimeCapabilityCardConfig(vllm_spec_num_tokens=2)
    with pytest.raises(ValueError, match="requires vllm_spec_draft_repo"):
        RuntimeCapabilityCardConfig(vllm_spec_method="dflash")
    with pytest.raises(ValueError, match="requires vllm_spec_method 'dflash'"):
        RuntimeCapabilityCardConfig(
            vllm_spec_method="mtp",
            vllm_spec_draft_repo="poolside/Laguna-XS-2.1-DFlash-FP8",
        )
    valid = RuntimeCapabilityCardConfig(
        vllm_spec_method="dflash",
        vllm_spec_num_tokens=15,
        vllm_spec_draft_repo="poolside/Laguna-XS-2.1-DFlash-FP8",
    )
    assert valid.vllm_spec_draft_repo == "poolside/Laguna-XS-2.1-DFlash-FP8"


@pytest.mark.anyio
async def test_gemma4_builtin_card_declares_context_length() -> None:
    """Built-in Gemma 4 cards should publish their context length to the UI."""
    card_path = (
        Path(RESOURCES_DIR)
        / "inference_model_cards"
        / "mlx-community--gemma-4-26b-a4b-it-4bit.toml"
    )

    card = await ModelCard.load_from_path(card_path)

    assert card.context_length == 262144
