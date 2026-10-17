# pyright: reportPrivateUsage=false
"""A card's ``min_skulk_version`` floor: syntax, comparison, and custom files.

Invariants:

1. The floor is a plain PEP 440 release, stored normalized; anything else
   (a pre-release, a local version, free text) is refused as malformed.
2. A build at or above the floor serves the card, and a pre-release or
   development build counts as its release; a build below the floor, or one
   that cannot read its own version, does not.
3. A custom card file this release is too old for is ignored with a warning,
   and its neighbours still load.

Registry and installed-record skips are covered in ``test_registry.py``.
"""

import pathlib
import tomllib

import pytest
from anyio import Path
from pydantic import ValidationError

from skulk.shared.models import model_cards as model_cards_module
from skulk.shared.models.model_cards import ModelCard, card_requires_newer_skulk
from skulk.shared.types.common import ModelId

_CARD = """\
model_id = "{model_id}"
n_layers = 4
hidden_size = 64
supports_tensor = false
tasks = ["TextEmbedding"]
quantization = "Q8_0"
{floor}
[storage_size]
in_bytes = 1024
"""


def _card(floor: str | None) -> ModelCard:
    line = f'min_skulk_version = "{floor}"\n' if floor is not None else ""
    return ModelCard.model_validate(
        tomllib.loads(_CARD.format(model_id="testorg/embed", floor=line))
    )


@pytest.mark.parametrize(
    ("raw", "stored"), [("2.0.3", "2.0.3"), ("2.1", "2.1"), ("v2.0.3", "2.0.3")]
)
def test_the_floor_is_a_normalized_release(raw: str, stored: str) -> None:
    assert _card(raw).min_skulk_version == stored


@pytest.mark.parametrize(
    "raw", ["latest", "", "2.0.3rc1", "2.0.3.dev1", "2.0.3.post1", "2.0.3+local"]
)
def test_the_floor_must_be_a_plain_release(raw: str) -> None:
    with pytest.raises(ValidationError, match="min_skulk_version"):
        _card(raw)


def test_a_card_without_a_floor_serves_on_any_build() -> None:
    assert card_requires_newer_skulk(_card(None), "0.1.0") is None


@pytest.mark.parametrize("running", ["2.0.3", "2.0.4", "2.1.0", "2.0.3rc1", "2.0.3.dev4"])
def test_a_build_at_or_above_the_floor_serves_the_card(running: str) -> None:
    assert card_requires_newer_skulk(_card("2.0.3"), running) is None


@pytest.mark.parametrize("running", ["2.0.2", "1.9.9", "2.0.2.post3"])
def test_a_build_below_the_floor_does_not(running: str) -> None:
    reason = card_requires_newer_skulk(_card("2.0.3"), running)
    assert reason is not None
    assert "requires Skulk 2.0.3 or newer" in reason
    assert running in reason


def test_a_build_that_cannot_read_its_version_does_not_serve_a_floored_card() -> None:
    reason = card_requires_newer_skulk(_card("2.0.3"), "unknown")
    assert reason is not None
    assert "cannot be read" in reason


async def test_a_custom_card_above_this_release_is_ignored(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from loguru import logger

    custom = tmp_path / "custom"
    custom.mkdir()
    (custom / "newer.toml").write_text(
        _CARD.format(model_id="testorg/needs-newer", floor='min_skulk_version = "9.0"\n')
    )
    (custom / "fits.toml").write_text(
        _CARD.format(model_id="testorg/fits", floor='min_skulk_version = "2.0.3"\n')
    )
    monkeypatch.setattr(model_cards_module, "_card_cache", {})
    monkeypatch.setattr(model_cards_module, "_card_floor_warnings", set[tuple[str, str]]())

    def this_release() -> str:
        return "2.0.3"

    monkeypatch.setattr(model_cards_module, "get_skulk_version", this_release)
    warnings: list[str] = []
    handler = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    try:
        await model_cards_module._load_cards_from_dir(Path(custom), is_custom=True)
        # A second pass (the periodic refresh) does not repeat the warning.
        await model_cards_module._load_cards_from_dir(Path(custom), is_custom=True)
    finally:
        logger.remove(handler)

    assert ModelId("testorg/fits") in model_cards_module._card_cache
    assert ModelId("testorg/needs-newer") not in model_cards_module._card_cache
    floor_warnings = [line for line in warnings if "requires Skulk 9.0" in line]
    assert len(floor_warnings) == 1
    assert "newer.toml" in floor_warnings[0]
