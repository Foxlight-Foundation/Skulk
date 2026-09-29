"""Real model cards the tests load by name, now that Skulk ships none.

Copies of registry-seed cards the tests exercise: the MiniMax H3 video
cards, the two audio.cpp music cards, the Gemma 4 family, the steward's
preferred models, the TTS cards with reference voices, and one GLM and one
Kimi card for the tokenizer suite. They are fixtures, not a catalog: nothing
outside the tests reads them.
"""

import tomllib
from pathlib import Path

from anyio import Path as AsyncPath

from skulk.shared.models.model_cards import ModelCard
from skulk.shared.types.common import ModelId

FIXTURE_CARDS_DIR = Path(__file__).with_name("fixtures") / "model_cards"
"""Directory holding the fixture cards, one TOML per model id."""


def fixture_card_path(model_id: str) -> Path:
    """The fixture TOML for ``model_id``."""

    return FIXTURE_CARDS_DIR / (ModelId(model_id).normalize() + ".toml")


def fixture_card_paths(pattern: str = "*.toml") -> list[Path]:
    """Every fixture card matching ``pattern``, sorted by file name."""

    return sorted(FIXTURE_CARDS_DIR.glob(pattern))


async def load_fixture_card(model_id: str) -> ModelCard:
    """Load the fixture card for ``model_id``."""

    return await ModelCard.load_from_path(AsyncPath(fixture_card_path(model_id)))


def fixture_cards() -> list[ModelCard]:
    """Every fixture card, parsed, in file-name order."""

    return [
        ModelCard.model_validate(tomllib.loads(path.read_text()))
        for path in fixture_card_paths()
    ]

