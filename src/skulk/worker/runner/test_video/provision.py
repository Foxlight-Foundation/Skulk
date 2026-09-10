"""Stand-in weights and card registration for the test video engine.

The engine needs no weights, but the worker's placement path expects a
complete model directory before it loads a runner. This writes the smallest
directory the download resolver accepts (a safetensors index, an empty
safetensors file, and a config) under the node's models directory and adds
that directory to the model search path, so the card places without a
download. It also registers the bundled card as a custom card so a node
whose catalog comes from the signed registry still lists it.
"""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

from anyio import Path as AsyncPath

from skulk.shared.constants import (
    RESOURCES_DIR,
    SKULK_CUSTOM_MODEL_CARDS_DIR,
    SKULK_MODELS_DIR,
    add_model_search_path,
)
from skulk.shared.models.model_cards import ModelCard, ModelId, add_to_card_cache

TEST_VIDEO_MODEL_ID = ModelId("foxlight/test-video")
"""The one card the test engine serves; bundled under ``resources/video_model_cards``."""

_WEIGHTS_FILENAME = "model.safetensors"
_CARD_FILENAME = TEST_VIDEO_MODEL_ID.normalize() + ".toml"


def bundled_card_path() -> Path:
    """The bundled TOML for the test engine's card."""

    return Path(RESOURCES_DIR) / "video_model_cards" / _CARD_FILENAME


def register_test_video_card(custom_cards_dir: Path = SKULK_CUSTOM_MODEL_CARDS_DIR) -> Path:
    """Make the card visible on a node whose catalog comes from the registry.

    Bundled cards load only when the signed registry is unavailable, so on a
    connected node the test card would never enter the catalog. The custom
    card directory is the operator's override path and always loads; placing
    the bundled TOML there is exactly what an operator adding the card by
    hand would do. Idempotent; an existing file is left alone.
    """

    custom_cards_dir.mkdir(parents=True, exist_ok=True)
    target = custom_cards_dir / _CARD_FILENAME
    if not target.is_file():
        shutil.copyfile(bundled_card_path(), target)
    return target


async def install_test_video_card(
    custom_cards_dir: Path = SKULK_CUSTOM_MODEL_CARDS_DIR,
) -> ModelCard:
    """Register the card and make it visible in this process's catalog now.

    The catalog may already have been preloaded from the registry before the
    worker provisions the engine, and the custom directory is only re-read
    on the next refresh. Adding the loaded card to the cache directly means
    the documented start-and-place flow works on the first start.
    """

    path = register_test_video_card(custom_cards_dir)
    card = await ModelCard.load_from_path(AsyncPath(path))
    card = card.model_copy(update={"is_custom": True})
    add_to_card_cache(card)
    return card


def provision_test_video_model(models_dir: Path = SKULK_MODELS_DIR) -> Path:
    """Ensure the stand-in model directory exists and is searchable.

    Idempotent: existing files are left alone. Returns the directory.
    """

    directory = models_dir / TEST_VIDEO_MODEL_ID.normalize()
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / _WEIGHTS_FILENAME
    if not weights.is_file():
        header = json.dumps({"__metadata__": {"engine": "test_video"}}).encode()
        weights.write_bytes(struct.pack("<Q", len(header)) + header)
    index = directory / "model.safetensors.index.json"
    if not index.is_file():
        index.write_text(
            json.dumps(
                {
                    "metadata": {"total_size": weights.stat().st_size},
                    "weight_map": {"test_video.marker": _WEIGHTS_FILENAME},
                }
            )
        )
    config = directory / "config.json"
    if not config.is_file():
        config.write_text(json.dumps({"model_type": "test_video"}))
    add_model_search_path(models_dir)
    return directory
