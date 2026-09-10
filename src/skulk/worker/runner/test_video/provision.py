"""Stand-in weights for the test video engine.

The engine needs no weights, but the worker's placement path expects a
complete model directory before it loads a runner. This writes the smallest
directory the download resolver accepts (a safetensors index, an empty
safetensors file, and a config) under the node's models directory and adds
that directory to the model search path, so the bundled card places without
a download.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from skulk.shared.constants import SKULK_MODELS_DIR, add_model_search_path
from skulk.shared.models.model_cards import ModelId

TEST_VIDEO_MODEL_ID = ModelId("foxlight/test-video")
"""The one card the test engine serves; bundled under ``resources/video_model_cards``."""

_WEIGHTS_FILENAME = "model.safetensors"


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
