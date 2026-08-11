"""Security boundaries for paths loaded from the model-store registry."""

import json
from pathlib import Path
from typing import cast

import pytest

from skulk.store.model_store import ModelStore, StoreRegistryIndex


def test_corrupted_registry_path_cannot_escape_store(tmp_path: Path) -> None:
    """Registry traversal must not read or delete a directory outside the store."""

    store_root = tmp_path / "store"
    store_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep")
    (store_root / "registry.json").write_text(
        json.dumps(
            {
                "org/model": {
                    "model_id": "org/model",
                    "store_path": "../outside",
                    "files": ["keep.txt"],
                    "downloaded_at": "2026-07-16T00:00:00+00:00",
                    "total_bytes": 4,
                    "source_revision": None,
                }
            }
        )
    )
    store = ModelStore(store_root)

    assert store.get_store_path("org/model") is None
    assert store.delete_model("org/model") is True
    assert sentinel.read_text() == "keep"


def test_register_model_rejects_path_outside_store(tmp_path: Path) -> None:
    """New registry entries cannot introduce an out-of-root canonical path."""

    store_root = tmp_path / "store"
    store_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(ValueError, match="contained by the store"):
        ModelStore(store_root).register_model(
            "org/model",
            outside,
            [],
            0,
        )


def test_legacy_registry_migrates_to_versioned_index_on_write(tmp_path: Path) -> None:
    """Existing unversioned store indexes remain readable and migrate atomically."""
    store_root = tmp_path / "store"
    model_root = store_root / "org--model"
    model_root.mkdir(parents=True)
    (model_root / "weights.bin").write_bytes(b"old")
    (store_root / "registry.json").write_text(
        json.dumps(
            {
                "org/model": {
                    "model_id": "org/model",
                    "store_path": "org--model",
                    "files": ["weights.bin"],
                    "downloaded_at": "2026-07-16T00:00:00+00:00",
                    "total_bytes": 3,
                }
            }
        )
    )
    store = ModelStore(store_root)

    store.register_model(
        "org/model",
        model_root,
        ["weights.bin"],
        3,
    )

    index = StoreRegistryIndex.model_validate_json(
        (store_root / "registry.json").read_bytes(), strict=False
    )
    assert index.schema_version == 1
    assert index.entries["org/model"].store_path == "org--model"
    backup = cast(
        "dict[str, dict[str, object]]",
        json.loads((store_root / "registry.pre-installed-cards.json").read_text()),
    )
    assert backup["org/model"]["store_path"] == "org--model"
