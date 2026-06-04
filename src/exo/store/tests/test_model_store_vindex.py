from pathlib import Path

from exo.store.model_store import ModelStore


def test_model_store_registers_vindex_directory(tmp_path: Path) -> None:
    root = tmp_path / "store"
    vindex = root / "skulk--gemma-vindex"
    vindex.mkdir(parents=True)
    (vindex / "metadata.json").write_text("{}")
    (vindex / "weights.bin").write_bytes(b"abc")

    store = ModelStore(root)
    store.register_vindex(
        "skulk/gemma-vindex",
        vindex,
        ["metadata.json", "weights.bin"],
        5,
    )

    entries = store.list_models()

    assert len(entries) == 1
    assert entries[0].artifact_kind == "vindex"
    assert entries[0].model_id == "skulk/gemma-vindex"
    assert store.get_store_path("skulk/gemma-vindex") == vindex

