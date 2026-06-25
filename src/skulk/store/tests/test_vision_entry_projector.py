"""Store-side stale-vision detection via the registered projector flag (#346).

``vision_entry_missing_projector`` decides whether an in-store GGUF entry is a
vision model missing its projector. The registration guard records whether the
upstream repo ships a projector, so the steady-state answer is local (no HF
probe on the store-availability hot path); only legacy entries (flag ``None``)
fall back to a probe.
"""

from pathlib import Path

from skulk.store.model_store import ModelStore


def _register(
    store: ModelStore,
    model_id: str,
    files: list[str],
    repo_has_projector: bool | None,
) -> None:
    model_dir = store.store_path / model_id.replace("/", "--")
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        (model_dir / name).write_text("x")
    store.register_model(
        model_id, model_dir, files, 1, repo_has_projector=repo_has_projector
    )


async def test_flagged_vision_entry_without_projector_is_stale(tmp_path: Path) -> None:
    store = ModelStore(tmp_path)
    _register(
        store, "org/vlm", ["model-Q4_K_M.gguf", "config.json"], repo_has_projector=True
    )
    assert await store.vision_entry_missing_projector("org/vlm") is True


async def test_flagged_text_entry_is_not_stale_without_hf(tmp_path: Path) -> None:
    store = ModelStore(tmp_path)
    _register(
        store, "org/text", ["model-Q4_K_M.gguf", "config.json"], repo_has_projector=False
    )
    assert await store.vision_entry_missing_projector("org/text") is False


async def test_entry_with_projector_is_complete(tmp_path: Path) -> None:
    store = ModelStore(tmp_path)
    _register(
        store,
        "org/vlm",
        ["model-Q4_K_M.gguf", "mmproj-F16.gguf", "config.json"],
        repo_has_projector=True,
    )
    assert await store.vision_entry_missing_projector("org/vlm") is False


async def test_absent_model_is_not_stale(tmp_path: Path) -> None:
    store = ModelStore(tmp_path)
    assert await store.vision_entry_missing_projector("org/missing") is False


async def test_legacy_non_gguf_entry_skips_hf_probe(tmp_path: Path) -> None:
    # Legacy entry (flag None) that is not a GGUF model must short-circuit to
    # False without any HF probe (an MLX model has no projector to be missing).
    store = ModelStore(tmp_path)
    _register(
        store,
        "org/mlx",
        ["model.safetensors", "config.json"],
        repo_has_projector=None,
    )
    assert await store.vision_entry_missing_projector("org/mlx") is False
