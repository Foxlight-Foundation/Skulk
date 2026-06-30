"""A cached download status must not outlive the model's files on disk.

``ModelStore._active_downloads`` caches per-model download status. A store-delete
(``delete_model``) or out-of-band file removal drops the registry entry and the
on-disk files but cannot reach that in-memory cache, so a stale ``"complete"``
can linger. If ``request_download`` trusted it, the re-download would be
short-circuited and the model would never come back, and a worker staging it
would fail "not found in store". These tests pin the two guards that prevent
that: ``delete_model`` clears the cache at the source, and ``request_download``
re-checks ``is_in_store`` as a backstop for any other cause of files-gone.
"""

from pathlib import Path

from skulk.store.model_store import ModelStore, StoreDownloadStatus


def _register(store: ModelStore, model_id: str, files: list[str]) -> None:
    model_dir = store.store_path / model_id.replace("/", "--")
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        (model_dir / name).write_text("x")
    store.register_model(model_id, model_dir, files, 1, repo_has_projector=False)


def _seed_complete(store: ModelStore, model_id: str) -> None:
    store._active_downloads[model_id] = StoreDownloadStatus(  # pyright: ignore[reportPrivateUsage]
        model_id=model_id, status="complete", progress=1.0
    )


def test_delete_model_clears_cached_download_status(tmp_path: Path) -> None:
    store = ModelStore(tmp_path)
    _register(store, "org/bundle", ["base-Q4_K_M.gguf", "config.json"])
    _seed_complete(store, "org/bundle")

    assert store.delete_model("org/bundle") is True

    # The stale "complete" is gone, so a later request_download re-fetches the
    # deleted model instead of short-circuiting on it.
    assert "org/bundle" not in store._active_downloads  # pyright: ignore[reportPrivateUsage]


async def test_request_download_redownloads_stale_complete_when_files_gone(
    tmp_path: Path,
) -> None:
    # Files were removed out-of-band (e.g. a delete that did not clear the cache,
    # or a manual rmtree) but a "complete" status lingers in memory.
    store = ModelStore(tmp_path)
    _seed_complete(store, "org/gone")
    assert store.is_in_store("org/gone") is False

    status = await store.request_download("org/gone", "base-Q4_K_M.gguf", None)

    # The stale complete is dropped and a real re-download is kicked off rather
    # than returning the lie.
    assert status.status in ("pending", "downloading")


async def test_request_download_keeps_cached_complete_when_still_in_store(
    tmp_path: Path,
) -> None:
    # Dedup is preserved for an entry that genuinely is still in the store.
    store = ModelStore(tmp_path)
    _register(store, "org/present", ["base-Q4_K_M.gguf", "config.json"])
    _seed_complete(store, "org/present")

    status = await store.request_download("org/present", "base-Q4_K_M.gguf", None)

    assert status.status == "complete"
