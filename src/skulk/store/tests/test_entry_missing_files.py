"""Store-side companion completeness via the registered file list (#422).

``entry_missing_files`` decides whether an in-store entry omits a requested
same-repo companion GGUF (a served-engine draft bundled with the base). A stale
base-only entry (registered before the card declared a draft) must report the
draft missing so a re-download recovers it before staging.
"""

from pathlib import Path

from skulk.store.model_store import ModelStore


def _register(store: ModelStore, model_id: str, files: list[str]) -> None:
    model_dir = store.store_path / model_id.replace("/", "--")
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        (model_dir / name).write_text("x")
    store.register_model(model_id, model_dir, files, 1, repo_has_projector=False)


def test_entry_missing_requested_companion_is_flagged(tmp_path: Path) -> None:
    store = ModelStore(tmp_path)
    _register(store, "org/bundle", ["base-IQ4_XS.gguf", "config.json"])
    assert store.entry_missing_files("org/bundle", ["draft.gguf"]) is True


def test_entry_with_companion_is_complete(tmp_path: Path) -> None:
    store = ModelStore(tmp_path)
    _register(store, "org/bundle", ["base-IQ4_XS.gguf", "config.json", "draft.gguf"])
    assert store.entry_missing_files("org/bundle", ["draft.gguf"]) is False


def test_no_required_files_is_complete(tmp_path: Path) -> None:
    store = ModelStore(tmp_path)
    _register(store, "org/bundle", ["base-IQ4_XS.gguf"])
    assert store.entry_missing_files("org/bundle", []) is False


def test_absent_model_is_not_flagged(tmp_path: Path) -> None:
    # Not in store: the normal download path handles it, not the recovery guard.
    store = ModelStore(tmp_path)
    assert store.entry_missing_files("org/missing", ["draft.gguf"]) is False


async def test_cached_complete_entry_redownloads_when_companion_missing(
    tmp_path: Path,
) -> None:
    # A prior base-only download leaves a "complete" status in _active_downloads.
    # A later request naming a same-repo companion must NOT return that cached
    # status (which would skip recovery): it drops the stale entry and re-runs
    # the download so entry_missing_files recovery can fetch the companion.
    from skulk.store.model_store import StoreDownloadStatus

    store = ModelStore(tmp_path)
    _register(store, "org/bundle", ["base-IQ4_XS.gguf", "config.json"])
    store._active_downloads["org/bundle"] = StoreDownloadStatus(  # pyright: ignore[reportPrivateUsage]
        model_id="org/bundle", status="complete", progress=1.0
    )
    status = await store.request_download(
        "org/bundle", "base-IQ4_XS.gguf", ["draft.gguf"]
    )
    # Recovery kicked off: a fresh pending/downloading status, not the cached
    # complete one (the draft is genuinely missing from the registry).
    assert status.status in ("pending", "downloading")


async def test_cached_complete_entry_returned_when_nothing_missing(
    tmp_path: Path,
) -> None:
    from skulk.store.model_store import StoreDownloadStatus

    store = ModelStore(tmp_path)
    _register(store, "org/bundle", ["base-IQ4_XS.gguf", "config.json", "draft.gguf"])
    cached = StoreDownloadStatus(model_id="org/bundle", status="complete", progress=1.0)
    store._active_downloads["org/bundle"] = cached  # pyright: ignore[reportPrivateUsage]
    status = await store.request_download(
        "org/bundle", "base-IQ4_XS.gguf", ["draft.gguf"]
    )
    # Nothing missing: the cached complete status is returned (dedup preserved).
    assert status.status == "complete"
