"""A cached download status must not outlive the model's files on disk.

``ModelStore._active_downloads`` caches per-model download status. A store-delete
(``delete_model``) or out-of-band file removal drops the registry entry and the
on-disk files but cannot reach that in-memory cache, so a stale ``"complete"``
can linger. If ``request_download`` trusted it, the re-download would be
short-circuited and the model would never come back, and a worker staging it
would fail "not found in store". These tests pin the two guards that prevent
that: ``delete_model`` clears the cache at the source, and ``request_download``
re-checks that the cached status's own artifact is still registered as a
backstop for any other cause of files-gone.

That backstop is artifact-level rather than alias-level (#916). Asking only
whether *some* generation of the alias is on disk let a surviving generation,
typically a legacy revision-``None`` install, vouch for a cached
pinned-revision ``"complete"`` whose own directory had been cancelled or
replaced, and placement then launched a runner against missing weight shards.
"""

from pathlib import Path

import pytest

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


def test_delete_model_leaves_in_flight_download_status(tmp_path: Path) -> None:
    # An in-flight (pending/downloading) status is owned by a live _do_download
    # task that reads self._active_downloads[model_id]; delete_model must not pop
    # it out from under that task. Only terminal (complete/failed) statuses are
    # cleared.
    store = ModelStore(tmp_path)
    _register(store, "org/bundle", ["base-Q4_K_M.gguf"])
    store._active_downloads["org/bundle"] = StoreDownloadStatus(  # pyright: ignore[reportPrivateUsage]
        model_id="org/bundle", status="downloading", progress=0.4
    )

    assert store.delete_model("org/bundle") is True

    # The in-flight entry survives so the running download task does not crash.
    assert "org/bundle" in store._active_downloads  # pyright: ignore[reportPrivateUsage]


def test_get_download_status_does_not_report_disappeared_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A concurrent directory removal must not produce a false complete state."""

    store = ModelStore(tmp_path)

    def present_before_removal(_model_id: str) -> bool:
        return True

    def missing_after_removal(_model_id: str) -> None:
        return None

    monkeypatch.setattr(store, "is_in_store", present_before_removal)
    monkeypatch.setattr(store, "get_entry", missing_after_removal)

    assert store.get_download_status("org/gone") is None


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


async def test_request_download_recovers_missing_requested_quant(
    tmp_path: Path,
) -> None:
    store = ModelStore(tmp_path)
    _register(store, "org/present", ["base-Q4_K_M.gguf", "config.json"])
    _seed_complete(store, "org/present")

    status = await store.request_download(
        "org/present",
        "base-IQ3_XXS.gguf",
        None,
    )

    assert status.status in ("pending", "downloading")


def _seed_complete_at_revision(
    store: ModelStore, model_id: str, source_revision: str
) -> None:
    """Cache a ``complete`` status pinned to one immutable revision."""
    store._active_downloads[model_id] = StoreDownloadStatus(  # pyright: ignore[reportPrivateUsage]
        model_id=model_id,
        source_revision=source_revision,
        status="complete",
        progress=1.0,
    )


async def test_cached_pinned_complete_is_stale_when_only_a_legacy_generation_remains(
    tmp_path: Path,
) -> None:
    """#916: a legacy generation must not vouch for a pinned-revision complete.

    The registered entry is a legacy install carrying no ``source_revision``,
    while the cached status claims a specific immutable revision completed. The
    directory that status referred to is gone. An alias-level presence check
    calls this fresh and short-circuits the re-download, which is how a runner
    ended up loading a directory with missing shards.
    """
    revision = "a" * 40
    store = ModelStore(tmp_path)
    _register(store, "org/legacy", ["model-00001-of-00002.safetensors", "config.json"])
    legacy_entry = store.get_entry("org/legacy")
    assert legacy_entry is not None
    assert legacy_entry.source_revision is None
    # The alias really is on disk, which is exactly why the old guard passed.
    assert store.is_in_store("org/legacy") is True
    _seed_complete_at_revision(store, "org/legacy", revision)

    status = await store.request_download("org/legacy", None, None, revision)

    assert status.status in ("pending", "downloading")


async def test_cached_complete_survives_when_the_registered_revision_agrees(
    tmp_path: Path,
) -> None:
    """The tightened guard must not break dedup for a genuinely present artifact."""
    revision = "b" * 40
    store = ModelStore(tmp_path)
    model_dir = store.store_path / "org--pinned"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in ("model.safetensors", "config.json"):
        _ = (model_dir / name).write_text("x")
    store.register_model(
        "org/pinned",
        model_dir,
        ["model.safetensors", "config.json"],
        1,
        repo_has_projector=False,
        source_revision=revision,
    )
    _seed_complete_at_revision(store, "org/pinned", revision)

    status = await store.request_download("org/pinned", None, None, revision)

    assert status.status == "complete"
