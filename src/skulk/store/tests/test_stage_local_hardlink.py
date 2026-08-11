"""Local staging must hardlink, not copy, when store and staging share a filesystem.

Staging by copy doubles the disk cost of every model on a store-host node: a
26GB GGUF needs 52GB of free disk to stage, which is exactly how a 50GB-disk
node ended up failing staging with ENOSPC after a successful store download
(issue #533). Store files are immutable once registered and staged files are
never mutated in place, so a hardlink is safe and costs no additional disk.
"""

from pathlib import Path

import pytest

from skulk.store.model_store import ModelStore
from skulk.store.model_store_client import ModelStoreClient


def _make_store_model(store_root: Path, sanitized_id: str) -> Path:
    """Create a fake registered store model with two files, returning its dir."""
    model_dir = store_root / sanitized_id
    (model_dir / "sub").mkdir(parents=True)
    (model_dir / "weights.gguf").write_bytes(b"g" * 4096)
    (model_dir / "sub" / "config.json").write_bytes(b"{}")
    return model_dir


async def test_local_path_lookup_never_runs_authoritative_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient client lookup must remain read-only and bounded."""

    store_root = tmp_path / "store"
    store_root.mkdir()

    def reject_recovery(
        _self: ModelStore,
        _current_registry_ids: dict[str, str] | None = None,
    ) -> None:
        raise AssertionError("transient client attempted authoritative recovery")

    monkeypatch.setattr(ModelStore, "_rebuild_registry_from_sidecars", reject_recovery)
    client = ModelStoreClient("localhost", local_store_path=store_root)

    assert await client.local_model_path("org/model", None) is None
    assert not (store_root / "registry.json").exists()


async def test_stage_local_hardlinks_on_same_filesystem(tmp_path: Path) -> None:
    """Staged files share an inode with the store files (no disk doubling)."""
    store_root = tmp_path / "store"
    store_root.mkdir()
    model_dir = _make_store_model(store_root, "org--model")
    client = ModelStoreClient("localhost", local_store_path=store_root)
    admitted_bytes: list[int] = []

    async def record_capacity(additional_bytes: int) -> None:
        admitted_bytes.append(additional_bytes)

    dest = tmp_path / "staging" / "org--model"
    dest.mkdir(parents=True)
    await client._stage_local(  # pyright: ignore[reportPrivateUsage]
        "org/model",
        dest,
        on_progress=None,
        capacity_preflight=record_capacity,
    )

    staged_weights = dest / "weights.gguf"
    staged_config = dest / "sub" / "config.json"
    assert staged_weights.read_bytes() == b"g" * 4096
    assert staged_config.read_bytes() == b"{}"
    # Same inode == hardlink == zero additional data blocks on disk.
    assert staged_weights.stat().st_ino == (model_dir / "weights.gguf").stat().st_ino
    assert staged_config.stat().st_ino == (model_dir / "sub" / "config.json").stat().st_ino
    assert admitted_bytes == [0]


async def test_stage_local_falls_back_to_copy_when_link_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EXDEV-style link failures degrade to a byte-identical copy."""
    store_root = tmp_path / "store"
    store_root.mkdir()
    model_dir = _make_store_model(store_root, "org--model")
    client = ModelStoreClient("localhost", local_store_path=store_root)

    def refuse_link(src: object, dst: object, **_kwargs: object) -> None:
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr("skulk.store.model_store_client.os.link", refuse_link)
    admitted_bytes: list[int] = []

    async def record_capacity(additional_bytes: int) -> None:
        admitted_bytes.append(additional_bytes)

    dest = tmp_path / "staging" / "org--model"
    dest.mkdir(parents=True)
    await client._stage_local(  # pyright: ignore[reportPrivateUsage]
        "org/model",
        dest,
        on_progress=None,
        capacity_preflight=record_capacity,
    )

    staged_weights = dest / "weights.gguf"
    assert staged_weights.read_bytes() == b"g" * 4096
    assert staged_weights.stat().st_ino != (model_dir / "weights.gguf").stat().st_ino
    assert admitted_bytes == [0, 4096, 2]


async def test_stage_local_replaces_stale_partial_destination(tmp_path: Path) -> None:
    """A size-mismatched (stale/partial) staged file is replaced, not kept."""
    store_root = tmp_path / "store"
    store_root.mkdir()
    _make_store_model(store_root, "org--model")
    client = ModelStoreClient("localhost", local_store_path=store_root)

    dest = tmp_path / "staging" / "org--model"
    (dest / "sub").mkdir(parents=True)
    (dest / "weights.gguf").write_bytes(b"partial")

    await client._stage_local("org/model", dest, on_progress=None)  # pyright: ignore[reportPrivateUsage]

    assert (dest / "weights.gguf").read_bytes() == b"g" * 4096


async def test_stage_local_replaces_same_size_installed_card(tmp_path: Path) -> None:
    """Metadata refresh cannot be skipped merely because JSON sizes match."""

    store_root = tmp_path / "store"
    store_root.mkdir()
    model_dir = _make_store_model(store_root, "org--model")
    source_sidecar = model_dir / ".skulk" / "installed-card.json"
    source_sidecar.parent.mkdir()
    source_sidecar.write_text("new")
    client = ModelStoreClient("localhost", local_store_path=store_root)

    destination = tmp_path / "staging" / "org--model"
    staged_sidecar = destination / ".skulk" / "installed-card.json"
    staged_sidecar.parent.mkdir(parents=True)
    staged_sidecar.write_text("old")

    await client._stage_local(  # pyright: ignore[reportPrivateUsage]
        "org/model", destination, on_progress=None
    )

    assert staged_sidecar.read_text() == "new"
