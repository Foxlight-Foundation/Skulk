"""Peer-cache imports are resumable, verified and atomically published."""

import socket
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from aiohttp import web

from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.memory import Memory
from skulk.store.installed_cards import (
    build_installed_card_record,
    write_installed_card,
)
from skulk.store.model_store import (
    MINIMUM_STAGING_FREE_DISK_BYTES,
    ModelStore,
    read_reconciliation_tombstones,
)
from skulk.store.peer_exports import ArtifactExportManager


def _card() -> ModelCard:
    return ModelCard(
        model_id=ModelId("org/model"),
        storage_size=Memory.from_mb(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        source_revision="a" * 40,
        registry_card_id=f"card_{'a' * 52}",
        registry_snapshot_id="snapshot_1_test",
        registry_provenance="foxlight",
    )


async def test_peer_import_resumes_and_verifies_before_publish(tmp_path: Path) -> None:
    source = tmp_path / "source" / "org--model"
    source.mkdir(parents=True)
    payload = b"verified-model-weights"
    (source / "model.safetensors").write_bytes(payload)
    (source / ".skulk-source-revision").write_text(f"{'a' * 40}\n")
    record = build_installed_card_record(source, _card())
    write_installed_card(source, record)
    manager = ArtifactExportManager()
    grant = manager.issue(
        source,
        manifest_sha256=record.manifest_sha256,
        target_node_id="store-node",
    )

    async def export(request: web.Request) -> web.Response:
        path, _ = manager.resolve(
            request.match_info["token"],
            target_node_id=request.headers["X-Skulk-Store-Node"],
            relative_path=request.match_info["path"],
            range_header=request.headers.get("Range"),
        )
        data = path.read_bytes()
        range_header = request.headers.get("Range")
        if range_header is None:
            return web.Response(body=data)
        offset = int(range_header.removeprefix("bytes=").removesuffix("-"))
        return web.Response(status=206, body=data[offset:])

    app = web.Application()
    app.router.add_get("/store/internal/exports/{token}/{path:.*}", export)
    runner = web.AppRunner(app)
    await runner.setup()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = cast("tuple[str, int]", probe.getsockname())[1]
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()

    store_root = tmp_path / "store"
    store = ModelStore(store_root)
    partial = (
        store_root
        / ".imports"
        / f"{record.installed_identity}.partial"
        / "model.safetensors.partial"
    )
    partial.parent.mkdir(parents=True)
    partial.write_bytes(payload[:5])
    try:
        entry = await store.import_peer_artifact(
            record,
            source_base_url=f"http://127.0.0.1:{port}",
            capability_token=grant.token,
            target_node_id="store-node",
        )
    finally:
        await runner.cleanup()

    assert entry.installed_card == record
    canonical = store.get_store_path("org/model")
    assert canonical is not None
    assert (canonical / "model.safetensors").read_bytes() == payload
    assert source.exists()

    # A duplicate replica is already satisfied by identity and digest and does
    # not need the now-stopped source server.
    duplicate = await store.import_peer_artifact(
        record,
        source_base_url=f"http://127.0.0.1:{port}",
        capability_token=grant.token,
        target_node_id="store-node",
    )
    assert duplicate.store_path == entry.store_path


async def test_peer_import_capacity_credits_resumable_partial_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capacity admission reserves only the suffix still needing transfer."""

    source = tmp_path / "source" / "org--model"
    source.mkdir(parents=True)
    payload = b"verified-model-weights"
    (source / "model.safetensors").write_bytes(payload)
    record = build_installed_card_record(source, _card())
    write_installed_card(source, record)
    manager = ArtifactExportManager()
    grant = manager.issue(
        source,
        manifest_sha256=record.manifest_sha256,
        target_node_id="store-node",
    )

    async def export(request: web.Request) -> web.Response:
        path, _ = manager.resolve(
            request.match_info["token"],
            target_node_id=request.headers["X-Skulk-Store-Node"],
            relative_path=request.match_info["path"],
            range_header=request.headers.get("Range"),
        )
        offset = int(request.headers["Range"].removeprefix("bytes=").removesuffix("-"))
        return web.Response(status=206, body=path.read_bytes()[offset:])

    app = web.Application()
    app.router.add_get("/store/internal/exports/{token}/{path:.*}", export)
    runner = web.AppRunner(app)
    await runner.setup()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = cast("tuple[str, int]", probe.getsockname())[1]
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()

    store_root = tmp_path / "store"
    store = ModelStore(store_root)
    partial = (
        store_root
        / ".imports"
        / f"{record.installed_identity}.partial"
        / "model.safetensors.partial"
    )
    partial.parent.mkdir(parents=True)
    prefix_bytes = 15
    partial.write_bytes(payload[:prefix_bytes])
    free_bytes = MINIMUM_STAGING_FREE_DISK_BYTES + len(payload) - prefix_bytes

    def _disk_usage(_path: Path) -> SimpleNamespace:
        return SimpleNamespace(free=free_bytes)

    monkeypatch.setattr(
        "skulk.store.model_store.shutil.disk_usage",
        _disk_usage,
    )
    try:
        entry = await store.import_peer_artifact(
            record,
            source_base_url=f"http://127.0.0.1:{port}",
            capability_token=grant.token,
            target_node_id="store-node",
        )
    finally:
        await runner.cleanup()

    assert entry.installed_card == record
    canonical = store.get_store_path("org/model")
    assert canonical is not None
    assert (canonical / "model.safetensors").read_bytes() == payload


async def test_peer_import_promotes_complete_partial_without_network(
    tmp_path: Path,
) -> None:
    """A crash after the final write but before rename resumes locally."""

    source = tmp_path / "source" / "org--model"
    source.mkdir(parents=True)
    payload = b"complete-before-rename"
    (source / "model.safetensors").write_bytes(payload)
    record = build_installed_card_record(source, _card())

    store_root = tmp_path / "store"
    store = ModelStore(store_root)
    partial = (
        store_root
        / ".imports"
        / f"{record.installed_identity}.partial"
        / "model.safetensors.partial"
    )
    partial.parent.mkdir(parents=True)
    partial.write_bytes(payload)

    entry = await store.import_peer_artifact(
        record,
        source_base_url="http://127.0.0.1:1",
        capability_token="unused-because-no-request-is-needed",
        target_node_id="store-node",
    )

    canonical = store.get_store_path("org/model")
    assert entry.installed_card == record
    assert canonical is not None
    assert (canonical / "model.safetensors").read_bytes() == payload


async def test_peer_import_repairs_corrupt_matching_canonical_generation(
    tmp_path: Path,
) -> None:
    """Identity equality cannot short-circuit manifest corruption recovery."""

    source = tmp_path / "source" / "org--model"
    source.mkdir(parents=True)
    payload = b"healthy-replica"
    (source / "model.safetensors").write_bytes(payload)
    record = build_installed_card_record(source, _card())
    write_installed_card(source, record)
    manager = ArtifactExportManager()
    grant = manager.issue(
        source,
        manifest_sha256=record.manifest_sha256,
        target_node_id="store-node",
    )

    async def export(request: web.Request) -> web.Response:
        path, _ = manager.resolve(
            request.match_info["token"],
            target_node_id=request.headers["X-Skulk-Store-Node"],
            relative_path=request.match_info["path"],
        )
        return web.Response(body=path.read_bytes())

    app = web.Application()
    app.router.add_get("/store/internal/exports/{token}/{path:.*}", export)
    runner = web.AppRunner(app)
    await runner.setup()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = cast("tuple[str, int]", probe.getsockname())[1]
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()

    store_root = tmp_path / "store"
    canonical = store_root / "org--model"
    canonical.mkdir(parents=True)
    (canonical / "model.safetensors").write_bytes(payload)
    write_installed_card(canonical, record)
    store = ModelStore(store_root)
    (canonical / "model.safetensors").write_bytes(b"corrupt")
    try:
        imported = await store.import_peer_artifact(
            record,
            source_base_url=f"http://127.0.0.1:{port}",
            capability_token=grant.token,
            target_node_id="store-node",
        )
    finally:
        await runner.cleanup()

    repaired = store.get_store_path("org/model")
    assert imported.installed_card == record
    assert repaired is not None
    assert (repaired / "model.safetensors").read_bytes() == payload


async def test_deleted_alias_tombstone_blocks_stale_peer_import(
    tmp_path: Path,
) -> None:
    """A missed node-cache eviction cannot recreate deleted store bytes."""

    store_root = tmp_path / "store"
    canonical = store_root / "org--model"
    canonical.mkdir(parents=True)
    (canonical / "model.safetensors").write_bytes(b"weights")
    record = build_installed_card_record(canonical, _card())
    store = ModelStore(store_root)
    store.register_model(
        "org/model",
        canonical,
        ["model.safetensors"],
        len(b"weights"),
        installed_card=record,
    )

    assert store.delete_model("org/model")
    assert read_reconciliation_tombstones(store_root) == frozenset({"org/model"})
    with pytest.raises(ValueError, match="operator-deletion tombstone"):
        await store.import_peer_artifact(
            record,
            source_base_url="http://127.0.0.1:1",
            capability_token="stale-replica",
            target_node_id="store-node",
        )
    assert store.get_entry("org/model") is None


async def test_failed_peer_replacement_preserves_old_generation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source" / "org--model"
    source.mkdir(parents=True)
    (source / "model.safetensors").write_bytes(b"new-generation")
    record = build_installed_card_record(source, _card())
    write_installed_card(source, record)
    manager = ArtifactExportManager()
    grant = manager.issue(
        source,
        manifest_sha256=record.manifest_sha256,
        target_node_id="store-node",
    )

    serve_corrupt_bytes = True

    async def corrupt_then_healthy_export(_request: web.Request) -> web.Response:
        return web.Response(
            body=(b"corrupt-bytes!" if serve_corrupt_bytes else b"new-generation")
        )

    app = web.Application()
    app.router.add_get(
        "/store/internal/exports/{token}/{path:.*}", corrupt_then_healthy_export
    )
    runner = web.AppRunner(app)
    await runner.setup()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = cast("tuple[str, int]", probe.getsockname())[1]
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()

    store_root = tmp_path / "store"
    old = store_root / "org--model"
    old.mkdir(parents=True)
    (old / "model.safetensors").write_bytes(b"old-generation")
    store = ModelStore(store_root)
    store.register_model(
        "org/model",
        old,
        ["model.safetensors"],
        len(b"old-generation"),
    )
    try:
        with pytest.raises(RuntimeError, match="length|digest"):
            await store.import_peer_artifact(
                record,
                source_base_url=f"http://127.0.0.1:{port}",
                capability_token=grant.token,
                target_node_id="store-node",
            )
        assert store.get_store_path("org/model") == old
        assert (old / "model.safetensors").read_bytes() == b"old-generation"
        serve_corrupt_bytes = False
        imported = await store.import_peer_artifact(
            record,
            source_base_url=f"http://127.0.0.1:{port}",
            capability_token=grant.token,
            target_node_id="store-node",
        )
    finally:
        await runner.cleanup()

    canonical = store.get_store_path("org/model")
    assert imported.installed_card == record
    assert canonical is not None and canonical != old
    assert (canonical / "model.safetensors").read_bytes() == b"new-generation"


def test_export_capability_is_target_and_manifest_bound(tmp_path: Path) -> None:
    source = tmp_path / "org--model"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"weights")
    record = build_installed_card_record(source, _card())
    write_installed_card(source, record)
    manager = ArtifactExportManager()
    grant = manager.issue(
        source,
        manifest_sha256=record.manifest_sha256,
        target_node_id="store-node",
    )

    try:
        manager.resolve(
            grant.token,
            target_node_id="another-node",
            relative_path="model.safetensors",
        )
    except PermissionError:
        pass
    else:
        raise AssertionError("export token was accepted by the wrong target")


def test_export_capability_enforces_file_snapshot_and_byte_ceiling(
    tmp_path: Path,
) -> None:
    source = tmp_path / "org--model"
    source.mkdir()
    weights = source / "model.safetensors"
    weights.write_bytes(b"weights")
    record = build_installed_card_record(source, _card())
    write_installed_card(source, record)
    manager = ArtifactExportManager()
    grant = manager.issue(
        source,
        manifest_sha256=record.manifest_sha256,
        target_node_id="store-node",
    )

    manager.resolve(
        grant.token,
        target_node_id="store-node",
        relative_path="model.safetensors",
        range_header="bytes=5-",
    )
    manager.resolve(
        grant.token,
        target_node_id="store-node",
        relative_path="model.safetensors",
        range_header="bytes=0-4",
    )
    with pytest.raises(PermissionError, match="byte ceiling"):
        manager.resolve(
            grant.token,
            target_node_id="store-node",
            relative_path="model.safetensors",
            range_header="bytes=0-0",
        )

    replacement_grant = manager.issue(
        source,
        manifest_sha256=record.manifest_sha256,
        target_node_id="store-node",
    )
    weights.write_bytes(b"changed")
    with pytest.raises(PermissionError, match="file changed"):
        manager.resolve(
            replacement_grant.token,
            target_node_id="store-node",
            relative_path="model.safetensors",
        )
