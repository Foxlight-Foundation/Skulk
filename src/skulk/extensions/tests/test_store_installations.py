"""Installations bound from the built-in store follow the store's trust while they run."""

import asyncio
import hashlib
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import JsonValue, TypeAdapter

from skulk.extensions.capability_store import (
    CAPABILITY_STORE_CATALOG_DOCUMENT,
    CAPABILITY_STORE_CATALOG_URL,
    StoreTrustClient,
)
from skulk.extensions.runtime_artifacts import RuntimeTrust, canonical_json
from skulk.extensions.runtime_attachment import HostSettings
from skulk.extensions.runtime_catalog import CatalogReview, CatalogSourceUpdate
from skulk.extensions.runtime_controller import LifecycleRequest
from skulk.extensions.runtime_download import RuntimeDownloads, SourceUpdate
from skulk.extensions.runtime_files import (
    private_directory,
    read_private,
    write_private,
)
from skulk.extensions.runtime_manager import (
    CatalogInstall,
    CatalogInstallation,
    CatalogInstallRequest,
    CatalogRequest,
    InstallationRequest,
    RuntimeManager,
    SourceRegistration,
    StoreTrustRefresh,
    StoreTrustRequest,
    SubmitRequest,
    store_bound,
)
from skulk.extensions.tests.store_repository import StoreRepository, trust_document
from skulk.extensions.tests.test_runtime_install import artifacts
from skulk.extensions.tests.test_runtime_service import OWNER_SOURCE, running, status

FEED = "https://skulkapps.foxlight.ai/releases/example/1/"
PRIVATE = "https://private.example.test/"
_OBJECT: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])


class _ShiftedClock:
    """The installer's clock, moved forward without moving anything else's."""

    def __init__(self) -> None:
        self.offset = 0

    def time(self) -> float:
        """Wall time as the installer sees it."""
        return time.time() + self.offset

    def monotonic(self) -> float:
        """Unshifted: waits and deadlines keep real time."""
        return time.monotonic()


def _listing(metadata: bytes, key: Ed25519PrivateKey) -> bytes:
    """A protocol 1 catalog listing the signed release the way a publisher derives it."""
    signed = _OBJECT.validate_json(metadata)
    runtime = _OBJECT.validate_python(signed["runtime"])
    release = _OBJECT.validate_python(runtime["release"])
    manifest = _OBJECT.validate_python(release["manifest"])
    artifact_size = TypeAdapter(int).validate_python(release["artifact_size"])
    wheel_bytes = sum(
        TypeAdapter(int).validate_python(_OBJECT.validate_python(wheel)["size"])
        for wheel in TypeAdapter(list[JsonValue]).validate_python(runtime["wheels"])
    )
    now = int(time.time())
    catalog: dict[str, JsonValue] = {
        "protocol": 1,
        "publisher": "fixture",
        "revision": 1,
        "created_at": now - 1,
        "expires_at": now + 3600,
        "entries": [
            {
                "bundle_id": manifest["bundle_id"],
                "bundle_version": manifest["bundle_version"],
                "title": "Example plugin",
                "publisher": release["publisher"],
                "sequence": release["sequence"],
                "feed_url": FEED,
                "release_sha256": hashlib.sha256(metadata).hexdigest(),
                "release_digest": hashlib.sha256(canonical_json(runtime)).hexdigest(),
                "runtime_platform": runtime["platform"],
                "artifact_sha256": manifest["executable_sha256"],
                "artifact_size": artifact_size,
                "transfer_bytes": artifact_size + wheel_bytes,
                "platforms": release["platforms"],
                "skulk_build_sha256": release["skulk_build_sha256"],
                "permissions": release["permissions"],
                "descriptors": ["example.echo@1.0.0"],
                "expires_at": release["expires_at"],
            }
        ],
    }
    signature = key.sign(canonical_json(catalog)).hex()
    return json.dumps({"catalog": catalog, "signature": signature}).encode()


@dataclass
class _Host:
    manager: RuntimeManager
    repository: StoreRepository
    key: Ed25519PrivateKey
    metadata: bytes
    source: Path
    clock: _ShiftedClock
    feed_up: list[bool]

    def publish(self, revision: int, **revocations: tuple[str, ...]) -> None:
        """Sign a store trust revision naming the release's publisher key."""
        self.repository.publish_trust(
            trust_document(
                self.key,
                revision=revision,
                expires_in=600 if revision == 1 else 86400,
                revoked_publishers=revocations.get("publishers", ()),
                revoked_artifacts=revocations.get("artifacts", ()),
            )
        )

    async def refresh(self, *, offline: bool = False) -> StoreTrustRefresh:
        """The node's periodic renewal request."""
        return StoreTrustRefresh.model_validate_json(
            json.dumps(await self.manager.dispatch(StoreTrustRequest(offline=offline)))
        )

    async def bind(self, plugin_id: str) -> CatalogInstallation:
        """Read the catalog and bind ``plugin_id`` to its one listing."""
        review = CatalogReview.model_validate_json(
            json.dumps(
                await self.manager.dispatch(CatalogRequest(action="read_catalog"))
            )
        )
        return CatalogInstallation.model_validate_json(
            json.dumps(
                await self.manager.dispatch(
                    CatalogInstallRequest(
                        request=CatalogInstall(
                            catalog_sha256=review.catalog_sha256,
                            bundle_id="example.plugin",
                            sequence=1,
                            runtime_platform="macos-arm64",
                            plugin_id=plugin_id,
                        )
                    )
                )
            )
        )

    def trust(self, plugin_id: str) -> RuntimeTrust:
        """The installation's own publisher trust, as its verification reads it."""
        return RuntimeTrust.model_validate_json(
            read_private(
                self.manager.downloads[plugin_id].root / "publisher-trust.json"
            )
        )

    async def run(self, plugin_id: str) -> None:
        """Stage the bound release and activate it, then wait for its owner."""
        controller = self.manager.controllers[plugin_id]
        staged = await controller.selector.installer.stage(self.metadata, self.source)
        await self.manager.dispatch(
            SubmitRequest(
                plugin_id=plugin_id,
                request=LifecycleRequest(
                    operation_id="1" * 32,
                    action="activate",
                    expected_revision=0,
                    runtime_digest=staged.runtime_digest,
                ),
            )
        )
        assert controller.work is not None
        await controller.work
        assert controller.service is not None
        await running(controller.service)


@asynccontextmanager
async def _store_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[_Host]:
    """A real manager whose catalog is the built-in store, with no network."""
    source = tmp_path / "source"
    key = Ed25519PrivateKey.generate()
    metadata, _, host = artifacts(source, owner_source=OWNER_SOURCE, signing_key=key)
    monkeypatch.setattr("skulk.extensions.runtime_install.measure_host", lambda: host)
    monkeypatch.setattr("skulk.extensions.runtime_manager.measure_host", lambda: host)
    clock = _ShiftedClock()
    monkeypatch.setattr("skulk.extensions.runtime_install.time", clock)
    # Owners re-verify their generation every 30 seconds in production.
    monkeypatch.setattr("skulk.extensions.runtime_service._CHECK_SECONDS", 0.05)
    listing = _listing(metadata, key)

    feed_up = [True]

    def feed(request: httpx.Request) -> httpx.Response:
        if feed_up[0] and str(request.url) == FEED + "release.json":
            return httpx.Response(200, content=metadata)
        return httpx.Response(404)

    def catalogs(request: httpx.Request) -> httpx.Response:
        if str(request.url) in (
            CAPABILITY_STORE_CATALOG_URL + CAPABILITY_STORE_CATALOG_DOCUMENT,
            PRIVATE + "catalog.json",
        ):
            return httpx.Response(200, content=listing)
        return httpx.Response(404)

    def downloads(root: Path) -> RuntimeDownloads:
        return RuntimeDownloads(root, transport=httpx.MockTransport(feed))

    monkeypatch.setattr("skulk.extensions.runtime_manager.RuntimeDownloads", downloads)
    root = tmp_path / "manager"
    private_directory(root)
    write_private(
        root / "host.json",
        HostSettings(transport_node_id="fixture-peer").model_dump_json().encode(),
    )
    manager = RuntimeManager(root)
    repository = StoreRepository()
    (tmp_path / "store-root.json").write_bytes(repository.embedded_root)
    manager.catalog.store = StoreTrustClient(
        root / "store-trust",
        embedded_root=tmp_path / "store-root.json",
        fetcher=repository.fetcher,
    )
    manager.catalog.transport = httpx.MockTransport(catalogs)
    await manager.start()
    try:
        yield _Host(manager, repository, key, metadata, source, clock, feed_up)
    finally:
        await manager.close()


async def _state(host: _Host, plugin_id: str, error_code: str) -> None:
    """Wait for the owner's service to report ``error_code``."""
    root = host.manager.controllers[plugin_id].root
    async with asyncio.timeout(10):
        while status(root).error_code != error_code:
            await asyncio.sleep(0.02)


async def test_a_running_store_installation_follows_renewal_and_revocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A renewal keeps the owner past its bind-time expiry; a revocation stops it."""
    async with _store_host(tmp_path, monkeypatch) as host:
        host.publish(1)
        bound = await host.bind("managed.store")
        installation = host.manager.downloads["managed.store"]
        assert store_bound(installation.root)
        bind_time = host.trust("managed.store")
        assert bind_time.revision == 1
        assert bind_time.publishers == {
            "fixture": host.key.public_key().public_bytes_raw().hex()
        }
        # A private installation beside it: its owner's trust is its own.
        await host.manager.dispatch(
            InstallationRequest(action="register", plugin_id="managed.private")
        )
        await host.manager.dispatch(
            SourceRegistration(
                plugin_id="managed.private",
                request=SourceUpdate(
                    expected_revision=0,
                    base_url=PRIVATE,
                    metadata_filename="release.json",
                    trust=bind_time.model_copy(
                        update={"expires_at": int(time.time()) + 3600}
                    ),
                ),
            )
        )
        private = host.trust("managed.private")
        await host.run("managed.store")
        service = host.manager.controllers["managed.store"].service
        assert service is not None and service.process is not None
        # The store renews its trust; the running installation takes it up
        # at its own next revision, with its source revision untouched.
        host.publish(2)
        refreshed = await host.refresh()
        assert (refreshed.trust_revision, refreshed.followed) == (2, ("managed.store",))
        renewed = host.trust("managed.store")
        assert renewed.revision == 2 and renewed.expires_at > bind_time.expires_at
        assert installation.source_status().revision == bound.source.revision
        # Twenty minutes on, past the bind-time expiry: still running.
        host.clock.offset = 1200
        await asyncio.sleep(0.5)
        assert status(service.root).state == "running"
        assert service.process.returncode is None
        # Nothing changed, so nothing is rewritten.
        assert (await host.refresh()).followed == ()
        # The store revokes the installed release: its owner stops.
        selection = host.manager.controllers["managed.store"].selector.current()
        assert selection is not None
        host.publish(3, artifacts=(selection.runtime_digest,))
        assert (await host.refresh()).followed == ("managed.store",)
        await _state(host, "managed.store", "verification_failed")
        # The private installation never followed any of it.
        assert host.trust("managed.private") == private
        assert not store_bound(host.manager.downloads["managed.private"].root)


async def test_an_offline_store_installation_runs_on_its_held_trust_until_expiry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline, nothing reaches the store; the held trust serves until it expires."""
    async with _store_host(tmp_path, monkeypatch) as host:
        host.publish(1)
        await host.bind("managed.store")
        await host.run("managed.store")
        host.publish(2)
        requests = len(host.repository.fetcher.requests)
        refreshed = await host.refresh(offline=True)
        assert (refreshed.trust_revision, refreshed.followed) == (1, ())
        assert host.repository.fetcher.requests[requests:] == []
        assert host.trust("managed.store").revision == 1
        await asyncio.sleep(0.3)
        root = host.manager.controllers["managed.store"].root
        assert status(root).state == "running"
        host.clock.offset = 1200
        await _state(host, "managed.store", "verification_failed")


async def test_only_a_binding_from_the_store_follows_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The owner's own source, or a private catalog's listing, ends following."""
    async with _store_host(tmp_path, monkeypatch) as host:
        host.publish(1)
        bound = await host.bind("managed.store")
        root = host.manager.downloads["managed.store"].root
        assert store_bound(root)
        # The owner configures the source directly: the owner's trust governs.
        await host.manager.dispatch(
            SourceRegistration(
                plugin_id="managed.store",
                request=SourceUpdate(
                    expected_revision=bound.source.revision, base_url=FEED
                ),
            )
        )
        assert not store_bound(root)
        host.publish(2)
        held = host.trust("managed.store")
        # Nothing follows the store now, so the renewal does not contact it.
        requests = len(host.repository.fetcher.requests)
        refreshed = await host.refresh()
        assert (refreshed.trust_revision, refreshed.followed) == (None, ())
        assert host.repository.fetcher.requests[requests:] == []
        assert host.trust("managed.store") == held
        # Bound from the store again, it follows again, and takes the
        # store's revocations exactly as published.
        await host.bind("managed.store")
        assert store_bound(root)
        current = host.manager.catalog.trust()
        assert current.revision == 2
        assert host.trust("managed.store").model_copy(update={"revision": 2}) == current
        # The host moves to a private catalog; a listing bound from it ends
        # following, as a private installation behaves today.
        status_now = host.manager.catalog.source_status()
        await host.manager.catalog.configure(
            CatalogSourceUpdate(
                expected_revision=status_now.revision,
                base_url=PRIVATE,
                trust=RuntimeTrust(
                    revision=(status_now.trust_revision or 0) + 1,
                    expires_at=int(time.time()) + 3600,
                    publishers={
                        "fixture": host.key.public_key().public_bytes_raw().hex()
                    },
                ),
            )
        )
        await host.bind("managed.store")
        assert not store_bound(root)
        followed = host.trust("managed.store")
        host.publish(3, publishers=("fixture",))
        assert (await host.refresh()).followed == ()
        assert host.trust("managed.store") == followed


async def test_a_refused_store_binding_restores_the_private_trust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store binding whose release is refused leaves the owner's authority as it was."""
    async with _store_host(tmp_path, monkeypatch) as host:
        # This test asserts verification explicitly; the owner's periodic
        # check must not race the binding for the installation fence.
        monkeypatch.setattr("skulk.extensions.runtime_service._CHECK_SECONDS", 30.0)
        other = Ed25519PrivateKey.generate().public_key().public_bytes_raw().hex()
        private = RuntimeTrust(
            revision=1,
            expires_at=int(time.time()) + 3600,
            publishers={
                "fixture": host.key.public_key().public_bytes_raw().hex(),
                "other": other,
            },
            revoked_artifacts=("e" * 64,),
        )
        await host.manager.dispatch(
            InstallationRequest(action="register", plugin_id="managed.private")
        )
        await host.manager.dispatch(
            SourceRegistration(
                plugin_id="managed.private",
                request=SourceUpdate(
                    expected_revision=0,
                    base_url=PRIVATE,
                    metadata_filename="release.json",
                    trust=private,
                ),
            )
        )
        await host.run("managed.private")
        installation = host.manager.downloads["managed.private"]
        before = installation.source_status()
        # The store lists the same release; binding to it fails at inspection.
        host.publish(1)
        host.feed_up[0] = False
        with pytest.raises(ValueError):
            await host.bind("managed.private")
        restored = host.trust("managed.private")
        # Bound at revision 2, restored at 3: the trust floor only rises.
        assert before.trust_revision == 1 and restored.revision == 3
        assert restored.model_copy(update={"revision": private.revision}) == private
        assert not store_bound(installation.root)
        assert installation.source().base_url == PRIVATE
        assert installation.source_status().revision > before.revision
        # The selected release still verifies under the restored trust.
        selection = host.manager.controllers["managed.private"].selector.current()
        assert selection is not None
        async with installation.installer.locked_generation(selection.runtime_digest):
            pass
        # A later store renewal leaves it alone.
        assert (await host.refresh()).followed == ()
        assert host.trust("managed.private") == restored


async def test_the_node_renews_store_trust_hourly_and_retries_busy_installations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The node states its offline mode; a deferred installation is retried soon."""
    from skulk.extensions import managed_services
    from skulk.extensions.managed_services import ManagedServices

    sent: list[object] = []
    replies: list[dict[str, JsonValue]] = [
        {"result": {"trust_revision": 2, "followed": [], "deferred": ["managed.x"]}},
        {"result": {"trust_revision": 2, "followed": ["managed.x"], "deferred": []}},
    ]

    async def request(root: Path, sent_request: object) -> dict[str, JsonValue]:
        del root
        sent.append(sent_request)
        return replies.pop(0)

    monkeypatch.setattr(managed_services, "manager_request", request)
    monkeypatch.setattr(managed_services, "offline_mode", lambda: True)
    services = ManagedServices(tmp_path / "connection.json")
    services._schedule_store_trust(tmp_path)  # pyright: ignore[reportPrivateUsage]
    await services._settle_store_trust()  # pyright: ignore[reportPrivateUsage]
    assert sent == [StoreTrustRequest(offline=True)]
    due = services.store_trust_due
    assert due is not None and due - time.monotonic() <= 300
    # Not due yet: nothing is sent.
    services._schedule_store_trust(tmp_path)  # pyright: ignore[reportPrivateUsage]
    assert len(sent) == 1
    services.store_trust_due = time.monotonic()
    services._schedule_store_trust(tmp_path)  # pyright: ignore[reportPrivateUsage]
    await services._settle_store_trust()  # pyright: ignore[reportPrivateUsage]
    assert len(sent) == 2
    due = services.store_trust_due
    assert due is not None and due - time.monotonic() > 3000
