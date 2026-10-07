"""The built-in capability store: trust anchored in an embedded TUF root, renewed by itself."""

import importlib.machinery
import importlib.util
import json
import sys
import sysconfig
import time
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import JsonValue, SecretStr, TypeAdapter
from tuf.api.metadata import (
    Metadata,
    Root,  # pyright: ignore[reportPrivateImportUsage]
)

from skulk.extensions import service_snapshot
from skulk.extensions.capability_store import (
    CAPABILITY_STORE_CATALOG_DOCUMENT,
    CAPABILITY_STORE_CATALOG_URL,
    EMBEDDED_CAPABILITY_STORE_ROOT,
    StoreTrustClient,
    StoreTrustUnavailableError,
)
from skulk.extensions.runtime_artifacts import RuntimeTrust, canonical_json
from skulk.extensions.runtime_catalog import (
    CatalogRefusal,
    CatalogRefusedError,
    CatalogSourceUpdate,
    HostCatalog,
)
from skulk.extensions.runtime_files import private_directory
from skulk.extensions.tests.store_repository import StoreRepository, trust_document

STORE_CATALOG = CAPABILITY_STORE_CATALOG_URL + CAPABILITY_STORE_CATALOG_DOCUMENT


def _catalog(
    key: Ed25519PrivateKey, *, revision: int = 1, publisher: str = "fixture"
) -> bytes:
    """A protocol 2 catalog listing one release, valid for a month."""
    now = int(time.time())
    catalog: dict[str, JsonValue] = {
        "protocol": 2,
        "publisher": publisher,
        "revision": revision,
        "created_at": now - 1,
        "expires_at": now + 30 * 86400,
        "entries": [
            {
                "bundle_id": "example.studio",
                "bundle_version": "1.0.0",
                "title": "Example Studio",
                "publisher": publisher,
                "sequence": 1,
                "feed_url": "https://skulkapps.foxlight.ai/releases/example/1/",
                "release_sha256": "1" * 64,
                "release_digest": "2" * 64,
                "artifact_sha256": "3" * 64,
                "artifact_size": 4096,
                "transfer_bytes": 4096,
                "platforms": ["darwin", "linux"],
                "skulk_build_sha256": "a" * 64,
                "skulk_requires": ">=2.0.1,<3",
                "permissions": ["local synthetic operation"],
                "descriptors": ["example.render@1.0.0"],
                "expires_at": now + 30 * 86400,
            }
        ],
    }
    signature = key.sign(canonical_json(catalog)).hex()
    return json.dumps({"catalog": catalog, "signature": signature}).encode()


class _Store:
    """One host on the store: its repository, client, catalog and served documents."""

    def __init__(self, root: Path) -> None:
        self.repository = StoreRepository()
        self.publisher = Ed25519PrivateKey.generate()
        manager = root / "manager"
        private_directory(manager)
        embedded = root / "capability_store_root.json"
        embedded.write_bytes(self.repository.embedded_root)
        self.client = StoreTrustClient(
            manager / "store-trust",
            embedded_root=embedded,
            fetcher=self.repository.fetcher,
        )
        self.original = _catalog(self.publisher)
        self.served = [self.original]
        self.catalog_requests: list[httpx.Request] = []
        self.catalog = HostCatalog(
            manager, transport=httpx.MockTransport(self.respond), store=self.client
        )

    def respond(self, request: httpx.Request) -> httpx.Response:
        self.catalog_requests.append(request)
        if str(request.url) == STORE_CATALOG or request.url.host == (
            "private.example.test"
        ):
            return httpx.Response(200, content=self.served[0])
        return httpx.Response(404)

    def publish(
        self,
        revision: int,
        *,
        expires_in: int = 86400,
        revoked_publishers: tuple[str, ...] = (),
        revoked_artifacts: tuple[str, ...] = (),
    ) -> None:
        """Sign a trust revision naming this store's publisher key."""
        self.repository.publish_trust(
            trust_document(
                self.publisher,
                revision=revision,
                expires_in=expires_in,
                revoked_publishers=revoked_publishers,
                revoked_artifacts=revoked_artifacts,
            )
        )

    def trust_revision(self) -> int | None:
        return self.catalog.source_status().trust_revision

    async def refused(
        self, *, now: int | None = None, offline: bool = False
    ) -> CatalogRefusal:
        with pytest.raises(CatalogRefusedError) as refused:
            await self.catalog.fetch(now=now, offline=offline)
        return refused.value.code


async def test_first_use_lists_the_store_with_nothing_to_configure(
    tmp_path: Path,
) -> None:
    """A fresh host reads the store from the embedded root, anonymously."""
    store = _Store(tmp_path)
    status = store.catalog.source_status()
    assert status.builtin_store and status.builtin_store_available
    assert status.configured and status.credential_ready
    assert (status.revision, status.trust_revision) == (0, None)
    store.publish(1)
    verified = await store.catalog.fetch()
    assert verified.claims.publisher == "fixture"
    assert [entry.bundle_id for entry in verified.entries] == ["example.studio"]
    assert [str(request.url) for request in store.catalog_requests] == [STORE_CATALOG]
    assert "authorization" not in store.catalog_requests[0].headers
    status = store.catalog.source_status()
    assert status.builtin_store and status.configured
    assert (status.revision, status.trust_revision) == (1, 1)
    assert store.catalog.trust().publishers == {
        "fixture": store.publisher.public_key().public_bytes_raw().hex()
    }
    # Every existing floor applies to the seeded source unchanged.
    store.served[0] = _catalog(store.publisher, revision=2)
    assert (await store.catalog.fetch()).claims.revision == 2
    store.served[0] = store.original
    assert await store.refused() == "catalog_rollback_refused"


async def test_a_renewed_trust_is_applied_and_an_older_one_never(
    tmp_path: Path,
) -> None:
    """The store renews trust by itself; a rollback through a valid TUF update is refused."""
    store = _Store(tmp_path)
    store.publish(1)
    await store.catalog.fetch()
    second = Ed25519PrivateKey.generate()
    store.repository.publish_trust(
        RuntimeTrust(
            revision=2,
            expires_at=int(time.time()) + 7 * 86400,
            publishers={
                "fixture": store.publisher.public_key().public_bytes_raw().hex(),
                "second": second.public_key().public_bytes_raw().hex(),
            },
        )
        .model_dump_json()
        .encode()
    )
    await store.catalog.fetch()
    assert store.trust_revision() == 2
    assert set(store.catalog.trust().publishers) == {"fixture", "second"}
    # A renewed timestamp alone changes nothing and is accepted.
    store.repository.renew_timestamp()
    await store.catalog.fetch()
    assert store.trust_revision() == 2
    # An older revision, signed under a newer targets version, is refused;
    # so is a different document at the revision already held.
    store.publish(1)
    await store.catalog.fetch()
    assert store.trust_revision() == 2
    store.publish(2, expires_in=3 * 86400)
    await store.catalog.fetch()
    assert store.trust_revision() == 2
    assert set(store.catalog.trust().publishers) == {"fixture", "second"}
    assert store.client.last_known_good().revision == 2
    store.publish(3)
    await store.catalog.fetch()
    assert store.trust_revision() == 3
    assert set(store.catalog.trust().publishers) == {"fixture"}


async def test_a_key_rotation_through_a_new_root_is_accepted(tmp_path: Path) -> None:
    """A new root version replaces root and targets keys; hosts follow it unaided."""
    store = _Store(tmp_path)
    repository = store.repository
    store.publish(1)
    await store.catalog.fetch()
    retired = repository.targets_signer
    repository.rotate_keys()
    current = repository.targets_signer
    store.publish(2)
    await store.catalog.fetch()
    assert store.trust_revision() == 2
    # The verified rotation is replayed from the host's own root history, so
    # the store need not keep serving it.
    rotated_url = repository.base_url + "metadata/2.root.json"
    rotated = repository.files.pop(rotated_url)
    store.publish(3)
    await store.catalog.fetch()
    assert store.trust_revision() == 3
    # The retired targets key no longer signs anything this host accepts.
    repository.targets_signer = retired
    store.publish(4)
    await store.catalog.fetch()
    assert store.trust_revision() == 3
    # A host that first starts after the rotation follows it from the root
    # its build shipped.
    repository.files[rotated_url] = rotated
    repository.targets_signer = current
    store.publish(5)
    fresh = StoreTrustClient(
        tmp_path / "fresh",
        embedded_root=store.client.embedded_root,
        fetcher=repository.fetcher,
    )
    assert fresh.load(offline=False).revision == 5


async def test_an_expired_trust_is_refused(tmp_path: Path) -> None:
    """Expired trust is never applied, and held trust that lapses must be renewed."""
    store = _Store(tmp_path)
    store.publish(1, expires_in=-60)
    assert await store.refused() == "catalog_store_trust_unavailable"
    assert store.trust_revision() is None
    store.publish(2, expires_in=3600)
    await store.catalog.fetch()
    assert store.trust_revision() == 2
    store.publish(3, expires_in=-60)
    await store.catalog.fetch()
    assert store.trust_revision() == 2
    # Two hours on, the held trust has lapsed. Without the store the host
    # refuses by name; once the store signs current trust, it is applied.
    later = int(time.time()) + 7200
    store.repository.fetcher.reachable = False
    assert await store.refused(now=later) == "catalog_store_trust_unavailable"
    store.repository.fetcher.reachable = True
    store.publish(4)
    await store.catalog.fetch(now=later)
    assert store.trust_revision() == 4


async def test_the_stores_current_revocations_are_what_applies(
    tmp_path: Path,
) -> None:
    """A revocation applies at once; a later document that drops it restores nothing held."""
    store = _Store(tmp_path)
    store.publish(1)
    await store.catalog.fetch()
    store.publish(2, revoked_artifacts=("2" * 64,))
    assert (await store.catalog.fetch()).entries == ()
    store.publish(3, revoked_publishers=("fixture",))
    assert await store.refused() == "catalog_trust_refused"
    # The store withdrew a mistaken revocation: its current document is the
    # whole truth, so the publisher and the artifact are offered again.
    store.publish(4)
    assert [entry.sequence for entry in (await store.catalog.fetch()).entries] == [1]
    trust = store.catalog.trust()
    assert trust.revision == 4
    assert (trust.revoked_publishers, trust.revoked_artifacts) == ((), ())


async def test_store_revocations_past_the_trust_bounds_never_accumulate(
    tmp_path: Path,
) -> None:
    """Two revisions whose revocations together exceed a bound still renew cleanly."""
    store = _Store(tmp_path)
    first = tuple(f"{index:064x}" for index in range(100))
    second = tuple(f"{index:064x}" for index in range(100, 140))
    store.publish(1, revoked_artifacts=first)
    await store.catalog.fetch()
    assert len(store.catalog.trust().revoked_artifacts) == 100
    store.publish(2, revoked_artifacts=second)
    await store.catalog.fetch()
    trust = store.catalog.trust()
    assert trust.revision == 2
    assert trust.revoked_artifacts == second


async def test_revocations_never_cross_between_the_store_and_a_private_catalog(
    tmp_path: Path,
) -> None:
    """A publisher name means nothing across sources; neither side's revocations follow."""
    store = _Store(tmp_path)
    store.publish(1, revoked_publishers=("fixture",))
    assert await store.refused() == "catalog_trust_refused"
    private_key = Ed25519PrivateKey.generate()
    private_public = private_key.public_key().public_bytes_raw().hex()
    # The store revoked "fixture"; a private catalog's own "fixture" is not.
    await store.catalog.configure(
        CatalogSourceUpdate(
            expected_revision=1,
            base_url="https://private.example.test/",
            trust=RuntimeTrust(
                revision=2,
                expires_at=int(time.time()) + 3600,
                publishers={"fixture": private_public},
            ),
        )
    )
    assert store.catalog.trust().revoked_publishers == ()
    store.served[0] = _catalog(private_key)
    assert (await store.catalog.fetch()).claims.publisher == "fixture"
    # Within the private catalog's own history revocations still carry.
    await store.catalog.configure(
        CatalogSourceUpdate(
            expected_revision=2,
            trust=RuntimeTrust(
                revision=3,
                expires_at=int(time.time()) + 3600,
                publishers={"fixture": private_public},
                revoked_publishers=("fixture",),
            ),
        )
    )
    await store.catalog.configure(
        CatalogSourceUpdate(
            expected_revision=3,
            trust=RuntimeTrust(
                revision=4,
                expires_at=int(time.time()) + 3600,
                publishers={"fixture": private_public},
            ),
        )
    )
    assert store.catalog.trust().revoked_publishers == ("fixture",)
    # The private revocation of "fixture" does not follow the host back to
    # the store, whose current document no longer revokes it.
    store.publish(2)
    status = await store.catalog.use_builtin_store(4)
    assert status.builtin_store and status.trust_revision == 2
    assert store.catalog.trust().revoked_publishers == ()
    store.served[0] = _catalog(store.publisher)
    assert (await store.catalog.fetch()).claims.publisher == "fixture"


async def test_offline_uses_the_last_verified_trust(tmp_path: Path) -> None:
    """Offline, nothing reaches the store's repository; the retained copy serves."""
    store = _Store(tmp_path)
    store.publish(1)
    assert await store.refused(offline=True) == "catalog_store_trust_unavailable"
    assert store.repository.fetcher.requests == []
    await store.catalog.fetch()
    store.publish(2)
    requests = len(store.repository.fetcher.requests)
    await store.catalog.fetch(offline=True)
    assert store.repository.fetcher.requests[requests:] == []
    assert store.trust_revision() == 1
    assert store.client.load(offline=True).revision == 1
    # An unreachable store reads the same as offline: the held trust serves.
    store.repository.fetcher.reachable = False
    await store.catalog.fetch()
    assert store.trust_revision() == 1
    store.repository.fetcher.reachable = True
    await store.catalog.fetch()
    assert store.trust_revision() == 2


async def test_a_tampered_target_is_refused(tmp_path: Path) -> None:
    """Bytes that differ from the signed target are never parsed or retained."""
    store = _Store(tmp_path)
    forged = trust_document(Ed25519PrivateKey.generate(), revision=9)
    store.repository.publish_trust(
        trust_document(store.publisher, revision=1), served=forged
    )
    assert await store.refused() == "catalog_store_trust_unavailable"
    store.publish(1)
    await store.catalog.fetch()
    store.repository.publish_trust(
        trust_document(store.publisher, revision=2), served=forged
    )
    await store.catalog.fetch()
    assert store.trust_revision() == 1
    assert store.client.last_known_good().revision == 1
    # A retained copy edited on disk no longer matches its digest.
    record = TypeAdapter(dict[str, JsonValue])
    retained = record.validate_json(store.client.retained_path.read_bytes())
    retained["document"] = forged.decode()
    store.client.retained_path.write_bytes(record.dump_json(retained))
    with pytest.raises(StoreTrustUnavailableError):
        store.client.last_known_good()


async def test_a_private_catalog_replaces_the_store_and_the_owner_switches_back(
    tmp_path: Path,
) -> None:
    """One source at a time: the store by default, a private catalog on request."""
    store = _Store(tmp_path)
    store.publish(1)
    await store.catalog.fetch()
    private_key = Ed25519PrivateKey.generate()
    private_trust = RuntimeTrust(
        revision=5,
        expires_at=int(time.time()) + 3600,
        publishers={"fixture": private_key.public_key().public_bytes_raw().hex()},
    )

    async def configure_refused(update: CatalogSourceUpdate) -> CatalogRefusal:
        with pytest.raises(CatalogRefusedError) as refused:
            await store.catalog.configure(update)
        return refused.value.code

    # Leaving the store is a first setup: an address and trust of its own.
    assert (
        await configure_refused(
            CatalogSourceUpdate(expected_revision=1, trust=private_trust)
        )
        == "catalog_setup_incomplete"
    )
    assert (
        await configure_refused(
            CatalogSourceUpdate(
                expected_revision=1, base_url="https://private.example.test/"
            )
        )
        == "catalog_setup_incomplete"
    )
    # The store's own address is not a private catalog.
    assert (
        await configure_refused(
            CatalogSourceUpdate(
                expected_revision=1,
                base_url=CAPABILITY_STORE_CATALOG_URL,
                trust=private_trust,
            )
        )
        == "catalog_source_reserved"
    )
    status = await store.catalog.configure(
        CatalogSourceUpdate(
            expected_revision=1,
            base_url="https://private.example.test/",
            trust=private_trust,
            token=SecretStr("private-catalog-token"),
        )
    )
    assert not status.builtin_store and status.builtin_store_available
    assert (status.revision, status.trust_revision) == (2, 5)
    store.served[0] = _catalog(private_key)
    await store.catalog.fetch()
    assert store.catalog_requests[-1].headers["Authorization"] == (
        "Bearer private-catalog-token"
    )
    # A stale switch is refused; the current one returns to the store. The
    # private trust's higher revision does not bind the store's own history.
    with pytest.raises(CatalogRefusedError) as stale:
        await store.catalog.use_builtin_store(1)
    assert stale.value.code == "catalog_source_conflict"
    status = await store.catalog.use_builtin_store(2)
    assert status.builtin_store
    assert (status.revision, status.trust_revision) == (3, 1)
    assert await store.catalog.use_builtin_store(3) == status
    # The store's address keeps its own catalog history across the detour.
    store.served[0] = _catalog(store.publisher, revision=2)
    assert (await store.catalog.fetch()).claims.revision == 2
    assert "authorization" not in store.catalog_requests[-1].headers
    store.served[0] = store.original
    assert await store.refused() == "catalog_rollback_refused"


async def test_a_build_without_the_store_root_behaves_as_before(
    tmp_path: Path,
) -> None:
    """No embedded root: no built-in store, and its address is an ordinary one."""
    private_directory(tmp_path)
    client = StoreTrustClient(tmp_path / "store", embedded_root=tmp_path / "absent")
    catalog = HostCatalog(tmp_path, store=client)
    status = catalog.source_status()
    assert not (status.configured or status.builtin_store)
    assert not status.builtin_store_available
    with pytest.raises(CatalogRefusedError) as unconfigured:
        await catalog.fetch()
    assert unconfigured.value.code == "catalog_unconfigured"
    with pytest.raises(CatalogRefusedError) as unavailable:
        await catalog.use_builtin_store(0)
    assert unavailable.value.code == "catalog_store_unavailable"
    with pytest.raises(StoreTrustUnavailableError):
        client.load(offline=False)
    key = Ed25519PrivateKey.generate()
    status = await catalog.configure(
        CatalogSourceUpdate(
            expected_revision=0,
            base_url=CAPABILITY_STORE_CATALOG_URL,
            trust=RuntimeTrust(
                revision=1,
                expires_at=int(time.time()) + 3600,
                publishers={"fixture": key.public_key().public_bytes_raw().hex()},
            ),
        )
    )
    assert status.configured and not status.builtin_store
    assert not (tmp_path / "store").exists()


def test_the_shipped_store_root_is_absent_or_a_self_signed_root(
    tmp_path: Path,
) -> None:
    """Until the root ceremony the store is inactive; after it, the root verifies."""
    client = StoreTrustClient(tmp_path, embedded_root=EMBEDDED_CAPABILITY_STORE_ROOT)
    if not EMBEDDED_CAPABILITY_STORE_ROOT.exists():
        assert not client.available
        return
    root = Metadata[Root].from_bytes(EMBEDDED_CAPABILITY_STORE_ROOT.read_bytes())
    root.signed.verify_delegate("root", root.signed_bytes, root.signatures)
    assert client.available


def test_the_manager_snapshot_stages_package_data_beside_its_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store root ships as a package file; the manager's staged copy carries it."""
    site = tmp_path / "site-packages"
    extensions = site / "skulk" / "extensions"
    (extensions / "__pycache__").mkdir(parents=True)
    (site / "skulk" / "__init__.py").write_text("")
    (extensions / "capability_store_root.json").write_text("{}")
    (extensions / "__pycache__" / "capability_store.cpython-313.pyc").write_bytes(b"0")
    (site / "skulk_pyo3_bindings").mkdir()
    (site / "skulk_pyo3_bindings" / "__init__.py").write_text("")
    resources = tmp_path / "resources"
    resources.mkdir()

    def site_path(name: str) -> str:
        del name  # purelib and platlib are the same isolated directory
        return str(site)

    def package_spec(name: str) -> importlib.machinery.ModuleSpec:
        return importlib.machinery.ModuleSpec(
            name, None, origin=str(site / name / "__init__.py")
        )

    def resource_root() -> Path:
        return resources

    monkeypatch.setattr(sys, "prefix", str(tmp_path / "environment"))
    monkeypatch.setattr(sysconfig, "get_path", site_path)
    monkeypatch.setattr(importlib.util, "find_spec", package_spec)
    monkeypatch.setattr(service_snapshot, "find_resources", resource_root)
    staged = {
        item.relative
        for item in service_snapshot._sources()  # pyright: ignore[reportPrivateUsage]
    }
    version = sys.version_info
    site_prefix = f"lib/python{version.major}.{version.minor}/site-packages/"
    assert site_prefix + "skulk/extensions/capability_store_root.json" in staged
    assert not any(name.endswith(".pyc") for name in staged)
    assert EMBEDDED_CAPABILITY_STORE_ROOT.parent.name == "extensions"
