"""Engine trust: a separate signed authority, read only to decide an installation."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from skulk.extensions.capability_store import StoreTrustClient
from skulk.extensions.tests.store_repository import StoreRepository, trust_document
from skulk.provisioning.engine_packages import (
    EnginePackageRefusedError,
    verify_engine_manifest,
)
from skulk.provisioning.engine_trust import (
    ENGINE_TRUST_DIR,
    ENGINE_TRUST_TARGET,
    engine_trust_client,
    load_engine_trust,
)
from skulk.provisioning.tests.engine_package_fixtures import (
    PUBLISHER,
    engine_claims,
    sign_engine_claims,
)


class _Node:
    """One node on the store: the repository, its engine client and a capability one."""

    def __init__(self, root: Path) -> None:
        self.repository = StoreRepository()
        embedded = root / "capability_store_root.json"
        embedded.write_bytes(self.repository.embedded_root)
        self.engine_key = Ed25519PrivateKey.generate()
        self.capability_key = Ed25519PrivateKey.generate()
        self.engines = engine_trust_client(
            root / "engine-trust",
            embedded_root=embedded,
            fetcher=self.repository.fetcher,
        )
        self.capabilities = StoreTrustClient(
            root / "store-trust",
            embedded_root=embedded,
            fetcher=self.repository.fetcher,
        )

    def publish_engine_trust(self, revision: int, *, expires_in: int = 86400) -> None:
        self.repository.publish_trust(
            trust_document(
                self.engine_key,
                revision=revision,
                publisher=PUBLISHER,
                expires_in=expires_in,
            ),
            trust_target=ENGINE_TRUST_TARGET,
        )

    def publish_capability_trust(self, revision: int) -> None:
        # The same publisher name, on purpose: only the key and the target differ.
        self.repository.publish_trust(
            trust_document(self.capability_key, revision=revision, publisher=PUBLISHER)
        )


def _manifest(key: Ed25519PrivateKey) -> tuple[bytes, int]:
    """A signed manifest valid now, and the time to verify it at."""
    now = int(time.time())
    claims = engine_claims()
    claims.update(issued_at=now - 60, expires_at=now + 3600)
    return sign_engine_claims(key, claims), now


def _refused(node: _Node, *, offline: bool, now: int | None = None) -> str:
    with pytest.raises(EnginePackageRefusedError) as caught:
        load_engine_trust(offline=offline, now=now, client=node.engines)
    return caught.value.code


def test_the_default_client_reads_the_engine_target_in_its_own_directory() -> None:
    client = engine_trust_client()

    assert client.trust_target == ENGINE_TRUST_TARGET
    assert client.cache_dir == ENGINE_TRUST_DIR
    assert not client.retained_path.name.startswith("last-known-good-trust")


def test_engine_trust_admits_a_manifest_signed_by_an_engine_publisher(
    tmp_path: Path,
) -> None:
    node = _Node(tmp_path)
    node.publish_capability_trust(4)
    node.publish_engine_trust(1)
    document, now = _manifest(node.engine_key)

    trust = load_engine_trust(offline=False, client=node.engines)
    package = verify_engine_manifest(document, trust, now=now)

    assert trust.revision == 1
    assert package.claims.engine_id == "foxlight.tensorfold"


def test_a_capability_publisher_key_cannot_sign_an_engine(tmp_path: Path) -> None:
    node = _Node(tmp_path)
    node.publish_capability_trust(4)
    node.publish_engine_trust(1)
    document, now = _manifest(node.capability_key)

    trust = load_engine_trust(offline=False, client=node.engines)
    with pytest.raises(EnginePackageRefusedError) as caught:
        verify_engine_manifest(document, trust, now=now)

    # Same publisher name in both documents; the engine document holds a
    # different key, so the capability key's signature does not verify.
    assert caught.value.code == "signature_refused"


def test_a_store_that_signs_no_engine_trust_never_lends_capability_trust(
    tmp_path: Path,
) -> None:
    node = _Node(tmp_path)
    node.publish_capability_trust(4)
    assert node.capabilities.load(offline=False).revision == 4

    assert _refused(node, offline=False) == "trust_unavailable"
    assert _refused(node, offline=True) == "trust_unavailable"


def test_a_client_for_another_target_is_refused_as_engine_trust(
    tmp_path: Path,
) -> None:
    node = _Node(tmp_path)
    node.publish_capability_trust(4)

    with pytest.raises(EnginePackageRefusedError) as caught:
        load_engine_trust(offline=False, client=node.capabilities)

    assert caught.value.code == "trust_unavailable"


def test_retained_engine_trust_serves_an_unreachable_or_offline_node(
    tmp_path: Path,
) -> None:
    node = _Node(tmp_path)
    node.publish_engine_trust(2)
    assert load_engine_trust(offline=False, client=node.engines).revision == 2

    node.repository.fetcher.reachable = False

    assert load_engine_trust(offline=False, client=node.engines).revision == 2
    assert load_engine_trust(offline=True, client=node.engines).revision == 2


def test_a_node_that_never_reached_the_store_has_no_engine_trust(
    tmp_path: Path,
) -> None:
    node = _Node(tmp_path)
    node.publish_engine_trust(1)
    node.repository.fetcher.reachable = False

    assert _refused(node, offline=False) == "trust_unavailable"
    assert _refused(node, offline=True) == "trust_unavailable"


def test_expired_engine_trust_refuses_a_new_installation(tmp_path: Path) -> None:
    node = _Node(tmp_path)
    node.publish_engine_trust(1, expires_in=600)
    assert load_engine_trust(offline=False, client=node.engines).revision == 1
    after_expiry = int(time.time()) + 601

    assert _refused(node, offline=False, now=after_expiry) == "trust_unavailable"
    assert _refused(node, offline=True, now=after_expiry) == "trust_unavailable"


def test_an_older_engine_trust_never_replaces_the_retained_one(tmp_path: Path) -> None:
    node = _Node(tmp_path)
    node.publish_capability_trust(9)
    node.publish_engine_trust(3)
    assert load_engine_trust(offline=False, client=node.engines).revision == 3

    node.publish_engine_trust(2)

    assert load_engine_trust(offline=False, client=node.engines).revision == 3
