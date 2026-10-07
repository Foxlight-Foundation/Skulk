"""A throwaway TUF repository for the built-in capability store, served from memory.

Built with python-tuf's metadata API and software keys, laid out as a real
store would be (consistent snapshots, hash-prefixed targets), and served to the
store client through an injected fetcher, so tests exercise the real updater
end to end without any network.
"""

import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import final

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

# These names are public API that their packages re-export without listing
# them in __all__, which strict import checking cannot tell apart.
from securesystemslib.signer import (
    CryptoSigner,  # pyright: ignore[reportPrivateImportUsage]
)
from tuf.api.exceptions import DownloadError, DownloadHTTPError
from tuf.api.metadata import (
    Metadata,
    MetaFile,  # pyright: ignore[reportPrivateImportUsage]
    Root,  # pyright: ignore[reportPrivateImportUsage]
    Snapshot,  # pyright: ignore[reportPrivateImportUsage]
    TargetFile,  # pyright: ignore[reportPrivateImportUsage]
    Targets,  # pyright: ignore[reportPrivateImportUsage]
    Timestamp,  # pyright: ignore[reportPrivateImportUsage]
)
from tuf.ngclient.fetcher import FetcherInterface

from skulk.extensions.capability_store import (
    CAPABILITY_STORE_TRUST_TARGET,
    CAPABILITY_STORE_TRUST_URL,
)
from skulk.extensions.runtime_artifacts import RuntimeTrust


def _expiry(days: int) -> datetime:
    return datetime.now(UTC).replace(microsecond=0) + timedelta(days=days)


@final
class MemoryFetcher(FetcherInterface):
    """Serve repository files by URL, record each request and simulate an outage."""

    def __init__(self, files: dict[str, bytes]) -> None:
        """Serve ``files``, keyed by absolute URL."""
        self.files = files
        self.reachable = True
        self.requests: list[str] = []

    def _fetch(self, url: str) -> Iterator[bytes]:
        self.requests.append(url)
        if not self.reachable:
            raise DownloadError("store unreachable")
        data = self.files.get(url)
        if data is None:
            raise DownloadHTTPError("not found", 404)
        return iter((data,))


@final
class StoreRepository:
    """The store's TUF repository: root, online roles and the one trust target.

    The root starts on an ECDSA P-256 key, as a hardware key holds it; the
    online roles use Ed25519. ``rotate_keys`` publishes a root version that
    swaps both the root and the targets key types, so both are verified.
    """

    def __init__(self, base_url: str = CAPABILITY_STORE_TRUST_URL) -> None:
        """Create and serve root version 1; no trust is published yet."""
        self.base_url = base_url
        self.files: dict[str, bytes] = {}
        self.fetcher = MemoryFetcher(self.files)
        self.root_signer = CryptoSigner.generate_ecdsa()
        self.targets_signer = CryptoSigner.generate_ed25519()
        self.snapshot_signer = CryptoSigner.generate_ed25519()
        self.timestamp_signer = CryptoSigner.generate_ed25519()
        root = Root(expires=_expiry(365))
        root.add_key(self.root_signer.public_key, "root")
        root.add_key(self.targets_signer.public_key, "targets")
        root.add_key(self.snapshot_signer.public_key, "snapshot")
        root.add_key(self.timestamp_signer.public_key, "timestamp")
        self.root = Metadata(root)
        self.root.sign(self.root_signer)
        # Root version 1, as a Skulk build would ship it.
        self.embedded_root = self.root.to_bytes()
        self.files[self._metadata_url("1.root.json")] = self.embedded_root
        self.targets = Metadata(Targets(expires=_expiry(30)))
        self.snapshot = Metadata(Snapshot(expires=_expiry(7)))
        self.timestamp = Metadata(Timestamp(expires=_expiry(1)))
        self.publications = 0
        self.timestamps = 0

    def _metadata_url(self, name: str) -> str:
        return f"{self.base_url}metadata/{name}"

    def publish_trust(self, document: bytes, *, served: bytes | None = None) -> None:
        """Sign ``document`` as the trust target under new role versions.

        Args:
            document: The trust bytes the targets role signs.
            served: Bytes served at the target's address instead, to model a
                tampered mirror; the signed document by default.
        """
        self.publications += 1
        target = TargetFile.from_data(CAPABILITY_STORE_TRUST_TARGET, document)
        self.targets.signed.version = self.publications
        self.targets.signed.targets = {CAPABILITY_STORE_TRUST_TARGET: target}
        self.targets.sign(self.targets_signer)
        self.files[self._metadata_url(f"{self.publications}.targets.json")] = (
            self.targets.to_bytes()
        )
        self.snapshot.signed.version = self.publications
        self.snapshot.signed.meta["targets.json"] = MetaFile(version=self.publications)
        self.snapshot.sign(self.snapshot_signer)
        self.files[self._metadata_url(f"{self.publications}.snapshot.json")] = (
            self.snapshot.to_bytes()
        )
        directory, _, name = CAPABILITY_STORE_TRUST_TARGET.rpartition("/")
        self.files[
            f"{self.base_url}targets/{directory}/{target.hashes['sha256']}.{name}"
        ] = served if served is not None else document
        self.renew_timestamp()

    def renew_timestamp(self) -> None:
        """Re-sign the timestamp at a new version, as the scheduled job does."""
        self.timestamps += 1
        self.timestamp.signed.version = self.timestamps
        self.timestamp.signed.expires = _expiry(1)
        self.timestamp.signed.snapshot_meta = MetaFile(version=self.publications)
        self.timestamp.sign(self.timestamp_signer)
        self.files[self._metadata_url("timestamp.json")] = self.timestamp.to_bytes()

    def rotate_keys(self) -> None:
        """Publish the next root version with a new root key and a new targets key.

        The new root is signed by the old root key and the new one, as a key
        ceremony does. The root moves to Ed25519 and targets to ECDSA P-256,
        the reverse of where they started. Trust must be published again
        under the new targets key.
        """
        root = self.root.signed
        old_root = self.root_signer
        self.root_signer = CryptoSigner.generate_ed25519()
        old_targets = self.targets_signer
        self.targets_signer = CryptoSigner.generate_ecdsa()
        root.revoke_key(old_root.public_key.keyid, "root")
        root.add_key(self.root_signer.public_key, "root")
        root.revoke_key(old_targets.public_key.keyid, "targets")
        root.add_key(self.targets_signer.public_key, "targets")
        root.version += 1
        self.root.signatures.clear()
        self.root.sign(old_root, append=True)
        self.root.sign(self.root_signer, append=True)
        self.files[self._metadata_url(f"{root.version}.root.json")] = (
            self.root.to_bytes()
        )


def trust_document(
    key: Ed25519PrivateKey,
    *,
    revision: int,
    publisher: str = "fixture",
    expires_in: int = 86400,
    revoked_publishers: tuple[str, ...] = (),
    revoked_artifacts: tuple[str, ...] = (),
) -> bytes:
    """A publisher trust document as the store's targets role signs it."""
    return (
        RuntimeTrust(
            revision=revision,
            expires_at=int(time.time()) + expires_in,
            publishers={publisher: key.public_key().public_bytes_raw().hex()},
            revoked_publishers=revoked_publishers,
            revoked_artifacts=revoked_artifacts,
        )
        .model_dump_json()
        .encode()
    )
