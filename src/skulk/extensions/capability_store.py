"""The built-in capability store's publisher trust, anchored and renewed through TUF.

Skulk ships the store's TUF root beside this module. The store's repository
signs one target, the publisher trust the store vouches for: the publishers
it lists with their Ed25519 release keys, the publishers and artifacts it
revokes, a revision and an expiry. A host verifies that target from the
embedded root, replaying the root rotations it has already verified, and
keeps the last verified copy bound to its digest. A host that cannot reach the
store, or that runs offline, keeps reading with the trust it last verified
until that trust's own expiry.

The trust never wraps the catalog. The catalog and every release keep their
own publisher signatures, revision floors and expiry; this module only decides
which publishers the built-in source trusts for discovery.

A build without the embedded root has no built-in store: the client reports
itself unavailable and the host behaves as it did before the store existed.
"""

import contextlib
import hashlib
import time
from pathlib import Path
from typing import Final, final

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field
from tuf.ngclient.fetcher import FetcherInterface
from tuf.ngclient.updater import Updater
from tuf.ngclient.urllib3_fetcher import Urllib3Fetcher

from skulk.extensions.runtime_artifacts import Digest, RuntimeTrust
from skulk.extensions.runtime_files import (
    private_directory,
    read_private,
    write_private,
)

CAPABILITY_STORE_TRUST_URL: Final = "https://skulkapps.foxlight.ai/trust/"
"""TUF repository of the built-in store; metadata and targets live beneath it."""

CAPABILITY_STORE_CATALOG_URL: Final = "https://skulkapps.foxlight.ai/catalog/"
"""Directory the built-in store serves its signed catalog from."""

CAPABILITY_STORE_CATALOG_DOCUMENT: Final = "catalog.json"
"""Basename of the built-in store's signed catalog."""

CAPABILITY_STORE_TRUST_TARGET: Final = "v1/publisher-trust.json"
"""The one TUF target the store signs: its publisher trust document."""

EMBEDDED_CAPABILITY_STORE_ROOT: Final = Path(__file__).with_name(
    "capability_store_root.json"
)
"""The store's TUF trust root, shipped as package data beside this module.

The plugin manager runs from a staged copy of the Skulk package, which copies
every file under the package, so the root travels with the manager. Absent,
the built-in store is inactive."""

_TRUST_BYTES: Final = 65536
"""Upper bound on the signed trust document; a real one is a few kilobytes."""

_ROOT_BYTES: Final = 512000
"""Upper bound on the embedded root, the updater's own root bound."""

_REFRESH_LOCK_SECONDS: Final = 20.0
"""How long a load waits for an earlier refresh still running in its thread."""


class StoreTrustUnavailableError(ValueError):
    """No verified, unexpired store trust is available to this host.

    Raised when the build ships no store root, or when neither a refresh nor
    the last verified copy yields trust that is still in force. Carries no
    address or document: callers name the refusal from a fixed vocabulary.
    """


class _RetainedTrust(BaseModel):
    """The last store trust this host verified, bound to the bytes it verified."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    document: str = Field(
        max_length=_TRUST_BYTES,
        description="The trust target exactly as TUF verified it.",
    )
    sha256: Digest = Field(description="Digest of the verified document bytes.")
    revision: int = Field(ge=1, description="Trust revision the document carries.")
    verified_at: int = Field(gt=0, description="Unix time the document was verified.")


@final
class StoreTrustClient:
    """Verify, renew and retain the built-in store's publisher trust.

    The pattern is the model registry's: an explicit bootstrap from the
    embedded root (so a cached ``root.json`` is never trusted on its own),
    python-tuf replaying its verified ``root_history`` before the network,
    and a hash-bound last-known-good copy. The retained copy is also the
    store trust's revision floor: a verified target older than it, or a
    different document at its revision, is refused and the retained copy is
    kept.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        embedded_root: Path = EMBEDDED_CAPABILITY_STORE_ROOT,
        base_url: str = CAPABILITY_STORE_TRUST_URL,
        catalog_url: str = CAPABILITY_STORE_CATALOG_URL,
        catalog_document: str = CAPABILITY_STORE_CATALOG_DOCUMENT,
        fetcher: FetcherInterface | None = None,
        socket_timeout_seconds: int = 4,
    ) -> None:
        """Configure the store without touching the network or the disk.

        Args:
            cache_dir: Owner-only directory for TUF metadata, the downloaded
                target and the retained trust; created on first load.
            embedded_root: The store's TUF root shipped with this build.
            base_url: The store's TUF repository directory.
            catalog_url: Directory the store serves its signed catalog from.
            catalog_document: Basename of the store's signed catalog.
            fetcher: Network access for TUF, injectable for tests; defaults
                to python-tuf's urllib3 fetcher.
            socket_timeout_seconds: Connect and between-bytes timeout of the
                default fetcher.
        """
        self.cache_dir = cache_dir
        self.embedded_root = embedded_root
        self.base_url = base_url.rstrip("/") + "/"
        self.catalog_url = catalog_url
        self.catalog_document = catalog_document
        self.fetcher = fetcher
        self.socket_timeout_seconds = socket_timeout_seconds
        self.retained_path = cache_dir / "last-known-good-trust.json"

    @property
    def available(self) -> bool:
        """Whether this build ships the store's root, so the store can be used."""
        return self.embedded_root.is_file()

    def load(self, *, offline: bool, now: int | None = None) -> RuntimeTrust:
        """The newest verified store trust still in force.

        Online, the store's TUF repository is refreshed first; a newer
        verified trust replaces the retained copy. Any refresh failure (no
        network, an expired or unsigned role, a tampered target, a trust
        that is expired, older than the retained one or different at its
        revision) leaves the retained copy in place and it is used instead.
        Offline, the retained copy is used without network access.

        Args:
            offline: Use only the retained copy; never reach the network.
            now: Unix time for expiry checks; the current time by default.

        Returns:
            Verified publisher trust whose expiry is still ahead.

        Raises:
            StoreTrustUnavailableError: The build ships no store root, or no
                verified trust in force is available.

        Side effects:
            Writes TUF metadata, the downloaded target and the retained copy
            under ``cache_dir``.
        """
        current = now if now is not None else int(time.time())
        if not self.available:
            raise StoreTrustUnavailableError("this build ships no store root")
        try:
            private_directory(self.cache_dir)
            with FileLock(
                str(self.cache_dir / "refresh.lock"), timeout=_REFRESH_LOCK_SECONDS
            ):
                if not offline:
                    # Security fallback boundary: whatever the refresh refused,
                    # nothing it fetched was used. The retained copy decides,
                    # and its absence or expiry is the refusal.
                    with contextlib.suppress(Exception):
                        return self._refresh(current)
                return self._last_known_good(current)
        except (OSError, ValueError, TimeoutError) as error:
            raise StoreTrustUnavailableError(
                "no verified store trust in force"
            ) from error

    def last_known_good(self, *, now: int | None = None) -> RuntimeTrust:
        """The retained trust without any network access.

        Args:
            now: Unix time for the expiry check; the current time by default.

        Returns:
            The retained trust when it is intact and still in force.

        Raises:
            StoreTrustUnavailableError: Nothing is retained, the copy no
                longer matches its digest, or it has expired.
        """
        try:
            return self._last_known_good(now if now is not None else int(time.time()))
        except (OSError, ValueError) as error:
            raise StoreTrustUnavailableError(
                "no verified store trust in force"
            ) from error

    def _refresh(self, now: int) -> RuntimeTrust:
        """Refresh TUF metadata, then verify and retain the trust target."""
        metadata = self.cache_dir / "metadata"
        targets = self.cache_dir / "targets"
        private_directory(metadata)
        private_directory(targets)
        with self.embedded_root.open("rb") as source:
            root = source.read(_ROOT_BYTES + 1)
        if len(root) > _ROOT_BYTES:
            raise ValueError("embedded store root exceeds bound")
        updater = Updater(
            metadata_dir=str(metadata),
            metadata_base_url=f"{self.base_url}metadata/",
            target_dir=str(targets),
            target_base_url=f"{self.base_url}targets/",
            fetcher=self.fetcher
            if self.fetcher is not None
            else Urllib3Fetcher(
                socket_timeout=self.socket_timeout_seconds,
                app_user_agent="Skulk capability-store client",
            ),
            # An explicit bootstrap ignores an arbitrary cached root.json;
            # python-tuf then replays its verified local root_history before
            # consulting the network, so legitimate rotations persist.
            bootstrap=root,
        )
        updater.refresh()
        target = updater.get_targetinfo(CAPABILITY_STORE_TRUST_TARGET)
        if target is None:
            raise ValueError("store repository signs no publisher trust")
        if target.length > _TRUST_BYTES:
            raise ValueError("store publisher trust exceeds bound")
        # download_target checks the signed length and hashes before the path
        # is returned, so a tampered target never reaches the parser.
        payload = Path(updater.download_target(target)).read_bytes()
        trust = RuntimeTrust.model_validate_json(payload)
        if now >= trust.expires_at:
            raise ValueError("store publisher trust has expired")
        digest = hashlib.sha256(payload).hexdigest()
        retained = self._retained_or_none()
        if retained is not None and (
            trust.revision < retained.revision
            or (trust.revision == retained.revision and digest != retained.sha256)
        ):
            raise ValueError("store publisher trust rollback refused")
        if retained is None or retained.sha256 != digest:
            write_private(
                self.retained_path,
                _RetainedTrust(
                    document=payload.decode("utf-8"),
                    sha256=digest,
                    revision=trust.revision,
                    verified_at=now,
                )
                .model_dump_json()
                .encode(),
            )
        return trust

    def _retained(self) -> _RetainedTrust:
        """The retained record, checked against its digest; raises when unusable."""
        record = _RetainedTrust.model_validate_json(
            read_private(self.retained_path, _TRUST_BYTES * 2 + 4096)
        )
        if hashlib.sha256(record.document.encode()).hexdigest() != record.sha256:
            raise ValueError("retained store trust differs from its digest")
        return record

    def _retained_or_none(self) -> _RetainedTrust | None:
        """The retained record, or ``None`` when absent or unusable.

        An unusable record is local damage, not a remote rollback: it must not
        stop the store from renewing, so a refresh treats it as first use.
        """
        try:
            return self._retained()
        except (OSError, ValueError):
            return None

    def _last_known_good(self, now: int) -> RuntimeTrust:
        """Read the retained trust and require it to still be in force."""
        record = self._retained()
        trust = RuntimeTrust.model_validate_json(record.document)
        if trust.revision != record.revision:
            raise ValueError("retained store trust revision differs")
        if now >= trust.expires_at:
            raise ValueError("retained store trust has expired")
        return trust
