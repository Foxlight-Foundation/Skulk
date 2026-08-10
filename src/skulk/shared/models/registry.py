"""TUF-verified external model-card catalog with bounded offline fallback."""

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field
from tuf.ngclient.updater import Updater
from tuf.ngclient.urllib3_fetcher import Urllib3Fetcher

EMBEDDED_REGISTRY_ROOT = Path(__file__).with_name("model_registry_root.json")
"""Public TUF trust root shipped inside the Skulk Python package."""


class RegistryUnavailableError(RuntimeError):
    """Raised when neither the signed registry nor a safe local snapshot exists."""


class RegistryArtifact(BaseModel):
    """Exact upstream artifact identity asserted by a registry card."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    repository: str = Field(min_length=3, max_length=512)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    selected_file: str | None = Field(default=None, max_length=2048)
    format: str = Field(min_length=1, max_length=80)
    quantization: str = Field(max_length=120)


class RegistryCard(BaseModel):
    """One immutable registry envelope in a signed catalog target."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    schema_version: int = Field(ge=1)
    card_id: str = Field(pattern=r"^card_[a-z2-7]{52}$")
    alias: str = Field(min_length=1, max_length=512)
    model_ref: str = Field(min_length=1, max_length=512)
    artifact: RegistryArtifact
    card: dict[str, Any]


class RegistryCardMetadata(BaseModel):
    """Mutable catalog metadata that must not alter immutable card identity."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    provenance: Literal["foxlight", "agent", "community"]


class RegistryCatalog(BaseModel):
    """Complete published model-card catalog verified as one TUF target."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    schema_version: int = Field(ge=1)
    snapshot_id: str = Field(min_length=1, max_length=120)
    generated_at: datetime
    published_by: str = Field(min_length=1, max_length=320)
    note: str
    cards: tuple[RegistryCard, ...]
    card_metadata: dict[str, RegistryCardMetadata]


class _VerifiedCacheRecord(BaseModel):
    """Local proof that the last-known-good bytes passed TUF verification."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verified_at: datetime
    snapshot_id: str


class TufRegistryClient:
    """Load a signed catalog and retain a hash-bound last-known-good copy."""

    def __init__(
        self,
        *,
        base_url: str,
        cache_dir: Path,
        embedded_root: Path,
        timeout_seconds: int,
        max_stale_days: int,
    ) -> None:
        """Configure registry transport, trust anchor, and offline policy."""
        self._base_url = base_url.rstrip("/") + "/"
        self._cache_dir = cache_dir
        self._embedded_root = embedded_root
        self._timeout_seconds = timeout_seconds
        self._max_stale_days = max_stale_days
        self._metadata_dir = cache_dir / "metadata"
        self._targets_dir = cache_dir / "targets"
        self._last_known_good_path = cache_dir / "last-known-good-catalog.json"
        self._cache_record_path = cache_dir / "last-known-good.json"
        self._lock = FileLock(str(cache_dir / "refresh.lock"))

    def load_catalog(self) -> RegistryCatalog:
        """Refresh from the signed repository or return a bounded verified cache."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            try:
                return self._refresh()
            except Exception as error:  # noqa: BLE001 - security fallback boundary
                try:
                    return self._load_last_known_good()
                except (OSError, ValueError):
                    raise RegistryUnavailableError(
                        "signed model registry is unavailable and no acceptable "
                        "last-known-good catalog exists"
                    ) from error

    def _refresh(self) -> RegistryCatalog:
        """Perform a normal python-tuf refresh and persist verified target bytes."""
        self._metadata_dir.mkdir(parents=True, exist_ok=True)
        self._targets_dir.mkdir(parents=True, exist_ok=True)
        updater = Updater(
            metadata_dir=str(self._metadata_dir),
            metadata_base_url=f"{self._base_url}metadata/",
            target_dir=str(self._targets_dir),
            target_base_url=f"{self._base_url}targets/",
            fetcher=Urllib3Fetcher(
                socket_timeout=self._timeout_seconds,
                app_user_agent="Skulk model-registry client",
            ),
            # Supplying an explicit bootstrap ignores an arbitrary cached
            # root.json. python-tuf then replays its verified local root_history
            # before consulting the network, preserving legitimate rotations.
            bootstrap=self._embedded_root.read_bytes(),
        )
        updater.refresh()
        target = updater.get_targetinfo("v1/catalog.json")
        if target is None:
            raise ValueError("signed repository has no v1/catalog.json target")
        downloaded = Path(updater.download_target(target))
        payload = downloaded.read_bytes()
        catalog = RegistryCatalog.model_validate_json(payload, strict=False)
        self._write_verified_cache(payload, catalog)
        return catalog

    def _write_verified_cache(
        self, payload: bytes, catalog: RegistryCatalog
    ) -> None:
        """Atomically retain bytes that the updater just verified."""
        record = _VerifiedCacheRecord(
            sha256=hashlib.sha256(payload).hexdigest(),
            verified_at=datetime.now(UTC),
            snapshot_id=catalog.snapshot_id,
        )
        self._atomic_write(self._last_known_good_path, payload)
        self._atomic_write(
            self._cache_record_path,
            record.model_dump_json().encode(),
        )

    def _load_last_known_good(self) -> RegistryCatalog:
        """Load only hash-bound, previously verified, sufficiently fresh bytes."""
        if self._max_stale_days == 0:
            raise ValueError("last-known-good registry fallback is disabled")
        record = _VerifiedCacheRecord.model_validate_json(
            self._cache_record_path.read_bytes(), strict=False
        )
        now = datetime.now(UTC)
        verified_at = record.verified_at.astimezone(UTC)
        if now - verified_at > timedelta(days=self._max_stale_days):
            raise ValueError("last-known-good registry catalog is too old")
        payload = self._last_known_good_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record.sha256:
            raise ValueError("last-known-good registry catalog hash mismatch")
        catalog = RegistryCatalog.model_validate_json(payload, strict=False)
        if catalog.snapshot_id != record.snapshot_id:
            raise ValueError("last-known-good snapshot identity mismatch")
        return catalog

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        """Replace one local cache file without exposing partial contents."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
