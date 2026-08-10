"""Durable, self-describing records for installed model artifacts.

The model bytes remain the source of local availability.  This module binds
those bytes to the complete card Skulk used to acquire and launch them so an
installed artifact remains understandable after a registry outage or an
explicit air-gapped restart.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal, final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from skulk.shared.constants import SKULK_INSTALLED_CARD_RECORDS_DIR
from skulk.shared.models.model_cards import ModelCard, ModelId

INSTALLED_CARD_RELATIVE_PATH = Path(".skulk") / "installed-card.json"
"""Reserved sidecar path copied with canonical and staged artifacts."""

_SOURCE_REVISION_MARKER = ".skulk-source-revision"
_IGNORED_MANIFEST_NAMES = {
    ".last_used",
    ".skulk-source-revision",
    ".skulk-source-revision-staging",
}

InstalledCardVerification = Literal[
    "registry_verified",
    "local_legacy",
    "custom",
    "unresolved",
]
InstalledArtifactRole = Literal[
    "base",
    "vision_weights",
    "mtp_sidecar",
    "assistant",
    "served_draft",
    "vllm_draft",
]


@final
class InstalledFileManifestEntry(BaseModel):
    """One immutable file observation inside an installed artifact."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    path: str = Field(min_length=1, max_length=4096)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        """Reject absolute, ambiguous, and cross-platform escaping paths."""

        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or value != path.as_posix()
            or "\\" in value
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("manifest path must be a canonical relative POSIX path")
        return value


@final
class InstalledCardRecord(BaseModel):
    """Complete local card and byte identity for one installed artifact.

    ``model_card`` is deliberately retained in full.  ``verification`` states
    whether the local bytes are proven to be the signed card's pinned artifact;
    retaining a signed card as operational metadata never upgrades unmarked
    legacy bytes to ``registry_verified``.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    schema_version: Literal[1] = 1
    installed_identity: str = Field(
        pattern=r"^(?:card_[a-z2-7]{52}|local_[a-z2-7]{52})$"
    )
    artifact_model_id: str = Field(min_length=3, max_length=512)
    model_card: ModelCard
    verification: InstalledCardVerification
    artifact_role: InstalledArtifactRole = "base"
    owner_model_id: str | None = Field(default=None, min_length=3, max_length=512)
    owner_card_id: str | None = Field(default=None, max_length=80)
    artifact_repository: str = Field(min_length=3, max_length=512)
    artifact_revision: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{40}$",
    )
    artifact_file: str | None = Field(default=None, max_length=2048)
    artifact_format: str = Field(default="unknown", min_length=1, max_length=80)
    quantization: str = Field(default="", max_length=120)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: tuple[InstalledFileManifestEntry, ...]
    captured_at: datetime

    @model_validator(mode="after")
    def validate_evidence_and_ownership(self) -> "InstalledCardRecord":
        """Keep verification and companion ownership internally consistent."""

        ModelId(self.artifact_model_id)
        ModelId(self.artifact_repository)
        if self.owner_model_id is not None:
            ModelId(self.owner_model_id)
        if self.artifact_role == "base" and (
            self.owner_model_id is not None or self.owner_card_id is not None
        ):
            raise ValueError("base artifacts cannot declare companion ownership")
        if self.artifact_role != "base" and self.owner_model_id is None:
            raise ValueError("companion artifacts require owner_model_id")
        if self.verification == "registry_verified" and (
            self.model_card.registry_card_id is None or self.artifact_revision is None
        ):
            raise ValueError(
                "registry verification requires card and revision identity"
            )
        if self.verification == "custom" and not self.model_card.is_custom:
            raise ValueError("custom verification requires a custom card")
        return self

    @property
    def is_registry_current_candidate(self) -> bool:
        """Return whether this record carries an immutable registry identity."""

        return self.model_card.registry_card_id is not None


@final
class ExternalInstalledCardRecord(BaseModel):
    """Path-bound fallback for an artifact whose root cannot hold a sidecar."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    schema_version: Literal[1] = 1
    artifact_path: str = Field(min_length=1, max_length=4096)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    record: InstalledCardRecord


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_file_manifest(
    model_directory: Path,
) -> tuple[InstalledFileManifestEntry, ...]:
    """Hash every stable artifact file below ``model_directory``.

    Runtime-owned markers and partial files are excluded.  The installed-card
    sidecar is metadata about the manifest and therefore cannot hash itself.
    Symlinks are rejected so a malicious or accidental cache entry cannot make
    reconciliation read outside the artifact directory.
    """

    resolved_root = model_directory.resolve()
    entries: list[InstalledFileManifestEntry] = []
    for candidate in sorted(model_directory.rglob("*")):
        if candidate.is_symlink():
            raise ValueError(f"installed artifact contains symlink: {candidate}")
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(model_directory)
        if relative == INSTALLED_CARD_RELATIVE_PATH:
            continue
        if candidate.name in _IGNORED_MANIFEST_NAMES or candidate.name.endswith(
            ".partial"
        ):
            continue
        resolved = candidate.resolve()
        if not resolved.is_relative_to(resolved_root):
            raise ValueError(f"installed artifact path escapes root: {candidate}")
        entries.append(
            InstalledFileManifestEntry(
                path=relative.as_posix(),
                size_bytes=candidate.stat().st_size,
                sha256=_sha256_file(candidate),
            )
        )
    if not entries:
        raise ValueError(f"installed artifact has no stable files: {model_directory}")
    return tuple(entries)


def manifest_sha256(files: tuple[InstalledFileManifestEntry, ...]) -> str:
    """Return a deterministic digest for one complete file manifest."""

    payload = [entry.model_dump(mode="json") for entry in files]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _local_identity(model_card: ModelCard, digest: str, role: str) -> str:
    payload = {
        "card": model_card.model_dump(mode="json", exclude={"is_custom"}),
        "manifest_sha256": digest,
        "artifact_role": role,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    encoded = base64.b32encode(hashlib.sha256(canonical).digest()).decode().lower()
    return f"local_{encoded.rstrip('=')}"


def _revision_marker(model_directory: Path) -> str | None:
    marker = model_directory / _SOURCE_REVISION_MARKER
    try:
        value = marker.read_text().strip()
    except OSError:
        return None
    return value if len(value) == 40 else None


def model_card_artifact_format(model_card: ModelCard) -> str:
    """Derive the human artifact format from immutable card selection truth."""

    if model_card.gguf_file is not None:
        return "gguf"
    backends = model_card.placement.compatible_backends
    if any(backend.startswith("mlx_audio") for backend in backends):
        return "mlx-audio"
    if any(backend.startswith("mlx") for backend in backends):
        return "mlx"
    return "safetensors"


def companion_artifact_role(
    owner_card: ModelCard,
    repository: str,
) -> InstalledArtifactRole:
    """Resolve a declared companion repository to its durable artifact role."""

    if owner_card.vision is not None and owner_card.vision.weights_repo == repository:
        return "vision_weights"
    runtime = owner_card.runtime
    if runtime is not None:
        if runtime.mtp_sidecar_repo == repository:
            return "mtp_sidecar"
        if runtime.assistant_model_repo == repository:
            return "assistant"
        if runtime.served_spec_draft_repo == repository:
            return "served_draft"
        if runtime.vllm_spec_draft_repo == repository:
            return "vllm_draft"
    raise ValueError(
        f"{repository} is not a declared companion of {owner_card.model_id}"
    )


def build_installed_card_record(
    model_directory: Path,
    model_card: ModelCard,
    *,
    artifact_role: InstalledArtifactRole = "base",
    artifact_model_id: str | None = None,
    owner_model_id: str | None = None,
    owner_card_id: str | None = None,
    artifact_repository: str | None = None,
    artifact_revision: str | None = None,
    artifact_file: str | None = None,
    artifact_format: str | None = None,
) -> InstalledCardRecord:
    """Create an honest installed-card record from local bytes and card truth."""

    files = build_file_manifest(model_directory)
    digest = manifest_sha256(files)
    effective_repository = artifact_repository or str(model_card.artifact_repository)
    effective_revision = (
        artifact_revision
        if artifact_revision is not None
        else model_card.source_revision
    )
    revision_verified = (
        model_card.registry_card_id is not None
        and effective_revision is not None
        and _revision_marker(model_directory) == effective_revision
    )
    if model_card.is_custom:
        verification: InstalledCardVerification = "custom"
    elif revision_verified:
        verification = "registry_verified"
    else:
        verification = "local_legacy"
    registry_card_id = model_card.registry_card_id
    installed_identity = (
        registry_card_id
        if revision_verified
        and artifact_role == "base"
        and registry_card_id is not None
        else _local_identity(model_card, digest, artifact_role)
    )
    return InstalledCardRecord(
        installed_identity=installed_identity,
        artifact_model_id=artifact_model_id or str(model_card.model_id),
        model_card=model_card,
        verification=verification,
        artifact_role=artifact_role,
        owner_model_id=owner_model_id,
        owner_card_id=owner_card_id,
        artifact_repository=effective_repository,
        artifact_revision=effective_revision,
        artifact_file=artifact_file
        if artifact_file is not None
        else model_card.gguf_file,
        artifact_format=artifact_format or model_card_artifact_format(model_card),
        quantization=model_card.quantization,
        manifest_sha256=digest,
        files=files,
        captured_at=datetime.now(UTC),
    )


def installed_card_path(model_directory: Path) -> Path:
    """Return the reserved installed-card sidecar path for an artifact."""

    return model_directory / INSTALLED_CARD_RELATIVE_PATH


def write_installed_card(model_directory: Path, record: InstalledCardRecord) -> Path:
    """Atomically persist ``record`` beside its artifact files."""

    destination = installed_card_path(model_directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(record.model_dump_json(indent=2))
    temporary.replace(destination)
    return destination


def _external_record_path(
    model_directory: Path,
    manifest_digest: str,
    *,
    fallback_root: Path = SKULK_INSTALLED_CARD_RECORDS_DIR,
) -> Path:
    """Return a stable fallback path keyed by artifact path and manifest digest."""

    identity = f"{model_directory.resolve()}\0{manifest_digest}".encode()
    return fallback_root / f"{hashlib.sha256(identity).hexdigest()}.json"


def write_installed_card_with_fallback(
    model_directory: Path,
    record: InstalledCardRecord,
    *,
    fallback_root: Path = SKULK_INSTALLED_CARD_RECORDS_DIR,
) -> Path:
    """Write beside the artifact, falling back to Skulk data for read-only roots."""

    try:
        return write_installed_card(model_directory, record)
    except OSError:
        fallback_root.mkdir(parents=True, exist_ok=True)
        destination = _external_record_path(
            model_directory,
            record.manifest_sha256,
            fallback_root=fallback_root,
        )
        external = ExternalInstalledCardRecord(
            artifact_path=str(model_directory.resolve()),
            manifest_sha256=record.manifest_sha256,
            record=record,
        )
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        temporary.write_text(external.model_dump_json(indent=2))
        temporary.replace(destination)
        return destination


def read_installed_card(model_directory: Path) -> InstalledCardRecord | None:
    """Load a local installed-card record, returning ``None`` when absent."""

    path = installed_card_path(model_directory)
    if not path.is_file():
        return None
    return InstalledCardRecord.model_validate_json(path.read_bytes(), strict=False)


def read_installed_card_with_fallback(
    model_directory: Path,
    *,
    fallback_root: Path = SKULK_INSTALLED_CARD_RECORDS_DIR,
) -> InstalledCardRecord | None:
    """Load an adjacent sidecar or its path-and-manifest-bound fallback."""

    adjacent = read_installed_card(model_directory)
    if adjacent is not None:
        return adjacent
    if not fallback_root.is_dir():
        return None
    resolved_directory = str(model_directory.resolve())
    for path in fallback_root.glob("*.json"):
        try:
            external = ExternalInstalledCardRecord.model_validate_json(
                path.read_bytes(), strict=False
            )
        except (OSError, ValueError):
            continue
        if (
            external.artifact_path == resolved_directory
            and external.manifest_sha256 == external.record.manifest_sha256
            # A detached record cannot rely on the artifact directory and
            # sidecar being moved together. Re-hash the bytes before trusting
            # that path-bound fallback as installed truth.
            and verify_installed_card(
                model_directory,
                external.record,
                verify_hashes=True,
            )
        ):
            return external.record
    return None


def installed_card_matches(
    model_directory: Path,
    model_card: ModelCard,
) -> bool:
    """Return whether a complete sidecar identifies the requested card generation."""

    try:
        record = read_installed_card_with_fallback(model_directory)
    except (OSError, ValueError):
        return False
    if record is None:
        return False
    if not verify_installed_card(model_directory, record):
        return False
    requested_id = model_card.registry_card_id
    if requested_id is not None and record.model_card.registry_card_id == requested_id:
        return True
    return (
        requested_id is None
        and record.model_card.model_id == model_card.model_id
        and record.model_card.source_revision == model_card.source_revision
        and record.model_card.gguf_file == model_card.gguf_file
    )


def installed_companion_matches(
    model_directory: Path,
    *,
    artifact_model_id: str,
    owner_card: ModelCard,
    artifact_role: InstalledArtifactRole,
) -> bool:
    """Return whether a complete local artifact belongs to an owning card."""

    try:
        record = read_installed_card_with_fallback(model_directory)
    except (OSError, ValueError):
        return False
    if record is None:
        return False
    owner_identity_matches = (
        record.owner_card_id == owner_card.registry_card_id
        if owner_card.registry_card_id is not None
        else record.model_card == owner_card
    )
    return (
        record.artifact_model_id == artifact_model_id
        and record.artifact_role == artifact_role
        and owner_identity_matches
        and verify_installed_card(model_directory, record)
    )


def verify_installed_card(
    model_directory: Path,
    record: InstalledCardRecord,
    *,
    verify_hashes: bool = False,
) -> bool:
    """Verify path boundaries, sizes and optionally hashes for one sidecar."""

    resolved_root = model_directory.resolve()
    for entry in record.files:
        candidate = (resolved_root / entry.path).resolve()
        if not candidate.is_relative_to(resolved_root) or not candidate.is_file():
            return False
        try:
            if candidate.stat().st_size != entry.size_bytes:
                return False
            if verify_hashes and _sha256_file(candidate) != entry.sha256:
                return False
        except OSError:
            return False
    return bool(record.files)


def discover_installed_cards(roots: Iterable[Path]) -> list[InstalledCardRecord]:
    """Discover complete installed records below model-search roots.

    Only direct artifact directories are inspected.  Invalid, partial or stale
    records are ignored without making registry startup depend on any one disk.
    """

    records_by_identity: dict[str, InstalledCardRecord] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for model_directory in root.iterdir():
            if not model_directory.is_dir() or model_directory.name.startswith("."):
                continue
            try:
                record = read_installed_card_with_fallback(model_directory)
            except (OSError, ValueError):
                continue
            if record is None or not verify_installed_card(model_directory, record):
                continue
            records_by_identity.setdefault(record.installed_identity, record)
    return list(records_by_identity.values())


def associate_installed_card(
    model_directory: Path,
    cards: Iterable[ModelCard],
) -> InstalledCardRecord | None:
    """Associate a trusted catalog card with one pre-existing artifact directory.

    The directory name alone is never enough to assert registry verification;
    it is only used to select a candidate card.  Verification still requires
    the immutable revision marker and the selected artifact file, when one is
    declared.  Companion repositories retain their owning full base card.
    """

    directory_name = model_directory.name
    matches: list[tuple[ModelCard, InstalledArtifactRole, str, str | None]] = []
    for card in cards:
        base_names = {
            card.model_id.normalize(),
            card.artifact_repository.normalize(),
        }
        if directory_name in base_names and (
            card.gguf_file is None or (model_directory / card.gguf_file).is_file()
        ):
            matches.append(
                (card, "base", str(card.artifact_repository), card.source_revision)
            )
        companion_candidates: list[tuple[str | None, str | None]] = []
        if card.vision is not None:
            companion_candidates.append(
                (card.vision.weights_repo or None, card.vision.weights_revision)
            )
        if card.runtime is not None:
            companion_candidates.extend(
                [
                    (card.runtime.mtp_sidecar_repo, card.runtime.mtp_sidecar_revision),
                    (
                        card.runtime.assistant_model_repo,
                        card.runtime.assistant_model_revision,
                    ),
                    (
                        card.runtime.served_spec_draft_repo,
                        card.runtime.served_spec_draft_revision,
                    ),
                    (
                        card.runtime.vllm_spec_draft_repo,
                        card.runtime.vllm_spec_draft_revision,
                    ),
                ]
            )
        for repository, revision in companion_candidates:
            if repository is None or ModelId(repository).normalize() != directory_name:
                continue
            matches.append(
                (card, companion_artifact_role(card, repository), repository, revision)
            )
    if not matches:
        return None
    matches.sort(
        key=lambda match: (
            match[0].registry_card_id is not None,
            match[0].registry_card_id or "",
        ),
        reverse=True,
    )
    card, role, repository, revision = matches[0]
    return build_installed_card_record(
        model_directory,
        card,
        artifact_role=role,
        artifact_model_id=(str(card.model_id) if role == "base" else repository),
        owner_model_id=(None if role == "base" else str(card.model_id)),
        owner_card_id=(None if role == "base" else card.registry_card_id),
        artifact_repository=repository,
        artifact_revision=revision,
    )


def ensure_installed_cards(
    root: Path,
    cards: Iterable[ModelCard],
) -> None:
    """Materialize missing sidecars for trusted, complete legacy directories."""

    if not root.is_dir():
        return
    card_list = tuple(cards)
    for model_directory in root.iterdir():
        if not model_directory.is_dir() or model_directory.name.startswith("."):
            continue
        try:
            existing = read_installed_card_with_fallback(model_directory)
            if existing is not None and verify_installed_card(
                model_directory, existing
            ):
                continue
            record = associate_installed_card(model_directory, card_list)
            if record is not None:
                write_installed_card_with_fallback(model_directory, record)
        except (OSError, ValueError):
            continue
