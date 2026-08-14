"""Validated bundled reference-voice profiles for speech synthesis."""

from __future__ import annotations

import hashlib
import tomllib
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import cast

from pydantic import field_validator

from skulk.shared.constants import RESOURCES_DIR
from skulk.utils.pydantic_ext import FrozenModel


class ReferenceVoiceManifestEntry(FrozenModel):
    """One packaged reference voice declared by the resource catalog.

    Attributes:
        id: Stable API identifier selected through the ``voice`` request field.
        name: Human-readable name shown by clients.
        preferred_languages: Ordered BCP 47 language tags for automatic selection.
        audio_file: Catalog-relative path to the conditioning audio.
        transcript_file: Catalog-relative path to its exact spoken transcript.
        sha256: Expected lowercase SHA-256 digest of the audio bytes.
    """

    id: str
    name: str
    preferred_languages: tuple[str, ...]
    audio_file: str
    transcript_file: str
    sha256: str

    @field_validator("id", "name", "audio_file", "transcript_file", mode="before")
    @classmethod
    def _validate_nonempty_text(cls, value: object) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("reference voice text fields must not be empty")
        return normalized

    @field_validator("preferred_languages", mode="before")
    @classmethod
    def _validate_languages(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
            raise ValueError("preferred_languages must be a list of language tags")
        normalized = tuple(
            str(item).strip().lower() for item in cast("Iterable[object]", value)
        )
        if not normalized or any(not item for item in normalized):
            raise ValueError("preferred_languages must contain non-empty language tags")
        return normalized

    @field_validator("sha256", mode="before")
    @classmethod
    def _validate_sha256(cls, value: object) -> str:
        normalized = str(value).strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("sha256 must be a lowercase hexadecimal digest")
        return normalized


class ReferenceVoiceCatalogManifest(FrozenModel):
    """Top-level packaged reference-voice catalog."""

    voices: tuple[ReferenceVoiceManifestEntry, ...]

    @field_validator("voices", mode="before")
    @classmethod
    def _validate_voices(cls, value: object) -> tuple[object, ...]:
        if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
            raise ValueError("voices must be a list")
        return tuple(cast("Iterable[object]", value))


class ReferenceVoiceProfile(FrozenModel):
    """Resolved immutable conditioning data for one bundled reference voice.

    Attributes:
        id: Stable API identifier.
        name: Human-readable voice name.
        preferred_languages: Ordered BCP 47 language tags.
        audio_path: Verified catalog-contained audio file.
        transcript: Exact non-empty transcript paired with the audio.
        sha256: Verified SHA-256 digest of the audio bytes.
    """

    id: str
    name: str
    preferred_languages: tuple[str, ...]
    audio_path: Path
    transcript: str
    sha256: str


def _contained_file(catalog_root: Path, relative_path: str) -> Path:
    """Resolve a catalog-relative regular file without permitting path escape."""

    requested = Path(relative_path)
    if requested.is_absolute():
        raise ValueError("reference voice asset paths must be relative")
    resolved_root = catalog_root.resolve()
    resolved = (resolved_root / requested).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError("reference voice asset path escapes the catalog")
    if not resolved.is_file():
        raise ValueError(f"reference voice asset is not a regular file: {relative_path}")
    return resolved


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one local file without retaining its bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_reference_voice_catalog(catalog_root: Path) -> tuple[ReferenceVoiceProfile, ...]:
    """Load and verify a reference-voice catalog rooted at ``catalog_root``."""

    manifest_path = catalog_root / "catalog.toml"
    with manifest_path.open("rb") as stream:
        manifest = ReferenceVoiceCatalogManifest.model_validate(tomllib.load(stream))
    if not manifest.voices:
        raise ValueError("reference voice catalog must contain at least one profile")

    profiles: list[ReferenceVoiceProfile] = []
    seen_ids: set[str] = set()
    for entry in manifest.voices:
        if entry.id in seen_ids:
            raise ValueError(f"duplicate reference voice id: {entry.id}")
        seen_ids.add(entry.id)
        audio_path = _contained_file(catalog_root, entry.audio_file)
        transcript_path = _contained_file(catalog_root, entry.transcript_file)
        actual_digest = _file_sha256(audio_path)
        if actual_digest != entry.sha256:
            raise ValueError(f"reference voice digest mismatch: {entry.id}")
        transcript = transcript_path.read_text(encoding="utf-8").strip()
        if not transcript:
            raise ValueError(f"reference voice transcript is empty: {entry.id}")
        profiles.append(
            ReferenceVoiceProfile(
                id=entry.id,
                name=entry.name,
                preferred_languages=entry.preferred_languages,
                audio_path=audio_path,
                transcript=transcript,
                sha256=entry.sha256,
            )
        )
    return tuple(profiles)


@lru_cache(maxsize=1)
def bundled_reference_voice_profiles() -> tuple[ReferenceVoiceProfile, ...]:
    """Return every checksummed reference voice shipped with this Skulk build."""

    return _load_reference_voice_catalog(Path(RESOURCES_DIR) / "speech_reference_voices")


def bundled_reference_voice_profile(profile_id: str) -> ReferenceVoiceProfile:
    """Resolve one bundled reference voice by stable identifier.

    Args:
        profile_id: Voice identifier declared by a model card.

    Returns:
        The verified profile and its local conditioning asset.

    Raises:
        ValueError: If the installed catalog does not contain ``profile_id``.
    """

    for profile in bundled_reference_voice_profiles():
        if profile.id == profile_id:
            return profile
    raise ValueError(f"unknown bundled reference voice profile: {profile_id}")
