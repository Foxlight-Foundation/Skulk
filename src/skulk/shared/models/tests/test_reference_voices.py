# pyright: reportPrivateUsage=false
"""Validation coverage for packaged TTS reference-voice profiles."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from skulk.shared.models.reference_voices import (
    _load_reference_voice_catalog,
    bundled_reference_voice_profile,
    bundled_reference_voice_profiles,
)

_EXPECTED_IDS = (
    "angus",
    "ember",
    "hannah",
    "ian",
    "jake",
    "kite",
    "rufus",
    "samson",
    "sydney",
    "sylvie",
)


def test_bundled_reference_voice_catalog_is_complete_and_verified() -> None:
    """The shipped catalog must resolve all approved checksummed assets."""

    profiles = bundled_reference_voice_profiles()

    assert tuple(profile.id for profile in profiles) == _EXPECTED_IDS
    assert all(profile.audio_path.is_file() for profile in profiles)
    assert all(profile.transcript.startswith("Good morning.") for profile in profiles)
    assert bundled_reference_voice_profile("angus") == profiles[0]


def _write_catalog(
    root: Path,
    *,
    voice_id: str = "voice",
    audio_file: str = "voice/reference.mp3",
    transcript_file: str = "voice/reference.txt",
    digest: str | None = None,
    transcript: str = "Exact transcript.",
) -> None:
    """Write a minimal catalog fixture with one reference profile."""

    asset_root = root / "voice"
    asset_root.mkdir(parents=True)
    audio = b"reference-audio"
    (asset_root / "reference.mp3").write_bytes(audio)
    (asset_root / "reference.txt").write_text(transcript, encoding="utf-8")
    expected_digest = digest or hashlib.sha256(audio).hexdigest()
    (root / "catalog.toml").write_text(
        "\n".join(
            (
                "[[voices]]",
                f'id = "{voice_id}"',
                'name = "Voice"',
                'preferred_languages = ["en"]',
                f'audio_file = "{audio_file}"',
                f'transcript_file = "{transcript_file}"',
                f'sha256 = "{expected_digest}"',
                "",
            )
        ),
        encoding="utf-8",
    )


def test_reference_voice_catalog_rejects_asset_path_escape(tmp_path: Path) -> None:
    """Catalog entries cannot read conditioning assets outside their root."""

    _write_catalog(tmp_path, audio_file="../outside.mp3")
    (tmp_path.parent / "outside.mp3").write_bytes(b"reference-audio")

    with pytest.raises(ValueError, match="escapes"):
        _load_reference_voice_catalog(tmp_path)


def test_reference_voice_catalog_rejects_digest_mismatch(tmp_path: Path) -> None:
    """Modified voice bytes must fail before a generation request uses them."""

    _write_catalog(tmp_path, digest="0" * 64)

    with pytest.raises(ValueError, match="digest mismatch"):
        _load_reference_voice_catalog(tmp_path)


def test_reference_voice_catalog_rejects_empty_transcript(tmp_path: Path) -> None:
    """Reference conditioning always needs the exact non-empty transcript."""

    _write_catalog(tmp_path, transcript="   ")

    with pytest.raises(ValueError, match="transcript is empty"):
        _load_reference_voice_catalog(tmp_path)


def test_reference_voice_catalog_rejects_duplicate_ids(tmp_path: Path) -> None:
    """Voice identifiers must select exactly one immutable profile."""

    _write_catalog(tmp_path)
    first_entry = (tmp_path / "catalog.toml").read_text(encoding="utf-8")
    with (tmp_path / "catalog.toml").open("a", encoding="utf-8") as stream:
        stream.write(first_entry)

    with pytest.raises(ValueError, match="duplicate reference voice id"):
        _load_reference_voice_catalog(tmp_path)


def test_unknown_bundled_reference_voice_is_rejected() -> None:
    """A stale model-card profile identifier must fail closed."""

    with pytest.raises(ValueError, match="unknown bundled"):
        bundled_reference_voice_profile("missing")
