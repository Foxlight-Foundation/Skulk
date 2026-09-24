"""Pinned audio.cpp preparation, cache reuse, and offline failure behavior."""

import hashlib
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

from skulk.provisioning import audio_cpp


def _wheel(tmp_path: Path) -> tuple[audio_cpp.AudioCppWheel, Path]:
    wheel_path = tmp_path / "skulk_audio_cpp_cpu-0.8.2.post1-py3-none-macosx_15_0_arm64.whl"
    with zipfile.ZipFile(wheel_path, "w") as archive:
        for member in audio_cpp._REQUIRED_MEMBERS:  # pyright: ignore[reportPrivateUsage]
            archive.writestr(member, b"#!/bin/sh\n" if member.endswith("audiocpp_server") else b"{}")
    wheel = audio_cpp.AudioCppWheel(
        filename=wheel_path.name,
        sha256=hashlib.sha256(wheel_path.read_bytes()).hexdigest(),
    )
    return wheel, wheel_path


def _fixture_downloader(source: Path) -> Callable[[audio_cpp.AudioCppWheel, Path], None]:
    """Return a typed local substitute for the package-channel downloader."""

    def download(_wheel: audio_cpp.AudioCppWheel, destination: Path) -> None:
        destination.write_bytes(source.read_bytes())

    return download


def test_prepare_download_then_offline_cache_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One verified download gives an offline-ready executable thereafter."""
    wheel, source = _wheel(tmp_path)
    monkeypatch.setattr(audio_cpp, "SKULK_ENGINES_DIR", tmp_path / "cache")
    monkeypatch.setattr(audio_cpp, "audio_cpp_wheel_for_host", lambda: wheel)
    monkeypatch.setattr(
        audio_cpp,
        "_download_wheel",
        _fixture_downloader(source),
    )
    monkeypatch.delenv("SKULK_AUDIO_CPP_BIN", raising=False)
    installed = audio_cpp.prepare_audio_cpp(allow_download=True, environ={})
    assert installed.is_file()
    monkeypatch.delenv("SKULK_AUDIO_CPP_BIN", raising=False)
    assert audio_cpp.prepare_audio_cpp(allow_download=False, environ={}) == installed


def test_offline_miss_and_tampered_cache_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An offline miss or changed binary cannot become ready by cache presence."""
    wheel, source = _wheel(tmp_path)
    monkeypatch.setattr(audio_cpp, "SKULK_ENGINES_DIR", tmp_path / "cache")
    monkeypatch.setattr(audio_cpp, "audio_cpp_wheel_for_host", lambda: wheel)
    monkeypatch.delenv("SKULK_AUDIO_CPP_BIN", raising=False)
    with pytest.raises(RuntimeError, match="not in the verified local cache"):
        audio_cpp.prepare_audio_cpp(allow_download=False, environ={})
    monkeypatch.setattr(
        audio_cpp,
        "_download_wheel",
        _fixture_downloader(source),
    )
    installed = audio_cpp.prepare_audio_cpp(allow_download=True, environ={})
    installed.write_bytes(b"replaced executable")
    monkeypatch.delenv("SKULK_AUDIO_CPP_BIN", raising=False)
    with pytest.raises(RuntimeError, match="not in the verified local cache"):
        audio_cpp.prepare_audio_cpp(allow_download=False, environ={})


def test_explicit_override_cannot_hide_invalid_path(tmp_path: Path) -> None:
    """A bad operator override remains a mount error, without downloading."""
    with pytest.raises(RuntimeError, match="names no executable"):
        audio_cpp.prepare_audio_cpp(
            allow_download=True,
            environ={"SKULK_AUDIO_CPP_BIN": str(tmp_path / "missing")},
        )


def test_cached_spec_mutation_invalidates_even_selected_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache record covers required model specs as well as the executable."""
    wheel, source = _wheel(tmp_path)
    monkeypatch.setattr(audio_cpp, "SKULK_ENGINES_DIR", tmp_path / "cache")
    monkeypatch.setattr(audio_cpp, "audio_cpp_wheel_for_host", lambda: wheel)
    monkeypatch.setattr(audio_cpp, "_download_wheel", _fixture_downloader(source))
    monkeypatch.delenv("SKULK_AUDIO_CPP_BIN", raising=False)
    installed = audio_cpp.prepare_audio_cpp(allow_download=True, environ={})
    (installed.parent.parent / "model_specs" / "ace_step.json").write_text("tampered")
    with pytest.raises(RuntimeError, match="integrity verification"):
        audio_cpp.prepare_audio_cpp(
            allow_download=False,
            environ={"SKULK_AUDIO_CPP_BIN": str(installed)},
        )
