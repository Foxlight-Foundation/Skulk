"""Tests for the designated gateway's protected local authority key."""

import os
from pathlib import Path

import pytest

from skulk.operator.key_provider import (
    AuthorityKeyUnavailableError,
    LocalFileAuthorityKeyProvider,
)


def test_local_key_provider_creates_and_reuses_protected_key(tmp_path: Path) -> None:
    """First use creates one 32-byte key and later calls reuse it."""

    key_path = tmp_path / "operator" / "authority-key.bin"
    provider = LocalFileAuthorityKeyProvider(key_path)

    first = provider.ensure_data_key()
    second = provider.ensure_data_key()

    assert len(first) == 32
    assert second == first
    assert provider.load_data_key(provider.active_key_id) == first
    if os.name == "posix":
        assert key_path.stat().st_mode & 0o777 == 0o600
        assert key_path.parent.stat().st_mode & 0o777 == 0o700


def test_local_key_provider_rejects_wrong_version(tmp_path: Path) -> None:
    """A record cannot silently decrypt under a different key identifier."""

    provider = LocalFileAuthorityKeyProvider(tmp_path / "key.bin")
    provider.ensure_data_key()

    with pytest.raises(AuthorityKeyUnavailableError, match="version"):
        provider.load_data_key("future-key")


def test_local_key_provider_rejects_broad_permissions(tmp_path: Path) -> None:
    """POSIX group or other access fails closed."""

    if os.name != "posix":
        pytest.skip("POSIX permission semantics are unavailable")
    key_path = tmp_path / "key.bin"
    provider = LocalFileAuthorityKeyProvider(key_path)
    provider.ensure_data_key()
    key_path.chmod(0o644)

    with pytest.raises(AuthorityKeyUnavailableError, match="permissions"):
        provider.load_data_key(provider.active_key_id)
