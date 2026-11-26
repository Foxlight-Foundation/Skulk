"""Gateway identity preserves private custody and atomic publication on each OS."""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from skulk.api.inference_gateway import GatewaySigner
from skulk.utils.runtime_files import (
    is_private_directory,
    private_directory,
    read_private,
    write_private,
)


@pytest.fixture
def key_path(tmp_path: Path) -> Path:
    """Use a new state folder, rather than inheriting the test runner's access."""
    return tmp_path / "inference" / "gateway.key"


def test_signer_key_stays_private_and_stable(key_path: Path) -> None:
    """Restart retains the private gateway identity without account credentials."""
    public = GatewaySigner(key_path).public_key()
    assert GatewaySigner(key_path).public_key() == public
    assert is_private_directory(key_path.parent)
    assert len(read_private(key_path, 32)) == 32


def test_existing_unprotected_key_is_refused(key_path: Path) -> None:
    """Mode bits or a protected access list are required, without repairing it."""
    private_directory(key_path.parent)
    key_path.write_bytes(b"k" * 32)
    if sys.platform != "win32":
        key_path.chmod(0o644)
    # On Windows this file inherits its parent's ACL; it is not protected
    # owner-only state, even though no unrelated account currently has access.
    with pytest.raises(ValueError):
        GatewaySigner(key_path).public_key()
    assert key_path.read_bytes() == b"k" * 32


def test_concurrent_first_signers_publish_one_complete_key(
    key_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A paused first writer cannot expose an empty key or replace the winner."""
    generating, resume = Event(), Event()
    generate = Ed25519PrivateKey.generate

    def pause_first_generation() -> Ed25519PrivateKey:
        if not generating.is_set():
            generating.set()
            assert resume.wait(5)
        return generate()

    monkeypatch.setattr(Ed25519PrivateKey, "generate", pause_first_generation)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(GatewaySigner(key_path).public_key)
        try:
            assert generating.wait(5)
            winner = executor.submit(GatewaySigner(key_path).public_key).result(
                timeout=5
            )
        finally:
            resume.set()
        assert first.result(timeout=5) == winner
    assert GatewaySigner(key_path).public_key() == winner
    assert len(read_private(key_path, 32)) == 32
    assert tuple(key_path.parent.iterdir()) == (key_path,)


def test_interrupted_signer_creation_does_not_publish_a_key(
    key_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed content sync leaves no published key and the next call recovers."""
    sync = os.fsync

    def fail_sync(descriptor: int) -> None:
        raise OSError("interrupted key creation")

    monkeypatch.setattr(os, "fsync", fail_sync)
    with pytest.raises(OSError, match="interrupted key creation"):
        GatewaySigner(key_path).public_key()
    assert not tuple(key_path.parent.iterdir())
    monkeypatch.setattr(os, "fsync", sync)
    assert GatewaySigner(key_path).public_key() == GatewaySigner(key_path).public_key()


@pytest.mark.parametrize("size", [0, 31, 33])
def test_malformed_private_key_is_never_replaced(key_path: Path, size: int) -> None:
    """Invalid existing bytes are refused instead of silently rotating identity."""
    original = b"k" * size
    write_private(key_path, original)
    with pytest.raises(ValueError):
        GatewaySigner(key_path).public_key()
    assert read_private(key_path, 64) == original


def test_key_symlink_is_refused(key_path: Path) -> None:
    """A link cannot redirect gateway signing to another private identity."""
    private_directory(key_path.parent)
    target = key_path.parent / "other.key"
    write_private(target, b"k" * 32)
    try:
        key_path.symlink_to(target)
    except OSError:
        pytest.skip("this account cannot create symbolic links")
    with pytest.raises((OSError, ValueError)):
        GatewaySigner(key_path).public_key()
    assert read_private(target, 32) == b"k" * 32
