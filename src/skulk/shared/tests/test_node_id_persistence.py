from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from skulk_pyo3_bindings import Keypair

from skulk.routing import router


def _owner_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.owner")


def test_node_identity_persists_across_restarts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Repeated starts on one physical host must keep one libp2p peer ID."""

    path = tmp_path / "node_id.keypair"
    monkeypatch.setattr(router, "_machine_identity_fingerprint", lambda: "host-a")

    first = router.get_node_id_keypair(path)
    second = router.get_node_id_keypair(path)

    assert first.to_bytes() == second.to_bytes()
    assert path.stat().st_mode & 0o777 == 0o600
    assert _owner_path(path).stat().st_mode & 0o777 == 0o600


def test_node_identity_rotates_when_snapshot_moves_to_another_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cloned config directory must not create duplicate live peer IDs."""

    path = tmp_path / "node_id.keypair"
    monkeypatch.setattr(router, "_machine_identity_fingerprint", lambda: "host-a")
    first = router.get_node_id_keypair(path)

    monkeypatch.setattr(router, "_machine_identity_fingerprint", lambda: "host-b")
    second = router.get_node_id_keypair(path)

    assert first.to_bytes() != second.to_bytes()
    assert _owner_path(path).read_text(encoding="ascii") == "host-b"


def test_legacy_ownerless_identity_rotates_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ownerless keys from old releases cannot be trusted after a clone."""

    path = tmp_path / "node_id.keypair"
    legacy = Keypair.generate()
    path.write_bytes(legacy.to_bytes())
    monkeypatch.setattr(router, "_machine_identity_fingerprint", lambda: "host-a")

    migrated = router.get_node_id_keypair(path)
    stable = router.get_node_id_keypair(path)

    assert migrated.to_bytes() != legacy.to_bytes()
    assert stable.to_bytes() == migrated.to_bytes()


def test_concurrent_identity_loaders_share_one_keypair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The file lock must serialize simultaneous startup identity creation."""

    path = tmp_path / "node_id.keypair"
    monkeypatch.setattr(router, "_machine_identity_fingerprint", lambda: "host-a")

    def load_identity(_attempt: int) -> bytes:
        return router.get_node_id_keypair(path).to_bytes()

    with ThreadPoolExecutor(max_workers=10) as executor:
        encoded_keypairs = list(executor.map(load_identity, range(20)))

    assert len(set(encoded_keypairs)) == 1


def test_unknown_machine_identity_still_persists_keypair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unsupported platforms retain restart stability without clone detection."""

    path = tmp_path / "node_id.keypair"
    monkeypatch.setattr(router, "_machine_identity_fingerprint", lambda: None)

    first = router.get_node_id_keypair(path)
    second = router.get_node_id_keypair(path)

    assert first.to_bytes() == second.to_bytes()
    assert not _owner_path(path).exists()


def test_default_test_identity_never_mutates_real_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Implicit test identities stay isolated from the developer's real key."""

    monkeypatch.setenv("SKULK_TESTS", "1")

    first = router.get_node_id_keypair()
    second = router.get_node_id_keypair()

    assert first.to_bytes() != second.to_bytes()
