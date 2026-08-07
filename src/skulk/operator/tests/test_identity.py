"""Persistence and integrity tests for stable operator identities."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from skulk.operator.identity import (
    ClusterPublicIdentity,
    OperatorIdentityRepository,
    create_cluster_identity,
)


def test_node_installation_identity_survives_repository_restart(tmp_path: Path) -> None:
    """A durable node reference must not follow the ephemeral libp2p lifecycle."""

    root = tmp_path / "operator"
    first = OperatorIdentityRepository(root).load_or_create_node_identity()
    second = OperatorIdentityRepository(root).load_or_create_node_identity()

    assert first == second
    assert root.stat().st_mode & 0o777 == 0o700
    assert (root / "node-installation.json").stat().st_mode & 0o777 == 0o600


def test_node_installation_identity_rejects_corrupt_record(tmp_path: Path) -> None:
    """Corruption fails loudly instead of silently changing durable node identity."""

    root = tmp_path / "operator"
    repository = OperatorIdentityRepository(root)
    repository.load_or_create_node_identity()
    path = root / "node-installation.json"
    payload = cast(
        dict[str, object],
        json.loads(path.read_text(encoding="utf-8")),
    )
    payload["nodeInstallId"] = "not-a-uuid"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValidationError):
        OperatorIdentityRepository(root).load_or_create_node_identity()


def test_cluster_identity_normalizes_name_and_binds_fingerprint() -> None:
    """The public fingerprint is deterministically bound to the generated key."""

    material = create_cluster_identity("  Fox   Den  ")

    assert material.public_identity.name == "Fox Den"
    assert material.public_identity.fingerprint.startswith("sha256:")
    assert len(material.private_key) == 32
    assert "private" not in material.public_identity.model_dump_json().lower()


def test_cluster_identity_rejects_substituted_fingerprint() -> None:
    """A persisted public key cannot be paired with an attacker-chosen fingerprint."""

    identity = create_cluster_identity().public_identity
    payload = identity.model_dump(mode="python")
    payload["fingerprint"] = "sha256:incorrect"

    with pytest.raises(ValidationError, match="fingerprint"):
        ClusterPublicIdentity.model_validate(payload)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions only")
def test_repository_repairs_overly_broad_directory_mode(tmp_path: Path) -> None:
    """Opening the repository narrows an accidentally broad directory mode."""

    root = tmp_path / "operator"
    root.mkdir(mode=0o755)

    OperatorIdentityRepository(root).load_or_create_node_identity()

    assert root.stat().st_mode & 0o777 == 0o700
