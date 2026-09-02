"""Cluster-wide Hugging Face token propagation.

A token entered in any node's Settings must converge onto the node that
actually fetches from Hugging Face: the store host, a worker doing an
``allow_hf_fallback`` direct download, or a node joining the fleet later.
The fabric is PSK-encrypted and trusted by doctrine, so the token rides the
ordinary config-sync and bootstrap payloads; ``GET /config`` still never
returns it. These tests pin the send sides and the bootstrap merge; the
ordinary config-sync receive side is pinned in
``download/tests/test_progress_throttle.py``.
"""

from pathlib import Path

import pytest

from skulk.main import merge_cluster_config_bootstrap


def test_bootstrap_adopts_the_masters_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A joining node downloads with the fleet's token, no restart needed."""

    monkeypatch.delenv("HF_TOKEN", raising=False)
    config_path = tmp_path / "skulk.yaml"

    merged = merge_cluster_config_bootstrap(
        "hf_token: fleet-token\nlogging:\n  enabled: false\n",
        config_path,
    )

    assert merged.get("hf_token") == "fleet-token"
    assert "hf_token: fleet-token" in config_path.read_text()
    assert config_path.stat().st_mode & 0o777 == 0o600
    import os

    assert os.environ.get("HF_TOKEN") == "fleet-token"


def test_bootstrap_without_a_token_preserves_the_local_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The previous raw overwrite erased local tokens on every bootstrap."""

    monkeypatch.delenv("HF_TOKEN", raising=False)
    config_path = tmp_path / "skulk.yaml"
    config_path.write_text("hf_token: local-secret\n")

    merged = merge_cluster_config_bootstrap(
        "logging:\n  enabled: false\n",
        config_path,
    )

    assert merged.get("hf_token") == "local-secret"
    assert "hf_token: local-secret" in config_path.read_text()


def test_bootstrap_blank_token_does_not_clobber_local(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    config_path = tmp_path / "skulk.yaml"
    config_path.write_text("hf_token: local-secret\n")

    merged = merge_cluster_config_bootstrap("hf_token: ''\n", config_path)

    assert merged.get("hf_token") == "local-secret"


def test_bootstrap_preserves_local_model_trust_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Deprecated compatibility state stays node-local through bootstrap."""

    monkeypatch.delenv("HF_TOKEN", raising=False)
    config_path = tmp_path / "skulk.yaml"
    card_id = f"card_{'a' * 52}"
    config_path.write_text(
        "model_trust:\n"
        "  approved_remote_code_identities:\n"
        f"    - {card_id}\n"
    )

    merged = merge_cluster_config_bootstrap(
        "logging:\n  enabled: false\n",
        config_path,
    )

    assert "model_trust" in merged
    assert card_id in config_path.read_text()


def test_bootstrap_does_not_overwrite_an_existing_env_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A token the operator exported at launch outranks the fleet's."""

    monkeypatch.setenv("HF_TOKEN", "operator-launch-token")
    config_path = tmp_path / "skulk.yaml"

    _ = merge_cluster_config_bootstrap("hf_token: fleet-token\n", config_path)

    import os

    assert os.environ.get("HF_TOKEN") == "operator-launch-token"
