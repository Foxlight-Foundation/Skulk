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


def test_bootstrap_does_not_overwrite_an_operator_launch_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A token the operator exported at launch outranks the fleet's.

    The user-set marker is what startup stamps when HF_TOKEN was present at
    launch; with it set, no sync path may replace the environment value.
    """
    from skulk.store.config import HF_TOKEN_USER_SET_MARKER

    monkeypatch.setenv("HF_TOKEN", "operator-launch-token")
    monkeypatch.setenv(HF_TOKEN_USER_SET_MARKER, "1")
    config_path = tmp_path / "skulk.yaml"

    _ = merge_cluster_config_bootstrap("hf_token: fleet-token\n", config_path)

    import os

    assert os.environ.get("HF_TOKEN") == "operator-launch-token"


def test_bootstrap_whitespace_token_does_not_clobber_local(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Whitespace is truthy but not a credential; it must merge as absent."""

    monkeypatch.delenv("HF_TOKEN", raising=False)
    config_path = tmp_path / "skulk.yaml"
    config_path.write_text("hf_token: local-secret\n")

    merged = merge_cluster_config_bootstrap("hf_token: '   '\n", config_path)

    assert merged.get("hf_token") == "local-secret"
    import os

    # The preserved local token is legitimately promoted; the whitespace
    # value must never be what lands in the environment.
    assert os.environ.get("HF_TOKEN") == "local-secret"


def test_bootstrap_rotates_a_config_derived_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Joining with an old config-promoted token must adopt the fleet's."""
    import os

    from skulk.store.config import HF_TOKEN_USER_SET_MARKER

    monkeypatch.setenv("HF_TOKEN", "old-token")
    monkeypatch.delenv(HF_TOKEN_USER_SET_MARKER, raising=False)
    config_path = tmp_path / "skulk.yaml"
    config_path.write_text("hf_token: old-token\n")

    merged = merge_cluster_config_bootstrap("hf_token: new-token\n", config_path)

    assert merged.get("hf_token") == "new-token"
    assert os.environ.get("HF_TOKEN") == "new-token"


def test_bootstrap_malformed_payload_keeps_local_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-mapping payload must keep the ENTIRE local config, not crash.

    Degrading to an empty mapping and merging would wipe every local field
    except the explicitly preserved ones (#922 review, second round).
    """
    monkeypatch.delenv("HF_TOKEN", raising=False)
    config_path = tmp_path / "skulk.yaml"
    config_path.write_text(
        "hf_token: local-secret\nlogging:\n  enabled: true\n"
    )

    merged = merge_cluster_config_bootstrap("- just\n- a\n- list\n", config_path)

    assert merged.get("hf_token") == "local-secret"
    # The rest of the local config survives too.
    assert merged.get("logging") == {"enabled": True}
    persisted = config_path.read_text()
    assert "logging" in persisted and "local-secret" in persisted
