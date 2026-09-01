# pyright: reportPrivateUsage=false, reportUnusedFunction=false
"""Hugging Face token resolution and the operator guidance built on it (#917).

A token is node-local and never broadcast, so the only thing standing between
an operator and a silent gated-download failure is that these messages name
the right mechanism on the right node.
"""

from pathlib import Path

import pytest

from skulk.download.download_utils import _build_auth_error_message
from skulk.download.huggingface_utils import (
    get_hf_token,
    get_hf_token_path,
    resolve_hf_token_source,
)
from skulk.shared.types.common import ModelId

_MODEL = ModelId("meta-llama/Llama-3.1-8B-Instruct")


@pytest.fixture(autouse=True)
def _hermetic_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No ambient HF_TOKEN and no real token file may leak into these tests."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))


def _write_token_file(text: str) -> None:
    path = get_hf_token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(text)


def test_resolves_absent_when_nothing_is_configured() -> None:
    assert resolve_hf_token_source() == (None, "absent")


def test_env_wins_over_file(monkeypatch: pytest.MonkeyPatch) -> None:
    _write_token_file("from_file")
    monkeypatch.setenv("HF_TOKEN", "from_env")
    assert resolve_hf_token_source() == ("from_env", "env")


def test_reads_token_file_and_strips_whitespace() -> None:
    _write_token_file("  from_file\n")
    assert resolve_hf_token_source() == ("from_file", "file")


def test_empty_token_file_is_absent_not_a_blank_token() -> None:
    """A blank file must not present as a configured token."""
    _write_token_file("\n  \n")
    assert resolve_hf_token_source() == (None, "absent")


@pytest.mark.parametrize(
    "file_contents",
    [None, "", "   \n", "a_real_token"],
)
async def test_sync_and_async_resolvers_agree(file_contents: str | None) -> None:
    """The doctor's answer must match what downloads actually use.

    Two implementations of one precedence rule is exactly how operator
    guidance drifts away from runtime behavior, so pin them together.
    """
    if file_contents is not None:
        _write_token_file(file_contents)
    sync_token, _source = resolve_hf_token_source()
    assert sync_token == await get_hf_token()


async def test_401_without_token_names_node_and_restart_free_mechanism() -> None:
    message = await _build_auth_error_message(401, _MODEL)
    assert "no Hugging Face token is configured" in message
    assert "hf auth login" in message
    assert "store host" in message
    assert "never broadcast" in message


async def test_401_with_token_reports_rejection_not_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured-but-rejected token must not be described as missing."""
    monkeypatch.setenv("HF_TOKEN", "expired")
    message = await _build_auth_error_message(401, _MODEL)
    assert "rejected the configured token" in message
    assert "expired, revoked, or mistyped" in message
    assert "no Hugging Face token is configured" not in message
    assert "HF_TOKEN environment variable" in message


async def test_403_without_token_covers_both_terms_and_token() -> None:
    message = await _build_auth_error_message(403, _MODEL)
    assert f"https://huggingface.co/{_MODEL}" in message
    assert "no Hugging Face token is configured" in message


async def test_403_with_token_points_at_the_account_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "valid")
    message = await _build_auth_error_message(403, _MODEL)
    assert "same" in message and "account" in message
    assert "no Hugging Face token is configured" not in message


async def test_auth_messages_never_leak_the_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_supersecret_value")
    for status in (401, 403, 500):
        message = await _build_auth_error_message(status, _MODEL)
        assert "hf_supersecret_value" not in message
