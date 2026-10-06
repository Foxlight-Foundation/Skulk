# pyright: reportPrivateUsage=false, reportUnusedFunction=false
"""Hugging Face token resolution and the operator guidance built on it (#917).

A token is node-local and never broadcast, so the only thing standing between
an operator and a silent gated-download failure is that these messages name
the right mechanism on the right node.
"""

from pathlib import Path

import pytest

from skulk.download.download_utils import build_auth_error_message
from skulk.download.huggingface_utils import (
    get_hf_token,
    get_hf_token_path,
    resolve_hf_token_source,
)
from skulk.shared.types.common import ModelId

_MODEL = ModelId("meta-llama/Llama-3.1-8B-Instruct")


@pytest.fixture(autouse=True)
def _hermetic_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No ambient token from any source may leak into these tests."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    # huggingface_hub resolves HF_TOKEN_PATH once at import, so setting HF_HOME
    # here would NOT redirect it and these tests would read and write the
    # developer's real ~/.cache/huggingface/token. Patch the constant itself.
    from huggingface_hub import constants as hf_constants

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setattr(
        hf_constants, "HF_TOKEN_PATH", str(tmp_path / "hf-home" / "token")
    )
    # Relocate HOME rather than stubbing the resolver, so these tests exercise
    # the shipped path resolution and no developer's real skulk.env leaks in.
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fake-home")
    monkeypatch.setattr("skulk.store.config.load_skulk_config", lambda: None)


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
    guidance drifts away from runtime behavior, so pin them together. The
    comparison excludes skulk.yaml because get_hf_token models an already
    running process, where startup has folded that value into HF_TOKEN.
    """
    if file_contents is not None:
        _write_token_file(file_contents)
    sync_token, _source = resolve_hf_token_source(include_config=False)
    assert sync_token == await get_hf_token()


async def test_401_without_token_names_node_and_the_dashboard_path() -> None:
    message = await build_auth_error_message(401, _MODEL)
    assert "sent no Hugging Face token" in message
    assert "hf auth login" in message
    # The easy path on a formed cluster: enter it once, it propagates.
    assert "dashboard Settings" in message
    assert "propagates" in message


async def test_401_with_token_reports_rejection_not_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured-but-rejected token must not be described as missing."""
    monkeypatch.setenv("HF_TOKEN", "expired")
    message = await build_auth_error_message(401, _MODEL)
    assert "rejected the configured token" in message
    assert "expired, revoked, or mistyped" in message
    assert "sent no Hugging Face token" not in message
    assert "HF_TOKEN environment variable" in message


async def test_403_without_token_covers_both_terms_and_token() -> None:
    message = await build_auth_error_message(403, _MODEL)
    assert f"https://huggingface.co/{_MODEL}" in message
    assert "sent no Hugging Face token" in message


async def test_403_with_token_points_at_the_account_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "valid")
    message = await build_auth_error_message(403, _MODEL)
    assert "same" in message and "account" in message
    assert "sent no Hugging Face token" not in message


async def test_auth_messages_never_leak_the_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_supersecret_value")
    for status in (401, 403, 500):
        message = await build_auth_error_message(status, _MODEL)
        assert "hf_supersecret_value" not in message


def _set_config_token(
    monkeypatch: pytest.MonkeyPatch, token: str | None
) -> None:
    """Point skulk.yaml resolution at a config carrying (or lacking) a token."""
    from skulk.store.config import SkulkConfig

    config = SkulkConfig(hf_token=token)
    monkeypatch.setattr("skulk.store.config.load_skulk_config", lambda: config)


def test_config_token_is_found_without_a_running_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dashboard-saved token lives in skulk.yaml, not the environment.

    Standalone commands run before node startup promotes it into HF_TOKEN, so
    resolving only env and file would report the most common setup as absent.
    """
    _set_config_token(monkeypatch, "from_config")
    assert resolve_hf_token_source() == ("from_config", "config")


def test_env_outranks_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_config_token(monkeypatch, "from_config")
    monkeypatch.setenv("HF_TOKEN", "from_env")
    assert resolve_hf_token_source() == ("from_env", "env")


def test_config_outranks_token_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirrors runtime: startup promotes hf_token before the file is read."""
    _write_token_file("from_file")
    _set_config_token(monkeypatch, "from_config")
    assert resolve_hf_token_source() == ("from_config", "config")


def test_blank_config_token_falls_through_to_the_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_token_file("from_file")
    _set_config_token(monkeypatch, "   ")
    assert resolve_hf_token_source() == ("from_file", "file")


def test_include_config_false_ignores_skulk_yaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_config_token(monkeypatch, "from_config")
    assert resolve_hf_token_source(include_config=False) == (None, "absent")


def test_unreadable_config_does_not_break_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed skulk.yaml is another check's problem, not a crash here."""

    def _raise() -> None:
        raise ValueError("malformed config")

    monkeypatch.setattr("skulk.store.config.load_skulk_config", _raise)
    _write_token_file("from_file")
    assert resolve_hf_token_source() == ("from_file", "file")


async def test_401_names_skulk_yaml_when_that_is_the_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_config_token(monkeypatch, "stale")
    message = await build_auth_error_message(401, _MODEL)
    assert "hf_token in skulk.yaml" in message
    assert "stale" not in message


def _write_service_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: str) -> None:
    """Create ~/.skulk/skulk.env under the relocated HOME, holding *body*."""
    del monkeypatch
    path = (tmp_path / "fake-home" / ".skulk" / "skulk.env")
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(body)


def test_service_env_file_token_is_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Doctor's own remediation writes here, so it must read it back."""
    _write_service_env(
        monkeypatch, tmp_path, "SKULK_VLLM_BIN=/x\nHF_TOKEN=from_service_env\n"
    )
    assert resolve_hf_token_source() == ("from_service_env", "service_env")


def test_service_env_quoting_and_comments_are_handled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_service_env(
        monkeypatch, tmp_path, '# HF_TOKEN=commented\nHF_TOKEN="quoted_value"\n'
    )
    assert resolve_hf_token_source() == ("quoted_value", "service_env")


def test_service_env_outranks_config_and_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The wrapper exports it before startup would apply hf_token."""
    _write_token_file("from_file")
    _set_config_token(monkeypatch, "from_config")
    _write_service_env(monkeypatch, tmp_path, "HF_TOKEN=from_service_env\n")
    assert resolve_hf_token_source() == ("from_service_env", "service_env")


def test_missing_service_env_file_is_not_an_error() -> None:
    _write_token_file("from_file")
    assert resolve_hf_token_source() == ("from_file", "file")


async def test_auth_message_ignores_tokens_this_process_never_loaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A 401 must describe the token actually sent, which was none.

    skulk.yaml and skulk.env only reach downloads through HF_TOKEN at startup,
    so describing them as "sent and rejected" would send the operator hunting a
    revoked token instead of restarting the node.
    """
    _set_config_token(monkeypatch, "never_loaded")
    message = await build_auth_error_message(401, _MODEL)
    assert "sent no Hugging Face token" in message
    assert "rejected the configured token" not in message
    # ...but it must still point at the token sitting unused in the config.
    assert "has not loaded it; restart the node" in message
    assert "never_loaded" not in message


async def test_auth_message_has_no_restart_hint_when_nothing_is_configured() -> None:
    message = await build_auth_error_message(401, _MODEL)
    assert "sent no Hugging Face token" in message
    assert "restart the node to apply it" not in message


def test_service_env_path_matches_the_shipped_wrappers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """~/.skulk/skulk.env, and SKULK_HOME must not move it.

    The systemd unit hardcodes %h/.skulk/skulk.env and skulk-startup.sh
    defaults to $HOME/.skulk/skulk.env; neither consults SKULK_HOME, so
    resolving from it would read a file the service never sources.
    """
    from skulk.download.huggingface_utils import get_service_env_path

    monkeypatch.setenv("SKULK_HOME", "somewhere-else")
    assert get_service_env_path() == tmp_path / "fake-home" / ".skulk" / "skulk.env"


def test_service_env_last_assignment_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Shell sourcing and systemd EnvironmentFile both take the final value.

    An operator appending a replacement token must not have doctor keep
    reporting the superseded one.
    """
    _write_service_env(
        monkeypatch,
        tmp_path,
        "HF_TOKEN=superseded\nSKULK_VLLM_BIN=/x\nHF_TOKEN=current\n",
    )
    assert resolve_hf_token_source() == ("current", "service_env")


def test_service_env_export_prefix_is_recognized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """skulk-startup.sh sources this file as shell, so `export` is valid there."""
    _write_service_env(monkeypatch, tmp_path, "export HF_TOKEN=exported_value\n")
    assert resolve_hf_token_source() == ("exported_value", "service_env")


def test_service_env_inline_comment_is_not_part_of_the_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """` #` starts a comment in the shell that sources this file."""
    _write_service_env(monkeypatch, tmp_path, "HF_TOKEN=real_value  # rotated 2026-09\n")
    assert resolve_hf_token_source() == ("real_value", "service_env")


def test_service_env_hash_without_space_stays_in_the_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Shell only starts a comment after whitespace, so a bare # is literal."""
    _write_service_env(monkeypatch, tmp_path, "HF_TOKEN=has#hash\n")
    assert resolve_hf_token_source() == ("has#hash", "service_env")


def test_blank_environment_token_pins_resolution_to_the_token_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A present-but-empty HF_TOKEN is not an absent one.

    Node startup only promotes hf_token when the key is missing, so a blank
    export leaves the downloader on the token file. Reporting the startup-only
    sources would claim a token that never gets sent.
    """
    _write_service_env(monkeypatch, tmp_path, "HF_TOKEN=from_service_env\n")
    _set_config_token(monkeypatch, "from_config")
    _write_token_file("from_file")
    monkeypatch.setenv("HF_TOKEN", "")
    assert resolve_hf_token_source() == ("from_file", "file")


def test_blank_environment_token_with_no_file_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_config_token(monkeypatch, "from_config")
    monkeypatch.setenv("HF_TOKEN", "")
    assert resolve_hf_token_source() == (None, "absent")


def test_token_path_follows_huggingface_hub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """HF_TOKEN_PATH and the XDG cache location must be honored as the Hub does.

    Recomputing $HF_HOME/token missed both, so `hf auth login` could report
    success while writing somewhere Skulk never read.
    """
    from huggingface_hub import constants as hf_constants

    override = tmp_path / "hub-token"
    monkeypatch.setattr(hf_constants, "HF_TOKEN_PATH", str(override))
    assert get_hf_token_path() == override

    _ = override.write_text("from_hub_path\n")
    assert resolve_hf_token_source() == ("from_hub_path", "file")


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_whitespace_environment_token_is_not_a_credential(
    monkeypatch: pytest.MonkeyPatch, blank: str
) -> None:
    """Whitespace in HF_TOKEN must not read as a configured token.

    Left unstripped it would be sent as a bearer token and rejected, and the
    guidance would tell the operator their token was revoked rather than that
    they never had one.
    """
    _write_token_file("from_file")
    monkeypatch.setenv("HF_TOKEN", blank)
    assert resolve_hf_token_source() == ("from_file", "file")


@pytest.mark.parametrize("blank", ["", "   "])
async def test_whitespace_environment_token_agrees_across_resolvers(
    monkeypatch: pytest.MonkeyPatch, blank: str
) -> None:
    monkeypatch.setenv("HF_TOKEN", blank)
    sync_token, _source = resolve_hf_token_source(include_config=False)
    assert sync_token == await get_hf_token()
    assert sync_token is None


async def test_whitespace_environment_token_reports_as_absent_not_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "   ")
    message = await build_auth_error_message(401, _MODEL)
    assert "sent no Hugging Face token" in message
    assert "rejected the configured token" not in message
