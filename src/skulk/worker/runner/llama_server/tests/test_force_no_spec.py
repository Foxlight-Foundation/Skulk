"""`SKULK_LLAMA_SERVER_FORCE_NO_SPEC` gates the served runner's spec flags.

When set on a node, the served runner serves a speculation-carded model with
`--spec-type` omitted (plain decode), which is the apples-to-apples "MTP off"
baseline for an on-vs-off throughput comparison. This pins the env parsing.
"""

import pytest

from skulk.worker.runner.llama_server.runner import (
    _force_no_spec,  # pyright: ignore[reportPrivateUsage]
)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_truthy_values_disable_spec(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("SKULK_LLAMA_SERVER_FORCE_NO_SPEC", value)
    assert _force_no_spec() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "  "])
def test_falsy_values_keep_spec(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("SKULK_LLAMA_SERVER_FORCE_NO_SPEC", value)
    assert _force_no_spec() is False


def test_unset_keeps_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SKULK_LLAMA_SERVER_FORCE_NO_SPEC", raising=False)
    assert _force_no_spec() is False
