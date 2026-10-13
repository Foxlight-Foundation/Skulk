"""Apple hardware matching uses measured chip truth and degrades when unavailable."""

import subprocess
from unittest.mock import Mock

import pytest

from skulk.utils.info_gatherer import system_info


@pytest.mark.parametrize("name", ["Apple M4", "Apple M5 Max", "Intel Core i7", ""])
def test_apple_chip_reader_requires_successful_apple_identity(
    monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    """Only the successful Apple Silicon brand string becomes an exact chip class."""
    monkeypatch.setattr(system_info.sys, "platform", "darwin")
    result = subprocess.CompletedProcess([], 0, name + "\n", "")
    execute = Mock(return_value=result)
    monkeypatch.setattr(system_info.subprocess, "run", execute)
    assert system_info.observe_apple_chip() == (name if name.startswith("Apple M") else None)
    assert execute.call_args.kwargs["timeout"] == 2


def test_apple_chip_reader_degrades_on_probe_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing sysctl and nonzero exits never fabricate an exact chip class."""
    monkeypatch.setattr(system_info.sys, "platform", "darwin")
    monkeypatch.setattr(system_info.subprocess, "run", Mock(side_effect=OSError("unavailable")))
    assert system_info.observe_apple_chip() is None
    monkeypatch.setattr(system_info.subprocess, "run", Mock(return_value=subprocess.CompletedProcess([], 1, "Apple M4", "")))
    assert system_info.observe_apple_chip() is None
