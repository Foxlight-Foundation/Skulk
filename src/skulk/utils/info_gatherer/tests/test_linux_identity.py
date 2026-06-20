# pyright: reportPrivateUsage=false
"""Tests for the Linux identity helpers in system_info (no real /sys needed)."""

import pytest

from skulk.utils.info_gatherer import system_info

# kite4's real values (Strix Halo / Nimo mini-PC) used as the fixture.
_FILES = {
    "/sys/class/dmi/id/product_name": "MME3L\n",
    "/sys/class/dmi/id/board_name": "NIMO Mini PC\n",
    "/sys/class/dmi/id/sys_vendor": "Nimo Direct Inc.\n",
    "/proc/cpuinfo": (
        "processor\t: 0\n"
        "vendor_id\t: AuthenticAMD\n"
        "model name\t: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S\n"
        "processor\t: 1\n"
        "model name\t: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S\n"
    ),
    "/etc/os-release": 'NAME="Ubuntu"\nPRETTY_NAME="Ubuntu 26.04 LTS"\nID=ubuntu\n',
}


@pytest.fixture
def fake_files(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(system_info, "_read_text", lambda path: _FILES.get(path, ""))


def test_linux_model_and_chip(fake_files: None) -> None:
    model, chip = system_info._linux_model_and_chip()
    assert model == "Nimo Direct Inc. MME3L"  # vendor prefixed onto product
    assert chip == "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S"  # from /proc/cpuinfo


def test_linux_os_pretty_name(fake_files: None) -> None:
    assert system_info._linux_os_pretty_name() == "Ubuntu 26.04 LTS"


def test_clean_dmi_filters_oem_junk() -> None:
    assert system_info._clean_dmi("To Be Filled By O.E.M.") == ""
    assert system_info._clean_dmi("Default string") == ""
    assert system_info._clean_dmi("  ") == ""
    assert system_info._clean_dmi("MME3L") == "MME3L"


def test_linux_model_falls_back_to_board_then_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # product_name is OEM junk -> use board_name; no cpuinfo -> Unknown Chip.
    files = {
        "/sys/class/dmi/id/product_name": "Default string",
        "/sys/class/dmi/id/board_name": "NIMO Mini PC",
        "/sys/class/dmi/id/sys_vendor": "Default string",
    }
    monkeypatch.setattr(system_info, "_read_text", lambda path: files.get(path, ""))
    model, chip = system_info._linux_model_and_chip()
    assert model == "NIMO Mini PC"  # vendor was junk, product junk -> board name
    assert chip == "Unknown Chip"
