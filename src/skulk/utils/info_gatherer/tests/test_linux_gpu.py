"""Tests for the AMD/Linux sysfs GPU collector (no GPU needed; sysfs faked)."""

from pathlib import Path

from skulk.utils.info_gatherer.linux_gpu import (
    find_amd_gpu_device,
    read_accelerator_metrics,
    read_system_profile,
)


def _make_device(tmp_path: Path, *, with_hwmon: bool = True, sclk: bool = True) -> Path:
    """Build a fake amdgpu sysfs device dir mirroring kite4's layout."""
    device = tmp_path / "card1" / "device"
    device.mkdir(parents=True)
    (device / "gpu_busy_percent").write_text("42\n")
    (device / "mem_info_vram_total").write_text("68719476736\n")
    (device / "mem_info_vram_used").write_text("163233792\n")
    if sclk:
        (device / "pp_dpm_sclk").write_text("0: 600Mhz \n1: 1100Mhz *\n2: 2000Mhz \n")
    if with_hwmon:
        hwmon = device / "hwmon" / "hwmon5"
        hwmon.mkdir(parents=True)
        (hwmon / "power1_average").write_text("9030000\n")  # microwatts
        (hwmon / "temp1_input").write_text("39000\n")  # millidegrees
    return device


def test_reads_and_normalizes_units(tmp_path: Path) -> None:
    acc = read_accelerator_metrics(_make_device(tmp_path))
    assert acc.vendor == "amd"
    assert acc.utilization_ratio == 0.42  # 42% -> 0..1
    assert acc.vram_total_bytes == 68719476736
    assert acc.vram_used_bytes == 163233792
    assert acc.power_watts == 9.03  # microwatts -> watts
    assert acc.temperature_celsius == 39.0  # millidegrees -> Celsius
    assert acc.clock_mhz == 1100  # the starred DPM state


def test_missing_nodes_yield_none_not_zero(tmp_path: Path) -> None:
    # A device without hwmon / sclk reports None for those, never a fake 0.
    device = tmp_path / "card1" / "device"
    device.mkdir(parents=True)
    (device / "gpu_busy_percent").write_text("0\n")
    acc = read_accelerator_metrics(device)
    assert acc.utilization_ratio == 0.0  # a real 0%, distinct from None
    assert acc.power_watts is None
    assert acc.temperature_celsius is None
    assert acc.clock_mhz is None
    assert acc.vram_total_bytes is None


def test_system_profile_fills_legacy_scalars(tmp_path: Path) -> None:
    # Legacy Mac-shaped readers (topology GPU bar, power sampler) must see real
    # AMD values, not a default 0%/0C/0W (the no-fake-zero contract).
    prof = read_system_profile(_make_device(tmp_path))
    assert prof.gpu_usage == 42.0  # ratio 0.42 surfaced back as a percentage
    assert prof.temp == 39.0
    assert prof.sys_power == 9.03
    assert prof.accelerator is not None
    assert prof.accelerator.vram_total_bytes == 68719476736


def test_find_device_picks_the_render_card(tmp_path: Path) -> None:
    # A connector subdir (no gpu_busy_percent) must be skipped.
    (tmp_path / "card1-DP-1" / "device").mkdir(parents=True)
    device = _make_device(tmp_path)
    assert find_amd_gpu_device(tmp_path) == device


def test_find_device_none_when_absent(tmp_path: Path) -> None:
    (tmp_path / "card0-eDP-1" / "device").mkdir(parents=True)  # no gpu_busy_percent
    assert find_amd_gpu_device(tmp_path) is None
