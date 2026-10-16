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


# --- unified-memory classification (kernel facts, not size heuristics) --------

# kite5, a 64 GB Strix Halo at AMD's recommended 512 MB carve, as read on
# 2026-10-16: KFD sizes the GPU's bank at the TTM limit (60 GiB), which sits
# just under the node's RAM, so no GTT-versus-RAM comparison can classify it.
_KITE5_VRAM = 536_870_912
_KITE5_TTM_PAGES = 15_728_640
_KITE5_BANK = 64_424_509_440
_PAGE = 4096


def _make_apu_device(
    tmp_path: Path,
    *,
    vram: int,
    render_minor: int = 128,
    uma_interface: bool = False,
) -> Path:
    """A fake amdgpu device with its render node and optional ``uma`` dir."""
    device = tmp_path / "drm" / "card1" / "device"
    (device / "drm" / f"renderD{render_minor}").mkdir(parents=True)
    (device / "drm" / "card1").mkdir()
    (device / "gpu_busy_percent").write_text("0\n")
    (device / "mem_info_vram_total").write_text(f"{vram}\n")
    (device / "mem_info_vram_used").write_text("236175360\n")
    (device / "mem_info_gtt_total").write_text(f"{_KITE5_BANK}\n")
    if uma_interface:
        (device / "uma").mkdir()
        (device / "uma" / "carveout").write_text("0\n")
    return device


def _make_kfd(tmp_path: Path, nodes: list[tuple[int, int, int]]) -> Path:
    """KFD topology with ``(gfx_target_version, drm_render_minor, bank)`` nodes.

    Node 0 is always a CPU node (gfx_target_version 0), as on real systems.
    """
    root = tmp_path / "kfd" / "nodes"
    cpu = root / "0"
    (cpu / "mem_banks" / "0").mkdir(parents=True)
    (cpu / "properties").write_text("cpu_cores_count 32\ngfx_target_version 0\ndrm_render_minor 0\n")
    (cpu / "mem_banks" / "0" / "properties").write_text("heap_type 0\nsize_in_bytes 65430781952\n")
    for index, (gfx, minor, bank) in enumerate(nodes, start=1):
        node = root / str(index)
        (node / "mem_banks" / "0").mkdir(parents=True)
        (node / "properties").write_text(
            "cpu_cores_count 0\nsimd_count 80\nlocal_mem_size 0\n"
            f"gfx_target_version {gfx}\ndrm_render_minor {minor}\n"
        )
        (node / "mem_banks" / "0" / "properties").write_text(
            f"heap_type 1\nsize_in_bytes {bank}\nflags 0\nwidth 256\n"
        )
    return root


def _ttm(tmp_path: Path, pages: int) -> Path:
    path = tmp_path / "ttm_pages_limit"
    path.write_text(f"{pages}\n")
    return path


def test_recommended_carve_apu_is_unified_and_sized_by_ttm(tmp_path: Path) -> None:
    """kite5's exact facts: classified unified, addressable = the TTM limit."""
    device = _make_apu_device(tmp_path, vram=_KITE5_VRAM)
    kfd = _make_kfd(tmp_path, [(110501, 128, _KITE5_BANK)])
    acc = read_accelerator_metrics(
        device, kfd_root=kfd, ttm_pages_limit=_ttm(tmp_path, _KITE5_TTM_PAGES), page_size=_PAGE
    )
    assert acc.unified_memory is True
    assert acc.addressable_memory_bytes == _KITE5_BANK == _KITE5_TTM_PAGES * _PAGE


def test_kite4_apu_is_unified_too(tmp_path: Path) -> None:
    """kite4 (128 GB): the same rule, with GTT/TTM above RAM."""
    bank = 133_143_986_176
    device = _make_apu_device(tmp_path, vram=_KITE5_VRAM)
    kfd = _make_kfd(tmp_path, [(110501, 128, bank)])
    acc = read_accelerator_metrics(
        device, kfd_root=kfd, ttm_pages_limit=_ttm(tmp_path, bank // _PAGE), page_size=_PAGE
    )
    assert acc.unified_memory is True
    assert acc.addressable_memory_bytes == bank


def test_discrete_card_bank_equals_vram(tmp_path: Path) -> None:
    """The kernel sizes a discrete card's bank at its VRAM: never unified."""
    vram = 25_769_803_776
    device = _make_apu_device(tmp_path, vram=vram)
    kfd = _make_kfd(tmp_path, [(110000, 128, vram)])
    # Even when its TTM default happens to match, equality with VRAM wins.
    acc = read_accelerator_metrics(
        device, kfd_root=kfd, ttm_pages_limit=_ttm(tmp_path, vram // _PAGE), page_size=_PAGE
    )
    assert acc.unified_memory is False
    assert acc.addressable_memory_bytes is None


def test_apu_with_carve_larger_than_gtt_is_served_from_its_carve(tmp_path: Path) -> None:
    """An APU outside the kernel's prefer-GTT mode reports its carve as the bank."""
    carve = 103_079_215_104  # 96 GiB carve
    device = _make_apu_device(tmp_path, vram=carve, uma_interface=True)
    kfd = _make_kfd(tmp_path, [(110501, 128, carve)])
    acc = read_accelerator_metrics(
        device, kfd_root=kfd, ttm_pages_limit=_ttm(tmp_path, 4_194_304), page_size=_PAGE
    )
    assert acc.unified_memory is False


def test_uma_interface_without_kfd_is_unified_and_sized_by_ttm(tmp_path: Path) -> None:
    """No KFD report, but the APU-only ``uma`` sysfs interface: unified."""
    device = _make_apu_device(tmp_path, vram=_KITE5_VRAM, uma_interface=True)
    acc = read_accelerator_metrics(
        device,
        kfd_root=tmp_path / "absent",
        ttm_pages_limit=_ttm(tmp_path, _KITE5_TTM_PAGES),
        page_size=_PAGE,
    )
    assert acc.unified_memory is True
    assert acc.addressable_memory_bytes == _KITE5_TTM_PAGES * _PAGE


def test_no_kernel_facts_is_unknown(tmp_path: Path) -> None:
    """Without KFD or the ``uma`` interface the collector does not guess."""
    device = _make_apu_device(tmp_path, vram=_KITE5_VRAM)
    acc = read_accelerator_metrics(
        device,
        kfd_root=tmp_path / "absent",
        ttm_pages_limit=_ttm(tmp_path, _KITE5_TTM_PAGES),
        page_size=_PAGE,
    )
    assert acc.unified_memory is None
    assert acc.addressable_memory_bytes is None


def test_partitioned_bank_matching_neither_is_unknown(tmp_path: Path) -> None:
    """A bank equal to neither VRAM nor TTM (e.g. a partition) stays unknown."""
    device = _make_apu_device(tmp_path, vram=206_158_430_208)
    kfd = _make_kfd(tmp_path, [(90402, 128, 51_539_607_552)])
    acc = read_accelerator_metrics(
        device, kfd_root=kfd, ttm_pages_limit=_ttm(tmp_path, _KITE5_TTM_PAGES), page_size=_PAGE
    )
    assert acc.unified_memory is None


def test_kfd_node_is_matched_by_render_minor(tmp_path: Path) -> None:
    """Two GPUs: the device's own render minor picks its KFD node exactly."""
    device = _make_apu_device(tmp_path, vram=_KITE5_VRAM, render_minor=129)
    kfd = _make_kfd(
        tmp_path,
        [(110000, 128, 25_769_803_776), (110501, 129, _KITE5_BANK)],
    )
    acc = read_accelerator_metrics(
        device, kfd_root=kfd, ttm_pages_limit=_ttm(tmp_path, _KITE5_TTM_PAGES), page_size=_PAGE
    )
    assert acc.unified_memory is True
    assert acc.addressable_memory_bytes == _KITE5_BANK
