"""Linux interface-type classification via sysfs.

Before `classify_linux_interface`, every Linux interface was labeled
``unknown`` (interface types came only from macOS ``networksetup``), so the
ring/RPC address prioritiser's thunderbolt-first ranking (#265) could never
fire between two Linux nodes: a USB4/TB point-to-point link and the ordinary
LAN tied at "unknown" priority. These tests exercise the classifier against a
synthetic sysfs tree; no hardware or privileged reads involved.
"""

import os
from pathlib import Path

from skulk.utils.info_gatherer.system_info import classify_linux_interface


def _make_iface(
    sysfs_root: Path,
    name: str,
    *,
    wireless: bool = False,
    device: bool = False,
    driver: str | None = None,
) -> None:
    iface_dir = sysfs_root / name
    iface_dir.mkdir(parents=True)
    if wireless:
        (iface_dir / "wireless").mkdir()
    if device or driver is not None:
        (iface_dir / "device").mkdir()
    if driver is not None:
        # Mirror the kernel layout: device/driver is a symlink into
        # /sys/bus/<bus>/drivers/<name>; only the basename matters here.
        drivers_dir = sysfs_root / "_drivers"
        drivers_dir.mkdir(exist_ok=True)
        target = drivers_dir / driver
        target.mkdir(exist_ok=True)
        os.symlink(target, iface_dir / "device" / "driver")


def test_wireless_dir_wins_regardless_of_name(tmp_path: Path) -> None:
    # Kernel creates the wireless dir for any Wi-Fi interface; name-agnostic.
    _make_iface(tmp_path, "wlp2s0", wireless=True, device=True, driver="ath11k_pci")
    assert classify_linux_interface("wlp2s0", str(tmp_path)) == "wifi"


def test_thunderbolt_by_driver_symlink(tmp_path: Path) -> None:
    # thunderbolt-net interfaces are the USB4/TB point-to-point links the
    # address prioritiser must rank first.
    _make_iface(tmp_path, "thunderbolt0", device=True, driver="thunderbolt-net")
    assert classify_linux_interface("thunderbolt0", str(tmp_path)) == "thunderbolt"


def test_thunderbolt_by_name_when_driver_unreadable(tmp_path: Path) -> None:
    _make_iface(tmp_path, "tb0", device=True)
    assert classify_linux_interface("tb0", str(tmp_path)) == "thunderbolt"


def test_ethernet_needs_physical_device_backing(tmp_path: Path) -> None:
    _make_iface(tmp_path, "enp92s0", device=True, driver="r8169")
    assert classify_linux_interface("enp92s0", str(tmp_path)) == "ethernet"


def test_virtual_en_named_interface_stays_unknown(tmp_path: Path) -> None:
    # A veth/bridge can carry an en-ish name but has no device backing; it must
    # not masquerade as wired transport for the prioritiser.
    _make_iface(tmp_path, "enveth0")
    assert classify_linux_interface("enveth0", str(tmp_path)) == "unknown"


def test_tunnels_and_loopback_stay_unknown(tmp_path: Path) -> None:
    # Tailscale/tun devices are handled by address-based VPN detection
    # downstream; the classifier must not give them a transport label.
    _make_iface(tmp_path, "tailscale0")
    _make_iface(tmp_path, "lo")
    assert classify_linux_interface("tailscale0", str(tmp_path)) == "unknown"
    assert classify_linux_interface("lo", str(tmp_path)) == "unknown"


def test_missing_sysfs_entry_is_unknown(tmp_path: Path) -> None:
    # Interface disappeared between enumeration and classification.
    assert classify_linux_interface("enp99s0", str(tmp_path)) == "unknown"
