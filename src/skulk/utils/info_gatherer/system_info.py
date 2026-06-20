import platform
import socket
import sys
from subprocess import CalledProcessError

import psutil
from anyio import run_process

from skulk.shared.types.profiling import InterfaceType, NetworkInterfaceInfo

# DMI fields are frequently populated with OEM placeholder junk; treat these
# (case-insensitively) as "not set" so a node reports a real name or nothing.
_DMI_JUNK = {
    "",
    "to be filled by o.e.m.",
    "default string",
    "system product name",
    "system manufacturer",
    "none",
    "not applicable",
    "not specified",
    "oem",
    "o.e.m.",
}


def _read_text(path: str) -> str:
    """Read and strip a small system file, or return '' if unavailable."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _clean_dmi(value: str) -> str:
    """Return a DMI value, or '' if it is empty or a known OEM placeholder."""
    return "" if value.strip().lower() in _DMI_JUNK else value.strip()


def _linux_os_pretty_name() -> str:
    """Return /etc/os-release PRETTY_NAME (e.g. 'Ubuntu 26.04 LTS') or ''."""
    for line in _read_text("/etc/os-release").splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip().strip('"')
    return ""


def get_os_version() -> str:
    """Return the OS version string for this node.

    On macOS this is the macOS version (e.g. ``"15.3"``). On Linux it is the
    distro pretty name from ``/etc/os-release`` (e.g. ``"Ubuntu 26.04 LTS"``),
    falling back to the bare platform name.
    """
    if sys.platform == "darwin":
        version = platform.mac_ver()[0]
        return version if version else "Unknown"
    if sys.platform.startswith("linux"):
        return _linux_os_pretty_name() or platform.system() or "Unknown"
    return platform.system() or "Unknown"


async def get_os_build_version() -> str:
    """Return the OS build version string.

    macOS: the build version (e.g. ``"24D5055b"``). Linux: the kernel release
    (e.g. ``"6.14.0-12-generic"``), the closest build analog. Other platforms:
    ``"Unknown"``.
    """
    if sys.platform.startswith("linux"):
        return platform.release() or "Unknown"
    if sys.platform != "darwin":
        return "Unknown"

    try:
        process = await run_process(["sw_vers", "-buildVersion"])
    except CalledProcessError:
        return "Unknown"

    return process.stdout.decode("utf-8", errors="replace").strip() or "Unknown"


async def get_friendly_name() -> str:
    """
    Asynchronously gets the 'Computer Name' (friendly name) of a Mac.
    e.g., "John's MacBook Pro"
    Returns the name as a string, or None if an error occurs or not on macOS.
    """
    hostname = socket.gethostname()

    if sys.platform != "darwin":
        return hostname

    try:
        process = await run_process(["scutil", "--get", "ComputerName"])
    except CalledProcessError:
        return hostname

    return process.stdout.decode("utf-8", errors="replace").strip() or hostname


def parse_hardware_port_types(listing: str) -> dict[str, InterfaceType]:
    """Classify devices from ``networksetup -listallhardwareports`` output.

    Pure parser (the subprocess lives at the caller). The port-name header
    gives the specific class; the device line applies one downgrade: an enX
    device beyond en0/en1 with a GENERIC "Ethernet" port name may be a USB
    dongle rather than built-in ethernet, so only that ambiguous case becomes
    ``maybe_ethernet``. A port the header classified specifically
    ("Thunderbolt N" → thunderbolt, "Wi-Fi" → wifi) keeps its label — Mac
    Thunderbolt ports are always en2+, so the previous unconditional
    downgrade meant "thunderbolt" could never survive on macOS and the ring's
    TB-first transport priority was dead code (#222). Unclassified enX ports
    (e.g. "iPhone USB") stay "unknown" (lowest priority) instead of being
    promoted to maybe_ethernet.
    """
    types: dict[str, InterfaceType] = {}
    current_type: InterfaceType = "unknown"

    for line in listing.splitlines():
        if line.startswith("Hardware Port:"):
            port_name = line.split(":", 1)[1].strip()
            if "Wi-Fi" in port_name:
                current_type = "wifi"
            elif "Ethernet" in port_name or "LAN" in port_name:
                current_type = "ethernet"
            elif port_name.startswith("Thunderbolt"):
                current_type = "thunderbolt"
            else:
                current_type = "unknown"
        elif line.startswith("Device:"):
            device = line.split(":", 1)[1].strip()
            device_type = current_type
            if (
                device.startswith("en")
                and device not in ("en0", "en1")
                and current_type == "ethernet"
            ):
                device_type = "maybe_ethernet"
            types[device] = device_type

    return types


async def _get_interface_types_from_networksetup() -> dict[str, InterfaceType]:
    """Parse networksetup -listallhardwareports to get interface types."""
    if sys.platform != "darwin":
        return {}

    try:
        result = await run_process(["networksetup", "-listallhardwareports"])
    except CalledProcessError:
        return {}

    return parse_hardware_port_types(result.stdout.decode())


async def get_network_interfaces() -> list[NetworkInterfaceInfo]:
    """
    Retrieves detailed network interface information on macOS.
    Parses output from 'networksetup -listallhardwareports' and 'ifconfig'
    to determine interface names, IP addresses, and types (ethernet, wifi, vpn, other).
    Returns a list of NetworkInterfaceInfo objects.
    """
    interfaces_info: list[NetworkInterfaceInfo] = []
    interface_types = await _get_interface_types_from_networksetup()

    for iface, services in psutil.net_if_addrs().items():
        for service in services:
            match service.family:
                case socket.AF_INET | socket.AF_INET6:
                    interfaces_info.append(
                        NetworkInterfaceInfo(
                            name=iface,
                            ip_address=service.address,
                            interface_type=interface_types.get(iface, "unknown"),
                        )
                    )
                case _:
                    pass

    return interfaces_info


def _linux_model_and_chip() -> tuple[str, str]:
    """Derive (model, chip) for a Linux node from sysfs/procfs (no subprocess).

    Model comes from DMI (``/sys/class/dmi/id``): the product name, falling back
    to the board name, prefixed with the vendor when it adds information (e.g.
    ``"Nimo Direct Inc. MME3L"``). Chip is the CPU brand string from
    ``/proc/cpuinfo`` (``"AMD RYZEN AI MAX+ 395 w/ Radeon 8060S"``), which on an
    APU usefully also names the integrated GPU. Either falls back to the Mac-style
    ``"Unknown Model"`` / ``"Unknown Chip"`` so callers stay uniform.
    """
    dmi = "/sys/class/dmi/id"
    product = _clean_dmi(_read_text(f"{dmi}/product_name")) or _clean_dmi(
        _read_text(f"{dmi}/board_name")
    )
    vendor = _clean_dmi(_read_text(f"{dmi}/sys_vendor"))
    if product and vendor and vendor.lower() not in product.lower():
        model = f"{vendor} {product}"
    else:
        model = product or "Unknown Model"

    chip = "Unknown Chip"
    for line in _read_text("/proc/cpuinfo").splitlines():
        if line.lower().startswith("model name"):
            chip = line.split(":", 1)[1].strip() or "Unknown Chip"
            break
    return (model, chip)


async def get_model_and_chip() -> tuple[str, str]:
    """Get this node's hardware model and chip names.

    macOS uses ``system_profiler``; Linux reads DMI + ``/proc/cpuinfo`` (see
    ``_linux_model_and_chip``); other platforms report ``Unknown``.
    """
    model = "Unknown Model"
    chip = "Unknown Chip"

    if sys.platform.startswith("linux"):
        return _linux_model_and_chip()
    if sys.platform != "darwin":
        return (model, chip)

    try:
        process = await run_process(
            [
                "system_profiler",
                "SPHardwareDataType",
            ]
        )
    except CalledProcessError:
        return (model, chip)

    # less interested in errors here because this value should be hard coded
    output = process.stdout.decode().strip()

    model_line = next(
        (line for line in output.split("\n") if "Model Name" in line), None
    )
    model = model_line.split(": ")[1] if model_line else "Unknown Model"

    chip_line = next((line for line in output.split("\n") if "Chip" in line), None)
    chip = chip_line.split(": ")[1] if chip_line else "Unknown Chip"

    return (model, chip)
