"""Observe owned AMD buffer allocations without entering Torch's allocator lock."""

from __future__ import annotations

import ctypes
import importlib
import os
import re
import sys
from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import Literal, Protocol, cast


class _PciDeviceQuery(Protocol):
    """The fixed HIP C signature resolved from Torch's already loaded dependencies."""

    argtypes: list[object]
    restype: type[ctypes.c_int]

    def __call__(
        self, buffer: ctypes.Array[ctypes.c_char], length: int, device: Literal[0]
    ) -> int:
        """Write the runtime-visible device zero PCI address, returning HIP status."""
        ...


def _pci_address(value: str) -> str:
    match = re.fullmatch(
        r"([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-1][0-9a-fA-F])\.([0-7])", value
    )
    if match is None:
        raise ValueError("native ROCm memory has an invalid PCI identity")
    domain, bus, slot, function = (int(part, 16) for part in match.groups())
    return f"{domain:04x}:{bus:02x}:{slot:02x}.{function:x}"


def _selected_pci_address() -> str:
    import torch

    if sys.platform != "linux" or torch.version.hip is None:
        raise ValueError("native ROCm memory requires the admitted Linux HIP runtime")
    if torch.cuda.current_device() != 0:
        raise ValueError("native ROCm memory requires runtime-visible device zero")
    location = importlib.import_module("torch._C").__file__
    if location is None:
        raise ValueError("native ROCm memory cannot identify its loaded Torch binding")
    # NOLOAD resolves HIP through this verified binding's existing dependencies.
    # Searching the host loader path could silently query a different HIP runtime.
    library = ctypes.CDLL(location, mode=os.RTLD_LOCAL | os.RTLD_NOLOAD)
    query = cast(_PciDeviceQuery, cast(object, library.hipDeviceGetPCIBusId))
    query.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int]
    query.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(32)
    if query(buffer, len(buffer), 0) != 0:
        raise ValueError("native ROCm memory cannot identify the selected PCI device")
    return _pci_address(bytes(buffer).partition(b"\0")[0].decode("ascii"))


def _read_fdinfos(directory: Path = Path("/proc/self/fdinfo")) -> tuple[str, ...]:
    with os.scandir(directory) as inventory:
        entries = tuple(Path(entry.path) for entry in islice(inventory, 4097))
    if len(entries) > 4096:
        raise ValueError("native ROCm memory descriptor inventory exceeds its bound")
    records: list[str] = []
    for entry in entries:
        try:
            with entry.open(encoding="utf-8") as source:
                record = source.read(65537)
        except FileNotFoundError:
            # The directory iterator's own descriptor can close before sampling.
            # A remaining matching client is still mandatory below.
            continue
        if len(record) > 65536:
            raise ValueError("native ROCm memory descriptor exceeds its bound")
        records.append(record)
    return tuple(records)


def _allocation_bytes(records: Iterable[str], selected_pci: str) -> int:
    # Kernel totals cover requested storage; resident/legacy counters omit
    # unbacked buffers. Identity and alias rules are specified here:
    # https://docs.kernel.org/gpu/drm-usage-stats.html
    selected = _pci_address(selected_pci)
    clients: dict[int, dict[str, int]] = {}
    for record in records:
        fields: dict[str, str] = {}
        for line in record.splitlines():
            key, delimiter, value = line.partition(":")
            if not delimiter or not key.startswith("drm-"):
                continue
            if key in fields:
                raise ValueError("native ROCm memory contains a duplicated DRM field")
            fields[key] = value.strip()
        if not fields:
            continue
        driver = fields.get("drm-driver")
        if driver is None:
            raise ValueError("native ROCm memory lacks its DRM driver identity")
        if driver != "amdgpu":
            continue
        pci = fields.get("drm-pdev")
        if pci is None:
            raise ValueError("native ROCm memory lacks its DRM PCI identity")
        if _pci_address(pci) != selected:
            continue
        identity = fields.get("drm-client-id", "")
        if not re.fullmatch(r"[0-9]+", identity):
            raise ValueError("native ROCm memory lacks its DRM client identity")
        if not {"drm-total-vram", "drm-total-gtt", "drm-total-cpu"}.issubset(fields):
            raise ValueError(
                "native ROCm memory lacks complete buffer allocation totals"
            )
        allocations: dict[str, int] = {}
        for key, value in fields.items():
            if not key.startswith("drm-total-") or key.startswith("drm-total-cycles-"):
                continue
            match = re.fullmatch(r"([0-9]+)(?: (KiB|MiB))?", value)
            if match is None:
                raise ValueError("native ROCm memory has an invalid allocation unit")
            scale = {None: 1, "KiB": 1024, "MiB": 1024**2}[match[2]]
            allocations[key] = int(match[1]) * scale
        # Duplicated/shared descriptors denote one client. Sequential aliases can
        # race allocations: retain each region's larger value, never sum aliases.
        client = clients.setdefault(int(identity), {})
        for key, size in allocations.items():
            client[key] = max(client.get(key, 0), size)
    if not clients:
        raise ValueError("native ROCm memory has no owned client for the selected GPU")
    return sum(sum(allocations.values()) for allocations in clients.values())


def rocm_allocated_bytes() -> int:
    """Read this process's requested AMD buffers on its actual HIP device zero.

    Kernel DRM totals include driver workspaces and nonresident buffers, not
    merely Torch's caching pool. Shared objects can be conservatively counted
    across distinct clients; no allocation is credited against free capacity.
    Missing identities, totals or sensing fail closed under the existing guard.
    """
    selected = _selected_pci_address()
    return _allocation_bytes(_read_fdinfos(), selected)
