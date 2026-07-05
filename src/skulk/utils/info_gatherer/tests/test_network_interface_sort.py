"""``get_network_interfaces`` must return a deterministically ordered list.

``psutil.net_if_addrs()`` iterates in an unspecified order that can differ
poll-to-poll (worst on Linux/AMD nodes, which carry the most interfaces). The
connectivity change-detection that stops the 5-node gossip storm compares the
serialized interface list, so an unsorted list would report a spurious change
every poll and defeat the fix. These tests pin the sort + stability.
"""

import socket
from typing import NamedTuple

import pytest

from skulk.utils.info_gatherer import system_info


class _Addr(NamedTuple):
    family: int
    address: str


async def _no_interface_types() -> dict[str, str]:
    return {}


@pytest.mark.asyncio
async def test_get_network_interfaces_is_sorted_and_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Interfaces in a deliberately unsorted order, with a mix of IPv4/IPv6 and a
    # non-IP address family that must be dropped (family 0 = AF_UNSPEC).
    unsorted = {
        "en3": [_Addr(socket.AF_INET, "10.0.0.3")],
        "en1": [
            _Addr(socket.AF_INET6, "fe80::2"),
            _Addr(socket.AF_INET, "10.0.0.1"),
        ],
        "en2": [
            _Addr(socket.AF_INET, "10.0.0.2"),
            _Addr(0, "should-be-dropped"),
        ],
    }
    monkeypatch.setattr(system_info.psutil, "net_if_addrs", lambda: unsorted)
    monkeypatch.setattr(
        system_info, "_get_interface_types_from_networksetup", _no_interface_types
    )

    result = await system_info.get_network_interfaces()

    keys = [(i.name, i.ip_address, i.interface_type) for i in result]
    assert keys == sorted(keys), "interfaces must come back in a deterministic order"
    assert "should-be-dropped" not in [i.ip_address for i in result]

    # Change-detection depends on byte-stability across repeated polls.
    again = await system_info.get_network_interfaces()
    assert result == again
