"""Regression tests for #222: the Thunderbolt interface label must survive.

The Device-line downgrade previously rewrote EVERY enX≥2 device to
``maybe_ethernet`` — and Mac Thunderbolt ports are always en2+, so
"thunderbolt" could never exist on macOS and the ring's TB-first transport
priority was dead code. Fixture captured verbatim from a 16GB M4 (kite2)
with active Thunderbolt links.
"""

from skulk.utils.info_gatherer.system_info import parse_hardware_port_types

_KITE2_LISTING = """\
Hardware Port: Ethernet
Device: en0
Ethernet Address: 1c:f6:4c:3d:45:fa

Hardware Port: Ethernet Adapter (en5)
Device: en5
Ethernet Address: aa:67:a6:29:f1:bf

Hardware Port: Ethernet Adapter (en6)
Device: en6
Ethernet Address: aa:67:a6:29:f1:c0

Hardware Port: Ethernet Adapter (en7)
Device: en7
Ethernet Address: aa:67:a6:29:f1:c2

Hardware Port: Wi-Fi
Device: en1
Ethernet Address: 1c:f6:4c:3a:e0:12

Hardware Port: Thunderbolt 1
Device: en2
Ethernet Address: 36:6f:45:4b:a9:80

Hardware Port: Thunderbolt 2
Device: en3
Ethernet Address: 36:6f:45:4b:a9:84

Hardware Port: Thunderbolt 4
Device: en4
Ethernet Address: 36:6f:45:4b:a9:8c

VLAN Configurations
===================
"""


def test_thunderbolt_label_survives():
    types = parse_hardware_port_types(_KITE2_LISTING)
    assert types["en2"] == "thunderbolt"
    assert types["en3"] == "thunderbolt"
    assert types["en4"] == "thunderbolt"


def test_builtin_ethernet_and_wifi_keep_their_labels():
    types = parse_hardware_port_types(_KITE2_LISTING)
    assert types["en0"] == "ethernet"
    assert types["en1"] == "wifi"


def test_generic_ethernet_adapters_downgrade_to_maybe():
    # "Ethernet Adapter (enX)" on en2+ may be a USB dongle — the one case
    # that legitimately downgrades.
    types = parse_hardware_port_types(_KITE2_LISTING)
    assert types["en5"] == "maybe_ethernet"
    assert types["en6"] == "maybe_ethernet"
    assert types["en7"] == "maybe_ethernet"


def test_unclassified_ports_stay_unknown():
    # Previously promoted to maybe_ethernet (rank just below thunderbolt) —
    # an iPhone tether must not outrank real ethernet for ring transport.
    listing = "Hardware Port: iPhone USB\nDevice: en8\nEthernet Address: aa:bb\n"
    assert parse_hardware_port_types(listing)["en8"] == "unknown"
