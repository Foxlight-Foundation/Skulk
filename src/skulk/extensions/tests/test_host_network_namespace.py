"""Only a non-blank cluster namespace counts as private."""

from __future__ import annotations

import pytest

from skulk.extensions.host_network import private_namespace_configured


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({}, False),
        ({"SKULK_LIBP2P_NAMESPACE": ""}, False),
        ({"SKULK_LIBP2P_NAMESPACE": "   "}, False),
        ({"SKULK_LIBP2P_NAMESPACE": "family-cluster"}, True),
        ({"SKULK_LIBP2P_NAMESPACE": " family-cluster "}, True),
    ],
)
def test_private_namespace_requires_a_non_blank_value(
    environ: dict[str, str], expected: bool
) -> None:
    assert private_namespace_configured(environ) is expected
