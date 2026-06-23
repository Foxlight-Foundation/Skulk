# pyright: reportPrivateUsage=false
"""Tests for the pre-load fit tolerance (#383).

The worker's pre-load guard refuses a shard only when its (already padded)
footprint exceeds live usable memory beyond a tolerance, so a borderline
placement the master admitted is not flipped to a refusal by a sub-GB
live-vs-gossip jitter, while a gross shortfall still refuses.
"""

from skulk.shared.types.memory import Memory
from skulk.worker.main import _LOAD_FIT_TOLERANCE, footprint_exceeds_usable


def _gb(value: float) -> Memory:
    return Memory.from_bytes(int(value * 1_000_000_000))


def test_borderline_miss_within_tolerance_admits() -> None:
    # The observed #383 case: 9.2GB needed vs 9.0GB usable (a 0.2GB / ~2% miss)
    # must NOT refuse at the default tolerance.
    assert (
        footprint_exceeds_usable(_gb(9.2), _gb(9.0), _LOAD_FIT_TOLERANCE) is False
    )


def test_gross_shortfall_still_refuses() -> None:
    # A node that genuinely lost memory (another model loaded) still refuses.
    assert footprint_exceeds_usable(_gb(9.2), _gb(4.0), _LOAD_FIT_TOLERANCE) is True


def test_exact_fit_admits() -> None:
    assert footprint_exceeds_usable(_gb(9.0), _gb(9.0), _LOAD_FIT_TOLERANCE) is False


def test_boundary_at_tolerance_admits() -> None:
    # footprint exactly at usable * (1 + tolerance) is not "beyond" it.
    usable = _gb(10.0)
    at_edge = Memory.from_bytes(int(usable.in_bytes * (1 + _LOAD_FIT_TOLERANCE)))
    assert footprint_exceeds_usable(at_edge, usable, _LOAD_FIT_TOLERANCE) is False
    just_over = Memory.from_bytes(at_edge.in_bytes + 1)
    assert footprint_exceeds_usable(just_over, usable, _LOAD_FIT_TOLERANCE) is True


def test_zero_tolerance_is_strict_comparison() -> None:
    assert footprint_exceeds_usable(_gb(9.2), _gb(9.0), 0.0) is True
    assert footprint_exceeds_usable(_gb(9.0), _gb(9.0), 0.0) is False
