"""Unit tests for the Zenoh data-plane connectivity sampler.

The sampler feeds NodeResources advertisement, so its contract is what keeps
the ``zenoh_isolated`` health reason trustworthy: ``None`` means unknown
(gossipsub, sample failure, or startup grace), 0 means positive isolation
evidence, and a positive count permanently ends the grace shield.
"""

from skulk.routing.zenoh_status import ZenohPeerSampler


class _Clock:
    """Manually advanced monotonic clock for grace-window tests."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _sampler(
    counts: list[int | None], clock: _Clock, *, grace_seconds: float = 90.0
) -> ZenohPeerSampler:
    """Build a sampler whose underlying counts are served from a list."""

    async def count_peers() -> int | None:
        return counts.pop(0)

    return ZenohPeerSampler(count_peers, grace_seconds=grace_seconds, clock=clock)


async def test_gossipsub_none_passes_through() -> None:
    clock = _Clock()
    sampler = _sampler([None], clock)
    clock.now += 1000.0  # far past grace; None still means "no zenoh", not 0
    assert await sampler.advertised_count() is None


async def test_zero_inside_grace_advertises_unknown() -> None:
    clock = _Clock()
    sampler = _sampler([0], clock)
    clock.now += 89.0
    assert await sampler.advertised_count() is None


async def test_zero_after_grace_advertises_isolation() -> None:
    clock = _Clock()
    sampler = _sampler([0], clock)
    clock.now += 90.0
    assert await sampler.advertised_count() == 0


async def test_positive_count_advertised_immediately() -> None:
    clock = _Clock()
    sampler = _sampler([2], clock)
    # No grace applies to a connected session, even right after startup.
    assert await sampler.advertised_count() == 2


async def test_grace_never_returns_after_first_connection() -> None:
    clock = _Clock()
    sampler = _sampler([1, 0], clock)
    assert await sampler.advertised_count() == 1
    # A later 0 is a real transition to isolation (peers left / link died),
    # not startup noise, so it must be advertised even inside the window.
    clock.now += 1.0
    assert await sampler.advertised_count() == 0


async def test_sample_failure_advertises_unknown() -> None:
    clock = _Clock()

    async def raising_count() -> int | None:
        raise RuntimeError("session gone")

    sampler = ZenohPeerSampler(raising_count, clock=clock)
    clock.now += 1000.0
    assert await sampler.advertised_count() is None
