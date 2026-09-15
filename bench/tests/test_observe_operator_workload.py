"""The control channel accepts flow labels, never arbitrary metadata or URLs."""

import asyncio

import pytest

from bench.observe_operator_workload import capture_controls
from bench.operator_fixture_observer import FixtureObservationError, FixtureObserver


@pytest.mark.parametrize(
    "command",
    [
        b"",
        b"begin private-user\n",
        b"begin chat",
        b"x" * 129 + b"\n",
        b"finish secret\n",
        b"begin \xff\n",
    ],
)
async def test_controls_reject_eof_unknown_and_oversized_input(command: bytes) -> None:
    """Invalid commands cannot be accepted as flow markers or echoed secrets."""
    reader = asyncio.StreamReader(limit=128)
    reader.feed_data(command)
    reader.feed_eof()
    with pytest.raises(FixtureObservationError):
        await capture_controls(reader, FixtureObserver(lambda _: None))


async def test_controls_require_idle_before_finishing() -> None:
    """A finish command cannot truncate an open TCP lifetime."""
    observer = FixtureObserver(lambda _: None)
    observer.begin("chat")
    observer.accepted()
    reader = asyncio.StreamReader(limit=128)
    reader.feed_data(b"finish\n")
    with pytest.raises(FixtureObservationError):
        await capture_controls(reader, observer)
