# pyright: reportPrivateUsage=false, reportArgumentType=false
"""Tests for skulk.log rotation policy (#382).

The durable log must rotate once per process start (fresh log per run, previous
run retained as a compressed archive) AND whenever a write would push the file
past the size cap, so a long-lived node cannot grow its log without bound.
"""

from __future__ import annotations

from skulk.shared import logging as skulk_logging


class _FakeFile:
    """Minimal stand-in for loguru's open file handle: only ``tell`` is used."""

    def __init__(self, size: int) -> None:
        self._size = size

    def tell(self) -> int:
        return self._size


def test_rotation_policy_rotates_once_on_startup() -> None:
    policy = skulk_logging._make_rotation_policy()
    # The first write of a run always rotates (begins a clean skulk.log).
    assert policy("first message", _FakeFile(0)) is True


def test_rotation_policy_no_rotate_below_cap_after_startup() -> None:
    policy = skulk_logging._make_rotation_policy()
    policy("startup", _FakeFile(0))  # consume the once-on-startup rotation
    # A small write into a small file does not rotate.
    assert policy("x" * 100, _FakeFile(1024)) is False


def test_rotation_policy_rotates_when_write_exceeds_cap() -> None:
    policy = skulk_logging._make_rotation_policy()
    policy("startup", _FakeFile(0))  # consume the once-on-startup rotation
    # A write that would push the file past the cap triggers rotation.
    near_cap = _FakeFile(skulk_logging._MAX_LOG_BYTES - 10)
    assert policy("this message is more than ten bytes", near_cap) is True


def test_rotation_policy_independent_instances() -> None:
    # Each call site gets its own once-on-startup latch.
    first = skulk_logging._make_rotation_policy()
    second = skulk_logging._make_rotation_policy()
    assert first("m", _FakeFile(0)) is True
    assert second("m", _FakeFile(0)) is True
