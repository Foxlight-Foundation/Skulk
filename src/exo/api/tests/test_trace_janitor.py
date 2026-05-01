"""Unit tests for ``prune_old_trace_files`` — the trace janitor's core logic.

The janitor itself is an async task on the API class; testing it end-to-end
would require spinning up the whole API. We instead exercise the pruning
function directly. It's pure (clock injected), so the tests pin time and
assert which files survive.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from exo.api.main import prune_old_trace_files


def _touch_trace(directory: Path, name: str, mtime: float) -> Path:
    path = directory / name
    path.write_text("{}")
    os.utime(path, (mtime, mtime))
    return path


def test_zero_retention_disables_pruning(tmp_path: Path) -> None:
    """retention_days=0 must short-circuit; nothing gets deleted."""
    now = time.time()
    old = _touch_trace(tmp_path, "trace_old.json", now - 100 * 86400)
    new = _touch_trace(tmp_path, "trace_new.json", now)

    removed = prune_old_trace_files(tmp_path, retention_days=0, now=now)

    assert removed == 0
    assert old.exists()
    assert new.exists()


def test_negative_retention_also_disables_pruning(tmp_path: Path) -> None:
    """Defensive: any non-positive value must disable pruning."""
    now = time.time()
    target = _touch_trace(tmp_path, "trace_old.json", now - 100 * 86400)

    removed = prune_old_trace_files(tmp_path, retention_days=-5, now=now)

    assert removed == 0
    assert target.exists()


def test_only_files_older_than_cutoff_are_removed(tmp_path: Path) -> None:
    """Cutoff = now - retention_days * 86400. Anything older goes; the rest stays."""
    now = time.time()
    retention_days = 3

    old_a = _touch_trace(tmp_path, "trace_old_a.json", now - 4 * 86400)
    old_b = _touch_trace(tmp_path, "trace_old_b.json", now - 5 * 86400)
    fresh = _touch_trace(tmp_path, "trace_fresh.json", now - 1 * 86400)
    boundary = _touch_trace(
        tmp_path,
        "trace_boundary.json",
        now - retention_days * 86400 + 1,  # just inside the window
    )

    removed = prune_old_trace_files(tmp_path, retention_days=retention_days, now=now)

    assert removed == 2
    assert not old_a.exists()
    assert not old_b.exists()
    assert fresh.exists()
    assert boundary.exists()


def test_non_trace_files_are_left_alone(tmp_path: Path) -> None:
    """Glob is `trace_*.json`; siblings of other names must survive."""
    now = time.time()
    sibling = tmp_path / "metadata.json"
    sibling.write_text("{}")
    os.utime(sibling, (now - 100 * 86400, now - 100 * 86400))

    removed = prune_old_trace_files(tmp_path, retention_days=3, now=now)

    assert removed == 0
    assert sibling.exists()


def test_missing_directory_is_a_no_op(tmp_path: Path) -> None:
    """Janitor should not raise when its directory doesn't exist yet."""
    nonexistent = tmp_path / "does_not_exist"

    removed = prune_old_trace_files(nonexistent, retention_days=3, now=time.time())

    assert removed == 0


def test_unlink_failure_does_not_crash_the_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single OSError on one file mustn't abort the whole sweep — counted
    out of the result, but other files still get pruned and the function
    returns normally."""
    now = time.time()
    bad = _touch_trace(tmp_path, "trace_bad.json", now - 100 * 86400)
    good = _touch_trace(tmp_path, "trace_good.json", now - 100 * 86400)

    real_unlink = Path.unlink

    def selective_unlink(self: Path, missing_ok: bool = False) -> None:
        if self == bad:
            raise OSError("simulated permission error")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", selective_unlink)

    removed = prune_old_trace_files(tmp_path, retention_days=3, now=now)

    assert removed == 1
    assert bad.exists()
    assert not good.exists()
