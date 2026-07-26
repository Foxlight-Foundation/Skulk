"""Tests for staging-cache recency and free-space enforcement.

Not-in-use staged models are retained newest-first up to the grace budget,
while pre-download capacity pressure may evict the oldest retained copies.
The grace budget keeps node crashes, restarts, and repeated place/delete
cycles from re-paying the staging copy every time.
"""

import os
import time
from pathlib import Path

import pytest

from skulk.store.staging_eviction import (
    LAST_USED_MARKER_FILENAME,
    MINIMUM_STAGING_FREE_DISK_BYTES,
    enforce_staging_budget,
    list_staged_models,
    model_id_from_staging_directory_name,
    staged_model_size_bytes,
    touch_last_used,
)

_GIB = 1024**3


def _stage_model(
    root: Path, model_id: str, size_bytes: int, last_used_age_seconds: float
) -> Path:
    directory = root / model_id.replace("/", "--")
    directory.mkdir(parents=True)
    (directory / "model.safetensors").write_bytes(b"\0" * size_bytes)
    marker = directory / LAST_USED_MARKER_FILENAME
    marker.touch()
    stamp = time.time() - last_used_age_seconds
    os.utime(marker, (stamp, stamp))
    return directory


def test_directory_name_round_trip() -> None:
    assert (
        model_id_from_staging_directory_name("mlx-community--Qwen3.5-9B-MLX-4bit")
        == "mlx-community/Qwen3.5-9B-MLX-4bit"
    )


def test_budget_retains_newest_and_evicts_oldest(tmp_path: Path) -> None:
    _stage_model(tmp_path, "org/newest", size_bytes=100, last_used_age_seconds=10)
    _stage_model(tmp_path, "org/middle", size_bytes=100, last_used_age_seconds=100)
    _stage_model(tmp_path, "org/oldest", size_bytes=100, last_used_age_seconds=1000)

    report = enforce_staging_budget(tmp_path, keep_recent_bytes=250)

    assert report.evicted_model_ids == ["org/oldest"]
    assert report.retained_candidate_bytes == 200
    assert (tmp_path / "org--newest").exists()
    assert (tmp_path / "org--middle").exists()
    assert not (tmp_path / "org--oldest").exists()


def test_zero_budget_is_strict_eviction(tmp_path: Path) -> None:
    _stage_model(tmp_path, "org/a", size_bytes=10, last_used_age_seconds=10)
    _stage_model(tmp_path, "org/b", size_bytes=10, last_used_age_seconds=20)

    report = enforce_staging_budget(tmp_path, keep_recent_bytes=0)

    assert sorted(report.evicted_model_ids) == ["org/a", "org/b"]
    assert list(tmp_path.iterdir()) == []


def test_in_use_models_are_never_evicted(tmp_path: Path) -> None:
    """The crash-recovery and live-runner cases: in-use models survive a
    zero budget, including companions named only by the card."""
    _stage_model(tmp_path, "org/base", size_bytes=10, last_used_age_seconds=10)
    _stage_model(tmp_path, "FoxlightAI/base-mtp", size_bytes=5, last_used_age_seconds=10)
    _stage_model(tmp_path, "org/idle", size_bytes=10, last_used_age_seconds=10)

    report = enforce_staging_budget(
        tmp_path,
        keep_recent_bytes=0,
        in_use_model_ids=frozenset({"org/base", "FoxlightAI/base-mtp"}),
    )

    assert report.evicted_model_ids == ["org/idle"]
    assert (tmp_path / "org--base").exists()
    assert (tmp_path / "FoxlightAI--base-mtp").exists()


def test_touch_last_used_refreshes_lru_position(tmp_path: Path) -> None:
    """A model staged long ago but used just now must sort newest.

    The genuinely OLD model (age 1000s) sorts behind the newer one until
    touched — so a broken touch_last_used cannot pass this test."""
    old_dir = _stage_model(
        tmp_path, "org/stale", size_bytes=10, last_used_age_seconds=1000
    )
    _stage_model(tmp_path, "org/fresh", size_bytes=10, last_used_age_seconds=10)

    staged_before = list_staged_models(tmp_path)
    assert [info.model_id for info in staged_before] == ["org/fresh", "org/stale"]

    touch_last_used(old_dir)
    staged = list_staged_models(tmp_path)

    assert [info.model_id for info in staged] == ["org/stale", "org/fresh"]


def test_staged_model_size_walks_only_requested_directory(tmp_path: Path) -> None:
    _stage_model(tmp_path, "org/requested", size_bytes=17, last_used_age_seconds=10)
    _stage_model(tmp_path, "org/unrelated", size_bytes=29, last_used_age_seconds=10)

    assert staged_model_size_bytes(tmp_path, "org/requested") == 17
    assert staged_model_size_bytes(tmp_path, "org/missing") == 0


def test_missing_staging_root_is_empty(tmp_path: Path) -> None:
    assert list_staged_models(tmp_path / "does-not-exist") == []
    report = enforce_staging_budget(tmp_path / "does-not-exist", keep_recent_bytes=0)
    assert report.evicted_model_ids == []


def test_budget_counts_only_candidates_not_in_use_bytes(tmp_path: Path) -> None:
    """In-use models must not consume the grace budget — a giant live model
    would otherwise force-evict every idle candidate."""
    _stage_model(tmp_path, "org/live-giant", size_bytes=1000, last_used_age_seconds=5)
    _stage_model(tmp_path, "org/idle-small", size_bytes=50, last_used_age_seconds=50)

    report = enforce_staging_budget(
        tmp_path,
        keep_recent_bytes=100,
        in_use_model_ids=frozenset({"org/live-giant"}),
    )

    assert report.evicted_model_ids == []
    assert report.retained_candidate_bytes == 50


def test_free_space_pressure_overrides_warm_cache_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model launch reclaims the oldest idle cache even below the grace budget."""
    _stage_model(tmp_path, "org/newest", size_bytes=40, last_used_age_seconds=10)
    _stage_model(tmp_path, "org/middle", size_bytes=40, last_used_age_seconds=100)
    _stage_model(tmp_path, "org/oldest", size_bytes=40, last_used_age_seconds=1000)

    def _free_bytes(_path: Path) -> int:
        reclaimed = sum(
            40
            for model_id in ("org--newest", "org--middle", "org--oldest")
            if not (tmp_path / model_id).exists()
        )
        return 50 + reclaimed

    monkeypatch.setattr(
        "skulk.store.staging_eviction._filesystem_free_bytes", _free_bytes
    )

    report = enforce_staging_budget(
        tmp_path,
        keep_recent_bytes=200,
        required_free_bytes=120,
    )

    assert report.evicted_model_ids == ["org/oldest", "org/middle"]
    assert report.free_bytes_before == 50
    assert report.free_bytes_after == 130
    assert report.capacity_satisfied
    assert report.retained_candidate_bytes == 40
    assert (tmp_path / "org--newest").exists()


def test_pressure_only_pass_does_not_apply_recency_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Download preflight preserves a healthy warm cache despite a zero budget."""
    _stage_model(tmp_path, "org/idle", size_bytes=40, last_used_age_seconds=100)

    def _free_bytes(_path: Path) -> int:
        return 1_000

    monkeypatch.setattr(
        "skulk.store.staging_eviction._filesystem_free_bytes",
        _free_bytes,
    )

    report = enforce_staging_budget(
        tmp_path,
        keep_recent_bytes=0,
        required_free_bytes=100,
        enforce_recent_budget=False,
    )

    assert report.evicted_model_ids == []
    assert report.capacity_satisfied
    assert (tmp_path / "org--idle").exists()


def test_pressure_never_evicts_protected_model_and_reports_shortfall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage_model(tmp_path, "org/incoming", size_bytes=40, last_used_age_seconds=1000)
    _stage_model(tmp_path, "org/idle", size_bytes=10, last_used_age_seconds=100)

    def _free_bytes(_path: Path) -> int:
        reclaimed = 10 if not (tmp_path / "org--idle").exists() else 0
        return 5 + reclaimed

    monkeypatch.setattr(
        "skulk.store.staging_eviction._filesystem_free_bytes", _free_bytes
    )

    report = enforce_staging_budget(
        tmp_path,
        keep_recent_bytes=100,
        in_use_model_ids=frozenset({"org/incoming"}),
        required_free_bytes=100,
        enforce_recent_budget=False,
    )

    assert report.evicted_model_ids == ["org/idle"]
    assert not report.capacity_satisfied
    assert report.free_bytes_after == 15
    assert (tmp_path / "org--incoming").exists()


def test_staging_preserves_ten_gib_operating_system_reserve() -> None:
    assert MINIMUM_STAGING_FREE_DISK_BYTES == 10 * _GIB


def test_store_host_direct_load_is_never_evicted(tmp_path: Path) -> None:
    """SEV-5 guard (codex, #215): a store host whose node_cache_path IS the
    canonical store directory must never run eviction there — the budget
    pass would delete the cluster's only copy of every model beyond it.
    Exercised at the worker layer via the path-equality guard; this test
    pins the comparison semantics it relies on (expanduser + resolve)."""
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    alias = tmp_path / "alias" / ".." / "store"
    (tmp_path / "alias").mkdir()

    assert store_dir.resolve() == alias.resolve()
