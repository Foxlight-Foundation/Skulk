"""Disk usage for a models folder that does not exist yet."""

from pathlib import Path

from skulk.shared.types.profiling import DiskUsage


def test_a_missing_folder_measures_its_nearest_existing_parent(tmp_path: Path) -> None:
    # A fresh install has no models folder before its first download; the
    # folder will be created on the partition its parent lives on.
    usage = DiskUsage.from_path(tmp_path / "not" / "created" / "models")

    assert usage.total == DiskUsage.from_path(tmp_path).total
