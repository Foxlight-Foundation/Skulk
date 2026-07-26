"""Staging-cache eviction with recency and free-space safety.

The staging directory holds node-local copies of store-served models. Left
unmanaged it grows without bound — every model ever staged survives both
instance deletion and node crashes (58-70 GB piles observed; one node died
of a full disk during the 2026-06-06 launch smoke).

The policy here is deliberately a single mechanism:

* A staged model is an **eviction candidate** when no live runner uses it
  (including as a companion — an MTP sidecar or assistant of an active
  model is in use even though no instance names it directly).
* Candidates are kept, newest-first by last use, up to the
  ``staging_keep_recent_gb`` grace budget; everything beyond it is
  deleted. The grace budget exists because node deaths, restarts, and
  repeated place/delete cycles of the same model should not re-pay the
  staging copy every time.
* Before a new store-backed download, the same least-recently-used
  eviction mechanism may override that grace budget to make room for the
  incoming model while preserving operating-system headroom.

Lifecycle enforcement runs at instance deactivation and node startup
(which reconciles orphans left by a crashed session). Capacity enforcement
runs immediately before a store-backed download.
"""

import contextlib
import shutil
import time
from pathlib import Path

from loguru import logger
from pydantic import Field

from skulk.utils.pydantic_ext import CamelCaseModel

LAST_USED_MARKER_FILENAME = ".last_used"
"""Marker file touched whenever a staged model is resolved for loading.

Directory mtimes change on content writes, not reads, so without the
marker a model staged long ago but used constantly would look idle to the
LRU ordering."""

MINIMUM_STAGING_FREE_DISK_BYTES = 10 * 1024**3
"""Free space Skulk preserves after staging an incoming model.

The staging cache contains reproducible copies while the rest of the volume
contains the operating system, logs, package caches, and user data. Keeping
10 GiB uncommitted prevents an otherwise successful model copy from leaving
the host at the filesystem's hard-stop boundary.
"""


class StagedModelInfo(CamelCaseModel):
    """One staged model directory, as seen by eviction and the storage API."""

    model_id: str
    """Model ID in repo form (``org/name``), reconstructed from the
    directory name."""

    directory: str
    """Absolute path of the staged copy."""

    size_bytes: int
    """Total size of all files in the staged copy."""

    last_used_epoch_seconds: float
    """Best-known last-use time: the ``.last_used`` marker when present,
    else the directory mtime."""

    in_use: bool = False
    """True when a live runner currently uses this model (directly or as a
    companion). In-use models are never eviction candidates."""


def model_id_from_staging_directory_name(directory_name: str) -> str:
    """Best-effort inverse of the ``org--name`` directory sanitization.

    Display/reporting only — the inverse is ambiguous when a repo name
    itself contains ``--`` next to the separator, so MATCHING never uses
    it: in-use checks compare forward-sanitized directory names instead
    (see ``list_staged_models``).
    """
    return directory_name.replace("--", "/", 1)


def staging_directory_name(model_id: str) -> str:
    """Forward sanitization for staging directory names (``org--name``)."""
    return model_id.replace("/", "--")


def touch_last_used(staged_model_directory: Path) -> None:
    """Record that a staged model was just resolved for loading.

    Best-effort: a failed touch only weakens LRU ordering, it must never
    interfere with the load path.
    """
    with contextlib.suppress(OSError):
        (staged_model_directory / LAST_USED_MARKER_FILENAME).touch()


def _directory_size_bytes(directory: Path) -> int:
    total = 0
    for file_path in directory.rglob("*"):
        try:
            if file_path.is_file():
                total += file_path.stat().st_size
        except OSError:
            continue
    return total


def staged_model_size_bytes(staging_root: Path, model_id: str) -> int:
    """Return the bytes already present for one resumable staged model."""
    directory = staging_root / staging_directory_name(model_id)
    if not directory.is_dir():
        return 0
    return _directory_size_bytes(directory)


def _last_used_epoch_seconds(directory: Path) -> float:
    marker = directory / LAST_USED_MARKER_FILENAME
    try:
        if marker.exists():
            return marker.stat().st_mtime
        return directory.stat().st_mtime
    except OSError:
        return 0.0


def list_staged_models(
    staging_root: Path,
    in_use_model_ids: frozenset[str] = frozenset(),
) -> list[StagedModelInfo]:
    """Inventory the staging directory, newest-used first.

    ``in_use_model_ids`` are repo-form IDs (``org/name``) of models a live
    runner currently depends on — including companion repos of active
    models.
    """
    if not staging_root.is_dir():
        return []
    # In-use matching is by forward-sanitized DIRECTORY name: the inverse
    # mapping is ambiguous for ids with "--" near the separator, and a
    # mis-match here could evict a live model's files.
    in_use_directory_names = {
        staging_directory_name(model_id) for model_id in in_use_model_ids
    }
    staged: list[StagedModelInfo] = []
    for entry in staging_root.iterdir():
        if not entry.is_dir():
            continue
        model_id = model_id_from_staging_directory_name(entry.name)
        staged.append(
            StagedModelInfo(
                model_id=model_id,
                directory=str(entry),
                size_bytes=_directory_size_bytes(entry),
                last_used_epoch_seconds=_last_used_epoch_seconds(entry),
                in_use=entry.name in in_use_directory_names,
            )
        )
    staged.sort(key=lambda info: info.last_used_epoch_seconds, reverse=True)
    return staged


class StagingEvictionReport(CamelCaseModel):
    """Result of one budget enforcement pass."""

    evicted_model_ids: list[str] = Field(default_factory=list)
    evicted_bytes: int = 0
    retained_candidate_bytes: int = 0
    """Bytes of not-in-use staged data kept under the grace budget."""
    required_free_bytes: int = 0
    """Free bytes the caller required after eviction."""
    free_bytes_before: int | None = None
    """Filesystem free bytes before capacity-driven eviction, when requested."""
    free_bytes_after: int | None = None
    """Filesystem free bytes after capacity-driven eviction, when requested."""
    capacity_satisfied: bool = True
    """Whether the requested free-space target was reached."""


def _filesystem_free_bytes(path: Path) -> int:
    """Return free bytes for ``path`` or its nearest existing parent."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def _remove_staged_model(
    info: StagedModelInfo,
    report: StagingEvictionReport,
    *,
    reason: str,
) -> bool:
    """Remove one idle staged model and record the successful reclamation."""
    try:
        shutil.rmtree(info.directory)
    except OSError as error:
        logger.warning(f"Staging eviction could not remove {info.directory}: {error}")
        return False
    report.evicted_model_ids.append(info.model_id)
    report.evicted_bytes += info.size_bytes
    age_hours = (time.time() - info.last_used_epoch_seconds) / 3600
    logger.info(
        f"Evicted staged model {info.model_id} "
        f"({info.size_bytes / 2**30:.1f} GiB, last used "
        f"~{age_hours:.1f}h ago) — {reason}"
    )
    return True


def enforce_staging_budget(
    staging_root: Path,
    keep_recent_bytes: int,
    in_use_model_ids: frozenset[str] = frozenset(),
    *,
    required_free_bytes: int = 0,
    enforce_recent_budget: bool = True,
) -> StagingEvictionReport:
    """Evict idle staged models to enforce recency and free-space constraints.

    In-use models are never touched. When ``enforce_recent_budget`` is true,
    candidates are retained newest-first until the grace budget is spent and
    the older tail is deleted. With a budget of 0 this is strict
    evict-on-deactivate.

    When ``required_free_bytes`` is non-zero, least-recently-used retained
    candidates are also removed until the filesystem reaches that target.
    This pressure pass intentionally overrides the warm-cache grace budget,
    but never the in-use set. Deletion failures are logged and skipped.
    """
    if keep_recent_bytes < 0:
        raise ValueError("keep_recent_bytes must be non-negative")
    if required_free_bytes < 0:
        raise ValueError("required_free_bytes must be non-negative")

    report = StagingEvictionReport(required_free_bytes=required_free_bytes)
    candidates = [
        info
        for info in list_staged_models(staging_root, in_use_model_ids)
        if not info.in_use
    ]

    retained: list[StagedModelInfo] = []
    over_budget: list[StagedModelInfo] = []
    retained_bytes = 0
    for info in candidates:
        if (
            not enforce_recent_budget
            or retained_bytes + info.size_bytes <= keep_recent_bytes
        ):
            retained.append(info)
            retained_bytes += info.size_bytes
            continue
        over_budget.append(info)

    for info in reversed(over_budget):
        _remove_staged_model(
            info,
            report,
            reason=(
                "staging held to the "
                f"{keep_recent_bytes / 2**30:.0f} GiB recent-use budget"
            ),
        )

    if required_free_bytes:
        report.free_bytes_before = _filesystem_free_bytes(staging_root)
        free_bytes = report.free_bytes_before
        for info in reversed(retained):
            if free_bytes >= required_free_bytes:
                break
            if _remove_staged_model(
                info,
                report,
                reason=(
                    "free-space pressure required "
                    f"{required_free_bytes / 2**30:.1f} GiB before staging"
                ),
            ):
                retained_bytes -= info.size_bytes
                free_bytes = _filesystem_free_bytes(staging_root)
        report.free_bytes_after = _filesystem_free_bytes(staging_root)
        report.capacity_satisfied = report.free_bytes_after >= required_free_bytes

    report.retained_candidate_bytes = retained_bytes
    return report
