"""Staging-cache eviction for lifecycle and pre-download disk pressure.

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

The same budget enforcement runs at instance deactivation, at node startup
(which reconciles orphans left by a crashed session), and may be invoked by
operator tooling. Before a new store-backed model is staged, the worker can
also sacrifice idle entries inside the grace budget when the filesystem would
otherwise run out of space.
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

STAGING_MIN_FREE_BYTES = 10 * 1024**3
"""Minimum filesystem headroom retained after a predicted staging transfer."""

STAGING_MIN_FREE_FRACTION = 0.05
"""Minimum fraction of the staging filesystem retained as free headroom."""


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
    """Return bytes already present for one staged or partially staged model.

    Args:
        staging_root: Root directory containing node-local staged models.
        model_id: Repository-form model identifier.

    Returns:
        Total bytes currently present below that model's staging directory.
    """

    return _directory_size_bytes(staging_root / staging_directory_name(model_id))


def required_staging_free_bytes(
    *,
    model_total_bytes: int,
    staged_model_bytes: int,
    filesystem_total_bytes: int,
) -> int:
    """Calculate free bytes required before resuming a staging transfer.

    The model's existing partial bytes are reusable, so only the remaining
    artifact bytes need new capacity. The transfer also preserves the larger
    of 10 GiB or five percent of the filesystem as operating-system headroom.

    Args:
        model_total_bytes: Expected final artifact size from the model card.
        staged_model_bytes: Bytes already present in the resumable staging
            directory.
        filesystem_total_bytes: Total capacity of the staging filesystem.

    Returns:
        Required free bytes before the transfer starts or resumes.
    """

    remaining_model_bytes = max(model_total_bytes - staged_model_bytes, 0)
    reserve_bytes = max(
        STAGING_MIN_FREE_BYTES,
        int(filesystem_total_bytes * STAGING_MIN_FREE_FRACTION),
    )
    return remaining_model_bytes + reserve_bytes


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


def enforce_staging_budget(
    staging_root: Path,
    keep_recent_bytes: int,
    in_use_model_ids: frozenset[str] = frozenset(),
) -> StagingEvictionReport:
    """Evict least-recently-used staging candidates beyond the grace budget.

    In-use models are never touched. Candidates are retained newest-first
    until the grace budget is spent; the rest are deleted. With a budget of
    0 this is strict evict-on-deactivate.

    Deletion failures are logged and skipped — a partially evicted cache is
    still a smaller cache, and the next enforcement pass retries.
    """
    report = StagingEvictionReport()
    candidates = [
        info
        for info in list_staged_models(staging_root, in_use_model_ids)
        if not info.in_use
    ]

    retained_bytes = 0
    for info in candidates:
        if retained_bytes + info.size_bytes <= keep_recent_bytes:
            retained_bytes += info.size_bytes
            continue
        try:
            shutil.rmtree(info.directory)
        except OSError as error:
            logger.warning(
                f"Staging eviction could not remove {info.directory}: {error}"
            )
            continue
        report.evicted_model_ids.append(info.model_id)
        report.evicted_bytes += info.size_bytes
        age_hours = (time.time() - info.last_used_epoch_seconds) / 3600
        logger.info(
            f"Evicted staged model {info.model_id} "
            f"({info.size_bytes / 2**30:.1f} GiB, last used "
            f"~{age_hours:.1f}h ago) — staging held to the "
            f"{keep_recent_bytes / 2**30:.0f} GiB recent-use budget"
        )
    report.retained_candidate_bytes = retained_bytes
    return report


def evict_staging_bytes(
    staging_root: Path,
    bytes_to_reclaim: int,
    in_use_model_ids: frozenset[str] = frozenset(),
) -> StagingEvictionReport:
    """Evict oldest idle staged models until the requested bytes are reclaimed.

    This is the disk-pressure counterpart to :func:`enforce_staging_budget`.
    It deliberately ignores the recent-use grace budget: retaining a warm
    cache cannot take priority over completing a model the user just launched.
    Live models, active downloads, and the incoming resumable directory remain
    protected through ``in_use_model_ids``.

    Args:
        staging_root: Root directory containing node-local staged models.
        bytes_to_reclaim: Minimum number of bytes to attempt to reclaim.
        in_use_model_ids: Repository-form model IDs that must never be evicted.

    Returns:
        Eviction report naming removed models and the bytes they occupied.
    """

    report = StagingEvictionReport()
    candidates = [
        info
        for info in list_staged_models(staging_root, in_use_model_ids)
        if not info.in_use
    ]
    if bytes_to_reclaim <= 0:
        report.retained_candidate_bytes = sum(
            candidate.size_bytes for candidate in candidates
        )
        return report

    evicted_directories: set[str] = set()
    for info in reversed(candidates):
        if report.evicted_bytes >= bytes_to_reclaim:
            break
        try:
            shutil.rmtree(info.directory)
        except OSError as error:
            logger.warning(
                f"Staging pressure eviction could not remove {info.directory}: "
                f"{error}"
            )
            continue
        report.evicted_model_ids.append(info.model_id)
        report.evicted_bytes += info.size_bytes
        evicted_directories.add(info.directory)
        age_hours = (time.time() - info.last_used_epoch_seconds) / 3600
        logger.info(
            f"Evicted staged model {info.model_id} "
            f"({info.size_bytes / 2**30:.1f} GiB, last used "
            f"~{age_hours:.1f}h ago) to make room for an incoming model"
        )

    report.retained_candidate_bytes = sum(
        candidate.size_bytes
        for candidate in candidates
        if candidate.directory not in evicted_directories
    )
    return report
