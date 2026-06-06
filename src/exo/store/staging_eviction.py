"""Staging-cache eviction: one budget mechanism, three triggers.

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

The same enforcement runs at instance deactivation, at node startup (which
is what reconciles orphans left by a crashed session), and may be invoked
by operator tooling.
"""

import contextlib
import shutil
import time
from pathlib import Path

from loguru import logger
from pydantic import Field

from exo.utils.pydantic_ext import CamelCaseModel

LAST_USED_MARKER_FILENAME = ".last_used"
"""Marker file touched whenever a staged model is resolved for loading.

Directory mtimes change on content writes, not reads, so without the
marker a model staged long ago but used constantly would look idle to the
LRU ordering."""


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
    """Invert the ``org--name`` sanitization used for staging directories."""
    return directory_name.replace("--", "/", 1)


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
                in_use=model_id in in_use_model_ids,
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
