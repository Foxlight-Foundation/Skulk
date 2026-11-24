"""What a node remembers about the last model load on each engine package build.

A load is the only thing that says whether a build serves a model on this
node, so the node keeps the newest answer per build and reports it with its
engine packages. The record lives in this process: it starts empty when the
node starts and is never written to disk, replicated or used by placement. It
is evidence for a person reading the engines report, nothing more.
"""

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, final

from skulk.shared.backends import engine_load_refusal_code
from skulk.shared.types.engine_packages import (
    MAX_NODE_ENGINE_PACKAGES,
    MAX_REPORTED_ENGINE_BUILDS,
    NodeEngineLoadResult,
    NodeEnginePackage,
)
from skulk.shared.types.worker.runners import (
    RunnerFailed,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerStatus,
    RunnerWarmingUp,
)

_LOAD_PHASES: Final = (RunnerLoading, RunnerLoaded, RunnerWarmingUp)


@final
@dataclass(frozen=True)
class LoadObservation:
    """How one runner's model load ended.

    Attributes:
        outcome: ``loaded`` when the runner became ready, ``refused`` when an
            engine package answered the load with a typed refusal, ``failed``
            when the runner ended while loading for any other reason.
        code: The engine's typed refusal code for ``refused``, else ``None``.
        at_unix_ms: When the runner supervisor saw the load end.
    """

    outcome: Literal["loaded", "refused", "failed"]
    code: str | None
    at_unix_ms: int


def observe_load(
    previous: RunnerStatus, new: RunnerStatus, *, now_unix_ms: int
) -> LoadObservation | None:
    """Read the end of a model load out of one runner status change.

    A runner reports its statuses in order, so a load ended exactly when a
    loading status is followed by ready or by failed. A failure that comes
    from any other status (before the load began, or while serving) is not a
    load result and returns ``None``.

    Args:
        previous: The status the runner held before this change.
        new: The status it now reports.
        now_unix_ms: The wall-clock time to stamp the observation with.

    Returns:
        The observation, or ``None`` when this change does not end a load.
    """
    if not isinstance(previous, _LOAD_PHASES):
        return None
    if isinstance(new, RunnerReady):
        return LoadObservation("loaded", None, now_unix_ms)
    if isinstance(new, RunnerFailed):
        code = engine_load_refusal_code(new.error_message or "")
        return LoadObservation(
            "failed" if code is None else "refused", code, now_unix_ms
        )
    return None


@final
class EngineLoadResults:
    """The newest load result for each engine package build on this node.

    Safe to call from any thread. Bounded: at most ``MAX_NODE_ENGINE_PACKAGES``
    backend tags and ``MAX_REPORTED_ENGINE_BUILDS`` builds per tag, the oldest
    result leaving first.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._results: dict[str, dict[str, NodeEngineLoadResult]] = {}

    def record(self, backend_tag: str, result: NodeEngineLoadResult) -> None:
        """Keep ``result`` unless a newer one for the same build is held.

        Recording the same observation again changes nothing, so a caller may
        report what it currently sees as often as it likes.

        Args:
            backend_tag: The backend tag the load was stamped with.
            result: The load result, which names its build.
        """
        with self._lock:
            builds = self._results.get(backend_tag)
            if builds is None:
                if len(self._results) >= MAX_NODE_ENGINE_PACKAGES:
                    return
                builds = self._results[backend_tag] = {}
            held = builds.get(result.build)
            if held is not None and held.at_unix_ms > result.at_unix_ms:
                return
            builds[result.build] = result
            while len(builds) > MAX_REPORTED_ENGINE_BUILDS:
                oldest = min(builds.values(), key=lambda entry: entry.at_unix_ms)
                del builds[oldest.build]

    def for_tag(self, backend_tag: str) -> tuple[NodeEngineLoadResult, ...]:
        """Return the held results for one backend tag, newest first."""
        with self._lock:
            held = tuple(self._results.get(backend_tag, {}).values())
        return tuple(
            sorted(held, key=lambda entry: (-entry.at_unix_ms, entry.build))
        )


PROCESS_ENGINE_LOAD_RESULTS: Final = EngineLoadResults()
"""The record this node's worker writes to. Process-wide because a worker is
replaced at every election change while the node, and what it has learned
about its builds, carries on."""


def with_load_results(
    packages: Sequence[NodeEnginePackage], results: EngineLoadResults
) -> tuple[NodeEnginePackage, ...]:
    """Attach each lane's held load results to the node's package report.

    Only builds the lane still names (its active build or one on disk) are
    attached: a result for a build the node no longer holds describes nothing
    a person can act on. A lane with no result is returned as it came, so a
    node that has loaded nothing reports exactly what it did before.

    Args:
        packages: The node's package report, one entry per lane.
        results: The node's load record.

    Returns:
        The report in the same order, with ``load_results`` filled in.
    """
    reported: list[NodeEnginePackage] = []
    for package in packages:
        named = {*package.installed_builds, package.build}
        held = tuple(
            result
            for result in results.for_tag(package.backend_tag)
            if result.build in named
        )
        reported.append(
            package.model_copy(update={"load_results": held}) if held else package
        )
    return tuple(reported)
