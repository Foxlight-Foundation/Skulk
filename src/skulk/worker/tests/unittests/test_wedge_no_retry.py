"""GPU-wedge deaths are never retried (the wedge-exit wired-memory leak).

A runner killed by the deadline watchdog mid-wedge leaks ~a shard of wired
GPU memory (measured 2026-06-09; reboot-only recovery), so the worker must
give the instance up on the FIRST wedge death instead of relaunching —
especially because wedges take ~300s each and would never trip the
3-in-60s crash window.
"""

from skulk.shared.types.worker.runners import RunnerFailed, RunnerReady
from skulk.worker.main import (
    _runner_failed_wedged,  # pyright: ignore[reportPrivateUsage] — unit under test
)
from skulk.worker.runner.bootstrap import WEDGE_EXIT_CODE, WEDGE_FAILURE_MARKER


def test_marker_failure_is_wedged() -> None:
    status = RunnerFailed(
        error_message=(
            f"Terminated ({WEDGE_FAILURE_MARKER}: deadline watchdog declared "
            "a GPU wedge (faulted Metal eval); wired memory may have leaked)"
        )
    )
    assert _runner_failed_wedged(status)


def test_ordinary_failures_are_not_wedged() -> None:
    assert not _runner_failed_wedged(
        RunnerFailed(error_message="Terminated (signal=6 (Abort trap: 6))")
    )
    assert not _runner_failed_wedged(RunnerFailed(error_message=None))
    assert not _runner_failed_wedged(RunnerReady())
    assert not _runner_failed_wedged(None)


def test_wedge_exit_code_is_distinct_from_common_codes() -> None:
    # 0 = clean, 1 = generic python failure, <0 = signals; the watchdog's
    # code must not collide with any of them or the supervisor would
    # misclassify ordinary deaths as wedges (and stop retrying transient
    # failures) or vice versa.
    assert WEDGE_EXIT_CODE not in (0, 1)
    assert WEDGE_EXIT_CODE > 0


def test_supervisor_maps_wedge_exit_code_to_marker() -> None:
    # Mirror the supervisor's cause-classification logic for the wedge code:
    # the marker must round-trip into RunnerFailed.error_message so the
    # worker-side matcher (_runner_failed_wedged) fires on it.
    rc: int = WEDGE_EXIT_CODE
    if rc < 0:
        cause = f"signal={-rc}"
    elif rc == WEDGE_EXIT_CODE:
        cause = (
            f"{WEDGE_FAILURE_MARKER}: deadline watchdog declared a GPU "
            "wedge (faulted Metal eval); wired memory may have leaked"
        )
    else:
        cause = f"exitcode={rc}"
    assert _runner_failed_wedged(RunnerFailed(error_message=f"Terminated ({cause})"))


def test_wedged_live_instances_sweep() -> None:
    """The planning-tick sweep catches LOCAL wedge deaths (single-node case).

    plan._kill_runner never emits Shutdown for a locally failed runner while
    its instance lives, so the sweep is the only path that frees a
    single-node placement from a wedged-dead runner.
    """
    from types import SimpleNamespace
    from typing import cast

    from skulk.worker.main import (
        _wedged_live_instances,  # pyright: ignore[reportPrivateUsage] — unit under test
    )

    def supervisor(instance_id: str, model_id: str, status: object):
        return SimpleNamespace(
            bound_instance=SimpleNamespace(
                instance=SimpleNamespace(instance_id=instance_id)
            ),
            shard_metadata=SimpleNamespace(
                model_card=SimpleNamespace(model_id=model_id)
            ),
            status=status,
        )

    wedged = RunnerFailed(
        error_message=f"Terminated ({WEDGE_FAILURE_MARKER}: ...)"
    )
    ordinary = RunnerFailed(error_message="Terminated (signal=6)")
    runners = cast(
        "dict[object, object]",
        {
            "r-wedged-live": supervisor("inst-a", "model-a", wedged),
            "r-wedged-deleted": supervisor("inst-gone", "model-b", wedged),
            "r-ordinary-failure": supervisor("inst-c", "model-c", ordinary),
            "r-healthy": supervisor("inst-d", "model-d", RunnerReady()),
        },
    )

    from skulk.shared.types.worker.instances import InstanceId
    from skulk.shared.types.worker.runners import RunnerId
    from skulk.worker.runner.runner_supervisor import RunnerSupervisor

    result = _wedged_live_instances(
        cast("dict[RunnerId, RunnerSupervisor]", cast(object, runners)),
        cast("set[InstanceId]", {"inst-a", "inst-c", "inst-d"}),
    )
    # Only the wedge-marked runner whose instance still lives is reported:
    # deleted instances follow the normal Shutdown cleanup, ordinary failures
    # keep the 3-in-60s breaker semantics, healthy runners are untouched.
    assert result == [("inst-a", "model-a")]
